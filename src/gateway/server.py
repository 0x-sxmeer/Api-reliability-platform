"""
Gateway Edge Server
Exposes the core engines over HTTP via FastAPI.

Security posture (audit remediation session):
  * F-02: every request to /v1/proxy must carry an identity. When
    GATEWAY_AUTH_TOKEN is configured, a matching `Authorization: Bearer`
    header is required and the token maps to a tenant identity; when it is
    NOT configured the server runs in explicit dev mode (identity "default")
    and logs that loudly once at startup. No silent unauthenticated prod.
  * F-03: the provider-pinned path applies policy -> budget -> quota gates
    exactly like the routed path before executing.
  * F-01: webhook ingress requires an HMAC-SHA256 signature over the raw
    body when GATEWAY_WEBHOOK_SECRET is configured; unsigned requests are
    rejected 403. Without the secret, webhooks are DISABLED outright (503)
    rather than left open — settlement mutates ledger state.
  * Info-leak: upstream failure details are no longer serialized to clients;
    the event_id is returned so operators can correlate internally.
"""
import asyncio
import hashlib
import hmac
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from gateway.adapters.anthropic import AnthropicAdapter
from gateway.adapters.openai import OpenAIAdapter
from gateway.adapters.rest import GenericRestAdapter
from gateway.core.adapter import AdapterRequest, ProviderAdapter
from gateway.core.executor import CallExecutor
from gateway.core.types import CallOutcome
from gateway.engines.auth import AuthLifecycleManager
from gateway.engines.budget import BudgetEngine
from gateway.engines.policy import (
    OperationPolicy,
    PolicyConfig,
    PolicyEngine,
    PolicyVerdict,
)
from gateway.engines.quota import QuotaEngine, Verdict
from gateway.engines.reconcile import ReconciliationEngine
from gateway.engines.router import RoutingConfig, RoutingEngine, RoutingTarget
from gateway.engines.validator import ResponseValidator, ValidationVerdict
from gateway.ledger.store import SqliteLedgerStore

logger = logging.getLogger(__name__)

# Global state
app_state = {}


def _resolve_identity(http_request: Request) -> str:
    """Audit F-02: derive the tenant identity from the request.

    When GATEWAY_AUTH_TOKEN is set, requests must carry
    `Authorization: Bearer <token>`; the token maps to an identity via
    GATEWAY_TENANT_MAP (JSON: {"<token>": "<identity_key>"}) or falls back
    to "authenticated". Without a constant-time compare this would be
    timing-oracle-prone, so hmac.compare_digest is used.

    When GATEWAY_AUTH_TOKEN is unset we are in explicit dev mode: identity
    "default" is returned, and startup logs a loud warning (see lifespan).
    """
    expected = os.getenv("GATEWAY_AUTH_TOKEN", "")
    if not expected:
        return "default"
    header = http_request.headers.get("authorization", "")
    token = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""
    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid credentials")
    try:
        mapping = json.loads(os.getenv("GATEWAY_TENANT_MAP", "{}"))
    except ValueError:
        mapping = {}
    return mapping.get(token, "authenticated")


def _safe_json(response) -> object:
    """Passthrough body extraction that never 500s on non-JSON upstreams.

    Fixes the audit-noted `response.json()` crash on HTML/text error bodies.
    """
    if response is None:
        return None
    try:
        return response.json()
    except ValueError:
        text = response.text
        # Cap echoed body: hostile/oversized upstreams must not be mirrored
        # wholesale into our own responses (DoS amplification).
        return {"_raw_body": text[:65536]}


def _build_wiring(db_path: str) -> dict:
    """Construct every engine/adapter for one gateway instance.

    Kept separate from the lifespan context manager so tests (and any
    future multi-tenant factory) can build an isolated wiring graph
    without touching global app_state or starting background tasks.
    """
    # Initialize Ledger
    ledger = SqliteLedgerStore(db_path)

    # Dev Mode awareness: when no real credentials are configured, tell the
    # operator (loudly, once) that outbound HTTP is being faked at the
    # transport layer. Engines above it run 100% for real.
    from gateway.devmode import describe as _describe_mode
    from gateway.devmode import dev_mode_enabled

    if dev_mode_enabled():
        logger.warning("GATEWAY STARTING IN %s", _describe_mode())

    # Initialize Engines
    quota_engine = QuotaEngine(ledger=ledger)
    validator = ResponseValidator()
    budget_engine = BudgetEngine(ledger=ledger)
    # Demo/dev policy: allow chat-style operations across all registered
    # providers. Production deployments should replace this with a real
    # least-privilege policy file; default_deny stays True so anything
    # not explicitly allowed is still blocked.
    policy_config = PolicyConfig(
        policies={
            "default": [OperationPolicy(allowed_providers="*", allowed_operations=[
                "chat.completions.create", "chat", "messages.create", "generate_content",
            ])]
        }
    )
    policy_engine = PolicyEngine(ledger=ledger, config=policy_config)

    # Initialize Router
    router = RoutingEngine(
        quota_engine=quota_engine,
        validator=validator,
        budget_engine=budget_engine,
        policy_engine=policy_engine,
        config=RoutingConfig(max_total_attempts=5)
    )

    # Provider credentials come from the environment at deploy time.
    # Default to "mock" so local dev / test boots never crash; production
    # deployments must set OPENAI_API_KEY / ANTHROPIC_API_KEY (and any
    # future per-vendor keys) — vendor-specific knowledge stays in adapters,
    # the server only reads generic env vars here.
    openai_api_key = os.getenv("OPENAI_API_KEY", "mock")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "mock")

    # Routing order = failover preference: OpenAI first, then Anthropic,
    # then the generic REST adapter as the terminal fallback target.
    adapters: list[ProviderAdapter] = [
        OpenAIAdapter(api_key=openai_api_key),
        AnthropicAdapter(api_key=anthropic_api_key),
        GenericRestAdapter(),
    ]
    for adapter in adapters:
        quota_engine.register(adapter)  # register with quota BEFORE routing targets
        router.register(RoutingTarget(
            adapter=adapter,
            executor=CallExecutor(adapter=adapter, ledger=ledger, identity_key="default"),
        ))

    return {
        "router": router,
        "reconcile": ReconciliationEngine(ledger=ledger),
        "auth": AuthLifecycleManager(),
        "ledger": ledger,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_path = os.getenv("GATEWAY_DB_PATH", "gateway.db")
    app_state.update(_build_wiring(db_path))
    reconcile_engine = app_state["reconcile"]

    # Audit F-02 / dev-mode footgun loudness: state the security posture at
    # startup so an operator who forgot env vars sees it in plain sight.
    if not os.getenv("GATEWAY_AUTH_TOKEN", ""):
        logger.warning(
            "GATEWAY AUTH IS OPEN: GATEWAY_AUTH_TOKEN unset — /v1/proxy accepts "
            "unauthenticated traffic as identity 'default'. Set the token before "
            "exposing this server to any network you do not control."
        )
    if not os.getenv("GATEWAY_WEBHOOK_SECRET", ""):
        logger.warning(
            "WEBHOOKS DISABLED: GATEWAY_WEBHOOK_SECRET unset — /v1/webhooks/* "
            "returns 503 until a signing secret is configured (audit F-01)."
        )

    # Background worker for stuck pending events
    async def poll_pending_events():
        while True:
            try:
                # Polling loop for Category 8 (human-in-the-loop) and stuck Category 7
                events = await reconcile_engine.get_unresolved_events(limit=100)
                if events:
                    logger.info(
                        "poll_pending_events: %d unresolved event(s) awaiting settlement",
                        len(events),
                    )
                # In production, this would execute refetches against the provider
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("poll_pending_events: error during reconciliation poll")
            await asyncio.sleep(60)

    bg_task = asyncio.create_task(poll_pending_events())
    app_state["bg_task"] = bg_task

    yield

    # Cleanup: cancel the background worker and wait for it to finish so
    # shutdown is deterministic (no pending-task warnings / lost logs).
    bg_task = app_state.get("bg_task")
    if bg_task:
        bg_task.cancel()
        try:
            await bg_task
        except asyncio.CancelledError:
            pass
    app_state.clear()


app = FastAPI(title="API Reliability Platform", lifespan=lifespan)


class ProxyRequest(BaseModel):
    provider: str | None = None
    operation: str
    payload: dict
    extra: dict | None = None


@app.post("/v1/proxy")
async def proxy(request: ProxyRequest, http_request: Request):
    """
    Primary ingress endpoint for the Gateway Edge.

    When `provider` is given, the request is dispatched to exactly that
    adapter (no cross-provider failover — the caller chose their vendor).
    Audit F-03 remediation: the provider-pinned path previously executed
    directly against the target's executor, bypassing policy/budget/quota
    gates entirely; it now runs the same gates before executing.

    Identity comes from `_resolve_identity` (audit F-02) and is threaded
    into route()/gates explicitly rather than inferred from targets[0]
    (audit F-05).
    """
    identity_key = _resolve_identity(http_request)
    router: RoutingEngine = app_state["router"]

    adapter_req = AdapterRequest(
        operation=request.operation,
        payload=request.payload,
        extra=request.extra,
    )

    if request.provider:
        target = next(
            (t for t in router.targets if t.adapter.provider_name == request.provider), None
        )
        if target is None:
            # Do not echo the full registered-provider list to unauthenticated
            # clients on an auth-enabled deployment? The list is non-secret
            # config; keep it — operators depend on this hint. (Info-leak fix
            # applies to *error classifications*, see below.)
            raise HTTPException(
                status_code=404,
                detail=f"Unknown provider '{request.provider}'. Registered: "
                       f"{[t.adapter.provider_name for t in router.targets]}",
            )

        # --- Gates that used to be bypassed (audit F-03) ---------------
        if router.policy is not None:
            assessment = await router.policy.assess(
                identity_key, target.adapter.provider_name, request.operation
            )
            if assessment.verdict != PolicyVerdict.ALLOWED:
                # Policy already wrote the audit event; do not leak the reason.
                raise HTTPException(status_code=403, detail="Request denied by policy.")
        if router.budget is not None:
            budget_assessment = await router.budget.assess(identity_key)
            if budget_assessment.verdict == Verdict.BLOCKED:
                raise HTTPException(status_code=402, detail="Monthly budget exhausted.")
        quota_assessment = await router.quota.assess(target.adapter, adapter_req)
        if quota_assessment.verdict == Verdict.BLOCKED:
            raise HTTPException(
                status_code=429,
                detail=f"{target.adapter.provider_name} is rate-limited; retry later.",
                # TODO(phase-9): honor Retry-After from quota_assessment.retry_at
            )

        result = await target.executor.execute(adapter_req)
        validation = router.validator.validate(result)
        if result.succeeded and validation.verdict != ValidationVerdict.INVALID:
            return {
                "status": "success",
                "provider": result.event.provider,
                "response": _safe_json(result.response),
                "warnings": [asdict(f) for f in validation.findings],
            }
        # Audit info-leak fix: internal classification no longer serialized
        # to clients; correlate via the logged event id instead.
        logger.warning(
            "proxy(pinned): %s call failed/invalid event=%s category=%s",
            request.provider,
            result.event.event_id,
            result.classified_error.category.value if result.classified_error else "validation-rejected",
        )
        raise HTTPException(
            status_code=502,
            detail="Call failed. Ask your gateway operator for the event id.",
        )

    outcome = await router.route(adapter_req, identity_key=identity_key)

    if outcome.final_result and outcome.succeeded:
        # Success — return the parsed body when one exists. Transport
        # failures carry no response object at all. Non-JSON bodies are
        # echoed as raw text (capped), never a 500 crash.
        return {
            "status": "success",
            "provider": outcome.final_result.event.provider,
            "response": _safe_json(outcome.final_result.response),
            "warnings": (
                [asdict(f) for f in outcome.final_validation.findings]
                if outcome.final_validation is not None
                else []
            ),
        }

    # Request ultimately failed (or was blocked by Policy/Budget).
    raise HTTPException(status_code=502, detail="Request blocked or failed across all routing attempts.")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


class WebhookPayload(BaseModel):
    event_id: UUID
    status: str
    reason: str | None = None


# Explicit webhook-status -> terminal-outcome mapping. Anything not listed
# here is rejected; we never guess FAILURE for an unknown receipt, because a
# misclassified webhook would corrupt the ledger's state machine.
_WEBHOOK_OUTCOMES: dict[str, CallOutcome] = {
    "delivered": CallOutcome.SUCCESS,
    "accepted": CallOutcome.SUCCESS,
    "failed": CallOutcome.FAILURE,
    "bounced": CallOutcome.FAILURE,
}


def _verify_webhook_signature(raw_body: bytes, signature_header: str | None) -> bool:
    """Audit F-01 remediation: HMAC-SHA256 over the raw request body.

    Expected header: `X-Gateway-Signature: sha256=<hexdigest>` (Stripe-style
    scheme prefix for future-proofing). Constant-time comparison via
    hmac.compare_digest. When GATEWAY_WEBHOOK_SECRET is unset the caller
    rejects webhooks outright (settlement mutates ledger state — an open
    webhook is a state-forgery endpoint, not a degraded feature).
    """
    secret = os.getenv("GATEWAY_WEBHOOK_SECRET", "")
    if not secret or not signature_header:
        return False
    expected = "sha256=" + hmac.new(
        secret.encode(), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header.strip())


@app.post("/v1/webhooks/{provider}")
async def webhook_ingress(provider: str, payload: WebhookPayload, request: Request):
    """
    Ingests downstream delivery receipts to settle pending events.
    Solves Category 7 (Downstream/third-party silent loss).

    Audit F-01 remediation: signature is verified BEFORE any ledger touch;
    unknown provider path values are rejected (the old code ignored
    `provider` entirely); replayed settlements are rejected by the store's
    state machine (already-settled -> ValueError -> HTTP 400).
    """
    if not os.getenv("GATEWAY_WEBHOOK_SECRET", ""):
        raise HTTPException(
            status_code=503,
            detail=(
                "Webhook ingress disabled: set GATEWAY_WEBHOOK_SECRET to enable "
                "HMAC-verified settlement."
            ),
        )
    registered = {
        t.adapter.provider_name for t in app_state["router"].targets
    } | {"stripe"}  # payments/messaging providers webhook without being routed
    if provider not in registered:
        raise HTTPException(status_code=404, detail=f"Unknown webhook provider '{provider}'.")

    raw_body = await request.body()
    if not _verify_webhook_signature(raw_body, request.headers.get("x-gateway-signature")):
        raise HTTPException(status_code=403, detail="Missing or invalid webhook signature.")

    reconcile: ReconciliationEngine = app_state["reconcile"]

    status = _WEBHOOK_OUTCOMES.get(payload.status)
    if status is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown webhook status '{payload.status}'. "
                f"Expected one of: {sorted(_WEBHOOK_OUTCOMES)}"
            ),
        )

    try:
        await reconcile.settle_event(
            event_id=payload.event_id,
            final_outcome=status,
            reason=payload.reason or f"Webhook receipt from {provider}",
        )
        return {"status": "accepted"}
    except ValueError as e:
        # Invalid state transition (already-settled / unknown event).
        raise HTTPException(status_code=400, detail=str(e)) from e
