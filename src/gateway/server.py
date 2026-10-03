"""
Gateway Edge Server
Exposes the core engines over HTTP via FastAPI.
"""
import asyncio
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
from gateway.core.adapter import AdapterRequest
from gateway.core.executor import CallExecutor
from gateway.core.types import CallOutcome
from gateway.engines.auth import AuthLifecycleManager
from gateway.engines.budget import BudgetEngine
from gateway.engines.policy import OperationPolicy, PolicyConfig, PolicyEngine
from gateway.engines.quota import QuotaEngine
from gateway.engines.reconcile import ReconciliationEngine
from gateway.engines.router import RoutingConfig, RoutingEngine, RoutingTarget
from gateway.engines.validator import ResponseValidator
from gateway.ledger.store import SqliteLedgerStore

logger = logging.getLogger(__name__)

# Global state
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Ledger
    db_path = os.getenv("GATEWAY_DB_PATH", "gateway.db")
    ledger = SqliteLedgerStore(db_path)
    
    # Initialize Engines
    quota_engine = QuotaEngine(ledger=ledger)
    validator = ResponseValidator()
    budget_engine = BudgetEngine(ledger=ledger)
    policy_config = PolicyConfig(
        policies={
            "default": [OperationPolicy(allowed_providers="*", allowed_operations=["chat.completions.create", "chat"])]
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
    
    openai_adapter = OpenAIAdapter(api_key="mock")
    quota_engine.register(openai_adapter)
    openai_target = RoutingTarget(
        adapter=openai_adapter,
        executor=CallExecutor(adapter=openai_adapter, ledger=ledger, identity_key="default")
    )
    anthropic_adapter = AnthropicAdapter(api_key="mock")
    quota_engine.register(anthropic_adapter)
    anthropic_target = RoutingTarget(
        adapter=anthropic_adapter,
        executor=CallExecutor(adapter=anthropic_adapter, ledger=ledger, identity_key="default")
    )
    rest_adapter = GenericRestAdapter()
    quota_engine.register(rest_adapter)
    rest_target = RoutingTarget(
        adapter=rest_adapter,
        executor=CallExecutor(adapter=rest_adapter, ledger=ledger, identity_key="default")
    )
    router.register(openai_target)
    router.register(anthropic_target)
    router.register(rest_target)

    reconcile_engine = ReconciliationEngine(ledger=ledger)
    auth_manager = AuthLifecycleManager()

    app_state["router"] = router
    app_state["reconcile"] = reconcile_engine
    app_state["auth"] = auth_manager
    app_state["ledger"] = ledger

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
    operation: str
    payload: dict
    extra: dict | None = None


@app.post("/v1/proxy")
async def proxy(request: ProxyRequest):
    """
    Primary ingress endpoint for the Gateway Edge.
    """
    router: RoutingEngine = app_state["router"]

    adapter_req = AdapterRequest(
        operation=request.operation,
        payload=request.payload,
        extra=request.extra,
    )

    outcome = await router.route(adapter_req)

    if outcome.final_result and outcome.succeeded:
        # Success — return the parsed body when one exists. Transport
        # failures carry no response object at all.
        response = outcome.final_result.response
        body = response.json() if response is not None else None
        warnings = []
        if outcome.final_validation is not None:
            # ValidationFinding is a plain frozen dataclass (no model_dump);
            # serialize via asdict().
            warnings = [asdict(f) for f in outcome.final_validation.findings]
        return {
            "status": "success",
            "provider": outcome.final_result.event.provider,
            "response": body,
            "warnings": warnings,
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


@app.post("/v1/webhooks/{provider}")
async def webhook_ingress(provider: str, payload: WebhookPayload, request: Request):
    """
    Ingests downstream delivery receipts to settle pending events.
    Solves Category 7 (Downstream/third-party silent loss).
    """
    # In a real system, verify signature here (e.g. Stripe-Signature)
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
