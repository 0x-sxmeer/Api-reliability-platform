"""
Gateway Edge Server
Exposes the core engines over HTTP via FastAPI.
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from gateway.ledger.store import SqliteLedgerStore
from gateway.engines.quota import QuotaEngine
from gateway.engines.validator import ResponseValidator
from gateway.engines.budget import BudgetEngine
from gateway.engines.policy import PolicyEngine
from gateway.engines.router import RoutingEngine, RoutingTarget, RoutingConfig
from gateway.core.executor import CallExecutor
from gateway.core.adapter import AdapterRequest

from gateway.adapters.openai import OpenAIAdapter
from gateway.adapters.anthropic import AnthropicAdapter
from gateway.adapters.rest import GenericRestAdapter
from gateway.engines.auth import AuthLifecycleManager
from gateway.engines.reconcile import ReconciliationEngine
from gateway.core.types import CallOutcome
import asyncio

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
    from gateway.engines.policy import PolicyConfig, OperationPolicy
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
                # In production, this would execute refetches against the provider
            except asyncio.CancelledError:
                break
            except Exception as e:
                pass
            await asyncio.sleep(60)

    bg_task = asyncio.create_task(poll_pending_events())
    app_state["bg_task"] = bg_task
    
    yield
    
    # Cleanup (if needed)
    bg_task = app_state.get("bg_task")
    if bg_task:
        bg_task.cancel()
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
        extra=request.extra
    )
    
    outcome = await router.route(adapter_req)
    
    if outcome.final_result:
        # Success!
        return {
            "status": "success",
            "provider": outcome.final_result.event.provider,
            "response": outcome.final_result.response.json() if hasattr(outcome.final_result.response, "json") and outcome.final_result.response else None,
            "warnings": [f.model_dump() for f in outcome.final_validation.findings] if outcome.final_validation else []
        }
    else:
        # Request ultimately failed (or was blocked by Policy/Budget)
        raise HTTPException(status_code=502, detail="Request blocked or failed across all routing attempts.")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

class WebhookPayload(BaseModel):
    event_id: str
    status: str
    reason: str | None = None

@app.post("/v1/webhooks/{provider}")
async def webhook_ingress(provider: str, payload: WebhookPayload, request: Request):
    """
    Ingests downstream delivery receipts to settle pending events.
    Solves Category 7 (Downstream/third-party silent loss).
    """
    # In a real system, verify signature here (e.g. Stripe-Signature)
    reconcile: ReconciliationEngine = app_state["reconcile"]
    
    status = CallOutcome.SUCCESS if payload.status == "delivered" else CallOutcome.FAILURE
    
    try:
        await reconcile.settle_event(
            event_id=payload.event_id,
            final_outcome=status,
            reason=payload.reason or f"Webhook receipt from {provider}"
        )
        return {"status": "accepted"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
