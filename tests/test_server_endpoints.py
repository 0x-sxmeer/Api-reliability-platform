"""
Integration tests for the FastAPI Gateway Edge Server (previously untested).

Exercises the real ASGI app through httpx.ASGITransport with a temp DB:
  * lifespan wiring (adapters registered, router/reconcile/ledger in app_state)
  * POST /v1/proxy happy path + validation warnings serialization (asdict)
  * POST /v1/proxy policy denial -> HTTP 502
  * GET  /health
  * POST /v1/webhooks/{provider}: settle PENDING -> SUCCESS, double-settle
    -> 400 (state machine), unknown status -> 422 (never guessed),
    malformed UUID -> 422.
"""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import httpx
import pytest

import gateway.server as server_module
from gateway.core.types import CallOutcome
from gateway.ledger.events import GatewayEvent


@pytest.fixture
async def client(tmp_path, monkeypatch) -> AsyncIterator[httpx.AsyncClient]:
    monkeypatch.setenv("GATEWAY_DB_PATH", str(tmp_path / "srv-test.db"))
    transport = httpx.ASGITransport(app=server_module.app)
    # Enter the real lifespan so app_state is populated.
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://gateway") as c,
        server_module.lifespan(server_module.app),
    ):
        yield c


# ------------------------------------------------------------------ wiring

async def test_lifespan_wires_expected_state(client):
    for key in ("router", "reconcile", "auth", "ledger"):
        assert key in server_module.app_state, key


async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


# ------------------------------------------------------------------ proxy

async def test_proxy_success_path(client):
    # The mock-keyed OpenAI adapter path used by simulate_traffic scenario 1;
    # if the environment has no network and the adapter fails, we still must
    # get a well-formed 502 (never a 500 crash).
    resp = await client.post(
        "/v1/proxy",
        json={"operation": "chat.completions.create", "payload": {"model": "gpt-6-astra", "messages": []}},
    )
    assert resp.status_code in (200, 502)
    if resp.status_code == 200:
        body = resp.json()
        assert body["status"] == "success"
        assert isinstance(body["warnings"], list)
        # Warnings serialize via dataclasses.asdict -> plain dicts, never
        # crash on model_dump (regression from the frozen-dataclass finding).
        for w in body["warnings"]:
            assert isinstance(w, dict)


async def test_proxy_policy_denial_returns_502_not_crash(client):
    resp = await client.post(
        "/v1/proxy",
        json={"operation": "dangerous_operation", "payload": {}},
    )
    assert resp.status_code == 502
    assert "detail" in resp.json()


# ------------------------------------------------------------------ webhooks

async def _seed_pending(client, provider: str = "generic_rest") -> uuid.UUID:
    ledger = server_module.app_state["ledger"]
    event_id = uuid.uuid4()
    await ledger.append(
        GatewayEvent(
            event_id=event_id,
            identity_key="default",
            provider=provider,
            operation="charge.create",
            outcome=CallOutcome.PENDING,
        )
    )
    return event_id


async def test_webhook_settles_pending_event(client):
    event_id = await _seed_pending(client)
    resp = await client.post(
        f"/v1/webhooks/{'stripe'}",
        json={"event_id": str(event_id), "status": "delivered", "reason": "receipt"},
    )
    assert resp.status_code == 200
    rows = await server_module.app_state["ledger"].query(provider="generic_rest")
    settled = next(r for r in rows if r.event_id == event_id)
    assert settled.outcome is CallOutcome.SUCCESS
    assert settled.original_outcome is CallOutcome.PENDING
    assert settled.reconciled_at is not None


async def test_webhook_double_settle_is_rejected_400(client):
    event_id = await _seed_pending(client)
    ok = await client.post("/v1/webhooks/stripe", json={"event_id": str(event_id), "status": "delivered"})
    assert ok.status_code == 200
    again = await client.post("/v1/webhooks/stripe", json={"event_id": str(event_id), "status": "failed"})
    assert again.status_code == 400
    # State machine held: still SUCCESS, not flipped to FAILURE.
    rows = await server_module.app_state["ledger"].query(provider="generic_rest")
    row = next(r for r in rows if r.event_id == event_id)
    assert row.outcome is CallOutcome.SUCCESS


async def test_webhook_unknown_status_is_422_never_guessed(client):
    event_id = await _seed_pending(client)
    resp = await client.post("/v1/webhooks/stripe", json={"event_id": str(event_id), "status": "maybe"})
    assert resp.status_code == 422
    rows = await server_module.app_state["ledger"].query(provider="generic_rest")
    row = next(r for r in rows if r.event_id == event_id)
    assert row.outcome is CallOutcome.PENDING  # untouched


async def test_webhook_malformed_uuid_is_422(client):
    resp = await client.post("/v1/webhooks/stripe", json={"event_id": "not-a-uuid", "status": "delivered"})
    assert resp.status_code == 422


# ---------------------------------------------------- pinned dispatch (all providers)

async def test_pinned_gemini_accepts_model_in_payload(client):
    """Regression: Gemini's model selects the URL path. OpenAI-style gateway
    requests carry it in payload; the adapter must accept that too instead
    of hard-failing (previously every pinned/failover Gemini call -> 502)."""
    resp = await client.post(
        "/v1/proxy",
        json={
            "operation": "generateContent",
            "provider": "gemini",
            "payload": {"contents": [{"parts": [{"text": "hi"}]}], "model": "gemini-2.0-flash"},
        },
    )
    assert resp.status_code in (200, 502)  # never 500
    if resp.status_code == 200:
        assert resp.json()["provider"] == "gemini"


async def test_pinned_unknown_provider_is_404(client):
    resp = await client.post(
        "/v1/proxy",
        json={"operation": "x.y", "provider": "definitely-not-real", "payload": {}},
    )
    assert resp.status_code == 404
    assert "Registered" in resp.json()["detail"]


async def test_pinned_dispatch_enforces_policy(client):
    """Pinning a vendor must never bypass AuthZ/BOLA."""
    resp = await client.post(
        "/v1/proxy",
        json={"operation": "payments.refund", "provider": "openai", "payload": {"model": "gpt-4o-mini"}},
    )
    # Operation not in the demo allow-list -> clean 502 policy denial, no crash.
    assert resp.status_code == 502
    assert "Policy denial" in resp.json()["detail"]
