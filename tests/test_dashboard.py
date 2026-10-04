"""Phase 9 dashboard: endpoint contracts, live-data propagation, and the
regression guards for bugs found during the full audit:

  * pinned dispatch must feed QuotaEngine (execute_observed) — otherwise the
    quota panel is permanently empty for pinned traffic;
  * providers endpoint must include stats fields even with zero ledger rows;
  * quota snapshot shape must match what the HTML renderer consumes
    (binding.requests_remaining / requests_limit etc.);
  * endpoints answer 503 before lifespan wiring exists.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from gateway.engines.budget import BudgetEngine
from gateway.engines.quota import QuotaEngine
from gateway.engines.router import RoutingTarget


@pytest.fixture()
async def wired_app(tmp_path, monkeypatch) -> AsyncIterator[tuple]:
    """Fresh temp DB + real lifespan wiring, isolated per test."""
    import gateway.server as server_mod

    monkeypatch.setenv("GATEWAY_DB_PATH", str(tmp_path / "dash-test.db"))
    monkeypatch.setenv("GATEWAY_FAKE_APIS", "1")
    app = server_mod.app
    async with server_mod.lifespan(app):
        yield app, server_mod.app_state


async def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_endpoints_503_before_wiring():
    """Dashboard APIs must refuse cleanly (503) when engines aren't up."""
    from fastapi import FastAPI

    from gateway.dashboard import add_dashboard_routes

    fresh = FastAPI()
    add_dashboard_routes(fresh)
    # No lifespan ever ran on this app object -> app.state has no wiring.
    async with await _client(fresh) as c:
        r = await c.get("/v1/dashboard/overview")
        assert r.status_code == 503


async def test_all_json_endpoints_ok_with_traffic(wired_app):
    app, _state = wired_app
    async with await _client(app) as c:
        for path in [
            "/v1/dashboard/overview",
            "/v1/dashboard/status",
            "/v1/dashboard/providers",
            "/v1/dashboard/quota",
            "/v1/dashboard/budget",
            "/v1/dashboard/spend",
            "/v1/dashboard/events?limit=5",
            "/v1/dashboard/reconciliation",
        ]:
            r = await c.get(path)
            assert r.status_code == 200, path

        pv = (await c.get("/v1/dashboard/providers")).json()
        # Every registered target appears WITH stats fields even at zero rows.
        assert len(pv["providers"]) >= 4
        for p in pv["providers"]:
            if p["in_failover_chain"]:
                assert {"calls", "successes", "failures", "spend_usd"} <= set(p)
                assert isinstance(p["calls"], int)
        assert pv["quota_snapshot"] == []
        assert pv["note"] is not None  # honest empty-state note


async def test_pinned_dispatch_feeds_quota_engine(wired_app):
    """REGRESSION GUARD: pinned calls must update QuotaEngine observations.

    Audit bug: the pinned path called executor.execute() directly, bypassing
    quota.observe(), so the dashboard's quota table stayed empty forever for
    pinned traffic. Now it goes through RoutingTarget.execute_observed().
    """
    _app, state = wired_app
    from gateway.server import ProxyRequest, proxy

    req = ProxyRequest(
        provider="openai",
        operation="chat.completions.create",
        payload={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
    )
    result = await proxy(req)
    assert result["status"] == "success"

    quota: QuotaEngine = state["quota"]
    snap = quota.snapshot()
    openai_buckets = [b for b in snap if b["provider"] == "openai"]
    assert openai_buckets, "pinned dispatch must create a quota observation"
    binding = openai_buckets[0]["binding"]
    assert binding is not None
    # Dev-mode fake sends x-ratelimit-limit-requests: 500 / remaining: 493.
    assert binding["requests_limit"] == 500
    assert binding["requests_remaining"] == 493


def test_execute_observed_helper_contract():
    """RoutingTarget.execute_observed feeds observe() exactly once per call."""
    import asyncio

    observed: list[Any] = []

    class FakeQuota:
        def observe(self, adapter, request, *, snapshot=None, error=None):
            observed.append((adapter.provider_name, snapshot, error))

    class StubAdapter:
        provider_name = "stub"

    class StubExecutor:
        identity_key = "default"

        async def execute(self, request):
            sentinel = object()
            return type("R", (), {"rate_limit": sentinel, "classified_error": None})()

    target = RoutingTarget(adapter=StubAdapter(), executor=StubExecutor())  # type: ignore[arg-type]
    from gateway.core.adapter import AdapterRequest

    req = AdapterRequest(operation="op", payload={})

    async def run():
        res = await target.execute_observed(req, FakeQuota())  # type: ignore[arg-type]
        assert res.rate_limit is not None
        assert len(observed) == 1
        assert observed[0][0] == "stub"
        # No quota engine passed -> plain execute, no crash.
        res2 = await target.execute_observed(req, None)
        assert res2 is not None

    asyncio.run(run())


async def test_events_endpoint_filters_and_validation(wired_app):
    app, _state = wired_app
    async with await _client(app) as c:
        r = await c.get("/v1/dashboard/events?outcome=bogus")
        assert r.status_code == 422
        r = await c.get("/v1/dashboard/events?limit=0")
        assert r.status_code == 422
        r = await c.get("/v1/dashboard/events?provider=openai&limit=5")
        assert r.status_code == 200
        assert isinstance(r.json()["events"], list)


async def test_dashboard_html_served_and_self_consistent(wired_app):
    app, _state = wired_app
    async with await _client(app) as c:
        r = await c.get("/dashboard")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        html = r.text
        # The page must fetch every JSON surface it renders from.
        for path in ["/overview", "/status", "/providers", "/budget",
                     "/reconciliation", "/events", "/spend"]:
            assert f"/v1/dashboard{path}" in html, path
        # Regression guards for renderer bugs fixed in the audit:
        assert "placeholder guard" not in html          # dead rows() call removed
        assert "fraction_remaining" not in html         # stale field name gone
        assert "requests_remaining" in html             # renderer matches snapshot() schema
        assert "declares_quota_windows" in html


async def test_budget_panel_shape(wired_app):
    app, state = wired_app
    budget: BudgetEngine = state["budget"]
    assessment = await budget.assess("default")
    async with await _client(app) as c:
        bg = (await c.get("/v1/dashboard/budget")).json()
        assert bg["verdict"] == assessment.verdict.value
        assert {"cost_headroom_usd", "block_reason", "assessed_at"} <= set(bg)


async def test_overview_reflects_recorded_traffic(wired_app):
    app, _state = wired_app
    from gateway.server import ProxyRequest, proxy

    async with await _client(app) as c:
        before = (await c.get("/v1/dashboard/overview")).json()["total_calls"]
        await proxy(ProxyRequest(
            provider="openai", operation="chat.completions.create",
            payload={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]}))
        after = (await c.get("/v1/dashboard/overview")).json()["total_calls"]
    assert after == before + 1

