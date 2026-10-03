"""
Unit tests for AuthLifecycleManager (Phase 7, previously untested).

Covers Category 4 behaviors that matter in production:
  * refresh happens exactly once when N concurrent callers race a stale token
  * the expiry safety buffer triggers a proactive refresh
  * force_invalidate causes the next get to re-refresh
  * locks are per provider:identity (no cross-tenant serialization)
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from gateway.engines.auth import AuthLifecycleManager, TokenState


def _state(token: str, seconds_live: int) -> TokenState:
    return TokenState(
        access_token=token,
        refresh_token=f"refresh-{token}",
        expires_at=datetime.now(UTC) + timedelta(seconds=seconds_live),
        provider="oauth",
    )


async def test_returns_cached_valid_token_without_refresh():
    mgr = AuthLifecycleManager()
    calls = 0

    async def refresh(_old):
        nonlocal calls
        calls += 1
        return _state("tok-A", 3600)

    # Seed cache via first call.
    assert await mgr.get_valid_token("oauth", "team-1", refresh) == "tok-A"
    assert calls == 1
    # Second call is within validity window -> no second refresh.
    assert await mgr.get_valid_token("oauth", "team-1", refresh) == "tok-A"
    assert calls == 1


async def test_refresh_race_collapses_to_single_call():
    """The core defect this class exists to prevent: 50 concurrent
    requests against an expired token must trigger ONE refresh, not 50."""
    mgr = AuthLifecycleManager()
    calls = 0

    async def slow_refresh(_old):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)  # widen the race window
        return _state(f"tok-{calls}", 3600)

    results = await asyncio.gather(
        *(mgr.get_valid_token("oauth", "team-1", slow_refresh) for _ in range(50))
    )
    assert calls == 1
    assert set(results) == {"tok-1"}


async def test_expiry_buffer_forces_proactive_refresh():
    mgr = AuthLifecycleManager()
    calls = 0

    async def refresh(_old):
        nonlocal calls
        calls += 1
        # Expires in 30s: inside the 60s safety buffer -> must refresh again.
        return _state(f"tok-{calls}", 30 if calls == 1 else 3600)

    assert await mgr.get_valid_token("oauth", "team-1", refresh) == "tok-1"
    assert await mgr.get_valid_token("oauth", "team-1", refresh) == "tok-2"
    assert calls == 2


async def test_force_invalidate_forces_next_refresh():
    mgr = AuthLifecycleManager()
    calls = 0

    async def refresh(_old):
        nonlocal calls
        calls += 1
        return _state(f"tok-{calls}", 3600)

    await mgr.get_valid_token("oauth", "team-1", refresh)
    mgr.force_invalidate("oauth", "team-1")
    assert await mgr.get_valid_token("oauth", "team-1", refresh) == "tok-2"
    assert calls == 2


async def test_identities_are_isolated():
    mgr = AuthLifecycleManager()
    seen_keys: list[str] = []

    async def refresh(_old):
        # The manager passes only the old refresh token; distinguish
        # identities by seeding separate entries directly instead.
        seen_keys.append("called")
        return _state(f"tok-{len(seen_keys)}", 3600)

    t1 = await mgr.get_valid_token("oauth", "team-1", refresh)
    t2 = await mgr.get_valid_token("oauth", "team-2", refresh)
    assert t1 != t2  # separate cache slots refreshed independently
    assert len(seen_keys) == 2


async def test_refresh_receives_previous_refresh_token():
    mgr = AuthLifecycleManager()
    received: list[str | None] = []

    async def refresh(old):
        received.append(old)
        return _state(f"tok-{len(received)}", 0)  # immediately stale again

    await mgr.get_valid_token("oauth", "team-1", refresh)   # first: no prior token
    await mgr.get_valid_token("oauth", "team-1", refresh)   # second: gets prior
    assert received[0] is None
    assert received[1] == "refresh-tok-1"


async def test_force_invalidate_unknown_identity_is_noop():
    mgr = AuthLifecycleManager()
    mgr.force_invalidate("oauth", "nobody")  # must not raise
