"""
Tests for the SQLite-backed ledger store — checking that filtering,
counting, and round-tripping events works correctly, since every later
engine (Quota, Budget) depends on these queries being right.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from gateway.core.types import CallOutcome, ErrorCategory, utcnow
from gateway.ledger.events import GatewayEvent
from gateway.ledger.store import SqliteLedgerStore


@pytest.fixture
def store(tmp_path) -> SqliteLedgerStore:
    return SqliteLedgerStore(tmp_path / "test_ledger.db")


def make_event(**overrides) -> GatewayEvent:
    defaults = {
        "identity_key": "team-a",
        "provider": "anthropic",
        "operation": "messages.create",
        "outcome": CallOutcome.SUCCESS,
        "http_status": 200,
        "latency_ms": 123.4,
        "cost_usd": 0.001,
    }
    defaults.update(overrides)
    return GatewayEvent(**defaults)


@pytest.mark.asyncio
async def test_append_and_query_roundtrip(store: SqliteLedgerStore) -> None:
    event = make_event()
    await store.append(event)

    results = await store.query(identity_key="team-a")
    assert len(results) == 1
    assert results[0].event_id == event.event_id
    assert results[0].cost_usd == 0.001


@pytest.mark.asyncio
async def test_query_filters_by_provider(store: SqliteLedgerStore) -> None:
    await store.append(make_event(provider="anthropic"))
    await store.append(make_event(provider="openai"))

    anthropic_only = await store.query(provider="anthropic")
    assert len(anthropic_only) == 1
    assert anthropic_only[0].provider == "anthropic"


@pytest.mark.asyncio
async def test_query_filters_by_outcome(store: SqliteLedgerStore) -> None:
    await store.append(make_event(outcome=CallOutcome.SUCCESS))
    await store.append(
        make_event(
            outcome=CallOutcome.FAILURE,
            error_category=ErrorCategory.RATE_LIMITED,
            http_status=429,
            cost_usd=None,
        )
    )

    failures = await store.query(outcome=CallOutcome.FAILURE)
    assert len(failures) == 1
    assert failures[0].error_category == ErrorCategory.RATE_LIMITED


@pytest.mark.asyncio
async def test_query_filters_by_since(store: SqliteLedgerStore) -> None:
    old_event = make_event(timestamp=utcnow() - timedelta(hours=2))
    recent_event = make_event(timestamp=utcnow())
    await store.append(old_event)
    await store.append(recent_event)

    recent_only = await store.query(since=utcnow() - timedelta(hours=1))
    assert len(recent_only) == 1
    assert recent_only[0].event_id == recent_event.event_id


@pytest.mark.asyncio
async def test_count_since(store: SqliteLedgerStore) -> None:
    for _ in range(5):
        await store.append(make_event())

    count = await store.count_since(identity_key="team-a")
    assert count == 5


@pytest.mark.asyncio
async def test_raw_provider_metadata_roundtrips_as_dict(store: SqliteLedgerStore) -> None:
    event = make_event(raw_provider_metadata={"anthropic-ratelimit-requests-remaining": "42"})
    await store.append(event)

    results = await store.query(identity_key="team-a")
    assert results[0].raw_provider_metadata == {"anthropic-ratelimit-requests-remaining": "42"}


@pytest.mark.asyncio
async def test_query_respects_limit(store: SqliteLedgerStore) -> None:
    for _ in range(10):
        await store.append(make_event())

    results = await store.query(identity_key="team-a", limit=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_concurrent_appends_none_lost(store: SqliteLedgerStore) -> None:
    """
    Phase 2 requirement: many concurrent appends, none lost.

    append() now runs the actual sqlite3 call inside asyncio.to_thread,
    so many coroutines calling it "concurrently" really do dispatch to
    separate worker threads hitting the same SQLite file at once. This
    is exactly the scenario the WAL-mode + busy-timeout configuration
    (see store.py's CONCURRENCY MODEL docstring) exists to survive
    without raising "database is locked" or silently dropping a write.
    """
    N = 50
    events = [make_event(identity_key="team-concurrent") for _ in range(N)]

    await asyncio.gather(*(store.append(e) for e in events))

    count = await store.count_since(identity_key="team-concurrent")
    assert count == N

    stored_ids = {e.event_id for e in await store.query(identity_key="team-concurrent", limit=N)}
    expected_ids = {e.event_id for e in events}
    assert stored_ids == expected_ids, "some concurrently-appended events were lost or duplicated"


# ----------------------------------------------------------- Phase 3: quota_bucket


@pytest.mark.asyncio
async def test_quota_bucket_roundtrips(store: SqliteLedgerStore) -> None:
    await store.append(make_event(quota_bucket="model-a"))
    await store.append(make_event())  # no bucket
    by_bucket = {e.quota_bucket for e in await store.query()}
    assert by_bucket == {"model-a", None}


@pytest.mark.asyncio
async def test_count_since_filters_by_bucket_and_never_matches_unbucketed_rows(store: SqliteLedgerStore) -> None:
    await store.append(make_event(quota_bucket="model-a"))
    await store.append(make_event(quota_bucket="model-a"))
    await store.append(make_event(quota_bucket="model-b"))
    await store.append(make_event(quota_bucket=None))
    assert await store.count_since(provider="anthropic", quota_bucket="model-a") == 2
    assert await store.count_since(provider="anthropic", quota_bucket="model-b") == 1
    assert await store.count_since(provider="anthropic", quota_bucket="nope") == 0
    assert await store.count_since(provider="anthropic") == 4, "no bucket filter sees everything"


@pytest.mark.asyncio
async def test_bucket_filter_composes_with_provider_and_time(store: SqliteLedgerStore) -> None:
    now = utcnow()
    await store.append(make_event(quota_bucket="m", timestamp=now - timedelta(hours=2)))
    await store.append(make_event(quota_bucket="m", timestamp=now))
    await store.append(make_event(quota_bucket="m", provider="openai", timestamp=now))
    got = await store.count_since(provider="anthropic", quota_bucket="m", since=now - timedelta(hours=1))
    assert got == 1
