"""
Tests for CallExecutor.

Priority order matches the defects it fixes:
  1. A network exception (defect 2) must still produce exactly one
     ledger event, classified via the adapter's classify_exception.
  2. A normal success, an HTTP error, and a known-bad-pattern 200 must
     each produce the right CallOutcome (defect 1: this logic used to
     live only in the demo script and was untested as a library unit).
"""

from __future__ import annotations

import httpx
import pytest

from gateway.adapters.anthropic import AnthropicAdapter
from gateway.core.adapter import AdapterRequest
from gateway.core.executor import CallExecutor
from gateway.core.types import CallOutcome, ErrorCategory
from gateway.ledger.store import SqliteLedgerStore


def make_request() -> AdapterRequest:
    return AdapterRequest(
        operation="messages.create",
        payload={"model": "claude-x", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
    )


@pytest.fixture
def ledger(tmp_path) -> SqliteLedgerStore:
    return SqliteLedgerStore(tmp_path / "executor_test.db")


def adapter_with_transport(handler) -> AnthropicAdapter:
    client = httpx.AsyncClient(
        base_url="https://api.anthropic.com/v1", transport=httpx.MockTransport(handler)
    )
    return AnthropicAdapter(api_key="test-key-not-used", http_client=client)


@pytest.mark.asyncio
async def test_network_exception_still_produces_one_event(ledger: SqliteLedgerStore) -> None:
    """
    The core fix for defect 2: send() raising a transport exception must
    not vanish. Before this executor existed, nothing caught this and no
    event was ever written for the worst class of failure.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connection timed out", request=request)

    adapter = adapter_with_transport(handler)
    executor = CallExecutor(adapter, ledger, identity_key="team-x")

    result = await executor.execute(make_request())

    assert result.succeeded is False
    assert result.response is None
    assert result.exception is not None
    assert isinstance(result.exception, httpx.ConnectTimeout)
    assert result.event.outcome == CallOutcome.FAILURE
    assert result.event.error_category == ErrorCategory.TRANSIENT
    assert result.event.http_status is None
    assert result.event.raw_provider_metadata["exception_type"] == "ConnectTimeout"

    # And it's actually IN the ledger, not just in the returned result —
    # this is the part defect 2 says was missing entirely.
    stored = await ledger.query(identity_key="team-x")
    assert len(stored) == 1
    assert stored[0].event_id == result.event.event_id


@pytest.mark.asyncio
async def test_unrecognized_exception_type_still_produces_one_event(ledger: SqliteLedgerStore) -> None:
    """
    The executor's broad except is deliberate (see its docstring) — an
    exception type the adapter's classify_exception doesn't specifically
    recognize must still land in the ledger, honestly marked UNKNOWN,
    rather than propagating and vanishing (recreating defect 2 one level
    removed for "exception types nobody anticipated").
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("something nobody anticipated")

    adapter = adapter_with_transport(handler)
    executor = CallExecutor(adapter, ledger, identity_key="team-x")

    result = await executor.execute(make_request())

    assert result.event.outcome == CallOutcome.FAILURE
    assert result.event.error_category == ErrorCategory.UNKNOWN
    stored = await ledger.query(identity_key="team-x")
    assert len(stored) == 1


@pytest.mark.asyncio
async def test_success_response_produces_success_event_with_cost(ledger: SqliteLedgerStore) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "anthropic-ratelimit-requests-remaining": "49",
                "anthropic-ratelimit-requests-limit": "50",
            },
            json={
                "id": "msg_1",
                "type": "message",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    adapter = adapter_with_transport(handler)
    executor = CallExecutor(adapter, ledger, identity_key="team-x")

    result = await executor.execute(make_request())

    assert result.succeeded is True
    assert result.event.outcome == CallOutcome.SUCCESS
    assert result.event.http_status == 200
    # Phase 6: pricing.py now carries Anthropic entries.
    # 10 input * $3/M + 5 output * $15/M = $0.000030 + $0.000075 = $0.000105
    assert result.event.cost_usd == 0.000105


@pytest.mark.asyncio
async def test_http_error_produces_failure_event_with_no_cost(ledger: SqliteLedgerStore) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            529, json={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        )

    adapter = adapter_with_transport(handler)
    executor = CallExecutor(adapter, ledger, identity_key="team-x")

    result = await executor.execute(make_request())

    assert result.succeeded is False
    assert result.event.error_category == ErrorCategory.PROVIDER_OVERLOADED
    assert result.event.cost_usd is None


@pytest.mark.asyncio
async def test_known_bad_pattern_on_200_is_suspected_silent_failure(ledger: SqliteLedgerStore) -> None:
    """
    This is the whole reason CallOutcome.SUSPECTED_SILENT_FAILURE exists:
    a 200 whose body is actually an Anthropic error object must NOT be
    logged as SUCCESS just because the status code looked fine.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"type": "error", "error": {"type": "overloaded_error", "message": "mid-stream"}}
        )

    adapter = adapter_with_transport(handler)
    executor = CallExecutor(adapter, ledger, identity_key="team-x")

    result = await executor.execute(make_request())

    assert result.event.outcome == CallOutcome.SUSPECTED_SILENT_FAILURE
    assert result.succeeded is False
    assert "json_error_body_on_2xx" in result.event.raw_provider_metadata["bad_pattern_names"]


@pytest.mark.asyncio
async def test_provider_request_id_is_captured(ledger: SqliteLedgerStore) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"request-id": "req_abc123"},
            json={"type": "message", "model": "claude-x", "content": [], "usage": {"input_tokens": 1, "output_tokens": 1}},
        )

    adapter = adapter_with_transport(handler)
    executor = CallExecutor(adapter, ledger, identity_key="team-x")

    result = await executor.execute(make_request())
    assert result.event.provider_request_id == "req_abc123"


# ------------------------------------------------ Phase 3: what the executor now exposes


@pytest.mark.asyncio
async def test_success_exposes_the_parsed_snapshot_and_stamps_the_bucket(ledger: SqliteLedgerStore) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "anthropic-ratelimit-requests-limit": "50",
                "anthropic-ratelimit-requests-remaining": "49",
                "anthropic-ratelimit-tokens-limit": "100000",
                "anthropic-ratelimit-tokens-remaining": "99000",
            },
            json={"model": "claude-x", "usage": {"input_tokens": 1, "output_tokens": 1}},
        )

    result = await CallExecutor(adapter_with_transport(handler), ledger, identity_key="t").execute(make_request())
    assert result.rate_limit is not None and result.rate_limit.requests_remaining == 49
    assert result.classified_error is None, "a 2xx has nothing to classify"
    assert result.event.quota_bucket == "claude-x"
    assert (await ledger.query(identity_key="t"))[0].quota_bucket == "claude-x"


@pytest.mark.asyncio
async def test_http_error_exposes_snapshot_and_classified_error_the_ledger_cannot_keep(
    ledger: SqliteLedgerStore,
) -> None:
    """The ledger stores raw headers and the classification MESSAGE only; the
    parsed retry-after/reset/resume_at would otherwise be lost for good."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "9"},
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}},
        )

    result = await CallExecutor(adapter_with_transport(handler), ledger, identity_key="t").execute(make_request())
    assert result.classified_error is not None
    assert result.classified_error.category == ErrorCategory.RATE_LIMITED
    assert result.classified_error.retry_after_seconds == 9.0
    assert result.rate_limit is not None and result.rate_limit.retry_after_seconds == 9.0
    assert result.event.quota_bucket == "claude-x"


@pytest.mark.asyncio
async def test_transport_failure_has_a_classified_error_no_snapshot_and_still_a_bucket(
    ledger: SqliteLedgerStore,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    result = await CallExecutor(adapter_with_transport(handler), ledger, identity_key="t").execute(make_request())
    assert result.rate_limit is None, "no response, so nothing to parse"
    assert result.classified_error is not None and result.classified_error.category == ErrorCategory.TRANSIENT
    assert result.event.quota_bucket == "claude-x"


@pytest.mark.asyncio
async def test_a_malformed_request_still_reaches_the_ledger_with_a_fallback_bucket(
    ledger: SqliteLedgerStore,
) -> None:
    """quota_bucket() is called before send(); it must not turn a bad request
    into an exception that skips the ledger."""
    adapter = adapter_with_transport(lambda r: httpx.Response(200, json={}))
    bad = AdapterRequest(operation="messages.create", payload={})  # no model
    result = await CallExecutor(adapter, ledger, identity_key="t").execute(bad)
    assert result.event.quota_bucket == "unknown-model"
    assert len(await ledger.query(identity_key="t")) == 1
