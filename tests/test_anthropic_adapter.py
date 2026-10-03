"""
Tests for the Anthropic adapter — focused on the parts that are easy to
get subtly wrong and expensive to get wrong silently: error
classification and rate-limit header parsing.

These don't hit the real network. Each test builds an httpx.Response
directly with the exact headers/body Anthropic is documented to send,
and checks the adapter's classification against it.
"""

from __future__ import annotations

import httpx
import pytest

from gateway.adapters.anthropic import AnthropicAdapter
from gateway.core.adapter import AdapterRequest
from gateway.core.types import ErrorCategory, LimitingUnit


@pytest.fixture
def adapter() -> AnthropicAdapter:
    return AnthropicAdapter(api_key="test-key-not-used")


def test_limiting_unit_is_organization(adapter: AnthropicAdapter) -> None:
    """
    This is the single most consequential fact this adapter encodes:
    get it wrong and the Quota Engine will incorrectly tell users that
    adding more API keys will increase their throughput.
    """
    assert adapter.get_limiting_unit() == LimitingUnit.ORGANIZATION


def test_529_is_provider_overloaded_not_rate_limited(adapter: AnthropicAdapter) -> None:
    """
    The core distinction this whole project exists to make: a 529 is
    NOT the same failure as a 429, even though both are "the request
    didn't succeed." Misclassifying this would cause the Routing Engine
    (Phase 5) to apply the wrong backoff strategy and miss the failover
    signal.
    """
    response = httpx.Response(
        529,
        json={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    )
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.PROVIDER_OVERLOADED
    assert result.is_definitively_classified


def test_429_with_retry_after_is_rate_limited(adapter: AnthropicAdapter) -> None:
    """A normal per-minute/token rate limit: retryable after the stated interval."""
    response = httpx.Response(
        429,
        headers={"retry-after": "12"},
        json={"type": "error", "error": {"type": "rate_limit_error", "message": "..."}},
    )
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.RATE_LIMITED
    assert result.retry_after_seconds == 12.0


def test_429_spend_cap_is_quota_exhausted_with_resume_time(adapter: AnthropicAdapter) -> None:
    """
    Defect 6. The documented signal is error.details.error_code ==
    "enforced_spend_limit_reached" (NOT error.code, which Phase 1 read),
    the message states when access resumes, and the category is
    QUOTA_EXHAUSTED (not PERMANENT: it clears at a known future time).
    """
    response = httpx.Response(
        429,
        json={
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "You have reached your API usage limits: your organization has "
                "crossed its monthly API usage threshold. You will regain access on "
                "2026-10-01 at 00:00 UTC.",
                "details": {"error_code": "enforced_spend_limit_reached"},
            },
        },
    )
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.QUOTA_EXHAUSTED
    assert result.is_definitively_classified is True
    assert result.resume_at is not None
    assert (result.resume_at.year, result.resume_at.month, result.resume_at.day) == (2026, 10, 1)


def test_429_no_retry_after_no_code_is_heuristic_quota(adapter: AnthropicAdapter) -> None:
    """A 429 with neither retry-after nor the documented code is only a
    *probable* spend cap. It must say so (is_definitively_classified=False)."""
    response = httpx.Response(429, json={"type": "error", "error": {"type": "rate_limit_error"}})
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.QUOTA_EXHAUSTED
    assert result.is_definitively_classified is False


def test_429_with_retry_after_stays_rate_limited_even_if_workspace_limit(adapter: AnthropicAdapter) -> None:
    """Regression guard: docs say a Claude Code workspace limit is a 429
    WITH retry-after. The Phase 1 'no retry-after => cap' heuristic was
    only safe because it keyed on absence; presence must stay RATE_LIMITED."""
    response = httpx.Response(
        429,
        headers={"retry-after": "30"},
        json={"type": "error", "error": {"type": "rate_limit_error", "message": "workspace"}},
    )
    assert adapter.classify_error(response).category == ErrorCategory.RATE_LIMITED


def test_user_set_spend_limit_400_is_quota_not_permanent(adapter: AnthropicAdapter) -> None:
    """Found while researching defect 6 (not in the brief): a user-configured
    spend limit is documented as HTTP 400 invalid_request_error. Phase 1
    classified every 400 as PERMANENT, losing the fact that it resumes."""
    response = httpx.Response(
        400,
        json={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "You have reached your specified API usage limits. You will "
                "regain access on 2026-10-01 at 00:00 UTC.",
            },
        },
    )
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.QUOTA_EXHAUSTED
    assert result.resume_at is not None


def test_plain_400_is_still_permanent(adapter: AnthropicAdapter) -> None:
    response = httpx.Response(
        400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens required"}}
    )
    assert adapter.classify_error(response).category == ErrorCategory.PERMANENT


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413])
def test_client_errors_are_permanent(adapter: AnthropicAdapter, status: int) -> None:
    response = httpx.Response(status, json={"type": "error", "error": {"type": "x", "message": "x"}})
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.PERMANENT


@pytest.mark.parametrize("status", [500, 504])
def test_server_errors_are_transient(adapter: AnthropicAdapter, status: int) -> None:
    response = httpx.Response(status, json={"type": "error", "error": {"type": "x", "message": "x"}})
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.TRANSIENT


def test_unrecognized_status_is_honestly_unknown(adapter: AnthropicAdapter) -> None:
    """
    An unmapped status code should come back as UNKNOWN with
    is_definitively_classified=False — NOT silently guessed as one of
    the known categories. Silent wrong guesses are worse than an honest
    "I don't know."
    """
    response = httpx.Response(418, json={})  # I'm a teapot — not a real Anthropic status
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.UNKNOWN
    assert result.is_definitively_classified is False


def test_parse_rate_limit_headers_normalizes_correctly(adapter: AnthropicAdapter) -> None:
    response = httpx.Response(
        200,
        headers={
            "anthropic-ratelimit-requests-limit": "50",
            "anthropic-ratelimit-requests-remaining": "42",
            "anthropic-ratelimit-requests-reset": "2026-09-28T12:00:00Z",
            "anthropic-ratelimit-tokens-limit": "100000",
            "anthropic-ratelimit-tokens-remaining": "87654",
        },
        json={},
    )
    snapshot = adapter.parse_rate_limit_headers(response)
    assert snapshot.requests_remaining == 42
    assert snapshot.requests_limit == 50
    assert snapshot.tokens_remaining == 87654
    assert snapshot.limiting_unit == LimitingUnit.ORGANIZATION
    assert snapshot.reset_at is not None


def test_parse_rate_limit_headers_handles_missing_headers_gracefully(
    adapter: AnthropicAdapter,
) -> None:
    """A 529 carries no rate-limit headers at all — parsing must return
    None fields, not crash."""
    response = httpx.Response(529, json={})
    snapshot = adapter.parse_rate_limit_headers(response)
    assert snapshot.requests_remaining is None
    assert snapshot.tokens_remaining is None


def test_known_bad_pattern_detects_error_shaped_200(adapter: AnthropicAdapter) -> None:
    """
    A 200 status whose body is actually an Anthropic error object
    (e.g. a mid-stream overload) must be flagged — this is the whole
    reason check_known_bad_patterns exists.
    """
    response = httpx.Response(
        200,
        json={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    )
    matches = adapter.check_known_bad_patterns(response)
    assert len(matches) == 1
    assert matches[0].pattern_name == "json_error_body_on_2xx"


def test_known_bad_pattern_ignores_genuine_success(adapter: AnthropicAdapter) -> None:
    response = httpx.Response(
        200,
        json={"type": "message", "role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    )
    matches = adapter.check_known_bad_patterns(response)
    assert matches == []


# ------------------------------------------------------------------ defect 3

SSE_WITH_MIDSTREAM_ERROR = (
    "event: message_start\n"
    'data: {"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-4-5",'
    '"usage":{"input_tokens":25,"output_tokens":1}}}\n\n'
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n'
    "event: error\n"
    'data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n'
)


def test_sse_midstream_error_after_200_is_detected(adapter: AnthropicAdapter) -> None:
    """
    DEFECT 3 regression. Phase 1's check called _safe_json, which returns
    None on an SSE body, so a stream that died mid-flight returned [] (all
    clear) - the exact response the check exists to catch.
    """
    response = httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=SSE_WITH_MIDSTREAM_ERROR.encode())
    matches = adapter.check_known_bad_patterns(response)
    assert [m.pattern_name for m in matches] == ["streaming_error_after_200"]
    assert "overloaded_error" in matches[0].description


def test_clean_sse_stream_is_not_flagged(adapter: AnthropicAdapter) -> None:
    clean = SSE_WITH_MIDSTREAM_ERROR.split("event: error")[0]
    response = httpx.Response(200, headers={"content-type": "text/event-stream"}, content=clean.encode())
    assert adapter.check_known_bad_patterns(response) == []


def test_usage_extractable_from_sse_stream(adapter: AnthropicAdapter) -> None:
    clean = SSE_WITH_MIDSTREAM_ERROR.split("event: error")[0]
    response = httpx.Response(200, content=clean.encode())
    usage = adapter.extract_usage(response)
    assert usage is not None and usage.model == "claude-sonnet-4-5" and usage.input_tokens == 25


# ------------------------------------------------------------------ defect 2 / 4

def test_transport_exception_classified_transient(adapter: AnthropicAdapter) -> None:
    from gateway.core.adapter import AdapterRequest
    req = AdapterRequest(operation="messages.create", payload={})
    result = adapter.classify_exception(httpx.ConnectTimeout("t"), req)
    assert result.category == ErrorCategory.TRANSIENT


def test_reset_at_is_absolute_utc(adapter: AnthropicAdapter) -> None:
    """DEFECT 4: reset_at must be tz-aware UTC regardless of provider format."""
    response = httpx.Response(200, headers={
        "anthropic-ratelimit-requests-reset": "2026-09-28T12:00:00+05:30",
        "anthropic-ratelimit-requests-limit": "50", "anthropic-ratelimit-requests-remaining": "1",
        "anthropic-ratelimit-tokens-limit": "9", "anthropic-ratelimit-tokens-remaining": "1",
    }, json={})
    snap = adapter.parse_rate_limit_headers(response)
    assert snap.reset_at.utcoffset().total_seconds() == 0
    assert snap.reset_at.hour == 6 and snap.reset_at.minute == 30
    assert snap.reset_at_source == "timestamp"
    assert snap.signal_quality.value == "full"


class TestQuotaBucket:
    """VERIFIED: Anthropic measures limits per model class. Class membership
    is not documented as a stable list, so the model string is used as a
    bucket finer than or equal to the real one."""

    def test_bucket_is_the_request_model(self) -> None:
        adapter = AnthropicAdapter(api_key="test-key-not-used")
        a = AdapterRequest(operation="messages.create", payload={"model": "model-a"})
        b = AdapterRequest(operation="messages.create", payload={"model": "model-b"})
        assert adapter.quota_bucket(a) == "model-a"
        assert adapter.quota_bucket(a) != adapter.quota_bucket(b)

    @pytest.mark.parametrize("payload", [{}, {"model": None}, {"model": ""}, {"model": 7}])
    def test_unusable_model_falls_back_instead_of_raising(self, payload) -> None:
        adapter = AnthropicAdapter(api_key="test-key-not-used")
        req = AdapterRequest(operation="messages.create", payload=payload)
        assert adapter.quota_bucket(req) == "unknown-model"

    def test_no_quota_windows_declared_because_headers_carry_the_limits(self) -> None:
        assert AnthropicAdapter(api_key="test-key-not-used").quota_windows() == ()
