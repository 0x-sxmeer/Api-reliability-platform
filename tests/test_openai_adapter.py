"""
Tests for the OpenAI adapter. Mirrors test_anthropic_adapter.py's shape:
focused on classification correctness, since that's where a subtle
mistake is expensive and silent.
"""

from __future__ import annotations

from typing import ClassVar

import httpx
import pytest

from gateway.adapters.openai import OpenAIAdapter, _parse_openai_duration
from gateway.core.adapter import AdapterRequest
from gateway.core.types import ErrorCategory, LimitingUnit, SignalQuality


@pytest.fixture
def adapter() -> OpenAIAdapter:
    return OpenAIAdapter(api_key="test-key-not-used")


def test_limiting_unit_is_organization(adapter: OpenAIAdapter) -> None:
    assert adapter.get_limiting_unit() == LimitingUnit.ORGANIZATION


class TestDurationParsing:
    """OpenAI's Go-style duration reset headers -- the defect-4 case."""

    def test_seconds_only(self) -> None:
        assert _parse_openai_duration("1s") == pytest.approx(1.0)

    def test_minutes_and_seconds(self) -> None:
        assert _parse_openai_duration("6m0s") == pytest.approx(360.0)

    def test_fractional_seconds(self) -> None:
        assert _parse_openai_duration("59.5s") == pytest.approx(59.5)

    def test_garbage_returns_none(self) -> None:
        assert _parse_openai_duration("not-a-duration") is None

    def test_empty_returns_none(self) -> None:
        assert _parse_openai_duration("") is None


def test_503_is_provider_overloaded(adapter: OpenAIAdapter) -> None:
    response = httpx.Response(
        503,
        json={"error": {"type": "service_unavailable_error", "code": "server_is_overloaded", "message": "x"}},
    )
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.PROVIDER_OVERLOADED


def test_429_insufficient_quota_is_quota_exhausted(adapter: OpenAIAdapter) -> None:
    """The defect-6-equivalent case for OpenAI: distinct `code` from an
    ordinary rate limit, same 429 status."""
    response = httpx.Response(
        429,
        json={
            "error": {
                "message": "You exceeded your current quota, please check your plan and billing details.",
                "type": "insufficient_quota",
                "code": "insufficient_quota",
            }
        },
    )
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.QUOTA_EXHAUSTED
    assert result.is_definitively_classified is True


def test_429_organization_usage_limit_exceeded_is_quota_exhausted(adapter: OpenAIAdapter) -> None:
    response = httpx.Response(
        429,
        json={"error": {"message": "monthly usage limit", "code": "organization_usage_limit_exceeded"}},
    )
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.QUOTA_EXHAUSTED


def test_429_rate_limit_exceeded_is_rate_limited_not_quota(adapter: OpenAIAdapter) -> None:
    """The critical discrimination: this must NOT be QUOTA_EXHAUSTED even
    though it's the same HTTP status as the insufficient_quota case."""
    response = httpx.Response(
        429,
        headers={"retry-after": "5"},
        json={"error": {"message": "Rate limit exceeded", "type": "requests", "code": "rate_limit_exceeded"}},
    )
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.RATE_LIMITED
    assert result.retry_after_seconds == 5.0
    assert result.is_definitively_classified is True


def test_429_slow_down_migration_code_is_rate_limited(adapter: OpenAIAdapter) -> None:
    """The 2026 migration case: rapid-traffic-increase throttling moved
    from 503/slow_down to 429/slow_down on some endpoints. Must still be
    classified as an ordinary (retryable) rate limit, not overload."""
    response = httpx.Response(429, json={"error": {"message": "Slow down", "code": "slow_down"}})
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.RATE_LIMITED


def test_429_with_no_code_is_heuristic_rate_limit(adapter: OpenAIAdapter) -> None:
    response = httpx.Response(429, json={"error": {"message": "Too many requests"}})
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.RATE_LIMITED
    assert result.is_definitively_classified is False


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 416])
def test_client_errors_are_permanent(adapter: OpenAIAdapter, status: int) -> None:
    response = httpx.Response(status, json={"error": {"type": "x", "message": "x"}})
    assert adapter.classify_error(response).category == ErrorCategory.PERMANENT


@pytest.mark.parametrize("status", [500, 502, 504])
def test_server_errors_are_transient(adapter: OpenAIAdapter, status: int) -> None:
    response = httpx.Response(status, json={"error": {"type": "x", "message": "x"}})
    assert adapter.classify_error(response).category == ErrorCategory.TRANSIENT


def test_parse_rate_limit_headers_includes_project_scoped_in_raw(adapter: OpenAIAdapter) -> None:
    """Project-scoped headers are ALSO promoted to additional_scopes (Phase
    3; see TestProjectScope below), but must remain visible in raw_headers
    too: raw_headers is the audit copy, and losing it would hide what the
    parser was given."""
    response = httpx.Response(
        200,
        headers={
            "x-ratelimit-remaining-requests": "100",
            "x-ratelimit-limit-requests": "200",
            "x-ratelimit-remaining-tokens": "5000",
            "x-ratelimit-limit-tokens": "10000",
            "x-ratelimit-reset-tokens": "6m0s",
            "x-ratelimit-remaining-project-tokens": "57000",
            "x-ratelimit-reset-project-tokens": "3s",
        },
        json={},
    )
    snapshot = adapter.parse_rate_limit_headers(response)
    assert snapshot.requests_remaining == 100
    assert snapshot.tokens_remaining == 5000
    assert snapshot.signal_quality == SignalQuality.FULL
    assert snapshot.reset_at is not None
    assert snapshot.reset_at_source == "duration"
    # project-scoped fields preserved, even though not first-class:
    assert snapshot.raw_headers["x-ratelimit-remaining-project-tokens"] == "57000"


def test_parse_rate_limit_headers_empty_is_none_quality(adapter: OpenAIAdapter) -> None:
    response = httpx.Response(503, json={})
    snapshot = adapter.parse_rate_limit_headers(response)
    assert snapshot.signal_quality == SignalQuality.NONE
    assert snapshot.requests_remaining is None


def test_extract_usage_reads_cached_tokens_from_nested_details(adapter: OpenAIAdapter) -> None:
    response = httpx.Response(
        200,
        json={
            "model": "gpt-6-sol",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 40},
            },
        },
    )
    usage = adapter.extract_usage(response)
    assert usage is not None
    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.cached_input_tokens == 40


def test_extract_request_id(adapter: OpenAIAdapter) -> None:
    response = httpx.Response(200, headers={"x-request-id": "req_xyz"}, json={})
    assert adapter.extract_request_id(response) == "req_xyz"


def test_known_bad_patterns_is_honestly_empty(adapter: OpenAIAdapter) -> None:
    """No verified OpenAI-specific silent-failure pattern exists yet --
    must return [] rather than a fabricated heuristic."""
    response = httpx.Response(200, json={"model": "gpt-6-sol", "choices": []})
    assert adapter.check_known_bad_patterns(response) == []


class TestProjectScope:
    """Phase 3: OpenAI enforces org AND project limits at once. The project
    headers (VERIFIED: limit, remaining, reset -- tokens only) ride along in
    RateLimitSnapshot.additional_scopes so the Quota Engine can take the
    tightest scope."""

    FULL: ClassVar[dict[str, str]] = {
        "x-ratelimit-remaining-requests": "100",
        "x-ratelimit-limit-requests": "200",
        "x-ratelimit-remaining-tokens": "5000",
        "x-ratelimit-limit-tokens": "10000",
        "x-ratelimit-reset-tokens": "6s",
        "x-ratelimit-limit-project-tokens": "60000",
        "x-ratelimit-remaining-project-tokens": "57000",
        "x-ratelimit-reset-project-tokens": "6s",
    }

    def test_three_documented_headers_become_a_full_project_scope(self, adapter: OpenAIAdapter) -> None:
        snap = adapter.parse_rate_limit_headers(httpx.Response(200, headers=self.FULL, json={}))
        assert len(snap.additional_scopes) == 1
        proj = snap.additional_scopes[0]
        assert proj.limiting_unit == LimitingUnit.PROJECT
        assert (proj.tokens_remaining, proj.tokens_limit) == (57000, 60000)
        assert proj.signal_quality == SignalQuality.FULL
        assert proj.reset_at is not None and proj.reset_at_source == "duration"
        # Tokens only: no project-scoped request headers are documented.
        assert proj.requests_remaining is None and proj.requests_limit is None

    def test_primary_scope_is_untouched_by_the_additional_one(self, adapter: OpenAIAdapter) -> None:
        snap = adapter.parse_rate_limit_headers(httpx.Response(200, headers=self.FULL, json={}))
        assert snap.limiting_unit == LimitingUnit.ORGANIZATION
        assert (snap.requests_remaining, snap.tokens_remaining) == (100, 5000)
        assert snap.additional_scopes[0].additional_scopes == (), "does not nest"

    def test_both_scopes_share_one_receipt_time(self, adapter: OpenAIAdapter) -> None:
        """Equal durations must give equal reset_at, i.e. one clock reading."""
        snap = adapter.parse_rate_limit_headers(httpx.Response(200, headers=self.FULL, json={}))
        assert snap.reset_at == snap.additional_scopes[0].reset_at

    def test_remaining_without_limit_is_partial_not_full(self, adapter: OpenAIAdapter) -> None:
        """The pre-Phase-3 notes believed only remaining+reset existed; be
        correct for a response that has just those."""
        headers = {k: v for k, v in self.FULL.items() if k != "x-ratelimit-limit-project-tokens"}
        proj = adapter.parse_rate_limit_headers(httpx.Response(200, headers=headers, json={})).additional_scopes[0]
        assert proj.signal_quality == SignalQuality.PARTIAL
        assert proj.tokens_limit is None and proj.tokens_remaining == 57000

    def test_no_project_headers_means_no_additional_scope(self, adapter: OpenAIAdapter) -> None:
        headers = {k: v for k, v in self.FULL.items() if "project" not in k}
        assert adapter.parse_rate_limit_headers(httpx.Response(200, headers=headers, json={})).additional_scopes == ()

    def test_garbage_project_headers_do_not_raise_or_invent_a_scope(self, adapter: OpenAIAdapter) -> None:
        headers = {
            "x-ratelimit-limit-project-tokens": "lots",
            "x-ratelimit-remaining-project-tokens": "some",
            "x-ratelimit-reset-project-tokens": "soon",
        }
        assert adapter.parse_rate_limit_headers(httpx.Response(200, headers=headers, json={})).additional_scopes == ()


class TestQuotaBucket:
    def test_bucket_is_the_request_model(self, adapter: OpenAIAdapter) -> None:
        a = AdapterRequest(operation="chat.completions.create", payload={"model": "model-a"})
        b = AdapterRequest(operation="chat.completions.create", payload={"model": "model-b"})
        assert adapter.quota_bucket(a) == "model-a"
        assert adapter.quota_bucket(a) != adapter.quota_bucket(b)

    @pytest.mark.parametrize("payload", [{}, {"model": None}, {"model": ""}, {"model": 7}])
    def test_unusable_model_falls_back_instead_of_raising(self, adapter: OpenAIAdapter, payload) -> None:
        req = AdapterRequest(operation="chat.completions.create", payload=payload)
        assert adapter.quota_bucket(req) == "unknown-model"

    def test_no_quota_windows_declared_because_headers_carry_the_limits(self, adapter: OpenAIAdapter) -> None:
        assert adapter.quota_windows() == ()
