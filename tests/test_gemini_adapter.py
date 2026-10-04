"""
Tests for the Gemini adapter. Heaviest focus: the undifferentiated
RESOURCE_EXHAUSTED case, since that's the single biggest way this
provider differs from Anthropic/OpenAI (see docs/provider-notes.md).
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfoNotFoundError

import httpx
import pytest

from gateway.adapters import gemini as gemini_module
from gateway.adapters.gemini import GeminiAdapter, _next_daily_reset
from gateway.core.adapter import AdapterRequest
from gateway.core.types import (
    ErrorCategory,
    LimitingUnit,
    QuotaDimension,
    SignalQuality,
    WindowKind,
)


@pytest.fixture
def adapter() -> GeminiAdapter:
    return GeminiAdapter(api_key="test-key-not-used")


def test_limiting_unit_is_project_not_key(adapter: GeminiAdapter) -> None:
    """VERIFIED verbatim from official docs: per project, not per key."""
    assert adapter.get_limiting_unit() == LimitingUnit.PROJECT


def test_send_requires_model_in_extra(adapter: GeminiAdapter) -> None:
    """Gemini's model selects the URL path, not a payload field -- unlike
    Anthropic/OpenAI. A request missing it must fail loudly, not silently
    hit a broken URL."""
    with pytest.raises(ValueError, match="model"):
        import asyncio

        asyncio.run(adapter.send(AdapterRequest(operation="generateContent", payload={})))


def test_send_accepts_model_in_payload_as_fallback(adapter: GeminiAdapter) -> None:
    """Canonical location is extra["model"], but gateway clients naturally
    put the model in the OpenAI-style payload. send() must accept that too
    instead of hard-failing (previously broke every pinned Gemini call)."""
    import asyncio

    resp = asyncio.run(
        adapter.send(
            AdapterRequest(
                operation="generateContent",
                payload={"contents": [], "model": "gemini-2.0-flash"},
            )
        )
    )
    assert resp.status_code == 200


def test_quota_bucket_falls_back_to_payload_model(adapter: GeminiAdapter) -> None:
    req = AdapterRequest(operation="generateContent", payload={"model": "m-payload"})
    assert adapter.quota_bucket(req) == "m-payload"


def test_rate_limit_headers_are_always_none_quality(adapter: GeminiAdapter) -> None:
    """The single most important Gemini-specific fact: generateContent
    documents NO rate-limit headers on any response, ever. This must
    never silently become PARTIAL or FULL."""
    success = httpx.Response(200, json={"candidates": []})
    failure = httpx.Response(429, json={"error": {}})

    assert adapter.parse_rate_limit_headers(success).signal_quality == SignalQuality.NONE
    assert adapter.parse_rate_limit_headers(failure).signal_quality == SignalQuality.NONE


def test_429_with_per_day_quota_id_is_quota_exhausted(adapter: GeminiAdapter) -> None:
    """The real body shape from docs/provider-notes.md's cited GitHub
    issues -- a QuotaFailure detail naming a daily quota."""
    response = httpx.Response(
        429,
        json={
            "error": {
                "code": 429,
                "message": "Quota exceeded.",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}
                        ],
                    }
                ],
            }
        },
    )
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.QUOTA_EXHAUSTED
    # Correctly flagged as a heuristic, per the UNVERIFIED-as-contract note.
    assert result.is_definitively_classified is False


def test_429_with_per_minute_quota_id_is_rate_limited(adapter: GeminiAdapter) -> None:
    response = httpx.Response(
        429,
        json={
            "error": {
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}
                        ],
                    },
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"},
                ],
            }
        },
    )
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.RATE_LIMITED
    assert result.retry_after_seconds == 34.0


def test_429_with_no_quota_id_defaults_to_rate_limited_not_quota(adapter: GeminiAdapter) -> None:
    """The honest, common case: no quotaId to disambiguate with. Must
    default to the less severe (RATE_LIMITED, retryable) classification
    per official guidance, rather than assuming the worse case, but must
    say it isn't sure."""
    response = httpx.Response(
        429, json={"error": {"code": 429, "message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}}
    )
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.RATE_LIMITED
    assert result.is_definitively_classified is False


def test_failed_precondition_400_is_permanent(adapter: GeminiAdapter) -> None:
    """Billing/region prerequisite -- not a malformed request, not fixed
    by retrying, but distinguishable from a real bad-request 400 via the
    gRPC `status` field."""
    response = httpx.Response(
        400,
        json={
            "error": {
                "code": 400,
                "message": "Gemini API free tier is not available in your country.",
                "status": "FAILED_PRECONDITION",
            }
        },
    )
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.PERMANENT


def test_plain_invalid_argument_400_is_also_permanent(adapter: GeminiAdapter) -> None:
    response = httpx.Response(
        400, json={"error": {"code": 400, "message": "bad field", "status": "INVALID_ARGUMENT"}}
    )
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.PERMANENT


def test_503_is_provider_overloaded(adapter: GeminiAdapter) -> None:
    response = httpx.Response(
        503, json={"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}
    )
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.PROVIDER_OVERLOADED


def test_500_is_transient_not_overloaded(adapter: GeminiAdapter) -> None:
    """500 and 503 are documented as different things -- 500 is a
    generic internal error, not specifically a capacity signal."""
    response = httpx.Response(500, json={"error": {"code": 500, "message": "internal", "status": "INTERNAL"}})
    result = adapter.classify_error(response)
    assert result.category == ErrorCategory.TRANSIENT


def test_extract_usage_reads_model_version_and_tokens(adapter: GeminiAdapter) -> None:
    response = httpx.Response(
        200,
        json={
            "candidates": [{"content": {"parts": [{"text": "hi"}]}}],
            "usageMetadata": {
                "promptTokenCount": 15,
                "candidatesTokenCount": 6,
                "totalTokenCount": 21,
                "cachedContentTokenCount": 0,
            },
            "modelVersion": "gemini-3.1-flash-lite",
        },
    )
    usage = adapter.extract_usage(response)
    assert usage is not None
    assert usage.model == "gemini-3.1-flash-lite"
    assert usage.input_tokens == 15
    assert usage.output_tokens == 6


def test_extract_request_id_is_honestly_none(adapter: GeminiAdapter) -> None:
    """No verified Gemini request-id field/header exists for
    generateContent -- must return None, not guess a header name."""
    response = httpx.Response(200, json={})
    assert adapter.extract_request_id(response) is None


def test_blocked_prompt_with_no_candidates_is_known_bad_pattern(adapter: GeminiAdapter) -> None:
    response = httpx.Response(
        200,
        json={
            "promptFeedback": {"blockReason": "SAFETY"},
        },
    )
    matches = adapter.check_known_bad_patterns(response)
    assert len(matches) == 1
    assert matches[0].pattern_name == "blocked_before_generation_no_candidates"


def test_blocked_feedback_with_candidates_present_is_not_flagged(adapter: GeminiAdapter) -> None:
    """If candidates ARE present, a blockReason on promptFeedback (e.g. a
    partial safety rating) is not the same failure -- must not
    over-trigger."""
    response = httpx.Response(
        200,
        json={
            "candidates": [{"content": {"parts": [{"text": "hi"}]}}],
            "promptFeedback": {"blockReason": None},
        },
    )
    assert adapter.check_known_bad_patterns(response) == []


def _quota_429(quota_id: str | None) -> httpx.Response:
    details = []
    if quota_id:
        details.append(
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaId": quota_id}],
            }
        )
    return httpx.Response(
        429, json={"error": {"code": 429, "message": "x", "status": "RESOURCE_EXHAUSTED", "details": details}}
    )


class TestDailyQuotaResumeAt:
    """Phase 3 fix: the PerDay branch's message has always claimed the VERIFIED
    midnight-Pacific reset but left resume_at None, discarding known data."""

    def test_daily_quota_reports_the_next_midnight_pacific(self, adapter: GeminiAdapter, monkeypatch) -> None:
        monkeypatch.setattr(gemini_module, "utcnow", lambda: datetime(2026, 9, 30, 12, 0, tzinfo=UTC))
        result = adapter.classify_error(_quota_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
        assert result.category == ErrorCategory.QUOTA_EXHAUSTED
        assert result.resume_at == datetime(2026, 10, 1, 7, 0, tzinfo=UTC)  # 00:00 PDT

    def test_the_daily_diagnosis_is_still_a_heuristic_so_still_not_definitive(self, adapter: GeminiAdapter) -> None:
        """A derived resume time must not launder an UNVERIFIED classification."""
        result = adapter.classify_error(_quota_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
        assert result.is_definitively_classified is False

    def test_per_minute_and_unlabelled_429s_state_no_resume_time(self, adapter: GeminiAdapter) -> None:
        assert adapter.classify_error(_quota_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")).resume_at is None
        assert adapter.classify_error(_quota_429(None)).resume_at is None

    @pytest.mark.parametrize(
        ("now", "expected"),
        [
            # PST (UTC-8) -> PDT (UTC-7) on 2026-03-08: the day after ends at 00:00 PDT.
            (datetime(2026, 3, 7, 12, 0, tzinfo=UTC), datetime(2026, 3, 8, 8, 0, tzinfo=UTC)),
            (datetime(2026, 3, 8, 12, 0, tzinfo=UTC), datetime(2026, 3, 9, 7, 0, tzinfo=UTC)),
            # PDT -> PST on 2026-11-01.
            (datetime(2026, 11, 1, 12, 0, tzinfo=UTC), datetime(2026, 11, 2, 8, 0, tzinfo=UTC)),
            # Exactly at local midnight, the NEXT midnight is a full day away.
            (datetime(2026, 9, 30, 7, 0, tzinfo=UTC), datetime(2026, 10, 1, 7, 0, tzinfo=UTC)),
        ],
    )
    def test_next_daily_reset_is_correct_across_dst(self, now: datetime, expected: datetime) -> None:
        assert _next_daily_reset(now) == expected

    def test_missing_tz_database_degrades_to_unknown_instead_of_raising(
        self, adapter: GeminiAdapter, monkeypatch
    ) -> None:
        """classify_error is contractually never-raises; slim images lack tzdata."""

        def boom(_name: str):
            raise ZoneInfoNotFoundError("no tz database")

        monkeypatch.setattr(gemini_module, "ZoneInfo", boom)
        result = adapter.classify_error(_quota_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
        assert result.category == ErrorCategory.QUOTA_EXHAUSTED
        assert result.resume_at is None


class TestQuotaWindowsAndBucket:
    def test_declares_exactly_the_verified_windows(self, adapter: GeminiAdapter) -> None:
        windows = {w.name: w for w in adapter.quota_windows()}
        assert set(windows) == {"rpm", "tpm", "rpd", "spend_10m"}
        assert windows["rpm"].dimension == QuotaDimension.REQUESTS and windows["rpm"].seconds == 60
        assert windows["tpm"].dimension == QuotaDimension.TOKENS
        assert windows["rpd"].kind == WindowKind.CALENDAR_DAY and windows["rpd"].tz == "America/Los_Angeles"
        assert windows["spend_10m"].dimension == QuotaDimension.SPEND and windows["spend_10m"].seconds == 600

    def test_bucket_is_the_model_from_extra(self, adapter: GeminiAdapter) -> None:
        a = AdapterRequest(operation="generateContent", payload={}, extra={"model": "model-a"})
        b = AdapterRequest(operation="generateContent", payload={}, extra={"model": "model-b"})
        assert adapter.quota_bucket(a) == "model-a"
        assert adapter.quota_bucket(a) != adapter.quota_bucket(b)

    @pytest.mark.parametrize("extra", [{}, {"model": ""}, {"model": None}, {"model": 3}])
    def test_missing_model_falls_back_because_send_not_bucket_reports_it(self, adapter: GeminiAdapter, extra) -> None:
        req = AdapterRequest(operation="generateContent", payload={}, extra=extra)
        assert adapter.quota_bucket(req) == "unknown-model"
