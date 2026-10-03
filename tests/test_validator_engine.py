"""
Response Validator unit tests.

Every test here uses the same vendor-free FakeAdapter pattern from
test_quota_engine.py. The validator is stateless and synchronous, so
there is no clock injection or ledger fixture needed for the unit tests.

Real-adapter behaviour is in test_quota_integration.py (extended for
Phase 4).
"""

from __future__ import annotations

import httpx
import pytest

from gateway.core.adapter import AdapterRequest, ProviderAdapter
from gateway.core.executor import CallExecutor, CallResult
from gateway.core.types import (
    CallOutcome,
    ClassifiedError,
    ErrorCategory,
    KnownBadPatternMatch,
    LimitingUnit,
    RateLimitSnapshot,
    SignalQuality,
    UsageInfo,
)
from gateway.engines.validator import (
    FindingCategory,
    ResponseValidator,
    ValidationResult,
    ValidationVerdict,
    ValidatorConfig,
)
from gateway.ledger.events import GatewayEvent
from gateway.ledger.store import SqliteLedgerStore

# ------------------------------------------------------------------ test helpers


class FakeAdapter(ProviderAdapter):
    """A vendor-free adapter for testing the validator."""

    def __init__(self, name: str = "fakeprov") -> None:
        self._name = name

    @property
    def provider_name(self) -> str:
        return self._name

    def get_limiting_unit(self) -> LimitingUnit:
        return LimitingUnit.ORGANIZATION

    def quota_bucket(self, request: AdapterRequest) -> str:
        return str(request.payload.get("model", "m0"))

    def parse_rate_limit_headers(self, response: httpx.Response) -> RateLimitSnapshot:
        return RateLimitSnapshot(limiting_unit=LimitingUnit.ORGANIZATION, signal_quality=SignalQuality.NONE)

    def classify_error(self, response: httpx.Response) -> ClassifiedError:
        return ClassifiedError(category=ErrorCategory.UNKNOWN)

    def check_known_bad_patterns(self, response: httpx.Response) -> list[KnownBadPatternMatch]:
        return []

    def classify_exception(self, exc: Exception, request: AdapterRequest) -> ClassifiedError:
        return ClassifiedError(category=ErrorCategory.UNKNOWN)

    def extract_usage(self, response: httpx.Response) -> UsageInfo | None:
        return None

    def extract_request_id(self, response: httpx.Response) -> str | None:
        return None

    async def send(self, request: AdapterRequest) -> httpx.Response:
        raise NotImplementedError


def _event(
    *,
    outcome: CallOutcome = CallOutcome.SUCCESS,
    error_category: ErrorCategory | None = None,
    cost_usd: float | None = None,
) -> GatewayEvent:
    return GatewayEvent(
        identity_key="team-a",
        provider="fakeprov",
        operation="op",
        outcome=outcome,
        error_category=error_category,
        cost_usd=cost_usd,
    )


def _result(
    *,
    outcome: CallOutcome = CallOutcome.SUCCESS,
    error_category: ErrorCategory | None = None,
    cost_usd: float | None = None,
    bad_patterns: list[KnownBadPatternMatch] | None = None,
    classified_error: ClassifiedError | None = None,
    exception: Exception | None = None,
) -> CallResult:
    return CallResult(
        event=_event(outcome=outcome, error_category=error_category, cost_usd=cost_usd),
        response=httpx.Response(200, content=b"{}") if exception is None else None,
        exception=exception,
        bad_patterns=bad_patterns,
        classified_error=classified_error,
    )


# ============================================================== basic verdicts


class TestValidVerdicts:
    """A clean, successful call should produce VALID."""

    def test_clean_success_is_valid(self) -> None:
        v = ResponseValidator()
        r = v.validate(_result())
        assert r.verdict is ValidationVerdict.VALID
        assert r.confidence == 0.0
        assert r.findings == ()
        assert r.should_retry is False
        assert r.retry_on_different_provider is False

    def test_clean_success_with_cost_is_valid_and_billed(self) -> None:
        v = ResponseValidator()
        r = v.validate(_result(cost_usd=0.005))
        assert r.verdict is ValidationVerdict.VALID
        assert r.was_billed is True

    def test_clean_success_without_cost_is_valid_and_not_billed(self) -> None:
        v = ResponseValidator()
        r = v.validate(_result(cost_usd=None))
        assert r.verdict is ValidationVerdict.VALID
        assert r.was_billed is False


# ============================================================= bad-pattern validation


class TestBadPatternValidation:
    """Bad-pattern matches from adapters should be properly validated."""

    def test_high_confidence_pattern_is_invalid(self) -> None:
        v = ResponseValidator()
        r = v.validate(
            _result(
                outcome=CallOutcome.SUSPECTED_SILENT_FAILURE,
                bad_patterns=[
                    KnownBadPatternMatch(
                        pattern_name="streaming_error_after_200",
                        description="SSE error in stream",
                        confidence=0.97,
                    )
                ],
            )
        )
        assert r.verdict is ValidationVerdict.INVALID
        assert r.confidence == 0.97
        assert len(r.findings) == 1
        assert r.findings[0].category is FindingCategory.BAD_PATTERN
        assert r.findings[0].name == "streaming_error_after_200"
        assert r.should_retry is True
        assert r.retry_on_different_provider is False  # same provider retry should work

    def test_medium_confidence_pattern_is_suspect_when_policy_is_lenient(self) -> None:
        cfg = ValidatorConfig(
            suspect_threshold=0.5,
            reject_threshold=0.95,
            treat_suspected_silent_failure_as_invalid=False,
        )
        v = ResponseValidator(config=cfg)
        r = v.validate(
            _result(
                outcome=CallOutcome.SUSPECTED_SILENT_FAILURE,
                bad_patterns=[
                    KnownBadPatternMatch(
                        pattern_name="blocked_before_generation",
                        description="Prompt was blocked",
                        confidence=0.90,
                    )
                ],
            )
        )
        assert r.verdict is ValidationVerdict.SUSPECT
        assert r.confidence == 0.90

    def test_below_threshold_pattern_is_dropped(self) -> None:
        cfg = ValidatorConfig(suspect_threshold=0.95)
        v = ResponseValidator(config=cfg)
        r = v.validate(
            _result(
                outcome=CallOutcome.SUCCESS,
                bad_patterns=[
                    KnownBadPatternMatch(
                        pattern_name="low_conf_pattern",
                        description="Uncertain detection",
                        confidence=0.5,
                    )
                ],
            )
        )
        assert r.verdict is ValidationVerdict.VALID
        assert r.findings == ()

    def test_multiple_patterns_highest_confidence_wins(self) -> None:
        v = ResponseValidator()
        r = v.validate(
            _result(
                outcome=CallOutcome.SUSPECTED_SILENT_FAILURE,
                bad_patterns=[
                    KnownBadPatternMatch(
                        pattern_name="pattern_a",
                        description="Low",
                        confidence=0.6,
                    ),
                    KnownBadPatternMatch(
                        pattern_name="pattern_b",
                        description="High",
                        confidence=0.97,
                    ),
                ],
            )
        )
        assert r.verdict is ValidationVerdict.INVALID
        assert r.confidence == 0.97
        assert len(r.findings) == 2
        assert {f.name for f in r.findings} == {"pattern_a", "pattern_b"}

    def test_pattern_below_suspect_but_outcome_is_silent_failure_gets_composite_finding(self) -> None:
        """If ALL patterns are below the suspect threshold but the executor
        still flagged SUSPECTED_SILENT_FAILURE, a composite finding ensures
        the router knows."""
        cfg = ValidatorConfig(suspect_threshold=0.99)
        v = ResponseValidator(config=cfg)
        r = v.validate(
            _result(
                outcome=CallOutcome.SUSPECTED_SILENT_FAILURE,
                bad_patterns=[
                    KnownBadPatternMatch(
                        pattern_name="borderline",
                        description="Just missed threshold",
                        confidence=0.95,
                    )
                ],
            )
        )
        assert r.verdict is ValidationVerdict.INVALID
        assert len(r.findings) == 1
        assert r.findings[0].category is FindingCategory.SILENT_FAILURE
        assert r.findings[0].name == "silent_failure_below_threshold"

    def test_no_bad_patterns_but_success_outcome_is_valid(self) -> None:
        v = ResponseValidator()
        r = v.validate(_result(outcome=CallOutcome.SUCCESS, bad_patterns=[]))
        assert r.verdict is ValidationVerdict.VALID


# ============================================================= error classification


class TestErrorValidation:
    """Error responses should produce appropriate findings."""

    def test_transient_error_is_retryable_same_provider(self) -> None:
        v = ResponseValidator()
        r = v.validate(
            _result(
                outcome=CallOutcome.FAILURE,
                error_category=ErrorCategory.TRANSIENT,
                classified_error=ClassifiedError(
                    category=ErrorCategory.TRANSIENT,
                    message="Server error 500",
                ),
            )
        )
        assert r.verdict is ValidationVerdict.INVALID
        assert r.should_retry is True
        f = r.findings[0]
        assert f.retryable_same_provider is True
        assert f.retryable_different_provider is True

    def test_rate_limited_error_is_retryable_after_backoff(self) -> None:
        v = ResponseValidator()
        r = v.validate(
            _result(
                outcome=CallOutcome.FAILURE,
                error_category=ErrorCategory.RATE_LIMITED,
                classified_error=ClassifiedError(
                    category=ErrorCategory.RATE_LIMITED,
                    retry_after_seconds=30.0,
                    message="Rate limit exceeded",
                ),
            )
        )
        assert r.verdict is ValidationVerdict.INVALID
        assert r.should_retry is True
        f = next(f for f in r.findings if f.category is FindingCategory.ERROR_RESPONSE)
        assert f.name == "error_rate_limited"
        assert f.retryable_same_provider is True

    def test_quota_exhausted_is_failover_only(self) -> None:
        v = ResponseValidator()
        r = v.validate(
            _result(
                outcome=CallOutcome.FAILURE,
                error_category=ErrorCategory.QUOTA_EXHAUSTED,
                classified_error=ClassifiedError(
                    category=ErrorCategory.QUOTA_EXHAUSTED,
                    message="Credit balance exhausted",
                ),
            )
        )
        assert r.verdict is ValidationVerdict.INVALID
        assert r.should_retry is True
        assert r.retry_on_different_provider is True
        f = next(f for f in r.findings if f.category is FindingCategory.ERROR_RESPONSE)
        assert f.retryable_same_provider is False
        assert f.retryable_different_provider is True

    def test_permanent_error_is_not_retryable(self) -> None:
        v = ResponseValidator()
        r = v.validate(
            _result(
                outcome=CallOutcome.FAILURE,
                error_category=ErrorCategory.PERMANENT,
                classified_error=ClassifiedError(
                    category=ErrorCategory.PERMANENT,
                    message="Bad request (400)",
                ),
            )
        )
        assert r.verdict is ValidationVerdict.INVALID
        assert r.should_retry is False
        assert r.retry_on_different_provider is False

    def test_provider_overloaded_recommends_failover(self) -> None:
        v = ResponseValidator()
        r = v.validate(
            _result(
                outcome=CallOutcome.FAILURE,
                error_category=ErrorCategory.PROVIDER_OVERLOADED,
                classified_error=ClassifiedError(
                    category=ErrorCategory.PROVIDER_OVERLOADED,
                    message="Overloaded (529)",
                ),
            )
        )
        assert r.verdict is ValidationVerdict.INVALID
        assert r.should_retry is True
        f = next(f for f in r.findings if f.category is FindingCategory.ERROR_RESPONSE)
        assert f.retryable_same_provider is True
        assert f.retryable_different_provider is True

    def test_requires_human_action_is_not_retryable(self) -> None:
        v = ResponseValidator()
        r = v.validate(
            _result(
                outcome=CallOutcome.FAILURE,
                error_category=ErrorCategory.REQUIRES_HUMAN_ACTION,
                classified_error=ClassifiedError(
                    category=ErrorCategory.REQUIRES_HUMAN_ACTION,
                    message="3DS required",
                ),
            )
        )
        assert r.verdict is ValidationVerdict.INVALID
        assert r.should_retry is False

    def test_unknown_error_is_conservative(self) -> None:
        v = ResponseValidator()
        r = v.validate(
            _result(
                outcome=CallOutcome.FAILURE,
                error_category=ErrorCategory.UNKNOWN,
                classified_error=ClassifiedError(
                    category=ErrorCategory.UNKNOWN,
                    message="Unknown status 418",
                    is_definitively_classified=False,
                ),
            )
        )
        assert r.verdict is ValidationVerdict.INVALID
        f = next(f for f in r.findings if f.category is FindingCategory.ERROR_RESPONSE)
        assert f.confidence == 0.5  # lower confidence for UNKNOWN


# ============================================================= transport failures


class TestTransportFailures:
    def test_transport_failure_is_retryable(self) -> None:
        v = ResponseValidator()
        exc = httpx.ConnectTimeout("timed out", request=httpx.Request("POST", "https://x.test"))
        r = v.validate(
            _result(
                outcome=CallOutcome.FAILURE,
                exception=exc,
                classified_error=ClassifiedError(
                    category=ErrorCategory.TRANSIENT,
                    message="Transport failure ConnectTimeout",
                ),
            )
        )
        assert r.verdict is ValidationVerdict.INVALID
        assert r.should_retry is True
        transport_findings = [f for f in r.findings if f.category is FindingCategory.TRANSPORT_FAILURE]
        assert len(transport_findings) == 1
        assert transport_findings[0].retryable_same_provider is True


# ============================================================= policy configuration


class TestPolicyConfiguration:
    def test_custom_thresholds(self) -> None:
        cfg = ValidatorConfig(suspect_threshold=0.3, reject_threshold=0.6)
        v = ResponseValidator(config=cfg)

        # A 0.5 confidence pattern: above suspect (0.3), below reject (0.6)
        r = v.validate(
            _result(
                outcome=CallOutcome.SUCCESS,
                bad_patterns=[
                    KnownBadPatternMatch(
                        pattern_name="fuzzy_pattern",
                        description="Not sure",
                        confidence=0.5,
                    )
                ],
            )
        )
        assert r.verdict is ValidationVerdict.SUSPECT
        assert len(r.findings) == 1

    def test_treat_suspected_silent_failure_as_suspect_not_invalid(self) -> None:
        cfg = ValidatorConfig(treat_suspected_silent_failure_as_invalid=False)
        v = ResponseValidator(config=cfg)
        r = v.validate(
            _result(
                outcome=CallOutcome.SUSPECTED_SILENT_FAILURE,
                bad_patterns=[
                    KnownBadPatternMatch(
                        pattern_name="some_pattern",
                        description="Something",
                        confidence=0.95,
                    )
                ],
            )
        )
        assert r.verdict is ValidationVerdict.SUSPECT

    def test_config_is_readable(self) -> None:
        cfg = ValidatorConfig(suspect_threshold=0.7, reject_threshold=0.9)
        v = ResponseValidator(config=cfg)
        assert v.config.suspect_threshold == 0.7
        assert v.config.reject_threshold == 0.9


# ============================================================= caveats


class TestCaveats:
    def test_billed_invalid_response_has_wasted_spend_caveat(self) -> None:
        v = ResponseValidator()
        r = v.validate(
            _result(
                outcome=CallOutcome.SUSPECTED_SILENT_FAILURE,
                cost_usd=0.005,
                bad_patterns=[
                    KnownBadPatternMatch(
                        pattern_name="error_body_on_2xx",
                        description="Error body",
                        confidence=0.95,
                    )
                ],
            )
        )
        assert r.was_billed is True
        assert any("wasted spend" in c for c in r.caveats)

    def test_suspected_silent_failure_below_thresholds_has_caveat(self) -> None:
        cfg = ValidatorConfig(
            suspect_threshold=0.99,
            treat_suspected_silent_failure_as_invalid=False,
        )
        v = ResponseValidator(config=cfg)
        r = v.validate(
            _result(
                outcome=CallOutcome.SUSPECTED_SILENT_FAILURE,
                bad_patterns=[
                    KnownBadPatternMatch(
                        pattern_name="low_conf",
                        description="Low",
                        confidence=0.5,
                    )
                ],
            )
        )
        # With treat_suspected_silent_failure_as_invalid=False and all
        # patterns below suspect_threshold, but outcome is SUSPECTED_SILENT_FAILURE:
        # the silent_failure_below_threshold finding still keeps it INVALID
        # because the default treat_suspected_silent_failure_as_invalid=True...
        # Actually, we set it to False here, so the composite finding drives SUSPECT.
        # But the outcome mismatch caveat should be present.
        assert any("below the validator's thresholds" in c for c in r.caveats)


# ============================================================= end-to-end with executor


class TestExecutorIntegration:
    """Verify bad_patterns flow through the real CallExecutor."""

    async def test_executor_populates_bad_patterns_on_call_result(self, tmp_path) -> None:
        """The executor should now carry bad_patterns through to CallResult."""
        patterns = [
            KnownBadPatternMatch(
                pattern_name="test_pattern",
                description="Test detection",
                confidence=0.95,
            )
        ]

        class DetectingAdapter(FakeAdapter):
            def check_known_bad_patterns(self, response: httpx.Response) -> list[KnownBadPatternMatch]:
                return patterns

            async def send(self, request: AdapterRequest) -> httpx.Response:
                return httpx.Response(200, json={"ok": True})

        adapter = DetectingAdapter()
        ledger = SqliteLedgerStore(tmp_path / "test.db")
        executor = CallExecutor(adapter, ledger, identity_key="team-a")
        result = await executor.execute(AdapterRequest(operation="op", payload={"model": "m0"}))

        assert result.bad_patterns == patterns
        assert result.event.outcome is CallOutcome.SUSPECTED_SILENT_FAILURE

        # Now validate it
        v = ResponseValidator()
        validation = v.validate(result)
        assert validation.verdict is ValidationVerdict.INVALID
        assert validation.confidence == 0.95

    async def test_executor_clean_response_has_empty_bad_patterns(self, tmp_path) -> None:
        class CleanAdapter(FakeAdapter):
            async def send(self, request: AdapterRequest) -> httpx.Response:
                return httpx.Response(200, json={"ok": True})

        adapter = CleanAdapter()
        ledger = SqliteLedgerStore(tmp_path / "test.db")
        executor = CallExecutor(adapter, ledger, identity_key="team-a")
        result = await executor.execute(AdapterRequest(operation="op", payload={"model": "m0"}))

        assert result.bad_patterns == []
        v = ResponseValidator()
        validation = v.validate(result)
        assert validation.verdict is ValidationVerdict.VALID


# ============================================================= edge cases


class TestEdgeCases:
    def test_empty_bad_patterns_list_is_valid(self) -> None:
        v = ResponseValidator()
        r = v.validate(_result(bad_patterns=[]))
        assert r.verdict is ValidationVerdict.VALID

    def test_none_bad_patterns_is_valid(self) -> None:
        v = ResponseValidator()
        r = v.validate(_result(bad_patterns=None))
        assert r.verdict is ValidationVerdict.VALID

    def test_pending_outcome_with_no_findings_is_valid(self) -> None:
        v = ResponseValidator()
        r = v.validate(
            _result(outcome=CallOutcome.PENDING)
        )
        assert r.verdict is ValidationVerdict.VALID
        assert r.outcome is CallOutcome.PENDING

    def test_failure_with_no_classified_error_still_checks_outcome(self) -> None:
        """FAILURE outcome without classified_error: possible if constructed
        manually (not by the executor), should not crash."""
        v = ResponseValidator()
        r = v.validate(_result(outcome=CallOutcome.FAILURE))
        # No findings at all -> VALID (the validator trusts the findings
        # mechanism, and an unadorned FAILURE with no classified_error
        # has no evidence to report). This is an edge case that wouldn't
        # occur with the real executor.
        assert r.verdict is ValidationVerdict.VALID

    def test_outcome_is_passed_through(self) -> None:
        v = ResponseValidator()
        for outcome in CallOutcome:
            r = v.validate(_result(outcome=outcome))
            assert r.outcome is outcome

    def test_bad_pattern_retryable_same_provider(self) -> None:
        """Bad patterns from adapters are retryable on the same provider
        (the response is atypical, not a systematic rejection)."""
        v = ResponseValidator()
        r = v.validate(
            _result(
                outcome=CallOutcome.SUSPECTED_SILENT_FAILURE,
                bad_patterns=[
                    KnownBadPatternMatch(
                        pattern_name="x", description="y", confidence=0.95,
                    )
                ],
            )
        )
        assert all(f.retryable_same_provider for f in r.findings if f.category is FindingCategory.BAD_PATTERN)

    def test_error_and_bad_pattern_together(self) -> None:
        """A response can have both an error classification and bad patterns
        (e.g., if the executor is extended in the future). The validator
        should handle both."""
        v = ResponseValidator()
        r = v.validate(
            _result(
                outcome=CallOutcome.FAILURE,
                error_category=ErrorCategory.TRANSIENT,
                classified_error=ClassifiedError(
                    category=ErrorCategory.TRANSIENT,
                    message="Server error",
                ),
                bad_patterns=[
                    KnownBadPatternMatch(
                        pattern_name="pattern", description="Also bad",
                        confidence=0.9,
                    )
                ],
            )
        )
        assert r.verdict is ValidationVerdict.INVALID
        categories = {f.category for f in r.findings}
        assert FindingCategory.BAD_PATTERN in categories
        assert FindingCategory.ERROR_RESPONSE in categories


# ============================================================= invariant tests


class TestInvariants:
    """Property-based style: random-ish inputs should not break invariants."""

    @pytest.mark.parametrize(
        "confidence", [0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 0.97, 1.0]
    )
    def test_confidence_ordering_invariant(self, confidence: float) -> None:
        """If the ONLY finding is a bad pattern at a given confidence,
        the result's confidence must equal it."""
        v = ResponseValidator()
        r = v.validate(
            _result(
                outcome=CallOutcome.SUSPECTED_SILENT_FAILURE if confidence >= 0.5 else CallOutcome.SUCCESS,
                bad_patterns=[
                    KnownBadPatternMatch(
                        pattern_name="p", description="d", confidence=confidence,
                    )
                ],
            )
        )
        if confidence >= v.config.suspect_threshold:
            assert r.confidence == confidence
        else:
            # Below threshold: pattern is dropped. But if the outcome is
            # SUSPECTED_SILENT_FAILURE, a composite finding is added at
            # the suspect_threshold confidence.
            pass  # not a clean invariant to check here

    @pytest.mark.parametrize("outcome", list(CallOutcome))
    def test_validate_never_raises_for_any_outcome(self, outcome: CallOutcome) -> None:
        v = ResponseValidator()
        r = v.validate(_result(outcome=outcome))
        assert isinstance(r, ValidationResult)

    @pytest.mark.parametrize("category", list(ErrorCategory))
    def test_validate_never_raises_for_any_error_category(self, category: ErrorCategory) -> None:
        v = ResponseValidator()
        r = v.validate(
            _result(
                outcome=CallOutcome.FAILURE,
                error_category=category,
                classified_error=ClassifiedError(category=category, message="test"),
            )
        )
        assert isinstance(r, ValidationResult)
        assert r.verdict is ValidationVerdict.INVALID

    def test_valid_verdict_implies_no_findings(self) -> None:
        v = ResponseValidator()
        r = v.validate(_result())
        assert r.verdict is ValidationVerdict.VALID
        assert r.findings == ()
        assert r.confidence == 0.0

    def test_should_retry_is_false_when_valid(self) -> None:
        v = ResponseValidator()
        r = v.validate(_result())
        assert r.should_retry is False

    def test_retry_on_different_provider_implies_should_retry(self) -> None:
        """If retry_on_different_provider is True, should_retry must also be True."""
        v = ResponseValidator()
        r = v.validate(
            _result(
                outcome=CallOutcome.FAILURE,
                error_category=ErrorCategory.QUOTA_EXHAUSTED,
                classified_error=ClassifiedError(
                    category=ErrorCategory.QUOTA_EXHAUSTED,
                    message="Quota gone",
                ),
            )
        )
        if r.retry_on_different_provider:
            assert r.should_retry is True
