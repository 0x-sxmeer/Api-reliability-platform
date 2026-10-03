"""
Response Validator (Phase 4): policy-configurable validation of completed calls.

WHAT THE ROUTING ENGINE (PHASE 5) CALLS, AND WHAT IT GETS BACK
    validator = ResponseValidator(config=ValidatorConfig(...))
    result = await executor.execute(request)
    quota_engine.observe(adapter, request, ...)
    validation = validator.validate(result)

    validate() returns a ValidationResult: a verdict (VALID / SUSPECT /
    INVALID), the highest confidence among any finding, a list of findings,
    operator-policy-driven routing recommendations (should_retry,
    retry_on_different_provider), whether the call was billed, and caveats.

    The validator does not retry, sleep, pick a provider, or touch the
    ledger. It answers "is this response usable, and how sure are we";
    deciding what to do is Phase 5.

WHERE THE DETECTION COMES FROM (and where it doesn't)
    All vendor-specific pattern detection lives in each adapter's
    check_known_bad_patterns() method. The validator does NOT have its own
    detection logic independent of the adapters. Phase 3's CallResult
    carries the already-computed KnownBadPatternMatch objects (added by
    Phase 4: CallResult.bad_patterns); the validator reads those plus the
    event's outcome and error classification. It is a POLICY LAYER above
    detection, not a second detection layer.

    check_known_bad_patterns returns data; this engine applies policy:
    "given operator-configured thresholds, is this response usable, and
    should the router retry or fail over?"

WHY THIS IS A SEPARATE ENGINE AND NOT INLINE IN THE EXECUTOR
    The executor's job is "send once, classify, log one event." It already
    sets CallOutcome.SUSPECTED_SILENT_FAILURE correctly; that ledger record
    is accurate. The validator adds:
    1. Confidence thresholding — a 0.90 pattern might be informational; a
       0.97 might be actionable. The operator's policy decides.
    2. Multi-finding composition — the executor reduces bad_patterns to a
       binary (any/none); the validator preserves per-finding detail.
    3. A Phase-5-ready vocabulary — the Routing Engine needs the same shape
       of structured answer as QuotaAssessment: verdict, confidence, and
       routing recommendations. ValidationResult is that.
    4. Classification-based validation — failed calls (non-None
       classified_error) with specific error categories also produce
       findings, so the router can see "this was rate-limited AND had a
       bad pattern" in one unified assessment.
    5. Retry recommendation — whether to retry on the same provider or
       fail over to a different one, based on the nature of the findings.

WHAT THIS ENGINE DELIBERATELY DOES NOT DO (and why)
    * No stateful / cross-call detection. Detecting "the provider is
      returning shorter and shorter responses" needs a window of recent
      calls; that data model depends on how Phase 5 structures its
      per-provider state. Declared as a plug point (ValidatorConfig has
      no cross-call knobs today), not built.
    * No ledger reads. The validator works from a single CallResult. If
      cross-call heuristics are added, they would be fed by Phase 5
      (which already holds per-provider history), not by ledger queries
      from here.
    * No cost enforcement. The validator reports `was_billed` so Phase 6
      can track wasted spend, but does not enforce budgets (that's the
      Budget Engine's job). The pricing gap (no Anthropic/Gemini prices)
      means cost_usd is None for those providers today.

ARCHITECTURE RULE: this module calls no adapter methods, imports no
adapter module, and branches on no vendor name. It reads only the
provider-agnostic types on CallResult and GatewayEvent.

Concurrency: validate() is synchronous and does no I/O, so it is trivially
safe on any event loop or thread.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from gateway.core.executor import CallResult
from gateway.core.types import (
    CallOutcome,
    ErrorCategory,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ vocabulary


class ValidationVerdict(str, Enum):
    """The validator's summary judgement on a completed call.

    VALID means "the response is usable output": no bad patterns, no error
    classification suggesting otherwise. SUSPECT means "the response has
    issues above the informational threshold but below the hard-reject
    threshold" — the router should prefer alternatives if available, but
    can use this response as a fallback. INVALID means "the response
    should not be used" — the router should retry (same or different
    provider) or surface the failure.

    These are deliberately NOT 1:1 with CallOutcome. A SUCCESS outcome can
    be SUSPECT (bad pattern below the reject threshold); a FAILURE outcome
    is always INVALID. The mapping is policy, and this engine is where
    that policy lives.
    """

    VALID = "valid"
    SUSPECT = "suspect"
    INVALID = "invalid"


class FindingCategory(str, Enum):
    """What kind of issue a finding describes.

    The validator does not invent new categories; it classifies findings
    from the data the executor already computed (bad patterns, error
    classification, outcome).
    """

    BAD_PATTERN = "bad_pattern"
    # A KnownBadPatternMatch from the adapter's check_known_bad_patterns.

    SILENT_FAILURE = "silent_failure"
    # The executor already flagged this as SUSPECTED_SILENT_FAILURE, so
    # the validator confirms it. This is a composite finding derived from
    # the outcome, not from a single pattern.

    ERROR_RESPONSE = "error_response"
    # The call returned an HTTP error (status >= 400). The validator
    # includes this so that a single ValidationResult describes the full
    # picture: the Routing Engine does not need to inspect CallResult and
    # ValidationResult separately.

    TRANSPORT_FAILURE = "transport_failure"
    # The call never received a response (transport exception). Same
    # rationale as ERROR_RESPONSE.


@dataclass(frozen=True)
class ValidationFinding:
    """One issue found during validation.

    Each finding carries its own confidence (from the source that
    produced it) and its own retry recommendation. The ValidationResult
    composes these into an overall recommendation.
    """

    category: FindingCategory
    name: str
    # A short, stable identifier for this specific finding. For
    # BAD_PATTERN findings, this is KnownBadPatternMatch.pattern_name.
    # For composite findings, a fixed string like "outcome_mismatch".
    description: str
    confidence: float
    # 0.0–1.0. For BAD_PATTERN findings, from KnownBadPatternMatch.
    # For ERROR_RESPONSE/TRANSPORT_FAILURE, 1.0 (these are facts).
    retryable_same_provider: bool
    # Whether retrying on the SAME provider could help. For bad patterns:
    # generally True (the response is atypical, not a systematic
    # rejection). For errors: depends on the error category (TRANSIENT
    # yes, PERMANENT no).
    retryable_different_provider: bool
    # Whether failing over to a DIFFERENT provider could help. For bad
    # patterns: True (a different provider may not have the same issue).
    # For QUOTA_EXHAUSTED: True (the defining property). For PERMANENT:
    # False (a 400 bad-request fails on any provider).


@dataclass(frozen=True)
class ValidatorConfig:
    """Operator-tunable policy.

    These are engineering defaults, not provider facts: there is no
    official specification for what confidence level makes a bad-pattern
    match actionable. The defaults are chosen to flag the known documented
    cases (Anthropic's SSE error at 0.97, Anthropic's error body at 0.95,
    Gemini's blocked prompt at 0.90) while leaving room for future
    lower-confidence heuristics to be informational only.
    """

    suspect_threshold: float = 0.5
    # Findings with confidence >= this ARE reported (and can make the
    # verdict SUSPECT or INVALID). Below this, they are dropped entirely
    # — not even informational. This is a noise gate.

    reject_threshold: float = 0.85
    # Findings with confidence >= this make the verdict INVALID (not
    # just SUSPECT). ALL documented cases today (0.90, 0.95, 0.97)
    # exceed this, so the default is actionable out of the box.

    treat_suspected_silent_failure_as_invalid: bool = True
    # When the executor's outcome is SUSPECTED_SILENT_FAILURE, should
    # the validator verdict be INVALID (True) or SUSPECT (False)?
    # Default True because the executor only sets that outcome when at
    # least one pattern matched, and all current patterns are high
    # confidence. An operator processing low-confidence experimental
    # patterns could set this to False to let the router use such
    # responses as fallbacks.


@dataclass(frozen=True)
class ValidationResult:
    """What validate() returns — the validator's structured answer.

    Shaped like QuotaAssessment: verdict, confidence, evidence, and
    routing recommendations, so Phase 5 can consume both through the
    same kind of logic.
    """

    verdict: ValidationVerdict
    confidence: float
    # The highest confidence among all findings. 0.0 when there are no
    # findings (i.e., the response is VALID). This is NOT the
    # validator's confidence in its own verdict — it is the strongest
    # evidence of a problem. A VALID verdict with confidence 0.0 means
    # "no evidence of any issue."

    findings: tuple[ValidationFinding, ...]
    # All findings that passed the suspect_threshold. Empty for VALID.

    should_retry: bool
    # True when the router should attempt the call again (on the same
    # provider or a different one). Derived from the findings: True if
    # any finding is retryable.

    retry_on_different_provider: bool
    # True when retrying on the SAME provider is unlikely to help (e.g.,
    # a persistent bad-pattern match, or a quota exhaustion). The router
    # should fail over rather than retry locally.

    was_billed: bool
    # True when the event has a non-None cost_usd. Phase 6's Budget
    # Engine can use this to track "money spent on unusable output."
    # NOTE: the pricing gap (no Anthropic/Gemini prices) means this is
    # False (unknown) for those providers today; it does NOT mean
    # "known to be free."

    outcome: CallOutcome
    # The executor's original outcome, for context. The validator does
    # not change it; this is passed through for convenience so the
    # router doesn't need to look at CallResult and ValidationResult
    # separately.

    caveats: tuple[str, ...] = ()


# -------------------------------------------------------------------- engine


class ResponseValidator:
    """Validates completed CallResults against operator-configurable policy.

    One instance per policy configuration. Stateless: each validate() call
    is independent, so there is no registration step (unlike QuotaEngine,
    which needs adapter-declared windows and operator-configured limits).
    """

    def __init__(self, *, config: ValidatorConfig | None = None) -> None:
        self._config = config or ValidatorConfig()

    @property
    def config(self) -> ValidatorConfig:
        """The active policy configuration. Read-only."""
        return self._config

    def validate(self, result: CallResult) -> ValidationResult:
        """Validate a completed call and return a structured assessment.

        This is the method Phase 5's Routing Engine calls after
        executor.execute() and quota_engine.observe(). It is synchronous
        (no I/O) and never raises.
        """
        findings: list[ValidationFinding] = []
        cfg = self._config

        # 1. Bad-pattern findings from the adapter's detection layer.
        #    These are already computed and carried on CallResult.bad_patterns.
        for match in result.bad_patterns:
            if match.confidence >= cfg.suspect_threshold:
                findings.append(
                    ValidationFinding(
                        category=FindingCategory.BAD_PATTERN,
                        name=match.pattern_name,
                        description=match.description,
                        confidence=match.confidence,
                        retryable_same_provider=True,
                        retryable_different_provider=True,
                    )
                )

        # 2. Error-classification findings — a single ValidationResult
        #    should describe the full picture so the router does not need
        #    to inspect both CallResult and ValidationResult.
        if result.classified_error is not None:
            cat = result.classified_error.category
            finding = _error_finding(cat, result.classified_error.message)
            findings.append(finding)

        # 3. Transport-failure finding.
        if result.exception is not None:
            findings.append(
                ValidationFinding(
                    category=FindingCategory.TRANSPORT_FAILURE,
                    name="transport_failure",
                    description=(
                        f"Transport failure ({type(result.exception).__name__}): "
                        f"{result.exception}"
                    ),
                    confidence=1.0,
                    retryable_same_provider=True,
                    retryable_different_provider=True,
                )
            )

        # 4. Silent-failure confirmation — the executor already flagged it;
        #    the validator confirms it as a composite finding if no
        #    individual bad-pattern finding covered it yet (which would
        #    only happen if all patterns were below suspect_threshold).
        event_outcome = result.event.outcome
        if event_outcome is CallOutcome.SUSPECTED_SILENT_FAILURE:
            bp_names = {f.name for f in findings if f.category is FindingCategory.BAD_PATTERN}
            if not bp_names:
                # All patterns were below suspect_threshold, but the
                # executor still flagged the outcome. Add a composite
                # finding so the router knows.
                findings.append(
                    ValidationFinding(
                        category=FindingCategory.SILENT_FAILURE,
                        name="silent_failure_below_threshold",
                        description=(
                            "The executor flagged this as SUSPECTED_SILENT_FAILURE "
                            "but all bad-pattern matches were below the configured "
                            f"suspect threshold ({cfg.suspect_threshold})."
                        ),
                        confidence=cfg.suspect_threshold,
                        retryable_same_provider=True,
                        retryable_different_provider=True,
                    )
                )

        # ---- Compose findings into a verdict and routing recommendations.
        verdict, overall_confidence = self._compute_verdict(
            event_outcome, findings, cfg
        )
        should_retry = any(
            f.retryable_same_provider or f.retryable_different_provider
            for f in findings
        )
        retry_different = (
            verdict is not ValidationVerdict.VALID
            and any(f.retryable_different_provider for f in findings)
            and not all(f.retryable_same_provider for f in findings)
        )
        was_billed = result.event.cost_usd is not None

        caveats: list[str] = []
        if was_billed and verdict is not ValidationVerdict.VALID:
            caveats.append(
                "this response was billed but validated as "
                f"{verdict.value}; cost may be wasted spend"
            )
        bp_findings_present = any(
            f.category is FindingCategory.BAD_PATTERN for f in findings
        )
        if (
            event_outcome is CallOutcome.SUSPECTED_SILENT_FAILURE
            and not bp_findings_present
        ):
            # No individual bad-pattern finding survived the suspect
            # threshold, even though the executor detected patterns.
            # This can leave the verdict as VALID (if no composite
            # finding was added) or SUSPECT (if one was). Either way,
            # the router should know the executor saw something.
            caveats.append(
                "executor flagged SUSPECTED_SILENT_FAILURE but all "
                "patterns were below the validator's thresholds"
            )

        return ValidationResult(
            verdict=verdict,
            confidence=overall_confidence,
            findings=tuple(findings),
            should_retry=should_retry and verdict is not ValidationVerdict.VALID,
            retry_on_different_provider=retry_different,
            was_billed=was_billed,
            outcome=event_outcome,
            caveats=tuple(caveats),
        )

    @staticmethod
    def _compute_verdict(
        outcome: CallOutcome,
        findings: list[ValidationFinding],
        cfg: ValidatorConfig,
    ) -> tuple[ValidationVerdict, float]:
        """Derive the verdict and overall confidence from findings + policy."""
        if not findings:
            return ValidationVerdict.VALID, 0.0

        max_confidence = max(f.confidence for f in findings)

        # Hard failures are always INVALID regardless of threshold.
        if outcome is CallOutcome.FAILURE:
            return ValidationVerdict.INVALID, max_confidence

        # SUSPECTED_SILENT_FAILURE: policy decides INVALID vs SUSPECT.
        if outcome is CallOutcome.SUSPECTED_SILENT_FAILURE:
            if cfg.treat_suspected_silent_failure_as_invalid:
                return ValidationVerdict.INVALID, max_confidence
            # Operator said "treat as SUSPECT, not INVALID" for responses
            # the executor flagged. The router can still use it as a
            # fallback or retry, depending on its own policy.
            return ValidationVerdict.SUSPECT, max_confidence

        # SUCCESS outcome with findings: threshold-based.
        if max_confidence >= cfg.reject_threshold:
            return ValidationVerdict.INVALID, max_confidence
        return ValidationVerdict.SUSPECT, max_confidence


def _error_finding(category: ErrorCategory, message: str) -> ValidationFinding:
    """Map an ErrorCategory to a ValidationFinding.

    This is a STATIC mapping, not a provider-specific one. The adapter
    already classified the error; the validator just translates the
    category into retry recommendations.
    """
    if category is ErrorCategory.TRANSIENT:
        return ValidationFinding(
            category=FindingCategory.ERROR_RESPONSE,
            name="error_transient",
            description=message,
            confidence=1.0,
            retryable_same_provider=True,
            retryable_different_provider=True,
        )
    if category is ErrorCategory.RATE_LIMITED:
        return ValidationFinding(
            category=FindingCategory.ERROR_RESPONSE,
            name="error_rate_limited",
            description=message,
            confidence=1.0,
            retryable_same_provider=True,  # after backoff
            retryable_different_provider=True,
        )
    if category is ErrorCategory.PROVIDER_OVERLOADED:
        return ValidationFinding(
            category=FindingCategory.ERROR_RESPONSE,
            name="error_provider_overloaded",
            description=message,
            confidence=1.0,
            retryable_same_provider=True,  # later
            retryable_different_provider=True,  # now
        )
    if category is ErrorCategory.QUOTA_EXHAUSTED:
        return ValidationFinding(
            category=FindingCategory.ERROR_RESPONSE,
            name="error_quota_exhausted",
            description=message,
            confidence=1.0,
            retryable_same_provider=False,
            retryable_different_provider=True,
        )
    if category is ErrorCategory.PERMANENT:
        return ValidationFinding(
            category=FindingCategory.ERROR_RESPONSE,
            name="error_permanent",
            description=message,
            confidence=1.0,
            retryable_same_provider=False,
            retryable_different_provider=False,
        )
    if category is ErrorCategory.REQUIRES_HUMAN_ACTION:
        return ValidationFinding(
            category=FindingCategory.ERROR_RESPONSE,
            name="error_requires_human_action",
            description=message,
            confidence=1.0,
            retryable_same_provider=False,
            retryable_different_provider=False,
        )
    # UNKNOWN
    return ValidationFinding(
        category=FindingCategory.ERROR_RESPONSE,
        name="error_unknown",
        description=message,
        confidence=0.5,
        retryable_same_provider=False,
        retryable_different_provider=False,
    )
