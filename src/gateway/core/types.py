"""
Core types shared across the entire gateway.

CRITICAL RULE: nothing in this file may reference a specific provider
(no "anthropic", "openai", "stripe", etc.). If you find yourself wanting
to add a provider-specific field or enum value here, that knowledge
belongs in an adapter instead. This file is the contract every adapter
promises to speak — it has to stay provider-agnostic or the whole
"one adapter per vendor, zero vendor knowledge above it" architecture
breaks down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class LimitingUnit(str, Enum):
    """
    What scope a provider actually enforces its rate/quota limits against.

    This exists because of a finding from research: providers differ on
    this in ways that break naive scaling strategies. Anthropic limits
    per-organization, so spinning up more API keys under the same org
    does nothing. AWS limits per-account (among other scopes), so
    spinning up more accounts genuinely does help. A gateway that
    doesn't model this explicitly will recommend the wrong fix.
    """

    KEY = "key"
    PROJECT = "project"
    ORGANIZATION = "organization"
    ACCOUNT = "account"
    REGION = "region"
    RESOURCE = "resource"  # e.g. a single Lambda's concurrency, a single phone number


class ErrorCategory(str, Enum):
    """
    How a failed (or suspicious) response should be handled.

    This is deliberately NOT just "retryable vs not" — that binary is
    exactly what causes the AWS Bedrock problem, where ThrottlingException
    and ModelNotReadyException are both HTTP 429 but need opposite fixes.
    An adapter's classify_error() has to pick one of these, and the
    Routing Engine (Phase 5) will branch on which one it got.
    """

    TRANSIENT = "transient"
    # A one-off glitch (network blip, brief 5xx). Safe to retry soon,
    # same request, normal backoff.

    RATE_LIMITED = "rate_limited"
    # The caller has exceeded a quota. Retrying immediately makes it
    # worse. Must back off according to the provider's own signal
    # (e.g. Retry-After) and should inform the Quota Engine, not just
    # the Routing Engine.

    PROVIDER_OVERLOADED = "provider_overloaded"
    # The provider itself is capacity-constrained (Anthropic's 529 is
    # the canonical example). This is NOT the caller's quota being
    # exceeded — it means the provider is under load. Different retry
    # curve than RATE_LIMITED, and a candidate for failover to another
    # provider rather than just waiting.

    PERMANENT = "permanent"
    # Bad request, invalid auth, malformed payload. Retrying the exact
    # same request will never succeed. No backoff — surface immediately.

    QUOTA_EXHAUSTED = "quota_exhausted"
    # A spend cap / credit balance / usage limit is used up. Retrying
    # provably fails until a reset or a human action, BUT failing over
    # to a different provider succeeds immediately. That last property
    # is why this is not PERMANENT (a 400 fails identically on any
    # provider) and not RATE_LIMITED (which says "retry soon"). Forced
    # by all three providers: Anthropic's enforced_spend_limit_reached,
    # OpenAI's credit_balance_exhausted / *_spend_limit_exceeded /
    # organization_usage_limit_exceeded, and Gemini's tier "limit: 0".
    # ClassifiedError.resume_at is set when the provider states when
    # access returns (Anthropic does; OpenAI credit exhaustion doesn't).

    REQUIRES_HUMAN_ACTION = "requires_human_action"
    # The request is correctly blocked pending a human step (e.g. a
    # payment requiring 3DS/SCA bank approval). Not a failure at all —
    # must never be retried automatically, and needs its own ledger
    # state so it doesn't get counted as an error.

    UNKNOWN = "unknown"
    # The adapter couldn't classify it. Better to be honest about this
    # than to guess wrong and silently misroute.


class CallOutcome(str, Enum):
    """
    The final state of a logged event, as far as the ledger is concerned.

    SUSPECTED_SILENT_FAILURE exists because of a finding from research:
    a response can be HTTP 200 and still be wrong (a known-bad-pattern
    match, or later, a schema mismatch from the Phase 4 Response
    Validator). Folding that into plain SUCCESS would make the whole
    reason this project exists invisible in its own ledger.

    PENDING exists for providers where the real outcome isn't known at
    call time — e.g. email deliverability, where you get an initial
    202 Accepted and the real signal (bounce, spam-fold, complaint)
    arrives later via a separate webhook. The ledger needs to be able
    to update a PENDING event to SUCCESS/FAILURE after the fact rather
    than treating the initial response as final.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    SUSPECTED_SILENT_FAILURE = "suspected_silent_failure"
    PENDING = "pending"


class SignalQuality(str, Enum):
    """
    How much live rate-limit information a provider actually gave us on
    THIS response. Forced by Gemini, which documents no rate-limit
    response headers: the Quota Engine (Phase 3) must know when it
    cannot trust headers and has to fall back to counting from the
    ledger instead. Never fake data to look FULL.
    """

    FULL = "full"        # remaining + limit + reset for the binding dimension
    PARTIAL = "partial"  # some fields, e.g. only retry-after, or only an error body
    NONE = "none"        # provider gave nothing usable on this response


class QuotaDimension(str, Enum):
    """What a quota window measures. Used by ProviderAdapter.quota_windows()."""

    REQUESTS = "requests"
    TOKENS = "tokens"
    SPEND = "spend"  # USD


class WindowKind(str, Enum):
    ROLLING = "rolling"  # the last N seconds, ending now
    CALENDAR_DAY = "calendar_day"  # resets at local midnight in an IANA timezone


@dataclass(frozen=True)
class QuotaWindow:
    """
    One documented limit window a provider enforces, declared by its
    adapter so the Quota Engine can count against it from the ledger when
    the provider returns no usable headers (Phase 3, forced by Gemini).

    This carries the SHAPE of a window, never the numeric limit. Limit
    values are operator configuration: Gemini documents that the active
    numbers are only visible in a dashboard and vary by tier and model,
    so no adapter could hard-code them truthfully.

    `name` is an opaque, adapter-chosen, stable id ("rpm", "rpd", ...).
    The operator's limit config is keyed by it; the engine never
    interprets the string.
    """

    name: str
    dimension: QuotaDimension
    kind: WindowKind
    seconds: int | None = None  # required for ROLLING
    tz: str | None = None  # IANA zone name; required for CALENDAR_DAY

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("QuotaWindow.name must be non-empty")
        if self.kind is WindowKind.ROLLING and (self.seconds is None or self.seconds <= 0):
            raise ValueError(f"ROLLING window {self.name!r} needs seconds > 0")
        if self.kind is WindowKind.CALENDAR_DAY and not self.tz:
            raise ValueError(f"CALENDAR_DAY window {self.name!r} needs an IANA tz name")


@dataclass(frozen=True)
class RateLimitSnapshot:
    """
    A normalized view of a provider's rate-limit state, parsed from
    whatever headers that specific provider actually returns.

    Every field is optional because providers expose different subsets
    of this information (Anthropic's 529 carries no quota headers at
    all, for instance). Fields being None means "this provider didn't
    tell us" — not "we don't have this data because we crashed."
    """

    limiting_unit: LimitingUnit
    signal_quality: SignalQuality = SignalQuality.NONE
    requests_remaining: int | None = None
    requests_limit: int | None = None
    tokens_remaining: int | None = None
    tokens_limit: int | None = None
    reset_at: datetime | None = None
    # ALWAYS an absolute, timezone-aware UTC instant, whatever the
    # provider sent. Anthropic sends an RFC 3339 timestamp (used as-is).
    # OpenAI sends a duration ("6m0s"); the adapter converts it to
    # (response received time + duration). CLOCK-SKEW CAVEAT: a
    # duration-derived reset_at is measured against OUR clock at receipt,
    # so it is only as accurate as the network latency plus our clock
    # (typically fine for a 60s window; do not use it for hour-scale
    # scheduling without NTP). A timestamp-derived one is measured
    # against the PROVIDER's clock, so skew between the two clocks
    # shifts it. Consumers should treat reset_at as an estimate.
    reset_at_source: str | None = None
    # "timestamp" | "duration" | None. Lets a consumer know which of the
    # two skew caveats above applies to this value.
    retry_after_seconds: float | None = None
    raw_headers: dict[str, str] = field(default_factory=dict)
    # raw_headers is kept for debugging/audit — the Quota Engine (Phase 3)
    # should use the parsed fields above, not re-parse this itself.
    additional_scopes: tuple[RateLimitSnapshot, ...] = ()
    # Phase 3 (forced by OpenAI): scopes that are enforced SIMULTANEOUSLY
    # with the primary one described by the fields above. A call succeeds
    # only if EVERY scope has room, so a consumer must treat the tightest
    # one as binding. Each entry is a full snapshot for ONE other scope
    # (its own limiting_unit / remaining / limit / reset_at); it carries no
    # raw_headers of its own (they live on the parent) and does not nest
    # further. Empty for single-scope providers, which is all of them
    # except OpenAI today. Adding this field changed no existing caller:
    # the default is an empty tuple.


@dataclass(frozen=True)
class ClassifiedError:
    """What an adapter's classify_error() returns."""

    category: ErrorCategory
    retry_after_seconds: float | None = None
    resume_at: datetime | None = None
    # When access returns, if the provider states it OR the adapter can
    # derive it from a documented reset rule (QUOTA_EXHAUSTED only).
    # None means "unknown", NOT "never". (Phase 3 widened this from
    # "states it" to include "derived from a documented rule" so the
    # Gemini adapter can report its VERIFIED midnight-Pacific reset; pair
    # a derived value with is_definitively_classified=False whenever the
    # classification that led to it is itself a heuristic.)
    message: str = ""
    is_definitively_classified: bool = True
    # False when the adapter is guessing (e.g. an unrecognized status
    # code). The Routing Engine should be more conservative — smaller
    # number of retries, more logging — when this is False.


@dataclass(frozen=True)
class KnownBadPatternMatch:
    """
    A hit against a provider's library of "response looks fine but
    isn't" signatures — e.g. an LLM grounding feature that returns
    200 with a body saying "I cannot find any data right now."
    """

    pattern_name: str
    description: str
    confidence: float  # 0.0-1.0 — some patterns are exact-match, others heuristic


@dataclass(frozen=True)
class UsageInfo:
    """Provider-agnostic token usage, extracted by an adapter."""

    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    # cached_input_tokens is the SUBSET of input_tokens served from cache.


def utcnow() -> datetime:
    """Single source of truth for 'now' so every timestamp in the
    system is timezone-aware and consistent. Avoids the classic bug
    of mixing naive and aware datetimes across adapters."""
    return datetime.now(UTC)
