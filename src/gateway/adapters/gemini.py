"""
Google Gemini provider adapter (targets the `generateContent` REST
surface -- see docs/provider-notes.md for why, over the newer
Interactions API).

All Gemini-specific knowledge lives in this file and nowhere else.
See docs/provider-notes.md ("Google Gemini" section) for full
VERIFIED/UNVERIFIED sourcing.

Key facts this adapter encodes:
- NO rate-limit response headers are documented for generateContent, on
  ANY response, success or failure. VERIFIED-absence (official rate-limits
  page read in full; nothing of the kind is mentioned). Every
  RateLimitSnapshot this adapter returns therefore has
  signal_quality=SignalQuality.NONE, always -- the Quota Engine (Phase 3)
  MUST fall back to ledger-derived counting for this provider; there is
  no header data to ever promote this past NONE.
- Rate limits (RPM/TPM/RPD) are applied per PROJECT, not per API key.
  VERIFIED, official docs, verbatim quote in provider-notes.md.
- Every rate/quota/spend limit -- RPM, TPM, RPD, AND the spend-based
  10-minute-window limit -- surfaces as the SAME HTTP 429 with the SAME
  gRPC status RESOURCE_EXHAUSTED. VERIFIED (official generateContent
  error-codes page: one row, "429 | RESOURCE_EXHAUSTED", explicitly
  covering "RPM, TPM, RPD, spend, etc." as one undifferentiated case).
- A real 429 body MAY carry a QuotaFailure.quotaId string that names
  which specific quota was hit (e.g. "...PerDay..." vs "...PerMinute...").
  This is UNVERIFIED AS A GUARANTEED CONTRACT (not documented on the
  official error-codes page itself; observed in real bodies via Google's
  own gemini-cli issue tracker). Distinguishing "will clear in seconds"
  from "will clear at midnight Pacific" from "will clear next billing
  window" is only possible via this heuristic string match -- so this
  adapter tries, but sets is_definitively_classified=False on every
  result derived from it, per the documented real-world failure of NOT
  doing this (see docs/provider-notes.md's citation of
  UKGovernmentBEIS/inspect_ai#5526: a client blind-retried a per-day
  quota 429 as if it were a per-minute one, backing off for 30 real
  minutes without ever succeeding).
- 400 FAILED_PRECONDITION can mean "billing not enabled for free tier in
  this region" -- not a malformed request, and not retryable without a
  billing change. Distinguished from plain 400 INVALID_ARGUMENT (a real
  bad request) via the gRPC `status` field, not the HTTP code alone.
- 402 payment_required (Interactions-API-only field per provider-notes;
  NOT observed on generateContent) is intentionally NOT handled here --
  this adapter targets generateContent, which does not document this
  code. Left out rather than speculatively added.
- No pricing entries exist for Gemini in core/pricing.py (never-guess
  rule: the official pricing page returned inconsistent cached content
  for the same models across fetches this session -- see
  docs/provider-notes.md). extract_usage() still extracts real token
  counts; compute_cost_usd() will correctly return None until verified
  prices are added.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from gateway.core.adapter import AdapterRequest, ProviderAdapter
from gateway.core.types import (
    ClassifiedError,
    ErrorCategory,
    KnownBadPatternMatch,
    LimitingUnit,
    QuotaDimension,
    QuotaWindow,
    RateLimitSnapshot,
    SignalQuality,
    UsageInfo,
    WindowKind,
    utcnow,
)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Substring match against QuotaFailure.quotaId (when present). UNVERIFIED
# as a stable contract -- see module docstring. Ordered so a "PerDay"
# check never gets shadowed by a broader match.
_QUOTA_ID_DAILY_MARKERS = ("PerDay",)
_QUOTA_ID_MINUTE_MARKERS = ("PerMinute",)

# VERIFIED (official rate-limits page): "Requests per day (RPD) quotas
# reset at midnight Pacific time."
_DAILY_RESET_TZ = "America/Los_Angeles"


def _next_daily_reset(now: datetime) -> datetime | None:
    """
    The next midnight Pacific after `now`, as an aware UTC instant, or None
    if the tz database is unavailable (classify_error must never raise, so
    a missing tzdata degrades to "resume time unknown", not a crash).
    Computed from local calendar arithmetic, so it is correct across the
    23- and 25-hour DST days.
    """
    try:
        zone = ZoneInfo(_DAILY_RESET_TZ)
    except ZoneInfoNotFoundError:
        return None
    local = now.astimezone(zone)
    tomorrow = datetime.combine(local.date() + timedelta(days=1), time.min, tzinfo=zone)
    return tomorrow.astimezone(UTC)


class GeminiAdapter(ProviderAdapter):
    """Adapter for the Gemini `generateContent` REST API."""

    def __init__(self, api_key: str, *, http_client: httpx.AsyncClient | None = None) -> None:
        self._api_key = api_key
        self._http = http_client or httpx.AsyncClient(base_url=GEMINI_API_BASE)

    @property
    def provider_name(self) -> str:
        return "gemini"

    def get_limiting_unit(self) -> LimitingUnit:
        # VERIFIED, verbatim from official docs: "Rate limits are applied
        # per project, not per API key."
        return LimitingUnit.PROJECT

    def quota_bucket(self, request: AdapterRequest) -> str:
        # VERIFIED (official rate-limits page): "Limits vary depending on
        # the specific model being used." Counters being per-model is
        # VERIFIED-by-example only (real 429 bodies name quotas like
        # `...PerProjectPerModel...`; see docs/provider-notes.md), not a
        # documented contract. Gemini's model is in `extra`, not the
        # payload (see send()). Must never raise -- send() is where a
        # missing model is reported -- so a fixed fallback is returned.
        # UNVERIFIED CAVEAT: the model string is used verbatim. If Google's
        # aliases (e.g. a "-latest" name) share counters with the concrete
        # version they point at -- not checked -- then an operator mixing an
        # alias and a version for one model splits its ledger count across
        # two buckets and UNDER-counts (the unsafe direction for ledger-
        # derived headroom). Use one spelling per model.
        model = request.extra.get("model") if isinstance(request.extra, dict) else None
        return model if isinstance(model, str) and model else "unknown-model"

    def quota_windows(self) -> tuple[QuotaWindow, ...]:
        # Gemini returns no rate-limit headers (VERIFIED-absence), so the
        # Quota Engine must count from the ledger; these are the windows it
        # may count against. SHAPES only -- the numeric limits are not
        # published by any API (VERIFIED: "View your active rate limits in
        # AI Studio") and vary by tier and model, so the operator
        # configures them.
        #   rpm / tpm   VERIFIED as dimensions ("measured across RPM, TPM
        #               (input), and RPD"). UNVERIFIED whether the minute is
        #               a sliding window or a fixed clock minute; a rolling
        #               60s window over-counts relative to a fixed one
        #               (it always contains the current fixed minute), so
        #               it errs toward LESS headroom -- the safe direction.
        #   rpd         VERIFIED: resets at midnight Pacific.
        #   spend_10m   VERIFIED: rolling 10-minute spend limit that
        #               applies by billing tier (Free: N/A).
        return (
            QuotaWindow("rpm", QuotaDimension.REQUESTS, WindowKind.ROLLING, seconds=60),
            QuotaWindow("tpm", QuotaDimension.TOKENS, WindowKind.ROLLING, seconds=60),
            QuotaWindow("rpd", QuotaDimension.REQUESTS, WindowKind.CALENDAR_DAY, tz=_DAILY_RESET_TZ),
            QuotaWindow("spend_10m", QuotaDimension.SPEND, WindowKind.ROLLING, seconds=600),
        )

    async def send(self, request: AdapterRequest) -> httpx.Response:
        if request.operation != "generateContent":
            raise ValueError(
                f"GeminiAdapter does not handle operation '{request.operation}'. "
                "Known: generateContent"
            )
        model = request.extra.get("model")
        if not model:
            raise ValueError(
                "GeminiAdapter requires request.extra['model'] (e.g. 'gemini-2.5-flash') "
                "since, unlike Anthropic/OpenAI, the model selects the URL path, not a "
                "payload field."
            )
        # VERIFIED (current official docs + independent third-party client
        # implementations, corroborating each other): the API key is sent
        # as the `x-goog-api-key` HEADER for standard generateContent
        # calls. A `?key=` query parameter also works (older docs/examples
        # use it, and it is not deprecated), but the header keeps the key
        # out of HTTP access logs and is what official examples lead with
        # today -- so that's what this adapter uses.
        return await self._http.post(
            f"/models/{model}:generateContent",
            json=request.payload,
            headers={"x-goog-api-key": self._api_key, "Content-Type": "application/json"},
        )

    # ---------------------------------------------------------------- headers

    def parse_rate_limit_headers(self, response: httpx.Response) -> RateLimitSnapshot:
        # ALWAYS NONE for generateContent -- there is nothing to parse.
        # This is not a bug or an unimplemented case; it's the documented
        # (by absence) reality of this API surface. See module docstring.
        return RateLimitSnapshot(
            limiting_unit=self.get_limiting_unit(),
            signal_quality=SignalQuality.NONE,
            raw_headers=dict(response.headers),
        )

    # ----------------------------------------------------------------- errors

    def classify_error(self, response: httpx.Response) -> ClassifiedError:
        status = response.status_code
        if status < 400:
            return ClassifiedError(
                category=ErrorCategory.UNKNOWN,
                message="classify_error called on a non-error response",
                is_definitively_classified=False,
            )

        body = self._safe_json(response)
        err = body.get("error", {}) if isinstance(body, dict) else {}
        err = err if isinstance(err, dict) else {}
        grpc_status = err.get("status")  # e.g. "RESOURCE_EXHAUSTED", SCREAMING_CASE
        message = str(err.get("message", ""))
        details = err.get("details") if isinstance(err.get("details"), list) else []

        if status == 400 and grpc_status == "FAILED_PRECONDITION":
            # VERIFIED: distinct from a malformed-request 400 -- typically
            # "free tier not available in your country, enable billing."
            # Not fixed by changing the request; needs a billing change.
            return ClassifiedError(
                category=ErrorCategory.PERMANENT,
                message=f"Gemini FAILED_PRECONDITION (billing/region prerequisite not met): {message}",
            )

        if status == 429:
            # ALWAYS RESOURCE_EXHAUSTED on this surface, whether the cause
            # is RPM, TPM, RPD, or the spend-based limit -- VERIFIED,
            # official docs treat all of these as one row/case. Try the
            # UNVERIFIED quotaId heuristic to say more; be honest when it
            # doesn't apply.
            quota_id = self._extract_quota_id(details)
            retry_delay_s = self._extract_retry_delay_seconds(details)

            if quota_id and any(marker in quota_id for marker in _QUOTA_ID_DAILY_MARKERS):
                return ClassifiedError(
                    category=ErrorCategory.QUOTA_EXHAUSTED,
                    # Phase 3 fix: this message has always claimed the
                    # VERIFIED midnight-Pacific reset but left resume_at
                    # None, discarding a fact the adapter knows. Still
                    # is_definitively_classified=False below: the DAILY
                    # diagnosis is the heuristic; the reset rule is not.
                    resume_at=_next_daily_reset(utcnow()),
                    message=f"Gemini RESOURCE_EXHAUSTED, quotaId={quota_id!r} (daily quota): "
                    f"{message} Will not clear until midnight Pacific (VERIFIED reset time "
                    "per official docs); retrying before then is futile, failover works now. "
                    "(quotaId-based classification is UNVERIFIED as a stable contract -- "
                    "see docs/provider-notes.md.)",
                    is_definitively_classified=False,
                )

            if quota_id and any(marker in quota_id for marker in _QUOTA_ID_MINUTE_MARKERS):
                return ClassifiedError(
                    category=ErrorCategory.RATE_LIMITED,
                    retry_after_seconds=retry_delay_s,
                    message=f"Gemini RESOURCE_EXHAUSTED, quotaId={quota_id!r} (per-minute "
                    f"quota): {message} Ordinary rate limit; retry after the stated delay. "
                    "(quotaId-based classification is UNVERIFIED as a stable contract.)",
                    is_definitively_classified=False,
                )

            # No quotaId (or an unrecognized one) to disambiguate with --
            # this is the honest, common case for this provider. Do NOT
            # guess which of RPM/TPM/RPD/spend caused this.
            return ClassifiedError(
                category=ErrorCategory.RATE_LIMITED,
                retry_after_seconds=retry_delay_s,
                message="Gemini RESOURCE_EXHAUSTED (429) with no quotaId to disambiguate "
                "daily-vs-per-minute-vs-spend. Treated as an ordinary rate limit as the "
                "conservative default (per official guidance: 'wait and retry'), but this "
                f"could be a daily quota that will not actually clear soon. Provider "
                f"message: {message} -- see docs/provider-notes.md.",
                is_definitively_classified=False,
            )

        if status in (400, 401, 403, 404, 416):
            return ClassifiedError(
                category=ErrorCategory.PERMANENT,
                message=f"Non-retryable client error (HTTP {status}, status={grpc_status!r}): {message}",
            )

        if status in (500, 503, 504):
            # VERIFIED: official table lists 503 as "temporarily overloaded
            # or down" -- Gemini's analogue of Anthropic's 529 / OpenAI's
            # 503. 500 and 504 are generic server-side failures.
            category = ErrorCategory.PROVIDER_OVERLOADED if status == 503 else ErrorCategory.TRANSIENT
            return ClassifiedError(
                category=category,
                message=f"Gemini server-side error (HTTP {status}, status={grpc_status!r}): {message}",
            )

        return ClassifiedError(
            category=ErrorCategory.UNKNOWN,
            message=f"Unrecognized status code {status} for Gemini API.",
            is_definitively_classified=False,
        )

    def classify_exception(self, exc: Exception, request: AdapterRequest) -> ClassifiedError:
        name = type(exc).__name__
        if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
            return ClassifiedError(
                category=ErrorCategory.TRANSIENT,
                message=f"Transport failure {name}: {exc}",
                is_definitively_classified=True,
            )
        return ClassifiedError(
            category=ErrorCategory.UNKNOWN,
            message=f"Unexpected exception {name}: {exc}",
            is_definitively_classified=False,
        )

    # ------------------------------------------------------- bad-pattern check

    def check_known_bad_patterns(self, response: httpx.Response) -> list[KnownBadPatternMatch]:
        """
        UNVERIFIED territory: I found no official documentation of a
        Gemini-specific "200 but actually wrong" behavior on
        generateContent (Anthropic's mid-stream SSE error and OpenAI's
        undocumented-here equivalent don't have a confirmed Gemini
        analogue). One real, VERIFIED structural signal exists and is
        checked: `generateContent` can return 200 with a `promptFeedback.
        blockReason` (content blocked before generation even started) and
        NO `candidates` array at all -- a genuine non-error 200 that
        nonetheless produced no usable output. This is check-able and
        documented in Gemini's safety-settings guide, so it ships;
        anything beyond it would be speculation.
        """
        matches: list[KnownBadPatternMatch] = []
        body = self._safe_json(response)
        if not isinstance(body, dict):
            return matches

        prompt_feedback = body.get("promptFeedback")
        has_candidates = bool(body.get("candidates"))
        if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason") and not has_candidates:
            matches.append(
                KnownBadPatternMatch(
                    pattern_name="blocked_before_generation_no_candidates",
                    description=(
                        "HTTP 200 but promptFeedback.blockReason="
                        f"{prompt_feedback.get('blockReason')!r} and no candidates were "
                        "returned -- the prompt was blocked before generation; there is no "
                        "model output to use."
                    ),
                    confidence=0.9,
                )
            )
        return matches

    # ---------------------------------------------------------- usage / req id

    def extract_usage(self, response: httpx.Response) -> UsageInfo | None:
        body = self._safe_json(response)
        if not isinstance(body, dict):
            return None
        usage = body.get("usageMetadata")
        if not isinstance(usage, dict):
            return None
        # VERIFIED: generateContent's response body DOES echo the model
        # via a top-level "modelVersion" field (confirmed across multiple
        # independent, mutually-consistent real response examples, e.g.
        # "modelVersion": "gemini-3.1-flash-lite"). Earlier drafting of
        # this adapter assumed this field was absent and planned to
        # thread the model in from the request instead -- that assumption
        # was wrong and has been corrected; extract_usage needs no
        # request context, consistent with the ProviderAdapter interface.
        model = body.get("modelVersion")
        if not isinstance(model, str):
            return None
        try:
            return UsageInfo(
                model=model,
                input_tokens=int(usage.get("promptTokenCount", 0)),
                output_tokens=int(usage.get("candidatesTokenCount", 0)),
                cached_input_tokens=int(usage.get("cachedContentTokenCount", 0)),
            )
        except (TypeError, ValueError):
            return None

    def extract_request_id(self, response: httpx.Response) -> str | None:
        """
        UNVERIFIED: I found no documented Gemini response header or body
        field equivalent to Anthropic's `request-id` or OpenAI's
        `x-request-id` for generateContent. Returns None honestly rather
        than guessing a header name that might not exist.
        """
        return None

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _safe_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except Exception:  # noqa: BLE001 -- deliberately broad; must never raise on garbage input (contract-tested)
            return None

    @staticmethod
    def _extract_quota_id(details: list) -> str | None:
        for d in details:
            if isinstance(d, dict) and d.get("@type", "").endswith("QuotaFailure"):
                violations = d.get("violations")
                if isinstance(violations, list) and violations:
                    first = violations[0]
                    if isinstance(first, dict):
                        qid = first.get("quotaId")
                        if isinstance(qid, str):
                            return qid
        return None

    @staticmethod
    def _extract_retry_delay_seconds(details: list) -> float | None:
        for d in details:
            if isinstance(d, dict) and d.get("@type", "").endswith("RetryInfo"):
                delay = d.get("retryDelay")
                if isinstance(delay, str) and delay.endswith("s"):
                    try:
                        return float(delay[:-1])
                    except ValueError:
                        return None
        return None
