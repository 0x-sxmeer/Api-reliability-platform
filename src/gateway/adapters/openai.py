"""
OpenAI provider adapter.

All OpenAI-specific knowledge lives in this file and nowhere else.
See docs/provider-notes.md ("OpenAI" section) for full VERIFIED/
UNVERIFIED sourcing on every claim below — this docstring summarizes,
it doesn't replace, that file.

Key facts this adapter encodes (VERIFIED against developers.openai.com
official docs, fetched this session, unless noted):
- Rate-limit headers: x-ratelimit-{limit,remaining,reset}-{requests,tokens},
  PLUS project-scoped x-ratelimit-{limit,remaining,reset}-project-tokens
  which "may be present when a project-scoped limit applies." Since
  Phase 3 these are promoted to RateLimitSnapshot.additional_scopes."
- reset-* headers are DURATION STRINGS ("1s", "6m0s", "59.5s"), not
  timestamps — opposite of Anthropic. Converted to absolute UTC using
  response-receipt-time + duration (see RateLimitSnapshot.reset_at_source
  for the clock-skew caveat this implies).
- Rate limits apply at BOTH the organization level AND the project level
  simultaneously — this is genuinely dual-scoped, unlike Anthropic
  (org only) or Gemini (project only). get_limiting_unit() reports
  ORGANIZATION as the primary/outer scope; the project-level headers
  are exposed as RateLimitSnapshot.additional_scopes (Phase 3 -- the
  Quota Engine needed the binding scope, see docs/interface-changes.md)
  and are still also preserved in raw_headers.
- 429 rate_limit_error/rate_limit_exceeded: ordinary per-minute/token
  limit, ALWAYS check `code`, never assume 429 == rate limit.
- 429 insufficient_quota: credit/spend exhausted. Retrying is provably
  futile; failover to another provider works. -> QUOTA_EXHAUSTED.
  (No documented resume_at for this one, unlike Anthropic's spend cap.)
- organization_usage_limit_exceeded: OpenAI-assigned MONTHLY usage cap
  (distinct from a configured spend limit). Also -> QUOTA_EXHAUSTED.
- 503 service_unavailable_error/server_is_overloaded: provider capacity,
  analogous to Anthropic's 529. -> PROVIDER_OVERLOADED.
- A 2026 documented migration moved SOME overload-adjacent throttling
  from 503/slow_down to 429/slow_down on some endpoints. This adapter
  checks the `code` field ("slow_down") ahead of assuming 429 always
  means ordinary rate-limiting, so it is migration-aware rather than
  hard-coded to the pre-migration shape.
- APIConnectionError-class failures are transport-level, not HTTP status
  codes -- handled in classify_exception, not classify_error.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from gateway.core.adapter import AdapterRequest, ProviderAdapter
from gateway.core.types import (
    ClassifiedError,
    ErrorCategory,
    KnownBadPatternMatch,
    LimitingUnit,
    RateLimitSnapshot,
    SignalQuality,
    UsageInfo,
)

OPENAI_API_BASE = "https://api.openai.com/v1"

# error.code values (JSON body) that mean "quota/credit gone, retry is
# futile, failover works" -- as opposed to error.code == "rate_limit_exceeded"
# or "slow_down", which mean "back off, retry works". VERIFIED distinct
# values per docs/provider-notes.md.
_QUOTA_EXHAUSTED_CODES = {"insufficient_quota", "organization_usage_limit_exceeded"}

# error.code values on a 429 that are still an ordinary, retry-able rate
# limit -- "slow_down" is the 2026-migration code for rapid-traffic-increase
# throttling that used to be a 503; treating it as anything other than
# RATE_LIMITED would be wrong per the official migration guidance.
_RATE_LIMIT_CODES = {"rate_limit_exceeded", "slow_down"}

_DURATION_RE = re.compile(
    r"(?:(?P<h>\d+(?:\.\d+)?)h)?(?:(?P<m>\d+(?:\.\d+)?)m)?(?:(?P<s>\d+(?:\.\d+)?)s)?$"
)

_IDEMPOTENT_OPS = {"chat.completions.create"}


def _parse_openai_duration(value: str) -> float | None:
    """
    Parse OpenAI's Go-style duration strings ("6m0s", "1s", "59.5s",
    "1h2m3s") into seconds. Returns None on anything unparseable rather
    than raising or guessing -- this feeds reset_at, and a wrong reset
    time is worse than a missing one (a consumer might schedule a retry
    for the wrong instant and either hammer the API early or wait too
    long).

    VERIFIED format examples ("1s", "6m0s") per docs/provider-notes.md;
    the hour component is UNVERIFIED (no observed example used one, but
    the format is consistent with Go's time.Duration.String(), which
    OpenAI's own reset-time helper explicitly parses via
    time.ParseDuration in a third-party client -- included defensively,
    not fabricated).
    """
    if not value:
        return None
    m = _DURATION_RE.match(value.strip())
    if not m or not any(m.groups()):
        return None
    hours = float(m.group("h") or 0)
    minutes = float(m.group("m") or 0)
    seconds = float(m.group("s") or 0)
    total = hours * 3600 + minutes * 60 + seconds
    return total if total > 0 or value.strip() in ("0s", "0m0s") else None


class OpenAIAdapter(ProviderAdapter):
    """Adapter for the OpenAI Chat Completions API."""

    def __init__(self, api_key: str, *, http_client: httpx.AsyncClient | None = None) -> None:
        self._api_key = api_key
        if http_client is not None:
            self._http = http_client
        else:
            from gateway.devmode import make_http_client

            self._http = make_http_client(OPENAI_API_BASE)

    @property
    def provider_name(self) -> str:
        return "openai"

    def get_limiting_unit(self) -> LimitingUnit:
        # OpenAI is genuinely dual-scoped (org AND project limits can
        # both apply to the same call -- VERIFIED, see module docstring).
        # This is a fixed fact, so it reports the OUTER/primary scope; a
        # static answer cannot say which scope is BINDING right now. That
        # is data-dependent, and it is why parse_rate_limit_headers()
        # also returns the project scope in `additional_scopes`: the
        # Quota Engine takes the tightest scope per response. A caller
        # who only checks this method and assumes "add more projects"
        # will fully solve throughput would be wrong if the ORG-level
        # limit is what's binding -- use QuotaAssessment.binding instead.
        return LimitingUnit.ORGANIZATION

    def quota_bucket(self, request: AdapterRequest) -> str:
        # VERIFIED (official rate-limits page): limits vary by model, and
        # models listed under a "shared limit" on the org's limits page
        # share one pool. Pool membership is per-organization
        # configuration, not something this adapter can know, so the model
        # string is used: finer than or equal to the real bucket, which is
        # the safe direction for header-derived state (see
        # ProviderAdapter.quota_bucket).
        model = request.payload.get("model") if isinstance(request.payload, dict) else None
        return model if isinstance(model, str) and model else "unknown-model"

    async def send(self, request: AdapterRequest) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if request.operation == "chat.completions.create":
            return await self._http.post(
                "/chat/completions", json=request.payload, headers=headers
            )
        raise ValueError(
            f"OpenAIAdapter does not handle operation '{request.operation}'. "
            "Known: chat.completions.create"
        )

    # ---------------------------------------------------------------- headers

    def parse_rate_limit_headers(self, response: httpx.Response) -> RateLimitSnapshot:
        h = response.headers

        def _int(key: str) -> int | None:
            v = h.get(key)
            try:
                return int(v) if v is not None else None
            except ValueError:
                return None

        # Prefer token-dimension fields as "the" remaining/limit figures,
        # matching the Anthropic adapter's convention of surfacing the
        # most-restrictive/most-commonly-binding dimension while keeping
        # every raw header available for anyone who needs the full detail
        # (including the project-scoped variants, which live in
        # raw_headers only -- see get_limiting_unit's note).
        rem_req = _int("x-ratelimit-remaining-requests")
        lim_req = _int("x-ratelimit-limit-requests")
        rem_tok = _int("x-ratelimit-remaining-tokens")
        lim_tok = _int("x-ratelimit-limit-tokens")

        received_at = datetime.now(UTC)
        reset_at = None
        reset_src = None
        raw_reset = h.get("x-ratelimit-reset-tokens") or h.get("x-ratelimit-reset-requests")
        if raw_reset:
            secs = _parse_openai_duration(raw_reset)
            if secs is not None:
                # Computed from OUR receipt time, not OpenAI's clock --
                # this is the "duration" branch of the clock-skew caveat
                # documented on RateLimitSnapshot.reset_at_source.
                reset_at = received_at + timedelta(seconds=secs)
                reset_src = "duration"

        # Project scope (Phase 3). VERIFIED (official rate-limits page and
        # API reference, re-fetched for Phase 3): THREE project-scoped
        # headers exist -- limit, remaining, reset -- "present when a
        # project-scoped token limit applies". (docs/provider-notes.md
        # previously listed only remaining and reset; corrected there.)
        # Tokens only: no project-scoped REQUEST headers are documented.
        # This scope is enforced SIMULTANEOUSLY with the one above, so it
        # rides along as an additional scope rather than replacing it.
        # ASSUMPTION (UNVERIFIED): the un-suffixed x-ratelimit-* headers
        # describe the organization-level figures. The docs say project
        # headers are separate and additional but never state what the
        # un-suffixed ones are scoped to; get_limiting_unit() has always
        # reported ORGANIZATION for them.
        proj_rem = _int("x-ratelimit-remaining-project-tokens")
        proj_lim = _int("x-ratelimit-limit-project-tokens")
        proj_reset_at = None
        proj_reset_src = None
        raw_proj_reset = h.get("x-ratelimit-reset-project-tokens")
        if raw_proj_reset:
            proj_secs = _parse_openai_duration(raw_proj_reset)
            if proj_secs is not None:
                proj_reset_at = received_at + timedelta(seconds=proj_secs)
                proj_reset_src = "duration"

        additional_scopes: tuple[RateLimitSnapshot, ...] = ()
        if proj_rem is not None or proj_lim is not None or proj_reset_at is not None:
            # FULL here means what SignalQuality documents: remaining +
            # limit + reset for the (only) dimension this scope exposes.
            proj_full = proj_rem is not None and proj_lim is not None and proj_reset_at is not None
            additional_scopes = (
                RateLimitSnapshot(
                    limiting_unit=LimitingUnit.PROJECT,
                    signal_quality=SignalQuality.FULL if proj_full else SignalQuality.PARTIAL,
                    tokens_remaining=proj_rem,
                    tokens_limit=proj_lim,
                    reset_at=proj_reset_at,
                    reset_at_source=proj_reset_src,
                ),
            )

        retry_after = None
        if "retry-after" in h:
            try:
                retry_after = float(h["retry-after"])
            except ValueError:
                retry_after = None

        if rem_req is not None and lim_req is not None and rem_tok is not None and lim_tok is not None:
            quality = SignalQuality.FULL
        elif any(v is not None for v in (rem_req, lim_req, rem_tok, lim_tok)) or retry_after is not None:
            quality = SignalQuality.PARTIAL
        else:
            quality = SignalQuality.NONE

        return RateLimitSnapshot(
            limiting_unit=self.get_limiting_unit(),
            signal_quality=quality,
            requests_remaining=rem_req,
            requests_limit=lim_req,
            tokens_remaining=rem_tok,
            tokens_limit=lim_tok,
            reset_at=reset_at,
            reset_at_source=reset_src,
            retry_after_seconds=retry_after,
            raw_headers=dict(h),
            additional_scopes=additional_scopes,
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
        code = err.get("code")
        err_type = err.get("type")
        message = str(err.get("message", ""))

        # 503: provider capacity, analogous to Anthropic's 529. VERIFIED.
        if status == 503:
            return ClassifiedError(
                category=ErrorCategory.PROVIDER_OVERLOADED,
                message=f"OpenAI service unavailable (503, type={err_type!r}). "
                "Provider capacity, not caller quota; consider failing over.",
            )

        if status == 429:
            retry_after = response.headers.get("retry-after")
            retry_after_s = None
            if retry_after is not None:
                try:
                    retry_after_s = float(retry_after)
                except ValueError:
                    retry_after_s = None

            if code in _QUOTA_EXHAUSTED_CODES:
                return ClassifiedError(
                    category=ErrorCategory.QUOTA_EXHAUSTED,
                    # No documented resume_at for OpenAI's quota-exhausted
                    # codes (unlike Anthropic's spend cap, which states a
                    # resume time in the message) -- VERIFIED-absence,
                    # left as None rather than guessed.
                    message=f"OpenAI quota exhausted (code={code!r}): {message} "
                    "Retrying will not succeed; failover to another provider works now.",
                )

            if code in _RATE_LIMIT_CODES or code is not None:
                # code is not None but also not in either known set: still
                # safer to treat an explicit-but-unrecognized 429 code as
                # an ordinary rate limit (the conservative default per
                # OpenAI's own docs, which say to back off) than to guess
                # QUOTA_EXHAUSTED -- but mark it as not definitively
                # classified so the Routing Engine can be more cautious.
                return ClassifiedError(
                    category=ErrorCategory.RATE_LIMITED,
                    retry_after_seconds=retry_after_s,
                    message=f"OpenAI rate limit (code={code!r}): {message}",
                    is_definitively_classified=code in _RATE_LIMIT_CODES,
                )

            # No code at all: treat as rate limit (OpenAI's documented
            # default guidance for a bare 429), but flag the uncertainty.
            return ClassifiedError(
                category=ErrorCategory.RATE_LIMITED,
                retry_after_seconds=retry_after_s,
                message=f"429 with no recognized error.code; treated as rate limit "
                f"(heuristic). Provider message: {message}",
                is_definitively_classified=False,
            )

        if status in (400, 401, 403, 404, 413, 416):
            return ClassifiedError(
                category=ErrorCategory.PERMANENT,
                message=f"Non-retryable client error (HTTP {status}, type={err_type!r}): {message}",
            )

        if status in (500, 502, 504):
            return ClassifiedError(
                category=ErrorCategory.TRANSIENT,
                message=f"Server-side error (HTTP {status}); safe to retry shortly. "
                f"Provider message: {message}",
            )

        return ClassifiedError(
            category=ErrorCategory.UNKNOWN,
            message=f"Unrecognized status code {status} for OpenAI API.",
            is_definitively_classified=False,
        )

    def classify_exception(self, exc: Exception, request: AdapterRequest) -> ClassifiedError:
        name = type(exc).__name__
        if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
            idempotent = request.operation in _IDEMPOTENT_OPS
            return ClassifiedError(
                category=ErrorCategory.TRANSIENT,
                message=f"Transport failure {name}: {exc}"
                + ("" if idempotent else " (operation not known idempotent; do not blind-retry)"),
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
        UNVERIFIED as a documented contract: I found no official OpenAI
        page describing a "200 but actually an error" case analogous to
        Anthropic's mid-stream SSE error, or Gemini's undifferentiated
        RESOURCE_EXHAUSTED. Rather than invent a speculative pattern to
        fill this method, it ships with zero patterns and says so --
        an honest empty list beats a fabricated heuristic.
        """
        return []

    # ---------------------------------------------------------- usage / req id

    def extract_usage(self, response: httpx.Response) -> UsageInfo | None:
        body = self._safe_json(response)
        if not isinstance(body, dict):
            return None
        usage = body.get("usage")
        model = body.get("model")
        if not isinstance(usage, dict) or not isinstance(model, str):
            return None
        try:
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
            # cached tokens live nested under prompt_tokens_details in the
            # current Chat Completions response shape.
            details = usage.get("prompt_tokens_details")
            cached = int(details.get("cached_tokens", 0)) if isinstance(details, dict) else 0
            return UsageInfo(
                model=model,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                cached_input_tokens=cached,
            )
        except (TypeError, ValueError):
            return None

    def extract_request_id(self, response: httpx.Response) -> str | None:
        # VERIFIED header name (x-request-id) per multiple corroborating
        # sources in docs/provider-notes.md (community-reproduced curl
        # output, consistent across all of them).
        rid = response.headers.get("x-request-id")
        return rid if rid else None

    @staticmethod
    def _safe_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except Exception:  # noqa: BLE001 -- deliberately broad; must never raise on garbage input (contract-tested)
            return None
