"""
Anthropic (Claude) provider adapter.

All Anthropic-specific knowledge lives in this file and nowhere else.

Sources (see docs/provider-notes.md for VERIFIED/UNVERIFIED tagging):
- Rate-limit headers: anthropic-ratelimit-{requests,tokens,input-tokens,
  output-tokens}-{limit,remaining,reset}; reset is RFC 3339. VERIFIED.
- retry-after (seconds) on per-minute 429s. VERIFIED.
- Tier spend-cap: 429 rate_limit_error, NO retry-after, body
  error.details.error_code == "enforced_spend_limit_reached", message
  states the resume time. VERIFIED (official rate-limits page).
- User-set spend limit: HTTP 400 invalid_request_error whose message
  begins "You have reached your specified API usage limits" (or
  "...workspace API usage limits"). VERIFIED.
- Claude Code workspace limit can be a 429 WITH retry-after. VERIFIED.
- 529 overloaded_error = provider capacity, not caller quota. VERIFIED.
- Errors can arrive as SSE `event: error` after a 200. VERIFIED.
- Limits are per ORGANIZATION. VERIFIED (multiple sources agree; the
  official page frames limits per-org, with optional workspace limits).
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
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

ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"

_SPEND_CAP_CODE = "enforced_spend_limit_reached"
_USER_LIMIT_PREFIX = "you have reached your specified"
_RESUME_RE = re.compile(r"regain access on (\d{4}-\d{2}-\d{2}) at (\d{2}:\d{2}) UTC", re.IGNORECASE)

# Operations that are safe to retry blindly after a transport failure.
_IDEMPOTENT_OPS = {"messages.create"}


class AnthropicAdapter(ProviderAdapter):
    """Adapter for the Anthropic Messages API."""

    def __init__(self, api_key: str, *, http_client: httpx.AsyncClient | None = None) -> None:
        self._api_key = api_key
        self._http = http_client or httpx.AsyncClient(base_url=ANTHROPIC_API_BASE)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def get_limiting_unit(self) -> LimitingUnit:
        # KNOWN GAP (VERIFIED in Phase 3, see docs/provider-notes.md): the
        # combined anthropic-ratelimit-tokens-* headers report "the most
        # restrictive limit currently in effect", and the docs' own example
        # is a WORKSPACE-level cap -- a scope below the organization. The
        # response does not say which scope the numbers describe, and
        # LimitingUnit has no WORKSPACE member (not forced: the data to
        # populate it is absent). So this is right when no workspace cap is
        # set and wrong-but-unfixable-from-headers when one is binding; the
        # Quota Engine's "would adding more X help?" advice inherits that.
        return LimitingUnit.ORGANIZATION

    def quota_bucket(self, request: AdapterRequest) -> str:
        # VERIFIED (official rate-limits page, re-fetched for Phase 3):
        # RPM / ITPM / OTPM are measured "for each model class". Which
        # models share a class is NOT documented as a stable list, so this
        # does not encode it: the model string is used as a bucket that is
        # finer than or equal to the real one. That is the safe direction
        # for header-derived state -- every response re-states the true
        # bucket's figures, so two model strings in one class simply each
        # converge on the same numbers. (It would NOT be safe for
        # ledger-derived counting; Anthropic never uses that path.)
        model = request.payload.get("model") if isinstance(request.payload, dict) else None
        return model if isinstance(model, str) and model else "unknown-model"

    async def send(self, request: AdapterRequest) -> httpx.Response:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        if request.operation == "messages.create":
            return await self._http.post("/messages", json=request.payload, headers=headers)
        raise ValueError(
            f"AnthropicAdapter does not handle operation '{request.operation}'. "
            "Known: messages.create"
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

        reset_at = None
        reset_src = None
        raw_reset = h.get("anthropic-ratelimit-requests-reset")
        if raw_reset:
            try:
                # Python 3.11+ fromisoformat() natively accepts a trailing
                # "Z" -- no manual replacement needed on this project's
                # target (3.12).
                reset_at = datetime.fromisoformat(raw_reset)
                if reset_at.tzinfo is None:
                    reset_at = reset_at.replace(tzinfo=UTC)
                reset_at = reset_at.astimezone(UTC)
                reset_src = "timestamp"
            except ValueError:
                reset_at = None

        retry_after = None
        if "retry-after" in h:
            try:
                retry_after = float(h["retry-after"])
            except ValueError:
                retry_after = None

        rem_req = _int("anthropic-ratelimit-requests-remaining")
        lim_req = _int("anthropic-ratelimit-requests-limit")
        rem_tok = _int("anthropic-ratelimit-tokens-remaining")
        lim_tok = _int("anthropic-ratelimit-tokens-limit")

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
        details = err.get("details") if isinstance(err.get("details"), dict) else {}
        error_code = details.get("error_code")
        message = str(err.get("message", ""))

        if status == 529:
            return ClassifiedError(
                category=ErrorCategory.PROVIDER_OVERLOADED,
                message="Anthropic API is overloaded (529). Provider capacity, not a "
                f"quota issue; consider failing over. Provider message: {message}",
            )

        # User-set spend limit: documented as HTTP 400, NOT 429.
        if status == 400 and message.lower().startswith(_USER_LIMIT_PREFIX):
            return ClassifiedError(
                category=ErrorCategory.QUOTA_EXHAUSTED,
                resume_at=self._parse_resume(message),
                message=f"User-configured spend limit reached (HTTP 400): {message} "
                "Raise/remove the limit or wait for the stated resume time.",
            )

        if status == 429:
            retry_after = response.headers.get("retry-after")

            # Primary signal: the documented machine-readable error code.
            if error_code == _SPEND_CAP_CODE:
                return ClassifiedError(
                    category=ErrorCategory.QUOTA_EXHAUSTED,
                    resume_at=self._parse_resume(message),
                    message=f"Tier monthly spend cap reached: {message} Retrying fails "
                    "until access resumes; failover works now.",
                )

            if retry_after is not None:
                try:
                    secs = float(retry_after)
                except ValueError:
                    secs = None
                return ClassifiedError(
                    category=ErrorCategory.RATE_LIMITED,
                    retry_after_seconds=secs,
                    message=f"Rate limit exceeded (per-minute/token/acceleration): {message}",
                )

            # FALLBACK heuristic only: 429 with neither the documented code
            # nor retry-after. Docs say spend-cap 429s have no retry-after,
            # so this is *probably* a cap, but we cannot prove it. Say so.
            return ClassifiedError(
                category=ErrorCategory.QUOTA_EXHAUSTED,
                message="429 without retry-after and without a recognised error_code; "
                f"treated as probable spend cap (heuristic). Provider message: {message}",
                is_definitively_classified=False,
            )

        if status in (400, 401, 403, 404, 413):
            return ClassifiedError(
                category=ErrorCategory.PERMANENT,
                message=f"Non-retryable client error (HTTP {status}): {message}",
            )

        if status in (500, 504):
            return ClassifiedError(
                category=ErrorCategory.TRANSIENT,
                message=f"Server-side error (HTTP {status}); safe to retry shortly. "
                f"Provider message: {message}",
            )

        return ClassifiedError(
            category=ErrorCategory.UNKNOWN,
            message=f"Unrecognized status code {status} for Anthropic API.",
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
        matches: list[KnownBadPatternMatch] = []
        try:
            text = response.text
        except Exception:  # noqa: BLE001 -- deliberately broad; must never raise on garbage input (contract-tested)
            return matches

        # Case A: a plain JSON error object under a 2xx.
        body = self._safe_json(response)
        if isinstance(body, dict) and body.get("type") == "error":
            matches.append(
                KnownBadPatternMatch(
                    pattern_name="json_error_body_on_2xx",
                    description="HTTP 2xx but the body is an Anthropic error object.",
                    confidence=0.95,
                )
            )
            return matches

        # Case B: SSE stream with an `event: error` block after a 200.
        # (Defect 3: Phase 1 only did Case A, so this returned [] here.)
        if "event:" in text or "data:" in text:
            for event_name, data in self._iter_sse(text):
                if event_name == "error":
                    err_type = None
                    try:
                        parsed = json.loads(data)
                        err_type = (parsed.get("error") or {}).get("type")
                    except Exception:  # noqa: BLE001, S110 -- best-effort err_type extraction; the pattern match itself doesn't depend on it succeeding
                        pass
                    matches.append(
                        KnownBadPatternMatch(
                            pattern_name="streaming_error_after_200",
                            description="SSE stream returned HTTP 200 but contained an "
                            f"`event: error` block (error type: {err_type or 'unknown'}). "
                            "Output may be truncated; do not treat as success.",
                            confidence=0.97,
                        )
                    )
                    break
        return matches

    # ---------------------------------------------------------- usage / req id

    def extract_usage(self, response: httpx.Response) -> UsageInfo | None:
        body = self._safe_json(response)
        if not isinstance(body, dict):
            body = self._usage_from_sse(response)
        if not isinstance(body, dict):
            return None
        usage = body.get("usage")
        model = body.get("model")
        if not isinstance(usage, dict) or not isinstance(model, str):
            return None
        try:
            return UsageInfo(
                model=model,
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                cached_input_tokens=int(usage.get("cache_read_input_tokens", 0)),
            )
        except (TypeError, ValueError):
            return None

    def extract_request_id(self, response: httpx.Response) -> str | None:
        rid = response.headers.get("request-id")
        if rid:
            return rid
        body = self._safe_json(response)
        if isinstance(body, dict) and isinstance(body.get("request_id"), str):
            return body["request_id"]
        return None

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _safe_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except Exception:  # noqa: BLE001 -- deliberately broad; must never raise on garbage input (contract-tested)
            return None

    @staticmethod
    def _iter_sse(text: str):
        """Yield (event_name, data_string) pairs from an SSE body."""
        event_name = None
        data_lines: list[str] = []
        for raw in text.splitlines():
            line = raw.rstrip("\r")
            if line == "":
                if event_name is not None or data_lines:
                    yield event_name, "\n".join(data_lines)
                event_name, data_lines = None, []
            elif line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if event_name is not None or data_lines:
            yield event_name, "\n".join(data_lines)

    def _usage_from_sse(self, response: httpx.Response) -> dict | None:
        """Best-effort usage from a stream: model/input from message_start,
        final output_tokens from message_delta. None if not derivable."""
        try:
            text = response.text
        except Exception:  # noqa: BLE001 -- deliberately broad; must never raise on garbage input (contract-tested)
            return None
        model = None
        usage: dict[str, Any] = {}
        for name, data in self._iter_sse(text):
            try:
                parsed = json.loads(data)
            except Exception:  # noqa: BLE001, S112 -- one malformed SSE event must not abort extraction from the well-formed ones
                continue
            if name == "message_start":
                msg = parsed.get("message", {})
                model = msg.get("model", model)
                usage.update(msg.get("usage", {}) or {})
            elif name == "message_delta":
                usage.update(parsed.get("usage", {}) or {})
        if model and usage:
            return {"model": model, "usage": usage}
        return None

    @staticmethod
    def _parse_resume(message: str) -> datetime | None:
        m = _RESUME_RE.search(message or "")
        if not m:
            return None
        try:
            return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M").replace(
                tzinfo=UTC
            )
        except ValueError:
            return None
