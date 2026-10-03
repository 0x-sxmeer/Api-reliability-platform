"""
Generic REST / OAuth2 Adapter
Handles standard REST APIs (e.g. Stripe, Twilio, HubSpot) proving the platform's
non-LLM applicability.

ALL generic-REST knowledge lives here (The One Rule): HTTP-status-to-category
mapping, IETF Draft / X-RateLimit header parsing, request-id conventions. The
shared engines see only ProviderAdapter return types.

Request contract: ``request.extra`` must carry the transport details, since a
generic adapter cannot know any vendor's URL scheme:
    extra["url"]      (required) absolute http(s) URL to call
    extra["method"]   HTTP method, default "POST"
    extra["headers"]  additional headers dict (merged over auth headers)
    extra["params"]   query parameters dict
A request without a usable URL is an unknown operation and fails loudly with
ValueError before any network I/O -- never a fabricated success.
"""

from __future__ import annotations

import logging
import math
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

logger = logging.getLogger(__name__)


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    # Guard against nan/inf sneaking through float() on hostile input.
    if not math.isfinite(result):
        return None
    return result


def _parse_reset(raw: str | None) -> datetime | None:
    """Normalize an epoch-seconds or ISO-8601 reset value to an aware UTC
    datetime, or None if absent/unparseable."""
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        # float() first so fractional epoch seconds ("1767225600.5") parse;
        # int() alone would raise ValueError and we'd misreport them as None.
        return datetime.fromtimestamp(float(value), tz=UTC)
    except (ValueError, OverflowError, OSError):
        pass
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


class GenericRestAdapter(ProviderAdapter):
    """
    A generic adapter for standard REST APIs.

    Maps plain HTTP semantics to the gateway's reliability primitives.
    Subclasses (a future Stripe/Twilio-specific adapter) can override any
    method to add vendor knowledge; this base stays deliberately generic.
    """

    def __init__(self, *, api_key: str | None = None, client: httpx.AsyncClient | None = None) -> None:
        self._api_key = api_key
        self._client = client

    @property
    def provider_name(self) -> str:
        return "generic_rest"

    def get_limiting_unit(self) -> LimitingUnit:
        # Standard REST APIs typically limit per API key unless a specific
        # vendor says otherwise; subclasses can override. Must return the
        # enum member (the Quota Engine compares it identity-wise), not a
        # bare string.
        return LimitingUnit.KEY

    def quota_bucket(self, request: AdapterRequest) -> str:
        # Opaque, stable, deterministic, never raises. Prefer finer than
        # the provider's real bucket: split by endpoint when the caller
        # gave one, else a fixed fallback.
        url = request.extra.get("url")
        if isinstance(url, str) and url:
            return url
        return "default"

    def parse_rate_limit_headers(self, response: httpx.Response) -> RateLimitSnapshot:
        """
        Parse whichever rate-limit header family the upstream actually uses:
        the IETF draft (RateLimit-Limit/-Remaining/-Reset) or the widely
        deployed X-RateLimit-* family. Anything unparseable becomes None —
        never a raise, never fake data. reset_at is normalized to an
        absolute UTC instant; whatever its wire form, it is measured
        against the provider's clock, hence source "timestamp".
        """
        headers = response.headers
        # Per-header fallback, NOT "if draft limit missing use X family for
        # everything": a provider may legitimately send only one member of a
        # family (e.g. bare RateLimit-Reset), and the old block-level gate
        # silently discarded it whenever RateLimit-Limit was absent.
        limit = _parse_int(headers.get("ratelimit-limit"))
        if limit is None:
            limit = _parse_int(headers.get("x-ratelimit-limit"))
        remaining = _parse_int(headers.get("ratelimit-remaining"))
        if remaining is None:
            remaining = _parse_int(headers.get("x-ratelimit-remaining"))
        reset_raw = headers.get("ratelimit-reset") or headers.get("x-ratelimit-reset")

        retry_after = _parse_float(headers.get("retry-after"))
        reset_at = _parse_reset(reset_raw)

        has_any = limit is not None or remaining is not None or reset_at is not None or retry_after is not None
        if limit is not None and remaining is not None:
            signal_quality = SignalQuality.FULL
        elif has_any:
            signal_quality = SignalQuality.PARTIAL
        else:
            signal_quality = SignalQuality.NONE

        return RateLimitSnapshot(
            limiting_unit=self.get_limiting_unit(),
            signal_quality=signal_quality,
            requests_remaining=remaining,
            requests_limit=limit,
            reset_at=reset_at,
            reset_at_source="timestamp" if reset_at is not None else None,
            retry_after_seconds=retry_after,
            raw_headers=dict(headers.items()),
        )

    def classify_error(self, response: httpx.Response) -> ClassifiedError:
        """
        Map standard HTTP status codes onto the gateway's ErrorCategory
        taxonomy. Must never raise on any status/body (contract-tested for
        every code 400-599 including malformed JSON bodies).
        """
        status = response.status_code
        retry_after = _parse_float(response.headers.get("retry-after"))

        if status == 429:
            return ClassifiedError(
                category=ErrorCategory.RATE_LIMITED,
                message="Rate limited",
                retry_after_seconds=retry_after,
            )
        if status == 402:
            return ClassifiedError(
                category=ErrorCategory.QUOTA_EXHAUSTED,
                message="Payment required / usage cap reached",
                # No documented reset rule for generic 402s: resume time is
                # unknown, so the flag is False and the Quota Engine takes
                # its provisional/probe path.
                is_definitively_classified=False,
            )
        if status in (401, 403):
            return ClassifiedError(
                category=ErrorCategory.PERMANENT,
                message="Authentication/authorization failed",
            )
        if 400 <= status < 500:
            return ClassifiedError(
                category=ErrorCategory.PERMANENT,
                message=f"Client error ({status})",
            )
        if status in (502, 503, 504):
            return ClassifiedError(
                category=ErrorCategory.TRANSIENT,
                message=f"Transient server error ({status})",
                retry_after_seconds=retry_after,
            )
        if status == 500:
            return ClassifiedError(
                category=ErrorCategory.TRANSIENT,
                message="Server error (500)",
                retry_after_seconds=retry_after,
                is_definitively_classified=False,
            )
        if 500 <= status < 600:
            return ClassifiedError(
                category=ErrorCategory.TRANSIENT,
                message=f"Server error ({status})",
                is_definitively_classified=False,
            )
        return ClassifiedError(
            category=ErrorCategory.UNKNOWN,
            message=f"Unclassifiable response status {status}",
            is_definitively_classified=False,
        )

    def check_known_bad_patterns(self, response: httpx.Response) -> list[KnownBadPatternMatch]:
        # A generic REST adapter has no library of provider-specific
        # "looks fine but isn't" signatures (unlike LLMs with grounding
        # refusals). Vendor subclasses can add their own. Never raises.
        return []

    def classify_exception(self, exc: Exception, request: AdapterRequest) -> ClassifiedError:
        """
        Transport-level failures on generic REST calls are classified
        TRANSIENT *provisionally* (is_definitively_classified=False):
        unlike a read-timeout on an idempotent LLM generation, we cannot
        know whether this operation is idempotent, so the Routing Engine
        should be conservative about blind retries.
        """
        return ClassifiedError(
            category=ErrorCategory.TRANSIENT,
            message=f"{type(exc).__name__}: {exc}",
            is_definitively_classified=False,
        )

    def extract_usage(self, response: httpx.Response) -> UsageInfo | None:
        # Non-LLM REST APIs have no token usage; cost accounting for them
        # is per-call pricing configuration, not response-derived. Must
        # never raise.
        return None

    def extract_request_id(self, response: httpx.Response) -> str | None:
        headers = response.headers
        for name in ("x-request-id", "request-id", "x-amzn-requestid", "x-correlation-id"):
            value = headers.get(name)
            if value:
                return value
        return None

    async def send(self, request: AdapterRequest) -> httpx.Response:
        """
        Exactly one HTTP attempt against the URL supplied in
        extra["url"]. An operation we cannot resolve to a URL is a
        programming error and raises ValueError synchronously (before any
        I/O) — the contract requires unknown operations to fail loudly,
        never to fabricate a 200. Retry policy is the Routing Engine's
        job; this makes a single attempt.
        """
        url = request.extra.get("url")
        if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
            raise ValueError(
                f"GenericRestAdapter cannot perform operation {request.operation!r}: "
                "request.extra['url'] must be an absolute http(s) URL"
            )
        method = str(request.extra.get("method", "POST")).upper()
        headers: dict[str, str] = dict(request.extra.get("headers") or {})
        if self._api_key and "authorization" not in {k.lower() for k in headers}:
            headers["Authorization"] = f"Bearer {self._api_key}"

        kwargs: dict[str, Any] = {"headers": headers}
        params = request.extra.get("params")
        if params:
            kwargs["params"] = params
        if method in ("POST", "PUT", "PATCH"):
            kwargs["json"] = request.payload

        if self._client is not None:
            return await self._client.request(method, url, **kwargs)
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            return await client.request(method, url, **kwargs)
