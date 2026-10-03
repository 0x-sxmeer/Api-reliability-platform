"""
The Provider Adapter contract.

Every vendor integration (Anthropic today; OpenAI, Gemini, Stripe,
Twilio, etc. in later phases) implements ProviderAdapter. This is the
ONLY place vendor-specific knowledge is allowed to live. The four
shared engines (Quota, Routing, Budget, Validator) and the Ledger all
operate purely in terms of this interface's return types — they never
import or reference a specific provider.

If a future engine needs to ask "is this Anthropic?" to decide what to
do, that's a sign the adapter interface is missing a method — the fix
is to add a method here, not to leak vendor checks into the engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from gateway.core.types import (
    ClassifiedError,
    KnownBadPatternMatch,
    LimitingUnit,
    QuotaWindow,
    RateLimitSnapshot,
    UsageInfo,
)


class ProviderAdapter(ABC):
    """
    Abstract base class for a single provider integration.

    Design note: methods that inspect a response (parse_rate_limit_headers,
    classify_error, check_known_bad_patterns) take the raw httpx.Response
    rather than something pre-parsed, because different providers put
    the signal in different places — headers, status code, AND body, in
    varying combinations. Forcing a narrower input type here would mean
    some future provider can't implement the interface properly.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short, stable identifier used as the 'provider' field in every
        ledger event, e.g. 'anthropic'. Must never change once used in
        production data, since the ledger keys on it."""
        raise NotImplementedError

    @abstractmethod
    def get_limiting_unit(self) -> LimitingUnit:
        """
        What scope this provider actually enforces limits against.

        This is a fixed fact about the provider (Anthropic = organization,
        AWS = varies by service, hence AWS's adapter would need a more
        nuanced answer — see the note in the Anthropic adapter for how
        a single-scope provider keeps this simple). The Quota Engine
        (Phase 3) uses this to warn the caller when a scaling strategy
        (e.g. "just add more keys") won't actually help.
        """
        raise NotImplementedError

    @abstractmethod
    def quota_bucket(self, request: AdapterRequest) -> str:
        """
        Which rate-limit counters this request is metered against, as an
        opaque, stable string (Phase 3; forced by all three providers).

        get_limiting_unit() says WHOSE limit applies (org / project / key);
        this says WHICH of that owner's counters. Every current provider
        partitions its limits further, by model: Anthropic per model class,
        OpenAI per model (with org-defined shared pools), Gemini per model.
        A response's rate-limit headers describe only the bucket that
        request drew from, so an engine that keyed its state by provider
        alone would let a cheap model's headroom overwrite an expensive
        model's. The Quota Engine keys by (provider_name, this string) and
        never interprets it.

        Returning a bucket FINER than the provider's real one is safe for
        header-derived state (each response re-states the true figures);
        COARSER is not. When unsure, be finer. Must be deterministic and
        must NEVER raise: the executor calls it on every request,
        including malformed ones whose failure belongs in the ledger, not
        in an exception. Return a fixed fallback string if the bucket
        can't be determined.
        """
        raise NotImplementedError

    def quota_windows(self) -> tuple[QuotaWindow, ...]:
        """
        The limit windows this provider enforces that the Quota Engine
        should be able to count against FROM THE LEDGER. Default: none.

        Only a provider that gives no usable rate-limit headers needs to
        override this (today: Gemini). For providers whose headers carry
        limit, remaining and reset, the engine reads those and there is
        nothing to count. Declares window SHAPES only; the numeric limits
        are operator configuration (see QuotaWindow). Concrete rather than
        abstract on purpose: requiring every adapter to declare windows
        nobody would read is exactly the speculative surface this project
        avoids.
        """
        return ()

    @abstractmethod
    def parse_rate_limit_headers(self, response: httpx.Response) -> RateLimitSnapshot:
        """
        Extract this provider's rate-limit signal from a response into
        the normalized RateLimitSnapshot shape.

        Must be called on EVERY response, not just ones that returned
        a 429 — the whole point (per the research) is to track headroom
        proactively rather than reactively discovering limits via errors.
        """
        raise NotImplementedError

    @abstractmethod
    def classify_error(self, response: httpx.Response) -> ClassifiedError:
        """
        Turn a non-2xx (or suspicious) response into one of the shared
        ErrorCategory buckets. This is where a provider's specific status
        codes get translated into the gateway's provider-agnostic model —
        e.g. Anthropic's 529 becomes PROVIDER_OVERLOADED, not just
        "some 5xx error."
        """
        raise NotImplementedError

    @abstractmethod
    def check_known_bad_patterns(self, response: httpx.Response) -> list[KnownBadPatternMatch]:
        """
        Check a nominally-successful (2xx) response against this
        provider's library of "looks fine but isn't" signatures.
        Returns an empty list if nothing matched — most calls will hit
        this path, so it should be cheap.
        """
        raise NotImplementedError

    @abstractmethod
    def classify_exception(self, exc: Exception, request: AdapterRequest) -> ClassifiedError:
        """
        Map a transport-level exception (timeout, connect error, ...)
        raised by send() to a ClassifiedError.

        Lives on the adapter, not the executor, because the RIGHT answer
        is provider- and operation-specific: a ReadTimeout on a long LLM
        generation is TRANSIENT and idempotent to retry, whereas a
        ReadTimeout on a payment capture is *ambiguous* (the charge may
        have succeeded) and must never be blindly retried. Only the
        adapter knows which operations are idempotent. Must never raise.
        """
        raise NotImplementedError

    @abstractmethod
    def extract_usage(self, response: httpx.Response) -> UsageInfo | None:
        """
        Pull model + token counts out of a SUCCESSFUL response, or None
        if this response carries no usage block. Each vendor puts these
        in a different place, so only the adapter can do this. Must
        never raise (a malformed body returns None).
        """
        raise NotImplementedError

    @abstractmethod
    def extract_request_id(self, response: httpx.Response) -> str | None:
        """The provider's own request id (for support tickets and for
        reconciling PENDING events later), or None. Must never raise."""
        raise NotImplementedError

    @abstractmethod
    async def send(self, request: AdapterRequest) -> httpx.Response:
        """
        Actually perform the HTTP call to the provider. Kept as a thin
        wrapper (auth headers, base URL) — NOT where retry/backoff logic
        lives. Retry policy is the Routing Engine's job (Phase 5); this
        method makes exactly one attempt.
        """
        raise NotImplementedError


class AdapterRequest:
    """
    A provider-agnostic description of a single call, before an adapter
    translates it into an actual HTTP request.

    Kept intentionally minimal for Phase 1 — just enough to make a real
    call. Phase 2, when a second and third adapter get built, is the
    right time to see what fields actually need to be here versus what
    can stay adapter-specific (passed via `extra`).
    """

    def __init__(
        self,
        *,
        operation: str,
        payload: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.operation = operation  # e.g. "messages.create" — adapter interprets this
        self.payload = payload
        self.extra = extra or {}
