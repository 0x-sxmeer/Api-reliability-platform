"""
CallExecutor — the orchestration Phase 1 left stranded in examples/demo.py
(defect 1), now a real library class, plus the fix for defect 2 (transport
exceptions never reaching the ledger).

This is deliberately the SIMPLEST thing that can be called "orchestration":
send once, classify whatever happened, write exactly one event. No retry,
no backoff, no fallback — that is Phase 5's Routing Engine, built on top
of this, not folded into it. Keeping this class ignorant of retry policy
is what lets Phase 5 wrap or replace it without a rewrite.

Nothing here references a specific provider. Every provider-specific
decision (is this exception retryable, what does this response cost,
what's the request id) is delegated to the ProviderAdapter passed in.
"""

from __future__ import annotations

import time

import httpx

from gateway.core.adapter import AdapterRequest, ProviderAdapter
from gateway.core.pricing import compute_cost_usd
from gateway.core.types import (
    CallOutcome,
    ClassifiedError,
    KnownBadPatternMatch,
    RateLimitSnapshot,
)
from gateway.ledger.events import GatewayEvent
from gateway.ledger.store import LedgerStore


class CallResult:
    """
    What execute() hands back to the caller — everything they might want
    to react to, in one place, so they don't have to re-derive it from
    the GatewayEvent that was just written.
    """

    def __init__(
        self,
        *,
        event: GatewayEvent,
        response: httpx.Response | None,
        exception: Exception | None,
        rate_limit: RateLimitSnapshot | None = None,
        classified_error: ClassifiedError | None = None,
        bad_patterns: list[KnownBadPatternMatch] | None = None,
    ) -> None:
        self.event = event
        # The raw response, if one was received — None on a transport
        # failure (defect 2's case), since there is nothing to inspect.
        self.response = response
        # The raw exception, if send() raised one — None on any response
        # (success or HTTP-level error), since httpx returning a Response
        # at all means the transport layer did its job.
        self.exception = exception
        # Phase 3: the two objects _handle_response()/the exception path
        # already computed and used to discard after copying a few fields
        # into the event. The Quota Engine's live feed comes from these --
        # the ledger keeps only raw_headers and the classification
        # MESSAGE, so the parsed reset time, retry-after and resume_at are
        # unrecoverable from it. Both default to None so nothing that
        # constructs a CallResult needed to change.
        #   rate_limit: parsed from the response's headers; None when no
        #     response arrived (transport failure).
        #   classified_error: from classify_error()/classify_exception();
        #     None on a 2xx/3xx outcome.
        self.rate_limit = rate_limit
        self.classified_error = classified_error
        # Phase 4: the KnownBadPatternMatch objects _handle_response()
        # already computed but discarded after extracting pattern names.
        # The Response Validator needs the confidence scores these carry;
        # the ledger keeps only the names (strings). Same pattern as
        # Phase 3 adding rate_limit and classified_error above.
        #   bad_patterns: from check_known_bad_patterns(); None when no
        #     response arrived (transport failure) or on an HTTP error
        #     (check is only run on <400 responses).
        self.bad_patterns = bad_patterns or []

    @property
    def succeeded(self) -> bool:
        return self.event.outcome == CallOutcome.SUCCESS


class CallExecutor:
    """
    Ties one ProviderAdapter to one LedgerStore under one identity_key.

    One executor instance per (adapter, identity) is the expected usage —
    it's cheap to construct, so callers making calls under different
    identities (different teams, different API keys) should use separate
    executor instances rather than passing identity_key per call. This
    mirrors how Phase 6's Budget Engine will need to reason about spend
    per identity: keeping identity fixed per executor means that engine
    can wrap an executor 1:1 rather than threading identity through every
    call site.
    """

    def __init__(self, adapter: ProviderAdapter, ledger: LedgerStore, *, identity_key: str) -> None:
        self._adapter = adapter
        self._ledger = ledger
        self._identity_key = identity_key
        
    @property
    def identity_key(self) -> str:
        return self._identity_key

    async def execute(self, request: AdapterRequest) -> CallResult:
        """
        Send the request exactly once, classify whatever came back
        (a response OR an exception), write exactly one GatewayEvent,
        and return a CallResult. Never raises: a transport exception is
        caught here (defect 2's fix) rather than propagated, because the
        whole point is that the ledger must record what happened even
        when "what happened" is "the network failed," not just when the
        provider answered.
        """
        # Must-never-raise by the adapter contract, so it is safe to call
        # outside the try below (a malformed request's failure belongs in
        # the ledger, not in an exception from here).
        bucket = self._adapter.quota_bucket(request)
        start = time.monotonic()
        try:
            response = await self._adapter.send(request)
        except Exception as exc:  # noqa: BLE001 — intentionally broad; see below
            # Deliberately broad: httpx raises many exception types
            # (ConnectTimeout, ReadTimeout, ConnectError, RemoteProtocolError,
            # PoolTimeout, ...) and new ones can appear in new httpx
            # versions. classify_exception's job is to make sense of
            # WHATEVER came out of send() — including exception types
            # nobody anticipated — via its own is_definitively_classified
            # flag when it doesn't recognize one. Catching narrowly here
            # would silently let an unrecognized exception type escape
            # uncaught, which is exactly the defect-2 failure mode
            # (a bad outcome that never reaches the ledger) recurring at
            # one level removed. Any exception RAISED BY classify_exception
            # itself, or by anything below (adapter methods are documented
            # to never raise, but "documented to" isn't "physically
            # cannot"), is deliberately NOT caught a second time here —
            # a bug in an adapter should be loud, not swallowed into a
            # miscategorized ledger row.
            latency_ms = (time.monotonic() - start) * 1000
            classified = self._adapter.classify_exception(exc, request)
            event = GatewayEvent(
                identity_key=self._identity_key,
                provider=self._adapter.provider_name,
                operation=request.operation,
                outcome=CallOutcome.FAILURE,
                error_category=classified.category,
                http_status=None,
                latency_ms=latency_ms,
                cost_usd=None,
                quota_bucket=bucket,
                raw_provider_metadata={"exception_type": type(exc).__name__, "exception_message": str(exc)},
            )
            await self._ledger.append(event)
            return CallResult(event=event, response=None, exception=exc, classified_error=classified)

        latency_ms = (time.monotonic() - start) * 1000
        return await self._handle_response(request, response, latency_ms, bucket)

    async def _handle_response(
        self, request: AdapterRequest, response: httpx.Response, latency_ms: float, bucket: str
    ) -> CallResult:
        snapshot = self._adapter.parse_rate_limit_headers(response)
        request_id = self._adapter.extract_request_id(response)

        if response.status_code >= 400:
            classified = self._adapter.classify_error(response)
            event = GatewayEvent(
                identity_key=self._identity_key,
                provider=self._adapter.provider_name,
                operation=request.operation,
                outcome=CallOutcome.FAILURE,
                error_category=classified.category,
                http_status=response.status_code,
                provider_request_id=request_id,
                latency_ms=latency_ms,
                cost_usd=None,
                quota_bucket=bucket,
                raw_provider_metadata={
                    "rate_limit_headers": snapshot.raw_headers,
                    "classification_message": classified.message,
                    "is_definitively_classified": classified.is_definitively_classified,
                },
            )
            await self._ledger.append(event)
            return CallResult(
                event=event,
                response=response,
                exception=None,
                rate_limit=snapshot,
                classified_error=classified,
            )

        # Status < 400: still have to check for a "looks fine but isn't"
        # match before calling this a real SUCCESS.
        bad_patterns = self._adapter.check_known_bad_patterns(response)
        usage = self._adapter.extract_usage(response)
        cost = (
            compute_cost_usd(
                self._adapter.provider_name,
                usage.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_input_tokens=usage.cached_input_tokens,
            )
            if usage is not None
            else None
        )

        outcome = CallOutcome.SUSPECTED_SILENT_FAILURE if bad_patterns else CallOutcome.SUCCESS
        event = GatewayEvent(
            identity_key=self._identity_key,
            provider=self._adapter.provider_name,
            operation=request.operation,
            outcome=outcome,
            error_category=None,
            http_status=response.status_code,
            provider_request_id=request_id,
            latency_ms=latency_ms,
            cost_usd=cost,
            quota_bucket=bucket,
            raw_provider_metadata={
                "rate_limit_headers": snapshot.raw_headers,
                "bad_pattern_names": [m.pattern_name for m in bad_patterns],
            },
        )
        await self._ledger.append(event)
        return CallResult(
            event=event, response=response, exception=None,
            rate_limit=snapshot, bad_patterns=bad_patterns,
        )
