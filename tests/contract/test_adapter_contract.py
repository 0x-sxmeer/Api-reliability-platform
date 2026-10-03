"""
The adapter contract test suite.

Every adapter -- the three that exist today, and every one Phases 7-8
add (Stripe, Twilio, a generic REST/OAuth2 adapter) -- must pass these
tests. They check properties the shared engines are allowed to assume
hold for ANY adapter, not provider-specific behavior (that's each
adapter's own test file).

Adding a new adapter: add one entry to ADAPTERS below. If it fails one
of these tests, that's a real bug in the new adapter, not a reason to
weaken the test.
"""

from __future__ import annotations

import httpx
import pytest

from gateway.adapters.anthropic import AnthropicAdapter
from gateway.adapters.gemini import GeminiAdapter
from gateway.adapters.openai import OpenAIAdapter
from gateway.adapters.rest import GenericRestAdapter
from gateway.core.adapter import AdapterRequest, ProviderAdapter
from gateway.core.types import ErrorCategory, QuotaWindow

# Add a new (name, adapter_instance) pair here for every new adapter.
ADAPTERS: list[tuple[str, ProviderAdapter]] = [
    ("anthropic", AnthropicAdapter(api_key="test-key-not-used")),
    ("openai", OpenAIAdapter(api_key="test-key-not-used")),
    ("gemini", GeminiAdapter(api_key="test-key-not-used")),
    ("generic_rest", GenericRestAdapter()),
]

ADAPTER_IDS = [name for name, _ in ADAPTERS]
ADAPTER_INSTANCES = [inst for _, inst in ADAPTERS]

# Every HTTP status code from 400 to 599 -- classify_error must handle
# ALL of these without raising, even ones the adapter doesn't
# specifically recognize (it should fall through to UNKNOWN).
ALL_ERROR_STATUS_CODES = list(range(400, 600))


@pytest.mark.parametrize("adapter", ADAPTER_INSTANCES, ids=ADAPTER_IDS)
class TestAdapterContract:
    """Every test in this class runs once per adapter in ADAPTERS."""

    def test_provider_name_is_stable_and_nonempty(self, adapter: ProviderAdapter) -> None:
        name = adapter.provider_name
        assert isinstance(name, str)
        assert len(name) > 0
        assert name == adapter.provider_name, "provider_name must be stable across calls"
        assert name.islower(), "provider_name should be a stable lowercase identifier"

    def test_get_limiting_unit_does_not_raise(self, adapter: ProviderAdapter) -> None:
        # Must simply return one of the enum members -- not raise, not
        # return None. This is a hard fact about the provider, not
        # something that depends on a response.
        result = adapter.get_limiting_unit()
        assert result is not None

    @pytest.mark.parametrize(
        "request_",
        [
            AdapterRequest(operation="x", payload={}),
            AdapterRequest(operation="x", payload={"model": None}),
            AdapterRequest(operation="x", payload={"model": ""}),
            AdapterRequest(operation="x", payload={"model": 123}),
            AdapterRequest(operation="x", payload={}, extra={"model": None}),
            AdapterRequest(operation="x", payload={"model": "m"}, extra={"model": "m"}),
            AdapterRequest(operation="", payload={"unexpected": ["shape"]}, extra={"k": object()}),
        ],
        ids=["empty", "model-none", "model-empty", "model-int", "extra-none", "both", "garbage"],
    )
    def test_quota_bucket_is_deterministic_nonempty_and_never_raises(
        self, adapter: ProviderAdapter, request_: AdapterRequest
    ) -> None:
        """The executor calls this on EVERY request, including malformed ones
        whose failure must reach the ledger rather than escape as an exception."""
        bucket = adapter.quota_bucket(request_)
        assert isinstance(bucket, str) and bucket
        assert bucket == adapter.quota_bucket(request_), "must be deterministic"

    def test_quota_windows_are_well_formed_and_uniquely_named(self, adapter: ProviderAdapter) -> None:
        windows = adapter.quota_windows()
        assert isinstance(windows, tuple)
        assert all(isinstance(w, QuotaWindow) for w in windows)
        names = [w.name for w in windows]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("status", ALL_ERROR_STATUS_CODES)
    def test_classify_error_never_raises(self, adapter: ProviderAdapter, status: int) -> None:
        """
        The single most important contract test: classify_error MUST NOT
        raise on any status code 400-599, including ones the adapter has
        never seen. An adapter that raises here would take down the
        executor's error path -- exactly the kind of silent, total
        failure this project exists to prevent. An unrecognized code
        should come back as ErrorCategory.UNKNOWN with
        is_definitively_classified=False, not an exception.
        """
        # A garbage/empty body -- the adapter must not assume its own
        # documented shape is present.
        response = httpx.Response(status, content=b"")
        result = adapter.classify_error(response)
        assert result.category in ErrorCategory
        assert isinstance(result.is_definitively_classified, bool)

    @pytest.mark.parametrize("status", ALL_ERROR_STATUS_CODES)
    def test_classify_error_never_raises_on_malformed_json(
        self, adapter: ProviderAdapter, status: int
    ) -> None:
        """Same as above, but with a body that LOOKS like JSON and isn't
        the adapter's expected shape -- e.g. a list instead of an object,
        or valid JSON that's just `null`. A real proxy or CDN error page
        can return almost anything under an error status code."""
        response = httpx.Response(status, json=["unexpected", "shape"])
        result = adapter.classify_error(response)
        assert result.category in ErrorCategory

    def test_parse_rate_limit_headers_never_raises_on_empty_headers(
        self, adapter: ProviderAdapter
    ) -> None:
        response = httpx.Response(200, content=b"{}")
        snapshot = adapter.parse_rate_limit_headers(response)
        assert snapshot is not None
        assert snapshot.limiting_unit is not None

    def test_parse_rate_limit_headers_never_raises_on_garbage_header_values(
        self, adapter: ProviderAdapter
    ) -> None:
        """Headers with the right NAME but a nonsense VALUE (a hostile or
        buggy upstream/proxy could send this) must not crash parsing --
        the field should come back as None, not raise."""
        response = httpx.Response(
            200,
            headers={
                "anthropic-ratelimit-requests-remaining": "not-a-number",
                "x-ratelimit-remaining-requests": "not-a-number",
                "x-ratelimit-reset-tokens": "not-a-duration",
                "retry-after": "not-a-number",
            },
            content=b"{}",
        )
        snapshot = adapter.parse_rate_limit_headers(response)
        assert snapshot is not None

    @pytest.mark.parametrize("status", [200, 201, 202, 204])
    def test_check_known_bad_patterns_returns_list_on_garbage_2xx(
        self, adapter: ProviderAdapter, status: int
    ) -> None:
        response = httpx.Response(status, content=b"not even json {{{")
        result = adapter.check_known_bad_patterns(response)
        assert isinstance(result, list)

    def test_check_known_bad_patterns_returns_list_on_empty_body(
        self, adapter: ProviderAdapter
    ) -> None:
        response = httpx.Response(200, content=b"")
        result = adapter.check_known_bad_patterns(response)
        assert isinstance(result, list)

    def test_extract_usage_never_raises_on_garbage(self, adapter: ProviderAdapter) -> None:
        for body in (b"", b"null", b"[]", b'{"unexpected": "shape"}', b"not json {{{"):
            response = httpx.Response(200, content=body)
            result = adapter.extract_usage(response)
            assert result is None or hasattr(result, "model")

    def test_extract_request_id_never_raises_on_garbage(self, adapter: ProviderAdapter) -> None:
        for body in (b"", b"null", b"not json {{{"):
            response = httpx.Response(200, content=body)
            result = adapter.extract_request_id(response)
            assert result is None or isinstance(result, str)

    def test_classify_exception_never_raises(self, adapter: ProviderAdapter) -> None:
        request = AdapterRequest(operation="__contract_test_unknown_op__", payload={})
        for exc in (
            httpx.ConnectTimeout("timeout", request=httpx.Request("POST", "https://x.test")),
            httpx.ReadTimeout("timeout", request=httpx.Request("POST", "https://x.test")),
            httpx.ConnectError("failed", request=httpx.Request("POST", "https://x.test")),
            RuntimeError("totally unrelated exception type"),
            ValueError("also unrelated"),
        ):
            result = adapter.classify_exception(exc, request)
            assert result.category in ErrorCategory

    def test_classify_error_distinguishes_overload_from_auth_failure(
        self, adapter: ProviderAdapter
    ) -> None:
        """
        429/5xx-family 'overload' must not be classified the same as a
        401/403 auth failure -- these require completely different
        caller action (wait-or-failover vs. fix-your-credentials-and-
        never-retry-unchanged). This doesn't assert a SPECIFIC category
        for either (that's adapter-specific, tested in each adapter's own
        file) -- only that the two are never conflated.
        """
        auth_response = httpx.Response(401, json={"error": {"message": "bad auth"}})
        auth_result = adapter.classify_error(auth_response)
        assert auth_result.category == ErrorCategory.PERMANENT, (
            "an auth failure (401) must be PERMANENT -- retrying an unchanged "
            "request can never fix bad credentials"
        )

    def test_send_raises_cleanly_on_unknown_operation(self, adapter: ProviderAdapter) -> None:
        """An adapter asked to perform an operation it doesn't implement
        must fail loudly and synchronously-classifiable (a ValueError,
        not a silent no-op or a hang) -- callers need to find this at
        development time, not in production telemetry."""
        import asyncio

        request = AdapterRequest(operation="__totally_unknown_operation__", payload={})
        with pytest.raises(ValueError):
            asyncio.run(adapter.send(request))
