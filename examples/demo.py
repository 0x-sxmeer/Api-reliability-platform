"""
Phase 2 end-to-end demo: three adapters (Anthropic, OpenAI, Gemini),
one CallExecutor, one shared ledger.

Orchestration now lives in gateway.core.executor.CallExecutor (defect 1's
fix) -- this script is a thin caller, not where the send/classify/log
logic lives. Every provider is driven through httpx.MockTransport with
REALISTIC response bodies (the exact shapes documented in
docs/provider-notes.md) so this runs immediately, with no API keys and
no network access, while still exercising real header-parsing and
error-classification against each provider's actual documented shape --
including a genuine transport-level exception (defect 2's fix) for one
of the calls.

To run any of the three against its REAL API instead, set the
corresponding environment variable (ANTHROPIC_API_KEY / OPENAI_API_KEY /
GEMINI_API_KEY) before running -- see build_adapter() below.

Run with:  python examples/demo.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx

from gateway.adapters.anthropic import AnthropicAdapter
from gateway.adapters.gemini import GeminiAdapter
from gateway.adapters.openai import OpenAIAdapter
from gateway.core.adapter import AdapterRequest, ProviderAdapter
from gateway.core.executor import CallExecutor
from gateway.ledger.store import SqliteLedgerStore

# ---------------------------------------------------------------- mock bodies


def _anthropic_transport() -> httpx.MockTransport:
    """Success, then a 429 rate limit (WITH retry-after), then a 529
    overload -- the same three-call sequence Phase 1 demonstrated,
    unchanged, so this remains a regression check on top of a demo."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200,
                headers={
                    "anthropic-ratelimit-requests-limit": "50",
                    "anthropic-ratelimit-requests-remaining": "49",
                    "anthropic-ratelimit-requests-reset": "2026-09-28T12:01:00Z",
                    "anthropic-ratelimit-tokens-limit": "100000",
                    "anthropic-ratelimit-tokens-remaining": "99500",
                    "request-id": "req_anthropic_001",
                },
                json={
                    "id": "msg_01demo",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello from Claude!"}],
                    "model": "claude-sonnet-4-5",
                    "usage": {"input_tokens": 12, "output_tokens": 8},
                },
            )
        if calls["n"] == 2:
            return httpx.Response(
                429,
                headers={"retry-after": "3"},
                json={
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error",
                        "message": "Number of request tokens has exceeded your per-minute rate limit.",
                    },
                },
            )
        return httpx.Response(
            529, json={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        )

    return httpx.MockTransport(handler)


def _openai_transport() -> httpx.MockTransport:
    """Success, then an insufficient_quota 429 (QUOTA_EXHAUSTED -- distinct
    from an ordinary rate limit), then a genuine transport-level timeout
    (defect 2's fix, demonstrated live)."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200,
                headers={
                    "x-ratelimit-remaining-requests": "199",
                    "x-ratelimit-limit-requests": "200",
                    "x-ratelimit-remaining-tokens": "9800",
                    "x-ratelimit-limit-tokens": "10000",
                    "x-ratelimit-reset-tokens": "6m0s",
                    "x-request-id": "req_openai_001",
                },
                json={
                    "id": "chatcmpl-demo",
                    "model": "gpt-6-sol",
                    "choices": [{"message": {"role": "assistant", "content": "Hello from GPT!"}}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 6,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    },
                },
            )
        if calls["n"] == 2:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "You exceeded your current quota, please check your plan and billing details.",
                        "type": "insufficient_quota",
                        "code": "insufficient_quota",
                    }
                },
            )
        raise httpx.ReadTimeout("simulated network timeout", request=request)

    return httpx.MockTransport(handler)


def _gemini_transport() -> httpx.MockTransport:
    """Success (no rate-limit headers at all -- this is documented,
    correct behavior for this provider, not a mock oversight), then a
    429 RESOURCE_EXHAUSTED WITH a daily quotaId (QUOTA_EXHAUSTED, correctly
    NOT treated as an ordinary retry-soon rate limit), then a prompt
    blocked before generation (SUSPECTED_SILENT_FAILURE on a 200)."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": "Hello from Gemini!"}], "role": "model"}}
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 8,
                        "candidatesTokenCount": 5,
                        "cachedContentTokenCount": 0,
                    },
                    "modelVersion": "gemini-2.5-flash",
                },
            )
        if calls["n"] == 2:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "code": 429,
                        "message": "Quota exceeded.",
                        "status": "RESOURCE_EXHAUSTED",
                        "details": [
                            {
                                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                "violations": [
                                    {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}
                                ],
                            }
                        ],
                    }
                },
            )
        return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})

    return httpx.MockTransport(handler)


# ------------------------------------------------------------- adapter setup


def build_adapter(name: str) -> ProviderAdapter:
    """Real API key from the environment if present; otherwise the
    realistic mock for that provider. Each branch is independent, so you
    can mix real and mocked providers (e.g. a real ANTHROPIC_API_KEY with
    the other two still mocked)."""
    if name == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            return AnthropicAdapter(api_key=key)
        client = httpx.AsyncClient(base_url="https://api.anthropic.com/v1", transport=_anthropic_transport())
        return AnthropicAdapter(api_key="mock-not-used", http_client=client)

    if name == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            return OpenAIAdapter(api_key=key)
        client = httpx.AsyncClient(base_url="https://api.openai.com/v1", transport=_openai_transport())
        return OpenAIAdapter(api_key="mock-not-used", http_client=client)

    if name == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            return GeminiAdapter(api_key=key)
        client = httpx.AsyncClient(
            base_url="https://generativelanguage.googleapis.com/v1beta", transport=_gemini_transport()
        )
        return GeminiAdapter(api_key="mock-not-used", http_client=client)

    raise ValueError(name)


def build_request(provider: str) -> AdapterRequest:
    if provider == "anthropic":
        return AdapterRequest(
            operation="messages.create",
            payload={
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            },
        )
    if provider == "openai":
        return AdapterRequest(
            operation="chat.completions.create",
            payload={
                "model": "gpt-6-sol",
                "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            },
        )
    if provider == "gemini":
        return AdapterRequest(
            operation="generateContent",
            payload={"contents": [{"parts": [{"text": "Say hello in one sentence."}]}]},
            extra={"model": "gemini-2.5-flash"},
        )
    raise ValueError(provider)


# ------------------------------------------------------------------- runner


async def run_provider(provider: str, ledger: SqliteLedgerStore, calls_to_make: int) -> None:
    adapter = build_adapter(provider)
    executor = CallExecutor(adapter, ledger, identity_key=f"demo-{provider}")

    print(f"\n=== {provider} ===")
    for i in range(1, calls_to_make + 1):
        result = await executor.execute(build_request(provider))
        snapshot_note = ""
        # Peek at what got stored, for narration -- the raw_provider_metadata
        # is exactly what a real consumer would read back from the ledger.
        rl = result.event.raw_provider_metadata.get("rate_limit_headers")
        if rl:
            snapshot_note = f" | headers present: {len(rl)} field(s)"
        elif result.response is not None:
            snapshot_note = " | no rate-limit headers on this response (may be expected)"

        if result.exception is not None:
            print(f"  call {i}: EXCEPTION {type(result.exception).__name__} "
                  f"-> outcome={result.event.outcome.value}, "
                  f"category={result.event.error_category.value if result.event.error_category else None}")
        else:
            status = result.event.http_status
            print(f"  call {i}: HTTP {status} -> outcome={result.event.outcome.value}"
                  f"{', category=' + result.event.error_category.value if result.event.error_category else ''}"
                  f"{snapshot_note}")


async def main() -> None:
    db_path = Path(__file__).parent / "demo_ledger.db"
    db_path.unlink(missing_ok=True)
    ledger = SqliteLedgerStore(db_path)

    using_mocks = [
        name for name in ("anthropic", "openai", "gemini")
        if not os.environ.get(f"{name.upper()}_API_KEY")
    ]
    if using_mocks:
        print(f"No API key set for: {', '.join(using_mocks)} -- using realistic mocked responses.")
        print("(set ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY to hit the real APIs instead)")
    print(
        "\nNote: you'll see 'pricing: no entry for ...' on stderr for Anthropic and "
        "Gemini models -- that's core/pricing.py correctly refusing to guess at "
        "unverified prices (see docs/provider-notes.md), not an error."
    )

    await run_provider("anthropic", ledger, calls_to_make=3)
    await run_provider("openai", ledger, calls_to_make=3)
    await run_provider("gemini", ledger, calls_to_make=3)

    # ---- one shared ledger, three providers -- summary by provider/outcome
    print("\n=== Ledger summary (one store, three providers) ===")
    for provider in ("anthropic", "openai", "gemini"):
        events = await ledger.query(provider=provider, limit=100)
        by_outcome: dict[str, int] = {}
        for e in events:
            by_outcome[e.outcome.value] = by_outcome.get(e.outcome.value, 0) + 1
        cost = await ledger.sum_cost_since(provider=provider)
        outcome_str = ", ".join(f"{k}={v}" for k, v in sorted(by_outcome.items()))
        print(f"  {provider:10s} total={len(events):2d}  ({outcome_str})  cost=${cost:.6f}")

    total_events = 0
    for p in ("anthropic", "openai", "gemini"):
        total_events += len(await ledger.query(provider=p, limit=100))
    print(f"\n  TOTAL events across all providers, one ledger: {total_events}")
    print(f"\nLedger persisted at: {db_path}")


if __name__ == "__main__":
    asyncio.run(main())
