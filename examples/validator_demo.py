"""
Phase 4 demo: the Response Validator catching "200-but-actually-failure"
across all three providers.

Runs offline: every provider uses httpx.MockTransport with realistic response
shapes from docs/provider-notes.md. Each scenario demonstrates a different
case the validator must handle:

  1. Anthropic SSE error-after-200 (the canonical Phase 4 case)
  2. Anthropic JSON error body on a 2xx
  3. Gemini blocked prompt (200, blockReason, no candidates)
  4. OpenAI clean response (no bad patterns -- honest VALID)
  5. Anthropic 429 rate limit (error response, not a bad pattern)
  6. Transport failure (ConnectTimeout)

Run with:  python examples/validator_demo.py

Structure note: this file builds adapters and mock bodies inside scenario
functions (the same vendor-shaped fixture wiring as demo.py/quota_demo.py).
The validator itself and everything after the scenarios is provider-agnostic.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from gateway.adapters.anthropic import AnthropicAdapter
from gateway.adapters.gemini import GeminiAdapter
from gateway.adapters.openai import OpenAIAdapter
from gateway.core.adapter import AdapterRequest, ProviderAdapter
from gateway.core.executor import CallExecutor
from gateway.engines.validator import (
    ResponseValidator,
    ValidationResult,
    ValidationVerdict,
    ValidatorConfig,
)
from gateway.ledger.store import SqliteLedgerStore


@dataclass
class Scenario:
    name: str
    adapter: ProviderAdapter
    request: AdapterRequest
    description: str


def _client(base_url: str, handler: Callable) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(handler))


# ----------------------------------------------------------- scenario builders


def anthropic_sse_error() -> Scenario:
    """THE canonical Phase 4 case: Anthropic streams a 200 with message_start,
    content_block_delta, then an `event: error` mid-stream."""

    sse_body = (
        "event: message_start\n"
        'data: {"type":"message_start","message":{"id":"msg_001","type":"message",'
        '"role":"assistant","model":"claude-sonnet","usage":{"input_tokens":25,"output_tokens":0}}}\n\n'
        "event: content_block_start\n"
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n'
        "event: error\n"
        'data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "request-id": "req_demo_sse",
                "anthropic-ratelimit-requests-limit": "50",
                "anthropic-ratelimit-requests-remaining": "45",
            },
            content=sse_body.encode(),
        )

    adapter = AnthropicAdapter("demo-key", http_client=_client("https://api.anthropic.com/v1", handler))
    return Scenario(
        name="Anthropic SSE error-after-200",
        adapter=adapter,
        request=AdapterRequest(
            operation="messages.create",
            payload={"model": "claude-sonnet", "max_tokens": 100, "messages": [{"role": "user", "content": "Hi"}]},
        ),
        description="A streaming response that starts normally but includes an `event: error` mid-stream.",
    )


def anthropic_json_error_body() -> Scenario:
    """Anthropic returns 200 but the body is {"type": "error", ...}."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"request-id": "req_demo_json"},
            json={"type": "error", "error": {"type": "server_error", "message": "Internal error"}},
        )

    adapter = AnthropicAdapter("demo-key", http_client=_client("https://api.anthropic.com/v1", handler))
    return Scenario(
        name="Anthropic JSON error body on 2xx",
        adapter=adapter,
        request=AdapterRequest(
            operation="messages.create",
            payload={"model": "claude-sonnet", "max_tokens": 100, "messages": [{"role": "user", "content": "Hi"}]},
        ),
        description="HTTP 200 but the body is an error object (not a message response).",
    )


def gemini_blocked_prompt() -> Scenario:
    """Gemini returns 200 but with blockReason and no candidates."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "promptFeedback": {
                    "blockReason": "SAFETY",
                    "safetyRatings": [
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "HIGH"},
                    ],
                },
                "modelVersion": "gemini-2.0-flash",
            },
        )

    adapter = GeminiAdapter("demo-key", http_client=_client("https://generativelanguage.googleapis.com/v1beta", handler))
    return Scenario(
        name="Gemini blocked prompt (safety)",
        adapter=adapter,
        request=AdapterRequest(
            operation="generateContent",
            payload={"contents": [{"parts": [{"text": "test"}]}]},
            extra={"model": "gemini-2.0-flash"},
        ),
        description="HTTP 200 but promptFeedback.blockReason is set and no candidates.",
    )


def openai_clean() -> Scenario:
    """OpenAI returns a normal, clean response with no bad patterns."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "x-ratelimit-limit-requests": "200",
                "x-ratelimit-remaining-requests": "195",
                "x-ratelimit-limit-tokens": "10000",
                "x-ratelimit-remaining-tokens": "9500",
                "x-ratelimit-reset-tokens": "6s",
                "x-request-id": "req_demo_clean",
            },
            json={
                "id": "chatcmpl-demo",
                "model": "gpt-6-sol",
                "choices": [{"message": {"role": "assistant", "content": "Hello! How can I help?"}, "index": 0}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 12},
            },
        )

    adapter = OpenAIAdapter("demo-key", http_client=_client("https://api.openai.com/v1", handler))
    return Scenario(
        name="OpenAI clean response",
        adapter=adapter,
        request=AdapterRequest(
            operation="chat.completions.create",
            payload={"model": "gpt-6-sol", "messages": [{"role": "user", "content": "Hi"}]},
        ),
        description="A normal successful response -- the validator should say VALID.",
    )


def anthropic_rate_limit() -> Scenario:
    """Anthropic returns 429 rate_limit_error -- an HTTP error, not a bad pattern."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "30", "request-id": "req_demo_429"},
            json={
                "type": "error",
                "error": {"type": "rate_limit_error", "message": "Rate limit exceeded; retry after 30 seconds."},
            },
        )

    adapter = AnthropicAdapter("demo-key", http_client=_client("https://api.anthropic.com/v1", handler))
    return Scenario(
        name="Anthropic 429 rate limit",
        adapter=adapter,
        request=AdapterRequest(
            operation="messages.create",
            payload={"model": "claude-sonnet", "max_tokens": 100, "messages": []},
        ),
        description="An HTTP 429 error -- the validator classifies this with routing advice.",
    )


def transport_failure() -> Scenario:
    """A transport-level failure (ConnectTimeout) -- no response at all."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("Connection timed out", request=request)

    adapter = OpenAIAdapter("demo-key", http_client=_client("https://api.openai.com/v1", handler))
    return Scenario(
        name="Transport failure (ConnectTimeout)",
        adapter=adapter,
        request=AdapterRequest(
            operation="chat.completions.create",
            payload={"model": "gpt-6-sol", "messages": []},
        ),
        description="No response received -- the validator says INVALID + retryable.",
    )


# ------------------------------------------------------------- display helpers


def _verdict_symbol(v: ValidationVerdict) -> str:
    return {"valid": "[OK]", "suspect": "[??]", "invalid": "[!!]"}[v.value]


def _print_validation(scenario: Scenario, validation: ValidationResult) -> None:
    sym = _verdict_symbol(validation.verdict)
    print(f"\n{'=' * 72}")
    print(f"  {sym}  {scenario.name}")
    print(f"     {scenario.description}")
    print(f"{'-' * 72}")
    print(f"  Verdict:    {validation.verdict.value.upper()}")
    print(f"  Confidence: {validation.confidence:.2f}")
    print(f"  Outcome:    {validation.outcome.value}")
    print(f"  Billed:     {'yes' if validation.was_billed else 'no (unknown or not billed)'}")
    print(f"  Retry:      {'yes' if validation.should_retry else 'no'}", end="")
    if validation.retry_on_different_provider:
        print(" (different provider recommended)", end="")
    print()

    if validation.findings:
        print(f"\n  Findings ({len(validation.findings)}):")
        for f in validation.findings:
            retry = []
            if f.retryable_same_provider:
                retry.append("same")
            if f.retryable_different_provider:
                retry.append("different")
            retry_str = f"retryable on {'/'.join(retry)}" if retry else "not retryable"
            print(f"    - [{f.category.value}] {f.name} (conf={f.confidence:.2f}, {retry_str})")
            print(f"      {f.description[:100]}")

    if validation.caveats:
        print("\n  Caveats:")
        for c in validation.caveats:
            print(f"    > {c}")


# -------------------------------------------------------------------- main


async def main() -> None:
    print("=" * 72)
    print("  Phase 4 Demo: Response Validator")
    print("=" * 72)
    print()
    print("  The Response Validator sits between the executor and the router.")
    print("  It validates completed CallResults against operator-configurable")
    print("  policy and produces structured routing recommendations.")
    print()

    scenarios = [
        anthropic_sse_error(),
        anthropic_json_error_body(),
        gemini_blocked_prompt(),
        openai_clean(),
        anthropic_rate_limit(),
        transport_failure(),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = SqliteLedgerStore(Path(tmpdir) / "demo.db")
        validator = ResponseValidator()

        for scenario in scenarios:
            executor = CallExecutor(scenario.adapter, ledger, identity_key="demo-team")
            result = await executor.execute(scenario.request)
            validation = validator.validate(result)
            _print_validation(scenario, validation)

    # ---- Summary
    print(f"\n{'=' * 72}")
    print("  Summary")
    print(f"{'=' * 72}")
    print()
    print("  The validator is:")
    print("    - Stateless -- each validate() call is independent")
    print("    - Synchronous -- no I/O, no async needed")
    print("    - Vendor-agnostic -- reads CallResult, not adapter internals")
    print("    - Policy-configurable -- thresholds are operator-tunable")
    print()
    print("  It answers the question Phase 5's Routing Engine needs:")
    print("  'Is this response usable, and if not, should I retry?'")
    print()

    # ---- Demonstrate policy configuration
    print(f"{'=' * 72}")
    print("  Policy Configuration Demo")
    print(f"{'=' * 72}")
    print()
    print("  Default config: suspect_threshold=0.5, reject_threshold=0.85")

    lenient = ValidatorConfig(
        suspect_threshold=0.3,
        reject_threshold=0.99,
        treat_suspected_silent_failure_as_invalid=False,
    )
    print("  Lenient config: suspect_threshold=0.3, reject_threshold=0.99,")
    print("                  treat_suspected_silent_failure_as_invalid=False")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = SqliteLedgerStore(Path(tmpdir) / "demo2.db")
        scenario = anthropic_sse_error()
        executor = CallExecutor(scenario.adapter, ledger, identity_key="demo-team")
        result = await executor.execute(scenario.request)

        default_v = ResponseValidator().validate(result)
        lenient_v = ResponseValidator(config=lenient).validate(result)

        print("  Same response (Anthropic SSE error, confidence=0.97):")
        print(f"    Default policy -> {default_v.verdict.value.upper()}")
        print(f"    Lenient policy -> {lenient_v.verdict.value.upper()}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
