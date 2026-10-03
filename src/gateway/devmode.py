"""
Dev Mode: fake-transport layer for zero-credential demos.

When no real provider API keys are configured (or GATEWAY_FAKE_APIS=1),
the gateway intercepts outbound HTTP at the httpx transport layer and
returns realistic canned responses. ALL gateway logic above the transport
(routing, quota, policy, budget, validation, ledger, reconciliation) runs
for real — only the network hop to the vendor is simulated.

This module is provider-agnostic plumbing: it never branches on
`provider == ...` inside the engines. It routes purely by URL host, which
is exactly what a transport does. Vendor response *shapes* live here as
static fixtures, mirroring how adapters own vendor request shapes.

Enable with:  GATEWAY_FAKE_APIS=1  (auto-enabled when OPENAI_API_KEY /
ANTHROPIC_API_KEY are unset). Disable entirely with GATEWAY_FAKE_APIS=0.
"""

from __future__ import annotations

import json
import os
from typing import Any, ClassVar

import httpx

# ---------------------------------------------------------------------------
# Canned response fixtures (status, body) per (host-pattern, path-fragment)
# ---------------------------------------------------------------------------

_OPENAI_CHAT = {
    "id": "chatcmpl-fake-001",
    "object": "chat.completion",
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello! (Dev Mode canned response)"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
}

_ANTHROPIC_MSG = {
    "id": "msg_fake_001",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello! (Dev Mode canned response)"}],
    "model": "claude-sonnet-4-20250514",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 15, "output_tokens": 9},
}

_GEMINI_GENERATE = {
    "candidates": [
        {
            "content": {"parts": [{"text": "Hello! (Dev Mode canned response)"}], "role": "model"},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 7, "totalTokenCount": 17},
}

_STRIPE_PAYMENT = {
    "id": "pi_fake_001",
    "object": "payment_intent",
    "status": "succeeded",
    "amount": 2000,
    "currency": "usd",
}


def _rate_headers() -> dict[str, str]:
    """Plausible rate-limit headers so QuotaEngine has real data to track."""
    return {
        "x-ratelimit-limit-requests": "500",
        "x-ratelimit-remaining-requests": "493",
        "x-ratelimit-limit-tokens": "200000",
        "x-ratelimit-remaining-tokens": "180000",
        "retry-after": "0",
    }


class FakeAPIResponses(httpx.AsyncBaseTransport):
    """httpx transport that answers known endpoints without touching the network.

    Unknown hosts/paths fall through to the wrapped real transport, so this
    can be layered safely even if some calls should go live.
    """

    def __init__(self, fallback: httpx.AsyncBaseTransport | None = None) -> None:
        self._fallback = fallback or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host.lower()
        path = request.url.path.lower()

        # Fault-injection hook for demos: GATEWAY_FAKE_FAIL_ONCE=n makes the
        # first n OpenAI chat calls return 503, proving router failover live.
        if "openai" in host and "chat/completions" in path and self._should_fail():
            return httpx.Response(
                503,
                headers={"retry-after": "1"},
                json={
                    "error": {
                        "message": "Service temporarily overloaded (injected)",
                        "type": "server_error",
                        "code": "service_unavailable",
                    }
                },
            )

        if "openai" in host and "chat/completions" in path:
            return httpx.Response(200, json=_OPENAI_CHAT, headers=_rate_headers())
        if "anthropic" in host and "messages" in path:
            # Anthropic requires an `anthropic-version` header on every real
            # request; mirror that here so the fake validates what production
            # would validate (catches adapters that forget to send it).
            if request.headers.get("anthropic-version") is None:
                return httpx.Response(
                    400,
                    json={"type": "error", "error": {
                        "type": "invalid_request_error",
                        "message": "missing anthropic-version header",
                    }},
                )
            return httpx.Response(
                200,
                json=_ANTHROPIC_MSG,
                headers={
                    "request-id": "req_devmode_anthropic_001",
                    "x-ratelimit-limit-requests": "1000",
                    "x-ratelimit-remaining-requests": "999",
                },
            )
        if ("generativelanguage.googleapis.com" in host) and "generatecontent" in path:
            return httpx.Response(200, json=_GEMINI_GENERATE)
        if "api.stripe.com" in host or "payments" in path or "charges" in path:
            return httpx.Response(
                200,
                json=_STRIPE_PAYMENT,
                headers={
                    "stripe-request-id": "req_devmode_stripe_001",
                    "ratelimit-limit": "100",
                    "ratelimit-remaining": "97",
                    "ratelimit-reset": "30",
                },
            )
        # Generic catch-all for any other REST target registered in dev mode.
        if host.startswith("api."):
            return httpx.Response(200, json={"status": "ok", "dev_mode": True})

        return await self._fallback.handle_async_request(request)

    # --- fault injection ---------------------------------------------------
    _fail_counter: ClassVar[list[int]] = []  # class-level mutable countdown

    @staticmethod
    def _should_fail() -> bool:
        if not FakeAPIResponses._fail_counter:
            raw = os.getenv("GATEWAY_FAKE_FAIL_ONCE", "")
            if raw.isdigit() and int(raw) > 0:
                FakeAPIResponses._fail_counter.append(int(raw))
        if FakeAPIResponses._fail_counter and FakeAPIResponses._fail_counter[0] > 0:
            FakeAPIResponses._fail_counter[0] -= 1
            return True
        return False


def dev_mode_enabled() -> bool:
    """True when fake APIs should be used (explicit flag, or missing creds)."""
    flag = os.getenv("GATEWAY_FAKE_APIS")
    if flag is not None:
        return flag != "0"
    return os.getenv("OPENAI_API_KEY", "") in ("", "mock")


def make_http_client(base_url: str) -> httpx.AsyncClient:
    """Build an AsyncClient wired to the fake transport in Dev Mode."""
    if dev_mode_enabled():
        return httpx.AsyncClient(base_url=base_url, transport=FakeAPIResponses())
    return httpx.AsyncClient(base_url=base_url)


def describe() -> str:
    enabled = dev_mode_enabled()
    return (
        "DEV MODE: external APIs are faked at the transport layer; all "
        "gateway engines run for real."
        if enabled
        else "LIVE MODE: requests will hit real provider endpoints."
    )


# Keep json import meaningful for fixture debugging utilities.
def dump_fixtures() -> dict[str, Any]:
    return json.loads(json.dumps({"openai": _OPENAI_CHAT, "anthropic": _ANTHROPIC_MSG}))
