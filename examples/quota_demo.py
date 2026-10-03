"""
Phase 3 demo: the Quota Engine giving DIFFERENT-CONFIDENCE answers for
Anthropic / OpenAI (live response headers) versus Gemini (no headers at all,
so counted from the ledger).

Runs offline: every provider is an httpx.MockTransport replaying the exact
response shapes documented in docs/provider-notes.md, pushed through the real
adapters, the real CallExecutor, a real SQLite ledger and the real engine.

Run with:  python examples/quota_demo.py

Structure note: this file contains no `if provider == "..."`. Each provider
is a Scenario built by its own function; the vendor-shaped mock bodies live
in those builders, and everything below them is provider-agnostic, which is
the same property the engine itself has.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import httpx

from gateway.adapters.anthropic import AnthropicAdapter
from gateway.adapters.gemini import GeminiAdapter
from gateway.adapters.openai import OpenAIAdapter
from gateway.core.adapter import AdapterRequest, ProviderAdapter
from gateway.core.executor import CallExecutor
from gateway.core.types import LimitingUnit, utcnow
from gateway.engines.quota import BlockReason, QuotaAssessment, QuotaEngine, Verdict
from gateway.ledger.store import SqliteLedgerStore

Responder = Callable[[], httpx.Response]


class DemoClock:
    """Real time plus an offset, so one section can jump ahead without sleeping."""

    def __init__(self) -> None:
        self.offset = timedelta(0)

    def __call__(self):
        return utcnow() + self.offset


@dataclass
class Scenario:
    name: str
    adapter: ProviderAdapter
    make_request: Callable[[str], AdapterRequest]
    # model -> queue of scripted responses (the last one repeats)
    script: dict[str, list[Responder]] = field(default_factory=dict)
    limits: dict[str, dict[str, float]] | None = None


def _transport(script: dict[str, list[Responder]], route: Callable[[httpx.Request], str]) -> httpx.MockTransport:
    cursors = dict.fromkeys(script, 0)

    def handler(request: httpx.Request) -> httpx.Response:
        model = route(request)
        queue = script[model]
        i = min(cursors[model], len(queue) - 1)
        cursors[model] += 1
        return queue[i]()

    return httpx.MockTransport(handler)


def _body_model(request: httpx.Request) -> str:
    return json.loads(request.content)["model"]


def _rfc3339(dt) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------- scenario builders


def anthropic_scenario() -> Scenario:
    def ok() -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "anthropic-ratelimit-requests-limit": "50",
                "anthropic-ratelimit-requests-remaining": "40",
                "anthropic-ratelimit-requests-reset": _rfc3339(utcnow() + timedelta(seconds=45)),
                "anthropic-ratelimit-tokens-limit": "100000",
                "anthropic-ratelimit-tokens-remaining": "30000",
                "request-id": "req_anthropic_demo",
            },
            json={"model": "model-a", "usage": {"input_tokens": 12, "output_tokens": 8}},
        )

    def too_fast() -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "8"},
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "per-minute limit"}},
        )

    script = {"model-a": [ok, too_fast]}
    client = httpx.AsyncClient(
        base_url="https://api.anthropic.com/v1", transport=_transport(script, _body_model)
    )
    return Scenario(
        name="anthropic",
        adapter=AnthropicAdapter("demo", http_client=client),
        make_request=lambda m: AdapterRequest(
            operation="messages.create", payload={"model": m, "max_tokens": 16, "messages": []}
        ),
        script=script,
    )


def openai_scenario() -> Scenario:
    def project_nearly_full() -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "x-ratelimit-limit-requests": "500",
                "x-ratelimit-remaining-requests": "480",
                "x-ratelimit-limit-tokens": "200000",
                "x-ratelimit-remaining-tokens": "150000",
                "x-ratelimit-reset-tokens": "6s",
                # the project scope is the tight one: 5% of 60,000 left
                "x-ratelimit-limit-project-tokens": "60000",
                "x-ratelimit-remaining-project-tokens": "3000",
                "x-ratelimit-reset-project-tokens": "3s",
            },
            json={"model": "model-a", "usage": {"prompt_tokens": 10, "completion_tokens": 6}},
        )

    def out_of_credit() -> httpx.Response:
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

    def topped_up() -> httpx.Response:
        return httpx.Response(
            200, json={"model": "model-broke", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        )

    script = {
        "model-a": [project_nearly_full],
        # out of credit for 3 calls (first failure, a failed probe, ...), then a human tops up
        "model-broke": [out_of_credit, out_of_credit, topped_up],
    }
    client = httpx.AsyncClient(base_url="https://api.openai.com/v1", transport=_transport(script, _body_model))
    return Scenario(
        name="openai",
        adapter=OpenAIAdapter("demo", http_client=client),
        make_request=lambda m: AdapterRequest(
            operation="chat.completions.create", payload={"model": m, "messages": []}
        ),
        script=script,
    )


def gemini_scenario() -> Scenario:
    def ok() -> httpx.Response:  # NOTE: no rate-limit headers at all -- documented, not a mock omission
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "hi"}], "role": "model"}}],
                "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 5},
                "modelVersion": "model-a",
            },
        )

    def daily_quota() -> httpx.Response:
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
                            "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
                        }
                    ],
                }
            },
        )

    script = {"model-a": [ok], "model-b": [daily_quota]}

    def route(request: httpx.Request) -> str:  # Gemini puts the model in the URL path
        return request.url.path.split("/models/")[1].split(":")[0]

    client = httpx.AsyncClient(
        base_url="https://generativelanguage.googleapis.com/v1beta", transport=_transport(script, route)
    )
    return Scenario(
        name="gemini",
        adapter=GeminiAdapter("demo", http_client=client),
        make_request=lambda m: AdapterRequest(
            operation="generateContent", payload={"contents": []}, extra={"model": m}
        ),
        script=script,
        # No provider publishes these through an API (Gemini: "view your active
        # limits in AI Studio"), so the operator supplies the numbers. The
        # ADAPTER supplies the window shapes; typo'd names are rejected.
        limits={"*": {"rpm": 5, "rpd": 1000, "tpm": 250_000, "spend_10m": 10.0}},
    )


# --------------------------------------------------------------- presentation


def describe(a: QuotaAssessment) -> None:
    head = f"verdict={a.verdict.value.upper():7s} confidence={a.confidence.value.upper():6s}"
    if a.block_reason:
        head += f" reason={a.block_reason.value}"
    if a.provisional:
        head += " (PROVISIONAL)"
    print(f"    {head}")
    if a.binding is not None:
        b = a.binding
        frac = f"{b.fraction_remaining:.0%} left" if b.fraction_remaining is not None else "fraction unknown"
        print(f"    binding: {b.label} @ {b.unit.value} via {b.source.value} ({frac}, remaining={b.remaining:g})")
    if a.retry_at is not None:
        line = f"    retry_at: {a.retry_at:%H:%M:%S}Z"
        if a.block_reason is BlockReason.QUOTA_EXHAUSTED:  # resume_at is only meaningful here
            line += f"   resume_at: {a.resume_at.isoformat() if a.resume_at else 'UNKNOWN (provider states none)'}"
        print(line)
    if a.binding is not None:
        helps = {u.value: a.adding_helps(u) for u in (LimitingUnit.KEY, LimitingUnit.PROJECT, LimitingUnit.ORGANIZATION)}
        print("    would adding more ... help?  " + "  ".join(f"{k}={'yes' if v else 'NO'}" for k, v in helps.items()))
    for u in a.unmonitored:
        print(f"    UNMONITORED  {u}")
    for c in a.caveats:
        print(f"    caveat: {c}")


async def step(engine: QuotaEngine, ledger: SqliteLedgerStore, sc: Scenario, model: str) -> None:
    request = sc.make_request(model)
    result = await CallExecutor(sc.adapter, ledger, identity_key="demo").execute(request)
    engine.observe(sc.adapter, request, snapshot=result.rate_limit, error=result.classified_error)
    status = result.event.http_status
    cat = result.event.error_category.value if result.event.error_category else "-"
    print(f"  -> {sc.name}/{model}: HTTP {status} outcome={result.event.outcome.value} category={cat}")


async def show(engine: QuotaEngine, sc: Scenario, model: str) -> QuotaAssessment:
    a = await engine.assess(sc.adapter, sc.make_request(model))
    print(f"  assess {sc.name}/{model}:")
    describe(a)
    return a


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


async def main() -> None:
    # The pricing module warns once per call that no price exists for these
    # made-up model names. That absence is narrated in section 3 instead.
    logging.getLogger("gateway.core.pricing").setLevel(logging.ERROR)
    anthropic, openai, gemini = anthropic_scenario(), openai_scenario(), gemini_scenario()
    clock = DemoClock()
    with tempfile.TemporaryDirectory() as tmp:
        ledger = SqliteLedgerStore(Path(tmp) / "quota_demo.db")
        engine = QuotaEngine(ledger, clock=clock)
        for sc in (anthropic, openai, gemini):
            engine.register(sc.adapter, limits=sc.limits)

        banner("1. ANTHROPIC -- live headers -> HIGH confidence")
        print("  Before any call the engine knows nothing and says so (not a hopeful 'open'):")
        await show(engine, anthropic, "model-a")
        await step(engine, ledger, anthropic, "model-a")
        a1 = await show(engine, anthropic, "model-a")
        print("  ^ caveat: Anthropic can also enforce a WORKSPACE cap that these combined headers can't")
        print("    distinguish from the org's (docs/provider-notes.md), so 'project=NO' is only as good as that.")
        print("  Then a 429 WITH retry-after: a stated cooldown, not a guess:")
        await step(engine, ledger, anthropic, "model-a")
        await show(engine, anthropic, "model-a")

        banner("2. OPENAI -- org AND project scopes enforced at once; the tighter one binds")
        await step(engine, ledger, openai, "model-a")
        o1 = await show(engine, openai, "model-a")
        print("  ^ the ORG has 75% of its tokens left; the PROJECT has 5%. A single-scope model would")
        print("    call this healthy and recommend the wrong fix. More keys won't help; a new project will.")

        banner("3. GEMINI -- NO headers, ever -> counted from the ledger, LOWER confidence")
        print("  Operator-configured limits (no API publishes them): rpm=5, rpd=1000, tpm=250k, spend_10m=$10")
        for _ in range(3):
            await step(engine, ledger, gemini, "model-a")
        g1 = await show(engine, gemini, "model-a")
        print("  ^ note what it REFUSES to claim: tpm (ledger stores no tokens) and spend_10m (Gemini has no")
        print("    prices, so SUM(cost) would read $0 -- a confident lie) are reported UNMONITORED.")
        print("  Two more calls reach the configured rpm of 5:")
        for _ in range(2):
            await step(engine, ledger, gemini, "model-a")
        g2 = await show(engine, gemini, "model-a")
        print("  ^ 'blocked' from the ledger is decent evidence (we provably used >= the limit);")
        print("    'open' from the ledger was weak evidence (others may share the project).")

        banner("4. GEMINI per-model isolation + a daily-quota 429 (heuristic diagnosis)")
        print("  model-a is at its rpm cap, but the limits are PER MODEL, so model-b is unaffected:")
        await show(engine, gemini, "model-b")
        await step(engine, ledger, gemini, "model-b")
        print("  ^ a PerDay quotaId 429 (UNVERIFIED as a stable contract => not definitively classified).")
        print("    The category is trusted for routing (block), the duration is not (LOW confidence, early probe);")
        print("    the adapter supplies the VERIFIED midnight-Pacific reset as resume_at.")
        await show(engine, gemini, "model-b")

        banner("5. OPENAI out of credit -- resume time UNKNOWN is a first-class state")
        await step(engine, ledger, openai, "model-broke")
        await show(engine, openai, "model-broke")
        print("  ...5 minutes later (clock jumped, nothing slept): a single probe is due.")
        clock.offset += timedelta(minutes=5, seconds=1)
        p = await show(engine, openai, "model-broke")
        assert p.verdict is Verdict.PROBE
        req = openai.make_request("model-broke")
        print(f"  claim_probe() -> {engine.claim_probe(openai.adapter, req)}   (a concurrent caller: "
              f"{engine.claim_probe(openai.adapter, req)})")
        await step(engine, ledger, openai, "model-broke")
        print("  The probe failed, so the wait doubles (5 min -> 10 min), capped at 6h:")
        await show(engine, openai, "model-broke")
        clock.offset += timedelta(minutes=10, seconds=1)
        assert engine.claim_probe(openai.adapter, req)
        await step(engine, ledger, openai, "model-broke")
        print("  Someone topped up the account; the next probe succeeds and closes the breaker:")
        await show(engine, openai, "model-broke")

        banner("SUMMARY -- same engine, three providers, honestly different confidence")
        rows = [("anthropic", a1), ("openai", o1), ("gemini (open)", g1), ("gemini (blocked)", g2)]
        print(f"  {'provider':18s} {'verdict':8s} {'confidence':11s} evidence")
        for name, a in rows:
            src = a.binding.source.value if a.binding else "-"
            print(f"  {name:18s} {a.verdict.value:8s} {a.confidence.value:11s} {src}")


if __name__ == "__main__":
    asyncio.run(main())
