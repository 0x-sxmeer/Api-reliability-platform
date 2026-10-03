"""
Quota Engine against the three real adapters, the real CallExecutor and a
real SQLite ledger (only the network is mocked). This is where the claims in
the Phase 3 report are demonstrated end to end: header-derived HIGH
confidence for Anthropic/OpenAI, ledger-derived lower confidence for Gemini,
the OpenAI project scope binding, and unknown-resume handling.

Also here: the architecture-rule tests. They inspect the engine's CODE via
the AST, so vendor names in explanatory docstrings/comments don't trip them
but a vendor name in an identifier, string literal or import would.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from gateway.adapters import gemini as gemini_module
from gateway.adapters.anthropic import AnthropicAdapter
from gateway.adapters.gemini import GeminiAdapter
from gateway.adapters.openai import OpenAIAdapter
from gateway.core.adapter import AdapterRequest, ProviderAdapter
from gateway.core.executor import CallExecutor
from gateway.core.types import CallOutcome, LimitingUnit, QuotaDimension, utcnow
from gateway.engines.quota import BlockReason, Confidence, QuotaEngine, Source, Verdict
from gateway.ledger.store import LedgerStore, SqliteLedgerStore

ORG, PROJECT, KEY = LimitingUnit.ORGANIZATION, LimitingUnit.PROJECT, LimitingUnit.KEY
FIXED_NOW = datetime(2026, 9, 30, 12, 0, 0, tzinfo=UTC)  # 05:00 Pacific


class FixedClock:
    def __init__(self, now: datetime = FIXED_NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def ledger(tmp_path) -> SqliteLedgerStore:
    return SqliteLedgerStore(tmp_path / "integration.db")


def _client(base_url: str, handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(handler))


def anthropic_adapter(handler) -> AnthropicAdapter:
    return AnthropicAdapter("k", http_client=_client("https://api.anthropic.com/v1", handler))


def openai_adapter(handler) -> OpenAIAdapter:
    return OpenAIAdapter("k", http_client=_client("https://api.openai.com/v1", handler))


def gemini_adapter(handler) -> GeminiAdapter:
    return GeminiAdapter("k", http_client=_client("https://generativelanguage.googleapis.com/v1beta", handler))


def anthropic_req(model: str = "model-a") -> AdapterRequest:
    return AdapterRequest(operation="messages.create", payload={"model": model, "max_tokens": 5, "messages": []})


def openai_req(model: str = "model-a") -> AdapterRequest:
    return AdapterRequest(operation="chat.completions.create", payload={"model": model, "messages": []})


def gemini_req(model: str = "model-a") -> AdapterRequest:
    return AdapterRequest(operation="generateContent", payload={"contents": []}, extra={"model": model})


async def call(engine: QuotaEngine, ledger: LedgerStore, adapter: ProviderAdapter, request: AdapterRequest):
    """What Phase 5's router will do: execute, then feed the engine."""
    result = await CallExecutor(adapter, ledger, identity_key="team-a").execute(request)
    engine.observe(adapter, request, snapshot=result.rate_limit, error=result.classified_error)
    return result


def rfc3339(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ============================================ Anthropic: header-derived, HIGH


async def test_anthropic_headers_give_a_high_confidence_answer(ledger) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "anthropic-ratelimit-requests-limit": "50",
                "anthropic-ratelimit-requests-remaining": "40",
                "anthropic-ratelimit-requests-reset": rfc3339(utcnow() + timedelta(seconds=45)),
                "anthropic-ratelimit-tokens-limit": "100000",
                "anthropic-ratelimit-tokens-remaining": "30000",
            },
            json={"model": "model-a", "usage": {"input_tokens": 1, "output_tokens": 1}},
        )

    adapter = anthropic_adapter(handler)
    engine = QuotaEngine(ledger)
    engine.register(adapter)
    await call(engine, ledger, adapter, anthropic_req())

    out = await engine.assess(adapter, anthropic_req())
    assert out.verdict is Verdict.OPEN
    assert out.confidence is Confidence.HIGH
    assert out.binding is not None
    assert out.binding.source is Source.HEADER
    assert out.binding.dimension is QuotaDimension.TOKENS  # 30% is tighter than 80%
    assert out.binding.unit is ORG
    assert out.adding_helps(KEY) is False, "the project's founding mistake: more keys under one org"
    assert out.adding_helps(ORG) is True


async def test_anthropic_529_does_not_erase_what_we_knew_or_block(ledger) -> None:
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(
                200,
                headers={
                    "anthropic-ratelimit-requests-limit": "50",
                    "anthropic-ratelimit-requests-remaining": "40",
                    "anthropic-ratelimit-requests-reset": rfc3339(utcnow() + timedelta(seconds=45)),
                    "anthropic-ratelimit-tokens-limit": "100000",
                    "anthropic-ratelimit-tokens-remaining": "90000",
                },
                json={"model": "model-a", "usage": {"input_tokens": 1, "output_tokens": 1}},
            )
        return httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})

    adapter = anthropic_adapter(handler)
    engine = QuotaEngine(ledger)
    engine.register(adapter)
    await call(engine, ledger, adapter, anthropic_req())
    await call(engine, ledger, adapter, anthropic_req())
    out = await engine.assess(adapter, anthropic_req())
    assert out.verdict is Verdict.OPEN, "overload is the provider's capacity, not our quota"
    assert out.confidence is Confidence.HIGH


async def test_anthropic_spend_cap_is_definitive_with_a_stated_resume_time(ledger) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "You will regain access on 2026-10-01 at 00:00 UTC.",
                    "details": {"error_code": "enforced_spend_limit_reached"},
                },
            },
        )

    clock = FixedClock()
    adapter = anthropic_adapter(handler)
    engine = QuotaEngine(ledger, clock=clock)
    engine.register(adapter)
    await call(engine, ledger, adapter, anthropic_req())

    out = await engine.assess(adapter, anthropic_req())
    resume = datetime(2026, 10, 1, 0, 0, tzinfo=UTC)
    assert out.verdict is Verdict.BLOCKED and out.block_reason is BlockReason.QUOTA_EXHAUSTED
    assert out.resume_at == resume and out.retry_at == resume
    assert out.confidence is Confidence.HIGH and out.provisional is False
    clock.now = resume
    assert (await engine.assess(adapter, anthropic_req())).verdict is Verdict.PROBE


async def test_anthropic_buckets_are_per_model_so_one_model_cannot_mask_another(ledger) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        big = b"model-big" in request.content
        remaining = "0" if big else "900"
        return httpx.Response(
            200,
            headers={
                "anthropic-ratelimit-requests-limit": "1000",
                "anthropic-ratelimit-requests-remaining": remaining,
                "anthropic-ratelimit-requests-reset": rfc3339(utcnow() + timedelta(seconds=30)),
                "anthropic-ratelimit-tokens-limit": "100000",
                "anthropic-ratelimit-tokens-remaining": "90000",
            },
            json={"model": "m", "usage": {"input_tokens": 1, "output_tokens": 1}},
        )

    adapter = anthropic_adapter(handler)
    engine = QuotaEngine(ledger)
    engine.register(adapter)
    await call(engine, ledger, adapter, anthropic_req("model-big"))
    await call(engine, ledger, adapter, anthropic_req("model-small"))
    assert (await engine.assess(adapter, anthropic_req("model-big"))).verdict is Verdict.BLOCKED
    assert (await engine.assess(adapter, anthropic_req("model-small"))).verdict is Verdict.OPEN


# ================================== OpenAI: dual scope, and unknown resume time


def _openai_ok(project_remaining: str | None, org_remaining_tokens: str = "9800"):
    def handler(request: httpx.Request) -> httpx.Response:
        headers = {
            "x-ratelimit-limit-requests": "200",
            "x-ratelimit-remaining-requests": "199",
            "x-ratelimit-limit-tokens": "10000",
            "x-ratelimit-remaining-tokens": org_remaining_tokens,
            "x-ratelimit-reset-tokens": "6s",
            "x-request-id": "req_1",
        }
        if project_remaining is not None:
            headers.update(
                {
                    "x-ratelimit-limit-project-tokens": "60000",
                    "x-ratelimit-remaining-project-tokens": project_remaining,
                    "x-ratelimit-reset-project-tokens": "3s",
                }
            )
        return httpx.Response(
            200, headers=headers, json={"model": "model-a", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        )

    return handler


async def test_openai_project_scope_binds_and_the_engine_says_which_fix_helps(ledger) -> None:
    adapter = openai_adapter(_openai_ok(project_remaining="3000"))  # 5% of the project's limit
    engine = QuotaEngine(ledger)
    engine.register(adapter)
    await call(engine, ledger, adapter, openai_req())

    out = await engine.assess(adapter, openai_req())
    assert out.verdict is Verdict.LOW
    assert out.confidence is Confidence.HIGH
    assert out.binding is not None and out.binding.unit is PROJECT, "org has room; the project does not"
    assert {c.unit for c in out.constraints} == {ORG, PROJECT}
    assert out.adding_helps(KEY) is False
    assert out.adding_helps(PROJECT) is True, "a new project has its own limit"


async def test_openai_org_binding_when_the_project_has_room(ledger) -> None:
    adapter = openai_adapter(_openai_ok(project_remaining="59000", org_remaining_tokens="500"))
    engine = QuotaEngine(ledger)
    engine.register(adapter)
    await call(engine, ledger, adapter, openai_req())
    out = await engine.assess(adapter, openai_req())
    assert out.binding is not None and out.binding.unit is ORG
    assert out.adding_helps(PROJECT) is False, "more projects would not add org-level capacity"


async def test_openai_without_project_headers_behaves_as_a_single_scope(ledger) -> None:
    adapter = openai_adapter(_openai_ok(project_remaining=None))
    engine = QuotaEngine(ledger)
    engine.register(adapter)
    await call(engine, ledger, adapter, openai_req())
    out = await engine.assess(adapter, openai_req())
    assert {c.unit for c in out.constraints} == {ORG}


async def test_openai_insufficient_quota_is_exhausted_with_unknown_resume_then_probes(ledger) -> None:
    """The common case the brief asked for: a major provider that never states
    when access returns."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"message": "You exceeded your current quota.", "type": "insufficient_quota",
                            "code": "insufficient_quota"}},
        )

    clock = FixedClock()
    adapter = openai_adapter(handler)
    engine = QuotaEngine(ledger, clock=clock)
    engine.register(adapter)
    await call(engine, ledger, adapter, openai_req())

    out = await engine.assess(adapter, openai_req())
    assert out.verdict is Verdict.BLOCKED and out.block_reason is BlockReason.QUOTA_EXHAUSTED
    assert out.resume_at is None
    assert out.retry_at == FIXED_NOW + timedelta(minutes=5)
    assert out.confidence is Confidence.HIGH and out.provisional is False
    clock.now += timedelta(minutes=5)
    assert (await engine.assess(adapter, openai_req())).verdict is Verdict.PROBE
    assert engine.claim_probe(adapter, openai_req()) is True
    assert engine.claim_probe(adapter, openai_req()) is False


# ================================== Gemini: no headers, ledger-derived, lower confidence


def _gemini_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [{"content": {"parts": [{"text": "hi"}], "role": "model"}}],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2},
            "modelVersion": "model-a",
        },
    )


async def test_gemini_has_no_header_signal_yet_the_engine_still_gives_a_sane_answer(ledger) -> None:
    adapter = gemini_adapter(_gemini_ok)
    engine = QuotaEngine(ledger)
    engine.register(adapter, limits={"*": {"rpm": 5, "rpd": 100}})
    r = await call(engine, ledger, adapter, gemini_req())
    assert r.rate_limit is not None and r.rate_limit.signal_quality.value == "none"

    out = await engine.assess(adapter, gemini_req())
    assert out.verdict is Verdict.OPEN
    assert out.confidence is Confidence.LOW, "lower confidence, as promised, not a fabricated HIGH"
    assert all(c.source is Source.LEDGER for c in out.constraints)
    assert out.binding is not None and out.binding.unit is PROJECT
    assert {c.label: c.remaining for c in out.constraints} == {"rpm": 4, "rpd": 99}
    assert any(u.startswith("tpm:") for u in out.unmonitored)
    assert out.adding_helps(KEY) is False, "Gemini limits are per project, NOT per key"


async def test_gemini_ledger_path_blocks_at_the_configured_limit(ledger) -> None:
    adapter = gemini_adapter(_gemini_ok)
    engine = QuotaEngine(ledger)
    engine.register(adapter, limits={"*": {"rpm": 3}})
    for _ in range(3):
        await call(engine, ledger, adapter, gemini_req())
    out = await engine.assess(adapter, gemini_req())
    assert out.verdict is Verdict.BLOCKED
    assert out.block_reason is BlockReason.HEADROOM
    assert out.confidence is Confidence.MEDIUM
    assert out.retry_at is not None and out.retry_at > utcnow()


async def test_gemini_limits_are_counted_per_model_from_the_ledger(ledger) -> None:
    adapter = gemini_adapter(_gemini_ok)
    engine = QuotaEngine(ledger)
    engine.register(adapter, limits={"*": {"rpm": 3}})
    for _ in range(3):
        await call(engine, ledger, adapter, gemini_req("model-a"))
    await call(engine, ledger, adapter, gemini_req("model-b"))
    assert (await engine.assess(adapter, gemini_req("model-a"))).verdict is Verdict.BLOCKED
    b = await engine.assess(adapter, gemini_req("model-b"))
    assert b.verdict is Verdict.OPEN
    assert {c.label: c.remaining for c in b.constraints}["rpm"] == 2


async def test_executor_stamps_the_bucket_on_ledger_events(ledger) -> None:
    adapter = gemini_adapter(_gemini_ok)
    await CallExecutor(adapter, ledger, identity_key="t").execute(gemini_req("model-z"))
    assert await ledger.count_since(provider="gemini", quota_bucket="model-z") == 1
    assert await ledger.count_since(provider="gemini", quota_bucket="other") == 0


async def test_gemini_spend_is_unmonitored_because_gemini_calls_are_unpriced(ledger) -> None:
    """The brief expected sum_cost_since to back the Gemini fallback. It can't
    today: there are no Gemini prices, so every cost is NULL and the sum is 0."""
    adapter = gemini_adapter(_gemini_ok)
    engine = QuotaEngine(ledger)
    engine.register(adapter, limits={"*": {"rpm": 50, "spend_10m": 10.0}})
    await call(engine, ledger, adapter, gemini_req())
    assert await ledger.sum_cost_since(provider="gemini") == 0.0  # the trap
    out = await engine.assess(adapter, gemini_req())
    assert any(u.startswith("spend_10m:") and "no known price" in u for u in out.unmonitored)
    assert "spend_10m" not in {c.label for c in out.constraints}


async def test_gemini_daily_quota_is_provisional_with_the_verified_midnight_pacific_resume(
    ledger, monkeypatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"code": 429, "message": "Quota exceeded.", "status": "RESOURCE_EXHAUSTED",
                            "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                         "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}},
        )

    monkeypatch.setattr(gemini_module, "utcnow", lambda: FIXED_NOW)
    clock = FixedClock()
    adapter = gemini_adapter(handler)
    engine = QuotaEngine(ledger, clock=clock)
    engine.register(adapter)
    await call(engine, ledger, adapter, gemini_req())

    out = await engine.assess(adapter, gemini_req())
    assert out.verdict is Verdict.BLOCKED and out.block_reason is BlockReason.QUOTA_EXHAUSTED
    assert out.provisional is True and out.confidence is Confidence.LOW
    assert out.resume_at == datetime(2026, 10, 1, 7, 0, tzinfo=UTC), "next midnight Pacific (PDT) in UTC"
    assert out.retry_at == FIXED_NOW + timedelta(seconds=30), "probe early: the PerDay diagnosis is a heuristic"


async def test_gemini_per_minute_429_is_a_short_cooldown_not_an_exhaustion(ledger) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"code": 429, "message": "x", "status": "RESOURCE_EXHAUSTED",
                            "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                         "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
                                        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"}]}},
        )

    clock = FixedClock()
    adapter = gemini_adapter(handler)
    engine = QuotaEngine(ledger, clock=clock)
    engine.register(adapter)
    await call(engine, ledger, adapter, gemini_req())
    out = await engine.assess(adapter, gemini_req())
    assert out.block_reason is BlockReason.COOLDOWN
    assert out.retry_at == FIXED_NOW + timedelta(seconds=34)


async def test_one_engine_serves_all_three_providers_at_once(ledger) -> None:
    a = anthropic_adapter(lambda r: httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error", "message": ""}}))
    o = openai_adapter(_openai_ok(project_remaining="30000"))
    g = gemini_adapter(_gemini_ok)
    engine = QuotaEngine(ledger)
    engine.register(a)
    engine.register(o)
    engine.register(g, limits={"*": {"rpm": 5}})
    await call(engine, ledger, o, openai_req())
    await call(engine, ledger, g, gemini_req())
    verdicts = {
        "a": (await engine.assess(a, anthropic_req())).verdict,
        "o": (await engine.assess(o, openai_req())).verdict,
        "g": (await engine.assess(g, gemini_req())).verdict,
    }
    assert verdicts == {"a": Verdict.UNKNOWN, "o": Verdict.OPEN, "g": Verdict.OPEN}


# ======================================================== architecture rule

ENGINES_DIR = Path(__file__).resolve().parent.parent / "src" / "gateway" / "engines"
VENDOR_WORDS = ("anthropic", "openai", "gemini", "claude", "gpt", "google", "stripe", "twilio")


def _code_nodes(tree: ast.Module):
    """Every node except docstrings. Comments never appear in an AST."""
    docstring_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstring_ids.add(id(body[0].value))
    for node in ast.walk(tree):
        if id(node) not in docstring_ids:
            yield node


def _engine_files() -> list[Path]:
    files = sorted(p for p in ENGINES_DIR.glob("*.py"))
    assert any(p.name == "quota.py" for p in files)
    assert any(p.name == "validator.py" for p in files), "validator.py must be in engines/"
    return files


@pytest.mark.parametrize("path", _engine_files(), ids=lambda p: p.name)
def test_engine_code_names_no_vendor(path: Path) -> None:
    offenders = []
    for node in _code_nodes(ast.parse(path.read_text())):
        texts: list[str] = []
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            texts.append(node.value)
        elif isinstance(node, ast.Name):
            texts.append(node.id)
        elif isinstance(node, ast.Attribute):
            texts.append(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            texts.extend(a.name for a in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                texts.append(node.module)
        for t in texts:
            if any(w in t.lower() for w in VENDOR_WORDS):
                offenders.append((getattr(node, "lineno", "?"), t))
    assert not offenders, f"vendor-specific knowledge outside adapters/: {offenders}"


@pytest.mark.parametrize("path", _engine_files(), ids=lambda p: p.name)
def test_engines_never_import_the_adapters_package(path: Path) -> None:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("gateway.adapters"), node.module
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("gateway.adapters") for a in node.names)


def _public_members(cls: type) -> set[str]:
    return {n for n in dir(cls) if not n.startswith("_")}


def test_the_engine_only_calls_methods_that_exist_on_the_abstract_interfaces() -> None:
    """'The Quota Engine must only call ProviderAdapter methods and
    LedgerStore methods.' Checked structurally: every attribute the engine
    reads off an adapter or the ledger must be declared on the ABC, so it
    cannot be reaching for something only one concrete class has."""
    tree = ast.parse((ENGINES_DIR / "quota.py").read_text())
    adapter_attrs: set[str] = set()
    ledger_attrs: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        v = node.value
        if (isinstance(v, ast.Name) and v.id == "adapter") or (
            isinstance(v, ast.Attribute) and v.attr == "adapter"  # reg.adapter.x
        ):
            adapter_attrs.add(node.attr)
        elif isinstance(v, ast.Attribute) and v.attr == "_ledger":
            ledger_attrs.add(node.attr)
    assert {"provider_name", "quota_bucket", "quota_windows", "get_limiting_unit"} <= adapter_attrs
    assert adapter_attrs <= _public_members(ProviderAdapter), adapter_attrs - _public_members(ProviderAdapter)
    assert {"count_since", "sum_cost_since", "count_unpriced_since"} <= ledger_attrs
    assert ledger_attrs <= _public_members(LedgerStore), ledger_attrs - _public_members(LedgerStore)


def test_validator_only_imports_core_types_not_adapter_classes() -> None:
    """The Response Validator must only import from gateway.core, not from
    gateway.adapters or gateway.ledger. It reads CallResult, not the raw
    adapter/ledger APIs."""
    tree = ast.parse((ENGINES_DIR / "validator.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("gateway.adapters"), (
                f"validator.py imports {node.module}: vendor knowledge leak"
            )
            # The validator may import from gateway.core (types, executor)
            # but NOT from gateway.ledger (it does no ledger I/O).
            assert not node.module.startswith("gateway.ledger"), (
                f"validator.py imports {node.module}: the validator is stateless "
                "and should not access the ledger directly"
            )


# ============================================ Phase 4: Validator integration tests

from gateway.engines.validator import (
    FindingCategory,
    ResponseValidator,
    ValidationVerdict,
)


async def test_anthropic_sse_error_after_200_is_caught_by_validator(ledger) -> None:
    """THE documented Phase 4 case: Anthropic's SSE `event: error` after a 200.
    The executor sets SUSPECTED_SILENT_FAILURE; the validator confirms INVALID."""

    def handler(request: httpx.Request) -> httpx.Response:
        # Realistic SSE stream: starts normally, then an error event mid-stream.
        sse_body = (
            "event: message_start\n"
            'data: {"type":"message_start","message":{"model":"claude-sonnet","usage":{"input_tokens":12,"output_tokens":0}}}\n\n'
            "event: content_block_start\n"
            'data: {"type":"content_block_start","index":0}\n\n'
            "event: content_block_delta\n"
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n'
            "event: error\n"
            'data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n'
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "request-id": "req_test"},
            content=sse_body.encode(),
        )

    adapter = anthropic_adapter(handler)
    executor = CallExecutor(adapter, ledger, identity_key="team-a")
    request = anthropic_req()
    result = await executor.execute(request)

    assert result.event.outcome is CallOutcome.SUSPECTED_SILENT_FAILURE
    assert len(result.bad_patterns) == 1
    assert result.bad_patterns[0].pattern_name == "streaming_error_after_200"
    assert result.bad_patterns[0].confidence == 0.97

    validator = ResponseValidator()
    validation = validator.validate(result)
    assert validation.verdict is ValidationVerdict.INVALID
    assert validation.confidence == 0.97
    assert len(validation.findings) == 1
    assert validation.findings[0].category is FindingCategory.BAD_PATTERN
    assert validation.findings[0].name == "streaming_error_after_200"
    assert validation.should_retry is True


async def test_anthropic_json_error_body_on_2xx_is_caught_by_validator(ledger) -> None:
    """Anthropic's other bad-pattern case: a JSON error object under a 2xx."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"request-id": "req_test2"},
            json={"type": "error", "error": {"type": "server_error", "message": "Internal error"}},
        )

    adapter = anthropic_adapter(handler)
    result = await CallExecutor(adapter, ledger, identity_key="team-a").execute(anthropic_req())
    assert result.event.outcome is CallOutcome.SUSPECTED_SILENT_FAILURE
    assert any(m.pattern_name == "json_error_body_on_2xx" for m in result.bad_patterns)

    validation = ResponseValidator().validate(result)
    assert validation.verdict is ValidationVerdict.INVALID
    assert validation.confidence == 0.95


async def test_gemini_blocked_prompt_is_caught_by_validator(ledger) -> None:
    """Gemini's bad-pattern case: 200 but promptFeedback.blockReason and no candidates."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "promptFeedback": {
                    "blockReason": "SAFETY",
                    "safetyRatings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "HIGH"}],
                },
                "modelVersion": "model-a",
            },
        )

    adapter = gemini_adapter(handler)
    result = await CallExecutor(adapter, ledger, identity_key="team-a").execute(gemini_req())
    assert result.event.outcome is CallOutcome.SUSPECTED_SILENT_FAILURE
    assert any(m.pattern_name == "blocked_before_generation_no_candidates" for m in result.bad_patterns)

    validation = ResponseValidator().validate(result)
    assert validation.verdict is ValidationVerdict.INVALID
    assert validation.confidence == 0.90
    assert validation.findings[0].retryable_different_provider is True


async def test_openai_clean_response_is_valid(ledger) -> None:
    """OpenAI has no documented bad patterns; a clean response should be VALID."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "x-ratelimit-limit-requests": "200",
                "x-ratelimit-remaining-requests": "199",
                "x-ratelimit-limit-tokens": "10000",
                "x-ratelimit-remaining-tokens": "9900",
                "x-ratelimit-reset-tokens": "6s",
                "x-request-id": "req_clean",
            },
            json={"model": "model-a", "usage": {"prompt_tokens": 10, "completion_tokens": 20}},
        )

    adapter = openai_adapter(handler)
    result = await CallExecutor(adapter, ledger, identity_key="team-a").execute(openai_req())
    assert result.event.outcome is CallOutcome.SUCCESS
    assert result.bad_patterns == []

    validation = ResponseValidator().validate(result)
    assert validation.verdict is ValidationVerdict.VALID
    assert validation.confidence == 0.0
    assert validation.findings == ()


async def test_validator_on_error_response_gives_routing_advice(ledger) -> None:
    """A 429 from Anthropic: the validator should say INVALID + retryable."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "30"},
            json={
                "type": "error",
                "error": {"type": "rate_limit_error", "message": "Rate limit exceeded"},
            },
        )

    adapter = anthropic_adapter(handler)
    result = await CallExecutor(adapter, ledger, identity_key="team-a").execute(anthropic_req())
    assert result.event.outcome is CallOutcome.FAILURE
    assert result.classified_error is not None

    validation = ResponseValidator().validate(result)
    assert validation.verdict is ValidationVerdict.INVALID
    assert validation.should_retry is True
    error_findings = [f for f in validation.findings if f.category is FindingCategory.ERROR_RESPONSE]
    assert len(error_findings) == 1
    assert error_findings[0].retryable_same_provider is True


async def test_validator_works_across_all_three_providers_in_one_session(ledger) -> None:
    """One validator instance validating results from all three providers."""
    validator = ResponseValidator()

    # Anthropic: SSE error -> INVALID
    a = anthropic_adapter(lambda r: httpx.Response(
        200, content=(
            b"event: error\n"
            b'data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n'
        ),
    ))
    r_a = await CallExecutor(a, ledger, identity_key="t").execute(anthropic_req())
    v_a = validator.validate(r_a)

    # OpenAI: clean -> VALID
    o = openai_adapter(lambda r: httpx.Response(
        200, json={"model": "m", "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    ))
    r_o = await CallExecutor(o, ledger, identity_key="t").execute(openai_req())
    v_o = validator.validate(r_o)

    # Gemini: blocked -> INVALID
    g = gemini_adapter(lambda r: httpx.Response(
        200, json={"promptFeedback": {"blockReason": "SAFETY"}, "modelVersion": "m"},
    ))
    r_g = await CallExecutor(g, ledger, identity_key="t").execute(gemini_req())
    v_g = validator.validate(r_g)

    assert v_a.verdict is ValidationVerdict.INVALID
    assert v_o.verdict is ValidationVerdict.VALID
    assert v_g.verdict is ValidationVerdict.INVALID

