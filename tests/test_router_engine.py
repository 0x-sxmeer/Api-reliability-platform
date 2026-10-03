"""
Unit tests for the Routing Engine (Phase 5).
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from gateway.core.adapter import AdapterRequest, ProviderAdapter
from gateway.core.executor import CallExecutor, CallResult
from gateway.core.types import (
    CallOutcome,
    ClassifiedError,
    ErrorCategory,
    KnownBadPatternMatch,
    LimitingUnit,
    RateLimitSnapshot,
    UsageInfo,
)
from gateway.engines.quota import (
    Confidence,
    QuotaAssessment,
    QuotaEngine,
    QuotaWindow,
    Verdict,
)
from gateway.engines.router import (
    RoutingConfig,
    RoutingEngine,
    RoutingTarget,
)
from gateway.engines.validator import (
    ResponseValidator,
    ValidationResult,
    ValidationVerdict,
)
from gateway.ledger.store import SqliteLedgerStore

# ------------------------------------------------------------------ Fakes

class FakeAdapter(ProviderAdapter):
    def __init__(self, name: str) -> None:
        self._name = name
        self.send_calls = 0

    @property
    def provider_name(self) -> str:
        return self._name

    def get_limiting_unit(self) -> LimitingUnit:
        return LimitingUnit.ORGANIZATION

    def quota_bucket(self, request: AdapterRequest) -> str:
        return "bucket"
        
    def quota_windows(self) -> tuple[QuotaWindow, ...]:
        return ()

    async def send(self, request: AdapterRequest) -> httpx.Response:
        self.send_calls += 1
        return httpx.Response(200, json={"fake": "response"})

    def parse_rate_limit_headers(self, response: httpx.Response) -> RateLimitSnapshot | None:
        return None

    def check_known_bad_patterns(self, response: httpx.Response) -> list[KnownBadPatternMatch]:
        return []

    def classify_error(self, response: httpx.Response) -> ClassifiedError:
        return ClassifiedError(ErrorCategory.UNKNOWN)
        
    def classify_exception(self, exc: Exception, request: AdapterRequest) -> ClassifiedError:
        return ClassifiedError(category=ErrorCategory.UNKNOWN)

    def extract_usage(self, response: httpx.Response) -> UsageInfo | None:
        return None

    def extract_request_id(self, response: httpx.Response) -> str | None:
        return "req-id"


class MockExecutor(CallExecutor):
    """An executor that yields pre-programmed results without hitting the network/ledger."""
    def __init__(self, adapter: ProviderAdapter, results: list[CallResult]):
        super().__init__(adapter, SqliteLedgerStore(":memory:"), identity_key="test-identity")
        self._results = results
        self.execute_calls = 0

    async def execute(self, request: AdapterRequest) -> CallResult:
        if self.execute_calls < len(self._results):
            res = self._results[self.execute_calls]
        else:
            res = self._results[-1]
        self.execute_calls += 1
        return res


class MockQuotaEngine(QuotaEngine):
    def __init__(self, assessments: list[QuotaAssessment] | None = None):
        super().__init__(ledger=SqliteLedgerStore(":memory:"))
        self._assessments = assessments or []
        self.assess_calls = 0
        self.observe_calls = 0

    async def assess(self, adapter: ProviderAdapter, request: AdapterRequest) -> QuotaAssessment:
        if self.assess_calls < len(self._assessments):
            res = self._assessments[self.assess_calls]
        else:
            # Default to OPEN if no more pre-programmed assessments
            res = QuotaAssessment(adapter.provider_name, "test-bucket", Verdict.OPEN, Confidence.NONE, datetime.now(UTC))
        self.assess_calls += 1
        return res

    def observe(self, adapter: ProviderAdapter, request: AdapterRequest, *, snapshot: RateLimitSnapshot | None = None, error: ClassifiedError | None = None, at: datetime | None = None) -> None:
        self.observe_calls += 1


class MockValidator(ResponseValidator):
    def __init__(self, results: list[ValidationResult]):
        super().__init__()
        self._results = results
        self.validate_calls = 0

    def validate(self, result: CallResult) -> ValidationResult:
        if self.validate_calls < len(self._results):
            res = self._results[self.validate_calls]
        else:
            res = self._results[-1]
        self.validate_calls += 1
        return res


def make_call_result() -> CallResult:
    # A dummy CallResult for testing
    from gateway.ledger.events import GatewayEvent
    event = GatewayEvent(
        id="test-id",
        timestamp=datetime.now(UTC),
        provider="test-provider",
        identity_key="test-identity",
        operation="test-operation",
        payload_hash="hash",
        outcome=CallOutcome.SUCCESS,
    )
    return CallResult(event=event, response=httpx.Response(200), exception=None)


# ------------------------------------------------------------------ Tests

@pytest.mark.asyncio
async def test_router_single_provider_clean_response():
    """1. Single provider, clean response -> VALID on first attempt."""
    adapter = FakeAdapter("provA")
    result1 = make_call_result()
    executor = MockExecutor(adapter, [result1])
    target = RoutingTarget(adapter, executor)

    quota = MockQuotaEngine()
    validator = MockValidator([
        ValidationResult(ValidationVerdict.VALID, 0.0, (), False, False, False, CallOutcome.SUCCESS, ())
    ])

    router = RoutingEngine(quota, validator, RoutingConfig(sleep_fn=asyncio.sleep))
    router.register(target)

    outcome = await router.route(AdapterRequest(operation="test", payload={}))

    assert outcome.succeeded is True
    assert outcome.final_result is result1
    assert len(outcome.attempts) == 1
    assert executor.execute_calls == 1
    assert quota.assess_calls == 1
    assert quota.observe_calls == 1
    assert validator.validate_calls == 1


@pytest.mark.asyncio
async def test_router_single_provider_transient_error_retries():
    """2. Single provider, transient error on attempt 1, clean on attempt 2 -> retries."""
    adapter = FakeAdapter("provA")
    result1 = make_call_result()
    result2 = make_call_result()
    executor = MockExecutor(adapter, [result1, result2])
    target = RoutingTarget(adapter, executor)

    quota = MockQuotaEngine()
    
    # First attempt: INVALID, should_retry=True, same provider (retry_on_different_provider=False)
    v1 = ValidationResult(ValidationVerdict.INVALID, 1.0, (), True, False, False, CallOutcome.FAILURE, ())
    # Second attempt: VALID
    v2 = ValidationResult(ValidationVerdict.VALID, 0.0, (), False, False, False, CallOutcome.SUCCESS, ())
    
    validator = MockValidator([v1, v2])

    sleeps = []
    async def fake_sleep(secs: float):
        sleeps.append(secs)

    router = RoutingEngine(quota, validator, RoutingConfig(sleep_fn=fake_sleep))
    router.register(target)

    outcome = await router.route(AdapterRequest(operation="test", payload={}))

    assert outcome.succeeded is True
    assert outcome.final_result is result2
    assert len(outcome.attempts) == 2
    assert executor.execute_calls == 2
    assert len(sleeps) == 1
    assert sleeps[0] == 1.0  # backoff_base_seconds


@pytest.mark.asyncio
async def test_router_single_provider_blocked_quota_waits():
    """3. Single provider, BLOCKED quota -> waits if within budget, retries."""
    adapter = FakeAdapter("provA")
    result1 = make_call_result()
    executor = MockExecutor(adapter, [result1])
    target = RoutingTarget(adapter, executor)

    now = datetime.now(UTC)
    # First assess: BLOCKED, retry_at in 5 seconds
    a1 = QuotaAssessment("provA", "test-bucket", Verdict.BLOCKED, Confidence.NONE, datetime.now(UTC), retry_at=now + timedelta(seconds=5))
    # Second assess (after sleep): OPEN
    a2 = QuotaAssessment("provA", "test-bucket", Verdict.OPEN, Confidence.NONE, datetime.now(UTC))
    
    quota = MockQuotaEngine([a1, a2])
    validator = MockValidator([
        ValidationResult(ValidationVerdict.VALID, 0.0, (), False, False, False, CallOutcome.SUCCESS, ())
    ])

    sleeps = []
    async def fake_sleep(secs: float):
        sleeps.append(secs)

    # config.max_wait_seconds defaults to 10.0, so 5s is within budget
    router = RoutingEngine(quota, validator, RoutingConfig(sleep_fn=fake_sleep))
    router.register(target)

    outcome = await router.route(AdapterRequest(operation="test", payload={}))

    assert outcome.succeeded is True
    assert len(outcome.attempts) == 1 # The block doesn't count as a failed API attempt in outcome.attempts because no CallResult was produced. Wait, the loop `continue`s, so it assesses again.
    assert executor.execute_calls == 1
    assert quota.assess_calls == 2
    assert len(sleeps) == 1
    assert 4.0 < sleeps[0] < 6.0


@pytest.mark.asyncio
async def test_router_two_providers_bad_pattern_failover():
    """4. Two providers, provider A returns bad pattern -> fails over to provider B."""
    adapterA = FakeAdapter("provA")
    resultA = make_call_result()
    executorA = MockExecutor(adapterA, [resultA])

    adapterB = FakeAdapter("provB")
    resultB = make_call_result()
    executorB = MockExecutor(adapterB, [resultB])

    quota = MockQuotaEngine()
    
    # provA: INVALID, should_retry=True, different provider
    vA = ValidationResult(ValidationVerdict.INVALID, 1.0, (), True, True, False, CallOutcome.SUSPECTED_SILENT_FAILURE, ())
    # provB: VALID
    vB = ValidationResult(ValidationVerdict.VALID, 0.0, (), False, False, False, CallOutcome.SUCCESS, ())
    
    validator = MockValidator([vA, vB])

    router = RoutingEngine(quota, validator, RoutingConfig())
    router.register(RoutingTarget(adapterA, executorA))
    router.register(RoutingTarget(adapterB, executorB))

    outcome = await router.route(AdapterRequest(operation="test", payload={}))

    assert outcome.succeeded is True
    assert outcome.final_result is resultB
    assert len(outcome.attempts) == 2
    assert executorA.execute_calls == 1
    assert executorB.execute_calls == 1


@pytest.mark.asyncio
async def test_router_two_providers_quota_blocked_failover():
    """5. Two providers, provider A quota BLOCKED -> immediately fails over to B."""
    adapterA = FakeAdapter("provA")
    executorA = MockExecutor(adapterA, [])

    adapterB = FakeAdapter("provB")
    resultB = make_call_result()
    executorB = MockExecutor(adapterB, [resultB])

    # provA: BLOCKED, no retry_at (immediate failover)
    aA = QuotaAssessment("provA", "test-bucket", Verdict.BLOCKED, Confidence.NONE, datetime.now(UTC))
    aB = QuotaAssessment("provB", "test-bucket", Verdict.OPEN, Confidence.NONE, datetime.now(UTC))
    quota = MockQuotaEngine([aA, aB])
    
    vB = ValidationResult(ValidationVerdict.VALID, 0.0, (), False, False, False, CallOutcome.SUCCESS, ())
    validator = MockValidator([vB])

    router = RoutingEngine(quota, validator, RoutingConfig())
    router.register(RoutingTarget(adapterA, executorA))
    router.register(RoutingTarget(adapterB, executorB))

    outcome = await router.route(AdapterRequest(operation="test", payload={}))

    assert outcome.succeeded is True
    assert outcome.final_result is resultB
    assert len(outcome.attempts) == 2  # ProvA block counts as an attempt record with no result
    assert outcome.attempts[0].target.adapter.provider_name == "provA"
    assert outcome.attempts[0].result is None
    assert outcome.attempts[1].target.adapter.provider_name == "provB"
    assert outcome.attempts[1].result is resultB
    assert executorA.execute_calls == 0
    assert executorB.execute_calls == 1


@pytest.mark.asyncio
async def test_router_all_providers_exhausted():
    """6. All providers exhausted -> returns last CallResult + RoutingOutcome."""
    adapterA = FakeAdapter("provA")
    resultA1 = make_call_result()
    resultA2 = make_call_result()
    executorA = MockExecutor(adapterA, [resultA1, resultA2])

    quota = MockQuotaEngine()
    
    # Always invalid, retry same provider
    v_invalid = ValidationResult(ValidationVerdict.INVALID, 1.0, (), True, False, False, CallOutcome.FAILURE, ())
    validator = MockValidator([v_invalid, v_invalid])

    sleeps = []
    async def fake_sleep(secs: float):
        sleeps.append(secs)

    router = RoutingEngine(quota, validator, RoutingConfig(max_attempts_per_provider=2, sleep_fn=fake_sleep))
    router.register(RoutingTarget(adapterA, executorA))

    outcome = await router.route(AdapterRequest(operation="test", payload={}))

    assert outcome.succeeded is False
    assert outcome.final_result is resultA2
    assert len(outcome.attempts) == 2
    assert executorA.execute_calls == 2
    assert len(sleeps) == 1


@pytest.mark.asyncio
async def test_router_bad_pattern_then_rate_limit_exhausted():
    """7. Bad pattern on A, rate limit on B, retries exhausted -> surfaced correctly."""
    adapterA = FakeAdapter("provA")
    resultA = make_call_result()
    executorA = MockExecutor(adapterA, [resultA])

    adapterB = FakeAdapter("provB")
    resultB1 = make_call_result()
    resultB2 = make_call_result()
    executorB = MockExecutor(adapterB, [resultB1, resultB2])

    quota = MockQuotaEngine()
    
    vA = ValidationResult(ValidationVerdict.INVALID, 1.0, (), True, True, False, CallOutcome.SUSPECTED_SILENT_FAILURE, ())
    vB1 = ValidationResult(ValidationVerdict.INVALID, 1.0, (), True, False, False, CallOutcome.FAILURE, ())
    vB2 = ValidationResult(ValidationVerdict.INVALID, 1.0, (), True, False, False, CallOutcome.FAILURE, ())
    validator = MockValidator([vA, vB1, vB2])

    sleeps = []
    async def fake_sleep(secs: float):
        sleeps.append(secs)

    # Total budget = 3
    router = RoutingEngine(quota, validator, RoutingConfig(max_total_attempts=3, sleep_fn=fake_sleep))
    router.register(RoutingTarget(adapterA, executorA))
    router.register(RoutingTarget(adapterB, executorB))

    outcome = await router.route(AdapterRequest(operation="test", payload={}))

    assert outcome.succeeded is False
    assert outcome.final_result is resultB2
    assert len(outcome.attempts) == 3
    assert executorA.execute_calls == 1
    assert executorB.execute_calls == 2


@pytest.mark.asyncio
async def test_router_max_total_attempts_respected():
    """8. max_total_attempts respected — never exceeds budget regardless of provider count."""
    adapterA = FakeAdapter("provA")
    executorA = MockExecutor(adapterA, [make_call_result(), make_call_result()])
    
    adapterB = FakeAdapter("provB")
    executorB = MockExecutor(adapterB, [make_call_result(), make_call_result()])
    
    adapterC = FakeAdapter("provC")
    executorC = MockExecutor(adapterC, [make_call_result(), make_call_result()])

    quota = MockQuotaEngine()
    # Always fail, failover to next provider
    v_failover = ValidationResult(ValidationVerdict.INVALID, 1.0, (), True, True, False, CallOutcome.FAILURE, ())
    validator = MockValidator([v_failover] * 10)

    # Budget is 2 total attempts across 3 providers
    router = RoutingEngine(quota, validator, RoutingConfig(max_total_attempts=2))
    router.register(RoutingTarget(adapterA, executorA))
    router.register(RoutingTarget(adapterB, executorB))
    router.register(RoutingTarget(adapterC, executorC))

    outcome = await router.route(AdapterRequest(operation="test", payload={}))

    assert outcome.succeeded is False
    assert len(outcome.attempts) == 2
    assert executorA.execute_calls == 1
    assert executorB.execute_calls == 1
    assert executorC.execute_calls == 0


@pytest.mark.asyncio
async def test_router_permanent_error_not_retried():
    """9. PERMANENT error is not retried (validator says should_retry=False)."""
    adapterA = FakeAdapter("provA")
    resultA = make_call_result()
    executorA = MockExecutor(adapterA, [resultA])

    adapterB = FakeAdapter("provB")
    executorB = MockExecutor(adapterB, [])

    quota = MockQuotaEngine()
    # INVALID but should_retry=False
    v_perm = ValidationResult(ValidationVerdict.INVALID, 1.0, (), False, False, False, CallOutcome.FAILURE, ())
    validator = MockValidator([v_perm])

    router = RoutingEngine(quota, validator, RoutingConfig())
    router.register(RoutingTarget(adapterA, executorA))
    router.register(RoutingTarget(adapterB, executorB))

    outcome = await router.route(AdapterRequest(operation="test", payload={}))

    assert outcome.succeeded is False
    assert len(outcome.attempts) == 1
    assert executorA.execute_calls == 1
    assert executorB.execute_calls == 0


@pytest.mark.asyncio
async def test_router_suspect_result_used_when_no_valid_alternative():
    """10. SUSPECT result is used when no VALID alternative is available."""
    adapterA = FakeAdapter("provA")
    resultA = make_call_result()
    executorA = MockExecutor(adapterA, [resultA])

    adapterB = FakeAdapter("provB")
    resultB = make_call_result()
    executorB = MockExecutor(adapterB, [resultB])

    quota = MockQuotaEngine()
    # A returns SUSPECT, failover recommended
    vA = ValidationResult(ValidationVerdict.SUSPECT, 0.6, (), True, True, False, CallOutcome.SUSPECTED_SILENT_FAILURE, ())
    # B returns INVALID, failover not recommended (exhausted)
    vB = ValidationResult(ValidationVerdict.INVALID, 1.0, (), True, False, False, CallOutcome.FAILURE, ())
    validator = MockValidator([vA, vB])

    sleeps = []
    async def fake_sleep(secs: float):
        sleeps.append(secs)

    router = RoutingEngine(quota, validator, RoutingConfig(max_attempts_per_provider=1, sleep_fn=fake_sleep))
    router.register(RoutingTarget(adapterA, executorA))
    router.register(RoutingTarget(adapterB, executorB))

    outcome = await router.route(AdapterRequest(operation="test", payload={}))

    assert outcome.succeeded is False  # Final state is not full success
    assert outcome.final_result is resultA  # Reverted back to the SUSPECT result A
    assert outcome.final_validation.verdict == ValidationVerdict.SUSPECT
    assert len(outcome.attempts) == 2
