"""
Quota Engine unit tests.

Every test here uses FakeAdapter, which names no vendor. That is deliberate
and is itself a check on the architecture rule: if the engine only works for
adapters it has special knowledge of, these fail. Real-adapter behaviour is
in test_quota_integration.py.

Time is injected (FakeClock) everywhere, so nothing sleeps and nothing is
flaky. Ledger events are written with explicit timestamps.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from gateway.core.adapter import AdapterRequest, ProviderAdapter
from gateway.core.types import (
    CallOutcome,
    ClassifiedError,
    ErrorCategory,
    KnownBadPatternMatch,
    LimitingUnit,
    QuotaDimension,
    QuotaWindow,
    RateLimitSnapshot,
    SignalQuality,
    UsageInfo,
    WindowKind,
)
from gateway.engines.quota import (
    BlockReason,
    Confidence,
    Demand,
    ProbePolicy,
    QuotaConfig,
    QuotaEngine,
    Source,
    Verdict,
    _window_bounds,
    adding_capacity_helps,
)
from gateway.ledger.events import GatewayEvent
from gateway.ledger.store import LedgerStore, SqliteLedgerStore

# 12:00 UTC on 2026-09-30 is 05:00 PDT, so the America/Los_Angeles calendar
# day is [2026-09-30T07:00Z, 2026-10-01T07:00Z).
NOW = datetime(2026, 9, 30, 12, 0, 0, tzinfo=UTC)
DAY_START = datetime(2026, 9, 30, 7, 0, 0, tzinfo=UTC)
DAY_END = datetime(2026, 10, 1, 7, 0, 0, tzinfo=UTC)

ORG, PROJECT, KEY = LimitingUnit.ORGANIZATION, LimitingUnit.PROJECT, LimitingUnit.KEY

WINDOWS = (
    QuotaWindow("rpm", QuotaDimension.REQUESTS, WindowKind.ROLLING, seconds=60),
    QuotaWindow("tpm", QuotaDimension.TOKENS, WindowKind.ROLLING, seconds=60),
    QuotaWindow("rpd", QuotaDimension.REQUESTS, WindowKind.CALENDAR_DAY, tz="America/Los_Angeles"),
    QuotaWindow("spend_10m", QuotaDimension.SPEND, WindowKind.ROLLING, seconds=600),
)


class FakeClock:
    def __init__(self, start: datetime = NOW) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kw: float) -> None:
        self.now += timedelta(**kw)


class FakeAdapter(ProviderAdapter):
    """A provider the engine has never heard of."""

    def __init__(
        self,
        name: str = "fakeprov",
        unit: LimitingUnit = ORG,
        windows: tuple[QuotaWindow, ...] = (),
    ) -> None:
        self._name, self._unit, self._windows = name, unit, windows

    @property
    def provider_name(self) -> str:
        return self._name

    def get_limiting_unit(self) -> LimitingUnit:
        return self._unit

    def quota_bucket(self, request: AdapterRequest) -> str:
        return str(request.payload.get("model", "m0"))

    def quota_windows(self) -> tuple[QuotaWindow, ...]:
        return self._windows

    def parse_rate_limit_headers(self, response: httpx.Response) -> RateLimitSnapshot:
        return RateLimitSnapshot(limiting_unit=self._unit)

    def classify_error(self, response: httpx.Response) -> ClassifiedError:
        return ClassifiedError(category=ErrorCategory.UNKNOWN)

    def check_known_bad_patterns(self, response: httpx.Response) -> list[KnownBadPatternMatch]:
        return []

    def classify_exception(self, exc: Exception, request: AdapterRequest) -> ClassifiedError:
        return ClassifiedError(category=ErrorCategory.UNKNOWN)

    def extract_usage(self, response: httpx.Response) -> UsageInfo | None:
        return None

    def extract_request_id(self, response: httpx.Response) -> str | None:
        return None

    async def send(self, request: AdapterRequest) -> httpx.Response:
        raise NotImplementedError


def req(model: str = "m0") -> AdapterRequest:
    return AdapterRequest(operation="op", payload={"model": model})


def snap(
    unit: LimitingUnit = ORG,
    *,
    requests: tuple[int, int] | None = (49, 50),
    tokens: tuple[int, int] | None = (99_500, 100_000),
    reset: datetime | None = None,
    quality: SignalQuality = SignalQuality.FULL,
    retry_after: float | None = None,
    extra: tuple[RateLimitSnapshot, ...] = (),
) -> RateLimitSnapshot:
    """(remaining, limit) pairs."""
    return RateLimitSnapshot(
        limiting_unit=unit,
        signal_quality=quality,
        requests_remaining=requests[0] if requests else None,
        requests_limit=requests[1] if requests else None,
        tokens_remaining=tokens[0] if tokens else None,
        tokens_limit=tokens[1] if tokens else None,
        reset_at=reset,
        retry_after_seconds=retry_after,
        additional_scopes=extra,
    )


def ev(
    ts: datetime,
    *,
    provider: str = "fakeprov",
    bucket: str | None = "m0",
    outcome: CallOutcome = CallOutcome.SUCCESS,
    cost: float | None = None,
    identity: str = "team-a",
) -> GatewayEvent:
    return GatewayEvent(
        timestamp=ts,
        identity_key=identity,
        provider=provider,
        operation="op",
        outcome=outcome,
        quota_bucket=bucket,
        cost_usd=cost,
    )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def ledger(tmp_path) -> SqliteLedgerStore:
    return SqliteLedgerStore(tmp_path / "quota.db")


def make_engine(ledger: LedgerStore, clock: FakeClock, adapter: FakeAdapter, **kw) -> QuotaEngine:
    limits = kw.pop("limits", None)
    engine = QuotaEngine(ledger, clock=clock, **kw)
    engine.register(adapter, limits=limits)
    return engine


# ============================================================ registration


def test_register_rejects_unknown_window_name_and_lists_valid_ones(ledger, clock) -> None:
    engine = QuotaEngine(ledger, clock=clock)
    with pytest.raises(ValueError, match=r"no quota window 'RPM'.*rpd.*rpm"):
        engine.register(FakeAdapter(windows=WINDOWS), limits={"*": {"RPM": 5}})


def test_register_says_so_when_adapter_declares_no_windows(ledger, clock) -> None:
    engine = QuotaEngine(ledger, clock=clock)
    with pytest.raises(ValueError, match="declares no countable windows"):
        engine.register(FakeAdapter(), limits={"*": {"rpm": 5}})


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_register_rejects_non_positive_limits(ledger, clock, bad) -> None:
    engine = QuotaEngine(ledger, clock=clock)
    with pytest.raises(ValueError, match="must be > 0"):
        engine.register(FakeAdapter(windows=WINDOWS), limits={"*": {"rpm": bad}})


def test_register_rejects_a_second_adapter_under_the_same_name(ledger, clock) -> None:
    engine = QuotaEngine(ledger, clock=clock)
    engine.register(FakeAdapter())
    with pytest.raises(ValueError, match="One credential per provider_name"):
        engine.register(FakeAdapter())


def test_register_same_adapter_twice_just_updates_limits(ledger, clock) -> None:
    engine = QuotaEngine(ledger, clock=clock)
    adapter = FakeAdapter(windows=WINDOWS)
    engine.register(adapter, limits={"*": {"rpm": 5}})
    engine.register(adapter, limits={"*": {"rpm": 9}})  # no error


def test_register_fails_fast_on_unusable_timezone(ledger, clock) -> None:
    bad = QuotaWindow("rpd", QuotaDimension.REQUESTS, WindowKind.CALENDAR_DAY, tz="Not/AZone")
    with pytest.raises(ValueError, match="unavailable"):
        QuotaEngine(ledger, clock=clock).register(FakeAdapter(windows=(bad,)))


def test_register_rejects_duplicate_window_names(ledger, clock) -> None:
    w = QuotaWindow("rpm", QuotaDimension.REQUESTS, WindowKind.ROLLING, seconds=60)
    with pytest.raises(ValueError, match="duplicate"):
        QuotaEngine(ledger, clock=clock).register(FakeAdapter(windows=(w, w)))


async def test_unregistered_adapter_is_rejected(ledger, clock) -> None:
    engine = QuotaEngine(ledger, clock=clock)
    with pytest.raises(ValueError, match="not registered"):
        await engine.assess(FakeAdapter(), req())
    with pytest.raises(ValueError, match="not registered"):
        engine.observe(FakeAdapter(), req())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": ""}, "non-empty"),
        ({"name": "x", "kind": WindowKind.ROLLING}, "needs seconds > 0"),
        ({"name": "x", "kind": WindowKind.ROLLING, "seconds": 0}, "needs seconds > 0"),
        ({"name": "x", "kind": WindowKind.CALENDAR_DAY}, "needs an IANA tz"),
    ],
)
def test_quota_window_validates_its_own_shape(kwargs, message) -> None:
    kwargs.setdefault("kind", WindowKind.ROLLING)
    with pytest.raises(ValueError, match=message):  # match: a bare ValueError could be the WRONG one
        QuotaWindow(dimension=QuotaDimension.REQUESTS, **kwargs)


# ======================================================== header-derived path


async def test_cold_start_is_unknown_not_optimistically_open(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.UNKNOWN
    assert out.confidence is Confidence.NONE
    assert out.binding is None
    assert out.adding_helps(KEY) is None, "no binding scope -> the engine must not guess"


async def test_fresh_full_snapshot_is_open_with_high_confidence(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), snapshot=snap(reset=NOW + timedelta(seconds=60)))
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.OPEN
    assert out.confidence is Confidence.HIGH
    # requests 49/50 = 0.98 is tighter than tokens 99.5k/100k = 0.995
    assert out.binding is not None
    assert out.binding.dimension is QuotaDimension.REQUESTS
    assert out.binding.source is Source.HEADER
    assert {c.dimension for c in out.constraints} == {QuotaDimension.REQUESTS, QuotaDimension.TOKENS}


async def test_stale_snapshot_drops_to_medium_confidence(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), snapshot=snap(reset=NOW + timedelta(seconds=300)))
    clock.advance(seconds=45)  # older than fresh_for (30s), before reset
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.OPEN
    assert out.confidence is Confidence.MEDIUM


async def test_partial_signal_is_capped_below_high(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), snapshot=snap(tokens=None, quality=SignalQuality.PARTIAL, reset=NOW + timedelta(seconds=60)))
    out = await engine.assess(a, req())
    assert out.confidence is Confidence.MEDIUM


async def test_zero_remaining_blocks_until_reset_with_high_confidence(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    reset = NOW + timedelta(seconds=20)
    engine.observe(a, req(), snapshot=snap(requests=(0, 50), reset=reset))
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.BLOCKED
    assert out.block_reason is BlockReason.HEADROOM
    assert out.retry_at == reset
    assert out.confidence is Confidence.HIGH
    assert out.binding is not None and out.binding.dimension is QuotaDimension.REQUESTS


async def test_block_expires_at_reset_assuming_replenishment(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), snapshot=snap(requests=(0, 50), reset=NOW + timedelta(seconds=20)))
    clock.advance(seconds=21)
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.OPEN
    assert out.confidence is Confidence.MEDIUM, "assumed, not measured"
    assert any(c.assumed_replenished for c in out.constraints)


async def test_block_with_no_reset_still_expires_so_it_cannot_deadlock(ledger, clock) -> None:
    """A blocked scope receives no calls, so nothing would ever refresh a
    remaining=0 reading. Without the no-reset expiry it would block forever."""
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), snapshot=snap(requests=(0, 50), reset=None))
    blocked = await engine.assess(a, req())
    assert blocked.verdict is Verdict.BLOCKED
    assert blocked.retry_at == NOW + timedelta(seconds=60)  # stale_after_no_reset
    clock.advance(seconds=61)
    assert (await engine.assess(a, req())).verdict is Verdict.OPEN


async def test_expired_observation_with_no_known_limit_is_ignored_not_trusted(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), snapshot=snap(requests=None, tokens=None))  # nothing -> not stored
    engine.observe(
        a, req(),
        snapshot=RateLimitSnapshot(limiting_unit=ORG, signal_quality=SignalQuality.PARTIAL, requests_remaining=3),
    )
    clock.advance(seconds=120)
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.UNKNOWN
    assert any("no limit known" in c for c in out.caveats)


async def test_low_headroom_verdict_below_threshold(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), snapshot=snap(requests=(4, 50), reset=NOW + timedelta(seconds=30)))  # 8%
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.LOW
    assert out.confidence is Confidence.HIGH


async def test_low_threshold_is_configurable(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a, config=QuotaConfig(low_headroom_fraction=0.5))
    engine.observe(a, req(), snapshot=snap(requests=(20, 50), reset=NOW + timedelta(seconds=30)))  # 40%
    assert (await engine.assess(a, req())).verdict is Verdict.LOW


async def test_response_with_no_numbers_does_not_overwrite_a_good_observation(ledger, clock) -> None:
    """A 529 or other headerless response must not erase what we knew."""
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), snapshot=snap(reset=NOW + timedelta(seconds=60)))
    engine.observe(a, req(), snapshot=snap(requests=None, tokens=None, quality=SignalQuality.NONE))
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.OPEN and out.confidence is Confidence.HIGH


async def test_buckets_are_isolated_from_each_other(ledger, clock) -> None:
    """The reason quota_bucket exists: a cheap model's headroom must not
    describe an expensive model's."""
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req("big"), snapshot=snap(requests=(0, 10), reset=NOW + timedelta(seconds=30)))
    engine.observe(a, req("small"), snapshot=snap(requests=(900, 1000), reset=NOW + timedelta(seconds=30)))
    assert (await engine.assess(a, req("big"))).verdict is Verdict.BLOCKED
    assert (await engine.assess(a, req("small"))).verdict is Verdict.OPEN
    third = await engine.assess(a, req("never-seen"))
    assert third.verdict is Verdict.UNKNOWN
    assert third.bucket == "never-seen"


async def test_token_demand_is_checked_against_remaining_tokens(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), snapshot=snap(tokens=(2_000, 100_000), reset=NOW + timedelta(seconds=30)))
    assert (await engine.assess(a, req(), demand=Demand(tokens=1_000))).verdict is not Verdict.BLOCKED
    out = await engine.assess(a, req(), demand=Demand(tokens=5_000))
    assert out.verdict is Verdict.BLOCKED
    assert out.binding is not None and out.binding.dimension is QuotaDimension.TOKENS


async def test_observe_rejects_naive_datetimes(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    with pytest.raises(ValueError, match="timezone-aware"):
        engine.observe(a, req(), snapshot=snap(), at=datetime(2026, 9, 30, 12, 0, 0))  # noqa: DTZ001 -- deliberately naive: asserting it is rejected


# ================================================================ multi-scope


def project_scope(remaining: int, limit: int = 60_000, reset: datetime | None = None) -> RateLimitSnapshot:
    return snap(PROJECT, requests=None, tokens=(remaining, limit), reset=reset or NOW + timedelta(seconds=30))


async def test_tightest_scope_binds_and_says_keys_will_not_help(ledger, clock) -> None:
    """OPEN question 1, answered: the org has lots of room, the project
    does not. A single-scope model would call this healthy org-level
    headroom and recommend the wrong fix."""
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    org = snap(ORG, requests=(190, 200), tokens=(9_000, 10_000), extra=(project_scope(15_000),))
    org = RateLimitSnapshot(**{**org.__dict__, "reset_at": NOW + timedelta(seconds=30)})
    engine.observe(a, req(), snapshot=org)
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.OPEN
    assert out.binding is not None and out.binding.unit is PROJECT  # 15k/60k = 25% is the tightest
    assert out.adding_helps(KEY) is False, "new keys share the project's limit"
    assert out.adding_helps(PROJECT) is True
    assert out.adding_helps(ORG) is True


async def test_org_binding_means_neither_keys_nor_projects_help(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    org = snap(ORG, requests=(10, 200), tokens=(9_000, 10_000), extra=(project_scope(59_000),), reset=NOW + timedelta(seconds=30))
    engine.observe(a, req(), snapshot=org)
    out = await engine.assess(a, req())
    assert out.binding is not None and out.binding.unit is ORG
    assert out.adding_helps(KEY) is False
    assert out.adding_helps(PROJECT) is False
    assert out.adding_helps(ORG) is True


async def test_exhausted_project_blocks_even_though_org_is_healthy(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    org = snap(ORG, requests=(190, 200), tokens=(9_000, 10_000), extra=(project_scope(0),), reset=NOW + timedelta(seconds=30))
    engine.observe(a, req(), snapshot=org)
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.BLOCKED
    assert out.binding is not None and out.binding.unit is PROJECT
    assert out.adding_helps(KEY) is False


async def test_a_later_response_without_the_project_scope_keeps_the_old_project_view(ledger, clock) -> None:
    """Project headers are 'present when a project-scoped limit applies', so
    their absence on one response is not proof the limit went away."""
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    reset = NOW + timedelta(seconds=60)
    engine.observe(a, req(), snapshot=snap(ORG, reset=reset, extra=(project_scope(0, reset=reset),)))
    engine.observe(a, req(), snapshot=snap(ORG, reset=reset))  # no additional_scopes this time
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.BLOCKED and out.binding is not None and out.binding.unit is PROJECT


@pytest.mark.parametrize(
    ("binding", "added", "helps"),
    [
        (KEY, KEY, True), (KEY, PROJECT, True), (KEY, ORG, True),
        (PROJECT, KEY, False), (PROJECT, PROJECT, True), (PROJECT, ORG, True),
        (ORG, KEY, False), (ORG, PROJECT, False), (ORG, ORG, True),
        # Outside the verified KEY<PROJECT<ORG nesting only the same kind is claimed to help.
        (LimitingUnit.REGION, LimitingUnit.REGION, True),
        (LimitingUnit.REGION, LimitingUnit.KEY, False),
        (LimitingUnit.REGION, LimitingUnit.ORGANIZATION, False),
        (ORG, LimitingUnit.ACCOUNT, False),
        (LimitingUnit.ACCOUNT, ORG, False),
    ],
)
def test_adding_capacity_helps_table(binding, added, helps) -> None:
    assert adding_capacity_helps(binding, added) is helps


# ============================================================== error-driven


async def test_rate_limited_with_retry_after_is_a_stated_cooldown(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), error=ClassifiedError(ErrorCategory.RATE_LIMITED, retry_after_seconds=7))
    clock.advance(seconds=3)
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.BLOCKED
    assert out.block_reason is BlockReason.COOLDOWN
    assert out.retry_at == NOW + timedelta(seconds=7)
    assert out.confidence is Confidence.HIGH
    clock.advance(seconds=5)
    assert (await engine.assess(a, req())).verdict is Verdict.UNKNOWN, "cooldown over, nothing else known"


async def test_rate_limited_without_retry_after_uses_default_and_says_so(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), error=ClassifiedError(ErrorCategory.RATE_LIMITED))
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.BLOCKED
    assert out.retry_at == NOW + timedelta(seconds=30)
    assert out.confidence is Confidence.MEDIUM
    assert any("default" in c for c in out.caveats)


async def test_retry_after_falls_back_to_the_snapshots_header(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(
        a, req(),
        snapshot=snap(requests=None, tokens=None, quality=SignalQuality.PARTIAL, retry_after=12.0),
        error=ClassifiedError(ErrorCategory.RATE_LIMITED),
    )
    out = await engine.assess(a, req())
    assert out.retry_at == NOW + timedelta(seconds=12)
    assert out.confidence is Confidence.HIGH


async def test_a_shorter_later_cooldown_never_shortens_an_existing_one(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), error=ClassifiedError(ErrorCategory.RATE_LIMITED, retry_after_seconds=60))
    engine.observe(a, req(), error=ClassifiedError(ErrorCategory.RATE_LIMITED, retry_after_seconds=5))
    assert (await engine.assess(a, req())).retry_at == NOW + timedelta(seconds=60)


@pytest.mark.parametrize(
    "category",
    [
        ErrorCategory.PROVIDER_OVERLOADED,
        ErrorCategory.TRANSIENT,
        ErrorCategory.PERMANENT,
        ErrorCategory.UNKNOWN,
        ErrorCategory.REQUIRES_HUMAN_ACTION,
    ],
)
async def test_errors_that_are_not_the_callers_quota_change_nothing(ledger, clock, category) -> None:
    """A 529 is the provider's capacity, not evidence about OUR headroom."""
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), snapshot=snap(reset=NOW + timedelta(seconds=60)))
    engine.observe(a, req(), error=ClassifiedError(category))
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.OPEN and out.confidence is Confidence.HIGH


async def test_a_success_closes_cooldown_and_exhaustion(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), error=ClassifiedError(ErrorCategory.RATE_LIMITED, retry_after_seconds=60))
    engine.observe(a, req(), error=ClassifiedError(ErrorCategory.QUOTA_EXHAUSTED))
    assert (await engine.assess(a, req())).verdict is Verdict.BLOCKED
    engine.observe(a, req())  # error=None: a call succeeded
    assert (await engine.assess(a, req())).verdict is Verdict.UNKNOWN


async def test_exhaustion_outranks_healthy_looking_headers(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), snapshot=snap(reset=NOW + timedelta(seconds=60)),
                   error=ClassifiedError(ErrorCategory.QUOTA_EXHAUSTED))
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.BLOCKED and out.block_reason is BlockReason.QUOTA_EXHAUSTED


async def test_exhaustion_is_per_bucket(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req("x"), error=ClassifiedError(ErrorCategory.QUOTA_EXHAUSTED))
    assert (await engine.assess(a, req("x"))).verdict is Verdict.BLOCKED
    assert (await engine.assess(a, req("y"))).verdict is Verdict.UNKNOWN


# ========================== exhaustion: resume time known, unknown, provisional


async def test_definitive_exhaustion_with_stated_resume_waits_then_probes(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    resume = NOW + timedelta(days=3)
    engine.observe(a, req(), error=ClassifiedError(ErrorCategory.QUOTA_EXHAUSTED, resume_at=resume))
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.BLOCKED
    assert out.block_reason is BlockReason.QUOTA_EXHAUSTED
    assert out.resume_at == resume and out.retry_at == resume
    assert out.confidence is Confidence.HIGH and out.provisional is False
    assert not any("no resume time" in c for c in out.caveats)
    clock.now = resume
    assert (await engine.assess(a, req())).verdict is Verdict.PROBE, "at resume we verify, we don't assume"


async def test_unknown_resume_is_a_first_class_state_with_a_backoff_schedule(ledger, clock) -> None:
    """OPEN question 4, answered."""
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), error=ClassifiedError(ErrorCategory.QUOTA_EXHAUSTED))
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.BLOCKED
    assert out.resume_at is None, "None means unknown, never 'never'"
    assert out.retry_at == NOW + timedelta(minutes=5)
    assert out.confidence is Confidence.HIGH, "we are sure it's exhausted; only WHEN is unknown"
    assert any("no resume time" in c for c in out.caveats)
    clock.advance(minutes=5)
    assert (await engine.assess(a, req())).verdict is Verdict.PROBE


async def test_claim_probe_hands_out_exactly_one_probe(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    assert engine.claim_probe(a, req()) is False, "nothing exhausted, nothing to probe"
    engine.observe(a, req(), error=ClassifiedError(ErrorCategory.QUOTA_EXHAUSTED))
    assert engine.claim_probe(a, req()) is False, "not due yet"
    clock.advance(minutes=5)
    assert (await engine.assess(a, req())).verdict is Verdict.PROBE
    assert engine.claim_probe(a, req()) is True
    assert engine.claim_probe(a, req()) is False, "a concurrent caller must not also probe"
    after = await engine.assess(a, req())
    assert after.verdict is Verdict.BLOCKED
    assert after.retry_at == clock.now + timedelta(minutes=10), "pushed out as if the probe will fail"


async def test_failed_probes_back_off_exponentially_to_the_cap(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), error=ClassifiedError(ErrorCategory.QUOTA_EXHAUSTED))
    seen = []
    for _ in range(9):
        out = await engine.assess(a, req())
        seen.append(int((out.retry_at - clock.now).total_seconds() // 60))
        clock.now = out.retry_at
        assert engine.claim_probe(a, req())
        engine.observe(a, req(), error=ClassifiedError(ErrorCategory.QUOTA_EXHAUSTED))  # probe failed
    assert seen == [5, 10, 20, 40, 80, 160, 320, 360, 360]  # capped at 6h


async def test_backoff_schedule_is_configurable(ledger, clock) -> None:
    a = FakeAdapter()
    cfg = QuotaConfig(probe=ProbePolicy(initial_delay=timedelta(seconds=10), factor=3.0, max_delay=timedelta(seconds=50)))
    engine = make_engine(ledger, clock, a, config=cfg)
    engine.observe(a, req(), error=ClassifiedError(ErrorCategory.QUOTA_EXHAUSTED))
    deltas = []
    for _ in range(3):
        out = await engine.assess(a, req())
        deltas.append((out.retry_at - clock.now).total_seconds())
        clock.now = out.retry_at
        engine.observe(a, req(), error=ClassifiedError(ErrorCategory.QUOTA_EXHAUSTED))
    assert deltas == [10, 30, 50]


async def test_successful_probe_closes_the_breaker(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), error=ClassifiedError(ErrorCategory.QUOTA_EXHAUSTED))
    clock.advance(minutes=5)
    assert engine.claim_probe(a, req())
    engine.observe(a, req())  # the probe succeeded
    assert (await engine.assess(a, req())).verdict is Verdict.UNKNOWN
    assert engine.claim_probe(a, req()) is False


async def test_provisional_exhaustion_blocks_but_is_low_confidence_and_probed_soon(ledger, clock) -> None:
    """OPEN question 3, answered: trust the category, distrust the duration."""
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), error=ClassifiedError(ErrorCategory.QUOTA_EXHAUSTED, is_definitively_classified=False))
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.BLOCKED, "trusted for routing: don't keep sending into it"
    assert out.provisional is True
    assert out.confidence is Confidence.LOW, "but not trusted as to duration"
    assert out.retry_at == NOW + timedelta(seconds=30), "short first probe, unlike a definitive 5 min"
    assert any("not-definitively-classified" in c for c in out.caveats)
    clock.advance(seconds=30)
    assert (await engine.assess(a, req())).verdict is Verdict.PROBE
    assert engine.claim_probe(a, req())
    engine.observe(a, req())  # it was only a per-minute limit after all
    assert (await engine.assess(a, req())).verdict is Verdict.UNKNOWN, "one success clears a false block"


async def test_provisional_with_a_far_resume_still_probes_early(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    resume = NOW + timedelta(hours=10)
    engine.observe(a, req(), error=ClassifiedError(
        ErrorCategory.QUOTA_EXHAUSTED, resume_at=resume, is_definitively_classified=False))
    out = await engine.assess(a, req())
    assert out.retry_at == NOW + timedelta(seconds=30)
    assert out.resume_at == resume, "the derived resume time is still reported"


async def test_provisional_never_probes_later_than_a_sooner_resume(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    resume = NOW + timedelta(seconds=10)
    engine.observe(a, req(), error=ClassifiedError(
        ErrorCategory.QUOTA_EXHAUSTED, resume_at=resume, is_definitively_classified=False))
    assert (await engine.assess(a, req())).retry_at == resume


async def test_repeat_failure_with_a_stale_resume_time_cannot_cause_a_probe_storm(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    stale = NOW - timedelta(hours=1)
    err = ClassifiedError(ErrorCategory.QUOTA_EXHAUSTED, resume_at=stale)
    engine.observe(a, req(), error=err)
    assert (await engine.assess(a, req())).verdict is Verdict.PROBE  # first time: resume has passed, verify
    engine.observe(a, req(), error=err)  # the probe failed, and the message still says "1h ago"
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.BLOCKED
    assert out.retry_at == NOW + timedelta(minutes=10), "fell back to backoff, not 'probe now' forever"


# ===================================================== ledger-derived path


def gemini_like(ledger, clock, **limits_kw):
    a = FakeAdapter(name="headerless", unit=PROJECT, windows=WINDOWS)
    limits = limits_kw.get("limits", {"*": {"rpm": 5, "rpd": 20}})
    return a, make_engine(ledger, clock, a, limits=limits)


async def test_no_limits_configured_is_unknown_and_lists_every_window_as_unmonitored(ledger, clock) -> None:
    a = FakeAdapter(name="headerless", unit=PROJECT, windows=WINDOWS)
    engine = make_engine(ledger, clock, a)
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.UNKNOWN and out.confidence is Confidence.NONE
    assert len(out.unmonitored) == 4
    assert all("no limit configured" in u for u in out.unmonitored)


async def test_ledger_counts_give_a_sane_low_confidence_open(ledger, clock) -> None:
    a, engine = gemini_like(ledger, clock)
    await ledger.append(ev(NOW - timedelta(seconds=30), provider="headerless"))
    await ledger.append(ev(NOW - timedelta(seconds=10), provider="headerless"))
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.OPEN
    assert out.confidence is Confidence.LOW, "ledger usage is a lower bound: 'open' is weak evidence"
    by_label = {c.label: c for c in out.constraints}
    assert by_label["rpm"].remaining == 3 and by_label["rpm"].limit == 5
    assert by_label["rpd"].remaining == 18
    assert all(c.source is Source.LEDGER for c in out.constraints)
    assert out.binding is not None and out.binding.label == "rpm"  # 60% is tighter than 90%
    assert out.binding.unit is PROJECT
    assert any("lower bound" in c for c in out.caveats)
    # tpm and spend_10m have no configured limit -> honestly unmonitored
    assert {u.split(":")[0] for u in out.unmonitored} == {"tpm", "spend_10m"}


async def test_ledger_exhaustion_is_a_stronger_claim_than_ledger_headroom(ledger, clock) -> None:
    a, engine = gemini_like(ledger, clock)
    for i in range(5):
        await ledger.append(ev(NOW - timedelta(seconds=i + 1), provider="headerless"))
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.BLOCKED
    assert out.block_reason is BlockReason.HEADROOM
    assert out.confidence is Confidence.MEDIUM, "we provably used >= the limit, whatever others used"
    assert out.binding is not None and out.binding.label == "rpm"
    assert out.retry_at == NOW + timedelta(seconds=60), "rolling window: a pessimistic upper bound"


async def test_events_outside_the_rolling_window_do_not_count(ledger, clock) -> None:
    a, engine = gemini_like(ledger, clock)
    for _ in range(3):
        await ledger.append(ev(NOW - timedelta(seconds=61), provider="headerless"))
    for i in range(4):
        await ledger.append(ev(NOW - timedelta(seconds=i + 1), provider="headerless"))
    out = await engine.assess(a, req())
    assert {c.label: c.remaining for c in out.constraints}["rpm"] == 1


async def test_ledger_counts_are_per_bucket_and_bucket_limits_override_the_wildcard(ledger, clock) -> None:
    a = FakeAdapter(name="headerless", unit=PROJECT, windows=WINDOWS)
    engine = make_engine(ledger, clock, a, limits={"*": {"rpm": 5}, "big": {"rpm": 2}})
    for _ in range(2):
        await ledger.append(ev(NOW - timedelta(seconds=5), provider="headerless", bucket="big"))
    await ledger.append(ev(NOW - timedelta(seconds=5), provider="headerless", bucket="small"))

    big = await engine.assess(a, req("big"))
    assert big.verdict is Verdict.BLOCKED, "2 used against big's own limit of 2"
    small = await engine.assess(a, req("small"))
    assert small.verdict is Verdict.OPEN
    assert {c.label: c.remaining for c in small.constraints}["rpm"] == 4, "wildcard 5, minus small's 1; big's 2 not counted"


async def test_ledger_counts_all_identities_because_the_quota_is_not_per_team(ledger, clock) -> None:
    a, engine = gemini_like(ledger, clock)
    for who in ("team-a", "team-b", "team-c"):
        await ledger.append(ev(NOW - timedelta(seconds=5), provider="headerless", identity=who))
    assert {c.label: c.remaining for c in (await engine.assess(a, req())).constraints}["rpm"] == 2


async def test_failed_calls_count_against_the_request_window_conservatively(ledger, clock) -> None:
    a, engine = gemini_like(ledger, clock)
    await ledger.append(ev(NOW - timedelta(seconds=5), provider="headerless", outcome=CallOutcome.FAILURE))
    assert {c.label: c.remaining for c in (await engine.assess(a, req())).constraints}["rpm"] == 4


async def test_calendar_day_window_resets_at_local_midnight_not_a_rolling_24h(ledger, clock) -> None:
    a = FakeAdapter(name="headerless", unit=PROJECT, windows=WINDOWS)
    engine = make_engine(ledger, clock, a, limits={"*": {"rpd": 1}})
    # 06:59:59Z is 23:59:59 the previous Pacific day: yesterday's quota.
    await ledger.append(ev(DAY_START - timedelta(seconds=1), provider="headerless"))
    assert (await engine.assess(a, req())).verdict is Verdict.OPEN
    # 07:00:00Z is exactly local midnight: today's first request, counted (>=).
    await ledger.append(ev(DAY_START, provider="headerless"))
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.BLOCKED
    assert out.retry_at == DAY_END, "exact next local midnight"


async def test_timestamp_boundaries_survive_the_ledgers_iso_string_comparison(ledger, clock) -> None:
    """The store filters with `timestamp >= ?` on ISO strings. isoformat()
    omits '.000000' when microseconds are zero, so mixed-precision strings
    must still order chronologically. Lock that in at the boundary."""
    a = FakeAdapter(name="headerless", unit=PROJECT, windows=WINDOWS)
    engine = make_engine(ledger, clock, a, limits={"*": {"rpm": 100}})
    start = NOW - timedelta(seconds=60)  # microsecond == 0 -> no fractional part in the string
    await ledger.append(ev(start - timedelta(microseconds=1), provider="headerless"))  # just outside
    await ledger.append(ev(start, provider="headerless"))  # exactly on the boundary: inside
    await ledger.append(ev(start + timedelta(microseconds=1), provider="headerless"))  # fractional: inside
    assert {c.label: c.remaining for c in (await engine.assess(a, req())).constraints}["rpm"] == 98


def test_window_bounds_are_correct_across_dst_days() -> None:
    w = QuotaWindow("rpd", QuotaDimension.REQUESTS, WindowKind.CALENDAR_DAY, tz="America/Los_Angeles")
    # Spring forward (2026-03-08): a 23-hour day. PST (UTC-8) at 00:00, PDT (UTC-7) at next 00:00.
    start, end = _window_bounds(w, datetime(2026, 3, 8, 12, 0, tzinfo=UTC))
    assert start == datetime(2026, 3, 8, 8, 0, tzinfo=UTC)
    assert end == datetime(2026, 3, 9, 7, 0, tzinfo=UTC)
    assert end - start == timedelta(hours=23)
    # Fall back (2026-11-01): a 25-hour day.
    start, end = _window_bounds(w, datetime(2026, 11, 1, 12, 0, tzinfo=UTC))
    assert start == datetime(2026, 11, 1, 7, 0, tzinfo=UTC)
    assert end == datetime(2026, 11, 2, 8, 0, tzinfo=UTC)
    assert end - start == timedelta(hours=25)
    # Always aware UTC: the ledger compares ISO strings, so a -07:00 offset would mis-sort.
    assert start.utcoffset() == timedelta(0) and end.utcoffset() == timedelta(0)


async def test_token_windows_are_reported_unmonitored_even_with_a_limit_configured(ledger, clock) -> None:
    a, engine = gemini_like(ledger, clock, limits={"*": {"rpm": 5, "tpm": 250_000}})
    out = await engine.assess(a, req())
    assert "tpm: the ledger records no token counts" in out.unmonitored
    assert out.confidence is Confidence.LOW


async def test_spend_is_unmonitored_when_calls_are_unpriced_not_reported_as_zero(ledger, clock) -> None:
    """With no price table entry every cost_usd is NULL, so SUM(cost_usd) is
    0.0. Reporting '$0 spent' would be a confident lie."""
    a = FakeAdapter(name="headerless", unit=PROJECT, windows=WINDOWS)
    engine = make_engine(ledger, clock, a, limits={"*": {"spend_10m": 1.0}})
    await ledger.append(ev(NOW - timedelta(seconds=30), provider="headerless", cost=None))
    out = await engine.assess(a, req())
    assert out.constraints == ()
    assert any("no known price" in u for u in out.unmonitored)
    assert out.verdict is Verdict.UNKNOWN


async def test_spend_with_no_calls_yet_is_monitored_and_truly_zero(ledger, clock) -> None:
    a = FakeAdapter(name="headerless", unit=PROJECT, windows=WINDOWS)
    engine = make_engine(ledger, clock, a, limits={"*": {"spend_10m": 1.0}})
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.OPEN
    assert out.constraints[0].remaining == 1.0


async def test_priced_spend_blocks_and_is_provider_wide_not_per_bucket(ledger, clock) -> None:
    a = FakeAdapter(name="headerless", unit=PROJECT, windows=WINDOWS)
    engine = make_engine(ledger, clock, a, limits={"*": {"spend_10m": 1.0}})
    await ledger.append(ev(NOW - timedelta(seconds=30), provider="headerless", bucket="m-a", cost=0.6))
    await ledger.append(ev(NOW - timedelta(seconds=20), provider="headerless", bucket="m-b", cost=0.5))
    out = await engine.assess(a, req("m-a"))
    assert out.verdict is Verdict.BLOCKED
    assert out.binding is not None and out.binding.dimension is QuotaDimension.SPEND
    assert out.retry_at == NOW + timedelta(seconds=600)


class BrokenLedger(LedgerStore):
    async def append(self, event): ...
    async def query(self, **kw): return []
    async def count_since(self, **kw): raise RuntimeError("database is locked")
    async def sum_cost_since(self, **kw): raise RuntimeError("database is locked")
    async def count_unpriced_since(self, **kw): raise RuntimeError("database is locked")
    async def reconcile(self, *args, **kw): raise RuntimeError("database is locked")


async def test_a_ledger_failure_degrades_to_unmonitored_instead_of_raising(clock) -> None:
    a = FakeAdapter(name="headerless", unit=PROJECT, windows=WINDOWS)
    engine = make_engine(BrokenLedger(), clock, a, limits={"*": {"rpm": 5, "spend_10m": 1.0}})
    out = await engine.assess(a, req())  # must not raise
    assert out.verdict is Verdict.UNKNOWN
    assert any("ledger unavailable (RuntimeError)" in u for u in out.unmonitored)


async def test_header_and_ledger_evidence_combine_and_the_tightest_binds(ledger, clock) -> None:
    """Both sources are equal citizens: the provider said 'plenty left', but
    our own ledger proves we are at the configured cap."""
    a = FakeAdapter(name="both", unit=PROJECT, windows=WINDOWS)
    engine = make_engine(ledger, clock, a, limits={"*": {"rpm": 2}})
    engine.observe(a, req(), snapshot=snap(PROJECT, reset=NOW + timedelta(seconds=60)))
    for _ in range(2):
        await ledger.append(ev(NOW - timedelta(seconds=5), provider="both"))
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.BLOCKED
    assert out.binding is not None and out.binding.source is Source.LEDGER
    assert {c.source for c in out.constraints} == {Source.HEADER, Source.LEDGER}


async def test_an_uncounted_window_caps_even_high_confidence_header_evidence(ledger, clock) -> None:
    """An OPEN answer needs EVERY declared limit to be known. Fresh, FULL
    header evidence alone would be HIGH, but this adapter also declares
    windows the engine cannot evaluate (here: none configured), so 'open'
    must be capped at LOW. Without the cap the only ledger-derived 'open'
    cases are already LOW and the rule would be dead code."""
    a = FakeAdapter(name="both", unit=PROJECT, windows=WINDOWS)
    engine = make_engine(ledger, clock, a)  # no limits configured -> all four windows unmonitored
    engine.observe(a, req(), snapshot=snap(PROJECT, reset=NOW + timedelta(seconds=60)))
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.OPEN
    assert len(out.unmonitored) == 4
    assert out.confidence is Confidence.LOW
    assert all(c.confidence is Confidence.HIGH for c in out.constraints), "the evidence itself is strong"


async def test_a_proven_block_is_not_weakened_by_windows_we_cannot_count(ledger, clock) -> None:
    """Blocking evidence is monotone: one proven exhaustion blocks no matter
    what else is unknown, so uncounted windows must NOT cap a BLOCKED."""
    a = FakeAdapter(name="both", unit=PROJECT, windows=WINDOWS)
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), snapshot=snap(PROJECT, requests=(0, 50), reset=NOW + timedelta(seconds=30)))
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.BLOCKED and out.unmonitored
    assert out.confidence is Confidence.HIGH


# ------------------------------------------- audit follow-ups (found after the first green run)


async def test_a_reset_time_already_past_on_arrival_is_not_trusted_as_replenishment(ledger, clock) -> None:
    """Found by audit. A provider-clock timestamp (reset_at_source 'timestamp')
    can already be in the past when it reaches us if their clock runs behind
    ours. A response saying remaining=0 with such a reset used to be read as
    'assumed replenished' -> OPEN: told zero, answered open. A reset that was
    past on arrival cannot be a future reset, so fall back to the default
    expiry instead."""
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), snapshot=snap(requests=(0, 50), reset=NOW - timedelta(seconds=5)))
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.BLOCKED
    assert out.retry_at == NOW + timedelta(seconds=60), "default no-reset expiry, not the stale stated one"
    assert any("already past" in c for c in out.caveats)
    clock.advance(seconds=61)
    assert (await engine.assess(a, req())).verdict is Verdict.OPEN, "and it still cannot deadlock"


async def test_a_reset_exactly_at_arrival_is_also_treated_as_untrustworthy(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), snapshot=snap(requests=(0, 50), reset=NOW))
    assert (await engine.assess(a, req())).verdict is Verdict.BLOCKED


# ---- branches the first coverage run showed were never exercised


async def test_a_constraint_with_no_known_limit_still_blocks_but_has_no_fraction(ledger, clock) -> None:
    """e.g. a partial snapshot carrying `remaining` only."""
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    only_remaining = RateLimitSnapshot(
        limiting_unit=ORG, signal_quality=SignalQuality.PARTIAL, requests_remaining=0,
        reset_at=NOW + timedelta(seconds=30),
    )
    engine.observe(a, req(), snapshot=only_remaining)
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.BLOCKED, "remaining < needed is provable without a limit"
    assert out.constraints[0].fraction_remaining is None


async def test_with_no_known_limits_open_has_no_binding_scope_and_no_scaling_advice(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    only_remaining = RateLimitSnapshot(
        limiting_unit=ORG, signal_quality=SignalQuality.PARTIAL, requests_remaining=7,
        reset_at=NOW + timedelta(seconds=30),
    )
    engine.observe(a, req(), snapshot=only_remaining)
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.OPEN  # cannot call it LOW without a limit to take a fraction of
    assert out.binding is None
    assert out.adding_helps(KEY) is None, "won't guess a scope it cannot identify"


async def test_a_limit_with_no_remaining_contributes_no_constraint(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    only_limit = RateLimitSnapshot(
        limiting_unit=ORG, signal_quality=SignalQuality.PARTIAL, requests_limit=50,
        reset_at=NOW + timedelta(seconds=30),
    )
    engine.observe(a, req(), snapshot=only_limit)
    assert (await engine.assess(a, req())).verdict is Verdict.UNKNOWN


async def test_a_none_quality_snapshot_that_nonetheless_has_numbers_is_low_confidence(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), snapshot=snap(quality=SignalQuality.NONE, reset=NOW + timedelta(seconds=30)))
    out = await engine.assess(a, req())
    assert out.verdict is Verdict.OPEN and out.confidence is Confidence.LOW


# ----------------------------------- a demand larger than a whole limit can never fit


async def test_a_demand_larger_than_the_whole_limit_blocks_with_no_retry_time(ledger, clock) -> None:
    """Found by fuzzing: this used to be BLOCKED/HEADROOM with retry_at=None,
    i.e. 'wait' with nothing to wait for. Waiting cannot make 500k tokens fit
    a 100k/min limit; the router needs to know to shrink or fail over."""
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), snapshot=snap(tokens=(90_000, 100_000), reset=NOW + timedelta(seconds=30)))
    out = await engine.assess(a, req(), demand=Demand(tokens=500_000))
    assert out.verdict is Verdict.BLOCKED
    assert out.block_reason is BlockReason.EXCEEDS_LIMIT
    assert out.retry_at is None, "nothing to wait for"
    assert out.confidence is Confidence.MEDIUM, "capped: that the provider rejects it is UNVERIFIED"
    assert any("waiting will not help" in c for c in out.caveats)
    assert out.binding is not None and out.binding.dimension is QuotaDimension.TOKENS


async def test_exceeds_limit_survives_the_observation_expiring(ledger, clock) -> None:
    """The exact path that produced the original bug: expired -> assumed
    replenished to the full limit -> demand still doesn't fit."""
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), snapshot=snap(tokens=(90_000, 100_000), reset=NOW + timedelta(seconds=30)))
    clock.advance(seconds=31)
    out = await engine.assess(a, req(), demand=Demand(tokens=500_000))
    assert out.block_reason is BlockReason.EXCEEDS_LIMIT and out.retry_at is None


async def test_a_demand_that_fits_is_unaffected(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), snapshot=snap(tokens=(90_000, 100_000), reset=NOW + timedelta(seconds=30)))
    assert (await engine.assess(a, req(), demand=Demand(tokens=50_000))).verdict is Verdict.OPEN


async def test_a_limit_of_zero_means_nothing_ever_fits(ledger, clock) -> None:
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), snapshot=snap(requests=(0, 0), tokens=None, reset=NOW + timedelta(seconds=30)))
    out = await engine.assess(a, req())
    assert out.block_reason is BlockReason.EXCEEDS_LIMIT and out.retry_at is None


async def test_exceeds_limit_outranks_a_timed_block(ledger, clock) -> None:
    """One dimension is merely exhausted (clears at its reset); another can
    never fit. Reporting the timed one would send the caller waiting."""
    a = FakeAdapter()
    engine = make_engine(ledger, clock, a)
    engine.observe(a, req(), snapshot=snap(requests=(0, 50), tokens=(90_000, 100_000),
                                           reset=NOW + timedelta(seconds=30)))
    out = await engine.assess(a, req(), demand=Demand(tokens=500_000))
    assert out.block_reason is BlockReason.EXCEEDS_LIMIT
    plain = await engine.assess(a, req())  # without the oversized demand: an ordinary timed block
    assert plain.block_reason is BlockReason.HEADROOM and plain.retry_at == NOW + timedelta(seconds=30)


async def test_ledger_path_also_detects_a_demand_larger_than_the_configured_limit(ledger, clock) -> None:
    a, engine = gemini_like(ledger, clock)  # rpm limit 5
    out = await engine.assess(a, req(), demand=Demand(requests=10))
    assert out.block_reason is BlockReason.EXCEEDS_LIMIT and out.retry_at is None
    assert out.binding is not None and out.binding.source is Source.LEDGER


# ------------------------------------------------ randomized state-machine invariants


def _invariant_violations(out, demand: Demand) -> list[str]:
    bad = []
    never_fits = out.block_reason is BlockReason.EXCEEDS_LIMIT
    if out.verdict in (Verdict.BLOCKED, Verdict.PROBE) and (out.retry_at is None) != never_fits:
        bad.append(f"retry_at is None iff EXCEEDS_LIMIT, got retry_at={out.retry_at} reason={out.block_reason}")
    if never_fits and out.verdict is not Verdict.BLOCKED:
        bad.append("EXCEEDS_LIMIT on a non-BLOCKED verdict")
    timed = out.verdict is Verdict.BLOCKED and out.block_reason in (BlockReason.COOLDOWN, BlockReason.QUOTA_EXHAUSTED)
    if timed and (out.retry_at is None or out.retry_at <= out.assessed_at):
        bad.append("timed non-headroom block must have a retry_at in the future")
    if out.verdict is Verdict.UNKNOWN and out.confidence is not Confidence.NONE:
        bad.append("UNKNOWN must have confidence NONE")
    if out.verdict in (Verdict.OPEN, Verdict.LOW) and out.confidence is Confidence.NONE:
        bad.append("OPEN/LOW must have a confidence")
    if out.verdict is Verdict.PROBE and out.block_reason is not BlockReason.QUOTA_EXHAUSTED:
        bad.append("PROBE only ever follows an exhaustion")
    if out.resume_at is not None and out.block_reason is not BlockReason.QUOTA_EXHAUSTED:
        bad.append("resume_at only ever accompanies an exhaustion")
    if out.verdict in (Verdict.OPEN, Verdict.LOW) and out.block_reason is not None:
        bad.append("an unblocked verdict has no block_reason")
    return bad


@pytest.mark.parametrize("seed", range(20))
async def test_random_sequences_never_violate_the_engines_stated_invariants(seed, tmp_path) -> None:
    """Seeded and deterministic. This is how the EXCEEDS_LIMIT gap and the
    stale-reset bug class were found: random observe/assess/claim/time-advance
    sequences, checked against what the docstrings promise."""
    import random

    rnd = random.Random(seed)
    ledger = SqliteLedgerStore(tmp_path / "fuzz.db")
    clock = FakeClock()
    a = FakeAdapter(name="fz", unit=PROJECT, windows=WINDOWS)
    engine = make_engine(ledger, clock, a, limits={"*": {"rpm": 4, "rpd": 9, "spend_10m": 1.0}})
    for step in range(100):
        m = rnd.choice(["m0", "m1"])
        op = rnd.choice(["snap", "err", "ok", "advance", "claim", "ledger"])
        if op == "snap":
            lim = rnd.choice([0, 1, 50, 1000])
            rem = min(rnd.choice([0, 0, 1, lim]), lim)
            reset = rnd.choice([None, clock.now - timedelta(seconds=3),
                                clock.now + timedelta(seconds=rnd.randint(1, 90))])
            extra = (snap(PROJECT, requests=None, tokens=(rnd.choice([0, 10]), 60), reset=reset),) if rnd.random() < 0.3 else ()
            engine.observe(a, req(m), snapshot=snap(
                rnd.choice([ORG, PROJECT]), requests=(rem, lim),
                tokens=None if rnd.random() < 0.5 else (rem, max(lim, 1)),
                reset=reset, extra=extra, quality=rnd.choice(list(SignalQuality))))
        elif op == "err":
            engine.observe(a, req(m), error=ClassifiedError(
                rnd.choice(list(ErrorCategory)),
                retry_after_seconds=rnd.choice([None, 0, 5, 300]),
                resume_at=rnd.choice([None, clock.now - timedelta(hours=1),
                                      clock.now + timedelta(hours=rnd.randint(1, 48))]),
                is_definitively_classified=rnd.random() < 0.5))
        elif op == "ok":
            engine.observe(a, req(m))
        elif op == "advance":
            clock.advance(seconds=rnd.choice([1, 20, 61, 400, 4000, 90_000]))
        elif op == "ledger":
            await ledger.append(ev(clock.now - timedelta(seconds=rnd.randint(0, 70)),
                                   provider="fz", bucket=m, cost=rnd.choice([None, 0.4])))
        elif op == "claim":
            first, second = engine.claim_probe(a, req(m)), engine.claim_probe(a, req(m))
            assert not (first and second), f"seed={seed} step={step}: two probes granted at one instant"
        demand = Demand(requests=rnd.choice([0, 1, 3]), tokens=rnd.choice([0, 0, 500, 10**7]))
        out = await engine.assess(a, req(m), demand=demand)  # must never raise
        problems = _invariant_violations(out, demand)
        assert not problems, f"seed={seed} step={step} demand={demand}: {problems}"
