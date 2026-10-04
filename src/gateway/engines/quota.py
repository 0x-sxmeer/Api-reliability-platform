"""
Quota Engine (Phase 3): proactive headroom tracking.

WHAT THE ROUTING ENGINE (PHASE 5) CALLS, AND WHAT IT GETS BACK
    engine.register(adapter, limits=...)            once, at startup
    a = await engine.assess(adapter, request)       BEFORE each call
    engine.claim_probe(adapter, request)            only if a.verdict is PROBE
    result = await executor.execute(request)
    engine.observe(adapter, request,                AFTER each call
                   snapshot=result.rate_limit,
                   error=result.classified_error)

    assess() returns a QuotaAssessment: a verdict (OPEN / LOW / PROBE /
    BLOCKED / UNKNOWN), a confidence (HIGH / MEDIUM / LOW / NONE), the
    binding constraint, when to come back (`retry_at`; None only for the
    EXCEEDS_LIMIT block, where the demand is bigger than a whole limit and
    waiting cannot help), when the provider says access returns
    (`resume_at`, None = genuinely unknown), and caveats in plain English. The engine never retries, sleeps, or picks a
    provider. It answers "how much room is there, and how sure are we";
    deciding what to do is Phase 5.

THE FOUR DECISIONS THIS MODULE ENCODES (each flagged in the Phase 3 report)

1. MULTI-SCOPE: model every scope a response describes, take the tightest.
   A call succeeds only if ALL simultaneously-enforced scopes have room, so
   headroom is the minimum across them and the binding scope is
   data-dependent (for OpenAI it can be org OR project, call to call).
   Knowing the binding scope is what lets the engine answer the question
   the project exists to answer -- "would adding more keys help?" -- see
   adding_capacity_helps(). State is also partitioned by the adapter's
   quota_bucket (per model), because headers describe one bucket only.

2. LEDGER FALLBACK: when an adapter declares quota_windows() (Gemini) the
   engine counts from the ledger against operator-configured limits. Ledger
   usage is a LOWER BOUND on true usage (it sees only this gateway's
   traffic; Gemini limits are per project and shared). So evidence is
   asymmetric, and the confidence rules encode it: "blocked" from the
   ledger is decent evidence (we provably used at least that much);
   "open" from the ledger is weak evidence (somebody else may have used the
   rest), and any window we cannot count (Gemini TPM: the ledger stores no
   tokens; Gemini spend: no prices exist, so SUM(cost) would read $0) is
   reported as UNMONITORED and caps an "open" at LOW confidence. The
   engine never reports "$0 spent" when the truth is "unpriced".

3. NOT-DEFINITIVELY-CLASSIFIED QUOTA_EXHAUSTED: trust the CATEGORY, distrust
   the DURATION, verify by probing. It blocks the scope (failover exists
   precisely for this state, so a false block is cheap), but is marked
   `provisional`, capped at LOW confidence, and probed on the short end of
   the schedule; one successful call clears it. Treating it as a plain
   RATE_LIMITED would re-create the documented failure of retrying a daily
   quota as if it were per-minute (see docs/provider-notes.md).

4. RESUME TIME UNKNOWN is a first-class state, not a special case. With no
   resume_at the engine runs a circuit-breaker: BLOCKED, then PROBE after an
   exponentially growing delay (claim_probe() hands out ONE probe at a time
   so concurrent callers don't all probe), and any success closes it.

WHAT THIS ENGINE DELIBERATELY DOES NOT DO (and why)
  * No reservation of in-flight demand: N concurrent callers can all see the
    same headroom. Fixing that needs a decision about how Phase 5 schedules
    calls; nothing in Phase 3 consumes it, so it isn't built.
  * No per-dimension reset times: a RateLimitSnapshot carries one reset_at
    for two dimensions. retry_at is therefore an estimate, and an expired
    observation is treated as stale, never as proof of replenishment of a
    dimension we can't attribute the reset to.
  * State is in memory. After a restart the engine re-learns an exhaustion
    from one failed call per bucket (the ledger keeps only the
    classification MESSAGE, not resume_at).
  * One credential per provider_name per engine: the ledger has no
    credential dimension, so two orgs behind one name would be merged.

ARCHITECTURE RULE: this module calls only ProviderAdapter methods and
LedgerStore methods. It names no vendor and branches on none.

Concurrency: observe() and claim_probe() are synchronous and never await,
so on a single asyncio event loop they are atomic with respect to each
other and to the synchronous parts of assess(). Not thread-safe.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, time, timedelta
from enum import Enum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from gateway.core.adapter import AdapterRequest, ProviderAdapter
from gateway.core.types import (
    ClassifiedError,
    ErrorCategory,
    LimitingUnit,
    QuotaDimension,
    QuotaWindow,
    RateLimitSnapshot,
    SignalQuality,
    WindowKind,
    utcnow,
)
from gateway.ledger.store import LedgerStore

logger = logging.getLogger(__name__)

WILDCARD_BUCKET = "*"


# ----------------------------------------------------------------- vocabulary


class Verdict(str, Enum):
    OPEN = "open"  # room for the demand; send
    LOW = "low"  # room, but under the caution threshold; prefer alternatives
    PROBE = "probe"  # exhausted, but one trial call is due (call claim_probe())
    BLOCKED = "blocked"  # do not send before retry_at (None = never fits: see BlockReason.EXCEEDS_LIMIT)
    UNKNOWN = "unknown"  # no information at all; policy is the caller's


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


_CONF_RANK = {Confidence.NONE: 0, Confidence.LOW: 1, Confidence.MEDIUM: 2, Confidence.HIGH: 3}


def _min_conf(confs: list[Confidence]) -> Confidence:
    return min(confs, key=_CONF_RANK.__getitem__) if confs else Confidence.NONE


def _max_conf(confs: list[Confidence]) -> Confidence:
    return max(confs, key=_CONF_RANK.__getitem__) if confs else Confidence.NONE


class BlockReason(str, Enum):
    HEADROOM = "headroom"  # a measured/counted limit has no room left
    COOLDOWN = "cooldown"  # recent RATE_LIMITED; wait it out
    QUOTA_EXHAUSTED = "quota_exhausted"  # spend cap / credits gone; fail over
    EXCEEDS_LIMIT = "exceeds_limit"
    # The demand is larger than an ENTIRE limit (e.g. 500k tokens against a
    # 100k/min limit, or a limit of 0). Waiting cannot help, so retry_at is
    # None by design: shrink the request or fail over.


class Source(str, Enum):
    HEADER = "header"
    LEDGER = "ledger"


@dataclass(frozen=True)
class QuotaKey:
    """State is partitioned by provider AND the adapter's quota bucket."""

    provider: str
    bucket: str


@dataclass(frozen=True)
class Demand:
    """What the caller is about to spend. tokens=0 means 'not estimated'."""

    requests: int = 1
    tokens: int = 0


@dataclass(frozen=True)
class ProbePolicy:
    """
    Circuit-breaker schedule for an exhausted scope. These are ENGINEERING
    DEFAULTS, not provider facts: no provider documents how quickly credit
    returns after a top-up, so the values trade probe cost (one cheap failed
    call) against recovery latency.
    """

    initial_delay: timedelta = timedelta(minutes=5)
    provisional_initial_delay: timedelta = timedelta(seconds=30)
    factor: float = 2.0
    max_delay: timedelta = timedelta(hours=6)


@dataclass(frozen=True)
class QuotaConfig:
    """Policy knobs. All are defaults to tune, none is a provider fact."""

    fresh_for: timedelta = timedelta(seconds=30)
    # A header observation younger than this is "fresh" (HIGH confidence).
    stale_after_no_reset: timedelta = timedelta(seconds=60)
    # An observation with no reset_at expires after this. Without it a
    # remaining=0 reading could block a scope forever: a blocked scope
    # receives no calls, so nothing would ever refresh the observation.
    low_headroom_fraction: float = 0.10
    default_cooldown: timedelta = timedelta(seconds=30)
    # Cooldown for a RATE_LIMITED error that states no retry-after.
    probe: ProbePolicy = field(default_factory=ProbePolicy)


@dataclass(frozen=True)
class ConstraintHeadroom:
    """One limit, as currently understood. The tightest one binds."""

    unit: LimitingUnit
    dimension: QuotaDimension
    label: str  # "requests"/"tokens" for headers; the window name for ledger
    source: Source
    remaining: float | None
    limit: float | None
    blocked: bool
    confidence: Confidence
    observed_at: datetime
    reset_at: datetime | None = None  # when this constraint is expected to clear
    assumed_replenished: bool = False  # observation expired; we assumed a reset
    exceeds_limit: bool = False  # the demand alone is bigger than the whole limit
    note: str = ""

    @property
    def fraction_remaining(self) -> float | None:
        if self.remaining is None or self.limit is None or self.limit <= 0:
            return None
        return max(0.0, self.remaining / self.limit)


# Nesting used ONLY for scaling advice. Keys live inside projects live inside
# organizations. True for every current provider; deliberately NOT extended
# to ACCOUNT/REGION/RESOURCE, whose nesting differs by vendor.
_NESTING = (LimitingUnit.KEY, LimitingUnit.PROJECT, LimitingUnit.ORGANIZATION)


def adding_capacity_helps(binding: LimitingUnit, added: LimitingUnit) -> bool:
    """
    Would creating more entities of kind `added` raise capacity, given the
    scope that is binding? This is the "naive key rotation" question the
    project was started to answer.

    New keys share their project's limit, new projects share their org's
    limit, so only entities AT OR ABOVE the binding scope add capacity.
    For scopes outside the KEY<PROJECT<ORGANIZATION nesting, only the same
    kind of entity is claimed to help (no cross-vendor nesting is assumed).
    """
    if added == binding:
        return True
    if binding in _NESTING and added in _NESTING:
        return _NESTING.index(added) > _NESTING.index(binding)
    return False


@dataclass(frozen=True)
class QuotaAssessment:
    provider: str
    bucket: str
    verdict: Verdict
    confidence: Confidence
    assessed_at: datetime
    block_reason: BlockReason | None = None
    retry_at: datetime | None = None
    # Earliest time a call (or probe) is worth attempting. Set whenever the
    # verdict is BLOCKED or PROBE, EXCEPT block_reason EXCEEDS_LIMIT, where it
    # is None because no amount of waiting makes the demand fit.
    resume_at: datetime | None = None
    # When the provider (or its adapter, from a documented rule) says access
    # returns. None means UNKNOWN, never "never" -- the same contract as
    # ClassifiedError.resume_at. Set only for QUOTA_EXHAUSTED.
    provisional: bool = False
    # True when a QUOTA_EXHAUSTED block rests on a not-definitively-
    # classified error: the category is trusted for routing, the duration is
    # not. Prefer "wait briefly and retry here" over "mark provider dead".
    binding: ConstraintHeadroom | None = None
    constraints: tuple[ConstraintHeadroom, ...] = ()
    unmonitored: tuple[str, ...] = ()
    # Declared limit windows the engine could NOT evaluate, each with why.
    caveats: tuple[str, ...] = ()

    def adding_helps(self, added: LimitingUnit) -> bool | None:
        """Would more `added` entities raise capacity? None if no binding
        scope is known (so the engine won't guess)."""
        if self.binding is None:
            return None
        return adding_capacity_helps(self.binding.unit, added)


# ------------------------------------------------------------ internal state


@dataclass
class _Observation:
    snapshot: RateLimitSnapshot
    at: datetime


@dataclass
class _Exhaustion:
    since: datetime
    count: int
    provisional: bool
    resume_at: datetime | None
    next_probe_at: datetime


@dataclass
class _KeyState:
    observations: dict[LimitingUnit, _Observation] = field(default_factory=dict)
    cooldown_until: datetime | None = None
    cooldown_stated: bool = False
    exhaustion: _Exhaustion | None = None


@dataclass
class _Registration:
    adapter: ProviderAdapter
    windows: dict[str, QuotaWindow]
    limits: dict[str, dict[str, float]]  # bucket-or-"*" -> window name -> limit


def _utc(dt: datetime, what: str) -> datetime:
    if dt.tzinfo is None:
        raise ValueError(f"{what} must be timezone-aware (got a naive datetime)")
    return dt.astimezone(UTC)


def _window_bounds(window: QuotaWindow, now: datetime) -> tuple[datetime, datetime]:
    """
    (start, end) of the window containing `now`, as aware UTC.

    `end` is when capacity is expected back. For CALENDAR_DAY it is exact
    (next local midnight, correct across 23/25-hour DST days). For ROLLING
    it is now + length: a pessimistic UPPER BOUND, because the ledger count
    doesn't say when the oldest counted event ages out.
    KNOWN LIMIT: a zone whose midnight doesn't exist on a DST-change day
    resolves to the pre-change offset (off by up to an hour). Not an issue
    for any zone a provider currently documents.
    """
    if window.kind is WindowKind.ROLLING:
        length = timedelta(seconds=window.seconds or 0)
        return now - length, now + length
    zone = ZoneInfo(window.tz or "UTC")
    local = now.astimezone(zone)
    start = datetime.combine(local.date(), time.min, tzinfo=zone)
    end = datetime.combine(local.date() + timedelta(days=1), time.min, tzinfo=zone)
    return start.astimezone(UTC), end.astimezone(UTC)


# -------------------------------------------------------------------- engine


class QuotaEngine:
    def __init__(
        self,
        ledger: LedgerStore,
        *,
        config: QuotaConfig | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._ledger = ledger
        self._config = config or QuotaConfig()
        self._clock = clock
        self._registry: dict[str, _Registration] = {}
        self._states: dict[QuotaKey, _KeyState] = {}

    # ------------------------------------------------------------ registration

    def register(
        self,
        adapter: ProviderAdapter,
        *,
        limits: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        """
        Make an adapter known to the engine.

        `limits` is only needed for ledger-derived counting (providers that
        declare quota_windows()): {bucket_or_"*": {window_name: limit}}. A
        bucket's entry overrides "*" per window name. Numbers are operator
        configuration because no provider publishes them through an API.

        Fails fast on a typo'd window name, a non-positive limit, or an
        unusable timezone, rather than silently never counting.
        """
        name = adapter.provider_name
        existing = self._registry.get(name)
        if existing is not None and existing.adapter is not adapter:
            raise ValueError(
                f"A different adapter is already registered as {name!r}. One credential per "
                "provider_name per engine: the ledger has no credential dimension, so two "
                "credentials behind one name would be counted together."
            )
        windows = {w.name: w for w in adapter.quota_windows()}
        if len(windows) != len(adapter.quota_windows()):
            raise ValueError(f"{name!r} declares duplicate quota window names")
        clean: dict[str, dict[str, float]] = {}
        for bucket, per_window in (limits or {}).items():
            clean[bucket] = {}
            for wname, value in per_window.items():
                if wname not in windows:
                    valid = sorted(windows) or "none (this adapter declares no countable windows)"
                    raise ValueError(f"{name!r} has no quota window {wname!r}; valid names: {valid}")
                if not value > 0:
                    raise ValueError(f"limit for {name!r}/{bucket!r}/{wname!r} must be > 0, got {value!r}")
                clean[bucket][wname] = float(value)
        for w in windows.values():
            if w.kind is WindowKind.CALENDAR_DAY:
                try:
                    ZoneInfo(w.tz or "")
                except (ZoneInfoNotFoundError, ValueError) as exc:
                    raise ValueError(
                        f"window {w.name!r} needs timezone {w.tz!r}, which is unavailable "
                        "(install the 'tzdata' package)"
                    ) from exc
        self._registry[name] = _Registration(adapter=adapter, windows=windows, limits=clean)

    def _key(self, adapter: ProviderAdapter, request: AdapterRequest) -> QuotaKey:
        reg = self._registry.get(adapter.provider_name)
        if reg is None or reg.adapter is not adapter:
            raise ValueError(f"adapter {adapter.provider_name!r} is not registered with this QuotaEngine")
        return QuotaKey(adapter.provider_name, adapter.quota_bucket(request))

    # ----------------------------------------------------------------- observe

    def observe(
        self,
        adapter: ProviderAdapter,
        request: AdapterRequest,
        *,
        snapshot: RateLimitSnapshot | None = None,
        error: ClassifiedError | None = None,
        at: datetime | None = None,
    ) -> None:
        """
        Feed the outcome of ONE call. Pass `error=result.classified_error`
        whenever the call failed: `error=None` means "this call succeeded",
        which closes any cooldown or exhaustion for the bucket.

        Only QUOTA_EXHAUSTED and RATE_LIMITED errors change quota state.
        PROVIDER_OVERLOADED, TRANSIENT, PERMANENT, UNKNOWN and
        REQUIRES_HUMAN_ACTION are not the caller's quota and are routing's
        business, so they are ignored here (a 529 is not evidence of anything
        about our headroom).
        """
        key = self._key(adapter, request)
        now = _utc(at, "at") if at is not None else _utc(self._clock(), "clock()")
        state = self._states.setdefault(key, _KeyState())

        if snapshot is not None:
            for view in (snapshot, *snapshot.additional_scopes):
                # A response with no numbers (a 529, a transport-adjacent
                # error) must NOT overwrite a useful earlier observation.
                if any(
                    v is not None
                    for v in (view.requests_remaining, view.requests_limit, view.tokens_remaining, view.tokens_limit)
                ):
                    state.observations[view.limiting_unit] = _Observation(view, now)

        if error is None:
            state.cooldown_until = None
            state.cooldown_stated = False
            state.exhaustion = None
        elif error.category is ErrorCategory.RATE_LIMITED:
            secs = error.retry_after_seconds
            if secs is None and snapshot is not None:
                secs = snapshot.retry_after_seconds
            stated = secs is not None
            until = now + (timedelta(seconds=secs) if secs is not None else self._config.default_cooldown)
            if state.cooldown_until is None or until > state.cooldown_until:
                state.cooldown_until = until
                state.cooldown_stated = stated
        elif error.category is ErrorCategory.QUOTA_EXHAUSTED:
            self._record_exhaustion(state, error, now)

    def _probe_delay(self, count: int, provisional: bool) -> timedelta:
        p = self._config.probe
        base = p.provisional_initial_delay if provisional else p.initial_delay
        return min(p.max_delay, base * (p.factor ** max(0, count - 1)))

    def _record_exhaustion(self, state: _KeyState, error: ClassifiedError, now: datetime) -> None:
        prev = state.exhaustion
        count = prev.count + 1 if prev else 1
        provisional = not error.is_definitively_classified
        delay = self._probe_delay(count, provisional)
        resume_at = _utc(error.resume_at, "resume_at") if error.resume_at is not None else None
        if resume_at is None:
            next_probe = now + delay
        elif provisional:
            # The diagnosis is a guess, so don't sit out the whole stated
            # duration before checking -- but never probe LATER than it.
            next_probe = min(resume_at, now + delay)
        else:
            next_probe = resume_at
        if count > 1 and next_probe <= now:
            # A repeat failure with a stale/past resume time must not cause
            # a probe storm; fall back to the backoff schedule.
            next_probe = now + delay
        state.exhaustion = _Exhaustion(
            since=prev.since if prev else now,
            count=count,
            provisional=provisional,
            resume_at=resume_at,
            next_probe_at=next_probe,
        )

    def claim_probe(self, adapter: ProviderAdapter, request: AdapterRequest) -> bool:
        """
        Atomically claim the right to send ONE probe call to an exhausted
        bucket. True only if a probe is due (assess() said PROBE) and nobody
        else has claimed it; the next probe is pushed out as if this one
        will fail, so concurrent callers see BLOCKED and a probe that is
        claimed but never reported doesn't cause a hammering loop. Report
        the probe's outcome via observe() as usual.
        """
        key = self._key(adapter, request)
        state = self._states.get(key)
        now = _utc(self._clock(), "clock()")
        if state is None or state.exhaustion is None or now < state.exhaustion.next_probe_at:
            return False
        ex = state.exhaustion
        ex.next_probe_at = now + self._probe_delay(ex.count + 1, ex.provisional)
        return True

    # ------------------------------------------------------------- snapshot

    def snapshot(self) -> list[dict]:
        """Read-only view of live quota state, for observability (Phase 9).

        Returns one dict per (provider, bucket) that has ever been observed,
        with the CURRENT verdict/confidence and the tightest known constraint.
        Pure data: no provider names or units are interpreted here — values
        come from each adapter's own observations. JSON-safe (datetimes as
        ISO strings) so the dashboard can render it without a second model.
        """
        now = _utc(self._clock(), "clock()")
        out: list[dict] = []
        for key, state in sorted(self._states.items(), key=lambda kv: (kv[0].provider, kv[0].bucket)):
            exhaustion = state.exhaustion
            cooldown_active = state.cooldown_until is not None and now < state.cooldown_until
            if exhaustion is not None and now < exhaustion.blocked_until:
                verdict = "blocked"
                reason = "quota_exhausted"
                retry_at = exhaustion.next_probe_at
            elif cooldown_active:
                verdict = "blocked"
                reason = "cooldown"
                retry_at = state.cooldown_until
            else:
                reason = None
                retry_at = None
                # Best-effort verdict from freshest header observations only;
                # full assessment (incl. ledger counting) needs a request,
                # which this read-only view deliberately does not fabricate.
                fresh = [
                    o for o in state.observations.values() if now - o.at <= self._config.fresh_for
                ]
                if not fresh:
                    verdict = "unknown"
                else:
                    fracs = []
                    for o in fresh:
                        s = o.snapshot
                        for rem, lim in (
                            (s.requests_remaining, s.requests_limit),
                            (s.tokens_remaining, s.tokens_limit),
                        ):
                            if rem is not None and lim:
                                fracs.append(max(0.0, rem / lim))
                    if not fracs:
                        verdict = "unknown"
                    elif min(fracs) < self._config.low_headroom_fraction:
                        verdict = "low"
                    else:
                        verdict = "open"
            binding = None
            obs_units = sorted(state.observations, key=lambda u: u.value)
            if obs_units:
                latest = max((state.observations[u] for u in obs_units), key=lambda o: o.at)
                s = latest.snapshot
                binding = {
                    "unit": s.limiting_unit.value,
                    "requests_remaining": s.requests_remaining,
                    "requests_limit": s.requests_limit,
                    "tokens_remaining": s.tokens_remaining,
                    "tokens_limit": s.tokens_limit,
                    "reset_at": s.reset_at.isoformat() if s.reset_at else None,
                    "observed_at": latest.at.isoformat(),
                }
            out.append(
                {
                    "provider": key.provider,
                    "bucket": key.bucket,
                    "verdict": verdict,
                    "block_reason": reason,
                    "retry_at": retry_at.isoformat() if retry_at else None,
                    "binding": binding,
                    "confidence": "high" if binding else "none",
                }
            )
        return out

    # ------------------------------------------------------------------ assess

    async def assess(
        self,
        adapter: ProviderAdapter,
        request: AdapterRequest,
        *,
        demand: Demand | None = None,
    ) -> QuotaAssessment:
        demand = demand or Demand()
        key = self._key(adapter, request)
        reg = self._registry[key.provider]
        now = _utc(self._clock(), "clock()")
        state = self._states.get(key)

        # 1. Error-driven states outrank measurements: the headers on a
        #    failing response are usually zeros and say nothing about WHEN.
        if state is not None:
            if state.exhaustion is not None:
                return self._exhausted_assessment(key, state.exhaustion, now)
            if state.cooldown_until is not None:
                if now < state.cooldown_until:
                    return QuotaAssessment(
                        provider=key.provider,
                        bucket=key.bucket,
                        verdict=Verdict.BLOCKED,
                        confidence=Confidence.HIGH if state.cooldown_stated else Confidence.MEDIUM,
                        assessed_at=now,
                        block_reason=BlockReason.COOLDOWN,
                        retry_at=state.cooldown_until,
                        caveats=(
                            ()
                            if state.cooldown_stated
                            else ("cooldown length is the engine's default; the provider stated none",)
                        ),
                    )
                state.cooldown_until = None  # expired; lazily cleared

        # 2. Measurements: header-derived and ledger-derived, equal citizens.
        caveats: list[str] = []
        constraints = self._header_constraints(state, now, demand, caveats)
        ledger_constraints, unmonitored = await self._ledger_constraints(reg, key, now, demand, caveats)
        constraints += ledger_constraints

        return self._compose(key, now, constraints, unmonitored, caveats)

    def _exhausted_assessment(self, key: QuotaKey, ex: _Exhaustion, now: datetime) -> QuotaAssessment:
        caveats: list[str] = []
        if ex.resume_at is None:
            caveats.append(
                "provider states no resume time; probing on a backoff schedule "
                f"(attempt {ex.count}) instead of waiting for one"
            )
        if ex.provisional:
            caveats.append(
                "exhaustion inferred from a not-definitively-classified error: blocked for "
                "safety, duration unknown; one successful call clears it"
            )
        return QuotaAssessment(
            provider=key.provider,
            bucket=key.bucket,
            verdict=Verdict.PROBE if now >= ex.next_probe_at else Verdict.BLOCKED,
            confidence=Confidence.LOW if ex.provisional else Confidence.HIGH,
            assessed_at=now,
            block_reason=BlockReason.QUOTA_EXHAUSTED,
            retry_at=ex.next_probe_at,
            resume_at=ex.resume_at,
            provisional=ex.provisional,
            caveats=tuple(caveats),
        )

    # -------------------------------------------------- header-derived evidence

    def _header_constraints(
        self, state: _KeyState | None, now: datetime, demand: Demand, caveats: list[str]
    ) -> list[ConstraintHeadroom]:
        out: list[ConstraintHeadroom] = []
        if state is None:
            return out
        cfg = self._config
        for unit, obs in state.observations.items():
            view, age = obs.snapshot, now - obs.at
            # A stated reset that was ALREADY past when we received it cannot
            # be a future reset: the provider's clock runs behind ours (the
            # skew risk RateLimitSnapshot.reset_at_source exists to flag).
            # Trusting it would read "remaining=0" as instantly replenished,
            # so fall back to the no-reset expiry instead.
            stated_reset = view.reset_at if view.reset_at is not None and view.reset_at > obs.at else None
            if view.reset_at is not None and stated_reset is None:
                caveats.append(
                    f"{unit.value}: stated reset time was already past when received "
                    "(provider clock skew?); using the default expiry instead"
                )
            expiry = stated_reset or (obs.at + cfg.stale_after_no_reset)
            dims = (
                (QuotaDimension.REQUESTS, view.requests_remaining, view.requests_limit, max(demand.requests, 1)),
                (QuotaDimension.TOKENS, view.tokens_remaining, view.tokens_limit, max(demand.tokens, 1)),
            )
            for dim, rem, lim, needed in dims:
                if rem is None and lim is None:
                    continue
                if now >= expiry:
                    # Stale. The only safe reading of "the window has
                    # passed" is "probably replenished" (true for both
                    # token-bucket and fixed-window providers) -- and if it
                    # was wrong, the next call fails and refreshes us.
                    if lim is None:
                        caveats.append(f"{unit.value} {dim.value}: observation expired and no limit known; ignored")
                        continue
                    out.append(
                        ConstraintHeadroom(
                            unit=unit, dimension=dim, label=dim.value, source=Source.HEADER,
                            remaining=float(lim), limit=float(lim), blocked=lim < needed,
                            confidence=Confidence.MEDIUM, observed_at=obs.at, assumed_replenished=True,
                            exceeds_limit=needed > lim,
                            note="observation expired; assumed replenished",
                        )
                    )
                    continue
                if rem is None:
                    continue
                out.append(
                    ConstraintHeadroom(
                        unit=unit, dimension=dim, label=dim.value, source=Source.HEADER,
                        remaining=float(rem), limit=float(lim) if lim is not None else None,
                        blocked=rem < needed, confidence=self._header_confidence(view, age),
                        observed_at=obs.at, reset_at=expiry,
                        exceeds_limit=lim is not None and needed > lim,
                    )
                )
        return out

    def _header_confidence(self, view: RateLimitSnapshot, age: timedelta) -> Confidence:
        fresh = age <= self._config.fresh_for
        if view.signal_quality is SignalQuality.FULL:
            return Confidence.HIGH if fresh else Confidence.MEDIUM
        if view.signal_quality is SignalQuality.PARTIAL:
            return Confidence.MEDIUM if fresh else Confidence.LOW
        return Confidence.LOW

    # -------------------------------------------------- ledger-derived evidence

    async def _ledger_constraints(
        self, reg: _Registration, key: QuotaKey, now: datetime, demand: Demand, caveats: list[str]
    ) -> tuple[list[ConstraintHeadroom], list[str]]:
        out: list[ConstraintHeadroom] = []
        unmonitored: list[str] = []
        unit = reg.adapter.get_limiting_unit()
        for name, window in reg.windows.items():
            limit = reg.limits.get(key.bucket, {}).get(name, reg.limits.get(WILDCARD_BUCKET, {}).get(name))
            if limit is None:
                unmonitored.append(f"{name}: no limit configured")
                continue
            if window.dimension is QuotaDimension.TOKENS:
                unmonitored.append(f"{name}: the ledger records no token counts")
                continue
            start, end = _window_bounds(window, now)
            try:
                if window.dimension is QuotaDimension.REQUESTS:
                    used = float(
                        await self._ledger.count_since(provider=key.provider, since=start, quota_bucket=key.bucket)
                    )
                    needed = float(max(demand.requests, 1))
                    blocked = limit - used < needed
                    exceeds = needed > limit
                else:  # SPEND -- provider-wide, not per bucket
                    unpriced = await self._ledger.count_unpriced_since(provider=key.provider, since=start)
                    if unpriced > 0:
                        unmonitored.append(
                            f"{name}: {unpriced} successful call(s) in the window have no known price, "
                            "so spend would be under-counted"
                        )
                        continue
                    used = await self._ledger.sum_cost_since(provider=key.provider, since=start)
                    blocked = limit - used <= 0
                    exceeds = False  # the spend a call will incur is unknown beforehand
            except Exception as exc:  # noqa: BLE001 -- an advisory engine must not take calls down with a ledger hiccup
                logger.warning("quota: ledger read failed for %s/%s: %r", key.provider, name, exc)
                unmonitored.append(f"{name}: ledger unavailable ({type(exc).__name__})")
                continue
            out.append(
                ConstraintHeadroom(
                    unit=unit, dimension=window.dimension, label=name, source=Source.LEDGER,
                    remaining=limit - used, limit=limit, blocked=blocked,
                    # Usage is a lower bound (see module docstring): a proven
                    # exhaustion is decent evidence, an apparent surplus is not.
                    confidence=Confidence.MEDIUM if blocked else Confidence.LOW,
                    observed_at=now, reset_at=end if blocked else None,
                    exceeds_limit=exceeds,
                    note="counted from this gateway's own ledger only",
                )
            )
        if out:
            caveats.append("ledger-derived figures count only this gateway's traffic (a lower bound on true usage)")
        return out, unmonitored

    # ---------------------------------------------------------------- verdict

    def _compose(
        self,
        key: QuotaKey,
        now: datetime,
        constraints: list[ConstraintHeadroom],
        unmonitored: list[str],
        caveats: list[str],
    ) -> QuotaAssessment:
        # Built with dataclasses.replace rather than **a-dict so every field
        # name stays type-checked (a typo in a **dict is silent).
        base = QuotaAssessment(
            provider=key.provider,
            bucket=key.bucket,
            verdict=Verdict.UNKNOWN,
            confidence=Confidence.NONE,
            assessed_at=now,
            constraints=tuple(constraints),
            unmonitored=tuple(unmonitored),
            caveats=tuple(caveats),
        )
        blocked = [c for c in constraints if c.blocked]
        impossible = [c for c in blocked if c.exceeds_limit]
        if impossible:
            # Waiting cannot fix this, so it outranks any timed block. Capped
            # at MEDIUM: that a provider rejects a request larger than the
            # whole limit (rather than, say, admitting it once the bucket is
            # full) is UNVERIFIED, and the token figure is the caller's estimate.
            return replace(
                base,
                verdict=Verdict.BLOCKED,
                confidence=_min_conf([Confidence.MEDIUM, _max_conf([c.confidence for c in impossible])]),
                block_reason=BlockReason.EXCEEDS_LIMIT,
                retry_at=None,
                binding=impossible[0],
                caveats=(
                    *base.caveats,
                    (
                        "the demand is larger than an entire limit, so waiting will not help: "
                        "shrink the request or fail over (assumes the provider rejects such a request: UNVERIFIED)"
                    ),
                ),
            )
        if blocked:
            # Any ONE proven-exhausted constraint blocks, whatever we don't
            # know about the rest -- so confidence is the best proof we hold.
            # Capacity returns only when ALL blocked constraints clear, so
            # the binding one is the one that clears last.
            return replace(
                base,
                verdict=Verdict.BLOCKED,
                confidence=_max_conf([c.confidence for c in blocked]),
                block_reason=BlockReason.HEADROOM,
                retry_at=max((c.reset_at for c in blocked if c.reset_at), default=None),
                binding=max(blocked, key=lambda c: c.reset_at or now),
            )
        if not constraints:
            return base  # UNKNOWN / NONE: nothing known at all

        with_fraction = [c for c in constraints if c.fraction_remaining is not None]
        tightest = min(with_fraction, key=lambda c: c.fraction_remaining or 0.0) if with_fraction else None
        is_low = (
            tightest is not None
            and tightest.fraction_remaining is not None
            and tightest.fraction_remaining < self._config.low_headroom_fraction
        )
        # An OPEN answer needs EVERY limit to be known; an uncounted one
        # means we may be wrong, so it caps confidence at LOW.
        confidence = _min_conf([c.confidence for c in constraints])
        if unmonitored and _CONF_RANK[confidence] > _CONF_RANK[Confidence.LOW]:
            confidence = Confidence.LOW
        return replace(
            base,
            verdict=Verdict.LOW if is_low else Verdict.OPEN,
            confidence=confidence,
            binding=tightest,
        )
