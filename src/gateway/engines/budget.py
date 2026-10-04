"""
Phase 6: Budget Engine
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from gateway.core.pricing import price_is_verified
from gateway.core.types import CallOutcome
from gateway.engines.quota import Verdict
from gateway.engines.validator import ValidationVerdict
from gateway.ledger.store import LedgerStore

if TYPE_CHECKING:
    from gateway.engines.router import RoutingOutcome

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BudgetConfig:
    """
    Specifies cost limits at different granularities.
    All limits are in USD.
    """
    global_monthly_usd: float | None = None
    identity_monthly_usd: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BudgetAssessment:
    verdict: Verdict
    cost_headroom: float | None
    assessed_at: datetime
    block_reason: str | None = None
    caveat: str | None = None
    """
    Audit remediation (F-09): headroom is a LOWER bound whenever spend
    cannot be fully accounted for. A non-None caveat means "the number
    you are looking at may understate actual spend" — never render an
    assessment with a caveat as a confident OPEN. The three sources of
    uncertainty, in priority order:

      1. unpriced successful events this month (cost_usd IS NULL on a
         SUCCESS row) — counted via count_unpriced_since(), the same
         discipline the Quota Engine's SPEND branch uses;
      2. any priced model whose `verified` flag is False or missing
         (Invariant #3: refuse to enforce caps on guessed prices);
      3. failed calls carry cost_usd=None by design (executor), so
         vendor-billed failures are invisible to SUM(cost_usd).
    """


class BudgetEngine:
    """
    The Budget Engine is the system's financial governor.
    It gates requests before execution based on spend, and tracks wasted spend.
    """
    def __init__(self, ledger: LedgerStore, config: BudgetConfig | None = None) -> None:
        self._ledger = ledger
        self._config = config or BudgetConfig()

    async def assess(self, identity_key: str) -> BudgetAssessment:
        """
        Check if the identity_key or global budget has been exceeded.

        Audit remediation (F-09): this used to render SUM(cost_usd) as a
        confident number while silently treating NULL-cost events as $0 —
        the exact "confident-looking 0.0" failure the Quota Engine's
        SPEND branch refuses. Now:

          * unpriced SUCCESS events since month start are counted via
            count_unpriced_since() and downgrade confidence (caveat);
          * spend on models whose price is not verified=True refuses to
            enforce (UNKNOWN verdict) rather than gate on guessed prices;
          * cost_headroom is always finite (None when unlimited) so it
            survives JSON serialization — no float('inf') leakage.
        """
        now = datetime.now(UTC)
        start_of_month = datetime(now.year, now.month, 1, tzinfo=UTC)

        identity_budget = self._config.identity_monthly_usd.get(identity_key)
        global_budget = self._config.global_monthly_usd

        if identity_budget is None and global_budget is None:
            return BudgetAssessment(Verdict.OPEN, None, now)

        # --- Refuse to enforce caps built on unverified prices ----------
        # Invariant #3: never guess. If any priced successful event seen
        # in the window carries a cost computed from a (provider, model)
        # whose price entry is not verified=True, USD totals are not
        # trustworthy enough to block traffic on. The model name lives in
        # raw_provider_metadata["model"] — GatewayEvent has no dedicated
        # model column, and quota_bucket is the closest indexed proxy
        # (adapters set it from the request's model field).
        for ev in await self._ledger.query(
            since=start_of_month, outcome=CallOutcome.SUCCESS, limit=500
        ):
            if ev.cost_usd is None:
                continue  # handled by the unpriced check below
            model = str(ev.raw_provider_metadata.get("model", ""))
            status = price_is_verified(ev.provider, model)
            if status is not True:
                reason = (
                    "is not verified=True" if status is False
                    else "has no price row (cost came from an unknown source)"
                )
                return BudgetAssessment(
                    Verdict.UNKNOWN,
                    None,
                    now,
                    block_reason=f"Budget enforcement refused: price for '{ev.provider}/{model}' {reason}",
                    caveat="Enforcing USD caps on unverified prices would be guessing.",
                )

        identity_spend = await self._ledger.sum_cost_since(since=start_of_month, identity_key=identity_key)

        # --- Honest accounting: unpriced successes are unknown spend -----
        caveats: list[str] = []
        unpriced_identity = await self._ledger.count_unpriced_since(
            since=start_of_month, identity_key=identity_key
        )
        if unpriced_identity > 0:
            caveats.append(
                f"{unpriced_identity} successful call(s) this month have no "
                "price entry (cost treated as unknown, NOT as $0)"
            )

        # Check identity budget
        if identity_budget is not None and identity_spend >= identity_budget:
            return BudgetAssessment(
                Verdict.BLOCKED,
                0.0,
                now,
                f"Identity monthly budget exceeded ({identity_budget} USD)",
                caveat="; ".join(caveats) or None,
            )

        global_spend = 0.0
        # Check global budget
        if global_budget is not None:
            unpriced_global = await self._ledger.count_unpriced_since(since=start_of_month)
            if unpriced_global > unpriced_identity:
                caveats.append(
                    f"{unpriced_global} unpriced successful call(s) globally "
                    "this month (spend above is a lower bound)"
                )
            global_spend = await self._ledger.sum_cost_since(since=start_of_month)
            if global_spend >= global_budget:
                return BudgetAssessment(
                    Verdict.BLOCKED,
                    0.0,
                    now,
                    f"Global monthly budget exceeded ({global_budget} USD)",
                    caveat="; ".join(caveats) or None,
                )

        # Calculate headroom — finite only where a cap exists (no inf leak).
        headroom: float | None = None
        if identity_budget is not None:
            headroom = identity_budget - identity_spend
        if global_budget is not None:
            remaining_global = global_budget - global_spend
            headroom = remaining_global if headroom is None else min(headroom, remaining_global)

        return BudgetAssessment(
            Verdict.OPEN, headroom, now,
            caveat="; ".join(caveats) or None,
        )

    async def record_waste(self, outcome: RoutingOutcome) -> float:
        """
        Post-routing hook that analyzes the final RoutingOutcome.
        Records and returns the total amount of wasted spend.
        Wasted spend: validation.was_billed is True AND validation.verdict is INVALID or SUSPECT.
        """
        wasted_usd = 0.0
        
        for attempt in outcome.attempts:
            val = attempt.validation
            res = attempt.result
            if (
                val and val.was_billed
                and val.verdict in (ValidationVerdict.INVALID, ValidationVerdict.SUSPECT)
                and res and res.event.cost_usd is not None
            ):
                        wasted_usd += res.event.cost_usd
                        
        if wasted_usd > 0:
            wasted_usd = round(wasted_usd, 8)
            logger.info("budget_engine: recorded wasted spend of %s USD", wasted_usd)
            # In later phases, we will persist this waste to the ledger
            # or a specific reconciliation table.
            
        return wasted_usd
