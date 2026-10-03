"""
Phase 6: Budget Engine
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

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
        """
        now = datetime.now(UTC)
        start_of_month = datetime(now.year, now.month, 1, tzinfo=UTC)
        
        identity_budget = self._config.identity_monthly_usd.get(identity_key)
        global_budget = self._config.global_monthly_usd
        
        if identity_budget is None and global_budget is None:
            return BudgetAssessment(Verdict.OPEN, None, now)
            
        identity_spend = await self._ledger.sum_cost_since(since=start_of_month, identity_key=identity_key)
        
        # Check identity budget
        if identity_budget is not None and identity_spend >= identity_budget:
                return BudgetAssessment(
                    Verdict.BLOCKED, 
                    0.0, 
                    now, 
                    f"Identity monthly budget exceeded ({identity_budget} USD)"
                )
        
        global_spend = 0.0
        # Check global budget
        if global_budget is not None:
            global_spend = await self._ledger.sum_cost_since(since=start_of_month)
            if global_spend >= global_budget:
                return BudgetAssessment(
                    Verdict.BLOCKED, 
                    0.0, 
                    now, 
                    f"Global monthly budget exceeded ({global_budget} USD)"
                )
                
        # Calculate headroom
        headroom = float("inf")
        if identity_budget is not None:
            headroom = min(headroom, identity_budget - identity_spend)
        if global_budget is not None:
            headroom = min(headroom, global_budget - global_spend)
            
        return BudgetAssessment(Verdict.OPEN, headroom, now)

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
