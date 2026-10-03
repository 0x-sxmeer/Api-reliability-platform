"""
Phase 7: Reconciliation Engine
"""
from __future__ import annotations

import logging
from uuid import UUID

from gateway.core.types import CallOutcome, ErrorCategory
from gateway.ledger.store import LedgerStore

logger = logging.getLogger(__name__)

class ReconciliationEngine:
    """
    The Reconciliation Engine processes downstream signals (like webhooks or polling)
    to definitively settle uncertain events in the ledger, enforcing strict state-machine rules.
    """
    def __init__(self, ledger: LedgerStore) -> None:
        self._ledger = ledger

    async def settle_event(
        self,
        event_id: UUID,
        final_outcome: CallOutcome,
        reason: str,
        final_cost: float | None = None,
        final_error: ErrorCategory | None = None,
    ) -> None:
        """
        Safely transition an event from PENDING or SUSPECTED_SILENT_FAILURE to a terminal state.
        """
        if final_outcome in (CallOutcome.PENDING, CallOutcome.SUSPECTED_SILENT_FAILURE):
            raise ValueError(f"Cannot settle to non-terminal outcome: {final_outcome}")
            
        await self._ledger.reconcile(
            event_id=event_id,
            final_outcome=final_outcome,
            reason=reason,
            final_cost=final_cost,
            final_error=final_error,
        )
        logger.info("reconciliation_engine: successfully settled event %s to %s", event_id, final_outcome.value)

    async def get_unresolved_events(self, provider: str | None = None, limit: int = 100):
        """
        Find unresolved events that might need reconciliation.
        """
        # In a real system, you might want both PENDING and SUSPECTED.
        # We'll just return PENDING for this simple API.
        return await self._ledger.query(
            provider=provider,
            outcome=CallOutcome.PENDING,
            limit=limit
        )
