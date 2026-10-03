"""
Unit tests for the Reconciliation Engine (Phase 7).
"""
from __future__ import annotations

import pytest
from uuid import uuid4

from gateway.core.types import CallOutcome, ErrorCategory
from gateway.ledger.events import GatewayEvent
from gateway.ledger.store import SqliteLedgerStore
from gateway.engines.reconcile import ReconciliationEngine


@pytest.mark.asyncio
async def test_reconciliation_engine_settles_pending_event(tmp_path):
    ledger = SqliteLedgerStore(tmp_path / "test.db")
    engine = ReconciliationEngine(ledger)
    
    event_id = uuid4()
    # Seed a PENDING event
    event = GatewayEvent(
        event_id=event_id,
        identity_key="team-a",
        provider="openai",
        operation="test",
        outcome=CallOutcome.PENDING,
    )
    await ledger.append(event)
    
    # Assert initial state
    rows = await ledger.query(identity_key="team-a")
    assert rows[0].outcome == CallOutcome.PENDING
    assert rows[0].reconciled_at is None
    
    # Settle it
    await engine.settle_event(
        event_id=event_id,
        final_outcome=CallOutcome.SUCCESS,
        reason="Webhook received",
        final_cost=0.5
    )
    
    # Assert final state
    rows = await ledger.query(identity_key="team-a")
    assert rows[0].outcome == CallOutcome.SUCCESS
    assert rows[0].original_outcome == CallOutcome.PENDING
    assert rows[0].reconciled_reason == "Webhook received"
    assert rows[0].reconciled_at is not None
    assert rows[0].cost_usd == 0.5


@pytest.mark.asyncio
async def test_reconciliation_engine_cannot_settle_to_non_terminal(tmp_path):
    ledger = SqliteLedgerStore(tmp_path / "test.db")
    engine = ReconciliationEngine(ledger)
    
    event_id = uuid4()
    event = GatewayEvent(
        event_id=event_id,
        identity_key="team-a",
        provider="openai",
        operation="test",
        outcome=CallOutcome.PENDING,
    )
    await ledger.append(event)
    
    with pytest.raises(ValueError, match="Cannot settle to non-terminal outcome"):
        await engine.settle_event(
            event_id=event_id,
            final_outcome=CallOutcome.PENDING,
            reason="Illegal",
        )


@pytest.mark.asyncio
async def test_ledger_rejects_reconciling_already_terminal_event(tmp_path):
    ledger = SqliteLedgerStore(tmp_path / "test.db")
    engine = ReconciliationEngine(ledger)
    
    event_id = uuid4()
    # Seed a SUCCESS event
    event = GatewayEvent(
        event_id=event_id,
        identity_key="team-a",
        provider="openai",
        operation="test",
        outcome=CallOutcome.SUCCESS,
    )
    await ledger.append(event)
    
    with pytest.raises(ValueError, match="cannot be reconciled"):
        await engine.settle_event(
            event_id=event_id,
            final_outcome=CallOutcome.FAILURE,
            reason="Too late",
            final_error=ErrorCategory.PERMANENT
        )


@pytest.mark.asyncio
async def test_get_unresolved_events(tmp_path):
    ledger = SqliteLedgerStore(tmp_path / "test.db")
    engine = ReconciliationEngine(ledger)
    
    event_id = uuid4()
    event = GatewayEvent(
        event_id=event_id,
        identity_key="team-a",
        provider="openai",
        operation="test",
        outcome=CallOutcome.PENDING,
    )
    await ledger.append(event)
    
    unresolved = await engine.get_unresolved_events(provider="openai")
    assert len(unresolved) == 1
    assert unresolved[0].event_id == event_id
