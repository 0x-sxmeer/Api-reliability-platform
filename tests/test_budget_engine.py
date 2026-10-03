"""
Unit tests for the Budget Engine (Phase 6).
"""
from __future__ import annotations

import httpx
import pytest

from gateway.core.types import CallOutcome
from gateway.engines.budget import BudgetConfig, BudgetEngine
from gateway.engines.quota import Verdict
from gateway.engines.router import AttemptRecord, RoutingOutcome
from gateway.engines.validator import ValidationResult, ValidationVerdict
from gateway.ledger.events import GatewayEvent
from gateway.ledger.store import SqliteLedgerStore

# ------------------------------------------------------------------ Tests

@pytest.mark.asyncio
async def test_budget_assessment_open_when_no_limits(tmp_path):
    ledger = SqliteLedgerStore(tmp_path / "test.db")
    engine = BudgetEngine(ledger)
    
    assessment = await engine.assess("team-a")
    assert assessment.verdict == Verdict.OPEN
    assert assessment.cost_headroom is None


@pytest.mark.asyncio
async def test_budget_assessment_blocks_when_identity_limit_exceeded(tmp_path):
    ledger = SqliteLedgerStore(tmp_path / "test.db")
    # Seed ledger with $10 spend for team-a
    event = GatewayEvent(
        identity_key="team-a",
        provider="openai",
        operation="test",
        outcome=CallOutcome.SUCCESS,
        cost_usd=10.0,
    )
    await ledger.append(event)
    
    # Configure budget limit of $5 for team-a
    config = BudgetConfig(identity_monthly_usd={"team-a": 5.0})
    engine = BudgetEngine(ledger, config)
    
    assessment = await engine.assess("team-a")
    assert assessment.verdict == Verdict.BLOCKED
    assert assessment.cost_headroom == 0.0
    assert "Identity monthly budget exceeded" in assessment.block_reason


@pytest.mark.asyncio
async def test_budget_assessment_open_when_identity_under_limit(tmp_path):
    ledger = SqliteLedgerStore(tmp_path / "test.db")
    # Seed ledger with $2 spend for team-a
    event = GatewayEvent(
        identity_key="team-a",
        provider="openai",
        operation="test",
        outcome=CallOutcome.SUCCESS,
        cost_usd=2.0,
    )
    await ledger.append(event)
    
    config = BudgetConfig(identity_monthly_usd={"team-a": 5.0})
    engine = BudgetEngine(ledger, config)
    
    assessment = await engine.assess("team-a")
    assert assessment.verdict == Verdict.OPEN
    assert assessment.cost_headroom == 3.0


@pytest.mark.asyncio
async def test_budget_assessment_blocks_when_global_limit_exceeded(tmp_path):
    ledger = SqliteLedgerStore(tmp_path / "test.db")
    # Seed ledger with $20 total spend across all users
    event = GatewayEvent(
        identity_key="team-x",
        provider="openai",
        operation="test",
        outcome=CallOutcome.SUCCESS,
        cost_usd=20.0,
    )
    await ledger.append(event)
    
    config = BudgetConfig(global_monthly_usd=15.0)
    engine = BudgetEngine(ledger, config)
    
    assessment = await engine.assess("team-y")
    assert assessment.verdict == Verdict.BLOCKED
    assert assessment.cost_headroom == 0.0
    assert "Global monthly budget exceeded" in assessment.block_reason


@pytest.mark.asyncio
async def test_budget_record_waste(tmp_path):
    ledger = SqliteLedgerStore(tmp_path / "test.db")
    engine = BudgetEngine(ledger)
    
    # Create fake attempts
    # Attempt 1: Billed, but VALID (Not waste)
    from gateway.core.executor import CallResult
    
    event1 = GatewayEvent(
        identity_key="team-a", provider="p", operation="o", outcome=CallOutcome.SUCCESS, cost_usd=1.0
    )
    res1 = CallResult(event=event1, response=httpx.Response(200), exception=None)
    val1 = ValidationResult(ValidationVerdict.VALID, 0.0, (), False, False, True, CallOutcome.SUCCESS, ())
    
    # Attempt 2: Billed and INVALID (Waste)
    event2 = GatewayEvent(
        identity_key="team-a", provider="p", operation="o", outcome=CallOutcome.SUCCESS, cost_usd=2.5
    )
    res2 = CallResult(event=event2, response=httpx.Response(200), exception=None)
    val2 = ValidationResult(ValidationVerdict.INVALID, 1.0, (), True, True, True, CallOutcome.SUSPECTED_SILENT_FAILURE, ())
    
    # Attempt 3: Not Billed and INVALID (Not waste, cost_usd=None or was_billed=False)
    event3 = GatewayEvent(
        identity_key="team-a", provider="p", operation="o", outcome=CallOutcome.SUCCESS, cost_usd=0.0
    )
    res3 = CallResult(event=event3, response=httpx.Response(200), exception=None)
    val3 = ValidationResult(ValidationVerdict.INVALID, 1.0, (), True, True, False, CallOutcome.FAILURE, ())
    
    # Attempt 4: Billed and SUSPECT (Waste)
    event4 = GatewayEvent(
        identity_key="team-a", provider="p", operation="o", outcome=CallOutcome.SUCCESS, cost_usd=3.1
    )
    res4 = CallResult(event=event4, response=httpx.Response(200), exception=None)
    val4 = ValidationResult(ValidationVerdict.SUSPECT, 0.8, (), True, False, True, CallOutcome.SUCCESS, ())
    
    class FakeTarget:
        adapter = None
        executor = None

    att1 = AttemptRecord(target=FakeTarget(), assessment=None, result=res1, validation=val1)
    att2 = AttemptRecord(target=FakeTarget(), assessment=None, result=res2, validation=val2)
    att3 = AttemptRecord(target=FakeTarget(), assessment=None, result=res3, validation=val3)
    att4 = AttemptRecord(target=FakeTarget(), assessment=None, result=res4, validation=val4)
    
    outcome = RoutingOutcome(
        final_result=res4,
        attempts=(att1, att2, att3, att4),
        succeeded=False,
        final_validation=val4
    )
    
    wasted = await engine.record_waste(outcome)
    assert wasted == 5.6  # 2.5 + 3.1
