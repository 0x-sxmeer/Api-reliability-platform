"""
Unit tests for the Policy Engine (Phase 8).
"""
from __future__ import annotations

import pytest

from gateway.core.adapter import AdapterRequest
from gateway.core.types import CallOutcome, ErrorCategory
from gateway.ledger.store import SqliteLedgerStore
from gateway.engines.policy import (
    PolicyEngine,
    PolicyConfig,
    OperationPolicy,
    PolicyVerdict,
)

@pytest.mark.asyncio
async def test_policy_engine_default_deny(tmp_path):
    ledger = SqliteLedgerStore(tmp_path / "test.db")
    config = PolicyConfig(default_deny=True)
    engine = PolicyEngine(ledger, config)
    
    assessment = await engine.assess("unknown-team", "openai", "chat")
    
    assert assessment.verdict == PolicyVerdict.DENIED
    assert "No policy configured" in assessment.reason
    
    # Assert it was logged to ledger
    rows = await ledger.query(identity_key="unknown-team")
    assert len(rows) == 1
    assert rows[0].outcome == CallOutcome.FAILURE
    assert rows[0].error_category == ErrorCategory.REQUIRES_HUMAN_ACTION
    assert rows[0].http_status == 403


@pytest.mark.asyncio
async def test_policy_engine_allows_matching_rules(tmp_path):
    ledger = SqliteLedgerStore(tmp_path / "test.db")
    config = PolicyConfig(
        policies={
            "team-a": [
                OperationPolicy(allowed_providers=["openai"], allowed_operations=["chat"])
            ]
        }
    )
    engine = PolicyEngine(ledger, config)
    
    assessment = await engine.assess("team-a", "openai", "chat")
    
    assert assessment.verdict == PolicyVerdict.ALLOWED
    
    # Assert nothing was logged for success (AuthZ successful, real request handles logging)
    rows = await ledger.query(identity_key="team-a")
    assert len(rows) == 0


@pytest.mark.asyncio
async def test_policy_engine_blocks_unauthorized_operation(tmp_path):
    ledger = SqliteLedgerStore(tmp_path / "test.db")
    config = PolicyConfig(
        policies={
            "team-a": [
                OperationPolicy(allowed_providers=["openai"], allowed_operations=["chat"])
            ]
        }
    )
    engine = PolicyEngine(ledger, config)
    
    # Trying to fine-tune
    assessment = await engine.assess("team-a", "openai", "fine-tune")
    
    assert assessment.verdict == PolicyVerdict.DENIED
    assert "not authorized to execute fine-tune on openai" in assessment.reason
    
    # Assert ledger log
    rows = await ledger.query(identity_key="team-a")
    assert len(rows) == 1
    assert rows[0].operation == "fine-tune"
    assert rows[0].http_status == 403


@pytest.mark.asyncio
async def test_policy_engine_wildcard_allow(tmp_path):
    ledger = SqliteLedgerStore(tmp_path / "test.db")
    config = PolicyConfig(
        policies={
            "admin-team": [
                OperationPolicy(allowed_providers="*", allowed_operations="*")
            ]
        }
    )
    engine = PolicyEngine(ledger, config)
    
    assessment = await engine.assess("admin-team", "anything", "everything")
    
    assert assessment.verdict == PolicyVerdict.ALLOWED
