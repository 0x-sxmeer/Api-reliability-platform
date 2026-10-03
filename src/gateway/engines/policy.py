"""
Phase 8: Policy Engine (Authorization Layer)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class PolicyVerdict(str, Enum):
    ALLOWED = "allowed"
    DENIED = "denied"

@dataclass(frozen=True)
class PolicyAssessment:
    verdict: PolicyVerdict
    reason: str | None = None

@dataclass(frozen=True)
class OperationPolicy:
    """
    Rules for which providers and operations an identity may call.
    Also supports optional BOLA protections via payload path inspections in the future.
    """
    allowed_providers: list[str] | str = "*"
    allowed_operations: list[str] | str = "*"

@dataclass(frozen=True)
class PolicyConfig:
    # Mapping of identity_key -> list of policies
    policies: dict[str, list[OperationPolicy]] = field(default_factory=dict)
    default_deny: bool = True

import uuid
from datetime import UTC, datetime

from gateway.core.types import CallOutcome, ErrorCategory
from gateway.ledger.events import GatewayEvent
from gateway.ledger.store import LedgerStore


class PolicyEngine:
    """
    Enforces Object-level and Function-level access rules (Phase 8).
    Evaluates an identity and their request against configured policies.
    """
    def __init__(self, ledger: LedgerStore, config: PolicyConfig | None = None) -> None:
        self._ledger = ledger
        self._config = config or PolicyConfig()

    async def assess(self, identity_key: str, provider: str, operation: str) -> PolicyAssessment:
        rules = self._config.policies.get(identity_key)
        
        assessment = None
        if not rules:
            if self._config.default_deny:
                assessment = PolicyAssessment(
                    PolicyVerdict.DENIED, 
                    f"No policy configured for identity {identity_key}"
                )
            else:
                assessment = PolicyAssessment(PolicyVerdict.ALLOWED)
        else:
            allowed = False
            for rule in rules:
                provider_match = rule.allowed_providers == "*" or provider in rule.allowed_providers
                operation_match = rule.allowed_operations == "*" or operation in rule.allowed_operations
                
                if provider_match and operation_match:
                    allowed = True
                    break
                    
            if allowed:
                assessment = PolicyAssessment(PolicyVerdict.ALLOWED)
            else:
                assessment = PolicyAssessment(
                    PolicyVerdict.DENIED,
                    f"Identity {identity_key} is not authorized to execute {operation} on {provider}"
                )

        if assessment.verdict == PolicyVerdict.DENIED:
            # Write to ledger immediately for security audit trail
            event = GatewayEvent(
                event_id=uuid.uuid4(),
                timestamp=datetime.now(UTC),
                identity_key=identity_key,
                provider=provider,
                operation=operation,
                outcome=CallOutcome.FAILURE,
                error_category=ErrorCategory.REQUIRES_HUMAN_ACTION,
                http_status=403,
            )
            await self._ledger.append(event)
            
        return assessment
