"""
Phase 5: Routing Engine
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from gateway.core.adapter import AdapterRequest, ProviderAdapter
from gateway.core.executor import CallExecutor, CallResult
from gateway.engines.budget import BudgetEngine
from gateway.engines.policy import PolicyAssessment, PolicyEngine, PolicyVerdict
from gateway.engines.quota import QuotaAssessment, QuotaEngine, Verdict
from gateway.engines.validator import (
    ResponseValidator,
    ValidationResult,
    ValidationVerdict,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingTarget:
    """A pre-built adapter and its dedicated executor."""

    adapter: ProviderAdapter
    executor: CallExecutor


@dataclass(frozen=True)
class RoutingConfig:
    max_attempts_per_provider: int = 2
    max_total_attempts: int = 4
    backoff_base_seconds: float = 1.0
    max_wait_seconds: float = 10.0
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep


@dataclass(frozen=True)
class AttemptRecord:
    target: RoutingTarget
    assessment: QuotaAssessment
    result: CallResult | None
    validation: ValidationResult | None


@dataclass(frozen=True)
class RoutingOutcome:
    final_result: CallResult | None
    attempts: tuple[AttemptRecord, ...]
    succeeded: bool
    final_validation: ValidationResult | None


class RoutingEngine:
    """
    Orchestrates pre-call quota assessment, execution, and post-call validation,
    with a policy-driven retry-and-failover loop.
    """

    def __init__(
        self,
        quota_engine: QuotaEngine,
        validator: ResponseValidator,
        config: RoutingConfig | None = None,
        budget_engine: BudgetEngine | None = None,
        policy_engine: PolicyEngine | None = None,
    ) -> None:
        self._quota = quota_engine
        self._validator = validator
        self._config = config or RoutingConfig()
        self._budget = budget_engine
        self._policy = policy_engine
        self._targets: list[RoutingTarget] = []

    def register(self, target: RoutingTarget) -> None:
        """Register a target for the routing pool."""
        self._targets.append(target)

    @property
    def targets(self) -> tuple[RoutingTarget, ...]:
        """Read-only view of registered targets (in failover preference order)."""
        return tuple(self._targets)

    @property
    def validator(self) -> ResponseValidator:
        return self._validator

    def target_for(self, provider_name: str) -> RoutingTarget | None:
        """Look up a registered target by the name its adapter publishes.

        Generic registry lookup keyed on each adapter's own ``provider_name``
        property (the same pattern as the quota engine's registry). This is
        dispatch-by-registry, not vendor branching: no vendor-specific
        knowledge or per-vendor behavior lives here — the caller passes an
        opaque name and gets back whichever target published it.
        """
        return {t.adapter.provider_name: t for t in self._targets}.get(provider_name)

    def registered_providers(self) -> list[str]:
        """Provider names of all registered targets, in failover order."""
        return [t.adapter.provider_name for t in self._targets]

    async def policy_assess(
        self, identity_key: str, provider_name: str, operation: str
    ) -> PolicyAssessment | None:
        """Run the policy gate for one (identity, provider, operation) triple.

        Returns None when no PolicyEngine is configured (gate disabled).
        Delegates to the engine so callers never touch provider internals.
        """
        if self._policy is None:
            return None
        return await self._policy.assess(identity_key, provider_name, operation)

    async def route(self, request: AdapterRequest) -> RoutingOutcome:
        """
        Attempt the request across registered targets according to policy.
        Returns the first VALID result, or the final outcome if all attempts are exhausted.
        """
        if not self._targets:
            raise ValueError("No routing targets registered.")

        # 0. Policy Gate (AuthZ)
        active_targets = self._targets
        if self._policy and self._targets:
            identity_key = self._targets[0].executor.identity_key
            allowed_targets = []
            auth_denial_reason = None
            
            for target in self._targets:
                policy_assessment = await self._policy.assess(
                    identity_key, target.adapter.provider_name, request.operation
                )
                if policy_assessment.verdict == PolicyVerdict.ALLOWED:
                    allowed_targets.append(target)
                else:
                    if not auth_denial_reason:
                        auth_denial_reason = policy_assessment.reason
                        
            if not allowed_targets:
                # Return an immediate blocked outcome for AuthZ failures
                logger.warning("AuthZ Denied for %s: %s", identity_key, auth_denial_reason)
                return RoutingOutcome(None, (), False, None)
                
            active_targets = allowed_targets

        # 1. Budget Gate
        if self._budget and active_targets:
            identity_key = active_targets[0].executor.identity_key
            budget_assessment = await self._budget.assess(identity_key)
            if budget_assessment.verdict is Verdict.BLOCKED:
                # Return an immediate blocked outcome
                return RoutingOutcome(None, (), False, None)

        attempts: list[AttemptRecord] = []
        total_attempts = 0
        target_idx = 0
        target_attempt_counts: dict[int, int] = {id(t): 0 for t in active_targets}

        fallback_result: CallResult | None = None
        fallback_validation: ValidationResult | None = None
        succeeded = False

        while total_attempts < self._config.max_total_attempts and target_idx < len(active_targets):
            target = active_targets[target_idx]
            target_id = id(target)

            if target_attempt_counts[target_id] >= self._config.max_attempts_per_provider:
                target_idx += 1
                continue

            # 1. Pre-call quota gate
            assessment = await self._quota.assess(target.adapter, request)

            if assessment.verdict == Verdict.BLOCKED:
                now = datetime.now(UTC)
                # Can we wait?
                if assessment.retry_at and assessment.retry_at > now:
                    wait_seconds = (assessment.retry_at - now).total_seconds()
                    if wait_seconds <= self._config.max_wait_seconds:
                        await self._config.sleep_fn(wait_seconds)
                        # We do not count waiting as an attempt on the total limit or provider limit
                        # until we actually try to execute. Since we slept, loop around to re-assess
                        continue
                    else:
                        # Beyond wait budget, immediately fail over
                        attempts.append(AttemptRecord(target, assessment, None, None))
                        target_idx += 1
                        continue
                else:
                    # Unknown or already passed retry_at, fail over immediately
                    attempts.append(AttemptRecord(target, assessment, None, None))
                    target_idx += 1
                    continue

            # 2. Dispatch
            result = await target.executor.execute(request)
            total_attempts += 1
            target_attempt_counts[target_id] += 1

            # 3. Post-call feed
            self._quota.observe(
                target.adapter,
                request,
                snapshot=result.rate_limit,
                error=result.classified_error,
            )

            # 4. Response validation
            validation = self._validator.validate(result)
            attempts.append(AttemptRecord(target, assessment, result, validation))

            # Keep track of the best fallback result if we don't succeed.
            # Usually the last one is fine, but if we had a SUSPECT we might want that over INVALID?
            # The prompt says: "SUSPECT result is used when no VALID alternative is available."
            # So if current is SUSPECT, we should hold onto it if fallback is currently None or INVALID.
            if validation.verdict == ValidationVerdict.SUSPECT:
                if fallback_validation is None or fallback_validation.verdict != ValidationVerdict.SUSPECT:
                    fallback_result = result
                    fallback_validation = validation
            else:
                # If it's INVALID, we only replace fallback if we don't already have a SUSPECT
                if fallback_validation is None or fallback_validation.verdict != ValidationVerdict.SUSPECT:
                    fallback_result = result
                    fallback_validation = validation

            # 5. Retry/failover decision
            if validation.verdict == ValidationVerdict.VALID:
                fallback_result = result
                fallback_validation = validation
                succeeded = True
                break

            if not validation.should_retry:
                # PERMANENT error, give up entirely
                break

            if validation.retry_on_different_provider:
                target_idx += 1
                continue

            # Otherwise, retry same provider (if budget allows)
            if target_attempt_counts[target_id] < self._config.max_attempts_per_provider:
                # Wait backoff before retrying same provider
                if total_attempts < self._config.max_total_attempts:
                    await self._config.sleep_fn(self._config.backoff_base_seconds)
            else:
                target_idx += 1

        # We exited the loop. `succeeded` was set True only on a VALID
        # response inside the loop; otherwise it stays False (including
        # when every attempt was quota-blocked and fallback_result is None).

        outcome = RoutingOutcome(
            final_result=fallback_result,
            attempts=tuple(attempts),
            succeeded=succeeded,
            final_validation=fallback_validation,
        )
        
        if self._budget:
            await self._budget.record_waste(outcome)
            
        return outcome
