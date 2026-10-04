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
from gateway.engines.policy import PolicyEngine, PolicyVerdict
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
    # Audit F-04 hardening: absolute ceiling on loop passes (waits + attempts),
    # guaranteeing termination independent of clock/sleep behavior.
    max_total_iterations: int = 32
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

    # Read-only accessors so a composition root (server.py) can apply the
    # same gates on non-routed paths without touching private attributes
    # (audit F-03: the provider-pinned proxy path must not bypass policy /
    # budget / quota).
    @property
    def policy(self) -> PolicyEngine | None:
        return self._policy

    @property
    def budget(self) -> BudgetEngine | None:
        return self._budget

    @property
    def quota(self) -> QuotaEngine:
        return self._quota

    async def route(
        self, request: AdapterRequest, identity_key: str | None = None
    ) -> RoutingOutcome:
        """
        Attempt the request across registered targets according to policy.
        Returns the first VALID result, or the final outcome if all attempts are exhausted.

        `identity_key` names the tenant this call is made FOR. Fixes audit F-05:
        when omitted it falls back to the first target's executor identity for
        backward compatibility, but that fallback is DEPRECATED — with multiple
        tenants sharing one router, targets[0].executor.identity_key would
        authorize/spend every request under whichever executor happens to be
        registered first. Callers must pass the authenticated identity.
        """
        if not self._targets:
            raise ValueError("No routing targets registered.")

        # 0. Policy Gate (AuthZ)
        active_targets = self._targets
        if self._policy and self._targets:
            if identity_key is None:
                logger.warning(
                    "router: route() called without identity_key; falling back to "
                    "targets[0].executor identity. Pass identity_key explicitly for "
                    "multi-tenant correctness (audit F-05)."
                )
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
            if identity_key is None:
                # No policy engine ran (identity not resolved yet) — same
                # deprecated fallback as the policy gate (audit F-05).
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

        # Fixes audit F-04: the BLOCKED-wait branch previously slept and looped
        # WITHOUT charging anything against a ceiling — a target whose retry_at
        # keeps landing inside the wait window could spin indefinitely (waits
        # never increment total_attempts). Cumulative sleep time is now charged
        # to max_wait_seconds, so route() always terminates.
        #
        # Hardening (this session): cumulative *loop iterations* are also
        # capped. The sleep-deadline alone bounds wall-clock time only when
        # every spin actually sleeps; a zero/near-zero wait (retry_at within
        # milliseconds, or an injected sleep_fn that returns instantly in
        # tests) would still allow thousands of re-assess spins per second.
        # max_total_iterations gives a deterministic termination guarantee
        # independent of clock behavior.
        deadline_slept = 0.0
        iterations = 0
        max_iterations = self._config.max_total_iterations

        while total_attempts < self._config.max_total_attempts and target_idx < len(active_targets):
            # Audit F-04 hardening: every pass through the loop (including pure
            # re-assess spins after a wait) is charged against a fixed ceiling,
            # so termination does not depend on clock behavior at all.
            iterations += 1
            if iterations > max_iterations:
                break
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
                    if wait_seconds <= self._config.max_wait_seconds and \
                            (deadline_slept + wait_seconds) <= self._config.max_wait_seconds:
                        await self._config.sleep_fn(wait_seconds)
                        deadline_slept += wait_seconds
                        # We do not count waiting as an attempt on the total limit or provider limit
                        # until we actually try to execute. Since we slept, loop around to re-assess.
                        # The cumulative sleep IS bounded by the deadline check above (audit F-04).
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

    def target_for(self, provider_name: str) -> RoutingTarget | None:
        """Find a target by its provider name without leaking provider names to the caller."""
        return next((t for t in self._targets if t.adapter.provider_name.startswith(provider_name) and len(t.adapter.provider_name) == len(provider_name)), None)

    def registered_providers(self) -> list[str]:
        """Return the names of all registered providers."""
        return [t.adapter.provider_name for t in self._targets]
