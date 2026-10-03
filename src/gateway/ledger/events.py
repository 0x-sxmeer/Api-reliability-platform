"""
The event schema for the Unified Event Ledger.

One event is written per call (and, in later phases, per webhook). This
schema is deliberately provider-agnostic — see the note on
GatewayEvent.raw_provider_metadata for where provider-specific detail
is still allowed to live without polluting the core queryable fields.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from gateway.core.types import CallOutcome, ErrorCategory, utcnow


class GatewayEvent(BaseModel):
    """
    One row in the ledger. This is intentionally flat rather than deeply
    nested — every field here is something the eventual dashboard (Phase 9)
    or the Budget Engine (Phase 6) will want to filter or aggregate on
    directly, and flat fields are trivial to index in SQLite/Postgres.
    """

    # --- identity ---
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=utcnow)

    # --- who made the call (Phase 6's Budget Engine keys on these) ---
    identity_key: str
    # A caller-supplied identifier for "who is responsible for this
    # spend" — could be an API key hash, a team slug, a customer ID.
    # Deliberately just a string in Phase 1: the identity/auth model
    # is Phase 7's job. This field exists now so later phases don't
    # have to do a schema migration to add attribution.

    # --- what was called ---
    provider: str  # matches ProviderAdapter.provider_name
    operation: str  # matches AdapterRequest.operation

    # --- outcome ---
    outcome: CallOutcome
    error_category: ErrorCategory | None = None
    http_status: int | None = None
    provider_request_id: str | None = None
    # The provider's own id for this call (Anthropic `request-id`, OpenAI
    # `x-request-id`, ...). Needed for support tickets, and for reconciling
    # a PENDING event with a later async signal (Category 7). Nullable:
    # transport failures never received a response, so have none.

    quota_bucket: str | None = None
    # Which rate-limit counters this call was metered against
    # (ProviderAdapter.quota_bucket(), opaque to everything but that
    # adapter). Promoted from raw_provider_metadata to a real column
    # because the Quota Engine (Phase 3) filters on it on every
    # ledger-derived assessment -- exactly the promotion trigger described
    # on raw_provider_metadata below. NULL for rows written before
    # migration 0004.

    latency_ms: float | None = None
    cost_usd: float | None = None

    # --- Phase 7: Reconciliation ---
    reconciled_at: datetime | None = None
    reconciled_reason: str | None = None
    original_outcome: CallOutcome | None = None
    # Optional because Phase 1's Anthropic adapter will populate this
    # from token counts, but not every provider/operation has a
    # meaningful per-call cost (e.g. a free-tier read).

    # --- extensibility escape hatch ---
    raw_provider_metadata: dict[str, Any] = Field(default_factory=dict)
    # Provider-specific detail that doesn't deserve a first-class column
    # (e.g. Anthropic's exact rate-limit headers, a Stripe event type).
    # This is the ONE place adapter-specific shape is allowed to leak
    # into the ledger — as an opaque dict, never as a new top-level
    # field. If a future engine needs to query into this dict routinely,
    # that's a sign it should be promoted to a real column instead.

    # Note: pydantic v2 serializes UUID and datetime to JSON-safe types
    # automatically via model_dump(mode="json") — no custom encoder
    # config needed (the old `json_encoders` config key is deprecated
    # in v2 and was removed here).


# Pydantic needs `datetime` resolvable for the forward reference above.
from datetime import datetime

GatewayEvent.model_rebuild()
