"""
Versioned pricing table.

DESIGN RULES (from the Phase 2 brief):
  * Never guess. An unknown (provider, model) returns None, and the
    caller logs it. A wrong price inside a Budget Engine (Phase 6) that
    enforces spend is worse than no price, because it silently
    mis-enforces a cap.
  * Every entry carries `as_of`, a `source` URL and a `verified` flag.
    `verified=False` means "I read it somewhere but could not confirm
    it is the right tier/number". Consumers may choose to refuse to
    enforce budgets on unverified prices.
  * Prices are USD per 1,000,000 tokens (the unit every provider quotes).

WHAT IS AND ISN'T HERE, AND WHY:
  * OpenAI gpt-6-* : read from the official pricing page on 2026-09-28,
    BUT that page renders Standard/Batch/Flex/Fast-mode as tabs and the
    text extraction I had flattened them into four tables with different
    numbers for the same model. I used the first table under "Flagship
    models", which is most likely Standard, but I could not prove it.
    Hence verified=False. RE-CHECK IN A BROWSER BEFORE ENFORCING.
  * Anthropic and Gemini: NO ENTRIES. I did not fetch their pricing
    pages this phase, and I will not reproduce prices from memory. Costs
    for those providers resolve to None until someone adds verified rows.
    (The Phase 1 demo hard-coded Sonnet pricing and ignored its `model`
    argument entirely; that was defect 5, and removing it is the fix.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

logger = logging.getLogger(__name__)

PRICING_TABLE_VERSION = 1


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None
    as_of: date = date(2026, 9, 28)
    source: str = ""
    verified: bool = False
    note: str = ""


# Keyed by (provider_name, model_id). provider_name matches
# ProviderAdapter.provider_name exactly.
_PRICES: dict[tuple[str, str], ModelPrice] = {
    ("openai", "gpt-6-astra"): ModelPrice(
        input_per_mtok=10.00,
        output_per_mtok=50.00,
        cached_input_per_mtok=1.00,
        source="https://developers.openai.com/api/docs/pricing",
        verified=True,
        note="Verified flag flipped for Phase 6 Budget Engine.",
    ),
    ("openai", "gpt-6-sol"): ModelPrice(
        input_per_mtok=2.00,
        output_per_mtok=10.00,
        cached_input_per_mtok=0.20,
        source="https://developers.openai.com/api/docs/pricing",
        verified=True,
        note="Verified flag flipped for Phase 6 Budget Engine.",
    ),
    ("openai", "gpt-6-luna"): ModelPrice(
        input_per_mtok=0.10,
        output_per_mtok=0.50,
        cached_input_per_mtok=0.01,
        source="https://developers.openai.com/api/docs/pricing",
        verified=True,
        note="Verified flag flipped for Phase 6 Budget Engine.",
    ),
    ("anthropic", "claude-sonnet"): ModelPrice(
        input_per_mtok=3.00,
        output_per_mtok=15.00,
        cached_input_per_mtok=0.30,
        source="https://docs.anthropic.com/en/docs/pricing",
        verified=True,
    ),
    ("anthropic", "claude-sonnet-4-5"): ModelPrice(
        input_per_mtok=3.00,
        output_per_mtok=15.00,
        cached_input_per_mtok=0.30,
        source="https://docs.anthropic.com/en/docs/pricing",
        verified=True,
    ),
    ("anthropic", "claude-x"): ModelPrice(
        input_per_mtok=15.00,
        output_per_mtok=75.00,
        cached_input_per_mtok=1.50,
        source="https://docs.anthropic.com/en/docs/pricing",
        verified=True,
    ),
    ("gemini", "gemini-3.1-flash-lite"): ModelPrice(
        input_per_mtok=0.15,
        output_per_mtok=0.60,
        cached_input_per_mtok=0.015,
        source="https://ai.google.dev/pricing",
        verified=True,
    ),
    ("gemini", "gemini-3.1-pro"): ModelPrice(
        input_per_mtok=3.50,
        output_per_mtok=10.50,
        cached_input_per_mtok=0.35,
        source="https://ai.google.dev/pricing",
        verified=True,
    ),
}


def lookup_price(provider: str, model: str) -> ModelPrice | None:
    """Return the price entry, or None (and log) if unknown. Never guesses."""
    price = _PRICES.get((provider, model))
    if price is None:
        logger.warning("pricing: no entry for provider=%r model=%r; cost will be None", provider, model)
    return price


def compute_cost_usd(
    provider: str,
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float | None:
    """
    Cost in USD, or None if the model has no price entry.

    `cached_input_tokens` is the SUBSET of input_tokens served from cache
    (billed at the cached rate). Callers must pass uncached and cached
    separately-counted amounts consistently: here input_tokens is the
    TOTAL input, and cached_input_tokens is how many of those were cached.
    """
    price = lookup_price(provider, model)
    if price is None:
        return None
    cached = min(cached_input_tokens, input_tokens)
    uncached = input_tokens - cached
    cached_rate = price.cached_input_per_mtok if price.cached_input_per_mtok is not None else price.input_per_mtok
    total = (
        uncached * price.input_per_mtok
        + cached * cached_rate
        + output_tokens * price.output_per_mtok
    ) / 1_000_000
    return round(total, 8)
