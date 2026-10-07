# API Reliability, Security & Cost-Governance Gateway

A unified layer between applications and the external APIs they call
(LLM providers, payment processors, messaging/email, maps/geo, cloud
infra), replacing per-integration hand-rolled retries, rate-limit
handling, and cost tracking with one shared reliability + governance
layer.

## Status: Active Development (Phase 4 of 9)

Phase 1 shipped the Unified Event Ledger and an Anthropic adapter.
Phase 2 fixed six defects found in Phase 1, added OpenAI and Gemini
adapters, and added a contract test suite that every future adapter
must pass. Phase 3 added the Quota Engine (`gateway.engines.quota`):
proactive headroom tracking from live response headers where a provider
sends them (Anthropic, OpenAI) and from the ledger where it doesn't
(Gemini), with an honest confidence level on every answer. **Phase 4
added the Response Validator** (`gateway.engines.validator`): policy-
configurable validation of completed calls, catching "200-but-actually-
failure" responses across all three providers. Phase 4's changes to the
shared interface are logged in `docs/interface-changes.md`. This repo now
implements:

- **`gateway.engines.validator.ResponseValidator`** -- `validate()` takes
  a completed `CallResult` and returns a `ValidationResult` with:
  verdict (`VALID`/`SUSPECT`/`INVALID`), confidence, typed findings,
  and routing recommendations (`should_retry`,
  `retry_on_different_provider`, `was_billed`). Stateless, synchronous,
  vendor-agnostic. Run `python examples/validator_demo.py` to see it.

- **`gateway.engines.quota.QuotaEngine`** -- `assess()` before a call
  returns a verdict (`OPEN`/`LOW`/`PROBE`/`BLOCKED`/`UNKNOWN`), a
  confidence, the binding constraint, `retry_at`, and `resume_at`
  (`None` = unknown); `observe()` after a call feeds it the executor's
  `CallResult.rate_limit` / `.classified_error`. It models every scope a
  response describes and takes the tightest, so it can answer "would
  adding more keys/projects actually help?". Run
  `python examples/quota_demo.py` to see it.

- The **Provider Adapter** contract (`gateway.core.adapter`) — the
  interface every vendor integration implements. Three real adapters
  exist and all pass a shared contract suite (see below).
- Shared, provider-agnostic types (`gateway.core.types`) — error
  categories (including `QUOTA_EXHAUSTED`, added this phase — see
  `docs/interface-changes.md`), rate-limit snapshots with a
  `signal_quality` field (`FULL` / `PARTIAL` / `NONE`, forced by
  Gemini's headerless responses), limiting-unit enum.
- **Three provider adapters**: `gateway.adapters.anthropic`,
  `gateway.adapters.openai`, `gateway.adapters.gemini`. Every claim
  each one encodes is sourced and tagged VERIFIED/UNVERIFIED in
  `docs/provider-notes.md` — read that file before trusting or
  modifying adapter behavior.
- **`gateway.core.executor.CallExecutor`** — the orchestration layer
  (send → classify → log) that Phase 1 left stranded inside the demo
  script. Handles transport-level exceptions (timeouts, connection
  errors) so they reach the ledger instead of vanishing.
- **`gateway.core.pricing`** — a versioned, never-guess pricing table.
  An unpriced (provider, model) pair returns `None` and logs a warning
  rather than silently using a wrong number.
- The **Unified Event Ledger** (`gateway.ledger`) — SQLite-backed,
  numbered-SQL migrations (a Phase 1 database upgrades cleanly), all
  blocking I/O off the event loop via `asyncio.to_thread`, and a
  `sum_cost_since()` for Phase 6's Budget Engine.
- **`tests/contract/test_adapter_contract.py`** — the suite every
  adapter (this phase's three, and Phases 7–8's Stripe/Twilio/generic-
  REST adapters) must pass: never raises on any status 400–599, never
  raises on garbage/empty bodies, never conflates overload with auth
  failure, etc.

See `docs/interface-changes.md` for exactly what changed in the shared
interface this phase and which real provider forced each change — the
project's rule is that nothing gets added speculatively.

## Quickstart

```bash
pip install -e ".[dev]"
python examples/demo.py
```

The demo runs with **no API keys required** — each of the three
adapters is driven through `httpx.MockTransport` with realistic,
documented response shapes (see `docs/provider-notes.md`), including a
genuine transport-level timeout for OpenAI, a 529 overload for
Anthropic, and a daily-quota 429 for Gemini. It prints a per-call
narration, then a summary by provider and outcome pulled from one
shared ledger.

To run any provider against its real API instead, set the matching
environment variable before running:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
python examples/demo.py
```

Each is independent — you can set one, two, or all three; unset ones
fall back to their mock.

## Running tests

```bash
python -m pytest tests/ -v          # everything: 1,592 passed + 1 skipped
ruff check src/ tests/ examples/    # the lint gate; clean
python examples/validator_demo.py   # Phase 4 demo, fully offline
python examples/quota_demo.py       # Phase 3 demo, fully offline
python -m pytest tests/contract/ -v # just the cross-adapter contract suite
```

## Project layout

```
src/gateway/
  core/
    types.py      # Provider-agnostic shared types. NEVER import a
                   # specific provider here.
    adapter.py     # The ProviderAdapter interface every vendor
                   # integration implements.
    executor.py    # CallExecutor: send -> classify -> log one event.
                   # No retry/backoff (that's Phase 5). Depends only on
                   # ProviderAdapter + LedgerStore.
    pricing.py     # Versioned (provider, model) -> price table.
                   # Unpriced pairs return None; never guesses.
  adapters/
    anthropic.py   # All Anthropic-specific knowledge lives here.
    openai.py      # All OpenAI-specific knowledge lives here.
    gemini.py      # All Gemini-specific knowledge lives here.
    # stripe.py, twilio.py, generic REST/OAuth2 -- Phase 7+.
  engines/
    quota.py       # Quota Engine (Phase 3). Calls ONLY ProviderAdapter and
                    # LedgerStore methods; names no vendor.
    validator.py   # Response Validator (Phase 4). Validates completed
                    # CallResults against operator policy. Stateless,
                    # synchronous, vendor-agnostic. Phase 5 (Routing)
                    # plugs in beside both engines.
  ledger/
    events.py      # The GatewayEvent schema.
    store.py       # LedgerStore interface + SqliteLedgerStore, with
                    # numbered migrations and off-loop I/O.
    migrations/    # 0001_baseline.sql, 0002_add_provider_request_id.sql,
                    # 0003_cost_index.sql, 0004_quota_bucket.sql -- see
                    # store.py's docstring for the migration mechanism.
docs/
  provider-notes.md    # Every provider claim, VERIFIED or UNVERIFIED,
                        # with its source. Read before trusting or
                        # editing adapter behavior.
  interface-changes.md # What changed in the shared interface this
                        # phase, which provider forced it, and what was
                        # considered and rejected as not-yet-forced.
examples/
  demo.py          # End-to-end Phase 2 demo: all three adapters, one
                    # ledger, one executor.
  quota_demo.py    # Phase 3 demo: header-derived (HIGH) vs ledger-derived
                    # (LOWER) confidence, OpenAI's binding project scope,
                    # unknown-resume probing. Offline; no vendor branches.
  validator_demo.py # Phase 4 demo: the validator catching 200-but-actually-
                    # failure across all three providers, plus policy config.
tests/
  test_anthropic_adapter.py
  test_openai_adapter.py
  test_gemini_adapter.py
  test_executor.py
  test_ledger_store.py
  test_ledger_migrations.py
  test_quota_engine.py       # Engine logic, via a vendor-free FakeAdapter.
  test_quota_integration.py  # Engine + real adapters + executor + ledger,
                              # and the AST-based architecture-rule tests.
  test_validator_engine.py   # Validator unit tests, vendor-free.
  contract/
    test_adapter_contract.py   # Runs once per adapter in its ADAPTERS
                                # list. Add a new adapter here when you
                                # build one.
```

## The one architectural rule that matters most

**Vendor-specific knowledge lives only inside `adapters/`.** If you're
adding logic to `core/`, `engines/`, or `ledger/` and find yourself
writing something like `if provider == "openai"`, stop — that almost
always means either:
1. The adapter interface (`core/adapter.py`) is missing a method that
   should express this as a provider-agnostic capability, or
2. The logic genuinely belongs inside a specific adapter file instead.

This was grep-verified clean at the end of Phase 2 and again at the end
of Phase 3 across `core/`, `engines/`, and `ledger/` — no `provider ==`
comparisons anywhere outside `adapters/`. Phase 3 also made it a *test*
(`tests/test_quota_integration.py`): an AST scan fails the build if engine
code names a vendor or imports `gateway.adapters`, and a structural check
fails it if the engine touches any adapter/ledger attribute that isn't
declared on the abstract interface. (Honest scope note: `examples/demo.py`
has six `== "<vendor>"` branches that build adapters and mock bodies — fixture
wiring, not engine logic — and is outside what the rule or the grep claim
covered.) The one place a provider's
*name* legitimately appears outside `adapters/` is `pricing.py`'s
lookup table, which is data keyed by `(provider, model)`, not a branch
in control flow — that distinction is what the rule actually protects.

## Design decisions flagged for review

Carried over from Phase 1, still open:
- **Async throughout, even where SQLite doesn't strictly need it** — now
  actually delivering on that: Phase 2 moved all blocking `sqlite3`
  calls into `asyncio.to_thread`, so this is no longer a gap, just a
  documented concurrency model (see `store.py`'s docstring) that is
  correct but not high-throughput. Right for now; Postgres is the
  planned Phase 9+ replacement.
- **`classify_error`/`check_known_bad_patterns` take the raw
  `httpx.Response`.** Held up across three genuinely different
  providers (Anthropic: body-and-headers; OpenAI: `error.code` field;
  Gemini: gRPC `status` field plus a `details[]` array) — the contract
  test suite proves none of the three needs a different shape.

Resolved in Phase 3 (the four questions Phase 2 left open; reasoning in
`docs/interface-changes.md` and the engine's module docstring):
- **OpenAI dual scope: model every scope, take the tightest.** The
  project headers (limit/remaining/reset, tokens only — the notes had
  listed only two of the three) now ride in `RateLimitSnapshot.
  additional_scopes`. Deciding argument: a call succeeds only if *all*
  scopes have room, and which one binds is data-dependent, so only
  modeling both can answer the project's founding question ("would more
  keys help?"). Bigger finding: scope has a second axis the brief did not
  mention — **per-model buckets** (all three providers) — added as
  `ProviderAdapter.quota_bucket()`.
- **Gemini has no headers: count from the ledger, with asymmetric
  confidence.** Ledger usage is a lower bound on true usage, so a
  ledger-derived "blocked" is decent evidence (MEDIUM) and a ledger-derived
  "open" is weak (LOW). What can't be counted is reported `UNMONITORED`,
  never as zero: Gemini TPM (the ledger stores no tokens) and Gemini spend
  (no prices exist, so `SUM(cost_usd)` is a confident-looking 0.0).
  Numeric limits must be operator-configured — no API publishes them.
- **Not-definitively-classified `QUOTA_EXHAUSTED`: trust the category,
  distrust the duration.** It blocks (failover exists for exactly this),
  is marked `provisional` at LOW confidence, is probed early (30s), and
  one success clears it. Treating it as `RATE_LIMITED` would repeat the
  documented retry-a-daily-quota-as-if-per-minute failure.
- **Resume time unknown is a first-class state:** a circuit breaker —
  BLOCKED, then PROBE after 5 min doubling to a 6h cap; `claim_probe()`
  hands out one probe at a time. These schedule values are engineering
  defaults, not provider facts, and are configurable.

Open going into Phase 4 (all also in `docs/interface-changes.md`):
- **`assess()` is a read, not a reservation** — concurrent callers can all
  see the same headroom. Phase 5 must decide how it schedules calls.
- **Gemini TPM and spend are unmonitored.** Cheapest fixes: a ledger
  `input_tokens` column (one migration) and verified Gemini prices.
- **Anthropic workspace caps** can bind and are indistinguishable in the
  headers; the engine's scaling advice is wrong in that case.
- **Engine state is in memory**; a restart re-learns exhaustion from one
  failed call per bucket.
- **One credential per `provider_name`** (the ledger has no credential
  dimension); `register()` refuses a second rather than merging silently.

Still open from Phase 2:
- **Gemini's `QUOTA_EXHAUSTED` vs `RATE_LIMITED` split relies on an
  UNVERIFIED heuristic** (string-matching `quotaId` for "PerDay" vs
  "PerMinute"). Every classification derived from it sets
  `is_definitively_classified=False`, which the Quota Engine now consumes
  (see above). Worth monitoring for `quotaId` format changes.
- **No pricing entries exist for Anthropic or Gemini at all**, and
  OpenAI's three entries are `verified=False` (see
  `docs/provider-notes.md` and `pricing.py`'s own docstring). Costs for
  every provider currently resolve to `None` in practice. This needs a
  deliberate follow-up before Phase 6's Budget Engine can enforce
  anything real — and, per Phase 3, before the Quota Engine can monitor
  Gemini's spend window.
