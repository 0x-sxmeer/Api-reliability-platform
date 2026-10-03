# Interface change log (Phase 1 -> Phase 2)

Rule applied: a change is only made if a real provider FORCED it. Anything
speculative was rejected (see "Considered and rejected").

| # | Change | Forced by | Why |
|---|--------|-----------|-----|
| 1 | `ErrorCategory.QUOTA_EXHAUSTED` | Anthropic, OpenAI, Gemini | All three have a "quota gone, retry provably fails, failover works" state (Anthropic `enforced_spend_limit_reached`; OpenAI `credit_balance_exhausted` / `*_spend_limit_exceeded` / `organization_usage_limit_exceeded`; Gemini tier `limit: 0`). `PERMANENT` can't express "failover works"; `RATE_LIMITED` says "retry soon". |
| 2 | `ClassifiedError.resume_at` | Anthropic | Anthropic's spend-cap body states the exact resume time. OpenAI credit exhaustion has none, so it is Optional. |
| 3 | `SignalQuality` + `RateLimitSnapshot.signal_quality` | Gemini | Gemini documents no rate-limit response headers. The Quota Engine must be told when headers can't be trusted. |
| 4 | `RateLimitSnapshot.reset_at` normalized to absolute UTC + `reset_at_source` | OpenAI | OpenAI sends durations ("6m0s"); Anthropic sends RFC 3339 timestamps. One internal form needed. |
| 5 | `ProviderAdapter.classify_exception(exc)` | Executor (defect 2) + all three | Transport exceptions must reach the ledger. Mapping lives in the adapter (see executor notes). |
| 6 | `ProviderAdapter.extract_usage(response)` -> `UsageInfo` | All three | Each vendor's usage block has a different shape (Anthropic `usage.input_tokens`, OpenAI `usage.prompt_tokens`/`input_tokens`, Gemini `usageMetadata.promptTokenCount`). Cost needs tokens + model; only the adapter knows where they are. |
| 7 | `ProviderAdapter.extract_request_id(response)` | All three | Every provider returns one; needed for the ledger's `provider_request_id`. |

## Considered and rejected (not forced)
- `LimitingUnit` new members: ORGANIZATION/PROJECT/KEY already cover Anthropic (org), OpenAI (org+project), Gemini (project). No new member needed.
- A dual-scope `LimitingUnit` (e.g. a set/list instead of a single enum) for OpenAI's genuinely simultaneous org+project limits: rejected this phase. `get_limiting_unit()` reports ORGANIZATION as the primary/outer scope; the project-scoped headers are preserved in `raw_headers` (verified present in `test_openai_adapter.py::test_parse_rate_limit_headers_includes_project_scoped_in_raw`) rather than lost. This is a real simplification, not a non-issue — flagged for Phase 3: if the Quota Engine needs to reason about project-level headroom specifically (not just org-level), this will need revisiting, and the fix is a documented gap, not a silent one. **Revisited in Phase 3: it WAS forced -- see change #8 below.**
- A `RateLimitSnapshot` per-dimension breakdown (requests/input/output tokens separately): Anthropic exposes it but none of the three *requires* it yet. Phase 3 may. Raw headers are preserved. **Revisited in Phase 3: still not forced -- see "Considered and rejected in Phase 3".**
- A generic "streaming" abstraction on the adapter: not needed until Phase 5 designs retry-after-partial-output.
- Folding Gemini's newer Interactions API (cleaner `rate_limit_exceeded`/`quota_exceeded`/`too_many_requests` codes) into the same `GeminiAdapter`: rejected. Mixing two different response shapes into one `classify_error` would risk exactly the silent-misclassification failure mode this project exists to prevent. `generateContent` is what's actually integrated against today (see docs/provider-notes.md); a second adapter variant is the right shape if/when the Interactions API is added, not a branch inside this one.

## Added this phase (not an interface change, but the missing orchestration layer)
- `gateway.core.executor.CallExecutor` — not a `ProviderAdapter` interface change (no abstract method added), but the library-level orchestration class defect 1 required. It depends only on the existing `ProviderAdapter` and `LedgerStore` interfaces; no adapter needed to change to support it. This is worth logging here because it's the one non-adapter, non-type change of substance this phase made to `core/`.

## Corrections made mid-build (logged, not silently fixed)
Two claims in `GeminiAdapter`'s first draft were wrong and were corrected after further verification rather than shipped:
1. API key transport was initially implemented as a `?key=` query parameter based on an early, narrower doc read; corrected to the `x-goog-api-key` header, which current official examples lead with and which independent third-party client implementations confirm (see docs/provider-notes.md).
2. `extract_usage` initially assumed `generateContent`'s response body does not echo the model name; this was wrong — it's present as a top-level `modelVersion` string, confirmed across three independent, mutually-consistent real response examples. The planned workaround (threading the model in from the request) was unnecessary and was removed.


---

# Interface change log (Phase 2 -> Phase 3: Quota Engine)

Same rule: a change is made only if a real provider FORCED it. Each change
below is additive (no existing signature changed, one abstract method added)
and **independently revertible** -- the "To revert" column says exactly what
goes with it. Everything here was verified against official docs this
session unless tagged otherwise; see `docs/provider-notes.md`.

| # | Change | Forced by | Why | To revert |
|---|--------|-----------|-----|-----------|
| 8 | `RateLimitSnapshot.additional_scopes: tuple[RateLimitSnapshot, ...] = ()` | OpenAI | OpenAI enforces org **and** project limits at once; the project scope has its own limit/remaining/reset (tokens only). A call succeeds only if every scope has room, so a consumer must take the tightest, and *which scope binds is data-dependent* -- a fixed `get_limiting_unit()` can't say. Reusing `RateLimitSnapshot` recursively added no new type. | Delete the field and the `additional_scopes` block in `OpenAIAdapter.parse_rate_limit_headers`; the engine then sees only the org view (and says ORGANIZATION binds even when the project does). |
| 9 | `ProviderAdapter.quota_bucket(request) -> str` (**abstract**) | Anthropic, OpenAI, Gemini | All three partition limits by model (Anthropic per model class; OpenAI per model with org-defined shared pools; Gemini per model). A response's headers describe only the bucket it drew from, so state keyed by provider alone lets one model's headroom overwrite another's -- and Phase 5's whole job is routing *between* models. Abstract so a future adapter author must decide consciously (a constant is a valid answer). Contract-tested to be deterministic, non-empty, and never raise. | Remove the method from the ABC and the three adapters; key engine state by provider only. |
| 10 | `ProviderAdapter.quota_windows() -> tuple[QuotaWindow, ...]` (concrete, default `()`) + `QuotaWindow`, `QuotaDimension`, `WindowKind` | Gemini | No headers, so the engine must count from the ledger, and counting needs the window *shape*: rolling 60s (RPM/TPM), **calendar day at midnight Pacific** (RPD, VERIFIED), rolling 10-min spend (VERIFIED). That is vendor knowledge, so it lives in the adapter. Numeric limits are deliberately NOT here (not publishable through any API -- operator config). Concrete default because only header-less providers need it; making Anthropic/OpenAI declare windows nobody reads is the speculative surface this project avoids. | Remove the method/types; the Gemini ledger path loses its window definitions. |
| 11 | `CallResult.rate_limit`, `CallResult.classified_error` (both default `None`) | Executor (Quota Engine's only live feed) | `_handle_response` already computed the parsed `RateLimitSnapshot` and `ClassifiedError`, then persisted only `raw_headers` and the classification *message*. The parsed reset time, `retry_after_seconds` and `resume_at` were unrecoverable. No new computation: the objects were simply discarded. | Delete the two attributes; the engine would have to re-parse `raw_headers` (which `types.py` says it must not). |
| 12 | `GatewayEvent.quota_bucket` + migration `0004_quota_bucket.sql` + `LedgerStore.count_since(quota_bucket=...)` | Gemini | Per-model limits can only be counted per-model if each event records its bucket. `events.py` already states the rule: a metadata field queried routinely should be promoted to a real column. Nullable; pre-0004 rows are simply unattributable (never match a bucket filter, still counted unfiltered). Executor stamps it on all three paths (success / HTTP error / transport failure). | Revert migration needs a *new* migration (never edit applied ones); until then a nullable column is inert. |
| 13 | `ClassifiedError.resume_at` doc widened: "stated by the provider **or derived from a documented reset rule**"; Gemini's `PerDay` branch now populates it | Gemini | The branch's own message claimed the VERIFIED midnight-Pacific reset but returned `resume_at=None` -- discarding a fact the adapter knows. Still `is_definitively_classified=False`: the *daily* diagnosis is the heuristic, the reset rule is not. | Remove `resume_at=` from the Gemini branch. |
| 14 | OpenAI adapter now parses the three project-scoped headers | OpenAI | Behavior change behind #8. Corrects a docs gap: the official page lists `limit`, `remaining` **and** `reset` (project tokens), not just remaining/reset as `provider-notes.md` said. `raw_headers` still carries them. | See #8. |

Support changes that are not interface changes: `pyproject.toml` gains
`tzdata` (zoneinfo needs a tz database; slim images/Windows ship none) and
`ruff` in dev extras (the lint gate was not reproducible from the repo).

## Considered and rejected in Phase 3 (not forced)

- **Token columns in the ledger** (to count Gemini TPM). Not forced for a *sane* answer: the engine reports `tpm` as UNMONITORED and caps an "open" at LOW confidence rather than guessing. It IS the highest-value follow-up: one migration (`input_tokens`), executor writes `usage.input_tokens`, one `sum_input_tokens_since` ledger method, ~15 lines in the engine. Left for your call because it widens the ledger's public surface.
- **Gemini prices.** `sum_cost_since(provider="gemini")` is structurally 0.0 because no Gemini price entry exists. This is data, not code: add *verified* entries to `pricing.py` and the `spend_10m` window starts being monitored with no engine change. I will not invent prices.
- **In-flight reservation (`reserve()/release()`).** N concurrent callers can see the same headroom. The correct design depends on how Phase 5 schedules calls (per-provider semaphore? token-bucket gate?), and nothing in Phase 3 consumes it. `QuotaAssessment` exposes `assessed_at`/`observed_at` so a router can at least see staleness.
- **Per-dimension reset times in `RateLimitSnapshot`** (the Phase 2 "per-dimension breakdown" bullet). Costs accuracy of `retry_at` only: Anthropic's `reset_at` is the *requests* reset, OpenAI's the *tokens* reset, each used for both dimensions. The engine copes by treating an expired observation as "probably replenished, and the next call corrects us" -- never as proof. Revisit if retry timing proves wrong in Phase 5.
- **A `WORKSPACE` `LimitingUnit`** (Anthropic). Docs say the combined `tokens-*` headers report "the most restrictive limit currently in effect", with a *workspace* cap as the example, and the response does not say which scope it is describing. The adapter labels it ORGANIZATION. Consequence: if a workspace cap binds, `adding_helps(PROJECT)` wrongly says "no". Not fixable from headers; flagged, not hidden.
- **Linear refill interpolation** for token-bucket providers (Anthropic). Needs a per-adapter "refill model". A stale snapshot already *under*-states headroom for a bucket, which is the safe direction.
- **Adaptive limit learning** (infer Gemini's ceiling from the count at which a 429 occurred). Heuristic stacked on a heuristic.
- **Multi-credential per provider.** The ledger has no credential dimension, so two orgs behind one `provider_name` would be summed. `register()` refuses a second adapter under one name instead of silently merging them. Phase 7 (auth/token lifecycle) is where this belongs.
- **Rehydrating exhaustion state after a restart.** State is in memory; a restart re-learns from one failed call per bucket. Making it durable needs `resume_at`/`retry_after` persisted (see finding (a) below) *and* a bucket filter on `query()`.

## Findings in pre-existing code (reported, and fixed only where noted)

| | Finding | Status |
|---|---|---|
| a | `CallExecutor` discarded the parsed snapshot and `ClassifiedError`; only `raw_headers` + message reached the ledger | **Fixed** (#11). Persisting `resume_at`/`retry_after_seconds` in `raw_provider_metadata` is *not* done (nothing consumes it yet). |
| b | Gemini `PerDay` message claimed a VERIFIED reset but `resume_at=None` | **Fixed** (#13) |
| c | `provider-notes.md` / OpenAI docstring listed 2 of the 3 project-scoped headers | **Fixed** (#14, notes updated) |
| d | `SignalQuality.FULL` is documented as "remaining + limit + reset for the binding dimension", but the Anthropic and OpenAI adapters set FULL on four request/token remaining+limit fields without checking `reset_at` | Not fixed. The engine tolerates it (it handles a missing `reset_at` explicitly). |
| e | One `reset_at` is shared by two dimensions (see "Considered and rejected") | Not fixed |
| f | `openai.py:311`: `code in _RATE_LIMIT_CODES or code is not None` -- the first operand is subsumed by the second | Not fixed; harmless, ruff does not flag it |
| g | `ledger/events.py` ends with `from datetime import datetime` + `model_rebuild()` -- a late import that a top-of-file import would make unnecessary | Not fixed; works, ruff's default set does not flag it |
| h | The only test proving a *real* Phase 1 database upgrades (`test_phase1_database_upgrades_cleanly`) is **skipped** without git commit `cb8b49b`, which the shipped tarball lacks; three other tests hard-coded schema version `3` | Version literals replaced by a derived `LATEST`. Added a git-independent upgrade test (v3 -> v4) -- an *imitation* built from our own migration files, so it cannot catch divergence from what Phase 1 really wrote. The skipped test remains the only one that can. |
| i | The "ruff clean" claim was not reproducible from the repo (no ruff in dev extras); `ruff format --check` was already failing on 16 files | `ruff` added to dev extras. Formatting not touched (never claimed, would be a noisy unrelated diff). |
| j | `LedgerStore.count_since` compares ISO *strings*, correct only for same-offset timestamps; the store does not normalize `since` or `timestamp` to UTC | Not fixed. The engine always passes aware UTC (tested, incl. microsecond-zero boundaries); a non-UTC caller would get silently wrong counts. |
| k | The brief expected `count_since`/`sum_cost_since` to back the Gemini fallback. `count_since` does (requests only); `sum_cost_since` returns 0.0 for Gemini (no prices) and nothing counts tokens | Handled by reporting UNMONITORED, not 0 |
| l | Headline "1,325 tests" is 84 distinct test functions; 1,242 items come from `range(400, 600)` x 3 adapters in the contract suite | Reported as-is |
| m | Anthropic workspace-scope caveat (above) | Flagged |
| n | `examples/demo.py` contains six `provider/name == "<vendor>"` branches (adapter + mock-body construction). `src/` is clean; the README's grep claim was scoped to `core/ engines/ ledger/` | Left as fixture wiring; `quota_demo.py` and all engine code have none |

## Self-audit after the first green run (Phase 3 follow-up)

The first delivery passed all tests and lint. A deeper pass (coverage, mypy, a
broader ruff selection, and a seeded random-sequence fuzz of the engine
against its own docstrings) found three defects in the engine and two things
worth recording. None touched the shared interface (`core/`); all are fixed
and regression-tested, and the fuzz test alone catches the second one.

| | Finding | Status |
|---|---|---|
| o | **Clock-skew bug.** A provider-clock `reset_at` that was already in the past on arrival (their clock behind ours) turned `remaining=0` into "assumed replenished" -> **OPEN**: told zero, answered open. | **Fixed**: a reset not strictly after the observation time is untrusted; falls back to the no-reset expiry, with a caveat. |
| p | **`BLOCKED` with `retry_at=None` labelled HEADROOM** when a demand exceeded an *entire* limit (e.g. 500k tokens vs a 100k/min limit, or a limit of 0): "wait" with nothing to wait for. Contradicted the `retry_at` docstring. | **Fixed**: new `BlockReason.EXCEEDS_LIMIT`, `retry_at=None` by design, confidence capped at MEDIUM (that a provider rejects a request larger than its whole limit is UNVERIFIED; the token figure is the caller's estimate). Header and ledger paths both covered. |
| q | mypy: Phase 2 baseline = 3 error lines; first Phase 3 delivery = 25. 22 came from one `**dict` into a dataclass constructor (which hides field-name typos) and one variable reused for two types. | **Fixed** with `dataclasses.replace`; mypy is back to **exactly the 3 pre-existing lines** (`gemini.py` x2, `anthropic.py` x1; untouched, not mine). mypy is not part of the project's gate; this was an audit. |
| r | Gemini model string is used verbatim as the bucket. If Google's aliases share counters with the version they point at (UNVERIFIED), mixing spellings undercounts the ledger. | Documented (adapter comment + `provider-notes.md`): one spelling per model. |
| s | Bare `pytest.raises(ValueError)` in a `QuotaWindow` test (would pass for the wrong error). Three engine branches had no test (limit-less constraint, remaining-less constraint, `NONE`-quality snapshot with numbers). | **Fixed** (`match=` added; branches covered). Engine line coverage 99% before this follow-up. |

Also noted, not built: engine state is keyed per bucket string and never
evicted, so a multi-tenant gateway that lets callers choose arbitrary model
names could grow it without bound. Each entry is small; eviction was left out
as speculative until a real deployment shape is known.

A broader `ruff --select ALL` pass over the new files reported 88 findings.
Reviewed, and deliberately not "fixed": composite asserts (`PT018`), unused
arguments on signatures that must match an ABC or `httpx.MockTransport`
(`ARG`), `(str, Enum)` vs `StrEnum` (the repo's own enums use `(str, Enum)`),
exception-message style (`TRY003`/`EM`). The one real item, `PT011`, is the
test fix in (s).


---

# Interface change log (Phase 3 -> Phase 4: Response Validator)

Same rule: a change is made only if a real, demonstrated need FORCED it.
Phase 4 has ONE additive change to an existing type and ONE new engine
module. No changes to `ProviderAdapter` (no new abstract method), no
changes to the ledger schema (no migration), no changes to `core/types.py`.

| # | Change | Forced by | Why | To revert |
|---|--------|-----------|-----|-----------|
| 15 | `CallResult.bad_patterns: list[KnownBadPatternMatch]` (default `[]`) | Response Validator (same rationale as Phase 3 #11) | `_handle_response()` already computed the `KnownBadPatternMatch` objects from `check_known_bad_patterns()`, then discarded them — only the pattern names (strings) were stored in `raw_provider_metadata`. The validator needs the confidence scores these objects carry; re-invoking `check_known_bad_patterns` on the response would duplicate work and the parsed confidence is unrecoverable from the stored name strings. Identical to Phase 3's reasoning for #11 (`rate_limit`, `classified_error`): "the objects were simply discarded." | Delete the field from `CallResult.__init__` and the `bad_patterns=bad_patterns` kwarg from the `_handle_response` return. The validator would need to re-invoke `check_known_bad_patterns` on `CallResult.response` (viable but wasteful). |

## New module: `gateway.engines.validator`

Not an interface change (no abstract method added to `ProviderAdapter`), but
the Phase 4 engine, logged here for the same reason Phase 2 logged
`CallExecutor`:

- **`ResponseValidator`** — validates a completed `CallResult` against
  operator-configurable policy and returns a `ValidationResult` with:
  verdict (`VALID` / `SUSPECT` / `INVALID`), confidence, typed findings,
  and routing recommendations (`should_retry`, `retry_on_different_provider`,
  `was_billed`). Stateless, synchronous, vendor-agnostic. Reads only
  `CallResult.bad_patterns`, `.classified_error`, `.exception`, and
  `.event.outcome/.cost_usd` — never touches the ledger or adapters directly.

- **`ValidatorConfig`** — operator-tunable policy: `suspect_threshold` (noise
  gate, default 0.5), `reject_threshold` (INVALID threshold, default 0.85),
  `treat_suspected_silent_failure_as_invalid` (default `True`).

Architecture verification: `validator.py` passes the existing
`test_engine_code_names_no_vendor` and `test_engines_never_import_the_adapters_package`
AST-based checks (it was automatically included by `_engine_files()`'s glob),
plus a new structural check (`test_validator_only_imports_core_types_not_adapter_classes`)
verifying it imports only from `gateway.core`, not from `gateway.adapters` or
`gateway.ledger`.

## Considered and rejected in Phase 4 (not forced)

- **Cross-call / stateful detection** (detecting patterns across a sequence of
  calls — e.g., progressively shorter responses, systematic safety blocks for
  a prompt pattern). The data model depends on how Phase 5 structures
  per-provider state; building it before the router exists is speculative.
  `ValidatorConfig` has no cross-call knobs; the interface is designed so
  Phase 5+ can add them without a rewrite.
- **Ledger reads in the validator** (querying recent events for trend
  detection). The validator is deliberately stateless and synchronous; if
  cross-call heuristics are added, they would be fed by Phase 5 (which already
  holds per-provider history), not by ledger queries from the validator.
- **New `ProviderAdapter` abstract method for validation policy.** Not forced:
  all current bad-pattern detection is already expressed through
  `check_known_bad_patterns()`, which returns `KnownBadPatternMatch` objects
  with confidence scores. The validator's job is applying *operator* policy to
  those scores, not asking the adapter for more adapter-side policy.
- **A `ValidationFinding` per rate-limit dimension** (separate findings for
  request-rate vs token-rate constraints). The validator already includes error
  findings from `classified_error`; adding per-dimension breakdown adds
  complexity without a consumer (Phase 5's routing decisions will use the
  Quota Engine's `QuotaAssessment`, not the validator, for rate-limit detail).
- **Wasted-spend tracking in the validator** (maintaining a running total of
  cost_usd on INVALID results). This is Phase 6's Budget Engine's job; the
  validator exposes `was_billed` as a flag for that engine to consume.

