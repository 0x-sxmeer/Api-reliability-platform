"""
The Ledger Store interface, and a SQLite-backed implementation.

Phase 2 changes vs Phase 1 (each driven by a stated need):
  * Numbered-SQL migrations + a schema_version table, because Phase 1's
    CREATE TABLE IF NOT EXISTS could never evolve. Existing UNVERSIONED
    Phase 1 databases are detected and adopted (see _migrate).
  * All blocking sqlite3 work runs via asyncio.to_thread so it never
    stalls the event loop. The LedgerStore interface is UNCHANGED.
  * provider_request_id column.
  * sum_cost_since(): Phase 6's Budget Engine needs a SQL SUM, not a
    fetch-everything-and-add-in-Python loop.

CONCURRENCY MODEL (a real trade-off; read this):
  One connection per operation, opened inside the worker thread. That is
  safe (sqlite3 connections must not cross threads by default) and
  simple. Under many concurrent writers SQLite serialises them with a
  file lock; we set a busy timeout and WAL mode so writers wait rather
  than fail with "database is locked". This is correct and lossless but
  NOT high-throughput: it is the right Phase 2 choice and the wrong
  final answer, which is exactly why LedgerStore is an interface and
  Postgres (Phase 9+) is the planned replacement.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from uuid import UUID

from gateway.core.types import CallOutcome, ErrorCategory
from gateway.ledger.events import GatewayEvent

_BUSY_TIMEOUT_MS = 30_000


class LedgerStore(ABC):
    """Abstract append-only event store."""

    @abstractmethod
    async def append(self, event: GatewayEvent) -> None:
        """Write one event. Append-only by design: there is deliberately
        no update. Category 7 (downstream silent loss) will need to
        reconcile a PENDING event later; that is an explicit future
        decision, not something to sneak in via a generic update."""
        raise NotImplementedError

    @abstractmethod
    async def reconcile(
        self,
        event_id: UUID,
        final_outcome: CallOutcome,
        reason: str,
        final_cost: float | None = None,
        final_error: ErrorCategory | None = None,
    ) -> None:
        """
        Transition an uncertain event (PENDING, SUSPECTED_SILENT_FAILURE) to a terminal state.
        Preserves original outcome in the audit trail.
        """
        raise NotImplementedError

    @abstractmethod
    async def query(
        self,
        *,
        provider: str | None = None,
        identity_key: str | None = None,
        outcome: CallOutcome | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[GatewayEvent]:
        raise NotImplementedError

    @abstractmethod
    async def count_since(
        self,
        *,
        provider: str | None = None,
        identity_key: str | None = None,
        since: datetime | None = None,
        quota_bucket: str | None = None,
    ) -> int:
        """Number of events matching the filters. Counts EVERY event
        regardless of outcome (a failed attempt may still have consumed
        provider quota), which over-counts: the safe direction for a
        headroom estimate. `since` must be timezone-aware UTC -- the
        SQLite implementation compares ISO strings, which is only correct
        when both sides use the same offset. `quota_bucket` (Phase 3)
        filters to one adapter-defined bucket; rows written before
        migration 0004 have none and never match a bucket filter."""
        raise NotImplementedError

    @abstractmethod
    async def sum_cost_since(
        self,
        *,
        provider: str | None = None,
        identity_key: str | None = None,
        since: datetime | None = None,
    ) -> float:
        """SUM(cost_usd) over matching events. Events whose cost is NULL
        (unknown price, failed call) contribute 0 and are NOT an error:
        callers that must not under-count should also check
        count_unpriced_since()."""
        raise NotImplementedError

    @abstractmethod
    async def count_unpriced_since(
        self,
        *,
        provider: str | None = None,
        identity_key: str | None = None,
        since: datetime | None = None,
    ) -> int:
        """Number of SUCCESS events with cost_usd IS NULL in the window.
        A Budget Engine that enforces spend needs to know how many
        successful calls it could not price, because sum_cost_since()
        silently treats those as $0 (an under-count). Kept separate so
        sum_cost_since() stays a single fast aggregate."""
        raise NotImplementedError


_MIGRATION_RE = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


def _load_migrations() -> list[tuple[int, str, str]]:
    """Return [(version, filename, sql)] sorted by version."""
    out: list[tuple[int, str, str]] = []
    pkg = resources.files("gateway.ledger.migrations")
    for entry in pkg.iterdir():
        m = _MIGRATION_RE.match(entry.name)
        if m:
            out.append((int(m.group(1)), entry.name, entry.read_text(encoding="utf-8")))
    out.sort(key=lambda t: t[0])
    versions = [v for v, _, _ in out]
    if versions != list(range(1, len(versions) + 1)):
        raise RuntimeError(f"Migration versions must be contiguous from 1; found {versions}")
    return out


class SqliteLedgerStore(LedgerStore):
    def __init__(self, db_path: str | Path = "gateway_ledger.db") -> None:
        self.db_path = str(db_path)
        self._migrate()

    # ------------------------------------------------------------ connections

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=_BUSY_TIMEOUT_MS / 1000)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        return conn

    # -------------------------------------------------------------- migrations

    def _migrate(self) -> None:
        """
        Bring the DB to the latest schema version.

        Three starting states, all handled:
          1. Brand new file            -> apply 0001..N.
          2. Versioned (Phase 2+)      -> apply only versions > current.
          3. UNVERSIONED Phase 1 DB    -> has an `events` table but no
             schema_version table. 0001 is written with IF NOT EXISTS so it
             adopts the existing table harmlessly and is then recorded as
             applied; 0002+ then add the new column/indexes.
        Each migration and its version-row insert commit in ONE transaction,
        so a crash cannot leave a half-applied migration. (Needed because
        ALTER TABLE ADD COLUMN is not re-runnable.)
        """
        migrations = _load_migrations()
        conn = self._connect()
        try:
            # WAL lets readers proceed during a write and makes concurrent
            # appends queue instead of erroring. Persisted in the DB file.
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version ("
                "version INTEGER PRIMARY KEY, filename TEXT NOT NULL, applied_at TEXT NOT NULL)"
            )
            conn.commit()
            row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            current = row["v"] or 0
            for version, filename, sql in migrations:
                if version <= current:
                    continue
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    for stmt in _split_sql(sql):
                        conn.execute(stmt)
                    conn.execute(
                        "INSERT INTO schema_version(version, filename, applied_at) "
                        "VALUES (?, ?, datetime('now'))",
                        (version, filename),
                    )
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
        finally:
            conn.close()

    def schema_version(self) -> int:
        conn = self._connect()
        try:
            return conn.execute("SELECT COALESCE(MAX(version),0) AS v FROM schema_version").fetchone()["v"]
        finally:
            conn.close()

    # --------------------------------------------------------------- operations

    async def append(self, event: GatewayEvent) -> None:
        await asyncio.to_thread(self._append_sync, event)

    async def reconcile(
        self,
        event_id: UUID,
        final_outcome: CallOutcome,
        reason: str,
        final_cost: float | None = None,
        final_error: ErrorCategory | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._reconcile_sync, event_id, final_outcome, reason, final_cost, final_error
        )

    def _reconcile_sync(self, event_id: UUID, final_outcome: CallOutcome, reason: str, final_cost: float | None, final_error: ErrorCategory | None) -> None:
        conn = self._connect()
        try:
            # Audit F-20 remediation: BEGIN IMMEDIATE takes the write lock
            # up front. Without it, two concurrent settles of one event
            # could BOTH read 'pending' before either UPDATE committed —
            # last writer silently overwrote the first's settlement.
            conn.execute("BEGIN IMMEDIATE")
            with conn:
                # First fetch the existing row to verify it can be reconciled
                row = conn.execute("SELECT outcome FROM events WHERE event_id = ?", (str(event_id),)).fetchone()
                if not row:
                    raise ValueError(f"Event {event_id} not found")
                
                old_outcome = row["outcome"]
                if old_outcome not in (CallOutcome.PENDING.value, CallOutcome.SUSPECTED_SILENT_FAILURE.value):
                    raise ValueError(f"Event {event_id} has outcome {old_outcome} which cannot be reconciled.")
                    
                now = datetime.now(UTC)
                conn.execute(
                    """
                    UPDATE events 
                    SET outcome = ?,
                        cost_usd = ?,
                        error_category = ?,
                        original_outcome = ?,
                        reconciled_at = ?,
                        reconciled_reason = ?
                    WHERE event_id = ?
                    """,
                    (
                        final_outcome.value,
                        final_cost,
                        final_error.value if final_error else None,
                        old_outcome,
                        now.isoformat(),
                        reason,
                        str(event_id),
                    ),
                )
        finally:
            conn.close()

    def _append_sync(self, event: GatewayEvent) -> None:
        conn = self._connect()
        try:
            with conn:  # commit/rollback
                conn.execute(
                    """
                    INSERT INTO events (
                        event_id, timestamp, identity_key, provider, operation,
                        outcome, error_category, http_status, provider_request_id,
                        latency_ms, cost_usd, input_tokens, output_tokens, raw_provider_metadata, quota_bucket,
                        reconciled_at, reconciled_reason, original_outcome
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(event.event_id),
                        event.timestamp.isoformat(),
                        event.identity_key,
                        event.provider,
                        event.operation,
                        event.outcome.value,
                        event.error_category.value if event.error_category else None,
                        event.http_status,
                        event.provider_request_id,
                        event.latency_ms,
                        event.cost_usd,
                        event.input_tokens,
                        event.output_tokens,
                        json.dumps(event.raw_provider_metadata),
                        event.quota_bucket,
                        event.reconciled_at.isoformat() if event.reconciled_at else None,
                        event.reconciled_reason,
                        event.original_outcome.value if event.original_outcome else None,
                    ),
                )
        finally:
            conn.close()

    async def query(
        self,
        *,
        provider: str | None = None,
        identity_key: str | None = None,
        outcome: CallOutcome | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[GatewayEvent]:
        return await asyncio.to_thread(
            self._query_sync, provider, identity_key, outcome, since, limit
        )

    def _query_sync(self, provider, identity_key, outcome, since, limit) -> list[GatewayEvent]:
        clauses, params = self._build_filters(provider, identity_key, outcome, since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM events {where} ORDER BY timestamp DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_event(r) for r in rows]

    async def count_since(
        self, *, provider=None, identity_key=None, since=None, quota_bucket=None
    ) -> int:
        return await asyncio.to_thread(
            self._scalar_sync, "COUNT(*)", None, provider, identity_key, since, quota_bucket
        )

    async def sum_cost_since(self, *, provider=None, identity_key=None, since=None) -> float:
        v = await asyncio.to_thread(
            self._scalar_sync, "COALESCE(SUM(cost_usd), 0.0)", None, provider, identity_key, since
        )
        return float(v)

    async def count_unpriced_since(self, *, provider=None, identity_key=None, since=None) -> int:
        return int(
            await asyncio.to_thread(
                self._scalar_sync,
                "COUNT(*)",
                "outcome = 'success' AND cost_usd IS NULL",
                provider,
                identity_key,
                since,
            )
        )

    def _scalar_sync(self, expr, extra_clause, provider, identity_key, since, quota_bucket=None):
        clauses, params = self._build_filters(provider, identity_key, None, since, quota_bucket)
        if extra_clause:
            clauses.append(extra_clause)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        conn = self._connect()
        try:
            return conn.execute(f"SELECT {expr} AS v FROM events {where}", params).fetchone()["v"]
        finally:
            conn.close()

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _build_filters(
        provider, identity_key, outcome, since, quota_bucket=None
    ) -> tuple[list[str], list]:
        clauses: list[str] = []
        params: list = []
        if provider is not None:
            clauses.append("provider = ?")
            params.append(provider)
        if identity_key is not None:
            clauses.append("identity_key = ?")
            params.append(identity_key)
        if outcome is not None:
            clauses.append("outcome = ?")
            params.append(outcome.value)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())
        if quota_bucket is not None:
            clauses.append("quota_bucket = ?")
            params.append(quota_bucket)
        return clauses, params

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> GatewayEvent:
        return GatewayEvent(
            event_id=UUID(row["event_id"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            identity_key=row["identity_key"],
            provider=row["provider"],
            operation=row["operation"],
            outcome=CallOutcome(row["outcome"]),
            error_category=ErrorCategory(row["error_category"]) if row["error_category"] else None,
            http_status=row["http_status"],
            provider_request_id=row["provider_request_id"],
            latency_ms=row["latency_ms"],
            cost_usd=row["cost_usd"],
            input_tokens=row["input_tokens"] if "input_tokens" in row.keys() else None,
            output_tokens=row["output_tokens"] if "output_tokens" in row.keys() else None,
            raw_provider_metadata=json.loads(row["raw_provider_metadata"]),
            quota_bucket=row["quota_bucket"],
            # NOTE: sqlite3.Row does NOT support .get(); use index access plus
            # truthiness checks. Do not "modernize" these to row.get(...).
            reconciled_at=datetime.fromisoformat(row["reconciled_at"]) if row["reconciled_at"] else None,
            reconciled_reason=row["reconciled_reason"],
            original_outcome=CallOutcome(row["original_outcome"]) if row["original_outcome"] else None,
        )


def _split_sql(sql: str) -> list[str]:
    """Split a migration file into statements. Our migrations are plain
    DDL with -- comments and no triggers/strings containing ';', so a
    simple splitter is sufficient and honest. Revisit if a migration ever
    needs a trigger body (use sqlite3.complete_statement then)."""
    stmts: list[str] = []
    buf: list[str] = []
    for line in sql.splitlines():
        if line.strip().startswith("--"):
            continue
        buf.append(line)
        joined = "\n".join(buf)
        if sqlite3.complete_statement(joined):
            stmts.append(joined.strip())
            buf = []
    tail = "\n".join(buf).strip()
    if tail:
        stmts.append(tail)
    return [s for s in stmts if s]
