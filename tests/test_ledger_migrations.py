"""
Migration tests. The key one builds a REAL Phase 1 database by executing the
actual Phase 1 store code from git (baseline commit cb8b49b), not a
hand-written imitation, so it catches divergence between what Phase 1
really wrote and what our migration assumes.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from gateway.core.types import CallOutcome
from gateway.ledger.events import GatewayEvent
from gateway.ledger.store import SqliteLedgerStore, _load_migrations

REPO = Path(__file__).resolve().parent.parent
PHASE1_COMMIT = "cb8b49b"

# Derived, not hard-coded: these tests assert "reaches the latest version",
# and a literal broke every time a migration was added (0004 did).
LATEST = _load_migrations()[-1][0]


def _build_phase1_db(path: Path, n_rows: int = 3) -> None:
    """Create a DB exactly as Phase 1 would, by running the Phase 1 store."""
    src = subprocess.run(
        ["git", "show", f"{PHASE1_COMMIT}:src/gateway/ledger/store.py"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    if src.returncode != 0:
        pytest.skip(f"Phase 1 baseline commit {PHASE1_COMMIT} not available in this checkout")
    mod_path = path.parent / "phase1_store.py"
    mod_path.write_text(src.stdout)
    spec = importlib.util.spec_from_file_location("phase1_store", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase1_store"] = mod
    spec.loader.exec_module(mod)

    # Phase 1's GatewayEvent had no provider_request_id; build with the
    # shared event class but only Phase 1 fields, then let the OLD store write.
    store = mod.SqliteLedgerStore(path)
    from gateway.ledger.events import GatewayEvent as Ev
    class _P1Event:  # duck-typed: exactly the attributes Phase 1's append() read
        pass
    async def go():
        for i in range(n_rows):
            e = Ev(identity_key="team-old", provider="anthropic", operation="messages.create",
                   outcome=CallOutcome.SUCCESS, http_status=200, latency_ms=10.0 + i, cost_usd=0.001)
            await store.append(e)
    asyncio.run(go())


def test_phase1_database_upgrades_cleanly(tmp_path: Path) -> None:
    db = tmp_path / "phase1.db"
    _build_phase1_db(db, n_rows=3)

    # Sanity: it really is an unversioned Phase 1 DB.
    raw = sqlite3.connect(db)
    tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    cols = {r[1] for r in raw.execute("PRAGMA table_info(events)")}
    raw.close()
    assert "events" in tables and "schema_version" not in tables
    assert "provider_request_id" not in cols

    store = SqliteLedgerStore(db)  # <- the upgrade

    assert store.schema_version() == LATEST
    rows = asyncio.run(store.query(identity_key="team-old"))
    assert len(rows) == 3, "existing Phase 1 rows must survive the upgrade"
    assert all(r.provider_request_id is None for r in rows), "old rows read back NULL"

    # New writes work, including the new column, against the upgraded DB.
    ev = GatewayEvent(identity_key="team-old", provider="anthropic", operation="messages.create",
                      outcome=CallOutcome.SUCCESS, provider_request_id="req_123", cost_usd=0.5)
    asyncio.run(store.append(ev))
    got = asyncio.run(store.query(identity_key="team-old", limit=10))
    assert len(got) == 4
    assert any(r.provider_request_id == "req_123" for r in got)


def test_fresh_database_reaches_latest_version(tmp_path: Path) -> None:
    store = SqliteLedgerStore(tmp_path / "fresh.db")
    assert store.schema_version() == LATEST


def test_migrating_twice_is_a_noop(tmp_path: Path) -> None:
    """ALTER TABLE ADD COLUMN is not re-runnable. Re-opening must not re-apply."""
    p = tmp_path / "twice.db"
    SqliteLedgerStore(p)
    SqliteLedgerStore(p)  # would raise 'duplicate column name' if 0002 re-ran
    assert SqliteLedgerStore(p).schema_version() == LATEST


def test_failed_migration_rolls_back_and_is_not_recorded(tmp_path: Path, monkeypatch) -> None:
    """A migration that errors midway must leave neither partial schema nor a version row."""
    import gateway.ledger.store as store_mod
    real = store_mod._load_migrations()
    bad_version = LATEST + 1
    bad = real + [
        (bad_version, f"{bad_version:04d}_bad.sql", "ALTER TABLE events ADD COLUMN ok_col TEXT;\nTHIS IS NOT SQL;")
    ]
    monkeypatch.setattr(store_mod, "_load_migrations", lambda: bad)
    p = tmp_path / "bad.db"
    with pytest.raises(sqlite3.Error):
        SqliteLedgerStore(p)
    monkeypatch.undo()
    raw = sqlite3.connect(p)
    cols = {r[1] for r in raw.execute("PRAGMA table_info(events)")}
    versions = [r[0] for r in raw.execute("SELECT version FROM schema_version")]
    raw.close()
    assert "ok_col" not in cols, "partial migration must be rolled back"
    assert bad_version not in versions


def _build_db_at_version(path: Path, upto: int) -> None:
    """
    Build a database at schema version `upto` by applying the real numbered
    migrations 1..upto by hand.

    HONEST LIMITATION: unlike _build_phase1_db this is an imitation built
    from our own migration files, so it cannot catch divergence between
    what Phase 1 really wrote and what 0001 assumes. The git-backed test
    above is the one that can -- but it is skipped in any checkout without
    the Phase 1 commit (including the shipped tarball), so this one at
    least proves the upgrade mechanics everywhere.
    """
    import gateway.ledger.store as store_mod

    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, filename TEXT NOT NULL, applied_at TEXT NOT NULL)"
    )
    for version, filename, sql in store_mod._load_migrations():
        if version > upto:
            break
        for stmt in store_mod._split_sql(sql):
            conn.execute(stmt)
        conn.execute("INSERT INTO schema_version VALUES (?, ?, datetime('now'))", (version, filename))
    conn.commit()
    conn.close()


def test_pre_0004_database_upgrades_and_old_rows_have_no_bucket(tmp_path: Path) -> None:
    db = tmp_path / "v3.db"
    _build_db_at_version(db, upto=3)
    raw = sqlite3.connect(db)
    cols = {r[1] for r in raw.execute("PRAGMA table_info(events)")}
    assert "quota_bucket" not in cols, "sanity: this really is a pre-0004 schema"
    raw.execute(
        "INSERT INTO events (event_id, timestamp, identity_key, provider, operation, outcome, "
        "raw_provider_metadata) VALUES ('11111111-1111-1111-1111-111111111111', "
        "'2026-01-01T00:00:00+00:00', 'team-old', 'anthropic', 'messages.create', 'success', '{}')"
    )
    raw.commit()
    raw.close()

    store = SqliteLedgerStore(db)  # <- the upgrade

    assert store.schema_version() == LATEST
    old_rows = asyncio.run(store.query(identity_key="team-old"))
    assert len(old_rows) == 1 and old_rows[0].quota_bucket is None

    ev = GatewayEvent(identity_key="team-old", provider="anthropic", operation="messages.create",
                      outcome=CallOutcome.SUCCESS, quota_bucket="claude-x")
    asyncio.run(store.append(ev))
    # A bucket filter matches only the attributable row; the unfiltered
    # count still sees both (old rows are not lost, just unattributable).
    assert asyncio.run(store.count_since(provider="anthropic", quota_bucket="claude-x")) == 1
    assert asyncio.run(store.count_since(provider="anthropic")) == 2
