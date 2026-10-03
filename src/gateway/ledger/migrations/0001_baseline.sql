-- 0001: the Phase 1 schema, verbatim, made idempotent (IF NOT EXISTS) so an
-- existing UNVERSIONED Phase 1 database is adopted rather than clobbered.
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    identity_key TEXT NOT NULL,
    provider TEXT NOT NULL,
    operation TEXT NOT NULL,
    outcome TEXT NOT NULL,
    error_category TEXT,
    http_status INTEGER,
    latency_ms REAL,
    cost_usd REAL,
    raw_provider_metadata TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_provider ON events(provider);
CREATE INDEX IF NOT EXISTS idx_identity ON events(identity_key);
CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp);
