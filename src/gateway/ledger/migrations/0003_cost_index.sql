-- 0003: supports sum_cost_since() (Phase 6 Budget Engine): filter by
-- identity/provider and a time window, then SUM(cost_usd). A composite
-- index lets SQLite answer without scanning the whole table.
CREATE INDEX IF NOT EXISTS idx_identity_ts ON events(identity_key, timestamp);
CREATE INDEX IF NOT EXISTS idx_provider_ts ON events(provider, timestamp);
