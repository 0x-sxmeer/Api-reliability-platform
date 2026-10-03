-- 0002: provider_request_id (nullable). SQLite ALTER TABLE ADD COLUMN is
-- cheap and does not rewrite existing rows; old rows read back as NULL.
ALTER TABLE events ADD COLUMN provider_request_id TEXT;
CREATE INDEX IF NOT EXISTS idx_request_id ON events(provider_request_id);
