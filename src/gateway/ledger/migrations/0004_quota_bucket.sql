-- 0004: quota_bucket (nullable). Which rate-limit counters the call was
-- metered against, as the adapter's opaque ProviderAdapter.quota_bucket()
-- string. Needed so the Quota Engine can count per-bucket (Gemini limits
-- vary by model) from the ledger. Pre-0004 rows read back NULL and are
-- simply not attributable to a bucket. The composite index serves the
-- engine's "count events for (provider, bucket) since T" query.
ALTER TABLE events ADD COLUMN quota_bucket TEXT;
CREATE INDEX IF NOT EXISTS idx_provider_bucket_ts ON events(provider, quota_bucket, timestamp);
