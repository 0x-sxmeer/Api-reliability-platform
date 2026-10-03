-- Phase 7: Reconciliation
-- Adding columns to track post-facto reconciliation of PENDING/SUSPECTED events.

ALTER TABLE events ADD COLUMN reconciled_at TIMESTAMP NULL;
ALTER TABLE events ADD COLUMN reconciled_reason TEXT NULL;
ALTER TABLE events ADD COLUMN original_outcome TEXT NULL;
