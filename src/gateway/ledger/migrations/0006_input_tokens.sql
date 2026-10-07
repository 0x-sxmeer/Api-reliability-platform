-- 0006: add input and output tokens for better usage tracking
ALTER TABLE events ADD COLUMN input_tokens INTEGER;
ALTER TABLE events ADD COLUMN output_tokens INTEGER;
