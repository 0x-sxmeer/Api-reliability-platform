-- Add input_tokens and output_tokens to track usage for providers like Gemini that don't publish spend directly

ALTER TABLE events ADD COLUMN input_tokens INTEGER;
ALTER TABLE events ADD COLUMN output_tokens INTEGER;
