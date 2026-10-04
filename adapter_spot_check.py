import httpx

from gateway.adapters.openai import (
    _QUOTA_EXHAUSTED_CODES,
    _RATE_LIMIT_CODES,
    OpenAIAdapter,
    _parse_openai_duration,
)

tests = [("6m0s", 360.0), ("59.5s", 59.5), ("1h2m3s", 3723.0), ("1s", 1.0)]
for s, expected in tests:
    got = _parse_openai_duration(s)
    ok = abs(got - expected) < 0.001 if got is not None else False
    print(f"  {s!r} -> {got} (expected {expected}) {'OK' if ok else 'FAIL'}")

print("_QUOTA_EXHAUSTED_CODES:", _QUOTA_EXHAUSTED_CODES)
print("_RATE_LIMIT_CODES:", _RATE_LIMIT_CODES)

adapter = OpenAIAdapter(api_key="test")

r1 = httpx.Response(429, json={"error": {"code": "slow_down", "message": "slow"}})
c1 = adapter.classify_error(r1)
print(
    "429 slow_down ->", c1.category.value, "PASS:", c1.category.value == "rate_limited"
)

r2 = httpx.Response(503, json={"error": {"type": "service_unavailable_error"}})
c2 = adapter.classify_error(r2)
print("503 ->", c2.category.value, "PASS:", c2.category.value == "provider_overloaded")

r3 = httpx.Response(
    429, json={"error": {"code": "insufficient_quota", "message": "quota"}}
)
c3 = adapter.classify_error(r3)
print(
    "insufficient_quota ->",
    c3.category.value,
    "PASS:",
    c3.category.value == "quota_exhausted",
)

# Anthropic spot checks
from gateway.adapters.anthropic import AnthropicAdapter

aa = AnthropicAdapter(api_key="test")

r4 = httpx.Response(
    529, json={"error": {"type": "overloaded_error", "message": "overloaded"}}
)
c4 = aa.classify_error(r4)
print(
    "anthropic 529 ->",
    c4.category.value,
    "PASS:",
    c4.category.value == "provider_overloaded",
)

# Gemini spot checks
from gateway.adapters.gemini import GeminiAdapter

ga = GeminiAdapter(api_key="test")
windows = ga.quota_windows()
print("gemini quota_windows:", len(windows), "entries, PASS:", len(windows) > 0)

r5 = httpx.Response(
    429, json={"error": {"status": "RESOURCE_EXHAUSTED", "message": "rate limit"}}
)
c5 = ga.classify_error(r5)
print("gemini 429 RESOURCE_EXHAUSTED ->", c5.category.value)
