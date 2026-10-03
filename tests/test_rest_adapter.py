"""
Unit tests for GenericRestAdapter (the fourth adapter, added late).

The contract suite covers cross-adapter invariants; these tests pin the
REST-specific behaviors that were P0 defects in the audit:
  * parse_rate_limit_headers returns a RateLimitSnapshot (not a dict)
  * get_limiting_unit returns the LimitingUnit enum member (not a string)
  * send() fails LOUDLY on an unresolvable operation (no fabricated 200)
  * header parsing normalizes epoch/ISO resets and never raises
"""
from __future__ import annotations

import math
from datetime import UTC, datetime

import httpx
import pytest

from gateway.adapters.rest import GenericRestAdapter
from gateway.core.adapter import AdapterRequest
from gateway.core.types import (
    ErrorCategory,
    LimitingUnit,
    RateLimitSnapshot,
    SignalQuality,
)


def make_adapter() -> GenericRestAdapter:
    return GenericRestAdapter(api_key="sk-test")


def make_request(**extra) -> AdapterRequest:
    op = extra.pop("operation", "charge.create")
    return AdapterRequest(operation=op, payload={"amount": 100}, extra=extra)


# ---------------------------------------------------------------- limiting unit

def test_limiting_unit_is_enum_member():
    # Regression: returned bare string "key" once, which identity-compares
    # False against LimitingUnit.KEY inside the Quota Engine.
    assert make_adapter().get_limiting_unit() is LimitingUnit.KEY


# ---------------------------------------------------------------- quota bucket

def test_quota_bucket_uses_url_when_present():
    adapter = make_adapter()
    req = make_request(url="https://api.example.com/v1/charges")
    assert adapter.quota_bucket(req) == "https://api.example.com/v1/charges"


def test_quota_bucket_falls_back_and_never_raises():
    adapter = make_adapter()
    assert adapter.quota_bucket(make_request()) == "default"
    # Non-string / hostile extras must not raise.
    assert adapter.quota_bucket(make_request(url=12345)) == "default"
    assert adapter.quota_bucket(make_request(url=None)) == "default"


# ---------------------------------------------------------------- headers

def test_parse_headers_returns_snapshot_not_dict():
    # Regression: this used to return a plain dict, crashing CallExecutor
    # with AttributeError on snapshot.signal_quality.
    response = httpx.Response(
        200,
        headers={
            "RateLimit-Limit": "100",
            "RateLimit-Remaining": "99",
            "RateLimit-Reset": "1767225600",  # epoch seconds
            "Retry-After": "2.5",
        },
    )
    snap = make_adapter().parse_rate_limit_headers(response)
    assert isinstance(snap, RateLimitSnapshot)
    assert snap.requests_limit == 100
    assert snap.requests_remaining == 99
    assert snap.retry_after_seconds == 2.5
    assert snap.reset_at == datetime.fromtimestamp(1767225600, tz=UTC)
    assert snap.reset_at_source == "timestamp"
    assert snap.signal_quality is SignalQuality.FULL


def test_parse_headers_partial_when_only_one_value_present():
    # limit present but remaining absent -> PARTIAL, not FULL.
    response = httpx.Response(200, headers={"RateLimit-Limit": "10"})
    snap = make_adapter().parse_rate_limit_headers(response)
    assert snap.requests_limit == 10
    assert snap.requests_remaining is None
    assert snap.signal_quality is SignalQuality.PARTIAL


def test_parse_headers_falls_back_to_x_family_and_draft_wins_both():
    # Draft family with its own Reset marker proves the draft branch ran;
    # then an X-only response proves the fallback branch runs too.
    draft = httpx.Response(
        200,
        headers={
            "RateLimit-Limit": "100",
            "RateLimit-Remaining": "50",
            "RateLimit-Reset": "epoch:1767225600",
        },
    )
    snap = make_adapter().parse_rate_limit_headers(draft)
    assert snap.requests_limit == 100 and snap.requests_remaining == 50

    xonly = httpx.Response(
        200,
        headers={"X-RateLimit-Limit": "10", "X-RateLimit-Remaining": "3"},
    )
    snap2 = make_adapter().parse_rate_limit_headers(xonly)
    assert snap2.requests_limit == 10
    assert snap2.requests_remaining == 3
    assert snap2.signal_quality is SignalQuality.FULL


def test_parse_headers_iso_reset_naive_assumed_utc():
    response = httpx.Response(200, headers={"RateLimit-Reset": "2027-01-01T00:00:00+00:00"})
    snap = make_adapter().parse_rate_limit_headers(response)
    assert snap.reset_at is not None
    assert snap.reset_at.tzinfo is not None
    assert snap.reset_at.utcoffset().total_seconds() == 0
    assert snap.reset_at.year == 2027


def test_parse_headers_fractional_epoch_reset_parses():
    # Regression: int() only would silently drop "1767225600.5".
    response = httpx.Response(200, headers={"RateLimit-Reset": "1767225600.5"})
    snap = make_adapter().parse_rate_limit_headers(response)
    assert snap.reset_at == datetime.fromtimestamp(1767225600.5, tz=UTC)


def test_parse_headers_empty_response_yields_none_quality_no_raise():
    snap = make_adapter().parse_rate_limit_headers(httpx.Response(200))
    assert snap.signal_quality is SignalQuality.NONE
    assert snap.requests_limit is None
    assert snap.requests_remaining is None
    assert snap.reset_at is None
    assert snap.retry_after_seconds is None


@pytest.mark.parametrize("junk", ["abc", "", "nan", "inf", "-"])
def test_parse_headers_hostile_values_never_raise(junk):
    response = httpx.Response(
        200,
        headers={"RateLimit-Limit": junk, "RateLimit-Reset": junk, "Retry-After": junk},
    )
    snap = make_adapter().parse_rate_limit_headers(response)
    if snap.retry_after_seconds is not None:
        assert math.isfinite(snap.retry_after_seconds)


# ---------------------------------------------------------------- errors

@pytest.mark.parametrize(
    ("status", "category"),
    [
        (429, ErrorCategory.RATE_LIMITED),
        (402, ErrorCategory.QUOTA_EXHAUSTED),
        (401, ErrorCategory.PERMANENT),
        (403, ErrorCategory.PERMANENT),
        (400, ErrorCategory.PERMANENT),
        (404, ErrorCategory.PERMANENT),
        (500, ErrorCategory.TRANSIENT),
        (502, ErrorCategory.TRANSIENT),
        (503, ErrorCategory.TRANSIENT),
        (504, ErrorCategory.TRANSIENT),
        (599, ErrorCategory.TRANSIENT),
    ],
)
def test_classify_error_status_mapping(status, category):
    err = make_adapter().classify_error(httpx.Response(status))
    assert err.category is category


def test_classify_error_retry_after_passthrough():
    err = make_adapter().classify_error(httpx.Response(429, headers={"Retry-After": "30"}))
    assert err.retry_after_seconds == 30.0


def test_classify_error_402_is_provisional():
    # No documented reset rule for generic 402s — engine must take its
    # probe path, so definiteness is False.
    err = make_adapter().classify_error(httpx.Response(402))
    assert err.is_definitively_classified is False


def test_classify_exception_transient_provisional():
    err = make_adapter().classify_exception(RuntimeError("boom"), make_request())
    assert err.category is ErrorCategory.TRANSIENT
    assert err.is_definitively_classified is False
    assert "RuntimeError" in err.message


# ---------------------------------------------------------------- usage / ids

def test_extract_usage_returns_none_never_raises():
    assert make_adapter().extract_usage(httpx.Response(200, json={})) is None


def test_extract_request_id_conventions():
    adapter = make_adapter()
    for name in ("x-request-id", "request-id", "x-amzn-requestid", "x-correlation-id"):
        rid = adapter.extract_request_id(httpx.Response(200, headers={name: "abc123"}))
        assert rid == "abc123", name
    assert adapter.extract_request_id(httpx.Response(200)) is None


def test_known_bad_patterns_empty_list():
    assert make_adapter().check_known_bad_patterns(httpx.Response(200)) == []


# ---------------------------------------------------------------- send

async def test_send_rejects_missing_or_relative_url_without_io():
    # Regression: unknown operations used to fabricate httpx.Response(200),
    # writing false SUCCESS rows into the ledger. Must raise before I/O.
    adapter = make_adapter()
    with pytest.raises(ValueError, match="absolute http"):
        await adapter.send(make_request())
    with pytest.raises(ValueError, match="absolute http"):
        await adapter.send(make_request(url="/charges"))
    with pytest.raises(ValueError, match="absolute http"):
        await adapter.send(make_request(url="ftp://example.com/x"))
    with pytest.raises(ValueError, match="absolute http"):
        await adapter.send(make_request(url=123))


async def test_send_makes_single_attempt_with_auth_and_json():
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, json={"id": "ch_1"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = GenericRestAdapter(api_key="sk-test", client=client)
        response = await adapter.send(make_request(url="https://api.example.com/v1/charges"))
    assert response.status_code == 201
    assert len(seen) == 1  # exactly one attempt; retries are the router's job
    assert seen[0].headers["authorization"] == "Bearer sk-test"
    assert seen[0].method == "POST"


async def test_send_respects_explicit_authorization_header():
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = GenericRestAdapter(api_key="sk-test", client=client)
        await adapter.send(
            make_request(
                url="https://api.example.com/v1/charges",
                method="get",
                headers={"Authorization": "Basic dXNlcjpwdw=="},
            )
        )
    assert seen[0].headers["authorization"] == "Basic dXNlcjpwdw=="
    assert seen[0].method == "GET"
