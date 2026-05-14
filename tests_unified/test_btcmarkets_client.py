"""
Tests for BTC Markets Exchange Client and WebSocket Feed.

Coverage:
1.  test_btcm_constants           — module-level fee / URL constants
2.  test_btcm_maker_fee_is_negative — BTCM_MAKER_FEE is strictly negative (rebate)
3.  test_btcm_symbol_normalisation  — various input forms → "BTC-AUD"
4.  test_btcm_from_symbol           — "BTC-AUD" → "BTC/AUD"
5.  test_btcm_sign                  — signature is deterministic base-64 string
6.  test_btcm_post_only_flag        — create_order body includes postOnly=True
7.  test_btcm_fetch_ticker_mock     — mock HTTP response, bid/ask normalised
8.  test_btcm_fetch_order_book_mock — bids sorted descending
9.  test_btcm_rate_limiter          — token-bucket respects 45 req/min ceiling
10. test_btcm_rebate_tracking       — record fill → get_rebate_earned > 0
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import sys
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from exchanges.btcmarkets_client import (
    BTCM_BASE_URL,
    BTCM_MAKER_FEE,
    BTCM_MARKETS,
    BTCM_TAKER_FEE,
    BTCM_WS_URL,
    BTCMarketsAPIError,
    BTCMarketsAuthError,
    BTCMarketsClient,
    _BTCMRateLimiter,
    from_btcm_symbol,
    get_exchange_info,
    to_btcm_symbol,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_client(
    api_key: str = "testkey",
    api_secret: str = "",
) -> BTCMarketsClient:
    """Create a BTCMarketsClient with a known (base-64-safe) secret."""
    # Use a stable base-64-encoded secret for deterministic signature tests
    if not api_secret:
        api_secret = base64.b64encode(b"supersecret123").decode()
    return BTCMarketsClient(api_key=api_key, api_secret=api_secret)


# ===========================================================================
# 1. Constants
# ===========================================================================

class TestBTCMConstants:
    """Test 1: Module-level constants have correct values."""

    def test_btcm_constants(self) -> None:
        assert BTCM_MAKER_FEE == -0.0005, "Maker fee must be -0.0005 (-0.05%)"
        assert BTCM_TAKER_FEE == 0.002,  "Taker fee must be 0.002 (0.20%)"
        assert BTCM_BASE_URL  == "https://api.btcmarkets.net/v3"
        assert BTCM_WS_URL    == "wss://socket.btcmarkets.net/v2"
        assert isinstance(BTCM_MARKETS, list)
        assert len(BTCM_MARKETS) >= 6
        assert "BTC-AUD" in BTCM_MARKETS
        assert "ETH-AUD" in BTCM_MARKETS

    def test_btcm_get_exchange_info(self) -> None:
        info = get_exchange_info()
        assert info["maker_fee"] == BTCM_MAKER_FEE
        assert info["taker_fee"] == BTCM_TAKER_FEE
        assert "BTC-AUD" in info["markets"]


# ===========================================================================
# 2. Maker fee is negative
# ===========================================================================

class TestBTCMMakerFeeIsNegative:
    """Test 2: Maker fee must be negative (rebate scenario)."""

    def test_btcm_maker_fee_is_negative(self) -> None:
        assert BTCM_MAKER_FEE < 0, (
            f"BTCM_MAKER_FEE={BTCM_MAKER_FEE} should be negative — "
            "BTC Markets pays the maker a rebate"
        )

    def test_btcm_rebate_is_income(self) -> None:
        """Maker fill of $1000 AUD notional should earn a positive rebate."""
        rebate_per_unit = abs(BTCM_MAKER_FEE)  # 0.0005
        notional = 1000.0
        rebate = rebate_per_unit * notional
        assert rebate == pytest.approx(0.50, rel=1e-6), (
            "Expected $0.50 AUD rebate on $1000 maker trade"
        )


# ===========================================================================
# 3. Symbol normalisation → BTC Markets format
# ===========================================================================

class TestBTCMSymbolNormalisation:
    """Test 3: to_btcm_symbol converts various input forms correctly."""

    def test_btcm_symbol_normalisation(self) -> None:
        assert to_btcm_symbol("BTC/AUD")  == "BTC-AUD"
        assert to_btcm_symbol("ETH/AUD")  == "ETH-AUD"
        assert to_btcm_symbol("SOL/AUD")  == "SOL-AUD"
        assert to_btcm_symbol("XRP/AUD")  == "XRP-AUD"
        assert to_btcm_symbol("USDT/AUD") == "USDT-AUD"

    def test_to_btcm_from_concatenated(self) -> None:
        assert to_btcm_symbol("BTCAUD")  == "BTC-AUD"
        assert to_btcm_symbol("ETHAUD")  == "ETH-AUD"

    def test_to_btcm_noop_if_already_correct(self) -> None:
        assert to_btcm_symbol("BTC-AUD") == "BTC-AUD"
        assert to_btcm_symbol("ETH-AUD") == "ETH-AUD"

    def test_to_btcm_lowercased_input(self) -> None:
        assert to_btcm_symbol("btc/aud") == "BTC-AUD"
        assert to_btcm_symbol("eth/aud") == "ETH-AUD"


# ===========================================================================
# 4. from_btcm_symbol
# ===========================================================================

class TestBTCMFromSymbol:
    """Test 4: from_btcm_symbol converts dash-separated back to slash-separated."""

    def test_btcm_from_symbol(self) -> None:
        assert from_btcm_symbol("BTC-AUD")  == "BTC/AUD"
        assert from_btcm_symbol("ETH-AUD")  == "ETH/AUD"
        assert from_btcm_symbol("SOL-AUD")  == "SOL/AUD"
        assert from_btcm_symbol("XRP-AUD")  == "XRP/AUD"
        assert from_btcm_symbol("USDT-AUD") == "USDT/AUD"

    def test_roundtrip_symbol(self) -> None:
        for sym in ("BTC/AUD", "ETH/AUD", "SOL/AUD", "DOGE/AUD"):
            assert from_btcm_symbol(to_btcm_symbol(sym)) == sym


# ===========================================================================
# 5. Signature generation
# ===========================================================================

class TestBTCMSign:
    """Test 5: _sign produces a deterministic base-64 HMAC-SHA256 signature."""

    def test_btcm_sign(self) -> None:
        secret_raw = b"test_secret_key"
        api_secret = base64.b64encode(secret_raw).decode()
        client = BTCMarketsClient(api_key="key", api_secret=api_secret)

        method    = "GET"
        path      = "/v3/markets/BTC-AUD/ticker"
        timestamp = "1700000000000"
        body      = ""

        sig = client._sign(method, path, timestamp, body)

        # Must be a non-empty base-64 string
        assert isinstance(sig, str)
        assert len(sig) > 0
        decoded = base64.b64decode(sig)
        assert len(decoded) == 32  # SHA-256 = 32 bytes

        # Must be deterministic
        sig2 = client._sign(method, path, timestamp, body)
        assert sig == sig2

    def test_btcm_sign_matches_manual(self) -> None:
        """Cross-check against a manually computed HMAC."""
        secret_raw = b"known_secret"
        api_secret = base64.b64encode(secret_raw).decode()
        client = BTCMarketsClient(api_key="k", api_secret=api_secret)

        message = "GET/v3/orders1700000000000"
        expected_bytes = hmac.new(
            secret_raw,
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        expected_sig = base64.b64encode(expected_bytes).decode()

        sig = client._sign("GET", "/v3/orders", "1700000000000", "")
        assert sig == expected_sig

    def test_btcm_sign_with_body(self) -> None:
        """Signature changes when body is non-empty."""
        secret_raw = b"body_test"
        api_secret = base64.b64encode(secret_raw).decode()
        client = BTCMarketsClient(api_key="k", api_secret=api_secret)

        sig_no_body   = client._sign("POST", "/v3/orders", "1700000000000", "")
        sig_with_body = client._sign("POST", "/v3/orders", "1700000000000", '{"marketId":"BTC-AUD"}')
        assert sig_no_body != sig_with_body


# ===========================================================================
# 6. Post-only flag in create_order
# ===========================================================================

class TestBTCMPostOnlyFlag:
    """Test 6: create_order sends postOnly=True for limit orders by default."""

    @pytest.mark.asyncio
    async def test_btcm_post_only_flag(self) -> None:
        """create_order should include postOnly=True in the request body."""
        client = _make_client()
        captured_body: Dict[str, Any] = {}

        async def fake_request(
            method: str, path: str, *, params=None, body=None, authenticated=False
        ) -> Dict[str, Any]:
            if method == "POST" and path == "/v3/orders":
                captured_body.update(body or {})
                return {
                    "orderId":      "order-001",
                    "status":       "Placed",
                    "creationTime": "2024-01-01T00:00:00.000000Z",
                }
            return {}

        client._request = fake_request  # type: ignore[method-assign]

        result = await client.create_order(
            symbol="BTC/AUD",
            side="buy",
            order_type="Limit",
            quantity=0.001,
            price=95000.0,
            post_only=True,
        )

        assert captured_body.get("postOnly") is True, (
            "postOnly flag must be True for maker-order guarantee"
        )
        assert captured_body.get("timeInForce") == "GTC"
        assert result["post_only"] is True
        assert result["order_id"] == "order-001"

    @pytest.mark.asyncio
    async def test_btcm_post_only_false_omits_flag(self) -> None:
        """post_only=False should NOT include postOnly in the body."""
        client = _make_client()
        captured_body: Dict[str, Any] = {}

        async def fake_request(method, path, *, params=None, body=None, authenticated=False):
            if method == "POST":
                captured_body.update(body or {})
                return {"orderId": "order-002", "status": "Placed", "creationTime": ""}
            return {}

        client._request = fake_request  # type: ignore[method-assign]
        await client.create_order("BTC/AUD", "sell", "Limit", 0.001, 95000.0, post_only=False)

        assert "postOnly" not in captured_body


# ===========================================================================
# 7. fetch_ticker with mocked HTTP
# ===========================================================================

class TestBTCMFetchTickerMock:
    """Test 7: fetch_ticker normalises bid/ask/last from BTC Markets response."""

    @pytest.mark.asyncio
    async def test_btcm_fetch_ticker_mock(self) -> None:
        client = _make_client()

        async def fake_request(method, path, *, params=None, body=None, authenticated=False):
            assert "BTC-AUD" in path, f"Expected BTC-AUD in path, got {path}"
            return {
                "bestBid":   "94500.00",
                "bestAsk":   "94600.00",
                "lastPrice": "94550.00",
                "volume24h": "12.345",
                "timestamp": "2024-01-15T10:00:00.000000Z",
            }

        client._request = fake_request  # type: ignore[method-assign]

        ticker = await client.fetch_ticker("BTC/AUD")

        assert ticker["symbol"] == "BTC/AUD"
        assert ticker["bid"]    == pytest.approx(94500.0)
        assert ticker["ask"]    == pytest.approx(94600.0)
        assert ticker["last"]   == pytest.approx(94550.0)
        assert ticker["volume"] == pytest.approx(12.345)
        assert ticker["exchange"] == "btcmarkets"

    @pytest.mark.asyncio
    async def test_btcm_fetch_ticker_symbol_forms(self) -> None:
        """Ticker works with BTCAUD, BTC/AUD, and BTC-AUD inputs."""
        client = _make_client()

        async def fake_request(method, path, *, params=None, body=None, authenticated=False):
            return {"bestBid": "1.0", "bestAsk": "2.0", "lastPrice": "1.5", "volume24h": "0"}

        client._request = fake_request  # type: ignore[method-assign]

        for sym_input in ("BTC/AUD", "BTC-AUD", "BTCAUD"):
            ticker = await client.fetch_ticker(sym_input)
            assert ticker["symbol"] == "BTC/AUD"


# ===========================================================================
# 8. fetch_order_book — bids sorted descending
# ===========================================================================

class TestBTCMFetchOrderBookMock:
    """Test 8: fetch_order_book returns bids sorted highest-first."""

    @pytest.mark.asyncio
    async def test_btcm_fetch_order_book_mock(self) -> None:
        client = _make_client()

        async def fake_request(method, path, *, params=None, body=None, authenticated=False):
            return {
                "snapshotId": 999,
                "bids": [
                    ["94400.00", "0.10"],
                    ["94500.00", "0.25"],
                    ["94450.00", "0.05"],
                ],
                "asks": [
                    ["94600.00", "0.08"],
                    ["94700.00", "0.15"],
                    ["94650.00", "0.20"],
                ],
            }

        client._request = fake_request  # type: ignore[method-assign]

        book = await client.fetch_order_book("BTC/AUD", limit=10)

        assert book["symbol"] == "BTC/AUD"
        bids = book["bids"]
        asks = book["asks"]

        # Bids must be sorted descending (highest price first)
        bid_prices = [b[0] for b in bids]
        assert bid_prices == sorted(bid_prices, reverse=True), (
            f"Bids not sorted descending: {bid_prices}"
        )

        # Asks must be sorted ascending (lowest price first)
        ask_prices = [a[0] for a in asks]
        assert ask_prices == sorted(ask_prices), (
            f"Asks not sorted ascending: {ask_prices}"
        )

        # Best bid < best ask
        assert bids[0][0] < asks[0][0], "Best bid must be less than best ask"

    @pytest.mark.asyncio
    async def test_btcm_order_book_limit_respected(self) -> None:
        """limit parameter truncates the returned levels."""
        client = _make_client()

        async def fake_request(method, path, *, params=None, body=None, authenticated=False):
            return {
                "bids": [[str(p), "1.0"] for p in range(100, 90, -1)],
                "asks": [[str(p), "1.0"] for p in range(101, 111)],
            }

        client._request = fake_request  # type: ignore[method-assign]

        book = await client.fetch_order_book("BTC/AUD", limit=5)
        assert len(book["bids"]) <= 5
        assert len(book["asks"]) <= 5


# ===========================================================================
# 9. Rate limiter
# ===========================================================================

class TestBTCMRateLimiter:
    """Test 9: Token-bucket rate limiter enforces 45 req/min ceiling."""

    @pytest.mark.asyncio
    async def test_btcm_rate_limiter(self) -> None:
        """Burst of 5 fast acquires should complete — none blocked more than a token allows."""
        limiter = _BTCMRateLimiter(max_per_min=45)
        # Pre-load all 45 tokens
        limiter._tokens = 45.0

        t0 = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - t0

        # With 45 tokens pre-loaded, 5 acquires should be near-instant
        assert elapsed < 1.0, (
            f"5 acquires with full token bucket took {elapsed:.3f}s — too slow"
        )
        assert limiter._tokens == pytest.approx(40.0, abs=0.5)

    @pytest.mark.asyncio
    async def test_btcm_rate_limiter_throttles(self) -> None:
        """Exhausted token bucket introduces a measurable delay."""
        limiter = _BTCMRateLimiter(max_per_min=60)  # 1 token/second
        limiter._tokens = 0.0
        limiter._last_refill = time.monotonic()

        t0 = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - t0

        # Should have waited ~1 second for 1 token to refill at 60/min
        assert elapsed >= 0.8, (
            f"Expected ~1s delay but got {elapsed:.3f}s"
        )


# ===========================================================================
# 10. Rebate tracking
# ===========================================================================

class TestBTCMRebateTracking:
    """Test 10: record_maker_fill accumulates and get_rebate_earned returns > 0."""

    def test_btcm_rebate_tracking(self) -> None:
        client = _make_client()

        # Initially zero
        assert client.get_rebate_earned_session() == pytest.approx(0.0)

        # Record a maker fill: 1 BTC at $95,000 AUD
        rebate_aud = client.record_maker_fill(
            fill_quantity=1.0, fill_price_aud=95_000.0
        )

        # Rebate per fill = 0.0005 * 95000 = $47.50 AUD
        assert rebate_aud == pytest.approx(47.50, rel=1e-4)

        # Session USD rebate = 47.50 * 0.62 AUD/USD = $29.45 USD
        usd_rebate = client.get_rebate_earned_session()
        assert usd_rebate > 0, "Session rebate must be positive after a maker fill"
        assert usd_rebate == pytest.approx(47.50 * 0.62, rel=1e-4)

    def test_btcm_rebate_accumulates(self) -> None:
        """Multiple fills accumulate correctly."""
        client = _make_client()

        client.record_maker_fill(0.5, 90_000.0)   # notional 45_000 AUD
        client.record_maker_fill(0.5, 90_000.0)   # notional 45_000 AUD

        total_notional_aud = 0.5 * 90_000.0 + 0.5 * 90_000.0   # 90_000 AUD
        expected_rebate_aud = abs(BTCM_MAKER_FEE) * total_notional_aud  # 45 AUD
        expected_usd = expected_rebate_aud * client.aud_usd_rate

        assert client.get_rebate_earned_session() == pytest.approx(expected_usd, rel=1e-4)
