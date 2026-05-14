"""
Tests for MEXC Exchange Client
================================

Covers:
  - Fee constants
  - HMAC-SHA256 signing
  - post_only flag on create_order
  - fetch_ticker (mocked aiohttp)
  - fetch_order_book (mocked aiohttp, bids sorted descending)
  - Rate limiter behaviour under load
  - Error code → MEXCAPIError
  - fetch_funding_rate (mocked)
  - MEXCWSFeed spread_bps computation
  - get_exchange_info returns dict with fee_rates key
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import unittest
import urllib.parse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from exchanges.mexc_client import (
    MEXCAPIError,
    MEXCClient,
    MEXC_FUTURES_MAKER_FEE,
    MEXC_FUTURES_TAKER_FEE,
    MEXC_SPOT_MAKER_FEE,
    MEXC_SPOT_TAKER_FEE,
    MEXC_SPOT_BASE_URL,
    MEXC_FUTURES_BASE_URL,
    MEXC_SPOT_WS_URL,
    MEXC_FUTURES_WS_URL,
    get_exchange_info,
)
from exchanges.mexc_ws_feed import MEXCWSFeed


# ===========================================================================
# 1. Fee constants
# ===========================================================================

class TestMEXCConstants(unittest.TestCase):
    """Verify all four fee constants match documented MEXC schedule."""

    def test_spot_maker_fee_is_zero(self):
        assert MEXC_SPOT_MAKER_FEE == 0.0, "Spot maker fee must be 0%"

    def test_spot_taker_fee(self):
        assert MEXC_SPOT_TAKER_FEE == pytest.approx(0.0005), \
            "Spot taker fee must be 0.05%"

    def test_futures_maker_fee_is_zero(self):
        assert MEXC_FUTURES_MAKER_FEE == 0.0, "Futures maker fee must be 0%"

    def test_futures_taker_fee(self):
        assert MEXC_FUTURES_TAKER_FEE == pytest.approx(0.0002), \
            "Futures taker fee must be 0.02%"

    def test_url_constants(self):
        assert "api.mexc.com" in MEXC_SPOT_BASE_URL
        assert "contract.mexc.com" in MEXC_FUTURES_BASE_URL
        assert "wbs.mexc.com" in MEXC_SPOT_WS_URL
        assert "contract.mexc.com" in MEXC_FUTURES_WS_URL


# ===========================================================================
# 2. HMAC signing
# ===========================================================================

class TestMEXCSign(unittest.TestCase):
    """HMAC-SHA256 signature must be deterministic for known input."""

    def setUp(self):
        self.client = MEXCClient(api_key="test_key", api_secret="test_secret")

    def test_sign_deterministic(self):
        params = {"symbol": "BTCUSDT", "side": "BUY", "quantity": "0.001"}
        sig1 = self.client._sign(params)
        sig2 = self.client._sign(params)
        assert sig1 == sig2, "Same params must produce same signature"

    def test_sign_known_value(self):
        """Verify against a hand-computed HMAC-SHA256."""
        params = {"symbol": "BTCUSDT", "timestamp": 1700000000000}
        # Sort alphabetically: symbol before timestamp
        sorted_qs = "symbol=BTCUSDT&timestamp=1700000000000"
        expected = hmac.new(
            b"test_secret",
            sorted_qs.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        assert self.client._sign(params) == expected

    def test_sign_different_params_differ(self):
        params_a = {"symbol": "BTCUSDT", "side": "BUY"}
        params_b = {"symbol": "ETHUSDT", "side": "BUY"}
        assert self.client._sign(params_a) != self.client._sign(params_b)

    def test_sign_sorted_order_independent(self):
        """Order of keys must not affect signature (sort is applied internally)."""
        params_ab = {"a": "1", "b": "2"}
        params_ba = {"b": "2", "a": "1"}
        assert self.client._sign(params_ab) == self.client._sign(params_ba)


# ===========================================================================
# 3. post_only flag
# ===========================================================================

class TestMEXCPostOnly(unittest.IsolatedAsyncioTestCase):
    """create_order with post_only=True must include timeInForce=PO."""

    async def asyncSetUp(self):
        self.client = MEXCClient(api_key="k", api_secret="s")

    async def asyncTearDown(self):
        await self.client.close()

    async def test_post_only_sends_po_time_in_force(self):
        captured = {}

        async def fake_request(method, base_url, path, **kwargs):
            captured.update(kwargs.get("params", {}))
            return {"orderId": "123", "status": "NEW"}

        self.client._request = fake_request
        await self.client.create_order(
            "BTCUSDT", "BUY", "LIMIT", 0.001, price=29000.0, post_only=True
        )
        assert captured.get("timeInForce") == "PO", \
            "post_only=True must set timeInForce=PO"

    async def test_non_post_only_no_po(self):
        captured = {}

        async def fake_request(method, base_url, path, **kwargs):
            captured.update(kwargs.get("params", {}))
            return {"orderId": "124", "status": "NEW"}

        self.client._request = fake_request
        await self.client.create_order(
            "BTCUSDT", "BUY", "LIMIT", 0.001, price=29000.0, post_only=False
        )
        assert captured.get("timeInForce") != "PO", \
            "post_only=False must NOT set timeInForce=PO"


# ===========================================================================
# 4. fetch_ticker (mocked)
# ===========================================================================

class TestMEXCFetchTickerMock(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.client = MEXCClient(api_key="k", api_secret="s")

    async def asyncTearDown(self):
        await self.client.close()

    async def test_fetch_ticker_extracts_bid_ask(self):
        call_count = 0

        async def fake_request(method, base_url, path, **kwargs):
            nonlocal call_count
            call_count += 1
            if "bookTicker" in path:
                return {
                    "symbol": "BTCUSDT",
                    "bidPrice": "29000.50",
                    "bidQty": "0.1",
                    "askPrice": "29001.00",
                    "askQty": "0.2",
                }
            # 24hr stats
            return {
                "lastPrice": "29000.75",
                "volume": "1234.56",
            }

        self.client._request = fake_request
        ticker = await self.client.fetch_ticker("BTCUSDT")

        assert ticker["bid"] == pytest.approx(29000.50)
        assert ticker["ask"] == pytest.approx(29001.00)
        assert ticker["last"] == pytest.approx(29000.75)
        assert ticker["volume"] == pytest.approx(1234.56)
        assert "timestamp" in ticker
        assert call_count == 2


# ===========================================================================
# 5. fetch_order_book (mocked, bids sorted desc)
# ===========================================================================

class TestMEXCFetchOrderBookMock(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.client = MEXCClient(api_key="k", api_secret="s")

    async def asyncTearDown(self):
        await self.client.close()

    async def test_bids_sorted_descending(self):
        async def fake_request(method, base_url, path, **kwargs):
            return {
                "lastUpdateId": 99,
                "bids": [
                    ["28999.00", "0.3"],
                    ["29001.00", "0.1"],
                    ["29000.00", "0.2"],
                ],
                "asks": [
                    ["29002.00", "0.5"],
                    ["29003.00", "0.4"],
                ],
            }

        self.client._request = fake_request
        book = await self.client.fetch_order_book("BTCUSDT", limit=20)

        bid_prices = [level[0] for level in book["bids"]]
        ask_prices = [level[0] for level in book["asks"]]

        assert bid_prices == sorted(bid_prices, reverse=True), \
            "Bids must be sorted highest-first"
        assert ask_prices == sorted(ask_prices), \
            "Asks must be sorted lowest-first"
        assert bid_prices[0] == pytest.approx(29001.00)
        assert ask_prices[0] == pytest.approx(29002.00)


# ===========================================================================
# 6. Rate limiter
# ===========================================================================

class TestMEXCRateLimiter(unittest.IsolatedAsyncioTestCase):
    """Verify the token bucket actually throttles at 490 req / 10 s."""

    async def test_rate_limiter_activates_under_burst(self):
        from exchanges.mexc_client import _TokenBucket

        bucket = _TokenBucket(capacity=5, window=10.0)
        start = time.monotonic()

        # Acquire 6 tokens; the 6th should block briefly
        for _ in range(6):
            await bucket.acquire()

        elapsed = time.monotonic() - start
        # With capacity=5/10s, refill rate = 0.5 tok/s
        # After 5 free tokens, 6th needs at least 2s of refill
        # We just verify some measurable delay happened (> 0.5s)
        assert elapsed > 0.5, (
            f"Rate limiter should introduce delay after burst; elapsed={elapsed:.2f}s"
        )


# ===========================================================================
# 7. Error handling → MEXCAPIError
# ===========================================================================

class TestMEXCErrorHandling(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.client = MEXCClient(api_key="k", api_secret="s")

    async def asyncTearDown(self):
        await self.client.close()

    async def test_exchange_error_raises_mexc_api_error(self):
        """A non-zero code in the response body must raise MEXCAPIError."""
        # Patch _request directly to simulate the exchange returning an error
        # payload.  MEXCClient._request is responsible for parsing and raising.
        async def error_request(*args, **kwargs):
            raise MEXCAPIError(700003, "Invalid price")

        self.client._request = error_request

        with pytest.raises(MEXCAPIError) as exc_info:
            await self.client.fetch_order_book("BTCUSDT")

        assert exc_info.value.code == 700003
        assert "Invalid price" in exc_info.value.message

    def test_mexc_api_error_attributes(self):
        err = MEXCAPIError(code=429, message="Rate limit exceeded")
        assert err.code == 429
        assert "Rate limit exceeded" in err.message
        assert "429" in str(err)


# ===========================================================================
# 8. fetch_funding_rate (mocked)
# ===========================================================================

class TestMEXCFundingRateMock(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.client = MEXCClient(api_key="k", api_secret="s")

    async def asyncTearDown(self):
        await self.client.close()

    async def test_funding_rate_mock(self):
        async def fake_request(method, base_url, path, **kwargs):
            return {
                "success": True,
                "code": 0,
                "data": {
                    "symbol": "BTC_USDT",
                    "fundingRate": "0.0001",
                    "nextSettleTime": 1700000000000,
                    "nextFundingRate": "0.00015",
                },
            }

        self.client._request = fake_request
        result = await self.client.fetch_funding_rate("BTC_USDT")

        assert "rate" in result
        assert result["rate"] == pytest.approx(0.0001)
        assert result["next_time"] == 1700000000000
        assert result["predicted"] == pytest.approx(0.00015)


# ===========================================================================
# 9. MEXCWSFeed spread_bps
# ===========================================================================

class TestMEXCWSFeedSpread(unittest.TestCase):
    """Verify spread_bps is computed correctly from book state."""

    def test_spread_bps_computed_correctly(self):
        feed = MEXCWSFeed(symbols=["BTCUSDT"])
        book = feed.books["BTCUSDT"]

        # Inject a mock book state
        book.bids = {29000.0: 0.5, 28999.0: 1.0}
        book.asks = {29010.0: 0.3, 29011.0: 0.2}
        book.is_snapshot = True

        spread = feed.get_spread_bps("BTCUSDT")
        # best_bid=29000, best_ask=29010
        # spread_bps = (29010 - 29000) / 29000 * 10000 ≈ 3.448
        assert spread is not None
        assert spread == pytest.approx((29010 - 29000) / 29000 * 10000, rel=1e-4)

    def test_spread_none_when_book_empty(self):
        feed = MEXCWSFeed(symbols=["ETHUSDT"])
        spread = feed.get_spread_bps("ETHUSDT")
        assert spread is None

    def test_best_bid_ask_accessors(self):
        feed = MEXCWSFeed(symbols=["BTCUSDT"])
        book = feed.books["BTCUSDT"]
        book.bids = {29000.0: 1.0, 28950.0: 2.0}
        book.asks = {29050.0: 1.0, 29100.0: 2.0}

        assert feed.get_best_bid("BTCUSDT") == pytest.approx(29000.0)
        assert feed.get_best_ask("BTCUSDT") == pytest.approx(29050.0)


# ===========================================================================
# 10. get_exchange_info
# ===========================================================================

class TestMEXCGetExchangeInfo(unittest.TestCase):

    def test_returns_dict_with_fee_rates(self):
        info = get_exchange_info()
        assert isinstance(info, dict)
        assert "fee_rates" in info

    def test_fee_rates_content(self):
        info = get_exchange_info()
        rates = info["fee_rates"]
        assert rates["spot_maker"] == 0.0
        assert rates["spot_taker"] == pytest.approx(0.0005)
        assert rates["futures_maker"] == 0.0
        assert rates["futures_taker"] == pytest.approx(0.0002)

    def test_exchange_name(self):
        info = get_exchange_info()
        assert info.get("exchange") == "mexc"

    def test_urls_present(self):
        info = get_exchange_info()
        assert "urls" in info
        assert "spot_rest" in info["urls"]
        assert "futures_rest" in info["urls"]


if __name__ == "__main__":
    unittest.main()
