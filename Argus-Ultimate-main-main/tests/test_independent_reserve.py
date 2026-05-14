"""
Tests for Independent Reserve connector.

Covers: symbol mapping, HMAC signing, ticker/orderbook/balance parsing,
order placement, cancellation, trade history, rate limiting, error handling,
health check, nonce incrementing, async context manager, exchange manager wiring.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.connectors.independent_reserve_connector import (
    IR_API_BASE,
    IndependentReserveAuthError,
    IndependentReserveConnector,
    IndependentReserveError,
    IndependentReserveRateLimitError,
    IRRateLimiter,
    _PRIMARY_CURRENCY_MAP,
    _SECONDARY_CURRENCY_MAP,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def connector():
    """Create a connector with test credentials."""
    return IndependentReserveConnector(
        api_key="test_key_abc123",
        api_secret="test_secret_xyz789",
        base_url="https://api.independentreserve.com",
    )


@pytest.fixture
def mock_session():
    """Create a mock aiohttp session."""
    session = AsyncMock()
    session.closed = False
    return session


# ---------------------------------------------------------------------------
# Helper to create mock HTTP responses
# ---------------------------------------------------------------------------

def _mock_response(status: int = 200, json_data: Any = None, text: str = ""):
    """Create a mock aiohttp response context manager."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data or {})
    resp.text = AsyncMock(return_value=text)

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


from typing import Any


# ===========================================================================
# 1. Symbol Mapping Tests
# ===========================================================================

class TestSymbolMapping:
    """Test IR symbol conversion (BTC/AUD -> Xbt/Aud and back)."""

    def test_btc_aud(self):
        primary, secondary = IndependentReserveConnector.parse_symbol("BTC/AUD")
        assert primary == "Xbt"
        assert secondary == "Aud"

    def test_eth_aud(self):
        primary, secondary = IndependentReserveConnector.parse_symbol("ETH/AUD")
        assert primary == "Eth"
        assert secondary == "Aud"

    def test_ltc_aud(self):
        primary, secondary = IndependentReserveConnector.parse_symbol("LTC/AUD")
        assert primary == "Ltc"
        assert secondary == "Aud"

    def test_xrp_aud(self):
        primary, secondary = IndependentReserveConnector.parse_symbol("XRP/AUD")
        assert primary == "Xrp"
        assert secondary == "Aud"

    def test_sol_aud(self):
        primary, secondary = IndependentReserveConnector.parse_symbol("SOL/AUD")
        assert primary == "Sol"
        assert secondary == "Aud"

    def test_usdt_aud(self):
        primary, secondary = IndependentReserveConnector.parse_symbol("USDT/AUD")
        assert primary == "Usdt"
        assert secondary == "Aud"

    def test_case_insensitive(self):
        primary, secondary = IndependentReserveConnector.parse_symbol("btc/aud")
        assert primary == "Xbt"
        assert secondary == "Aud"

    def test_reverse_mapping_btc(self):
        sym = IndependentReserveConnector.to_standard_symbol("Xbt", "Aud")
        assert sym == "BTC/AUD"

    def test_reverse_mapping_eth(self):
        sym = IndependentReserveConnector.to_standard_symbol("Eth", "Aud")
        assert sym == "ETH/AUD"

    def test_invalid_no_slash(self):
        with pytest.raises(ValueError, match="Invalid symbol format"):
            IndependentReserveConnector.parse_symbol("BTCAUD")

    def test_unsupported_primary(self):
        with pytest.raises(ValueError, match="Unsupported primary currency"):
            IndependentReserveConnector.parse_symbol("SHIB/AUD")

    def test_unsupported_secondary(self):
        with pytest.raises(ValueError, match="Unsupported secondary currency"):
            IndependentReserveConnector.parse_symbol("BTC/EUR")

    def test_all_supported_pairs_parse(self):
        """All declared SUPPORTED_PAIRS should parse without error."""
        for pair in IndependentReserveConnector.SUPPORTED_PAIRS:
            primary, secondary = IndependentReserveConnector.parse_symbol(pair)
            assert primary in _PRIMARY_CURRENCY_MAP.values()
            assert secondary in _SECONDARY_CURRENCY_MAP.values()


# ===========================================================================
# 2. HMAC Signing Tests
# ===========================================================================

class TestHMACSigning:
    """Verify HMAC-SHA256 signing matches known test vectors."""

    def test_sign_basic(self, connector):
        """Verify signature for a known URL + params."""
        url = "https://api.independentreserve.com/Private/GetAccounts"
        params = [
            ("apiKey", "test_key_abc123"),
            ("nonce", "1234567890"),
        ]
        sig = connector._sign(url, params)

        # Recompute expected
        message = "https://api.independentreserve.com/Private/GetAccounts,apiKey=test_key_abc123,nonce=1234567890"
        expected = hmac.new(
            b"test_secret_xyz789", message.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        assert sig == expected

    def test_sign_with_extra_params(self, connector):
        """Params should be sorted alphabetically in the signed message."""
        url = "https://api.independentreserve.com/Private/PlaceLimitOrder"
        params = [
            ("apiKey", "test_key_abc123"),
            ("nonce", "999"),
            ("volume", "0.1"),
            ("price", "50000"),
            ("primaryCurrencyCode", "xbt"),
        ]
        sig = connector._sign(url, params)

        # Sorted by key: apiKey, nonce, price, primaryCurrencyCode, volume
        message = (
            "https://api.independentreserve.com/Private/PlaceLimitOrder,"
            "apiKey=test_key_abc123,nonce=999,price=50000,"
            "primaryCurrencyCode=xbt,volume=0.1"
        )
        expected = hmac.new(
            b"test_secret_xyz789", message.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        assert sig == expected

    def test_sign_deterministic(self, connector):
        """Same inputs produce same signature."""
        url = "https://example.com/test"
        params = [("apiKey", "k"), ("nonce", "1")]
        sig1 = connector._sign(url, params)
        sig2 = connector._sign(url, params)
        assert sig1 == sig2

    def test_sign_different_secret(self):
        """Different secret produces different signature."""
        c1 = IndependentReserveConnector(api_key="k", api_secret="secret1")
        c2 = IndependentReserveConnector(api_key="k", api_secret="secret2")
        url = "https://example.com/test"
        params = [("apiKey", "k"), ("nonce", "1")]
        assert c1._sign(url, params) != c2._sign(url, params)


# ===========================================================================
# 3. Ticker Parsing Tests
# ===========================================================================

class TestTickerParsing:

    @pytest.mark.asyncio
    async def test_get_ticker_success(self, connector, mock_session):
        """Successful ticker response is parsed correctly."""
        ir_response = {
            "LastPrice": 95000.50,
            "CurrentHighestBidPrice": 94990.00,
            "CurrentLowestOfferPrice": 95010.00,
            "DayVolumeXbt": 12.345,
            "DayHighestPrice": 96000.00,
            "DayLowestPrice": 93000.00,
            "CreatedTimestampUtc": "2026-03-18T10:00:00Z",
        }

        mock_session.get = MagicMock(return_value=_mock_response(200, ir_response))
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        ticker = await connector.get_ticker("BTC/AUD")

        assert ticker is not None
        assert ticker["symbol"] == "BTC/AUD"
        assert ticker["bid"] == 94990.00
        assert ticker["ask"] == 95010.00
        assert ticker["last"] == 95000.50
        assert ticker["volume_24h"] == 12.345
        assert ticker["exchange"] == "independent_reserve"

    @pytest.mark.asyncio
    async def test_get_ticker_invalid_symbol(self, connector):
        """Invalid symbol returns None, does not raise."""
        ticker = await connector.get_ticker("INVALID")
        assert ticker is None


# ===========================================================================
# 4. Orderbook Parsing Tests
# ===========================================================================

class TestOrderbookParsing:

    @pytest.mark.asyncio
    async def test_get_orderbook_success(self, connector, mock_session):
        ir_response = {
            "BuyOrders": [
                {"Price": 94990.0, "Volume": 0.5},
                {"Price": 94980.0, "Volume": 1.2},
            ],
            "SellOrders": [
                {"Price": 95010.0, "Volume": 0.3},
                {"Price": 95020.0, "Volume": 0.8},
            ],
            "CreatedTimestampUtc": "2026-03-18T10:00:00Z",
        }

        mock_session.get = MagicMock(return_value=_mock_response(200, ir_response))
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        ob = await connector.get_orderbook("BTC/AUD")

        assert ob is not None
        assert ob["symbol"] == "BTC/AUD"
        assert len(ob["bids"]) == 2
        assert len(ob["asks"]) == 2
        assert ob["bids"][0] == (94990.0, 0.5)
        assert ob["asks"][0] == (95010.0, 0.3)

    @pytest.mark.asyncio
    async def test_get_orderbook_depth_truncation(self, connector, mock_session):
        """Orderbook truncates to requested depth."""
        ir_response = {
            "BuyOrders": [{"Price": 100.0 - i, "Volume": 1.0} for i in range(100)],
            "SellOrders": [{"Price": 101.0 + i, "Volume": 1.0} for i in range(100)],
            "CreatedTimestampUtc": "2026-03-18T10:00:00Z",
        }

        mock_session.get = MagicMock(return_value=_mock_response(200, ir_response))
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        ob = await connector.get_orderbook("BTC/AUD", depth=10)
        assert ob is not None
        assert len(ob["bids"]) == 10
        assert len(ob["asks"]) == 10


# ===========================================================================
# 5. Balance Parsing Tests
# ===========================================================================

class TestBalanceParsing:

    @pytest.mark.asyncio
    async def test_get_balances_success(self, connector, mock_session):
        ir_response = [
            {"CurrencyCode": "Xbt", "TotalBalance": 1.5, "AvailableBalance": 1.0},
            {"CurrencyCode": "Aud", "TotalBalance": 50000.0, "AvailableBalance": 45000.0},
            {"CurrencyCode": "Eth", "TotalBalance": 10.0, "AvailableBalance": 8.5},
        ]

        mock_session.post = MagicMock(return_value=_mock_response(200, ir_response))
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        balances = await connector.get_balances()

        assert "BTC" in balances
        assert balances["BTC"]["total"] == 1.5
        assert balances["BTC"]["available"] == 1.0
        assert "AUD" in balances
        assert balances["AUD"]["total"] == 50000.0
        assert balances["AUD"]["available"] == 45000.0
        assert "ETH" in balances

    @pytest.mark.asyncio
    async def test_get_balance_alias(self, connector, mock_session):
        """get_balance() is an alias for get_balances()."""
        ir_response = [
            {"CurrencyCode": "Aud", "TotalBalance": 1000.0, "AvailableBalance": 900.0},
        ]
        mock_session.post = MagicMock(return_value=_mock_response(200, ir_response))
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        result = await connector.get_balance()
        assert "AUD" in result


# ===========================================================================
# 6. Order Placement Tests (mock HTTP)
# ===========================================================================

class TestOrderPlacement:

    @pytest.mark.asyncio
    async def test_place_limit_buy(self, connector, mock_session):
        ir_response = {
            "OrderGuid": "abc-123-def",
            "Status": "Open",
            "CreatedTimestampUtc": "2026-03-18T10:00:00Z",
        }
        mock_session.post = MagicMock(return_value=_mock_response(200, ir_response))
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        result = await connector.place_order(
            symbol="BTC/AUD", side="buy", order_type="limit",
            quantity=0.01, price=90000.0,
        )

        assert result["order_id"] == "abc-123-def"
        assert result["side"] == "buy"
        assert result["order_type"] == "limit"
        assert result["symbol"] == "BTC/AUD"

    @pytest.mark.asyncio
    async def test_place_market_sell(self, connector, mock_session):
        ir_response = {
            "OrderGuid": "sell-456",
            "Status": "Filled",
            "CreatedTimestampUtc": "2026-03-18T10:00:00Z",
        }
        mock_session.post = MagicMock(return_value=_mock_response(200, ir_response))
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        result = await connector.place_order(
            symbol="ETH/AUD", side="sell", order_type="market",
            quantity=1.0,
        )

        assert result["order_id"] == "sell-456"
        assert result["side"] == "sell"
        assert result["order_type"] == "market"

    @pytest.mark.asyncio
    async def test_place_order_invalid_side(self, connector):
        with pytest.raises(ValueError, match="Invalid side"):
            await connector.place_order("BTC/AUD", "hold", "limit", 0.1, 90000)

    @pytest.mark.asyncio
    async def test_place_order_invalid_type(self, connector):
        with pytest.raises(ValueError, match="Invalid order_type"):
            await connector.place_order("BTC/AUD", "buy", "stop", 0.1, 90000)

    @pytest.mark.asyncio
    async def test_place_order_zero_quantity(self, connector):
        with pytest.raises(ValueError, match="Quantity must be positive"):
            await connector.place_order("BTC/AUD", "buy", "limit", 0, 90000)

    @pytest.mark.asyncio
    async def test_place_limit_order_no_price(self, connector):
        with pytest.raises(ValueError, match="Price is required"):
            await connector.place_order("BTC/AUD", "buy", "limit", 0.1, None)


# ===========================================================================
# 7. Order Cancellation Tests
# ===========================================================================

class TestOrderCancellation:

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, connector, mock_session):
        mock_session.post = MagicMock(return_value=_mock_response(200, {}))
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        result = await connector.cancel_order("order-guid-123")
        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_order_api_error(self, connector, mock_session):
        mock_session.post = MagicMock(
            return_value=_mock_response(400, text="Order not found")
        )
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        with pytest.raises(IndependentReserveError):
            await connector.cancel_order("nonexistent-guid")


# ===========================================================================
# 8. Trade History Tests
# ===========================================================================

class TestTradeHistory:

    @pytest.mark.asyncio
    async def test_get_trade_history(self, connector, mock_session):
        ir_response = {
            "Data": [
                {
                    "TradeGuid": "trade-1",
                    "OrderGuid": "order-1",
                    "Taker": "Bid",
                    "PrimaryCurrencyAmount": 0.5,
                    "SecondaryCurrencyTradePrice": 95000.0,
                    "BrokeFee": 2.50,
                    "TradeTimestampUtc": "2026-03-18T10:00:00Z",
                },
                {
                    "TradeGuid": "trade-2",
                    "OrderGuid": "order-2",
                    "Taker": "Offer",
                    "PrimaryCurrencyAmount": 1.0,
                    "SecondaryCurrencyTradePrice": 94500.0,
                    "BrokeFee": 5.00,
                    "TradeTimestampUtc": "2026-03-18T09:30:00Z",
                },
            ],
            "PageSize": 50,
            "TotalItems": 2,
        }

        mock_session.post = MagicMock(return_value=_mock_response(200, ir_response))
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        trades = await connector.get_trade_history("BTC/AUD", limit=50)

        assert len(trades) == 2
        assert trades[0]["trade_id"] == "trade-1"
        assert trades[0]["side"] == "buy"  # Taker=Bid -> buy
        assert trades[0]["quantity"] == 0.5
        assert trades[0]["price"] == 95000.0
        assert trades[1]["side"] == "sell"  # Taker=Offer -> sell


# ===========================================================================
# 9. Rate Limiting Tests
# ===========================================================================

class TestRateLimiting:

    @pytest.mark.asyncio
    async def test_rate_limiter_basic(self):
        """Rate limiter allows immediate first request."""
        limiter = IRRateLimiter(max_per_second=10)
        t0 = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.05  # First request should be near-instant

    @pytest.mark.asyncio
    async def test_rate_limiter_throttles(self):
        """Rate limiter throttles when bucket is exhausted."""
        limiter = IRRateLimiter(max_per_second=2)
        # Drain the bucket
        await limiter.acquire()
        await limiter.acquire()
        t0 = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - t0
        # Should have waited ~0.5s (1/2 req per second)
        assert elapsed >= 0.1

    @pytest.mark.asyncio
    async def test_429_retry(self, connector, mock_session):
        """HTTP 429 triggers retry with backoff."""
        call_count = 0

        def make_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _mock_response(429)
            return _mock_response(200, {"LastPrice": 95000})

        mock_session.get = MagicMock(side_effect=make_response)
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        result = await connector._public_get(
            "/Public/GetMarketSummary",
            {"primaryCurrencyCode": "xbt", "secondaryCurrencyCode": "aud"},
        )
        assert result["LastPrice"] == 95000
        assert call_count == 3  # 2 retries + 1 success


# ===========================================================================
# 10. Error Handling Tests
# ===========================================================================

class TestErrorHandling:

    @pytest.mark.asyncio
    async def test_http_500_raises(self, connector, mock_session):
        mock_session.get = MagicMock(
            return_value=_mock_response(500, text="Internal Server Error")
        )
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        with pytest.raises(IndependentReserveError) as exc_info:
            await connector._public_get("/Public/GetMarketSummary")
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_auth_failure_401(self, connector, mock_session):
        mock_session.post = MagicMock(
            return_value=_mock_response(401, text="Invalid API key")
        )
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        with pytest.raises(IndependentReserveAuthError):
            await connector._private_post("/Private/GetAccounts")

    @pytest.mark.asyncio
    async def test_auth_failure_403(self, connector, mock_session):
        mock_session.get = MagicMock(
            return_value=_mock_response(403, text="Forbidden")
        )
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        with pytest.raises(IndependentReserveAuthError):
            await connector._public_get("/Public/Forbidden")

    @pytest.mark.asyncio
    async def test_429_exhausted_raises(self, connector, mock_session):
        """After max retries on 429, raise IndependentReserveRateLimitError."""
        mock_session.get = MagicMock(return_value=_mock_response(429))
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        with pytest.raises(IndependentReserveRateLimitError):
            await connector._public_get("/Public/GetMarketSummary")

    @pytest.mark.asyncio
    async def test_missing_credentials_private(self):
        """Private endpoint without credentials raises AuthError."""
        c = IndependentReserveConnector(api_key="", api_secret="")
        with pytest.raises(IndependentReserveAuthError, match="API key and secret"):
            await c._private_post("/Private/GetAccounts")

    def test_exception_hierarchy(self):
        """Auth and RateLimit errors are subclasses of base error."""
        assert issubclass(IndependentReserveAuthError, IndependentReserveError)
        assert issubclass(IndependentReserveRateLimitError, IndependentReserveError)


# ===========================================================================
# 11. Health Check Tests
# ===========================================================================

class TestHealthCheck:

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, connector, mock_session):
        ir_response = {"LastPrice": 95000.0}
        mock_session.get = MagicMock(return_value=_mock_response(200, ir_response))
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        result = await connector.health_check()
        assert result["healthy"] is True
        assert result["exchange"] == "independent_reserve"
        assert result["latency_ms"] >= 0
        assert result["last_price"] == 95000.0

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, connector, mock_session):
        mock_session.get = MagicMock(
            return_value=_mock_response(500, text="Down")
        )
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        result = await connector.health_check()
        assert result["healthy"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_connect_success(self, connector, mock_session):
        ir_response = {"LastPrice": 95000.0}
        mock_session.get = MagicMock(return_value=_mock_response(200, ir_response))
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        ok = await connector.connect()
        assert ok is True
        assert connector.connected is True

    @pytest.mark.asyncio
    async def test_connect_failure(self, connector, mock_session):
        mock_session.get = MagicMock(
            return_value=_mock_response(500, text="Down")
        )
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        ok = await connector.connect()
        assert ok is False
        assert connector.connected is False


# ===========================================================================
# 12. Nonce Tests
# ===========================================================================

class TestNonce:

    @pytest.mark.asyncio
    async def test_nonce_increments(self, connector):
        """Each nonce call returns a strictly increasing value."""
        n1 = await connector._next_nonce()
        n2 = await connector._next_nonce()
        n3 = await connector._next_nonce()
        assert n2 > n1
        assert n3 > n2

    @pytest.mark.asyncio
    async def test_nonce_is_integer(self, connector):
        n = await connector._next_nonce()
        assert isinstance(n, int)

    @pytest.mark.asyncio
    async def test_nonce_based_on_time(self, connector):
        """Nonce should be approximately current time in milliseconds."""
        n = await connector._next_nonce()
        now_ms = int(time.time() * 1000)
        # Within 5 seconds
        assert abs(n - now_ms) < 5000


# ===========================================================================
# 13. Async Context Manager Tests
# ===========================================================================

class TestAsyncContextManager:

    @pytest.mark.asyncio
    async def test_context_manager_connect_disconnect(self, mock_session):
        """async with connector connects and disconnects."""
        ir_response = {"LastPrice": 95000.0}
        mock_session.get = MagicMock(return_value=_mock_response(200, ir_response))
        mock_session.close = AsyncMock()

        c = IndependentReserveConnector(api_key="k", api_secret="s")
        c._session = mock_session
        c._rate_limiter = AsyncMock()
        c._rate_limiter.acquire = AsyncMock()

        async with c:
            assert c.connected is True

        assert c.connected is False

    @pytest.mark.asyncio
    async def test_disconnect_closes_session(self, connector, mock_session):
        mock_session.close = AsyncMock()
        connector._session = mock_session

        await connector.disconnect()
        assert connector.connected is False
        mock_session.close.assert_awaited_once()


# ===========================================================================
# 14. Exchange Manager Wiring Tests
# ===========================================================================

class TestExchangeManagerWiring:

    def test_create_ir_connector(self):
        """ExchangeManager._create_connector creates IR connector for 'independent_reserve'."""
        from core.exchange_manager import ExchangeManager
        em = ExchangeManager()
        connector = em._create_connector("independent_reserve", "key", "secret")
        assert isinstance(connector, IndependentReserveConnector)
        assert connector.api_key == "key"
        assert connector.api_secret == "secret"

    def test_create_ir_connector_alias(self):
        """ExchangeManager recognizes 'ir' as alias."""
        from core.exchange_manager import ExchangeManager
        em = ExchangeManager()
        connector = em._create_connector("ir", "key", "secret")
        assert isinstance(connector, IndependentReserveConnector)

    def test_create_ir_connector_case_insensitive(self):
        """ExchangeManager recognizes 'IndependentReserve' (any case)."""
        from core.exchange_manager import ExchangeManager
        em = ExchangeManager()
        connector = em._create_connector("IndependentReserve", "key", "secret")
        assert isinstance(connector, IndependentReserveConnector)


# ===========================================================================
# 15. Get Order / Open Orders Tests
# ===========================================================================

class TestGetOrder:

    @pytest.mark.asyncio
    async def test_get_order_details(self, connector, mock_session):
        ir_response = {
            "OrderGuid": "order-789",
            "Status": "Filled",
            "OrderType": "LimitBid",
            "Volume": 0.5,
            "Outstanding": 0.0,
            "Price": 90000.0,
            "AvgPrice": 89950.0,
            "CreatedTimestampUtc": "2026-03-18T10:00:00Z",
        }
        mock_session.post = MagicMock(return_value=_mock_response(200, ir_response))
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        order = await connector.get_order("order-789")
        assert order["order_id"] == "order-789"
        assert order["status"] == "Filled"
        assert order["volume"] == 0.5
        assert order["avg_price"] == 89950.0

    @pytest.mark.asyncio
    async def test_get_open_orders(self, connector, mock_session):
        ir_response = {
            "Data": [
                {
                    "OrderGuid": "open-1",
                    "Status": "Open",
                    "OrderType": "LimitBid",
                    "Volume": 0.1,
                    "Outstanding": 0.1,
                    "Price": 88000.0,
                    "CreatedTimestampUtc": "2026-03-18T09:00:00Z",
                },
            ],
            "PageSize": 50,
            "TotalItems": 1,
        }
        mock_session.post = MagicMock(return_value=_mock_response(200, ir_response))
        connector._session = mock_session
        connector._rate_limiter = AsyncMock()
        connector._rate_limiter.acquire = AsyncMock()

        orders = await connector.get_open_orders("BTC/AUD")
        assert len(orders) == 1
        assert orders[0]["order_id"] == "open-1"
        assert orders[0]["status"] == "Open"


# ===========================================================================
# 16. Constants and Config Tests
# ===========================================================================

class TestConfig:

    def test_default_base_url(self):
        c = IndependentReserveConnector()
        assert c.base_url == IR_API_BASE

    def test_custom_base_url(self):
        c = IndependentReserveConnector(base_url="https://custom.example.com/")
        assert c.base_url == "https://custom.example.com"  # trailing slash stripped

    def test_health_check_symbol(self):
        assert IndependentReserveConnector.health_check_symbol == "BTC/AUD"

    def test_supported_pairs_non_empty(self):
        assert len(IndependentReserveConnector.SUPPORTED_PAIRS) >= 6

    def test_env_var_fallback(self):
        """Connector reads from env vars when no explicit key provided."""
        import os
        c = IndependentReserveConnector()
        # Should not crash with empty env vars
        assert c.api_key == os.environ.get("INDEPENDENT_RESERVE_API_KEY", "")
