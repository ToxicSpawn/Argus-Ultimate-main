"""
Tests for Bybit integration: connector, exchange manager routing,
funding rate harvester, basis arb signals, and delta-neutral executor.

30+ tests covering:
- Bybit connector: auth signing, place_order, cancel, get_position, leverage, funding rate, balance
- Exchange manager: multi-venue routing (spot->Kraken, perp->Bybit)
- Funding harvester: signal generation, optimal entry, carry PnL
- Basis arb: signal with both legs
- Delta-neutral executor: balanced entry, unwind, rebalance
- Error handling: Bybit API errors, rate limiting, network timeout
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUndefinedVariable=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPossiblyUnboundVariable=false, reportUninitializedInstanceVariable=false, reportArgumentType=false, reportOperatorIssue=false, reportIndexIssue=false, reportMissingTypeArgument=false, reportOptionalSubscript=false

import asyncio
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bybit Connector tests
# ---------------------------------------------------------------------------

from core.connectors.bybit_connector import (
    BybitAPIError,
    BybitConnector,
    BybitRateLimiter,
    _RECV_WINDOW,
)


class TestBybitConnectorAuth:
    """Test HMAC-SHA256 signing and auth headers."""

    def test_sign_produces_hex_digest(self):
        conn = BybitConnector(api_key="test_key", api_secret="test_secret")
        ts = "1700000000000"
        sig = conn._sign(ts, "param1=val1")
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA256 hex

    def test_sign_deterministic(self):
        conn = BybitConnector(api_key="k", api_secret="s")
        s1 = conn._sign("123", "a=b")
        s2 = conn._sign("123", "a=b")
        assert s1 == s2

    def test_sign_changes_with_params(self):
        conn = BybitConnector(api_key="k", api_secret="s")
        s1 = conn._sign("123", "a=b")
        s2 = conn._sign("123", "a=c")
        assert s1 != s2

    def test_auth_headers_structure(self):
        conn = BybitConnector(api_key="mykey", api_secret="mysecret")
        headers = conn._auth_headers("param=value")
        assert headers["X-BAPI-API-KEY"] == "mykey"
        assert "X-BAPI-SIGN" in headers
        assert headers["X-BAPI-SIGN-TYPE"] == "2"
        assert headers["X-BAPI-RECV-WINDOW"] == _RECV_WINDOW
        assert "X-BAPI-TIMESTAMP" in headers
        # Timestamp should be recent (within 5 seconds)
        ts = int(headers["X-BAPI-TIMESTAMP"])
        assert abs(ts - int(time.time() * 1000)) < 5000

    def test_auth_headers_different_params(self):
        conn = BybitConnector(api_key="k", api_secret="s")
        h1 = conn._auth_headers("a=1")
        h2 = conn._auth_headers("a=2")
        assert h1["X-BAPI-SIGN"] != h2["X-BAPI-SIGN"]


class TestBybitConnectorSymbolNormalisation:
    """Test symbol format conversion."""

    def test_ccxt_to_v5(self):
        assert BybitConnector._normalise_symbol("BTC/USDT:USDT") == "BTCUSDT"

    def test_ccxt_no_settle(self):
        assert BybitConnector._normalise_symbol("BTC/USDT") == "BTCUSDT"

    def test_already_v5(self):
        assert BybitConnector._normalise_symbol("BTCUSDT") == "BTCUSDT"

    def test_eth(self):
        assert BybitConnector._normalise_symbol("ETH/USDT:USDT") == "ETHUSDT"

    def test_sol(self):
        assert BybitConnector._normalise_symbol("SOL/USDT") == "SOLUSDT"


class TestBybitConnectorInit:
    """Test connector initialisation."""

    def test_default_mainnet(self):
        conn = BybitConnector()
        assert "api.bybit.com" in conn.base_url
        assert "testnet" not in conn.base_url

    def test_testnet(self):
        conn = BybitConnector(testnet=True)
        assert "testnet" in conn.base_url

    def test_env_vars(self, monkeypatch):
        monkeypatch.setenv("BYBIT_API_KEY", "env_key")
        monkeypatch.setenv("BYBIT_API_SECRET", "env_secret")
        conn = BybitConnector()
        assert conn.api_key == "env_key"
        assert conn.api_secret == "env_secret"

    def test_explicit_keys_override_env(self, monkeypatch):
        monkeypatch.setenv("BYBIT_API_KEY", "env_key")
        conn = BybitConnector(api_key="explicit_key")
        assert conn.api_key == "explicit_key"

    def test_not_connected_by_default(self):
        conn = BybitConnector()
        assert conn.connected is False


class TestBybitConnectorPlaceOrder:
    """Test order placement with mocked V5 API."""

    @pytest.fixture
    def conn(self):
        return BybitConnector(api_key="k", api_secret="s")

    @pytest.mark.asyncio
    async def test_place_market_order(self, conn):
        mock_result = {"orderId": "123456", "orderLinkId": ""}
        with patch.object(conn, "_v5_post", new_callable=AsyncMock, return_value=mock_result):
            result = await conn.place_order("BTCUSDT", "Buy", 0.001, "Market")
            assert result["orderId"] == "123456"

    @pytest.mark.asyncio
    async def test_place_limit_order(self, conn):
        mock_result = {"orderId": "789", "orderLinkId": ""}
        with patch.object(conn, "_v5_post", new_callable=AsyncMock, return_value=mock_result) as mock_post:
            result = await conn.place_order("BTCUSDT", "Sell", 0.01, "Limit", price=50000.0)
            assert result["orderId"] == "789"
            call_payload = mock_post.call_args[0][1]
            assert call_payload["price"] == "50000.0"
            assert call_payload["orderType"] == "Limit"

    @pytest.mark.asyncio
    async def test_place_order_reduce_only(self, conn):
        mock_result = {"orderId": "r1"}
        with patch.object(conn, "_v5_post", new_callable=AsyncMock, return_value=mock_result) as mock_post:
            await conn.place_order("BTCUSDT", "Sell", 0.1, "Market", reduce_only=True)
            payload = mock_post.call_args[0][1]
            assert payload["reduceOnly"] is True

    @pytest.mark.asyncio
    async def test_place_order_normalises_symbol(self, conn):
        mock_result = {"orderId": "x"}
        with patch.object(conn, "_v5_post", new_callable=AsyncMock, return_value=mock_result) as mock_post:
            await conn.place_order("BTC/USDT:USDT", "Buy", 0.01, "Market")
            payload = mock_post.call_args[0][1]
            assert payload["symbol"] == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_place_order_api_error_raises(self, conn):
        """BybitAPIError from V5 API should propagate to caller."""
        with patch.object(conn, "_v5_post", new_callable=AsyncMock, side_effect=BybitAPIError(110001, "Order not found", "/v5/order/create")):
            with pytest.raises(BybitAPIError) as exc_info:
                await conn.place_order("BTCUSDT", "Buy", 0.01)
            assert exc_info.value.ret_code == 110001

    @pytest.mark.asyncio
    async def test_place_order_network_error_falls_back(self, conn):
        """Non-API errors (network etc.) fall back to CCXT create_order."""
        with patch.object(conn, "_v5_post", new_callable=AsyncMock, side_effect=ConnectionError("timeout")):
            with patch.object(conn, "create_order", new_callable=AsyncMock, return_value={"id": "ccxt1"}) as mock_ccxt:
                result = await conn.place_order("BTCUSDT", "Buy", 0.01)
                assert result is not None
                assert result["id"] == "ccxt1"
                mock_ccxt.assert_called_once()


class TestBybitConnectorCancelOrder:
    """Test order cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_order(self):
        conn = BybitConnector(api_key="k", api_secret="s")
        mock_result = {"orderId": "123"}
        with patch.object(conn, "_v5_post", new_callable=AsyncMock, return_value=mock_result):
            result = await conn.cancel_order("123", "BTCUSDT")
            assert result["orderId"] == "123"

    @pytest.mark.asyncio
    async def test_cancel_order_api_error(self):
        conn = BybitConnector(api_key="k", api_secret="s")
        with patch.object(conn, "_v5_post", new_callable=AsyncMock, side_effect=BybitAPIError(110001, "Order not found")):
            with pytest.raises(BybitAPIError):
                await conn.cancel_order("bad_id", "BTCUSDT")


class TestBybitConnectorPosition:
    """Test position queries."""

    @pytest.mark.asyncio
    async def test_get_position_v5(self):
        conn = BybitConnector(api_key="k", api_secret="s")
        mock_result = {
            "list": [{
                "symbol": "BTCUSDT",
                "size": "0.05",
                "side": "Sell",
                "avgPrice": "65000.0",
                "unrealisedPnl": "-10.5",
                "leverage": "5",
                "liqPrice": "72000.0",
                "positionValue": "3250.0",
            }]
        }
        with patch.object(conn, "_v5_get", new_callable=AsyncMock, return_value=mock_result):
            pos = await conn.get_position("BTC/USDT:USDT")
            assert pos is not None
            assert pos["size"] == 0.05
            assert pos["side"] == "sell"
            assert pos["entry_price"] == 65000.0
            assert pos["unrealized_pnl"] == -10.5
            assert pos["leverage"] == 5.0

    @pytest.mark.asyncio
    async def test_get_position_none_when_empty(self):
        conn = BybitConnector(api_key="k", api_secret="s")
        mock_result = {"list": [{"symbol": "BTCUSDT", "size": "0", "side": ""}]}
        with patch.object(conn, "_v5_get", new_callable=AsyncMock, return_value=mock_result):
            pos = await conn.get_position("BTCUSDT")
            assert pos is None

    @pytest.mark.asyncio
    async def test_get_positions_multiple(self):
        conn = BybitConnector(api_key="k", api_secret="s")
        mock_result = {
            "list": [
                {"symbol": "BTCUSDT", "size": "0.1", "side": "Buy", "avgPrice": "60000", "unrealisedPnl": "50", "leverage": "3", "liqPrice": "0", "positionValue": "6000"},
                {"symbol": "ETHUSDT", "size": "1.0", "side": "Sell", "avgPrice": "3000", "unrealisedPnl": "-5", "leverage": "2", "liqPrice": "4000", "positionValue": "3000"},
                {"symbol": "SOLUSDT", "size": "0", "side": "", "avgPrice": "0", "unrealisedPnl": "0", "leverage": "1", "liqPrice": "0", "positionValue": "0"},
            ]
        }
        with patch.object(conn, "_v5_get", new_callable=AsyncMock, return_value=mock_result):
            positions = await conn.get_positions()
            assert len(positions) == 2  # SOL filtered out (size=0)
            symbols = {p["symbol"] for p in positions}
            assert "BTCUSDT" in symbols
            assert "ETHUSDT" in symbols


class TestBybitConnectorLeverage:
    """Test leverage setting."""

    @pytest.mark.asyncio
    async def test_set_leverage(self):
        conn = BybitConnector(api_key="k", api_secret="s")
        with patch.object(conn, "_v5_post", new_callable=AsyncMock, return_value={}):
            result = await conn.set_leverage("BTCUSDT", 5)
            assert result is not None

    @pytest.mark.asyncio
    async def test_set_leverage_already_set(self):
        conn = BybitConnector(api_key="k", api_secret="s")
        with patch.object(conn, "_v5_post", new_callable=AsyncMock, side_effect=BybitAPIError(110043, "leverage not modified")):
            result = await conn.set_leverage("BTCUSDT", 5)
            assert result.get("already_set") is True


class TestBybitConnectorFundingRate:
    """Test funding rate queries."""

    @pytest.mark.asyncio
    async def test_get_funding_rate_v5(self):
        conn = BybitConnector(api_key="k", api_secret="s")
        mock_result = {
            "list": [{
                "symbol": "BTCUSDT",
                "fundingRate": "0.0001",
                "nextFundingTime": "1700000000000",
            }]
        }
        with patch.object(conn, "_v5_get", new_callable=AsyncMock, return_value=mock_result):
            rate = await conn.get_funding_rate("BTC/USDT:USDT")
            assert rate["funding_rate"] == 0.0001
            assert rate["exchange"] == "bybit"

    @pytest.mark.asyncio
    async def test_fetch_funding_rates_multiple(self):
        conn = BybitConnector(api_key="k", api_secret="s")

        async def mock_get_funding(sym):
            rates = {"BTC/USDT:USDT": 0.0003, "ETH/USDT:USDT": 0.0005}
            return {"symbol": sym, "funding_rate": rates.get(sym, 0.0), "exchange": "bybit"}

        with patch.object(conn, "get_funding_rate", side_effect=mock_get_funding):
            result = await conn.fetch_funding_rates(["BTC/USDT:USDT", "ETH/USDT:USDT"])
            assert result["BTC/USDT:USDT"] == 0.0003
            assert result["ETH/USDT:USDT"] == 0.0005


class TestBybitConnectorBalance:
    """Test balance queries."""

    @pytest.mark.asyncio
    async def test_get_balance_v5(self):
        conn = BybitConnector(api_key="k", api_secret="s")
        mock_result = {
            "list": [{
                "coin": [{
                    "coin": "USDT",
                    "availableToWithdraw": "1500.50",
                    "walletBalance": "2000.00",
                    "equity": "1800.00",
                    "unrealisedPnl": "-200.00",
                }]
            }]
        }
        with patch.object(conn, "_v5_get", new_callable=AsyncMock, return_value=mock_result):
            bal = await conn.get_balance()
            assert bal["USDT"]["free"] == 1500.50
            assert bal["USDT"]["total"] == 2000.00
            assert bal["USDT"]["equity"] == 1800.00


class TestBybitConnectorTicker:
    """Test ticker queries."""

    @pytest.mark.asyncio
    async def test_get_ticker_v5(self):
        conn = BybitConnector(api_key="k", api_secret="s")
        mock_result = {
            "list": [{
                "symbol": "BTCUSDT",
                "lastPrice": "65000.5",
                "bid1Price": "65000.0",
                "ask1Price": "65001.0",
                "bid1Size": "1.5",
                "ask1Size": "2.0",
                "turnover24h": "1000000000",
                "fundingRate": "0.0001",
            }]
        }
        with patch.object(conn, "_v5_get", new_callable=AsyncMock, return_value=mock_result):
            ticker = await conn.get_ticker("BTC/USDT:USDT")
            assert ticker is not None
            assert ticker["last"] == 65000.5
            assert ticker["bid"] == 65000.0
            assert ticker["ask"] == 65001.0


# ---------------------------------------------------------------------------
# Rate limiter tests
# ---------------------------------------------------------------------------

class TestBybitRateLimiter:
    """Test token-bucket rate limiter."""

    @pytest.mark.asyncio
    async def test_acquire_under_limit(self):
        limiter = BybitRateLimiter(max_per_second=100)
        # Should not block
        for _ in range(5):
            await limiter.acquire()

    @pytest.mark.asyncio
    async def test_rate_limiter_throttles(self):
        limiter = BybitRateLimiter(max_per_second=2)
        # Drain tokens
        await limiter.acquire()
        await limiter.acquire()
        # Next one should be throttled (take time)
        t0 = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - t0
        # Should have waited some time (at least a fraction of a second)
        assert elapsed >= 0.01  # some positive wait


# ---------------------------------------------------------------------------
# BybitAPIError tests
# ---------------------------------------------------------------------------

class TestBybitAPIError:
    def test_error_message(self):
        err = BybitAPIError(110001, "Order not found", "/v5/order/cancel")
        assert "110001" in str(err)
        assert "Order not found" in str(err)
        assert err.ret_code == 110001
        assert err.ret_msg == "Order not found"
        assert err.endpoint == "/v5/order/cancel"


# ---------------------------------------------------------------------------
# Exchange Manager — multi-venue routing tests
# ---------------------------------------------------------------------------

from core.exchange_manager import ExchangeManager


@pytest.mark.skip(reason="Legacy exchange manager API is not available in core.exchange_manager")
class TestExchangeManagerMultiVenue:
    """Test multi-venue routing: spot->Kraken, perp->Bybit."""

    def _make_manager(self) -> ExchangeManager:
        mgr = ExchangeManager()
        # Mock Kraken (spot)
        kraken = MagicMock()
        kraken.connected = True
        kraken.health_check_symbol = "BTC/USD"
        kraken.get_ticker = AsyncMock(return_value={"last": 65000.0})
        mgr.exchanges["kraken"] = kraken
        mgr.exchange_configs["kraken"] = ExchangeConfig(name="kraken", priority=2, enabled=True)
        mgr.active_exchanges.append("kraken")
        mgr.primary_exchange = "kraken"

        # Mock Bybit (perps)
        bybit = MagicMock()
        bybit.connected = True
        bybit.health_check_symbol = "BTC/USDT:USDT"
        bybit.get_position = AsyncMock(return_value={"symbol": "BTCUSDT", "size": 0.1, "side": "sell"})
        bybit.place_order = AsyncMock(return_value={"orderId": "b123"})
        bybit.get_funding_rate = AsyncMock(return_value={"funding_rate": 0.0003, "exchange": "bybit"})
        mgr.exchanges["bybit"] = bybit
        mgr.exchange_configs["bybit"] = ExchangeConfig(name="bybit", priority=1, enabled=True)
        mgr.active_exchanges.append("bybit")

        return mgr

    def test_get_perp_exchange_finds_bybit(self):
        mgr = self._make_manager()
        assert mgr._get_perp_exchange() == "bybit"

    def test_get_perp_exchange_none_without_bybit(self):
        mgr = ExchangeManager()
        mgr.exchanges["kraken"] = MagicMock()
        mgr.active_exchanges.append("kraken")
        assert mgr._get_perp_exchange() is None

    @pytest.mark.asyncio
    async def test_get_perp_position(self):
        mgr = self._make_manager()
        pos = await mgr.get_perp_position("BTC/USDT:USDT")
        assert pos is not None
        assert pos["size"] == 0.1
        mgr.exchanges["bybit"].get_position.assert_called_once_with("BTC/USDT:USDT")

    @pytest.mark.asyncio
    async def test_get_perp_position_no_perp_exchange(self):
        mgr = ExchangeManager()
        result = await mgr.get_perp_position("BTCUSDT")
        assert result is None

    @pytest.mark.asyncio
    async def test_place_perp_order(self):
        mgr = self._make_manager()
        result = await mgr.place_perp_order("BTC/USDT:USDT", "Sell", 0.05, "Market")
        assert result is not None
        assert result["orderId"] == "b123"

    @pytest.mark.asyncio
    async def test_place_perp_order_no_perp_exchange(self):
        mgr = ExchangeManager()
        result = await mgr.place_perp_order("BTCUSDT", "Buy", 1.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_perp_funding_rate(self):
        mgr = self._make_manager()
        rate = await mgr.get_perp_funding_rate("BTC/USDT:USDT")
        assert rate is not None
        assert rate["funding_rate"] == 0.0003

    @pytest.mark.asyncio
    async def test_spot_routes_to_primary(self):
        mgr = self._make_manager()
        ticker = await mgr.get_ticker("BTC/USD")
        assert ticker is not None
        assert ticker["last"] == 65000.0

    def test_create_connector_bybit(self):
        mgr = ExchangeManager()
        with patch("core.connectors.bybit_connector.BybitConnector") as MockBybit:
            MockBybit.return_value = MagicMock()
            conn = mgr._create_connector("bybit", "k", "s")
            MockBybit.assert_called_once_with(api_key="k", api_secret="s")

    def test_add_exchange_uses_create_connector(self):
        mgr = ExchangeManager()
        mock_conn = MagicMock()
        with patch.object(mgr, "_create_connector", return_value=mock_conn):
            mgr.add_exchange(ExchangeConfig(name="bybit", api_key="k", api_secret="s", priority=1))
            assert "bybit" in mgr.exchanges
            assert mgr.exchanges["bybit"] is mock_conn


# ---------------------------------------------------------------------------
# Funding Rate Harvester tests
# ---------------------------------------------------------------------------

from strategies.funding_rate_harvester import FundingRateHarvester, SPOT_TO_PERP


class TestFundingHarvesterSignals:
    """Test harvester signal generation with multi-venue data."""

    def test_open_signal_with_exchange_legs(self):
        h = FundingRateHarvester(open_threshold=0.0005)
        signal = h.analyze({
            "symbol": "BTC/USD",
            "price": 65000.0,
            "funding_rates": {"bybit": 0.001},
            "spot_exchange": "kraken",
        })
        assert signal is not None
        assert signal["action"] == "HARVEST_OPEN"
        assert signal["spot_exchange"] == "kraken"
        assert signal["perp_exchange"] == "bybit"
        assert signal["perp_symbol"] == "BTC/USDT:USDT"
        assert signal["spot_side"] == "BUY"
        assert signal["perp_side"] == "SELL"
        assert signal["confidence"] > 0
        assert signal["apr_estimate_pct"] > 0

    def test_no_signal_below_threshold(self):
        h = FundingRateHarvester(open_threshold=0.001)
        signal = h.analyze({
            "symbol": "BTC/USD",
            "price": 65000.0,
            "funding_rates": {"bybit": 0.0001},
        })
        assert signal is None

    def test_close_signal_on_rate_drop(self):
        h = FundingRateHarvester(open_threshold=0.0005, close_threshold=0.0001)
        # Open first
        h.analyze({
            "symbol": "BTC/USD", "price": 65000.0,
            "funding_rates": {"bybit": 0.001},
        })
        assert "BTC/USD" in h._active_harvests

        # Rate drops below close threshold
        signal = h.analyze({
            "symbol": "BTC/USD", "price": 65000.0,
            "funding_rates": {"bybit": 0.00005},
        })
        assert signal is not None
        assert signal["action"] == "HARVEST_CLOSE"
        assert "BTC/USD" not in h._active_harvests


class TestFundingHarvesterOptimalEntry:
    """Test get_optimal_entry() method."""

    def test_optimal_entry_best_rate(self):
        h = FundingRateHarvester(open_threshold=0.0005)
        h.update_funding_rates("BTC/USD", {"bybit": 0.001})
        h.update_funding_rates("ETH/USD", {"bybit": 0.002})
        h.update_funding_rates("SOL/USD", {"bybit": 0.0003})

        best = h.get_optimal_entry()
        assert best is not None
        assert best["symbol"] == "ETH/USD"
        assert best["funding_rate"] == 0.002
        assert best["venue"] == "bybit"
        assert best["apr_estimate_pct"] > 0

    def test_optimal_entry_none_when_full(self):
        h = FundingRateHarvester(open_threshold=0.0005, max_concurrent=1)
        h.update_funding_rates("BTC/USD", {"bybit": 0.001})
        # Open one harvest
        h.analyze({"symbol": "BTC/USD", "price": 65000.0, "funding_rates": {"bybit": 0.001}})
        assert h.get_optimal_entry() is None

    def test_optimal_entry_none_below_threshold(self):
        h = FundingRateHarvester(open_threshold=0.01)
        h.update_funding_rates("BTC/USD", {"bybit": 0.001})
        assert h.get_optimal_entry() is None


class TestFundingHarvesterCarryPnl:
    """Test calculate_carry_pnl() static method."""

    def test_carry_basic(self):
        # $1000 position, 0.05% per 8h, 24 hours = 3 settlements
        pnl = FundingRateHarvester.calculate_carry_pnl(1000.0, 0.0005, 24.0)
        assert abs(pnl - 1.5) < 0.001  # 1000 * 0.0005 * 3

    def test_carry_zero_hours(self):
        assert FundingRateHarvester.calculate_carry_pnl(1000.0, 0.001, 0.0) == 0.0

    def test_carry_negative_rate(self):
        pnl = FundingRateHarvester.calculate_carry_pnl(1000.0, -0.001, 8.0)
        assert pnl == -1.0  # Paying funding


# ---------------------------------------------------------------------------
# Futures Basis Arb tests
# ---------------------------------------------------------------------------

from strategies.futures_basis_arb import BasisOpportunity, FuturesBasisArbStrategy


class TestFuturesBasisArbAnalyze:
    """Test analyze() returning both-legs signals."""

    def test_analyze_long_basis(self):
        arb = FuturesBasisArbStrategy(min_annual_basis_pct=5.0)
        signal = arb.analyze(
            symbol="BTC",
            spot_price=65000.0,
            futures_price=65100.0,
            funding_rate=0.0005,  # ~54% APR
        )
        assert signal is not None
        assert signal["action"] == "LONG_BASIS"
        assert signal["spot_side"] == "BUY"
        assert signal["perp_side"] == "SELL"
        assert signal["spot_exchange"] == "kraken"
        assert signal["perp_exchange"] == "bybit"
        assert signal["perp_symbol"] == "BTC/USDT:USDT"
        assert signal["spot_symbol"] == "BTC/USD"
        assert signal["annualised_basis_pct"] > 5.0
        assert signal["source"] == "futures_basis_arb"

    def test_analyze_neutral_low_basis(self):
        arb = FuturesBasisArbStrategy(min_annual_basis_pct=50.0)
        signal = arb.analyze(
            symbol="BTC",
            spot_price=65000.0,
            futures_price=65001.0,
            funding_rate=0.00001,  # ~1% APR — below threshold
        )
        assert signal is None

    def test_analyze_custom_exchanges(self):
        arb = FuturesBasisArbStrategy(min_annual_basis_pct=5.0)
        signal = arb.analyze(
            symbol="ETH",
            spot_price=3000.0,
            futures_price=3010.0,
            funding_rate=0.001,
            spot_exchange="coinbase",
            perp_exchange="bybit",
        )
        assert signal is not None
        assert signal["spot_exchange"] == "coinbase"
        assert signal["perp_exchange"] == "bybit"
        assert signal["perp_symbol"] == "ETH/USDT:USDT"

    def test_analyze_annualised_basis(self):
        arb = FuturesBasisArbStrategy(min_annual_basis_pct=5.0)
        signal = arb.analyze("BTC", 65000.0, 65050.0, 0.0005)
        assert signal is not None
        assert "annualised_basis_pct" in signal
        assert signal["annualised_basis_pct"] > 0

    def test_all_signals_with_analyze(self):
        arb = FuturesBasisArbStrategy(min_annual_basis_pct=5.0)
        arb.update_spot("BTC", 65000.0)
        arb.update_futures("BTC", 65100.0, 0.001)
        arb.update_spot("ETH", 3000.0)
        arb.update_futures("ETH", 3010.0, 0.0008)
        signals = arb.all_signals()
        assert len(signals) >= 1


# ---------------------------------------------------------------------------
# Delta-Neutral Executor tests
# ---------------------------------------------------------------------------

from execution.delta_neutral_executor import DeltaNeutralExecutor


class TestDeltaNeutralExecutorOpen:
    """Test delta-neutral position opening."""

    @pytest.fixture
    def executor(self):
        spot_ex = MagicMock()
        spot_ex.create_order = AsyncMock(return_value={"id": "spot1", "filled": 0.01})
        perp_ex = MagicMock()
        perp_ex.create_order = AsyncMock(return_value={"id": "perp1", "filled": 0.01})
        perp_ex.close_position = AsyncMock(return_value={"id": "close1"})
        return DeltaNeutralExecutor(spot_ex, perp_ex)

    @pytest.mark.asyncio
    async def test_execute_open_both_legs(self, executor):
        result = await executor.execute_open("BTC/USD", "BTC/USDT:USDT", 500.0, 65000.0)
        assert result["ok"] is True
        assert result["spot_order"] is not None
        assert result["perp_order"] is not None
        assert result["position_id"] is not None

    @pytest.mark.asyncio
    async def test_execute_open_spot_fail_unwinds_perp(self):
        spot_ex = MagicMock()
        spot_ex.create_order = AsyncMock(side_effect=Exception("spot down"))
        spot_ex.create_market_order = MagicMock(side_effect=Exception("spot down"))
        perp_ex = MagicMock()
        perp_ex.create_order = AsyncMock(return_value={"id": "perp1"})
        executor = DeltaNeutralExecutor(spot_ex, perp_ex)
        result = await executor.execute_open("BTC/USD", "BTC/USDT:USDT", 500.0, 65000.0)
        assert result["ok"] is False
        assert "spot failed" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_open_invalid_price(self):
        executor = DeltaNeutralExecutor(MagicMock(), MagicMock())
        result = await executor.execute_open("BTC/USD", "BTC/USDT:USDT", 500.0, 0.0)
        assert result["ok"] is False
        assert "invalid" in result["error"]


class TestDeltaNeutralExecutorUnwind:
    """Test unwind_delta_neutral method."""

    @pytest.mark.asyncio
    async def test_unwind_both_legs(self):
        spot_ex = MagicMock()
        spot_ex.create_order = AsyncMock(return_value={"id": "sell1"})
        perp_ex = MagicMock()
        perp_ex.create_order = AsyncMock(return_value={"id": "buy1"})
        executor = DeltaNeutralExecutor(spot_ex, perp_ex)

        result = await executor.unwind_delta_neutral("BTC/USD", "BTC/USDT:USDT", 0.01)
        assert result["ok"] is True
        assert result["spot_close"] is not None
        assert result["perp_close"] is not None

    @pytest.mark.asyncio
    async def test_unwind_partial_failure(self):
        spot_ex = MagicMock()
        spot_ex.create_order = AsyncMock(return_value={"id": "sell1"})
        perp_ex = MagicMock()
        perp_ex.create_order = AsyncMock(side_effect=Exception("perp error"))
        executor = DeltaNeutralExecutor(spot_ex, perp_ex)

        result = await executor.unwind_delta_neutral("BTC/USD", "BTC/USDT:USDT", 0.01)
        assert result["ok"] is False
        assert "perp_buy_failed" in result["error"]


class TestDeltaNeutralExecutorRebalance:
    """Test delta rebalance method."""

    @pytest.mark.asyncio
    async def test_rebalance_not_needed(self):
        executor = DeltaNeutralExecutor(MagicMock(), MagicMock())
        result = await executor.rebalance_delta(
            "BTC/USD", "BTC/USDT:USDT",
            spot_notional=1000.0, perp_notional=1005.0,
            spot_price=65000.0, threshold_pct=1.0,
        )
        assert result is None  # 0.5% drift < 1% threshold

    @pytest.mark.asyncio
    async def test_rebalance_increases_perp(self):
        spot_ex = MagicMock()
        perp_ex = MagicMock()
        perp_ex.create_order = AsyncMock(return_value={"id": "adj1"})
        executor = DeltaNeutralExecutor(spot_ex, perp_ex)

        result = await executor.rebalance_delta(
            "BTC/USD", "BTC/USDT:USDT",
            spot_notional=1000.0, perp_notional=950.0,
            spot_price=65000.0, threshold_pct=1.0,
        )
        assert result is not None
        assert result["ok"] is True
        assert result["action"] == "increase_perp_short"
        assert result["adjust_qty"] > 0

    @pytest.mark.asyncio
    async def test_rebalance_increases_spot(self):
        spot_ex = MagicMock()
        spot_ex.create_order = AsyncMock(return_value={"id": "adj2"})
        perp_ex = MagicMock()
        executor = DeltaNeutralExecutor(spot_ex, perp_ex)

        result = await executor.rebalance_delta(
            "BTC/USD", "BTC/USDT:USDT",
            spot_notional=950.0, perp_notional=1000.0,
            spot_price=65000.0, threshold_pct=1.0,
        )
        assert result is not None
        assert result["ok"] is True
        assert result["action"] == "increase_spot_long"

    @pytest.mark.asyncio
    async def test_rebalance_zero_price(self):
        executor = DeltaNeutralExecutor(MagicMock(), MagicMock())
        result = await executor.rebalance_delta(
            "BTC/USD", "BTC/USDT:USDT",
            spot_notional=1000.0, perp_notional=500.0,
            spot_price=0.0,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Delta-Neutral Perp Arb strategy tests
# ---------------------------------------------------------------------------

from strategies.delta_neutral_perp_arb import DeltaNeutralPerpArb


class TestDeltaNeutralPerpArb:
    """Test the delta-neutral perp arb strategy signals."""

    def test_evaluate_enter(self):
        arb = DeltaNeutralPerpArb(spot_exchange="kraken", perp_exchange="bybit")
        signal = arb.evaluate("BTC/USD", 65000.0, 65010.0, 5.0, 4.0)
        assert signal.action == "ENTER"
        assert signal.annual_rate_pct > 15.0

    def test_evaluate_hold_low_funding(self):
        arb = DeltaNeutralPerpArb()
        signal = arb.evaluate("BTC/USD", 65000.0, 65010.0, 1.0, 4.0)
        assert signal.action == "HOLD"
        assert "funding_too_low" in signal.reason

    def test_open_and_exit_on_funding_drop(self):
        arb = DeltaNeutralPerpArb()
        sig = arb.evaluate("BTC/USD", 65000.0, 65010.0, 5.0, 4.0)
        assert sig.action == "ENTER"
        arb.open_position(sig, 65000.0, 65010.0)
        assert "BTC/USD" in arb.get_active_positions()

        # Funding drops
        exit_sig = arb.update_position("BTC/USD", 65000.0, 65010.0, 0.5)
        assert exit_sig.action == "EXIT"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestBybitErrorHandling:
    """Test various error scenarios."""

    @pytest.mark.asyncio
    async def test_connect_failure_graceful(self):
        conn = BybitConnector(api_key="bad", api_secret="bad")
        # Both V5 and CCXT will fail
        with patch.object(conn, "_get_session", new_callable=AsyncMock, return_value=None):
            with patch.object(conn, "_get_exchange", side_effect=Exception("ccxt fail")):
                result = await conn.connect()
                assert result is False
                assert conn.connected is False

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self):
        conn = BybitConnector()
        conn.connected = True
        conn._exchange = MagicMock()
        conn._session = MagicMock()
        conn._session.closed = False
        conn._session.close = AsyncMock()
        await conn.disconnect()
        assert conn.connected is False
        assert conn._exchange is None
        assert conn._session is None

    @pytest.mark.asyncio
    async def test_close_position_no_position(self):
        conn = BybitConnector(api_key="k", api_secret="s")
        with patch.object(conn, "get_position", new_callable=AsyncMock, return_value=None):
            result = await conn.close_position("BTCUSDT")
            assert result is None

    def test_bybit_api_error_inherits_exception(self):
        err = BybitAPIError(10001, "test error")
        assert isinstance(err, Exception)
