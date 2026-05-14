"""
Tests for LiveMarketDataManager, ExchangeValidator, and position sync.

Covers:
  - LiveMarketDataManager: subscribe, get_latest, staleness, orderbook, callbacks, REST fallback
  - ExchangeValidator: key validation, balance check, pair validation, rate limits
  - Position sync: startup sync, empty exchange, paper mode skip
  - Data fallback: WS failure -> REST
"""
import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# LiveMarketDataManager tests
# ---------------------------------------------------------------------------

from core.live_market_data import LiveMarketDataManager


class TestLiveMarketDataManagerInit:
    """Test LiveMarketDataManager initialization."""

    def test_init_default(self):
        mgr = LiveMarketDataManager()
        assert mgr._exchanges == {}
        assert mgr._market_data_service is None
        assert mgr._latest == {}
        assert mgr._tick_callbacks == []
        assert mgr._running is False

    def test_init_with_exchanges(self):
        ex = {"kraken": MagicMock()}
        mgr = LiveMarketDataManager(exchanges=ex)
        assert mgr._exchanges == ex

    def test_init_with_rest_interval(self):
        mgr = LiveMarketDataManager(rest_fallback_interval_s=10.0)
        assert mgr._rest_fallback_interval_s == 10.0


class TestLiveMarketDataGetLatest:
    """Test get_latest and data storage."""

    def test_get_latest_no_data(self):
        mgr = LiveMarketDataManager()
        assert mgr.get_latest("BTC/USD") is None

    def test_get_latest_with_data(self):
        mgr = LiveMarketDataManager()
        now = datetime.now(timezone.utc)
        mgr._latest["BTC/USD"] = {
            "symbol": "BTC/USD",
            "price": 65000.0,
            "bid": 64990.0,
            "ask": 65010.0,
            "spread": 20.0,
            "volume": 1234.5,
            "timestamp": now,
            "_mono": time.monotonic(),
        }
        result = mgr.get_latest("BTC/USD")
        assert result is not None
        assert result["price"] == 65000.0
        assert result["bid"] == 64990.0
        assert result["ask"] == 65010.0
        assert result["spread"] == 20.0
        assert result["volume"] == 1234.5
        assert "age_ms" in result
        assert result["age_ms"] >= 0

    def test_get_latest_returns_copy(self):
        """Modifying returned dict should not affect internal state."""
        mgr = LiveMarketDataManager()
        mgr._latest["BTC/USD"] = {
            "symbol": "BTC/USD",
            "price": 65000.0,
            "bid": 64990.0,
            "ask": 65010.0,
            "spread": 20.0,
            "volume": 0.0,
            "timestamp": datetime.now(timezone.utc),
            "_mono": time.monotonic(),
        }
        result = mgr.get_latest("BTC/USD")
        result["price"] = 99999.0
        assert mgr._latest["BTC/USD"]["price"] == 65000.0


class TestLiveMarketDataStaleness:
    """Test is_stale detection."""

    def test_stale_no_data(self):
        mgr = LiveMarketDataManager()
        assert mgr.is_stale("BTC/USD") is True

    def test_not_stale_fresh_data(self):
        mgr = LiveMarketDataManager()
        mgr._latest["BTC/USD"] = {
            "price": 65000.0,
            "_mono": time.monotonic(),
        }
        assert mgr.is_stale("BTC/USD", max_age_ms=5000) is False

    def test_stale_old_data(self):
        mgr = LiveMarketDataManager()
        mgr._latest["BTC/USD"] = {
            "price": 65000.0,
            "_mono": time.monotonic() - 10.0,  # 10 seconds ago
        }
        assert mgr.is_stale("BTC/USD", max_age_ms=5000) is True

    def test_stale_custom_threshold(self):
        mgr = LiveMarketDataManager()
        mgr._latest["BTC/USD"] = {
            "price": 65000.0,
            "_mono": time.monotonic() - 0.5,  # 500ms ago
        }
        assert mgr.is_stale("BTC/USD", max_age_ms=1000) is False
        assert mgr.is_stale("BTC/USD", max_age_ms=100) is True


class TestLiveMarketDataOrderbook:
    """Test get_orderbook functionality."""

    def test_orderbook_no_data(self):
        mgr = LiveMarketDataManager()
        assert mgr.get_orderbook("BTC/USD") is None

    def test_orderbook_from_l2_feed(self):
        mgr = LiveMarketDataManager()

        # Mock L2 feed with an OrderBook-like object
        mock_book = MagicMock()
        mock_book.bids = [MagicMock(price=65000.0, size=1.0), MagicMock(price=64990.0, size=2.0)]
        mock_book.asks = [MagicMock(price=65010.0, size=0.5), MagicMock(price=65020.0, size=1.5)]
        mock_book.mid_price = 65005.0
        mock_book.spread_bps = 1.5

        mock_l2 = MagicMock()
        mock_l2.get_book.return_value = mock_book
        mgr.set_l2_feed(mock_l2)

        result = mgr.get_orderbook("BTC/USD")
        assert result is not None
        assert result["mid_price"] == 65005.0
        assert result["spread_bps"] == 1.5
        assert len(result["bids"]) == 2
        assert len(result["asks"]) == 2

    def test_orderbook_fallback_to_cached(self):
        mgr = LiveMarketDataManager()
        mgr._orderbooks["BTC/USD"] = {
            "bids": [[65000.0, 1.0]],
            "asks": [[65010.0, 0.5]],
            "mid_price": 65005.0,
            "spread_bps": 1.5,
        }
        result = mgr.get_orderbook("BTC/USD")
        assert result is not None
        assert result["mid_price"] == 65005.0


class TestLiveMarketDataCallbacks:
    """Test on_tick callback registration and firing."""

    def test_register_callback(self):
        mgr = LiveMarketDataManager()
        cb = MagicMock()
        mgr.on_tick(cb)
        assert len(mgr._tick_callbacks) == 1

    def test_fire_callbacks(self):
        mgr = LiveMarketDataManager()
        cb1 = MagicMock()
        cb2 = MagicMock()
        mgr.on_tick(cb1)
        mgr.on_tick(cb2)

        data = {"symbol": "BTC/USD", "price": 65000.0}
        mgr._fire_tick_callbacks(data)

        cb1.assert_called_once_with(data)
        cb2.assert_called_once_with(data)

    def test_callback_error_does_not_propagate(self):
        mgr = LiveMarketDataManager()
        cb_bad = MagicMock(side_effect=ValueError("test error"))
        cb_good = MagicMock()
        mgr.on_tick(cb_bad)
        mgr.on_tick(cb_good)

        data = {"symbol": "BTC/USD", "price": 65000.0}
        mgr._fire_tick_callbacks(data)

        cb_bad.assert_called_once()
        cb_good.assert_called_once_with(data)


class TestLiveMarketDataCoinbaseTick:
    """Test Coinbase ticker handling."""

    def test_handle_coinbase_tick(self):
        mgr = LiveMarketDataManager()
        cb = MagicMock()
        mgr.on_tick(cb)

        tick = {
            "symbol": "BTC-AUD",
            "bid": 100000.0,
            "ask": 100020.0,
            "last": 100010.0,
            "volume_24h": 500.0,
            "timestamp": datetime.now(timezone.utc),
        }
        mgr._handle_coinbase_tick(tick)

        result = mgr.get_latest("BTC/AUD")
        assert result is not None
        assert result["price"] == 100010.0
        assert result["bid"] == 100000.0
        assert result["ask"] == 100020.0
        assert result["spread"] == 20.0
        assert result["volume"] == 500.0
        cb.assert_called_once()

    def test_handle_coinbase_tick_zero_last(self):
        """When last=0, price falls back to mid of bid/ask."""
        mgr = LiveMarketDataManager()
        tick = {
            "symbol": "ETH-AUD",
            "bid": 5000.0,
            "ask": 5010.0,
            "last": 0,
            "volume_24h": 0,
        }
        mgr._handle_coinbase_tick(tick)
        result = mgr.get_latest("ETH/AUD")
        assert result is not None
        assert result["price"] == 5005.0


class TestLiveMarketDataSubscribe:
    """Test subscribe with mocked WS connectors."""

    @pytest.mark.asyncio
    async def test_subscribe_ws_fails_starts_rest_fallback(self):
        mock_mds = AsyncMock()
        mock_mds.fetch_ticker = AsyncMock(return_value={
            "bid": 65000.0, "ask": 65010.0, "last": 65005.0, "baseVolume": 100.0
        })
        mgr = LiveMarketDataManager(market_data_service=mock_mds, rest_fallback_interval_s=0.1)

        # Patch WS start to fail
        with patch.object(mgr, "_start_ws", return_value=False):
            result = await mgr.subscribe(["BTC/USD"], exchange="kraken")
            assert result is True  # REST fallback still succeeds
            assert "kraken" in mgr._rest_fallback_tasks

        # Let REST poll run briefly
        await asyncio.sleep(0.25)
        await mgr.disconnect()

        # Should have received data via REST
        latest = mgr.get_latest("BTC/USD")
        assert latest is not None
        assert latest["price"] == 65005.0

    @pytest.mark.asyncio
    async def test_subscribe_ws_succeeds(self):
        mgr = LiveMarketDataManager()

        with patch.object(mgr, "_start_ws", return_value=True):
            result = await mgr.subscribe(["BTC/USD"], exchange="kraken")
            assert result is True

        await mgr.disconnect()


class TestLiveMarketDataDisconnect:
    """Test disconnect lifecycle."""

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self):
        mgr = LiveMarketDataManager()
        mgr._running = True

        # Add a mock WS connector
        mock_conn = AsyncMock()
        mock_conn.disconnect = AsyncMock()
        mgr._ws_connectors["kraken"] = mock_conn

        await mgr.disconnect()
        assert mgr._running is False
        mock_conn.disconnect.assert_called_once()
        assert len(mgr._ws_connectors) == 0


class TestLiveMarketDataProperties:
    """Test property accessors."""

    def test_connected_exchanges_empty(self):
        mgr = LiveMarketDataManager()
        assert mgr.connected_exchanges == []

    def test_connected_exchanges_with_connected(self):
        mgr = LiveMarketDataManager()
        mock_conn = MagicMock()
        mock_conn.connected = True
        mgr._ws_connectors["kraken"] = mock_conn
        assert mgr.connected_exchanges == ["kraken"]

    def test_subscribed_symbols(self):
        mgr = LiveMarketDataManager()
        mgr._subscriptions = {"BTC/USD": "kraken", "ETH/USD": "kraken"}
        assert sorted(mgr.subscribed_symbols) == ["BTC/USD", "ETH/USD"]


# ---------------------------------------------------------------------------
# ExchangeValidator tests
# ---------------------------------------------------------------------------

from core.exchange_validator import ExchangeValidator, validate_exchange_startup


class TestExchangeValidatorApiKeys:
    """Test API key validation."""

    @pytest.mark.asyncio
    async def test_api_keys_valid(self):
        mock_ex = AsyncMock()
        mock_ex.fetch_balance = AsyncMock(return_value={"USD": {"total": 1000}})
        validator = ExchangeValidator({"kraken": mock_ex})
        result = await validator.validate_api_keys("kraken", mock_ex)
        assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_api_keys_invalid(self):
        mock_ex = AsyncMock()
        mock_ex.fetch_balance = AsyncMock(side_effect=Exception("Invalid API key"))
        validator = ExchangeValidator({"kraken": mock_ex})
        result = await validator.validate_api_keys("kraken", mock_ex)
        assert result["ok"] is False
        assert "Invalid API key" in result["error"]

    @pytest.mark.asyncio
    async def test_api_keys_no_endpoint(self):
        mock_ex = MagicMock(spec=[])  # No methods at all
        validator = ExchangeValidator({"kraken": mock_ex})
        result = await validator.validate_api_keys("kraken", mock_ex)
        assert result["ok"] is False


class TestExchangeValidatorBalances:
    """Test balance validation."""

    @pytest.mark.asyncio
    async def test_sufficient_balance(self):
        mock_ex = AsyncMock()
        mock_ex.fetch_balance = AsyncMock(return_value={
            "total": {"USD": 500.0, "BTC": 0.01},
            "info": {},
        })
        validator = ExchangeValidator()
        result = await validator.validate_balances("kraken", mock_ex, min_balance_usd=100.0)
        assert result["ok"] is True
        assert result["total_usd"] > 100.0

    @pytest.mark.asyncio
    async def test_insufficient_balance(self):
        mock_ex = AsyncMock()
        mock_ex.fetch_balance = AsyncMock(return_value={
            "total": {"USD": 5.0},
            "info": {},
        })
        validator = ExchangeValidator()
        result = await validator.validate_balances("kraken", mock_ex, min_balance_usd=100.0)
        assert result["ok"] is False
        assert "insufficient" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_balance_aud_conversion(self):
        mock_ex = AsyncMock()
        mock_ex.fetch_balance = AsyncMock(return_value={
            "total": {"AUD": 1000.0},
            "info": {},
        })
        validator = ExchangeValidator()
        result = await validator.validate_balances("kraken", mock_ex, min_balance_usd=10.0)
        assert result["ok"] is True
        assert result["total_usd"] == pytest.approx(650.0, rel=0.1)


class TestExchangeValidatorPairs:
    """Test trading pair validation."""

    @pytest.mark.asyncio
    async def test_pairs_available(self):
        mock_ex = AsyncMock()
        mock_ex.load_markets = AsyncMock()
        mock_ex.markets = {"BTC/USD": {}, "ETH/USD": {}}
        validator = ExchangeValidator()
        result = await validator.validate_trading_pairs("kraken", mock_ex, ["BTC/USD"])
        assert result["ok"] is True
        assert "BTC/USD" in result["available"]

    @pytest.mark.asyncio
    async def test_pairs_unavailable(self):
        mock_ex = AsyncMock()
        mock_ex.load_markets = AsyncMock()
        mock_ex.markets = {"BTC/USD": {}}
        validator = ExchangeValidator()
        result = await validator.validate_trading_pairs("kraken", mock_ex, ["BTC/USD", "DOGE/USD"])
        assert result["ok"] is False
        assert "DOGE/USD" in result["unavailable"]


class TestExchangeValidatorRateLimits:
    """Test rate limit validation."""

    @pytest.mark.asyncio
    async def test_rate_limits_ok(self):
        mock_ex = AsyncMock()
        mock_ex.fetch_ticker = AsyncMock(return_value={"last": 65000.0})
        validator = ExchangeValidator()
        result = await validator.validate_rate_limits("kraken", mock_ex)
        assert result["ok"] is True
        assert result["latency_ms"] >= 0  # mock returns instantly, latency may round to 0

    @pytest.mark.asyncio
    async def test_rate_limits_error(self):
        mock_ex = AsyncMock()
        mock_ex.fetch_ticker = AsyncMock(side_effect=Exception("rate limited"))
        validator = ExchangeValidator()
        result = await validator.validate_rate_limits("kraken", mock_ex)
        assert result["ok"] is False


class TestExchangeValidatorRunAll:
    """Test the full run_all validation."""

    @pytest.mark.asyncio
    async def test_run_all_passes(self):
        mock_ex = AsyncMock()
        mock_ex.fetch_balance = AsyncMock(return_value={"total": {"USD": 1000.0}, "info": {}})
        mock_ex.load_markets = AsyncMock()
        mock_ex.markets = {"BTC/USD": {}}
        mock_ex.fetch_ticker = AsyncMock(return_value={"last": 65000.0})

        report = await validate_exchange_startup(
            exchanges={"kraken": mock_ex},
            min_balance_usd=10.0,
            pairs=["BTC/USD"],
        )
        assert report["all_passed"] is True
        assert report["exchanges"]["kraken"]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_run_all_fails_bad_keys(self):
        mock_ex = AsyncMock()
        mock_ex.fetch_balance = AsyncMock(side_effect=Exception("bad key"))
        mock_ex.load_markets = AsyncMock()
        mock_ex.markets = {"BTC/USD": {}}
        mock_ex.fetch_ticker = AsyncMock(return_value={"last": 65000.0})

        report = await validate_exchange_startup(
            exchanges={"kraken": mock_ex},
            min_balance_usd=10.0,
            pairs=["BTC/USD"],
        )
        assert report["all_passed"] is False
        assert len(report["exchanges"]["kraken"]["issues"]) > 0


class TestExchangeValidatorTimeout:
    """Test timeout handling."""

    @pytest.mark.asyncio
    async def test_check_with_timeout(self):
        validator = ExchangeValidator()

        async def slow_check():
            await asyncio.sleep(10)
            return {"ok": True}

        result = await validator._check_with_timeout(slow_check(), timeout=0.1, label="test")
        assert result["ok"] is False
        assert "timed out" in result["error"]


# ---------------------------------------------------------------------------
# Position sync tests
# ---------------------------------------------------------------------------

class TestPositionSync:
    """Test _sync_positions_from_exchange."""

    def _make_system(self, run_mode="live"):
        """Create a minimal mock system with position sync method."""
        system = MagicMock()
        system.config = MagicMock()
        system.config.run_mode = run_mode
        system.config.trading_pairs = ["BTC/USD", "ETH/USD"]
        system.positions = {}
        system.exchanges = {}
        system.exchange_manager = None
        # Bind the real method
        from unified_trading_system import UnifiedSystemArchitecture
        system._sync_positions_from_exchange = UnifiedSystemArchitecture._sync_positions_from_exchange.__get__(system)
        return system

    @pytest.mark.asyncio
    async def test_paper_mode_skips_sync(self):
        system = self._make_system(run_mode="paper")
        result = await system._sync_positions_from_exchange()
        assert result["synced"] is True
        assert result["positions_found"] == 0

    @pytest.mark.asyncio
    async def test_live_mode_syncs_from_exchange(self):
        system = self._make_system(run_mode="live")
        mock_ex = AsyncMock()
        mock_ex.fetch_balance = AsyncMock(return_value={
            "total": {
                "BTC": 0.5,
                "ETH": 10.0,
                "USD": 5000.0,  # should be skipped (not a base currency)
            },
            "info": {},
        })
        system.exchanges = {"kraken": mock_ex}

        result = await system._sync_positions_from_exchange()
        assert result["synced"] is True
        assert result["positions_found"] == 2
        assert "BTC/USD" in system.positions
        assert system.positions["BTC/USD"]["quantity"] == 0.5
        assert "ETH/USD" in system.positions
        assert system.positions["ETH/USD"]["quantity"] == 10.0

    @pytest.mark.asyncio
    async def test_live_mode_empty_exchange(self):
        system = self._make_system(run_mode="live")
        mock_ex = AsyncMock()
        mock_ex.fetch_balance = AsyncMock(return_value={"total": {}, "info": {}})
        system.exchanges = {"kraken": mock_ex}

        result = await system._sync_positions_from_exchange()
        assert result["synced"] is True
        assert result["positions_found"] == 0

    @pytest.mark.asyncio
    async def test_live_mode_exchange_error(self):
        system = self._make_system(run_mode="live")
        mock_ex = AsyncMock()
        mock_ex.fetch_balance = AsyncMock(side_effect=Exception("connection error"))
        system.exchanges = {"kraken": mock_ex}

        result = await system._sync_positions_from_exchange()
        # Should not crash, just log warning
        assert result["synced"] is True
        assert result["positions_found"] == 0


# ---------------------------------------------------------------------------
# REST fallback tests
# ---------------------------------------------------------------------------

class TestRESTFallback:
    """Test REST fallback when WebSocket fails."""

    @pytest.mark.asyncio
    async def test_rest_fallback_fetch_ticker_via_mds(self):
        mock_mds = AsyncMock()
        mock_mds.fetch_ticker = AsyncMock(return_value={
            "bid": 65000.0, "ask": 65010.0, "last": 65005.0
        })
        mgr = LiveMarketDataManager(market_data_service=mock_mds)
        result = await mgr._fetch_rest_ticker("BTC/USD", "kraken")
        assert result is not None
        assert result["last"] == 65005.0

    @pytest.mark.asyncio
    async def test_rest_fallback_fetch_ticker_via_ccxt(self):
        mock_ex = AsyncMock()
        mock_ex.fetch_ticker = AsyncMock(return_value={
            "bid": 65000.0, "ask": 65010.0, "last": 65005.0
        })
        mgr = LiveMarketDataManager(exchanges={"kraken": mock_ex})
        result = await mgr._fetch_rest_ticker("BTC/USD", "kraken")
        assert result is not None
        assert result["last"] == 65005.0

    @pytest.mark.asyncio
    async def test_rest_fallback_both_fail(self):
        mock_mds = AsyncMock()
        mock_mds.fetch_ticker = AsyncMock(side_effect=Exception("mds fail"))
        mock_ex = AsyncMock()
        mock_ex.fetch_ticker = AsyncMock(side_effect=Exception("ccxt fail"))
        mgr = LiveMarketDataManager(
            exchanges={"kraken": mock_ex},
            market_data_service=mock_mds,
        )
        result = await mgr._fetch_rest_ticker("BTC/USD", "kraken")
        assert result is None

    @pytest.mark.asyncio
    async def test_rest_poll_loop_populates_data(self):
        mock_mds = AsyncMock()
        mock_mds.fetch_ticker = AsyncMock(return_value={
            "bid": 65000.0, "ask": 65010.0, "last": 65005.0, "baseVolume": 100.0
        })
        mgr = LiveMarketDataManager(
            market_data_service=mock_mds,
            rest_fallback_interval_s=0.05,
        )
        mgr._running = True
        mgr._rest_fallback_active["kraken"] = True

        task = asyncio.create_task(mgr._rest_poll_loop(["BTC/USD"], "kraken"))
        await asyncio.sleep(0.15)
        mgr._running = False
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        latest = mgr.get_latest("BTC/USD")
        assert latest is not None
        assert latest["price"] == 65005.0
        assert latest.get("_source") == "rest"


# ---------------------------------------------------------------------------
# L2 feed integration test
# ---------------------------------------------------------------------------

class TestL2FeedIntegration:
    """Test set_l2_feed and L2 tick polling."""

    def test_set_l2_feed(self):
        mgr = LiveMarketDataManager()
        mock_l2 = MagicMock()
        mgr.set_l2_feed(mock_l2)
        assert mgr._l2_feed is mock_l2

    def test_orderbook_l2_none_returns_none(self):
        mgr = LiveMarketDataManager()
        mock_l2 = MagicMock()
        mock_l2.get_book.return_value = None
        mgr.set_l2_feed(mock_l2)
        # L2 returns None, and no cached orderbook
        assert mgr.get_orderbook("BTC/USD") is None
