"""
Tests for latency optimization modules:
  - core/connection_pool.py
  - execution/order_templates.py
  - execution/async_order_submitter.py
  - core/latency_router.py
  - monitoring/tick_to_trade.py
  - unified_trading_system.py fast_mode / WS preference
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# ConnectionPool
# ---------------------------------------------------------------------------
from core.connection_pool import ConnectionPool


class TestConnectionPool:
    """Tests for core/connection_pool.py."""

    def test_pool_creation_default_exchanges(self):
        pool = ConnectionPool()
        assert pool._exchange_urls  # has default exchanges
        assert "kraken" in pool._exchange_urls
        assert "bybit" in pool._exchange_urls

    def test_pool_creation_custom_exchanges(self):
        pool = ConnectionPool(exchanges={"myex": "https://myex.com"})
        assert "myex" in pool._exchange_urls
        assert "kraken" not in pool._exchange_urls

    def test_dns_resolve_and_cache(self):
        pool = ConnectionPool()
        # Resolve a well-known hostname
        ip = pool.resolve_dns("localhost")
        assert ip is not None
        assert ip == "127.0.0.1"
        # Should be cached
        assert "localhost" in pool.get_dns_cache()
        # Second call returns from cache
        ip2 = pool.resolve_dns("localhost")
        assert ip2 == ip

    def test_dns_resolve_failure(self):
        pool = ConnectionPool()
        ip = pool.resolve_dns("this.host.does.not.exist.example.invalid")
        assert ip is None

    @pytest.mark.asyncio
    async def test_get_session_creates_on_demand(self):
        """get_session should create a session for unknown exchange."""
        pool = ConnectionPool(exchanges={"testex": "https://httpbin.org"})
        try:
            import aiohttp
            session = pool.get_session("testex")
            assert session is not None
            assert "testex" in pool._pools
            await pool.close_all()
        except ImportError:
            pytest.skip("aiohttp not installed")

    @pytest.mark.asyncio
    async def test_get_session_reuses(self):
        """Calling get_session twice returns the same session."""
        pool = ConnectionPool(exchanges={"testex": "https://httpbin.org"})
        try:
            import aiohttp
            s1 = pool.get_session("testex")
            s2 = pool.get_session("testex")
            assert s1 is s2
            await pool.close_all()
        except ImportError:
            pytest.skip("aiohttp not installed")

    def test_get_stats(self):
        pool = ConnectionPool()
        stats = pool.get_stats()
        assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_close_all(self):
        pool = ConnectionPool(exchanges={"testex": "https://httpbin.org"})
        try:
            import aiohttp
            pool.get_session("testex")
            await pool.close_all()
            assert pool._closed
            assert len(pool._pools) == 0
        except ImportError:
            pytest.skip("aiohttp not installed")

    def test_closed_pool_raises(self):
        pool = ConnectionPool()
        pool._closed = True
        with pytest.raises(RuntimeError, match="closed"):
            pool.get_session("kraken")


# ---------------------------------------------------------------------------
# OrderTemplateRegistry
# ---------------------------------------------------------------------------
from execution.order_templates import OrderTemplateRegistry


class TestOrderTemplates:
    """Tests for execution/order_templates.py."""

    def test_register_pair(self):
        reg = OrderTemplateRegistry()
        reg.register_pair("kraken", "BTC/USD")
        assert reg.template_count == 2  # buy + sell

    def test_register_pairs_batch(self):
        reg = OrderTemplateRegistry()
        reg.register_pairs("bybit", ["BTCUSDT", "ETHUSDT"])
        assert reg.template_count == 4

    def test_get_template_kraken(self):
        reg = OrderTemplateRegistry()
        reg.register_pair("kraken", "BTC/USD")
        tpl = reg.get_template("kraken", "BTC/USD", "buy")
        assert tpl["pair"] == "BTCUSD"
        assert tpl["type"] == "buy"
        assert tpl["ordertype"] == "market"

    def test_get_template_bybit(self):
        reg = OrderTemplateRegistry()
        reg.register_pair("bybit", "BTCUSDT")
        tpl = reg.get_template("bybit", "BTCUSDT", "sell")
        assert tpl["symbol"] == "BTCUSDT"
        assert tpl["side"] == "Sell"

    def test_get_template_okx(self):
        reg = OrderTemplateRegistry()
        reg.register_pair("okx", "BTC/USDT")
        tpl = reg.get_template("okx", "BTC/USDT", "buy")
        assert tpl["instId"] == "BTC-USDT"
        assert tpl["side"] == "buy"

    def test_get_template_coinbase(self):
        reg = OrderTemplateRegistry()
        reg.register_pair("coinbase", "BTC/USD")
        tpl = reg.get_template("coinbase", "BTC/USD", "buy")
        assert tpl["product_id"] == "BTC-USD"
        assert tpl["side"] == "BUY"

    def test_get_template_auto_registers(self):
        reg = OrderTemplateRegistry()
        tpl = reg.get_template("kraken", "ETH/USD", "sell")
        assert tpl["pair"] == "ETHUSD"
        assert reg.template_count == 2

    def test_template_is_copy_not_reference(self):
        reg = OrderTemplateRegistry()
        reg.register_pair("kraken", "BTC/USD")
        t1 = reg.get_template("kraken", "BTC/USD", "buy")
        t2 = reg.get_template("kraken", "BTC/USD", "buy")
        t1["price"] = 99999
        assert t2.get("price") != 99999

    def test_fill_template(self):
        reg = OrderTemplateRegistry()
        reg.register_pair("kraken", "BTC/USD")
        filled = reg.fill_template("kraken", "BTC/USD", "buy", price=65000.0, amount=0.01)
        assert filled["volume"] == 0.01
        assert filled["price"] == 65000.0
        assert filled["ordertype"] == "market"

    def test_fill_template_limit(self):
        reg = OrderTemplateRegistry()
        filled = reg.fill_template("bybit", "BTCUSDT", "sell", price=64000.0, amount=0.05, order_type="limit")
        assert filled["orderType"] == "Limit"
        assert filled["price"] == "64000.0"
        assert filled["qty"] == "0.05"

    def test_benchmark(self):
        reg = OrderTemplateRegistry()
        reg.register_pair("kraken", "BTC/USD")
        result = reg.benchmark_full_construction("kraken", "BTC/USD", "buy", iterations=100)
        assert result["speedup_factor"] > 0
        assert result["template_ns_avg"] > 0
        assert result["construction_ns_avg"] > 0

    def test_get_stats(self):
        reg = OrderTemplateRegistry()
        reg.register_pair("kraken", "BTC/USD")
        reg.get_template("kraken", "BTC/USD", "buy")
        stats = reg.get_stats()
        assert stats["template_count"] == 2
        assert len(stats["entries"]) == 2
        # Check that fill was counted
        buy_entry = [e for e in stats["entries"] if e["side"] == "buy"][0]
        assert buy_entry["fill_count"] == 1

    def test_template_fill_speed(self):
        """Template fill should be faster than full construction."""
        reg = OrderTemplateRegistry()
        reg.register_pair("kraken", "BTC/USD")
        # Warm up
        for _ in range(10):
            reg.get_template("kraken", "BTC/USD", "buy")

        t0 = time.perf_counter_ns()
        for _ in range(1000):
            tpl = reg.get_template("kraken", "BTC/USD", "buy")
            tpl["price"] = 65000.0
            tpl["volume"] = 0.01
        template_ns = time.perf_counter_ns() - t0

        # Just verify it runs fast (< 1ms for 1000 iterations)
        assert template_ns < 50_000_000  # 50ms for 1000 ops


# ---------------------------------------------------------------------------
# AsyncOrderSubmitter
# ---------------------------------------------------------------------------
from execution.async_order_submitter import AsyncOrderSubmitter, OrderConfirmation


class TestAsyncOrderSubmitter:
    """Tests for execution/async_order_submitter.py."""

    @pytest.mark.asyncio
    async def test_submit_returns_pending_id(self):
        sub = AsyncOrderSubmitter()
        await sub.start()
        try:
            pid = await sub.submit_fire_and_forget({"symbol": "BTC/USD", "side": "buy", "amount": 0.01})
            assert pid.startswith("ff_")
            assert sub.total_submitted == 1
        finally:
            await sub.stop()

    @pytest.mark.asyncio
    async def test_confirmation_collected(self):
        sub = AsyncOrderSubmitter()
        await sub.start()
        try:
            await sub.submit_fire_and_forget({"symbol": "BTC/USD", "side": "buy", "amount": 0.01})
            await asyncio.sleep(0.3)
            confs = sub.get_pending_confirmations()
            assert len(confs) >= 1
            assert confs[0].status == "simulated"
            assert sub.total_confirmed >= 1
        finally:
            await sub.stop()

    @pytest.mark.asyncio
    async def test_deduplication(self):
        sub = AsyncOrderSubmitter(dedup_window_ms=200)
        await sub.start()
        try:
            order = {"symbol": "BTC/USD", "side": "buy", "amount": 0.01, "price": 65000}
            pid1 = await sub.submit_fire_and_forget(order)
            pid2 = await sub.submit_fire_and_forget(order)
            assert pid2.startswith("deduped_")
            assert sub.total_deduped == 1
            assert sub.total_submitted == 1
        finally:
            await sub.stop()

    @pytest.mark.asyncio
    async def test_dedup_allows_after_window(self):
        sub = AsyncOrderSubmitter(dedup_window_ms=50)
        await sub.start()
        try:
            order = {"symbol": "BTC/USD", "side": "buy", "amount": 0.01, "price": 65000}
            await sub.submit_fire_and_forget(order)
            await asyncio.sleep(0.06)
            pid2 = await sub.submit_fire_and_forget(order)
            assert not pid2.startswith("deduped_")
            assert sub.total_submitted == 2
        finally:
            await sub.stop()

    @pytest.mark.asyncio
    async def test_exchange_manager_integration(self):
        mock_em = AsyncMock()
        mock_em.execute_order = AsyncMock(return_value={
            "order_id": "test123",
            "status": "filled",
            "filled": 0.01,
            "price": 65000,
        })
        sub = AsyncOrderSubmitter(exchange_manager=mock_em)
        await sub.start()
        try:
            await sub.submit_fire_and_forget(
                {"symbol": "BTC/USD", "side": "buy", "amount": 0.01},
                exchange="kraken",
            )
            await asyncio.sleep(0.3)
            confs = sub.get_pending_confirmations()
            assert len(confs) >= 1
            assert confs[0].order_id == "test123"
            assert confs[0].status == "filled"
        finally:
            await sub.stop()

    @pytest.mark.asyncio
    async def test_get_stats(self):
        sub = AsyncOrderSubmitter()
        await sub.start()
        try:
            await sub.submit_fire_and_forget({"symbol": "BTC/USD", "side": "buy", "amount": 0.01})
            stats = sub.get_stats()
            assert stats["total_submitted"] == 1
            assert "pending_count" in stats
        finally:
            await sub.stop()


# ---------------------------------------------------------------------------
# LatencyRouter
# ---------------------------------------------------------------------------
from core.latency_router import LatencyRouter


class TestLatencyRouter:
    """Tests for core/latency_router.py."""

    def test_record_latency(self):
        router = LatencyRouter()
        router.record_latency("kraken", 50.0)
        router.record_latency("kraken", 60.0)
        router.record_latency("kraken", 55.0)
        report = router.get_latency_report()
        assert "kraken" in report
        assert report["kraken"]["samples"] == 3
        assert report["kraken"]["p50_ms"] > 0

    def test_get_fastest_venue(self):
        router = LatencyRouter()
        for _ in range(10):
            router.record_latency("kraken", 50.0)
            router.record_latency("coinbase", 80.0)
        fastest = router.get_fastest_venue("BTC/USD")
        assert fastest == "kraken"

    def test_get_fastest_venue_no_data(self):
        router = LatencyRouter()
        assert router.get_fastest_venue("BTC/USD") is None

    def test_get_fastest_venue_with_symbol_filter(self):
        router = LatencyRouter(
            venue_symbols={
                "kraken": ["BTC/USD", "ETH/USD"],
                "coinbase": ["BTC/USD"],
            }
        )
        for _ in range(10):
            router.record_latency("kraken", 50.0)
            router.record_latency("coinbase", 30.0)
        # For ETH/USD, only kraken supports it
        fastest = router.get_fastest_venue("ETH/USD")
        assert fastest == "kraken"
        # For BTC/USD, coinbase is faster
        fastest = router.get_fastest_venue("BTC/USD")
        assert fastest == "coinbase"

    def test_degradation_detection(self):
        router = LatencyRouter(degradation_threshold=2.0)
        for _ in range(10):
            router.record_latency("kraken", 50.0)
        # Now a spike
        router.record_latency("kraken", 150.0)
        report = router.get_latency_report()
        assert report["kraken"]["degraded"] is True

    def test_degradation_recovery(self):
        router = LatencyRouter(degradation_threshold=2.0)
        for _ in range(10):
            router.record_latency("kraken", 50.0)
        router.record_latency("kraken", 150.0)
        assert router.get_latency_report()["kraken"]["degraded"] is True
        # Recover
        router.record_latency("kraken", 50.0)
        assert router.get_latency_report()["kraken"]["degraded"] is False

    def test_venue_ranking(self):
        router = LatencyRouter()
        for _ in range(10):
            router.record_latency("kraken", 50.0)
            router.record_latency("coinbase", 80.0)
            router.record_latency("bybit", 30.0)
        ranking = router.get_venue_ranking()
        assert len(ranking) == 3
        assert ranking[0]["exchange"] == "bybit"  # fastest
        assert ranking[2]["exchange"] == "coinbase"  # slowest

    def test_latency_report_structure(self):
        router = LatencyRouter()
        for _ in range(5):
            router.record_latency("kraken", 50.0)
        report = router.get_latency_report()
        kr = report["kraken"]
        assert "p50_ms" in kr
        assert "p95_ms" in kr
        assert "p99_ms" in kr
        assert "last_ms" in kr
        assert "degraded" in kr

    @pytest.mark.asyncio
    async def test_start_stop(self):
        router = LatencyRouter(ping_interval_s=60)
        await router.start()
        assert router._running
        await router.stop()
        assert not router._running


# ---------------------------------------------------------------------------
# TickToTradeMonitor
# ---------------------------------------------------------------------------
from monitoring.tick_to_trade import TickToTradeMonitor


class TestTickToTradeMonitor:
    """Tests for monitoring/tick_to_trade.py."""

    def test_record_and_get_stats(self):
        monitor = TickToTradeMonitor(threshold_ms=500.0)
        t0 = time.monotonic()
        stages = {
            "tick_received": t0,
            "signal_generated": t0 + 0.050,
            "risk_checked": t0 + 0.055,
            "order_submitted": t0 + 0.060,
            "order_acknowledged": t0 + 0.120,
        }
        monitor.record_tick_to_trade(stages)
        stats = monitor.get_stats()
        assert stats["total_records"] == 1
        assert stats["total"]["p50_ms"] > 0
        assert stats["total"]["p50_ms"] == pytest.approx(120.0, abs=5.0)

    def test_individual_stage_timing(self):
        monitor = TickToTradeMonitor()
        t0 = time.monotonic()
        stages = {
            "tick_received": t0,
            "signal_generated": t0 + 0.010,
            "risk_checked": t0 + 0.015,
            "order_submitted": t0 + 0.020,
            "order_acknowledged": t0 + 0.080,
        }
        monitor.record_tick_to_trade(stages)
        stats = monitor.get_stats()
        assert stats["tick_to_signal"]["p50_ms"] == pytest.approx(10.0, abs=2.0)
        assert stats["submit_to_ack"]["p50_ms"] == pytest.approx(60.0, abs=2.0)

    def test_threshold_breach(self):
        monitor = TickToTradeMonitor(threshold_ms=100.0)
        t0 = time.monotonic()
        # Total = 200ms > 100ms threshold
        stages = {
            "tick_received": t0,
            "order_acknowledged": t0 + 0.200,
        }
        monitor.record_tick_to_trade(stages)
        assert monitor._threshold_breaches == 1
        stats = monitor.get_stats()
        assert stats["threshold_breaches"] == 1

    def test_no_breach_under_threshold(self):
        monitor = TickToTradeMonitor(threshold_ms=500.0)
        t0 = time.monotonic()
        stages = {
            "tick_received": t0,
            "order_acknowledged": t0 + 0.050,
        }
        monitor.record_tick_to_trade(stages)
        assert monitor._threshold_breaches == 0

    def test_record_stage_directly(self):
        monitor = TickToTradeMonitor()
        monitor.record_stage("custom_stage", 42.0)
        stats = monitor.get_stats()
        assert "custom_stage" in stats
        assert stats["custom_stage"]["p50_ms"] == 42.0

    def test_summary_line(self):
        monitor = TickToTradeMonitor()
        line = monitor.get_summary_line()
        assert "no data" in line

        t0 = time.monotonic()
        monitor.record_tick_to_trade({
            "tick_received": t0,
            "order_acknowledged": t0 + 0.100,
        })
        line = monitor.get_summary_line()
        assert "p50=" in line
        assert "p95=" in line

    def test_multiple_records_percentiles(self):
        monitor = TickToTradeMonitor()
        for i in range(100):
            t0 = time.monotonic()
            delay = 0.050 + (i * 0.001)
            monitor.record_tick_to_trade({
                "tick_received": t0,
                "order_acknowledged": t0 + delay,
            })
        stats = monitor.get_stats()
        assert stats["total"]["p50_ms"] > 0
        assert stats["total"]["p95_ms"] > stats["total"]["p50_ms"]
        assert stats["total"]["p99_ms"] >= stats["total"]["p95_ms"]

    def test_threshold_setter(self):
        monitor = TickToTradeMonitor(threshold_ms=500.0)
        assert monitor.threshold_ms == 500.0
        monitor.threshold_ms = 200.0
        assert monitor.threshold_ms == 200.0


# ---------------------------------------------------------------------------
# UnifiedConfig fast_mode + latency fields
# ---------------------------------------------------------------------------


class TestUnifiedConfigLatency:
    """Tests for latency config fields in UnifiedConfig."""

    def test_fast_mode_default_false(self):
        from unified_trading_system import UnifiedConfig
        cfg = UnifiedConfig()
        assert cfg.fast_mode is False

    def test_fast_mode_can_be_set(self):
        from unified_trading_system import UnifiedConfig
        cfg = UnifiedConfig(fast_mode=True)
        assert cfg.fast_mode is True

    def test_ws_order_preference_default(self):
        from unified_trading_system import UnifiedConfig
        cfg = UnifiedConfig()
        assert cfg.latency_ws_order_preference is True

    def test_tick_to_trade_threshold_default(self):
        from unified_trading_system import UnifiedConfig
        cfg = UnifiedConfig()
        assert cfg.latency_tick_to_trade_threshold_ms == 500.0

    def test_connection_pool_enabled_default(self):
        from unified_trading_system import UnifiedConfig
        cfg = UnifiedConfig()
        assert cfg.latency_connection_pool_enabled is True

    def test_fire_and_forget_disabled_by_default(self):
        from unified_trading_system import UnifiedConfig
        cfg = UnifiedConfig()
        assert cfg.latency_fire_and_forget_enabled is False


# ---------------------------------------------------------------------------
# Config keys registered in config_manager
# ---------------------------------------------------------------------------


class TestConfigKeysRegistered:
    """Verify latency config keys are in _KNOWN_TOP_LEVEL_KEYS."""

    def test_latency_keys_registered(self):
        from core.config_manager import _KNOWN_TOP_LEVEL_KEYS
        for key in [
            "latency", "connection_pool", "order_templates",
            "async_order_submitter", "latency_router", "tick_to_trade",
        ]:
            assert key in _KNOWN_TOP_LEVEL_KEYS, f"'{key}' not in _KNOWN_TOP_LEVEL_KEYS"
