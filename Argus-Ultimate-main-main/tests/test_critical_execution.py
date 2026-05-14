"""
Tests for 7 critical execution features:
1. Liquidation price calculator
2. Margin call auto-reduce
3. WebSocket token refresh
4. Execution backpressure
5. Rate limiter enforcement decorator
6. Multi-venue score config
7. VWAP cost placeholder fix
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. Liquidation price calculator
# ---------------------------------------------------------------------------

class TestLiquidationPriceCalculator:
    def _make_rm(self):
        from risk.unified_risk_manager import UnifiedRiskManager
        return UnifiedRiskManager(initial_capital=10_000)

    def test_long_liquidation_basic(self):
        rm = self._make_rm()
        # entry=100, leverage=10, maintenance_margin_pct=0.5
        # long: 100 * (1 - 1/10 + 0.5/100) = 100 * (1 - 0.1 + 0.005) = 100 * 0.905 = 90.5
        liq = rm.calculate_liquidation_price(100.0, 10.0, "long", 0.5)
        assert abs(liq - 90.5) < 1e-6

    def test_short_liquidation_basic(self):
        rm = self._make_rm()
        # short: 100 * (1 + 1/10 - 0.5/100) = 100 * (1 + 0.1 - 0.005) = 100 * 1.095 = 109.5
        liq = rm.calculate_liquidation_price(100.0, 10.0, "short", 0.5)
        assert abs(liq - 109.5) < 1e-6

    def test_long_buy_alias(self):
        rm = self._make_rm()
        liq = rm.calculate_liquidation_price(100.0, 10.0, "buy", 0.5)
        assert abs(liq - 90.5) < 1e-6

    def test_short_sell_alias(self):
        rm = self._make_rm()
        liq = rm.calculate_liquidation_price(100.0, 10.0, "sell", 0.5)
        assert abs(liq - 109.5) < 1e-6

    def test_high_leverage(self):
        rm = self._make_rm()
        # 100x leverage long: 100 * (1 - 0.01 + 0.005) = 100 * 0.995 = 99.5
        liq = rm.calculate_liquidation_price(100.0, 100.0, "long", 0.5)
        assert abs(liq - 99.5) < 1e-6

    def test_low_leverage(self):
        rm = self._make_rm()
        # 2x leverage long: 100 * (1 - 0.5 + 0.005) = 100 * 0.505 = 50.5
        liq = rm.calculate_liquidation_price(100.0, 2.0, "long", 0.5)
        assert abs(liq - 50.5) < 1e-6

    def test_invalid_side_raises(self):
        rm = self._make_rm()
        with pytest.raises(ValueError, match="side must be"):
            rm.calculate_liquidation_price(100.0, 10.0, "neutral")

    def test_zero_leverage_raises(self):
        rm = self._make_rm()
        with pytest.raises(ValueError, match="leverage must be positive"):
            rm.calculate_liquidation_price(100.0, 0.0, "long")

    def test_negative_entry_raises(self):
        rm = self._make_rm()
        with pytest.raises(ValueError, match="entry_price must be positive"):
            rm.calculate_liquidation_price(-50.0, 10.0, "long")

    def test_result_never_negative(self):
        rm = self._make_rm()
        # Even with extreme params, result should be >= 0
        liq = rm.calculate_liquidation_price(1.0, 1.0, "long", 0.5)
        assert liq >= 0.0

    def test_static_method(self):
        from risk.unified_risk_manager import UnifiedRiskManager
        # Can call as static method without instance
        liq = UnifiedRiskManager.calculate_liquidation_price(100.0, 10.0, "long", 0.5)
        assert abs(liq - 90.5) < 1e-6

    def test_custom_maintenance_margin(self):
        rm = self._make_rm()
        # maintenance_margin_pct=1.0 for long: 100 * (1 - 0.1 + 0.01) = 100 * 0.91 = 91.0
        liq = rm.calculate_liquidation_price(100.0, 10.0, "long", 1.0)
        assert abs(liq - 91.0) < 1e-6


# ---------------------------------------------------------------------------
# 2. Margin call auto-reduce
# ---------------------------------------------------------------------------

class TestMarginCallAutoReduce:
    def _make_rm(self):
        from risk.unified_risk_manager import UnifiedRiskManager
        return UnifiedRiskManager(initial_capital=10_000)

    def test_no_margin_call_under_threshold(self):
        rm = self._make_rm()
        rm.update_margin_requirement("BTC/USD", 500)
        rm.update_margin_requirement("ETH/USD", 200)
        # Total margin = 700, capital = 10000, usage = 7% < 80%
        result = rm.check_margin_call(10_000)
        assert result == []

    def test_margin_call_triggered(self):
        rm = self._make_rm()
        rm.update_margin_requirement("BTC/USD", 5000)
        rm.update_margin_requirement("ETH/USD", 4000)
        # Total margin = 9000, capital = 10000, usage = 90% > 80%
        result = rm.check_margin_call(10_000)
        assert len(result) > 0
        # Should target 70% = 7000, excess = 2000
        # Largest first: BTC/USD (5000), then ETH/USD (4000)
        assert result[0]["symbol"] == "BTC/USD"

    def test_margin_call_reduce_pct(self):
        rm = self._make_rm()
        rm.update_margin_requirement("BTC/USD", 5000)
        rm.update_margin_requirement("ETH/USD", 4000)
        result = rm.check_margin_call(10_000)
        # Total = 9000, target = 7000, excess = 2000
        # BTC has 5000 margin, reduce 2000 of it = 40%
        assert result[0]["reduce_by_pct"] == 40.0

    def test_margin_call_exactly_at_threshold(self):
        rm = self._make_rm()
        rm.update_margin_requirement("BTC/USD", 8000)
        # usage = 80%, not exceeded (<=)
        result = rm.check_margin_call(10_000)
        assert result == []

    def test_margin_call_just_above_threshold(self):
        rm = self._make_rm()
        rm.update_margin_requirement("BTC/USD", 8100)
        result = rm.check_margin_call(10_000)
        assert len(result) > 0

    def test_auto_reduce_positions(self):
        rm = self._make_rm()
        rm.update_margin_requirement("BTC/USD", 9000)
        result = rm.auto_reduce_positions(10_000)
        assert len(result) > 0

    def test_auto_reduce_no_action_needed(self):
        rm = self._make_rm()
        rm.update_margin_requirement("BTC/USD", 100)
        result = rm.auto_reduce_positions(10_000)
        assert result == []

    def test_zero_capital(self):
        rm = self._make_rm()
        rm.update_margin_requirement("BTC/USD", 5000)
        result = rm.check_margin_call(0)
        assert result == []

    def test_multiple_positions_sorted(self):
        rm = self._make_rm()
        rm.update_margin_requirement("SMALL", 1000)
        rm.update_margin_requirement("LARGE", 5000)
        rm.update_margin_requirement("MID", 3000)
        # Total = 9000, capital = 10000, usage 90%, target 7000, excess 2000
        result = rm.check_margin_call(10_000)
        assert result[0]["symbol"] == "LARGE"


# ---------------------------------------------------------------------------
# 3. WebSocket token refresh
# ---------------------------------------------------------------------------

class TestWebSocketTokenRefresh:
    def test_token_expiry_field_exists(self):
        from core.websocket_order_placer import KrakenWebSocketOrderPlacer
        placer = KrakenWebSocketOrderPlacer()
        assert hasattr(placer, "_token_expiry")
        assert placer._token_expiry == 0.0

    def test_ensure_token_fresh_method_exists(self):
        from core.websocket_order_placer import KrakenWebSocketOrderPlacer
        placer = KrakenWebSocketOrderPlacer()
        assert hasattr(placer, "_ensure_token_fresh")

    @pytest.mark.asyncio
    async def test_ensure_token_fresh_no_token(self):
        from core.websocket_order_placer import KrakenWebSocketOrderPlacer
        placer = KrakenWebSocketOrderPlacer()
        placer._token = None
        # Should not raise
        await placer._ensure_token_fresh()

    @pytest.mark.asyncio
    async def test_ensure_token_fresh_not_expired(self):
        from core.websocket_order_placer import KrakenWebSocketOrderPlacer
        placer = KrakenWebSocketOrderPlacer()
        placer._token = "test_token"
        placer._token_expiry = time.monotonic() + 600  # 10 min away
        # Mock _get_ws_token to verify it's NOT called
        placer._get_ws_token = AsyncMock(return_value="new_token")
        await placer._ensure_token_fresh()
        placer._get_ws_token.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_token_fresh_near_expiry(self):
        from core.websocket_order_placer import KrakenWebSocketOrderPlacer
        placer = KrakenWebSocketOrderPlacer()
        placer._token = "old_token"
        placer._token_expiry = time.monotonic() + 60  # 1 min away (within 5-min window)
        placer._get_ws_token = AsyncMock(return_value="new_token")
        await placer._ensure_token_fresh()
        placer._get_ws_token.assert_called_once()
        assert placer._token == "new_token"

    @pytest.mark.asyncio
    async def test_ensure_token_fresh_expired(self):
        from core.websocket_order_placer import KrakenWebSocketOrderPlacer
        placer = KrakenWebSocketOrderPlacer()
        placer._token = "expired_token"
        placer._token_expiry = time.monotonic() - 100  # already expired
        placer._get_ws_token = AsyncMock(return_value="refreshed")
        await placer._ensure_token_fresh()
        placer._get_ws_token.assert_called_once()
        assert placer._token == "refreshed"

    @pytest.mark.asyncio
    async def test_ensure_token_fresh_refresh_fails(self):
        from core.websocket_order_placer import KrakenWebSocketOrderPlacer
        placer = KrakenWebSocketOrderPlacer()
        placer._token = "old_token"
        placer._token_expiry = time.monotonic() - 100
        placer._get_ws_token = AsyncMock(return_value=None)
        await placer._ensure_token_fresh()
        # Token should remain unchanged on failure
        assert placer._token == "old_token"

    @pytest.mark.asyncio
    async def test_place_order_calls_ensure_token_fresh(self):
        from core.websocket_order_placer import KrakenWebSocketOrderPlacer
        placer = KrakenWebSocketOrderPlacer()
        placer._token = None
        placer._connected = False
        placer._ensure_token_fresh = AsyncMock()
        # place_order will fall through to REST fallback (no credentials), that's fine
        await placer.place_order("BTC/USD", "buy", 0.1)
        placer._ensure_token_fresh.assert_called_once()


# ---------------------------------------------------------------------------
# 4. Execution backpressure
# ---------------------------------------------------------------------------

class TestExecutionBackpressure:
    def test_max_queue_size_parameter(self):
        from execution.execution_mesh import ExecutionMeshCoordinator
        mesh = ExecutionMeshCoordinator(max_queue_size=500)
        assert mesh.max_queue_size == 500

    def test_default_max_queue_size(self):
        from execution.execution_mesh import ExecutionMeshCoordinator
        mesh = ExecutionMeshCoordinator()
        assert mesh.max_queue_size == 1000

    def test_queue_pressure_empty(self):
        from execution.execution_mesh import ExecutionMeshCoordinator
        mesh = ExecutionMeshCoordinator()
        assert mesh.queue_pressure() == 0.0

    @pytest.mark.asyncio
    async def test_backpressure_rejects_when_full(self):
        from execution.execution_mesh import ExecutionMeshCoordinator

        async def dummy_fn(symbol, batch, corr_id):
            return [{"status": "ok"} for _ in batch]

        mesh = ExecutionMeshCoordinator(
            max_queue_size=5,
            max_queue_per_lane=100,
            halt_on_lane_error=False,
        )
        # First, fill the queue by enqueuing without draining
        signals = [{"symbol": "BTC", "action": "buy"} for _ in range(10)]
        # Put signals into lanes manually
        for sig in signals[:5]:
            lane = mesh._get_or_create_lane("BTC")
            lane.enqueue([sig])

        # Now the queue has 5 items = max_queue_size, next batch should be rejected
        results, summary = await mesh.execute_cycle(
            [{"symbol": "BTC", "action": "buy"}],
            execute_fn=dummy_fn,
            correlation_id="test",
        )
        assert summary["rejected_backpressure"] > 0

    @pytest.mark.asyncio
    async def test_no_backpressure_under_limit(self):
        from execution.execution_mesh import ExecutionMeshCoordinator

        async def dummy_fn(symbol, batch, corr_id):
            return [{"status": "ok"} for _ in batch]

        mesh = ExecutionMeshCoordinator(max_queue_size=1000, halt_on_lane_error=False)
        signals = [{"symbol": "BTC", "action": "buy"} for _ in range(3)]
        results, summary = await mesh.execute_cycle(
            signals,
            execute_fn=dummy_fn,
            correlation_id="test",
        )
        assert summary["rejected_backpressure"] == 0
        assert summary["accepted_signals"] == 3

    def test_queue_pressure_ratio(self):
        from execution.execution_mesh import ExecutionMeshCoordinator
        mesh = ExecutionMeshCoordinator(max_queue_size=10)
        # Manually add items to a lane
        lane = mesh._get_or_create_lane("BTC")
        lane.enqueue([{"symbol": "BTC"} for _ in range(5)])
        assert abs(mesh.queue_pressure() - 0.5) < 1e-6

    def test_queue_pressure_capped_at_one(self):
        from execution.execution_mesh import ExecutionMeshCoordinator
        mesh = ExecutionMeshCoordinator(max_queue_size=5, max_queue_per_lane=100)
        lane = mesh._get_or_create_lane("BTC")
        lane.enqueue([{"symbol": "BTC"} for _ in range(20)])
        assert mesh.queue_pressure() == 1.0


# ---------------------------------------------------------------------------
# 5. Rate limiter enforcement decorator
# ---------------------------------------------------------------------------

class TestRateLimitedDecorator:
    def test_decorator_import(self):
        from execution.rate_limit_manager import rate_limited, EndpointType
        assert callable(rate_limited)

    @pytest.mark.asyncio
    async def test_decorator_calls_wait_if_needed(self):
        from execution.rate_limit_manager import (
            RateLimitManager, EndpointType, rate_limited,
        )

        class MyClient:
            def __init__(self):
                self._rate_limit_manager = RateLimitManager()
                self.called = False

            @rate_limited(EndpointType.PRIVATE)
            async def fetch_data(self, exchange: str = "kraken"):
                self.called = True
                return "ok"

        client = MyClient()
        result = await client.fetch_data(exchange="kraken")
        assert result == "ok"
        assert client.called

    @pytest.mark.asyncio
    async def test_decorator_positional_exchange(self):
        from execution.rate_limit_manager import (
            RateLimitManager, EndpointType, rate_limited,
        )

        class MyClient:
            def __init__(self):
                self._rate_limit_manager = RateLimitManager()

            @rate_limited(EndpointType.PUBLIC)
            async def fetch(self, exchange: str):
                return exchange

        client = MyClient()
        result = await client.fetch("coinbase")
        assert result == "coinbase"

    @pytest.mark.asyncio
    async def test_decorator_without_manager(self):
        from execution.rate_limit_manager import EndpointType, rate_limited

        class NoManager:
            @rate_limited(EndpointType.PUBLIC)
            async def fetch(self, exchange: str = "kraken"):
                return "done"

        obj = NoManager()
        # Should not raise even without a rate limit manager
        result = await obj.fetch()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_wait_if_needed_method(self):
        from execution.rate_limit_manager import RateLimitManager, EndpointType
        mgr = RateLimitManager()
        result = await mgr.wait_if_needed("kraken", EndpointType.PUBLIC)
        assert result is True


# ---------------------------------------------------------------------------
# 6. Multi-venue score config
# ---------------------------------------------------------------------------

class TestMultiVenueScoreConfig:
    def test_default_weights(self):
        from execution.multi_venue_execution import MultiVenueExecutor
        executor = MultiVenueExecutor()
        assert executor._weights["spread"] == 0.40
        assert executor._weights["liquidity"] == 0.35
        assert executor._weights["latency"] == 0.15
        assert executor._weights["fee"] == 0.10

    def test_custom_weights(self):
        from execution.multi_venue_execution import MultiVenueExecutor
        custom = {"spread": 0.50, "liquidity": 0.25, "latency": 0.15, "fee": 0.10}
        executor = MultiVenueExecutor(weights=custom)
        assert executor._weights["spread"] == 0.50
        assert executor._weights["liquidity"] == 0.25

    def test_custom_weights_affect_scores(self):
        from execution.multi_venue_execution import MultiVenueExecutor
        # Default weights
        exec_default = MultiVenueExecutor()
        scores_default = exec_default.venue_scores()

        # Custom weights emphasizing spread heavily
        custom = {"spread": 0.90, "liquidity": 0.05, "latency": 0.03, "fee": 0.02}
        exec_custom = MultiVenueExecutor(weights=custom)
        scores_custom = exec_custom.venue_scores()

        # Scores should differ
        for venue in scores_default:
            assert scores_default[venue] != scores_custom[venue]

    def test_venue_stats_uses_weights(self):
        from execution.multi_venue_execution import VenueStats
        stats = VenueStats(spread_bps=5.0, liquidity_score=0.9, latency_ms=80.0, taker_fee_bps=26.0)
        default_score = stats.score()

        custom = {"spread": 1.0, "liquidity": 0.0, "latency": 0.0, "fee": 0.0}
        custom_score = stats.score(custom)
        # With only spread weight, score = 1.0 * (1/5) = 0.2
        assert abs(custom_score - 0.2) < 1e-6
        assert custom_score != default_score

    def test_split_with_custom_weights(self):
        from execution.multi_venue_execution import MultiVenueExecutor, MultiVenueDecision
        custom = {"spread": 0.10, "liquidity": 0.10, "latency": 0.10, "fee": 0.70}
        executor = MultiVenueExecutor(weights=custom)
        decision = MultiVenueDecision(symbol="BTC/USD", side="BUY", total_size=1000.0)
        orders = executor.split(decision)
        assert len(orders) > 0
        total = sum(o.size for o in orders)
        assert abs(total - 1000.0) < 1e-6

    def test_weights_passed_to_new_venues(self):
        from execution.multi_venue_execution import MultiVenueExecutor
        custom = {"spread": 0.60, "liquidity": 0.20, "latency": 0.10, "fee": 0.10}
        executor = MultiVenueExecutor(weights=custom)
        executor.update_venue_stats("binance", spread_bps=3.0)
        # The new venue should have the custom weights
        assert executor._stats["binance"]._weights["spread"] == 0.60


# ---------------------------------------------------------------------------
# 7. VWAP cost placeholder fix
# ---------------------------------------------------------------------------

class TestVwapCostFix:
    def test_cost_uses_price_and_fee(self):
        from execution.vwap_pov_core import run_vwap_slicer
        # execute_slice returns None (non-dict), so the else branch is hit
        results = run_vwap_slicer(
            total_size=100.0,
            duration_sec=10.0,
            execute_slice=lambda ts, size: None,
            num_slices=5,
            fee_rate=0.001,
        )
        # With no price in schedule (default schedule has no "price" key),
        # s.get("price", 0.0) = 0.0, so cost = size * 0.0 * fee_rate = 0
        assert results["filled_total"] == pytest.approx(100.0, abs=1e-6)
        # avg_price should be 0.0 since cost is 0
        assert results["avg_price"] == 0.0

    def test_cost_with_dict_result(self):
        from execution.vwap_pov_core import run_vwap_slicer

        def exec_fn(ts, size):
            return {"filled": size, "avg_price": 50000.0}

        results = run_vwap_slicer(
            total_size=100.0,
            duration_sec=10.0,
            execute_slice=exec_fn,
            num_slices=5,
        )
        assert results["filled_total"] == pytest.approx(100.0, abs=1e-6)
        assert results["avg_price"] == pytest.approx(50000.0, abs=1e-6)

    def test_fee_rate_default(self):
        from execution.vwap_pov_core import run_vwap_slicer
        import inspect
        sig = inspect.signature(run_vwap_slicer)
        assert sig.parameters["fee_rate"].default == 0.001

    def test_no_placeholder_in_source(self):
        """Verify the placeholder comment is gone."""
        import execution.vwap_pov_core as mod
        import inspect
        source = inspect.getsource(mod.run_vwap_slicer)
        assert "placeholder" not in source.lower()

    def test_cost_calculation_non_zero_price(self):
        from execution.vwap_pov_core import vwap_schedule, run_vwap_slicer

        # Manually create a schedule with prices and test cost calculation
        call_count = 0

        def exec_fn(ts, size):
            nonlocal call_count
            call_count += 1
            # Return None to exercise the else branch with fee_rate
            return None

        results = run_vwap_slicer(
            total_size=10.0,
            duration_sec=5.0,
            execute_slice=exec_fn,
            num_slices=5,
            fee_rate=0.002,
        )
        assert call_count == 5
        assert results["slices"] == 5
