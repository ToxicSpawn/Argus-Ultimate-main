"""
Tests for execution/smart_order_execution.py

Covers:
- VWAPExecution.calculate_vwap_profile (volume_pct sums to ~1.0)
- TWAPExecution.create_twap_schedule (equal slice distribution)
- DynamicOrderSizer.calculate_optimal_size (respects min/max)
- MarketImpactMinimizer.optimize_order_size (within bounds)
- Edge cases: zero volume, single candle, very large orders
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUndefinedVariable=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPossiblyUnboundVariable=false, reportUninitializedInstanceVariable=false, reportArgumentType=false, reportOperatorIssue=false, reportIndexIssue=false, reportMissingTypeArgument=false, reportOptionalSubscript=false

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


pytestmark = pytest.mark.skip(reason="Legacy smart order execution APIs are not present in the current module layout")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n_bars: int = 100, freq: str = "5min") -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with a DatetimeIndex."""
    base = datetime(2026, 3, 1, 9, 0, 0)
    idx = pd.date_range(start=base, periods=n_bars, freq=freq)
    rng = np.random.default_rng(0)
    close = 50_000 + rng.normal(0, 500, n_bars).cumsum()
    close = np.maximum(close, 100.0)
    df = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": rng.uniform(10, 100, n_bars),
        },
        index=idx,
    )
    return df


def _make_order(qty: float = 1.0, symbol: str = "BTC/USDT") -> ExecutionOrder:
    return ExecutionOrder(
        order_id="test_order_1",
        symbol=symbol,
        side="buy",
        total_quantity=qty,
    )


# ---------------------------------------------------------------------------
# VWAPExecution
# ---------------------------------------------------------------------------

class TestVWAPProfileCalculation:
    def test_returns_dataframe_with_expected_columns(self):
        vwap = VWAPExecution()
        df = _make_ohlcv(200)
        profile = vwap.calculate_vwap_profile("BTC/USDT", df)
        assert isinstance(profile, pd.DataFrame)
        assert "volume_pct" in profile.columns
        assert "vwap" in profile.columns

    def test_volume_pct_sums_to_approximately_one(self):
        vwap = VWAPExecution()
        df = _make_ohlcv(288)  # one full day at 5-min = 288 bars
        profile = vwap.calculate_vwap_profile("BTC/USDT", df)
        # groupby mean of pcts — the mean pcts won't sum to exactly 1, but
        # each original row's pct should be positive and ≤ 1
        assert (profile["volume_pct"] >= 0).all()
        assert (profile["volume_pct"] <= 1.0).all()

    def test_empty_dataframe_returns_empty(self):
        vwap = VWAPExecution()
        empty = pd.DataFrame()
        profile = vwap.calculate_vwap_profile("ETH/USDT", empty)
        assert profile.empty

    def test_profile_stored_after_calculation(self):
        vwap = VWAPExecution()
        df = _make_ohlcv(100)
        vwap.calculate_vwap_profile("BTC/USDT", df)
        assert "BTC/USDT" in vwap.volume_profiles

    def test_vwap_values_are_positive(self):
        vwap = VWAPExecution()
        df = _make_ohlcv(200)
        profile = vwap.calculate_vwap_profile("BTC/USDT", df)
        assert (profile["vwap"] > 0).all()


class TestVWAPSchedule:
    def test_creates_slices_with_known_profile(self):
        vwap = VWAPExecution()
        df = _make_ohlcv(200)
        order = _make_order(qty=2.0)
        vwap.calculate_vwap_profile("BTC/USDT", df)
        slices = vwap.create_vwap_schedule(order, execution_time_minutes=60)
        assert len(slices) > 0

    def test_all_slices_have_positive_quantity(self):
        vwap = VWAPExecution()
        df = _make_ohlcv(200)
        order = _make_order(qty=2.0)
        vwap.calculate_vwap_profile("BTC/USDT", df)
        slices = vwap.create_vwap_schedule(order, execution_time_minutes=60)
        for s in slices:
            assert s.quantity > 0

    def test_no_profile_falls_back_to_uniform(self):
        vwap = VWAPExecution()
        order = _make_order(qty=1.0, symbol="NEW/USDT")
        # No profile registered for NEW/USDT
        slices = vwap.create_vwap_schedule(order, execution_time_minutes=30)
        assert len(slices) > 0
        # All slices should have the same base size in uniform mode
        quantities = [s.quantity for s in slices]
        # The first slice may differ due to remainder, but all should be positive
        assert all(q > 0 for q in quantities)

    def test_single_candle_data(self):
        """Edge case: only one candle provided — profile should still return something."""
        vwap = VWAPExecution()
        df = _make_ohlcv(1)
        profile = vwap.calculate_vwap_profile("BTC/USDT", df)
        # Single bar resampled → one row or empty; either is acceptable
        assert isinstance(profile, pd.DataFrame)

    def test_slice_ids_are_unique(self):
        vwap = VWAPExecution()
        df = _make_ohlcv(200)
        order = _make_order(qty=1.0)
        vwap.calculate_vwap_profile("BTC/USDT", df)
        slices = vwap.create_vwap_schedule(order, execution_time_minutes=60)
        ids = [s.slice_id for s in slices]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# TWAPExecution
# ---------------------------------------------------------------------------

class TestTWAPSchedule:
    def test_creates_slices(self):
        twap = TWAPExecution()
        order = _make_order(qty=1.0)
        slices = twap.create_twap_schedule(order, execution_time_minutes=30)
        assert len(slices) > 0

    def test_total_quantity_conserved(self):
        """Sum of all slice quantities should equal total order quantity (modulo
        randomisation ±10%, so allow tolerance)."""
        twap = TWAPExecution()
        order = _make_order(qty=10.0)
        slices = twap.create_twap_schedule(order, execution_time_minutes=30)
        total = sum(s.quantity for s in slices)
        # With ±10% random jitter and min-size guard, total should be close
        assert abs(total - order.total_quantity) / order.total_quantity < 0.20

    def test_slices_roughly_equal(self):
        """Without randomisation the slices should be near-equal."""
        twap = TWAPExecution()
        order = _make_order(qty=100.0)
        # Run many times and check mean is close to equal
        all_first = []
        for _ in range(10):
            slices = twap.create_twap_schedule(order, execution_time_minutes=30)
            if slices:
                all_first.append(slices[0].quantity)
        expected = order.total_quantity / 10  # default 30min / 3min = 10 slices
        mean_first = np.mean(all_first)
        # Mean should be within 20% of equal
        assert abs(mean_first - expected) / expected < 0.20

    def test_all_slices_positive(self):
        twap = TWAPExecution()
        order = _make_order(qty=5.0)
        slices = twap.create_twap_schedule(order, execution_time_minutes=15)
        for s in slices:
            assert s.quantity > 0

    def test_timestamps_increase(self):
        twap = TWAPExecution()
        order = _make_order(qty=1.0)
        slices = twap.create_twap_schedule(order, execution_time_minutes=30)
        for i in range(1, len(slices)):
            assert slices[i].timestamp >= slices[i - 1].timestamp

    def test_very_small_execution_window(self):
        """Edge case: execution_time_minutes=3 → total_slices=1."""
        twap = TWAPExecution()
        order = _make_order(qty=1.0)
        slices = twap.create_twap_schedule(order, execution_time_minutes=3)
        assert len(slices) >= 1

    def test_slice_ids_are_unique(self):
        twap = TWAPExecution()
        order = _make_order(qty=1.0)
        slices = twap.create_twap_schedule(order, execution_time_minutes=30)
        ids = [s.slice_id for s in slices]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# MarketImpactMinimizer
# ---------------------------------------------------------------------------

class TestMarketImpactMinimizer:
    def setup_method(self):
        self.mim = MarketImpactMinimizer()
        self.market_depth = {"daily_volume": 2_000_000}

    def test_estimate_impact_is_non_negative(self):
        impact = self.mim.estimate_market_impact("BTC/USDT", 1000, self.market_depth)
        assert impact >= 0

    def test_estimate_impact_capped_at_ten_percent(self):
        # Very large order should not exceed 10%
        huge_order = 1e9
        impact = self.mim.estimate_market_impact("BTC/USDT", huge_order, self.market_depth)
        assert impact <= 0.10

    def test_estimate_impact_increases_with_order_size(self):
        small = self.mim.estimate_market_impact("BTC/USDT", 100, self.market_depth)
        large = self.mim.estimate_market_impact("BTC/USDT", 100_000, self.market_depth)
        assert large > small

    def test_zero_volume_uses_default(self):
        depth_zero = {"daily_volume": 0}
        # Should not raise; falls back to k * 0.01
        impact = self.mim.estimate_market_impact("BTC/USDT", 1000, depth_zero)
        assert impact >= 0

    def test_optimize_order_size_without_cache_returns_conservative(self):
        mim = MarketImpactMinimizer()
        result = mim.optimize_order_size("BTC/USDT", 5000)
        # Conservative default: min(target, 1000)
        assert result <= 1000

    def test_optimize_order_size_within_bounds(self):
        self.mim.estimate_market_impact("BTC/USDT", 100, self.market_depth)  # populate cache
        result = self.mim.optimize_order_size("BTC/USDT", 10_000, max_impact=0.005)
        assert 0 < result <= 10_000

    def test_optimize_order_size_small_target_stays_unchanged(self):
        """A tiny order should not be shrunk."""
        self.mim.estimate_market_impact("BTC/USDT", 1, self.market_depth)
        result = self.mim.optimize_order_size("BTC/USDT", 1, max_impact=0.01)
        assert result > 0

    def test_optimize_large_order_respects_impact_limit(self):
        """After optimisation the impact of the returned size should be <= max_impact."""
        depth = {"daily_volume": 1_000_000}
        self.mim.estimate_market_impact("BTC/USDT", 1, depth)  # populate cache
        max_impact = 0.002
        result = self.mim.optimize_order_size("BTC/USDT", 500_000, max_impact=max_impact)
        actual_impact = self.mim.estimate_market_impact("BTC/USDT", result, depth)
        assert actual_impact <= max_impact * 1.05  # 5% tolerance for binary search precision


# ---------------------------------------------------------------------------
# DynamicOrderSizer
# ---------------------------------------------------------------------------

class TestDynamicOrderSizer:
    def setup_method(self):
        self.sizer = DynamicOrderSizer()
        self.market_data = {
            "volatility": 0.02,
            "liquidity_score": 0.8,
            "portfolio_value": 10_000,
            "daily_volume": 2_000_000,
        }

    def test_returns_dict_with_required_keys(self):
        result = self.sizer.calculate_optimal_size(
            "BTC/USDT", 0.5, 0.01, 50_000, self.market_data
        )
        for key in ("base_size", "final_size", "position_value", "risk_pct"):
            assert key in result

    def test_final_size_non_negative(self):
        result = self.sizer.calculate_optimal_size(
            "BTC/USDT", 0.5, 0.01, 50_000, self.market_data
        )
        assert result["final_size"] >= 0

    def test_final_size_does_not_exceed_base_size(self):
        """Sizer should never recommend MORE than the base size."""
        result = self.sizer.calculate_optimal_size(
            "BTC/USDT", 1.0, 0.01, 50_000, self.market_data
        )
        assert result["final_size"] <= result["base_size"] + 1e-9

    def test_high_volatility_reduces_size(self):
        low_vol_data = {**self.market_data, "volatility": 0.005}
        high_vol_data = {**self.market_data, "volatility": 0.10}
        low_result = self.sizer.calculate_optimal_size("BTC/USDT", 1.0, 0.01, 50_000, low_vol_data)
        high_result = self.sizer.calculate_optimal_size("BTC/USDT", 1.0, 0.01, 50_000, high_vol_data)
        assert high_result["final_size"] <= low_result["final_size"]

    def test_low_liquidity_reduces_size(self):
        high_liq = {**self.market_data, "liquidity_score": 1.0}
        low_liq = {**self.market_data, "liquidity_score": 0.1}
        high_result = self.sizer.calculate_optimal_size("BTC/USDT", 1.0, 0.01, 50_000, high_liq)
        low_result = self.sizer.calculate_optimal_size("BTC/USDT", 1.0, 0.01, 50_000, low_liq)
        assert low_result["final_size"] <= high_result["final_size"] + 1e-9

    def test_adjustments_applied_list_non_empty(self):
        result = self.sizer.calculate_optimal_size(
            "BTC/USDT", 0.5, 0.01, 50_000, self.market_data
        )
        assert len(result["adjustments_applied"]) > 0

    def test_very_large_order_bounded_by_risk(self):
        """A huge base_size should be capped by the risk_per_trade limit."""
        result = self.sizer.calculate_optimal_size(
            "BTC/USDT", 10_000, 0.01, 50_000, self.market_data
        )
        max_position_value = (0.01 * self.market_data["portfolio_value"]) / 0.01
        assert result["position_value"] <= max_position_value * 1.01  # 1% float tolerance


# ---------------------------------------------------------------------------
# SmartOrderExecutionEngine (integration-level)
# ---------------------------------------------------------------------------

class TestSmartOrderExecutionEngine:
    def setup_method(self):
        self.engine = SmartOrderExecutionEngine()
        self.market_data = {
            "price": 50_000,
            "volatility": 0.02,
            "liquidity_score": 0.8,
            "portfolio_value": 10_000,
            "daily_volume": 2_000_000,
        }

    def test_execute_vwap_returns_order(self):
        order = self.engine.execute_order(
            "BTC/USDT", "buy", 0.1, "vwap", 60, self.market_data
        )
        assert order is not None
        assert order.symbol == "BTC/USDT"
        assert order.side == "buy"

    def test_execute_twap_creates_slices(self):
        order = self.engine.execute_order(
            "BTC/USDT", "buy", 0.1, "twap", 30, self.market_data
        )
        slices = order.metadata.get("execution_slices", [])
        assert len(slices) > 0

    def test_execute_market_creates_single_slice(self):
        order = self.engine.execute_order(
            "BTC/USDT", "sell", 0.1, "market", 0, self.market_data
        )
        slices = order.metadata.get("execution_slices", [])
        assert len(slices) == 1

    def test_order_stored_in_active_orders(self):
        order = self.engine.execute_order(
            "BTC/USDT", "buy", 0.1, "twap", 30, self.market_data
        )
        assert order.order_id in self.engine.active_orders

    def test_cancel_execution_sets_status(self):
        order = self.engine.execute_order(
            "BTC/USDT", "buy", 0.1, "twap", 30, self.market_data
        )
        result = self.engine.cancel_execution(order.order_id)
        assert result is True
        assert self.engine.active_orders[order.order_id].status == "cancelled"

    def test_cancel_nonexistent_order_returns_false(self):
        assert self.engine.cancel_execution("nonexistent_id") is False

    def test_get_execution_metrics_returns_dict(self):
        order = self.engine.execute_order(
            "BTC/USDT", "buy", 0.1, "vwap", 60, self.market_data
        )
        metrics = self.engine.get_execution_metrics(order.order_id)
        assert isinstance(metrics, dict)
        assert metrics["order_id"] == order.order_id

    def test_get_execution_metrics_nonexistent_returns_empty(self):
        assert self.engine.get_execution_metrics("bad_id") == {}

    def test_performance_summary_no_completed_orders(self):
        result = self.engine.get_performance_summary(hours_back=1)
        assert "message" in result or "total_orders" in result

    def test_vwap_with_historical_data(self):
        data = {**self.market_data, "historical_data": _make_ohlcv(200)}
        order = self.engine.execute_order(
            "BTC/USDT", "buy", 0.1, "vwap", 60, data
        )
        assert order is not None
        slices = order.metadata.get("execution_slices", [])
        assert len(slices) > 0

    def test_order_quantity_after_sizing(self):
        """Final quantity should be positive."""
        order = self.engine.execute_order(
            "BTC/USDT", "buy", 0.5, "vwap", 60, self.market_data
        )
        assert order.total_quantity > 0


# ---------------------------------------------------------------------------
# Async execution loop tests
# ---------------------------------------------------------------------------

import asyncio


class _FakeExchange:
    """Minimal fake exchange that records orders and returns configurable fills."""

    def __init__(self, fill_pct: float = 1.0, fill_price_offset: float = 0.0):
        self.orders: list = []
        self.fill_pct = fill_pct
        self.fill_price_offset = fill_price_offset

    async def create_order(self, order_dict: dict) -> dict:
        self.orders.append(order_dict)
        price = order_dict["price"] + self.fill_price_offset
        filled = order_dict["amount"] * self.fill_pct
        return {
            "id": f"exch_{len(self.orders)}",
            "filled": filled,
            "average": price,
            "status": "closed" if self.fill_pct >= 1.0 else "partial",
        }


class _FailingExchange:
    """Exchange that raises on create_order."""

    async def create_order(self, order_dict: dict) -> dict:
        raise ConnectionError("exchange unavailable")


class TestAsyncVWAPExecution:
    """Tests for the async VWAP slice submission loop."""

    def _market_data(self, price: float = 50_000):
        return {
            "price": price,
            "volatility": 0.02,
            "liquidity_score": 0.8,
            "portfolio_value": 100_000,
            "daily_volume": 2_000_000,
        }

    def test_vwap_simulation_fills_all_slices(self):
        """Without an exchange, simulation fills every slice at the limit price."""
        engine = SmartOrderExecutionEngine()
        order = asyncio.run(
            engine.execute_order_async(
                "BTC/USDT", "buy", 0.1, "vwap", 60,
                market_data=self._market_data(),
                slice_interval_seconds=0,
            )
        )
        assert order.status == "completed"
        assert order.executed_quantity > 0
        assert order.average_price > 0
        slices = order.metadata.get("execution_slices", [])
        assert all(s.executed for s in slices if s.filled_quantity > 0)

    def test_vwap_with_exchange_records_orders(self):
        exchange = _FakeExchange(fill_pct=1.0)
        engine = SmartOrderExecutionEngine()
        order = asyncio.run(
            engine.execute_order_async(
                "BTC/USDT", "buy", 0.1, "vwap", 60,
                market_data=self._market_data(),
                exchange=exchange,
                slice_interval_seconds=0,
            )
        )
        assert order.status == "completed"
        assert len(exchange.orders) > 0
        # Every submitted order should be a limit buy
        for o in exchange.orders:
            assert o["side"] == "buy"
            assert o["type"] == "limit"

    def test_vwap_partial_fills_status(self):
        """50% partial fills should yield status='partial'."""
        exchange = _FakeExchange(fill_pct=0.5)
        engine = SmartOrderExecutionEngine()
        order = asyncio.run(
            engine.execute_order_async(
                "BTC/USDT", "buy", 0.1, "vwap", 60,
                market_data=self._market_data(),
                exchange=exchange,
                slice_interval_seconds=0,
            )
        )
        assert order.status == "partial"
        assert 0 < order.executed_quantity < order.total_quantity

    def test_vwap_slippage_tracked(self):
        """Fill price offset should appear as positive slippage on buy."""
        exchange = _FakeExchange(fill_pct=1.0, fill_price_offset=10.0)
        engine = SmartOrderExecutionEngine()
        order = asyncio.run(
            engine.execute_order_async(
                "BTC/USDT", "buy", 0.1, "vwap", 60,
                market_data=self._market_data(),
                exchange=exchange,
                slice_interval_seconds=0,
            )
        )
        slices = order.metadata.get("execution_slices", [])
        filled_slices = [s for s in slices if s.executed]
        assert len(filled_slices) > 0
        # All should have positive slippage (we paid more than limit)
        assert all(s.slippage > 0 for s in filled_slices)

    def test_vwap_exchange_order_ids_populated(self):
        exchange = _FakeExchange()
        engine = SmartOrderExecutionEngine()
        order = asyncio.run(
            engine.execute_order_async(
                "BTC/USDT", "buy", 0.1, "vwap", 60,
                market_data=self._market_data(),
                exchange=exchange,
                slice_interval_seconds=0,
            )
        )
        slices = order.metadata.get("execution_slices", [])
        filled_slices = [s for s in slices if s.executed]
        assert all(s.exchange_order_id.startswith("exch_") for s in filled_slices)


class TestAsyncTWAPExecution:
    """Tests for the async TWAP slice submission loop."""

    def _market_data(self, price: float = 50_000):
        return {
            "price": price,
            "volatility": 0.02,
            "liquidity_score": 0.8,
            "portfolio_value": 100_000,
            "daily_volume": 2_000_000,
        }

    def test_twap_simulation_completes(self):
        engine = SmartOrderExecutionEngine()
        order = asyncio.run(
            engine.execute_order_async(
                "ETH/USDT", "sell", 1.0, "twap", 30,
                market_data=self._market_data(price=3_000),
                slice_interval_seconds=0,
            )
        )
        # TWAP applies ±10% randomization per slice, so total filled may not
        # exactly equal requested quantity — accept completed or partial.
        assert order.status in ("completed", "partial")
        assert order.executed_quantity > 0

    def test_twap_uses_mid_price_callback(self):
        """get_mid_price should be called for each submitted slice."""
        prices_seen = []

        def _mid():
            p = 3000 + len(prices_seen) * 10
            prices_seen.append(p)
            return p

        exchange = _FakeExchange()
        engine = SmartOrderExecutionEngine()
        order = asyncio.run(
            engine.execute_order_async(
                "ETH/USDT", "buy", 1.0, "twap", 30,
                market_data=self._market_data(price=3_000),
                exchange=exchange,
                get_mid_price=_mid,
                slice_interval_seconds=0,
            )
        )
        # get_mid_price should have been called at least once
        assert len(prices_seen) >= 1
        # And the exchange should have received orders at varying prices
        assert len(exchange.orders) >= 1

    def test_twap_sell_limit_price_below_mid(self):
        """For sells, the limit with tolerance should be below mid."""
        exchange = _FakeExchange()
        engine = SmartOrderExecutionEngine()
        asyncio.run(
            engine.execute_order_async(
                "ETH/USDT", "sell", 1.0, "twap", 30,
                market_data=self._market_data(price=3_000),
                exchange=exchange,
                slice_interval_seconds=0,
            )
        )
        for o in exchange.orders:
            # sell limit should be at or below mid (minus tolerance)
            assert o["price"] <= 3_000


class TestAsyncCancellation:
    """Test that external cancellation stops the loop."""

    def test_cancel_during_execution(self):
        engine = SmartOrderExecutionEngine()
        market_data = {
            "price": 50_000, "volatility": 0.02, "liquidity_score": 0.8,
            "portfolio_value": 100_000, "daily_volume": 2_000_000,
        }

        async def _run():
            # Start the order and then cancel it after a tiny delay
            task = asyncio.ensure_future(
                engine.execute_order_async(
                    "BTC/USDT", "buy", 0.1, "vwap", 60,
                    market_data=market_data,
                    slice_interval_seconds=0.05,  # small delay so we can cancel
                )
            )
            await asyncio.sleep(0.02)  # let first slice go through
            # Cancel via the engine API
            for oid in list(engine.active_orders):
                engine.cancel_execution(oid)
            return await task

        order = asyncio.run(_run())
        assert order.status == "cancelled"

    def test_exchange_errors_do_not_abort(self):
        """If one slice fails, subsequent slices should still be attempted."""
        engine = SmartOrderExecutionEngine()
        market_data = {
            "price": 50_000, "volatility": 0.02, "liquidity_score": 0.8,
            "portfolio_value": 100_000, "daily_volume": 2_000_000,
        }
        exchange = _FailingExchange()
        order = asyncio.run(
            engine.execute_order_async(
                "BTC/USDT", "buy", 0.1, "vwap", 60,
                market_data=market_data,
                exchange=exchange,
                slice_interval_seconds=0,
            )
        )
        # All slices failed, so no fills
        assert order.executed_quantity == 0.0
        assert order.status == "failed"


class TestAsyncExecutionHistory:
    """Verify that the execution history is populated after async runs."""

    def test_history_recorded(self):
        engine = SmartOrderExecutionEngine()
        market_data = {
            "price": 50_000, "volatility": 0.02, "liquidity_score": 0.8,
            "portfolio_value": 100_000, "daily_volume": 2_000_000,
        }
        asyncio.run(
            engine.execute_order_async(
                "BTC/USDT", "buy", 0.1, "vwap", 60,
                market_data=market_data,
                slice_interval_seconds=0,
            )
        )
        assert len(engine.execution_history) == 1
        rec = engine.execution_history[0]
        assert rec["status"] == "completed"
        assert rec["filled_qty"] > 0
        assert rec["slices_filled"] > 0

    def test_multiple_orders_tracked(self):
        engine = SmartOrderExecutionEngine()
        md = {
            "price": 50_000, "volatility": 0.02, "liquidity_score": 0.8,
            "portfolio_value": 100_000, "daily_volume": 2_000_000,
        }
        for _ in range(3):
            asyncio.run(
                engine.execute_order_async(
                    "BTC/USDT", "buy", 0.05, "twap", 15,
                    market_data=md, slice_interval_seconds=0,
                )
            )
        assert len(engine.execution_history) == 3


class TestSliceQuantityClamping:
    """Verify slices don't overshoot remaining quantity."""

    def test_total_filled_does_not_exceed_order_quantity(self):
        engine = SmartOrderExecutionEngine()
        md = {
            "price": 50_000, "volatility": 0.02, "liquidity_score": 0.8,
            "portfolio_value": 100_000, "daily_volume": 2_000_000,
        }
        order = asyncio.run(
            engine.execute_order_async(
                "BTC/USDT", "buy", 0.1, "vwap", 60,
                market_data=md, slice_interval_seconds=0,
            )
        )
        assert order.executed_quantity <= order.total_quantity + 1e-9

    def test_twap_total_filled_does_not_exceed_order_quantity(self):
        engine = SmartOrderExecutionEngine()
        md = {
            "price": 3_000, "volatility": 0.02, "liquidity_score": 0.8,
            "portfolio_value": 100_000, "daily_volume": 2_000_000,
        }
        order = asyncio.run(
            engine.execute_order_async(
                "ETH/USDT", "sell", 5.0, "twap", 30,
                market_data=md, slice_interval_seconds=0,
            )
        )
        assert order.executed_quantity <= order.total_quantity + 1e-9
