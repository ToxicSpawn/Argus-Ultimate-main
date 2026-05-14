"""
tests_unified/test_tier4_mm_quality.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for Tier 4 — Market Making Quality components:
  - SpreadAttributionEngine  (execution/spread_attribution.py)
  - StreamingPnL             (core/streaming_pnl.py)
  - SessionSpreadSchedule    (execution/session_spread_schedule.py)
  - InventoryUnwindScheduler (execution/inventory_unwind.py)
"""

from __future__ import annotations

import asyncio
import time
import sys
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

# Ensure repo root is on sys.path regardless of how pytest is invoked
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from execution.spread_attribution import SpreadAttributionEngine, FillAttribution
from core.streaming_pnl import StreamingPnL
from execution.session_spread_schedule import SessionSpreadSchedule
from execution.inventory_unwind import (
    InventoryUnwindScheduler,
    UnwindState,
    UnwindStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = 1_700_000_000_000_000_000  # fixed reference timestamp (ns)


def _ns() -> int:
    return time.time_ns()


# ---------------------------------------------------------------------------
# SpreadAttributionEngine tests
# ---------------------------------------------------------------------------

class TestSpreadAttributionSpreadCapture(unittest.TestCase):
    """Buy fill below mid → positive spread capture."""

    def setUp(self):
        self.engine = SpreadAttributionEngine()

    def test_buy_below_mid_positive_capture(self):
        mid = 50_000.0
        fill_price = 49_995.0  # 1 bps below mid
        self.engine.record_fill(
            fill_id="f1",
            symbol="BTC-USD",
            side="buy",
            fill_price=fill_price,
            fill_size=1.0,
            quote_mid_at_fill=mid,
            timestamp_ns=_TS,
        )
        attr = self.engine.get_attribution("f1")
        self.assertIsNotNone(attr)
        # spread_captured_bps = (mid - fill_price) / mid * 10000
        expected_bps = (mid - fill_price) / mid * 10_000
        self.assertAlmostEqual(attr.spread_captured_bps, expected_bps, places=4)
        self.assertGreater(attr.spread_captured_bps, 0.0)

    def test_sell_above_mid_positive_capture(self):
        mid = 50_000.0
        fill_price = 50_005.0  # 1 bps above mid
        self.engine.record_fill(
            fill_id="f2",
            symbol="BTC-USD",
            side="sell",
            fill_price=fill_price,
            fill_size=1.0,
            quote_mid_at_fill=mid,
            timestamp_ns=_TS,
        )
        attr = self.engine.get_attribution("f2")
        expected_bps = (fill_price - mid) / mid * 10_000
        self.assertAlmostEqual(attr.spread_captured_bps, expected_bps, places=4)
        self.assertGreater(attr.spread_captured_bps, 0.0)


class TestSpreadAttributionAdverseSelection(unittest.TestCase):
    """Mid moves against us post-fill → toxic flagged."""

    def setUp(self):
        self.engine = SpreadAttributionEngine()

    def _setup_fill(self, spread_bps: float) -> str:
        mid = 50_000.0
        fill_price = mid - (mid * spread_bps / 10_000)  # buy below mid
        self.engine.record_fill(
            fill_id="fx",
            symbol="BTC-USD",
            side="buy",
            fill_price=fill_price,
            fill_size=1.0,
            quote_mid_at_fill=mid,
            timestamp_ns=_TS,
        )
        return "fx"

    def test_toxic_when_mid_drops_more_than_spread(self):
        """After a buy fill, if mid drops 3× the captured spread → toxic."""
        spread_bps = 1.0
        fid = self._setup_fill(spread_bps)
        attr = self.engine.get_attribution(fid)
        mid_at_fill = attr.quote_mid_at_fill

        # 500ms: mid drops a little
        post_500ms = mid_at_fill - mid_at_fill * 0.0001
        self.engine.record_post_fill_mid(fid, post_500ms, delay_ms=500.0)

        # 5s: mid drops by 3 bps (more than 1 bps captured)
        drop_bps = 3.0
        post_5s = mid_at_fill - mid_at_fill * (drop_bps / 10_000)
        self.engine.record_post_fill_mid(fid, post_5s, delay_ms=5000.0)

        attr = self.engine.get_attribution(fid)
        self.assertTrue(attr.is_toxic, "Fill should be toxic when adverse selection > spread")
        self.assertGreater(attr.adverse_selection_bps_5s, attr.spread_captured_bps)
        self.assertTrue(attr.resolved)

    def test_not_toxic_when_mid_stable(self):
        """If mid stays near fill level, fill is not toxic."""
        fid = self._setup_fill(2.0)
        attr = self.engine.get_attribution(fid)
        mid_at_fill = attr.quote_mid_at_fill

        # 500ms: no movement
        self.engine.record_post_fill_mid(fid, mid_at_fill * 1.0001, delay_ms=500.0)
        # 5s: mid drifts UP (good for long)
        self.engine.record_post_fill_mid(fid, mid_at_fill * 1.0005, delay_ms=5000.0)

        attr = self.engine.get_attribution(fid)
        self.assertFalse(attr.is_toxic)


class TestSpreadAttributionShouldWiden(unittest.TestCase):
    """High adverse selection ratio → should_widen_spread returns True."""

    def setUp(self):
        self.engine = SpreadAttributionEngine()

    def _add_toxic_fill(self, fid: str) -> None:
        mid = 50_000.0
        fill_price = mid - mid * 0.0001  # 1 bps captured
        self.engine.record_fill(
            fill_id=fid,
            symbol="BTC-USD",
            side="buy",
            fill_price=fill_price,
            fill_size=1.0,
            quote_mid_at_fill=mid,
            timestamp_ns=_TS,
        )
        # Adverse of 5 bps (>> 1 bps captured)
        post_5s = mid - mid * 0.0005
        self.engine.record_post_fill_mid(fid, mid * 0.9999, delay_ms=500.0)
        self.engine.record_post_fill_mid(fid, post_5s, delay_ms=5000.0)

    def test_should_widen_when_mostly_toxic(self):
        for i in range(8):
            self._add_toxic_fill(f"toxic_{i}")
        self.assertTrue(self.engine.should_widen_spread("BTC-USD"))

    def test_should_not_widen_when_healthy(self):
        engine = SpreadAttributionEngine()
        mid = 50_000.0
        for i in range(5):
            fid = f"good_{i}"
            engine.record_fill(
                fill_id=fid,
                symbol="ETH-USD",
                side="buy",
                fill_price=mid - mid * 0.0002,
                fill_size=1.0,
                quote_mid_at_fill=mid,
                timestamp_ns=_TS,
            )
            engine.record_post_fill_mid(fid, mid * 0.9999, delay_ms=500.0)
            # Adverse only 0.5 bps, much less than 2 bps captured
            engine.record_post_fill_mid(fid, mid - mid * 0.00005, delay_ms=5000.0)
        self.assertFalse(engine.should_widen_spread("ETH-USD"))


# ---------------------------------------------------------------------------
# StreamingPnL tests
# ---------------------------------------------------------------------------

class TestStreamingPnLUnrealised(unittest.TestCase):
    """Long 1 BTC at 50k, mid=51k → unrealised = 1000."""

    def test_basic_unrealised(self):
        pnl = StreamingPnL()
        pnl.update_position("BTC-USD", "binance", 1.0, 50_000.0, _TS)
        pnl.update_mid("BTC-USD", "binance", 51_000.0, _TS + 1)
        unreal = pnl.get_unrealised_pnl("BTC-USD", "binance")
        self.assertAlmostEqual(unreal, 1_000.0, places=4)

    def test_short_unrealised(self):
        pnl = StreamingPnL()
        pnl.update_position("BTC-USD", "kraken", -1.0, 50_000.0, _TS)
        pnl.update_mid("BTC-USD", "kraken", 49_000.0, _TS + 1)
        unreal = pnl.get_unrealised_pnl("BTC-USD", "kraken")
        # Short: (mid - cost) × size = (49000 - 50000) × -1 = 1000
        self.assertAlmostEqual(unreal, 1_000.0, places=4)

    def test_zero_position_unrealised(self):
        pnl = StreamingPnL()
        pnl.update_position("ETH-USD", "ftx", 0.0, 0.0, _TS)
        pnl.update_mid("ETH-USD", "ftx", 3_000.0, _TS + 1)
        self.assertAlmostEqual(pnl.get_unrealised_pnl("ETH-USD", "ftx"), 0.0)


class TestStreamingPnLDrawdown(unittest.TestCase):
    """Peak 500 then drops to 200 → drawdown computed."""

    def test_drawdown_from_peak(self):
        pnl = StreamingPnL()
        # Build a position and simulate mid moving to create a 500 PnL peak
        pnl.update_position("BTC-USD", "binance", 1.0, 50_000.0, _TS)
        pnl.update_mid("BTC-USD", "binance", 50_500.0, _TS + 1)   # unrealised = 500
        # Access stats to register peak
        _ = pnl.get_session_stats()
        # Mid drops so unrealised falls to 200
        pnl.update_mid("BTC-USD", "binance", 50_200.0, _TS + 2)

        drawdown = pnl.get_drawdown()
        self.assertGreaterEqual(drawdown, 300.0 - 1e-6)  # 500 - 200 = 300

    def test_no_drawdown_when_always_rising(self):
        pnl = StreamingPnL()
        pnl.update_position("BTC-USD", "binance", 1.0, 50_000.0, _TS)
        for price in [50_100.0, 50_200.0, 50_300.0]:
            pnl.update_mid("BTC-USD", "binance", price, _TS + 1)
        # Peak is current — no drawdown
        stats = pnl.get_session_stats()
        self.assertAlmostEqual(stats["current_drawdown_pct"], 0.0, places=4)


class TestStreamingPnLTotal(unittest.TestCase):
    """Realised + unrealised summed correctly."""

    def test_total_combines_realised_and_unrealised(self):
        pnl = StreamingPnL()
        pnl.update_position("BTC-USD", "binance", 1.0, 50_000.0, _TS)
        pnl.update_mid("BTC-USD", "binance", 51_000.0, _TS + 1)  # unrealised = 1000
        pnl.add_realised_pnl("BTC-USD", "binance", 250.0)          # realised = 250

        total = pnl.get_total_pnl("BTC-USD", "binance")
        self.assertAlmostEqual(total, 1_250.0, places=4)

    def test_total_across_all_symbols(self):
        pnl = StreamingPnL()
        pnl.update_position("BTC-USD", "binance", 1.0, 50_000.0, _TS)
        pnl.update_mid("BTC-USD", "binance", 51_000.0, _TS + 1)  # +1000
        pnl.update_position("ETH-USD", "binance", 5.0, 2_000.0, _TS)
        pnl.update_mid("ETH-USD", "binance", 2_100.0, _TS + 1)   # +500

        total = pnl.get_total_pnl()
        self.assertAlmostEqual(total, 1_500.0, places=4)


# ---------------------------------------------------------------------------
# SessionSpreadSchedule tests
# ---------------------------------------------------------------------------

class TestSessionSpreadSchedulePeak(unittest.TestCase):
    """UTC 14:30 → multiplier < 1 (peak_liquidity zone)."""

    def test_peak_multiplier_less_than_one(self):
        schedule = SessionSpreadSchedule(base_spread_bps=5.0)
        mult = schedule.get_multiplier_at_hour(14.5)  # UTC 14:30
        self.assertLess(mult, 1.0, "Peak liquidity zone should have multiplier < 1")
        self.assertAlmostEqual(mult, 0.7, places=5)

    def test_peak_spread_tighter(self):
        schedule = SessionSpreadSchedule(base_spread_bps=10.0)
        mult = schedule.get_multiplier_at_hour(15.0)
        spread = 10.0 * mult
        self.assertLess(spread, 10.0)


class TestSessionSpreadScheduleDeadZone(unittest.TestCase):
    """UTC 02:00 → multiplier > 1.5."""

    def test_dead_zone_multiplier_greater_than_1_5(self):
        schedule = SessionSpreadSchedule()
        mult = schedule.get_multiplier_at_hour(2.0)  # UTC 02:00
        self.assertGreater(mult, 1.5)
        self.assertAlmostEqual(mult, 1.8, places=5)

    def test_dead_zone_spread_wider(self):
        schedule = SessionSpreadSchedule(base_spread_bps=5.0)
        mult = schedule.get_multiplier_at_hour(2.5)
        spread = 5.0 * mult
        self.assertGreater(spread, 5.0)


class TestSessionSpreadOverride(unittest.TestCase):
    """set_override(2.0) → returns 2.0 × base."""

    def test_override_applies(self):
        schedule = SessionSpreadSchedule(base_spread_bps=5.0)
        schedule.set_override(2.0, duration_s=60.0)
        mult = schedule.get_current_spread_multiplier()
        self.assertAlmostEqual(mult, 2.0, places=5)
        spread = schedule.get_spread_bps()
        self.assertAlmostEqual(spread, 10.0, places=5)

    def test_override_cleared(self):
        schedule = SessionSpreadSchedule(base_spread_bps=5.0)
        schedule.set_override(3.0, duration_s=60.0)
        schedule.clear_override()
        mult = schedule.get_current_spread_multiplier()
        # After clearing, should revert to schedule-based value (not 3.0)
        self.assertNotAlmostEqual(mult, 3.0, places=1)

    def test_override_expiry(self):
        """Override with 0 s should immediately expire."""
        schedule = SessionSpreadSchedule(base_spread_bps=5.0)
        # Force instant expiry by backdating the expiry
        schedule._override_multiplier = 99.0
        schedule._override_expires_at = time.monotonic() - 1.0  # already expired
        mult = schedule.get_current_spread_multiplier()
        self.assertNotAlmostEqual(mult, 99.0, places=1)

    def test_symbol_override(self):
        schedule = SessionSpreadSchedule(base_spread_bps=5.0)
        schedule.register_symbol_override("ETH-USD", 8.0)
        schedule.set_override(1.0, duration_s=60.0)  # multiplier = 1.0
        spread = schedule.get_spread_bps("ETH-USD")
        self.assertAlmostEqual(spread, 8.0, places=5)


# ---------------------------------------------------------------------------
# InventoryUnwindScheduler tests
# ---------------------------------------------------------------------------

class _MockExchange:
    """Synchronous-friendly mock exchange for testing unwind scheduler."""

    def __init__(self, mid: float = 50_000.0):
        self._mid = mid
        self._order_counter = 0

    async def get_mid_price(self, symbol: str) -> float:
        return self._mid

    async def place_limit_order(
        self, symbol: str, side: str, size: float, price: float
    ) -> str:
        self._order_counter += 1
        return f"order_{self._order_counter}"

    async def get_order_fill(self, order_id: str) -> float:
        # Immediately fill at the mid
        return self._mid

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        return True

    async def place_market_order(
        self, symbol: str, side: str, size: float
    ) -> float:
        return self._mid


class TestInventoryUnwindTriggers(unittest.TestCase):
    """Position > threshold + near session end → schedules."""

    def test_triggers_when_near_session_end(self):
        mock_ex = _MockExchange()
        scheduler = InventoryUnwindScheduler(
            exchange=mock_ex,
            unwind_threshold=0.01,
            session_end_utc_hour=22,
        )
        # Use a timestamp that corresponds to 21:30 UTC (30 min to session end)
        # time.gmtime(ts_s) should give ~21:30 UTC
        # We patch _seconds_to_session_end to return 1700 (within 1800s trigger)
        with patch.object(
            scheduler,
            "_seconds_to_session_end",
            return_value=1700.0,
        ):
            scheduled = scheduler.check_and_schedule(
                symbol="BTC-USD",
                net_position=0.5,
                avg_cost=50_000.0,
                current_mid=50_000.0,
                timestamp_ns=_TS,
            )
        self.assertTrue(scheduled, "Should schedule unwind when near session end")
        active = scheduler.get_active_unwinds()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].symbol, "BTC-USD")

    def test_does_not_trigger_far_from_session_end(self):
        mock_ex = _MockExchange()
        scheduler = InventoryUnwindScheduler(
            exchange=mock_ex,
            unwind_threshold=0.01,
            session_end_utc_hour=22,
        )
        with patch.object(
            scheduler,
            "_seconds_to_session_end",
            return_value=7200.0,  # 2 hours away — no trigger
        ):
            scheduled = scheduler.check_and_schedule(
                symbol="BTC-USD",
                net_position=0.5,
                avg_cost=50_000.0,
                current_mid=50_000.0,
                timestamp_ns=_TS,
            )
        self.assertFalse(scheduled)

    def test_does_not_trigger_below_threshold(self):
        mock_ex = _MockExchange()
        scheduler = InventoryUnwindScheduler(
            exchange=mock_ex,
            unwind_threshold=0.01,
        )
        with patch.object(scheduler, "_seconds_to_session_end", return_value=500.0):
            scheduled = scheduler.check_and_schedule(
                symbol="BTC-USD",
                net_position=0.005,  # below threshold
                avg_cost=50_000.0,
                current_mid=50_000.0,
                timestamp_ns=_TS,
            )
        self.assertFalse(scheduled)


class TestInventoryUnwindSliceCount(unittest.TestCase):
    """1800 s remaining → correct slice count."""

    def setUp(self):
        self.mock_ex = _MockExchange()
        self.scheduler = InventoryUnwindScheduler(
            exchange=self.mock_ex,
            unwind_threshold=0.01,
        )

    def test_slice_count_1800s(self):
        seconds_remaining = 1800.0
        n = self.scheduler._num_slices(seconds_remaining)
        # 1800 / 120 = 15 → max(5, 15) = 15
        self.assertEqual(n, 15)

    def test_slice_count_minimum(self):
        # Very little time remaining → minimum of 5 slices
        n = self.scheduler._num_slices(300.0)  # 300 / 120 = 2.5 → max(5, 2) = 5
        self.assertEqual(n, 5)

    def test_slice_count_large(self):
        # 3600 / 120 = 30
        n = self.scheduler._num_slices(3600.0)
        self.assertEqual(n, 30)

    def test_execute_unwind_async(self):
        """Run a full unwind cycle via asyncio.run and verify result."""
        scheduler = InventoryUnwindScheduler(
            exchange=self.mock_ex,
            unwind_threshold=0.001,
            limit_fill_timeout_s=0.1,  # fast timeout for test
        )

        async def _run():
            # Override sleep to run without real delays
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await scheduler.execute_unwind(
                    symbol="BTC-USD",
                    net_position=0.1,
                    target_price_tolerance_bps=10.0,
                )
            return result

        result = asyncio.run(_run())
        self.assertAlmostEqual(result.total_filled, 0.1, places=6)
        self.assertEqual(result.status, UnwindStatus.COMPLETED)
        self.assertGreater(result.slices_used, 0)


if __name__ == "__main__":
    unittest.main()
