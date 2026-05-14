"""
Tests for real-time data and execution modules (Batch — Realtime/Execution).

Covers:
  - data/tick_engine.py         (TickEngine, OHLCVCandle)
  - data/liquidation_heatmap.py (LiquidationHeatmap, LiquidationLevel)
  - data/funding_rate_predictor.py (FundingRatePredictor, FundingPrediction)
  - execution/adaptive_twap.py  (AdaptiveTWAP, TWAPPlan, TWAPSlice)
  - execution/cross_venue_executor.py (CrossVenueExecutor, VenueOrder)
  - execution/fee_optimizer.py  (FeeOptimizer, FeeTier)

60+ tests total.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock

# ── TickEngine ──────────────────────────────────────────────────────
from data.tick_engine import TickEngine, OHLCVCandle, TIMEFRAME_SECONDS

# ── LiquidationHeatmap ─────────────────────────────────────────────
from data.liquidation_heatmap import LiquidationHeatmap, LiquidationLevel

# ── FundingRatePredictor ───────────────────────────────────────────
from data.funding_rate_predictor import FundingRatePredictor, FundingPrediction

# ── AdaptiveTWAP ───────────────────────────────────────────────────
from execution.adaptive_twap import AdaptiveTWAP, TWAPPlan, TWAPSlice

# ── CrossVenueExecutor ─────────────────────────────────────────────
from execution.cross_venue_executor import (
    CrossVenueExecutor, VenueOrder, VenueBook, OrderBookLevel,
)

# ── FeeOptimizer ───────────────────────────────────────────────────
from execution.fee_optimizer import FeeOptimizer, FeeTier


# ====================================================================
# TickEngine Tests
# ====================================================================


class TestTickEngineBasic(unittest.TestCase):
    """Basic TickEngine functionality."""

    def test_instantiation(self):
        engine = TickEngine()
        self.assertEqual(engine.tick_count, 0)

    def test_single_tick(self):
        engine = TickEngine(timeframes=["1m"])
        engine.on_tick("BTC/AUD", 98_000.0, 0.5, timestamp=1_000_000.0)
        candle = engine.get_candle("BTC/AUD", "1m")
        self.assertIsNotNone(candle)
        self.assertEqual(candle.open, 98_000.0)
        self.assertEqual(candle.close, 98_000.0)
        self.assertEqual(candle.volume, 0.5)
        self.assertEqual(candle.tick_count, 1)
        self.assertFalse(candle.complete)

    def test_multiple_ticks_same_candle(self):
        engine = TickEngine(timeframes=["1m"])
        # Use a base that is already on a 60s boundary so all 4 ticks fit in one candle
        base = 999_960.0
        engine.on_tick("BTC/AUD", 100.0, 1.0, timestamp=base)
        engine.on_tick("BTC/AUD", 105.0, 2.0, timestamp=base + 10)
        engine.on_tick("BTC/AUD", 95.0, 1.0, timestamp=base + 20)
        engine.on_tick("BTC/AUD", 102.0, 1.5, timestamp=base + 30)

        candle = engine.get_candle("BTC/AUD", "1m")
        self.assertEqual(candle.open, 100.0)
        self.assertEqual(candle.high, 105.0)
        self.assertEqual(candle.low, 95.0)
        self.assertEqual(candle.close, 102.0)
        self.assertAlmostEqual(candle.volume, 5.5)
        self.assertEqual(candle.tick_count, 4)

    def test_candle_close_fires_callback(self):
        engine = TickEngine(timeframes=["1m"])
        closed = []
        engine.subscribe_candle_close(lambda c: closed.append(c))
        # First candle: boundary at 1_000_000 (aligned to 60s)
        base = 999_960.0  # aligns to 999_960
        engine.on_tick("BTC/AUD", 100.0, 1.0, timestamp=base)
        # Move to next candle boundary (60s later)
        engine.on_tick("BTC/AUD", 101.0, 1.0, timestamp=base + 60)
        self.assertEqual(len(closed), 1)
        self.assertTrue(closed[0].complete)

    def test_vwap_calculation(self):
        engine = TickEngine(timeframes=["1m"])
        base = 1_000_000.0
        engine.on_tick("BTC/AUD", 100.0, 2.0, timestamp=base)
        engine.on_tick("BTC/AUD", 110.0, 3.0, timestamp=base + 5)
        candle = engine.get_candle("BTC/AUD", "1m")
        expected_vwap = (100.0 * 2.0 + 110.0 * 3.0) / 5.0
        self.assertAlmostEqual(candle.vwap, expected_vwap, places=4)

    def test_invalid_tick_ignored(self):
        engine = TickEngine(timeframes=["1m"])
        engine.on_tick("BTC/AUD", -1, 1.0, timestamp=1_000_000.0)
        engine.on_tick("BTC/AUD", 100, 0.0, timestamp=1_000_000.0)
        engine.on_tick("BTC/AUD", 100, -1.0, timestamp=1_000_000.0)
        self.assertEqual(engine.tick_count, 0)

    def test_multiple_timeframes(self):
        engine = TickEngine(timeframes=["1m", "5m"])
        base = 1_000_000.0
        engine.on_tick("BTC/AUD", 100.0, 1.0, timestamp=base)
        c1m = engine.get_candle("BTC/AUD", "1m")
        c5m = engine.get_candle("BTC/AUD", "5m")
        self.assertIsNotNone(c1m)
        self.assertIsNotNone(c5m)

    def test_completed_only(self):
        engine = TickEngine(timeframes=["1m"])
        engine.on_tick("BTC/AUD", 100.0, 1.0, timestamp=1_000_000.0)
        candle = engine.get_candle("BTC/AUD", "1m", completed_only=True)
        self.assertIsNone(candle)

    def test_symbols_list(self):
        engine = TickEngine(timeframes=["1m"])
        engine.on_tick("BTC/AUD", 100.0, 1.0, timestamp=1_000_000.0)
        engine.on_tick("ETH/AUD", 3000.0, 1.0, timestamp=1_000_000.0)
        self.assertEqual(engine.symbols(), ["BTC/AUD", "ETH/AUD"])

    def test_unsupported_timeframe_raises(self):
        with self.assertRaises(ValueError):
            TickEngine(timeframes=["2m"])

    def test_thread_safety(self):
        engine = TickEngine(timeframes=["1m"])
        base = 1_000_000.0

        def writer():
            for i in range(100):
                engine.on_tick("BTC/AUD", 100.0 + i, 0.01, timestamp=base + i * 0.1)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(engine.tick_count, 400)


# ====================================================================
# LiquidationHeatmap Tests
# ====================================================================


class TestLiquidationHeatmap(unittest.TestCase):
    """LiquidationHeatmap tests."""

    def test_instantiation(self):
        hm = LiquidationHeatmap()
        self.assertEqual(hm.symbols(), [])

    def test_update_oi(self):
        hm = LiquidationHeatmap()
        hm.update_oi("BTC/USD", price=68_000, open_interest=5_000_000, avg_leverage=10)
        self.assertEqual(hm.snapshot_count("BTC/USD"), 1)

    def test_invalid_oi_ignored(self):
        hm = LiquidationHeatmap()
        hm.update_oi("BTC/USD", price=-1, open_interest=1_000, avg_leverage=10)
        hm.update_oi("BTC/USD", price=100, open_interest=-1, avg_leverage=10)
        hm.update_oi("BTC/USD", price=100, open_interest=1000, avg_leverage=0.5)
        self.assertEqual(hm.snapshot_count("BTC/USD"), 0)

    def test_liquidation_levels_both_sides(self):
        hm = LiquidationHeatmap()
        hm.update_oi("BTC/USD", price=68_000, open_interest=5_000_000, avg_leverage=10)
        levels = hm.estimate_liquidation_levels("BTC/USD", current_price=67_000)
        sides = {lv.side for lv in levels}
        self.assertEqual(sides, {"long", "short"})

    def test_long_liquidation_price(self):
        hm = LiquidationHeatmap()
        hm.update_oi("BTC/USD", price=100_000, open_interest=1_000_000, avg_leverage=10)
        levels = hm.estimate_liquidation_levels("BTC/USD", current_price=95_000)
        long_levels = [lv for lv in levels if lv.side == "long"]
        # Long liq = 100_000 * (1 - 1/10) = 90_000
        self.assertAlmostEqual(long_levels[0].price, 90_000, places=0)

    def test_short_liquidation_price(self):
        hm = LiquidationHeatmap()
        hm.update_oi("BTC/USD", price=100_000, open_interest=1_000_000, avg_leverage=10)
        levels = hm.estimate_liquidation_levels("BTC/USD", current_price=105_000)
        short_levels = [lv for lv in levels if lv.side == "short"]
        # Short liq = 100_000 * (1 + 1/10) = 110_000
        self.assertAlmostEqual(short_levels[0].price, 110_000, places=0)

    def test_nearest_liquidation(self):
        hm = LiquidationHeatmap()
        hm.update_oi("BTC/USD", price=68_000, open_interest=5_000_000, avg_leverage=10)
        nearest = hm.get_nearest_liquidation("BTC/USD", 67_000)
        self.assertIsNotNone(nearest)
        self.assertIsInstance(nearest, LiquidationLevel)

    def test_nearest_no_data(self):
        hm = LiquidationHeatmap()
        self.assertIsNone(hm.get_nearest_liquidation("BTC/USD", 67_000))

    def test_cascade_risk(self):
        hm = LiquidationHeatmap()
        hm.update_oi("BTC/USD", price=100_000, open_interest=5_000_000, avg_leverage=10)
        risk = hm.get_cascade_risk("BTC/USD", current_price=100_000, move_pct=15)
        # Both long liq (90k, 10% away) and short liq (110k, 10% away) within 15% move
        self.assertGreater(risk, 0)

    def test_cascade_risk_zero_for_narrow_move(self):
        hm = LiquidationHeatmap()
        hm.update_oi("BTC/USD", price=100_000, open_interest=5_000_000, avg_leverage=10)
        # Liq levels at 90k and 110k -> 10% away, so 1% move triggers nothing
        risk = hm.get_cascade_risk("BTC/USD", current_price=100_000, move_pct=1)
        self.assertEqual(risk, 0.0)

    def test_pruning(self):
        hm = LiquidationHeatmap(window_s=10)
        now = time.time()
        hm.update_oi("BTC/USD", price=100, open_interest=1000, avg_leverage=5, timestamp=now - 20)
        hm.update_oi("BTC/USD", price=101, open_interest=1000, avg_leverage=5, timestamp=now)
        # Old snapshot should be pruned
        self.assertEqual(hm.snapshot_count("BTC/USD"), 1)


# ====================================================================
# FundingRatePredictor Tests
# ====================================================================


class TestFundingRatePredictor(unittest.TestCase):
    """FundingRatePredictor tests."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._db = os.path.join(self._tmpdir, "test_funding.db")

    def _make(self, **kwargs):
        return FundingRatePredictor(db_path=self._db, **kwargs)

    def test_instantiation(self):
        p = self._make()
        self.assertEqual(p.symbols(), [])

    def test_update_and_symbols(self):
        p = self._make()
        p.update("BTC/USD", 0.0001, 0.05, 300e6, 1e9)
        self.assertEqual(p.symbols(), ["BTC/USD"])

    def test_predict_single_observation(self):
        p = self._make()
        p.update("BTC/USD", 0.0003, 0.1, 500e6, 2e9)
        pred = p.predict_next_funding("BTC/USD")
        self.assertIsNotNone(pred)
        self.assertIsInstance(pred, FundingPrediction)
        self.assertEqual(pred.symbol, "BTC/USD")

    def test_predict_returns_none_for_unknown(self):
        p = self._make()
        self.assertIsNone(p.predict_next_funding("XRP/USD"))

    def test_predict_with_multiple_observations(self):
        p = self._make()
        for i in range(10):
            p.update("BTC/USD", 0.0001 * (i + 1), 0.05 * (i + 1), 300e6 + i * 1e6, 1e9 + i * 1e7,
                     timestamp=1_000_000 + i * 3600)
        pred = p.predict_next_funding("BTC/USD")
        self.assertIsNotNone(pred)
        self.assertTrue(0.0 <= pred.confidence <= 1.0)

    def test_direction_classification(self):
        p = self._make()
        for i in range(5):
            p.update("BTC/USD", 0.005, 0.3, 500e6, 2e9, timestamp=1e6 + i * 3600)
        pred = p.predict_next_funding("BTC/USD")
        self.assertEqual(pred.direction, "long_pay")

    def test_arb_opportunities(self):
        p = self._make()
        for i in range(5):
            p.update("BTC/USD", 0.02, 0.5, 500e6, 2e9, timestamp=1e6 + i * 3600)
            p.update("ETH/USD", 0.001, 0.01, 100e6, 500e6, timestamp=1e6 + i * 3600)
        opps = p.get_arb_opportunities(min_rate=0.005)
        symbols = [o.symbol for o in opps]
        self.assertIn("BTC/USD", symbols)

    def test_persistence_reload(self):
        p1 = self._make()
        p1.update("BTC/USD", 0.0003, 0.1, 500e6, 2e9, timestamp=1e6)
        # Create a new instance pointing to the same DB
        p2 = self._make()
        self.assertIn("BTC/USD", p2.symbols())

    def test_max_history_trim(self):
        p = self._make(max_history=20)
        for i in range(50):
            p.update("BTC/USD", 0.0001, 0.01, 100e6, 1e9, timestamp=1e6 + i)
        # In-memory should be trimmed to 20
        pred = p.predict_next_funding("BTC/USD")
        self.assertIsNotNone(pred)


# ====================================================================
# AdaptiveTWAP Tests
# ====================================================================


class TestAdaptiveTWAP(unittest.TestCase):
    """AdaptiveTWAP tests."""

    def test_instantiation(self):
        twap = AdaptiveTWAP()
        self.assertEqual(twap.active_plans(), [])

    def test_create_twap_plan(self):
        twap = AdaptiveTWAP()
        plan = twap.create_plan("BTC/AUD", total_size=1.0, duration_s=300, style="twap")
        self.assertEqual(plan.symbol, "BTC/AUD")
        self.assertEqual(plan.style, "twap")
        self.assertEqual(plan.total_size, 1.0)
        self.assertEqual(len(plan.slices), 10)  # default

    def test_twap_equal_slices(self):
        twap = AdaptiveTWAP(default_slices=5)
        plan = twap.create_plan("BTC/AUD", total_size=1.0, duration_s=300, style="twap")
        sizes = [s.size for s in plan.slices]
        self.assertEqual(len(sizes), 5)
        for s in sizes:
            self.assertAlmostEqual(s, 0.2, places=6)

    def test_create_vwap_plan(self):
        twap = AdaptiveTWAP(default_slices=5)
        plan = twap.create_plan("BTC/AUD", total_size=1.0, duration_s=300, style="vwap")
        self.assertEqual(plan.style, "vwap")
        total = sum(s.size for s in plan.slices)
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_vwap_slices_not_all_equal(self):
        twap = AdaptiveTWAP(default_slices=10)
        plan = twap.create_plan("BTC/AUD", total_size=1.0, duration_s=600, style="vwap")
        sizes = [s.size for s in plan.slices]
        # VWAP U-shape: edge slices should be larger than middle
        self.assertGreater(sizes[0], sizes[len(sizes) // 2])

    def test_get_next_slice(self):
        twap = AdaptiveTWAP(default_slices=3)
        plan = twap.create_plan("BTC/AUD", total_size=0.3, duration_s=180, style="twap")
        s1 = twap.get_next_slice(plan.plan_id)
        s2 = twap.get_next_slice(plan.plan_id)
        s3 = twap.get_next_slice(plan.plan_id)
        s4 = twap.get_next_slice(plan.plan_id)
        self.assertIsNotNone(s1)
        self.assertIsNotNone(s2)
        self.assertIsNotNone(s3)
        self.assertIsNone(s4)  # exhausted

    def test_vwap_participation_cap(self):
        twap = AdaptiveTWAP(default_slices=3, max_participation_rate=0.05)
        plan = twap.create_plan("BTC/AUD", total_size=100.0, duration_s=180, style="vwap")
        sl = twap.get_next_slice(plan.plan_id, current_volume=10.0)
        # 10 * 0.05 = 0.5, so slice should be capped at 0.5
        self.assertLessEqual(sl.size, 0.5)

    def test_cancel_plan(self):
        twap = AdaptiveTWAP()
        plan = twap.create_plan("BTC/AUD", total_size=1.0, duration_s=300)
        self.assertTrue(twap.cancel_plan(plan.plan_id))
        self.assertIsNone(twap.get_next_slice(plan.plan_id))

    def test_adjust_for_volume(self):
        twap = AdaptiveTWAP(default_slices=5)
        plan = twap.create_plan("BTC/AUD", total_size=1.0, duration_s=300, style="twap")
        twap.adjust_for_volume(plan.plan_id, current_volume=200, avg_volume=100)
        # Slices should be scaled up
        sl = twap.get_next_slice(plan.plan_id)
        self.assertIsNotNone(sl)

    def test_invalid_style_raises(self):
        twap = AdaptiveTWAP()
        with self.assertRaises(ValueError):
            twap.create_plan("BTC/AUD", total_size=1.0, duration_s=300, style="invalid")

    def test_invalid_size_raises(self):
        twap = AdaptiveTWAP()
        with self.assertRaises(ValueError):
            twap.create_plan("BTC/AUD", total_size=-1, duration_s=300)

    def test_price_limit_propagated(self):
        twap = AdaptiveTWAP(default_slices=3)
        plan = twap.create_plan("BTC/AUD", total_size=1.0, duration_s=180,
                                style="twap", price_limit=99_000)
        for sl in plan.slices:
            self.assertEqual(sl.price_limit, 99_000)

    def test_mark_executed(self):
        twap = AdaptiveTWAP(default_slices=2)
        plan = twap.create_plan("BTC/AUD", total_size=1.0, duration_s=120, style="twap")
        twap.mark_executed(plan.plan_id, 0, fill_price=98_000, fill_size=0.5)
        self.assertTrue(plan.slices[0].executed)
        self.assertEqual(plan.slices[0].actual_fill_price, 98_000)

    def test_get_plan(self):
        twap = AdaptiveTWAP()
        plan = twap.create_plan("BTC/AUD", total_size=1.0, duration_s=300)
        fetched = twap.get_plan(plan.plan_id)
        self.assertEqual(fetched.plan_id, plan.plan_id)

    def test_unknown_plan_id(self):
        twap = AdaptiveTWAP()
        self.assertIsNone(twap.get_next_slice("nonexistent"))


# ====================================================================
# CrossVenueExecutor Tests
# ====================================================================


class TestCrossVenueExecutor(unittest.TestCase):
    """CrossVenueExecutor tests."""

    def test_instantiation(self):
        ex = CrossVenueExecutor()
        self.assertIsNotNone(ex)

    def test_split_order_basic(self):
        ex = CrossVenueExecutor()
        orders = ex.split_order(
            "BTC/AUD", size=1.0,
            venues=["kraken", "coinbase"],
            venue_prices={"kraken": 98_000, "coinbase": 98_100},
            side="buy",
        )
        self.assertTrue(len(orders) >= 1)
        total = sum(o.size for o in orders)
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_split_favours_better_price(self):
        ex = CrossVenueExecutor()
        orders = ex.split_order(
            "BTC/AUD", size=1.0,
            venues=["kraken", "coinbase"],
            venue_prices={"kraken": 90_000, "coinbase": 100_000},
            side="buy",
        )
        # Kraken has significantly better price
        kraken_orders = [o for o in orders if o.venue == "kraken"]
        coinbase_orders = [o for o in orders if o.venue == "coinbase"]
        if kraken_orders and coinbase_orders:
            self.assertGreater(kraken_orders[0].size, coinbase_orders[0].size)

    def test_split_order_sell(self):
        ex = CrossVenueExecutor()
        orders = ex.split_order(
            "BTC/AUD", size=1.0,
            venues=["kraken", "coinbase"],
            venue_prices={"kraken": 98_000, "coinbase": 98_100},
            side="sell",
        )
        total = sum(o.size for o in orders)
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_empty_venues(self):
        ex = CrossVenueExecutor()
        orders = ex.split_order("BTC/AUD", 1.0, [], {})
        self.assertEqual(orders, [])

    def test_zero_size(self):
        ex = CrossVenueExecutor()
        orders = ex.split_order("BTC/AUD", 0, ["kraken"], {"kraken": 98_000})
        self.assertEqual(orders, [])

    def test_best_split_with_orderbooks(self):
        ex = CrossVenueExecutor()
        books = {
            "kraken": VenueBook(
                venue="kraken",
                asks=[OrderBookLevel(98_000, 0.5), OrderBookLevel(98_100, 1.0)],
                fee_bps=10,
            ),
            "coinbase": VenueBook(
                venue="coinbase",
                asks=[OrderBookLevel(97_900, 0.3), OrderBookLevel(98_200, 2.0)],
                fee_bps=15,
            ),
        }
        orders = ex.get_best_split("BTC/AUD", 1.0, books, side="buy")
        total = sum(o.size for o in orders)
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_best_split_exhausted_books(self):
        ex = CrossVenueExecutor()
        books = {
            "kraken": VenueBook(
                venue="kraken",
                asks=[OrderBookLevel(98_000, 0.1)],
                fee_bps=10,
            ),
        }
        orders = ex.get_best_split("BTC/AUD", 1.0, books, side="buy")
        # Only 0.1 available
        total = sum(o.size for o in orders)
        self.assertAlmostEqual(total, 0.1, places=6)

    def test_estimate_savings(self):
        ex = CrossVenueExecutor()
        naive = [VenueOrder("kraken", "BTC/AUD", 1.0, 98_100, "buy", fee_bps=20)]
        optimised = [
            VenueOrder("kraken", "BTC/AUD", 0.5, 98_000, "buy", fee_bps=10),
            VenueOrder("coinbase", "BTC/AUD", 0.5, 98_050, "buy", fee_bps=15),
        ]
        savings = ex.estimate_savings_bps(naive, optimised)
        self.assertGreater(savings, 0)

    def test_estimate_savings_empty(self):
        ex = CrossVenueExecutor()
        self.assertEqual(ex.estimate_savings_bps([], []), 0.0)

    def test_sell_side_orderbook_split(self):
        ex = CrossVenueExecutor()
        books = {
            "kraken": VenueBook(
                venue="kraken",
                bids=[OrderBookLevel(98_000, 0.5), OrderBookLevel(97_900, 1.0)],
                fee_bps=10,
            ),
            "coinbase": VenueBook(
                venue="coinbase",
                bids=[OrderBookLevel(98_100, 0.3), OrderBookLevel(97_800, 2.0)],
                fee_bps=15,
            ),
        }
        orders = ex.get_best_split("BTC/AUD", 0.5, books, side="sell")
        total = sum(o.size for o in orders)
        self.assertAlmostEqual(total, 0.5, places=6)


# ====================================================================
# FeeOptimizer Tests
# ====================================================================


class TestFeeOptimizer(unittest.TestCase):
    """FeeOptimizer tests."""

    def test_instantiation(self):
        opt = FeeOptimizer()
        self.assertIn("kraken", opt.exchanges())
        self.assertIn("coinbase", opt.exchanges())
        self.assertIn("bybit", opt.exchanges())
        self.assertIn("okx", opt.exchanges())
        self.assertIn("binance", opt.exchanges())

    def test_kraken_starter_tier(self):
        opt = FeeOptimizer()
        tier = opt.get_fee_tier("kraken", 10_000)
        self.assertEqual(tier.tier_name, "Starter")
        self.assertEqual(tier.maker_bps, 16.0)
        self.assertEqual(tier.taker_bps, 26.0)

    def test_kraken_expert_tier(self):
        opt = FeeOptimizer()
        tier = opt.get_fee_tier("kraken", 600_000)
        self.assertEqual(tier.tier_name, "Expert")

    def test_coinbase_top_tier(self):
        opt = FeeOptimizer()
        tier = opt.get_fee_tier("coinbase", 100_000_000)
        self.assertEqual(tier.tier_name, "Level 6")
        self.assertEqual(tier.maker_bps, 0.0)

    def test_unknown_exchange_raises(self):
        opt = FeeOptimizer()
        with self.assertRaises(ValueError):
            opt.get_fee_tier("unknown_exchange", 1000)

    def test_case_insensitive(self):
        opt = FeeOptimizer()
        tier = opt.get_fee_tier("KRAKEN", 10_000)
        self.assertEqual(tier.exchange, "kraken")

    def test_should_use_limit_low_urgency(self):
        opt = FeeOptimizer()
        # Low urgency + decent fee saving → limit recommended
        self.assertTrue(opt.should_use_limit(spread_bps=5.0, urgency=0.1, fee_saving_bps=10.0))

    def test_should_use_limit_high_urgency(self):
        opt = FeeOptimizer()
        # High urgency + wide spread → market recommended
        self.assertFalse(opt.should_use_limit(spread_bps=20.0, urgency=0.9, fee_saving_bps=5.0))

    def test_volume_to_next_tier(self):
        opt = FeeOptimizer()
        to_next = opt.get_volume_to_next_tier("kraken", 10_000)
        self.assertEqual(to_next, 40_000)  # next tier at 50k

    def test_volume_to_next_tier_top(self):
        opt = FeeOptimizer()
        to_next = opt.get_volume_to_next_tier("kraken", 50_000_000)
        self.assertEqual(to_next, 0.0)

    def test_estimate_monthly_savings(self):
        opt = FeeOptimizer()
        savings = opt.estimate_monthly_fee_savings(
            trades_per_day=10, avg_size_usd=500, exchange="kraken",
        )
        self.assertGreater(savings, 0)

    def test_all_tiers(self):
        opt = FeeOptimizer()
        tiers = opt.all_tiers("kraken")
        self.assertEqual(len(tiers), 8)

    def test_custom_schedule(self):
        custom = {
            "my_exchange": [
                ("Tier 1", 5.0, 10.0, 0, 100_000),
                ("Tier 2", 2.0, 5.0, 100_000, float("inf")),
            ]
        }
        opt = FeeOptimizer(custom_schedules=custom)
        tier = opt.get_fee_tier("my_exchange", 50_000)
        self.assertEqual(tier.tier_name, "Tier 1")

    def test_binance_regular_tier(self):
        opt = FeeOptimizer()
        tier = opt.get_fee_tier("binance", 500_000)
        self.assertEqual(tier.tier_name, "Regular")
        self.assertEqual(tier.maker_bps, 10.0)

    def test_bybit_vip3(self):
        opt = FeeOptimizer()
        tier = opt.get_fee_tier("bybit", 750_000)
        self.assertEqual(tier.tier_name, "VIP 3")

    def test_okx_regular(self):
        opt = FeeOptimizer()
        tier = opt.get_fee_tier("okx", 50_000)
        self.assertEqual(tier.tier_name, "Regular")


if __name__ == "__main__":
    unittest.main()
