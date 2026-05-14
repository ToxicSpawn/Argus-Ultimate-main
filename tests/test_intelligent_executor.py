"""Tests for intelligent execution engine."""
import unittest
from execution.intelligent_executor import (
    FillProbabilityEstimator, FillProbability,
    AdaptiveSlicer, AdaptiveSlicePlan, SliceResult,
    ExecutionAlphaTracker, ExecutionQuality,
)


class TestFillProbabilityEstimator(unittest.TestCase):
    def test_high_prob_tight_spread(self):
        fpe = FillProbabilityEstimator()
        fp = fpe.estimate(size_usd=100, book_depth_usd=100000, spread_bps=1.5, volatility=0.01)
        self.assertGreater(fp.probability, 0.60)
        self.assertEqual(fp.recommended_type, "limit")

    def test_low_prob_wide_spread(self):
        fpe = FillProbabilityEstimator()
        fp = fpe.estimate(size_usd=50000, book_depth_usd=10000, spread_bps=15, volatility=0.005)
        self.assertLess(fp.probability, 0.40)

    def test_toxic_market_pauses(self):
        fpe = FillProbabilityEstimator()
        fp = fpe.estimate(size_usd=100, book_depth_usd=100000, spread_bps=2, volatility=0.01,
                          toxicity=0.85)
        self.assertEqual(fp.recommended_type, "pause")

    def test_urgent_low_prob_uses_market(self):
        fpe = FillProbabilityEstimator()
        fp = fpe.estimate(size_usd=50000, book_depth_usd=10000, spread_bps=20, volatility=0.005,
                          edge_bps=15)
        self.assertIn(fp.recommended_type, ("market", "vwap"))

    def test_learns_from_history(self):
        fpe = FillProbabilityEstimator()
        # Record good fills at tight spread
        for _ in range(15):
            fpe.record_fill("BTC/USD", "buy", 0.01, 2.0, 0.02, True, 0.5, 1.0)
        # Should learn that similar conditions have high fill prob
        fp = fpe.estimate(100, 100000, 2.0, 0.02)
        self.assertGreater(fp.probability, 0.50)

    def test_limit_offset_scales_with_spread(self):
        fpe = FillProbabilityEstimator()
        fp_tight = fpe.estimate(100, 100000, 1.0, 0.01)
        fp_wide = fpe.estimate(100, 100000, 10.0, 0.01)
        self.assertLess(fp_tight.limit_offset_bps, fp_wide.limit_offset_bps)

    def test_get_stats(self):
        fpe = FillProbabilityEstimator()
        fpe.record_fill("BTC/USD", "buy", 0.01, 2.0, 0.02, True, 0.5, 1.0)
        stats = fpe.get_stats()
        self.assertEqual(stats["observations"], 1)


class TestAdaptiveSlicer(unittest.TestCase):
    def test_create_plan(self):
        slicer = AdaptiveSlicer()
        plan = slicer.create_plan(total_qty=1.0, target_duration_s=60)
        self.assertEqual(plan.status, "active")
        self.assertGreater(plan.slices_planned, 1)
        self.assertAlmostEqual(plan.remaining_qty, 1.0)

    def test_adapt_reduces_on_high_slippage(self):
        slicer = AdaptiveSlicer(slippage_threshold_bps=3.0)
        plan = slicer.create_plan(1.0, 60)
        initial_slice = plan.current_slice_qty

        result = SliceResult(0, 0.1, 0.1, 50000, slippage_bps=8.0,
                             fill_time_s=1.0, market_moved_bps=5.0)
        plan = slicer.adapt(plan, result)
        self.assertLess(plan.current_slice_qty, initial_slice)

    def test_adapt_increases_aggression_on_low_fills(self):
        slicer = AdaptiveSlicer()
        plan = slicer.create_plan(1.0, 60, initial_aggression=0.3)
        result = SliceResult(0, 0.1, 0.05, 50000, slippage_bps=1.0,
                             fill_time_s=5.0, market_moved_bps=0)  # 50% fill
        plan = slicer.adapt(plan, result)
        self.assertGreater(plan.aggression, 0.3)

    def test_toxicity_pauses(self):
        slicer = AdaptiveSlicer(toxicity_pause_threshold=0.6)
        plan = slicer.create_plan(1.0, 60)
        result = SliceResult(0, 0.1, 0.1, 50000, 1.0, 0.5, 0)
        plan = slicer.adapt(plan, result, current_toxicity=0.8)
        self.assertEqual(plan.status, "paused")

    def test_completes_when_filled(self):
        slicer = AdaptiveSlicer()
        plan = slicer.create_plan(0.1, 10)
        result = SliceResult(0, 0.1, 0.1, 50000, 1.0, 0.5, 0)
        plan = slicer.adapt(plan, result)
        self.assertEqual(plan.status, "completed")

    def test_depth_limits_slice_size(self):
        slicer = AdaptiveSlicer()
        plan = slicer.create_plan(total_qty=100, target_duration_s=60, book_depth_qty=10)
        self.assertLessEqual(plan.current_slice_qty, 0.5)  # 5% of 10


class TestExecutionAlphaTracker(unittest.TestCase):
    def test_record_buy(self):
        tracker = ExecutionAlphaTracker()
        eq = tracker.record("o1", "BTC/USD", "buy", "momentum",
                            decision_price=50000, arrival_price=50010,
                            fill_price=50020, spread_bps=2.0)
        self.assertIsInstance(eq, ExecutionQuality)
        self.assertGreater(eq.implementation_shortfall_bps, 0)  # paid more than decision

    def test_excellent_execution(self):
        tracker = ExecutionAlphaTracker()
        eq = tracker.record("o1", "BTC/USD", "buy", "momentum",
                            decision_price=50000, arrival_price=50000,
                            fill_price=50000.5, spread_bps=1.0)
        self.assertEqual(eq.grade, "EXCELLENT")

    def test_poor_execution(self):
        tracker = ExecutionAlphaTracker()
        eq = tracker.record("o1", "BTC/USD", "buy", "momentum",
                            decision_price=50000, arrival_price=50000,
                            fill_price=50100, spread_bps=2.0)
        self.assertEqual(eq.grade, "POOR")

    def test_sell_direction(self):
        tracker = ExecutionAlphaTracker()
        # Selling at 50050 when decision was 50000 = good (sold higher than expected)
        eq = tracker.record("o1", "BTC/USD", "sell", "mean_reversion",
                            decision_price=50000, arrival_price=50000,
                            fill_price=50050, spread_bps=2.0)
        self.assertLess(eq.implementation_shortfall_bps, 0)  # negative IS = good for sells

    def test_strategy_quality(self):
        tracker = ExecutionAlphaTracker()
        for i in range(10):
            tracker.record(f"o{i}", "BTC/USD", "buy", "momentum",
                           decision_price=50000, arrival_price=50000,
                           fill_price=50000 + i, spread_bps=2.0)
        quality = tracker.get_strategy_quality("momentum")
        self.assertEqual(quality["trade_count"], 10)
        self.assertIn("avg_is_bps", quality)

    def test_overall_quality(self):
        tracker = ExecutionAlphaTracker()
        tracker.record("o1", "BTC/USD", "buy", "test",
                       decision_price=50000, arrival_price=50000,
                       fill_price=50005, spread_bps=2.0)
        overall = tracker.get_overall_quality()
        self.assertEqual(overall["trades"], 1)
        self.assertIn("excellent_pct", overall)


if __name__ == "__main__":
    unittest.main()
