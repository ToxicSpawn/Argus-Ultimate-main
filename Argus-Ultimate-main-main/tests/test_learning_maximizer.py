"""Tests for learning maximizer."""
import unittest
from core.learning_maximizer import LearningMaximizer, ShadowTrade


class TestLearningMaximizer(unittest.TestCase):
    def test_should_explore(self):
        lm = LearningMaximizer(exploration_rate=1.0)  # always explore
        self.assertTrue(lm.should_explore(True, "meta_cognition"))

    def test_should_not_explore_active_trade(self):
        lm = LearningMaximizer()
        self.assertFalse(lm.should_explore(False, ""))

    def test_record_shadow(self):
        lm = LearningMaximizer()
        lm.record_shadow("BTC/USD", "buy", "momentum", 0.7, "entropy_filter", 50000)
        self.assertEqual(lm._total_shadows, 1)

    def test_shadow_outcome_tracking(self):
        lm = LearningMaximizer()
        lm.record_shadow("BTC/USD", "buy", "momentum", 0.7, "skip", 50000)
        lm.update_shadow_outcomes({"BTC/USD": 51000}, 10)
        shadow = list(lm._shadow_trades)[0]
        self.assertAlmostEqual(shadow.price_after_10_bars, 51000)

    def test_shadow_pnl_computed(self):
        lm = LearningMaximizer()
        lm.record_shadow("BTC/USD", "buy", "momentum", 0.7, "skip", 50000)
        lm.update_shadow_outcomes({"BTC/USD": 52000}, 50)
        shadow = list(lm._shadow_trades)[0]
        self.assertGreater(shadow.would_have_pnl_pct, 0)

    def test_exploration_outcome(self):
        lm = LearningMaximizer()
        lm.record_exploration_outcome(2.5, "meta_cognition")
        self.assertEqual(lm._explorations_that_profited, 1)

    def test_cycle_snapshot(self):
        lm = LearningMaximizer()
        lm.record_cycle_snapshot(1, {"BTC/USD": 50000}, {"regime": "trending"})
        self.assertEqual(len(lm._snapshots), 1)

    def test_strategy_tournament(self):
        lm = LearningMaximizer()
        lm.record_strategy_tournament("momentum", "BTC/USD", 1.5, 0.6, 20)
        lm.record_strategy_tournament("mean_reversion", "ETH/USD", 0.8, 0.55, 15)
        winners = lm.get_tournament_winners(2)
        self.assertEqual(len(winners), 2)
        self.assertEqual(winners[0]["strategy"], "momentum")

    def test_fast_intervals(self):
        lm = LearningMaximizer()
        intervals = lm.get_fast_intervals()
        self.assertEqual(intervals["evolution_interval"], 50)
        self.assertEqual(intervals["scanner_interval"], 20)

    def test_shadow_insights(self):
        lm = LearningMaximizer()
        # Record shadows with different skip reasons
        for i in range(10):
            lm.record_shadow("BTC/USD", "buy", "momentum", 0.7, "entropy_filter", 50000)
        # Complete them
        for shadow in lm._shadow_trades:
            shadow.price_after_50_bars = 51000
            shadow.would_have_pnl_pct = 2.0
        lm._shadows_that_would_profit = 10
        insights = lm.get_shadow_insights()
        self.assertGreater(insights["completed"], 0)
        self.assertIn("by_skip_reason", insights)
        self.assertIn("entropy_filter", insights["by_skip_reason"])

    def test_get_stats(self):
        lm = LearningMaximizer()
        lm.record_shadow("BTC/USD", "buy", "test", 0.5, "skip", 50000)
        stats = lm.get_stats()
        self.assertEqual(stats["total_shadows"], 1)


if __name__ == "__main__":
    unittest.main()
