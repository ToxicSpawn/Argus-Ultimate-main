"""Tests for strategy promotion pipeline."""
import unittest
from core.strategy_promotion import (
    StrategyPromotionPipeline, StrategyCandidate, PromotionStage,
)


class TestPromotionPipeline(unittest.TestCase):
    def _make_pipeline(self):
        return StrategyPromotionPipeline(
            validation_min_sharpe=0.3, validation_min_trades=5,
            validation_max_dd=15.0, paper_min_cycles=10,
            paper_min_sharpe=0.0, max_live_strategies=3,
        )

    def test_submit_candidate(self):
        pp = self._make_pipeline()
        c = pp.submit_candidate("s1", "generator", "gp_strat", {}, "RSI < 30",
                                fitness=1.0, sharpe=1.0, win_rate=0.6,
                                trade_count=20, max_dd=5.0)
        self.assertEqual(c.stage, PromotionStage.CANDIDATE)
        self.assertEqual(pp.get_stats()["total_candidates"], 1)

    def test_validation_passes(self):
        pp = self._make_pipeline()
        pp.submit_candidate("s1", "generator", "gp", {}, "rule",
                            fitness=1.0, sharpe=1.0, win_rate=0.6,
                            trade_count=20, max_dd=5.0)
        result = pp.validate("s1", oos_sharpe=0.5, oos_win_rate=0.55, oos_trade_count=15)
        self.assertTrue(result)
        self.assertEqual(pp.get_all()["s1"].stage, PromotionStage.PAPER_TESTING)

    def test_validation_fails_low_sharpe(self):
        pp = self._make_pipeline()
        pp.submit_candidate("s1", "generator", "gp", {}, "rule",
                            fitness=0.5, sharpe=0.5, win_rate=0.5,
                            trade_count=20, max_dd=5.0)
        result = pp.validate("s1", oos_sharpe=0.1, oos_win_rate=0.45, oos_trade_count=15)
        self.assertFalse(result)
        self.assertEqual(pp.get_all()["s1"].stage, PromotionStage.RETIRED)

    def test_paper_promotion(self):
        pp = self._make_pipeline()
        pp.submit_candidate("s1", "generator", "gp", {}, "rule",
                            fitness=1.0, sharpe=1.0, win_rate=0.6,
                            trade_count=20, max_dd=5.0)
        pp.validate("s1", 0.5, 0.55, 15)
        for _ in range(15):
            pp.record_paper_cycle("s1")
        for _ in range(5):
            pp.record_paper_trade("s1", pnl=0.5)
        result = pp.check_paper_promotion("s1")
        self.assertTrue(result)
        self.assertEqual(pp.get_all()["s1"].stage, PromotionStage.PROMOTED)

    def test_activate_live(self):
        pp = self._make_pipeline()
        pp.submit_candidate("s1", "generator", "gp", {}, "rule",
                            fitness=1.0, sharpe=1.0, win_rate=0.6,
                            trade_count=20, max_dd=5.0)
        pp.validate("s1", 0.5, 0.55, 15)
        for _ in range(15):
            pp.record_paper_cycle("s1")
        for _ in range(5):
            pp.record_paper_trade("s1", pnl=0.5)
        pp.check_paper_promotion("s1")
        pp.activate_live("s1")
        self.assertEqual(pp.get_all()["s1"].stage, PromotionStage.LIVE)
        self.assertEqual(len(pp.get_live_strategies()), 1)

    def test_live_retirement(self):
        pp = StrategyPromotionPipeline(
            validation_min_sharpe=0.0, validation_min_trades=1,
            paper_min_cycles=1, paper_min_sharpe=-999.0,
            live_retire_sharpe=-0.1, live_retire_trades=5,
        )
        pp.submit_candidate("s1", "gen", "gp", {}, "rule",
                            fitness=1.0, sharpe=1.0, win_rate=0.6,
                            trade_count=20, max_dd=5.0)
        pp.validate("s1", 0.5, 0.55, 10)
        for _ in range(3):
            pp.record_paper_cycle("s1")
            pp.record_paper_trade("s1", 0.1)
        pp.check_paper_promotion("s1")
        pp.activate_live("s1")
        for _ in range(6):
            pp.record_live_trade("s1", pnl=-1.0)
        retired = pp.check_live_retirement("s1")
        self.assertTrue(retired)
        self.assertEqual(pp.get_all()["s1"].stage, PromotionStage.RETIRED)

    def test_max_live_displaces_worst(self):
        pp = StrategyPromotionPipeline(
            validation_min_sharpe=0.0, validation_min_trades=1,
            paper_min_cycles=1, paper_min_sharpe=-999, max_live_strategies=2,
        )
        for i in range(3):
            sid = f"s{i}"
            pp.submit_candidate(sid, "gen", "gp", {}, "rule",
                                fitness=1.0, sharpe=1.0, win_rate=0.6,
                                trade_count=20, max_dd=5.0)
            pp.validate(sid, 0.5, 0.55, 10)
            pp.record_paper_cycle(sid)
            pp.record_paper_trade(sid, 0.1)
            pp.check_paper_promotion(sid)
            pp.activate_live(sid)
        live = pp.get_live_strategies()
        self.assertLessEqual(len(live), 2)

    def test_get_stats(self):
        pp = self._make_pipeline()
        pp.submit_candidate("s1", "gen", "gp", {}, "rule",
                            fitness=1.0, sharpe=1.0, win_rate=0.6,
                            trade_count=20, max_dd=5.0)
        stats = pp.get_stats()
        self.assertEqual(stats["total_candidates"], 1)
        self.assertIn("stages", stats)
        self.assertIn("promotions", stats)

    def test_full_pipeline_flow(self):
        """Test the complete CANDIDATE → LIVE flow."""
        pp = StrategyPromotionPipeline(
            validation_min_sharpe=0.2, validation_min_trades=3,
            paper_min_cycles=5, paper_min_sharpe=-1.0,
            max_live_strategies=5,
        )
        # Submit
        pp.submit_candidate("test_strat", "generator", "gp_42", {},
                            "RSI(14) < 30 AND VOL > 2*SMA(20)",
                            fitness=0.8, sharpe=0.9, win_rate=0.55,
                            trade_count=15, max_dd=8.0)
        # Validate
        self.assertTrue(pp.validate("test_strat", 0.4, 0.50, 10))
        # Paper test
        for _ in range(10):
            pp.record_paper_cycle("test_strat")
        for _ in range(5):
            pp.record_paper_trade("test_strat", pnl=0.3)
        # Promote
        self.assertTrue(pp.check_paper_promotion("test_strat"))
        # Go live
        self.assertTrue(pp.activate_live("test_strat"))
        self.assertEqual(pp.get_all()["test_strat"].stage, PromotionStage.LIVE)


if __name__ == "__main__":
    unittest.main()
