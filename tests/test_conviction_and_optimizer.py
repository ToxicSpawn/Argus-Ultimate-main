"""Tests for conviction sizer and self-optimizer."""
import unittest
from core.conviction_sizer import ConvictionSizer, ConvictionResult
from core.self_optimizer import SelfOptimizer, OptimizationDirective


class TestConvictionSizer(unittest.TestCase):
    def test_neutral_conviction(self):
        cs = ConvictionSizer()
        result = cs.compute(0.05, "BTC/USD", "BUY", "breakout")
        self.assertIsInstance(result, ConvictionResult)
        self.assertGreater(result.final_size_pct, 0)

    def test_high_conviction_boosts_size(self):
        cs = ConvictionSizer()
        # Provide strong agreement from all sources
        advisory = {
            "strategy_scanner": {"top_opportunities": [{"symbol": "BTC/USD", "strategy": "breakout"}]},
            "strategy_evolver": {"best_symbol": "BTC/USD", "best_composite": 0.8},
            "edge_monitor": {"edge_score": 0.9},
            "health_score": {"score": 90},
        }
        result = cs.compute(0.05, "BTC/USD", "BUY", "breakout",
                            advisory=advisory, mtf_bias={"1h": "BUY", "4h": "BUY"},
                            regime="TRENDING_UP")
        self.assertGreater(result.multiplier, 1.5)
        self.assertGreater(result.final_size_pct, 0.05)

    def test_low_conviction_reduces_size(self):
        cs = ConvictionSizer()
        # Everything disagrees
        advisory = {
            "strategy_scanner": {"top_opportunities": [{"symbol": "ETH/USD"}]},
            "strategy_evolver": {"best_symbol": "ETH/USD", "best_composite": 0.1},
            "edge_monitor": {"edge_score": 0.1},
            "health_score": {"score": 20},
        }
        result = cs.compute(0.05, "BTC/USD", "BUY", "mean_reversion",
                            advisory=advisory, mtf_bias={"1h": "SELL", "4h": "SELL"},
                            regime="TRENDING_UP")
        self.assertLess(result.multiplier, 1.0)

    def test_max_pos_cap(self):
        cs = ConvictionSizer(max_multiplier=5.0)
        result = cs.compute(0.10, "BTC/USD", "BUY", "breakout", max_pos_pct=0.15)
        self.assertLessEqual(result.final_size_pct, 0.15)

    def test_sources_breakdown(self):
        cs = ConvictionSizer()
        result = cs.compute(0.05, "BTC/USD", "BUY", "breakout")
        self.assertIn("scanner", result.sources)
        self.assertIn("evolver", result.sources)
        self.assertIn("regime", result.sources)
        self.assertIn("mtf", result.sources)
        self.assertIn("edge", result.sources)
        self.assertIn("health", result.sources)


class TestSelfOptimizer(unittest.TestCase):
    def test_no_directives_initially(self):
        so = SelfOptimizer()
        report = so.optimize(cycle=1)
        self.assertEqual(len(report.directives), 0)

    def test_disable_losing_strategy(self):
        so = SelfOptimizer(min_trades_to_judge=5)
        for _ in range(15):
            so.record_trade("bad_strat", "BTC/USD", pnl_pct=-2.0)
        report = so.optimize(cycle=100)
        disable_dirs = [d for d in report.directives if d.action == "DISABLE_STRATEGY"]
        self.assertGreater(len(disable_dirs), 0)
        self.assertEqual(disable_dirs[0].target, "bad_strat")

    def test_boost_winning_strategy(self):
        so = SelfOptimizer(min_trades_to_judge=5)
        for _ in range(20):
            so.record_trade("good_strat", "BTC/USD", pnl_pct=3.0)
        report = so.optimize(cycle=100)
        boost_dirs = [d for d in report.directives if d.action == "BOOST_STRATEGY"]
        self.assertGreater(len(boost_dirs), 0)

    def test_reduce_bad_symbol(self):
        so = SelfOptimizer(min_trades_to_judge=5, symbol_min_edge=-0.01)
        for _ in range(15):
            so.record_trade("strat", "DOGE/USD", pnl_pct=-1.5)
        report = so.optimize(cycle=100)
        reduce_dirs = [d for d in report.directives if d.action == "REDUCE_SYMBOL_ALLOCATION"]
        self.assertGreater(len(reduce_dirs), 0)

    def test_execution_quality_alert(self):
        so = SelfOptimizer()
        for _ in range(30):
            so.record_trade("strat", "BTC/USD", pnl_pct=0.5, slippage_bps=8.0)
        report = so.optimize(cycle=100)
        exec_dirs = [d for d in report.directives if d.action == "IMPROVE_EXECUTION"]
        self.assertGreater(len(exec_dirs), 0)

    def test_signal_quality_tracking(self):
        so = SelfOptimizer()
        for _ in range(25):
            so.record_trade("strat", "BTC/USD", pnl_pct=-1.0, signal_source="bad_source")
        report = so.optimize(cycle=100)
        signal_dirs = [d for d in report.directives if d.action == "REDUCE_SIGNAL_WEIGHT"]
        self.assertGreater(len(signal_dirs), 0)

    def test_stagnation_trigger(self):
        so = SelfOptimizer()
        advisory = {"strategy_evolver": {"stagnation_counter": 10}}
        report = so.optimize(cycle=100, advisory=advisory)
        trigger_dirs = [d for d in report.directives if d.action == "TRIGGER_EVOLUTION"]
        self.assertGreater(len(trigger_dirs), 0)

    def test_critical_health_reduces_exposure(self):
        so = SelfOptimizer()
        advisory = {"health_score": {"score": 15}}
        report = so.optimize(cycle=100, advisory=advisory)
        reduce_dirs = [d for d in report.directives if d.action == "REDUCE_EXPOSURE"]
        self.assertGreater(len(reduce_dirs), 0)

    def test_get_stats(self):
        so = SelfOptimizer()
        so.record_trade("strat", "BTC/USD", pnl_pct=1.0, slippage_bps=2.0, fill_rate=0.95)
        stats = so.get_stats()
        self.assertEqual(stats["strategies_tracked"], 1)
        self.assertGreater(stats["avg_fill_rate"], 0)


if __name__ == "__main__":
    unittest.main()
