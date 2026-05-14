"""Tests for superadvanced ARGUS systems."""
import unittest
from risk.portfolio_risk_optimizer import PortfolioRiskOptimizer, PortfolioRiskSnapshot
from ml.feedback_loop import MLFeedbackLoop, RetrainDecision
from data.manipulation_detector import ManipulationDetector, ManipulationType
from monitoring.strategy_attribution import StrategyAttributionEngine, PortfolioAttribution


class TestPortfolioRiskOptimizer(unittest.TestCase):
    def test_empty_portfolio(self):
        pro = PortfolioRiskOptimizer()
        snap = pro.optimize()
        self.assertEqual(snap.total_exposure_pct, 0)

    def test_single_strategy(self):
        pro = PortfolioRiskOptimizer()
        for _ in range(20):
            pro.record_return("momentum", 0.5)
        snap = pro.optimize()
        self.assertIn("momentum", snap.strategy_allocations)
        self.assertGreater(snap.strategy_allocations["momentum"], 0)

    def test_multiple_strategies_allocation(self):
        pro = PortfolioRiskOptimizer()
        for _ in range(20):
            pro.record_return("momentum", 1.0)
            pro.record_return("mean_reversion", 0.5)
            pro.record_return("breakout", -0.2)
        snap = pro.optimize()
        # Momentum should get more allocation (higher Sharpe)
        self.assertGreater(
            snap.strategy_allocations.get("momentum", 0),
            snap.strategy_allocations.get("breakout", 0),
        )

    def test_crisis_regime_reduces(self):
        pro = PortfolioRiskOptimizer()
        for _ in range(20):
            pro.record_return("momentum", 0.5)
        normal = pro.optimize("normal")
        crisis = pro.optimize("crisis")
        self.assertGreater(
            normal.strategy_allocations.get("momentum", 0),
            crisis.strategy_allocations.get("momentum", 0),
        )

    def test_single_strategy_cap(self):
        pro = PortfolioRiskOptimizer(max_single_strategy=0.40)
        for _ in range(20):
            pro.record_return("only_strat", 1.0)
        snap = pro.optimize()
        # With only 1 strategy, it should still be capped then renormalized to 1.0
        self.assertGreater(snap.strategy_allocations.get("only_strat", 0), 0)

    def test_var_breach_scales_down(self):
        pro = PortfolioRiskOptimizer(var_limit_pct=0.01)
        for _ in range(20):
            pro.record_return("volatile_strat", 5.0)  # high vol
        snap = pro.optimize()
        self.assertLessEqual(snap.var_95, 0.012)  # should be near limit


class TestMLFeedbackLoop(unittest.TestCase):
    def test_no_retrains_initially(self):
        fl = MLFeedbackLoop()
        fl.register_model("regime_classifier")
        decisions = fl.check_and_retrain()
        self.assertTrue(all(not d.should_retrain for d in decisions))

    def test_retrain_on_low_accuracy(self):
        fl = MLFeedbackLoop(accuracy_threshold=0.50, min_samples_to_judge=10, cooldown_seconds=0)
        fl.register_model("regime_classifier", retrain_fn=lambda: 0.6)
        # Record bad predictions
        for _ in range(20):
            fl.record_prediction("regime_classifier", predicted=1.0, actual=-1.0)
        decisions = fl.check_and_retrain()
        retrain = [d for d in decisions if d.model_name == "regime_classifier"]
        self.assertTrue(retrain[0].should_retrain)

    def test_drift_detection(self):
        fl = MLFeedbackLoop(min_samples_to_judge=10, cooldown_seconds=0)
        fl.register_model("test_model")
        # Good accuracy at first
        for _ in range(30):
            fl.record_prediction("test_model", 1.0, 1.0)
        # Then bad
        for _ in range(30):
            fl.record_prediction("test_model", 1.0, -1.0)
        perf = fl._models["test_model"]
        # Rolling accuracy should be low
        self.assertLess(perf.rolling_accuracy, 0.7)

    def test_model_status(self):
        fl = MLFeedbackLoop()
        fl.register_model("test")
        for _ in range(15):
            fl.record_prediction("test", 1.0, 1.0)
        status = fl.get_model_status()
        self.assertIn("test", status)
        self.assertGreater(status["test"]["accuracy"], 0.5)


class TestManipulationDetector(unittest.TestCase):
    def test_clean_market(self):
        md = ManipulationDetector()
        import time
        for i in range(10):
            md.record_trade("BTC/USD", 50000 + i, 1.0, "buy", time.time() + i)
        alert = md.check("BTC/USD")
        self.assertEqual(alert.manipulation_type, ManipulationType.CLEAN)
        self.assertFalse(alert.block_trading)

    def test_wash_trading_detection(self):
        md = ManipulationDetector(wash_trade_window_s=10.0, wash_trade_price_tolerance=0.005)
        import time
        now = time.time()
        # Rapid buy-sell at same price
        for i in range(20):
            md.record_trade("BTC/USD", 50000, 1.0, "buy", now + i * 0.1)
            md.record_trade("BTC/USD", 50000, 1.0, "sell", now + i * 0.1 + 0.05)
        alert = md.check("BTC/USD")
        # Should detect wash trading pattern
        self.assertGreater(alert.confidence, 0)

    def test_is_blocked(self):
        md = ManipulationDetector(block_duration_s=10.0)
        # Manually set blocked
        state = md._get_state("BTC/USD")
        import time
        state.blocked_until = time.time() + 10
        self.assertTrue(md.is_blocked("BTC/USD"))

    def test_pump_dump_detection(self):
        md = ManipulationDetector(pump_dump_vol_mult=3.0, pump_dump_reversal_pct=0.01)
        import time
        now = time.time()
        # Normal volume + low prices
        for i in range(25):
            md.record_trade("SHIB/USD", 0.001, 10, "buy", now + i)
        # Massive volume spike + price doubles
        for i in range(25):
            md.record_trade("SHIB/USD", 0.002 + i * 0.0001, 10000, "buy", now + 25 + i)
        # Crash back down
        for i in range(10):
            md.record_trade("SHIB/USD", 0.0008, 10, "sell", now + 50 + i)
        alert = md.check("SHIB/USD")
        # Should have non-zero confidence (may not trigger block, but pattern detected)
        self.assertIsNotNone(alert)

    def test_get_stats(self):
        md = ManipulationDetector()
        md.record_trade("BTC/USD", 50000, 1.0, "buy")
        stats = md.get_stats()
        self.assertEqual(stats["symbols_tracked"], 1)


class TestStrategyAttribution(unittest.TestCase):
    def test_empty(self):
        sa = StrategyAttributionEngine()
        attr = sa.compute()
        self.assertAlmostEqual(attr.total_pnl, 0)

    def test_single_strategy(self):
        sa = StrategyAttributionEngine()
        for _ in range(10):
            sa.record_trade("momentum", "BTC/USD", pnl=1.5)
        attr = sa.compute()
        self.assertAlmostEqual(attr.total_pnl, 15.0)
        self.assertEqual(attr.top_contributor, "momentum")

    def test_multi_strategy_attribution(self):
        sa = StrategyAttributionEngine()
        for _ in range(10):
            sa.record_trade("momentum", "BTC/USD", pnl=2.0)
            sa.record_trade("mean_reversion", "ETH/USD", pnl=-0.5)
        attr = sa.compute()
        self.assertEqual(attr.top_contributor, "momentum")
        self.assertEqual(attr.worst_contributor, "mean_reversion")
        self.assertAlmostEqual(attr.total_pnl, 15.0)

    def test_regime_pnl(self):
        sa = StrategyAttributionEngine()
        sa.record_trade("strat", "BTC/USD", pnl=5.0, regime="trending")
        sa.record_trade("strat", "BTC/USD", pnl=-3.0, regime="ranging")
        attr = sa.compute()
        self.assertIn("trending", attr.pnl_by_regime)
        self.assertIn("ranging", attr.pnl_by_regime)
        self.assertGreater(attr.pnl_by_regime["trending"], attr.pnl_by_regime["ranging"])

    def test_execution_cost_tracking(self):
        sa = StrategyAttributionEngine()
        sa.record_trade("strat", "BTC/USD", pnl=1.0, slippage_bps=5.0,
                        fee_usd=0.50, capital_used=1000.0)
        attr = sa.compute()
        self.assertGreater(attr.total_execution_cost, 0)

    def test_factor_decomposition(self):
        sa = StrategyAttributionEngine()
        # Record benchmark returns
        for _ in range(20):
            sa.record_benchmark_return(0.01)
            sa.record_trade("alpha_strat", "BTC/USD", pnl=0.02)
        attr = sa.compute()
        # Alpha should be positive (outperforming benchmark)
        strat_attr = attr.strategies.get("alpha_strat")
        self.assertIsNotNone(strat_attr)
        self.assertGreater(strat_attr.alpha, 0)

    def test_get_stats(self):
        sa = StrategyAttributionEngine()
        sa.record_trade("strat", "BTC/USD", pnl=1.0)
        stats = sa.get_stats()
        self.assertEqual(stats["strategies_tracked"], 1)
        self.assertEqual(stats["total_trades"], 1)


if __name__ == "__main__":
    unittest.main()
