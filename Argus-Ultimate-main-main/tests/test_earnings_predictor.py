"""Tests for earnings predictor."""
import unittest
from core.earnings_predictor import EarningsPredictor, EarningsForecast


class TestEarningsPredictor(unittest.TestCase):
    def test_insufficient_data(self):
        ep = EarningsPredictor()
        forecast = ep.predict(horizon_days=30, capital=1000)
        self.assertAlmostEqual(forecast.expected_return_pct, 0.0)
        self.assertAlmostEqual(forecast.data_quality, 0.0)
        self.assertTrue(any("INSUFFICIENT" in w for w in forecast.warnings))

    def test_winning_strategy(self):
        ep = EarningsPredictor(n_simulations=500)
        for _ in range(100):
            ep.record_trade(0.5, strategy="momentum", regime="trending")
        forecast = ep.predict(30, 1000, "trending", trades_per_day=5)
        self.assertGreater(forecast.expected_return_pct, 0)
        self.assertGreater(forecast.prob_profit, 0.5)
        self.assertGreater(forecast.expected_pnl, 0)

    def test_losing_strategy(self):
        ep = EarningsPredictor(n_simulations=500)
        for _ in range(100):
            ep.record_trade(-0.3, strategy="bad", regime="ranging")
        forecast = ep.predict(30, 1000, "ranging", trades_per_day=5)
        self.assertLess(forecast.expected_return_pct, 0)
        self.assertLess(forecast.prob_profit, 0.5)

    def test_mixed_returns(self):
        ep = EarningsPredictor(n_simulations=500)
        import random
        rng = random.Random(42)
        for _ in range(200):
            pnl = rng.gauss(0.1, 0.5)  # slight positive edge
            ep.record_trade(pnl, "mixed", "normal")
        forecast = ep.predict(30, 1000, trades_per_day=10)
        self.assertGreater(forecast.data_quality, 0.5)
        # With slight positive edge, p50 should be positive
        self.assertGreater(forecast.p50_return_pct, forecast.p5_return_pct)

    def test_percentile_ordering(self):
        import random
        rng = random.Random(42)
        ep = EarningsPredictor(n_simulations=500)
        for _ in range(100):
            ep.record_trade(rng.gauss(0.3, 1.0), "test", "normal")
        forecast = ep.predict(30, 1000, trades_per_day=5)
        self.assertLess(forecast.p5_return_pct, forecast.p25_return_pct)
        self.assertLess(forecast.p25_return_pct, forecast.p75_return_pct)
        self.assertLess(forecast.p75_return_pct, forecast.p95_return_pct)

    def test_pnl_matches_return(self):
        ep = EarningsPredictor(n_simulations=500)
        for _ in range(100):
            ep.record_trade(1.0, "test", "normal")
        forecast = ep.predict(30, 1000, trades_per_day=3)
        # p50_pnl should be roughly capital * p50_return / 100
        expected_pnl = 1000 * forecast.p50_return_pct / 100
        self.assertAlmostEqual(forecast.p50_pnl, expected_pnl, places=0)

    def test_compounding_projection(self):
        ep = EarningsPredictor(n_simulations=200)
        for _ in range(100):
            ep.record_trade(0.3, "test", "normal")
        projections = ep.project_compounding(months=6, capital=1000)
        self.assertEqual(len(projections), 6)
        self.assertEqual(projections[0]["month"], 1)
        # Capital should grow if edge is positive
        self.assertGreater(projections[-1]["expected_capital"], 1000)

    def test_regime_specific_prediction(self):
        ep = EarningsPredictor(n_simulations=500)
        for _ in range(50):
            ep.record_trade(1.0, "trend_strat", "trending")
        for _ in range(50):
            ep.record_trade(-0.5, "trend_strat", "ranging")
        # Forecast for trending should be positive
        trending = ep.predict(30, 1000, "trending", trades_per_day=5)
        # Forecast for ranging should be negative
        ranging = ep.predict(30, 1000, "ranging", trades_per_day=5)
        self.assertGreater(trending.expected_return_pct, ranging.expected_return_pct)

    def test_strategy_contributions(self):
        ep = EarningsPredictor(n_simulations=200)
        for _ in range(30):
            ep.record_trade(1.0, "winner", "normal")
            ep.record_trade(-0.5, "loser", "normal")
        forecast = ep.predict(30, 1000, active_strategies=["winner", "loser"])
        self.assertIn("winner", forecast.strategy_contributions)

    def test_data_quality_scales(self):
        ep = EarningsPredictor(n_simulations=100)
        for i in range(250):
            ep.record_trade(0.1, "test", "normal")
            if i == 49:
                f50 = ep.predict(30, 1000).data_quality
            if i == 199:
                f200 = ep.predict(30, 1000).data_quality
        self.assertGreater(f200, f50)

    def test_get_stats(self):
        ep = EarningsPredictor()
        ep.record_trade(1.0, "test", "normal")
        stats = ep.get_stats()
        self.assertEqual(stats["total_trades"], 1)
        self.assertAlmostEqual(stats["avg_return_pct"], 1.0)


if __name__ == "__main__":
    unittest.main()
