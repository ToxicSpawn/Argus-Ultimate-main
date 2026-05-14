"""
Tests for Ultimate Adaptation Engine.
"""

import unittest

from ml.ultimate_adaptation import (
    UltimateAdaptationEngine,
    AdaptationDecision,
    SentimentData,
    LiquidityMetrics,
    RegimeForecast,
    create_ultimate_engine,
)


class TestUltimateAdaptationEngine(unittest.TestCase):
    """Tests for UltimateAdaptationEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = create_ultimate_engine()

    def test_engine_creation(self):
        """Test engine creation."""
        self.assertIsInstance(self.engine, UltimateAdaptationEngine)
        self.assertTrue(self.engine.use_sentiment)
        self.assertTrue(self.engine.use_regime_forecast)

    def test_regime_forecast_trending(self):
        """Test regime forecast when trending."""
        history = ["trending_up"] * 10
        forecast = self.engine.forecast_regime(history, 0.03, 0.02)

        self.assertEqual(forecast.predicted_next, "trending_up")
        self.assertGreater(forecast.confidence, 0.5)

    def test_regime_forecast_ranging(self):
        """Test regime forecast when ranging."""
        history = ["ranging"] * 10
        forecast = self.engine.forecast_regime(history, 0.001, 0.015)

        self.assertEqual(forecast.predicted_next, "ranging")
        self.assertLess(forecast.confidence, 0.7)

    def test_correlation_adjustment(self):
        """Test correlation-based adjustment."""
        self.engine._correlation_matrix = {
            "BTC/USD": {"ETH/USD": 0.85},
            "ETH/USD": {},
        }
        open_positions = {"ETH/USD": 0.02}

        adjustment = self.engine.get_correlation_adjustment("BTC/USD", open_positions)

        self.assertLess(adjustment, 1.0)

    def test_sentiment_adjustment_positive(self):
        """Test sentiment with positive bias."""
        sentiment = SentimentData(
            fear_greed_index=70,
            news_sentiment=0.5,
            social_sentiment=0.3,
            whale_activity=0.2,
            funding_rate_bias=0.1,
        )

        adjustment = self.engine.get_sentiment_adjustment(sentiment)

        self.assertGreater(adjustment, 0)

    def test_sentiment_adjustment_negative(self):
        """Test sentiment with negative bias."""
        sentiment = SentimentData(
            fear_greed_index=30,
            news_sentiment=-0.4,
            social_sentiment=-0.3,
            whale_activity=-0.2,
            funding_rate_bias=-0.1,
        )

        adjustment = self.engine.get_sentiment_adjustment(sentiment)

        self.assertLess(adjustment, 0)

    def test_volatility_surface_high(self):
        """Test volatility surface with high vol."""
        adjustment = self.engine.get_volatility_surface_adjustment(0.045, 0.02)

        self.assertLess(adjustment, 1.0)

    def test_volatility_surface_low(self):
        """Test volatility surface with low vol."""
        adjustment = self.engine.get_volatility_surface_adjustment(0.01, 0.02)

        self.assertGreater(adjustment, 1.0)

    def test_streak_psychology_losses(self):
        """Test streak psychology with losses."""
        factor = self.engine.get_streak_psychology(0, 3)

        self.assertLess(factor, 1.0)

    def test_streak_psychology_wins(self):
        """Test streak psychology with wins."""
        factor = self.engine.get_streak_psychology(5, 0)

        self.assertGreater(factor, 1.0)

    def test_partial_exit_trigger(self):
        """Test partial exit calculation."""
        trigger = self.engine.calculate_partial_exit(0.02, 0.03)

        self.assertIsNotNone(trigger)
        self.assertGreater(trigger, 0)

    def test_adapt_method(self):
        """Test complete adapt method."""
        decision = self.engine.adapt(
            symbol="BTC/USD",
            regime_history=["trending_up"] * 10,
            open_positions={},
            current_momentum=0.02,
            current_volatility=0.02,
            recent_volatility=0.02,
            historical_volatility=0.02,
            sentiment=SentimentData(),
            liquidity=LiquidityMetrics(volume_ratio=1.5),
            consecutive_wins=2,
            consecutive_losses=0,
            avg_profit=0.02,
            current_equity=10500,
        )

        self.assertIsInstance(decision, AdaptationDecision)
        self.assertIsInstance(decision.reasoning, list)


class TestSentimentData(unittest.TestCase):
    """Tests for SentimentData."""

    def test_sentiment_defaults(self):
        """Test sentiment defaults."""
        sentiment = SentimentData()

        self.assertEqual(sentiment.fear_greed_index, 50.0)
        self.assertEqual(sentiment.news_sentiment, 0.0)


class TestLiquidityMetrics(unittest.TestCase):
    """Tests for LiquidityMetrics."""

    def test_liquidity_defaults(self):
        """Test liquidity defaults."""
        liquidity = LiquidityMetrics()

        self.assertEqual(liquidity.bid_ask_spread, 0.0)
        self.assertEqual(liquidity.volume_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()