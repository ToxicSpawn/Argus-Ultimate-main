"""
Tests for Market Flow Risk Adapter.
"""

import unittest

from ml.market_flow_risk import (
    MarketFlowRiskAdapter,
    MarketFlowRisk,
    RiskDecision,
    RiskCondition,
    create_risk_adapter,
)


class TestMarketFlowRiskAdapter(unittest.TestCase):
    """Tests for MarketFlowRiskAdapter."""

    def setUp(self):
        """Set up test fixtures."""
        self.adapter = create_risk_adapter()

    def test_assess_normal_conditions(self):
        """Test normal market conditions."""
        risk = self.adapter.assess_market_flow_risk(
            current_volatility=0.02,
            historical_volatility=0.02,
            current_volume=1000,
            average_volume=1000,
            bid_ask_spread_bps=5,
            order_book_depth=5000,
            current_regime="trending_up",
            fear_greed_index=50,
            price_change_pct=1.0,
        )

        # Should not be crisis
        self.assertNotEqual(risk.condition, "crisis")

    def test_assess_crisiss_conditions(self):
        """Test crisis market conditions."""
        risk = self.adapter.assess_market_flow_risk(
            current_volatility=0.08,
            historical_volatility=0.02,
            current_volume=100,
            average_volume=1000,
            bid_ask_spread_bps=200,
            order_book_depth=100,
            current_regime="high_volatility",
            fear_greed_index=20,
            price_change_pct=-15.0,
        )

        self.assertIn(risk.condition, ["high", "extreme", "crisis"])
        self.assertGreater(risk.overall_score, 0.6)

    def test_adapt_risk_low_volatility(self):
        """Test adaptation with low volatility."""
        risk = MarketFlowRisk(
            condition="normal",
            overall_score=0.2,
            volatility_score=0.2,
            liquidity_score=0.3,
            sentiment_score=0.3,
            regime_score=0.3,
            spread_score=0.2,
            volume_score=0.3,
        )

        performance = {"win_rate": 0.5}

        decision = self.adapter.adapt_risk(risk, performance)

        self.assertEqual(decision.risk_condition, "normal")

    def test_adapt_risk_high_volatility(self):
        """Test adaptation with high volatility."""
        risk = MarketFlowRisk(
            condition="high",
            overall_score=0.8,
            volatility_score=0.8,
            liquidity_score=0.3,
            sentiment_score=0.7,
            regime_score=0.8,
            spread_score=0.7,
            volume_score=0.7,
        )

        performance = {"win_rate": 0.5}

        decision = self.adapter.adapt_risk(risk, performance)

        self.assertGreater(decision.stop_loss_multiplier, 1.0)
        self.assertLess(decision.position_size_multiplier, 1.0)

    def test_adapt_risk_fear_sentiment(self):
        """Test adaptation with fear sentiment."""
        risk = MarketFlowRisk(
            condition="elevated",
            overall_score=0.5,
            volatility_score=0.4,
            liquidity_score=0.3,
            sentiment_score=0.8,  # Fear
            regime_score=0.5,
            spread_score=0.3,
            volume_score=0.3,
        )

        performance = {"win_rate": 0.5}

        decision = self.adapter.adapt_risk(risk, performance)

        self.assertLess(decision.position_size_multiplier, 1.0)

    def test_adapt_risk_poor_performance(self):
        """Test adaptation with poor performance."""
        risk = MarketFlowRisk(
            condition="normal",
            overall_score=0.3,
            volatility_score=0.3,
            liquidity_score=0.3,
            sentiment_score=0.3,
            regime_score=0.3,
            spread_score=0.3,
            volume_score=0.3,
        )

        # Poor win rate
        performance = {"win_rate": 0.25}

        decision = self.adapter.adapt_risk(risk, performance)

        # Should make risk tighter
        self.assertGreater(decision.stop_loss_multiplier, 0.0)

    def test_adapt_risk_crisiss_should_halt(self):
        """Test crisis should halt trading."""
        risk = MarketFlowRisk(
            condition="crisis",
            overall_score=0.9,
            volatility_score=0.9,
            liquidity_score=0.8,
            sentiment_score=0.9,
            regime_score=0.9,
            spread_score=0.9,
            volume_score=0.9,
        )

        performance = {"win_rate": 0.5}

        decision = self.adapter.adapt_risk(risk, performance)

        self.assertTrue(decision.should_halt_new_positions)

    def test_get_current_parameters(self):
        """Test getting current parameters."""
        risk = MarketFlowRisk(
            condition="normal",
            overall_score=0.3,
            volatility_score=0.3,
            liquidity_score=0.3,
            sentiment_score=0.3,
            regime_score=0.3,
            spread_score=0.3,
            volume_score=0.3,
        )

        performance = {"win_rate": 0.5}
        decision = self.adapter.adapt_risk(risk, performance)

        params = self.adapter.get_current_parameters(decision)

        self.assertGreater(params.stop_loss_pct, 0)
        self.assertGreater(params.take_profit_pct, 0)
        self.assertLessEqual(params.max_position_pct, 0.20)

    def test_check_should_trade_normal(self):
        """Test should trade normal."""
        decision = RiskDecision(risk_condition="normal")

        should_trade, reason = self.adapter.check_should_trade(decision)

        self.assertTrue(should_trade)
        self.assertEqual(reason, "OK")

    def test_check_should_trade_crisis(self):
        """Test should not trade crisis."""
        decision = RiskDecision(risk_condition="crisis", should_halt_new_positions=True)

        should_trade, reason = self.adapter.check_should_trade(decision)

        self.assertFalse(should_trade)


class TestMarketFlowRisk(unittest.TestCase):
    """Tests for MarketFlowRisk."""

    def test_default_values(self):
        """Test default values."""
        risk = MarketFlowRisk(condition="normal", overall_score=0.5)

        self.assertEqual(risk.condition, "normal")
        self.assertEqual(risk.overall_score, 0.5)


if __name__ == "__main__":
    unittest.main()