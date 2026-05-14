# pyright: reportMissingImports=false
"""
Tests for Strategy Intelligence Hub.

Tests cover:
- Strategy registration and categorization
- Performance tracking
- Decay detection
- Champion-challenger testing
- Allocation calculation
- Regime-based strategy selection
"""

from __future__ import annotations

import logging
import unittest
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class TestStrategyIntelligenceHub(unittest.TestCase):
    """Tests for Strategy Intelligence Hub."""

    def setUp(self):
        try:
            from strategies.strategy_intelligence_hub import (
                StrategyIntelligenceHub,
                StrategyCategory,
                MarketRegime,
                StrategyConfig,
                StrategyStatus
            )
            self.hub = StrategyIntelligenceHub()
            self.StrategyCategory = StrategyCategory
            self.MarketRegime = MarketRegime
            self.StrategyConfig = StrategyConfig
            self.StrategyStatus = StrategyStatus
        except ImportError:
            self.skipTest("Strategy Intelligence Hub not available")

    def test_initialization(self):
        """Test hub initialization with default strategies."""
        self.assertIsNotNone(self.hub)
        # Should have registered many default strategies
        self.assertGreater(len(self.hub.strategies), 20)

    def test_strategy_categories(self):
        """Test strategies are properly categorized."""
        categories = set()
        for config in self.hub.strategies.values():
            categories.add(config.category)
        
        # Should have all major categories
        self.assertIn(self.StrategyCategory.ARBITRAGE, categories)
        self.assertIn(self.StrategyCategory.MARKET_MAKING, categories)
        self.assertIn(self.StrategyCategory.MOMENTUM, categories)

    def test_record_trade(self):
        """Test trade recording."""
        strategy_name = "funding_rate_arb"
        
        # Record some trades
        for i in range(15):
            pnl = 10.0 if i % 3 != 0 else -5.0  # 2/3 wins
            self.hub.record_trade(
                strategy_name=strategy_name,
                pnl=pnl,
                edge_bps=15.0,
                hold_time_minutes=30.0
            )
        
        metrics = self.hub.metrics[strategy_name]
        self.assertEqual(metrics.total_trades, 15)
        self.assertGreater(metrics.winning_trades, 0)
        self.assertGreater(metrics.total_pnl, 0)

    def test_decay_detection(self):
        """Test strategy decay detection."""
        strategy_name = "momentum"
        
        # First, record good trades
        for i in range(25):
            self.hub.record_trade(
                strategy_name=strategy_name,
                pnl=20.0,  # Consistent wins
                edge_bps=10.0,
                hold_time_minutes=15.0
            )
        
        metrics = self.hub.metrics[strategy_name]
        self.assertEqual(metrics.recent_win_rate, 1.0)
        
        # Now record bad trades to trigger decay
        for i in range(25):
            self.hub.record_trade(
                strategy_name=strategy_name,
                pnl=-15.0,  # Consistent losses
                edge_bps=-5.0,
                hold_time_minutes=15.0
            )
        
        # Should detect decay
        self.assertGreater(metrics.decay_score, 0.0)

    def test_update_regime(self):
        """Test regime-based strategy selection."""
        # Test trending up regime
        active = self.hub.update_regime(self.MarketRegime.TRENDING_UP)
        
        self.assertGreater(len(active), 0)
        # Should include momentum strategies
        self.assertIn("momentum", active)
        self.assertIn("trend_following", active)
        
        # Test ranging regime
        active = self.hub.update_regime(self.MarketRegime.RANGING)
        
        # Should include mean reversion and market making
        self.assertIn("mean_reversion", active)
        self.assertIn("market_maker", active)

    def test_calculate_allocations(self):
        """Test capital allocation calculation."""
        # Record some performance data
        for name in ["funding_rate_arb", "momentum", "mean_reversion"]:
            for i in range(20):
                self.hub.record_trade(
                    strategy_name=name,
                    pnl=np.random.randn() * 10,
                    edge_bps=10.0,
                    hold_time_minutes=30.0
                )
        
        allocations = self.hub.calculate_allocations()
        
        # Should have allocations for all strategies
        self.assertGreater(len(allocations), 0)
        
        # Total should be 100%
        total = sum(allocations.values())
        self.assertAlmostEqual(total, 100.0, delta=0.1)

    def test_champion_challenger(self):
        """Test champion-challenger testing."""
        # Set up two strategies in same category with different performance
        strategy_a = "momentum"
        strategy_b = "breakout"
        
        # Strategy A: good performance
        for i in range(25):
            self.hub.record_trade(
                strategy_name=strategy_a,
                pnl=15.0 + np.random.randn() * 2,
                edge_bps=10.0,
                hold_time_minutes=15.0
            )
        
        # Strategy B: worse performance
        for i in range(25):
            self.hub.record_trade(
                strategy_name=strategy_b,
                pnl=5.0 + np.random.randn() * 2,
                edge_bps=5.0,
                hold_time_minutes=15.0
            )
        
        # Run champion-challenger test
        result = self.hub.run_champion_challenger_test(self.StrategyCategory.MOMENTUM)
        
        # Should have result
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.champion_name)

    def test_should_execute_strategy(self):
        """Test strategy execution decision."""
        # First set up allocations
        self.hub.update_regime(self.MarketRegime.TRENDING_UP)
        self.hub.calculate_allocations()
        
        # Test enabled strategy with allocation
        strategy_name = "funding_rate_arb"
        self.hub.allocations[strategy_name] = 10.0  # Give it allocation
        should_execute, reason = self.hub.should_execute_strategy(strategy_name)
        self.assertTrue(should_execute)
        
        # Test non-existent strategy
        should_execute, reason = self.hub.should_execute_strategy("nonexistent")
        self.assertFalse(should_execute)

    def test_get_strategy_report(self):
        """Test report generation."""
        # Record some trades
        for i in range(10):
            self.hub.record_trade(
                strategy_name="funding_rate_arb",
                pnl=10.0,
                edge_bps=15.0,
                hold_time_minutes=30.0
            )
        
        report = self.hub.get_strategy_report()
        
        self.assertIn("total_strategies", report)
        self.assertIn("categories", report)
        self.assertIn("top_performers", report)
        self.assertGreater(report["total_strategies"], 0)


class TestStrategyIntelligenceIntegration(unittest.TestCase):
    """Integration tests for Strategy Intelligence Hub."""

    def test_all_categories_have_strategies(self):
        """Test that all categories have registered strategies."""
        from strategies.strategy_intelligence_hub import (
            StrategyIntelligenceHub,
            StrategyCategory
        )
        
        hub = StrategyIntelligenceHub()
        
        for category in StrategyCategory:
            strategies_in_cat = [
                name for name, config in hub.strategies.items()
                if config.category == category
            ]
            self.assertGreater(
                len(strategies_in_cat), 0,
                f"Category {category.name} has no strategies"
            )

    def test_regime_strategy_mapping(self):
        """Test that regimes map to appropriate strategies."""
        from strategies.strategy_intelligence_hub import (
            StrategyIntelligenceHub,
            MarketRegime
        )
        
        hub = StrategyIntelligenceHub()
        
        # Trending should favor momentum
        trending = hub.update_regime(MarketRegime.TRENDING_UP)
        self.assertIn("momentum", trending)
        
        # Ranging should favor mean reversion
        ranging = hub.update_regime(MarketRegime.RANGING)
        self.assertIn("mean_reversion", ranging)


def run_tests():
    """Run all tests."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestStrategyIntelligenceHub))
    suite.addTest(unittest.makeSuite(TestStrategyIntelligenceIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
