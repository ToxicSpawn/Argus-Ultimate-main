"""
Tests for Market Flow Adaptive Strategy and Constant Adaptation Loop.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from ml.market_flow_strategy import (
    MarketFlowAdaptiveStrategy,
    MarketRegime,
    SignalDirection,
    StrategyPerformance,
    create_market_flow_strategy,
)
from ml.constant_adaptation import (
    AdaptationConfig,
    ConstantAdaptationLoop,
    create_default_loop,
)


class TestMarketFlowAdaptiveStrategy(unittest.TestCase):
    """Tests for MarketFlowAdaptiveStrategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = create_market_flow_strategy()

    def test_detect_regime_unknown_with_insufficient_data(self):
        """Test regime detection with insufficient data."""
        prices = [100 + i for i in range(30)]
        volumes = [1000] * 30

        regime = self.strategy.detect_regime(prices, volumes)
        self.assertEqual(regime, MarketRegime.UNKNOWN)

    def test_calculate_momentum(self):
        """Test momentum calculation."""
        # Upward momentum
        prices = [100 + i for i in range(10)]
        momentum, strength = self.strategy.calculate_momentum(prices, 5)

        self.assertGreater(momentum, 0)
        self.assertEqual(strength, 1.0)

    def test_calculate_position_size(self):
        """Test position size calculation."""
        # Normal volatility, low confidence
        size = self.strategy.calculate_position_size(
            volatility=0.02,  # 2% daily volatility
            equity=10000,
            confidence=0.5,
            regime=MarketRegime.RANGING,
        )

        self.assertGreater(size, 0)
        self.assertLessEqual(size, self.strategy.max_position_pct)

    def test_calculate_confidence(self):
        """Test confidence calculation."""
        # Mock flow data
        flow_data = MagicMock()
        flow_data.momentum = 0.02
        flow_data.volume_ratio = 1.5
        flow_data.trend_strength = 0.8
        flow_data.order_book_imbalance = 0.6

        # Mock regime performance
        regime_perf = {"ranging": {"win_rate": 0.6}}

        confidence = self.strategy.calculate_confidence(flow_data, regime_perf)

        self.assertGreater(confidence, 0)
        self.assertLessEqual(confidence, 1)

    def test_generate_signals_insufficient_data(self):
        """Test signal generation with insufficient data."""
        # Not enough data
        ohlcv_data = [[0, 100, 105, 95, 100, 1000]] * 5

        import asyncio
        signals = asyncio.run(self.strategy.generate_signals("BTC/USD", ohlcv_data))

        self.assertEqual(len(signals), 0)

    def test_strategy_factory(self):
        """Test strategy factory."""
        strategy = create_market_flow_strategy(
            min_confidence=0.6,
            base_position_pct=0.03,
        )

        self.assertEqual(strategy.min_confidence, 0.6)
        self.assertEqual(strategy.base_position_pct, 0.03)


class TestConstantAdaptationLoop(unittest.TestCase):
    """Tests for ConstantAdaptationLoop."""

    def setUp(self):
        """Set up test fixtures."""
        config = AdaptationConfig(
            cycle_interval_seconds=15.0,
            adaptation_enabled=True,
            min_win_rate=0.40,
            max_drawdown=0.10,
        )
        strategy = create_market_flow_strategy()
        self.loop = ConstantAdaptationLoop(config=config, strategy=strategy)

    def test_ingest_market_data(self):
        """Test market data ingestion."""
        ohlcv_data = [
            [0, 100, 105, 95, 100, 1000],
            [1, 101, 106, 96, 101, 1100],
        ]

        import asyncio

        async def test():
            await self.loop.ingest_market_data("BTC/USD", ohlcv_data)
            self.assertIn("BTC/USD", self.loop._market_data_cache)

        asyncio.run(test())

    def test_adapt_parameters_no_trades(self):
        """Test adaptation with no trades."""
        adaptations = self.loop.adapt_parameters()

        # Should wait for more trades
        self.assertEqual(adaptations.get("action"), "wait")

    def test_adapt_parameters_low_win_rate(self):
        """Test adaptation with low win rate."""
        # Add losing trades
        for _ in range(10):
            self.loop.strategy.record_trade("BTC/USD", "buy", 100, 95, 0.02)

        adaptations = self.loop.adapt_parameters()

        # Should tighten
        self.assertEqual(adaptations.get("action"), "tighten")

    def test_adapt_parameters_good_win_rate(self):
        """Test adaptation with good win rate."""
        # Add winning trades
        for _ in range(10):
            self.loop.strategy.record_trade("BTC/USD", "buy", 100, 105, 0.02)

        adaptations = self.loop.adapt_parameters()

        # Should relax
        self.assertEqual(adaptations.get("action"), "relax")

    def test_record_trade(self):
        """Test recording trade."""
        self.loop.record_trade("BTC/USD", "buy", 100, 105, 0.02)

        self.assertEqual(self.loop._state.total_trades, 1)
        self.assertGreater(self.loop._state.equity, 10000)

    def test_emergency_stop_disabled(self):
        """Test emergency stop when disabled."""
        config = AdaptationConfig()
        config.emergency_stop_enabled = False
        
        strategy = create_market_flow_strategy()
        loop = ConstantAdaptationLoop(config=config, strategy=strategy)
        
        # Even with high drawdown, should not stop when disabled
        loop._state.peak_equity = 10000
        loop._state.equity = 8500
        loop._state.total_trades = 30

        should_stop = loop.check_emergency_stop()
        self.assertFalse(should_stop)

    def test_get_state(self):
        """Test getting state."""
        state = self.loop.get_state()

        self.assertIsNotNone(state.cycle)
        self.assertEqual(state.equity, 10000)  # Default


class TestStrategyPerformance(unittest.TestCase):
    """Tests for StrategyPerformance."""

    def test_record_trade_winning(self):
        """Test recording winning trade."""
        strategy = create_market_flow_strategy()
        strategy.record_trade("BTC/USD", "buy", 100, 110, 0.02)

        perf = strategy.get_performance()
        self.assertEqual(perf.total_trades, 1)
        self.assertEqual(perf.winning_trades, 1)
        self.assertGreater(perf.total_pnl, 0)

    def test_record_trade_losing(self):
        """Test recording losing trade."""
        strategy = create_market_flow_strategy()
        strategy.record_trade("BTC/USD", "buy", 100, 95, 0.02)

        perf = strategy.get_performance()
        self.assertEqual(perf.total_trades, 1)
        self.assertEqual(perf.losing_trades, 1)
        self.assertLess(perf.total_pnl, 0)

    def test_consecutive_streaks(self):
        """Test consecutive win/loss tracking."""
        strategy = create_market_flow_strategy()

        # Add wins
        for _ in range(3):
            strategy.record_trade("BTC/USD", "buy", 100, 105, 0.02)

        # Add losses
        for _ in range(2):
            strategy.record_trade("BTC/USD", "buy", 100, 95, 0.02)

        perf = strategy.get_performance()
        self.assertEqual(perf.consecutive_wins, 0)  # Reset after loss
        self.assertEqual(perf.consecutive_losses, 2)


if __name__ == "__main__":
    unittest.main()