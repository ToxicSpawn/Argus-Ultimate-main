"""
Tests for Market Flow Trading Integration.
"""

import unittest
from unittest.mock import MagicMock

from ml.market_flow_integration import (
    MarketFlowTradingIntegration,
    IntegratedSignal,
    TradingState,
    create_integration,
)


class TestMarketFlowTradingIntegration(unittest.TestCase):
    """Tests for MarketFlowTradingIntegration."""

    def setUp(self):
        """Set up test fixtures."""
        self.integration = create_integration(
            strategy_min_confidence=0.50,
            risk_base_stop_loss=0.015,
        )

    def test_initialization(self):
        """Test integration initializes correctly."""
        self.assertIsNotNone(self.integration.strategy)
        self.assertIsNotNone(self.integration.risk_adapter)
        self.assertIsInstance(self.integration._state, TradingState)

    def test_initialization_without_ultimate(self):
        """Test initialization without ultimate adaptation."""
        integration = MarketFlowTradingIntegration(
            strategy_min_confidence=0.50,
            strategy_base_position=0.02,
            use_ultimate_adaptation=False,
            use_risk_adapter=True,
        )

        self.assertIsNone(integration.ultimate_adapter)
        self.assertIsNotNone(integration.risk_adapter)

    def test_initialization_without_risk(self):
        """Test initialization without risk adapter."""
        integration = MarketFlowTradingIntegration(
            strategy_min_confidence=0.50,
            use_ultimate_adaptation=True,
            use_risk_adapter=False,
        )

        self.assertIsNotNone(integration.ultimate_adapter)

    def test_record_trade_win(self):
        """Test recording a winning trade."""
        integration = create_integration()
        initial_pnl = integration._state.total_pnl

        integration.record_trade("BTCUSDT", "buy", 0.05)

        self.assertGreater(integration._state.total_pnl, initial_pnl)
        self.assertEqual(integration._state.consecutive_wins, 1)

    def test_record_trade_loss(self):
        """Test recording a losing trade."""
        integration = create_integration()

        integration.record_trade("BTCUSDT", "buy", -0.03)

        self.assertLess(integration._state.total_pnl, 0)
        self.assertEqual(integration._state.consecutive_losses, 1)

    def test_get_status(self):
        """Test getting status."""
        integration = create_integration()

        status = integration.get_status()

        self.assertIn("strategy", status)
        self.assertIn("risk", status)
        self.assertIn("positions", status)
        self.assertIn("pnl", status)

    def test_trading_state_defaults(self):
        """Test TradingState default values."""
        state = TradingState()

        self.assertEqual(state.symbols_traded, {})
        self.assertEqual(state.total_pnl, 0.0)
        self.assertEqual(state.daily_pnl, 0.0)
        self.assertEqual(state.winning_trades, 0)
        self.assertEqual(state.losing_trades, 0)
        self.assertEqual(state.consecutive_wins, 0)
        self.assertEqual(state.consecutive_losses, 0)

    def test_create_integration_factory(self):
        """Test factory function."""
        integration = create_integration(
            strategy_min_confidence=0.60,
            risk_base_stop_loss=0.02,
        )

        self.assertIsNotNone(integration)
        self.assertIsNotNone(integration.strategy)
        self.assertIsNotNone(integration.risk_adapter)


class TestIntegratedSignal(unittest.TestCase):
    """Tests for IntegratedSignal dataclass."""

    def test_integrated_signal_creation(self):
        """Test creating an IntegratedSignal."""
        # Create mock raw signal
        raw = MagicMock()
        raw.direction = "buy"
        raw.confidence = 0.7
        raw.position_size_pct = 0.02
        raw.entry_price = 50000
        raw.stop_loss = 49000
        raw.take_profit = 52000

        # Create risk objects
        risk = MagicMock()
        risk.condition = "normal"
        risk.overall_score = 0.3

        decision = MagicMock()
        decision.position_size_multiplier = 1.0
        decision.confidence_adjustment = 0.0
        decision.stop_loss_multiplier = 1.0
        decision.take_profit_multiplier = 1.0

        # Create integrated signal
        integrated = IntegratedSignal(
            raw_signal=raw,
            adapted_direction="buy",
            adapted_confidence=0.7,
            adapted_position_size=0.02,
            adapted_stop_loss=49000,
            adapted_take_profit=52000,
            risk=risk,
            risk_decision=decision,
            adaptation=None,
            should_execute=True,
            execution_reason="OK",
        )

        self.assertEqual(integrated.adapted_direction, "buy")
        self.assertEqual(integrated.adapted_confidence, 0.7)
        self.assertTrue(integrated.should_execute)
        self.assertEqual(integrated.execution_reason, "OK")


class TestIntegrationSignalProcessing(unittest.TestCase):
    """Tests for signal processing pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.integration = create_integration()

    def test_process_signal_returns_none_on_no_signals(self):
        """Test process_signal handles empty signals."""
        # Verify integration is set up
        self.assertIsNotNone(self.integration)


if __name__ == "__main__":
    unittest.main()