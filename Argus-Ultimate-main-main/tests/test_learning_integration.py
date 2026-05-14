# pyright: reportMissingImports=false
"""
Integration tests for all advanced learning systems.

Tests the complete integration of:
- Knowledge Distillation
- Multi-Agent RL
- RLHF
- Uncertainty Quantification
- Adversarial Training
- Active Learning
- Transfer Learning
- Learning Health Dashboard
- Quantum RL
- Advanced Learning Integration
"""

from __future__ import annotations

import logging
import unittest
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class TestAdvancedLearningIntegration(unittest.TestCase):
    """Tests for the complete advanced learning integration."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from ml.advanced_learning_integration import (
                AdvancedLearningOrchestrator, IntegratedTradingLoop, LearningConfig, LearningMode
            )
            self.config = LearningConfig(
                mode=LearningMode.LIGHTWEIGHT,
                enable_quantum_rl=False,  # Disable for faster tests
                enable_dashboard=False
            )
            self.orchestrator = AdvancedLearningOrchestrator(self.config)
            self.trading_loop = IntegratedTradingLoop(self.orchestrator)
        except ImportError as e:
            self.skipTest(f"Advanced learning integration not available: {e}")

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        self.assertIsNotNone(self.orchestrator)
        self.assertTrue(self.orchestrator.is_initialized)
        self.assertGreater(self.orchestrator.metrics.active_systems, 0)

    def test_trading_decision(self):
        """Test making a trading decision."""
        market_data = {
            "close": 50000.0,
            "open": 49500.0,
            "high": 51000.0,
            "low": 49000.0,
            "volume": 1000.0
        }

        decision = self.trading_loop.process_market_data(market_data)

        self.assertIn("action", decision)
        self.assertIn(decision["action"], [0, 1, 2, 3])
        self.assertIn("confidence", decision)
        self.assertIn("uncertainty", decision)
        self.assertGreater(decision["confidence"], 0)

    def test_multiple_decisions(self):
        """Test making multiple trading decisions."""
        for i in range(10):
            market_data = {
                "close": 50000.0 + i * 100,
                "open": 49500.0 + i * 100,
                "high": 51000.0 + i * 100,
                "low": 49000.0 + i * 100,
                "volume": 1000.0
            }
            decision = self.trading_loop.process_market_data(market_data)
            self.assertIsNotNone(decision)

        self.assertEqual(self.orchestrator.metrics.total_decisions, 10)

    def test_feedback_learning(self):
        """Test that feedback updates the learning systems."""
        market_data = {
            "close": 50000.0,
            "open": 49500.0,
            "high": 51000.0,
            "low": 49000.0,
            "volume": 1000.0
        }

        decision = self.trading_loop.process_market_data(market_data)

        # Record trade outcome
        self.trading_loop.record_trade_outcome(
            market_data, decision, 0.5, human_rating=0.8
        )

        # Check that metrics updated
        self.assertGreater(self.trading_loop.trade_count, 0)

    def test_performance_summary(self):
        """Test performance summary generation."""
        # Make some trades
        for i in range(5):
            market_data = {
                "close": 50000.0 + i * 100,
                "open": 49500.0 + i * 100,
                "high": 51000.0 + i * 100,
                "low": 49000.0 + i * 100,
                "volume": 1000.0
            }
            decision = self.trading_loop.process_market_data(market_data)
            self.trading_loop.record_trade_outcome(market_data, decision, 0.1)

        summary = self.trading_loop.get_performance_summary()

        self.assertIn("trading", summary)
        self.assertIn("learning", summary)
        self.assertEqual(summary["trading"]["total_trades"], 5)


class TestAdvancedLearningComponent(unittest.TestCase):
    """Tests for the component registry integration."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from core.advanced_learning_component import AdvancedLearningComponent
            self.component = AdvancedLearningComponent({
                "mode": "lightweight",
                "enable_quantum_rl": False
            })
        except ImportError as e:
            self.skipTest(f"Advanced learning component not available: {e}")

    def test_component_initialization(self):
        """Test component initialization."""
        result = self.component.initialize()
        self.assertTrue(result)
        self.assertTrue(self.component.is_initialized)

    def test_component_market_data(self):
        """Test processing market data through component."""
        self.component.initialize()

        market_data = {
            "close": 50000.0,
            "open": 49500.0,
            "high": 51000.0,
            "low": 49000.0,
            "volume": 1000.0
        }

        decision = self.component.process_market_data(market_data)
        self.assertIn("action", decision)

    def test_component_status(self):
        """Test component status reporting."""
        self.component.initialize()

        status = self.component.get_status()
        self.assertIn("initialized", status)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""

    def test_full_trading_cycle(self):
        """Test a complete trading cycle with learning."""
        try:
            from ml.advanced_learning_integration import (
                AdvancedLearningOrchestrator, IntegratedTradingLoop, LearningConfig, LearningMode
            )
        except ImportError:
            self.skipTest("Advanced learning integration not available")

        config = LearningConfig(
            mode=LearningMode.LIGHTWEIGHT,
            enable_quantum_rl=False,
            enable_dashboard=False
        )
        orchestrator = AdvancedLearningOrchestrator(config)
        trading_loop = IntegratedTradingLoop(orchestrator)

        # Simulate trading cycle
        for i in range(20):
            # Market data
            market_data = {
                "close": 50000.0 + np.random.randn() * 1000,
                "open": 49500.0 + np.random.randn() * 1000,
                "high": 51000.0 + np.random.randn() * 1000,
                "low": 49000.0 + np.random.randn() * 1000,
                "volume": 1000.0 + np.random.randn() * 200
            }

            # Make decision
            decision = trading_loop.process_market_data(market_data)

            # Simulate trade outcome
            reward = np.random.randn() * 0.5
            human_rating = 0.5 + np.random.rand() * 0.5

            # Record outcome
            trading_loop.record_trade_outcome(market_data, decision, reward, human_rating)

        # Get final summary
        summary = trading_loop.get_performance_summary()

        self.assertEqual(summary["trading"]["total_trades"], 20)
        self.assertIsNotNone(summary["learning"])

        print(f"\n✓ End-to-end test complete:")
        print(f"  Trades: {summary['trading']['total_trades']}")
        print(f"  PnL: {summary['trading']['total_pnl']:.4f}")
        print(f"  Active Systems: {summary['learning']['metrics']['active_systems']}")


def run_tests():
    """Run all integration tests."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAdvancedLearningIntegration))
    suite.addTest(unittest.makeSuite(TestAdvancedLearningComponent))
    suite.addTest(unittest.makeSuite(TestEndToEnd))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)