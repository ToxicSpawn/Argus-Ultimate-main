# pyright: reportMissingImports=false
"""
Integration tests for Advanced Learning System with Component Registry and Trading Loop.

Tests verify:
- ComponentRegistry correctly initializes the Advanced Learning Orchestrator
- on_cycle() provides learning advisory
- Trading loop records outcomes correctly
- Learning systems update properly
"""

from __future__ import annotations

import asyncio
import logging
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np

logger = logging.getLogger(__name__)


class TestComponentRegistryIntegration(unittest.TestCase):
    """Tests for ComponentRegistry integration with Advanced Learning."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the system
        self.mock_system = MagicMock()
        self.mock_system.config = MagicMock()

    def test_advanced_learning_property(self):
        """Test that the advanced_learning property is accessible."""
        from core.component_registry import ComponentRegistry
        
        registry = ComponentRegistry(system=self.mock_system)
        
        # Check property exists
        self.assertTrue(hasattr(registry, "advanced_learning"))

    def test_learning_trading_loop_property_exists(self):
        """Test that learning_trading_loop can be accessed via get()."""
        from core.component_registry import ComponentRegistry
        
        registry = ComponentRegistry(system=self.mock_system)
        
        # Check get method works
        result = registry.get("learning_trading_loop")
        # Initially should be None since we haven't registered
        self.assertIsNone(result)

    def test_component_registry_registers_advanced_learning(self):
        """Test that register_all() attempts to register advanced learning."""
        from core.component_registry import ComponentRegistry
        
        registry = ComponentRegistry(system=self.mock_system)
        
        # Check that register_all can be called
        # (It may not fully succeed due to missing dependencies, but should not crash)
        try:
            asyncio.run(registry.register_all())
        except Exception as e:
            # Expected if dependencies not installed
            logger.info(f"register_all() raised: {e}")
        
        # Check that components dict has the entry (even if None)
        self.assertIn("advanced_learning", registry._components)
        self.assertIn("learning_trading_loop", registry._components)


class TestAdvancedLearningOrchestrator(unittest.TestCase):
    """Tests for AdvancedLearningOrchestrator integration."""

    def test_orchestrator_initialization(self):
        """Test orchestrator can be initialized."""
        from ml.advanced_learning_integration import (
            AdvancedLearningOrchestrator, LearningConfig, LearningMode
        )
        
        config = LearningConfig(
            mode=LearningMode.LIGHTWEIGHT,  # Faster initialization
            enable_quantum_rl=False,  # Disable for faster tests
            enable_dashboard=False,
        )
        
        orchestrator = AdvancedLearningOrchestrator(config=config)
        
        self.assertIsNotNone(orchestrator)
        self.assertTrue(orchestrator.is_initialized)

    def test_trading_loop_initialization(self):
        """Test IntegratedTradingLoop can be initialized."""
        from ml.advanced_learning_integration import (
            AdvancedLearningOrchestrator, IntegratedTradingLoop, LearningConfig, LearningMode
        )
        
        config = LearningConfig(mode=LearningMode.LIGHTWEIGHT)
        orchestrator = AdvancedLearningOrchestrator(config=config)
        trading_loop = IntegratedTradingLoop(learning_orchestrator=orchestrator)
        
        self.assertIsNotNone(trading_loop)
        self.assertEqual(trading_loop.orchestrator, orchestrator)

    def test_make_decision(self):
        """Test making a trading decision."""
        from ml.advanced_learning_integration import (
            AdvancedLearningOrchestrator, IntegratedTradingLoop, LearningConfig, LearningMode
        )
        
        config = LearningConfig(mode=LearningMode.LIGHTWEIGHT)
        orchestrator = AdvancedLearningOrchestrator(config=config)
        trading_loop = IntegratedTradingLoop(learning_orchestrator=orchestrator)
        
        # Create sample market data
        market_data = {
            "close": 50000.0,
            "open": 49800.0,
            "high": 50200.0,
            "low": 49700.0,
            "volume": 1000.0,
        }
        
        result = trading_loop.process_market_data(market_data)
        
        self.assertIn("action", result)
        self.assertIn("confidence", result)
        self.assertIn("uncertainty", result)
        self.assertIn("position_size", result)
        self.assertIn("action_name", result)
        
        # Action should be 0-3
        self.assertIn(result["action"], [0, 1, 2, 3])

    def test_record_trade_outcome(self):
        """Test recording trade outcomes."""
        from ml.advanced_learning_integration import (
            AdvancedLearningOrchestrator, IntegratedTradingLoop, LearningConfig, LearningMode
        )
        
        config = LearningConfig(mode=LearningMode.LIGHTWEIGHT)
        orchestrator = AdvancedLearningOrchestrator(config=config)
        trading_loop = IntegratedTradingLoop(learning_orchestrator=orchestrator)
        
        # First make a decision
        market_data = {"close": 50000.0, "volume": 1000.0}
        decision = trading_loop.process_market_data(market_data)
        
        # Record outcome
        trading_loop.record_trade_outcome(
            market_data=market_data,
            decision=decision,
            actual_reward=0.02,  # 2% profit
            human_rating=None
        )
        
        # Check that PnL was updated
        self.assertEqual(trading_loop.total_pnl, 0.02)
        self.assertEqual(trading_loop.trade_count, 1)

    def test_performance_summary(self):
        """Test getting performance summary."""
        from ml.advanced_learning_integration import (
            AdvancedLearningOrchestrator, IntegratedTradingLoop, LearningConfig, LearningMode
        )
        
        config = LearningConfig(mode=LearningMode.LIGHTWEIGHT)
        orchestrator = AdvancedLearningOrchestrator(config=config)
        trading_loop = IntegratedTradingLoop(learning_orchestrator=orchestrator)
        
        # Make a few decisions and record outcomes
        for i in range(3):
            market_data = {"close": 50000.0 + i * 100, "volume": 1000.0}
            decision = trading_loop.process_market_data(market_data)
            trading_loop.record_trade_outcome(
                market_data=market_data,
                decision=decision,
                actual_reward=0.01 * (i + 1),
            )
        
        summary = trading_loop.get_performance_summary()
        
        self.assertEqual(summary["trading"]["total_trades"], 3)
        self.assertIn("learning", summary)
        self.assertIn("metrics", summary["learning"])


class TestLearningMode(unittest.TestCase):
    """Tests for LearningMode configuration."""

    def test_lightweight_mode_disables_quantum(self):
        """Test that lightweight mode disables quantum RL."""
        from ml.advanced_learning_integration import (
            AdvancedLearningOrchestrator, LearningConfig, LearningMode
        )
        
        config = LearningConfig(
            mode=LearningMode.LIGHTWEIGHT,
            enable_quantum_rl=True,  # Initially True
        )
        
        orchestrator = AdvancedLearningOrchestrator(config=config)
        orchestrator.enable_learning_mode(LearningMode.LIGHTWEIGHT)
        
        self.assertFalse(orchestrator.config.enable_quantum_rl)

    def test_quantum_mode_enables_quantum(self):
        """Test that quantum mode enables quantum RL."""
        from ml.advanced_learning_integration import (
            AdvancedLearningOrchestrator, LearningConfig, LearningMode
        )
        
        config = LearningConfig(
            mode=LearningMode.CLASSICAL,
            enable_quantum_rl=False,  # Initially False
        )
        
        orchestrator = AdvancedLearningOrchestrator(config=config)
        orchestrator.enable_learning_mode(LearningMode.QUANTUM)
        
        self.assertTrue(orchestrator.config.enable_quantum_rl)


class TestSystemStatus(unittest.TestCase):
    """Tests for system status reporting."""

    def test_get_system_status(self):
        """Test getting system status."""
        from ml.advanced_learning_integration import (
            AdvancedLearningOrchestrator, LearningConfig, LearningMode
        )
        
        config = LearningConfig(mode=LearningMode.LIGHTWEIGHT)
        orchestrator = AdvancedLearningOrchestrator(config=config)
        
        status = orchestrator.get_system_status()
        
        self.assertIn("initialized", status)
        self.assertIn("config", status)
        self.assertIn("metrics", status)
        self.assertIn("systems", status)
        
        self.assertTrue(status["initialized"])
        self.assertGreater(status["metrics"]["active_systems"], 0)


class TestOnCycleAdvisory(unittest.TestCase):
    """Tests for on_cycle learning advisory."""

    def test_on_cycle_includes_advanced_learning(self):
        """Test that on_cycle includes advanced learning advisory when registered."""
        from core.component_registry import ComponentRegistry
        from ml.advanced_learning_integration import (
            AdvancedLearningOrchestrator, IntegratedTradingLoop, LearningConfig, LearningMode
        )
        
        mock_system = MagicMock()
        mock_system.config = MagicMock()
        
        registry = ComponentRegistry(system=mock_system)
        
        # Manually register the advanced learning system
        config = LearningConfig(mode=LearningMode.LIGHTWEIGHT)
        orchestrator = AdvancedLearningOrchestrator(config=config)
        trading_loop = IntegratedTradingLoop(learning_orchestrator=orchestrator)
        
        registry._components["advanced_learning"] = orchestrator
        registry._components["learning_trading_loop"] = trading_loop
        
        # Call on_cycle
        advisory = asyncio.run(registry.on_cycle({"BTC/USD": 50000.0}))
        
        # Should include advanced learning advisory
        self.assertIn("advanced_learning", advisory)
        self.assertTrue(advisory["advanced_learning"]["enabled"])
        self.assertGreater(advisory["advanced_learning"]["active_systems"], 0)


def run_tests():
    """Run all integration tests."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestComponentRegistryIntegration))
    suite.addTest(unittest.makeSuite(TestAdvancedLearningOrchestrator))
    suite.addTest(unittest.makeSuite(TestLearningMode))
    suite.addTest(unittest.makeSuite(TestSystemStatus))
    suite.addTest(unittest.makeSuite(TestOnCycleAdvisory))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
