# pyright: reportMissingImports=false
"""
Tests for Universal Parameter Learning Engine.

Tests cover:
- Parameter registration and learning
- Context-aware parameter optimization
- Trade outcome recording
- Learning cycle execution
- Integration with trading system
"""

from __future__ import annotations

import logging
import unittest
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class TestParameterLearner(unittest.TestCase):
    """Tests for ParameterLearner class."""

    def setUp(self):
        try:
            from learning.universal_parameter_learner import (
                ParameterLearner,
                ParameterDefinition,
                ParameterType,
                ParameterCategory,
                ParameterObservation,
            )
            self.ParameterLearner = ParameterLearner
            self.ParameterDefinition = ParameterDefinition
            self.ParameterType = ParameterType
            self.ParameterCategory = ParameterCategory
            self.ParameterObservation = ParameterObservation
        except ImportError:
            self.skipTest("Parameter learning module not available")

    def test_learner_initialization(self):
        """Test learner initialization."""
        definition = self.ParameterDefinition(
            name="test_param",
            full_path="test.param",
            parameter_type=self.ParameterType.WEIGHT,
            category=self.ParameterCategory.SIGNAL,
            default_value=0.5,
            current_value=0.5,
            min_value=0.0,
            max_value=1.0,
        )
        
        learner = self.ParameterLearner(definition)
        self.assertIsNotNone(learner)
        self.assertEqual(learner.alpha, 1.0)
        self.assertEqual(learner.beta, 1.0)

    def test_observation_recording(self):
        """Test recording observations."""
        definition = self.ParameterDefinition(
            name="test_param",
            full_path="test.param",
            parameter_type=self.ParameterType.WEIGHT,
            category=self.ParameterCategory.SIGNAL,
            default_value=0.5,
            current_value=0.5,
            min_value=0.0,
            max_value=1.0,
        )
        
        learner = self.ParameterLearner(definition)
        
        # Record observations
        for i in range(20):
            observation = self.ParameterObservation(
                timestamp=datetime.now(),
                parameter_name="test_param",
                parameter_value=0.5,
                regime="trending",
                asset="BTC",
                outcome=10.0 if i % 3 != 0 else -5.0
            )
            learner.observe(observation)
        
        self.assertEqual(len(learner.observations), 20)
        self.assertGreater(learner.alpha, 1.0)


class TestParameterRegistry(unittest.TestCase):
    """Tests for ParameterRegistry class."""

    def setUp(self):
        try:
            from learning.universal_parameter_learner import ParameterRegistry
            self.registry = ParameterRegistry()
        except ImportError:
            self.skipTest("Parameter learning module not available")

    def test_registry_initialization(self):
        """Test registry initialization."""
        self.assertIsNotNone(self.registry)
        self.assertGreater(len(self.registry.parameters), 20)  # At least 20 parameters

    def test_parameter_categories(self):
        """Test parameter categorization."""
        categories = set(p.category for p in self.registry.parameters.values())
        self.assertGreater(len(categories), 3)

    def test_get_parameter(self):
        """Test parameter retrieval."""
        param = self.registry.get_parameter("whale_tracking_weight")
        self.assertIsNotNone(param)
        self.assertEqual(param.default_value, 0.15)

    def test_statistics(self):
        """Test registry statistics."""
        stats = self.registry.get_statistics()
        self.assertIn("total_parameters", stats)
        self.assertIn("by_category", stats)
        self.assertGreater(stats["total_parameters"], 20)  # At least 20 parameters


class TestUniversalParameterLearningEngine(unittest.TestCase):
    """Tests for UniversalParameterLearningEngine class."""

    def setUp(self):
        try:
            from learning.universal_parameter_learner import (
                UniversalParameterLearningEngine
            )
            self.engine = UniversalParameterLearningEngine()
        except ImportError:
            self.skipTest("Parameter learning module not available")

    def test_engine_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine)
        self.assertGreater(len(self.engine.registry.parameters), 0)

    def test_record_outcome(self):
        """Test outcome recording."""
        params = {
            "whale_tracking_weight": 0.2,
            "confidence_threshold": 0.75,
        }
        
        self.engine.record_outcome(params, 15.0)
        self.assertEqual(self.engine.total_observations, 1)

    def test_get_parameter_value(self):
        """Test parameter value retrieval."""
        value = self.engine.get_parameter_value("whale_tracking_weight")
        self.assertEqual(value, 0.15)  # Default value

    def test_context_update(self):
        """Test context updates."""
        self.engine.update_context("trending", "ETH")
        self.assertEqual(self.engine.current_regime, "trending")
        self.assertEqual(self.engine.current_asset, "ETH")

    def test_get_all_learned_values(self):
        """Test getting all learned values."""
        values = self.engine.get_all_learned_values()
        self.assertIn("whale_tracking_weight", values)
        self.assertIn("confidence_threshold", values)

    def test_learning_report(self):
        """Test learning report generation."""
        report = self.engine.get_learning_report()
        self.assertIn("total_observations", report)
        self.assertIn("registry", report)
        self.assertIn("learned_parameters", report)

    def test_reset_parameters(self):
        """Test parameter reset."""
        self.engine.reset_all_parameters()
        for param in self.engine.registry.parameters.values():
            self.assertEqual(param.current_value, param.default_value)


class TestParameterLearningIntegrator(unittest.TestCase):
    """Tests for ParameterLearningIntegrator class."""

    def setUp(self):
        try:
            from learning.universal_parameter_learner import reset_parameter_learning_engine
            reset_parameter_learning_engine()  # Reset singleton for clean tests
            
            from learning.parameter_learning_integration import (
                ParameterLearningIntegrator
            )
            self.integrator = ParameterLearningIntegrator()
            # Update cache for testing
            self.integrator._update_parameter_cache()
        except ImportError:
            self.skipTest("Parameter learning integration module not available")

    def test_integrator_initialization(self):
        """Test integrator initialization."""
        self.assertIsNotNone(self.integrator)
        self.assertIsNotNone(self.integrator.engine)

    def test_get_parameters_for_decision(self):
        """Test getting parameters for decision."""
        params = self.integrator.get_parameters_for_decision(
            {"regime": "trending", "asset": "BTC"}
        )
        self.assertIsInstance(params, dict)
        # Cache should be populated after _update_parameter_cache
        self.assertGreater(len(params), 0)

    def test_record_trade_outcome(self):
        """Test recording trade outcomes."""
        params = self.integrator.get_parameters_for_decision({"regime": "trending", "asset": "BTC"})
        self.integrator.record_trade_outcome(params, 10.0)
        
        self.assertEqual(self.integrator.total_trades, 1)

    def test_get_signal_weights(self):
        """Test getting signal weights."""
        weights = self.integrator.get_signal_weights()
        
        self.assertIn("whale_tracking", weights)
        self.assertIn("exchange_flow", weights)
        self.assertIn("social_sentiment", weights)
        self.assertIn("news_sentiment", weights)
        self.assertIn("derivatives", weights)

    def test_get_risk_parameters(self):
        """Test getting risk parameters."""
        risk_params = self.integrator.get_risk_parameters()
        
        self.assertIn("confidence_threshold", risk_params)
        self.assertIn("position_size_multiplier", risk_params)

    def test_get_status(self):
        """Test getting status."""
        status = self.integrator.get_status()
        
        self.assertIn("total_trades", status)
        self.assertIn("parameters_tracked", status)
        self.assertIn("learning_mode", status)

    def test_create_hook(self):
        """Test hook creation."""
        from learning.parameter_learning_integration import create_parameter_learning_hook
        
        hook = create_parameter_learning_hook(
            learning_interval_seconds=0.001,
            auto_start=False  # Don't auto-start in tests
        )
        
        self.assertIn("get_params", hook)
        self.assertIn("record_outcome", hook)
        self.assertIn("record_signal", hook)
        self.assertIn("record_tick", hook)
        self.assertIn("run_learning", hook)
        self.assertIn("get_status", hook)
        self.assertIn("start_learning", hook)
        self.assertIn("stop_learning", hook)
        self.assertIn("set_interval", hook)
        self.assertIn("enable_tick_learning", hook)


class TestMarketSpeedLearning(unittest.TestCase):
    """Tests for market-speed event-driven learning."""

    def setUp(self):
        try:
            from learning.universal_parameter_learner import reset_parameter_learning_engine
            reset_parameter_learning_engine()  # Reset singleton for clean tests
            
            from learning.parameter_learning_integration import (
                ParameterLearningIntegrator
            )
            self.integrator = ParameterLearningIntegrator()
            # Update cache for testing
            self.integrator._update_parameter_cache()
        except ImportError:
            self.skipTest("Parameter learning integration module not available")

    def test_start_stop_market_speed_learning(self):
        """Test starting and stopping market-speed learning."""
        self.assertFalse(self.integrator._background_running)
        
        self.integrator.start_market_speed_learning()
        self.assertTrue(self.integrator._background_running)
        self.assertIsNotNone(self.integrator._background_thread)
        
        # Let it run briefly
        import time
        time.sleep(0.3)
        
        self.integrator.stop_continuous_learning()
        self.assertFalse(self.integrator._background_running)

    def test_event_driven_learning_on_trade(self):
        """Test that trades trigger instant learning."""
        import numpy as np
        
        # Start learning
        self.integrator.start_market_speed_learning()
        
        # Simulate 20 trades
        for i in range(20):
            params = {
                "whale_tracking_weight": np.random.uniform(0.1, 0.3),
                "confidence_threshold": np.random.uniform(0.5, 0.9),
            }
            pnl = np.random.randn() * 10 + 5  # Mostly positive
            
            self.integrator.record_trade_outcome(params, pnl, {"regime": "trending"})
        
        status = self.integrator.get_status()
        self.assertEqual(status["total_trades"], 20)
        
        self.integrator.stop_continuous_learning()

    def test_signal_learning(self):
        """Test signal-based learning."""
        import numpy as np
        
        self.integrator.start_market_speed_learning()
        
        # Simulate 10 signals
        for i in range(10):
            params = {"rsi_period": 14.0 + np.random.uniform(-2, 2)}
            was_correct = np.random.random() > 0.4  # 60% accuracy
            strength = np.random.uniform(0.3, 0.8)
            
            self.integrator.record_signal(params, strength, was_correct)
        
        status = self.integrator.get_status()
        self.assertEqual(status["total_signals"], 10)
        
        self.integrator.stop_continuous_learning()

    def test_tick_learning(self):
        """Test tick-level learning (HFT mode)."""
        self.integrator.enable_tick_learning()
        self.assertTrue(self.integrator._learn_on_tick)
        
        self.integrator.start_market_speed_learning()
        
        # Simulate 100 ticks
        for i in range(100):
            self.integrator.record_tick(
                price=50000.0 + i,
                volume=100.0,
                metadata={"price_change": 0.001}
            )
        
        status = self.integrator.get_status()
        self.assertEqual(status["total_ticks"], 100)
        
        self.integrator.stop_continuous_learning()

    def test_regime_change_learning(self):
        """Test regime change triggers learning."""
        self.integrator.start_market_speed_learning()
        
        # Record regime change
        self.integrator.record_regime_change("trending", "ranging", 0.8)
        
        # Check context was updated
        self.assertEqual(self.integrator.engine.current_regime, "ranging")
        
        self.integrator.stop_continuous_learning()

    def test_instant_parameter_read(self):
        """Test that parameter reads are instant (cached)."""
        import time
        
        self.integrator.start_market_speed_learning()
        
        # Time parameter read
        start = time.perf_counter()
        for _ in range(1000):
            params = self.integrator.get_parameters_for_decision()
        elapsed = time.perf_counter() - start
        
        # 1000 reads should take <10ms (10 microseconds each)
        self.assertLess(elapsed, 0.01)
        
        self.integrator.stop_continuous_learning()

    def test_market_speed_stats(self):
        """Test market-speed learning statistics."""
        self.integrator.start_market_speed_learning()
        
        # Simulate some events
        for i in range(5):
            self.integrator.record_trade_outcome({}, 1.0)
        
        status = self.integrator.get_status()
        self.assertEqual(status["learning_mode"], "MARKET_SPEED_EVENT_DRIVEN")
        self.assertIn("market_speed", status)
        self.assertIn("total_events", status["market_speed"])
        
        self.integrator.stop_continuous_learning()

    def test_update_callback(self):
        """Test update callback registration."""
        callback_called = []
        
        def callback(result):
            callback_called.append(result)
        
        self.integrator.register_update_callback(callback)
        self.assertEqual(len(self.integrator._on_update_callbacks), 1)

    def test_get_learning_hook_global(self):
        """Test global hook getter with market-speed."""
        from learning.parameter_learning_integration import get_learning_hook
        
        hook = get_learning_hook(learning_interval_seconds=0.001)
        
        self.assertIn("get_params", hook)
        self.assertIn("record_signal", hook)
        self.assertIn("record_tick", hook)
        self.assertIn("integrator", hook)
        
        # Stop the learning thread
        hook["stop_learning"]()

    def test_integration_with_mock_trading(self):
        """Test integration with mock trading."""
        from learning.parameter_learning_integration import (
            ParameterLearningIntegrator
        )
        
        integrator = ParameterLearningIntegrator()
        integrator.start_market_speed_learning()
        
        # Simulate 50 trades
        import numpy as np
        for i in range(50):
            params = integrator.get_parameters_for_decision(
                {"regime": "trending", "asset": "BTC"}
            )
            pnl = np.random.randn() * 10 + 2
            integrator.record_trade_outcome(params, pnl)
        
        status = integrator.get_status()
        self.assertEqual(status["total_trades"], 50)
        
        integrator.stop_continuous_learning()


def run_tests():
    """Run all tests."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestParameterLearner))
    suite.addTest(unittest.makeSuite(TestParameterRegistry))
    suite.addTest(unittest.makeSuite(TestUniversalParameterLearningEngine))
    suite.addTest(unittest.makeSuite(TestParameterLearningIntegrator))
    suite.addTest(unittest.makeSuite(TestMarketSpeedLearning))
    suite.addTest(unittest.makeSuite(TestParameterLearningEndToEnd))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
