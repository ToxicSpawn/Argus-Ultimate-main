"""
Tests for ML master orchestrator module.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ml.master_ml_orchestrator import (
    MLOrchestrator,
    MLOrchestratorConfig,
    MLOrchestratorResult,
    create_orchestrator,
)


class TestMLOrchestratorConfig(unittest.TestCase):
    """Tests for MLOrchestratorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = MLOrchestratorConfig()

        self.assertEqual(config.feature_categories, ["price", "technical", "volume"])
        self.assertEqual(config.models, ["xgb", "lgb"])
        self.assertEqual(config.ensemble_method, "dynamic")
        self.assertTrue(config.online_learning_enabled)
        self.assertTrue(config.drift_detection_enabled)

    def test_custom_config(self):
        """Test custom configuration."""
        config = MLOrchestratorConfig(
            models=["xgb", "lgb", "rfr"],
            ensemble_method="stacking",
            n_validation_splits=10,
        )

        self.assertEqual(len(config.models), 3)
        self.assertEqual(config.ensemble_method, "stacking")
        self.assertEqual(config.n_validation_splits, 10)


class TestMLOrchestrator(unittest.TestCase):
    """Tests for MLOrchestrator class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 20
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = (np.sum(self.X, axis=1) > 0).astype(float)

    def test_init(self):
        """Test orchestrator initialization."""
        orchestrator = MLOrchestrator()

        self.assertIsInstance(orchestrator.config, MLOrchestratorConfig)
        self.assertFalse(orchestrator._fitted)

    def test_configure(self):
        """Test configuration update."""
        orchestrator = MLOrchestrator()
        orchestrator.configure(
            models=["xgb"],
            ensemble_method="weighted_average",
        )

        self.assertEqual(orchestrator.config.models, ["xgb"])
        self.assertEqual(orchestrator.config.ensemble_method, "weighted_average")

    def test_fit_simple(self):
        """Test simple fitting."""
        orchestrator = MLOrchestrator()
        orchestrator.fit(self.X, self.y)

        self.assertTrue(orchestrator._fitted)

    def test_fit_min_samples(self):
        """Test fitting with insufficient samples."""
        orchestrator = MLOrchestrator()

        with self.assertRaises(ValueError):
            orchestrator.fit(self.X[:10], self.y[:10])

    def test_predict_not_fitted(self):
        """Test prediction when not fitted."""
        orchestrator = MLOrchestrator()
        result = orchestrator.predict(self.X[:10])

        self.assertIsInstance(result, MLOrchestratorResult)
        self.assertEqual(result.action, "hold")
        self.assertEqual(result.confidence, 0.0)

    def test_predict_fitted(self):
        """Test prediction when fitted."""
        orchestrator = MLOrchestrator()
        orchestrator.fit(self.X, self.y)

        result = orchestrator.predict(self.X[:10])

        self.assertIsInstance(result, MLOrchestratorResult)
        self.assertIn(result.action, ["buy", "sell", "hold"])
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_health_check(self):
        """Test health check."""
        orchestrator = MLOrchestrator()
        orchestrator.fit(self.X, self.y)

        health = orchestrator.health_check()

        self.assertTrue(health["fitted"])
        self.assertIn("n_predictions", health)

    def test_get_feature_importance(self):
        """Test getting feature importance."""
        orchestrator = MLOrchestrator()
        orchestrator.fit(self.X, self.y)

        importance = orchestrator.get_feature_importance()

        self.assertIsInstance(importance, dict)

    def test_get_model_diversity(self):
        """Test getting model diversity."""
        orchestrator = MLOrchestrator()
        orchestrator.fit(self.X, self.y)

        diversity = orchestrator.get_model_diversity()

        self.assertGreaterEqual(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)


class TestMLOrchestratorResult(unittest.TestCase):
    """Tests for MLOrchestratorResult dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        result = MLOrchestratorResult(
            prediction=np.array([0.7]),
            confidence=0.8,
            action="buy",
            model_weights={"xgb": 0.5, "lgb": 0.5},
            features_used=["feat1", "feat2"],
            regime="TREND_UP",
            regime_confidence=0.9,
            metadata={"key": "value"},
        )

        d = result.to_dict()

        self.assertIn("prediction", d)
        self.assertEqual(d["action"], "buy")
        self.assertEqual(d["regime"], "TREND_UP")
        self.assertIn("model_weights", d)


class TestCreateOrchestrator(unittest.TestCase):
    """Tests for factory function."""

    def test_create_default(self):
        """Test creating with defaults."""
        orchestrator = create_orchestrator()

        self.assertIsInstance(orchestrator, MLOrchestrator)

    def test_create_custom(self):
        """Test creating with custom config."""
        orchestrator = create_orchestrator(
            models=["xgb", "rfr"],
            validation="walk_forward",
            online_learning=True,
        )

        self.assertIsInstance(orchestrator, MLOrchestrator)
        self.assertEqual(len(orchestrator.config.models), 2)


if __name__ == "__main__":
    unittest.main()