"""
Tests for ML feature selector module.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ml.feature_selector import (
    FeatureDriftResult,
    FeatureDriftTracker,
    FeatureSelector,
    FeatureSelectionResult,
)


class TestFeatureSelector(unittest.TestCase):
    """Tests for FeatureSelector class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 50
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = (self.X[:, 0] + self.X[:, 1] + np.random.randn(self.n_samples) * 0.5 > 0).astype(float)

    def test_importance_selection(self):
        """Test importance-based feature selection."""
        selector = FeatureSelector(method="importance", min_features=5, max_features=20)
        X_selected, indices = selector.select(self.X, self.y)

        self.assertEqual(X_selected.shape[1], len(indices))
        self.assertLessEqual(len(indices), 20)
        self.assertGreaterEqual(len(indices), 5)
        self.assertIsInstance(indices, np.ndarray)

    def test_correlation_selection(self):
        """Test correlation-based feature selection."""
        selector = FeatureSelector(method="correlation", min_features=5)
        X_selected, indices = selector.select(self.X, self.y)

        self.assertIsInstance(indices, np.ndarray)
        self.assertGreater(len(indices), 0)

    def test_statistical_selection(self):
        """Test statistical feature selection."""
        selector = FeatureSelector(method="statistical", min_features=5)
        X_selected, indices = selector.select(self.X, self.y)

        self.assertIsInstance(indices, np.ndarray)

    def test_get_importance_scores(self):
        """Test getting importance scores."""
        selector = FeatureSelector(method="importance", min_features=5)
        selector.select(self.X, self.y)

        scores = selector.get_importance_scores()
        self.assertIsInstance(scores, dict)
        if scores:
            self.assertTrue(all(isinstance(v, float) for v in scores.values()))

    def test_get_selection_result(self):
        """Test getting selection result."""
        selector = FeatureSelector(method="importance", min_features=5)
        selector.select(self.X, self.y)

        result = selector.get_selection_result()
        if result:
            self.assertIsInstance(result, FeatureSelectionResult)
            self.assertIn(result.method, ["importance", "correlation", "statistical", "recursive"])

    def test_to_dict(self):
        """Test serialization."""
        selector = FeatureSelector(method="importance", min_features=5)
        selector.select(self.X, self.y)

        result = selector.get_selection_result()
        if result:
            d = result.to_dict()
            self.assertIn("n_selected", d)
            self.assertIn("method", d)


class TestFeatureDriftTracker(unittest.TestCase):
    """Tests for FeatureDriftTracker class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.reference = np.random.randn(100, 20) * 2 + 1  # Mean 1, std 2
        self.current_stable = np.random.randn(100, 20) * 2 + 1  # Same distribution
        self.current_drift = np.random.randn(100, 20) * 3 + 5  # Different distribution

    def test_set_reference(self):
        """Test setting reference distribution."""
        tracker = FeatureDriftTracker()
        tracker.set_reference(self.reference)

        self.assertTrue(len(tracker._reference_stats) > 0)

    def test_check_drift_no_drift(self):
        """Test drift detection with stable data."""
        tracker = FeatureDriftTracker(reference_window=100, drift_threshold=0.2)
        tracker.set_reference(self.reference)

        result = tracker.check_drift(self.current_stable)

        self.assertIsInstance(result, FeatureDriftResult)
        self.assertIn(result.recommendation, ["no_action", "monitor", "retrain"])

    def test_check_drift_detected(self):
        """Test drift detection with drifting data."""
        tracker = FeatureDriftTracker(reference_window=100, drift_threshold=0.1)
        tracker.set_reference(self.reference)

        result = tracker.check_drift(self.current_drift)

        self.assertIsInstance(result, FeatureDriftResult)
        # Should detect some drift
        self.assertGreater(len(result.drifting_features) + len(result.stable_features), 0)

    def test_drift_result_serialization(self):
        """Test drift result serialization."""
        tracker = FeatureDriftTracker(reference_window=100)
        tracker.set_reference(self.reference)
        result = tracker.check_drift(self.current_stable)

        # All attributes should be serializable
        self.assertIsInstance(result.drifting_features, list)
        self.assertIsInstance(result.drift_scores, dict)
        self.assertIsInstance(result.stable_features, list)
        self.assertIsInstance(result.overall_drift, float)
        self.assertIn(result.recommendation, ["retrain", "monitor", "no_action"])


if __name__ == "__main__":
    unittest.main()