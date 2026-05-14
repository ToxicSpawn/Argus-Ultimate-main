"""
Tests for ML validation (bootstrap and stress testing) module.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np

from ml.ml_validation import (
    BootstrapResult,
    BootstrapValidator,
    ModelComparisonResult,
    RobustnessReport,
    StressTestResult,
    StressTester,
    run_robustness_analysis,
)


class TestBootstrapValidator(unittest.TestCase):
    """Tests for BootstrapValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 10
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = (np.sum(self.X, axis=1) > 0).astype(float)

    def test_validate_basic(self):
        """Test basic bootstrap validation."""
        validator = BootstrapValidator(n_bootstrap=50, seed=42)

        def evaluate(X_test, y_test):
            pred = np.mean(X_test, axis=1) > 0
            return float(np.mean(pred == y_test))

        result = validator.validate(evaluate, self.X, self.y)

        self.assertIsInstance(result, BootstrapResult)
        self.assertGreater(result.n_bootstrap, 0)
        self.assertGreater(result.mean, 0)
        self.assertGreaterEqual(result.ci_lower, 0)
        self.assertLessEqual(result.ci_upper, 1)

    def test_validate_block_bootstrap(self):
        """Test block bootstrap validation."""
        validator = BootstrapValidator(n_bootstrap=30, block_size=20, seed=42)

        def evaluate(X_test, y_test):
            return float(np.mean(y_test))

        result = validator.validate(evaluate, self.X, self.y)

        self.assertIsInstance(result, BootstrapResult)
        self.assertGreater(result.n_bootstrap, 0)

    def test_validate_stratified(self):
        """Test stratified bootstrap validation."""
        validator = BootstrapValidator(n_bootstrap=30, seed=42)

        def evaluate(X_test, y_test):
            return float(np.mean(y_test))

        result = validator.validate(evaluate, self.X, self.y, stratify=True)

        self.assertIsInstance(result, BootstrapResult)

    def test_compare_metrics(self):
        """Test model comparison via bootstrap."""
        validator = BootstrapValidator(n_bootstrap=30, seed=42)

        def model_a(X, y):
            return 0.75

        def model_b(X, y):
            return 0.70

        result = validator.compare_metrics(model_a, model_b, self.X, self.y)

        self.assertIsInstance(result, ModelComparisonResult)
        self.assertIn(result.winner, ["model_a", "model_b"])

    def test_to_dict(self):
        """Test serialization."""
        validator = BootstrapValidator(n_bootstrap=10, seed=42)

        def evaluate(X_test, y_test):
            return float(np.mean(y_test))

        result = validator.validate(evaluate, self.X, self.y)
        d = result.to_dict()

        self.assertIn("metric", d)
        self.assertIn("mean", d)
        self.assertIn("ci_95", d)


class TestStressTester(unittest.TestCase):
    """Tests for StressTester class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 10
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = (np.sum(self.X, axis=1) > 0).astype(float)
        self.returns = np.random.randn(self.n_samples) * 0.02

    def test_classify_regime(self):
        """Test regime classification."""
        tester = StressTester()

        # Normal regime
        regime = tester.classify_regime(self.returns)
        self.assertIsInstance(regime, str)
        self.assertIn(regime, [
            "HIGH_VOL_BULL", "HIGH_VOL_BEAR", "HIGH_VOL_NEUTRAL",
            "LOW_VOL_RANGE", "NORMAL_BULL", "NORMAL_BEAR", "NORMAL_RANGE", "UNKNOWN"
        ])

    def test_test_regimes(self):
        """Test regime stress testing."""
        tester = StressTester()

        predict_fn = MagicMock(return_value=np.random.rand(self.n_samples))

        results = tester.test_regimes(predict_fn, self.X, self.y, returns=self.returns)

        self.assertIsInstance(results, dict)

    def test_test_perturbation_robustness(self):
        """Test perturbation robustness."""
        tester = StressTester()

        predict_fn = MagicMock(return_value=np.random.rand(50))

        results = tester.test_perturbation_robustness(
            predict_fn,
            self.X[:50],
            noise_levels=[0.01, 0.1],
            feature_dropout=[0.1],
        )

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for result in results:
            self.assertIsInstance(result, StressTestResult)

    def test_get_summary(self):
        """Test getting summary."""
        tester = StressTester()
        summary = tester.get_summary()

        self.assertIn("status", summary)

    def test_no_regimes_tested(self):
        """Test summary when no regimes tested."""
        tester = StressTester()
        summary = tester.get_summary()

        self.assertEqual(summary["status"], "no_tests_run")


class TestRobustnessReport(unittest.TestCase):
    """Tests for RobustnessReport dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        report = RobustnessReport(
            bootstrap_results=[
                BootstrapResult(
                    metric_name="accuracy",
                    mean=0.8,
                    std=0.05,
                    ci_lower=0.7,
                    ci_upper=0.9,
                    values=[0.8, 0.75, 0.85],
                    n_bootstrap=3,
                    n_samples=100,
                )
            ],
            stress_results=[
                StressTestResult(
                    regime="noise",
                    metric="noise_0.1",
                    value=0.9,
                    threshold=0.5,
                    passed=True,
                )
            ],
            regime_results={},
            overall_score=0.85,
            recommendation="deploy",
        )

        d = report.to_dict()
        self.assertIn("bootstrap", d)
        self.assertIn("stress", d)
        self.assertIn("overall_score", d)
        self.assertEqual(d["recommendation"], "deploy")


class TestRunRobustnessAnalysis(unittest.TestCase):
    """Tests for run_robustness_analysis function."""

    def test_run_analysis(self):
        """Test running comprehensive robustness analysis."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        X = np.random.randn(n_samples, n_features)
        y = (np.sum(X, axis=1) > 0).astype(float)
        returns = np.random.randn(n_samples) * 0.02

        # Mock model
        model = MagicMock()
        model.predict = MagicMock(return_value=np.random.rand(n_samples))

        report = run_robustness_analysis(
            model,
            X,
            y,
            n_bootstrap=20,
            returns=returns,
        )

        self.assertIsInstance(report, RobustnessReport)
        self.assertIn(report.recommendation, ["deploy", "monitor", "retrain"])


if __name__ == "__main__":
    unittest.main()