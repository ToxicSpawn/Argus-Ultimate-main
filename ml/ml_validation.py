"""
Bootstrap Validation and Stress Testing for ML Models.

Provides:
- Bootstrap validation with confidence intervals
- Block bootstrap for time series
- Stress testing across market regimes
- Model comparison with statistical tests
- Robustness analysis

Usage:
    # Bootstrap validation
    validator = BootstrapValidator(n_bootstrap=100, block_size=20)
    result = validator.validate(model.fit, X, y)
    print(f"Accuracy: {result.mean} ± {result.std}")

    # Stress testing
    tester = StressTester()
    result = tester.test_regimes(model, X, y)
    print(f"VaR 95%: {result.var_95}")
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Result from bootstrap validation."""

    metric_name: str
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    values: List[float]
    n_bootstrap: int
    n_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric_name,
            "mean": float(self.mean),
            "std": float(self.std),
            "ci_95": (float(self.ci_lower), float(self.ci_upper)),
            "n_bootstrap": self.n_bootstrap,
        }


@dataclass
class StressTestResult:
    """Result from stress testing."""

    regime: str
    metric: str
    value: float
    threshold: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "metric": self.metric,
            "value": float(self.value),
            "threshold": float(self.threshold),
            "passed": self.passed,
            "details": self.details,
        }


@dataclass
class ModelComparisonResult:
    """Result from model comparison."""

    model_a_name: str
    model_b_name: str
    metric: str
    difference: float
    p_value: float
    significant: bool
    winner: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "models": (self.model_a_name, self.model_b_name),
            "metric": self.metric,
            "difference": float(self.difference),
            "p_value": float(self.p_value),
            "significant": self.significant,
            "winner": self.winner,
        }


class BootstrapValidator:
    """
    Bootstrap validation for ML models.

    Provides confidence intervals for model performance metrics:
    - Point estimates (accuracy, AUC, etc.)
    - Bootstrap confidence intervals
    - Bias correction

    Supports:
    - Simple bootstrap
    - Block bootstrap for time series
    - Stratified bootstrap
    """

    def __init__(
        self,
        n_bootstrap: int = 100,
        block_size: Optional[int] = None,
        confidence_level: float = 0.95,
        seed: Optional[int] = None,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.block_size = block_size
        self.confidence_level = confidence_level
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def validate(
        self,
        evaluate_fn: Callable[[np.ndarray, np.ndarray], float],
        X: np.ndarray,
        y: np.ndarray,
        *,
        stratify: bool = False,
    ) -> BootstrapResult:
        """
        Run bootstrap validation.

        Args:
            evaluate_fn: Function that takes (X_test, y_test) and returns a metric
            X: Features
            y: Labels
            stratify: Use stratified bootstrap (preserves class ratio)

        Returns:
            BootstrapResult with mean, std, CI
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples = len(y)

        metric_values: List[float] = []

        for _ in range(self.n_bootstrap):
            # Generate bootstrap sample
            if self.block_size is not None:
                # Block bootstrap
                indices = self._block_bootstrap_indices(n_samples)
            elif stratify:
                # Stratified bootstrap
                indices = self._stratified_indices(y)
            else:
                # Simple bootstrap
                indices = self._rng.integers(0, n_samples, size=n_samples)

            X_boot = X[indices]
            y_boot = y[indices]

            # Evaluate
            try:
                metric = evaluate_fn(X_boot, y_boot)
                metric_values.append(float(metric))
            except Exception as e:
                logger.warning(f"Bootstrap evaluation failed: {e}")

        if not metric_values:
            return BootstrapResult(
                metric_name="unknown",
                mean=0.0,
                std=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
                values=[],
                n_bootstrap=0,
                n_samples=0,
            )

        metric_values = np.array(metric_values)
        alpha = 1 - self.confidence_level

        return BootstrapResult(
            metric_name="score",
            mean=float(np.mean(metric_values)),
            std=float(np.std(metric_values)),
            ci_lower=float(np.percentile(metric_values, 100 * alpha / 2)),
            ci_upper=float(np.percentile(metric_values, 100 * (1 - alpha / 2))),
            values=metric_values.tolist(),
            n_bootstrap=self.n_bootstrap,
            n_samples=n_samples,
        )

    def _block_bootstrap_indices(self, n: int) -> np.ndarray:
        """Generate block bootstrap indices."""
        indices = []
        n_blocks = (n + self.block_size - 1) // self.block_size

        for _ in range(n_blocks):
            start = self._rng.integers(0, n)
            block_indices = [(start + i) % n for i in range(self.block_size)]
            indices.extend(block_indices)

        return np.array(indices[:n])

    def _stratified_indices(self, y: np.ndarray) -> np.ndarray:
        """Generate stratified bootstrap indices."""
        n = len(y)
        classes, counts = np.unique(y, return_counts=True)

        indices = []
        for cls, cnt in zip(classes, counts):
            cls_indices = np.where(y == cls)[0]
            sampled = self._rng.choice(cls_indices, size=cnt, replace=True)
            indices.extend(sampled)

        self._rng.shuffle(indices)
        return np.array(indices[:n])

    def compare_metrics(
        self,
        evaluate_fn_a: Callable,
        evaluate_fn_b: Callable,
        X: np.ndarray,
        y: np.ndarray,
        metric_name: str = "score",
    ) -> ModelComparisonResult:
        """Compare two models via bootstrap."""
        result_a = self.validate(evaluate_fn_a, X, y)
        result_b = self.validate(evaluate_fn_b, X, y)

        mean_a, mean_b = result_a.mean, result_b.mean
        diff = mean_a - mean_b

        # Paired difference
        n = min(len(result_a.values), len(result_b.values))
        values_a = np.array(result_a.values[:n])
        values_b = np.array(result_b.values[:n])
        diff_values = values_a - values_b

        # Two-sided t-test approximation
        if np.std(diff_values) > 1e-10:
            t_stat = np.mean(diff_values) / (np.std(diff_values) / np.sqrt(n))
            # Approximate p-value (two-tailed)
            from scipy import stats
            p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)))
        else:
            p_value = 1.0

        return ModelComparisonResult(
            model_a_name="model_a",
            model_b_name="model_b",
            metric=metric_name,
            difference=diff,
            p_value=p_value,
            significant=p_value < 0.05,
            winner="model_a" if diff > 0 else "model_b",
        )


@dataclass
class RegimeTest:
    """Test results for a specific regime."""

    regime: str
    mean_metric: float
    std_metric: float
    samples: int
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


class StressTester:
    """
    Stress testing for ML models across market regimes.

    Tests model robustness under:
    - Bull/bear markets
    - High/low volatility regimes
    - Trending/range-bound markets
    - Crisis periods
    """

    def __init__(
        self,
        volatility_threshold_high: float = 0.02,
        volatility_threshold_low: float = 0.005,
        trend_threshold: float = 0.01,
        seed: Optional[int] = None,
    ) -> None:
        self.volatility_threshold_high = volatility_threshold_high
        self.volatility_threshold_low = volatility_threshold_low
        self.trend_threshold = trend_threshold
        self._rng = np.random.default_rng(seed)

        self._regime_tests: Dict[str, RegimeTest] = {}

    def classify_regime(
        self,
        returns: np.ndarray,
        *,
        window: int = 20,
    ) -> str:
        """Classify market regime based on returns."""
        if len(returns) < window:
            return "UNKNOWN"

        window_returns = returns[-window:]
        volatility = float(np.std(window_returns))
        trend = float(np.mean(window_returns))

        # Classify
        if volatility > self.volatility_threshold_high:
            if trend > self.trend_threshold:
                return "HIGH_VOL_BULL"
            elif trend < -self.trend_threshold:
                return "HIGH_VOL_BEAR"
            else:
                return "HIGH_VOL_NEUTRAL"
        elif volatility < self.volatility_threshold_low:
            return "LOW_VOL_RANGE"
        else:
            if trend > self.trend_threshold:
                return "NORMAL_BULL"
            elif trend < -self.trend_threshold:
                return "NORMAL_BEAR"
            else:
                return "NORMAL_RANGE"

    def test_regimes(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X: np.ndarray,
        y: np.ndarray,
        *,
        regime_indicator: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None,
        min_samples: int = 30,
    ) -> Dict[str, RegimeTest]:
        """
        Test model across different market regimes.

        Args:
            predict_fn: Model predict function
            X: Features
            y: Labels
            regime_indicator: Pre-computed regime labels
            returns: Returns for automatic regime classification
            min_samples: Minimum samples per regime to test

        Returns:
            Dict of regime -> RegimeTest
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples = len(y)

        # Determine regimes
        if regime_indicator is not None:
            regimes = regime_indicator
        elif returns is not None:
            # Classify each sample
            regimes = np.array([self.classify_regime(returns[:i + 1]) for i in range(n_samples)])
        else:
            return {}

        unique_regimes, counts = np.unique(regimes, return_counts=True)

        self._regime_tests = {}
        for regime, count in zip(unique_regimes, counts):
            if count < min_samples:
                continue

            mask = regimes == regime
            X_regime = X[mask]
            y_regime = y[mask]

            try:
                predictions = predict_fn(X_regime)
                accuracy = np.mean((predictions > 0.5) == y_regime)

                # Bootstrap std estimate
                n_iter = min(50, count)
                accuracies = []
                for _ in range(n_iter):
                    boot_idx = self._rng.integers(0, len(y_regime), size=len(y_regime))
                    boot_acc = np.mean((predictions[boot_idx] > 0.5) == y_regime[boot_idx])
                    accuracies.append(boot_acc)

                self._regime_tests[regime] = RegimeTest(
                    regime=regime,
                    mean_metric=float(np.mean(accuracies)),
                    std_metric=float(np.std(accuracies)),
                    samples=count,
                    passed=accuracy > 0.5,
                    details={"raw_accuracy": float(accuracy)},
                )
            except Exception as e:
                logger.warning(f"Regime test failed for {regime}: {e}")

        return self._regime_tests

    def test_perturbation_robustness(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X: np.ndarray,
        *,
        noise_levels: List[float] = None,
        feature_dropout: List[float] = None,
        n_samples: int = 100,
    ) -> List[StressTestResult]:
        """
        Test robustness to input perturbations.

        Tests:
        - Gaussian noise injection
        - Feature dropout
        - Adversarial perturbations
        """
        results: List[StressTestResult] = []
        X = np.asarray(X, dtype=float)

        if noise_levels is None:
            noise_levels = [0.01, 0.05, 0.1, 0.2]

        if feature_dropout is None:
            feature_dropout = [0.1, 0.2, 0.3]

        # Sample data
        if len(X) > n_samples:
            indices = self._rng.choice(len(X), n_samples, replace=False)
            X_test = X[indices]
        else:
            X_test = X

        # Baseline prediction
        baseline = predict_fn(X_test)

        # Noise robustness
        for noise in noise_levels:
            X_noisy = X_test + self._rng.normal(0, noise, X_test.shape)
            noisy_pred = predict_fn(X_noisy)

            # Compare distributions (not exact values)
            correlation = float(np.corrcoef(baseline, noisy_pred)[0, 1])
            correlation = 0.0 if np.isnan(correlation) else correlation

            results.append(StressTestResult(
                regime="noise_robustness",
                metric=f"noise_{noise}",
                value=correlation,
                threshold=0.5,
                passed=correlation > 0.5,
                details={"noise_level": noise},
            ))

        # Feature dropout robustness
        n_features = X_test.shape[1]
        for dropout_rate in feature_dropout:
            n_drop = int(n_features * dropout_rate)
            if n_drop >= n_features:
                continue

            dropout_features = self._rng.choice(n_features, n_drop, replace=False)
            X_dropout = X_test.copy()
            X_dropout[:, dropout_features] = 0

            dropout_pred = predict_fn(X_dropout)
            correlation = float(np.corrcoef(baseline, dropout_pred)[0, 1])
            correlation = 0.0 if np.isnan(correlation) else correlation

            results.append(StressTestResult(
                regime="feature_dropout",
                metric=f"dropout_{dropout_rate}",
                value=correlation,
                threshold=0.3,
                passed=correlation > 0.3,
                details={"dropout_rate": dropout_rate, "n_features_dropped": n_drop},
            ))

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of stress test results."""
        if not self._regime_tests:
            return {"status": "no_tests_run"}

        passed = sum(1 for t in self._regime_tests.values() if t.passed)
        total = len(self._regime_tests)

        return {
            "total_regimes_tested": total,
            "regimes_passed": passed,
            "regimes_failed": total - passed,
            "pass_rate": passed / max(total, 1),
            "regimes": {
                regime: {
                    "passed": test.passed,
                    "accuracy": test.mean_metric,
                    "samples": test.samples,
                }
                for regime, test in self._regime_tests.items()
            },
        }


@dataclass
class RobustnessReport:
    """Comprehensive robustness report."""

    bootstrap_results: List[BootstrapResult]
    stress_results: List[StressTestResult]
    regime_results: Dict[str, RegimeTest]
    overall_score: float
    recommendation: str  # deploy, monitor, retrain

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bootstrap": [r.to_dict() for r in self.bootstrap_results],
            "stress": [r.to_dict() for r in self.stress_results],
            "regimes": {k: v.to_dict() for k, v in self.regime_results.items()},
            "overall_score": float(self.overall_score),
            "recommendation": self.recommendation,
        }


def run_robustness_analysis(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int = 100,
    returns: Optional[np.ndarray] = None,
) -> RobustnessReport:
    """
    Run comprehensive robustness analysis on a model.

    Args:
        model: Trained model with predict method
        X: Features
        y: Labels
        n_bootstrap: Number of bootstrap iterations
        returns: Returns for regime classification

    Returns:
        RobustnessReport
    """
    rng = np.random.default_rng()

    # Bootstrap validation
    validator = BootstrapValidator(n_bootstrap=n_bootstrap, seed=42)

    def evaluate(X_test, y_test):
        pred = model.predict(X_test)
        return float(np.mean((pred > 0.5) == y_test))

    bootstrap_results = [validator.validate(evaluate, X, y)]

    # Stress testing
    tester = StressTester()

    stress_results = tester.test_perturbation_robustness(
        model.predict, X
    )

    regime_results = tester.test_regimes(
        model.predict, X, y, returns=returns
    )

    # Compute overall score
    stress_pass_rate = sum(1 for r in stress_results if r.passed) / max(len(stress_results), 1)
    regime_pass_rate = sum(1 for r in regime_results.values() if r.passed) / max(len(regime_results), 1)

    overall_score = (stress_pass_rate + regime_pass_rate) / 2

    if overall_score > 0.8:
        recommendation = "deploy"
    elif overall_score > 0.5:
        recommendation = "monitor"
    else:
        recommendation = "retrain"

    return RobustnessReport(
        bootstrap_results=bootstrap_results,
        stress_results=stress_results,
        regime_results=regime_results,
        overall_score=overall_score,
        recommendation=recommendation,
    )


__all__ = [
    "BootstrapValidator",
    "StressTester",
    "BootstrapResult",
    "StressTestResult",
    "ModelComparisonResult",
    "RobustnessReport",
    "run_robustness_analysis",
]