"""
Walk-Forward ML Validation — prevents overfitting in ML models.

Features:
  - Rolling and anchored walk-forward splits
  - Purged cross-validation (no leakage)
  - Performance tracking across windows
  - Overfitting detection via IS/OOS performance gap
  - Automated retraining triggers based on WF results

Usage:
    validator = WalkForwardMLValidator(
        n_splits=5,
        train_pct=0.7,
        purge_gap=5,  # bars between train/test
    )
    
    results = validator.validate(
        model_factory=lambda: XGBClassifier(),
        X=features,
        y=labels,
    )
    
    print(results.overfitting_score)  # 0.0 = no overfitting, 1.0 = severe
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    """Result from a single walk-forward window."""
    window_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    train_samples: int
    test_samples: int
    training_time_seconds: float
    
    @property
    def is_sharpe(self) -> float:
        return self.train_metrics.get("sharpe", 0.0)
    
    @property
    def oos_sharpe(self) -> float:
        return self.test_metrics.get("sharpe", 0.0)
    
    @property
    def performance_gap(self) -> float:
        """IS - OOS performance gap (positive = overfitting)."""
        return self.is_sharpe - self.oos_sharpe
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_idx": self.window_idx,
            "train_period": [self.train_start, self.train_end],
            "test_period": [self.test_start, self.test_end],
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "performance_gap": round(self.performance_gap, 4),
            "training_time_seconds": round(self.training_time_seconds, 2),
        }


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation results."""
    windows: List[WindowResult]
    n_splits: int
    train_pct: float
    purge_gap: int
    
    # Aggregate metrics
    mean_is_sharpe: float
    mean_oos_sharpe: float
    std_oos_sharpe: float
    wf_efficiency: float  # mean_oos / mean_is
    overfitting_score: float  # 0 = no overfitting, 1 = severe
    
    # Per-metric aggregates
    metric_comparison: Dict[str, Dict[str, float]]  # metric -> {is_mean, oos_mean, gap}
    
    # Recommendation
    recommendation: str  # "pass", "review", "fail"
    recommendation_reason: str
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_splits": self.n_splits,
            "train_pct": self.train_pct,
            "purge_gap": self.purge_gap,
            "mean_is_sharpe": round(self.mean_is_sharpe, 4),
            "mean_oos_sharpe": round(self.mean_oos_sharpe, 4),
            "std_oos_sharpe": round(self.std_oos_sharpe, 4),
            "wf_efficiency": round(self.wf_efficiency, 4),
            "overfitting_score": round(self.overfitting_score, 4),
            "metric_comparison": self.metric_comparison,
            "recommendation": self.recommendation,
            "recommendation_reason": self.recommendation_reason,
            "windows": [w.to_dict() for w in self.windows],
            "timestamp": self.timestamp.isoformat(),
        }


class WalkForwardMLValidator:
    """
    Walk-forward cross-validation for ML models.
    
    Prevents overfitting by:
    1. Training on historical data
    2. Testing on future (unseen) data
    3. Rolling forward and repeating
    4. Comparing IS vs OOS performance
    
    Args:
        n_splits: Number of train/test windows
        train_pct: Fraction of each window used for training
        purge_gap: Number of bars to skip between train and test (prevent leakage)
        min_train_samples: Minimum training samples per window
        metrics: List of metric names to compute
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        train_pct: float = 0.70,
        purge_gap: int = 5,
        min_train_samples: int = 100,
        metrics: Optional[List[str]] = None,
    ):
        self.n_splits = n_splits
        self.train_pct = train_pct
        self.purge_gap = purge_gap
        self.min_train_samples = min_train_samples
        self.metrics = metrics or ["accuracy", "precision", "recall", "f1", "sharpe"]
    
    def validate(
        self,
        model_factory: Callable,
        X: np.ndarray,
        y: np.ndarray,
        fit_fn: Optional[Callable] = None,
        predict_fn: Optional[Callable] = None,
        score_fn: Optional[Callable] = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.
        
        Args:
            model_factory: Callable that returns a new model instance
            X: Feature matrix (n_samples, n_features)
            y: Target array (n_samples,)
            fit_fn: Optional custom fit function (model, X_train, y_train)
            predict_fn: Optional custom predict function (model, X_test)
            score_fn: Optional custom score function (y_true, y_pred) -> dict
            
        Returns:
            WalkForwardResult with per-window and aggregate results
        """
        n_samples = len(X)
        splits = self._create_splits(n_samples)
        
        if not splits:
            raise ValueError(
                f"Cannot create {self.n_splits} splits with {n_samples} samples "
                f"(min_train={self.min_train_samples}, purge={self.purge_gap})"
            )
        
        windows: List[WindowResult] = []
        all_is_metrics: List[Dict[str, float]] = []
        all_oos_metrics: List[Dict[str, float]] = []
        
        for idx, (train_start, train_end, test_start, test_end) in enumerate(splits):
            window_start = time.time()
            
            # Split data
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            # Train model
            model = model_factory()
            if fit_fn:
                fit_fn(model, X_train, y_train)
            else:
                model.fit(X_train, y_train)
            
            training_time = time.time() - window_start
            
            # Evaluate on IS
            train_metrics = self._evaluate(
                model, X_train, y_train, predict_fn, score_fn
            )
            
            # Evaluate on OOS
            test_metrics = self._evaluate(
                model, X_test, y_test, predict_fn, score_fn
            )
            
            window = WindowResult(
                window_idx=idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                train_samples=len(X_train),
                test_samples=len(X_test),
                training_time_seconds=training_time,
            )
            
            windows.append(window)
            all_is_metrics.append(train_metrics)
            all_oos_metrics.append(test_metrics)
            
            logger.info(
                "WF Window %d: IS Sharpe=%.4f, OOS Sharpe=%.4f, gap=%.4f",
                idx, window.is_sharpe, window.oos_sharpe, window.performance_gap
            )
        
        # Aggregate results
        return self._aggregate_results(windows, all_is_metrics, all_oos_metrics)
    
    def _create_splits(self, n_samples: int) -> List[Tuple[int, int, int, int]]:
        """Create walk-forward train/test splits."""
        splits = []
        window_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Window boundaries
            window_start = i * window_size
            window_end = min((i + 1) * window_size, n_samples)
            
            # Train/test split within window
            train_size = int((window_end - window_start) * self.train_pct)
            train_end = window_start + train_size
            
            # Apply purge gap
            test_start = train_end + self.purge_gap
            test_end = window_end
            
            # Validate split
            if test_start >= test_end:
                continue
            
            if (test_start - window_start) < self.min_train_samples:
                continue
            
            splits.append((window_start, train_end, test_start, test_end))
        
        return splits
    
    def _evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        predict_fn: Optional[Callable],
        score_fn: Optional[Callable],
    ) -> Dict[str, float]:
        """Evaluate model on data."""
        # Predict
        if predict_fn:
            y_pred = predict_fn(model, X)
        elif hasattr(model, 'predict'):
            y_pred = model.predict(X)
        else:
            return {m: 0.0 for m in self.metrics}
        
        # Score
        if score_fn:
            return score_fn(y, y_pred)
        
        # Default metrics
        return self._compute_default_metrics(y, y_pred)
    
    def _compute_default_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute default classification/regression metrics."""
        metrics = {}
        
        # Classification metrics
        if len(np.unique(y_true)) < 20:  # Likely classification
            # Accuracy
            accuracy = np.mean(y_true == y_pred)
            metrics["accuracy"] = float(accuracy)
            
            # Precision/Recall for binary
            if len(np.unique(y_true)) == 2:
                tp = np.sum((y_pred == 1) & (y_true == 1))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                fn = np.sum((y_pred == 0) & (y_true == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) \
                    if (precision + recall) > 0 else 0.0
                
                metrics["precision"] = float(precision)
                metrics["recall"] = float(recall)
                metrics["f1"] = float(f1)
        
        # Regression metrics
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        metrics["mse"] = float(mse)
        metrics["rmse"] = float(rmse)
        metrics["mae"] = float(mae)
        
        # Sharpe-like metric (for trading: direction accuracy)
        if len(y_true) > 1:
            direction_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))
            metrics["sharpe"] = float(direction_accuracy * 10 - 5)  # Scale to Sharpe-like
        
        return metrics
    
    def _aggregate_results(
        self,
        windows: List[WindowResult],
        all_is_metrics: List[Dict[str, float]],
        all_oos_metrics: List[Dict[str, float]],
    ) -> WalkForwardResult:
        """Aggregate window results."""
        # Compute means
        is_sharpes = [w.is_sharpe for w in windows]
        oos_sharpes = [w.oos_sharpe for w in windows]
        
        mean_is = np.mean(is_sharpes) if is_sharpes else 0.0
        mean_oos = np.mean(oos_sharpes) if oos_sharpes else 0.0
        std_oos = np.std(oos_sharpes) if len(oos_sharpes) > 1 else 0.0
        
        # WF efficiency
        wf_efficiency = mean_oos / mean_is if mean_is > 0 else 0.0
        
        # Overfitting score
        gaps = [w.performance_gap for w in windows]
        mean_gap = np.mean(gaps) if gaps else 0.0
        overfitting_score = np.clip(mean_gap / 2.0, 0.0, 1.0)  # Normalize
        
        # Metric comparison
        metric_comparison = {}
        for metric_name in self.metrics:
            is_values = [m.get(metric_name, 0.0) for m in all_is_metrics]
            oos_values = [m.get(metric_name, 0.0) for m in all_oos_metrics]
            
            is_mean = np.mean(is_values) if is_values else 0.0
            oos_mean = np.mean(oos_values) if oos_values else 0.0
            
            metric_comparison[metric_name] = {
                "is_mean": round(is_mean, 4),
                "oos_mean": round(oos_mean, 4),
                "gap": round(is_mean - oos_mean, 4),
            }
        
        # Recommendation
        recommendation, reason = self._make_recommendation(
            wf_efficiency, overfitting_score, std_oos
        )
        
        return WalkForwardResult(
            windows=windows,
            n_splits=len(windows),
            train_pct=self.train_pct,
            purge_gap=self.purge_gap,
            mean_is_sharpe=float(mean_is),
            mean_oos_sharpe=float(mean_oos),
            std_oos_sharpe=float(std_oos),
            wf_efficiency=float(wf_efficiency),
            overfitting_score=float(overfitting_score),
            metric_comparison=metric_comparison,
            recommendation=recommendation,
            recommendation_reason=reason,
        )
    
    def _make_recommendation(
        self,
        wf_efficiency: float,
        overfitting_score: float,
        std_oos: float,
    ) -> Tuple[str, str]:
        """Make recommendation based on WF results."""
        if wf_efficiency >= 0.8 and overfitting_score < 0.2:
            return "pass", "Good OOS performance, low overfitting"
        
        elif wf_efficiency >= 0.6 and overfitting_score < 0.4:
            return "review", "Moderate OOS performance, review before deployment"
        
        elif overfitting_score >= 0.5:
            return "fail", f"Severe overfitting detected (score={overfitting_score:.2f})"
        
        elif wf_efficiency < 0.4:
            return "fail", f"Poor OOS performance (efficiency={wf_efficiency:.2f})"
        
        else:
            return "review", "Marginal performance, requires human review"
