"""
Feature Selection and Importance Tracking.

Provides feature importance analysis, selection, and monitoring:
- Importance-based selection (model-based, statistical)
- Correlation-based pruning (remove redundant features)
- Recursive feature elimination
- Feature drift detection
- Cross-validation importance stability

Usage:
    selector = FeatureSelector(method="importance")
    X_selected, selected_idx = selector.select(X, y)
    importance = selector.get_importance_scores()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionResult:
    """Result of feature selection."""

    selected_indices: np.ndarray
    selected_names: List[str]
    removed_indices: np.ndarray
    removed_names: List[str]
    importance_scores: Dict[str, float]
    correlation_matrix: Optional[np.ndarray] = None
    method: str = "importance"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_selected": len(self.selected_indices),
            "n_removed": len(self.removed_indices),
            "importance_scores": self.importance_scores,
            "method": self.method,
        }


class FeatureSelector:
    """
    Feature selection with multiple methods.

    Methods:
    - importance: Use model-based feature importance
    - correlation: Remove highly correlated features
    - recursive: Recursive feature elimination
    - statistical: Use statistical tests (ANOVA, chi-square)
    """

    def __init__(
        self,
        *,
        method: str = "importance",
        min_features: int = 10,
        max_features: int = 200,
        correlation_threshold: float = 0.95,
        importance_threshold: float = 0.01,
        seed: Optional[int] = None,
    ) -> None:
        self.method = method
        self.min_features = max(1, int(min_features))
        self.max_features = max(max_features, min_features)
        self.correlation_threshold = correlation_threshold
        self.importance_threshold = importance_threshold
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self._importance_scores: Dict[int, float] = {}
        self._selected_indices: Optional[np.ndarray] = None
        self._model = None

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select features from X.

        Returns:
            X_selected: Feature matrix with selected features
            selected_indices: Indices of selected features
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        n_samples, n_features = X.shape

        # If too few features, return all
        if n_features <= self.min_features:
            self._selected_indices = np.arange(n_features)
            return X, self._selected_indices

        if self.method == "importance":
            selected = self._select_by_importance(X, y)
        elif self.method == "correlation":
            selected = self._select_by_correlation(X)
        elif self.method == "recursive":
            selected = self._select_recursive(X, y)
        elif self.method == "statistical":
            selected = self._select_by_statistical(X, y)
        else:
            # Default: keep all
            selected = np.arange(n_features)

        # Enforce min/max bounds
        if len(selected) < self.min_features:
            # Keep top min_features by variance
            variances = np.var(X[:, selected], axis=0)
            top_idx = np.argsort(variances)[-self.min_features:]
            selected = selected[top_idx]

        if len(selected) > self.max_features:
            selected = selected[:self.max_features]

        self._selected_indices = selected
        return X[:, selected], selected

    def _select_by_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Select features based on model importance."""
        try:
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.seed,
            )
            model.fit(X, y)
            importances = model.feature_importances_

            # Store importance scores
            self._importance_scores = {i: float(imp) for i, imp in enumerate(importances)}

            # Select features with importance above threshold
            selected = np.where(importances > self.importance_threshold)[0]

            # If too few, select top features by importance
            if len(selected) < self.min_features:
                n_keep = min(self.max_features, len(importances))
                selected = np.argsort(importances)[-n_keep:]

            return selected

        except Exception as e:
            logger.warning(f"Importance selection failed: {e}")
            # Fallback: select by variance
            variances = np.var(X, axis=0)
            return np.argsort(variances)[-self.min_features:]

    def _select_by_correlation(self, X: np.ndarray) -> np.ndarray:
        """Select features with low correlation."""
        # Compute correlation matrix
        corr = np.corrcoef(X.T)
        corr = np.nan_to_num(corr, nan=0.0)

        # Remove features with correlation > threshold
        n_features = X.shape[1]
        to_keep = []

        for i in range(n_features):
            # Check correlation with already-kept features
            keep = True
            for j in to_keep:
                if abs(corr[i, j]) > self.correlation_threshold:
                    keep = False
                    break
            if keep:
                to_keep.append(i)

        selected = np.array(to_keep)

        # If too few, add random features
        if len(selected) < self.min_features:
            remaining = [i for i in range(n_features) if i not in selected]
            needed = self.min_features - len(selected)
            selected = np.concatenate([selected, self._rng.choice(remaining, needed, replace=False)])

        return selected

    def _select_recursive(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Recursive feature elimination."""
        try:
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=self.seed)
            rfe = RFE(model, n_features_to_select=self.max_features)
            rfe.fit(X, y)

            selected = np.where(rfe.support_)[0]

            # Store rankings
            rankings = rfe.ranking_
            self._importance_scores = {i: float(1 / (r + 1)) for i, r in enumerate(rankings)}

            return selected

        except Exception as e:
            logger.warning(f"Recursive selection failed: {e}")
            return self._select_by_importance(X, y)

    def _select_by_statistical(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Select features using statistical tests."""
        try:
            from sklearn.feature_selection import SelectKBest, f_classif

            # Select k best features using ANOVA F-test
            n_select = min(self.max_features, X.shape[1])
            selector = SelectKBest(f_classif, k=n_select)
            selector.fit(X, y)

            selected = selector.get_support(indices=True)
            scores = selector.scores_
            self._importance_scores = {i: float(scores[i]) for i in range(len(scores))}

            return selected

        except Exception as e:
            logger.warning(f"Statistical selection failed: {e}")
            return self._select_by_importance(X, y)

    def get_importance_scores(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return {f"feature_{k}": v for k, v in self._importance_scores.items()}

    def get_selection_result(self) -> Optional[FeatureSelectionResult]:
        """Get detailed selection result."""
        if self._selected_indices is None:
            return None

        n_features = len(self._importance_scores) + len(self._selected_indices)
        all_indices = np.arange(n_features)
        removed_indices = np.setdiff1d(all_indices, self._selected_indices)

        return FeatureSelectionResult(
            selected_indices=self._selected_indices,
            selected_names=[f"feature_{i}" for i in self._selected_indices],
            removed_indices=removed_indices,
            removed_names=[f"feature_{i}" for i in removed_indices],
            importance_scores=self.get_importance_scores(),
            method=self.method,
        )


@dataclass
class FeatureDriftResult:
    """Result of feature drift detection."""

    drifting_features: List[str]
    drift_scores: Dict[str, float]
    stable_features: List[str]
    overall_drift: float
    recommendation: str  # retrain, monitor, no_action


class FeatureDriftTracker:
    """
    Track feature drift over time and detect distribution shifts.

    Usage:
        tracker = FeatureDriftTracker(reference_window=100)
        tracker.set_reference(X_reference)
        drift_result = tracker.check_drift(X_current)
    """

    def __init__(
        self,
        *,
        reference_window: int = 100,
        drift_threshold: float = 0.1,
        n_check_features: int = 50,
    ) -> None:
        self.reference_window = reference_window
        self.drift_threshold = drift_threshold
        self.n_check_features = n_check_features
        self._reference_stats: Dict[int, Tuple[float, float]] = {}  # mean, std per feature

    def set_reference(self, X: np.ndarray) -> None:
        """Set reference distribution."""
        X = np.asarray(X, dtype=float)
        for i in range(min(X.shape[1], self.n_check_features)):
            self._reference_stats[i] = (float(np.mean(X[:, i])), float(np.std(X[:, i])))

    def check_drift(self, X: np.ndarray) -> FeatureDriftResult:
        """Check for feature drift vs reference."""
        X = np.asarray(X, dtype=float)

        drifting = []
        drift_scores = {}
        stable = []

        for i in range(min(X.shape[1], self.n_check_features)):
            if i not in self._reference_stats:
                continue

            ref_mean, ref_std = self._reference_stats[i]
            curr_mean = float(np.mean(X[:, i]))
            curr_std = float(np.std(X[:, i]))

            # Compute drift score (normalized difference)
            std_avg = (ref_std + curr_std) / 2
            if std_avg > 1e-10:
                drift_score = abs(curr_mean - ref_mean) / std_avg
            else:
                drift_score = abs(curr_mean - ref_mean)

            drift_scores[f"feature_{i}"] = drift_score

            if drift_score > self.drift_threshold:
                drifting.append(f"feature_{i}")
            else:
                stable.append(f"feature_{i}")

        n_drift = len(drifting)
        n_total = len(drift_scores)
        overall_drift = n_drift / max(n_total, 1)

        if overall_drift > 0.3:
            recommendation = "retrain"
        elif overall_drift > 0.1:
            recommendation = "monitor"
        else:
            recommendation = "no_action"

        return FeatureDriftResult(
            drifting_features=drifting,
            drift_scores=drift_scores,
            stable_features=stable,
            overall_drift=overall_drift,
            recommendation=recommendation,
        )


__all__ = [
    "FeatureSelector",
    "FeatureSelectionResult",
    "FeatureDriftTracker",
    "FeatureDriftResult",
]
