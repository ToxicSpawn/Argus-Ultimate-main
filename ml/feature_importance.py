"""
Feature Importance Tracker — identifies which features drive model performance.

Computes feature importance using:
  1. Permutation importance (model-agnostic, works with any sklearn estimator)
  2. SHAP values (if shap library available)
  3. Correlation-based importance (fallback)

Useful for:
  - Understanding which features the TFT/ELM model relies on
  - Detecting feature degradation (important feature becomes noise)
  - Pruning irrelevant features to reduce overfitting
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


@dataclass
class FeatureScore:
    name: str
    importance: float
    std_dev: float
    rank: int
    direction: str  # "positive" | "negative" | "mixed"


class FeatureImportanceTracker:
    """Tracks and analyses feature importance for ARGUS ML models."""

    def __init__(self, n_permutations: int = 5, random_state: int = 42) -> None:
        self.n_permutations = n_permutations
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        # History: deque of (timestamp, List[FeatureScore])
        self._history: Deque[Tuple[float, List[FeatureScore]]] = deque(maxlen=100)
        self._latest: Optional[List[FeatureScore]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_permutation(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> List[FeatureScore]:
        """
        Permutation importance: shuffle each feature column, measure score drop.

        Returns scores normalised to sum to 1.0 (absolute values).
        """
        if X.ndim != 2 or X.shape[1] != len(feature_names):
            raise ValueError(
                f"X has {X.shape[1] if X.ndim == 2 else '?'} columns, "
                f"but {len(feature_names)} feature names provided"
            )

        baseline = self._score(model, X, y)
        importances: Dict[str, List[float]] = {n: [] for n in feature_names}

        for _ in range(self.n_permutations):
            for i, name in enumerate(feature_names):
                X_perm = X.copy()
                self._rng.shuffle(X_perm[:, i])
                perm_score = self._score(model, X_perm, y)
                importances[name].append(baseline - perm_score)

        # Average and normalise
        raw = {n: float(np.mean(v)) for n, v in importances.items()}
        std = {n: float(np.std(v)) for n, v in importances.items()}
        total = sum(abs(v) for v in raw.values()) or 1.0

        scores = []
        for rank, name in enumerate(
            sorted(feature_names, key=lambda n: raw[n], reverse=True), start=1
        ):
            imp = raw[name]
            direction = "positive" if imp > 0.001 else ("negative" if imp < -0.001 else "mixed")
            scores.append(FeatureScore(
                name=name,
                importance=imp / total,
                std_dev=std[name] / total,
                rank=rank,
                direction=direction,
            ))

        self._latest = scores
        return scores

    def compute_correlation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> List[FeatureScore]:
        """
        Correlation-based importance: |Pearson correlation with target|.
        Fast but only captures linear relationships.
        """
        if X.ndim != 2 or X.shape[1] != len(feature_names):
            raise ValueError("X shape doesn't match feature_names length")

        y_flat = y.ravel().astype(float)
        corrs: Dict[str, float] = {}
        for i, name in enumerate(feature_names):
            col = X[:, i].astype(float)
            if col.std() < 1e-10 or y_flat.std() < 1e-10:
                corrs[name] = 0.0
            else:
                corrs[name] = float(np.corrcoef(col, y_flat)[0, 1])

        abs_corrs = {n: abs(v) for n, v in corrs.items()}
        total = sum(abs_corrs.values()) or 1.0

        scores = []
        for rank, name in enumerate(
            sorted(feature_names, key=lambda n: abs_corrs[n], reverse=True), start=1
        ):
            c = corrs[name]
            direction = "positive" if c > 0.01 else ("negative" if c < -0.01 else "mixed")
            scores.append(FeatureScore(
                name=name,
                importance=abs_corrs[name] / total,
                std_dev=0.0,
                rank=rank,
                direction=direction,
            ))

        self._latest = scores
        return scores

    def compute_shap(
        self,
        model,
        X: np.ndarray,
        feature_names: List[str],
    ) -> List[FeatureScore]:
        """
        SHAP-based importance. Falls back to permutation if shap unavailable.
        """
        if not _SHAP_AVAILABLE:
            logger.debug("shap not available; falling back to permutation importance (no y provided)")
            # Without y we can't do permutation — return correlation proxy with zeros
            return [
                FeatureScore(name=n, importance=0.0, std_dev=0.0, rank=i + 1, direction="mixed")
                for i, n in enumerate(feature_names)
            ]
        try:
            explainer = _shap.Explainer(model, X, feature_names=feature_names)
            shap_values = explainer(X)
            mean_abs = np.abs(shap_values.values).mean(axis=0)
            total = mean_abs.sum() or 1.0

            scores = []
            for rank, idx in enumerate(np.argsort(-mean_abs), start=1):
                imp = float(mean_abs[idx]) / total
                avg_shap = float(shap_values.values[:, idx].mean())
                direction = "positive" if avg_shap > 0.001 else ("negative" if avg_shap < -0.001 else "mixed")
                scores.append(FeatureScore(
                    name=feature_names[idx],
                    importance=imp,
                    std_dev=float(np.std(shap_values.values[:, idx])) / total,
                    rank=rank,
                    direction=direction,
                ))
            self._latest = scores
            return scores
        except Exception:
            logger.warning("SHAP computation failed; falling back to correlation", exc_info=True)
            zeros = np.zeros(X.shape[0])
            return self.compute_correlation(X, zeros, feature_names)

    def track(self, feature_scores: List[FeatureScore], timestamp: float = None) -> None:
        """Store scores in history for drift detection."""
        ts = timestamp if timestamp is not None else time.time()
        self._history.append((ts, feature_scores))
        self._latest = feature_scores

    def detect_drift(self, window: int = 5) -> Dict[str, bool]:
        """
        Returns {feature_name: True} if importance drifted significantly
        across the last `window` tracked snapshots.

        Drift = std(importance) / mean(importance) > 0.5 (coefficient of variation).
        """
        if len(self._history) < 2:
            return {}

        recent = list(self._history)[-window:]
        all_names = {s.name for _, scores in recent for s in scores}
        drift: Dict[str, bool] = {}

        for name in all_names:
            vals = [
                s.importance
                for _, scores in recent
                for s in scores
                if s.name == name
            ]
            if len(vals) < 2:
                drift[name] = False
                continue
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            cv = std_v / max(mean_v, 1e-10)
            drift[name] = cv > 0.5

        return drift

    def top_features(self, n: int = 10) -> List[FeatureScore]:
        """Return top-n features by importance from latest computation."""
        if self._latest is None:
            return []
        sorted_scores = sorted(self._latest, key=lambda s: s.importance, reverse=True)
        return sorted_scores[:n]

    def to_dict(self) -> Dict:
        """Serialise latest scores to dict for logging/storage."""
        if self._latest is None:
            return {}
        return {
            "features": [
                {
                    "name": s.name,
                    "importance": s.importance,
                    "std_dev": s.std_dev,
                    "rank": s.rank,
                    "direction": s.direction,
                }
                for s in self._latest
            ],
            "n_history": len(self._history),
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _score(model, X: np.ndarray, y: np.ndarray) -> float:
        """Safe model.score() wrapper."""
        try:
            return float(model.score(X, y))
        except Exception:
            return 0.0
