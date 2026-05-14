"""
Market Anomaly Detector — pure-numpy Isolation Forest implementation.

Implements the Isolation Forest algorithm from scratch (Liu et al., 2008)
with extensions for real-time crypto market anomaly detection.

Design:
  - Ensemble of 100 isolation trees (configurable)
  - Random feature selection + random split values
  - Path length scoring (shorter path = more anomalous)
  - Online update capability (add samples without full retrain)
  - Anomaly type classification (flash crash, spoofing, whale manipulation, etc.)
  - Thread-safe for concurrent reads

Dependencies: numpy only.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Average path length of unsuccessful search in BST (used for normalisation)
def _avg_path_length(n: int) -> float:
    """Expected average path length c(n) of unsuccessful search in BST."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    # Harmonic number approximation: H(n-1) ≈ ln(n-1) + 0.5772
    return 2.0 * (math.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


# Anomaly type definitions
ANOMALY_TYPES = {
    "FLASH_CRASH": "Sudden extreme price drop",
    "SPOOFING": "Unusual orderbook imbalance pattern",
    "WHALE_MANIPULATION": "Large single-entity activity",
    "LIQUIDITY_VACUUM": "Sudden liquidity withdrawal",
    "VOLUME_SPIKE": "Abnormal volume increase",
    "SPREAD_BLOWOUT": "Extreme spread widening",
    "FUNDING_ANOMALY": "Unusual funding rate deviation",
    "CORRELATION_BREAK": "Cross-asset correlation breakdown",
    "UNKNOWN": "Unclassified anomaly",
}

# Feature names for real-time detection
REALTIME_FEATURES = [
    "spread_bps",
    "volume_ratio",
    "price_velocity",
    "orderbook_imbalance",
    "funding_rate_deviation",
]


# ---------------------------------------------------------------------------
# Isolation Tree Node
# ---------------------------------------------------------------------------

@dataclass
class _ITreeNode:
    """A node in an isolation tree."""
    left: Optional[_ITreeNode] = None
    right: Optional[_ITreeNode] = None
    split_feature: int = 0
    split_value: float = 0.0
    size: int = 0          # number of samples at this node (for external nodes)
    is_external: bool = False


# ---------------------------------------------------------------------------
# Single Isolation Tree
# ---------------------------------------------------------------------------

class _IsolationTree:
    """A single isolation tree built from a subsample of data."""

    def __init__(self, height_limit: int, rng: np.random.RandomState):
        self._height_limit = height_limit
        self._rng = rng
        self.root: Optional[_ITreeNode] = None

    def fit(self, X: np.ndarray) -> None:
        """Build the tree from data matrix X (n_samples, n_features)."""
        self.root = self._build(X, depth=0)

    def _build(self, X: np.ndarray, depth: int) -> _ITreeNode:
        n_samples, n_features = X.shape

        # External node conditions
        if depth >= self._height_limit or n_samples <= 1:
            node = _ITreeNode(is_external=True, size=n_samples)
            return node

        # Check if all values identical (no split possible)
        if np.all(X == X[0]):
            node = _ITreeNode(is_external=True, size=n_samples)
            return node

        # Random feature selection
        feat_idx = self._rng.randint(0, n_features)
        col = X[:, feat_idx]
        col_min, col_max = col.min(), col.max()

        if col_min == col_max:
            # No variance in this feature — try another
            # Find a feature with variance
            found = False
            for _ in range(min(n_features, 5)):
                feat_idx = self._rng.randint(0, n_features)
                col = X[:, feat_idx]
                col_min, col_max = col.min(), col.max()
                if col_min < col_max:
                    found = True
                    break
            if not found:
                return _ITreeNode(is_external=True, size=n_samples)

        # Random split value between min and max
        split_val = self._rng.uniform(col_min, col_max)

        # Partition
        left_mask = X[:, feat_idx] < split_val
        right_mask = ~left_mask

        X_left = X[left_mask]
        X_right = X[right_mask]

        node = _ITreeNode(
            split_feature=feat_idx,
            split_value=split_val,
        )
        node.left = self._build(X_left, depth + 1)
        node.right = self._build(X_right, depth + 1)

        return node

    def path_length(self, x: np.ndarray) -> float:
        """Compute path length for a single sample x (1D array)."""
        return self._traverse(x, self.root, 0)

    def _traverse(self, x: np.ndarray, node: Optional[_ITreeNode], depth: int) -> float:
        if node is None or node.is_external:
            size = node.size if node is not None else 1
            return depth + _avg_path_length(size)

        if x[node.split_feature] < node.split_value:
            return self._traverse(x, node.left, depth + 1)
        else:
            return self._traverse(x, node.right, depth + 1)


# ---------------------------------------------------------------------------
# Anomaly event record
# ---------------------------------------------------------------------------

@dataclass
class AnomalyEvent:
    """Record of a detected anomaly."""
    timestamp: float
    score: float          # anomaly score (0 = normal, 1 = highly anomalous)
    is_anomaly: bool
    anomaly_type: str
    severity: float       # 0–1
    features: Dict[str, float] = field(default_factory=dict)
    description: str = ""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MarketAnomalyDetector:
    """
    Pure-numpy Isolation Forest for market anomaly detection.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies (default 0.05 = 5%).
    n_trees : int
        Number of isolation trees (default 100).
    subsample_size : int
        Subsample size per tree (default 256).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_trees: int = 100,
        subsample_size: int = 256,
        random_state: int = 42,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or {}
        self._contamination = float(cfg.get("contamination", contamination))
        self._n_trees = int(cfg.get("n_trees", n_trees))
        self._subsample_size = int(cfg.get("subsample_size", subsample_size))
        self._rng = np.random.RandomState(random_state)

        self._trees: List[_IsolationTree] = []
        self._fitted = False
        self._n_samples = 0
        self._n_features = 0
        self._threshold: float = 0.5  # anomaly score threshold (set after fit)

        # Online buffer for incremental updates
        self._buffer: List[np.ndarray] = []
        self._max_buffer_size = int(cfg.get("max_buffer_size", 1000))

        # Feature statistics for normalisation
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

        # Anomaly history
        self._history: List[AnomalyEvent] = []
        self._max_history = int(cfg.get("max_history", 500))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, features: np.ndarray) -> None:
        """
        Train the isolation forest on historical market features.

        Parameters
        ----------
        features : np.ndarray
            Shape (n_samples, n_features). Each row is a feature vector.
        """
        X = np.asarray(features, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._n_samples, self._n_features = X.shape

        # Compute feature statistics for normalisation
        self._feature_means = np.nanmean(X, axis=0)
        self._feature_stds = np.nanstd(X, axis=0)
        self._feature_stds[self._feature_stds < 1e-10] = 1.0

        # Normalise
        X_norm = (X - self._feature_means) / self._feature_stds

        # Height limit: ceil(log2(subsample_size))
        psi = min(self._subsample_size, self._n_samples)
        height_limit = int(math.ceil(math.log2(max(psi, 2))))

        # Build trees
        self._trees = []
        for _ in range(self._n_trees):
            tree = _IsolationTree(height_limit=height_limit, rng=self._rng)
            # Subsample
            if self._n_samples > psi:
                indices = self._rng.choice(self._n_samples, size=psi, replace=False)
                X_sub = X_norm[indices]
            else:
                X_sub = X_norm
            tree.fit(X_sub)
            self._trees.append(tree)

        # Compute anomaly scores on training data to set threshold
        scores = self._score_samples(X_norm)
        # Threshold at contamination percentile
        self._threshold = float(np.percentile(scores, 100 * (1 - self._contamination)))

        self._fitted = True
        logger.info(
            "MarketAnomalyDetector: fitted %d trees on %d samples (%d features), "
            "threshold=%.4f",
            self._n_trees, self._n_samples, self._n_features, self._threshold,
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Return anomaly labels: -1 = anomaly, 1 = normal.

        Parameters
        ----------
        features : np.ndarray
            Shape (n_samples, n_features) or (n_features,) for single sample.
        """
        scores = self.score_samples(features)
        labels = np.where(scores >= self._threshold, -1, 1)
        return labels

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores for each sample (higher = more anomalous).

        Scores are in [0, 1] range: 0 = definitely normal, 1 = definitely anomalous.

        Parameters
        ----------
        features : np.ndarray
            Shape (n_samples, n_features) or (n_features,) for single sample.
        """
        if not self._fitted:
            raise RuntimeError("MarketAnomalyDetector not fitted. Call fit() first.")

        X = np.asarray(features, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Normalise using training statistics
        X_norm = (X - self._feature_means) / self._feature_stds
        return self._score_samples(X_norm)

    def detect_realtime(self, current_features: dict) -> dict:
        """
        Real-time anomaly detection for current market state.

        Parameters
        ----------
        current_features : dict
            Keys from REALTIME_FEATURES:
                spread_bps, volume_ratio, price_velocity,
                orderbook_imbalance, funding_rate_deviation

        Returns
        -------
        dict with keys:
            is_anomaly: bool
            score: float (0–1, higher = more anomalous)
            type: str (anomaly type classification)
            severity: float (0–1)
        """
        # Build feature vector
        feat_vec = np.array([
            float(current_features.get("spread_bps", 0.0)),
            float(current_features.get("volume_ratio", 1.0)),
            float(current_features.get("price_velocity", 0.0)),
            float(current_features.get("orderbook_imbalance", 0.0)),
            float(current_features.get("funding_rate_deviation", 0.0)),
        ], dtype=np.float64)

        # If not fitted, use heuristic detection
        if not self._fitted:
            return self._heuristic_detect(current_features, feat_vec)

        # Pad or truncate to match training dimensionality
        if len(feat_vec) < self._n_features:
            feat_vec = np.pad(feat_vec, (0, self._n_features - len(feat_vec)))
        elif len(feat_vec) > self._n_features:
            feat_vec = feat_vec[:self._n_features]

        # Score
        score = float(self.score_samples(feat_vec.reshape(1, -1))[0])
        is_anomaly = score >= self._threshold

        # Classify anomaly type
        anomaly_type = self._classify_anomaly(current_features, score)

        # Severity: map score to 0–1 with steeper curve near threshold
        severity = min(1.0, max(0.0, (score - 0.3) / 0.5))

        # Add to buffer for online updates
        self._buffer.append(feat_vec)
        if len(self._buffer) > self._max_buffer_size:
            self._buffer = self._buffer[-self._max_buffer_size:]

        # Record event
        event = AnomalyEvent(
            timestamp=time.time(),
            score=score,
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            severity=severity,
            features=current_features,
            description=ANOMALY_TYPES.get(anomaly_type, ""),
        )
        if is_anomaly:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            logger.warning(
                "Anomaly detected: type=%s score=%.3f severity=%.2f",
                anomaly_type, score, severity,
            )

        return {
            "is_anomaly": is_anomaly,
            "score": round(score, 4),
            "type": anomaly_type,
            "severity": round(severity, 4),
        }

    def update_online(self, new_samples: np.ndarray) -> None:
        """
        Add new samples to the model without full retrain.

        Rebuilds a fraction of the trees using the new data combined
        with the existing buffer.

        Parameters
        ----------
        new_samples : np.ndarray
            New observations to incorporate.
        """
        X = np.asarray(new_samples, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        for row in X:
            self._buffer.append(row)

        if len(self._buffer) > self._max_buffer_size:
            self._buffer = self._buffer[-self._max_buffer_size:]

        # Rebuild 10% of trees with updated buffer
        if self._fitted and len(self._buffer) >= self._subsample_size:
            buf = np.array(self._buffer)
            # Normalise
            buf_norm = (buf - self._feature_means) / self._feature_stds
            n_rebuild = max(1, self._n_trees // 10)
            psi = min(self._subsample_size, len(buf_norm))
            height_limit = int(math.ceil(math.log2(max(psi, 2))))

            for i in range(n_rebuild):
                idx = self._rng.randint(0, self._n_trees)
                tree = _IsolationTree(height_limit=height_limit, rng=self._rng)
                indices = self._rng.choice(len(buf_norm), size=min(psi, len(buf_norm)), replace=False)
                tree.fit(buf_norm[indices])
                self._trees[idx] = tree

            logger.debug(
                "MarketAnomalyDetector: online update rebuilt %d/%d trees",
                n_rebuild, self._n_trees,
            )

    def get_anomaly_history(self, limit: int = 50) -> list:
        """Return recent anomaly events."""
        return [
            {
                "timestamp": e.timestamp,
                "score": e.score,
                "type": e.anomaly_type,
                "severity": e.severity,
                "features": e.features,
                "description": e.description,
            }
            for e in self._history[-limit:]
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Return detector statistics."""
        return {
            "fitted": self._fitted,
            "n_trees": self._n_trees,
            "n_features": self._n_features,
            "n_training_samples": self._n_samples,
            "threshold": round(self._threshold, 4),
            "buffer_size": len(self._buffer),
            "anomaly_count": len(self._history),
            "contamination": self._contamination,
        }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _score_samples(self, X_norm: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores on already-normalised data.

        Score formula: s(x, n) = 2^(-E[h(x)] / c(n))
        where h(x) is path length and c(n) is average path length.
        """
        n = X_norm.shape[0]
        c_n = _avg_path_length(self._n_samples)
        if c_n < 1e-10:
            c_n = 1.0

        scores = np.zeros(n)
        for tree in self._trees:
            for i in range(n):
                scores[i] += tree.path_length(X_norm[i])

        # Average path length across trees
        avg_path = scores / len(self._trees)

        # Anomaly score: 2^(-E[h(x)] / c(n))
        anomaly_scores = np.power(2.0, -avg_path / c_n)

        return anomaly_scores

    def _classify_anomaly(self, features: dict, score: float) -> str:
        """Classify the type of anomaly based on feature values."""
        spread = float(features.get("spread_bps", 0.0))
        volume = float(features.get("volume_ratio", 1.0))
        velocity = float(features.get("price_velocity", 0.0))
        imbalance = float(features.get("orderbook_imbalance", 0.0))
        funding = float(features.get("funding_rate_deviation", 0.0))

        # Classification rules (ordered by priority)
        if abs(velocity) > 5.0 and volume > 3.0:
            return "FLASH_CRASH" if velocity < 0 else "VOLUME_SPIKE"

        if spread > 50.0:
            return "SPREAD_BLOWOUT"

        if abs(imbalance) > 0.8 and volume < 0.5:
            return "SPOOFING"

        if volume > 5.0:
            return "WHALE_MANIPULATION"

        if spread > 20.0 and volume < 0.3:
            return "LIQUIDITY_VACUUM"

        if volume > 3.0:
            return "VOLUME_SPIKE"

        if abs(funding) > 3.0:
            return "FUNDING_ANOMALY"

        if abs(imbalance) > 0.7:
            return "CORRELATION_BREAK"

        return "UNKNOWN"

    def _heuristic_detect(self, features: dict, feat_vec: np.ndarray) -> dict:
        """Fallback heuristic detection when model is not fitted."""
        spread = float(features.get("spread_bps", 0.0))
        volume = float(features.get("volume_ratio", 1.0))
        velocity = float(features.get("price_velocity", 0.0))
        imbalance = float(features.get("orderbook_imbalance", 0.0))
        funding = float(features.get("funding_rate_deviation", 0.0))

        # Simple z-score-like heuristic
        anomaly_score = 0.0
        if spread > 30:
            anomaly_score += 0.3
        if abs(velocity) > 3.0:
            anomaly_score += 0.3
        if volume > 4.0:
            anomaly_score += 0.2
        if abs(imbalance) > 0.7:
            anomaly_score += 0.1
        if abs(funding) > 2.0:
            anomaly_score += 0.1

        anomaly_score = min(1.0, anomaly_score)
        is_anomaly = anomaly_score >= 0.5
        anomaly_type = self._classify_anomaly(features, anomaly_score) if is_anomaly else "UNKNOWN"

        return {
            "is_anomaly": is_anomaly,
            "score": round(anomaly_score, 4),
            "type": anomaly_type,
            "severity": round(anomaly_score, 4),
        }
