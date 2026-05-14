"""
Meta-Labeling — secondary ML model that filters primary strategy signals.

Concept (Lopez de Prado): primary model generates directional signals;
meta-model predicts PROBABILITY that the primary signal will be profitable.
Only trade when meta-model confidence exceeds threshold.

Dramatically reduces false positives without changing the primary strategy.
The meta-labeler accumulates (features, outcome) pairs from live trading,
retrains periodically, and gates each new signal through the learned classifier.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional sklearn — GBM preferred, logistic regression is the fallback
# ---------------------------------------------------------------------------

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    GradientBoostingClassifier = None  # type: ignore[assignment,misc]
    LogisticRegression = None  # type: ignore[assignment,misc]
    _SKLEARN_AVAILABLE = False
    logger.info("sklearn not available — MetaLabeler will use built-in logistic regression.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RETRAIN_EVERY_N = 500          # auto-retrain trigger (new outcomes)
_DEQUE_MAXLEN = 2000            # maximum stored (features, outcome) pairs
_DEFAULT_THRESHOLD = 0.55       # minimum meta-confidence to trade
_MIN_SAMPLES_DEFAULT = 100      # pass-through below this sample count

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PrimarySignal:
    """Signal produced by a primary directional strategy."""

    symbol: str
    direction: int              # 1 = long, -1 = short, 0 = flat
    confidence: float           # [0, 1] primary model confidence
    strategy: str               # e.g. "mtf_confluence", "stat_arb"
    timestamp: float = field(default_factory=time.time)
    features_json: str = "{}"   # JSON-encoded additional features (optional)
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class MetaLabel:
    """Decision produced by the meta-labeler for a single primary signal."""

    signal_id: str
    trade: bool                  # True → execute; False → skip
    meta_confidence: float       # probability estimate from meta-model
    expected_return_bps: float   # estimated return in basis points (may be 0 pre-training)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# MetaLabeler
# ---------------------------------------------------------------------------


class MetaLabeler:
    """
    Secondary binary classifier that decides whether to act on primary signals.

    Parameters
    ----------
    threshold : float
        Minimum meta-model probability required to approve a trade.
    min_samples : int
        Number of labelled outcomes required before the model becomes active.
        Below this count every signal is approved (pass-through mode).
    model_type : str
        "logistic" | "gbm" | "auto".
        "auto" selects GBM when sklearn is available, otherwise logistic.
    """

    def __init__(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        min_samples: int = _MIN_SAMPLES_DEFAULT,
        model_type: str = "auto",
    ) -> None:
        self.threshold = max(0.5, min(1.0, threshold))
        self.min_samples = max(1, min_samples)

        if model_type == "auto":
            self._model_type = "gbm" if _SKLEARN_AVAILABLE else "logistic"
        elif model_type in ("gbm", "logistic"):
            self._model_type = model_type
        else:
            logger.warning("Unknown model_type %r — defaulting to 'logistic'.", model_type)
            self._model_type = "logistic"

        # Storage: each entry is (features_list, outcome_int)
        self._buffer: deque[Tuple[List[float], int]] = deque(maxlen=_DEQUE_MAXLEN)

        # Map signal_id → features (kept until outcome arrives)
        self._pending: Dict[str, List[float]] = {}

        # Trained model — either sklearn object or weight vector
        self._model: Any = None
        self._model_version: int = 0

        # Counters for auto-retrain
        self._outcomes_since_retrain: int = 0

        # Stats
        self._n_evaluated: int = 0
        self._n_traded: int = 0
        self._last_retrain_ts: Optional[float] = None
        self._last_retrain_stats: Dict[str, Any] = {}

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, signal: PrimarySignal) -> MetaLabel:
        """
        Gate a primary signal through the meta-model.

        Returns a MetaLabel indicating whether to trade and the estimated
        probability that the trade will be profitable.
        """
        with self._lock:
            self._n_evaluated += 1
            features = self._build_features(signal)
            self._pending[signal.signal_id] = features

            n_samples = len(self._buffer)

            # Pass-through before min_samples
            if n_samples < self.min_samples or self._model is None:
                label = MetaLabel(
                    signal_id=signal.signal_id,
                    trade=True,
                    meta_confidence=0.5,
                    expected_return_bps=0.0,
                )
                self._n_traded += 1
                return label

            prob = self._predict_proba(features)
            trade = prob >= self.threshold

            if trade:
                self._n_traded += 1

            # Rough expected return: scale probability linearly around 0.5
            expected_bps = (prob - 0.5) * 200.0  # 0.5 → 0 bps; 1.0 → 100 bps

            return MetaLabel(
                signal_id=signal.signal_id,
                trade=trade,
                meta_confidence=prob,
                expected_return_bps=expected_bps,
            )

    def record_outcome(self, signal_id: str, actual_pnl_bps: float) -> None:
        """
        Record the realised outcome of a trade to the training buffer.

        Call this after a trade closes.  The outcome label is 1 when the
        trade was profitable, 0 otherwise.
        """
        with self._lock:
            features = self._pending.pop(signal_id, None)
            if features is None:
                logger.debug("meta_labeling: no pending features for signal_id=%s", signal_id)
                return

            outcome = 1 if actual_pnl_bps > 0 else 0
            self._buffer.append((features, outcome))
            self._outcomes_since_retrain += 1

            if self._outcomes_since_retrain >= _RETRAIN_EVERY_N:
                logger.info(
                    "meta_labeling: auto-retrain triggered after %d outcomes",
                    self._outcomes_since_retrain,
                )
                self._retrain_locked()

    def retrain(self) -> Dict[str, Any]:
        """Retrain the meta-model on buffered outcomes. Returns performance metrics."""
        with self._lock:
            return self._retrain_locked()

    def get_stats(self) -> Dict[str, Any]:
        """Return runtime statistics."""
        with self._lock:
            n_samples = len(self._buffer)
            return {
                "model_type": self._model_type,
                "model_version": self._model_version,
                "n_evaluated": self._n_evaluated,
                "n_traded": self._n_traded,
                "n_buffer": n_samples,
                "n_pending": len(self._pending),
                "threshold": self.threshold,
                "min_samples": self.min_samples,
                "pass_through_mode": n_samples < self.min_samples or self._model is None,
                "last_retrain_ts": self._last_retrain_ts,
                "last_retrain_stats": self._last_retrain_stats,
            }

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_features(self, signal: PrimarySignal) -> List[float]:
        """
        Build a fixed-length numeric feature vector from a PrimarySignal.

        Features:
          0: direction (float: -1.0, 0.0, 1.0)
          1: confidence
          2: hour_of_day / 23.0  (normalised)
          3: day_of_week / 6.0   (normalised)
          4: confidence_squared
        """
        import time as _time
        import datetime as _dt

        dt = _dt.datetime.utcfromtimestamp(signal.timestamp)
        hour_norm = dt.hour / 23.0
        dow_norm = dt.weekday() / 6.0

        conf = float(max(0.0, min(1.0, signal.confidence)))

        return [
            float(signal.direction),
            conf,
            hour_norm,
            dow_norm,
            conf * conf,
        ]

    # ------------------------------------------------------------------
    # Model fitting helpers
    # ------------------------------------------------------------------

    def _fit_logistic(
        self, X: List[List[float]], y: List[int]
    ) -> List[float]:
        """
        Fit logistic regression using gradient descent.

        Returns weight vector w (len = n_features + 1 for bias).
        Pure numpy — no sklearn required.
        """
        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y, dtype=float)

        # Append bias column
        ones = np.ones((X_arr.shape[0], 1))
        X_b = np.hstack([X_arr, ones])

        n_features = X_b.shape[1]
        w = np.zeros(n_features)

        lr = 0.1
        for epoch in range(300):
            logits = X_b @ w
            preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
            grad = X_b.T @ (preds - y_arr) / len(y_arr)
            w -= lr * grad

        return w.tolist()

    def _fit_gbm(self, X: List[List[float]], y: List[int]) -> Any:
        """
        Fit a gradient boosting classifier via sklearn.

        Raises ImportError if sklearn is unavailable (caller handles this).
        """
        clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
        clf.fit(np.array(X, dtype=float), y)
        return clf

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_proba(self, features: List[float]) -> float:
        """Return probability of profitable trade given features."""
        if self._model is None:
            return 0.5

        x = np.array(features, dtype=float)

        if self._model_type == "gbm" and _SKLEARN_AVAILABLE:
            try:
                prob: float = float(self._model.predict_proba(x.reshape(1, -1))[0, 1])
                return prob
            except Exception as exc:
                logger.warning("meta_labeling: GBM predict failed: %s", exc)
                return 0.5

        # Logistic weights vector (includes bias as last element)
        w = np.array(self._model, dtype=float)
        x_b = np.append(x, 1.0)  # bias
        logit = float(np.dot(w, x_b))
        return float(1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit)))))

    def _retrain_locked(self) -> Dict[str, Any]:
        """Retrain the model. Must be called with self._lock held."""
        if len(self._buffer) < self.min_samples:
            logger.debug(
                "meta_labeling: retrain skipped — only %d samples (need %d)",
                len(self._buffer),
                self.min_samples,
            )
            return {"status": "insufficient_data", "n_samples": len(self._buffer)}

        X = [entry[0] for entry in self._buffer]
        y = [entry[1] for entry in self._buffer]

        try:
            if self._model_type == "gbm" and _SKLEARN_AVAILABLE:
                self._model = self._fit_gbm(X, y)
                y_pred = list(self._model.predict(np.array(X, dtype=float)))
            else:
                weights = self._fit_logistic(X, y)
                self._model = weights
                y_pred = [
                    1 if self._predict_proba(x) >= 0.5 else 0 for x in X
                ]

            # Compute metrics
            correct = sum(1 for a, b in zip(y, y_pred) if a == b)
            accuracy = correct / len(y)

            tp = sum(1 for a, b in zip(y, y_pred) if a == 1 and b == 1)
            fp = sum(1 for a, b in zip(y, y_pred) if a == 0 and b == 1)
            fn = sum(1 for a, b in zip(y, y_pred) if a == 1 and b == 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            self._model_version += 1
            self._outcomes_since_retrain = 0
            self._last_retrain_ts = time.time()

            stats: Dict[str, Any] = {
                "status": "ok",
                "model_type": self._model_type,
                "model_version": self._model_version,
                "n_samples": len(y),
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            }
            self._last_retrain_stats = stats
            logger.info(
                "meta_labeling: retrained v%d — acc=%.3f prec=%.3f rec=%.3f n=%d",
                self._model_version,
                accuracy,
                precision,
                recall,
                len(y),
            )
            return stats

        except Exception as exc:
            logger.error("meta_labeling: retrain failed: %s", exc, exc_info=True)
            return {"status": "error", "error": str(exc)}
