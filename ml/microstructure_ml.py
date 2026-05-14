"""
Microstructure ML — gradient boosting on L2 order book features to predict short-term price.

Features extracted from L2 order book snapshots:
  - bid/ask imbalance at 5 price levels
  - weighted mid-price vs simple mid
  - bid/ask spread in bps
  - volume-at-price distribution skew
  - recent trade flow (buys vs sells last 30s)
  - price momentum (1m, 5m)

Target: price direction in next N seconds (binary: up/flat=1, down=0)

Falls back to logistic regression if sklearn not installed.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional sklearn
# ---------------------------------------------------------------------------

try:
    from sklearn.ensemble import GradientBoostingClassifier as _GBC
    _SKLEARN_AVAILABLE = True
except ImportError:
    _GBC = None  # type: ignore[assignment,misc]
    _SKLEARN_AVAILABLE = False
    logger.info("sklearn not available — MicrostructureML will use built-in logistic regression.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_FEATURES = 17      # fixed feature vector length (see _extract_features)
_DEQUE_MAXLEN = 5000  # maximum stored training samples
_MIN_RETRAIN_INTERVAL_S = 60.0  # minimum seconds between consecutive retrains

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BookSnapshot:
    """A point-in-time snapshot of the L2 order book."""

    symbol: str
    bids: List[Tuple[float, float]]   # [(price, qty), ...] best bid first
    asks: List[Tuple[float, float]]   # [(price, qty), ...] best ask first
    timestamp: float
    recent_buy_vol: float = 0.0       # buy volume in last 30 s
    recent_sell_vol: float = 0.0      # sell volume in last 30 s


@dataclass
class MicrostructurePrediction:
    """Prediction output from MicrostructureML."""

    symbol: str
    direction: int          # +1 = up, -1 = down, 0 = flat/uncertain
    confidence: float       # [0, 1]
    horizon_seconds: int
    features_used: int      # number of features actually populated (may be < 17 if book thin)
    model_type: str         # "gbm" | "logistic" | "prior"


# ---------------------------------------------------------------------------
# MicrostructureML
# ---------------------------------------------------------------------------


class MicrostructureML:
    """
    Gradient-boosting model (sklearn GBC) that predicts short-term price direction
    from L2 order book features.  Falls back to a built-in logistic regression
    when sklearn is unavailable.

    Parameters
    ----------
    symbol : str
        Instrument symbol (e.g., "BTC/USD").
    horizon_seconds : int
        Prediction horizon.  Labelled samples require a realized_direction.
    min_samples : int
        Minimum labelled samples before the model trains for the first time.
    n_levels : int
        Number of price levels to use from each side of the book.
    """

    def __init__(
        self,
        symbol: str,
        horizon_seconds: int = 30,
        min_samples: int = 200,
        n_levels: int = 5,
    ) -> None:
        self.symbol = symbol
        self.horizon_seconds = horizon_seconds
        self.min_samples = max(10, min_samples)
        self.n_levels = max(1, min(n_levels, 5))  # cap at 5 (feature count assumes 5)

        # Training buffer: (features, label) — label is 1 (up) or 0 (down)
        self._buffer: deque[Tuple[List[float], int]] = deque(maxlen=_DEQUE_MAXLEN)

        # Model: sklearn GBC or weight vector (list) for built-in logistic
        self._model: Any = None
        self._model_type: str = "prior"
        self._model_version: int = 0
        self._last_retrain_ts: float = 0.0
        self._last_retrain_stats: Dict[str, Any] = {}

        # Mid-price history for momentum features
        self._mid_history: deque[Tuple[float, float]] = deque(maxlen=600)  # (ts, mid)

        # Stats
        self._n_fed: int = 0
        self._n_labelled: int = 0
        self._n_predicted: int = 0

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(
        self,
        snapshot: BookSnapshot,
        realized_direction: Optional[int] = None,
    ) -> None:
        """
        Ingest a new order book snapshot.

        Parameters
        ----------
        snapshot : BookSnapshot
            Latest L2 snapshot.
        realized_direction : Optional[int]
            +1 if price went up after horizon_seconds, -1 if down, None if unknown.
            When provided, adds a labelled sample to the training buffer.
        """
        with self._lock:
            self._n_fed += 1
            features = self._extract_features(snapshot)

            # Track mid-price for momentum
            mid = self._simple_mid(snapshot)
            if mid > 0:
                self._mid_history.append((snapshot.timestamp, mid))

            if realized_direction is not None and realized_direction in (1, -1):
                label = 1 if realized_direction == 1 else 0
                self._buffer.append((features, label))
                self._n_labelled += 1

                # Auto-retrain every min_samples new labels
                if (
                    self._n_labelled % self.min_samples == 0
                    and len(self._buffer) >= self.min_samples
                ):
                    elapsed = time.time() - self._last_retrain_ts
                    if elapsed >= _MIN_RETRAIN_INTERVAL_S:
                        self._retrain_locked()

    def predict(self, snapshot: BookSnapshot) -> MicrostructurePrediction:
        """
        Predict price direction from a book snapshot.

        Returns a MicrostructurePrediction.  Before min_samples are collected
        the model is not yet trained; the prediction returns direction=0 with
        confidence=0.5 and model_type="prior".
        """
        with self._lock:
            self._n_predicted += 1
            features = self._extract_features(snapshot)
            n_nonzero = sum(1 for f in features if f != 0.0)

            if self._model is None or len(self._buffer) < self.min_samples:
                return MicrostructurePrediction(
                    symbol=self.symbol,
                    direction=0,
                    confidence=0.5,
                    horizon_seconds=self.horizon_seconds,
                    features_used=n_nonzero,
                    model_type="prior",
                )

            prob = self._predict_proba_locked(features)
            if prob >= 0.55:
                direction = 1
            elif prob <= 0.45:
                direction = -1
            else:
                direction = 0

            confidence = abs(prob - 0.5) * 2.0 + 0.5  # re-scale to [0.5, 1.0]

            return MicrostructurePrediction(
                symbol=self.symbol,
                direction=direction,
                confidence=min(1.0, confidence),
                horizon_seconds=self.horizon_seconds,
                features_used=n_nonzero,
                model_type=self._model_type,
            )

    def retrain(self) -> Dict[str, Any]:
        """Retrain the model on buffered samples. Returns performance metrics."""
        with self._lock:
            return self._retrain_locked()

    def get_stats(self) -> Dict[str, Any]:
        """Return runtime statistics."""
        with self._lock:
            return {
                "symbol": self.symbol,
                "horizon_seconds": self.horizon_seconds,
                "n_fed": self._n_fed,
                "n_labelled": self._n_labelled,
                "n_predicted": self._n_predicted,
                "n_buffer": len(self._buffer),
                "min_samples": self.min_samples,
                "model_type": self._model_type,
                "model_version": self._model_version,
                "last_retrain_ts": self._last_retrain_ts,
                "last_retrain_stats": self._last_retrain_stats,
            }

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, snapshot: BookSnapshot) -> List[float]:
        """
        Build a 17-element feature vector from a BookSnapshot.

        Features:
          [0]     bid_ask_spread_bps
          [1-5]   obi_level_1..5  (order book imbalance per level)
          [6]     weighted_mid_vs_simple_mid_bps
          [7]     buy_sell_ratio_30s
          [8-12]  ask_depth_bps_1..5  (distance from mid to each ask level)
          [13-17] bid_depth_bps_1..5  (distance from mid to each bid level)
          BUT we only have index 0..16 → 17 features.

        Note: ask_depth and bid_depth only go to level 5 (indices 8-12 and 13-16, not 17).
        Actual layout:
          [0]     spread_bps
          [1..5]  obi_level_1..5
          [6]     weighted_mid_vs_simple_mid_bps
          [7]     buy_sell_ratio_30s
          [8..12] ask_depth_bps_1..5
          [13..16] bid_depth_bps_1..4   (total 17)
        Plus volume_skew at [16] — see below for exact layout used.
        """
        feats = [0.0] * _N_FEATURES

        bids = snapshot.bids or []
        asks = snapshot.asks or []

        if not bids or not asks:
            return feats

        best_bid = bids[0][0]
        best_ask = asks[0][0]

        if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
            return feats

        simple_mid = (best_bid + best_ask) / 2.0

        # [0] Spread in bps
        feats[0] = (best_ask - best_bid) / simple_mid * 10_000.0

        # [1..5] Order book imbalance per level
        for lvl in range(self.n_levels):
            bid_px, bid_qty = bids[lvl] if lvl < len(bids) else (0.0, 0.0)
            ask_px, ask_qty = asks[lvl] if lvl < len(asks) else (0.0, 0.0)
            denom = bid_qty + ask_qty
            feats[1 + lvl] = (bid_qty - ask_qty) / denom if denom > 0 else 0.0

        # [6] Weighted mid vs simple mid in bps
        feats[6] = self._weighted_mid_diff_bps(bids, asks, simple_mid)

        # [7] Buy/sell ratio over last 30 s
        buy_vol = max(snapshot.recent_buy_vol, 0.0)
        sell_vol = max(snapshot.recent_sell_vol, 0.0)
        total_flow = buy_vol + sell_vol
        feats[7] = buy_vol / total_flow if total_flow > 0 else 0.5

        # [8..12] Ask depth in bps (distance from mid to each ask level)
        for lvl in range(self.n_levels):
            ask_px = asks[lvl][0] if lvl < len(asks) else simple_mid
            feats[8 + lvl] = (ask_px - simple_mid) / simple_mid * 10_000.0 if simple_mid > 0 else 0.0

        # [13..16] Bid depth in bps (distance from mid to each bid level), 4 levels to fill 17
        for lvl in range(min(self.n_levels - 1, 4)):
            bid_px = bids[lvl][0] if lvl < len(bids) else simple_mid
            feats[13 + lvl] = (simple_mid - bid_px) / simple_mid * 10_000.0 if simple_mid > 0 else 0.0

        # [16] (overwrite slot 16) volume skew: log(bid_top5_vol / ask_top5_vol)
        bid_vol_top5 = sum(bids[i][1] for i in range(min(5, len(bids))))
        ask_vol_top5 = sum(asks[i][1] for i in range(min(5, len(asks))))
        if bid_vol_top5 > 0 and ask_vol_top5 > 0:
            feats[16] = math.log(bid_vol_top5 / ask_vol_top5)
        else:
            feats[16] = 0.0

        return feats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simple_mid(self, snapshot: BookSnapshot) -> float:
        if not snapshot.bids or not snapshot.asks:
            return 0.0
        return (snapshot.bids[0][0] + snapshot.asks[0][0]) / 2.0

    def _weighted_mid_diff_bps(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        simple_mid: float,
    ) -> float:
        """Weighted mid price minus simple mid, expressed in bps."""
        if not bids or not asks or simple_mid <= 0:
            return 0.0
        best_bid_px, best_bid_qty = bids[0]
        best_ask_px, best_ask_qty = asks[0]
        denom = best_bid_qty + best_ask_qty
        if denom <= 0:
            return 0.0
        weighted_mid = (best_bid_px * best_ask_qty + best_ask_px * best_bid_qty) / denom
        return (weighted_mid - simple_mid) / simple_mid * 10_000.0

    def _predict_proba_locked(self, features: List[float]) -> float:
        """Return probability of up-move. Must be called with lock held."""
        x = np.array(features, dtype=float)
        if self._model_type == "gbm" and _SKLEARN_AVAILABLE:
            try:
                prob = float(self._model.predict_proba(x.reshape(1, -1))[0, 1])
                return prob
            except Exception as exc:
                logger.warning("microstructure_ml: GBM predict failed: %s", exc)
                return 0.5

        # Built-in logistic
        w = np.array(self._model, dtype=float)
        x_b = np.append(x, 1.0)
        logit = float(np.dot(w, x_b))
        return float(1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit)))))

    def _retrain_locked(self) -> Dict[str, Any]:
        """Retrain. Must be called with self._lock held."""
        n = len(self._buffer)
        if n < self.min_samples:
            return {"status": "insufficient_data", "n_samples": n}

        X = [e[0] for e in self._buffer]
        y = [e[1] for e in self._buffer]
        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y, dtype=int)

        try:
            if _SKLEARN_AVAILABLE:
                clf = _GBC(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
                clf.fit(X_arr, y_arr)
                self._model = clf
                self._model_type = "gbm"
                y_pred = list(clf.predict(X_arr))
            else:
                weights = self._fit_logistic(X, y)
                self._model = weights
                self._model_type = "logistic"
                y_pred = [
                    1 if self._predict_proba_locked(x) >= 0.5 else 0 for x in X
                ]

            correct = sum(1 for a, b in zip(y, y_pred) if a == b)
            accuracy = correct / len(y)
            pos_rate = sum(y) / len(y)

            self._model_version += 1
            self._last_retrain_ts = time.time()
            stats: Dict[str, Any] = {
                "status": "ok",
                "model_type": self._model_type,
                "model_version": self._model_version,
                "n_samples": n,
                "accuracy": round(accuracy, 4),
                "positive_rate": round(pos_rate, 4),
            }
            self._last_retrain_stats = stats
            logger.info(
                "microstructure_ml[%s]: retrained v%d — acc=%.3f n=%d",
                self.symbol,
                self._model_version,
                accuracy,
                n,
            )
            return stats

        except Exception as exc:
            logger.error("microstructure_ml[%s]: retrain failed: %s", self.symbol, exc, exc_info=True)
            return {"status": "error", "error": str(exc)}

    @staticmethod
    def _fit_logistic(X: List[List[float]], y: List[int]) -> List[float]:
        """Pure-numpy logistic regression. Returns weight vector with bias appended."""
        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y, dtype=float)
        ones = np.ones((X_arr.shape[0], 1))
        X_b = np.hstack([X_arr, ones])
        w = np.zeros(X_b.shape[1])
        lr = 0.1
        for _ in range(300):
            logits = np.clip(X_b @ w, -20, 20)
            preds = 1.0 / (1.0 + np.exp(-logits))
            grad = X_b.T @ (preds - y_arr) / len(y_arr)
            w -= lr * grad
        return w.tolist()
