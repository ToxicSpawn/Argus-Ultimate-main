"""
Orderbook Predictor — ensemble predictor for 30-second price direction from L2 data.

Combines:
  1. MicrostructureML (gradient boosting on order book features)
  2. OrderFlowImbalance (direct OBI signal from top-5 bid/ask volumes)
  3. MomentumFilter (price momentum confirmation from recent mid-price history)

Returns: direction, ensemble_confidence, contributing_signals, horizon_seconds.

The ensemble uses fixed weights:
  microstructure = 0.5, OBI = 0.3, momentum = 0.2

A final direction is assigned only when abs(weighted_sum) > 0.1; otherwise 0 (flat).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import of MicrostructureML from sibling module
try:
    from ml.microstructure_ml import BookSnapshot, MicrostructureML
    _MICRO_AVAILABLE = True
except ImportError:
    MicrostructureML = None  # type: ignore[assignment,misc]
    BookSnapshot = None  # type: ignore[assignment]
    _MICRO_AVAILABLE = False
    logger.info(
        "orderbook_predictor: ml.microstructure_ml not importable; "
        "microstructure component will be skipped."
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WEIGHT_MICRO = 0.5
_WEIGHT_OBI = 0.3
_WEIGHT_MOMENTUM = 0.2
_DIRECTION_THRESHOLD = 0.1   # abs(weighted_sum) must exceed this for non-flat direction
_MID_HISTORY_MAXLEN = 1800   # ~30 min at 1 update/s

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PredictorSignal:
    """Contribution from a single ensemble component."""

    source: str              # "microstructure" | "obi" | "momentum"
    direction: int           # +1, -1, 0
    confidence: float        # [0, 1]
    weight: float            # ensemble weight


@dataclass
class OrderbookPrediction:
    """Final ensemble prediction for a single book update."""

    symbol: str
    direction: int                      # +1, -1, 0
    ensemble_confidence: float          # abs(weighted_sum), clipped to [0, 1]
    signals: List[PredictorSignal] = field(default_factory=list)
    horizon_seconds: int = 30
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# OrderbookPredictor
# ---------------------------------------------------------------------------


class OrderbookPredictor:
    """
    Ensemble predictor that combines MicrostructureML, direct OBI, and momentum.

    Parameters
    ----------
    symbol : str
        Instrument symbol (e.g., "BTC/USD").
    horizon_seconds : int
        Prediction horizon used by the microstructure sub-model.
    """

    def __init__(self, symbol: str, horizon_seconds: int = 30) -> None:
        self.symbol = symbol
        self.horizon_seconds = horizon_seconds

        # Microstructure sub-model
        if _MICRO_AVAILABLE:
            self._micro = MicrostructureML(
                symbol=symbol,
                horizon_seconds=horizon_seconds,
            )
        else:
            self._micro = None

        # Mid-price history for momentum calculation
        self._mid_history: deque[Tuple[float, float]] = deque(maxlen=_MID_HISTORY_MAXLEN)

        # Outcome tracking for accuracy computation
        self._predictions: deque[Tuple[int, float]] = deque(maxlen=2000)  # (direction, ts)
        self._outcomes: List[Tuple[int, int]] = []   # (predicted, actual)

        # Per-source accuracy tracking
        self._source_outcomes: Dict[str, List[Tuple[int, int]]] = {
            "microstructure": [],
            "obi": [],
            "momentum": [],
        }

        # Stats
        self._n_updates: int = 0
        self._last_prediction: Optional[OrderbookPrediction] = None

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        snapshot: "BookSnapshot",
        recent_trades: List[Dict[str, Any]],
    ) -> OrderbookPrediction:
        """
        Ingest a new L2 snapshot and recent trade list; return an ensemble prediction.

        Parameters
        ----------
        snapshot : BookSnapshot
            Current L2 order book state.
        recent_trades : List[Dict]
            List of dicts with keys: price, qty, side ("buy"|"sell"), ts.

        Returns
        -------
        OrderbookPrediction
        """
        with self._lock:
            self._n_updates += 1
            signals: List[PredictorSignal] = []

            bids = getattr(snapshot, "bids", []) or []
            asks = getattr(snapshot, "asks", []) or []

            # -- Track mid-price history --
            simple_mid = self._simple_mid(bids, asks)
            if simple_mid > 0:
                self._mid_history.append((snapshot.timestamp, simple_mid))

            # Enrich snapshot buy/sell volumes from recent_trades
            buy_vol, sell_vol = self._aggregate_trade_flow(
                recent_trades, snapshot.timestamp, window_s=30
            )

            # -- Signal 1: Microstructure ML --
            micro_sig = self._microstructure_signal(snapshot, buy_vol, sell_vol)
            if micro_sig is not None:
                signals.append(micro_sig)

            # -- Signal 2: OBI (direct order book imbalance) --
            obi_sig = self._obi_signal(bids, asks)
            signals.append(obi_sig)

            # -- Signal 3: Momentum --
            mom_sig = self._momentum_signal(snapshot.timestamp)
            signals.append(mom_sig)

            # -- Ensemble --
            weighted_sum = sum(
                s.direction * s.confidence * s.weight for s in signals
            )
            direction = (
                int(np.sign(weighted_sum))
                if abs(weighted_sum) > _DIRECTION_THRESHOLD
                else 0
            )
            ensemble_confidence = min(1.0, abs(weighted_sum))

            prediction = OrderbookPrediction(
                symbol=self.symbol,
                direction=direction,
                ensemble_confidence=round(ensemble_confidence, 4),
                signals=signals,
                horizon_seconds=self.horizon_seconds,
                timestamp=snapshot.timestamp,
            )
            self._last_prediction = prediction
            self._predictions.append((direction, snapshot.timestamp))
            return prediction

    def record_outcome(self, direction_actual: int) -> None:
        """
        Record the realised price direction (called horizon_seconds after prediction).

        Parameters
        ----------
        direction_actual : int
            +1 if price rose, -1 if fell.
        """
        with self._lock:
            if not self._predictions:
                return
            direction_pred, _ = self._predictions[-1]
            self._outcomes.append((direction_pred, direction_actual))

    def get_accuracy(self) -> Dict[str, Any]:
        """Return overall and per-source accuracy metrics."""
        with self._lock:
            if not self._outcomes:
                return {"overall": None, "per_source": {}, "n_outcomes": 0}

            correct = sum(1 for p, a in self._outcomes if p == a and p != 0)
            non_flat = sum(1 for p, _ in self._outcomes if p != 0)
            overall = correct / non_flat if non_flat > 0 else None

            per_source: Dict[str, Optional[float]] = {}
            for src, pairs in self._source_outcomes.items():
                if not pairs:
                    per_source[src] = None
                    continue
                src_correct = sum(1 for p, a in pairs if p == a and p != 0)
                src_non_flat = sum(1 for p, _ in pairs if p != 0)
                per_source[src] = src_correct / src_non_flat if src_non_flat > 0 else None

            return {
                "overall": overall,
                "per_source": per_source,
                "n_outcomes": len(self._outcomes),
            }

    def snapshot(self) -> Dict[str, Any]:
        """Return a serialisable summary of current state."""
        with self._lock:
            last = self._last_prediction
            return {
                "symbol": self.symbol,
                "horizon_seconds": self.horizon_seconds,
                "n_updates": self._n_updates,
                "n_outcomes": len(self._outcomes),
                "mid_history_len": len(self._mid_history),
                "last_direction": last.direction if last else None,
                "last_confidence": last.ensemble_confidence if last else None,
                "last_ts": last.timestamp if last else None,
                "has_micro_model": self._micro is not None,
            }

    # ------------------------------------------------------------------
    # Signal generators
    # ------------------------------------------------------------------

    def _microstructure_signal(
        self,
        snapshot: "BookSnapshot",
        buy_vol: float,
        sell_vol: float,
    ) -> Optional[PredictorSignal]:
        """Run MicrostructureML prediction and return a PredictorSignal."""
        if self._micro is None:
            return None
        try:
            # Update buy/sell volumes on the snapshot before predicting
            snapshot.recent_buy_vol = buy_vol
            snapshot.recent_sell_vol = sell_vol
            pred = self._micro.predict(snapshot)
            return PredictorSignal(
                source="microstructure",
                direction=pred.direction,
                confidence=pred.confidence,
                weight=_WEIGHT_MICRO,
            )
        except Exception as exc:
            logger.warning("orderbook_predictor: microstructure predict failed: %s", exc)
            return None

    def _obi_signal(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> PredictorSignal:
        """
        Compute direct Order Book Imbalance signal from top-5 levels.

        OBI = (bid_vol_top5 - ask_vol_top5) / (bid_vol_top5 + ask_vol_top5)
        Direction: +1 if OBI > 0.1, -1 if OBI < -0.1, else 0.
        Confidence: abs(OBI).
        """
        bid_vol = sum(bids[i][1] for i in range(min(5, len(bids))))
        ask_vol = sum(asks[i][1] for i in range(min(5, len(asks))))
        total = bid_vol + ask_vol

        if total <= 0:
            return PredictorSignal(source="obi", direction=0, confidence=0.0, weight=_WEIGHT_OBI)

        obi = (bid_vol - ask_vol) / total
        direction = 1 if obi > 0.1 else (-1 if obi < -0.1 else 0)
        confidence = min(1.0, abs(obi))

        return PredictorSignal(
            source="obi",
            direction=direction,
            confidence=confidence,
            weight=_WEIGHT_OBI,
        )

    def _momentum_signal(self, current_ts: float) -> PredictorSignal:
        """
        Compute 30-second price momentum signal.

        Momentum = (current_mid - mid_30s_ago) / mid_30s_ago * 10_000 (bps)
        Direction: sign(momentum), confidence: clipped abs(momentum) / 20 bps.
        """
        if len(self._mid_history) < 2:
            return PredictorSignal(
                source="momentum", direction=0, confidence=0.0, weight=_WEIGHT_MOMENTUM
            )

        current_mid = self._mid_history[-1][1]
        target_ts = current_ts - 30.0

        # Find closest entry at or before target_ts
        past_mid: Optional[float] = None
        for ts, mid in reversed(self._mid_history):
            if ts <= target_ts:
                past_mid = mid
                break

        if past_mid is None or past_mid <= 0 or current_mid <= 0:
            return PredictorSignal(
                source="momentum", direction=0, confidence=0.0, weight=_WEIGHT_MOMENTUM
            )

        momentum_bps = (current_mid - past_mid) / past_mid * 10_000.0
        direction = 1 if momentum_bps > 0 else (-1 if momentum_bps < 0 else 0)
        # Confidence: scale by 20 bps — a 20 bps move gives confidence 1.0
        confidence = min(1.0, abs(momentum_bps) / 20.0)

        return PredictorSignal(
            source="momentum",
            direction=direction,
            confidence=confidence,
            weight=_WEIGHT_MOMENTUM,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _simple_mid(
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> float:
        if not bids or not asks:
            return 0.0
        return (bids[0][0] + asks[0][0]) / 2.0

    @staticmethod
    def _aggregate_trade_flow(
        trades: List[Dict[str, Any]],
        now_ts: float,
        window_s: float = 30.0,
    ) -> Tuple[float, float]:
        """Sum buy and sell volumes within the last window_s seconds."""
        cutoff = now_ts - window_s
        buy_vol = 0.0
        sell_vol = 0.0
        for t in trades:
            ts = float(t.get("ts", 0))
            if ts < cutoff:
                continue
            qty = float(t.get("qty", 0))
            side = str(t.get("side", "")).lower()
            if side == "buy":
                buy_vol += qty
            elif side == "sell":
                sell_vol += qty
        return buy_vol, sell_vol
