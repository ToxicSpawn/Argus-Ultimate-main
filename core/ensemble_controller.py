"""EnsembleController — dynamic model weight management via rolling Sharpe.

Improvements over the original EMA-based weight updates:
1. Rolling Sharpe weighting — each model's weight is proportional to its
   rolling Sharpe ratio over the last N trades, not just win-rate EMA.
   Sharpe captures both returns AND consistency, which is what matters.
2. Drawdown suppression — if a model is in active drawdown (peak-to-current
   PnL drop > drawdown_threshold), its weight is halved until it recovers.
   This replaces the crude "3 consecutive losses → halve" heuristic.
3. MTF-aligned boost — models that agree with higher-timeframe bias get a
   configurable boost multiplier on their weight before normalisation.
4. Minimum floor and maximum cap enforced after every update to prevent
   both starvation and concentration risk.
5. Thread-safe with no external dependencies.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default rolling window for Sharpe calculation
DEFAULT_SHARPE_WINDOW = 30

# Drawdown threshold that triggers weight suppression
DEFAULT_DRAWDOWN_THRESHOLD = 0.10   # 10% peak-to-trough

# Weight suppression multiplier when in drawdown
DRAWDOWN_SUPPRESSION = 0.5

# MTF alignment boost when model agrees with higher-TF bias
MTF_BOOST = 1.25


@dataclass
class ModelWeight:
    model_id: str
    weight: float
    win_rate: float = 0.5
    recent_pnl: float = 0.0
    sharpe: float = 0.0
    in_drawdown: bool = False
    peak_cumulative_pnl: float = 0.0
    cumulative_pnl: float = 0.0
    last_updated: float = field(default_factory=time.time)
    _pnl_history: deque = field(
        default_factory=lambda: deque(maxlen=DEFAULT_SHARPE_WINDOW)
    )

    def rolling_sharpe(self) -> float:
        """Compute rolling Sharpe from stored PnL history."""
        hist = list(self._pnl_history)
        if len(hist) < 3:
            return 0.0
        n = len(hist)
        mean = sum(hist) / n
        var = sum((x - mean) ** 2 for x in hist) / max(1, n - 1)
        std = math.sqrt(var) if var > 0 else 1e-9
        # Annualise with sqrt(252) assuming daily trades; scale-invariant for ranking
        return (mean / std) * math.sqrt(min(n, DEFAULT_SHARPE_WINDOW))

    def update_drawdown(self, drawdown_threshold: float) -> None:
        """Update peak PnL and set in_drawdown flag."""
        if self.cumulative_pnl > self.peak_cumulative_pnl:
            self.peak_cumulative_pnl = self.cumulative_pnl
        dd = (
            (self.peak_cumulative_pnl - self.cumulative_pnl)
            / max(abs(self.peak_cumulative_pnl), 1e-9)
        ) if self.peak_cumulative_pnl != 0 else 0.0
        self.in_drawdown = dd > drawdown_threshold


class EnsembleController:
    """
    Manages dynamic model weights for ensemble signal aggregation.

    Weight update rules (in order):
    1. Record trade PnL into rolling window
    2. Recompute rolling Sharpe for the model
    3. Check drawdown status
    4. Set raw weight = max(0, Sharpe), with floor MIN_WEIGHT
    5. Apply drawdown suppression multiplier if in drawdown
    6. Apply MTF boost if model agrees with higher-TF bias
    7. Normalise all weights to sum to 1.0
    8. Clip to [MIN_WEIGHT, MAX_WEIGHT]
    """

    MIN_WEIGHT = 0.02
    MAX_WEIGHT = 0.55

    def __init__(
        self,
        model_ids: Optional[List[str]] = None,
        sharpe_window: int = DEFAULT_SHARPE_WINDOW,
        drawdown_threshold: float = DEFAULT_DRAWDOWN_THRESHOLD,
    ) -> None:
        self._lock = threading.Lock()
        self._weights: Dict[str, ModelWeight] = {}
        self._sharpe_window = sharpe_window
        self._drawdown_threshold = drawdown_threshold

        if model_ids:
            equal_w = 1.0 / len(model_ids)
            for mid in model_ids:
                self._weights[mid] = ModelWeight(model_id=mid, weight=equal_w)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_model(self, model_id: str, initial_weight: float = 0.1) -> None:
        with self._lock:
            if model_id not in self._weights:
                self._weights[model_id] = ModelWeight(
                    model_id=model_id, weight=float(initial_weight)
                )
                self._normalise()

    # ------------------------------------------------------------------
    # Weight accessors
    # ------------------------------------------------------------------

    def get_weight(self, model_id: str) -> float:
        with self._lock:
            mw = self._weights.get(model_id)
            return mw.weight if mw else 0.0

    def get_all_weights(self) -> Dict[str, float]:
        with self._lock:
            return {mid: mw.weight for mid, mw in self._weights.items()}

    def get_sharpe_scores(self) -> Dict[str, float]:
        with self._lock:
            return {mid: mw.rolling_sharpe() for mid, mw in self._weights.items()}

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        model_id: str,
        pnl: float,
        win: bool,
        mtf_aligned: bool = False,
    ) -> None:
        """
        Update model performance and rebalance all weights.

        Args:
            model_id:    Strategy/model identifier.
            pnl:         Trade P&L (signed, absolute dollar or %).
            win:         Whether the trade was a winner.
            mtf_aligned: True if this model's signal agreed with the
                         higher-timeframe bias on this bar.
        """
        with self._lock:
            mw = self._weights.get(model_id)
            if mw is None:
                logger.warning("EnsembleController: unknown model '%s'", model_id)
                return

            # Record PnL into rolling window
            mw._pnl_history.append(float(pnl))
            mw.cumulative_pnl += float(pnl)
            mw.recent_pnl = 0.9 * mw.recent_pnl + 0.1 * float(pnl)

            # EMA win rate (kept for diagnostics)
            mw.win_rate = 0.9 * mw.win_rate + 0.1 * (1.0 if win else 0.0)
            mw.last_updated = time.time()

            # Recompute Sharpe
            mw.sharpe = mw.rolling_sharpe()

            # Drawdown check
            mw.update_drawdown(self._drawdown_threshold)

            # Raw weight = rectified Sharpe (negative Sharpe → floor)
            raw = max(0.0, mw.sharpe)
            mw.weight = max(self.MIN_WEIGHT, raw)

            # Drawdown suppression
            if mw.in_drawdown:
                mw.weight *= DRAWDOWN_SUPPRESSION
                mw.weight = max(self.MIN_WEIGHT, mw.weight)
                logger.info(
                    "EnsembleController: model '%s' in drawdown — weight suppressed to %.4f",
                    model_id, mw.weight,
                )

            # MTF alignment boost
            if mtf_aligned:
                mw.weight = min(self.MAX_WEIGHT, mw.weight * MTF_BOOST)

            self._normalise()

            logger.debug(
                "Ensemble updated: model=%s pnl=%.2f sharpe=%.3f weight=%.4f "
                "drawdown=%s mtf_boost=%s",
                model_id, pnl, mw.sharpe, mw.weight, mw.in_drawdown, mtf_aligned,
            )

    # ------------------------------------------------------------------
    # Weighted signal aggregation helper
    # ------------------------------------------------------------------

    def aggregate_signals(
        self,
        signals: Dict[str, float],   # model_id -> signal value (-1 to +1)
    ) -> Tuple[float, str]:
        """
        Weighted average of signals from all registered models.

        Returns:
            (aggregated_signal, dominant_model_id)
            aggregated_signal: -1.0 to +1.0
            dominant_model_id: model with highest weight that contributed
        """
        with self._lock:
            total_weight = 0.0
            weighted_sum = 0.0
            dominant = ""
            dominant_w = -1.0

            for model_id, signal in signals.items():
                w = self._weights.get(model_id)
                if w is None:
                    continue
                weighted_sum += w.weight * float(signal)
                total_weight += w.weight
                if w.weight > dominant_w:
                    dominant_w = w.weight
                    dominant = model_id

            if total_weight <= 0:
                return 0.0, ""

            return weighted_sum / total_weight, dominant

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_model_stats(self) -> List[dict]:
        with self._lock:
            stats = []
            for mid, mw in self._weights.items():
                stats.append({
                    "model_id":      mid,
                    "weight":        round(mw.weight, 6),
                    "win_rate":      round(mw.win_rate, 4),
                    "sharpe":        round(mw.sharpe, 4),
                    "in_drawdown":   mw.in_drawdown,
                    "cumulative_pnl": round(mw.cumulative_pnl, 4),
                    "n_trades":      len(mw._pnl_history),
                })
            return sorted(stats, key=lambda x: x["sharpe"], reverse=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _normalise(self) -> None:
        """Normalise all weights to sum to 1.0, then clip to [MIN, MAX]."""
        total = sum(mw.weight for mw in self._weights.values())
        if total <= 0:
            # Reset to equal weights
            n = max(1, len(self._weights))
            for mw in self._weights.values():
                mw.weight = 1.0 / n
            return
        for mw in self._weights.values():
            mw.weight = mw.weight / total
            mw.weight = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, mw.weight))
        # Re-normalise after clipping
        total2 = sum(mw.weight for mw in self._weights.values())
        if total2 > 0:
            for mw in self._weights.values():
                mw.weight /= total2
