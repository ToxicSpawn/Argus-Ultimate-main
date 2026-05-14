"""
Triple Barrier Labeling — Marcos Lopez de Prado's labeling method.

Implements the triple-barrier method from *Advances in Financial Machine
Learning* (Ch. 3).  Each observation receives one of three labels based on
which barrier is touched first:

  +1  upper profit-taking barrier hit first
  -1  lower stop-loss barrier hit first
   0  maximum holding period expired (timeout)

Also provides **meta-labeling** (Ch. 3.6): given a primary model's side
prediction, the meta-label indicates whether that prediction was *correct*
(1) or *wrong* (0), enabling a secondary model to size positions by
confidence.

Pure Python + numpy.  No exchange or config dependencies — works standalone.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BarrierResult:
    """Result for a single triple-barrier evaluation."""

    label: int               # -1, 0, or 1
    touch_time_idx: int      # bar index where the barrier was touched
    return_at_touch: float   # raw return at touch point
    barrier_type: str        # "upper", "lower", or "timeout"


class TripleBarrierLabeler:
    """Marcos Lopez de Prado triple-barrier labeling engine.

    Parameters
    ----------
    upper_barrier : float
        Fractional return threshold for profit-taking (default 2 %).
    lower_barrier : float
        Fractional return threshold for stop-loss — must be negative
        (default -2 %).
    max_holding_bars : int
        Maximum holding period in bars before a timeout label is assigned.
    """

    def __init__(
        self,
        upper_barrier: float = 0.02,
        lower_barrier: float = -0.02,
        max_holding_bars: int = 20,
    ) -> None:
        if upper_barrier <= 0:
            raise ValueError(f"upper_barrier must be > 0, got {upper_barrier}")
        if lower_barrier >= 0:
            raise ValueError(f"lower_barrier must be < 0, got {lower_barrier}")
        if max_holding_bars < 1:
            raise ValueError(f"max_holding_bars must be >= 1, got {max_holding_bars}")

        self.upper_barrier = upper_barrier
        self.lower_barrier = lower_barrier
        self.max_holding_bars = max_holding_bars

        logger.info(
            "TripleBarrierLabeler initialised: upper=%.4f lower=%.4f max_bars=%d",
            self.upper_barrier, self.lower_barrier, self.max_holding_bars,
        )

    # ------------------------------------------------------------------
    # Core labeling
    # ------------------------------------------------------------------

    def _label_single(
        self,
        prices: np.ndarray,
        entry_idx: int,
        upper: float,
        lower: float,
        max_bars: int,
    ) -> BarrierResult:
        """Label a single entry point against the three barriers."""
        entry_price = prices[entry_idx]
        horizon = min(entry_idx + max_bars, len(prices) - 1)

        for t in range(entry_idx + 1, horizon + 1):
            ret = (prices[t] - entry_price) / entry_price

            if ret >= upper:
                return BarrierResult(
                    label=1,
                    touch_time_idx=t,
                    return_at_touch=ret,
                    barrier_type="upper",
                )
            if ret <= lower:
                return BarrierResult(
                    label=-1,
                    touch_time_idx=t,
                    return_at_touch=ret,
                    barrier_type="lower",
                )

        # Timeout — use return at horizon
        final_ret = (prices[horizon] - entry_price) / entry_price if horizon > entry_idx else 0.0
        return BarrierResult(
            label=0,
            touch_time_idx=horizon,
            return_at_touch=final_ret,
            barrier_type="timeout",
        )

    def label(
        self,
        prices: Sequence[float],
        timestamps: Optional[Sequence] = None,
        upper_barrier: Optional[float] = None,
        lower_barrier: Optional[float] = None,
        max_holding_bars: Optional[int] = None,
    ) -> List[int]:
        """Label every bar in *prices* using the triple-barrier method.

        Parameters
        ----------
        prices : list[float]
            Ordered price series (close prices).
        timestamps : list, optional
            Parallel timestamp series (currently informational only).
        upper_barrier : float, optional
            Override the instance upper barrier.
        lower_barrier : float, optional
            Override the instance lower barrier.
        max_holding_bars : int, optional
            Override the instance max holding bars.

        Returns
        -------
        list[int]
            Labels for each bar: -1 (stop-loss), 0 (timeout), +1 (profit).
            The last *max_holding_bars* entries will typically be 0 (timeout)
            because there is insufficient future data.
        """
        arr = np.asarray(prices, dtype=np.float64)
        n = len(arr)
        if n < 2:
            logger.warning("Price series too short (%d bars) for labeling", n)
            return [0] * n

        ub = upper_barrier if upper_barrier is not None else self.upper_barrier
        lb = lower_barrier if lower_barrier is not None else self.lower_barrier
        mb = max_holding_bars if max_holding_bars is not None else self.max_holding_bars

        labels: List[int] = []
        for i in range(n):
            result = self._label_single(arr, i, ub, lb, mb)
            labels.append(result.label)

        counts = {-1: labels.count(-1), 0: labels.count(0), 1: labels.count(1)}
        logger.info(
            "Triple-barrier labeling complete: n=%d  +1=%d  0=%d  -1=%d",
            n, counts[1], counts[0], counts[-1],
        )
        return labels

    def label_detailed(
        self,
        prices: Sequence[float],
        timestamps: Optional[Sequence] = None,
        upper_barrier: Optional[float] = None,
        lower_barrier: Optional[float] = None,
        max_holding_bars: Optional[int] = None,
    ) -> List[BarrierResult]:
        """Like :meth:`label` but returns full :class:`BarrierResult` objects."""
        arr = np.asarray(prices, dtype=np.float64)
        n = len(arr)
        if n < 2:
            return [BarrierResult(0, 0, 0.0, "timeout")] * n

        ub = upper_barrier if upper_barrier is not None else self.upper_barrier
        lb = lower_barrier if lower_barrier is not None else self.lower_barrier
        mb = max_holding_bars if max_holding_bars is not None else self.max_holding_bars

        return [self._label_single(arr, i, ub, lb, mb) for i in range(n)]

    # ------------------------------------------------------------------
    # Meta-labeling
    # ------------------------------------------------------------------

    def get_meta_labels(
        self,
        prices: Sequence[float],
        primary_signal: Sequence[int],
        upper_barrier: Optional[float] = None,
        lower_barrier: Optional[float] = None,
        max_holding_bars: Optional[int] = None,
    ) -> List[int]:
        """Produce meta-labels that evaluate whether a primary signal's side
        prediction leads to a profitable outcome.

        Parameters
        ----------
        prices : list[float]
            Price series.
        primary_signal : list[int]
            Primary model predictions — +1 (long), -1 (short), 0 (no trade).
            Must be same length as *prices*.

        Returns
        -------
        list[int]
            1 if the primary signal was correct (profitable), 0 otherwise.
            For bars where primary_signal is 0, the meta-label is 0.
        """
        arr = np.asarray(prices, dtype=np.float64)
        n = len(arr)
        if len(primary_signal) != n:
            raise ValueError(
                f"primary_signal length ({len(primary_signal)}) must match "
                f"prices length ({n})"
            )

        ub = upper_barrier if upper_barrier is not None else self.upper_barrier
        lb = lower_barrier if lower_barrier is not None else self.lower_barrier
        mb = max_holding_bars if max_holding_bars is not None else self.max_holding_bars

        meta_labels: List[int] = []
        for i in range(n):
            side = primary_signal[i]
            if side == 0:
                meta_labels.append(0)
                continue

            result = self._label_single(arr, i, ub, lb, mb)

            # Meta-label is 1 when the barrier outcome aligns with the bet side
            if side == 1 and result.label == 1:
                meta_labels.append(1)
            elif side == -1 and result.label == -1:
                # Short side: hitting the lower barrier is profitable for shorts
                meta_labels.append(1)
            elif result.label == 0 and result.return_at_touch * side > 0:
                # Timeout but still in the right direction
                meta_labels.append(1)
            else:
                meta_labels.append(0)

        correct = sum(meta_labels)
        total_signals = sum(1 for s in primary_signal if s != 0)
        accuracy = correct / total_signals if total_signals > 0 else 0.0
        logger.info(
            "Meta-labeling complete: %d signals, %.1f%% correct",
            total_signals, accuracy * 100,
        )
        return meta_labels

    # ------------------------------------------------------------------
    # Adaptive barriers from volatility
    # ------------------------------------------------------------------

    def label_with_volatility_scaling(
        self,
        prices: Sequence[float],
        volatilities: Sequence[float],
        vol_multiplier: float = 1.0,
        max_holding_bars: Optional[int] = None,
    ) -> List[int]:
        """Label using per-bar volatility-scaled barriers.

        Instead of fixed barriers, each bar's barriers are set to
        ``+/- vol_multiplier * volatility[i]``.  This normalises the
        label distribution across high- and low-vol regimes.

        Parameters
        ----------
        prices : list[float]
            Price series.
        volatilities : list[float]
            Per-bar volatility estimate (e.g. realised vol over trailing window).
        vol_multiplier : float
            Scaling factor applied to volatility for barrier width.

        Returns
        -------
        list[int]
            Labels: -1, 0, +1.
        """
        arr = np.asarray(prices, dtype=np.float64)
        vols = np.asarray(volatilities, dtype=np.float64)
        n = len(arr)
        if len(vols) != n:
            raise ValueError("volatilities must be same length as prices")

        mb = max_holding_bars if max_holding_bars is not None else self.max_holding_bars

        labels: List[int] = []
        for i in range(n):
            v = max(abs(vols[i]) * vol_multiplier, 1e-8)
            result = self._label_single(arr, i, v, -v, mb)
            labels.append(result.label)

        logger.info("Volatility-scaled labeling complete: n=%d", n)
        return labels
