"""Batch 2 — Dynamic Kelly position-sizer.

Computes the full and fractional Kelly fraction from a rolling window of
trade returns, with regime-specific half-Kelly adjustments and a hard
max-position cap.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Regime multipliers applied to the full Kelly fraction
REGIME_KELLY_SCALE = {
    "trending": 0.5,
    "mean_reverting": 0.35,
    "volatile": 0.2,
    "ranging": 0.3,
}


class DynamicKellySizer:
    """Rolling Kelly criterion with regime adjustment."""

    def __init__(
        self,
        window: int = 50,
        max_fraction: float = 0.25,
        min_fraction: float = 0.01,
        full_kelly_cap: float = 1.0,
    ) -> None:
        self._window = window
        self._max_f = max_fraction
        self._min_f = min_fraction
        self._full_kelly_cap = full_kelly_cap
        self._returns: Deque[float] = deque(maxlen=window)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_trade(self, pnl_pct: float) -> None:
        """Record a trade outcome as a fraction of capital (e.g. 0.02 = 2%)."""
        self._returns.append(pnl_pct)

    def kelly_fraction(
        self,
        regime: str = "ranging",
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
    ) -> float:
        """Return position fraction [0, max_fraction].

        If win_rate / avg_win / avg_loss are provided they override the
        rolling-window estimates.
        """
        if len(self._returns) >= 5:
            arr = np.array(self._returns)
            wins = arr[arr > 0]
            losses = arr[arr < 0]
            w = win_rate if win_rate is not None else (len(wins) / len(arr) if arr.size else 0.5)
            b = avg_win if avg_win is not None else (wins.mean() if wins.size else 0.01)
            a = abs(avg_loss) if avg_loss is not None else (abs(losses.mean()) if losses.size else 0.01)
        else:
            w, b, a = 0.5, 0.01, 0.01

        # Kelly = W/a - (1-W)/b  (discrete)
        kelly = w / a - (1 - w) / b if b > 0 and a > 0 else 0.0
        kelly = min(kelly, self._full_kelly_cap)

        scale = REGIME_KELLY_SCALE.get(regime, 0.3)
        fraction = kelly * scale
        fraction = float(np.clip(fraction, self._min_f, self._max_f))
        logger.debug(
            "Kelly: W=%.3f b=%.4f a=%.4f full_kelly=%.4f scale=%.2f → fraction=%.4f",
            w,
            b,
            a,
            kelly,
            scale,
            fraction,
        )
        return fraction

    def size_from_capital(
        self,
        capital: float,
        price: float,
        regime: str = "ranging",
    ) -> float:
        """Return notional quantity to trade."""
        f = self.kelly_fraction(regime=regime)
        notional = capital * f
        return notional / price if price > 0 else 0.0
