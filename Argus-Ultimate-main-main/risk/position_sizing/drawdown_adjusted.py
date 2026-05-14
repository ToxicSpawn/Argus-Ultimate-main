"""
Drawdown-Adjusted Position Sizing
Tiered position reduction during drawdowns with recovery detection.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict

import numpy as np


class DrawdownAdjustedSizer:
    """Scale position size based on current drawdown depth."""

    def __init__(
        self,
        tiers: list = None,
        recovery_threshold: float = 0.02,
        max_position_pct: float = 0.15,
        lookback: int = 200,
    ):
        if tiers is None:
            tiers = [
                (0.03, 0.80),
                (0.05, 0.60),
                (0.08, 0.40),
                (0.10, 0.25),
                (0.15, 0.10),
            ]
        self.tiers = sorted(tiers, key=lambda x: x[0])
        self.recovery_threshold = float(recovery_threshold)
        self.max_position_pct = float(max_position_pct)
        self._equity_curve: deque = deque(maxlen=lookback)
        self._peak_equity: float = 0.0
        self._in_recovery: bool = False

    def update_equity(self, equity: float) -> None:
        eq = float(equity)
        self._equity_curve.append(eq)
        if eq > self._peak_equity:
            self._peak_equity = eq
            self._in_recovery = False
        elif self._peak_equity > 0:
            dd = (self._peak_equity - eq) / self._peak_equity
            if dd < self.recovery_threshold and len(self._equity_curve) > 5:
                recent = list(self._equity_curve)[-5:]
                if all(recent[i] <= recent[i + 1] for i in range(len(recent) - 1)):
                    self._in_recovery = True

    def _current_drawdown(self) -> float:
        if self._peak_equity <= 0 or not self._equity_curve:
            return 0.0
        return (self._peak_equity - self._equity_curve[-1]) / self._peak_equity

    def _dd_scale(self, dd: float) -> float:
        scale = 1.0
        for threshold, factor in self.tiers:
            if dd >= threshold:
                scale = factor
            else:
                break
        return scale

    def _volatility_of_equity(self) -> float:
        if len(self._equity_curve) < 10:
            return 0.0
        eq = np.array(list(self._equity_curve))
        rets = np.diff(eq) / np.maximum(eq[:-1], 1.0)
        return float(np.std(rets))

    def calculate(
        self,
        capital: float,
        risk_per_trade: float,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        cap = max(float(capital), 1.0)
        dd = self._current_drawdown()
        dd_scale = self._dd_scale(dd)
        if self._in_recovery:
            dd_scale = min(dd_scale * 1.3, 1.0)
        eq_vol = self._volatility_of_equity()
        vol_penalty = 1.0
        if eq_vol > 0.03:
            vol_penalty = max(0.5, 1.0 - (eq_vol - 0.03) * 10)
        base_size = cap * float(risk_per_trade) * float(confidence)
        adjusted_size = base_size * dd_scale * vol_penalty
        max_size = cap * self.max_position_pct
        adjusted_size = min(adjusted_size, max_size)
        return {
            "position_size": adjusted_size,
            "pct_of_capital": (adjusted_size / cap) * 100,
            "current_drawdown": dd,
            "dd_scale": dd_scale,
            "equity_vol": eq_vol,
            "vol_penalty": vol_penalty,
            "in_recovery": self._in_recovery,
            "peak_equity": self._peak_equity,
            "method": "drawdown_adjusted",
        }
