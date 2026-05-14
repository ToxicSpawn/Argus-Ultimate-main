"""
Volatility-Adjusted Position Sizing
Targets constant portfolio volatility via inverse vol scaling.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict

import numpy as np


class VolatilityAdjustedSizer:
    """Inverse volatility targeting: size inversely proportional to realized vol."""

    def __init__(
        self,
        target_vol: float = 0.15,
        vol_lookback: int = 30,
        vol_floor: float = 0.05,
        vol_cap: float = 1.0,
        max_position_pct: float = 0.15,
        use_ewm: bool = True,
        ewm_span: int = 20,
    ):
        self.target_vol = float(target_vol)
        self.vol_lookback = int(vol_lookback)
        self.vol_floor = float(vol_floor)
        self.vol_cap = float(vol_cap)
        self.max_position_pct = float(max_position_pct)
        self.use_ewm = use_ewm
        self.ewm_span = int(ewm_span)
        self._returns: deque = deque(maxlen=max(vol_lookback * 2, 100))

    def update_return(self, ret: float) -> None:
        self._returns.append(float(ret))

    def _realized_vol(self) -> float:
        if len(self._returns) < 5:
            return self.target_vol
        rets = np.array(list(self._returns))
        if self.use_ewm and len(rets) >= self.ewm_span:
            alpha = 2.0 / (self.ewm_span + 1)
            weights = np.array([(1 - alpha) ** i for i in range(len(rets) - 1, -1, -1)])
            weights /= weights.sum()
            mean = np.sum(weights * rets)
            var = np.sum(weights * (rets - mean) ** 2)
            vol = float(np.sqrt(var))
        else:
            vol = float(np.std(rets[-self.vol_lookback:]))
        annualized = vol * np.sqrt(365 * 24)
        return max(annualized, self.vol_floor)

    def calculate(
        self,
        capital: float,
        risk_per_trade: float,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        cap = max(float(capital), 1.0)
        realized = self._realized_vol()
        vol_scale = self.target_vol / max(realized, self.vol_floor)
        vol_scale = min(vol_scale, self.vol_cap / self.vol_floor)
        vol_scale = max(vol_scale, 0.1)
        base_size = cap * float(risk_per_trade) * float(confidence)
        adjusted_size = base_size * vol_scale
        max_size = cap * self.max_position_pct
        adjusted_size = min(adjusted_size, max_size)
        return {
            "position_size": adjusted_size,
            "pct_of_capital": (adjusted_size / cap) * 100,
            "realized_vol": realized,
            "target_vol": self.target_vol,
            "vol_scale": vol_scale,
            "method": "volatility_adjusted",
        }
