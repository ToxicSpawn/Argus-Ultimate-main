"""Volatility Stop Loss - widens in high-vol, tightens in low-vol."""
from __future__ import annotations
import numpy as np


class VolatilityStop:
    """Stop distance scales with current vs historical volatility."""
    def __init__(self, lookback: int = 20, vol_multiplier: float = 2.0, bb_period: int = 20, bb_std: float = 2.0):
        self.lookback = lookback
        self.vol_multiplier = vol_multiplier
        self.bb_period = bb_period
        self.bb_std = bb_std

    def calculate(self, entry_price: float, side: str, close: np.ndarray, high: np.ndarray = None, low: np.ndarray = None) -> dict:
        if len(close) < self.lookback:
            fallback = entry_price * 0.02
            sp = entry_price - fallback if side == "long" else entry_price + fallback
            return {"stop_price": sp, "method": "volatility_stop", "vol_ratio": 1.0}
        rets = np.diff(close[-self.lookback - 1:]) / close[-self.lookback - 1:-1]
        current_vol = float(np.std(rets[-self.lookback // 2:]))
        hist_vol = float(np.std(rets))
        vol_ratio = current_vol / max(hist_vol, 1e-9)
        bb_mid = float(np.mean(close[-self.bb_period:]))
        bb_width = float(np.std(close[-self.bb_period:])) * self.bb_std
        bb_pct = bb_width / max(bb_mid, 1e-9)
        stop_dist = entry_price * bb_pct * max(0.5, min(2.5, vol_ratio))
        if side == "long":
            stop_price = entry_price - stop_dist
        else:
            stop_price = entry_price + stop_dist
        return {"stop_price": stop_price, "vol_ratio": vol_ratio, "bb_pct": bb_pct, "method": "volatility_stop"}
