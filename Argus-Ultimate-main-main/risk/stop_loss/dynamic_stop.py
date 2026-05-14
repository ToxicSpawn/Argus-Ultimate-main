"""Dynamic Stop Loss - adjusts based on momentum, trend strength, and time in trade."""
from __future__ import annotations
import numpy as np


class DynamicStop:
    """Combines multiple factors for intelligent stop placement."""
    def __init__(self, base_pct: float = 0.02, momentum_weight: float = 0.3, trend_weight: float = 0.3, time_decay_weight: float = 0.2, atr_weight: float = 0.2):
        self.base_pct = base_pct
        self.momentum_weight = momentum_weight
        self.trend_weight = trend_weight
        self.time_decay_weight = time_decay_weight
        self.atr_weight = atr_weight

    def calculate(self, entry_price: float, side: str, close: np.ndarray, current_atr: float = 0.0, bars_held: int = 0, max_hold_bars: int = 200) -> dict:
        if len(close) < 14:
            fallback = entry_price * self.base_pct
            sp = entry_price - fallback if side == "long" else entry_price + fallback
            return {"stop_price": sp, "method": "dynamic_stop"}
        mom_10 = (close[-1] - close[-min(10, len(close))]) / close[-min(10, len(close))]
        trend_sign = 1.0 if mom_10 > 0 else -1.0
        if (side == "long" and trend_sign > 0) or (side == "short" and trend_sign < 0):
            trend_factor = 1.0 + abs(mom_10) * 5  # Widen when trend is favorable
        else:
            trend_factor = max(0.5, 1.0 - abs(mom_10) * 5)  # Tighten against trend
        time_factor = 1.0 - self.time_decay_weight * min(1.0, bars_held / max(max_hold_bars, 1))
        atr_factor = 1.0
        if current_atr > 0:
            atr_factor = current_atr / (entry_price * self.base_pct)
            atr_factor = max(0.5, min(3.0, atr_factor))
        effective_pct = self.base_pct * (self.momentum_weight * abs(mom_10) * 20 + self.trend_weight * trend_factor + self.time_decay_weight * time_factor + self.atr_weight * atr_factor)
        effective_pct = max(0.005, min(0.10, effective_pct))
        stop_dist = entry_price * effective_pct
        if side == "long":
            stop_price = entry_price - stop_dist
        else:
            stop_price = entry_price + stop_dist
        return {"stop_price": stop_price, "effective_pct": effective_pct, "trend_factor": trend_factor, "time_factor": time_factor, "method": "dynamic_stop"}
