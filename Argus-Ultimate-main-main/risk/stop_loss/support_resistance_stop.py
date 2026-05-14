"""Support/Resistance Stop Loss - places stops at structural price levels."""
from __future__ import annotations
import numpy as np


class SupportResistanceStop:
    """Detects swing highs/lows for structural stop placement."""
    def __init__(self, lookback: int = 50, swing_window: int = 5, buffer_pct: float = 0.002, fallback_atr_mult: float = 2.0):
        self.lookback = lookback
        self.swing_window = swing_window
        self.buffer_pct = buffer_pct
        self.fallback_atr_mult = fallback_atr_mult

    def _find_swing_lows(self, low: np.ndarray) -> list:
        swings = []
        w = self.swing_window
        for i in range(w, len(low) - w):
            if low[i] == min(low[i - w:i + w + 1]):
                swings.append(float(low[i]))
        return swings

    def _find_swing_highs(self, high: np.ndarray) -> list:
        swings = []
        w = self.swing_window
        for i in range(w, len(high) - w):
            if high[i] == max(high[i - w:i + w + 1]):
                swings.append(float(high[i]))
        return swings

    def calculate(self, entry_price: float, side: str, high: np.ndarray, low: np.ndarray, close: np.ndarray, current_atr: float = 0.0) -> dict:
        h = high[-self.lookback:] if len(high) > self.lookback else high
        l = low[-self.lookback:] if len(low) > self.lookback else low
        if side == "long":
            swings = self._find_swing_lows(l)
            below_entry = [s for s in swings if s < entry_price]
            if below_entry:
                nearest = max(below_entry)
                stop_price = nearest * (1.0 - self.buffer_pct)
            elif current_atr > 0:
                stop_price = entry_price - self.fallback_atr_mult * current_atr
            else:
                stop_price = entry_price * 0.98
            return {"stop_price": stop_price, "sr_levels_found": len(below_entry), "method": "support_resistance_stop"}
        else:
            swings = self._find_swing_highs(h)
            above_entry = [s for s in swings if s > entry_price]
            if above_entry:
                nearest = min(above_entry)
                stop_price = nearest * (1.0 + self.buffer_pct)
            elif current_atr > 0:
                stop_price = entry_price + self.fallback_atr_mult * current_atr
            else:
                stop_price = entry_price * 1.02
            return {"stop_price": stop_price, "sr_levels_found": len(above_entry), "method": "support_resistance_stop"}
