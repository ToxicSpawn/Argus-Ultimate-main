"""ATR-Based Stop Loss - uses Average True Range for dynamic stop distance."""
from __future__ import annotations
import numpy as np


class AtrStopStop:
    """Stop loss based on ATR (Average True Range) multiplier."""
    def __init__(self, atr_multiplier: float = 2.0, atr_period: int = 14, min_stop_pct: float = 0.005):
        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period
        self.min_stop_pct = min_stop_pct

    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        if len(high) < 2:
            return 0.0
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        period = min(self.atr_period, len(tr))
        return float(np.mean(tr[-period:]))

    def calculate(self, entry_price: float, side: str, high: np.ndarray, low: np.ndarray, close: np.ndarray, regime_widen: float = 1.0) -> dict:
        atr = self.calculate_atr(high, low, close)
        stop_dist = max(self.atr_multiplier * atr * regime_widen, entry_price * self.min_stop_pct)
        if side == "long":
            stop_price = entry_price - stop_dist
        else:
            stop_price = entry_price + stop_dist
        return {"stop_price": stop_price, "atr": atr, "stop_distance": stop_dist, "method": "atr_stop"}
