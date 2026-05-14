"""Time-Based Stop Loss - forces exit after maximum holding period."""
from __future__ import annotations
import time


class TimeBasedStop:
    """Exit if position held too long (decaying edge)."""
    def __init__(self, max_hold_seconds: int = 3600, max_hold_bars: int = 200, tighten_after_pct: float = 0.5):
        self.max_hold_seconds = max_hold_seconds
        self.max_hold_bars = max_hold_bars
        self.tighten_after_pct = tighten_after_pct

    def should_exit(self, bars_held: int = 0, seconds_held: float = 0.0) -> dict:
        bar_expired = bars_held >= self.max_hold_bars if self.max_hold_bars > 0 else False
        time_expired = seconds_held >= self.max_hold_seconds if self.max_hold_seconds > 0 else False
        bar_pct = bars_held / max(self.max_hold_bars, 1) if self.max_hold_bars > 0 else 0
        time_pct = seconds_held / max(self.max_hold_seconds, 1) if self.max_hold_seconds > 0 else 0
        hold_pct = max(bar_pct, time_pct)
        should_tighten = hold_pct >= self.tighten_after_pct
        tighten_factor = 1.0
        if should_tighten:
            tighten_factor = max(0.3, 1.0 - (hold_pct - self.tighten_after_pct) / (1.0 - self.tighten_after_pct) * 0.7)
        return {"should_exit": bar_expired or time_expired, "hold_pct": hold_pct, "tighten_factor": tighten_factor, "should_tighten": should_tighten, "method": "time_based_stop"}

    def calculate(self, entry_price: float, side: str, base_stop_dist: float, bars_held: int = 0, seconds_held: float = 0.0) -> dict:
        result = self.should_exit(bars_held, seconds_held)
        adjusted_dist = base_stop_dist * result["tighten_factor"]
        if side == "long":
            stop_price = entry_price - adjusted_dist
        else:
            stop_price = entry_price + adjusted_dist
        result["stop_price"] = stop_price
        result["adjusted_distance"] = adjusted_dist
        return result
