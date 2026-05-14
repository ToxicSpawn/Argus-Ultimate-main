"""
Trend Following strategy for Argus-Ultimate v5.0.0.
Moving average alignment with simple regime confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class TrendSignal:
    signal: str
    score: float
    fast_ma: float
    slow_ma: float


class TrendFollowingStrategy:
    def __init__(self, fast_window: int = 12, slow_window: int = 48):
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate(self, prices: Sequence[float]) -> TrendSignal:
        if len(prices) < self.slow_window:
            return TrendSignal("hold", 0.0, 0.0, 0.0)

        fast_ma = sum(prices[-self.fast_window:]) / self.fast_window
        slow_ma = sum(prices[-self.slow_window:]) / self.slow_window
        diff = (fast_ma - slow_ma) / max(slow_ma, 1e-9)

        if diff > 0.002:
            return TrendSignal("buy", min(diff / 0.01, 1.0), fast_ma, slow_ma)
        if diff < -0.002:
            return TrendSignal("sell", min(abs(diff) / 0.01, 1.0), fast_ma, slow_ma)
        return TrendSignal("hold", 0.0, fast_ma, slow_ma)
