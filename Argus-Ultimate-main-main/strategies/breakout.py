"""
Breakout strategy for Argus-Ultimate v5.0.0.
Detects range expansion with confirmation buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class BreakoutSignal:
    signal: str
    score: float
    breakout_level: float
    breakdown_level: float
    price: float


class BreakoutStrategy:
    def __init__(self, lookback: int = 30, buffer_pct: float = 0.0015):
        self.lookback = lookback
        self.buffer_pct = buffer_pct

    def generate(self, highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]) -> BreakoutSignal:
        if len(closes) < self.lookback:
            return BreakoutSignal("hold", 0.0, 0.0, 0.0, 0.0)

        breakout_level = max(highs[-self.lookback:])
        breakdown_level = min(lows[-self.lookback:])
        price = closes[-1]

        if price > breakout_level * (1 + self.buffer_pct):
            score = min((price - breakout_level) / max(breakout_level * self.buffer_pct, 1e-9), 1.0)
            return BreakoutSignal("buy", score, breakout_level, breakdown_level, price)
        if price < breakdown_level * (1 - self.buffer_pct):
            score = min((breakdown_level - price) / max(breakdown_level * self.buffer_pct, 1e-9), 1.0)
            return BreakoutSignal("sell", score, breakout_level, breakdown_level, price)
        return BreakoutSignal("hold", 0.0, breakout_level, breakdown_level, price)
