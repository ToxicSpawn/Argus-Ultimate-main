"""
Volatility Breakout strategy for Argus-Ultimate v5.0.0.
ATR-style breakout sensitivity scaffold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class VolatilityBreakoutSignal:
    signal: str
    score: float
    breakout_up: float
    breakout_down: float


class VolatilityBreakoutStrategy:
    def __init__(self, lookback: int = 20, multiplier: float = 1.5):
        self.lookback = lookback
        self.multiplier = multiplier

    def generate(self, highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]) -> VolatilityBreakoutSignal:
        if len(closes) < self.lookback + 1:
            return VolatilityBreakoutSignal("hold", 0.0, 0.0, 0.0)

        ranges = []
        for i in range(-self.lookback, 0):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            ranges.append(tr)
        atr = sum(ranges) / len(ranges)
        breakout_up = closes[-2] + atr * self.multiplier
        breakout_down = closes[-2] - atr * self.multiplier
        price = closes[-1]

        if price > breakout_up:
            return VolatilityBreakoutSignal("buy", min((price - breakout_up) / max(atr, 1e-9), 1.0), breakout_up, breakout_down)
        if price < breakout_down:
            return VolatilityBreakoutSignal("sell", min((breakout_up - price) / max(atr, 1e-9), 1.0), breakout_up, breakout_down)
        return VolatilityBreakoutSignal("hold", 0.0, breakout_up, breakout_down)
