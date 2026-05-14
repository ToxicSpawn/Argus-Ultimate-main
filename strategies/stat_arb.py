"""
Statistical Arbitrage strategy for Argus-Ultimate v5.0.0.
Pairs spread z-score signal scaffold for co-integrated instruments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math


@dataclass
class StatArbSignal:
    signal: str
    score: float
    spread: float
    zscore: float


class StatisticalArbitrageStrategy:
    def __init__(self, lookback: int = 60, entry_z: float = 2.0, hedge_ratio: float = 1.0):
        self.lookback = lookback
        self.entry_z = entry_z
        self.hedge_ratio = hedge_ratio

    def generate(self, x: Sequence[float], y: Sequence[float]) -> StatArbSignal:
        if len(x) < self.lookback or len(y) < self.lookback:
            return StatArbSignal("hold", 0.0, 0.0, 0.0)

        spreads = [a - self.hedge_ratio * b for a, b in zip(x[-self.lookback:], y[-self.lookback:])]
        spread = spreads[-1]
        mean_spread = sum(spreads) / len(spreads)
        variance = sum((s - mean_spread) ** 2 for s in spreads) / max(len(spreads), 1)
        std = math.sqrt(max(variance, 1e-9))
        z = (spread - mean_spread) / std

        if z > self.entry_z:
            return StatArbSignal("sell_spread", min(abs(z) / (self.entry_z * 2), 1.0), spread, z)
        if z < -self.entry_z:
            return StatArbSignal("buy_spread", min(abs(z) / (self.entry_z * 2), 1.0), spread, z)
        return StatArbSignal("hold", 0.0, spread, z)
