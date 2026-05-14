"""
Pairs Trading strategy for Argus-Ultimate v5.0.0.
Uses normalized spread divergence and convergence exit signal scaffold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math


@dataclass
class PairsSignal:
    signal: str
    score: float
    ratio: float
    zscore: float


class PairsTradingStrategy:
    def __init__(self, lookback: int = 80, entry_z: float = 1.8, exit_z: float = 0.5):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate(self, a: Sequence[float], b: Sequence[float]) -> PairsSignal:
        if len(a) < self.lookback or len(b) < self.lookback:
            return PairsSignal("hold", 0.0, 0.0, 0.0)

        ratios = [x / y for x, y in zip(a[-self.lookback:], b[-self.lookback:]) if y != 0]
        if not ratios:
            return PairsSignal("hold", 0.0, 0.0, 0.0)

        ratio = ratios[-1]
        mean_ratio = sum(ratios) / len(ratios)
        variance = sum((r - mean_ratio) ** 2 for r in ratios) / max(len(ratios), 1)
        std = math.sqrt(max(variance, 1e-9))
        z = (ratio - mean_ratio) / std

        if z > self.entry_z:
            return PairsSignal("short_a_long_b", min(abs(z) / (self.entry_z * 2), 1.0), ratio, z)
        if z < -self.entry_z:
            return PairsSignal("long_a_short_b", min(abs(z) / (self.entry_z * 2), 1.0), ratio, z)
        if abs(z) < self.exit_z:
            return PairsSignal("exit", 0.3, ratio, z)
        return PairsSignal("hold", 0.0, ratio, z)
