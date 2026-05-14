"""
VWAP Reversion strategy for Argus-Ultimate v5.0.0.
Trades deviations from session VWAP with mean-reverting bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class VWAPSignal:
    signal: str
    score: float
    vwap: float
    deviation_pct: float


class VWAPReversionStrategy:
    def __init__(self, threshold_pct: float = 0.0025):
        self.threshold_pct = threshold_pct

    def generate(self, prices: Sequence[float], volumes: Sequence[float]) -> VWAPSignal:
        if not prices or not volumes or len(prices) != len(volumes):
            return VWAPSignal("hold", 0.0, 0.0, 0.0)

        total_volume = sum(volumes)
        if total_volume <= 0:
            return VWAPSignal("hold", 0.0, 0.0, 0.0)

        vwap = sum(p * v for p, v in zip(prices, volumes)) / total_volume
        deviation = (prices[-1] - vwap) / vwap

        if deviation > self.threshold_pct:
            return VWAPSignal("sell", min(abs(deviation) / (self.threshold_pct * 3), 1.0), vwap, deviation)
        if deviation < -self.threshold_pct:
            return VWAPSignal("buy", min(abs(deviation) / (self.threshold_pct * 3), 1.0), vwap, deviation)
        return VWAPSignal("hold", 0.0, vwap, deviation)
