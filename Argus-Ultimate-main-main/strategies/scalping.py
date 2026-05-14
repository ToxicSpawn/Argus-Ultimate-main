"""
Scalping strategy for Argus-Ultimate v5.0.0.
Microstructure-aware short-horizon impulse detection.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScalpingSignal:
    signal: str
    score: float
    imbalance: float
    spread_bps: float


class ScalpingStrategy:
    def __init__(self, imbalance_threshold: float = 0.15, max_spread_bps: float = 3.0):
        self.imbalance_threshold = imbalance_threshold
        self.max_spread_bps = max_spread_bps

    def generate(self, imbalance: float, spread_bps: float) -> ScalpingSignal:
        if spread_bps > self.max_spread_bps:
            return ScalpingSignal("hold", 0.0, imbalance, spread_bps)
        if imbalance > self.imbalance_threshold:
            return ScalpingSignal("buy", min(imbalance / 0.5, 1.0), imbalance, spread_bps)
        if imbalance < -self.imbalance_threshold:
            return ScalpingSignal("sell", min(abs(imbalance) / 0.5, 1.0), imbalance, spread_bps)
        return ScalpingSignal("hold", 0.0, imbalance, spread_bps)
