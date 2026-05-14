"""
Momentum strategy for Argus-Ultimate v5.0.0.
Uses multi-horizon returns and acceleration to score trend persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class MomentumConfig:
    """Configuration for momentum strategy.
    
    Attributes
    ----------
    short_window : int
        Short-term lookback window (default 10)
    long_window : int
        Long-term lookback window (default 40)
    min_strength : float
        Minimum signal strength to trigger a trade (default 0.002)
    acceleration_threshold : float
        Minimum acceleration to confirm momentum (default 0.0)
    """
    short_window: int = 10
    long_window: int = 40
    min_strength: float = 0.002
    acceleration_threshold: float = 0.0


@dataclass
class MomentumSignal:
    signal: str
    score: float
    short_return: float
    long_return: float
    acceleration: float


class MomentumStrategy:
    def __init__(self, short_window: int = 10, long_window: int = 40, min_strength: float = 0.002):
        self.short_window = short_window
        self.long_window = long_window
        self.min_strength = min_strength

    def generate(self, prices: Sequence[float]) -> MomentumSignal:
        if len(prices) <= self.long_window:
            return MomentumSignal("hold", 0.0, 0.0, 0.0, 0.0)

        short_ret = (prices[-1] - prices[-self.short_window]) / prices[-self.short_window]
        long_ret = (prices[-1] - prices[-self.long_window]) / prices[-self.long_window]
        prev_short_ret = (prices[-2] - prices[-self.short_window - 1]) / prices[-self.short_window - 1]
        acceleration = short_ret - prev_short_ret
        score = (0.6 * short_ret) + (0.3 * long_ret) + (0.1 * acceleration)

        if score > self.min_strength:
            return MomentumSignal("buy", min(score / (self.min_strength * 4), 1.0), short_ret, long_ret, acceleration)
        if score < -self.min_strength:
            return MomentumSignal("sell", min(abs(score) / (self.min_strength * 4), 1.0), short_ret, long_ret, acceleration)
        return MomentumSignal("hold", 0.0, short_ret, long_ret, acceleration)
