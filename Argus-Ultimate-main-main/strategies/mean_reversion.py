"""
Mean Reversion strategy for Argus-Ultimate v5.0.0.
Uses z-score distance from rolling mean with volatility-aware thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import math


@dataclass
class MeanReversionConfig:
    """Configuration for mean reversion strategy.
    
    Attributes
    ----------
    lookback : int
        Lookback period for rolling mean (default 50)
    base_threshold : float
        Base z-score threshold for signals (default 1.5)
    vol_scale : float
        Volatility scaling factor (default 1.0)
    """
    lookback: int = 50
    base_threshold: float = 1.5
    vol_scale: float = 1.0


@dataclass
class MeanReversionSignal:
    signal: str
    score: float
    zscore: float
    upper_band: float
    lower_band: float


class MeanReversionStrategy:
    def __init__(self, lookback: int = 50, base_threshold: float = 1.5, vol_scale: float = 1.0):
        self.lookback = lookback
        self.base_threshold = base_threshold
        self.vol_scale = vol_scale

    def generate(self, prices, volatility: Optional[float] = None) -> MeanReversionSignal:
        if len(prices) < self.lookback:
            return MeanReversionSignal("hold", 0.0, 0.0, 0.0, 0.0)

        window = prices[-self.lookback:]
        mean_price = sum(window) / len(window)
        variance = sum((p - mean_price) ** 2 for p in window) / max(len(window), 1)
        std = math.sqrt(max(variance, 1e-9))
        z = (prices[-1] - mean_price) / std

        vol_adj = 1.0 + ((volatility or 0.0) * self.vol_scale)
        threshold = self.base_threshold * vol_adj

        if z < -threshold:
            return MeanReversionSignal("buy", min(abs(z) / threshold, 1.0), z, threshold, -threshold)
        if z > threshold:
            return MeanReversionSignal("sell", min(abs(z) / threshold, 1.0), z, threshold, -threshold)
        return MeanReversionSignal("hold", 0.0, z, threshold, -threshold)
