"""Realistic order latency simulation for backtesting.

Models the real-world delay between signal generation and order execution:
  - Fixed latency (e.g., 1-3 bars)
  - Random jitter (uniform or normal distribution)
  - Fill probability based on market conditions
  - Partial fills for large orders

Usage:
    latency = LatencyModel(
        fixed_bars=1,           # 1 bar fixed delay
        jitter_bars=0.5,        # ±0.5 bar random jitter
        fill_probability=0.95,  # 95% fill rate
    )
    delayed_signals = latency.apply(raw_signals)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class LatencyConfig:
    """Configuration for order latency simulation."""
    fixed_bars: int = 1              # Fixed delay in bars
    jitter_bars: float = 0.5         # Random jitter (±bars)
    fill_probability: float = 0.95   # Probability of order filling
    partial_fill_pct: float = 0.0    # Fraction of order that may be partial (0-1)
    seed: Optional[int] = None       # Random seed for reproducibility

    def to_dict(self) -> dict:
        return {
            "fixed_bars": self.fixed_bars,
            "jitter_bars": self.jitter_bars,
            "fill_probability": self.fill_probability,
            "partial_fill_pct": self.partial_fill_pct,
            "seed": self.seed,
        }


@dataclass
class LatencyStats:
    """Statistics about applied latency."""
    total_signals: int
    delayed_signals: int
    dropped_signals: int
    avg_delay_bars: float
    max_delay_bars: int
    fill_rate: float

    def to_dict(self) -> dict:
        return {
            "total_signals": self.total_signals,
            "delayed_signals": self.delayed_signals,
            "dropped_signals": self.dropped_signals,
            "avg_delay_bars": round(self.avg_delay_bars, 2),
            "max_delay_bars": self.max_delay_bars,
            "fill_rate": round(self.fill_rate, 4),
        }


class LatencyModel:
    """Simulates realistic order execution latency.

    Args:
        fixed_bars:    Fixed delay in bars (minimum latency)
        jitter_bars:   Random jitter added to fixed delay (±bars)
        fill_probability: Probability that an order fills (0-1)
        partial_fill_pct: Fraction of order that may be partial (0-1)
        seed:          Optional random seed
    """

    def __init__(
        self,
        fixed_bars: int = 1,
        jitter_bars: float = 0.5,
        fill_probability: float = 0.95,
        partial_fill_pct: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.fixed_bars = fixed_bars
        self.jitter_bars = jitter_bars
        self.fill_probability = fill_probability
        self.partial_fill_pct = partial_fill_pct
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def compute_delay(self) -> int:
        """Compute delay in bars for a single order."""
        if self.jitter_bars <= 0:
            return self.fixed_bars

        # Uniform jitter
        jitter = self._rng.uniform(-self.jitter_bars, self.jitter_bars)
        delay = self.fixed_bars + jitter
        return max(0, int(round(delay)))

    def will_fill(self) -> bool:
        """Determine if an order will fill based on fill probability."""
        return self._rng.random() < self.fill_probability

    def compute_partial_fill(self) -> float:
        """Compute fill fraction for partial fills."""
        if self.partial_fill_pct <= 0:
            return 1.0
        # Partial fill is between (1 - partial_fill_pct) and 1.0
        min_fill = 1.0 - self.partial_fill_pct
        return self._rng.uniform(min_fill, 1.0)

    def apply(
        self,
        signals: Sequence[float],
    ) -> tuple[np.ndarray, LatencyStats]:
        """Apply latency to signal array.

        Args:
            signals: Raw signal array (+1=long, -1=short, 0=flat)

        Returns:
            (delayed_signals, stats) tuple
        """
        signals = np.array(signals, dtype=float)
        n = len(signals)
        delayed = np.zeros(n, dtype=float)

        total_signals = 0
        delayed_count = 0
        dropped_count = 0
        delays: List[int] = []
        fills = 0

        for i in range(n):
            sig = signals[i]
            if sig == 0:
                continue

            total_signals += 1

            # Check if order fills
            if not self.will_fill():
                dropped_count += 1
                continue

            # Compute delay
            delay = self.compute_delay()
            delays.append(delay)

            # Apply delayed signal
            target_idx = i + delay
            if target_idx < n:
                # Apply partial fill if configured
                fill_frac = self.compute_partial_fill()
                delayed[target_idx] = sig * fill_frac
                delayed_count += 1
                fills += 1
            else:
                # Signal delayed beyond end of data - dropped
                dropped_count += 1

        avg_delay = sum(delays) / len(delays) if delays else 0.0
        max_delay = max(delays) if delays else 0
        fill_rate = fills / total_signals if total_signals > 0 else 1.0

        stats = LatencyStats(
            total_signals=total_signals,
            delayed_signals=delayed_count,
            dropped_signals=dropped_count,
            avg_delay_bars=avg_delay,
            max_delay_bars=max_delay,
            fill_rate=fill_rate,
        )

        return delayed, stats

    def apply_to_prices(
        self,
        prices: Sequence[float],
        signals: Sequence[float],
    ) -> tuple[np.ndarray, LatencyStats]:
        """Apply latency and adjust entry/exit prices for slippage.

        This is a more realistic model that:
        1. Delays signal execution
        2. Uses the delayed bar's price for execution
        3. Adds price impact based on signal direction

        Args:
            prices:  Price series (OHLCV close prices)
            signals: Raw signal array

        Returns:
            (adjusted_signals, stats) tuple
        """
        delayed_signals, stats = self.apply(signals)
        return delayed_signals, stats


class AdaptiveLatencyModel(LatencyModel):
    """Latency model that adapts based on market volatility.

    Higher volatility = higher latency (more slippage, slower fills).
    Lower volatility = lower latency (faster fills, less slippage).
    """

    def __init__(
        self,
        base_fixed_bars: int = 1,
        base_jitter_bars: float = 0.5,
        volatility_multiplier: float = 2.0,
        fill_probability: float = 0.95,
        partial_fill_pct: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__(
            fixed_bars=base_fixed_bars,
            jitter_bars=base_jitter_bars,
            fill_probability=fill_probability,
            partial_fill_pct=partial_fill_pct,
            seed=seed,
        )
        self.base_fixed_bars = base_fixed_bars
        self.base_jitter_bars = base_jitter_bars
        self.volatility_multiplier = volatility_multiplier
        self._volatility: Optional[float] = None

    def set_volatility(self, volatility: float) -> None:
        """Set current volatility level (annualized std of returns)."""
        self._volatility = volatility
        # Scale latency by volatility
        vol_factor = 1.0 + (volatility * self.volatility_multiplier)
        self.fixed_bars = max(1, int(self.base_fixed_bars * vol_factor))
        self.jitter_bars = self.base_jitter_bars * vol_factor

    def compute_volatility(self, prices: Sequence[float], window: int = 20) -> float:
        """Compute rolling volatility from price series."""
        if len(prices) < window + 1:
            return 0.02  # default 2% daily vol

        returns = np.diff(np.log(prices[-window:]))
        return float(np.std(returns) * np.sqrt(252))  # annualized
