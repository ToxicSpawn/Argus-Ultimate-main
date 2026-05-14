"""Push 75 — MeanReversionStrategy: Bollinger Band z-score + RSI extreme.

Logic:
  - Price touches lower BB (z < -band_mult) + RSI < rsi_oversold => LONG
  - Price touches upper BB (z > +band_mult) + RSI > rsi_overbought => SHORT
  - Flat signal when price returns to mean (|z| < 0.5)
  - Signal strength = normalised z-score distance from band
  - Stop loss = 1 std below/above entry

Default params:
  bb_period=20, band_mult=2.0, rsi_period=14,
  rsi_oversold=35, rsi_overbought=65
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional

from core.strategy.base_strategy import BaseStrategy, StrategyConfig
from core.strategy.signal import Signal, SignalSide


class MeanReversionStrategy(BaseStrategy):
    """Bollinger Band mean-reversion strategy."""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        p = config.params
        self.bb_period      = int(p.get("bb_period",      20))
        self.band_mult      = float(p.get("band_mult",    2.0))
        self.rsi_period     = int(p.get("rsi_period",     14))
        self.rsi_oversold   = float(p.get("rsi_oversold", 35.0))
        self.rsi_overbought = float(p.get("rsi_overbought", 65.0))

        maxlen = max(self.bb_period, self.rsi_period) * 3
        self._prices: Deque[float] = deque(maxlen=maxlen)
        self._in_long  = False
        self._in_short = False

    def _bb(self, prices: list) -> tuple[float, float, float]:
        """Return (mean, upper, lower) Bollinger Bands."""
        window = prices[-self.bb_period:]
        mean   = sum(window) / len(window)
        std    = math.sqrt(sum((x - mean) ** 2 for x in window) / len(window))
        return mean, mean + self.band_mult * std, mean - self.band_mult * std

    def _rsi(self, prices: list) -> float:
        if len(prices) < self.rsi_period + 1:
            return 50.0
        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains  = [max(d, 0)   for d in deltas[-self.rsi_period:]]
        losses = [abs(min(d, 0)) for d in deltas[-self.rsi_period:]]
        ag = sum(gains)  / self.rsi_period
        al = sum(losses) / self.rsi_period
        if al == 0:
            return 100.0
        return 100 - 100 / (1 + ag / al)

    def tick(
        self,
        price: float,
        volume: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> Optional[Signal]:
        self._prices.append(price)
        prices = list(self._prices)

        if len(prices) < self.bb_period:
            return None

        mean, upper, lower = self._bb(prices)
        rsi = self._rsi(prices)
        std = (upper - mean) / self.band_mult if self.band_mult else 0
        z   = (price - mean) / std if std > 0 else 0

        # Exit: price returns to mean
        if abs(z) < 0.5:
            if self._in_long or self._in_short:
                self._in_long = self._in_short = False
                return self._make_signal(SignalSide.FLAT, strength=1.0)

        # Long: oversold at lower band
        if z < -self.band_mult and rsi < self.rsi_oversold and not self._in_long:
            strength = min(abs(z) / (self.band_mult * 2), 1.0)
            sl = price - std
            tp = mean
            self._in_long = True
            return self._make_signal(
                SignalSide.LONG, strength,
                stop_loss=sl, take_profit=tp,
                z_score=z, rsi=rsi,
            )

        # Short: overbought at upper band
        if z > self.band_mult and rsi > self.rsi_overbought and not self._in_short:
            strength = min(abs(z) / (self.band_mult * 2), 1.0)
            sl = price + std
            tp = mean
            self._in_short = True
            return self._make_signal(
                SignalSide.SHORT, strength,
                stop_loss=sl, take_profit=tp,
                z_score=z, rsi=rsi,
            )

        return None
