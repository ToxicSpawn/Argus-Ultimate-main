"""Push 75 — MomentumStrategy: EMA crossover + RSI filter.

Logic:
  - Fast EMA crosses above slow EMA + RSI > rsi_bull  => LONG signal
  - Fast EMA crosses below slow EMA + RSI < rsi_bear  => SHORT signal
  - Signal strength = normalised RSI distance from 50
  - ATR-based stop loss (atr_mult * ATR below entry)
  - Take profit = 2x ATR above entry (RR = 2.0)

Default params:
  fast_period=9, slow_period=21, rsi_period=14,
  atr_period=14, atr_mult=1.5,
  rsi_bull=55, rsi_bear=45
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional

from core.strategy.base_strategy import BaseStrategy, StrategyConfig
from core.strategy.signal import Signal, SignalSide


def _ema(values: list, period: int) -> float:
    """Exponential moving average of the last `period` values."""
    if len(values) < period:
        return sum(values) / len(values) if values else 0.0
    k = 2.0 / (period + 1)
    ema = values[-period]
    for v in values[-period + 1:]:
        ema = v * k + ema * (1 - k)
    return ema


def _rsi(prices: list, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [max(d, 0) for d in deltas[-period:]]
    losses = [abs(min(d, 0)) for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(highs: list, lows: list, closes: list, period: int = 14) -> float:
    if len(closes) < 2:
        return 0.0
    trs = []
    for i in range(1, min(period + 1, len(closes))):
        tr = max(
            highs[-i] - lows[-i],
            abs(highs[-i] - closes[-i - 1]),
            abs(lows[-i] - closes[-i - 1]),
        )
        trs.append(tr)
    return sum(trs) / len(trs) if trs else 0.0


class MomentumStrategy(BaseStrategy):
    """EMA crossover momentum strategy with RSI + ATR filters."""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        p = config.params
        self.fast_period = int(p.get("fast_period", 9))
        self.slow_period = int(p.get("slow_period", 21))
        self.rsi_period  = int(p.get("rsi_period",  14))
        self.atr_period  = int(p.get("atr_period",  14))
        self.atr_mult    = float(p.get("atr_mult",   1.5))
        self.rsi_bull    = float(p.get("rsi_bull",   55.0))
        self.rsi_bear    = float(p.get("rsi_bear",   45.0))

        maxlen = max(self.slow_period, self.rsi_period, self.atr_period) * 3
        self._prices: Deque[float] = deque(maxlen=maxlen)
        self._prev_fast_ema: Optional[float] = None
        self._prev_slow_ema: Optional[float] = None

    def tick(
        self,
        price: float,
        volume: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> Optional[Signal]:
        self._prices.append(price)
        prices = list(self._prices)

        if len(prices) < self.slow_period + 1:
            return None

        fast_ema = _ema(prices, self.fast_period)
        slow_ema = _ema(prices, self.slow_period)
        rsi      = _rsi(prices, self.rsi_period)
        atr      = _atr(prices, prices, prices, self.atr_period)  # simplified: OHLC=close

        signal = None

        # Bullish crossover
        if (self._prev_fast_ema is not None and
                self._prev_fast_ema <= self._prev_slow_ema and
                fast_ema > slow_ema and
                rsi > self.rsi_bull):
            strength = min((rsi - 50) / 50, 1.0)
            sl = price - self.atr_mult * atr if atr > 0 else None
            tp = price + 2 * self.atr_mult * atr if atr > 0 else None
            signal = self._make_signal(
                SignalSide.LONG, strength,
                stop_loss=sl, take_profit=tp,
                fast_ema=fast_ema, slow_ema=slow_ema, rsi=rsi,
            )

        # Bearish crossover
        elif (self._prev_fast_ema is not None and
              self._prev_fast_ema >= self._prev_slow_ema and
              fast_ema < slow_ema and
              rsi < self.rsi_bear):
            strength = min((50 - rsi) / 50, 1.0)
            sl = price + self.atr_mult * atr if atr > 0 else None
            tp = price - 2 * self.atr_mult * atr if atr > 0 else None
            signal = self._make_signal(
                SignalSide.SHORT, strength,
                stop_loss=sl, take_profit=tp,
                fast_ema=fast_ema, slow_ema=slow_ema, rsi=rsi,
            )

        self._prev_fast_ema = fast_ema
        self._prev_slow_ema = slow_ema
        return signal
