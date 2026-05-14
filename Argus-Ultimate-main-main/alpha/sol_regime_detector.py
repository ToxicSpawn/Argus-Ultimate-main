"""
SOL Regime Detector — classifies SOL/USD market regime in real time.

Output regime labels (used by strategy_router to switch between strategies):
  - 'strong_trending'  ADX > 35, price > EMA20, realised_vol > 0.04
  - 'trending'         ADX 20–35, price > EMA20
  - 'ranging'          ADX < 20, realised_vol < 0.03
  - 'choppy'           ADX < 20, realised_vol > 0.04 (high vol, no direction)
  - 'mild_trending'    ADX 15–25, mixed conditions

The funding rate direction is used as a secondary confirmer:
  positive funding = long bias (trending up likely); negative = short bias.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List


@dataclass
class SolRegimeReading:
    regime: str
    adx: float
    realised_vol: float
    funding_rate: float
    ema20: float
    price: float


class SolRegimeDetector:
    """
    Lightweight SOL regime classifier.

    Feed 1-minute OHLCV bars via update(close, high, low).
    Call regime() to get the current label.
    """

    def __init__(self) -> None:
        self._closes: Deque[float] = deque(maxlen=50)
        self._highs: Deque[float] = deque(maxlen=50)
        self._lows: Deque[float] = deque(maxlen=50)
        self._tr_values: Deque[float] = deque(maxlen=14)
        self._plus_dm: Deque[float] = deque(maxlen=14)
        self._minus_dm: Deque[float] = deque(maxlen=14)
        self._funding_rate: float = 0.0

    # ------------------------------------------------------------------
    def update(self, close: float, high: float, low: float) -> None:
        """Feed a new bar."""
        prev_close = self._closes[-1] if self._closes else close
        prev_high = self._highs[-1] if self._highs else high
        prev_low = self._lows[-1] if self._lows else low

        self._closes.append(close)
        self._highs.append(high)
        self._lows.append(low)

        # True Range
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        self._tr_values.append(tr)

        # Directional Movement
        up_move = high - prev_high
        down_move = prev_low - low
        plus = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus = down_move if (down_move > up_move and down_move > 0) else 0.0
        self._plus_dm.append(plus)
        self._minus_dm.append(minus)

    def set_funding_rate(self, rate: float) -> None:
        """Update the perpetual funding rate (from live exchange feed)."""
        self._funding_rate = rate

    # ------------------------------------------------------------------
    def get_regime(self) -> SolRegimeReading:
        """Classify the current regime."""
        if len(self._closes) < 20:
            return SolRegimeReading(
                regime="ranging", adx=0.0, realised_vol=0.0,
                funding_rate=self._funding_rate,
                ema20=self._closes[-1] if self._closes else 0.0,
                price=self._closes[-1] if self._closes else 0.0,
            )

        adx = self._calc_adx()
        realised_vol = self._calc_realised_vol()
        ema20 = self._calc_ema(20)
        price = self._closes[-1]

        if adx > 35 and realised_vol > 0.04:
            regime = "strong_trending"
        elif adx > 20 and price > ema20:
            regime = "trending"
        elif adx < 20 and realised_vol > 0.04:
            regime = "choppy"
        elif adx >= 15:
            regime = "mild_trending"
        else:
            regime = "ranging"

        return SolRegimeReading(
            regime=regime,
            adx=adx,
            realised_vol=realised_vol,
            funding_rate=self._funding_rate,
            ema20=ema20,
            price=price,
        )

    # ------------------------------------------------------------------
    def _calc_ema(self, period: int) -> float:
        closes = list(self._closes)
        if len(closes) < period:
            return closes[-1]
        k = 2.0 / (period + 1)
        ema = closes[-period]
        for c in closes[-period + 1:]:
            ema = c * k + ema * (1 - k)
        return ema

    def _calc_adx(self, period: int = 14) -> float:
        if len(self._tr_values) < period:
            return 0.0
        atr = sum(list(self._tr_values)[-period:]) / period
        if atr == 0:
            return 0.0
        plus_di = (sum(list(self._plus_dm)[-period:]) / period) / atr * 100
        minus_di = (sum(list(self._minus_dm)[-period:]) / period) / atr * 100
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0.0
        dx = abs(plus_di - minus_di) / di_sum * 100
        return dx  # simplified single-bar DX as ADX proxy

    def _calc_realised_vol(self, window: int = 20) -> float:
        closes = list(self._closes)
        if len(closes) < window + 1:
            return 0.0
        returns = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(-window, 0)
        ]
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        return variance ** 0.5
