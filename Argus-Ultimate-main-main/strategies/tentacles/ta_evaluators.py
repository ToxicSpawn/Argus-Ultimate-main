"""
ta_evaluators.py — OctoBot-style TA evaluator tentacles.

Implements four independent TA evaluators, each inheriting BaseTentacle
and self-registering in TENTACLE_REGISTRY:

  RSIEvaluator      — RSI overbought/oversold signal
  MACDEvaluator     — MACD histogram momentum signal
  BollingerEvaluator— Bollinger Band %B mean-reversion signal
  StochEvaluator    — Stochastic %K/%D crossover signal

All return EvalResult with signal in [-1.0, 1.0]:
  +1.0 = strong buy
  -1.0 = strong sell
   0.0 = neutral
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from .base_tentacle import (
    BaseTentacle, EvalResult, TentacleType, register_tentacle,
    candles_close, candles_high, candles_low, ema, rsi,
)


# ---------------------------------------------------------------------------
# RSI Evaluator
# ---------------------------------------------------------------------------

@register_tentacle
class RSIEvaluator(BaseTentacle):
    """
    RSI overbought / oversold evaluator.

    Signal mapping:
      RSI <= oversold  -> +1.0 (buy)
      RSI >= overbought-> -1.0 (sell)
      Linear interpolation between thresholds -> [−1, +1]

    Config keys: rsi_period(14), oversold(30), overbought(70)
    """

    name = "RSIEvaluator"
    tentacle_type = TentacleType.TA_EVALUATOR
    version = "1.0.0"
    weight = 1.0

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._period     = int(self.config.get("rsi_period", 14))
        self._oversold   = float(self.config.get("oversold", 30))
        self._overbought = float(self.config.get("overbought", 70))

    def evaluate(self, candles: np.ndarray, **kwargs: Any) -> EvalResult:
        close = candles_close(candles)
        if len(close) < self._period + 1:
            return EvalResult(tentacle_name=self.name, signal=0.0, confidence=0.0)

        rsi_val = float(rsi(close, self._period)[-1])

        if rsi_val <= self._oversold:
            signal = 1.0
        elif rsi_val >= self._overbought:
            signal = -1.0
        else:
            mid = (self._oversold + self._overbought) / 2.0
            half = (self._overbought - self._oversold) / 2.0
            signal = -(rsi_val - mid) / half

        confidence = abs(signal) * 0.9 + 0.1
        return EvalResult(
            tentacle_name=self.name,
            signal=round(signal, 4),
            confidence=round(confidence, 4),
            metadata={"rsi": round(rsi_val, 2)},
        )


# ---------------------------------------------------------------------------
# MACD Evaluator
# ---------------------------------------------------------------------------

@register_tentacle
class MACDEvaluator(BaseTentacle):
    """
    MACD histogram momentum evaluator.

    Signal = tanh(histogram / scale) where scale is ATR-normalised.
    Positive histogram -> bullish momentum (+)
    Negative histogram -> bearish momentum (-)

    Config keys: fast(12), slow(26), signal_period(9), scale(0.002)
    """

    name = "MACDEvaluator"
    tentacle_type = TentacleType.TA_EVALUATOR
    version = "1.0.0"
    weight = 1.0

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._fast   = int(self.config.get("fast", 12))
        self._slow   = int(self.config.get("slow", 26))
        self._signal = int(self.config.get("signal_period", 9))
        self._scale  = float(self.config.get("scale", 0.002))

    def evaluate(self, candles: np.ndarray, **kwargs: Any) -> EvalResult:
        close = candles_close(candles)
        min_bars = self._slow + self._signal + 1
        if len(close) < min_bars:
            return EvalResult(tentacle_name=self.name, signal=0.0, confidence=0.0)

        ema_fast  = ema(close, self._fast)
        ema_slow  = ema(close, self._slow)
        macd_line = ema_fast - ema_slow
        sig_line  = ema(macd_line, self._signal)
        histogram = float(macd_line[-1] - sig_line[-1])

        # Normalise by current price
        price = float(close[-1])
        norm  = histogram / price if price > 0 else histogram
        signal = float(math.tanh(norm / self._scale))

        # Confirm with histogram direction
        hist_prev = float(macd_line[-2] - sig_line[-2])
        diverging = (histogram > 0 and histogram > hist_prev) or \
                    (histogram < 0 and histogram < hist_prev)
        confidence = 0.8 if diverging else 0.5

        return EvalResult(
            tentacle_name=self.name,
            signal=round(signal, 4),
            confidence=confidence,
            metadata={
                "macd": round(float(macd_line[-1]), 6),
                "signal_line": round(float(sig_line[-1]), 6),
                "histogram": round(histogram, 6),
            },
        )


# ---------------------------------------------------------------------------
# Bollinger Band Evaluator
# ---------------------------------------------------------------------------

@register_tentacle
class BollingerEvaluator(BaseTentacle):
    """
    Bollinger Band %B mean-reversion evaluator.

    %B = (price - lower) / (upper - lower)
    %B < 0  -> price below lower band -> strong buy (+1)
    %B > 1  -> price above upper band -> strong sell (-1)
    %B = 0.5 -> at midband -> neutral

    Signal = 1 - 2 * %B, clamped to [-1, 1]

    Config keys: period(20), std_dev(2.0)
    """

    name = "BollingerEvaluator"
    tentacle_type = TentacleType.TA_EVALUATOR
    version = "1.0.0"
    weight = 1.0

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._period  = int(self.config.get("period", 20))
        self._std_dev = float(self.config.get("std_dev", 2.0))

    def evaluate(self, candles: np.ndarray, **kwargs: Any) -> EvalResult:
        close = candles_close(candles)
        if len(close) < self._period:
            return EvalResult(tentacle_name=self.name, signal=0.0, confidence=0.0)

        window = close[-self._period:]
        mid    = float(np.mean(window))
        std    = float(np.std(window, ddof=1))

        if std == 0:
            return EvalResult(tentacle_name=self.name, signal=0.0, confidence=0.0)

        upper  = mid + self._std_dev * std
        lower  = mid - self._std_dev * std
        price  = float(close[-1])
        band_w = upper - lower

        pct_b  = (price - lower) / band_w if band_w > 0 else 0.5
        signal = max(-1.0, min(1.0, 1.0 - 2.0 * pct_b))

        # Squeeze detection: narrow bands reduce confidence
        avg_bw = float(np.mean([
            (np.mean(close[i-self._period:i]) + self._std_dev * np.std(close[i-self._period:i], ddof=1)) -
            (np.mean(close[i-self._period:i]) - self._std_dev * np.std(close[i-self._period:i], ddof=1))
            for i in range(self._period, len(close))
        ])) if len(close) > self._period * 2 else band_w
        squeeze = band_w < avg_bw * 0.5
        confidence = 0.5 if squeeze else min(0.95, abs(signal) + 0.3)

        return EvalResult(
            tentacle_name=self.name,
            signal=round(signal, 4),
            confidence=round(confidence, 4),
            metadata={
                "pct_b": round(pct_b, 4),
                "upper": round(upper, 4),
                "lower": round(lower, 4),
                "mid": round(mid, 4),
                "squeeze": squeeze,
            },
        )


# ---------------------------------------------------------------------------
# Stochastic Evaluator
# ---------------------------------------------------------------------------

@register_tentacle
class StochEvaluator(BaseTentacle):
    """
    Stochastic %K/%D crossover evaluator.

    %K = (close - lowest_low) / (highest_high - lowest_low) * 100
    %D = SMA(%K, d_period)

    Signal:
      %K crosses above %D in oversold zone (<20) -> buy (+)
      %K crosses below %D in overbought zone (>80) -> sell (-)
      Magnitude proportional to distance from midline (50)

    Config keys: k_period(14), d_period(3), oversold(20), overbought(80)
    """

    name = "StochEvaluator"
    tentacle_type = TentacleType.TA_EVALUATOR
    version = "1.0.0"
    weight = 0.9

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._k_period   = int(self.config.get("k_period", 14))
        self._d_period   = int(self.config.get("d_period", 3))
        self._oversold   = float(self.config.get("oversold", 20))
        self._overbought = float(self.config.get("overbought", 80))

    def evaluate(self, candles: np.ndarray, **kwargs: Any) -> EvalResult:
        close = candles_close(candles)
        high  = candles_high(candles)
        low   = candles_low(candles)
        min_bars = self._k_period + self._d_period + 1

        if len(close) < min_bars:
            return EvalResult(tentacle_name=self.name, signal=0.0, confidence=0.0)

        # Compute %K
        k_vals = np.zeros(len(close))
        for i in range(self._k_period - 1, len(close)):
            hh = float(np.max(high[i - self._k_period + 1:i + 1]))
            ll = float(np.min(low[i  - self._k_period + 1:i + 1]))
            rng = hh - ll
            k_vals[i] = ((close[i] - ll) / rng * 100.0) if rng > 0 else 50.0

        # %D = SMA of %K
        d_vals = np.convolve(
            k_vals[self._k_period - 1:],
            np.ones(self._d_period) / self._d_period,
            mode="valid",
        )

        k_now  = float(k_vals[-1])
        k_prev = float(k_vals[-2])
        d_now  = float(d_vals[-1])
        d_prev = float(d_vals[-2]) if len(d_vals) >= 2 else d_now

        crossover_up   = k_prev <= d_prev and k_now > d_now
        crossover_down = k_prev >= d_prev and k_now < d_now

        if crossover_up and k_now < self._overbought:
            # Stronger signal the more oversold
            signal = min(1.0, (self._overbought - k_now) / (self._overbought - self._oversold))
            signal = max(0.1, signal)
        elif crossover_down and k_now > self._oversold:
            signal = -min(1.0, (k_now - self._oversold) / (self._overbought - self._oversold))
            signal = min(-0.1, signal)
        else:
            # No crossover — linear position signal
            signal = (50.0 - k_now) / 50.0
            signal = max(-0.5, min(0.5, signal))

        confidence = 0.85 if (crossover_up or crossover_down) else 0.4

        return EvalResult(
            tentacle_name=self.name,
            signal=round(signal, 4),
            confidence=round(confidence, 4),
            metadata={
                "k": round(k_now, 2),
                "d": round(d_now, 2),
                "crossover_up": crossover_up,
                "crossover_down": crossover_down,
            },
        )
