"""MultiTimeframeFeatures — Push 38.

Resamples 1-minute OHLCV candles to 5m / 15m / 1h / 4h timeframes and
computes a 24-feature vector used for:
  1. MTF directional bias signal fed into the SignalGateway as LLM_OVERLAY.
  2. Conviction boost on VOID_BREAKER when MTF agrees with matrix signal.
  3. Future: direct input to DeepLOB retraining pipeline.

Features per timeframe (6 features × 4 timeframes = 24 total)
--------------------------------------------------------------
  rsi_14          Relative Strength Index (14-period)
  ema_cross       EMA(9) - EMA(21) normalised by close price
  atr_ratio       ATR(14) / close  (volatility proxy)
  bb_width        (BB_upper - BB_lower) / BB_mid  (squeeze proxy)
  vol_ratio       current_volume / rolling_mean_volume(20)
  trend_slope     Linear regression slope of last 20 closes, normalised
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Candle column indices
_TS, _O, _H, _L, _C, _V = 0, 1, 2, 3, 4, 5

_TIMEFRAMES: Dict[str, int] = {
    "5m":  5,
    "15m": 15,
    "1h":  60,
    "4h":  240,
}


@dataclass
class MTFResult:
    """Output of MultiTimeframeFeatures.compute().

    Attributes
    ----------
    features:           Flat 24-float vector (6 features × 4 timeframes).
    timeframe_biases:   Per-timeframe directional bias [-1, 1].
    aggregate_bias:     Weighted average bias across timeframes.
    direction:          'long', 'short', or 'flat'.
    confidence:         abs(aggregate_bias) clipped to [0.1, 1.0].
    feature_names:      Human-readable feature labels.
    """

    features: List[float] = field(default_factory=list)
    timeframe_biases: Dict[str, float] = field(default_factory=dict)
    aggregate_bias: float = 0.0
    direction: str = "flat"
    confidence: float = 0.1
    feature_names: List[str] = field(default_factory=list)


class MultiTimeframeFeatures:
    """Computes multi-timeframe features from 1-minute OHLCV candles.

    Parameters
    ----------
    timeframes : dict mapping label → minutes (default: 5m/15m/1h/4h)
    tf_weights : per-timeframe weight for aggregate_bias (higher = more influence)
    """

    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "5m":  1.0,
        "15m": 1.5,
        "1h":  2.0,
        "4h":  2.5,
    }

    def __init__(
        self,
        timeframes: Optional[Dict[str, int]] = None,
        tf_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._timeframes = timeframes or dict(_TIMEFRAMES)
        self._weights = tf_weights or dict(self._DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, candles_1m: np.ndarray) -> MTFResult:
        """Compute MTF features from *candles_1m* (shape N×6, col order T/O/H/L/C/V).

        Returns an MTFResult. If insufficient data, returns a neutral result.
        """
        if candles_1m.shape[0] < 60:
            return MTFResult()  # not enough data

        features: List[float] = []
        feature_names: List[str] = []
        biases: Dict[str, float] = {}

        for tf_label, tf_minutes in self._timeframes.items():
            resampled = self._resample(candles_1m, tf_minutes)
            if resampled.shape[0] < 22:  # need at least 22 bars for EMA-21
                biases[tf_label] = 0.0
                features.extend([0.0] * 6)
                feature_names.extend(
                    [f"{tf_label}_{n}" for n in
                     ("rsi14", "ema_cross", "atr_ratio", "bb_width", "vol_ratio", "trend_slope")]
                )
                continue

            closes  = resampled[:, _C].astype(float)
            highs   = resampled[:, _H].astype(float)
            lows    = resampled[:, _L].astype(float)
            volumes = resampled[:, _V].astype(float)

            rsi     = self._rsi(closes, 14)
            ema9    = self._ema(closes, 9)
            ema21   = self._ema(closes, 21)
            atr     = self._atr(highs, lows, closes, 14)
            bb_w    = self._bb_width(closes, 20)
            vol_r   = self._vol_ratio(volumes, 20)
            slope   = self._trend_slope(closes, 20)

            last_close = closes[-1] if closes[-1] != 0.0 else 1.0
            ema_cross  = (ema9 - ema21) / last_close
            atr_ratio  = atr / last_close

            tf_features = [rsi / 100.0, ema_cross, atr_ratio, bb_w, vol_r, slope]
            features.extend(tf_features)
            feature_names.extend(
                [f"{tf_label}_{n}" for n in
                 ("rsi14", "ema_cross", "atr_ratio", "bb_width", "vol_ratio", "trend_slope")]
            )

            # Directional bias for this timeframe
            # Positive ema_cross + rsi > 50 + positive slope → bullish
            bias = (
                (1.0 if rsi > 55 else (-1.0 if rsi < 45 else 0.0)) * 0.4
                + np.sign(ema_cross) * 0.4
                + np.sign(slope) * 0.2
            )
            biases[tf_label] = float(np.clip(bias, -1.0, 1.0))

        # Weighted aggregate bias
        total_w = sum(self._weights.get(tf, 1.0) for tf in biases)
        agg_bias = (
            sum(biases[tf] * self._weights.get(tf, 1.0) for tf in biases) / total_w
            if total_w > 0 else 0.0
        )
        agg_bias = float(np.clip(agg_bias, -1.0, 1.0))

        direction = "long" if agg_bias > 0.15 else ("short" if agg_bias < -0.15 else "flat")
        confidence = float(np.clip(abs(agg_bias), 0.1, 1.0))

        return MTFResult(
            features=features,
            timeframe_biases=biases,
            aggregate_bias=agg_bias,
            direction=direction,
            confidence=confidence,
            feature_names=feature_names,
        )

    def agrees_with(self, mtf: MTFResult, matrix_direction: str) -> bool:
        """Return True if MTF aggregate bias agrees with *matrix_direction*."""
        if mtf.direction == "flat" or matrix_direction == "flat":
            return False
        return mtf.direction == matrix_direction

    # ------------------------------------------------------------------
    # Resampler
    # ------------------------------------------------------------------

    @staticmethod
    def _resample(candles_1m: np.ndarray, minutes: int) -> np.ndarray:
        """Aggregate 1-minute candles into *minutes*-period bars."""
        n = candles_1m.shape[0]
        n_bars = n // minutes
        if n_bars == 0:
            return np.empty((0, 6))

        # Trim to exact multiple
        trimmed = candles_1m[n - n_bars * minutes:]
        reshaped = trimmed.reshape(n_bars, minutes, 6)

        out = np.empty((n_bars, 6))
        out[:, _TS] = reshaped[:, 0,  _TS]   # open timestamp
        out[:, _O]  = reshaped[:, 0,  _O]    # open
        out[:, _H]  = reshaped[:, :,  _H].max(axis=1)
        out[:, _L]  = reshaped[:, :,  _L].min(axis=1)
        out[:, _C]  = reshaped[:, -1, _C]    # close
        out[:, _V]  = reshaped[:, :,  _V].sum(axis=1)
        return out

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-(period + 1):])
        gains  = deltas[deltas > 0].sum() / period
        losses = -deltas[deltas < 0].sum() / period
        if losses == 0:
            return 100.0
        rs = gains / losses
        return float(100.0 - 100.0 / (1.0 + rs))

    @staticmethod
    def _ema(closes: np.ndarray, period: int) -> float:
        if len(closes) < period:
            return float(closes[-1]) if len(closes) > 0 else 0.0
        k   = 2.0 / (period + 1)
        ema = float(closes[-period])
        for c in closes[-period + 1:]:
            ema = c * k + ema * (1 - k)
        return ema

    @staticmethod
    def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
             period: int = 14) -> float:
        if len(closes) < period + 1:
            return float((highs - lows).mean()) if len(highs) > 0 else 0.0
        tr_list = []
        for i in range(1, period + 1):
            idx = -(period + 1 - i)
            tr = max(
                highs[idx] - lows[idx],
                abs(highs[idx] - closes[idx - 1]),
                abs(lows[idx]  - closes[idx - 1]),
            )
            tr_list.append(tr)
        return float(np.mean(tr_list))

    @staticmethod
    def _bb_width(closes: np.ndarray, period: int = 20) -> float:
        if len(closes) < period:
            return 0.0
        window = closes[-period:]
        mid  = window.mean()
        std  = window.std()
        if mid == 0:
            return 0.0
        return float((2.0 * 2.0 * std) / mid)  # 2σ BB width / mid

    @staticmethod
    def _vol_ratio(volumes: np.ndarray, period: int = 20) -> float:
        if len(volumes) < period:
            return 1.0
        mean_vol = volumes[-period:-1].mean()
        if mean_vol == 0:
            return 1.0
        return float(np.clip(volumes[-1] / mean_vol, 0.0, 5.0))

    @staticmethod
    def _trend_slope(closes: np.ndarray, period: int = 20) -> float:
        if len(closes) < period:
            return 0.0
        y = closes[-period:]
        x = np.arange(period, dtype=float)
        # OLS slope normalised by mean close
        x_mean = x.mean()
        y_mean = y.mean()
        slope = (
            ((x - x_mean) * (y - y_mean)).sum()
            / ((x - x_mean) ** 2).sum()
        )
        return float(np.clip(slope / (y_mean if y_mean != 0 else 1.0), -1.0, 1.0))
