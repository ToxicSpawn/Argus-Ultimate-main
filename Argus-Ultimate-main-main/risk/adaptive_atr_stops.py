"""Batch 1 — Adaptive ATR-based stop-loss and take-profit manager.

Computes dynamic stops that widen in volatile regimes and tighten in
calm/ranging ones.  ATR multipliers are adjusted by the current
volatility z-score relative to a rolling window.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StopLevels:
    entry: float
    stop_loss: float
    take_profit: float
    atr: float
    multiplier_used: float
    regime: str


class AdaptiveATRStops:
    """Compute ATR-based stops that adapt to market volatility regime."""

    # Default ATR multipliers per regime
    REGIME_MULTIPLIERS = {
        "trending": {"sl": 2.0, "tp": 3.5},
        "mean_reverting": {"sl": 1.2, "tp": 1.8},
        "volatile": {"sl": 3.0, "tp": 2.5},
        "ranging": {"sl": 1.5, "tp": 2.0},
    }

    def __init__(
        self,
        atr_period: int = 14,
        vol_lookback: int = 100,
        vol_z_scale: float = 0.5,
        min_sl_mult: float = 0.5,
        max_sl_mult: float = 6.0,
    ) -> None:
        self._atr_period = atr_period
        self._vol_lookback = vol_lookback
        self._vol_z_scale = vol_z_scale
        self._min_sl = min_sl_mult
        self._max_sl = max_sl_mult

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def compute(self, ohlcv: pd.DataFrame, entry_price: float, regime: str) -> StopLevels:
        """Compute stop/TP levels from OHLCV data.

        Parameters
        ----------
        ohlcv:  DataFrame with columns [open, high, low, close, volume].
        entry_price: Trade entry price.
        regime: One of the 4 recognised regime labels.
        """
        atr = self._atr(ohlcv)
        base_mults = self.REGIME_MULTIPLIERS.get(
            regime, self.REGIME_MULTIPLIERS["ranging"]
        )
        sl_mult = self._adapt_multiplier(ohlcv, base_mults["sl"])
        tp_mult = self._adapt_multiplier(ohlcv, base_mults["tp"])

        sl_dist = atr * sl_mult
        tp_dist = atr * tp_mult

        # Assume long direction; caller negates for shorts
        stop_loss = entry_price - sl_dist
        take_profit = entry_price + tp_dist

        logger.debug(
            "ATR=%.6f sl_mult=%.2f sl=%.6f tp=%.6f regime=%s",
            atr,
            sl_mult,
            stop_loss,
            take_profit,
            regime,
        )
        return StopLevels(
            entry=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr,
            multiplier_used=sl_mult,
            regime=regime,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _atr(self, df: pd.DataFrame) -> float:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return float(tr.ewm(span=self._atr_period, adjust=False).mean().iloc[-1])

    def _adapt_multiplier(self, df: pd.DataFrame, base: float) -> float:
        """Scale base multiplier by current vol z-score."""
        returns = df["close"].pct_change().dropna()
        if len(returns) < self._vol_lookback:
            return base
        rolling_vol = returns.rolling(self._vol_lookback).std().dropna()
        if len(rolling_vol) < 2:
            return base
        current_vol = rolling_vol.iloc[-1]
        mu = rolling_vol.mean()
        sigma = rolling_vol.std()
        if sigma == 0:
            return base
        z = (current_vol - mu) / sigma
        scaled = base * (1 + self._vol_z_scale * z)
        return float(np.clip(scaled, self._min_sl, self._max_sl))
