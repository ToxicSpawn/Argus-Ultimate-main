"""Batch 2 — Cross-asset regime detector.

Classifies the macro/crypto market regime using BTC dominance, VIX proxy,
correlation regime and momentum across BTC/ETH/DXY/GOLD.  Outputs one of
the 4 regime labels used throughout the Argus stack.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIME_LABELS = ["trending", "mean_reverting", "volatile", "ranging"]


class CrossAssetRegimeDetector:
    """Detect market regime from cross-asset price data."""

    def __init__(
        self,
        lookback: int = 60,
        vol_threshold: float = 0.025,
        trend_threshold: float = 0.015,
        corr_window: int = 30,
    ) -> None:
        self._lookback = lookback
        self._vol_threshold = vol_threshold
        self._trend_threshold = trend_threshold
        self._corr_window = corr_window

    def detect(
        self,
        prices: Dict[str, pd.Series],
        primary: str = "BTC/USDT",
    ) -> str:
        """Return the current regime label given a dict of price series.

        Parameters
        ----------
        prices : dict {asset_name: pd.Series of close prices}
        primary : primary asset to base regime on
        """
        if primary not in prices or len(prices[primary]) < self._lookback:
            return "ranging"

        btc = prices[primary].iloc[-self._lookback :]
        returns = btc.pct_change().dropna()

        current_vol = returns.std() * np.sqrt(252)
        trend_signal = abs((btc.iloc[-1] / btc.iloc[0]) - 1)

        # Cross-asset correlation regime
        cross_corr = self._cross_correlation(prices)

        if current_vol > self._vol_threshold * 3:
            regime = "volatile"
        elif trend_signal > self._trend_threshold:
            regime = "trending"
        elif cross_corr > 0.7:
            # High correlation → risk-on/off regime, treat as trending
            regime = "trending"
        else:
            # Low vol + low trend → mean reverting or ranging
            hurst = self._hurst(returns)
            regime = "mean_reverting" if hurst < 0.45 else "ranging"

        logger.debug(
            "Regime=%s vol=%.4f trend=%.4f corr=%.4f",
            regime,
            current_vol,
            trend_signal,
            cross_corr,
        )
        return regime

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cross_correlation(self, prices: Dict[str, pd.Series]) -> float:
        """Average pairwise correlation of recent returns."""
        series = []
        for p in prices.values():
            if len(p) >= self._corr_window:
                r = p.pct_change().dropna().iloc[-self._corr_window :]
                series.append(r)
        if len(series) < 2:
            return 0.0
        df = pd.concat(series, axis=1).dropna()
        if df.empty or df.shape[1] < 2:
            return 0.0
        corr = df.corr().values
        n = corr.shape[0]
        upper = corr[np.triu_indices(n, k=1)]
        return float(np.abs(upper).mean())

    @staticmethod
    def _hurst(returns: pd.Series, lags: int = 20) -> float:
        """Estimate Hurst exponent; <0.5 → mean-reverting, >0.5 → trending."""
        ts = returns.cumsum().values
        if len(ts) < lags + 2:
            return 0.5
        lag_range = range(2, min(lags, len(ts) // 2))
        rs_list = []
        for lag in lag_range:
            sub = ts[:lag]
            if sub.std() == 0:
                continue
            rs_list.append(np.log((sub.max() - sub.min()) / sub.std()))
        if not rs_list:
            return 0.5
        log_lags = [np.log(l) for l in lag_range[: len(rs_list)]]
        if len(log_lags) < 2:
            return 0.5
        hurst = float(np.polyfit(log_lags, rs_list, 1)[0])
        return hurst
