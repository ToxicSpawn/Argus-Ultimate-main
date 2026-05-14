"""
Feature computation for ML model training.

Computes technical features from OHLCV DataFrames:
  - Returns (1d, 5d, 20d)
  - Volatility (rolling 10d, 20d, 60d)
  - RSI (14), MACD, Bollinger Band width
  - Volume ratio (current / 20d avg)
  - Momentum (rate of change 10d, 20d)
  - Mean reversion z-score (20d)
  - Trend strength (ADX approximation)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

FEATURE_NAMES: List[str] = [
    "return_1d",
    "return_5d",
    "return_20d",
    "vol_10d",
    "vol_20d",
    "vol_60d",
    "rsi_14",
    "macd",
    "bollinger_width",
    "volume_ratio",
    "momentum_10d",
    "momentum_20d",
    "mean_reversion_zscore",
    "trend_strength",
]


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Compute MACD line (fast EMA - slow EMA)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def compute_bollinger_width(close: pd.Series, period: int = 20) -> pd.Series:
    """Compute Bollinger Band width (upper - lower) / middle."""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    width = (upper - lower) / (sma + 1e-10)
    return width


def compute_adx_approx(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    """
    Simplified ADX approximation using directional movement.
    Computes ratio of absolute mean return to volatility over rolling window.
    """
    log_returns = np.log(close / close.shift(1))
    rolling_mean = log_returns.rolling(window=period).mean().abs()
    rolling_std = log_returns.rolling(window=period).std()
    adx = rolling_mean / (rolling_std + 1e-10)
    # Scale to 0-100 range roughly
    adx = adx * 100
    return adx.clip(0, 100)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features from an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: close, high, low, volume (and optionally open, timestamp, symbol).

    Returns
    -------
    pd.DataFrame
        Original columns plus all FEATURE_NAMES columns.
    """
    result = df.copy()
    close = result["close"]
    high = result["high"]
    low = result["low"]
    volume = result["volume"]

    log_close = np.log(close)

    # Returns
    result["return_1d"] = log_close.diff(1)
    result["return_5d"] = log_close.diff(5)
    result["return_20d"] = log_close.diff(20)

    # Volatility (annualized)
    daily_returns = log_close.diff(1)
    result["vol_10d"] = daily_returns.rolling(10).std() * np.sqrt(252)
    result["vol_20d"] = daily_returns.rolling(20).std() * np.sqrt(252)
    result["vol_60d"] = daily_returns.rolling(60).std() * np.sqrt(252)

    # RSI
    result["rsi_14"] = compute_rsi(close, period=14)

    # MACD (normalized by price)
    result["macd"] = compute_macd(close) / (close + 1e-10)

    # Bollinger Band width
    result["bollinger_width"] = compute_bollinger_width(close, period=20)

    # Volume ratio (current / 20-day average)
    vol_avg_20 = volume.rolling(20).mean()
    result["volume_ratio"] = volume / (vol_avg_20 + 1e-10)

    # Momentum (rate of change)
    result["momentum_10d"] = close.pct_change(10)
    result["momentum_20d"] = close.pct_change(20)

    # Mean reversion z-score (close vs 20-day SMA)
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    result["mean_reversion_zscore"] = (close - sma_20) / (std_20 + 1e-10)

    # Trend strength (ADX approximation)
    result["trend_strength"] = compute_adx_approx(close, high, low, period=14)

    return result


def label_regimes(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Label market regimes from features.

    Rules:
      BULL:     return_20d > 0 AND vol_20d < median vol
      BEAR:     return_20d < 0 AND vol_20d < median vol
      HIGH_VOL: vol_20d > 75th percentile
      CRISIS:   return_20d < -2*std(return_20d) AND vol_20d > 90th percentile

    Returns
    -------
    labels : np.ndarray of int
    label_names : list of str
    """
    label_names = ["BULL", "BEAR", "HIGH_VOL", "CRISIS"]
    label_map = {name: i for i, name in enumerate(label_names)}

    ret_20d = df["return_20d"].values
    vol_20d = df["vol_20d"].values

    vol_median = np.nanmedian(vol_20d)
    vol_75 = np.nanpercentile(vol_20d, 75)
    vol_90 = np.nanpercentile(vol_20d, 90)
    ret_std = np.nanstd(ret_20d)

    labels = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        r = ret_20d[i]
        v = vol_20d[i]

        # Crisis: extreme drawdown + extreme vol
        if r < -2 * ret_std and v > vol_90:
            labels[i] = label_map["CRISIS"]
        # High vol
        elif v > vol_75:
            labels[i] = label_map["HIGH_VOL"]
        # Bear
        elif r < 0 and v <= vol_median:
            labels[i] = label_map["BEAR"]
        # Bull
        elif r >= 0 and v <= vol_median:
            labels[i] = label_map["BULL"]
        # Default: HIGH_VOL (above median but below 75th)
        else:
            labels[i] = label_map["HIGH_VOL"]

    return labels, label_names
