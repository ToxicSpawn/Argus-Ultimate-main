"""
Precomputed feature store: precompute rolling indicators (RSI, volatility, etc.)
and reuse in all strategies so signal generation is cheaper.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_cache: Dict[str, Dict[str, Any]] = {}
_cache_ts: Dict[str, float] = {}
CACHE_TTL_S = 30.0


def _cache_key(symbol: str, timeframe: str) -> str:
    return f"{symbol}_{timeframe}"


def precompute_indicators(
    closes: np.ndarray,
    highs: Optional[np.ndarray] = None,
    lows: Optional[np.ndarray] = None,
    volumes: Optional[np.ndarray] = None,
    rsi_period: int = 14,
    vol_window: int = 20,
) -> Dict[str, float]:
    """Compute RSI, rolling volatility, and optional other indicators."""
    out: Dict[str, float] = {}
    if closes is None or len(closes) < max(rsi_period, vol_window):
        return out
    c = np.asarray(closes).ravel()
    # RSI
    delta = np.diff(c)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.convolve(gain, np.ones(rsi_period) / rsi_period, mode="valid")
    avg_loss = np.convolve(loss, np.ones(rsi_period) / rsi_period, mode="valid")
    if len(avg_gain) and len(avg_loss):
        rs = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        out["rsi"] = float(rsi[-1])
    # Rolling volatility (annualized)
    ret = np.diff(c) / (c[:-1] + 1e-12)
    if len(ret) >= vol_window:
        out["volatility"] = float(np.std(ret[-vol_window:]) * np.sqrt(252 * 24 * 60))
    if len(c):
        out["last_close"] = float(c[-1])
    return out


def get_cached_features(symbol: str, timeframe: str = "1m") -> Optional[Dict[str, float]]:
    """Return precomputed features for (symbol, timeframe) if valid."""
    key = _cache_key(symbol, timeframe)
    if key in _cache and key in _cache_ts and (time.time() - _cache_ts[key]) < CACHE_TTL_S:
        return _cache[key]
    return None


def set_cached_features(symbol: str, timeframe: str, features: Dict[str, float]) -> None:
    """Store precomputed features for (symbol, timeframe)."""
    key = _cache_key(symbol, timeframe)
    _cache[key] = features
    _cache_ts[key] = time.time()
