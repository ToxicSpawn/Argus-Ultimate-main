"""
Minimal regime/next-bar boost from recent price series (numpy-only; optional LSTM later).
Outputs a scalar in [0.5, 1.0] to scale strategy confidence when use_regime_lstm_boost is True.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def regime_boost_from_closes(closes: Sequence[float], lookback: int = 20) -> float:
    """
    Compute a simple momentum/volatility-based boost from the last `lookback` closes.
    Returns a value in [0.5, 1.0]: higher when recent move aligns with trend, lower in chop.
    No LSTM dependency; use as baseline. Replace with LSTM/GRU when ml stack is ready.
    """
    if not closes or len(closes) < max(5, lookback):
        return 0.75  # neutral
    arr = np.array(closes[-lookback:], dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
        return 0.75
    rets = np.diff(arr) / np.maximum(arr[:-1], 1e-12)
    if len(rets) < 2:
        return 0.75
    vol = float(np.std(rets))
    trend = float(np.mean(rets))
    # Simple rule: low vol + consistent trend -> higher boost; high vol or no trend -> lower
    vol_scale = 1.0 - min(1.0, vol * 50.0)  # high vol -> lower
    trend_scale = 0.5 + 0.5 * min(1.0, abs(trend) * 100.0)  # stronger trend -> higher
    boost = 0.5 + 0.5 * (0.5 * vol_scale + 0.5 * trend_scale)
    return float(np.clip(boost, 0.5, 1.0))


def apply_regime_boost(confidence: float, closes: Optional[Sequence[float]] = None, lookback: int = 20) -> float:
    """Scale confidence by regime boost; if closes is None or short, return confidence unchanged."""
    if not closes or len(closes) < lookback:
        return confidence
    boost = regime_boost_from_closes(closes, lookback=lookback)
    return float(min(1.0, max(0.0, confidence * boost)))
