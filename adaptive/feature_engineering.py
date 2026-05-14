"""
Automated feature engineering: generate and evaluate new features for strategies.

Stub: produces candidate features (e.g. RSI(7), vol_20, close/ema_50) and scores them
against recent performance; best can be fed into strategy param space.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_candidate_features(close: pd.Series, limit: int = 20) -> List[Tuple[str, pd.Series]]:
    """
    Generate candidate feature series (e.g. RSI variants, momentum, vol).
    Returns [(name, series), ...].
    """
    out: List[Tuple[str, pd.Series]] = []
    try:
        for period in (7, 14, 21):
            if len(close) < period + 5:
                continue
            delta = close.diff()
            up = delta.clip(lower=0)
            down = (-delta).clip(lower=0)
            roll_up = up.ewm(span=period, adjust=False).mean()
            roll_down = down.ewm(span=period, adjust=False).mean()
            rs = roll_up / roll_down.replace(0, np.nan)
            rsi = (100 - (100 / (1 + rs))).fillna(50)
            out.append((f"rsi_{period}", rsi))
        for period in (20, 50):
            if len(close) >= period:
                out.append((f"ret_{period}", close.pct_change(period)))
                out.append((f"vol_{period}", close.pct_change().rolling(period).std()))
        if len(out) > limit:
            out = out[:limit]
    except Exception as e:
        logger.debug("Feature gen: %s", e)
    return out


def score_feature(feature_series: pd.Series, forward_returns: pd.Series, method: str = "ic") -> float:
    """
    Score feature by information coefficient or correlation with forward returns.
    Stub: align lengths and return correlation.
    """
    try:
        common = feature_series.dropna().index.intersection(forward_returns.dropna().index)
        if len(common) < 10:
            return 0.0
        a = feature_series.reindex(common).fillna(0)
        b = forward_returns.reindex(common).fillna(0)
        return float(np.corrcoef(a, b)[0, 1]) if method == "ic" else 0.0
    except Exception:
        return 0.0


class AutomatedFeatureEngine:
    """Stub: generate candidates, score, return top for strategy use."""

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = int(top_k)

    def fit_and_select(self, ohlcv: pd.DataFrame) -> List[str]:
        """Generate features, score vs forward return, return top_k names."""
        if ohlcv is None or ohlcv.empty or "close" not in ohlcv.columns:
            return []
        close = ohlcv["close"].astype(float)
        fwd = close.pct_change(5).shift(5)  # 5-period PAST return (no look-ahead bias)
        candidates = generate_candidate_features(close)
        scored = [(score_feature(s, fwd), name) for name, s in candidates]
        scored.sort(key=lambda x: -abs(x[0]))
        return [name for _, name in scored[: self.top_k]]
