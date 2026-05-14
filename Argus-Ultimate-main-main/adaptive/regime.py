from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    HIGH_VOL = "high_vol"


@dataclass(frozen=True)
class RegimeSnapshot:
    regime: MarketRegime
    trend_score: float  # signed, roughly in [-1, 1]
    vol_annualized: float  # heuristic annualized vol (not for risk reporting)
    sentiment_score: float = 0.0  # optional: -1 (fear) to 1 (greed), from emotion/sentiment
    macro_score: float = 0.0  # optional: macro bias, e.g. rates/growth


class RegimeDetector:
    """
    Lightweight regime detector using only OHLCV closes.
    Optional multi-factor: sentiment_score, macro_score (from external modules).

    - trend_score: normalized EMA(fast)-EMA(slow)
    - vol_annualized: std of returns * sqrt(365*24*60 / minutes_per_bar)
    """

    def __init__(
        self,
        *,
        fast_ema: int = 12,
        slow_ema: int = 48,
        vol_window: int = 48,
        trend_threshold: float = 0.0025,
        high_vol_threshold: float = 1.20,
        minutes_per_bar: float = 60.0,
        sentiment_provider: Optional[Any] = None,
        macro_provider: Optional[Any] = None,
    ) -> None:
        self.fast_ema = int(fast_ema)
        self.slow_ema = int(slow_ema)
        self.vol_window = int(vol_window)
        self.trend_threshold = float(trend_threshold)
        self.high_vol_threshold = float(high_vol_threshold)
        self.minutes_per_bar = float(minutes_per_bar)
        self._sentiment_provider = sentiment_provider
        self._macro_provider = macro_provider
        # Online regime learning: adaptive thresholds from recent observations
        self._online_enabled = True
        self._trend_scores: list = []  # rolling window
        self._vol_scores: list = []
        self._window_size = 200
        self._alpha_ema = 0.05

    def update_thresholds(self, trend_score: float, vol_annualized: float) -> None:
        """Update adaptive thresholds from new observation (online regime learning)."""
        if not getattr(self, "_online_enabled", True):
            return
        ts = getattr(self, "_trend_scores", [])
        vs = getattr(self, "_vol_scores", [])
        ts.append(float(trend_score))
        vs.append(float(vol_annualized))
        n = getattr(self, "_window_size", 200)
        if len(ts) > n:
            ts.pop(0)
        if len(vs) > n:
            vs.pop(0)
        self._trend_scores = ts
        self._vol_scores = vs
        if len(ts) >= 20:
            # Adaptive: threshold = percentile of recent distribution
            try:
                self.trend_threshold = float(np.percentile(np.abs(ts), 75)) * 0.5
                self.trend_threshold = max(0.0008, min(0.01, self.trend_threshold))
            except Exception as e:
                logger.debug("Adaptive trend threshold update failed: %s", e)
        if len(vs) >= 20:
            try:
                self.high_vol_threshold = float(np.percentile(vs, 85))
                self.high_vol_threshold = max(0.5, min(2.5, self.high_vol_threshold))
            except Exception as e:
                logger.debug("Adaptive vol threshold update failed: %s", e)

    def _get_sentiment(self, symbol: str = "") -> float:
        """Optional: -1 (fear) to 1 (greed). Stub returns 0 if no provider."""
        p = getattr(self, "_sentiment_provider", None)
        if p is None:
            return 0.0
        try:
            if callable(getattr(p, "fear_greed", None)):
                return float(p.fear_greed(symbol=symbol) or 0.0)
            if callable(getattr(p, "get_score", None)):
                return float(p.get_score() or 0.0)
        except Exception as e:
            logger.debug("Sentiment provider error: %s", e)
        return 0.0

    def _get_macro(self) -> float:
        """Optional: macro bias. Stub returns 0 if no provider."""
        p = getattr(self, "_macro_provider", None)
        if p is None:
            return 0.0
        try:
            if callable(getattr(p, "bias", None)):
                return float(p.bias() or 0.0)
        except Exception as e:
            logger.debug("Macro provider error: %s", e)
        return 0.0

    def detect(self, df: pd.DataFrame, symbol: str = "") -> Optional[RegimeSnapshot]:
        if df is None or df.empty or "close" not in df.columns:
            return None

        close = df["close"].astype(float)
        if len(close) < max(10, self.slow_ema + 5, self.vol_window + 5):
            return None

        ema_fast = close.ewm(span=self.fast_ema, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_ema, adjust=False).mean()
        last = float(close.iloc[-1])
        spread = float(ema_fast.iloc[-1] - ema_slow.iloc[-1])
        trend_score = float(spread / max(abs(last), 1e-9))

        rets = close.pct_change().dropna()
        w = int(min(self.vol_window, len(rets)))
        vol = float(np.std(rets.values[-w:])) if w > 3 else float(np.std(rets.values))

        # Heuristic annualization.
        bars_per_year = (365.0 * 24.0 * 60.0) / max(self.minutes_per_bar, 1e-9)
        vol_ann = float(vol * np.sqrt(max(bars_per_year, 1.0)))

        # Regime classification.
        if vol_ann >= self.high_vol_threshold:
            regime = MarketRegime.HIGH_VOL
        elif trend_score >= self.trend_threshold:
            regime = MarketRegime.TREND_UP
        elif trend_score <= -self.trend_threshold:
            regime = MarketRegime.TREND_DOWN
        else:
            regime = MarketRegime.RANGE

        if getattr(self, "_online_enabled", True):
            self.update_thresholds(trend_score, vol_ann)
        sentiment_score = self._get_sentiment(symbol=symbol)
        macro_score = self._get_macro()
        return RegimeSnapshot(
            regime=regime,
            trend_score=trend_score,
            vol_annualized=vol_ann,
            sentiment_score=float(sentiment_score),
            macro_score=float(macro_score),
        )

