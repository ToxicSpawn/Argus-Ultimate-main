"""
Multi-Factor Alpha Model — combines momentum, mean-reversion, microstructure,
and carry signals into a single composite alpha score per symbol.

Factors:
  - Time-series momentum (1d, 7d, 30d returns)
  - Short-term reversal (1h return flip)
  - Volatility-adjusted momentum (Sharpe ratio of recent returns)
  - Bid-ask spread penalty (microstructure cost)
  - Funding carry (perpetual vs spot)

Output: AlphaScore with signal in [-1, +1] and factor breakdown.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default factor weights (sum to 1.0)
DEFAULT_WEIGHTS = {
    "momentum_1d": 0.25,
    "momentum_7d": 0.20,
    "reversal_1h": 0.15,
    "vol_adjusted_momentum": 0.25,
    "carry": 0.15,
}

BARS_PER_HOUR = 12    # 5-minute bars
BARS_PER_DAY = 288
BARS_PER_WEEK = 2016


@dataclass
class AlphaScore:
    symbol: str
    composite: float              # [-1, +1] final alpha
    factors: Dict[str, float]     # individual factor scores
    signal: str                   # STRONG_LONG / LONG / NEUTRAL / SHORT / STRONG_SHORT
    confidence: float             # 0-1 based on data completeness


@dataclass
class _SymbolData:
    prices: Deque[float] = field(default_factory=lambda: deque(maxlen=3000))
    funding_rates: Deque[float] = field(default_factory=lambda: deque(maxlen=48))
    spread_bps: float = 0.0


def _zscore(series: np.ndarray, lookback: int) -> float:
    """Z-score of most recent value vs lookback window."""
    if len(series) < lookback + 1:
        return 0.0
    window = series[-lookback:]
    mu, sigma = float(np.mean(window)), float(np.std(window))
    if sigma < 1e-10:
        return 0.0
    return float((series[-1] - mu) / sigma)


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _classify(score: float) -> str:
    if score >= 0.5:
        return "STRONG_LONG"
    if score >= 0.15:
        return "LONG"
    if score <= -0.5:
        return "STRONG_SHORT"
    if score <= -0.15:
        return "SHORT"
    return "NEUTRAL"


class AlphaModel:
    """
    Multi-factor alpha model for crypto assets.

    Usage::

        model = AlphaModel()
        for price in price_series:
            model.update("BTC/USD", price)
        score = model.score("BTC/USD")
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        min_bars: int = BARS_PER_DAY,
    ) -> None:
        w = weights or DEFAULT_WEIGHTS
        total = sum(w.values())
        self._weights = {k: v / total for k, v in w.items()}
        self._min_bars = min_bars
        self._data: Dict[str, _SymbolData] = {}

    # ------------------------------------------------------------------
    def update(
        self,
        symbol: str,
        price: float,
        funding_rate: Optional[float] = None,
        spread_bps: float = 0.0,
    ) -> None:
        if symbol not in self._data:
            self._data[symbol] = _SymbolData()
        d = self._data[symbol]
        d.prices.append(price)
        d.spread_bps = spread_bps
        if funding_rate is not None:
            d.funding_rates.append(funding_rate)

    # ------------------------------------------------------------------
    def score(self, symbol: str) -> Optional[AlphaScore]:
        d = self._data.get(symbol)
        if d is None or len(d.prices) < self._min_bars:
            return None

        prices = np.array(d.prices)
        log_returns = np.diff(np.log(prices))
        factors: Dict[str, float] = {}

        # 1. Momentum 1-day
        if len(log_returns) >= BARS_PER_DAY:
            ret_1d = float(np.sum(log_returns[-BARS_PER_DAY:]))
            factors["momentum_1d"] = _clamp(ret_1d * 5)  # scale: ±20% → ±1
        else:
            factors["momentum_1d"] = 0.0

        # 2. Momentum 7-day
        if len(log_returns) >= BARS_PER_WEEK:
            ret_7d = float(np.sum(log_returns[-BARS_PER_WEEK:]))
            factors["momentum_7d"] = _clamp(ret_7d * 2)
        else:
            factors["momentum_7d"] = 0.0

        # 3. Short-term reversal (last 1h return, sign-flipped)
        if len(log_returns) >= BARS_PER_HOUR:
            ret_1h = float(np.sum(log_returns[-BARS_PER_HOUR:]))
            factors["reversal_1h"] = _clamp(-ret_1h * 20)
        else:
            factors["reversal_1h"] = 0.0

        # 4. Volatility-adjusted momentum (Sharpe of recent daily returns)
        if len(log_returns) >= BARS_PER_DAY * 2:
            daily_rets = np.array([
                float(np.sum(log_returns[i:i + BARS_PER_DAY]))
                for i in range(0, len(log_returns) - BARS_PER_DAY, BARS_PER_DAY)
            ])
            if len(daily_rets) >= 5:
                sharpe = float(np.mean(daily_rets) / (np.std(daily_rets) + 1e-10))
                factors["vol_adjusted_momentum"] = _clamp(sharpe / 2)
            else:
                factors["vol_adjusted_momentum"] = 0.0
        else:
            factors["vol_adjusted_momentum"] = 0.0

        # 5. Carry factor (positive funding = shorts pay longs → bullish bias)
        if len(d.funding_rates) >= 3:
            avg_funding = float(np.mean(list(d.funding_rates)[-3:]))
            # Positive funding: longs pay shorts → bearish for long-term holders
            # Negative funding: shorts pay longs → bullish signal
            factors["carry"] = _clamp(-avg_funding * 1000)
        else:
            factors["carry"] = 0.0

        # Spread penalty: high spread → reduce signal magnitude
        spread_penalty = 1.0 - min(0.5, d.spread_bps / 100)

        # Composite weighted score
        composite = sum(
            self._weights.get(k, 0.0) * v for k, v in factors.items()
        ) * spread_penalty

        composite = _clamp(composite)
        data_completeness = min(1.0, len(d.prices) / (self._min_bars * 2))

        return AlphaScore(
            symbol=symbol,
            composite=composite,
            factors=factors,
            signal=_classify(composite),
            confidence=data_completeness,
        )

    def all_scores(self) -> Dict[str, AlphaScore]:
        result = {}
        for sym in self._data:
            s = self.score(sym)
            if s is not None:
                result[sym] = s
        return result

    def set_weights(self, weights: Dict[str, float]) -> None:
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        self._weights = {k: v / total for k, v in weights.items()}
