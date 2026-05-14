"""
Adaptive ATR-based stop-loss and take-profit calculator.

Stop multiplier varies by market regime:
  HIGH_VOL  -> ATR x 2.5  (avoids stop-outs during volatile news events)
  TRENDING  -> ATR x 1.0  (tight stops, ride the trend)
  default   -> ATR x 1.5  (balanced)

Take-profit is always R:R >= 2:1 relative to stop distance.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Regime string -> ATR stop multiplier
_REGIME_ATR_MULT: dict = {
    "high_volatility": 2.5,
    "high_vol": 2.5,
    "volatile": 2.5,
    "trending": 1.0,
    "trend_up": 1.0,
    "trend_down": 1.0,
    "ranging": 1.5,
    "range": 1.5,
    "unknown": 1.5,
}
_DEFAULT_MULT = 1.5
_MIN_RR = 2.0  # minimum reward:risk ratio


def compute_adaptive_stops(
    df: pd.DataFrame,
    entry_price: float,
    regime: str = "unknown",
    atr_period: int = 14,
    rr_ratio: float = _MIN_RR,
    existing_stop: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Compute adaptive stop-loss and take-profit prices.

    Parameters
    ----------
    df          : OHLCV DataFrame (must have high, low, close columns)
    entry_price : trade entry price
    regime      : lowercase regime string (e.g. 'high_volatility', 'trending')
    atr_period  : ATR lookback period
    rr_ratio    : minimum reward-to-risk ratio for take-profit
    existing_stop : if provided and tighter than computed stop, keep existing

    Returns
    -------
    (stop_loss_price, take_profit_price)
    """
    try:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.ewm(span=atr_period, min_periods=atr_period // 2).mean().iloc[-1])
    except Exception as exc:
        logger.debug("ATR calculation error: %s", exc)
        atr = entry_price * 0.015  # fallback: 1.5% of price

    mult = _REGIME_ATR_MULT.get(regime.lower(), _DEFAULT_MULT)
    stop_dist = atr * mult
    stop_loss = entry_price - stop_dist

    # Don't widen stop if an existing tighter stop is already in place
    if existing_stop is not None and existing_stop > stop_loss:
        stop_loss = existing_stop

    take_profit = entry_price + (stop_dist * rr_ratio)

    logger.debug(
        "AdaptiveStop: entry=%.4f atr=%.4f mult=%.1f regime=%s "
        "stop=%.4f tp=%.4f",
        entry_price, atr, mult, regime, stop_loss, take_profit,
    )
    return round(stop_loss, 8), round(take_profit, 8)
