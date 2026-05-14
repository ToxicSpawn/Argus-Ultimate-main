"""
Adaptive ATR Stop-Loss patch for peak_alpha.py and momentum.py.

Import and call `compute_atr_stop` in both strategies to replace
static stop distances with regime-aware ATR multiples:

  HIGH_VOL  → ATR × 2.5
  LOW_VOL / TRENDING → ATR × 1.0
  default   → ATR × 1.5

Usage:
    from strategies.peak_alpha_atr_patch import compute_atr_stop

    stop_price = compute_atr_stop(
        entry_price=entry,
        atr=atr_value,
        regime=current_regime,
        side="buy",
    )
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Regime → ATR multiplier
_REGIME_ATR_MULT: dict[str, float] = {
    "HIGH_VOL": 2.5,
    "HIGHVOL": 2.5,
    "HIGH_VOLATILITY": 2.5,
    "LOW_VOL": 1.0,
    "LOWVOL": 1.0,
    "LOW_VOLATILITY": 1.0,
    "TRENDING": 1.0,
    "TREND_UP": 1.0,
    "TREND_DOWN": 1.0,
}
_DEFAULT_ATR_MULT = 1.5


def _resolve_mult(regime: Any) -> float:
    if regime is None:
        return _DEFAULT_ATR_MULT
    key = str(regime).upper().replace(" ", "_")
    for k, v in _REGIME_ATR_MULT.items():
        if k in key:
            return v
    return _DEFAULT_ATR_MULT


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> float:
    """
    Compute the latest ATR value from OHLC series.
    Returns 0.0 if insufficient data.
    """
    if len(close) < period + 1:
        return 0.0
    try:
        h = high.values[-period - 1:]
        l = low.values[-period - 1:]
        c = close.values[-period - 1:]
        tr_list = []
        for i in range(1, len(c)):
            tr = max(
                h[i] - l[i],
                abs(h[i] - c[i - 1]),
                abs(l[i] - c[i - 1]),
            )
            tr_list.append(tr)
        return float(sum(tr_list) / len(tr_list))
    except Exception as exc:
        logger.debug("compute_atr error: %s", exc)
        return 0.0


def compute_atr_stop(
    entry_price: float,
    atr: float,
    regime: Any = None,
    side: str = "buy",
    fallback_pct: float = 0.02,
) -> float:
    """
    Compute an ATR-based stop-loss price.

    Parameters
    ----------
    entry_price : float
        Order entry price.
    atr : float
        Current ATR value. If 0 or missing, falls back to pct-based stop.
    regime : str | MarketRegime | None
        Current market regime used to select ATR multiplier.
    side : str
        "buy" (long) → stop below entry; "sell" (short) → stop above entry.
    fallback_pct : float
        Fallback stop distance as fraction of entry when ATR=0.

    Returns
    -------
    float
        Stop-loss price.
    """
    mult = _resolve_mult(regime)
    if atr > 0:
        stop_dist = atr * mult
    else:
        stop_dist = entry_price * fallback_pct
        logger.debug(
            "compute_atr_stop: ATR=0, using fallback pct=%.2f%% → dist=%.6f",
            fallback_pct * 100, stop_dist,
        )

    if side.lower() == "buy":
        stop = entry_price - stop_dist
    else:
        stop = entry_price + stop_dist

    return max(0.0, stop)
