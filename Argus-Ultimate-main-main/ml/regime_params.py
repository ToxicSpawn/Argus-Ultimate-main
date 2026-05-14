"""
Regime-keyed Kelly fractions — maps the current market regime to a Kelly multiplier.

Different regimes have different levels of edge predictability:

  TRENDING / BULL / BREAKOUT  → full Kelly fraction  (0.50)
  RANGE / SIDEWAYS / NEUTRAL  → moderate Kelly       (0.30)
  HIGH_VOL / ELEVATED         → half Kelly           (0.15)
  CRISIS / EXTREME / BEAR     → minimal Kelly        (0.05)
  UNKNOWN / default           → conservative Kelly   (0.20)

The Kelly multiplier is NOT the Kelly fraction itself — it is a scalar applied
to whatever position size the caller would otherwise use.  The product keeps
the system inside a sensible fraction of the theoretically optimal Kelly bet.

Usage::

    rp = RegimeParams()
    kelly_mult = rp.kelly(current_regime)   # e.g. 0.50 in TRENDING
    sized_capital = base_capital * kelly_mult

You can also inject per-regime overrides at construction time::

    rp = RegimeParams(overrides={"RANGE": 0.25, "CRISIS": 0.03})
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Default Kelly fractions per regime keyword ─────────────────────────────────

_DEFAULT_KELLY: Dict[str, float] = {
    # Trending / bullish regimes — strong edge predictability
    "TRENDING":    0.50,
    "BULL":        0.50,
    "BREAKOUT":    0.50,
    "MOMENTUM":    0.45,
    "STRONG_BULL": 0.50,

    # Ranging / neutral regimes — moderate edge
    "RANGE":       0.30,
    "SIDEWAYS":    0.30,
    "NEUTRAL":     0.30,
    "MEAN_REVERT": 0.30,
    "LOW_VOL":     0.35,

    # Elevated volatility — reduced sizing
    "HIGH_VOL":    0.15,
    "HIGH_VOLATILITY": 0.15,
    "ELEVATED":    0.15,
    "VOLATILE":    0.15,

    # Crisis / extreme / bear — minimal sizing, capital preservation
    "CRISIS":      0.05,
    "EXTREME":     0.05,
    "BEAR":        0.08,
    "STRONG_BEAR": 0.05,
    "CRASH":       0.05,

    # Fallback
    "UNKNOWN":     0.20,
}

_DEFAULT_KELLY_FALLBACK = 0.20


class RegimeParams:
    """
    Maps regime labels to Kelly fraction multipliers.

    Parameters
    ----------
    overrides : dict, optional
        Per-regime Kelly overrides that take precedence over defaults.
        Keys are uppercase regime strings; values are fractions in (0, 1].
    """

    def __init__(
        self,
        overrides: Optional[Dict[str, float]] = None,
    ) -> None:
        self._table: Dict[str, float] = dict(_DEFAULT_KELLY)
        if overrides:
            for k, v in overrides.items():
                self._table[k.upper()] = float(v)

    # ── Core API ───────────────────────────────────────────────────────────────

    def kelly(self, regime: str) -> float:
        """
        Return Kelly fraction for the given regime label.

        The lookup is case-insensitive and falls back to ``_DEFAULT_KELLY_FALLBACK``
        (0.20) when the label is unrecognised.

        Parameters
        ----------
        regime : str
            Current market regime label (e.g. "TRENDING", "CRISIS").

        Returns
        -------
        float
            Kelly fraction in (0, 1].
        """
        key = (regime or "UNKNOWN").upper().strip()
        fraction = self._table.get(key)
        if fraction is None:
            # Try substring match for compound labels like "STRONG_BULL_2026"
            for k, v in self._table.items():
                if k in key:
                    fraction = v
                    break
        if fraction is None:
            logger.debug(
                "RegimeParams.kelly: unknown regime %r — using default %.2f",
                regime, _DEFAULT_KELLY_FALLBACK,
            )
            fraction = _DEFAULT_KELLY_FALLBACK
        return float(fraction)

    def snapshot(self) -> Dict[str, float]:
        """Return a copy of the full regime → Kelly table."""
        return dict(self._table)

    def set_override(self, regime: str, fraction: float) -> None:
        """Runtime override of a regime's Kelly fraction."""
        key = regime.upper().strip()
        self._table[key] = float(fraction)
        logger.info("RegimeParams: set %s → %.3f", key, fraction)
