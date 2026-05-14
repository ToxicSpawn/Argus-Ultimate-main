"""
Regime-Conditional Hyperparameter Store.

Stores per-regime strategy parameters and returns the active set based on the
current market regime.  Allows the system to use wider take-profits in TRENDING
regimes and tighter stops in RANGING/HIGH_VOL regimes.

Default parameter sets (overridable):

  TRENDING  : tp_pct=0.060, sl_pct=0.015, size_mult=1.0, entry_score_min=0.55
  RANGE     : tp_pct=0.030, sl_pct=0.008, size_mult=0.7, entry_score_min=0.60
  HIGH_VOL  : tp_pct=0.040, sl_pct=0.012, size_mult=0.5, entry_score_min=0.65
  CRISIS    : tp_pct=0.025, sl_pct=0.010, size_mult=0.2, entry_score_min=0.70
  UNKNOWN   : tp_pct=0.035, sl_pct=0.010, size_mult=0.6, entry_score_min=0.60

Usage::

    store = RegimeParamStore()
    params = store.get("TRENDING")
    # {'tp_pct': 0.06, 'sl_pct': 0.015, 'size_mult': 1.0, 'entry_score_min': 0.55}

    # Per-strategy override
    store.set_override("RANGE", "cross_exchange_arb", {"tp_pct": 0.02, "sl_pct": 0.005})
    params = store.get("RANGE", strategy="cross_exchange_arb")
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Default parameters per regime ─────────────────────────────────────────────

_DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "TRENDING": {
        "tp_pct":          0.060,   # take-profit: 6% from entry
        "sl_pct":          0.015,   # stop-loss:   1.5% from entry
        "size_mult":       1.0,     # full position size
        "entry_score_min": 0.55,    # minimum signal score to enter
        "hold_max_bars":   72,      # max bars to hold before time-exit
    },
    "BULL": {
        "tp_pct":          0.060,
        "sl_pct":          0.015,
        "size_mult":       1.0,
        "entry_score_min": 0.55,
        "hold_max_bars":   72,
    },
    "RANGE": {
        "tp_pct":          0.030,   # tighter TP for mean-reversion plays
        "sl_pct":          0.008,
        "size_mult":       0.7,
        "entry_score_min": 0.60,
        "hold_max_bars":   48,
    },
    "SIDEWAYS": {
        "tp_pct":          0.030,
        "sl_pct":          0.008,
        "size_mult":       0.7,
        "entry_score_min": 0.60,
        "hold_max_bars":   48,
    },
    "HIGH_VOL": {
        "tp_pct":          0.040,   # wider TP to avoid stop-outs on spikes
        "sl_pct":          0.012,   # but tighter size
        "size_mult":       0.5,
        "entry_score_min": 0.65,
        "hold_max_bars":   24,
    },
    "ELEVATED": {
        "tp_pct":          0.040,
        "sl_pct":          0.012,
        "size_mult":       0.5,
        "entry_score_min": 0.65,
        "hold_max_bars":   24,
    },
    "CRISIS": {
        "tp_pct":          0.025,   # very tight — capital preservation
        "sl_pct":          0.010,
        "size_mult":       0.2,
        "entry_score_min": 0.70,
        "hold_max_bars":   12,
    },
    "EXTREME": {
        "tp_pct":          0.025,
        "sl_pct":          0.010,
        "size_mult":       0.2,
        "entry_score_min": 0.70,
        "hold_max_bars":   12,
    },
    "BEAR": {
        "tp_pct":          0.030,
        "sl_pct":          0.010,
        "size_mult":       0.3,
        "entry_score_min": 0.65,
        "hold_max_bars":   24,
    },
    "UNKNOWN": {
        "tp_pct":          0.035,
        "sl_pct":          0.010,
        "size_mult":       0.6,
        "entry_score_min": 0.60,
        "hold_max_bars":   48,
    },
}

_UNKNOWN_PARAMS = _DEFAULT_PARAMS["UNKNOWN"]


class RegimeParamStore:
    """
    Regime-conditional hyperparameter store.

    Parameters
    ----------
    overrides : dict, optional
        Global per-regime overrides that take precedence over defaults.
        Format: ``{regime: {param: value, ...}}``.
    """

    def __init__(self, overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self._base: Dict[str, Dict[str, Any]] = deepcopy(_DEFAULT_PARAMS)
        # strategy-level overrides: {regime: {strategy: {param: value}}}
        self._strategy_overrides: Dict[str, Dict[str, Dict[str, Any]]] = {}

        if overrides:
            for regime, params in overrides.items():
                key = regime.upper()
                if key in self._base:
                    self._base[key].update(params)
                else:
                    self._base[key] = dict(params)

    # ── Core API ──────────────────────────────────────────────────────────────

    def get(
        self,
        regime: str,
        strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return parameter dict for the given regime (and optionally strategy).

        Strategy-level overrides layer on top of regime defaults.
        Falls back to UNKNOWN if regime is unrecognised.

        Parameters
        ----------
        regime : str
            Current market regime label.
        strategy : str, optional
            Strategy name for per-strategy parameter overrides.

        Returns
        -------
        dict
            Copy of the parameter dict — safe to mutate.
        """
        key = (regime or "UNKNOWN").upper().strip()

        # Try exact match; then substring match
        params = self._base.get(key)
        if params is None:
            for k in self._base:
                if k in key:
                    params = self._base[k]
                    break
        if params is None:
            logger.debug(
                "RegimeParamStore.get: unknown regime %r — using UNKNOWN defaults",
                regime,
            )
            params = _UNKNOWN_PARAMS

        result = dict(params)

        # Apply strategy-level override on top
        if strategy:
            strat_key = strategy.lower()
            strat_overrides = self._strategy_overrides.get(key, {})
            if strat_key in strat_overrides:
                result.update(strat_overrides[strat_key])

        return result

    def set_override(
        self,
        regime: str,
        strategy: str,
        params: Dict[str, Any],
    ) -> None:
        """
        Set per-strategy parameter overrides for a regime.

        Parameters
        ----------
        regime : str
            Regime label (case-insensitive).
        strategy : str
            Strategy name.
        params : dict
            Parameter overrides.
        """
        key = regime.upper().strip()
        strat_key = strategy.lower()
        if key not in self._strategy_overrides:
            self._strategy_overrides[key] = {}
        self._strategy_overrides[key][strat_key] = dict(params)
        logger.info("RegimeParamStore: set override [%s][%s] = %s", key, strat_key, params)

    def set_regime_defaults(self, regime: str, params: Dict[str, Any]) -> None:
        """Update or add default parameters for an entire regime."""
        key = regime.upper().strip()
        if key in self._base:
            self._base[key].update(params)
        else:
            self._base[key] = dict(params)

    def snapshot(self) -> Dict[str, Any]:
        """Return full state for diagnostics."""
        return {
            "regimes": list(self._base.keys()),
            "strategy_overrides": {
                k: list(v.keys())
                for k, v in self._strategy_overrides.items()
                if v
            },
        }
