"""
adaptive/regime_strategy_router.py — Regime-Aware Strategy Switching

Maps market regime labels to strategy weight adjustments so that the system
favours strategies whose edge aligns with the current regime and penalises
strategies that historically underperform in that regime.

Usage::

    router = RegimeStrategyRouter()
    adjusted = router.get_weights("trending", {"momentum": 1.0, "mean_reversion": 1.0})
    # → {"momentum": 1.5, "mean_reversion": 0.3, ...}

Works standalone — no dependency on the rest of ARGUS.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime → strategy preference map
# ---------------------------------------------------------------------------
# Each entry maps a regime label to:
#   "boost"   – set of strategies whose weight is multiplied UP
#   "penalise" – set of strategies whose weight is multiplied DOWN
#   "boost_factor"   – multiplier applied to boosted strategies (>1.0)
#   "penalise_factor" – multiplier applied to penalised strategies (<1.0)
#   "size_scale"     – global size scalar (1.0 = normal, 0.5 = half-size)
#   "skip"           – strategies to skip entirely (hard block)

_REGIME_STRATEGY_MAP: Dict[str, Dict[str, Any]] = {
    # ---- Trending / bullish ----
    "trending": {
        "boost": {"momentum", "breakout", "peak_alpha", "trend_following", "mev_sandwich", "cross_chain_arb"},
        "penalise": {"mean_reversion", "scalping", "market_maker", "oracle_deviation", "grid_mean_reversion"},
        "boost_factor": 1.5,
        "penalise_factor": 0.3,
        "size_scale": 1.0,
        "skip": set(),
    },
    "bull": {
        "boost": {"momentum", "breakout", "peak_alpha", "trend_following", "mev_sandwich"},
        "penalise": {"mean_reversion", "market_maker", "grid_mean_reversion"},
        "boost_factor": 1.4,
        "penalise_factor": 0.4,
        "size_scale": 1.0,
        "skip": set(),
    },
    # ---- Mean-reverting / range-bound ----
    "mean_revert": {
        "boost": {"mean_reversion", "scalping", "market_maker", "stat_arb", "grid_mean_reversion", "oracle_deviation", "triangular_arb"},
        "penalise": {"momentum", "breakout", "trend_following", "mev_sandwich"},
        "boost_factor": 1.5,
        "penalise_factor": 0.3,
        "size_scale": 1.0,
        "skip": set(),
    },
    "range": {
        "boost": {"mean_reversion", "scalping", "market_maker", "stat_arb", "grid_mean_reversion", "oracle_deviation"},
        "penalise": {"momentum", "breakout", "trend_following"},
        "boost_factor": 1.4,
        "penalise_factor": 0.35,
        "size_scale": 1.0,
        "skip": set(),
    },
    # ---- Crisis / bear ----
    "crisis": {
        "boost": {"tail_hedge", "delta_neutral", "options_vol_arb"},
        "penalise": {"momentum", "breakout", "peak_alpha", "mean_reversion", "scalping", "mev_sandwich", "cross_chain_arb"},
        "boost_factor": 2.0,
        "penalise_factor": 0.2,
        "size_scale": 0.4,
        "skip": {"trend_following"},
    },
    "bear": {
        "boost": {"tail_hedge", "delta_neutral", "mean_reversion", "options_vol_arb"},
        "penalise": {"momentum", "breakout", "peak_alpha", "mev_sandwich"},
        "boost_factor": 1.6,
        "penalise_factor": 0.3,
        "size_scale": 0.6,
        "skip": set(),
    },
    # ---- High volatility ----
    "volatile": {
        "boost": {"scalping", "market_maker", "options_vol_arb", "oracle_deviation"},
        "penalise": {"momentum", "breakout", "trend_following", "cross_chain_arb"},
        "boost_factor": 1.3,
        "penalise_factor": 0.5,
        "size_scale": 0.5,
        "skip": set(),
    },
    "high_vol": {
        "boost": {"scalping", "market_maker", "options_vol_arb"},
        "penalise": {"momentum", "breakout", "trend_following", "mev_sandwich"},
        "boost_factor": 1.3,
        "penalise_factor": 0.5,
        "size_scale": 0.5,
        "skip": set(),
    },
    # ---- Low volatility / calm ----
    "calm": {
        "boost": {"grid_mean_reversion", "triangular_arb", "oracle_deviation", "mean_reversion"},
        "penalise": {"options_vol_arb", "tail_hedge"},
        "boost_factor": 1.4,
        "penalise_factor": 0.6,
        "size_scale": 1.0,
        "skip": set(),
    },
    # ---- Recovery ----
    "recovery": {
        "boost": {"momentum", "mean_reversion", "grid_mean_reversion"},
        "penalise": {"tail_hedge", "delta_neutral"},
        "boost_factor": 1.3,
        "penalise_factor": 0.5,
        "size_scale": 0.8,
        "skip": set(),
    },
}


class RegimeStrategyRouter:
    """Route strategy weights based on the current market regime.

    Parameters
    ----------
    regime_map : dict, optional
        Override the built-in ``_REGIME_STRATEGY_MAP`` with a custom mapping.
    """

    def __init__(
        self,
        regime_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self._map: Dict[str, Dict[str, Any]] = regime_map or deepcopy(_REGIME_STRATEGY_MAP)
        self._last_regime: Optional[str] = None
        self._last_weights: Dict[str, float] = {}
        log.info(
            "RegimeStrategyRouter initialised — %d regime profiles loaded",
            len(self._map),
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_weights(
        self,
        regime_label: str,
        base_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Return adjusted strategy weights for the given *regime_label*.

        Parameters
        ----------
        regime_label : str
            Current regime (e.g. ``"trending"``, ``"crisis"``).
        base_weights : dict
            Mapping of strategy name to its base weight (e.g. ``1.0``).

        Returns
        -------
        dict
            Adjusted weights.  Strategies in the ``skip`` set are given
            weight ``0.0``.  All weights are clamped to ``>= 0.0``.
        """
        regime_label = regime_label.lower().strip()
        profile = self._map.get(regime_label)

        if profile is None:
            log.debug(
                "RegimeStrategyRouter: unknown regime '%s' — returning base weights",
                regime_label,
            )
            self._log_transition(regime_label, base_weights)
            return dict(base_weights)

        boost: Set[str] = profile.get("boost", set())
        penalise: Set[str] = profile.get("penalise", set())
        skip: Set[str] = profile.get("skip", set())
        boost_factor: float = profile.get("boost_factor", 1.0)
        penalise_factor: float = profile.get("penalise_factor", 1.0)
        size_scale: float = profile.get("size_scale", 1.0)

        adjusted: Dict[str, float] = {}
        for strat, w in base_weights.items():
            if strat in skip:
                adjusted[strat] = 0.0
            elif strat in boost:
                adjusted[strat] = max(0.0, w * boost_factor * size_scale)
            elif strat in penalise:
                adjusted[strat] = max(0.0, w * penalise_factor * size_scale)
            else:
                adjusted[strat] = max(0.0, w * size_scale)

        self._log_transition(regime_label, adjusted)
        return adjusted

    def should_skip_strategy(
        self,
        strategy_name: str,
        regime_label: str,
    ) -> bool:
        """Return ``True`` if *strategy_name* should be entirely skipped
        under the current *regime_label*."""
        regime_label = regime_label.lower().strip()
        profile = self._map.get(regime_label)
        if profile is None:
            return False
        return strategy_name in profile.get("skip", set())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_regime_profile(self, regime_label: str) -> Optional[Dict[str, Any]]:
        """Return the raw profile dict for a regime, or ``None``."""
        return self._map.get(regime_label.lower().strip())

    @property
    def supported_regimes(self) -> List[str]:
        """Return sorted list of known regime labels."""
        return sorted(self._map.keys())

    # ------------------------------------------------------------------
    # Transition logging
    # ------------------------------------------------------------------

    def _log_transition(
        self,
        regime_label: str,
        new_weights: Dict[str, float],
    ) -> None:
        """Log regime transitions with old and new weight snapshots."""
        if self._last_regime is not None and regime_label != self._last_regime:
            log.info(
                "RegimeStrategyRouter: regime transition '%s' → '%s'  "
                "old_weights=%s  new_weights=%s",
                self._last_regime,
                regime_label,
                self._last_weights,
                new_weights,
            )
        elif self._last_regime is None:
            log.info(
                "RegimeStrategyRouter: initial regime '%s'  weights=%s",
                regime_label,
                new_weights,
            )
        self._last_regime = regime_label
        self._last_weights = dict(new_weights)
