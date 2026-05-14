"""
core/strategy_weight_router.py

StrategyWeightRouter — returns per-strategy weight multipliers keyed by
capital tier and market regime.

The multipliers are applied on top of whatever base allocation the
StrategyBandit / MoE router produces; they are *not* replacements for
the full routing stack.

Design goals
------------
* NANO / MICRO  — survive fees; bias toward carry (funding_rate_harvester)
  and momentum; zero-out strategies that need large notional to clear fees
  (stat_arb, kalman_pairs, market_maker).
* SMALL / MID   — widen strategy set; reintroduce stat_arb at a discount;
  full momentum + carry;
* LARGE         — no artificial restrictions; all weights = 1.0 (neutral).
* Regime overlay — regardless of tier, boost momentum in TRENDING_UP /
  TRENDING_DOWN and mean-reversion in RANGE_BOUND; flatten everything in
  HIGH_VOLATILITY.

Usage (already wired in full_wiring.py)::

    swr = StrategyWeightRouter()
    weights = swr.get_weights(regime="TRENDING_UP", tier=CapitalTier.MICRO)
    # → {"momentum": 1.21, "funding_rate_harvester": 1.20, ...}

All weights are floats in [0.0, 2.0].  A weight of 0.0 means the strategy
is disabled for that tier; 1.0 is neutral.
"""
from __future__ import annotations

from typing import Dict

try:
    from core.capital_tier import CapitalTier
except ImportError:  # graceful fallback so the file is importable standalone
    from enum import Enum
    class CapitalTier(str, Enum):  # type: ignore[no-redef]
        NANO  = "NANO"
        MICRO = "MICRO"
        SMALL = "SMALL"
        MID   = "MID"
        LARGE = "LARGE"


# ---------------------------------------------------------------------------
# Base weights per tier  (applied in all regimes before regime overlay)
# ---------------------------------------------------------------------------

_TIER_BASE: Dict[str, Dict[str, float]] = {
    # ── NANO ($0-$499) ────────────────────────────────────────────────────
    # Almost certainly paper / test money.  Only the two cheapest strategies.
    "NANO": {
        "funding_rate_harvester":  1.20,
        "momentum":                1.00,
        "mean_reversion":          0.50,
        "breakout":                0.50,
        "bb_squeeze":              0.50,
        "grid_trader":             0.00,
        "market_maker":            0.00,
        "cross_exchange_arb":      0.00,
        "liquidation_cascade":     0.00,
        "stat_arb":                0.00,
        "kalman_pairs":            0.00,
    },
    # ── MICRO ($500-$2 499) ───────────────────────────────────────────────
    # Fee drag is existential.  Carry + momentum only.
    "MICRO": {
        "funding_rate_harvester":  1.20,
        "momentum":                1.00,
        "mean_reversion":          0.70,
        "breakout":                0.80,
        "bb_squeeze":              0.70,
        "grid_trader":             0.00,
        "market_maker":            0.00,
        "cross_exchange_arb":      0.00,
        "liquidation_cascade":     0.50,
        "stat_arb":                0.00,
        "kalman_pairs":            0.00,
    },
    # ── SMALL ($2 500-$9 999) ─────────────────────────────────────────────
    # Stat-arb back at a discount; grid + market-maker still off.
    "SMALL": {
        "funding_rate_harvester":  1.20,
        "momentum":                1.10,
        "mean_reversion":          1.00,
        "breakout":                1.00,
        "bb_squeeze":              0.90,
        "grid_trader":             0.50,
        "market_maker":            0.40,
        "cross_exchange_arb":      0.60,
        "liquidation_cascade":     0.80,
        "stat_arb":                0.80,
        "kalman_pairs":            0.60,
    },
    # ── MID ($10 000-$49 999) ─────────────────────────────────────────────
    # All strategies viable; market-maker still slightly discounted.
    "MID": {
        "funding_rate_harvester":  1.10,
        "momentum":                1.10,
        "mean_reversion":          1.00,
        "breakout":                1.00,
        "bb_squeeze":              1.00,
        "grid_trader":             1.00,
        "market_maker":            0.80,
        "cross_exchange_arb":      1.00,
        "liquidation_cascade":     1.00,
        "stat_arb":                1.00,
        "kalman_pairs":            1.00,
    },
    # ── LARGE ($50 000+) ──────────────────────────────────────────────────
    # Neutral — let the bandit / MoE decide everything.
    "LARGE": {
        "funding_rate_harvester":  1.00,
        "momentum":                1.00,
        "mean_reversion":          1.00,
        "breakout":                1.00,
        "bb_squeeze":              1.00,
        "grid_trader":             1.00,
        "market_maker":            1.00,
        "cross_exchange_arb":      1.00,
        "liquidation_cascade":     1.00,
        "stat_arb":                1.00,
        "kalman_pairs":            1.00,
    },
}


# ---------------------------------------------------------------------------
# Regime overlay multipliers  (multiplied on top of tier base)
# ---------------------------------------------------------------------------

_REGIME_OVERLAY: Dict[str, Dict[str, float]] = {
    "TRENDING_UP": {
        "momentum":                1.10,
        "breakout":                1.10,
        "mean_reversion":          0.70,
        "funding_rate_harvester":  1.00,
        "stat_arb":                0.80,
        "kalman_pairs":            0.80,
        "market_maker":            0.80,
        "liquidation_cascade":     1.10,
    },
    "TRENDING_DOWN": {
        "momentum":                1.10,  # short momentum
        "breakout":                1.10,
        "mean_reversion":          0.70,
        "funding_rate_harvester":  1.10,  # funding often negative = short premium
        "stat_arb":                0.80,
        "kalman_pairs":            0.80,
        "market_maker":            0.70,
        "liquidation_cascade":     1.20,
    },
    "RANGE_BOUND": {
        "momentum":                0.80,
        "breakout":                0.60,
        "mean_reversion":          1.20,
        "bb_squeeze":              1.20,
        "grid_trader":             1.30,
        "funding_rate_harvester":  1.10,
        "stat_arb":                1.10,
        "kalman_pairs":            1.10,
        "market_maker":            1.20,
    },
    "HIGH_VOLATILITY": {
        "momentum":                0.60,
        "breakout":                0.70,
        "mean_reversion":          0.50,
        "bb_squeeze":              0.80,
        "grid_trader":             0.30,
        "funding_rate_harvester":  0.90,
        "stat_arb":                0.40,
        "kalman_pairs":            0.40,
        "market_maker":            0.30,
        "cross_exchange_arb":      0.70,
        "liquidation_cascade":     1.10,
    },
    "CRISIS": {
        "momentum":                0.40,
        "breakout":                0.40,
        "mean_reversion":          0.30,
        "bb_squeeze":              0.50,
        "grid_trader":             0.00,
        "funding_rate_harvester":  0.70,
        "stat_arb":                0.20,
        "kalman_pairs":            0.20,
        "market_maker":            0.00,
        "cross_exchange_arb":      0.50,
        "liquidation_cascade":     0.80,
    },
    # NORMAL / unknown — no overlay (all 1.0)
    "NORMAL": {},
}


class StrategyWeightRouter:
    """
    Combines tier base weights and regime overlay into a single dict of
    per-strategy float multipliers.

    Thread-safe (read-only after construction).
    """

    # Strategies in canonical order — new strategies added here automatically
    # get a 1.0 base weight on all tiers.
    KNOWN_STRATEGIES: tuple[str, ...] = (
        "funding_rate_harvester",
        "momentum",
        "mean_reversion",
        "breakout",
        "bb_squeeze",
        "grid_trader",
        "market_maker",
        "cross_exchange_arb",
        "liquidation_cascade",
        "stat_arb",
        "kalman_pairs",
    )

    def get_weights(
        self,
        regime: str,
        tier: "CapitalTier",
    ) -> Dict[str, float]:
        """
        Return a dict mapping strategy name → weight multiplier.

        Parameters
        ----------
        regime:
            One of TRENDING_UP, TRENDING_DOWN, RANGE_BOUND,
            HIGH_VOLATILITY, CRISIS, NORMAL (or any unknown string
            which falls back to NORMAL).
        tier:
            CapitalTier enum value (or string).

        Returns
        -------
        dict[str, float]
            All known strategies present; values in [0.0, 2.0].
        """
        tier_key   = tier.value if hasattr(tier, "value") else str(tier)
        regime_key = str(regime).upper()

        base    = _TIER_BASE.get(tier_key, _TIER_BASE["LARGE"])
        overlay = _REGIME_OVERLAY.get(regime_key, {})

        result: Dict[str, float] = {}
        for strategy in self.KNOWN_STRATEGIES:
            b = base.get(strategy, 1.0)
            o = overlay.get(strategy, 1.0)
            # Zero in base means disabled regardless of regime
            w = round(b * o, 4) if b > 0.0 else 0.0
            # Hard cap at 2.0
            result[strategy] = min(w, 2.0)

        return result

    def is_enabled(
        self,
        strategy: str,
        regime: str,
        tier: "CapitalTier",
    ) -> bool:
        """Convenience: returns True iff weight > 0."""
        return self.get_weights(regime, tier).get(strategy, 0.0) > 0.0
