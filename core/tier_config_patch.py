"""
core/tier_config_patch.py

Tier-keyed configuration dicts for stops, order slicing, fee gates, and
portfolio heat limits.  Import and merge into core/config.py at runtime
or reference directly from execution layers.

Structure
---------
TIER_STOPS          — ATR multipliers and fixed stop floors per tier
TIER_SLICING        — TWAP/VWAP slice counts and notional thresholds
TIER_FEE_GATES      — min trade notional to clear fee drag threshold
TIER_HEAT_LIMITS    — portfolio heat caps (mirrors position_sizing.py)
"""
from __future__ import annotations

from core.capital_tier import CapitalTier

# ── Stop parameters ───────────────────────────────────────────────────
TIER_STOPS: dict[CapitalTier, dict] = {
    CapitalTier.MICRO: {
        "atr_stop_mult":      2.0,
        "atr_trail_mult":     2.5,
        "hard_stop_pct":      0.03,    # 3 % hard floor
        "breakeven_r":        1.0,     # move to breakeven at 1R profit
    },
    CapitalTier.SMALL: {
        "atr_stop_mult":      2.0,
        "atr_trail_mult":     2.5,
        "hard_stop_pct":      0.025,
        "breakeven_r":        1.0,
    },
    CapitalTier.MEDIUM: {
        "atr_stop_mult":      1.8,
        "atr_trail_mult":     2.2,
        "hard_stop_pct":      0.020,
        "breakeven_r":        1.2,
    },
    CapitalTier.LARGE: {
        "atr_stop_mult":      1.5,
        "atr_trail_mult":     2.0,
        "hard_stop_pct":      0.015,
        "breakeven_r":        1.5,
    },
    CapitalTier.WHALE: {
        "atr_stop_mult":      1.5,
        "atr_trail_mult":     2.0,
        "hard_stop_pct":      0.010,
        "breakeven_r":        1.5,
    },
}

# ── Order slicing parameters ──────────────────────────────────────────
TIER_SLICING: dict[CapitalTier, dict] = {
    CapitalTier.MICRO: {
        "max_slices":          1,       # single market order — size too small to slice
        "min_slice_notional":  5.0,     # AUD
        "use_twap":            False,
        "use_vwap":            False,
    },
    CapitalTier.SMALL: {
        "max_slices":          2,
        "min_slice_notional":  10.0,
        "use_twap":            False,
        "use_vwap":            False,
    },
    CapitalTier.MEDIUM: {
        "max_slices":          3,
        "min_slice_notional":  50.0,
        "use_twap":            True,
        "use_vwap":            False,
    },
    CapitalTier.LARGE: {
        "max_slices":          5,
        "min_slice_notional":  200.0,
        "use_twap":            True,
        "use_vwap":            True,
    },
    CapitalTier.WHALE: {
        "max_slices":          10,
        "min_slice_notional":  500.0,
        "use_twap":            True,
        "use_vwap":            True,
    },
}

# ── Fee gate thresholds ───────────────────────────────────────────────
# Minimum trade notional (AUD) required so fees don't exceed fee_drag_bps
# basis points of trade value at the exchange's taker rate.
TIER_FEE_GATES: dict[CapitalTier, dict] = {
    CapitalTier.MICRO: {
        "min_notional_aud":    10.0,
        "max_fee_drag_bps":    50.0,   # 0.50 % — acceptable for micro
        "taker_rate_bps":      10.0,   # Binance spot taker 0.10 %
    },
    CapitalTier.SMALL: {
        "min_notional_aud":    20.0,
        "max_fee_drag_bps":    30.0,
        "taker_rate_bps":      10.0,
    },
    CapitalTier.MEDIUM: {
        "min_notional_aud":    50.0,
        "max_fee_drag_bps":    20.0,
        "taker_rate_bps":      8.0,    # VIP-1 equivalent
    },
    CapitalTier.LARGE: {
        "min_notional_aud":    200.0,
        "max_fee_drag_bps":    15.0,
        "taker_rate_bps":      6.0,    # VIP-2+
    },
    CapitalTier.WHALE: {
        "min_notional_aud":    500.0,
        "max_fee_drag_bps":    10.0,
        "taker_rate_bps":      4.0,    # VIP-4+
    },
}

# ── Portfolio heat limits (mirrored from position_sizing.py) ──────────
TIER_HEAT_LIMITS: dict[CapitalTier, float] = {
    CapitalTier.MICRO:  0.06,
    CapitalTier.SMALL:  0.08,
    CapitalTier.MEDIUM: 0.10,
    CapitalTier.LARGE:  0.12,
    CapitalTier.WHALE:  0.15,
}


def get_stop_params(tier: CapitalTier) -> dict:
    return dict(TIER_STOPS[tier])


def get_slicing_params(tier: CapitalTier) -> dict:
    return dict(TIER_SLICING[tier])


def get_fee_gate(tier: CapitalTier) -> dict:
    return dict(TIER_FEE_GATES[tier])


def get_heat_limit(tier: CapitalTier) -> float:
    return TIER_HEAT_LIMITS[tier]
