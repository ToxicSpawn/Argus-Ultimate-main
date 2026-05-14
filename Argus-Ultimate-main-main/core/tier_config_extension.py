"""
core/tier_config_extension.py

get_tier_cfg(tier) — returns the scalar risk / ops parameters that
OpsMetrics and _ensure_tier_patch need but that are not stored on the
TierDefinition dataclass in capital_tier_manager.py.

These values are intentionally conservative.  They can be overridden
per-deployment via environment variables (ARGUS_FEE_DRAG_ALERT_BPS etc.)
or by loading a YAML profile — but the hardcoded values here are always
the fallback, ensuring the system is safe even if no config file loads.

Tier reference (from capital_tier.py)
--------------------------------------
  NANO   : $0 – $499
  MICRO  : $500 – $2 499
  SMALL  : $2 500 – $9 999
  MID    : $10 000 – $49 999
  LARGE  : $50 000+
"""
from __future__ import annotations

import os
from typing import Any, Dict

try:
    from core.capital_tier import CapitalTier
except ImportError:
    from enum import Enum
    class CapitalTier(str, Enum):  # type: ignore[no-redef]
        NANO  = "NANO"
        MICRO = "MICRO"
        SMALL = "SMALL"
        MID   = "MID"
        LARGE = "LARGE"


_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # ── NANO ────────────────────────────────────────────────────────────
    "NANO": {
        # Risk / sizing
        "kelly_cap_pct":           0.04,   # 4% max Kelly fraction
        "portfolio_heat_limit":    0.08,   # 8% max open risk
        "max_open_positions":      2,
        "stop_atr_mult":           1.5,
        "atr_risk_mult":           0.60,
        # Execution
        "max_slices":              1,
        "max_slice_usd":           25.0,
        "min_notional_usd":        5.0,
        # Fee monitoring
        "fee_drag_alert_bps":      40.0,   # alert if 24h fee drag > 40 bps
        # Daily kill
        "daily_loss_limit_pct":    0.04,   # 4% daily loss kill
    },
    # ── MICRO ────────────────────────────────────────────────────────────
    "MICRO": {
        "kelly_cap_pct":           0.06,
        "portfolio_heat_limit":    0.10,
        "max_open_positions":      3,
        "stop_atr_mult":           1.5,
        "atr_risk_mult":           0.75,
        "max_slices":              2,
        "max_slice_usd":           50.0,
        "min_notional_usd":        5.0,
        "fee_drag_alert_bps":      25.0,
        "daily_loss_limit_pct":    0.05,
    },
    # ── SMALL ────────────────────────────────────────────────────────────
    "SMALL": {
        "kelly_cap_pct":           0.08,
        "portfolio_heat_limit":    0.15,
        "max_open_positions":      5,
        "stop_atr_mult":           2.0,
        "atr_risk_mult":           1.00,
        "max_slices":              3,
        "max_slice_usd":           200.0,
        "min_notional_usd":        10.0,
        "fee_drag_alert_bps":      20.0,
        "daily_loss_limit_pct":    0.06,
    },
    # ── MID ──────────────────────────────────────────────────────────────
    "MID": {
        "kelly_cap_pct":           0.10,
        "portfolio_heat_limit":    0.20,
        "max_open_positions":      10,
        "stop_atr_mult":           2.0,
        "atr_risk_mult":           1.00,
        "max_slices":              5,
        "max_slice_usd":           1_000.0,
        "min_notional_usd":        10.0,
        "fee_drag_alert_bps":      15.0,
        "daily_loss_limit_pct":    0.08,
    },
    # ── LARGE ────────────────────────────────────────────────────────────
    "LARGE": {
        "kelly_cap_pct":           0.12,
        "portfolio_heat_limit":    0.25,
        "max_open_positions":      20,
        "stop_atr_mult":           2.5,
        "atr_risk_mult":           1.00,
        "max_slices":              10,
        "max_slice_usd":           10_000.0,
        "min_notional_usd":        20.0,
        "fee_drag_alert_bps":      10.0,
        "daily_loss_limit_pct":    0.10,
    },
}


def get_tier_cfg(tier: "CapitalTier") -> Dict[str, Any]:
    """
    Return the config dict for *tier*, with any env-var overrides applied.

    Environment variable convention (all optional)::

        ARGUS_FEE_DRAG_ALERT_BPS=20
        ARGUS_HEAT_LIMIT=0.12
        ARGUS_KELLY_CAP=0.05
        ARGUS_DAILY_LOSS_LIMIT=0.04

    The env overrides apply to **all** tiers uniformly — useful for
    conservative cloud deployments.
    """
    tier_key = tier.value if hasattr(tier, "value") else str(tier)
    cfg = dict(_DEFAULTS.get(tier_key, _DEFAULTS["MICRO"]))

    # Optional env overrides
    for env_key, cfg_key in (
        ("ARGUS_FEE_DRAG_ALERT_BPS", "fee_drag_alert_bps"),
        ("ARGUS_HEAT_LIMIT",         "portfolio_heat_limit"),
        ("ARGUS_KELLY_CAP",          "kelly_cap_pct"),
        ("ARGUS_DAILY_LOSS_LIMIT",   "daily_loss_limit_pct"),
    ):
        raw = os.getenv(env_key)
        if raw is not None:
            try:
                cfg[cfg_key] = float(raw)
            except ValueError:
                pass

    return cfg
