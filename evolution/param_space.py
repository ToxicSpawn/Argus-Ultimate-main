"""
Single source of truth for evolvable strategy parameters.

All evolution paths (unified GA, godmode, ultimate) should use these bounds
for params that are applied to UnifiedConfig. Keys here must exist as
UnifiedConfig attributes (or be mapped in apply_to_config) so auto_apply is never a no-op.
"""

from __future__ import annotations

from typing import Dict, Tuple

# Unified param bounds: keys = UnifiedConfig attribute names used by StrategyEngine/execution.
# Extended set can be added as UnifiedConfig gains more evolvable fields.
EVOLVABLE_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    # StrategyEngine (paper/backtest optimization) - all exist on UnifiedConfig
    "se_buy_rsi": (20.0, 45.0),
    "se_sell_rsi": (55.0, 80.0),
    "se_buy_bb": (0.10, 0.45),
    "se_sell_bb": (0.55, 0.90),
    "se_trend_rsi_buy": (45.0, 65.0),
    "se_trend_rsi_sell": (35.0, 55.0),
    "min_signal_confidence": (0.55, 0.85),
}


def get_bounds() -> Dict[str, Tuple[float, float]]:
    """Return a copy of evolvable param bounds for the unified system."""
    return dict(EVOLVABLE_PARAM_BOUNDS)


def filter_to_config_keys(params: Dict[str, float], config_attrs: set) -> Dict[str, float]:
    """
    Return only param keys that exist as config attributes so apply never no-ops.
    config_attrs should be set(dir(config)) or similar.
    """
    return {k: v for k, v in params.items() if k in config_attrs}
