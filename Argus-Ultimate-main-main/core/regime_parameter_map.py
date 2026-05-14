"""
RegimeParameterMap — Pre-defined parameter multiplier sets per market regime.

Every entry maps a "component.attr" key to a MULTIPLIER (1.0 = keep default).
Values > 1.0 increase the parameter; values < 1.0 decrease it.
DynamicParamHub reads this map and applies the multipliers to live components.

Regimes covered: CRISIS, HIGH_VOL, TRENDING_DOWN, RANGING, TRENDING_UP
Any unknown regime returns all 1.0 multipliers (no change from defaults).

Output: advisory["regime_parameters"]
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Multiplier maps — 1.0 = keep default, >1.0 increase, <1.0 decrease
# ---------------------------------------------------------------------------

_REGIME_MAP: Dict[str, Dict[str, float]] = {
    # ── CRISIS: maximum conservatism ─────────────────────────────────────
    "CRISIS": {
        # MetaGate: tighten all thresholds
        "meta_gate.regime_conf_threshold":    1.44,   # 0.45 → 0.65
        "meta_gate.regime_conf_panic":        1.33,   # 0.30 → 0.40
        "meta_gate.staleness_threshold":      0.86,   # 0.70 → 0.60
        "meta_gate.min_recent_sharpe":        2.00,   # -0.50 → -1.00 (less negative = stricter)
        "meta_gate.pause_dd_fraction":        0.94,   # 0.85 → 0.80
        # EscalatingGate: escalate faster in crisis
        "escalating_gate.reduce_cycles":      0.50,   # 100 → 50
        "escalating_gate.pause_cycles":       0.50,   # 200 → 100
        # ConsecutiveLossGuard: stricter — pause after 3 losses, not 5
        "consecutive_loss_guard.n_threshold": 0.60,   # 5 → 3
        "consecutive_loss_guard.cooldown_cycles": 1.50,  # 200 → 300
        "consecutive_loss_guard.max_daily_loss_pct": 0.50,  # 3% → 1.5%
        # StrategyEnforcer: disable faster
        "strategy_enforcer.disable_sharpe":   0.50,   # -0.20 → -0.10
        "strategy_enforcer.probation_cycles": 1.50,   # 500 → 750
        # EdgeMonitor: stricter degraded threshold
        "edge_monitor.degraded_threshold":    1.43,   # 0.35 → 0.50
        # IntradayVaR: tighten risk limit
        "intraday_var.var_limit_pct":         0.50,   # 0.02 → 0.01
        # SelfDiagnosis: tighten warn thresholds (warn earlier)
        "self_diagnosis.slippage_warn_threshold": 0.75,  # 0.20 → 0.15
        "self_diagnosis.slippage_critical_threshold": 0.60,  # 0.50 → 0.30
        # RegimeTransitionMonitor: hedge earlier
        "regime_transition_monitor.pre_hedge_threshold": 0.75,  # 0.40 → 0.30
        # DrawdownController: tighten max drawdown
        "drawdown_controller.max_drawdown_pct": 0.67,  # 0.12 → 0.08
        # AlphaModel: shift weight to reversal/carry in crisis
        "alpha_model.momentum_1d_weight":     0.60,   # 0.25 → 0.15
        "alpha_model.momentum_7d_weight":     0.50,   # 0.20 → 0.10
        "alpha_model.reversal_1h_weight":     1.67,   # 0.15 → 0.25
        "alpha_model.carry_weight":           1.67,   # 0.15 → 0.25
        # ML model params: tighten confidence requirements, slow drift, shorten LLM timeout
        "hmm_regime.min_confidence":          1.25,   # 0.40 → 0.50
        "ensemble_hub.min_confidence":        1.20,   # 0.50 → 0.60
        "llm_signal.timeout":                 0.75,   # 10s → 7.5s (faster fail)
        "online_learner.drift_threshold":     0.60,   # 50 → 30 (detect drift sooner)
    },

    # ── HIGH_VOL: elevated caution ────────────────────────────────────────
    "HIGH_VOL": {
        "meta_gate.regime_conf_threshold":    1.22,   # 0.45 → 0.55
        "meta_gate.staleness_threshold":      0.93,   # 0.70 → 0.65
        "escalating_gate.reduce_cycles":      0.75,   # 100 → 75
        "consecutive_loss_guard.n_threshold": 0.80,   # 5 → 4
        "consecutive_loss_guard.max_daily_loss_pct": 0.67,  # 3% → 2%
        "strategy_enforcer.disable_sharpe":   0.75,   # -0.20 → -0.15
        "edge_monitor.degraded_threshold":    1.14,   # 0.35 → 0.40
        "intraday_var.var_limit_pct":         0.75,   # 0.02 → 0.015
        "regime_transition_monitor.pre_hedge_threshold": 0.875,  # 0.40 → 0.35
        "alpha_model.momentum_1d_weight":     0.80,   # 0.25 → 0.20
        "alpha_model.reversal_1h_weight":     1.33,   # 0.15 → 0.20
        # ML model params: slightly tighten confidence thresholds
        "hmm_regime.min_confidence":          1.125,  # 0.40 → 0.45
        "ensemble_hub.min_confidence":        1.10,   # 0.50 → 0.55
        "online_learner.drift_threshold":     0.75,   # 50 → 37.5 (detect drift earlier)
    },

    # ── TRENDING_DOWN: moderate caution ──────────────────────────────────
    "TRENDING_DOWN": {
        "meta_gate.regime_conf_threshold":    1.11,   # 0.45 → 0.50
        "escalating_gate.reduce_cycles":      0.80,   # 100 → 80
        "consecutive_loss_guard.n_threshold": 0.80,   # 5 → 4
        "strategy_enforcer.disable_sharpe":   0.75,   # -0.20 → -0.15
        "edge_monitor.degraded_threshold":    1.14,   # 0.35 → 0.40
        "intraday_var.var_limit_pct":         0.75,   # 0.02 → 0.015
        "alpha_model.momentum_1d_weight":     0.80,
        "alpha_model.reversal_1h_weight":     1.33,
        # ML model params: mild tightening
        "hmm_regime.min_confidence":          1.10,   # 0.40 → 0.44
        "ensemble_hub.min_confidence":        1.08,   # 0.50 → 0.54
    },

    # ── RANGING: pure defaults — no changes ──────────────────────────────
    "RANGING": {},

    # ── UNKNOWN / fallback: same as RANGING ──────────────────────────────
    "UNKNOWN": {},

    # ── TRENDING_UP: opportunistic expansion ─────────────────────────────
    "TRENDING_UP": {
        "meta_gate.regime_conf_threshold":    0.89,   # 0.45 → 0.40
        "meta_gate.min_recent_sharpe":        0.80,   # -0.50 → -0.40 (still lenient)
        "escalating_gate.reduce_cycles":      1.25,   # 100 → 125 (slower escalation)
        "escalating_gate.de_escalation_cycles": 0.80,  # 50 → 40 (de-escalate faster)
        "consecutive_loss_guard.n_threshold": 1.40,   # 5 → 7 (more lenient)
        "strategy_enforcer.disable_sharpe":   1.50,   # -0.20 → -0.30 (more lenient)
        "strategy_enforcer.probation_cycles": 0.80,   # 500 → 400 (recover faster)
        "edge_monitor.degraded_threshold":    0.86,   # 0.35 → 0.30
        "intraday_var.var_limit_pct":         1.50,   # 0.02 → 0.03
        "drawdown_controller.max_drawdown_pct": 1.25,  # 0.12 → 0.15
        "alpha_model.momentum_1d_weight":     1.20,   # 0.25 → 0.30
        "alpha_model.momentum_7d_weight":     1.25,   # 0.20 → 0.25
        "alpha_model.reversal_1h_weight":     0.67,   # 0.15 → 0.10
        "alpha_model.carry_weight":           0.67,   # 0.15 → 0.10
        # ML model params: loosen confidence requirements, allow more drift
        "ensemble_hub.min_confidence":        0.85,   # 0.50 → 0.425 (more lenient)
        "hmm_regime.min_confidence":          0.875,  # 0.40 → 0.35
        "online_learner.drift_threshold":     1.25,   # 50 → 62.5 (less sensitive to drift)
    },
}

# Aliases — map regime label variants to canonical entries
_ALIASES: Dict[str, str] = {
    "BEAR":          "TRENDING_DOWN",
    "BULL":          "TRENDING_UP",
    "HIGH_VOLATILITY": "HIGH_VOL",
    "CRASH":         "CRISIS",
    "SIDEWAYS":      "RANGING",
    "FLAT":          "RANGING",
}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegimeParameters:
    regime: str
    multipliers: Dict[str, float]   # param_key → multiplier (1.0 = default)
    is_default: bool                # True when no specific regime overrides found
    ts: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# RegimeParameterMap
# ---------------------------------------------------------------------------

class RegimeParameterMap:
    """
    Lookup table for regime-specific parameter multipliers.

    Used by DynamicParamHub to compute target values:
        target_value = default_value × multiplier

    Parameters
    ----------
    config : optional config object (unused, reserved for overrides)
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = config
        self._map: Dict[str, Dict[str, float]] = dict(_REGIME_MAP)
        self._last_regime: Optional[str] = None
        self._last_params: Optional[RegimeParameters] = None

    # ── Public API ─────────────────────────────────────────────────────────

    def get_params(self, regime: str) -> RegimeParameters:
        """Return multiplier dict for the given regime (empty dict = all 1.0)."""
        canonical = self._resolve(regime)
        mults = dict(self._map.get(canonical, {}))
        is_default = len(mults) == 0
        rp = RegimeParameters(
            regime=canonical,
            multipliers=mults,
            is_default=is_default,
        )
        self._last_regime = canonical
        self._last_params = rp
        return rp

    def get_multiplier(self, regime: str, param_key: str) -> float:
        """Return single multiplier for a (regime, param_key) pair. 1.0 if not found."""
        canonical = self._resolve(regime)
        return self._map.get(canonical, {}).get(param_key, 1.0)

    def all_regimes(self) -> List[str]:
        """Return list of all regime labels in the map."""
        return list(self._map.keys())

    def snapshot(self) -> Dict[str, Any]:
        if self._last_params is None:
            return {"regime": "UNKNOWN", "multipliers": {}, "is_default": True}
        p = self._last_params
        return {
            "regime":      p.regime,
            "multipliers": dict(p.multipliers),
            "is_default":  p.is_default,
            "ts":          p.ts,
        }

    # ── Internal ───────────────────────────────────────────────────────────

    def _resolve(self, regime: str) -> str:
        """Resolve alias and title-case variants to canonical regime name."""
        if not regime:
            return "UNKNOWN"
        upper = str(regime).upper().strip()
        # Direct match
        if upper in self._map:
            return upper
        # Alias match
        if upper in _ALIASES:
            return _ALIASES[upper]
        # Partial match (e.g. "TREND_UP" → "TRENDING_UP")
        for canonical in self._map:
            if upper in canonical or canonical in upper:
                return canonical
        return "UNKNOWN"
