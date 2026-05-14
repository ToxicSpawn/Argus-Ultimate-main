"""
DynamicParamHub — Universal parameter adapter.

Knows about every tunable numeric attribute across all self-awareness and
risk/ML components. Applies regime-based multipliers directly to live
component instances (no restart required).

Coverage — 45 parameters across 19 components:
  MetaGate (6), EscalatingGate (3), ConsecutiveLossGuard (3),
  StrategyEnforcer (4), EdgeMonitor (4), RegimeTransitionMonitor (2),
  SelfDiagnosis (5), IntradayVaR (2), DrawdownController (1),
  AlphaModel (5), VolatilityForecaster (1), OrderFlowToxicity (1),
  HMMRegime (1), OnlineLearner (2), EnsembleHub (2),
  SignalStacker (1), RegimeEnsemble (1), LLMSignal (1), MetaLabeler (1)

Safety guarantees:
  • Every write is clamped to PARAM_REGISTRY safe [min, max]
  • Maintains change log (last 50 actions) for audit + rollback
  • Re-applies only when regime or health_score has changed, or every
    tune_interval_cycles — avoids thrashing
  • reset_to_defaults() emergency rollback

Output: advisory["dynamic_params"]
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter registry
# Format: "component.attr": (default, min_safe, max_safe, type_)
# ---------------------------------------------------------------------------

PARAM_REGISTRY: Dict[str, Tuple] = {
    # MetaGate — 6 params
    "meta_gate.regime_conf_threshold":      (0.45, 0.10, 0.90, float),
    "meta_gate.regime_conf_panic":          (0.30, 0.05, 0.50, float),
    "meta_gate.staleness_threshold":        (0.70, 0.20, 0.95, float),
    "meta_gate.min_recent_sharpe":          (-0.50, -2.00, 0.50, float),
    "meta_gate.pause_dd_fraction":          (0.85, 0.50, 0.99, float),
    "meta_gate.halt_dd_fraction":           (0.95, 0.70, 0.99, float),

    # EscalatingGate — 3 params
    "escalating_gate.reduce_escalation_cycles": (100, 10, 500, int),
    "escalating_gate.pause_escalation_cycles":  (200, 20, 1000, int),
    "escalating_gate.de_escalation_cycles":     (50, 5, 500, int),

    # ConsecutiveLossGuard — 3 params
    "consecutive_loss_guard.n_consecutive_threshold": (5, 1, 20, int),
    "consecutive_loss_guard.cooldown_cycles":         (200, 10, 1000, int),
    "consecutive_loss_guard.max_daily_loss_pct":      (0.03, 0.005, 0.10, float),

    # StrategyEnforcer — 4 params
    "strategy_enforcer.disable_sharpe_threshold":  (-0.20, -1.00, 0.00, float),
    "strategy_enforcer.min_trades_to_enforce":     (15, 5, 100, int),
    "strategy_enforcer.re_enable_sharpe_threshold": (0.30, 0.00, 1.00, float),
    "strategy_enforcer.probation_cycles":          (500, 100, 2000, int),

    # EdgeMonitor — 4 params
    "edge_monitor.degraded_threshold":    (0.35, 0.10, 0.70, float),
    "edge_monitor.lookback_trades":       (20, 5, 100, int),
    "edge_monitor.min_fills":             (5, 1, 20, int),
    "edge_monitor.baseline_slippage_bps": (8.0, 0.1, 50.0, float),

    # RegimeTransitionMonitor — 2 params
    "regime_transition_monitor.pre_hedge_threshold": (0.40, 0.10, 0.90, float),
    "regime_transition_monitor.horizon_steps":       (12, 1, 100, int),

    # SelfDiagnosis — 5 params
    "self_diagnosis.slippage_warn_threshold":        (0.20, 0.05, 0.50, float),
    "self_diagnosis.slippage_critical_threshold":    (0.50, 0.20, 1.00, float),
    "self_diagnosis.win_rate_decline_threshold":     (0.05, 0.01, 0.20, float),
    "self_diagnosis.staleness_increase_threshold":   (0.15, 0.05, 0.50, float),
    "self_diagnosis.regime_accuracy_warn_threshold": (0.50, 0.30, 0.90, float),

    # IntradayVaR — 2 params
    "intraday_var.var_limit_pct": (0.02, 0.001, 0.10, float),
    "intraday_var.ewma_lambda":   (0.94, 0.80, 0.99, float),

    # DrawdownController — 1 param
    "drawdown_controller.max_drawdown_pct": (0.12, 0.02, 0.50, float),

    # AlphaModel factor weights — 5 params (normalised by hub after tuning)
    "alpha_model.momentum_1d_weight":      (0.25, 0.05, 0.60, float),
    "alpha_model.momentum_7d_weight":      (0.20, 0.05, 0.60, float),
    "alpha_model.reversal_1h_weight":      (0.15, 0.00, 0.40, float),
    "alpha_model.vol_adj_momentum_weight": (0.25, 0.05, 0.60, float),
    "alpha_model.carry_weight":            (0.15, 0.00, 0.40, float),

    # VolatilityForecaster — 1 param
    "vol_forecaster.lambda_ewma": (0.94, 0.80, 0.99, float),

    # OrderFlowToxicity — 1 param
    "order_flow_toxicity.bucket_size": (100.0, 1.0, 10000.0, float),

    # ── ML model params ────────────────────────────────────────────────────
    # HMMRegime — 1 param
    "hmm_regime.min_confidence":        (0.40, 0.10, 0.90, float),   # attr: min_confidence

    # OnlineLearner — 2 params
    "online_learner.pa_cost":           (1.0,  0.1,  10.0, float),   # attr: _C
    "online_learner.drift_threshold":   (50.0, 5.0,  500.0, float),  # attr: drift_threshold

    # EnsembleHub — 2 params
    "ensemble_hub.min_confidence":      (0.50, 0.20, 0.90, float),   # attr: _bullish_threshold
    "ensemble_hub.ttl_seconds":         (300.0, 30.0, 3600.0, float),# attr: _cache_ttl (int cast)

    # SignalStacker — 1 param
    "signal_stacker.stale_threshold":   (10,   1,    100,  int),     # attr: stale_threshold

    # RegimeEnsemble — 1 param
    "regime_ensemble.min_weight":       (0.10, 0.01, 0.50, float),   # attr: min_weight

    # LLMSignal — 1 param
    "llm_signal.timeout":               (10.0, 1.0,  60.0, float),   # attr: inference_timeout

    # MetaLabeler — 1 param
    "meta_labeler.lookback_window":     (20,   5,    200,  int),     # attr: lookback_window
}

# Keys where the registry type is float but the actual component attribute is int
_INT_CAST_FLOAT_KEYS = {"ensemble_hub.ttl_seconds"}

# AlphaModel weight keys (must re-normalise after tuning so they sum to ~1)
_ALPHA_WEIGHT_KEYS = {
    "alpha_model.momentum_1d_weight",
    "alpha_model.momentum_7d_weight",
    "alpha_model.reversal_1h_weight",
    "alpha_model.vol_adj_momentum_weight",
    "alpha_model.carry_weight",
}

# Component-to-attribute name mapping
# Format: "registry_key_prefix" → (component_key_in_dict, attr_name)
_COMPONENT_ATTR_MAP: Dict[str, Tuple[str, str]] = {
    "meta_gate.regime_conf_threshold":      ("meta_gate", "regime_conf_threshold"),
    "meta_gate.regime_conf_panic":          ("meta_gate", "regime_conf_panic"),
    "meta_gate.staleness_threshold":        ("meta_gate", "staleness_threshold"),
    "meta_gate.min_recent_sharpe":          ("meta_gate", "min_recent_sharpe"),
    "meta_gate.pause_dd_fraction":          ("meta_gate", "pause_dd_fraction"),
    "meta_gate.halt_dd_fraction":           ("meta_gate", "halt_dd_fraction"),
    "escalating_gate.reduce_escalation_cycles": ("escalating_gate", "reduce_escalation_cycles"),
    "escalating_gate.pause_escalation_cycles":  ("escalating_gate", "pause_escalation_cycles"),
    "escalating_gate.de_escalation_cycles":     ("escalating_gate", "de_escalation_cycles"),
    "consecutive_loss_guard.n_consecutive_threshold": ("consecutive_loss_guard", "n_consecutive_threshold"),
    "consecutive_loss_guard.cooldown_cycles":         ("consecutive_loss_guard", "cooldown_cycles"),
    "consecutive_loss_guard.max_daily_loss_pct":      ("consecutive_loss_guard", "max_daily_loss_pct"),
    "strategy_enforcer.disable_sharpe_threshold":  ("strategy_enforcer", "disable_sharpe_threshold"),
    "strategy_enforcer.min_trades_to_enforce":     ("strategy_enforcer", "min_trades_to_enforce"),
    "strategy_enforcer.re_enable_sharpe_threshold": ("strategy_enforcer", "re_enable_sharpe_threshold"),
    "strategy_enforcer.probation_cycles":          ("strategy_enforcer", "probation_cycles"),
    "edge_monitor.degraded_threshold":    ("edge_monitor", "degraded_threshold"),
    "edge_monitor.lookback_trades":       ("edge_monitor", "lookback_trades"),
    "edge_monitor.min_fills":             ("edge_monitor", "min_fills"),
    "edge_monitor.baseline_slippage_bps": ("edge_monitor", "baseline_slippage_bps"),
    "regime_transition_monitor.pre_hedge_threshold": ("regime_transition_monitor", "pre_hedge_threshold"),
    "regime_transition_monitor.horizon_steps":       ("regime_transition_monitor", "horizon_steps"),
    "self_diagnosis.slippage_warn_threshold":        ("self_diagnosis", "slippage_warn_threshold"),
    "self_diagnosis.slippage_critical_threshold":    ("self_diagnosis", "slippage_critical_threshold"),
    "self_diagnosis.win_rate_decline_threshold":     ("self_diagnosis", "win_rate_decline_threshold"),
    "self_diagnosis.staleness_increase_threshold":   ("self_diagnosis", "staleness_increase_threshold"),
    "self_diagnosis.regime_accuracy_warn_threshold": ("self_diagnosis", "regime_accuracy_warn_threshold"),
    "intraday_var.var_limit_pct": ("intraday_var", "var_limit_pct"),
    "intraday_var.ewma_lambda":   ("intraday_var", "ewma_lambda"),
    "drawdown_controller.max_drawdown_pct": ("drawdown_controller", "max_drawdown_pct"),
    "alpha_model.momentum_1d_weight":      ("alpha_model", "_weights"),  # dict attr
    "alpha_model.momentum_7d_weight":      ("alpha_model", "_weights"),
    "alpha_model.reversal_1h_weight":      ("alpha_model", "_weights"),
    "alpha_model.vol_adj_momentum_weight": ("alpha_model", "_weights"),
    "alpha_model.carry_weight":            ("alpha_model", "_weights"),
    "vol_forecaster.lambda_ewma": ("vol_forecaster", "_lambda"),
    "order_flow_toxicity.bucket_size": ("order_flow_toxicity", "_bucket_size"),

    # ML model params
    "hmm_regime.min_confidence":      ("hmm_regime",       "min_confidence"),
    "online_learner.pa_cost":         ("online_learner",    "_C"),
    "online_learner.drift_threshold": ("online_learner",    "drift_threshold"),
    "ensemble_hub.min_confidence":    ("ensemble_hub",      "_bullish_threshold"),
    "ensemble_hub.ttl_seconds":       ("ensemble_hub",      "_cache_ttl"),
    "signal_stacker.stale_threshold": ("signal_stacker",    "stale_threshold"),
    "regime_ensemble.min_weight":     ("regime_ensemble",   "min_weight"),
    "llm_signal.timeout":             ("llm_signal",        "inference_timeout"),
    "meta_labeler.lookback_window":   ("meta_labeler",      "lookback_window"),
}

# AlphaModel factor name → weight dict key mapping
_ALPHA_FACTOR_MAP = {
    "alpha_model.momentum_1d_weight":      "momentum_1d",
    "alpha_model.momentum_7d_weight":      "momentum_7d",
    "alpha_model.reversal_1h_weight":      "reversal_1h",
    "alpha_model.vol_adj_momentum_weight": "vol_adjusted_momentum",
    "alpha_model.carry_weight":            "carry",
}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParamChange:
    key: str
    old_value: float
    new_value: float
    reason: str
    regime: str
    health_score: int
    ts: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# DynamicParamHub
# ---------------------------------------------------------------------------

class DynamicParamHub:
    """
    Universal parameter adapter. Holds references to all live component
    instances and applies regime-driven multipliers to their attributes.

    Parameters
    ----------
    components : dict
        Map of {component_key: component_instance}, e.g.:
        {"meta_gate": registry.meta_gate, "escalating_gate": ...}
    regime_parameter_map : RegimeParameterMap
        Lookup table for regime multipliers.
    tune_interval_cycles : int
        Re-apply params every N cycles (avoids thrashing).
    config : optional config object
    """

    def __init__(
        self,
        components: Dict[str, Any],
        regime_parameter_map: Any,
        tune_interval_cycles: int = 50,
        config: Optional[Any] = None,
    ) -> None:
        self._components = components or {}
        self._rpm = regime_parameter_map
        self.tune_interval_cycles = max(1, int(tune_interval_cycles))
        self.config = config

        # Track last applied state to skip unchanged
        self._last_regime: Optional[str] = None
        self._last_health: Optional[int] = None
        self._last_tune_cycle: int = -9999
        self._position_scale: float = 1.0

        # Change history (last 50)
        self._change_log: Deque[ParamChange] = deque(maxlen=50)

    # ── Public API ─────────────────────────────────────────────────────────

    def tune(
        self,
        regime: str,
        health_score: int,
        cycle: int,
    ) -> List[ParamChange]:
        """
        Compute and apply regime-based parameter overrides to all live
        component instances. Returns list of changes made this call.
        """
        # Rate-limit: skip if nothing meaningful changed and not due
        regime_changed = regime != self._last_regime
        health_band = health_score // 20   # buckets: 0-19, 20-39, ...
        health_changed = (self._last_health is None or
                          self._last_health // 20 != health_band)
        due = (cycle - self._last_tune_cycle) >= self.tune_interval_cycles

        if not (regime_changed or health_changed or due):
            return []

        self._last_regime = regime
        self._last_health = health_score
        self._last_tune_cycle = cycle

        changes: List[ParamChange] = []

        # Get regime multipliers
        if self._rpm is None:
            rp_mults: Dict[str, float] = {}
        else:
            try:
                rp = self._rpm.get_params(regime)
                rp_mults = rp.multipliers
            except Exception as exc:
                logger.debug("RegimeParameterMap error: %s", exc)
                rp_mults = {}

        # Apply health-score modifier on top of regime multipliers
        # health < 40 → tighten remaining params by extra 10%
        # health >= 80 → slightly relax
        health_modifier = 1.0
        if health_score < 40:
            health_modifier = 0.90
        elif health_score >= 80:
            health_modifier = 1.05

        # Collect all alpha weights for post-normalisation
        alpha_targets: Dict[str, float] = {}

        # Only iterate over keys that are explicitly in the regime's multiplier map.
        # RANGING/UNKNOWN have empty rp_mults → no changes applied (correct behaviour:
        # don't snap back to defaults, just leave live values alone).
        if not rp_mults:
            # No regime-specific overrides — nothing to change
            self._position_scale = 1.0
            return changes

        for key in list(rp_mults.keys()):
            mult = rp_mults[key]
            if key not in PARAM_REGISTRY:
                continue
            default, min_safe, max_safe, type_ = PARAM_REGISTRY[key]
            # For safety-critical params only apply health modifier when worsening
            target = default * mult
            # Apply health modifier for non-alpha params
            if key not in _ALPHA_WEIGHT_KEYS:
                # For thresholds where larger = more conservative, tighten in poor health
                # For thresholds where smaller = more conservative, loosen in good health
                target = target * health_modifier if health_modifier != 1.0 else target

            # Clamp to safe range
            if type_ is int:
                target = int(round(max(min_safe, min(max_safe, target))))
            else:
                target = float(max(min_safe, min(max_safe, target)))

            if key in _ALPHA_WEIGHT_KEYS:
                alpha_targets[key] = target
                continue

            old_val = self._get_current(key)
            if old_val is None:
                continue

            # Only apply if meaningfully different (avoids log spam)
            if type_ is float and abs(float(old_val) - target) < 1e-6:
                continue
            if type_ is int and int(old_val) == target:
                continue

            if self._apply(key, target, type_):
                change = ParamChange(
                    key=key,
                    old_value=float(old_val),
                    new_value=float(target),
                    reason=f"regime={regime} health={health_score}",
                    regime=regime,
                    health_score=health_score,
                )
                changes.append(change)
                self._change_log.append(change)

        # Normalise alpha weights so they sum to ~1.0
        if alpha_targets:
            changes.extend(self._apply_alpha_weights(alpha_targets, regime, health_score))

        # Compute position_scale for advisory output
        self._position_scale = float(rp_mults.get("position_scale", 1.0))
        # health modifier on position scale
        if health_score < 40:
            self._position_scale = max(0.10, self._position_scale * 0.80)
        elif health_score >= 80:
            self._position_scale = min(1.50, self._position_scale * 1.05)

        if changes:
            logger.debug(
                "DynamicParamHub: %d param changes for regime=%s health=%d",
                len(changes), regime, health_score,
            )

        return changes

    def get_current(self, key: str) -> Optional[float]:
        """
        Read current value of a registered parameter from its live component.
        Returns PARAM_REGISTRY default when component is not registered.
        """
        val = self._get_current(key)
        if val is None and key in PARAM_REGISTRY:
            return float(PARAM_REGISTRY[key][0])  # return default
        return val

    def reset_to_defaults(self) -> List[ParamChange]:
        """Emergency: reset every registered param to its default value."""
        changes: List[ParamChange] = []
        alpha_targets: Dict[str, float] = {}

        for key, (default, min_safe, max_safe, type_) in PARAM_REGISTRY.items():
            if key in _ALPHA_WEIGHT_KEYS:
                alpha_targets[key] = default
                continue
            old_val = self._get_current(key)
            if old_val is None:
                continue
            if self._apply(key, default, type_):
                change = ParamChange(
                    key=key,
                    old_value=float(old_val) if old_val is not None else 0.0,
                    new_value=float(default),
                    reason="reset_to_defaults",
                    regime="RESET",
                    health_score=-1,
                )
                changes.append(change)
                self._change_log.append(change)

        if alpha_targets:
            changes.extend(self._apply_alpha_weights(alpha_targets, "RESET", -1))

        self._position_scale = 1.0
        logger.info("DynamicParamHub: reset %d params to defaults", len(changes))
        return changes

    def snapshot(self) -> Dict[str, Any]:
        recent = [
            {"key": c.key, "old": c.old_value, "new": c.new_value,
             "reason": c.reason, "ts": c.ts}
            for c in list(self._change_log)[-5:]
        ]
        return {
            "last_regime":    self._last_regime,
            "last_health":    self._last_health,
            "position_scale": self._position_scale,
            "total_changes":  len(self._change_log),
            "recent_changes": recent,
            "registered_params": len(PARAM_REGISTRY),
        }

    # ── Internal helpers ───────────────────────────────────────────────────

    def _get_current(self, key: str) -> Optional[float]:
        """Read the current attribute value from the live component."""
        mapping = _COMPONENT_ATTR_MAP.get(key)
        if not mapping:
            return None
        comp_key, attr_name = mapping
        comp = self._components.get(comp_key)
        if comp is None:
            return None
        try:
            if key in _ALPHA_WEIGHT_KEYS:
                weights_dict = getattr(comp, attr_name, None)
                if not isinstance(weights_dict, dict):
                    return None
                factor = _ALPHA_FACTOR_MAP.get(key)
                return float(weights_dict.get(factor, 0.0))
            return float(getattr(comp, attr_name, None) or 0.0)
        except Exception:
            return None

    def _apply(self, key: str, value: Any, type_: type = float) -> bool:
        """Write value to the live component attribute. Returns True on success."""
        mapping = _COMPONENT_ATTR_MAP.get(key)
        if not mapping:
            return False
        comp_key, attr_name = mapping
        comp = self._components.get(comp_key)
        if comp is None:
            return False
        try:
            # Special case: some float-registry keys require int on the component
            if key in _INT_CAST_FLOAT_KEYS:
                cast_val = int(round(float(value)))
            else:
                cast_val = type_(value)
            setattr(comp, attr_name, cast_val)
            return True
        except Exception as exc:
            logger.debug("DynamicParamHub._apply %s=%s error: %s", key, value, exc)
            return False

    def _apply_alpha_weights(
        self,
        targets: Dict[str, float],
        regime: str,
        health_score: int,
    ) -> List[ParamChange]:
        """Normalise and apply AlphaModel factor weights."""
        changes: List[ParamChange] = []
        comp = self._components.get("alpha_model")
        if comp is None:
            return changes

        weights_dict = getattr(comp, "_weights", None)
        if not isinstance(weights_dict, dict):
            return changes

        # Normalise so weights sum to 1.0
        total = sum(targets.values())
        if total < 1e-6:
            return changes
        normalised = {k: v / total for k, v in targets.items()}

        for key, norm_val in normalised.items():
            factor = _ALPHA_FACTOR_MAP.get(key)
            if not factor:
                continue
            old_val = float(weights_dict.get(factor, 0.0))
            if abs(old_val - norm_val) < 1e-6:
                continue
            weights_dict[factor] = float(norm_val)
            change = ParamChange(
                key=key,
                old_value=old_val,
                new_value=norm_val,
                reason=f"alpha_reweight regime={regime} health={health_score}",
                regime=regime,
                health_score=health_score,
            )
            changes.append(change)
            self._change_log.append(change)

        return changes
