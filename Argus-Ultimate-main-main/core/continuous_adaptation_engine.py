"""
Continuous Adaptation Engine — master coordinator for ARGUS self-modification.

This is the conductor that orchestrates the entire continuous learning loop:

  1. ObservationRecorder       — captures every (input, decision, outcome) triple
  2. ParameterDriftOptimizer   — drifts numerical parameters toward optimal
  3. AdaptiveGateManager       — adapts intelligence gate thresholds
  4. AdaptationHealthMonitor   — verifies adaptations actually help
  5. (External) FeatureDiscoverer — finds new alpha features
  6. (External) StrategyEvolver  — generates new strategies
  7. (External) Hyperopt         — Bayesian parameter search

The engine runs every cycle but does light work most of the time. Heavy
adaptation happens at scheduled intervals:

  - Every cycle:        record observations, check health
  - Every 50 cycles:    drift parameters slightly
  - Every 100 cycles:   adapt gate thresholds
  - Every 500 cycles:   evaluate adaptation effectiveness
  - Every 1000 cycles:  full optimization sweep

If the AdaptationHealthMonitor detects that recent adaptations are hurting
performance, the engine throttles down and reverts to safe baselines.

This is the highest-level adaptation primitive in ARGUS — everything below
it (online learner, drift detection, GP evolver) feeds INTO this coordinator,
which decides what to actually apply.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AdaptationMode(Enum):
    AGGRESSIVE = "aggressive"   # adapt every 50 cycles, 2% max drift
    NORMAL = "normal"           # adapt every 100 cycles, 1% max drift
    CONSERVATIVE = "conservative"  # adapt every 200 cycles, 0.5% max drift
    PAUSED = "paused"           # no adaptation (when health is bad)


@dataclass
class AdaptationConfig:
    """Tunables for the continuous adaptation engine."""
    enabled: bool = True
    mode: AdaptationMode = AdaptationMode.NORMAL
    parameter_drift_cycles: int = 50
    gate_adapt_cycles: int = 100
    health_check_cycles: int = 500
    full_sweep_cycles: int = 1000
    measurement_window_cycles: int = 100
    max_simultaneous_adaptations: int = 10
    auto_throttle_enabled: bool = True
    auto_revert_enabled: bool = True


class ContinuousAdaptationEngine:
    """
    Master coordinator for ARGUS continuous adaptation.

    Usage::

        engine = ContinuousAdaptationEngine(
            observation_recorder=recorder,
            parameter_optimizer=opt,
            gate_manager=gate_mgr,
            health_monitor=monitor,
        )

        # Each cycle:
        engine.tick(
            cycle_number=42,
            portfolio_value_aud=1000.0,
            current_regime="TRENDING_UP",
        )
    """

    def __init__(
        self,
        observation_recorder: Any = None,
        parameter_optimizer: Any = None,
        gate_manager: Any = None,
        health_monitor: Any = None,
        config: Optional[AdaptationConfig] = None,
    ) -> None:
        self._config = config or AdaptationConfig()
        self._recorder = observation_recorder
        self._param_opt = parameter_optimizer
        self._gate_mgr = gate_manager
        self._health = health_monitor
        self._cycle_count = 0
        self._last_drift_cycle = 0
        self._last_gate_adapt_cycle = 0
        self._last_health_cycle = 0
        self._last_sweep_cycle = 0
        self._mode = self._config.mode
        self._stats: Dict[str, int] = {
            "drifts_applied": 0,
            "gate_adaptations": 0,
            "reverts": 0,
            "throttles": 0,
            "full_sweeps": 0,
        }
        # Default parameters to register
        self._registered_defaults = False
        logger.info(
            "ContinuousAdaptationEngine: initialized (mode=%s, drift_cycles=%d, gate_cycles=%d)",
            self._mode.value, self._config.parameter_drift_cycles, self._config.gate_adapt_cycles,
        )

    def register_default_parameters(self) -> None:
        """Register the standard ARGUS parameters for drift optimization."""
        if self._registered_defaults or self._param_opt is None:
            return

        # Sizing parameters
        self._param_opt.register(
            "max_position_pct", 0.25, 0.10, 0.50,
            description="Maximum position size as % of portfolio",
        )
        self._param_opt.register(
            "confidence_threshold", 0.55, 0.40, 0.75,
            description="Minimum signal confidence to trade",
        )

        # Risk parameters
        self._param_opt.register(
            "stop_loss_pct", 0.012, 0.005, 0.030,
            description="Stop loss as % of entry price",
        )
        self._param_opt.register(
            "take_profit_pct", 0.035, 0.015, 0.080,
            description="Take profit as % of entry price",
        )
        self._param_opt.register(
            "trailing_stop_pct", 0.015, 0.005, 0.040,
            description="Trailing stop distance as % of price",
        )

        # Sizing factors
        self._param_opt.register(
            "vol_adjustment_factor", 1.0, 0.5, 2.0,
            description="Volatility adjustment factor for position sizing",
        )
        self._param_opt.register(
            "kelly_fraction", 0.5, 0.25, 1.0,
            description="Fraction of full Kelly to use for sizing",
        )

        # Gate adaptations
        if self._gate_mgr is not None:
            self._gate_mgr.register_gate(
                "confidence_gate", 0.55, 0.40, 0.75,
                description="Minimum confidence for trade to pass",
            )
            self._gate_mgr.register_gate(
                "regime_confidence_gate", 0.40, 0.25, 0.65,
                description="Minimum regime confidence",
            )
            self._gate_mgr.register_gate(
                "edge_score_gate", 0.35, 0.20, 0.60,
                description="Minimum edge score from edge monitor",
            )
            self._gate_mgr.register_gate(
                "drawdown_gate", 0.05, 0.02, 0.15,
                description="Maximum allowed drawdown before reducing size",
            )

        self._registered_defaults = True
        logger.info("ContinuousAdaptationEngine: registered default parameters")

    def tick(
        self,
        cycle_number: int,
        portfolio_value_aud: float,
        current_regime: str = "NORMAL",
        cumulative_pnl: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Run one cycle of the continuous adaptation engine.
        Returns a status dict.
        """
        self._cycle_count = cycle_number

        if not self._config.enabled or self._mode == AdaptationMode.PAUSED:
            return {"enabled": False, "mode": self._mode.value}

        if not self._registered_defaults:
            self.register_default_parameters()

        result: Dict[str, Any] = {
            "cycle": cycle_number,
            "mode": self._mode.value,
            "actions": [],
        }

        # 1. Health check (every 50 cycles by default in monitor)
        if self._health is not None:
            health_state = self._health.update(
                portfolio_value=portfolio_value_aud,
                cycle=cycle_number,
                cumulative_pnl=cumulative_pnl,
            )

            if health_state.get("should_evaluate"):
                outcome = health_state.get("outcome", {})
                result["evaluated_adaptation"] = outcome
                if outcome.get("should_revert", False) and self._config.auto_revert_enabled:
                    self._handle_revert()
                    result["actions"].append("reverted")

            if self._health.should_throttle() and self._config.auto_throttle_enabled:
                self._throttle()
                result["actions"].append("throttled")

        # 2. Parameter drift (cadence based on mode)
        drift_cadence = self._effective_drift_cadence()
        if (cycle_number - self._last_drift_cycle) >= drift_cadence:
            self._last_drift_cycle = cycle_number
            drift_count = self._run_parameter_drift(current_regime, portfolio_value_aud, cycle_number, cumulative_pnl)
            if drift_count > 0:
                result["actions"].append(f"drifted_{drift_count}_params")
                self._stats["drifts_applied"] += drift_count

        # 3. Gate adaptation
        gate_cadence = self._effective_gate_cadence()
        if (cycle_number - self._last_gate_adapt_cycle) >= gate_cadence:
            self._last_gate_adapt_cycle = cycle_number
            gate_count = self._run_gate_adaptation()
            if gate_count > 0:
                result["actions"].append(f"adapted_{gate_count}_gates")
                self._stats["gate_adaptations"] += gate_count

        # 4. Full optimization sweep
        if (cycle_number - self._last_sweep_cycle) >= self._config.full_sweep_cycles:
            self._last_sweep_cycle = cycle_number
            self._run_full_sweep()
            result["actions"].append("full_sweep")
            self._stats["full_sweeps"] += 1

        return result

    def _effective_drift_cadence(self) -> int:
        """Return drift cadence adjusted for current mode."""
        base = self._config.parameter_drift_cycles
        if self._mode == AdaptationMode.AGGRESSIVE:
            return max(int(base * 0.5), 10)
        if self._mode == AdaptationMode.CONSERVATIVE:
            return int(base * 2)
        return base

    def _effective_gate_cadence(self) -> int:
        base = self._config.gate_adapt_cycles
        if self._mode == AdaptationMode.AGGRESSIVE:
            return max(int(base * 0.5), 25)
        if self._mode == AdaptationMode.CONSERVATIVE:
            return int(base * 2)
        return base

    def _run_parameter_drift(
        self,
        regime: str,
        portfolio_value: float,
        cycle: int,
        cumulative_pnl: float,
    ) -> int:
        """Compute and apply parameter drifts."""
        if self._param_opt is None:
            return 0

        # Snapshot for health monitor
        if self._health is not None:
            self._health.before_adaptation(
                portfolio_value=portfolio_value,
                cycle=cycle,
                cumulative_pnl=cumulative_pnl,
            )

        # Compute drifts
        new_values = self._param_opt.compute_drifts(regime=regime)

        # Limit number of simultaneous changes
        if len(new_values) > self._config.max_simultaneous_adaptations:
            sorted_changes = sorted(
                new_values.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:self._config.max_simultaneous_adaptations]
            new_values = dict(sorted_changes)

        applied = self._param_opt.apply_drifts(new_values)

        if applied > 0 and self._health is not None:
            self._health.after_adaptation(adaptations_applied=applied)

        return applied

    def _run_gate_adaptation(self) -> int:
        """Compute and apply gate adaptations."""
        if self._gate_mgr is None:
            return 0
        adaptations = self._gate_mgr.compute_adaptations()

        # Limit simultaneous changes
        if len(adaptations) > self._config.max_simultaneous_adaptations:
            limited_adaptations = dict(list(adaptations.items())[:self._config.max_simultaneous_adaptations])
            adaptations = limited_adaptations

        return self._gate_mgr.apply_adaptations(adaptations)

    def _run_full_sweep(self) -> None:
        """Run a full optimization pass — slower but more thorough."""
        logger.info("ContinuousAdaptationEngine: running full sweep at cycle %d", self._cycle_count)
        # Currently just logs — extension point for hyperopt integration

    def _throttle(self) -> None:
        """Reduce adaptation rate when health is bad."""
        old_mode = self._mode
        if self._mode == AdaptationMode.AGGRESSIVE:
            self._mode = AdaptationMode.NORMAL
        elif self._mode == AdaptationMode.NORMAL:
            self._mode = AdaptationMode.CONSERVATIVE
        elif self._mode == AdaptationMode.CONSERVATIVE:
            self._mode = AdaptationMode.PAUSED

        if old_mode != self._mode:
            self._stats["throttles"] += 1
            logger.warning(
                "ContinuousAdaptationEngine: throttled %s → %s",
                old_mode.value, self._mode.value,
            )

    def _handle_revert(self) -> None:
        """Handle a revert request from the health monitor."""
        self._stats["reverts"] += 1
        if self._param_opt is not None:
            # Revert parameters that have drifted
            for name in list(self._param_opt._params.keys()):
                state = self._param_opt.get_parameter_state(name)
                if state and abs(state["current_value"] - state["initial_value"]) > 1e-6:
                    self._param_opt.check_for_reverts({name: -10.0})  # force revert
        if self._health is not None:
            self._health.mark_reverted()
        logger.warning("ContinuousAdaptationEngine: REVERT executed")

    def set_mode(self, mode: AdaptationMode) -> None:
        old = self._mode
        self._mode = mode
        logger.info(
            "ContinuousAdaptationEngine: mode changed %s → %s",
            old.value, mode.value,
        )

    def resume(self) -> None:
        """Resume from paused state."""
        if self._mode == AdaptationMode.PAUSED:
            self._mode = AdaptationMode.CONSERVATIVE
            logger.info("ContinuousAdaptationEngine: resumed (CONSERVATIVE mode)")

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": self._config.enabled,
            "mode": self._mode.value,
            "cycle": self._cycle_count,
            "stats": dict(self._stats),
            "registered_params": (
                len(self._param_opt._params) if self._param_opt else 0
            ),
            "registered_gates": (
                len(self._gate_mgr._gates) if self._gate_mgr else 0
            ),
            "health_effectiveness": (
                self._health.get_effectiveness() if self._health else 0.5
            ),
        }
