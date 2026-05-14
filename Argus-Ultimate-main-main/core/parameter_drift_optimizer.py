"""
Parameter Drift Optimizer — online gradient adaptation of numerical parameters.

Every parameter ARGUS uses (stop-loss, take-profit, confidence threshold,
position sizing, gate thresholds, etc.) has a "current value" and an
"observed gradient" — how much its current setting helped or hurt recent P&L.

This module:
  1. Registers parameters with safe min/max bounds
  2. Observes outcomes correlated with parameter values
  3. Computes gradient (does increasing this help?)
  4. Drifts the parameter slowly toward optimal
  5. Reverts if drift hurts performance (anti-overfitting)

Key principles:
  - Slow drift (max 1-2% per cycle) — never sudden changes
  - Bounded — never violate hard safety limits
  - Reversible — can undo any change if it backfires
  - Per-regime — track gradients separately per market regime
  - Confidence-weighted — small sample size = small drift
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ParameterDefinition:
    """Definition of a tunable parameter with bounds."""
    name: str
    current_value: float
    min_value: float
    max_value: float
    initial_value: float                # for revert
    learning_rate: float = 0.01         # max change per cycle (as fraction)
    description: str = ""
    regime_specific: bool = True        # track per-regime values


@dataclass
class ParameterObservation:
    """Records a (parameter_value, outcome_pnl) pair for gradient computation."""
    timestamp: float
    parameter_name: str
    parameter_value: float
    pnl_contribution: float
    regime: str
    sample_weight: float = 1.0


@dataclass
class ParameterDriftHistory:
    """Tracks the history of one parameter's drift."""
    name: str
    history: deque = field(default_factory=lambda: deque(maxlen=1000))
    observations: deque = field(default_factory=lambda: deque(maxlen=10000))
    last_drift_direction: float = 0.0
    drift_count: int = 0
    revert_count: int = 0
    best_value_seen: float = 0.0
    best_pnl_at_value: float = -float("inf")


class ParameterDriftOptimizer:
    """
    Online gradient optimizer for ARGUS numerical parameters.

    Usage::

        opt = ParameterDriftOptimizer()

        # Register parameters
        opt.register("stop_loss_pct", current=0.012, min=0.005, max=0.030)
        opt.register("confidence_threshold", current=0.55, min=0.40, max=0.75)

        # On every fill:
        opt.observe_outcome(
            parameter_values={"stop_loss_pct": 0.012, "confidence_threshold": 0.55},
            pnl_aud=15.0,
            regime="TRENDING_UP",
        )

        # Periodically drift:
        new_values = opt.compute_drifts()
        # Apply new_values to system
    """

    # Min observations before computing gradient
    MIN_OBSERVATIONS_FOR_DRIFT = 20

    # If drift hurts, revert after this many consecutive bad cycles
    REVERT_THRESHOLD_CYCLES = 5

    # Max drift per cycle as fraction of current value
    MAX_DRIFT_FRACTION = 0.02

    def __init__(self) -> None:
        self._params: Dict[str, ParameterDefinition] = {}
        self._history: Dict[str, ParameterDriftHistory] = {}
        self._regime_observations: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1000))
        )
        self._consecutive_bad_drifts: Dict[str, int] = defaultdict(int)
        self._cycle_count = 0
        self._drift_count = 0
        self._revert_count = 0
        logger.info("ParameterDriftOptimizer: initialized")

    def register(
        self,
        name: str,
        current: float,
        min_value: float,
        max_value: float,
        learning_rate: float = 0.01,
        description: str = "",
        regime_specific: bool = True,
    ) -> None:
        """Register a parameter for drift optimization."""
        if name in self._params:
            return
        # Sanity bounds
        current = max(min_value, min(current, max_value))
        self._params[name] = ParameterDefinition(
            name=name,
            current_value=current,
            min_value=min_value,
            max_value=max_value,
            initial_value=current,
            learning_rate=learning_rate,
            description=description,
            regime_specific=regime_specific,
        )
        self._history[name] = ParameterDriftHistory(
            name=name,
            best_value_seen=current,
        )
        logger.info(
            "ParameterDriftOptimizer: registered %s (init=%.4f, range=[%.4f, %.4f])",
            name, current, min_value, max_value,
        )

    def get_value(self, name: str, regime: Optional[str] = None) -> Optional[float]:
        """Get the current value of a parameter."""
        param = self._params.get(name)
        if param is None:
            return None
        return param.current_value

    def observe_outcome(
        self,
        parameter_values: Dict[str, float],
        pnl_aud: float,
        regime: str = "NORMAL",
        sample_weight: float = 1.0,
    ) -> None:
        """Record an outcome correlated with parameter values."""
        ts = time.time()
        for name, value in parameter_values.items():
            if name not in self._params:
                continue
            obs = ParameterObservation(
                timestamp=ts,
                parameter_name=name,
                parameter_value=value,
                pnl_contribution=pnl_aud,
                regime=regime,
                sample_weight=sample_weight,
            )
            self._history[name].observations.append(obs)
            self._regime_observations[regime][name].append(obs)

            # Update best-value tracker
            hist = self._history[name]
            if pnl_aud > hist.best_pnl_at_value:
                hist.best_pnl_at_value = pnl_aud
                hist.best_value_seen = value

    def compute_drifts(self, regime: Optional[str] = None) -> Dict[str, float]:
        """
        Compute new parameter values via gradient on observed outcomes.
        Returns dict of {parameter_name: new_value}.
        """
        self._cycle_count += 1
        new_values: Dict[str, float] = {}

        for name, param in self._params.items():
            history = self._history[name]
            observations = list(history.observations)

            # Need minimum observations
            if len(observations) < self.MIN_OBSERVATIONS_FOR_DRIFT:
                continue

            # Use regime-specific observations if available
            if regime and param.regime_specific:
                regime_obs = list(self._regime_observations.get(regime, {}).get(name, []))
                if len(regime_obs) >= self.MIN_OBSERVATIONS_FOR_DRIFT:
                    observations = regime_obs

            # Compute gradient: did higher values lead to higher P&L?
            gradient = self._compute_gradient(observations)

            if abs(gradient) < 1e-6:
                continue

            # Compute drift
            drift = self._compute_drift(param, gradient, observations)

            if abs(drift) < 1e-9:
                continue

            new_value = param.current_value + drift
            new_value = max(param.min_value, min(new_value, param.max_value))

            # Record proposal
            new_values[name] = new_value

        return new_values

    def apply_drifts(self, new_values: Dict[str, float]) -> int:
        """Apply computed drifts to current values. Returns count applied."""
        applied = 0
        for name, new_value in new_values.items():
            param = self._params.get(name)
            if param is None:
                continue
            if abs(new_value - param.current_value) < 1e-9:
                continue
            old_value = param.current_value
            param.current_value = new_value

            history = self._history[name]
            history.history.append({
                "timestamp": time.time(),
                "old": old_value,
                "new": new_value,
                "direction": new_value - old_value,
            })
            history.last_drift_direction = new_value - old_value
            history.drift_count += 1
            self._drift_count += 1
            applied += 1
            logger.debug(
                "ParameterDriftOptimizer: %s drifted %.6f → %.6f",
                name, old_value, new_value,
            )
        return applied

    def check_for_reverts(self, recent_pnl_per_param: Dict[str, float]) -> List[str]:
        """
        Check if any drifted parameters are hurting performance.
        Returns list of parameters that were reverted.
        """
        reverted: List[str] = []
        for name, recent_pnl in recent_pnl_per_param.items():
            if recent_pnl >= 0:
                self._consecutive_bad_drifts[name] = 0
                continue

            self._consecutive_bad_drifts[name] += 1
            if self._consecutive_bad_drifts[name] >= self.REVERT_THRESHOLD_CYCLES:
                # Revert to initial value
                param = self._params.get(name)
                if param is None:
                    continue
                old = param.current_value
                param.current_value = param.initial_value
                self._history[name].revert_count += 1
                self._consecutive_bad_drifts[name] = 0
                self._revert_count += 1
                reverted.append(name)
                logger.warning(
                    "ParameterDriftOptimizer: REVERTED %s from %.6f → %.6f (5 bad cycles)",
                    name, old, param.initial_value,
                )
        return reverted

    def _compute_gradient(self, observations: List[ParameterObservation]) -> float:
        """
        Compute correlation-based gradient: does increasing param → higher P&L?
        Returns positive gradient → higher param is better.
        """
        if len(observations) < 5:
            return 0.0

        values = [o.parameter_value for o in observations]
        pnls = [o.pnl_contribution for o in observations]
        weights = [o.sample_weight for o in observations]

        n = len(values)
        sum_w = sum(weights)
        if sum_w < 1e-9:
            return 0.0

        # Weighted means
        mean_v = sum(v * w for v, w in zip(values, weights)) / sum_w
        mean_p = sum(p * w for p, w in zip(pnls, weights)) / sum_w

        # Weighted covariance and variance
        cov = sum(w * (v - mean_v) * (p - mean_p) for v, p, w in zip(values, pnls, weights)) / sum_w
        var = sum(w * (v - mean_v) ** 2 for v, w in zip(values, weights)) / sum_w

        if var < 1e-12:
            return 0.0

        # Correlation as gradient signal (sign indicates direction)
        gradient = cov / math.sqrt(var)
        return gradient

    def _compute_drift(
        self,
        param: ParameterDefinition,
        gradient: float,
        observations: List[ParameterObservation],
    ) -> float:
        """Compute the drift step for a parameter."""
        # Confidence factor — small sample = small drift
        n_obs = len(observations)
        confidence = min(1.0, n_obs / 100.0)

        # Direction is sign of gradient
        direction = 1.0 if gradient > 0 else -1.0

        # Magnitude is bounded by max drift fraction
        max_step = param.current_value * self.MAX_DRIFT_FRACTION
        max_step = min(max_step, abs(param.max_value - param.min_value) * 0.05)

        drift = direction * max_step * confidence * param.learning_rate * 100  # scale lr

        return drift

    def get_parameter_state(self, name: str) -> Optional[Dict[str, Any]]:
        param = self._params.get(name)
        history = self._history.get(name)
        if not param or not history:
            return None
        return {
            "name": name,
            "current_value": param.current_value,
            "initial_value": param.initial_value,
            "min": param.min_value,
            "max": param.max_value,
            "drift_count": history.drift_count,
            "revert_count": history.revert_count,
            "best_value_seen": history.best_value_seen,
            "best_pnl_at_value": history.best_pnl_at_value,
            "observation_count": len(history.observations),
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "registered_params": len(self._params),
            "cycle_count": self._cycle_count,
            "total_drifts": self._drift_count,
            "total_reverts": self._revert_count,
            "params": {
                name: {
                    "current": p.current_value,
                    "initial": p.initial_value,
                    "drifted_pct": (p.current_value - p.initial_value) / max(p.initial_value, 1e-9) * 100,
                }
                for name, p in self._params.items()
            },
        }
