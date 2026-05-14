"""
Adaptation Health Monitor — verifies adaptations actually help.

Without this safety net, the parameter drift / gate adaptation could easily
overfit to recent noise and degrade performance. This module:

  1. Snapshots P&L before each adaptation cycle
  2. Measures P&L for N cycles after the adaptation
  3. Computes whether the adaptation helped or hurt
  4. If the cumulative effect is negative for K cycles, REVERTS all changes
  5. Tracks adaptation effectiveness by component

The goal: make the adaptation system anti-fragile. It should converge toward
better parameters but never get stuck in a worse state.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AdaptationSnapshot:
    """A snapshot of system state before/after an adaptation."""
    timestamp: float
    cycle: int
    portfolio_value_aud: float
    cumulative_pnl: float
    win_rate: float
    sharpe: float
    parameter_state: Dict[str, float] = field(default_factory=dict)
    gate_state: Dict[str, float] = field(default_factory=dict)


@dataclass
class AdaptationOutcome:
    """Records the effect of one adaptation cycle."""
    timestamp: float
    cycle: int
    adaptations_applied: int
    pre_pnl: float
    post_pnl: float
    measurement_cycles: int
    delta_pnl: float
    helped: bool
    reverted: bool = False


class AdaptationHealthMonitor:
    """
    Monitors whether parameter and gate adaptations help or hurt.

    Usage::

        monitor = AdaptationHealthMonitor(measurement_cycles=50)

        # Before adapting:
        monitor.before_adaptation(portfolio_value=1000.0, cycle=42)

        # After applying adaptations:
        monitor.after_adaptation(adaptations_applied=5)

        # Each cycle, update with current P&L:
        monitor.update(portfolio_value=1010.0, cycle=43)

        # When measurement window completes:
        if monitor.should_revert():
            # Revert all recent adaptations
            ...
    """

    # Cycles to measure adaptation effect
    DEFAULT_MEASUREMENT_CYCLES = 100

    # If N consecutive adaptations hurt, reduce adaptation rate
    HURT_STREAK_THRESHOLD = 3

    # If hurt streak hits this, revert all recent adaptations
    REVERT_THRESHOLD = 5

    # Minimum P&L delta to consider "hurt" (avoids noise triggers)
    SIGNIFICANCE_THRESHOLD_AUD = 1.0

    def __init__(
        self,
        measurement_cycles: int = DEFAULT_MEASUREMENT_CYCLES,
    ) -> None:
        self._measurement_cycles = measurement_cycles
        self._snapshots: deque[AdaptationSnapshot] = deque(maxlen=100)
        self._outcomes: deque[AdaptationOutcome] = deque(maxlen=1000)
        self._pre_snapshot: Optional[AdaptationSnapshot] = None
        self._cycles_since_adaptation: int = 0
        self._current_adaptations_count: int = 0
        self._hurt_streak: int = 0
        self._help_streak: int = 0
        self._total_helped: int = 0
        self._total_hurt: int = 0
        self._reverts_triggered: int = 0
        logger.info(
            "AdaptationHealthMonitor: initialized (measurement=%d cycles)",
            measurement_cycles,
        )

    @property
    def is_measuring(self) -> bool:
        return self._pre_snapshot is not None

    def before_adaptation(
        self,
        portfolio_value: float,
        cycle: int,
        cumulative_pnl: float = 0.0,
        win_rate: float = 0.5,
        sharpe: float = 0.0,
        parameter_state: Optional[Dict[str, float]] = None,
        gate_state: Optional[Dict[str, float]] = None,
    ) -> None:
        """Snapshot current state right before applying adaptations."""
        self._pre_snapshot = AdaptationSnapshot(
            timestamp=time.time(),
            cycle=cycle,
            portfolio_value_aud=portfolio_value,
            cumulative_pnl=cumulative_pnl,
            win_rate=win_rate,
            sharpe=sharpe,
            parameter_state=parameter_state or {},
            gate_state=gate_state or {},
        )
        self._cycles_since_adaptation = 0

    def after_adaptation(self, adaptations_applied: int) -> None:
        """Record that adaptations were just applied."""
        self._current_adaptations_count = adaptations_applied
        if self._pre_snapshot is not None:
            self._snapshots.append(self._pre_snapshot)

    def update(
        self,
        portfolio_value: float,
        cycle: int,
        cumulative_pnl: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Called every cycle to track adaptation effect.
        Returns dict with measurement state.
        """
        result = {
            "is_measuring": self.is_measuring,
            "cycles_since_adaptation": self._cycles_since_adaptation,
            "should_evaluate": False,
        }

        if not self.is_measuring:
            return result

        self._cycles_since_adaptation += 1

        if self._cycles_since_adaptation >= self._measurement_cycles:
            outcome = self._evaluate_adaptation(portfolio_value, cumulative_pnl, cycle)
            result["should_evaluate"] = True
            result["outcome"] = outcome
            self._pre_snapshot = None
            self._cycles_since_adaptation = 0

        return result

    def _evaluate_adaptation(
        self,
        portfolio_value: float,
        cumulative_pnl: float,
        cycle: int,
    ) -> Dict[str, Any]:
        """Compute whether the recent adaptation helped or hurt."""
        if self._pre_snapshot is None:
            return {"helped": True, "delta_pnl": 0.0}

        delta_pnl = cumulative_pnl - self._pre_snapshot.cumulative_pnl
        helped = delta_pnl > self.SIGNIFICANCE_THRESHOLD_AUD

        outcome = AdaptationOutcome(
            timestamp=time.time(),
            cycle=cycle,
            adaptations_applied=self._current_adaptations_count,
            pre_pnl=self._pre_snapshot.cumulative_pnl,
            post_pnl=cumulative_pnl,
            measurement_cycles=self._cycles_since_adaptation,
            delta_pnl=delta_pnl,
            helped=helped,
        )
        self._outcomes.append(outcome)

        if helped:
            self._help_streak += 1
            self._hurt_streak = 0
            self._total_helped += 1
            logger.info(
                "AdaptationHealthMonitor: adaptation HELPED (Δpnl=$%.2f, streak=%d)",
                delta_pnl, self._help_streak,
            )
        elif delta_pnl < -self.SIGNIFICANCE_THRESHOLD_AUD:
            self._hurt_streak += 1
            self._help_streak = 0
            self._total_hurt += 1
            logger.warning(
                "AdaptationHealthMonitor: adaptation HURT (Δpnl=$%.2f, streak=%d)",
                delta_pnl, self._hurt_streak,
            )
        # else: noise, neither helped nor hurt

        return {
            "helped": helped,
            "delta_pnl": delta_pnl,
            "hurt_streak": self._hurt_streak,
            "help_streak": self._help_streak,
            "should_revert": self._hurt_streak >= self.REVERT_THRESHOLD,
        }

    def should_revert(self) -> bool:
        """True if the system should revert recent adaptations."""
        return self._hurt_streak >= self.REVERT_THRESHOLD

    def should_throttle(self) -> bool:
        """True if adaptation rate should be reduced."""
        return self._hurt_streak >= self.HURT_STREAK_THRESHOLD

    def mark_reverted(self) -> None:
        """Reset hurt streak after a revert."""
        self._reverts_triggered += 1
        self._hurt_streak = 0
        self._pre_snapshot = None
        logger.warning(
            "AdaptationHealthMonitor: REVERT TRIGGERED (total=%d)",
            self._reverts_triggered,
        )

    def get_recent_outcomes(self, n: int = 20) -> List[Dict[str, Any]]:
        recent = list(self._outcomes)[-n:]
        return [
            {
                "timestamp": o.timestamp,
                "cycle": o.cycle,
                "adaptations_applied": o.adaptations_applied,
                "delta_pnl": o.delta_pnl,
                "helped": o.helped,
                "reverted": o.reverted,
            }
            for o in reversed(recent)
        ]

    def get_effectiveness(self) -> float:
        """Return fraction of adaptations that helped (0.0 to 1.0)."""
        total = self._total_helped + self._total_hurt
        if total == 0:
            return 0.5  # neutral
        return self._total_helped / total

    def snapshot(self) -> Dict[str, Any]:
        return {
            "is_measuring": self.is_measuring,
            "cycles_since_adaptation": self._cycles_since_adaptation,
            "measurement_cycles": self._measurement_cycles,
            "total_helped": self._total_helped,
            "total_hurt": self._total_hurt,
            "effectiveness": round(self.get_effectiveness(), 3),
            "hurt_streak": self._hurt_streak,
            "help_streak": self._help_streak,
            "reverts_triggered": self._reverts_triggered,
        }
