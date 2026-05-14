"""
Adaptive Gate Manager — adapts intelligence gate thresholds and weights based on outcomes.

ARGUS has 75+ intelligence gates that decide whether/how to size each trade.
Each gate has thresholds (e.g. confidence > 0.55, drawdown < 5%, etc.).

This module observes gate decisions vs actual outcomes:
  - Did the gate block trades that would have been profitable? → loosen
  - Did the gate let through trades that lost money? → tighten
  - Was the gate's threshold predictive? → keep adapting

Adapts:
  - Threshold values (move toward optimal)
  - Gate weights in voting (boost predictive gates, demote noisy ones)
  - Gate ordering (cheap gates first, expensive gates last)
  - Gate disabling (turn off gates that hurt performance)
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GateDefinition:
    """Definition of an adaptive gate with bounds."""
    name: str
    threshold: float
    min_threshold: float
    max_threshold: float
    initial_threshold: float
    weight: float = 1.0
    enabled: bool = True
    description: str = ""


@dataclass
class GateOutcome:
    """One observation: gate decision + actual trade outcome."""
    timestamp: float
    gate_name: str
    decision: str           # "pass", "block", "reduce"
    threshold_used: float
    pnl_aud: float
    would_have_been_profitable: bool


@dataclass
class GatePerformance:
    """Tracks how well a gate performs over time."""
    name: str
    decisions: deque = field(default_factory=lambda: deque(maxlen=1000))
    total_passed: int = 0
    total_blocked: int = 0
    total_reduced: int = 0
    correct_blocks: int = 0      # blocks that prevented losses
    bad_blocks: int = 0          # blocks that prevented profits
    correct_passes: int = 0      # passes that led to wins
    bad_passes: int = 0          # passes that led to losses
    total_pnl_attributed: float = 0.0


class AdaptiveGateManager:
    """
    Adapts intelligence gate thresholds, weights, and enabled state.

    Usage::

        mgr = AdaptiveGateManager()
        mgr.register_gate("confidence_gate", current=0.55, min=0.40, max=0.75)

        # On each gate decision:
        mgr.record_decision(
            gate_name="confidence_gate",
            decision="pass",
            threshold_used=0.55,
        )

        # On trade outcome:
        mgr.record_outcome(
            gate_name="confidence_gate",
            pnl_aud=15.0,
            would_have_been_profitable=True,
        )

        # Periodically adapt:
        new_thresholds = mgr.compute_adaptations()
    """

    MIN_DECISIONS_FOR_ADAPT = 30
    MAX_THRESHOLD_DRIFT_PCT = 0.05  # max 5% change per adaptation

    def __init__(self) -> None:
        self._gates: Dict[str, GateDefinition] = {}
        self._performance: Dict[str, GatePerformance] = {}
        self._cycle_count = 0
        self._adaptation_count = 0
        self._disable_count = 0
        logger.info("AdaptiveGateManager: initialized")

    def register_gate(
        self,
        name: str,
        current: float,
        min_value: float,
        max_value: float,
        weight: float = 1.0,
        description: str = "",
    ) -> None:
        """Register an adaptive gate."""
        if name in self._gates:
            return
        current = max(min_value, min(current, max_value))
        self._gates[name] = GateDefinition(
            name=name,
            threshold=current,
            min_threshold=min_value,
            max_threshold=max_value,
            initial_threshold=current,
            weight=weight,
            enabled=True,
            description=description,
        )
        self._performance[name] = GatePerformance(name=name)
        logger.info(
            "AdaptiveGateManager: registered %s (init=%.4f, range=[%.4f, %.4f])",
            name, current, min_value, max_value,
        )

    def get_threshold(self, name: str) -> Optional[float]:
        gate = self._gates.get(name)
        return gate.threshold if gate else None

    def get_weight(self, name: str) -> Optional[float]:
        gate = self._gates.get(name)
        return gate.weight if gate else None

    def is_enabled(self, name: str) -> bool:
        gate = self._gates.get(name)
        return gate.enabled if gate else False

    def record_decision(
        self,
        gate_name: str,
        decision: str,
        threshold_used: Optional[float] = None,
    ) -> None:
        """Record that a gate made a decision (before outcome is known)."""
        if gate_name not in self._gates:
            return
        perf = self._performance[gate_name]
        if decision == "pass":
            perf.total_passed += 1
        elif decision == "block":
            perf.total_blocked += 1
        elif decision == "reduce":
            perf.total_reduced += 1

    def record_outcome(
        self,
        gate_name: str,
        decision: str,
        pnl_aud: float,
        would_have_been_profitable: Optional[bool] = None,
    ) -> None:
        """Record the actual outcome of a trade that this gate evaluated."""
        if gate_name not in self._gates:
            return
        perf = self._performance[gate_name]
        gate = self._gates[gate_name]

        outcome = GateOutcome(
            timestamp=time.time(),
            gate_name=gate_name,
            decision=decision,
            threshold_used=gate.threshold,
            pnl_aud=pnl_aud,
            would_have_been_profitable=bool(would_have_been_profitable) if would_have_been_profitable is not None else (pnl_aud > 0),
        )
        perf.decisions.append(outcome)
        perf.total_pnl_attributed += pnl_aud

        # Classify the gate's decision quality
        if decision == "pass":
            if pnl_aud > 0:
                perf.correct_passes += 1
            else:
                perf.bad_passes += 1
        elif decision == "block":
            # Use the would_have_been_profitable hint
            if would_have_been_profitable:
                perf.bad_blocks += 1
            else:
                perf.correct_blocks += 1

    def compute_adaptations(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute new threshold + weight + enabled state for each gate.
        Returns dict of {gate_name: {"threshold": x, "weight": y, "enabled": z}}.
        """
        self._cycle_count += 1
        adaptations: Dict[str, Dict[str, Any]] = {}

        for name, gate in self._gates.items():
            perf = self._performance[name]
            total = perf.total_passed + perf.total_blocked

            if total < self.MIN_DECISIONS_FOR_ADAPT:
                continue

            # Compute pass quality (correct passes / total passes)
            pass_quality = (
                perf.correct_passes / max(perf.correct_passes + perf.bad_passes, 1)
            )
            # Compute block quality (correct blocks / total blocks)
            block_quality = (
                perf.correct_blocks / max(perf.correct_blocks + perf.bad_blocks, 1)
            )

            # Average gate accuracy
            gate_accuracy = (pass_quality + block_quality) / 2

            adaptation = {}

            # If pass quality is poor (most passes lose), tighten threshold
            if perf.total_passed >= 20 and pass_quality < 0.40:
                new_threshold = self._tighten(gate)
                adaptation["threshold"] = new_threshold
                adaptation["reason"] = f"pass_quality={pass_quality:.2f} too low"

            # If block quality is poor (blocks miss profits), loosen threshold
            elif perf.total_blocked >= 20 and block_quality < 0.40:
                new_threshold = self._loosen(gate)
                adaptation["threshold"] = new_threshold
                adaptation["reason"] = f"block_quality={block_quality:.2f} too low"

            # Update weight based on overall accuracy
            new_weight = self._compute_new_weight(gate.weight, gate_accuracy)
            if abs(new_weight - gate.weight) > 0.05:
                adaptation["weight"] = new_weight

            # Disable if consistently bad
            if gate_accuracy < 0.30 and total > 100:
                adaptation["enabled"] = False
                adaptation["disable_reason"] = f"accuracy={gate_accuracy:.2f} consistently bad"

            if adaptation:
                adaptations[name] = adaptation

        return adaptations

    def apply_adaptations(self, adaptations: Dict[str, Dict[str, Any]]) -> int:
        """Apply computed adaptations to gates. Returns count applied."""
        applied = 0
        for name, change in adaptations.items():
            gate = self._gates.get(name)
            if gate is None:
                continue

            if "threshold" in change:
                old = gate.threshold
                gate.threshold = max(
                    gate.min_threshold,
                    min(change["threshold"], gate.max_threshold),
                )
                self._adaptation_count += 1
                applied += 1
                logger.info(
                    "AdaptiveGateManager: %s threshold %.4f → %.4f (%s)",
                    name, old, gate.threshold, change.get("reason", ""),
                )

            if "weight" in change:
                gate.weight = max(0.0, min(change["weight"], 2.0))
                applied += 1

            if "enabled" in change and not change["enabled"]:
                gate.enabled = False
                self._disable_count += 1
                logger.warning(
                    "AdaptiveGateManager: DISABLED %s — %s",
                    name, change.get("disable_reason", ""),
                )

        return applied

    def _tighten(self, gate: GateDefinition) -> float:
        """Increase threshold (more selective)."""
        delta = gate.threshold * self.MAX_THRESHOLD_DRIFT_PCT
        return min(gate.threshold + delta, gate.max_threshold)

    def _loosen(self, gate: GateDefinition) -> float:
        """Decrease threshold (less selective)."""
        delta = gate.threshold * self.MAX_THRESHOLD_DRIFT_PCT
        return max(gate.threshold - delta, gate.min_threshold)

    def _compute_new_weight(self, current_weight: float, accuracy: float) -> float:
        """Boost weight if accurate, demote if inaccurate."""
        # Scale: 0.5 accuracy → weight 1.0; 0.8 accuracy → weight 1.5; 0.3 → weight 0.5
        target_weight = 0.5 + (accuracy * 1.5)
        # Smooth: only move 20% toward target
        return current_weight + 0.2 * (target_weight - current_weight)

    def reset_gate(self, name: str) -> bool:
        """Revert a gate to its initial state."""
        gate = self._gates.get(name)
        if gate is None:
            return False
        gate.threshold = gate.initial_threshold
        gate.weight = 1.0
        gate.enabled = True
        return True

    def get_gate_performance(self, name: str) -> Optional[Dict[str, Any]]:
        gate = self._gates.get(name)
        perf = self._performance.get(name)
        if not gate or not perf:
            return None

        total = perf.total_passed + perf.total_blocked
        accuracy = 0.0
        if total > 0:
            correct = perf.correct_passes + perf.correct_blocks
            accuracy = correct / total

        return {
            "name": name,
            "threshold": gate.threshold,
            "weight": gate.weight,
            "enabled": gate.enabled,
            "total_decisions": total,
            "passed": perf.total_passed,
            "blocked": perf.total_blocked,
            "accuracy": accuracy,
            "total_pnl": perf.total_pnl_attributed,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "registered_gates": len(self._gates),
            "enabled_gates": sum(1 for g in self._gates.values() if g.enabled),
            "cycle_count": self._cycle_count,
            "total_adaptations": self._adaptation_count,
            "total_disabled": self._disable_count,
        }
