"""
MomentReadiness — single authoritative answer to "is RIGHT NOW a good time to enter?"

Synthesises health_score, timing_action, signal_conviction, regime_compatibility,
and learned win rates (from OutcomeCorrelator) into a 0-100 composite score.

Five readiness labels:
  PRIME    85-100  All systems aligned — trade full size
  READY    65-84   Good conditions — trade normal size
  CAUTIOUS 45-64   Mixed signals — trade reduced size
  WAIT     25-44   Conditions poor but improving — queue signal, retry soon
  STANDBY  0-24    Conditions bad — discard signal

Special overrides (hard rules applied after scoring):
  pre_hedge_signal=True → cap at CAUTIOUS regardless of score
  var_breach=True        → cap at WAIT
  timing_block=True      → force STANDBY
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from monitoring.outcome_correlator import OutcomeCorrelator

logger = logging.getLogger(__name__)


class ReadinessLabel(str, Enum):
    PRIME = "PRIME"
    READY = "READY"
    CAUTIOUS = "CAUTIOUS"
    WAIT = "WAIT"
    STANDBY = "STANDBY"

    def __str__(self) -> str:
        return self.value


_SIZE_MULTIPLIERS = {
    ReadinessLabel.PRIME: 1.00,
    ReadinessLabel.READY: 0.85,
    ReadinessLabel.CAUTIOUS: 0.60,
    ReadinessLabel.WAIT: 0.00,
    ReadinessLabel.STANDBY: 0.00,
}

_ENTRY_ALLOWED = {
    ReadinessLabel.PRIME: True,
    ReadinessLabel.READY: True,
    ReadinessLabel.CAUTIOUS: True,
    ReadinessLabel.WAIT: False,
    ReadinessLabel.STANDBY: False,
}

# Score thresholds (inclusive lower bound)
_LABEL_THRESHOLDS = [
    (85.0, ReadinessLabel.PRIME),
    (65.0, ReadinessLabel.READY),
    (45.0, ReadinessLabel.CAUTIOUS),
    (25.0, ReadinessLabel.WAIT),
    (0.0, ReadinessLabel.STANDBY),
]

# Component weights (must sum to 1.0)
_W_HEALTH = 0.30
_W_TIMING = 0.25
_W_SIGNAL = 0.20
_W_REGIME = 0.15
_W_LEARNED = 0.10


@dataclass
class ReadinessState:
    score: float
    label: ReadinessLabel
    entry_allowed: bool
    size_multiplier: float
    queue_signal: bool          # True only for WAIT
    insights: List[str]
    component_scores: Dict[str, float]
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "label": str(self.label.value),
            "entry_allowed": self.entry_allowed,
            "size_multiplier": self.size_multiplier,
            "queue_signal": self.queue_signal,
            "insights": self.insights,
            "component_scores": {k: round(v, 2) for k, v in self.component_scores.items()},
            "ts": self.ts,
        }


class MomentReadiness:
    """Evaluates whether right now is a good time to enter a trade."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        outcome_correlator: Optional["OutcomeCorrelator"] = None,
    ) -> None:
        self.outcome_correlator = outcome_correlator

        cfg = (config or {}).get("self_directed_trading") or {}
        self._prime_threshold = float(cfg.get("mr_prime_threshold", 85))
        self._ready_threshold = float(cfg.get("mr_ready_threshold", 65))
        self._cautious_threshold = float(cfg.get("mr_cautious_threshold", 45))
        self._wait_threshold = float(cfg.get("mr_wait_threshold", 25))

        # Rebuild threshold table from config
        self._thresholds = [
            (self._prime_threshold, ReadinessLabel.PRIME),
            (self._ready_threshold, ReadinessLabel.READY),
            (self._cautious_threshold, ReadinessLabel.CAUTIOUS),
            (self._wait_threshold, ReadinessLabel.WAIT),
            (0.0, ReadinessLabel.STANDBY),
        ]

        self._last_state: Optional[ReadinessState] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        advisory: Dict[str, Any],
        strategy_name: str = "",
    ) -> ReadinessState:
        """
        Compute the current MomentReadiness from the advisory dict.

        Returns a ReadinessState. Stores result as self._last_state.
        """
        advisory = advisory or {}
        insights: List[str] = []

        # ------ Compute individual components (0-100 each) ------
        h_score, h_insight = self._health_component(advisory)
        t_score, t_insight = self._timing_component(advisory)
        s_score, s_insight = self._signal_component(advisory)
        r_score, r_insight = self._regime_component(advisory, strategy_name)
        l_score, l_insight = self._learned_component(advisory, strategy_name)

        if h_insight:
            insights.append(h_insight)
        if t_insight:
            insights.append(t_insight)
        if s_insight:
            insights.append(s_insight)
        if r_insight:
            insights.append(r_insight)
        if l_insight:
            insights.append(l_insight)

        # ------ Composite score ------
        raw_score = (
            h_score * _W_HEALTH
            + t_score * _W_TIMING
            + s_score * _W_SIGNAL
            + r_score * _W_REGIME
            + l_score * _W_LEARNED
        )
        score = max(0.0, min(100.0, raw_score))

        # ------ Hard overrides ------
        # Check for override signals in advisory
        timing_block = self._is_timing_block(advisory)
        var_breach = self._is_var_breach(advisory)
        pre_hedge = self._is_pre_hedge(advisory)

        if timing_block:
            score = min(score, 10.0)  # Force STANDBY range
            insights.append("timing_block→STANDBY")
        elif var_breach:
            score = min(score, 44.0)  # Cap at WAIT range
            insights.append("var_breach→WAIT_cap")
        elif pre_hedge:
            score = min(score, 64.0)  # Cap at CAUTIOUS range
            insights.append("pre_hedge→CAUTIOUS_cap")

        # ------ Determine label ------
        label = self._score_to_label(score)

        state = ReadinessState(
            score=round(score, 2),
            label=label,
            entry_allowed=_ENTRY_ALLOWED[label],
            size_multiplier=_SIZE_MULTIPLIERS[label],
            queue_signal=(label == ReadinessLabel.WAIT),
            insights=insights,
            component_scores={
                "health": round(h_score, 2),
                "timing": round(t_score, 2),
                "signal": round(s_score, 2),
                "regime": round(r_score, 2),
                "learned": round(l_score, 2),
            },
        )
        self._last_state = state
        return state

    def snapshot(self) -> Dict[str, Any]:
        """Return the last computed state as a dict (for advisory injection)."""
        if self._last_state is None:
            return {
                "score": 50.0,
                "label": "CAUTIOUS",
                "entry_allowed": True,
                "size_multiplier": 0.60,
                "queue_signal": False,
                "insights": ["no_evaluation_yet"],
                "component_scores": {},
            }
        return self._last_state.to_dict()

    # ------------------------------------------------------------------
    # Component scorers (each returns 0-100 score + optional insight str)
    # ------------------------------------------------------------------

    def _health_component(self, advisory: Dict[str, Any]) -> tuple[float, str]:
        """Maps health_score 0-100 directly. CRITICAL label → hard cap at 20."""
        hs = advisory.get("health_score") or {}
        raw = float(hs.get("score", 50.0) or 50.0)
        label = str(hs.get("label") or "").upper()

        score = max(0.0, min(100.0, raw))

        if label == "CRITICAL":
            score = min(score, 20.0)
            return score, f"health=CRITICAL({raw:.0f})→cap20"
        if label == "POOR":
            score = min(score, 40.0)
            return score, f"health=POOR({raw:.0f})→cap40"

        insight = f"health={label}({raw:.0f})" if label else ""
        return score, insight

    def _timing_component(self, advisory: Dict[str, Any]) -> tuple[float, str]:
        """Maps timing action to a 0-100 score."""
        ti = advisory.get("timing_intelligence") or {}
        action = str(ti.get("action") or "OK").upper()

        mapping = {
            "IDEAL": 100.0,
            "OK": 75.0,
            "DEFER": 40.0,
            "BLOCK": 10.0,
        }
        score = mapping.get(action, 75.0)
        return score, f"timing={action}"

    def _signal_component(self, advisory: Dict[str, Any]) -> tuple[float, str]:
        """Maps signal conviction 0-1 → 0-100."""
        sig = advisory.get("signal_intelligence") or {}
        conviction = float(sig.get("conviction", 0.5) or 0.5)
        conviction = max(0.0, min(1.0, conviction))
        score = conviction * 100.0
        return score, f"conviction={conviction:.2f}"

    def _regime_component(
        self, advisory: Dict[str, Any], strategy_name: str
    ) -> tuple[float, str]:
        """Maps regime compatibility 0-1 → 0-100."""
        # Try strategy_regime_matrix fitness for this strategy
        srm = advisory.get("strategy_regime_matrix") or {}
        fitness_map = srm.get("fitness") or {}
        if strategy_name and strategy_name in fitness_map:
            # fitness is in [0.25, 1.5] → normalise to 0-100
            # fitness=1.0 → 67, fitness=1.5 → 100, fitness=0.25 → 17
            fitness = float(fitness_map[strategy_name] or 1.0)
            score = max(0.0, min(100.0, (fitness / 1.5) * 100.0))
            return score, f"regime_fitness={fitness:.2f}"

        # Fall back to execution_quality_gate regime_compatibility
        eqg = advisory.get("execution_quality_gate") or {}
        compat = float(eqg.get("regime_compatibility", 0.6) or 0.6)
        compat = max(0.0, min(1.0, compat))
        score = compat * 100.0
        return score, f"regime_compat={compat:.2f}"

    def _learned_component(
        self, advisory: Dict[str, Any], strategy_name: str
    ) -> tuple[float, str]:
        """
        Returns learned win rate as a 0-100 score.
        Falls back to neutral 60 if OutcomeCorrelator has insufficient data.
        """
        neutral = 60.0

        if self.outcome_correlator is None:
            return neutral, "learned=neutral(no_correlator)"

        # Extract condition keys
        hs = advisory.get("health_score") or {}
        health_label = str(hs.get("label") or "UNKNOWN")

        regime_adv = advisory.get("regime_parameters") or advisory.get("strategy_regime_matrix") or {}
        regime = str(regime_adv.get("regime") or "UNKNOWN")

        ti = advisory.get("timing_intelligence") or {}
        timing_action = str(ti.get("action") or "OK").upper()
        if timing_action not in ("IDEAL", "OK", "DEFER", "BLOCK"):
            timing_action = "OK"

        try:
            win_rate = self.outcome_correlator.win_rate_for_conditions(
                health_label, regime, timing_action
            )
        except Exception as exc:
            logger.debug("MomentReadiness._learned_component error: %s", exc)
            win_rate = None

        if win_rate is None:
            return neutral, "learned=neutral(insufficient_data)"

        score = win_rate * 100.0
        return score, f"learned_wr={win_rate:.2%}"

    # ------------------------------------------------------------------
    # Override detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_timing_block(advisory: Dict[str, Any]) -> bool:
        ti = advisory.get("timing_intelligence") or {}
        action = str(ti.get("action") or "").upper()
        return action == "BLOCK"

    @staticmethod
    def _is_var_breach(advisory: Dict[str, Any]) -> bool:
        # Check intraday_var advisory
        ivar = advisory.get("intraday_var") or advisory.get("adaptive_risk") or {}
        if bool(ivar.get("breach", False)):
            return True
        # Check trade_gate advisory
        tg = advisory.get("trade_gate") or {}
        decision = str(tg.get("decision") or "").upper()
        return decision == "HALT"

    @staticmethod
    def _is_pre_hedge(advisory: Dict[str, Any]) -> bool:
        rt = advisory.get("regime_transition") or {}
        return bool(rt.get("pre_hedge_signal", False))

    # ------------------------------------------------------------------
    # Label from score
    # ------------------------------------------------------------------

    def _score_to_label(self, score: float) -> ReadinessLabel:
        for threshold, label in self._thresholds:
            if score >= threshold:
                return label
        return ReadinessLabel.STANDBY
