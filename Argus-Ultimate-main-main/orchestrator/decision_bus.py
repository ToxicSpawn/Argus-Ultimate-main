"""
orchestrator/decision_bus.py
=============================
Blackboard architecture for agent coordination.

All agents communicate through a shared Decision Bus:
  - Perception agents post Observations
  - Reasoning agents post Hypotheses
  - Acting agents receive Decisions and return Actions
  - A ConflictResolver handles disagreements

This enables emergent coordination without central planning.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ObservationType(Enum):
    """Types of observations agents can make."""
    REGIME          = "regime"
    VOLATILITY      = "volatility"
    ORDER_FLOW      = "order_flow"
    SENTIMENT       = "sentiment"
    LIQUIDITY       = "liquidity"
    CORRELATION     = "correlation"
    ANOMALY         = "anomaly"
    PREDICTION      = "prediction"
    HEALTH          = "health"


class HypothesisType(Enum):
    """Types of hypotheses reasoning agents can form."""
    CAUSAL          = "causal"          # "X caused Y"
    COUNTERFACTUAL  = "counterfactual"  # "If X, then Y"
    PREDICTIVE      = "predictive"      # "Y will happen"
    DIAGNOSTIC      = "diagnostic"      # "The problem is X"
    STRATEGIC       = "strategic"       # "We should do X"


class DecisionType(Enum):
    """Types of decisions."""
    EXECUTE         = "execute"         # Place an order
    ADJUST          = "adjust"          # Modify existing order
    CANCEL          = "cancel"          # Cancel order
    HEDGE           = "hedge"           # Place hedge
    REALLOCATE      = "reallocate"      # Rebalance capital
    PAUSE           = "pause"           # Pause trading
    RESUME          = "resume"          # Resume trading


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """An observation from a perception agent."""
    agent           : str
    type            : ObservationType
    data            : Dict[str, Any]
    confidence      : float  # 0-1
    timestamp       : float  = field(default_factory=time.time)
    metadata        : Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent"     : self.agent,
            "type"      : self.type.value,
            "data"      : self.data,
            "confidence": self.confidence,
            "timestamp" : self.timestamp,
        }


@dataclass
class Hypothesis:
    """A hypothesis from a reasoning agent."""
    agent           : str
    type            : HypothesisType
    description     : str
    evidence        : List[str]    # observation IDs supporting this
    confidence      : float        # 0-1
    implications    : Dict[str, Any] = field(default_factory=dict)
    timestamp       : float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent"       : self.agent,
            "type"        : self.type.value,
            "description" : self.description,
            "evidence"    : self.evidence,
            "confidence"  : self.confidence,
            "implications": self.implications,
        }


@dataclass
class Decision:
    """A coordinated decision."""
    type            : DecisionType
    action          : Dict[str, Any]
    reasoning       : List[str]    # supporting hypotheses
    confidence      : float        # 0-1
    timestamp       : float = field(default_factory=time.time)
    metadata        : Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type"      : self.type.value,
            "action"    : self.action,
            "reasoning" : self.reasoning,
            "confidence": self.confidence,
            "timestamp" : self.timestamp,
        }


@dataclass
class Conflict:
    """A conflict between hypotheses or decisions."""
    agent_a         : str
    agent_b         : str
    hypothesis_a    : Hypothesis
    hypothesis_b    : Hypothesis
    resolution      : Optional[str] = None
    winner          : Optional[str] = None


# ---------------------------------------------------------------------------
# Conflict Resolver
# ---------------------------------------------------------------------------

class ConflictResolver:
    """
    Resolves conflicts between agent hypotheses.

    Uses weighted voting based on:
      - Agent historical accuracy
      - Agent confidence
      - Recency of observations
      - Domain expertise match
    """

    def __init__(self) -> None:
        self._agent_weights: Dict[str, float] = {}  # agent -> weight
        self._conflict_history: List[Conflict] = []

    def set_agent_weight(self, agent: str, weight: float) -> None:
        """Set the decision weight for an agent."""
        self._agent_weights[agent] = max(0.0, min(1.0, weight))

    def resolve(self, hypotheses: List[Hypothesis]) -> Optional[Hypothesis]:
        """
        Resolve conflicting hypotheses.

        Returns the winning hypothesis, or None if no consensus.
        """
        if not hypotheses:
            return None

        if len(hypotheses) == 1:
            return hypotheses[0]

        # Group by implication type
        groups: Dict[str, List[Hypothesis]] = {}
        for h in hypotheses:
            key = str(sorted(h.implications.keys()))
            if key not in groups:
                groups[key] = []
            groups[key].append(h)

        # Score each group
        best_group: Optional[str] = None
        best_score = -1.0

        for key, group in groups.items():
            score = self._score_group(group)
            if score > best_score:
                best_score = score
                best_group = key

        if best_group is None:
            return None

        # Return highest-confidence hypothesis from winning group
        winning = max(groups[best_group], key=lambda h: h.confidence)
        return winning

    def _score_group(self, hypotheses: List[Hypothesis]) -> float:
        """Score a group of aligned hypotheses."""
        total = 0.0
        for h in hypotheses:
            weight = self._agent_weights.get(h.agent, 0.5)
            total += h.confidence * weight
        return total

    def record_conflict(self, conflict: Conflict) -> None:
        """Record a conflict for analysis."""
        self._conflict_history.append(conflict)


# ---------------------------------------------------------------------------
# Decision Bus
# ---------------------------------------------------------------------------

class DecisionBus:
    """
    Blackboard architecture for agent coordination.

    All agents post to and read from the bus.
    The bus coordinates, resolves conflicts, and produces decisions.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        # Blackboard
        self._observations: Deque[Observation] = deque(maxlen=10000)
        self._hypotheses  : Dict[str, Hypothesis] = {}  # id -> hypothesis
        self._decisions   : Deque[Decision] = deque(maxlen=5000)

        # Conflict resolver
        self._resolver = ConflictResolver()

        # Agent weights (updated by meta-learner)
        self._agent_weights: Dict[str, float] = {}

        # Callbacks for decision listeners
        self._decision_callbacks: List[Callable[[Decision], None]] = []

        # Statistics
        self._total_observations: int = 0
        self._total_hypotheses: int = 0
        self._total_decisions: int = 0
        self._conflicts_resolved: int = 0

        logger.info("DecisionBus: initialised")

    # ------------------------------------------------------------------ Observation posting

    def post_observation(self, obs: Observation) -> str:
        """
        Post an observation from a perception agent.
        Returns observation ID.
        """
        with self._lock:
            self._observations.append(obs)
            self._total_observations += 1

            # Update agent weight from confidence
            self._update_agent_weight(obs.agent, obs.confidence)

            obs_id = f"obs_{self._total_observations}"
            return obs_id

    def post_observations(self, observations: List[Observation]) -> List[str]:
        """Post multiple observations."""
        return [self.post_observation(obs) for obs in observations]

    # ------------------------------------------------------------------ Hypothesis posting

    def post_hypothesis(self, hyp: Hypothesis) -> str:
        """
        Post a hypothesis from a reasoning agent.
        Returns hypothesis ID.
        """
        with self._lock:
            hyp_id = f"hyp_{self._total_hypotheses}"
            self._hypotheses[hyp_id] = hyp
            self._total_hypotheses += 1

            # Update agent weight
            self._update_agent_weight(hyp.agent, hyp.confidence)

            return hyp_id

    # ------------------------------------------------------------------ Decision request

    def request_decision(
        self,
        context_type: str,
        context_data: Dict[str, Any],
        min_confidence: float = 0.3,
    ) -> Optional[Decision]:
        """
        Request a coordinated decision based on current hypotheses.

        Parameters
        ----------
        context_type : str — type of decision needed ("execute", "hedge", etc.)
        context_data : dict — decision context
        min_confidence : float — minimum confidence threshold
        """
        with self._lock:
            # Gather relevant hypotheses
            relevant = self._gather_relevant_hypotheses(context_type, context_data)

            if not relevant:
                logger.debug("DecisionBus: no relevant hypotheses for %s", context_type)
                return None

            # Resolve conflicts
            consensus = self._resolver.resolve(relevant)

            if consensus is None:
                logger.debug("DecisionBus: no consensus for %s", context_type)
                return None

            if consensus.confidence < min_confidence:
                logger.debug(
                    "DecisionBus: confidence %.2f below threshold %.2f",
                    consensus.confidence, min_confidence,
                )
                return None

            # Build decision
            decision = Decision(
                type=self._map_decision_type(context_type),
                action=consensus.implications,
                reasoning=[h.description for h in relevant],
                confidence=consensus.confidence,
                metadata={
                    "n_hypotheses": len(relevant),
                    "agents": list(set(h.agent for h in relevant)),
                },
            )

            self._decisions.append(decision)
            self._total_decisions += 1

            # Notify listeners
            for cb in self._decision_callbacks:
                try:
                    cb(decision)
                except Exception as e:
                    logger.warning("Decision callback error: %s", e)

            return decision

    def _gather_relevant_hypotheses(
        self,
        context_type: str,
        context_data: Dict[str, Any],
    ) -> List[Hypothesis]:
        """Gather hypotheses relevant to this decision context."""
        relevant = []

        for hyp in self._hypotheses.values():
            # Check if hypothesis type is relevant
            if self._is_relevant(hyp, context_type, context_data):
                relevant.append(hyp)

        # Sort by confidence * agent weight
        relevant.sort(
            key=lambda h: h.confidence * self._agent_weights.get(h.agent, 0.5),
            reverse=True,
        )

        return relevant[:20]  # top 20

    def _is_relevant(
        self,
        hyp: Hypothesis,
        context_type: str,
        context_data: Dict[str, Any],
    ) -> bool:
        """Check if a hypothesis is relevant to the decision context."""
        # Simple relevance: check if any implication keys match context
        for key in hyp.implications:
            if key in context_data:
                return True

        # Type-based relevance
        if context_type == "execute" and hyp.type in (HypothesisType.PREDICTIVE, HypothesisType.STRATEGIC):
            return True
        if context_type == "hedge" and hyp.type in (HypothesisType.DIAGNOSTIC,):
            return True

        return False

    def _map_decision_type(self, context_type: str) -> DecisionType:
        """Map context type to DecisionType enum."""
        mapping = {
            "execute"  : DecisionType.EXECUTE,
            "adjust"   : DecisionType.ADJUST,
            "cancel"   : DecisionType.CANCEL,
            "hedge"    : DecisionType.HEDGE,
            "rebalance": DecisionType.REALLOCATE,
            "pause"    : DecisionType.PAUSE,
            "resume"   : DecisionType.RESUME,
        }
        return mapping.get(context_type, DecisionType.EXECUTE)

    # ------------------------------------------------------------------ Agent weight management

    def _update_agent_weight(self, agent: str, confidence: float) -> None:
        """Update agent weight based on recent confidence."""
        current = self._agent_weights.get(agent, 0.5)
        # Exponential moving average
        alpha = 0.1
        self._agent_weights[agent] = alpha * confidence + (1 - alpha) * current

    def set_agent_weight(self, agent: str, weight: float) -> None:
        """Explicitly set agent weight (for meta-learner)."""
        self._agent_weights[agent] = max(0.0, min(1.0, weight))
        self._resolver.set_agent_weight(agent, weight)

    # ------------------------------------------------------------------ Callbacks

    def on_decision(self, callback: Callable[[Decision], None]) -> None:
        """Register a callback for new decisions."""
        self._decision_callbacks.append(callback)

    # ------------------------------------------------------------------ Queries

    def get_recent_observations(
        self,
        agent: Optional[str] = None,
        type: Optional[ObservationType] = None,
        n: int = 100,
    ) -> List[Observation]:
        """Get recent observations, optionally filtered."""
        with self._lock:
            obs = list(self._observations)
            if agent:
                obs = [o for o in obs if o.agent == agent]
            if type:
                obs = [o for o in obs if o.type == type]
            return obs[-n:]

    def get_recent_decisions(self, n: int = 50) -> List[Decision]:
        """Get recent decisions."""
        with self._lock:
            return list(self._decisions)[-n:]

    def get_agent_weights(self) -> Dict[str, float]:
        """Get current agent weights."""
        with self._lock:
            return dict(self._agent_weights)

    def get_stats(self) -> Dict[str, Any]:
        """Get bus statistics."""
        with self._lock:
            return {
                "total_observations" : self._total_observations,
                "total_hypotheses"   : self._total_hypotheses,
                "total_decisions"    : self._total_decisions,
                "conflicts_resolved" : self._conflicts_resolved,
                "active_agents"      : len(self._agent_weights),
                "avg_agent_weight"   : sum(self._agent_weights.values()) / max(1, len(self._agent_weights)),
            }

    # ------------------------------------------------------------------ Reset

    def reset(self) -> None:
        """Reset the decision bus (for new session)."""
        with self._lock:
            self._observations.clear()
            self._hypotheses.clear()
            self._decisions.clear()
            self._total_observations = 0
            self._total_hypotheses = 0
            self._total_decisions = 0
            self._conflicts_resolved = 0
            logger.info("DecisionBus: reset")
