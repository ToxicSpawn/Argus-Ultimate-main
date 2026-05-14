from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

from .agent_roles import AgentAnalysis

logger = logging.getLogger(__name__)


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


@dataclass(slots=True)
class AggregatedSignal:
    final_stance: str
    confidence: float
    net_score: float
    weighted_votes: Dict[str, float] = field(default_factory=dict)
    normalized_weights: Dict[str, float] = field(default_factory=dict)
    consensus_reached: bool = False
    threshold: float = 0.0
    reasoning: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class SignalAggregator:
    """Aggregates multi-agent outputs with regime-aware weight adjustments."""

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "fundamental_analyst": 1.10,
        "sentiment_analyst": 0.95,
        "technical_analyst": 1.05,
        "bull_researcher": 0.85,
        "bear_researcher": 0.85,
        "risk_manager": 1.25,
        "fund_manager": 1.15,
        "market_regime_agent": 1.15,
    }

    REGIME_MULTIPLIERS: Dict[str, Dict[str, float]] = {
        "trending_bull": {
            "technical_analyst": 1.15,
            "bull_researcher": 1.10,
            "fund_manager": 1.10,
        },
        "stress_bear": {
            "risk_manager": 1.25,
            "bear_researcher": 1.15,
            "fund_manager": 0.90,
        },
        "high_volatility_range": {
            "risk_manager": 1.20,
            "technical_analyst": 0.90,
            "sentiment_analyst": 0.90,
        },
        "range_bound": {
            "technical_analyst": 1.05,
            "fundamental_analyst": 0.95,
        },
    }

    def __init__(self, consensus_threshold: float = 0.18, base_weights: Optional[Mapping[str, float]] = None) -> None:
        self.consensus_threshold = max(float(consensus_threshold), 0.0)
        self.base_weights = dict(base_weights or self.DEFAULT_WEIGHTS)

    def aggregate(self, analyses: Sequence[AgentAnalysis], regime: str = "unknown") -> AggregatedSignal:
        if not analyses:
            return AggregatedSignal(
                final_stance="hold",
                confidence=0.0,
                net_score=0.0,
                consensus_reached=False,
                threshold=self.consensus_threshold,
                reasoning=["No agent analyses available for aggregation."],
            )

        adjusted_weights = self._regime_adjusted_weights(regime)
        total_weight = 0.0
        weighted_sum = 0.0
        weighted_votes: Dict[str, float] = {}
        normalized_weights: Dict[str, float] = {}

        for analysis in analyses:
            weight = max(adjusted_weights.get(analysis.agent_name, 1.0), 0.0)
            directional_score = analysis.score
            if analysis.agent_name == "risk_manager":
                directional_score = min(analysis.score, 0.0)
            effective_vote = directional_score * analysis.confidence * weight
            weighted_votes[analysis.agent_name] = effective_vote
            total_weight += weight
            weighted_sum += effective_vote

        if total_weight > 0.0:
            participating_weights = {
                analysis.agent_name: max(adjusted_weights.get(analysis.agent_name, 1.0), 0.0)
                for analysis in analyses
            }
            participating_total = sum(participating_weights.values())
            if participating_total > 0.0:
                normalized_weights = {
                    name: weight / participating_total for name, weight in participating_weights.items() if weight > 0.0
                }
        net_score = _clamp(weighted_sum / total_weight if total_weight > 0.0 else 0.0, -1.0, 1.0)
        stance = "buy" if net_score > self.consensus_threshold else "sell" if net_score < -self.consensus_threshold else "hold"
        confidence = _clamp(abs(net_score) + (0.10 * self._vote_alignment(analyses)), 0.0, 1.0)
        consensus_reached = abs(net_score) >= self.consensus_threshold
        reasoning = [
            f"Regime={regime}, consensus threshold={self.consensus_threshold:.2f}.",
            f"Weighted net score={net_score:.2f} across {len(analyses)} agents.",
            f"Vote alignment={self._vote_alignment(analyses):.2f}; final stance={stance}.",
        ]
        logger.info("SignalAggregator: final stance=%s confidence=%.2f net_score=%.2f", stance, confidence, net_score)
        return AggregatedSignal(
            final_stance=stance,
            confidence=confidence,
            net_score=net_score,
            weighted_votes=weighted_votes,
            normalized_weights=normalized_weights,
            consensus_reached=consensus_reached,
            threshold=self.consensus_threshold,
            reasoning=reasoning,
        )

    def _regime_adjusted_weights(self, regime: str) -> Dict[str, float]:
        regime_key = str(regime or "unknown").lower()
        adjusted = dict(self.base_weights)
        for agent_name, multiplier in self.REGIME_MULTIPLIERS.get(regime_key, {}).items():
            adjusted[agent_name] = adjusted.get(agent_name, 1.0) * multiplier
        return adjusted

    def _vote_alignment(self, analyses: Sequence[AgentAnalysis]) -> float:
        if not analyses:
            return 0.0
        signs = [math.copysign(1.0, analysis.score) if abs(analysis.score) > 1e-9 else 0.0 for analysis in analyses]
        dominant = max(signs.count(1.0), signs.count(-1.0), signs.count(0.0))
        return dominant / len(signs)
