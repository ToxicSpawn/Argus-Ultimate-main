from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .agents import AgentAnalysis, AgentDebate, MarketContext

logger = logging.getLogger(__name__)


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


@dataclass(slots=True)
class ScoredArgument:
    agent_name: str
    side: str
    thesis: str
    score: float
    confidence: float
    supporting_points: List[str] = field(default_factory=list)
    rebuttals: List[str] = field(default_factory=list)


@dataclass(slots=True)
class DebateRound:
    round_number: int
    bull_argument: ScoredArgument
    bear_argument: ScoredArgument
    round_winner: str
    margin: float
    notes: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class DebateOutcome:
    symbol: str
    winner: str
    confidence: float
    reasoning: List[str] = field(default_factory=list)
    consensus_action: str = "hold"
    bull_score: float = 0.0
    bear_score: float = 0.0
    rounds: List[DebateRound] = field(default_factory=list)
    historical_adjustment: float = 0.0
    timestamp: float = field(default_factory=time.time)


class DebateEngine:
    def __init__(self, rounds: int = 2, confidence_threshold: float = 0.62) -> None:
        self.rounds = max(1, int(rounds))
        self.confidence_threshold = _clamp(confidence_threshold, 0.0, 1.0)
        self.argument_history: Dict[str, List[Tuple[str, float, float]]] = {}

    def run(
        self,
        context: MarketContext,
        bull_analysis: AgentAnalysis,
        bear_analysis: AgentAnalysis,
        bull_debate: AgentDebate,
        bear_debate: AgentDebate,
        peer_analyses: Sequence[AgentAnalysis],
    ) -> DebateOutcome:
        rounds: List[DebateRound] = []
        bull_total = 0.0
        bear_total = 0.0
        for round_number in range(1, self.rounds + 1):
            bull_argument = self._score_argument("bull", bull_analysis, bull_debate, peer_analyses, round_number)
            bear_argument = self._score_argument("bear", bear_analysis, bear_debate, peer_analyses, round_number)
            margin = bull_argument.score - bear_argument.score
            winner = "bull" if margin > 0.03 else "bear" if margin < -0.03 else "tie"
            bull_total += bull_argument.score
            bear_total += bear_argument.score
            rounds.append(
                DebateRound(
                    round_number=round_number,
                    bull_argument=bull_argument,
                    bear_argument=bear_argument,
                    round_winner=winner,
                    margin=margin,
                    notes=[f"Round {round_number} emphasized confidence, evidence breadth, and peer support."],
                )
            )
        historical_adjustment = self._historical_adjustment(context.symbol)
        bull_total = _clamp((bull_total / self.rounds) + historical_adjustment, 0.0, 1.0)
        bear_total = _clamp((bear_total / self.rounds) - historical_adjustment, 0.0, 1.0)
        net = bull_total - bear_total
        confidence = self.score_confidence(bull_total, bear_total, rounds)
        winner = "bull" if net > 0.04 else "bear" if net < -0.04 else "tie"
        consensus_action = "buy" if winner == "bull" and confidence >= self.confidence_threshold else "sell" if winner == "bear" and confidence >= self.confidence_threshold else "hold"
        reasoning = [
            f"Bull average score={bull_total:.2f}, bear average score={bear_total:.2f}.",
            f"Historical adjustment={historical_adjustment:.2f} for {context.symbol}.",
            f"Consensus action={consensus_action} with confidence {confidence:.2f} after {self.rounds} rounds.",
        ]
        outcome = DebateOutcome(
            symbol=context.symbol,
            winner=winner,
            confidence=confidence,
            reasoning=self.build_consensus(reasoning, rounds, winner, confidence, consensus_action),
            consensus_action=consensus_action,
            bull_score=bull_total,
            bear_score=bear_total,
            rounds=rounds,
            historical_adjustment=historical_adjustment,
        )
        self.record_outcome(outcome)
        logger.info("DebateEngine: %s winner=%s action=%s confidence=%.2f", context.symbol, winner, consensus_action, confidence)
        return outcome

    def score_confidence(self, bull_score: float, bear_score: float, rounds: Sequence[DebateRound]) -> float:
        round_decisiveness = sum(abs(item.margin) for item in rounds) / max(len(rounds), 1)
        net = abs(bull_score - bear_score)
        return _clamp((net * 0.65) + (round_decisiveness * 0.20) + 0.15, 0.0, 1.0)

    def build_consensus(
        self,
        base_reasoning: Sequence[str],
        rounds: Sequence[DebateRound],
        winner: str,
        confidence: float,
        consensus_action: str,
    ) -> List[str]:
        reasoning = list(base_reasoning)
        if rounds:
            decisive_rounds = [item for item in rounds if abs(item.margin) >= 0.05]
            reasoning.append(f"{len(decisive_rounds)} of {len(rounds)} rounds were materially decisive.")
        if winner == "tie":
            reasoning.append("Bull and bear cases were too balanced to justify a directional consensus.")
        elif consensus_action == "hold":
            reasoning.append(f"{winner.title()} side edged the debate, but confidence stayed below the action threshold.")
        else:
            reasoning.append(f"Consensus promoted a {consensus_action} bias with debate confidence {confidence:.2f}.")
        return reasoning

    def record_outcome(self, outcome: DebateOutcome) -> None:
        symbol_history = self.argument_history.setdefault(outcome.symbol, [])
        symbol_history.append((outcome.winner, outcome.bull_score, outcome.bear_score))
        if len(symbol_history) > 100:
            del symbol_history[:-100]

    def _score_argument(
        self,
        side: str,
        analysis: AgentAnalysis,
        debate: AgentDebate,
        peer_analyses: Sequence[AgentAnalysis],
        round_number: int,
    ) -> ScoredArgument:
        supportive = sum(1 for item in peer_analyses if item.action == analysis.action and item.agent_name != analysis.agent_name)
        conflicting = sum(1 for item in peer_analyses if item.action not in {analysis.action, "hold"})
        evidence_breadth = min(len(analysis.evidence) / 6.0, 1.0)
        concession_penalty = min(len(debate.concessions) * 0.08, 0.24)
        cadence_bonus = 0.02 if round_number > 1 else 0.0
        score = _clamp(
            (analysis.confidence * 0.35)
            + (abs(analysis.score) * 0.25)
            + (debate.argument_score * 0.20)
            + (min(supportive / 4.0, 1.0) * 0.15)
            + (evidence_breadth * 0.10)
            + cadence_bonus
            - (min(conflicting / 5.0, 1.0) * 0.10)
            - concession_penalty,
            0.0,
            1.0,
        )
        return ScoredArgument(
            agent_name=analysis.agent_name,
            side=side,
            thesis=analysis.thesis,
            score=score,
            confidence=analysis.confidence,
            supporting_points=list(analysis.reasoning[:3]),
            rebuttals=list(debate.rebuttals[:2]),
        )

    def _historical_adjustment(self, symbol: str) -> float:
        history = self.argument_history.get(symbol, [])
        if not history:
            return 0.0
        bull_wins = sum(1 for winner, _, _ in history[-20:] if winner == "bull")
        bear_wins = sum(1 for winner, _, _ in history[-20:] if winner == "bear")
        total = max(bull_wins + bear_wins, 1)
        return _clamp((bull_wins - bear_wins) / total * 0.08, -0.08, 0.08)
