from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .agent_roles import AgentAnalysis, AgentContext

logger = logging.getLogger(__name__)


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


@dataclass(slots=True)
class StructuredArgument:
    side: str
    thesis: str
    supporting_points: List[str] = field(default_factory=list)
    rebuttals: List[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.side = str(self.side or "neutral").lower().strip()
        self.confidence = _clamp(self.confidence, 0.0, 1.0)


@dataclass(slots=True)
class DebateOutcome:
    symbol: str
    bull_argument: StructuredArgument
    bear_argument: StructuredArgument
    consensus_stance: str
    consensus_confidence: float
    net_score: float
    reasoning_chain: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class DebateHistory:
    outcomes: List[DebateOutcome] = field(default_factory=list)

    def append(self, outcome: DebateOutcome) -> None:
        self.outcomes.append(outcome)

    def recent(self, limit: int = 20) -> List[DebateOutcome]:
        return self.outcomes[-max(1, int(limit)) :]


class DebateCoordinator:
    """Runs structured bull-versus-bear debates with durable history."""

    def __init__(self, history_path: str = "data/multi_agent_trading/debate_history.jsonl") -> None:
        self.history_path = Path(history_path)
        self.persistence_enabled = True
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.persistence_enabled = False
            logger.warning("DebateCoordinator: persistence disabled during initialization: %s", exc)
        self.history = DebateHistory()
        self._load_history()

    def run_debate(
        self,
        context: AgentContext,
        bull_analysis: AgentAnalysis,
        bear_analysis: AgentAnalysis,
        peer_analyses: Optional[Sequence[AgentAnalysis]] = None,
    ) -> DebateOutcome:
        peer_analyses = list(peer_analyses or [])
        bull_argument = self._build_argument("bull", bull_analysis, bear_analysis, peer_analyses)
        bear_argument = self._build_argument("bear", bear_analysis, bull_analysis, peer_analyses)
        consensus = self._build_consensus(context.symbol, bull_analysis, bear_analysis, bull_argument, bear_argument, peer_analyses)
        self.history.append(consensus)
        self._persist_outcome(consensus)
        logger.info(
            "DebateCoordinator: %s consensus=%s confidence=%.2f",
            context.symbol,
            consensus.consensus_stance,
            consensus.consensus_confidence,
        )
        return consensus

    def _build_argument(
        self,
        side: str,
        primary: AgentAnalysis,
        counterparty: AgentAnalysis,
        peers: Sequence[AgentAnalysis],
    ) -> StructuredArgument:
        supportive = [analysis.summary for analysis in peers if analysis.stance == primary.stance and analysis.agent_name != primary.agent_name]
        conflicting = [analysis.summary for analysis in peers if analysis.stance == counterparty.stance and analysis.agent_name != counterparty.agent_name]
        supporting_points = list(primary.rationale[:3]) + supportive[:2]
        rebuttals = [counterparty.summary] + conflicting[:2]
        thesis = primary.summary
        return StructuredArgument(
            side=side,
            thesis=thesis,
            supporting_points=supporting_points,
            rebuttals=rebuttals,
            confidence=primary.confidence,
            evidence=primary.evidence,
        )

    def _build_consensus(
        self,
        symbol: str,
        bull_analysis: AgentAnalysis,
        bear_analysis: AgentAnalysis,
        bull_argument: StructuredArgument,
        bear_argument: StructuredArgument,
        peers: Sequence[AgentAnalysis],
    ) -> DebateOutcome:
        peer_signal = sum(analysis.score * analysis.confidence for analysis in peers if analysis.agent_name not in {bull_analysis.agent_name, bear_analysis.agent_name})
        bull_strength = max(bull_analysis.score, 0.0) * bull_analysis.confidence
        bear_strength = abs(min(bear_analysis.score, 0.0)) * bear_analysis.confidence
        net_score = _clamp(bull_strength - bear_strength + (0.25 * peer_signal), -1.0, 1.0)
        consensus_stance = "buy" if net_score > 0.12 else "sell" if net_score < -0.12 else "hold"
        confidence = _clamp((abs(net_score) * 0.65) + (0.20 * max(bull_analysis.confidence, bear_analysis.confidence)) + (0.15 * min(len(peers) / 6.0, 1.0)), 0.0, 1.0)
        reasoning_chain = [
            f"Bull thesis: {bull_argument.thesis}",
            f"Bear thesis: {bear_argument.thesis}",
            f"Peer net contribution={peer_signal:.2f} across {len(peers)} supporting analyses.",
            f"Consensus selected {consensus_stance} with net score {net_score:.2f} and confidence {confidence:.2f}.",
        ]
        return DebateOutcome(
            symbol=symbol,
            bull_argument=bull_argument,
            bear_argument=bear_argument,
            consensus_stance=consensus_stance,
            consensus_confidence=confidence,
            net_score=net_score,
            reasoning_chain=reasoning_chain,
        )

    def _persist_outcome(self, outcome: DebateOutcome) -> None:
        if not self.persistence_enabled:
            return
        try:
            payload = asdict(outcome)
            with self.history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        except OSError as exc:
            logger.warning("DebateCoordinator: unable to persist debate history: %s", exc)

    def _load_history(self) -> None:
        if not self.persistence_enabled:
            return
        if not self.history_path.exists():
            return
        try:
            with self.history_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    payload = json.loads(raw)
                    self.history.append(
                        DebateOutcome(
                            symbol=str(payload.get("symbol", "UNKNOWN")),
                            bull_argument=StructuredArgument(**payload.get("bull_argument", {})),
                            bear_argument=StructuredArgument(**payload.get("bear_argument", {})),
                            consensus_stance=str(payload.get("consensus_stance", "hold")),
                            consensus_confidence=float(payload.get("consensus_confidence", 0.0)),
                            net_score=float(payload.get("net_score", 0.0)),
                            reasoning_chain=list(payload.get("reasoning_chain", []) or []),
                            created_at=float(payload.get("created_at", time.time())),
                        )
                    )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("DebateCoordinator: failed to load history from %s: %s", self.history_path, exc)
