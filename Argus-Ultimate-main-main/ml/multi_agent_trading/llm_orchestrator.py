from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .agent_debate import DebateCoordinator, DebateOutcome
from .agent_memory import AgentMemory
from .agent_roles import (
    AgentAnalysis,
    AgentContext,
    AgentSynthesis,
    BearResearcher,
    BullResearcher,
    FundamentalAnalyst,
    FundManager,
    MarketRegimeAgent,
    RiskManager,
    SentimentAnalyst,
    TechnicalAnalyst,
)
from .prompt_optimizer import PromptOptimizer
from .signal_aggregator import AggregatedSignal, SignalAggregator

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TradingDecision:
    decision_id: str
    symbol: str
    action: str
    confidence: float
    net_score: float
    detected_regime: str
    recommended_size: float
    reasoning_chain: List[str] = field(default_factory=list)
    agent_analyses: Dict[str, AgentAnalysis] = field(default_factory=dict)
    debate_outcome: Optional[DebateOutcome] = None
    aggregated_signal: Optional[AggregatedSignal] = None
    synthesis: List[AgentSynthesis] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class MultiAgentTradingOrchestrator:
    """Coordinates TradingAgents-style specialists into a final trading decision."""

    def __init__(
        self,
        *,
        aggregator: Optional[SignalAggregator] = None,
        debate_coordinator: Optional[DebateCoordinator] = None,
        prompt_optimizer: Optional[PromptOptimizer] = None,
        memory: Optional[AgentMemory] = None,
    ) -> None:
        self.aggregator = aggregator or SignalAggregator()
        self.debate_coordinator = debate_coordinator or DebateCoordinator()
        self.prompt_optimizer = prompt_optimizer or PromptOptimizer()
        self.memory = memory or AgentMemory()
        self.agents = [
            FundamentalAnalyst(),
            SentimentAnalyst(),
            TechnicalAnalyst(),
            BullResearcher(),
            BearResearcher(),
            RiskManager(),
            FundManager(),
            MarketRegimeAgent(),
        ]
        self._bootstrap_prompt_variants()

    def generate_decision(self, context: AgentContext | Mapping[str, Any]) -> TradingDecision:
        parsed = context if isinstance(context, AgentContext) else AgentContext.from_mapping(context)
        parsed.metadata.setdefault("prompt_variants", {})
        agent_analyses: Dict[str, AgentAnalysis] = {}

        for agent in self.agents:
            self.prompt_optimizer.ensure_default_variants(agent.agent_name)
            variant = self.prompt_optimizer.choose_variant(agent.agent_name, parsed.market_regime)
            if variant is not None:
                parsed.metadata["prompt_variants"][agent.agent_name] = {
                    "variant_id": variant.variant_id,
                    "name": variant.name,
                    "prompt_text": variant.prompt_text,
                    "metadata": variant.metadata,
                }
            analysis = agent.analyze(parsed)
            agent_analyses[agent.agent_name] = analysis

        regime_analysis = agent_analyses["market_regime_agent"]
        detected_regime = str(regime_analysis.metadata.get("detected_regime", parsed.market_regime or "unknown"))
        parsed.market_regime = detected_regime

        debate_outcome = self.debate_coordinator.run_debate(
            parsed,
            agent_analyses["bull_researcher"],
            agent_analyses["bear_researcher"],
            list(agent_analyses.values()),
        )

        aggregated_signal = self.aggregator.aggregate(list(agent_analyses.values()), regime=detected_regime)
        synthesis = [
            agent.synthesize(list(agent_analyses.values()), parsed)
            for agent in (FundManager(), RiskManager(), MarketRegimeAgent())
        ]
        recommended_size = self._recommended_size(agent_analyses, aggregated_signal)
        reasoning_chain = self._build_reasoning_chain(parsed, agent_analyses, debate_outcome, aggregated_signal, synthesis, recommended_size)
        action = self._final_action(aggregated_signal, debate_outcome, agent_analyses["risk_manager"])
        confidence = min(1.0, (aggregated_signal.confidence * 0.65) + (debate_outcome.consensus_confidence * 0.35))
        decision_id = self.memory.store_decision(
            symbol=parsed.symbol,
            action=action,
            confidence=confidence,
            net_score=aggregated_signal.net_score,
            reasoning_chain=reasoning_chain,
            metadata={
                "detected_regime": detected_regime,
                "recommended_size": recommended_size,
                "market_regime": detected_regime,
                "prompt_variants": {
                    agent_name: str(analysis.metadata.get("prompt_variant_id", ""))
                    for agent_name, analysis in agent_analyses.items()
                    if analysis.metadata.get("prompt_variant_id")
                },
            },
        )
        decision = TradingDecision(
            decision_id=decision_id,
            symbol=parsed.symbol,
            action=action,
            confidence=confidence,
            net_score=aggregated_signal.net_score,
            detected_regime=detected_regime,
            recommended_size=recommended_size,
            reasoning_chain=reasoning_chain,
            agent_analyses=agent_analyses,
            debate_outcome=debate_outcome,
            aggregated_signal=aggregated_signal,
            synthesis=synthesis,
            metadata={
                "consensus_reached": aggregated_signal.consensus_reached,
                "debate_confidence": debate_outcome.consensus_confidence,
            },
        )
        _ = self.memory.track_reasoning_chain(decision_id, decision.reasoning_chain)
        logger.info(
            "MultiAgentTradingOrchestrator: %s action=%s confidence=%.2f regime=%s",
            decision.symbol,
            decision.action,
            decision.confidence,
            decision.detected_regime,
        )
        return decision

    def update_outcome(
        self,
        decision_id: str,
        realized_return: float,
        horizon: str = "1d",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.memory.store_outcome(
            decision_id,
            {
                "realized_return": float(realized_return),
                "horizon": horizon,
                **dict(metadata or {}),
            },
        )
        decision_feedback = self.memory.learn_from_decision(decision_id, realized_return=realized_return, horizon=horizon)
        decision_record = self.memory.decisions.get(decision_id)
        if decision_record is None:
            return decision_feedback
        symbol_feedback = self.memory.learn_from_feedback(decision_record.symbol, realized_return=realized_return, horizon=horizon)
        for analysis in self._latest_prompt_variant_analyses(decision_record):
            self.prompt_optimizer.record_performance(analysis)
        return {
            "decision_feedback": decision_feedback,
            "symbol_feedback": symbol_feedback,
        }

    def _recommended_size(self, analyses: Mapping[str, AgentAnalysis], aggregated_signal: AggregatedSignal) -> float:
        fund_manager = analyses.get("fund_manager")
        risk_manager = analyses.get("risk_manager")
        base_size = 0.0
        if fund_manager is not None:
            base_size = float(fund_manager.metadata.get("target_weight", 0.0))
        risk_gate = 1.0
        if risk_manager is not None:
            utilization = max(risk_manager.evidence.values()) if risk_manager.evidence else 0.0
            risk_gate = max(0.0, 1.0 - utilization)
        size = base_size * aggregated_signal.confidence * risk_gate
        return max(0.0, min(size, 0.10))

    def _final_action(
        self,
        aggregated_signal: AggregatedSignal,
        debate_outcome: DebateOutcome,
        risk_analysis: AgentAnalysis,
    ) -> str:
        if "risk_limit_breached" in risk_analysis.risk_flags:
            return "hold"
        if aggregated_signal.final_stance == debate_outcome.consensus_stance:
            return aggregated_signal.final_stance
        if aggregated_signal.confidence >= debate_outcome.consensus_confidence:
            return aggregated_signal.final_stance
        return debate_outcome.consensus_stance

    def _build_reasoning_chain(
        self,
        context: AgentContext,
        analyses: Mapping[str, AgentAnalysis],
        debate_outcome: DebateOutcome,
        aggregated_signal: AggregatedSignal,
        synthesis: Sequence[AgentSynthesis],
        recommended_size: float,
    ) -> List[str]:
        chain = [
            f"Collected {len(analyses)} specialist views for {context.symbol}.",
            f"Market regime agent detected {analyses['market_regime_agent'].metadata.get('detected_regime', 'unknown')}.",
            f"Bull/Bear debate resolved to {debate_outcome.consensus_stance} with confidence {debate_outcome.consensus_confidence:.2f}.",
            f"Weighted vote aggregation produced {aggregated_signal.final_stance} with net score {aggregated_signal.net_score:.2f}.",
            f"Recommended size after portfolio and risk overlays is {recommended_size:.2%}.",
        ]
        chain.extend(item.conclusion for item in synthesis)
        return chain

    def _bootstrap_prompt_variants(self) -> None:
        for agent in self.agents:
            self.prompt_optimizer.ensure_default_variants(agent.agent_name)

    def _latest_prompt_variant_analyses(self, record: Any) -> List[Any]:
        outcomes = dict(record.outcome or {})
        variant_results: List[Any] = []
        realized_return = float(outcomes.get("realized_return", 0.0))
        market_regime = str(outcomes.get("market_regime", "unknown"))
        decision_quality = 1.0 if (record.action == "buy" and realized_return > 0.0) or (record.action == "sell" and realized_return < 0.0) else 0.5 if record.action == "hold" else 0.0
        agent_variant_map = outcomes.get("prompt_variants", {})
        if not isinstance(agent_variant_map, dict):
            return variant_results
        from .prompt_optimizer import PromptPerformance

        for variant_id in agent_variant_map.values():
            variant_results.append(
                PromptPerformance(
                    variant_id=str(variant_id),
                    market_regime=market_regime,
                    reward=realized_return,
                    accuracy=decision_quality,
                    latency_ms=0.0,
                    feedback={"decision_id": record.decision_id},
                )
            )
        return variant_results
