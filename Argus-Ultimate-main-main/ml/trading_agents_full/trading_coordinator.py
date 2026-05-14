from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .agents import (
    AgentAnalysis,
    AgentDebate,
    AgentSynthesis,
    BearResearcher,
    BullResearcher,
    FundamentalAnalyst,
    MarketContext,
    NewsAnalyst,
    RiskManager,
    SentimentAnalyst,
    TechnicalAnalyst,
)
from .debate_engine import DebateEngine, DebateOutcome
from .llm_integration import LLMProvider, ProviderConfig
from .memory_store import DecisionMemory, DecisionRecord, PatternInsight, SimilarSituation
from .signal_synthesizer import SignalSynthesizer, TradingSignal

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CoordinatorMetrics:
    cycles_run: int = 0
    debates_triggered: int = 0
    signals_published: int = 0
    llm_backed_cycles: int = 0
    heuristic_cycles: int = 0
    average_confidence: float = 0.0
    cumulative_realized_return: float = 0.0


@dataclass(slots=True)
class TradingCycleResult:
    context: MarketContext
    analyses: Dict[str, AgentAnalysis]
    debates: Dict[str, AgentDebate]
    synthesis: Dict[str, AgentSynthesis]
    debate_outcome: Optional[DebateOutcome]
    signal: TradingSignal
    decision_record: DecisionRecord
    similar_situations: List[SimilarSituation] = field(default_factory=list)
    pattern_insight: Optional[PatternInsight] = None
    published_to_bus: bool = False
    timestamp: float = field(default_factory=time.time)


class TradingCoordinator:
    def __init__(
        self,
        *,
        llm_provider: Optional[LLMProvider] = None,
        memory_store: Optional[DecisionMemory] = None,
        debate_engine: Optional[DebateEngine] = None,
        signal_synthesizer: Optional[SignalSynthesizer] = None,
        signal_bus: Optional[Any] = None,
        event_bus: Optional[Any] = None,
        risk_manager: Optional[Any] = None,
    ) -> None:
        self.llm_provider = llm_provider or LLMProvider(ProviderConfig())
        self.memory_store = memory_store or DecisionMemory()
        self.debate_engine = debate_engine or DebateEngine()
        self.signal_synthesizer = signal_synthesizer or SignalSynthesizer()
        self.signal_bus = signal_bus
        self.event_bus = event_bus
        self.external_risk_manager = risk_manager
        self.metrics = CoordinatorMetrics()
        self.agents = {
            "fundamental_analyst": FundamentalAnalyst(self.llm_provider),
            "sentiment_analyst": SentimentAnalyst(self.llm_provider),
            "technical_analyst": TechnicalAnalyst(self.llm_provider),
            "news_analyst": NewsAnalyst(self.llm_provider),
            "bull_researcher": BullResearcher(self.llm_provider),
            "bear_researcher": BearResearcher(self.llm_provider),
            "risk_manager": RiskManager(self.llm_provider),
        }

    def run_cycle(self, context: MarketContext | Mapping[str, Any], publish: bool = False) -> TradingCycleResult:
        parsed = context if isinstance(context, MarketContext) else MarketContext.from_mapping(context)
        analyses: Dict[str, AgentAnalysis] = {}
        for name, agent in self.agents.items():
            try:
                analyses[name] = agent.analyze(parsed)
            except Exception as exc:  # noqa: BLE001
                logger.warning("TradingCoordinator agent %s failed for %s: %s", name, parsed.symbol, exc)
                analyses[name] = AgentAnalysis(
                    agent_name=name,
                    action="hold",
                    confidence=0.0,
                    score=0.0,
                    thesis=f"{name} failed and defaulted to hold.",
                    reasoning=[f"Agent failure: {exc}"],
                    risk_flags=["agent_failure"],
                    metadata={"fallback": True},
                )
        debates: Dict[str, AgentDebate] = self._build_debates(parsed, analyses)
        debate_outcome = self._run_debate_if_needed(parsed, analyses, debates)
        signal = self.signal_synthesizer.synthesize(
            list(analyses.values()),
            regime=parsed.regime,
            symbol=parsed.symbol,
            debate_action=debate_outcome.consensus_action if debate_outcome is not None else "hold",
            debate_confidence=debate_outcome.confidence if debate_outcome is not None else 0.0,
        )
        signal = self._apply_external_risk_overlay(signal, parsed, analyses["risk_manager"])
        synthesis = {name: self.agents[name].synthesize(analyses[name], debates.get(name), parsed) for name in self.agents}
        context_signature = self._build_context_signature(parsed, analyses)
        similar = self.memory_store.query_similar_situations(parsed.symbol, parsed.regime, context_signature)
        pattern = self.memory_store.learn_pattern_summary(parsed.symbol, parsed.regime)
        signal.reasoning_chain.extend(
            [
                f"Found {len(similar)} similar historical situations in memory.",
                pattern.summary,
            ]
        )
        decision_record = self.memory_store.store_decision(
            symbol=parsed.symbol,
            action=signal.action,
            confidence=signal.confidence,
            regime=parsed.regime,
            net_score=signal.net_score,
            reasoning=signal.reasoning_chain,
            context_signature=context_signature,
        )
        published = self._publish_signal(signal, publish)
        self._publish_event(signal, decision_record, publish)
        self._update_metrics(analyses, debate_outcome, signal, published)
        logger.info("TradingCoordinator: %s action=%s confidence=%.2f publish=%s", parsed.symbol, signal.action, signal.confidence, published)
        return TradingCycleResult(
            context=parsed,
            analyses=analyses,
            debates=debates,
            synthesis=synthesis,
            debate_outcome=debate_outcome,
            signal=signal,
            decision_record=decision_record,
            similar_situations=similar,
            pattern_insight=pattern,
            published_to_bus=published,
        )

    def record_outcome(self, decision_id: str, realized_return: float) -> Optional[DecisionRecord]:
        self.memory_store.update_outcome(decision_id, realized_return)
        self.metrics.cumulative_realized_return += float(realized_return)
        return self.memory_store.get_decision(decision_id)

    async def run_cycle_async(self, context: MarketContext | Mapping[str, Any], publish: bool = False) -> TradingCycleResult:
        return await asyncio.to_thread(self.run_cycle, context, publish)

    def _build_debates(self, context: MarketContext, analyses: Mapping[str, AgentAnalysis]) -> Dict[str, AgentDebate]:
        debates: Dict[str, AgentDebate] = {}
        all_analyses = list(analyses.values())
        for name, agent in self.agents.items():
            analysis = analyses[name]
            debates[name] = agent.debate(analysis, [item for item in all_analyses if item.agent_name != name], context)
        return debates

    def _run_debate_if_needed(
        self,
        context: MarketContext,
        analyses: Mapping[str, AgentAnalysis],
        debates: Mapping[str, AgentDebate],
    ) -> Optional[DebateOutcome]:
        bull = analyses["bull_researcher"]
        bear = analyses["bear_researcher"]
        disagreement = bull.action != bear.action or abs(bull.score - bear.score) > 0.20
        if not disagreement:
            return None
        self.metrics.debates_triggered += 1
        return self.debate_engine.run(
            context,
            bull,
            bear,
            debates["bull_researcher"],
            debates["bear_researcher"],
            list(analyses.values()),
        )

    def _apply_external_risk_overlay(self, signal: TradingSignal, context: MarketContext, risk_analysis: AgentAnalysis) -> TradingSignal:
        if "risk_limit_breached" in risk_analysis.risk_flags:
            signal.action = "hold"
            signal.target_position = 0.0
            signal.risk_overrides.append("internal_risk_limit_breached")
        if self.external_risk_manager is None:
            return signal
        approved = self._call_external_risk_manager(signal, context)
        if not approved:
            signal.action = "hold"
            signal.target_position = 0.0
            signal.risk_overrides.append("external_risk_manager_rejected")
        return signal

    def _call_external_risk_manager(self, signal: TradingSignal, context: MarketContext) -> bool:
        for method_name in ("approve_signal", "validate_signal", "check_signal", "approve_trade"):
            method = getattr(self.external_risk_manager, method_name, None)
            if callable(method):
                try:
                    result = method(signal, context)
                    return bool(result if not isinstance(result, tuple) else result[0])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("External risk manager method %s failed: %s", method_name, exc)
        return True

    def _publish_signal(self, signal: TradingSignal, publish: bool) -> bool:
        if not publish or self.signal_bus is None:
            return False
        argus_signal = signal.to_argus_signal()
        if argus_signal is None:
            return False
        try:
            publish_method = getattr(self.signal_bus, "publish_sync", None)
            if callable(publish_method):
                publish_method(argus_signal)
                return True
            publish_async = getattr(self.signal_bus, "publish", None)
            if callable(publish_async):
                result = publish_async(argus_signal)
                if asyncio.iscoroutine(result):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(result)
                    except RuntimeError:
                        asyncio.run(result)
                return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Signal bus publish failed: %s", exc)
        return False

    def _publish_event(self, signal: TradingSignal, decision_record: DecisionRecord, publish: bool) -> None:
        if not publish or self.event_bus is None:
            return
        payload = {
            "symbol": signal.metadata.get("symbol", "UNKNOWN"),
            "action": signal.action,
            "confidence": signal.confidence,
            "net_score": signal.net_score,
            "decision_id": decision_record.decision_id,
            "target_position": signal.target_position,
            "reasoning_chain": signal.reasoning_chain,
            "unified_signal": signal.to_unified_signal(),
        }
        try:
            publish_method = getattr(self.event_bus, "publish", None)
            if callable(publish_method):
                publish_method("signal_generated", payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Event bus publish failed: %s", exc)

    def _build_context_signature(self, context: MarketContext, analyses: Mapping[str, AgentAnalysis]) -> Dict[str, float]:
        return {
            "trend_strength": float(context.technical_indicators.get("trend_strength", 0.0)),
            "realized_volatility": float(context.market_data.get("realized_volatility", 0.0)),
            "news_score": float(context.sentiment_data.get("news_score", 0.0)),
            "valuation_discount": float(context.fundamentals.get("valuation_discount", 0.0)),
            "risk_utilization": float(analyses["risk_manager"].metadata.get("risk_utilization", 0.0)),
        }

    def _update_metrics(
        self,
        analyses: Mapping[str, AgentAnalysis],
        debate_outcome: Optional[DebateOutcome],
        signal: TradingSignal,
        published: bool,
    ) -> None:
        self.metrics.cycles_run += 1
        if published:
            self.metrics.signals_published += 1
        if any(analysis.llm_used for analysis in analyses.values()):
            self.metrics.llm_backed_cycles += 1
        else:
            self.metrics.heuristic_cycles += 1
        rolling_sum = (self.metrics.average_confidence * (self.metrics.cycles_run - 1)) + signal.confidence
        self.metrics.average_confidence = rolling_sum / self.metrics.cycles_run
        if debate_outcome is not None:
            signal.reasoning_chain.extend(debate_outcome.reasoning)
