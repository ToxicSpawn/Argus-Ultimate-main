from __future__ import annotations

from .agent_debate import DebateCoordinator, DebateHistory, DebateOutcome, StructuredArgument
from .agent_memory import AgentDecisionRecord, AgentMemory, MemoryEdge, MemoryNode
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
from .llm_orchestrator import MultiAgentTradingOrchestrator, TradingDecision
from .prompt_optimizer import PromptOptimizer, PromptPerformance, PromptVariant
from .signal_aggregator import AggregatedSignal, SignalAggregator

LLMOrchestrator = MultiAgentTradingOrchestrator

__all__ = [
    "AgentAnalysis",
    "AgentContext",
    "AgentDecisionRecord",
    "AgentMemory",
    "AgentSynthesis",
    "AggregatedSignal",
    "BearResearcher",
    "BullResearcher",
    "DebateCoordinator",
    "DebateHistory",
    "DebateOutcome",
    "FundManager",
    "FundamentalAnalyst",
    "MarketRegimeAgent",
    "LLMOrchestrator",
    "MemoryEdge",
    "MemoryNode",
    "MultiAgentTradingOrchestrator",
    "PromptOptimizer",
    "PromptPerformance",
    "PromptVariant",
    "RiskManager",
    "SentimentAnalyst",
    "SignalAggregator",
    "StructuredArgument",
    "TechnicalAnalyst",
    "TradingDecision",
]
