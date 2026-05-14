from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence

from .agents import AgentAnalysis

logger = logging.getLogger(__name__)

try:
    from core.strategy.signal import Signal, SignalSide
except Exception:  # noqa: BLE001
    Signal = None
    SignalSide = None

try:
    from unified_types import TradingSignal as UnifiedTradingSignal
except Exception:  # noqa: BLE001
    UnifiedTradingSignal = None


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value))
    except (TypeError, ValueError):
        return default


class TradingAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(slots=True)
class WeightedVote:
    agent_name: str
    weight: float
    weighted_score: float
    confidence: float
    action: str


@dataclass(slots=True)
class TradingSignal:
    action: str
    confidence: float
    net_score: float
    target_position: float
    regime: str
    reasoning: List[str] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    weighted_votes: List[WeightedVote] = field(default_factory=list)
    risk_overrides: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_argus_signal(self, strategy_id: str = "trading_agents_full") -> Optional[object]:
        if Signal is None or SignalSide is None:
            return None
        side = SignalSide.FLAT
        if self.action == TradingAction.BUY.value:
            side = SignalSide.LONG
        elif self.action == TradingAction.SELL.value:
            side = SignalSide.SHORT
        return Signal(
            symbol=str(self.metadata.get("symbol", "UNKNOWN")),
            side=side,
            strength=_clamp(self.confidence, 0.0, 1.0),
            strategy_id=strategy_id,
            metadata={
                "reasoning": self.reasoning,
                "reasoning_chain": self.reasoning_chain,
                "target_position": self.target_position,
                "regime": self.regime,
                **self.metadata,
            },
        )

    def to_unified_signal(self) -> Optional[object]:
        if UnifiedTradingSignal is None:
            return None
        return UnifiedTradingSignal(
            symbol=str(self.metadata.get("symbol", "UNKNOWN")),
            action=self.action.upper(),
            confidence=_clamp(self.confidence, 0.0, 1.0),
            strength=_clamp(abs(self.net_score), 0.0, 1.0),
            entry_price=_safe_float(self.metadata.get("entry_price", 0.0), 0.0),
            stop_loss=_safe_float(self.metadata.get("stop_loss"), 0.0) if self.metadata.get("stop_loss") is not None else None,
            take_profit=_safe_float(self.metadata.get("take_profit"), 0.0) if self.metadata.get("take_profit") is not None else None,
            reasoning=" | ".join(self.reasoning_chain[:4]),
            agent_consensus=self.confidence,
        )


class SignalSynthesizer:
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "fundamental_analyst": 1.00,
        "sentiment_analyst": 0.90,
        "technical_analyst": 1.05,
        "news_analyst": 1.00,
        "bull_researcher": 0.85,
        "bear_researcher": 0.85,
        "risk_manager": 1.25,
    }
    REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
        "trending_bull": {"technical_analyst": 1.15, "bull_researcher": 1.10},
        "stress_bear": {"risk_manager": 1.25, "bear_researcher": 1.10, "sentiment_analyst": 1.05},
        "high_volatility_range": {"risk_manager": 1.20, "technical_analyst": 0.90},
        "range_bound": {"fundamental_analyst": 0.95, "technical_analyst": 1.05},
    }

    def __init__(self, base_weights: Optional[Mapping[str, float]] = None, decision_threshold: float = 0.16) -> None:
        self.base_weights = dict(base_weights or self.DEFAULT_WEIGHTS)
        self.decision_threshold = _clamp(decision_threshold, 0.01, 0.5)

    def synthesize(
        self,
        analyses: Sequence[AgentAnalysis],
        regime: str,
        symbol: str,
        debate_action: str = "hold",
        debate_confidence: float = 0.0,
    ) -> TradingSignal:
        if not analyses:
            return TradingSignal(action=TradingAction.HOLD.value, confidence=0.0, net_score=0.0, target_position=0.0, regime=regime, metadata={"symbol": symbol})
        weights = self._weights_for_regime(regime)
        weighted_votes: List[WeightedVote] = []
        total_weight = 0.0
        weighted_sum = 0.0
        risk_overrides: List[str] = []
        for analysis in analyses:
            weight = max(weights.get(analysis.agent_name, 1.0), 0.0)
            score = analysis.score
            if analysis.agent_name == "risk_manager":
                if "risk_limit_breached" in analysis.risk_flags:
                    risk_overrides.append("risk_limit_breached")
                score = min(score, 0.0)
            weighted_score = score * analysis.confidence * weight
            weighted_votes.append(WeightedVote(analysis.agent_name, weight, weighted_score, analysis.confidence, analysis.action))
            total_weight += weight
            weighted_sum += weighted_score
        net_score = _clamp(weighted_sum / total_weight if total_weight else 0.0, -1.0, 1.0)
        net_score = _clamp((net_score * 0.75) + (self._action_to_score(debate_action) * debate_confidence * 0.25), -1.0, 1.0)
        action = self._score_to_action(net_score)
        if risk_overrides:
            action = TradingAction.HOLD.value
        confidence = _clamp(abs(net_score) + (0.10 * self._alignment(weighted_votes)), 0.0, 1.0)
        target_position = 0.0 if action == TradingAction.HOLD.value else _clamp(confidence * 0.10, 0.0, 0.10)
        reasoning = [
            f"Regime {regime} applied regime-aware weights across {len(weighted_votes)} agents.",
            f"Debate overlay contributed action={debate_action} confidence={debate_confidence:.2f}.",
            f"Weighted net score resolved to {net_score:.2f}, producing action={action}.",
        ]
        logger.info("SignalSynthesizer: %s action=%s confidence=%.2f", symbol, action, confidence)
        return TradingSignal(
            action=action,
            confidence=confidence,
            net_score=net_score,
            target_position=target_position,
            regime=regime,
            reasoning=reasoning,
            reasoning_chain=list(reasoning),
            weighted_votes=weighted_votes,
            risk_overrides=risk_overrides,
            metadata={"symbol": symbol, "decision_threshold": self.decision_threshold},
        )

    def _weights_for_regime(self, regime: str) -> Dict[str, float]:
        adjusted = dict(self.base_weights)
        for agent_name, multiplier in self.REGIME_WEIGHTS.get(str(regime or "unknown").lower(), {}).items():
            adjusted[agent_name] = adjusted.get(agent_name, 1.0) * multiplier
        return adjusted

    def _alignment(self, votes: Sequence[WeightedVote]) -> float:
        if not votes:
            return 0.0
        actions = [vote.action for vote in votes]
        dominant = max(actions.count("buy"), actions.count("sell"), actions.count("hold"))
        return dominant / len(actions)

    def _score_to_action(self, score: float) -> str:
        if score >= self.decision_threshold:
            return TradingAction.BUY.value
        if score <= -self.decision_threshold:
            return TradingAction.SELL.value
        return TradingAction.HOLD.value

    def _action_to_score(self, action: str) -> float:
        if action == TradingAction.BUY.value:
            return 1.0
        if action == TradingAction.SELL.value:
            return -1.0
        return 0.0
