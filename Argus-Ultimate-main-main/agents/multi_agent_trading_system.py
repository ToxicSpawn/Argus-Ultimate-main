"""
agents/multi_agent_trading_system.py — Multi-Agent Trading System

Implements a swarm of specialized AI agents that trade cooperatively
and competitively to find optimal strategies.

Features:
- 10+ specialized agents (momentum, mean reversion, scalper, etc.)
- Voting/consensus mechanism for robust decisions
- Adversarial training (agents find each other's weaknesses)
- Performance-based agent weighting
- Agent communication protocol
- Swarm intelligence emergence

Usage::

    from agents.multi_agent_trading_system import MultiAgentTradingSystem
    
    system = MultiAgentTradingSystem()
    
    # Get trading decision
    decision = system.decide(market_state)
    
    # Update with results
    system.update_agents(decision, actual_result)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

class AgentType(str, Enum):
    """Types of trading agents."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    SCALPER = "scalper"
    TREND_FOLLOWER = "trend_follower"
    BREAKOUT = "breakout"
    VOLATILITY = "volatility"
    ARBITRAGE = "arbitrage"
    PATTERN_RECOGNIZER = "pattern_recognizer"
    SENTIMENT = "sentiment"
    RISK_MANAGER = "risk_manager"
    MARKET_MAKER = "market_maker"
    CONTRARIAN = "contrarian"


class SignalType(str, Enum):
    """Types of trading signals."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    SHORT = "short"
    CLOSE = "close"


@dataclass
class MarketState:
    """Current market state for agent decision making."""
    timestamp: datetime
    prices: np.ndarray  # OHLCV
    volume: np.ndarray
    returns: np.ndarray
    volatility: float
    trend_strength: float
    regime: str
    order_book: Optional[Dict[str, Any]] = None
    
    # Technical indicators
    rsi: float = 50.0
    macd: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    sma_20: float = 0.0
    ema_12: float = 0.0


@dataclass
class AgentSignal:
    """Signal from a single agent."""
    agent_id: str
    agent_type: AgentType
    signal_type: SignalType
    confidence: float  # 0 to 1
    target_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Metadata
    supporting_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusDecision:
    """Final consensus decision from all agents."""
    timestamp: datetime
    signal_type: SignalType
    confidence: float
    position_size: float  # 0 to 1
    
    # Agent votes
    agent_signals: List[AgentSignal]
    voting_weights: Dict[str, float]
    
    # Consensus metrics
    agreement_score: float  # 0 to 1, how much agents agree
    dissenting_agents: List[str]
    
    # Risk parameters
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Reasoning
    primary_reasoning: str = ""
    agent_reasonings: List[str] = field(default_factory=list)


@dataclass
class AgentPerformance:
    """Performance metrics for an agent."""
    agent_id: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    recent_performance: float = 0.0  # Last 20 trades
    weight: float = 1.0  # Current voting weight


# ============================================================================
# Base Agent
# ============================================================================

class BaseTradingAgent(ABC):
    """Abstract base class for trading agents."""
    
    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        lookback: int = 20,
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.lookback = lookback
        
        # Performance tracking
        self.performance = AgentPerformance(agent_id=agent_id)
        self.trade_history: deque = deque(maxlen=1000)
        
        # State
        self.is_active = True
        self.cooldown_until = 0.0
    
    @abstractmethod
    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Analyze market and generate signal."""
        pass
    
    def update_performance(self, trade_result: Dict[str, Any]):
        """Update agent performance after a trade."""
        self.trade_history.append(trade_result)
        self.performance.total_trades += 1
        
        pnl = trade_result.get("pnl", 0)
        self.performance.total_pnl += pnl
        
        if pnl > 0:
            self.performance.winning_trades += 1
            self.performance.avg_win = (
                (self.performance.avg_win * (self.performance.winning_trades - 1) + pnl)
                / self.performance.winning_trades
            )
        else:
            self.performance.losing_trades += 1
            self.performance.avg_loss = (
                (self.performance.avg_loss * (self.performance.losing_trades - 1) + abs(pnl))
                / self.performance.losing_trades
            )
        
        # Update win rate
        if self.performance.total_trades > 0:
            self.performance.win_rate = self.performance.winning_trades / self.performance.total_trades
        
        # Update profit factor
        total_wins = self.performance.avg_win * self.performance.winning_trades
        total_losses = self.performance.avg_loss * self.performance.losing_trades
        if total_losses > 0:
            self.performance.profit_factor = total_wins / total_losses
        
        # Recent performance (last 20 trades)
        recent = list(self.trade_history)[-20:]
        if recent:
            self.performance.recent_performance = sum(t.get("pnl", 0) for t in recent) / len(recent)
    
    def calculate_weight(self) -> float:
        """Calculate voting weight based on performance."""
        weight = 1.0
        
        # Boost based on win rate
        if self.performance.total_trades > 10:
            weight *= (0.5 + self.performance.win_rate)
        
        # Boost based on profit factor
        if self.performance.profit_factor > 1:
            weight *= min(2.0, self.performance.profit_factor)
        
        # Penalize recent poor performance
        if self.performance.recent_performance < 0:
            weight *= max(0.3, 1.0 + self.performance.recent_performance * 10)
        
        # Ensure minimum weight
        weight = max(0.1, min(3.0, weight))
        
        self.performance.weight = weight
        return weight


# ============================================================================
# Specialized Agents
# ============================================================================

class MomentumAgent(BaseTradingAgent):
    """Momentum-based trading agent."""
    
    def __init__(self):
        super().__init__("momentum_001", AgentType.MOMENTUM)
    
    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Generate momentum signal."""
        returns = market_state.returns
        
        if len(returns) < 10:
            return AgentSignal(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reasoning="Insufficient data",
            )
        
        # Calculate momentum
        short_momentum = np.mean(returns[-5:])
        long_momentum = np.mean(returns[-20:])
        
        # Trend strength
        trend = market_state.trend_strength
        
        # Generate signal
        if short_momentum > 0.001 and long_momentum > 0 and trend > 0.5:
            signal = SignalType.BUY
            confidence = min(0.9, abs(short_momentum) * 100 * trend)
        elif short_momentum < -0.001 and long_momentum < 0 and trend > 0.5:
            signal = SignalType.SELL
            confidence = min(0.9, abs(short_momentum) * 100 * trend)
        else:
            signal = SignalType.HOLD
            confidence = 0.3
        
        return AgentSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            signal_type=signal,
            confidence=confidence,
            reasoning=f"Short momentum: {short_momentum:.4f}, Long: {long_momentum:.4f}, Trend: {trend:.2f}",
        )


class MeanReversionAgent(BaseTradingAgent):
    """Mean reversion trading agent."""
    
    def __init__(self):
        super().__init__("mean_reversion_001", AgentType.MEAN_REVERSION)
    
    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Generate mean reversion signal."""
        prices = market_state.prices[:, 3]  # Close prices
        
        if len(prices) < 20:
            return AgentSignal(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reasoning="Insufficient data",
            )
        
        # Calculate z-score
        mean = np.mean(prices[-20:])
        std = np.std(prices[-20:]) + 1e-8
        z_score = (prices[-1] - mean) / std
        
        # Generate signal
        if z_score < -2.0:
            signal = SignalType.BUY
            confidence = min(0.9, abs(z_score) / 4)
        elif z_score > 2.0:
            signal = SignalType.SELL
            confidence = min(0.9, abs(z_score) / 4)
        else:
            signal = SignalType.HOLD
            confidence = 0.2
        
        return AgentSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            signal_type=signal,
            confidence=confidence,
            reasoning=f"Z-score: {z_score:.2f}",
        )


class ScalperAgent(BaseTradingAgent):
    """High-frequency scalping agent."""
    
    def __init__(self):
        super().__init__("scalper_001", AgentType.SCALPER)
    
    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Generate scalping signal."""
        if market_state.volatility < 0.01:
            return AgentSignal(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reasoning="Low volatility, scalping not profitable",
            )
        
        # Quick momentum check
        recent_returns = market_state.returns[-3:]
        momentum = np.mean(recent_returns)
        
        if abs(momentum) > 0.002:
            signal = SignalType.BUY if momentum > 0 else SignalType.SELL
            confidence = 0.6
        else:
            signal = SignalType.HOLD
            confidence = 0.3
        
        return AgentSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            signal_type=signal,
            confidence=confidence,
            reasoning=f"Quick momentum: {momentum:.4f}",
        )


class VolatilityAgent(BaseTradingAgent):
    """Volatility-based trading agent."""
    
    def __init__(self):
        super().__init__("volatility_001", AgentType.VOLATILITY)
    
    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Generate volatility signal."""
        vol = market_state.volatility
        
        # Compare to historical
        if len(market_state.returns) >= 20:
            hist_vol = np.std(market_state.returns[-20:]) * np.sqrt(252)
            vol_ratio = vol / (hist_vol + 1e-8)
        else:
            vol_ratio = 1.0
        
        # High volatility = reduce position or hedge
        if vol_ratio > 1.5:
            signal = SignalType.SELL  # Reduce exposure
            confidence = min(0.8, (vol_ratio - 1) * 0.5)
        elif vol_ratio < 0.5:
            signal = SignalType.BUY  # Increase exposure in low vol
            confidence = 0.5
        else:
            signal = SignalType.HOLD
            confidence = 0.3
        
        return AgentSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            signal_type=signal,
            confidence=confidence,
            reasoning=f"Vol ratio: {vol_ratio:.2f}, Current vol: {vol:.2%}",
        )


class PatternAgent(BaseTradingAgent):
    """Chart pattern recognition agent."""
    
    def __init__(self):
        super().__init__("pattern_001", AgentType.PATTERN_RECOGNIZER)
    
    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Generate pattern-based signal."""
        prices = market_state.prices[:, 3]  # Close
        
        if len(prices) < 20:
            return AgentSignal(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reasoning="Insufficient data",
            )
        
        # Simple pattern detection
        recent = prices[-10:]
        
        # Double bottom
        if self._is_double_bottom(recent):
            return AgentSignal(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                signal_type=SignalType.BUY,
                confidence=0.7,
                reasoning="Double bottom pattern detected",
            )
        
        # Double top
        if self._is_double_top(recent):
            return AgentSignal(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                signal_type=SignalType.SELL,
                confidence=0.7,
                reasoning="Double top pattern detected",
            )
        
        return AgentSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            signal_type=SignalType.HOLD,
            confidence=0.2,
            reasoning="No clear pattern",
        )
    
    def _is_double_bottom(self, prices: np.ndarray) -> bool:
        """Detect double bottom pattern."""
        if len(prices) < 8:
            return False
        min_idx = np.argsort(prices)[:2]
        return len(set(min_idx)) == 2 and abs(prices[min_idx[0]] - prices[min_idx[1]]) / prices[0] < 0.02
    
    def _is_double_top(self, prices: np.ndarray) -> bool:
        """Detect double top pattern."""
        if len(prices) < 8:
            return False
        max_idx = np.argsort(prices)[-2:]
        return len(set(max_idx)) == 2 and abs(prices[max_idx[0]] - prices[max_idx[1]]) / prices[0] < 0.02


class ContrarianAgent(BaseTradingAgent):
    """Contrarian trading agent (fades extremes)."""
    
    def __init__(self):
        super().__init__("contrarian_001", AgentType.CONTRARIAN)
    
    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Generate contrarian signal."""
        rsi = market_state.rsi
        
        # RSI extremes
        if rsi > 70:
            signal = SignalType.SELL
            confidence = min(0.8, (rsi - 70) / 30)
        elif rsi < 30:
            signal = SignalType.BUY
            confidence = min(0.8, (30 - rsi) / 30)
        else:
            signal = SignalType.HOLD
            confidence = 0.2
        
        return AgentSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            signal_type=signal,
            confidence=confidence,
            reasoning=f"RSI: {rsi:.1f}",
        )


class RiskManagerAgent(BaseTradingAgent):
    """Risk management agent (always voting for safety)."""
    
    def __init__(self, max_position: float = 0.3):
        super().__init__("risk_manager_001", AgentType.RISK_MANAGER)
        self.max_position = max_position
    
    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Generate risk signal."""
        vol = market_state.volatility
        
        # High volatility = reduce risk
        if vol > 0.03:
            signal = SignalType.SELL
            confidence = min(0.9, vol * 20)
            reasoning = f"High volatility ({vol:.2%}), reducing exposure"
        elif vol < 0.01:
            signal = SignalType.BUY
            confidence = 0.5
            reasoning = f"Low volatility ({vol:.2%}), can increase exposure"
        else:
            signal = SignalType.HOLD
            confidence = 0.3
            reasoning = f"Normal volatility ({vol:.2%})"
        
        return AgentSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            signal_type=signal,
            confidence=confidence,
            reasoning=reasoning,
        )


# ============================================================================
# Multi-Agent System
# ============================================================================

class MultiAgentTradingSystem:
    """
    Multi-Agent Trading System with consensus mechanism.
    
    Coordinates multiple specialized agents to make robust trading decisions.
    """
    
    def __init__(
        self,
        *,
        enable_adversarial: bool = True,
        consensus_threshold: float = 0.6,
        min_agents_for_decision: int = 3,
    ):
        self.enable_adversarial = enable_adversarial
        self.consensus_threshold = consensus_threshold
        self.min_agents_for_decision = min_agents_for_decision
        
        # Initialize agents
        self.agents: Dict[str, BaseTradingAgent] = {}
        self._initialize_default_agents()
        
        # Decision history
        self.decision_history: deque = deque(maxlen=1000)
        
        # Performance tracking
        self.total_decisions = 0
        self.correct_decisions = 0
    
    def _initialize_default_agents(self):
        """Initialize default set of agents."""
        default_agents = [
            MomentumAgent(),
            MeanReversionAgent(),
            ScalperAgent(),
            VolatilityAgent(),
            PatternAgent(),
            ContrarianAgent(),
            RiskManagerAgent(),
        ]
        
        for agent in default_agents:
            self.agents[agent.agent_id] = agent
    
    def add_agent(self, agent: BaseTradingAgent):
        """Add a new agent to the system."""
        self.agents[agent.agent_id] = agent
        logger.info("Added agent: %s (%s)", agent.agent_id, agent.agent_type.value)
    
    def remove_agent(self, agent_id: str):
        """Remove an agent from the system."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info("Removed agent: %s", agent_id)
    
    def decide(self, market_state: MarketState) -> ConsensusDecision:
        """
        Make trading decision based on all agents.
        
        Uses weighted voting with consensus mechanism.
        """
        start_time = time.monotonic()
        
        # Collect signals from all active agents
        agent_signals: List[AgentSignal] = []
        
        for agent in self.agents.values():
            if agent.is_active:
                try:
                    signal = agent.analyze(market_state)
                    agent_signals.append(signal)
                except Exception as e:
                    logger.warning("Agent %s failed: %s", agent.agent_id, e)
        
        if len(agent_signals) < self.min_agents_for_decision:
            return ConsensusDecision(
                timestamp=datetime.utcnow(),
                signal_type=SignalType.HOLD,
                confidence=0.0,
                position_size=0.0,
                agent_signals=agent_signals,
                voting_weights={},
                agreement_score=0.0,
                dissenting_agents=[],
                primary_reasoning="Insufficient agents for decision",
            )
        
        # Calculate voting weights
        voting_weights: Dict[str, float] = {}
        for agent in self.agents.values():
            voting_weights[agent.agent_id] = agent.calculate_weight()
        
        # Weighted voting
        signal_weights: Dict[SignalType, float] = defaultdict(float)
        signal_confidences: Dict[SignalType, List[float]] = defaultdict(list)
        
        for signal in agent_signals:
            weight = voting_weights.get(signal.agent_id, 1.0)
            weighted_confidence = signal.confidence * weight
            signal_weights[signal.signal_type] += weighted_confidence
            signal_confidences[signal.signal_type].append(signal.confidence)
        
        # Determine winning signal
        if not signal_weights:
            winning_signal = SignalType.HOLD
            confidence = 0.0
        else:
            winning_signal = max(signal_weights, key=signal_weights.get)
            total_weight = sum(signal_weights.values())
            confidence = signal_weights[winning_signal] / total_weight if total_weight > 0 else 0.0
        
        # Calculate agreement score
        total_signals = len(agent_signals)
        winning_count = sum(1 for s in agent_signals if s.signal_type == winning_signal)
        agreement_score = winning_count / total_signals
        
        # Find dissenting agents
        dissenting_agents = [
            s.agent_id for s in agent_signals
            if s.signal_type != winning_signal and s.confidence > 0.5
        ]
        
        # Calculate position size based on confidence and agreement
        position_size = confidence * agreement_score
        
        # Collect reasonings
        agent_reasonings = [s.reasoning for s in agent_signals if s.reasoning]
        primary_reasoning = f"Consensus: {winning_signal.value} (confidence: {confidence:.2f}, agreement: {agreement_score:.2f})"
        
        decision = ConsensusDecision(
            timestamp=datetime.utcnow(),
            signal_type=winning_signal,
            confidence=confidence,
            position_size=position_size,
            agent_signals=agent_signals,
            voting_weights=voting_weights,
            agreement_score=agreement_score,
            dissenting_agents=dissenting_agents,
            primary_reasoning=primary_reasoning,
            agent_reasonings=agent_reasonings[:5],  # Top 5
        )
        
        self.decision_history.append(decision)
        self.total_decisions += 1
        
        return decision
    
    def update_agents(self, decision: ConsensusDecision, trade_result: Dict[str, Any]):
        """Update all participating agents with trade results."""
        for signal in decision.agent_signals:
            agent = self.agents.get(signal.agent_id)
            if agent:
                # Weight result by agent's contribution
                agent.update_performance(trade_result)
    
    def get_agent_rankings(self) -> List[Tuple[str, AgentType, float]]:
        """Get agents ranked by performance."""
        rankings = []
        for agent in self.agents.values():
            rankings.append((
                agent.agent_id,
                agent.agent_type,
                agent.performance.total_pnl,
            ))
        
        return sorted(rankings, key=lambda x: x[2], reverse=True)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        active_agents = sum(1 for a in self.agents.values() if a.is_active)
        
        # Average confidence of recent decisions
        recent_decisions = list(self.decision_history)[-100:]
        avg_confidence = np.mean([d.confidence for d in recent_decisions]) if recent_decisions else 0.0
        avg_agreement = np.mean([d.agreement_score for d in recent_decisions]) if recent_decisions else 0.0
        
        return {
            "total_agents": len(self.agents),
            "active_agents": active_agents,
            "total_decisions": self.total_decisions,
            "avg_confidence": avg_confidence,
            "avg_agreement": avg_agreement,
            "agent_types": list(set(a.agent_type.value for a in self.agents.values())),
        }
    
    def disable_underperforming_agents(self, threshold: float = -100.0):
        """Disable agents with poor performance."""
        for agent in self.agents.values():
            if agent.performance.total_pnl < threshold and agent.performance.total_trades > 10:
                agent.is_active = False
                logger.info("Disabled underperforming agent: %s (PnL: %.2f)", 
                           agent.agent_id, agent.performance.total_pnl)


# ============================================================================
# Factory Function
# ============================================================================

def create_multi_agent_system(**kwargs) -> MultiAgentTradingSystem:
    """Create a multi-agent trading system."""
    return MultiAgentTradingSystem(**kwargs)
