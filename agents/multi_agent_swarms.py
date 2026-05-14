"""
multi_agent_swarms.py — Multi-Agent Trading Swarm

Multiple AI agents compete and collaborate to find the best trades.

Architecture:
- SwarmOrchestrator: Manages all agents, runs competition
- Specialized Agents: Each agent has different strategy/focus
- Competition: Agents compete on paper, best performer gets real capital
- Consensus: When multiple agents agree, confidence is high
- Evolution: Poor performers get replaced with new strategies

Why Swarm Intelligence?
- Multiple perspectives reduce blind spots
- Competition drives improvement
- Consensus reduces false signals
- Diversity handles different market conditions
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of trading agents."""
    MOMENTUM = "momentum"           # Follows trends
    MEAN_REVERSION = "mean_reversion"  # Fades extremes
    BREAKOUT = "breakout"           # Trades breakouts
    SCALPER = "scalper"             # Quick small gains
    SWING = "swing"                 # Multi-day holds
    WHALE_FOLLOWER = "whale_follower"  # Follows big money
    CONTRARIAN = "contrarian"       # Goes against crowd
    ML_PREDICTOR = "ml_predictor"   # ML-based predictions


class AgentVerdict(Enum):
    """Agent trading verdicts."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class TradeSignal:
    """A trade signal from an agent."""
    agent_id: str
    agent_type: AgentType
    symbol: str
    verdict: AgentVerdict
    confidence: float  # 0-1
    target_price: float
    stop_loss: float
    take_profit: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "target_price": self.target_price,
            "reasoning": self.reasoning,
        }


@dataclass
class AgentPerformance:
    """Track agent performance."""
    agent_id: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    elo_rating: float = 1500.0  # Chess-style rating
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.5
        return self.winning_trades / self.total_trades
    
    @property
    def avg_pnl(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades
    
    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "total_trades": self.total_trades,
            "win_rate": f"{self.win_rate*100:.1f}%",
            "total_pnl": f"${self.total_pnl:.2f}",
            "elo_rating": f"{self.elo_rating:.0f}",
        }


class TradingAgent(ABC):
    """Base class for trading agents."""
    
    def __init__(self, agent_id: str, agent_type: AgentType):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.performance = AgentPerformance(agent_id=agent_id)
        self.is_active = True
        
    @abstractmethod
    def analyze(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        additional_data: Optional[Dict] = None,
    ) -> TradeSignal:
        """Analyze market and return signal."""
        pass
    
    def update_performance(self, pnl: float, won: bool):
        """Update agent performance."""
        self.performance.total_trades += 1
        self.performance.total_pnl += pnl
        
        if won:
            self.performance.winning_trades += 1
        else:
            self.performance.losing_trades += 1
        
        # Update ELO rating
        expected = 1 / (1 + 10 ** ((1500 - self.performance.elo_rating) / 400))
        actual = 1.0 if won else 0.0
        k_factor = 32
        self.performance.elo_rating += k_factor * (actual - expected)


class MomentumAgent(TradingAgent):
    """Follows trends - buys when price above moving averages."""
    
    def __init__(self):
        super().__init__("momentum_001", AgentType.MOMENTUM)
    
    def analyze(self, prices, volumes, additional_data=None):
        if len(prices) < 20:
            return self._neutral_signal("Insufficient data")
        
        ma_10 = np.mean(prices[-10:])
        ma_20 = np.mean(prices[-20:])
        current = prices[-1]
        
        # Bullish: price > MA10 > MA20
        if current > ma_10 > ma_20:
            strength = (current - ma_20) / ma_20
            if strength > 0.05:
                verdict = AgentVerdict.STRONG_BUY
            else:
                verdict = AgentVerdict.BUY
            confidence = min(0.9, 0.5 + strength * 2)
            reasoning = f"Bullish MA alignment: {current:.2f} > MA10 {ma_10:.2f} > MA20 {ma_20:.2f}"
        
        # Bearish: price < MA10 < MA20
        elif current < ma_10 < ma_20:
            verdict = AgentVerdict.SELL
            confidence = 0.6
            reasoning = f"Bearish MA alignment: {current:.2f} < MA10 {ma_10:.2f} < MA20 {ma_20:.2f}"
        
        else:
            verdict = AgentVerdict.NEUTRAL
            confidence = 0.4
            reasoning = "Mixed MA signals"
        
        return TradeSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol="BTC/USDT",
            verdict=verdict,
            confidence=confidence,
            target_price=current,
            stop_loss=current * 0.97,
            take_profit=current * 1.06,
            reasoning=reasoning,
        )
    
    def _neutral_signal(self, reason: str) -> TradeSignal:
        return TradeSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol="BTC/USDT",
            verdict=AgentVerdict.NEUTRAL,
            confidence=0.0,
            target_price=0,
            stop_loss=0,
            take_profit=0,
            reasoning=reason,
        )


class MeanReversionAgent(TradingAgent):
    """Fades extremes - sells when overbought, buys when oversold."""
    
    def __init__(self):
        super().__init__("mean_reversion_001", AgentType.MEAN_REVERSION)
    
    def analyze(self, prices, volumes, additional_data=None):
        if len(prices) < 14:
            return self._neutral_signal("Insufficient data")
        
        # RSI calculation
        deltas = np.diff(prices[-15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        
        current = prices[-1]
        
        if rsi < 30:
            verdict = AgentVerdict.STRONG_BUY
            confidence = min(0.9, (30 - rsi) / 30 + 0.5)
            reasoning = f"Oversold RSI {rsi:.1f} - expecting mean reversion bounce"
        elif rsi < 40:
            verdict = AgentVerdict.BUY
            confidence = 0.6
            reasoning = f"Approaching oversold RSI {rsi:.1f}"
        elif rsi > 70:
            verdict = AgentVerdict.STRONG_SELL
            confidence = min(0.9, (rsi - 70) / 30 + 0.5)
            reasoning = f"Overbought RSI {rsi:.1f} - expecting pullback"
        elif rsi > 60:
            verdict = AgentVerdict.SELL
            confidence = 0.6
            reasoning = f"Approaching overbought RSI {rsi:.1f}"
        else:
            verdict = AgentVerdict.NEUTRAL
            confidence = 0.3
            reasoning = f"RSI neutral {rsi:.1f}"
        
        return TradeSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol="BTC/USDT",
            verdict=verdict,
            confidence=confidence,
            target_price=current,
            stop_loss=current * 0.97,
            take_profit=current * 1.03,
            reasoning=reasoning,
        )
    
    def _neutral_signal(self, reason):
        return TradeSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol="BTC/USDT",
            verdict=AgentVerdict.NEUTRAL,
            confidence=0.0,
            target_price=0, stop_loss=0, take_profit=0,
            reasoning=reason,
        )


class WhaleFollowerAgent(TradingAgent):
    """Follows whale activity - buys when whales buy."""
    
    def __init__(self):
        super().__init__("whale_follower_001", AgentType.WHALE_FOLLOWER)
    
    def analyze(self, prices, volumes, additional_data=None):
        if len(prices) < 10 or len(volumes) < 10:
            return self._neutral_signal("Insufficient data")
        
        current = prices[-1]
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
        price_change = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0
        
        # High volume + price up = whale buying
        if volume_ratio > 2.0 and price_change > 0.01:
            verdict = AgentVerdict.STRONG_BUY
            confidence = min(0.85, 0.5 + volume_ratio * 0.1)
            reasoning = f"Whale accumulation: {volume_ratio:.1f}x volume, +{price_change*100:.1f}% price"
        
        # High volume + price down = whale selling
        elif volume_ratio > 2.0 and price_change < -0.01:
            verdict = AgentVerdict.STRONG_SELL
            confidence = min(0.85, 0.5 + volume_ratio * 0.1)
            reasoning = f"Whale distribution: {volume_ratio:.1f}x volume, {price_change*100:.1f}% price"
        
        else:
            verdict = AgentVerdict.NEUTRAL
            confidence = 0.3
            reasoning = f"No whale activity ({volume_ratio:.1f}x volume)"
        
        return TradeSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol="BTC/USDT",
            verdict=verdict,
            confidence=confidence,
            target_price=current,
            stop_loss=current * 0.97,
            take_profit=current * 1.05,
            reasoning=reasoning,
        )
    
    def _neutral_signal(self, reason):
        return TradeSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol="BTC/USDT",
            verdict=AgentVerdict.NEUTRAL,
            confidence=0.0,
            target_price=0, stop_loss=0, take_profit=0,
            reasoning=reason,
        )


class BreakoutAgent(TradingAgent):
    """Trades breakouts - buys when price breaks resistance."""
    
    def __init__(self):
        super().__init__("breakout_001", AgentType.BREAKOUT)
        self.resistance_level = None
        self.support_level = None
    
    def analyze(self, prices, volumes, additional_data=None):
        if len(prices) < 20:
            return self._neutral_signal("Insufficient data")
        
        current = prices[-1]
        recent_high = np.max(prices[-20:])
        recent_low = np.min(prices[-20:])
        
        # Update levels
        if self.resistance_level is None or current > self.resistance_level:
            self.resistance_level = recent_high
        if self.support_level is None or current < self.support_level:
            self.support_level = recent_low
        
        # Breakout above resistance
        if current > recent_high * 0.995:  # Within 0.5% of high
            volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 else 1.0
            
            if volume_ratio > 1.5:
                verdict = AgentVerdict.BUY
                confidence = min(0.8, 0.4 + volume_ratio * 0.1)
                reasoning = f"Breakout attempt: {current:.2f} near resistance {recent_high:.2f}, volume {volume_ratio:.1f}x"
            else:
                verdict = AgentVerdict.NEUTRAL
                confidence = 0.4
                reasoning = f"Near resistance {recent_high:.2f} but low volume"
        
        # Breakdown below support
        elif current < recent_low * 1.005:  # Within 0.5% of low
            verdict = AgentVerdict.SELL
            confidence = 0.6
            reasoning = f"Breakdown risk: {current:.2f} near support {recent_low:.2f}"
        
        else:
            verdict = AgentVerdict.NEUTRAL
            confidence = 0.3
            reasoning = f"Range-bound: {recent_low:.2f} - {recent_high:.2f}"
        
        return TradeSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol="BTC/USDT",
            verdict=verdict,
            confidence=confidence,
            target_price=current,
            stop_loss=recent_low * 0.99,
            take_profit=recent_high * 1.1,
            reasoning=reasoning,
        )
    
    def _neutral_signal(self, reason):
        return TradeSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol="BTC/USDT",
            verdict=AgentVerdict.NEUTRAL,
            confidence=0.0,
            target_price=0, stop_loss=0, take_profit=0,
            reasoning=reason,
        )


class SwarmOrchestrator:
    """
    Orchestrates multiple trading agents.
    
    Features:
    - Runs all agents in parallel
    - Collects signals
    - Builds consensus
    - Tracks performance
    - Evolves agent weights
    """
    
    VERDICT_SCORES = {
        AgentVerdict.STRONG_BUY: 2.0,
        AgentVerdict.BUY: 1.0,
        AgentVerdict.NEUTRAL: 0.0,
        AgentVerdict.SELL: -1.0,
        AgentVerdict.STRONG_SELL: -2.0,
    }
    
    def __init__(self):
        self.agents: Dict[str, TradingAgent] = {}
        self.signal_history: List[Dict] = []
        self.consensus_history: List[Dict] = []
        
        # Initialize default swarm
        self._initialize_default_swarm()
        
        logger.info("=" * 60)
        logger.info("MULTI-AGENT SWARM INITIALIZED")
        logger.info(f"Agents: {len(self.agents)}")
        logger.info("=" * 60)
    
    def _initialize_default_swarm(self):
        """Initialize default set of agents."""
        self.agents = {
            "momentum": MomentumAgent(),
            "mean_reversion": MeanReversionAgent(),
            "whale_follower": WhaleFollowerAgent(),
            "breakout": BreakoutAgent(),
        }
    
    def add_agent(self, agent: TradingAgent):
        """Add agent to swarm."""
        self.agents[agent.agent_id] = agent
        logger.info("Agent added: %s (%s)", agent.agent_id, agent.agent_type.value)
    
    def remove_agent(self, agent_id: str):
        """Remove agent from swarm."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info("Agent removed: %s", agent_id)
    
    def get_signals(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        additional_data: Optional[Dict] = None,
    ) -> List[TradeSignal]:
        """Get signals from all active agents."""
        signals = []
        
        for agent in self.agents.values():
            if agent.is_active:
                try:
                    signal = agent.analyze(prices, volumes, additional_data)
                    signals.append(signal)
                except Exception as e:
                    logger.warning("Agent %s failed: %s", agent.agent_id, e)
        
        self.signal_history.append({
            "timestamp": datetime.now().isoformat(),
            "signals": [s.to_dict() for s in signals],
        })
        
        return signals
    
    def build_consensus(
        self,
        signals: List[TradeSignal],
    ) -> Dict[str, Any]:
        """
        Build consensus from multiple agent signals.
        
        Methods:
        1. Weighted voting (by ELO rating)
        2. Confidence weighting
        3. Majority voting
        """
        if not signals:
            return {
                "verdict": AgentVerdict.NEUTRAL,
                "confidence": 0.0,
                "method": "no_signals",
                "details": {},
            }
        
        # Method 1: Weighted score by ELO
        weighted_score = 0.0
        total_weight = 0.0
        
        for signal in signals:
            agent = self.agents.get(signal.agent_id)
            weight = agent.performance.elo_rating if agent else 1500
            score = self.VERDICT_SCORES.get(signal.verdict, 0)
            
            weighted_score += score * weight * signal.confidence
            total_weight += weight * signal.confidence
        
        avg_weighted_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Method 2: Majority vote
        verdict_counts = {}
        for signal in signals:
            verdict_counts[signal.verdict] = verdict_counts.get(signal.verdict, 0) + 1
        
        majority_verdict = max(verdict_counts.items(), key=lambda x: x[1])[0] if verdict_counts else AgentVerdict.NEUTRAL
        
        # Method 3: Confidence-weighted average
        conf_weighted_score = sum(
            self.VERDICT_SCORES.get(s.verdict, 0) * s.confidence
            for s in signals
        ) / len(signals)
        
        # Combine methods
        final_score = (avg_weighted_score + conf_weighted_score) / 2
        
        # Determine final verdict
        if final_score > 1.0:
            verdict = AgentVerdict.STRONG_BUY
        elif final_score > 0.3:
            verdict = AgentVerdict.BUY
        elif final_score < -1.0:
            verdict = AgentVerdict.STRONG_SELL
        elif final_score < -0.3:
            verdict = AgentVerdict.SELL
        else:
            verdict = AgentVerdict.NEUTRAL
        
        # Calculate consensus confidence
        agreement = sum(1 for s in signals if s.verdict == verdict) / len(signals)
        avg_confidence = np.mean([s.confidence for s in signals])
        consensus_confidence = (agreement + avg_confidence) / 2
        
        consensus = {
            "verdict": verdict,
            "confidence": consensus_confidence,
            "score": final_score,
            "agreement": agreement,
            "method": "weighted_consensus",
            "n_agents": len(signals),
            "agent_signals": [s.to_dict() for s in signals],
            "majority_verdict": majority_verdict.value,
        }
        
        self.consensus_history.append(consensus)
        
        return consensus
    
    def get_agent_rankings(self) -> List[Dict[str, Any]]:
        """Get agents ranked by performance."""
        rankings = []
        
        for agent in self.agents.values():
            rankings.append({
                "rank": 0,  # Will be set after sorting
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type.value,
                "elo_rating": agent.performance.elo_rating,
                "win_rate": agent.performance.win_rate,
                "total_pnl": agent.performance.total_pnl,
                "total_trades": agent.performance.total_trades,
                "is_active": agent.is_active,
            })
        
        # Sort by ELO rating
        rankings.sort(key=lambda x: x["elo_rating"], reverse=True)
        
        # Assign ranks
        for i, r in enumerate(rankings):
            r["rank"] = i + 1
        
        return rankings
    
    def evolve_swarm(self, min_elo: float = 1400):
        """
        Evolve swarm - remove underperformers, add new agents.
        
        If an agent's ELO drops below threshold, replace with new strategy.
        """
        removed = []
        added = []
        
        for agent_id, agent in list(self.agents.items()):
            if agent.performance.elo_rating < min_elo and agent.performance.total_trades > 10:
                # Remove underperformer
                del self.agents[agent_id]
                removed.append(agent_id)
                
                # Add new agent with different type
                new_agent = self._create_new_agent()
                self.agents[new_agent.agent_id] = new_agent
                added.append(new_agent.agent_id)
        
        if removed or added:
            logger.info("Swarm evolved: removed=%s, added=%s", removed, added)
        
        return {"removed": removed, "added": added}
    
    def _create_new_agent(self) -> TradingAgent:
        """Create a new agent with random strategy."""
        agent_types = [MomentumAgent, MeanReversionAgent, WhaleFollowerAgent, BreakoutAgent]
        agent_class = np.random.choice(agent_types)
        return agent_class()
    
    def get_swarm_report(self) -> Dict[str, Any]:
        """Get comprehensive swarm report."""
        return {
            "n_agents": len(self.agents),
            "active_agents": sum(1 for a in self.agents.values() if a.is_active),
            "agent_rankings": self.get_agent_rankings(),
            "total_signals": len(self.signal_history),
            "total_consensus": len(self.consensus_history),
        }


# Factory function
def create_trading_swarm() -> SwarmOrchestrator:
    """Create trading swarm."""
    return SwarmOrchestrator()
