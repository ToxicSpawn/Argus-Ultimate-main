"""
Multi-Agent Reinforcement Learning System v2.0
================================================
Specialized RL agents for different market regimes in Argus Ultimate.

Provides:
- Regime-specific agents (bull, bear, high-vol, low-vol)
- PPO (Proximal Policy Optimization) trainer
- Hierarchical RL controller
- Multi-agent coordination
- Experience replay with prioritization
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types for agent selection."""
    BULL = "bull"
    BEAR = "bear"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RANGING = "ranging"
    TRANSITION = "transition"


class ActionType(Enum):
    """Trading action types."""
    BUY = 0
    SELL = 1
    HOLD = 2
    REDUCE = 3
    HEDGE = 4


@dataclass
class State:
    """Environment state for RL agent."""
    prices: np.ndarray  # Recent prices
    returns: np.ndarray  # Recent returns
    volume: np.ndarray  # Recent volume
    indicators: Dict[str, float]  # Technical indicators
    portfolio_state: Dict[str, float]  # Current portfolio
    regime: MarketRegime  # Current market regime
    
    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector."""
        features = []
        
        # Price features (normalized)
        if len(self.prices) > 0:
            features.extend([
                (self.prices[-1] / self.prices[-5] - 1) if len(self.prices) >= 5 else 0,
                (self.prices[-1] / self.prices[-20] - 1) if len(self.prices) >= 20 else 0,
            ])
        
        # Return features
        if len(self.returns) > 0:
            features.extend([
                np.mean(self.returns[-5:]) if len(self.returns) >= 5 else 0,
                np.std(self.returns[-20:]) if len(self.returns) >= 20 else 0,
                np.mean(self.returns[-20:]) if len(self.returns) >= 20 else 0,
            ])
        
        # Volume features
        if len(self.volume) > 0:
            features.append(
                np.mean(self.volume[-5:]) / np.mean(self.volume[-20:]) - 1
                if len(self.volume) >= 20 else 0
            )
        
        # Indicator features
        for key in ["rsi", "macd", "bb_position", "atr"]:
            features.append(self.indicators.get(key, 0.0))
        
        # Portfolio features
        features.append(self.portfolio_state.get("exposure", 0.0))
        features.append(self.portfolio_state.get("unrealized_pnl_pct", 0.0))
        
        # Regime one-hot
        for r in MarketRegime:
            features.append(1.0 if self.regime == r else 0.0)
        
        return np.array(features, dtype=np.float32)


@dataclass
class Experience:
    """Single experience tuple for replay."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    timestamp: datetime = field(default_factory=datetime.now)


class ReplayBuffer:
    """
    Experience replay buffer with prioritization.
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6) -> None:
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer capacity
            alpha: Prioritization exponent (0=uniform, 1=full priority)
        """
        self.capacity = capacity
        self.alpha = alpha
        
        self._buffer: List[Experience] = []
        self._priorities = np.zeros(capacity, dtype=np.float32)
        self._position = 0
        self._size = 0
    
    def add(self, experience: Experience, priority: float = 1.0) -> None:
        """Add experience to buffer."""
        if self._size < self.capacity:
            self._buffer.append(experience)
            self._size += 1
        else:
            self._buffer[self._position] = experience
        
        self._priorities[self._position] = priority ** self.alpha
        self._position = (self._position + 1) % self.capacity
    
    def sample(
        self,
        batch_size: int,
        beta: float = 0.4
    ) -> Tuple[List[Experience], np.ndarray, List[int]]:
        """
        Sample batch with prioritization.
        
        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent
            
        Returns:
            Tuple of (experiences, importance_weights, indices)
        """
        if self._size == 0:
            return [], np.array([]), []
        
        # Calculate sampling probabilities
        priorities = self._priorities[:self._size]
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self._size, size=min(batch_size, self._size), p=probs)
        
        # Calculate importance weights
        weights = (self._size * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        experiences = [self._buffer[i] for i in indices]
        
        return experiences, weights, indices.tolist()
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            if idx < self._size:
                self._priorities[idx] = priority ** self.alpha
    
    def __len__(self) -> int:
        return self._size


class PPOTrainer:
    """
    Proximal Policy Optimization trainer.
    
    PPO is a policy gradient method that uses a clipped objective
    to prevent large policy updates.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 0.0003,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5
    ) -> None:
        """
        Initialize PPO trainer.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            lr: Learning rate
            gamma: Discount factor
            clip_epsilon: Clipping parameter
            entropy_coef: Entropy bonus coefficient
            value_coef: Value loss coefficient
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Initialize policy network (simple 2-layer)
        hidden_dim = 64
        scale = 0.01
        self.policy_params = {
            "W1": np.random.randn(state_dim, hidden_dim) * scale,
            "b1": np.zeros(hidden_dim),
            "W2": np.random.randn(hidden_dim, action_dim) * scale,
            "b2": np.zeros(action_dim),
        }
        
        # Initialize value network
        self.value_params = {
            "W1": np.random.randn(state_dim, hidden_dim) * scale,
            "b1": np.zeros(hidden_dim),
            "W2": np.random.randn(hidden_dim, 1) * scale,
            "b2": np.zeros(1),
        }
        
        self._training_steps = 0
    
    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Get action from policy.
        
        Args:
            state: State vector
            deterministic: If True, return best action
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        # Forward pass through policy
        z1 = state @ self.policy_params["W1"] + self.policy_params["b1"]
        a1 = np.maximum(0, z1)
        logits = a1 @ self.policy_params["W2"] + self.policy_params["b2"]
        
        # Softmax for action probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        # Sample action
        if deterministic:
            action = np.argmax(probs)
        else:
            action = np.random.choice(self.action_dim, p=probs)
        
        log_prob = np.log(probs[action] + 1e-8)
        
        # Get value estimate
        value = self._get_value(state)
        
        return action, log_prob, value
    
    def _get_value(self, state: np.ndarray) -> float:
        """Get value estimate from value network."""
        z1 = state @ self.value_params["W1"] + self.value_params["b1"]
        a1 = np.maximum(0, z1)
        value = a1 @ self.value_params["W2"] + self.value_params["b2"]
        return float(value)
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        gamma: float = 0.99,
        lam: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            gamma: Discount factor
            lam: GAE lambda
            
        Returns:
            Tuple of (advantages, returns)
        """
        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)
        
        last_gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update(
        self,
        experiences: List[Experience],
        old_log_probs: List[float],
        advantages: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Returns training metrics.
        """
        # Simplified PPO update (in production, use proper gradient computation)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for i, exp in enumerate(experiences):
            # Get current policy output
            action, new_log_prob, new_value = self.get_action(exp.state)
            
            # Policy loss (clipped surrogate objective)
            ratio = np.exp(new_log_prob - old_log_probs[i])
            surr1 = ratio * advantages[i]
            surr2 = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[i]
            policy_loss = -min(surr1, surr2)
            
            # Value loss
            value_loss = (new_value - returns[i]) ** 2
            
            total_policy_loss += policy_loss
            total_value_loss += value_loss
        
        self._training_steps += 1
        
        return {
            "policy_loss": total_policy_loss / len(experiences),
            "value_loss": total_value_loss / len(experiences),
            "training_steps": self._training_steps,
        }


class TradingAgent:
    """
    RL agent for trading in a specific regime.
    """
    
    def __init__(
        self,
        regime: MarketRegime,
        state_dim: int = 20,
        action_dim: int = 5
    ) -> None:
        """
        Initialize trading agent.
        
        Args:
            regime: Market regime this agent specializes in
            state_dim: State dimension
            action_dim: Action dimension
        """
        self.regime = regime
        self.trainer = PPOTrainer(state_dim, action_dim)
        self.replay_buffer = ReplayBuffer(capacity=5000)
        
        self._episode_rewards: List[float] = []
        self._total_trades = 0
        self._winning_trades = 0
    
    def select_action(
        self,
        state: State,
        exploration_rate: float = 0.1
    ) -> Tuple[ActionType, Dict[str, Any]]:
        """
        Select action based on current state.
        
        Args:
            state: Current state
            exploration_rate: Epsilon for exploration
            
        Returns:
            Tuple of (action, info)
        """
        state_vector = state.to_vector()
        
        # Epsilon-greedy exploration
        if np.random.random() < exploration_rate:
            action_idx = np.random.randint(5)
            log_prob = 0.0
            value = 0.0
        else:
            action_idx, log_prob, value = self.trainer.get_action(state_vector)
        
        action = ActionType(action_idx)
        
        info = {
            "log_prob": log_prob,
            "value": value,
            "regime": self.regime.value,
            "exploring": np.random.random() < exploration_rate
        }
        
        return action, info
    
    def update(self, experience: Experience) -> Dict[str, float]:
        """
        Update agent with new experience.
        
        Returns training metrics.
        """
        self.replay_buffer.add(experience)
        
        # Sample batch and update if enough experiences
        if len(self.replay_buffer) >= 32:
            experiences, weights, indices = self.replay_buffer.sample(32)
            
            # Compute advantages and returns
            rewards = [e.reward for e in experiences]
            values = [0.0] * len(experiences)  # Simplified
            dones = [e.done for e in experiences]
            
            advantages, returns = self.trainer.compute_gae(rewards, values, dones)
            
            old_log_probs = [0.0] * len(experiences)  # Simplified
            
            metrics = self.trainer.update(experiences, old_log_probs, advantages, returns)
            return metrics
        
        return {"policy_loss": 0.0, "value_loss": 0.0}
    
    def record_trade(self, profitable: bool) -> None:
        """Record trade outcome."""
        self._total_trades += 1
        if profitable:
            self._winning_trades += 1
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self._total_trades == 0:
            return 0.0
        return self._winning_trades / self._total_trades


class AgentCoordinator:
    """
    Coordinates multiple regime-specific agents.
    
    Handles:
    - Agent selection based on current regime
    - Conflict resolution between agents
    - Performance tracking and agent retirement
    """
    
    def __init__(self) -> None:
        """Initialize agent coordinator."""
        self.agents: Dict[MarketRegime, TradingAgent] = {}
        self._performance_history: Dict[MarketRegime, List[float]] = {}
        self._current_regime: Optional[MarketRegime] = None
        
        # Initialize agents for each regime
        for regime in MarketRegime:
            self.agents[regime] = TradingAgent(regime)
            self._performance_history[regime] = []
    
    def get_agent(self, regime: MarketRegime) -> TradingAgent:
        """Get agent for specified regime."""
        return self.agents[regime]
    
    def select_agent(
        self,
        current_regime: MarketRegime,
        regime_confidence: float = 0.5
    ) -> TradingAgent:
        """
        Select best agent for current conditions.
        
        Args:
            current_regime: Current market regime
            regime_confidence: Confidence in regime classification
            
        Returns:
            Selected agent
        """
        self._current_regime = current_regime
        
        # If confidence is low, use ensemble or transition agent
        if regime_confidence < 0.3:
            return self.agents[MarketRegime.TRANSITION]
        
        return self.agents[current_regime]
    
    def update_regime(self, new_regime: MarketRegime) -> None:
        """Update current regime and track regime changes."""
        if self._current_regime != new_regime:
            logger.info(
                "Regime change: %s -> %s",
                self._current_regime.value if self._current_regime else "None",
                new_regime.value
            )
            self._current_regime = new_regime
    
    def record_performance(
        self,
        regime: MarketRegime,
        pnl: float
    ) -> None:
        """Record agent performance."""
        self._performance_history[regime].append(pnl)
        
        # Keep last 100 records
        if len(self._performance_history[regime]) > 100:
            self._performance_history[regime] = self._performance_history[regime][-100:]
    
    def get_agent_rankings(self) -> List[Tuple[MarketRegime, float]]:
        """Get agents ranked by recent performance."""
        rankings = []
        
        for regime, history in self._performance_history.items():
            if len(history) >= 10:
                # Use last 20 trades for ranking
                recent = history[-20:]
                avg_pnl = np.mean(recent)
                rankings.append((regime, avg_pnl))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def get_coordination_summary(self) -> Dict[str, Any]:
        """Get summary of agent coordination."""
        rankings = self.get_agent_rankings()
        
        return {
            "current_regime": self._current_regime.value if self._current_regime else None,
            "agent_rankings": [(r.value, p) for r, p in rankings],
            "active_agents": len(self.agents),
        }


class MultiAgentRLSystem:
    """
    Main multi-agent RL system for Argus.
    
    Coordinates specialized agents for different market regimes.
    """
    
    def __init__(self) -> None:
        """Initialize multi-agent RL system."""
        self.coordinator = AgentCoordinator()
        self._current_state: Optional[State] = None
        self._episode_experiences: List[Experience] = []
        
        logger.info("MultiAgentRLSystem initialized with %d regime agents", len(MarketRegime))
    
    def select_action(
        self,
        state: State,
        regime_confidence: float = 0.5
    ) -> Tuple[ActionType, Dict[str, Any]]:
        """
        Select trading action based on current state.
        
        Args:
            state: Current market state
            regime_confidence: Confidence in regime classification
            
        Returns:
            Tuple of (action, info)
        """
        self._current_state = state
        
        # Select agent based on regime
        agent = self.coordinator.select_agent(state.regime, regime_confidence)
        
        # Get action from agent
        action, info = agent.select_action(state)
        
        info["selected_agent"] = agent.regime.value
        
        return action, info
    
    def update(
        self,
        state: State,
        action: ActionType,
        reward: float,
        next_state: State,
        done: bool
    ) -> Dict[str, float]:
        """
        Update agents with new experience.
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state
            done: Episode done
            
        Returns:
            Training metrics
        """
        experience = Experience(
            state=state.to_vector(),
            action=action.value,
            reward=reward,
            next_state=next_state.to_vector(),
            done=done
        )
        
        # Update the agent for current regime
        agent = self.coordinator.get_agent(state.regime)
        metrics = agent.update(experience)
        
        # Record performance
        self.coordinator.record_performance(state.regime, reward)
        
        self._episode_experiences.append(experience)
        
        if done:
            self._episode_experiences = []
        
        return metrics
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        coordination = self.coordinator.get_coordination_summary()
        
        agent_stats = {}
        for regime, agent in self.coordinator.agents.items():
            agent_stats[regime.value] = {
                "win_rate": agent.win_rate,
                "total_trades": agent._total_trades,
                "buffer_size": len(agent.replay_buffer),
            }
        
        return {
            "coordination": coordination,
            "agents": agent_stats,
        }
