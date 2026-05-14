"""
Deep Reinforcement Learning Trading Agent
=========================================

PPO and SAC agents that learn optimal trading policies.
Self-improving through market interaction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import random
import time

from core.gpu_acceleration import initialize_gpu, GPUBatchProcessor

logger = logging.getLogger(__name__)


@dataclass
class TradingState:
    """State representation for trading environment."""
    price_history: np.ndarray
    portfolio_value: float
    position: float  # Current position size
    cash: float
    unrealized_pnl: float
    market_regime: str
    volatility: float
    timestamp: float


@dataclass
class TradingAction:
    """Action representation."""
    action_type: int  # 0: hold, 1: buy, 2: sell
    position_size: float  # 0.0 to 1.0
    confidence: float


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""
    state: TradingState
    action: TradingAction
    reward: float
    next_state: TradingState
    done: bool


class PolicyNetwork(nn.Module):
    """
    Policy network for PPO.
    Outputs action probabilities.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.device = initialize_gpu()
        
        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (action probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Position sizing head (continuous)
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output 0-1 for position size
        )
        
        self.to(self.device)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            action_probs: Probability distribution over actions
            position_size: Suggested position size (0-1)
        """
        features = self.feature_extractor(state)
        action_probs = self.policy_head(features)
        position_size = self.position_head(features)
        
        return action_probs, position_size


class ValueNetwork(nn.Module):
    """
    Value network for PPO.
    Estimates expected return from state.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.device = initialize_gpu()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.to(self.device)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate value of state."""
        return self.network(state)


class PPOMemory:
    """Memory buffer for PPO."""
    
    def __init__(self, batch_size: int = 64):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
    
    def store(self, state, action, prob, val, reward, done):
        """Store single experience."""
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        """Clear memory."""
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
    
    def generate_batches(self):
        """Generate training batches."""
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            batches
        )


class DRLTradingAgent:
    """
    Deep Reinforcement Learning Trading Agent using PPO.
    
    Learns optimal trading policy through market interaction.
    """
    
    def __init__(
        self,
        state_dim: int = 50,  # Price history + portfolio state
        action_dim: int = 3,   # Hold, Buy, Sell
        lr: float = 3e-4,
        gamma: float = 0.99,   # Discount factor
        gae_lambda: float = 0.95,  # GAE parameter
        policy_clip: float = 0.2,  # PPO clip parameter
        n_epochs: int = 10,
        batch_size: int = 64
    ):
        self.device = initialize_gpu()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value = ValueNetwork(state_dim).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        # PPO parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        
        # Memory
        self.memory = PPOMemory(batch_size)
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.win_rate = 0.5
        
        logger.info(f"DRL Agent initialized (state_dim={state_dim}, action_dim={action_dim})")
    
    def select_action(self, state: TradingState) -> TradingAction:
        """
        Select action using current policy.
        
        Args:
            state: Current trading state
            
        Returns:
            Selected action
        """
        # Convert state to tensor
        state_tensor = self._state_to_tensor(state)
        
        with torch.no_grad():
            action_probs, position_size = self.policy(state_tensor)
            value = self.value(state_tensor)
        
        # Sample action from distribution
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        prob = dist.log_prob(action)
        
        return TradingAction(
            action_type=action.item(),
            position_size=position_size.item(),
            confidence=action_probs[action].item()
        )
    
    def _state_to_tensor(self, state: TradingState) -> torch.Tensor:
        """Convert TradingState to tensor."""
        # Normalize price history
        prices = state.price_history[-50:] if len(state.price_history) > 50 else state.price_history
        if len(prices) < 50:
            prices = np.pad(prices, (50 - len(prices), 0), mode='edge')
        
        # Calculate features
        returns = np.diff(prices) / (prices[:-1] + 1e-9)
        volatility = np.std(returns) if len(returns) > 1 else 0.0
        momentum = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0.0
        
        # Portfolio features
        portfolio_features = np.array([
            state.portfolio_value / 10000.0,  # Normalized
            state.position,
            state.cash / 10000.0,
            state.unrealized_pnl / 100.0,
            volatility,
            momentum,
            1.0 if state.market_regime == 'trending' else 0.0,
            1.0 if state.market_regime == 'ranging' else 0.0,
        ])
        
        # Combine
        state_vector = np.concatenate([prices / prices[-1] - 1.0, portfolio_features])
        
        # Pad or truncate to state_dim
        if len(state_vector) < self.state_dim:
            state_vector = np.pad(state_vector, (0, self.state_dim - len(state_vector)))
        else:
            state_vector = state_vector[:self.state_dim]
        
        return torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
    
    def calculate_reward(self, prev_state: TradingState, 
                        action: TradingAction,
                        next_state: TradingState) -> float:
        """
        Calculate reward for state-action-next_state transition.
        
        Uses risk-adjusted returns (Sharpe-like).
        """
        # PnL reward
        pnl = next_state.portfolio_value - prev_state.portfolio_value
        pnl_reward = pnl / 100.0  # Scale to reasonable range
        
        # Penalty for large drawdowns
        if next_state.portfolio_value < prev_state.portfolio_value * 0.95:
            drawdown_penalty = -0.5
        else:
            drawdown_penalty = 0.0
        
        # Reward for consistent wins
        if pnl > 0:
            consistency_bonus = 0.1
        else:
            consistency_bonus = -0.1
        
        # Combine
        reward = pnl_reward + drawdown_penalty + consistency_bonus
        
        return reward
    
    def store_transition(self, state: TradingState, action: TradingAction,
                        reward: float, next_state: TradingState, done: bool):
        """Store transition in memory."""
        state_tensor = self._state_to_tensor(state)
        
        with torch.no_grad():
            state_tensor_batch = state_tensor
            action_probs, position = self.policy(state_tensor_batch)
            value = self.value(state_tensor_batch)
            
            dist = torch.distributions.Categorical(action_probs)
            prob = dist.log_prob(torch.tensor([action.action_type]))
        
        self.memory.store(
            state_tensor.cpu().numpy(),
            action.action_type,
            prob.cpu().numpy(),
            value.cpu().numpy(),
            reward,
            done
        )
    
    def learn(self):
        """Perform PPO update."""
        if len(self.memory.states) < self.memory.batch_size:
            return
        
        # Generate batches
        states, actions, old_probs, vals, rewards, dones, batches = \
            self.memory.generate_batches()
        
        # Calculate advantages
        advantages = self._calculate_advantages(rewards, vals, dones)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_probs = torch.FloatTensor(old_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        values = torch.FloatTensor(vals).to(self.device)
        
        # PPO updates
        for _ in range(self.n_epochs):
            for batch in batches:
                # Get batch data
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old_probs = old_probs[batch]
                batch_advantages = advantages[batch]
                batch_values = values[batch]
                
                # Forward pass
                action_probs, position_sizes = self.policy(batch_states)
                critic_value = self.value(batch_states).squeeze()
                
                # Calculate policy loss
                dist = torch.distributions.Categorical(action_probs)
                new_probs = dist.log_prob(batch_actions)
                prob_ratio = torch.exp(new_probs - batch_old_probs)
                
                weighted_probs = batch_advantages * prob_ratio
                weighted_clipped_probs = batch_advantages * torch.clamp(
                    prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip
                )
                policy_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                # Calculate value loss
                returns = batch_advantages + batch_values
                value_loss = F.mse_loss(critic_value, returns)
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss
                
                # Backpropagation
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
        
        # Clear memory
        self.memory.clear()
        
        logger.debug(f"PPO update complete, loss={loss.item():.4f}")
    
    def _calculate_advantages(self, rewards: np.ndarray, values: np.ndarray,
                             dones: np.ndarray) -> np.ndarray:
        """Calculate advantages using GAE."""
        advantages = np.zeros_like(rewards, dtype=np.float32)
        
        for t in range(len(rewards) - 1):
            discount = 1
            a_t = 0
            
            for k in range(t, len(rewards) - 1):
                a_t += discount * (rewards[k] + self.gamma * values[k + 1] * (1 - dones[k]) - values[k])
                discount *= self.gamma * self.gae_lambda
            
            advantages[t] = a_t
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        logger.info(f"Model loaded from {filepath}")


class ReplayBuffer:
    """Prioritized experience replay buffer for SAC."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def push(self, experience: Experience, priority: float = 1.0):
        """Add experience with priority."""
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch with priority weighting."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            p=probabilities,
            replace=False
        )
        
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


# Factory function
def create_drl_agent(agent_type: str = 'ppo', **kwargs) -> DRLTradingAgent:
    """Create DRL agent."""
    if agent_type.lower() == 'ppo':
        return DRLTradingAgent(**kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# Global agent instance
_drl_agent: Optional[DRLTradingAgent] = None


def get_drl_agent() -> Optional[DRLTradingAgent]:
    """Get global DRL agent instance."""
    global _drl_agent
    if _drl_agent is None:
        _drl_agent = create_drl_agent()
    return _drl_agent
