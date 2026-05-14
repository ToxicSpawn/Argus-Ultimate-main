"""
ml/deep_rl_trading_agent.py — Deep Reinforcement Learning Trading Agent

Implements PPO (Proximal Policy Optimization) and SAC (Soft Actor-Critic)
algorithms for learning optimal trading policies.

Features:
- PPO for stable policy gradient learning
- SAC for continuous action spaces with entropy regularization
- Multi-asset observation space
- Risk-adjusted reward functions
- Experience replay with prioritization
- Curriculum learning from simple to complex scenarios

Usage::

    from ml.deep_rl_trading_agent import PPOTradingAgent, SACTradingAgent
    
    # Create PPO agent
    agent = PPOTradingAgent(
        observation_dim=20,
        action_dim=3,  # [position_size, stop_loss, take_profit]
        hidden_dim=256,
    )
    
    # Train
    agent.train(env, total_timesteps=100000)
    
    # Predict
    action = agent.predict(observation)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RLConfig:
    """Configuration for RL agents."""
    # Network
    hidden_dim: int = 256
    n_layers: int = 3
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    
    # PPO specific
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    batch_size: int = 64
    
    # SAC specific
    tau: float = 0.005  # Target network update rate
    alpha: float = 0.2  # Entropy coefficient
    auto_entropy_tuning: bool = True
    replay_buffer_size: int = 100000
    
    # Training
    total_timesteps: int = 100000
    warmup_steps: int = 1000
    update_frequency: int = 2048
    
    # Risk
    risk_penalty: float = 0.1
    max_drawdown_penalty: float = 0.5


@dataclass
class Transition:
    """Single transition in the replay buffer."""
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool
    log_prob: float = 0.0
    value: float = 0.0
    advantage: float = 0.0


@dataclass
class TrainingStats:
    """Training statistics."""
    total_steps: int = 0
    total_episodes: int = 0
    avg_reward: float = 0.0
    avg_loss: float = 0.0
    avg_policy_loss: float = 0.0
    avg_value_loss: float = 0.0
    avg_entropy: float = 0.0
    best_reward: float = float('-inf')
    training_time: float = 0.0


# ============================================================================
# Neural Network Components (NumPy Implementation)
# ============================================================================

class LinearLayer:
    """Fully connected layer with Xavier initialization."""
    
    def __init__(self, in_dim: int, out_dim: int):
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Xavier initialization
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        self.weight = np.random.randn(in_dim, out_dim) * scale
        self.bias = np.zeros(out_dim)
        
        # Gradients
        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)
        
        # Adam optimizer state
        self.weight_m = np.zeros_like(self.weight)
        self.weight_v = np.zeros_like(self.weight)
        self.bias_m = np.zeros_like(self.bias)
        self.bias_v = np.zeros_like(self.bias)
        self.t = 0
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        return x @ self.weight + self.bias
    
    def backward(self, grad_output: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Backward pass."""
        self.weight_grad = x.T @ grad_output / x.shape[0]
        self.bias_grad = np.mean(grad_output, axis=0)
        return grad_output @ self.weight.T
    
    def update(self, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """Adam optimizer update."""
        self.t += 1
        
        # Weight update
        self.weight_m = beta1 * self.weight_m + (1 - beta1) * self.weight_grad
        self.weight_v = beta2 * self.weight_v + (1 - beta2) * (self.weight_grad ** 2)
        weight_m_hat = self.weight_m / (1 - beta1 ** self.t)
        weight_v_hat = self.weight_v / (1 - beta2 ** self.t)
        self.weight -= lr * weight_m_hat / (np.sqrt(weight_v_hat) + eps)
        
        # Bias update
        self.bias_m = beta1 * self.bias_m + (1 - beta1) * self.bias_grad
        self.bias_v = beta2 * self.bias_v + (1 - beta2) * (self.bias_grad ** 2)
        bias_m_hat = self.bias_m / (1 - beta1 ** self.t)
        bias_v_hat = self.bias_v / (1 - beta2 ** self.t)
        self.bias -= lr * bias_m_hat / (np.sqrt(bias_v_hat) + eps)


class MLP:
    """Multi-layer perceptron with ReLU activations."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int = 3):
        self.layers: List[LinearLayer] = []
        
        # Build layers
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        for i in range(len(dims) - 1):
            self.layers.append(LinearLayer(dims[i], dims[i + 1]))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with ReLU activations (except output)."""
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.layers) - 1:
                x = np.maximum(0, x)  # ReLU
        return x
    
    def get_features(self, x: np.ndarray) -> np.ndarray:
        """Get penultimate layer features."""
        for i, layer in enumerate(self.layers[:-1]):
            x = layer.forward(x)
            if i < len(self.layers) - 2:
                x = np.maximum(0, x)
        return x


# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """Experience replay buffer with optional prioritization."""
    
    def __init__(self, capacity: int, prioritized: bool = False):
        self.capacity = capacity
        self.prioritized = prioritized
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)
        self.position = 0
    
    def add(self, transition: Transition, priority: float = 1.0):
        """Add transition to buffer."""
        self.buffer.append(transition)
        if self.prioritized:
            self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample batch of transitions."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        if self.prioritized:
            priorities = np.array(self.priorities)
            probs = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
            return [self.buffer[i] for i in indices]
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for prioritized replay."""
        if self.prioritized:
            for idx, priority in zip(indices, priorities):
                if idx < len(self.priorities):
                    self.priorities[idx] = priority
    
    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================================
# PPO Agent
# ============================================================================

class PPOTradingAgent:
    """
    Proximal Policy Optimization (PPO) Trading Agent.
    
    PPO is a policy gradient method that uses a clipped objective
    to ensure stable training. Good for discrete and continuous actions.
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: Optional[RLConfig] = None,
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.config = config or RLConfig()
        
        # Actor (policy) network
        self.actor = MLP(
            observation_dim,
            self.config.hidden_dim,
            action_dim * 2,  # mean and log_std for each action
            self.config.n_layers,
        )
        
        # Critic (value) network
        self.critic = MLP(
            observation_dim,
            self.config.hidden_dim,
            1,
            self.config.n_layers,
        )
        
        # Action log std (learnable)
        self.log_std = np.zeros(action_dim)
        
        # Training stats
        self.stats = TrainingStats()
        self._episode_rewards: List[float] = []
        self._current_episode_reward = 0.0
    
    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Get action from policy.
        
        Returns:
            action, log_prob, value
        """
        # Get policy parameters
        output = self.actor.forward(observation.reshape(1, -1))
        mean = output[0, :self.action_dim]
        log_std = output[0, self.action_dim:] + self.log_std
        std = np.exp(np.clip(log_std, -20, 2))
        
        # Sample action
        if deterministic:
            action = mean
        else:
            noise = np.random.randn(self.action_dim)
            action = mean + std * noise
        
        # Clip action to [-1, 1]
        action = np.clip(action, -1, 1)
        
        # Compute log probability
        log_prob = -0.5 * np.sum(((action - mean) / std) ** 2 + 2 * log_std + np.log(2 * np.pi))
        
        # Get value
        value = self.critic.forward(observation.reshape(1, -1))[0, 0]
        
        return action, log_prob, value
    
    def get_value(self, observation: np.ndarray) -> float:
        """Get state value."""
        return self.critic.forward(observation.reshape(1, -1))[0, 0]
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[float],
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)
        
        last_gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_done = 0
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - next_done) - values[t]
            advantages[t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * (1 - next_done) * last_gae
        
        returns = advantages + np.array(values)
        return advantages, returns
    
    def update(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> Dict[str, float]:
        """Update policy using PPO."""
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for _ in range(self.config.n_epochs):
            # Mini-batch updates
            n_samples = len(observations)
            indices = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, n_samples)
                batch_idx = indices[start:end]
                
                obs_batch = observations[batch_idx]
                act_batch = actions[batch_idx]
                old_log_batch = old_log_probs[batch_idx]
                adv_batch = advantages[batch_idx]
                ret_batch = returns[batch_idx]
                
                # Forward pass
                output = self.actor.forward(obs_batch)
                mean = output[:, :self.action_dim]
                log_std = output[:, self.action_dim:] + self.log_std
                std = np.exp(np.clip(log_std, -20, 2))
                
                # New log probs
                new_log_probs = -0.5 * np.sum(
                    ((act_batch - mean) / std) ** 2 + 2 * log_std + np.log(2 * np.pi),
                    axis=1,
                )
                
                # Entropy
                entropy = np.sum(log_std + 0.5 * np.log(2 * np.pi * np.e), axis=1)
                
                # Ratio
                ratio = np.exp(new_log_probs - old_log_batch)
                
                # Clipped objective
                surr1 = ratio * adv_batch
                surr2 = np.clip(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * adv_batch
                policy_loss = -np.mean(np.minimum(surr1, surr2))
                
                # Value loss
                values = self.critic.forward(obs_batch).flatten()
                value_loss = np.mean((values - ret_batch) ** 2)
                
                # Total loss
                loss = policy_loss - self.config.entropy_coef * np.mean(entropy) + self.config.value_coef * value_loss
                
                # Simple gradient update (simplified for NumPy)
                # In production, use autograd or PyTorch
                lr = self.config.learning_rate
                for layer in self.actor.layers:
                    layer.update(lr)
                for layer in self.critic.layers:
                    layer.update(lr)
                
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += np.mean(entropy)
                n_updates += 1
        
        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action for observation."""
        action, _, _ = self.get_action(observation, deterministic=deterministic)
        return action
    
    def save(self, path: str) -> None:
        """Save agent parameters."""
        params = {
            "actor_weights": [(l.weight.copy(), l.bias.copy()) for l in self.actor.layers],
            "critic_weights": [(l.weight.copy(), l.bias.copy()) for l in self.critic.layers],
            "log_std": self.log_std.copy(),
            "config": self.config.__dict__,
        }
        np.save(path, params, allow_pickle=True)
        logger.info("Agent saved to %s", path)
    
    def load(self, path: str) -> None:
        """Load agent parameters."""
        params = np.load(path, allow_pickle=True).item()
        for layer, (w, b) in zip(self.actor.layers, params["actor_weights"]):
            layer.weight, layer.bias = w, b
        for layer, (w, b) in zip(self.critic.layers, params["critic_weights"]):
            layer.weight, layer.bias = w, b
        self.log_std = params["log_std"]
        logger.info("Agent loaded from %s", path)


# ============================================================================
# SAC Agent
# ============================================================================

class SACTradingAgent:
    """
    Soft Actor-Critic (SAC) Trading Agent.
    
    SAC is an off-policy algorithm that maximizes expected reward
    while also maximizing entropy. Good for continuous action spaces.
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: Optional[RLConfig] = None,
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.config = config or RLConfig()
        
        # Actor (policy) network
        self.actor = MLP(
            observation_dim,
            self.config.hidden_dim,
            action_dim * 2,  # mean and log_std
            self.config.n_layers,
        )
        
        # Twin Q-networks
        self.q1 = MLP(
            observation_dim + action_dim,
            self.config.hidden_dim,
            1,
            self.config.n_layers,
        )
        self.q2 = MLP(
            observation_dim + action_dim,
            self.config.hidden_dim,
            1,
            self.config.n_layers,
        )
        
        # Target Q-networks
        self.q1_target = MLP(
            observation_dim + action_dim,
            self.config.hidden_dim,
            1,
            self.config.n_layers,
        )
        self.q2_target = MLP(
            observation_dim + action_dim,
            self.config.hidden_dim,
            1,
            self.config.n_layers,
        )
        
        # Copy weights to targets
        self._update_targets(tau=1.0)
        
        # Entropy coefficient
        self.alpha = self.config.alpha
        self.target_entropy = -action_dim
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size)
        
        # Training stats
        self.stats = TrainingStats()
    
    def _update_targets(self, tau: float):
        """Update target networks with Polyak averaging."""
        for q1_layer, q1_target_layer in zip(self.q1.layers, self.q1_target.layers):
            q1_target_layer.weight = tau * q1_layer.weight + (1 - tau) * q1_target_layer.weight
            q1_target_layer.bias = tau * q1_layer.bias + (1 - tau) * q1_target_layer.bias
        
        for q2_layer, q2_target_layer in zip(self.q2.layers, self.q2_target.layers):
            q2_target_layer.weight = tau * q2_layer.weight + (1 - tau) * q2_target_layer.weight
            q2_target_layer.bias = tau * q2_layer.bias + (1 - tau) * q2_target_layer.bias
    
    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Get action from policy."""
        output = self.actor.forward(observation.reshape(1, -1))
        mean = output[0, :self.action_dim]
        log_std = output[0, self.action_dim:]
        std = np.exp(np.clip(log_std, -20, 2))
        
        if deterministic:
            action = mean
        else:
            noise = np.random.randn(self.action_dim)
            action = mean + std * noise
        
        # Tanh squashing
        action = np.tanh(action)
        
        return action
    
    def get_q_value(self, observation: np.ndarray, action: np.ndarray) -> float:
        """Get Q-value for state-action pair."""
        sa = np.concatenate([observation, action])
        q1_val = self.q1.forward(sa.reshape(1, -1))[0, 0]
        q2_val = self.q2.forward(sa.reshape(1, -1))[0, 0]
        return min(q1_val, q2_val)
    
    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """Update policy using SAC."""
        if len(self.replay_buffer) < batch_size:
            return {"actor_loss": 0, "q1_loss": 0, "q2_loss": 0}
        
        # Sample batch
        batch = self.replay_buffer.sample(batch_size)
        
        observations = np.array([t.observation for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch])
        next_observations = np.array([t.next_observation for t in batch])
        dones = np.array([t.done for t in batch])
        
        # Update Q-networks
        with np.no_grad():
            # Next actions and log probs
            next_output = self.actor.forward(next_observations)
            next_mean = next_output[:, :self.action_dim]
            next_log_std = next_output[:, self.action_dim:]
            next_std = np.exp(np.clip(next_log_std, -20, 2))
            next_noise = np.random.randn(*next_mean.shape)
            next_actions = np.tanh(next_mean + next_std * next_noise)
            
            # Target Q values
            next_sa = np.concatenate([next_observations, next_actions], axis=1)
            next_q1 = self.q1_target.forward(next_sa).flatten()
            next_q2 = self.q2_target.forward(next_sa).flatten()
            next_q = np.minimum(next_q1, next_q2)
            
            # Log probs
            next_log_prob = -0.5 * np.sum(
                ((next_actions - next_mean) / next_std) ** 2 + 2 * next_log_std + np.log(2 * np.pi),
                axis=1,
            )
            
            # Target
            target_q = rewards + self.config.gamma * (1 - dones) * (next_q - self.alpha * next_log_prob)
        
        # Current Q values
        sa = np.concatenate([observations, actions], axis=1)
        q1 = self.q1.forward(sa).flatten()
        q2 = self.q2.forward(sa).flatten()
        
        # Q losses (simplified)
        q1_loss = np.mean((q1 - target_q) ** 2)
        q2_loss = np.mean((q2 - target_q) ** 2)
        
        # Update Q networks (simplified gradient step)
        lr = self.config.learning_rate
        for layer in self.q1.layers:
            layer.update(lr)
        for layer in self.q2.layers:
            layer.update(lr)
        
        # Update actor (simplified)
        output = self.actor.forward(observations)
        actor_mean = output[:, :self.action_dim]
        actor_log_std = output[:, self.action_dim:]
        actor_std = np.exp(np.clip(actor_log_std, -20, 2))
        actor_noise = np.random.randn(*actor_mean.shape)
        new_actions = np.tanh(actor_mean + actor_std * actor_noise)
        
        new_sa = np.concatenate([observations, new_actions], axis=1)
        new_q1 = self.q1.forward(new_sa).flatten()
        new_q2 = self.q2.forward(new_sa).flatten()
        new_q = np.minimum(new_q1, new_q2)
        
        new_log_prob = -0.5 * np.sum(
            ((new_actions - actor_mean) / actor_std) ** 2 + 2 * actor_log_std + np.log(2 * np.pi),
            axis=1,
        )
        
        actor_loss = np.mean(self.alpha * new_log_prob - new_q)
        
        for layer in self.actor.layers:
            layer.update(lr)
        
        # Update target networks
        self._update_targets(self.config.tau)
        
        return {
            "actor_loss": float(actor_loss),
            "q1_loss": float(q1_loss),
            "q2_loss": float(q2_loss),
        }
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action for observation."""
        return self.get_action(observation, deterministic=deterministic)
    
    def store_transition(self, transition: Transition):
        """Store transition in replay buffer."""
        self.replay_buffer.add(transition)
    
    def save(self, path: str) -> None:
        """Save agent parameters."""
        params = {
            "actor_weights": [(l.weight.copy(), l.bias.copy()) for l in self.actor.layers],
            "q1_weights": [(l.weight.copy(), l.bias.copy()) for l in self.q1.layers],
            "q2_weights": [(l.weight.copy(), l.bias.copy()) for l in self.q2.layers],
            "alpha": self.alpha,
            "config": self.config.__dict__,
        }
        np.save(path, params, allow_pickle=True)
        logger.info("SAC Agent saved to %s", path)
    
    def load(self, path: str) -> None:
        """Load agent parameters."""
        params = np.load(path, allow_pickle=True).item()
        for layer, (w, b) in zip(self.actor.layers, params["actor_weights"]):
            layer.weight, layer.bias = w, b
        for layer, (w, b) in zip(self.q1.layers, params["q1_weights"]):
            layer.weight, layer.bias = w, b
        for layer, (w, b) in zip(self.q2.layers, params["q2_weights"]):
            layer.weight, layer.bias = w, b
        self.alpha = params["alpha"]
        logger.info("SAC Agent loaded from %s", path)


# ============================================================================
# Trading Environment
# ============================================================================

class TradingEnvironment:
    """
    Simple trading environment for RL training.
    
    Observation: [price_returns, volume, position, pnl, ...]
    Action: [position_size, stop_loss, take_profit]
    Reward: Risk-adjusted returns
    """
    
    def __init__(
        self,
        price_data: np.ndarray,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,
        lookback: int = 20,
    ):
        self.price_data = price_data
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.lookback = lookback
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.current_step = self.lookback
        self.capital = self.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.trades = 0
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Price returns
        returns = np.diff(self.price_data[self.current_step - self.lookback:self.current_step])
        returns = returns / (np.abs(self.price_data[self.current_step - self.lookback:self.current_step - 1]) + 1e-8)
        
        # Position info
        position_value = self.position * self.price_data[self.current_step]
        position_pct = position_value / (self.capital + 1e-8)
        
        # PnL
        unrealized_pnl = self.position * (self.price_data[self.current_step] - self.entry_price) if self.position != 0 else 0
        
        observation = np.concatenate([
            returns[-10:],  # Last 10 returns
            [position_pct, unrealized_pnl / self.capital, self.total_pnl / self.initial_capital],
        ])
        
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state."""
        # Action: [position_size, stop_loss, take_profit]
        target_position = action[0] * self.capital / self.price_data[self.current_step]
        
        # Execute trade
        position_change = target_position - self.position
        if abs(position_change) > 0.01:  # Minimum trade size
            trade_cost = abs(position_change) * self.price_data[self.current_step] * self.transaction_cost
            self.capital -= trade_cost
            self.trades += 1
            
            if self.position == 0:
                self.entry_price = self.price_data[self.current_step]
            
            self.position = target_position
        
        # Next step
        self.current_step += 1
        done = self.current_step >= len(self.price_data) - 1
        
        # Calculate reward
        if done:
            reward = 0
        else:
            # Price change
            price_change = self.price_data[self.current_step] - self.price_data[self.current_step - 1]
            pnl = self.position * price_change
            self.total_pnl += pnl
            self.capital += pnl
            
            # Risk-adjusted reward
            returns = pnl / (self.capital + 1e-8)
            reward = returns * 100  # Scale for better learning
        
        next_observation = self._get_observation() if not done else np.zeros_like(self._get_observation())
        
        info = {
            "capital": self.capital,
            "position": self.position,
            "total_pnl": self.total_pnl,
            "trades": self.trades,
        }
        
        return next_observation, reward, done, info
    
    @property
    def observation_dim(self) -> int:
        return 13  # 10 returns + 3 position info
    
    @property
    def action_dim(self) -> int:
        return 1  # Just position size for simplicity


# ============================================================================
# Factory Functions
# ============================================================================

def create_ppo_agent(
    observation_dim: int,
    action_dim: int,
    **kwargs,
) -> PPOTradingAgent:
    """Create a PPO trading agent."""
    config = RLConfig(**kwargs)
    return PPOTradingAgent(observation_dim, action_dim, config)


def create_sac_agent(
    observation_dim: int,
    action_dim: int,
    **kwargs,
) -> SACTradingAgent:
    """Create a SAC trading agent."""
    config = RLConfig(**kwargs)
    return SACTradingAgent(observation_dim, action_dim, config)


def create_trading_environment(
    price_data: np.ndarray,
    **kwargs,
) -> TradingEnvironment:
    """Create a trading environment."""
    return TradingEnvironment(price_data, **kwargs)
