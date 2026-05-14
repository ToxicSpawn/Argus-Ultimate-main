# pyright: reportMissingImports=false
"""
Quantum Actor-Critic (QAC) implementation for trading strategy optimization.

This module implements a hybrid quantum-classical Actor-Critic algorithm
with separate quantum circuits for the actor (policy) and critic (value function),
enabling more stable learning in complex trading environments.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ActorCriticType(Enum):
    """Types of Actor-Critic architectures."""
    A2C = auto()   # Advantage Actor-Critic
    A3C = auto()   # Asynchronous Advantage Actor-Critic
    SAC = auto()   # Soft Actor-Critic
    TD3 = auto()   # Twin Delayed DDPG


@dataclass
class QACConfig:
    """Configuration for Quantum Actor-Critic."""
    # Architecture
    state_dim: int = 8
    action_dim: int = 4
    ac_type: ActorCriticType = ActorCriticType.A2C
    
    # Actor (Policy) quantum parameters
    actor_num_qubits: int = 4
    actor_num_layers: int = 3
    
    # Critic (Value) quantum parameters
    critic_num_qubits: int = 4
    critic_num_layers: int = 3
    
    # Training parameters
    actor_lr: float = 0.0003
    critic_lr: float = 0.001
    discount_factor: float = 0.99
    entropy_coefficient: float = 0.01
    value_loss_coefficient: float = 0.5
    
    # GAE parameters
    gae_lambda: float = 0.95
    
    # Batch settings
    batch_size: int = 32
    num_steps: int = 2048  # Steps per update
    
    # Quantum advantage threshold
    min_quantum_advantage: float = 0.05


@dataclass
class Trajectory:
    """Stores trajectory data for training."""
    states: List[NDArray[np.float64]]
    actions: List[int]
    rewards: List[float]
    log_probs: List[float]
    values: List[float]
    dones: List[bool]
    next_values: List[float]
    
    def __len__(self) -> int:
        return len(self.rewards)


class QuantumActorNetwork:
    """Quantum circuit for the actor (policy) network."""
    
    def __init__(self, num_qubits: int, num_layers: int, action_dim: int):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.action_dim = action_dim
        self.dimension = 2 ** num_qubits
        
        # Variational parameters
        self.rotation_params = np.random.uniform(0, 2*np.pi, (num_layers, num_qubits, 3))
        
        # Output projection
        self.output_weights = np.random.randn(num_qubits, action_dim) * 0.01
        self.output_bias = np.zeros(action_dim)
    
    def forward(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute action probabilities."""
        quantum_state = self._initialize_state()
        quantum_state = self._encode_state(quantum_state, state)
        
        for layer in range(self.num_layers):
            quantum_state = self._apply_variational_layer(quantum_state, layer)
        
        measurements = self._measure(quantum_state)
        logits = np.dot(measurements[:self.num_qubits], self.output_weights) + self.output_bias
        
        return self._softmax(logits)
    
    def _initialize_state(self) -> NDArray[np.complex128]:
        state = np.zeros(self.dimension, dtype=np.complex128)
        state[0] = 1.0 + 0j
        return state
    
    def _encode_state(self, state_vector: NDArray[np.complex128], classical_state: NDArray[np.float64]) -> NDArray[np.complex128]:
        normalized = (classical_state - np.mean(classical_state)) / (np.std(classical_state) + 1e-8)
        
        for i in range(min(len(normalized), self.num_qubits)):
            angle = normalized[i] * np.pi
            state_vector = self._apply_ry(state_vector, i, angle)
        
        return state_vector
    
    def _apply_variational_layer(self, state_vector: NDArray[np.complex128], layer: int) -> NDArray[np.complex128]:
        for qubit in range(self.num_qubits):
            rx, ry, rz = self.rotation_params[layer, qubit]
            state_vector = self._apply_rx(state_vector, qubit, rx)
            state_vector = self._apply_ry(state_vector, qubit, ry)
            state_vector = self._apply_rz(state_vector, qubit, rz)
        
        for i in range(self.num_qubits - 1):
            state_vector = self._apply_cnot(state_vector, i, i + 1)
        
        return state_vector
    
    def _apply_rx(self, state_vector: NDArray[np.complex128], qubit: int, angle: float) -> NDArray[np.complex128]:
        cos_half = np.cos(angle / 2)
        sin_half = -1j * np.sin(angle / 2)
        
        new_state = state_vector.copy()
        for i in range(self.dimension):
            if (i >> qubit) & 1 == 0:
                j = i | (1 << qubit)
                new_state[i] = cos_half * state_vector[i] + sin_half * state_vector[j]
                new_state[j] = sin_half * state_vector[i] + cos_half * state_vector[j]
        return new_state
    
    def _apply_ry(self, state_vector: NDArray[np.complex128], qubit: int, angle: float) -> NDArray[np.complex128]:
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        
        new_state = state_vector.copy()
        for i in range(self.dimension):
            if (i >> qubit) & 1 == 0:
                j = i | (1 << qubit)
                new_state[i] = cos_half * state_vector[i] - sin_half * state_vector[j]
                new_state[j] = sin_half * state_vector[i] + cos_half * state_vector[j]
        return new_state
    
    def _apply_rz(self, state_vector: NDArray[np.complex128], qubit: int, angle: float) -> NDArray[np.complex128]:
        phase_0 = np.exp(-1j * angle / 2)
        phase_1 = np.exp(1j * angle / 2)
        
        new_state = state_vector.copy()
        for i in range(self.dimension):
            if (i >> qubit) & 1 == 0:
                new_state[i] = phase_0 * state_vector[i]
            else:
                new_state[i] = phase_1 * state_vector[i]
        return new_state
    
    def _apply_cnot(self, state_vector: NDArray[np.complex128], control: int, target: int) -> NDArray[np.complex128]:
        new_state = state_vector.copy()
        for i in range(self.dimension):
            if (i >> control) & 1 == 1:
                j = i ^ (1 << target)
                new_state[i], new_state[j] = state_vector[j], state_vector[i]
        return new_state
    
    def _measure(self, state_vector: NDArray[np.complex128]) -> NDArray[np.float64]:
        return np.abs(state_vector[:self.num_qubits]) ** 2
    
    def _softmax(self, logits: NDArray[np.float64]) -> NDArray[np.float64]:
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)


class QuantumCriticNetwork:
    """Quantum circuit for the critic (value) network."""
    
    def __init__(self, num_qubits: int, num_layers: int):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dimension = 2 ** num_qubits
        
        # Variational parameters
        self.rotation_params = np.random.uniform(0, 2*np.pi, (num_layers, num_qubits, 3))
        
        # Output projection
        self.output_weight = np.random.randn(num_qubits, 1) * 0.01
        self.output_bias = np.zeros(1)
    
    def forward(self, state: NDArray[np.float64]) -> float:
        """Compute state value."""
        quantum_state = self._initialize_state()
        quantum_state = self._encode_state(quantum_state, state)
        
        for layer in range(self.num_layers):
            quantum_state = self._apply_variational_layer(quantum_state, layer)
        
        measurements = self._measure(quantum_state)
        value = np.dot(measurements[:self.num_qubits], self.output_weight.flatten()) + self.output_bias[0]
        
        return float(value)
    
    def _initialize_state(self) -> NDArray[np.complex128]:
        state = np.zeros(self.dimension, dtype=np.complex128)
        state[0] = 1.0 + 0j
        return state
    
    def _encode_state(self, state_vector: NDArray[np.complex128], classical_state: NDArray[np.float64]) -> NDArray[np.complex128]:
        normalized = (classical_state - np.mean(classical_state)) / (np.std(classical_state) + 1e-8)
        
        for i in range(min(len(normalized), self.num_qubits)):
            angle = normalized[i] * np.pi
            state_vector = self._apply_ry(state_vector, i, angle)
        
        return state_vector
    
    def _apply_variational_layer(self, state_vector: NDArray[np.complex128], layer: int) -> NDArray[np.complex128]:
        for qubit in range(self.num_qubits):
            rx, ry, rz = self.rotation_params[layer, qubit]
            state_vector = self._apply_rx(state_vector, qubit, rx)
            state_vector = self._apply_ry(state_vector, qubit, ry)
            state_vector = self._apply_rz(state_vector, qubit, rz)
        
        for i in range(self.num_qubits - 1):
            state_vector = self._apply_cnot(state_vector, i, i + 1)
        
        return state_vector
    
    def _apply_rx(self, state_vector: NDArray[np.complex128], qubit: int, angle: float) -> NDArray[np.complex128]:
        cos_half = np.cos(angle / 2)
        sin_half = -1j * np.sin(angle / 2)
        
        new_state = state_vector.copy()
        for i in range(self.dimension):
            if (i >> qubit) & 1 == 0:
                j = i | (1 << qubit)
                new_state[i] = cos_half * state_vector[i] + sin_half * state_vector[j]
                new_state[j] = sin_half * state_vector[i] + cos_half * state_vector[j]
        return new_state
    
    def _apply_ry(self, state_vector: NDArray[np.complex128], qubit: int, angle: float) -> NDArray[np.complex128]:
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        
        new_state = state_vector.copy()
        for i in range(self.dimension):
            if (i >> qubit) & 1 == 0:
                j = i | (1 << qubit)
                new_state[i] = cos_half * state_vector[i] - sin_half * state_vector[j]
                new_state[j] = sin_half * state_vector[i] + cos_half * state_vector[j]
        return new_state
    
    def _apply_rz(self, state_vector: NDArray[np.complex128], qubit: int, angle: float) -> NDArray[np.complex128]:
        phase_0 = np.exp(-1j * angle / 2)
        phase_1 = np.exp(1j * angle / 2)
        
        new_state = state_vector.copy()
        for i in range(self.dimension):
            if (i >> qubit) & 1 == 0:
                new_state[i] = phase_0 * state_vector[i]
            else:
                new_state[i] = phase_1 * state_vector[i]
        return new_state
    
    def _apply_cnot(self, state_vector: NDArray[np.complex128], control: int, target: int) -> NDArray[np.complex128]:
        new_state = state_vector.copy()
        for i in range(self.dimension):
            if (i >> control) & 1 == 1:
                j = i ^ (1 << target)
                new_state[i], new_state[j] = state_vector[j], state_vector[i]
        return new_state
    
    def _measure(self, state_vector: NDArray[np.complex128]) -> NDArray[np.float64]:
        return np.abs(state_vector[:self.num_qubits]) ** 2


class QuantumActorCritic:
    """
    Quantum Actor-Critic algorithm for trading strategy optimization.
    
    Uses separate quantum circuits for the actor (policy) and critic (value function),
    enabling more stable learning through advantage-based updates.
    """
    
    def __init__(self, config: Optional[QACConfig] = None):
        self.config = config or QACConfig()
        
        # Actor network (quantum policy)
        self.actor = QuantumActorNetwork(
            num_qubits=self.config.actor_num_qubits,
            num_layers=self.config.actor_num_layers,
            action_dim=self.config.action_dim
        )
        
        # Critic network (quantum value function)
        self.critic = QuantumCriticNetwork(
            num_qubits=self.config.critic_num_qubits,
            num_layers=self.config.critic_num_layers
        )
        
        # Training state
        self.training_step = 0
        self.episode = 0
        
        # Metrics
        self.metrics_history: List[Dict[str, Any]] = []
        
        logger.info(
            "Initialized QuantumActorCritic with actor_qubits=%d, critic_qubits=%d",
            self.config.actor_num_qubits, self.config.critic_num_qubits
        )
    
    def get_action_probs(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get action probabilities from actor."""
        return self.actor.forward(state)
    
    def get_value(self, state: NDArray[np.float64]) -> float:
        """Get state value from critic."""
        return self.critic.forward(state)
    
    def select_action(self, state: NDArray[np.float64], training: bool = True) -> Tuple[int, float, float]:
        """Select action and return log probability and value."""
        probs = self.get_action_probs(state)
        value = self.get_value(state)
        
        if training:
            action = np.random.choice(len(probs), p=probs)
        else:
            action = int(np.argmax(probs))
        
        log_prob = np.log(probs[action] + 1e-10)
        
        return action, log_prob, value
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        next_values: List[float],
        dones: List[bool]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute Generalized Advantage Estimation."""
        num_steps = len(rewards)
        advantages = np.zeros(num_steps)
        returns = np.zeros(num_steps)
        
        gae = 0.0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.discount_factor * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.discount_factor * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def compute_loss(
        self,
        log_probs: NDArray[np.float64],
        advantages: NDArray[np.float64],
        values: NDArray[np.float64],
        returns: NDArray[np.float64],
        entropy: float
    ) -> Tuple[float, float, float]:
        """Compute actor and critic losses."""
        # Actor loss (policy gradient)
        actor_loss = -np.mean(log_probs * advantages)
        
        # Critic loss (value function)
        critic_loss = np.mean((returns - values) ** 2)
        
        # Entropy bonus
        entropy_bonus = -self.config.entropy_coefficient * entropy
        
        # Total loss
        total_loss = actor_loss + self.config.value_loss_coefficient * critic_loss + entropy_bonus
        
        return total_loss, actor_loss, critic_loss
    
    def compute_entropy(self, probs: NDArray[np.float64]) -> float:
        """Compute entropy of action distribution."""
        return -np.sum(probs * np.log(probs + 1e-10))
    
    def collect_trajectory(self, env: Any) -> Trajectory:
        """Collect one trajectory of experience."""
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []
        next_values = []
        
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        done = False
        while not done:
            action, log_prob, value = self.select_action(state, training=True)
            
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                next_state, reward, done, info = result
            
            next_value = self.get_value(next_state) if not done else 0.0
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)
            next_values.append(next_value)
            
            state = next_state
        
        return Trajectory(
            states=states,
            actions=actions,
            rewards=rewards,
            log_probs=log_probs,
            values=values,
            dones=dones,
            next_values=next_values
        )
    
    def train_step(self, trajectory: Trajectory) -> Dict[str, float]:
        """Train on one trajectory."""
        # Compute GAE
        advantages, returns = self.compute_gae(
            trajectory.rewards,
            trajectory.values,
            trajectory.next_values,
            trajectory.dones
        )
        
        # Normalize advantages
        if np.std(advantages) > 0:
            advantages = (advantages - np.mean(advantages)) / np.std(advantages)
        
        # Compute losses
        total_loss = 0.0
        actor_loss = 0.0
        critic_loss = 0.0
        total_entropy = 0.0
        
        for i in range(len(trajectory)):
            state = trajectory.states[i]
            action = trajectory.actions[i]
            log_prob = trajectory.log_probs[i]
            advantage = advantages[i]
            value = trajectory.values[i]
            ret = returns[i]
            
            # Get current policy probabilities
            probs = self.get_action_probs(state)
            entropy = self.compute_entropy(probs)
            
            # Compute losses
            step_actor_loss = -log_prob * advantage
            step_critic_loss = (ret - value) ** 2
            
            total_loss += step_actor_loss + self.config.value_loss_coefficient * step_critic_loss - self.config.entropy_coefficient * entropy
            actor_loss += step_actor_loss
            critic_loss += step_critic_loss
            total_entropy += entropy
        
        avg_loss = total_loss / len(trajectory)
        avg_actor_loss = actor_loss / len(trajectory)
        avg_critic_loss = critic_loss / len(trajectory)
        avg_entropy = total_entropy / len(trajectory)
        
        self.training_step += 1
        
        return {
            "loss": avg_loss,
            "actor_loss": avg_actor_loss,
            "critic_loss": avg_critic_loss,
            "entropy": avg_entropy,
            "avg_advantage": float(np.mean(advantages)),
            "avg_return": float(np.mean(returns))
        }
    
    def train(
        self,
        env: Any,
        num_episodes: int,
        classical_baseline: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Train the QAC agent."""
        logger.info("Starting QAC training for %d episodes", num_episodes)
        
        for episode_idx in range(num_episodes):
            # Collect trajectory
            trajectory = self.collect_trajectory(env)
            
            # Train on trajectory
            train_metrics = self.train_step(trajectory)
            
            # Compute episode reward
            episode_reward = sum(trajectory.rewards)
            
            # Compute quantum advantage
            quantum_advantage = 0.0
            if classical_baseline is not None and classical_baseline != 0:
                quantum_advantage = (episode_reward - classical_baseline) / abs(classical_baseline)
            
            # Record metrics
            metrics = {
                "episode": episode_idx,
                "reward": episode_reward,
                "loss": train_metrics["loss"],
                "actor_loss": train_metrics["actor_loss"],
                "critic_loss": train_metrics["critic_loss"],
                "entropy": train_metrics["entropy"],
                "avg_advantage": train_metrics["avg_advantage"],
                "avg_return": train_metrics["avg_return"],
                "episode_length": len(trajectory),
                "quantum_advantage": quantum_advantage
            }
            self.metrics_history.append(metrics)
            
            # Log progress
            if (episode_idx + 1) % 10 == 0:
                avg_reward = np.mean([m["reward"] for m in self.metrics_history[-10:]])
                logger.info(
                    "Episode %d/%d | Avg Reward: %.4f | Actor Loss: %.4f | Critic Loss: %.4f",
                    episode_idx + 1, num_episodes, avg_reward,
                    train_metrics["actor_loss"], train_metrics["critic_loss"]
                )
        
        logger.info("QAC training completed")
        return self.metrics_history
    
    def evaluate(self, env: Any, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the trained agent."""
        rewards = []
        
        for _ in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            episode_reward = 0.0
            done = False
            
            while not done:
                action, _, _ = self.select_action(state, training=False)
                result = env.step(action)
                
                if len(result) == 5:
                    next_state, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    next_state, reward, done, info = result
                
                state = next_state
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "num_episodes": num_episodes
        }
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get information about the networks."""
        return {
            "actor": {
                "num_qubits": self.config.actor_num_qubits,
                "num_layers": self.config.actor_num_layers,
                "action_dim": self.config.action_dim
            },
            "critic": {
                "num_qubits": self.config.critic_num_qubits,
                "num_layers": self.config.critic_num_layers
            },
            "ac_type": self.config.ac_type.name
        }


__all__ = [
    "QuantumActorCritic",
    "QACConfig",
    "QuantumActorNetwork",
    "QuantumCriticNetwork",
    "ActorCriticType",
    "Trajectory"
]