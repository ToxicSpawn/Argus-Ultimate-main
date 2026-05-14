# pyright: reportMissingImports=false
"""
Quantum Policy Gradient (QPG) implementation for trading strategy optimization.

This module implements a hybrid quantum-classical policy gradient algorithm
that uses variational quantum circuits to parameterize the policy for
continuous and discrete action spaces in trading environments.
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


class PolicyType(Enum):
    """Types of policy networks."""
    DISCRETE = auto()   # Discrete action space
    CONTINUOUS = auto() # Continuous action space


class AdvantageEstimation(Enum):
    """Methods for advantage estimation."""
    MONTE_CARLO = auto()
    GAE = auto()  # Generalized Advantage Estimation
    TD = auto()   # Temporal Difference


@dataclass
class QPGConfig:
    """Configuration for Quantum Policy Gradient."""
    # Architecture
    state_dim: int = 8
    action_dim: int = 4
    policy_type: PolicyType = PolicyType.DISCRETE
    
    # Quantum parameters
    num_qubits: int = 4
    num_quantum_layers: int = 3
    encoding_type: str = "angle"
    
    # Training parameters
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    entropy_coefficient: float = 0.01  # Exploration bonus
    
    # Advantage estimation
    advantage_method: AdvantageEstimation = AdvantageEstimation.GAE
    gae_lambda: float = 0.95
    
    # Batch settings
    batch_size: int = 32
    num_episodes: int = 1000
    
    # Quantum advantage threshold
    min_quantum_advantage: float = 0.05


@dataclass
class Episode:
    """Stores trajectory for one episode."""
    states: List[NDArray[np.float64]]
    actions: List[int]
    rewards: List[float]
    log_probs: List[float]
    values: List[float]
    dones: List[bool]
    
    def __len__(self) -> int:
        return len(self.rewards)


class QuantumPolicyNetwork:
    """Quantum circuit for policy approximation."""
    
    def __init__(self, num_qubits: int, num_layers: int, action_dim: int):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.action_dim = action_dim
        self.dimension = 2 ** num_qubits
        
        # Initialize variational parameters
        self.rotation_params = np.random.uniform(0, 2*np.pi, (num_layers, num_qubits, 3))
        
        # Output projection (quantum measurements to action logits)
        self.output_weights = np.random.randn(num_qubits, action_dim) * 0.1
        self.output_bias = np.zeros(action_dim)
    
    def forward(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute action probabilities from state."""
        # Encode state into quantum state
        quantum_state = self._initialize_state()
        quantum_state = self._encode_state(quantum_state, state)
        
        # Apply variational layers
        for layer in range(self.num_layers):
            quantum_state = self._apply_variational_layer(quantum_state, layer)
        
        # Measure and project to action space
        measurements = self._measure(quantum_state)
        
        # Compute action logits
        logits = np.dot(measurements[:self.num_qubits], self.output_weights) + self.output_bias
        
        # Softmax for discrete actions
        return self._softmax(logits)
    
    def _initialize_state(self) -> NDArray[np.complex128]:
        """Initialize quantum state."""
        state = np.zeros(self.dimension, dtype=np.complex128)
        state[0] = 1.0 + 0j
        return state
    
    def _encode_state(self, state_vector: NDArray[np.complex128], classical_state: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Encode classical state using angle encoding."""
        normalized = (classical_state - np.mean(classical_state)) / (np.std(classical_state) + 1e-8)
        
        for i in range(min(len(normalized), self.num_qubits)):
            angle = normalized[i] * np.pi
            state_vector = self._apply_ry(state_vector, i, angle)
        
        return state_vector
    
    def _apply_variational_layer(self, state_vector: NDArray[np.complex128], layer: int) -> NDArray[np.complex128]:
        """Apply one variational layer."""
        # Single-qubit rotations
        for qubit in range(self.num_qubits):
            rx, ry, rz = self.rotation_params[layer, qubit]
            state_vector = self._apply_rx(state_vector, qubit, rx)
            state_vector = self._apply_ry(state_vector, qubit, ry)
            state_vector = self._apply_rz(state_vector, qubit, rz)
        
        # Entangling gates
        for i in range(self.num_qubits - 1):
            state_vector = self._apply_cnot(state_vector, i, i + 1)
        
        return state_vector
    
    def _apply_rx(self, state_vector: NDArray[np.complex128], qubit: int, angle: float) -> NDArray[np.complex128]:
        """Apply RX rotation gate."""
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
        """Apply RY rotation gate."""
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
        """Apply RZ rotation gate."""
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
        """Apply CNOT gate."""
        new_state = state_vector.copy()
        for i in range(self.dimension):
            if (i >> control) & 1 == 1:
                j = i ^ (1 << target)
                new_state[i], new_state[j] = state_vector[j], state_vector[i]
        return new_state
    
    def _measure(self, state_vector: NDArray[np.complex128]) -> NDArray[np.float64]:
        """Measure quantum state."""
        return np.abs(state_vector[:self.num_qubits]) ** 2
    
    def _softmax(self, logits: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute softmax."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def get_parameters(self) -> NDArray[np.float64]:
        """Get all trainable parameters."""
        return np.concatenate([
            self.rotation_params.flatten(),
            self.output_weights.flatten(),
            self.output_bias
        ])
    
    def set_parameters(self, params: NDArray[np.float64]) -> None:
        """Set trainable parameters."""
        rot_size = self.num_layers * self.num_qubits * 3
        output_w_size = self.num_qubits * self.action_dim
        
        idx = 0
        self.rotation_params = params[idx:idx + rot_size].reshape(
            (self.num_layers, self.num_qubits, 3)
        )
        idx += rot_size
        
        self.output_weights = params[idx:idx + output_w_size].reshape(
            (self.num_qubits, self.action_dim)
        )
        idx += output_w_size
        
        self.output_bias = params[idx:idx + self.action_dim]


class QuantumValueNetwork:
    """Quantum circuit for value function approximation."""
    
    def __init__(self, num_qubits: int, num_layers: int):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dimension = 2 ** num_qubits
        
        # Initialize variational parameters
        self.rotation_params = np.random.uniform(0, 2*np.pi, (num_layers, num_qubits, 3))
        
        # Output projection
        self.output_weight = np.random.randn(num_qubits, 1) * 0.1
        self.output_bias = np.zeros(1)
    
    def forward(self, state: NDArray[np.float64]) -> float:
        """Compute state value."""
        # Encode state
        quantum_state = self._initialize_state()
        quantum_state = self._encode_state(quantum_state, state)
        
        # Apply variational layers
        for layer in range(self.num_layers):
            quantum_state = self._apply_variational_layer(quantum_state, layer)
        
        # Measure
        measurements = self._measure(quantum_state)
        
        # Project to scalar value
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


class QuantumPolicyGradient:
    """
    Quantum Policy Gradient algorithm for trading strategy optimization.
    
    Uses variational quantum circuits to parameterize the policy,
    enabling quantum superposition of action preferences for enhanced exploration.
    """
    
    def __init__(self, config: Optional[QPGConfig] = None):
        self.config = config or QPGConfig()
        
        # Policy network (quantum)
        self.policy_network = QuantumPolicyNetwork(
            num_qubits=self.config.num_qubits,
            num_layers=self.config.num_quantum_layers,
            action_dim=self.config.action_dim
        )
        
        # Value network (optional, for advantage estimation)
        self.value_network = QuantumValueNetwork(
            num_qubits=self.config.num_qubits,
            num_layers=self.config.num_quantum_layers
        )
        
        # Training state
        self.training_step = 0
        self.episode = 0
        
        # Metrics
        self.metrics_history: List[Dict[str, Any]] = []
        
        logger.info(
            "Initialized QuantumPolicyGradient with %d qubits, %d layers",
            self.config.num_qubits, self.config.num_quantum_layers
        )
    
    def get_action_probs(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get action probabilities from policy."""
        return self.policy_network.forward(state)
    
    def select_action(self, state: NDArray[np.float64], training: bool = True) -> Tuple[int, float]:
        """Select action and return log probability."""
        probs = self.get_action_probs(state)
        
        if training:
            action = np.random.choice(len(probs), p=probs)
        else:
            action = int(np.argmax(probs))
        
        log_prob = np.log(probs[action] + 1e-10)
        
        return action, log_prob
    
    def compute_advantages(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[float]
    ) -> NDArray[np.float64]:
        """Compute advantages using GAE."""
        advantages = np.zeros(len(rewards))
        
        if self.config.advantage_method == AdvantageEstimation.GAE:
            # Generalized Advantage Estimation
            gae = 0.0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0.0
                else:
                    next_value = values[t + 1] * (1 - dones[t])
                
                delta = rewards[t] + self.config.discount_factor * next_value - values[t]
                gae = delta + self.config.discount_factor * self.config.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
        
        elif self.config.advantage_method == AdvantageEstimation.MONTE_CARLO:
            # Monte Carlo returns
            returns = np.zeros(len(rewards))
            running_return = 0.0
            for t in reversed(range(len(rewards))):
                running_return = rewards[t] + self.config.discount_factor * running_return * (1 - dones[t])
                returns[t] = running_return
            advantages = returns - np.array(values)
        
        else:  # TD
            # Temporal difference
            for t in range(len(rewards) - 1):
                next_value = values[t + 1] * (1 - dones[t])
                advantages[t] = rewards[t] + self.config.discount_factor * next_value - values[t]
        
        return advantages
    
    def compute_loss(
        self,
        log_probs: NDArray[np.float64],
        advantages: NDArray[np.float64],
        entropy: float
    ) -> float:
        """Compute policy gradient loss."""
        # Policy gradient loss
        policy_loss = -np.mean(log_probs * advantages)
        
        # Entropy bonus for exploration
        entropy_bonus = -self.config.entropy_coefficient * entropy
        
        return policy_loss + entropy_bonus
    
    def compute_entropy(self, probs: NDArray[np.float64]) -> float:
        """Compute entropy of action distribution."""
        return -np.sum(probs * np.log(probs + 1e-10))
    
    def train_step(self, episode: Episode) -> Dict[str, float]:
        """Train on one episode."""
        # Compute advantages
        advantages = self.compute_advantages(
            episode.rewards,
            episode.values,
            [float(d) for d in episode.dones]
        )
        
        # Normalize advantages
        if np.std(advantages) > 0:
            advantages = (advantages - np.mean(advantages)) / np.std(advantages)
        
        # Compute loss for each step
        total_loss = 0.0
        total_entropy = 0.0
        
        for i in range(len(episode)):
            state = episode.states[i]
            action = episode.actions[i]
            log_prob = episode.log_probs[i]
            advantage = advantages[i]
            
            # Get current policy probabilities
            probs = self.get_action_probs(state)
            entropy = self.compute_entropy(probs)
            
            # Compute loss
            loss = -log_prob * advantage - self.config.entropy_coefficient * entropy
            total_loss += loss
            total_entropy += entropy
        
        avg_loss = total_loss / len(episode)
        avg_entropy = total_entropy / len(episode)
        
        # Update training step
        self.training_step += 1
        
        return {
            "loss": avg_loss,
            "entropy": avg_entropy,
            "avg_advantage": float(np.mean(advantages))
        }
    
    def train(
        self,
        env: Any,
        num_episodes: int,
        classical_baseline: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Train the QPG agent."""
        logger.info("Starting QPG training for %d episodes", num_episodes)
        
        for episode_idx in range(num_episodes):
            # Collect episode
            episode = self.collect_episode(env)
            
            # Train on episode
            train_metrics = self.train_step(episode)
            
            # Compute episode reward
            episode_reward = sum(episode.rewards)
            
            # Compute quantum advantage
            quantum_advantage = 0.0
            if classical_baseline is not None and classical_baseline != 0:
                quantum_advantage = (episode_reward - classical_baseline) / abs(classical_baseline)
            
            # Record metrics
            metrics = {
                "episode": episode_idx,
                "reward": episode_reward,
                "loss": train_metrics["loss"],
                "entropy": train_metrics["entropy"],
                "avg_advantage": train_metrics["avg_advantage"],
                "episode_length": len(episode),
                "quantum_advantage": quantum_advantage
            }
            self.metrics_history.append(metrics)
            
            # Log progress
            if (episode_idx + 1) % 10 == 0:
                avg_reward = np.mean([m["reward"] for m in self.metrics_history[-10:]])
                logger.info(
                    "Episode %d/%d | Avg Reward: %.4f | Entropy: %.4f",
                    episode_idx + 1, num_episodes, avg_reward, train_metrics["entropy"]
                )
        
        logger.info("QPG training completed")
        return self.metrics_history
    
    def collect_episode(self, env: Any) -> Episode:
        """Collect one episode of experience."""
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []
        
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        done = False
        while not done:
            # Select action
            action, log_prob = self.select_action(state, training=True)
            
            # Get value estimate
            value = self.value_network.forward(state)
            
            # Take step
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                next_state, reward, done, info = result
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)
            
            state = next_state
        
        return Episode(
            states=states,
            actions=actions,
            rewards=rewards,
            log_probs=log_probs,
            values=values,
            dones=dones
        )
    
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
                action, _ = self.select_action(state, training=False)
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
    
    def get_policy_info(self) -> Dict[str, Any]:
        """Get information about the policy network."""
        return {
            "num_qubits": self.config.num_qubits,
            "num_layers": self.config.num_quantum_layers,
            "action_dim": self.config.action_dim,
            "num_parameters": len(self.policy_network.get_parameters())
        }


__all__ = [
    "QuantumPolicyGradient",
    "QPGConfig",
    "QuantumPolicyNetwork",
    "QuantumValueNetwork",
    "PolicyType",
    "AdvantageEstimation",
    "Episode"
]