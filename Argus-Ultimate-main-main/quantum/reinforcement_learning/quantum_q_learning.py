# pyright: reportMissingImports=false
"""
Quantum Q-Learning (QQL) implementation for trading strategy optimization.

This module implements a hybrid quantum-classical Q-Learning algorithm that uses
quantum circuits to represent and update Q-values. The quantum approach enables
superposition of action preferences and entanglement-based exploration strategies.
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class QuantumStateEncoding(Enum):
    """Methods for encoding classical states into quantum states."""
    ANGLE = auto()      # Angle encoding via rotation gates
    AMPLITUDE = auto()  # Amplitude encoding
    BASIS = auto()      # Basis state encoding


class ExplorationStrategy(Enum):
    """Exploration strategies for quantum Q-learning."""
    EPSILON_GREEDY = auto()
    QUANTUM_EXPLOITATION = auto()
    BOLTZMANN = auto()
    UCB = auto()


@dataclass
class QQLConfig:
    """Configuration for Quantum Q-Learning."""
    # Quantum parameters
    num_qubits: int = 4
    num_layers: int = 3
    encoding: QuantumStateEncoding = QuantumStateEncoding.ANGLE
    
    # Learning parameters
    learning_rate: float = 0.01
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Training parameters
    batch_size: int = 32
    replay_buffer_size: int = 10000
    target_update_freq: int = 100
    
    # Exploration
    exploration_strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY
    
    # Quantum advantage threshold
    min_quantum_advantage: float = 0.05  # 5% improvement required


@dataclass
class QuantumState:
    """Quantum representation of a state."""
    state_vector: NDArray[np.complex128]
    num_qubits: int
    encoding: QuantumStateEncoding
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Experience:
    """Experience tuple for replay buffer."""
    state: NDArray[np.float64]
    action: int
    reward: float
    next_state: NDArray[np.float64]
    done: bool
    timestamp: int = 0


@dataclass
class QQLMetrics:
    """Metrics for QQL training."""
    episode: int
    total_reward: float
    epsilon: float
    avg_q_value: float
    quantum_advantage: float = 0.0
    episodes_since_update: int = 0
    convergence_score: float = 0.0


class QuantumCircuitSimulator:
    """Simulates quantum circuit operations for QQL."""
    
    def __init__(self, num_qubits: int, num_layers: int):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dimension = 2 ** num_qubits
        self.parameters = np.random.uniform(0, 2 * np.pi, (num_layers, num_qubits))
        
    def encode_state(self, state: NDArray[np.float64]) -> QuantumState:
        """Encode classical state into quantum state using angle encoding."""
        # Normalize state
        normalized = (state - np.mean(state)) / (np.std(state) + 1e-8)
        
        # Create quantum state vector
        state_vector = np.zeros(self.dimension, dtype=np.complex128)
        state_vector[0] = 1.0 + 0j  # Start in |0...0⟩
        
        # Apply rotation gates based on state values
        for i, val in enumerate(normalized[:self.num_qubits]):
            angle = val * np.pi
            self._apply_rotation(state_vector, i, angle)
        
        # Normalize
        norm = np.linalg.norm(state_vector)
        if norm > 0:
            state_vector = state_vector / norm
            
        return QuantumState(
            state_vector=state_vector,
            num_qubits=self.num_qubits,
            encoding=QuantumStateEncoding.ANGLE,
            metadata={"original_state": state.tolist()}
        )
    
    def _apply_rotation(self, state_vector: NDArray[np.complex128], qubit: int, angle: float) -> None:
        """Apply rotation gate to a qubit."""
        # Simplified rotation simulation
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        
        for i in range(self.dimension):
            if (i >> qubit) & 1 == 0:
                j = i | (1 << qubit)
                temp_i = state_vector[i]
                temp_j = state_vector[j]
                state_vector[i] = cos_half * temp_i - sin_half * temp_j
                state_vector[j] = sin_half * temp_i + cos_half * temp_j
    
    def measure_q_values(self, quantum_state: QuantumState) -> NDArray[np.float64]:
        """Extract Q-values from quantum state through measurement."""
        probabilities = np.abs(quantum_state.state_vector) ** 2
        
        # Map probabilities to Q-values (simplified)
        q_values = np.zeros(self.dimension)
        for i in range(min(len(probabilities), self.dimension)):
            q_values[i] = probabilities[i] * self.dimension
        
        return q_values
    
    def update_parameters(self, gradients: NDArray[np.float64], learning_rate: float) -> None:
        """Update quantum circuit parameters using gradients."""
        self.parameters -= learning_rate * gradients


class ExperienceReplay:
    """Experience replay buffer for QQL."""
    
    def __init__(self, capacity: int):
        self.buffer: Deque[Experience] = deque(maxlen=capacity)
        self.position = 0
        
    def add(self, experience: Experience) -> None:
        """Add experience to buffer."""
        experience.timestamp = self.position
        self.position += 1
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch from buffer."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


class QuantumQLearning:
    """
    Quantum Q-Learning algorithm for trading strategy optimization.
    
    This implementation uses quantum circuits to represent Q-values and enables
    quantum superposition of action preferences for enhanced exploration.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[QQLConfig] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or QQLConfig()
        
        # Initialize quantum simulator
        self.quantum_sim = QuantumCircuitSimulator(
            num_qubits=self.config.num_qubits,
            num_layers=self.config.num_layers
        )
        
        # Classical Q-table backup (for fallback)
        self.q_table: Dict[str, NDArray[np.float64]] = {}
        
        # Experience replay
        self.replay_buffer = ExperienceReplay(self.config.replay_buffer_size)
        
        # Training state
        self.epsilon = self.config.epsilon_start
        self.training_step = 0
        self.episode = 0
        
        # Metrics
        self.metrics_history: List[QQLMetrics] = []
        self.best_avg_reward = float('-inf')
        
        logger.info(
            "Initialized QuantumQLearning with %d qubits, %d layers, state_dim=%d, action_dim=%d",
            self.config.num_qubits, self.config.num_layers, state_dim, action_dim
        )
    
    def _get_state_key(self, state: NDArray[np.float64]) -> str:
        """Convert state to hashable key."""
        return ",".join(f"{v:.4f}" for v in state[:self.state_dim])
    
    def get_q_values(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get Q-values for a state using quantum circuit."""
        try:
            # Encode state into quantum state
            quantum_state = self.quantum_sim.encode_state(state)
            
            # Measure to get Q-values
            raw_q_values = self.quantum_sim.measure_q_values(quantum_state)
            
            # Extract action-dimension Q-values
            q_values = raw_q_values[:self.action_dim]
            
            # Add classical bias from Q-table
            state_key = self._get_state_key(state)
            if state_key in self.q_table:
                q_values = 0.7 * q_values + 0.3 * self.q_table[state_key]
            
            return q_values
            
        except Exception as e:
            logger.warning("Quantum circuit failed, using classical fallback: %s", e)
            return self._classical_q_values(state)
    
    def _classical_q_values(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Classical fallback for Q-value computation."""
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        return self.q_table[state_key]
    
    def select_action(self, state: NDArray[np.float64], training: bool = True) -> int:
        """Select action using epsilon-greedy strategy."""
        if training and random.random() < self.epsilon:
            # Explore
            return random.randint(0, self.action_dim - 1)
        
        # Exploit - use quantum Q-values
        q_values = self.get_q_values(state)
        return int(np.argmax(q_values))
    
    def compute_td_error(
        self,
        state: NDArray[np.float64],
        action: int,
        reward: float,
        next_state: NDArray[np.float64],
        done: bool
    ) -> float:
        """Compute TD error for Q-learning update."""
        current_q = self.get_q_values(state)[action]
        
        if done:
            target_q = reward
        else:
            next_q_values = self.get_q_values(next_state)
            target_q = reward + self.config.discount_factor * np.max(next_q_values)
        
        return target_q - current_q
    
    def update(self, experience: Experience) -> float:
        """Update Q-values based on experience."""
        td_error = self.compute_td_error(
            experience.state,
            experience.action,
            experience.reward,
            experience.next_state,
            experience.done
        )
        
        # Update classical Q-table
        state_key = self._get_state_key(experience.state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        
        self.q_table[state_key][experience.action] += self.config.learning_rate * td_error
        
        # Update quantum circuit parameters
        gradients = self._compute_quantum_gradients(experience, td_error)
        self.quantum_sim.update_parameters(gradients, self.config.learning_rate)
        
        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
        
        self.training_step += 1
        
        return abs(td_error)
    
    def _compute_quantum_gradients(
        self,
        experience: Experience,
        td_error: float
    ) -> NDArray[np.float64]:
        """Compute gradients for quantum circuit parameters."""
        # Simplified gradient computation (parameter shift rule approximation)
        gradients = np.zeros_like(self.quantum_sim.parameters)
        
        for layer in range(self.config.num_layers):
            for qubit in range(self.config.num_qubits):
                # Parameter shift rule: gradient = (Q(θ+π/2) - Q(θ-π/2)) / 2
                shift = np.pi / 2
                
                # Positive shift
                original = self.quantum_sim.parameters[layer, qubit]
                self.quantum_sim.parameters[layer, qubit] = original + shift
                q_pos = self.get_q_values(experience.state)[experience.action]
                
                # Negative shift
                self.quantum_sim.parameters[layer, qubit] = original - shift
                q_neg = self.get_q_values(experience.state)[experience.action]
                
                # Restore original
                self.quantum_sim.parameters[layer, qubit] = original
                
                # Compute gradient
                gradients[layer, qubit] = td_error * (q_pos - q_neg) / 2
        
        return gradients
    
    def train_step(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Perform one training step with batch of experiences."""
        batch_size = batch_size or self.config.batch_size
        
        if len(self.replay_buffer) < batch_size:
            return {"loss": 0.0, "td_error": 0.0}
        
        batch = self.replay_buffer.sample(batch_size)
        
        total_td_error = 0.0
        total_loss = 0.0
        
        for exp in batch:
            td_error = self.update(exp)
            total_td_error += abs(td_error)
            total_loss += td_error ** 2
        
        avg_td_error = total_td_error / len(batch)
        avg_loss = total_loss / len(batch)
        
        return {
            "loss": avg_loss,
            "td_error": avg_td_error,
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer)
        }
    
    def compute_quantum_advantage(self, classical_performance: float) -> float:
        """Compute quantum advantage over classical baseline."""
        if not self.metrics_history:
            return 0.0
        
        quantum_performance = np.mean([m.total_reward for m in self.metrics_history[-10:]])
        
        if classical_performance == 0:
            return 0.0
        
        advantage = (quantum_performance - classical_performance) / abs(classical_performance)
        return advantage
    
    def record_metrics(
        self,
        episode_reward: float,
        classical_baseline: Optional[float] = None
    ) -> QQLMetrics:
        """Record training metrics for current episode."""
        # Get average Q-values
        if self.q_table:
            all_q_values = np.concatenate(list(self.q_table.values()))
            avg_q_value = np.mean(all_q_values)
        else:
            avg_q_value = 0.0
        
        # Compute quantum advantage
        quantum_advantage = 0.0
        if classical_baseline is not None:
            quantum_advantage = self.compute_quantum_advantage(classical_baseline)
        
        # Compute convergence score (based on reward stability)
        convergence_score = 0.0
        if len(self.metrics_history) >= 10:
            recent_rewards = [m.total_reward for m in self.metrics_history[-10:]]
            reward_std = np.std(recent_rewards)
            reward_mean = np.mean(recent_rewards)
            if reward_mean != 0:
                convergence_score = 1.0 - min(1.0, abs(reward_std / reward_mean))
        
        metrics = QQLMetrics(
            episode=self.episode,
            total_reward=episode_reward,
            epsilon=self.epsilon,
            avg_q_value=avg_q_value,
            quantum_advantage=quantum_advantage,
            episodes_since_update=self.training_step,
            convergence_score=convergence_score
        )
        
        self.metrics_history.append(metrics)
        
        # Update best performance
        if episode_reward > self.best_avg_reward:
            self.best_avg_reward = episode_reward
        
        self.episode += 1
        
        return metrics
    
    def train(
        self,
        env: Any,
        num_episodes: int,
        classical_baseline: Optional[float] = None
    ) -> List[QQLMetrics]:
        """
        Train the QQL agent on an environment.
        
        Args:
            env: Environment with reset() and step() methods
            num_episodes: Number of training episodes
            classical_baseline: Classical algorithm performance for comparison
            
        Returns:
            List of metrics for each episode
        """
        logger.info("Starting QQL training for %d episodes", num_episodes)
        
        for episode in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            episode_reward = 0.0
            done = False
            step = 0
            
            while not done:
                # Select action
                action = self.select_action(state, training=True)
                
                # Take step
                result = env.step(action)
                if len(result) == 5:
                    next_state, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    next_state, reward, done, info = result
                
                # Store experience
                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                self.replay_buffer.add(experience)
                
                # Train
                train_metrics = self.train_step()
                
                # Update state
                state = next_state
                episode_reward += reward
                step += 1
            
            # Record metrics
            metrics = self.record_metrics(episode_reward, classical_baseline)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                logger.info(
                    "Episode %d/%d | Reward: %.4f | Epsilon: %.3f | Q-Value: %.4f | Advantage: %.2f%%",
                    episode + 1, num_episodes, episode_reward, self.epsilon,
                    metrics.avg_q_value, metrics.quantum_advantage * 100
                )
        
        logger.info("QQL training completed. Best reward: %.4f", self.best_avg_reward)
        
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
                action = self.select_action(state, training=False)
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
    
    def get_policy(self) -> Dict[str, int]:
        """Extract deterministic policy from learned Q-values."""
        policy = {}
        
        for state_key, q_values in self.q_table.items():
            policy[state_key] = int(np.argmax(q_values))
        
        return policy
    
    def get_quantum_circuit_info(self) -> Dict[str, Any]:
        """Get information about the quantum circuit."""
        return {
            "num_qubits": self.quantum_sim.num_qubits,
            "num_layers": self.quantum_sim.num_layers,
            "dimension": self.quantum_sim.dimension,
            "num_parameters": self.quantum_sim.parameters.size,
            "parameters": self.quantum_sim.parameters.tolist()
        }


__all__ = [
    "QuantumQLearning",
    "QQLConfig",
    "QuantumState",
    "QuantumStateEncoding",
    "ExplorationStrategy",
    "QQLMetrics",
    "Experience"
]