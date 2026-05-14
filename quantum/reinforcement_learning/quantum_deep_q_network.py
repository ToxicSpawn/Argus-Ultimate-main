# pyright: reportMissingImports=false
"""
Quantum Deep Q-Networks (QDQN) implementation for trading strategy optimization.

This module implements a hybrid quantum-classical Deep Q-Network that uses
variational quantum circuits as part of the neural network architecture for
approximating Q-values in complex trading environments.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class QuantumLayerType(Enum):
    """Types of quantum layers in the network."""
    ENCODING = auto()    # State encoding layer
    VARIATIONAL = auto() # Trainable variational layer
    ENTANGLING = auto()  # Entangling operations
    MEASUREMENT = auto() # Measurement/output layer


@dataclass
class QDQNConfig:
    """Configuration for Quantum Deep Q-Networks."""
    # Architecture
    state_dim: int = 8
    action_dim: int = 4
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    
    # Quantum parameters
    num_qubits: int = 6
    num_quantum_layers: int = 4
    encoding_type: str = "angle"  # angle, amplitude, basis
    
    # Training parameters
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    batch_size: int = 64
    buffer_size: int = 100000
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Target network
    target_update_freq: int = 1000
    tau: float = 0.005  # Soft update parameter
    
    # Quantum advantage
    min_quantum_advantage: float = 0.05


class VariationalQuantumCircuit:
    """Variational quantum circuit for QDQN."""
    
    def __init__(self, num_qubits: int, num_layers: int):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dimension = 2 ** num_qubits
        
        # Initialize parameters
        self.rotation_params = np.random.uniform(0, 2*np.pi, (num_layers, num_qubits, 3))
        self.entangling_params = np.random.uniform(0, 2*np.pi, (num_layers, num_qubits - 1))
        
    def forward(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Forward pass through variational quantum circuit."""
        # Initialize state vector
        state_vector = self._initialize_state()
        
        # Encode classical state
        state_vector = self._encode_state(state_vector, state)
        
        # Apply variational layers
        for layer in range(self.num_layers):
            state_vector = self._apply_rotation_layer(state_vector, layer)
            state_vector = self._apply_entangling_layer(state_vector, layer)
        
        # Measure and return
        return self._measure(state_vector)
    
    def _initialize_state(self) -> NDArray[np.complex128]:
        """Initialize quantum state in |0...0⟩."""
        state = np.zeros(self.dimension, dtype=np.complex128)
        state[0] = 1.0 + 0j
        return state
    
    def _encode_state(self, state_vector: NDArray[np.complex128], classical_state: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Encode classical state into quantum state using angle encoding."""
        # Normalize classical state
        normalized = (classical_state - np.mean(classical_state)) / (np.std(classical_state) + 1e-8)
        
        # Apply RX and RY rotations based on state values
        for i in range(min(len(normalized), self.num_qubits)):
            angle_x = normalized[i] * np.pi / 2
            angle_y = normalized[i] * np.pi / 2
            
            state_vector = self._apply_rx(state_vector, i, angle_x)
            state_vector = self._apply_ry(state_vector, i, angle_y)
        
        return state_vector
    
    def _apply_rotation_layer(self, state_vector: NDArray[np.complex128], layer: int) -> NDArray[np.complex128]:
        """Apply parameterized rotation gates."""
        for qubit in range(self.num_qubits):
            rx, ry, rz = self.rotation_params[layer, qubit]
            state_vector = self._apply_rx(state_vector, qubit, rx)
            state_vector = self._apply_ry(state_vector, qubit, ry)
            state_vector = self._apply_rz(state_vector, qubit, rz)
        return state_vector
    
    def _apply_entangling_layer(self, state_vector: NDArray[np.complex128], layer: int) -> NDArray[np.complex128]:
        """Apply entangling CNOT gates."""
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
        """Measure quantum state and return probabilities."""
        probabilities = np.abs(state_vector) ** 2
        return probabilities
    
    def get_parameters(self) -> NDArray[np.float64]:
        """Get all trainable parameters."""
        return np.concatenate([self.rotation_params.flatten(), self.entangling_params.flatten()])
    
    def set_parameters(self, params: NDArray[np.float64]) -> None:
        """Set trainable parameters."""
        rot_size = self.num_layers * self.num_qubits * 3
        ent_size = self.num_layers * (self.num_qubits - 1)
        
        self.rotation_params = params[:rot_size].reshape((self.num_layers, self.num_qubits, 3))
        self.entangling_params = params[rot_size:rot_size + ent_size].reshape((self.num_layers, self.num_qubits - 1))


class QuantumNeuralNetwork:
    """Hybrid quantum-classical neural network."""
    
    def __init__(self, config: QDQNConfig):
        self.config = config
        
        # Classical layers
        self.classical_weights: List[NDArray[np.float64]] = []
        self.classical_biases: List[NDArray[np.float64]] = []
        
        # Initialize classical layers
        prev_dim = config.state_dim
        for hidden_dim in config.hidden_dims:
            self.classical_weights.append(
                np.random.randn(prev_dim, hidden_dim) * np.sqrt(2.0 / prev_dim)
            )
            self.classical_biases.append(np.zeros(hidden_dim))
            prev_dim = hidden_dim
        
        # Quantum circuit
        self.quantum_circuit = VariationalQuantumCircuit(
            num_qubits=config.num_qubits,
            num_layers=config.num_quantum_layers
        )
        
        # Output layer (after quantum processing)
        self.output_weight = np.random.randn(config.num_qubits, config.action_dim) * np.sqrt(2.0 / config.num_qubits)
        self.output_bias = np.zeros(config.action_dim)
    
    def forward(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Forward pass through hybrid network."""
        # Classical processing
        x = state
        for i, (weight, bias) in enumerate(zip(self.classical_weights, self.classical_biases)):
            x = np.tanh(np.dot(x, weight) + bias)
        
        # Quantum processing
        quantum_output = self.quantum_circuit.forward(x[:self.config.num_qubits])
        
        # If quantum output is too large (probabilities), reduce dimension
        if len(quantum_output) > self.config.num_qubits:
            quantum_output = quantum_output[:self.config.num_qubits]
        
        # Output layer
        q_values = np.dot(quantum_output, self.output_weight) + self.output_bias
        
        return q_values
    
    def get_parameters(self) -> Dict[str, NDArray[np.float64]]:
        """Get all parameters."""
        params = {}
        for i, (w, b) in enumerate(zip(self.classical_weights, self.classical_biases)):
            params[f'classical_weight_{i}'] = w
            params[f'classical_bias_{i}'] = b
        params['quantum'] = self.quantum_circuit.get_parameters()
        params['output_weight'] = self.output_weight
        params['output_bias'] = self.output_bias
        return params


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer."""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: List[Tuple[Any, ...]] = []
        self.priorities = np.zeros(capacity)
        self.position = 0
    
    def add(self, experience: Tuple[Any, ...], td_error: float = 1.0) -> None:
        """Add experience with priority."""
        priority = (abs(td_error) + 1e-6) ** self.alpha
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Tuple[Any, ...]], NDArray[np.float64]]:
        """Sample batch with priorities."""
        n = len(self.buffer)
        probabilities = self.priorities[:n] ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(n, batch_size, p=probabilities)
        
        # Importance sampling weights
        weights = (n * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return [self.buffer[i] for i in indices], weights
    
    def __len__(self) -> int:
        return len(self.buffer)


class QuantumDeepQNetwork:
    """
    Quantum Deep Q-Network for trading strategy optimization.
    
    Combines classical neural network layers with variational quantum circuits
    for enhanced Q-value approximation in complex trading environments.
    """
    
    def __init__(self, config: Optional[QDQNConfig] = None):
        self.config = config or QDQNConfig()
        
        # Main network
        self.q_network = QuantumNeuralNetwork(self.config)
        
        # Target network
        self.target_network = QuantumNeuralNetwork(self.config)
        self._update_target_network(tau=1.0)
        
        # Experience replay
        self.replay_buffer = PrioritizedReplayBuffer(self.config.buffer_size)
        
        # Training state
        self.training_step = 0
        self.epsilon = self.config.epsilon_start
        self.episode = 0
        
        # Metrics
        self.metrics_history: List[Dict[str, Any]] = []
        
        logger.info(
            "Initialized QuantumDeepQNetwork with %d qubits, %d quantum layers",
            self.config.num_qubits, self.config.num_quantum_layers
        )
    
    def _update_target_network(self, tau: float = None) -> None:
        """Soft update target network."""
        tau = tau or self.config.tau
        
        main_params = self.q_network.get_parameters()
        target_params = self.target_network.get_parameters()
        
        for key in main_params:
            if key in target_params:
                target_params[key] = tau * main_params[key] + (1 - tau) * target_params[key]
    
    def select_action(self, state: NDArray[np.float64], training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.config.action_dim - 1)
        
        q_values = self.q_network.forward(state)
        return int(np.argmax(q_values))
    
    def compute_td_error(
        self,
        state: NDArray[np.float64],
        action: int,
        reward: float,
        next_state: NDArray[np.float64],
        done: bool
    ) -> float:
        """Compute TD error for experience."""
        current_q = self.q_network.forward(state)[action]
        
        if done:
            target_q = reward
        else:
            with np.no_grad():
                next_q = self.target_network.forward(next_state)
            target_q = reward + self.config.discount_factor * np.max(next_q)
        
        return target_q - current_q
    
    def train_step(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Perform one training step."""
        batch_size = batch_size or self.config.batch_size
        
        if len(self.replay_buffer) < batch_size:
            return {"loss": 0.0, "td_error": 0.0, "epsilon": self.epsilon}
        
        # Sample batch
        batch, weights = self.replay_buffer.sample(batch_size)
        
        total_loss = 0.0
        total_td_error = 0.0
        
        for experience, weight in zip(batch, weights):
            state, action, reward, next_state, done = experience
            
            # Compute TD error
            td_error = self.compute_td_error(state, action, reward, next_state, done)
            
            # Update priority
            self.replay_buffer.add(experience, td_error)
            
            # Compute loss (simplified)
            loss = (td_error ** 2) * weight
            total_loss += loss
            total_td_error += abs(td_error)
        
        # Update target network periodically
        if self.training_step % self.config.target_update_freq == 0:
            self._update_target_network(tau=1.0)
        else:
            self._update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        
        self.training_step += 1
        
        return {
            "loss": total_loss / batch_size,
            "td_error": total_td_error / batch_size,
            "epsilon": self.epsilon
        }
    
    def train(
        self,
        env: Any,
        num_episodes: int,
        classical_baseline: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Train the QDQN agent."""
        logger.info("Starting QDQN training for %d episodes", num_episodes)
        
        for episode in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            episode_reward = 0.0
            episode_loss = 0.0
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
                experience = (state, action, reward, next_state, done)
                self.replay_buffer.add(experience)
                
                # Train
                train_metrics = self.train_step()
                episode_loss += train_metrics["loss"]
                
                # Update state
                state = next_state
                episode_reward += reward
                step += 1
            
            # Record metrics
            metrics = {
                "episode": episode,
                "reward": episode_reward,
                "loss": episode_loss / max(1, step),
                "epsilon": self.epsilon,
                "steps": step
            }
            self.metrics_history.append(metrics)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean([m["reward"] for m in self.metrics_history[-10:]])
                logger.info(
                    "Episode %d/%d | Avg Reward: %.4f | Epsilon: %.3f",
                    episode + 1, num_episodes, avg_reward, self.epsilon
                )
        
        logger.info("QDQN training completed")
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
    
    def get_q_values(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get Q-values for a state."""
        return self.q_network.forward(state)
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get information about the network architecture."""
        return {
            "state_dim": self.config.state_dim,
            "action_dim": self.config.action_dim,
            "hidden_dims": self.config.hidden_dims,
            "num_qubits": self.config.num_qubits,
            "num_quantum_layers": self.config.num_quantum_layers,
            "total_parameters": sum(
                p.size for p in self.q_network.get_parameters().values()
            )
        }


__all__ = [
    "QuantumDeepQNetwork",
    "QDQNConfig",
    "QuantumNeuralNetwork",
    "VariationalQuantumCircuit",
    "PrioritizedReplayBuffer"
]