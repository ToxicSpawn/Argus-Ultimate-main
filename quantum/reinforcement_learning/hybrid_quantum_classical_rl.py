# pyright: reportMissingImports=false
"""
Hybrid Quantum-Classical Reinforcement Learning implementation.

This module provides a unified framework for combining quantum and classical
reinforcement learning components, enabling flexible architectures where
quantum circuits handle specific aspects of learning while classical networks
provide robustness and scalability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class HybridArchitecture(Enum):
    """Architecture types for hybrid quantum-classical RL."""
    QUANTUM_POLICY_CLASSICAL_VALUE = auto()  # Quantum actor, classical critic
    CLASSICAL_POLICY_QUANTUM_VALUE = auto()  # Classical actor, quantum critic
    QUANTUM_FEATURE_EXTRACTION = auto()       # Quantum features → classical RL
    QUANTUM_ADVANTAGE_ESTIMATION = auto()     # Classical policy, quantum advantage
    ENSEMBLE = auto()                         # Ensemble of quantum and classical


@dataclass
class HybridRLConfig:
    """Configuration for Hybrid Quantum-Classical RL."""
    # Architecture
    architecture: HybridArchitecture = HybridArchitecture.QUANTUM_POLICY_CLASSICAL_VALUE
    state_dim: int = 8
    action_dim: int = 4
    
    # Quantum parameters
    quantum_num_qubits: int = 4
    quantum_num_layers: int = 3
    quantum_weight: float = 0.5  # Weight of quantum component in ensemble
    
    # Classical parameters
    classical_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    
    # Training parameters
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    entropy_coefficient: float = 0.01
    
    # Hybrid-specific
    quantum_fallback_threshold: float = 0.1  # Switch to classical if quantum fails
    adaptive_weighting: bool = True  # Dynamically adjust quantum/classical weights
    min_quantum_advantage: float = 0.05


class ClassicalPolicyNetwork:
    """Classical neural network for policy approximation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        self.layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append({
                'weight': np.random.randn(prev_dim, hidden_dim) * np.sqrt(2.0 / prev_dim),
                'bias': np.zeros(hidden_dim)
            })
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = {
            'weight': np.random.randn(prev_dim, action_dim) * np.sqrt(2.0 / prev_dim),
            'bias': np.zeros(action_dim)
        }
    
    def forward(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Forward pass through classical network."""
        x = state
        
        for layer in self.layers:
            x = np.tanh(np.dot(x, layer['weight']) + layer['bias'])
        
        logits = np.dot(x, self.output_layer['weight']) + self.output_layer['bias']
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)


class ClassicalValueNetwork:
    """Classical neural network for value approximation."""
    
    def __init__(self, state_dim: int, hidden_dims: List[int]):
        self.layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append({
                'weight': np.random.randn(prev_dim, hidden_dim) * np.sqrt(2.0 / prev_dim),
                'bias': np.zeros(hidden_dim)
            })
            prev_dim = hidden_dim
        
        # Output layer (scalar value)
        self.output_layer = {
            'weight': np.random.randn(prev_dim, 1) * np.sqrt(2.0 / prev_dim),
            'bias': np.zeros(1)
        }
    
    def forward(self, state: NDArray[np.float64]) -> float:
        """Forward pass to get state value."""
        x = state
        
        for layer in self.layers:
            x = np.tanh(np.dot(x, layer['weight']) + layer['bias'])
        
        value = np.dot(x, self.output_layer['weight']) + self.output_layer['bias']
        return float(value[0])


class QuantumFeatureExtractor:
    """Quantum circuit for feature extraction."""
    
    def __init__(self, num_qubits: int, num_layers: int, output_dim: int):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dimension = 2 ** num_qubits
        
        # Variational parameters
        self.rotation_params = np.random.uniform(0, 2*np.pi, (num_layers, num_qubits, 3))
        
        # Measurement projection
        self.projection_weights = np.random.randn(num_qubits, output_dim) * 0.1
    
    def extract_features(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Extract quantum features from state."""
        # Encode state
        quantum_state = self._initialize_state()
        quantum_state = self._encode_state(quantum_state, state)
        
        # Apply variational layers
        for layer in range(self.num_layers):
            quantum_state = self._apply_variational_layer(quantum_state, layer)
        
        # Measure and project
        measurements = self._measure(quantum_state)
        features = np.dot(measurements[:self.num_qubits], self.projection_weights)
        
        return features
    
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


class QuantumAdvantageEstimator:
    """Quantum circuit for advantage estimation."""
    
    def __init__(self, num_qubits: int, num_layers: int):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dimension = 2 ** num_qubits
        
        # Variational parameters
        self.rotation_params = np.random.uniform(0, 2*np.pi, (num_layers, num_qubits, 3))
        
        # Output projection
        self.output_weight = np.random.randn(num_qubits, 1) * 0.01
    
    def estimate_advantage(self, state: NDArray[np.float64], value: float) -> float:
        """Estimate advantage using quantum circuit."""
        # Encode state and value
        quantum_state = self._initialize_state()
        quantum_state = self._encode_state(quantum_state, state, value)
        
        # Apply variational layers
        for layer in range(self.num_layers):
            quantum_state = self._apply_variational_layer(quantum_state, layer)
        
        # Measure and project
        measurements = self._measure(quantum_state)
        advantage = np.dot(measurements[:self.num_qubits], self.output_weight.flatten())
        
        return float(advantage)
    
    def _initialize_state(self) -> NDArray[np.complex128]:
        state = np.zeros(self.dimension, dtype=np.complex128)
        state[0] = 1.0 + 0j
        return state
    
    def _encode_state(self, state_vector: NDArray[np.complex128], classical_state: NDArray[np.float64], value: float) -> NDArray[np.complex128]:
        normalized = (classical_state - np.mean(classical_state)) / (np.std(classical_state) + 1e-8)
        
        # Encode state
        for i in range(min(len(normalized), self.num_qubits - 1)):
            angle = normalized[i] * np.pi
            state_vector = self._apply_ry(state_vector, i, angle)
        
        # Encode value in last qubit
        value_angle = np.clip(value, -1, 1) * np.pi / 2
        state_vector = self._apply_ry(state_vector, self.num_qubits - 1, value_angle)
        
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


class HybridQuantumClassicalRL:
    """
    Hybrid Quantum-Classical Reinforcement Learning system.
    
    Combines quantum and classical RL components in various architectures
    to leverage the strengths of both approaches.
    """
    
    def __init__(self, config: Optional[HybridRLConfig] = None):
        self.config = config or HybridRLConfig()
        
        # Initialize components based on architecture
        self.quantum_weight = self.config.quantum_weight
        self.classical_weight = 1.0 - self.config.quantum_weight
        
        # Classical components
        self.classical_policy = ClassicalPolicyNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dims=self.config.classical_hidden_dims
        )
        
        self.classical_value = ClassicalValueNetwork(
            state_dim=self.config.state_dim,
            hidden_dims=self.config.classical_hidden_dims
        )
        
        # Quantum components (initialized based on architecture)
        self.quantum_feature_extractor = QuantumFeatureExtractor(
            num_qubits=self.config.quantum_num_qubits,
            num_layers=self.config.quantum_num_layers,
            output_dim=self.config.state_dim
        )
        
        self.quantum_advantage_estimator = QuantumAdvantageEstimator(
            num_qubits=self.config.quantum_num_qubits,
            num_layers=self.config.quantum_num_layers
        )
        
        # Training state
        self.training_step = 0
        self.episode = 0
        
        # Performance tracking
        self.quantum_performance_history: List[float] = []
        self.classical_performance_history: List[float] = []
        
        # Metrics
        self.metrics_history: List[Dict[str, Any]] = []
        
        logger.info(
            "Initialized HybridQuantumClassicalRL with architecture=%s, quantum_weight=%.2f",
            self.config.architecture.name, self.config.quantum_weight
        )
    
    def get_action_probs(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get action probabilities using hybrid approach."""
        if self.config.architecture == HybridArchitecture.QUANTUM_POLICY_CLASSICAL_VALUE:
            # Quantum policy (simplified: use quantum features with classical policy)
            quantum_features = self.quantum_feature_extractor.extract_features(state)
            combined_state = state + self.quantum_weight * quantum_features[:len(state)]
            return self.classical_policy.forward(combined_state)
        
        elif self.config.architecture == HybridArchitecture.CLASSICAL_POLICY_QUANTUM_VALUE:
            # Classical policy
            return self.classical_policy.forward(state)
        
        elif self.config.architecture == HybridArchitecture.QUANTUM_FEATURE_EXTRACTION:
            # Quantum feature extraction only
            quantum_features = self.quantum_feature_extractor.extract_features(state)
            return self.classical_policy.forward(quantum_features[:self.config.state_dim])
        
        elif self.config.architecture == HybridArchitecture.ENSEMBLE:
            # Ensemble of quantum and classical
            classical_probs = self.classical_policy.forward(state)
            quantum_features = self.quantum_feature_extractor.extract_features(state)
            quantum_probs = self.classical_policy.forward(quantum_features[:self.config.state_dim])
            
            # Weighted combination
            combined_probs = (
                self.quantum_weight * quantum_probs + 
                self.classical_weight * classical_probs
            )
            return combined_probs / np.sum(combined_probs)
        
        else:  # QUANTUM_ADVANTAGE_ESTIMATION
            return self.classical_policy.forward(state)
    
    def get_value(self, state: NDArray[np.float64]) -> float:
        """Get state value using hybrid approach."""
        if self.config.architecture == HybridArchitecture.QUANTUM_POLICY_CLASSICAL_VALUE:
            return self.classical_value.forward(state)
        
        elif self.config.architecture == HybridArchitecture.CLASSICAL_POLICY_QUANTUM_VALUE:
            # Classical value + quantum advantage
            classical_value = self.classical_value.forward(state)
            quantum_advantage = self.quantum_advantage_estimator.estimate_advantage(state, classical_value)
            return classical_value + self.quantum_weight * quantum_advantage
        
        elif self.config.architecture == HybridArchitecture.QUANTUM_ADVANTAGE_ESTIMATION:
            classical_value = self.classical_value.forward(state)
            quantum_advantage = self.quantum_advantage_estimator.estimate_advantage(state, classical_value)
            return classical_value + quantum_advantage
        
        else:
            return self.classical_value.forward(state)
    
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
    
    def update_weights(self, quantum_performance: float, classical_performance: float) -> None:
        """Dynamically update quantum/classical weights based on performance."""
        if not self.config.adaptive_weighting:
            return
        
        self.quantum_performance_history.append(quantum_performance)
        self.classical_performance_history.append(classical_performance)
        
        # Use recent performance to adjust weights
        if len(self.quantum_performance_history) >= 10:
            recent_quantum = np.mean(self.quantum_performance_history[-10:])
            recent_classical = np.mean(self.classical_performance_history[-10:])
            
            total = recent_quantum + recent_classical
            if total > 0:
                self.quantum_weight = recent_quantum / total
                self.classical_weight = recent_classical / total
                
                logger.debug(
                    "Updated weights: quantum=%.3f, classical=%.3f",
                    self.quantum_weight, self.classical_weight
                )
    
    def train_step(
        self,
        states: List[NDArray[np.float64]],
        actions: List[int],
        rewards: List[float],
        next_states: List[NDArray[np.float64]],
        dones: List[bool]
    ) -> Dict[str, float]:
        """Perform one training step."""
        # Compute advantages using GAE
        advantages = self._compute_gae(states, rewards, next_states, dones)
        
        # Normalize advantages
        if np.std(advantages) > 0:
            advantages = (advantages - np.mean(advantages)) / np.std(advantages)
        
        total_loss = 0.0
        
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            advantage = advantages[i]
            
            # Get current policy probabilities
            probs = self.get_action_probs(state)
            log_prob = np.log(probs[action] + 1e-10)
            
            # Policy gradient loss
            loss = -log_prob * advantage
            total_loss += loss
        
        avg_loss = total_loss / len(states)
        
        self.training_step += 1
        
        return {
            "loss": avg_loss,
            "avg_advantage": float(np.mean(advantages)),
            "quantum_weight": self.quantum_weight,
            "classical_weight": self.classical_weight
        }
    
    def _compute_gae(
        self,
        states: List[NDArray[np.float64]],
        rewards: List[float],
        next_states: List[NDArray[np.float64]],
        dones: List[bool]
    ) -> NDArray[np.float64]:
        """Compute Generalized Advantage Estimation."""
        num_steps = len(rewards)
        advantages = np.zeros(num_steps)
        
        values = [self.get_value(s) for s in states]
        next_values = [self.get_value(s) for s in next_states]
        
        gae = 0.0
        for t in reversed(range(num_steps)):
            next_value = 0.0 if dones[t] else next_values[t]
            delta = rewards[t] + self.config.discount_factor * next_value - values[t]
            gae = delta + self.config.discount_factor * 0.95 * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages
    
    def train(
        self,
        env: Any,
        num_episodes: int,
        classical_baseline: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Train the hybrid agent."""
        logger.info("Starting Hybrid Q-C RL training for %d episodes", num_episodes)
        
        for episode_idx in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            
            episode_reward = 0.0
            done = False
            
            while not done:
                action, log_prob, value = self.select_action(state, training=True)
                
                result = env.step(action)
                if len(result) == 5:
                    next_state, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    next_state, reward, done, info = result
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                
                state = next_state
                episode_reward += reward
            
            # Train step
            train_metrics = self.train_step(states, actions, rewards, next_states, dones)
            
            # Compute quantum advantage
            quantum_advantage = 0.0
            if classical_baseline is not None and classical_baseline != 0:
                quantum_advantage = (episode_reward - classical_baseline) / abs(classical_baseline)
                
                # Update weights based on performance
                self.update_weights(episode_reward, classical_baseline)
            
            # Record metrics
            metrics = {
                "episode": episode_idx,
                "reward": episode_reward,
                "loss": train_metrics["loss"],
                "avg_advantage": train_metrics["avg_advantage"],
                "quantum_weight": train_metrics["quantum_weight"],
                "classical_weight": train_metrics["classical_weight"],
                "quantum_advantage": quantum_advantage,
                "episode_length": len(states)
            }
            self.metrics_history.append(metrics)
            
            # Log progress
            if (episode_idx + 1) % 10 == 0:
                avg_reward = np.mean([m["reward"] for m in self.metrics_history[-10:]])
                logger.info(
                    "Episode %d/%d | Avg Reward: %.4f | Q-Weight: %.3f | C-Weight: %.3f",
                    episode_idx + 1, num_episodes, avg_reward,
                    self.quantum_weight, self.classical_weight
                )
        
        logger.info("Hybrid Q-C RL training completed")
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
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get information about the hybrid architecture."""
        return {
            "architecture": self.config.architecture.name,
            "state_dim": self.config.state_dim,
            "action_dim": self.config.action_dim,
            "quantum_num_qubits": self.config.quantum_num_qubits,
            "quantum_num_layers": self.config.quantum_num_layers,
            "classical_hidden_dims": self.config.classical_hidden_dims,
            "quantum_weight": self.quantum_weight,
            "classical_weight": self.classical_weight,
            "adaptive_weighting": self.config.adaptive_weighting
        }


__all__ = [
    "HybridQuantumClassicalRL",
    "HybridRLConfig",
    "HybridArchitecture",
    "ClassicalPolicyNetwork",
    "ClassicalValueNetwork",
    "QuantumFeatureExtractor",
    "QuantumAdvantageEstimator"
]