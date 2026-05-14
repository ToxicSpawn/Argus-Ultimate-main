# pyright: reportMissingImports=false
"""
Core utilities for Quantum Reinforcement Learning.

This module provides essential components for quantum RL including:
- Quantum state representation and encoding
- Quantum action selection mechanisms
- Quantum reward processing
- Quantum memory components (experience replay)
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ============================================================================
# Quantum State Representation
# ============================================================================

class StateEncoding(Enum):
    """Methods for encoding classical states into quantum states."""
    ANGLE = auto()       # Angle encoding via rotation gates
    AMPLITUDE = auto()   # Amplitude encoding
    BASIS = auto()       # Basis state encoding
    HADAMARD = auto()    # Hadamard test encoding


@dataclass
class QuantumState:
    """Quantum representation of a classical state."""
    state_vector: NDArray[np.complex128]
    num_qubits: int
    encoding: StateEncoding
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def measure_probabilities(self) -> NDArray[np.float64]:
        """Measure quantum state to get probabilities."""
        return np.abs(self.state_vector) ** 2
    
    def measure_expectation(self, observable: NDArray[np.complex128]) -> complex:
        """Compute expectation value of an observable."""
        return np.vdot(self.state_vector, observable @ self.state_vector)
    
    def fidelity(self, other: QuantumState) -> float:
        """Compute fidelity between two quantum states."""
        return float(np.abs(np.vdot(self.state_vector, other.state_vector)) ** 2)
    
    def entropy(self) -> float:
        """Compute von Neumann entropy of the state."""
        probs = self.measure_probabilities()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))


class QuantumStateEncoder:
    """Encodes classical states into quantum states."""
    
    def __init__(self, num_qubits: int, encoding: StateEncoding = StateEncoding.ANGLE):
        self.num_qubits = num_qubits
        self.encoding = encoding
        self.dimension = 2 ** num_qubits
    
    def encode(self, classical_state: NDArray[np.float64]) -> QuantumState:
        """Encode a classical state into a quantum state."""
        if self.encoding == StateEncoding.ANGLE:
            return self._encode_angle(classical_state)
        elif self.encoding == StateEncoding.AMPLITUDE:
            return self._encode_amplitude(classical_state)
        elif self.encoding == StateEncoding.BASIS:
            return self._encode_basis(classical_state)
        else:  # HADAMARD
            return self._encode_hadamard(classical_state)
    
    def _encode_angle(self, classical_state: NDArray[np.float64]) -> QuantumState:
        """Encode using angle encoding (RX/RY rotations)."""
        # Normalize state to [0, pi]
        normalized = (classical_state - np.min(classical_state)) / (
            np.max(classical_state) - np.min(classical_state) + 1e-8
        )
        
        # Initialize quantum state
        state_vector = np.zeros(self.dimension, dtype=np.complex128)
        state_vector[0] = 1.0 + 0j
        
        # Apply rotation gates
        for i in range(min(len(normalized), self.num_qubits)):
            angle = normalized[i] * np.pi
            state_vector = self._apply_ry(state_vector, i, angle)
        
        return QuantumState(
            state_vector=state_vector,
            num_qubits=self.num_qubits,
            encoding=StateEncoding.ANGLE,
            metadata={"original_state": classical_state.tolist()}
        )
    
    def _encode_amplitude(self, classical_state: NDArray[np.float64]) -> QuantumState:
        """Encode using amplitude encoding."""
        # Pad or truncate to fit quantum dimension
        padded = np.zeros(self.dimension, dtype=np.complex128)
        for i in range(min(len(classical_state), self.dimension)):
            padded[i] = classical_state[i]
        
        # Normalize
        norm = np.linalg.norm(padded)
        if norm > 0:
            padded = padded / norm
        
        return QuantumState(
            state_vector=padded,
            num_qubits=self.num_qubits,
            encoding=StateEncoding.AMPLITUDE,
            metadata={"original_state": classical_state.tolist()}
        )
    
    def _encode_basis(self, classical_state: NDArray[np.float64]) -> QuantumState:
        """Encode using basis state encoding."""
        # Convert to binary representation
        binary_states = []
        for val in classical_state[:self.num_qubits]:
            binary_states.append(1 if val > 0 else 0)
        
        # Create basis state
        state_vector = np.zeros(self.dimension, dtype=np.complex128)
        idx = 0
        for i, bit in enumerate(binary_states):
            idx += bit * (2 ** i)
        state_vector[idx] = 1.0 + 0j
        
        return QuantumState(
            state_vector=state_vector,
            num_qubits=self.num_qubits,
            encoding=StateEncoding.BASIS,
            metadata={"original_state": classical_state.tolist(), "binary": binary_states}
        )
    
    def _encode_hadamard(self, classical_state: NDArray[np.float64]) -> QuantumState:
        """Encode using Hadamard test encoding."""
        # Initialize with Hadamard on all qubits
        state_vector = np.ones(self.dimension, dtype=np.complex128) / np.sqrt(self.dimension)
        
        # Apply phase rotations based on classical state
        normalized = (classical_state - np.mean(classical_state)) / (np.std(classical_state) + 1e-8)
        
        for i in range(min(len(normalized), self.num_qubits)):
            phase = normalized[i] * np.pi
            state_vector = self._apply_rz(state_vector, i, phase)
        
        return QuantumState(
            state_vector=state_vector,
            num_qubits=self.num_qubits,
            encoding=StateEncoding.HADAMARD,
            metadata={"original_state": classical_state.tolist()}
        )
    
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


# ============================================================================
# Quantum Action Selection
# ============================================================================

class ActionSelectionMethod(Enum):
    """Methods for selecting actions in quantum RL."""
    EPSILON_GREEDY = auto()
    QUANTUM_SAMPLING = auto()
    BOLTZMANN = auto()
    THOMPSON_SAMPLING = auto()


@dataclass
class QuantumAction:
    """Quantum representation of an action."""
    action_id: int
    quantum_probability: float
    classical_probability: float
    state_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumActionSelector:
    """Selects actions using quantum-enhanced methods."""
    
    def __init__(
        self,
        action_dim: int,
        method: ActionSelectionMethod = ActionSelectionMethod.EPSILON_GREEDY,
        epsilon: float = 0.1
    ):
        self.action_dim = action_dim
        self.method = method
        self.epsilon = epsilon
        self.temperature = 1.0
    
    def select(
        self,
        quantum_state: QuantumState,
        training: bool = True
    ) -> QuantumAction:
        """Select an action based on quantum state."""
        if self.method == ActionSelectionMethod.EPSILON_GREEDY:
            return self._epsilon_greedy(quantum_state, training)
        elif self.method == ActionSelectionMethod.QUANTUM_SAMPLING:
            return self._quantum_sampling(quantum_state)
        elif self.method == ActionSelectionMethod.BOLTZMANN:
            return self._boltzmann(quantum_state)
        else:  # THOMPSON_SAMPLING
            return self._thompson_sampling(quantum_state)
    
    def _epsilon_greedy(self, quantum_state: QuantumState, training: bool) -> QuantumAction:
        """Epsilon-greedy action selection."""
        probabilities = quantum_state.measure_probabilities()[:self.action_dim]
        probabilities = probabilities / (np.sum(probabilities) + 1e-10)
        
        if training and random.random() < self.epsilon:
            # Explore
            action_id = random.randint(0, self.action_dim - 1)
            quantum_prob = probabilities[action_id]
            classical_prob = 1.0 / self.action_dim
        else:
            # Exploit
            action_id = int(np.argmax(probabilities))
            quantum_prob = probabilities[action_id]
            classical_prob = quantum_prob
        
        return QuantumAction(
            action_id=action_id,
            quantum_probability=float(quantum_prob),
            classical_probability=float(classical_prob),
            state_value=float(probabilities[action_id]),
            metadata={"method": "epsilon_greedy", "epsilon": self.epsilon}
        )
    
    def _quantum_sampling(self, quantum_state: QuantumState) -> QuantumAction:
        """Sample action based on quantum measurement probabilities."""
        probabilities = quantum_state.measure_probabilities()[:self.action_dim]
        probabilities = probabilities / (np.sum(probabilities) + 1e-10)
        
        action_id = np.random.choice(self.action_dim, p=probabilities)
        
        return QuantumAction(
            action_id=int(action_id),
            quantum_probability=float(probabilities[action_id]),
            classical_probability=float(probabilities[action_id]),
            state_value=float(probabilities[action_id]),
            metadata={"method": "quantum_sampling"}
        )
    
    def _boltzmann(self, quantum_state: QuantumState) -> QuantumAction:
        """Boltzmann (softmax) action selection."""
        probabilities = quantum_state.measure_probabilities()[:self.action_dim]
        
        # Apply temperature
        logits = np.log(probabilities + 1e-10) / self.temperature
        exp_logits = np.exp(logits - np.max(logits))
        softmax_probs = exp_logits / np.sum(exp_logits)
        
        action_id = np.random.choice(self.action_dim, p=softmax_probs)
        
        return QuantumAction(
            action_id=int(action_id),
            quantum_probability=float(softmax_probs[action_id]),
            classical_probability=float(softmax_probs[action_id]),
            state_value=float(softmax_probs[action_id]),
            metadata={"method": "boltzmann", "temperature": self.temperature}
        )
    
    def _thompson_sampling(self, quantum_state: QuantumState) -> QuantumAction:
        """Thompson sampling action selection."""
        # Use quantum state as prior for Beta distribution
        probabilities = quantum_state.measure_probabilities()[:self.action_dim]
        
        # Sample from Beta distributions
        samples = []
        for i in range(self.action_dim):
            alpha = probabilities[i] * 10 + 1
            beta = (1 - probabilities[i]) * 10 + 1
            samples.append(np.random.beta(alpha, beta))
        
        action_id = int(np.argmax(samples))
        
        return QuantumAction(
            action_id=action_id,
            quantum_probability=float(probabilities[action_id]),
            classical_probability=float(samples[action_id]),
            state_value=float(samples[action_id]),
            metadata={"method": "thompson_sampling"}
        )
    
    def update_epsilon(self, decay: float = 0.995, min_epsilon: float = 0.01) -> None:
        """Update epsilon for exploration decay."""
        self.epsilon = max(min_epsilon, self.epsilon * decay)
    
    def update_temperature(self, decay: float = 0.995, min_temperature: float = 0.1) -> None:
        """Update temperature for Boltzmann exploration."""
        self.temperature = max(min_temperature, self.temperature * decay)


# ============================================================================
# Quantum Reward Processing
# ============================================================================

class RewardShapingMethod(Enum):
    """Methods for shaping rewards in quantum RL."""
    POTENTIAL_BASED = auto()
    QUANTUM_POTENTIAL = auto()
    INTRINSIC_MOTIVATION = auto()
    HIERARCHICAL = auto()


@dataclass
class QuantumReward:
    """Quantum-processed reward."""
    raw_reward: float
    shaped_reward: float
    intrinsic_reward: float
    total_reward: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumRewardProcessor:
    """Processes rewards using quantum-inspired methods."""
    
    def __init__(
        self,
        method: RewardShapingMethod = RewardShapingMethod.POTENTIAL_BASED,
        discount_factor: float = 0.99,
        potential_scale: float = 0.1
    ):
        self.method = method
        self.discount_factor = discount_factor
        self.potential_scale = potential_scale
        
        # State potential tracker
        self.state_potentials: Dict[str, float] = {}
    
    def process(
        self,
        raw_reward: float,
        state: NDArray[np.float64],
        next_state: NDArray[np.float64],
        done: bool
    ) -> QuantumReward:
        """Process raw reward into shaped reward."""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        if self.method == RewardShapingMethod.POTENTIAL_BASED:
            shaped = self._potential_based_shaping(raw_reward, state_key, next_state_key)
        elif self.method == RewardShapingMethod.QUANTUM_POTENTIAL:
            shaped = self._quantum_potential_shaping(raw_reward, state, next_state)
        elif self.method == RewardShapingMethod.INTRINSIC_MOTIVATION:
            shaped = self._intrinsic_motivation(raw_reward, state, next_state)
        else:  # HIERARCHICAL
            shaped = self._hierarchical_shaping(raw_reward, state, next_state, done)
        
        return QuantumReward(
            raw_reward=raw_reward,
            shaped_reward=shaped["shaped"],
            intrinsic_reward=shaped["intrinsic"],
            total_reward=shaped["total"],
            metadata=shaped.get("metadata", {})
        )
    
    def _potential_based_shaping(
        self,
        raw_reward: float,
        state_key: str,
        next_state_key: str
    ) -> Dict[str, float]:
        """Potential-based reward shaping."""
        gamma = self.discount_factor
        
        # Get or compute potentials
        if state_key not in self.state_potentials:
            self.state_potentials[state_key] = random.uniform(0, 1)
        if next_state_key not in self.state_potentials:
            self.state_potentials[next_state_key] = random.uniform(0, 1)
        
        phi_s = self.state_potentials[state_key]
        phi_s_next = self.state_potentials[next_state_key]
        
        # Shaped reward: r + gamma * phi(s') - phi(s)
        shaped = raw_reward + gamma * phi_s_next - phi_s
        
        return {
            "shaped": shaped,
            "intrinsic": 0.0,
            "total": shaped,
            "metadata": {"phi_s": phi_s, "phi_s_next": phi_s_next}
        }
    
    def _quantum_potential_shaping(
        self,
        raw_reward: float,
        state: NDArray[np.float64],
        next_state: NDArray[np.float64]
    ) -> Dict[str, float]:
        """Quantum-inspired potential-based shaping."""
        gamma = self.discount_factor
        
        # Compute quantum-inspired potentials using state similarity
        state_norm = np.linalg.norm(state)
        next_state_norm = np.linalg.norm(next_state)
        
        # Quantum potential based on state "amplitude"
        phi_s = np.tanh(state_norm * self.potential_scale)
        phi_s_next = np.tanh(next_state_norm * self.potential_scale)
        
        shaped = raw_reward + gamma * phi_s_next - phi_s
        
        return {
            "shaped": shaped,
            "intrinsic": 0.0,
            "total": shaped,
            "metadata": {"quantum_potential_s": phi_s, "quantum_potential_next": phi_s_next}
        }
    
    def _intrinsic_motivation(
        self,
        raw_reward: float,
        state: NDArray[np.float64],
        next_state: NDArray[np.float64]
    ) -> Dict[str, float]:
        """Intrinsic motivation reward shaping."""
        # Curiosity-driven intrinsic reward
        state_change = np.linalg.norm(next_state - state)
        intrinsic = 0.1 * np.tanh(state_change)
        
        total = raw_reward + intrinsic
        
        return {
            "shaped": raw_reward,
            "intrinsic": intrinsic,
            "total": total,
            "metadata": {"state_change": state_change}
        }
    
    def _hierarchical_shaping(
        self,
        raw_reward: float,
        state: NDArray[np.float64],
        next_state: NDArray[np.float64],
        done: bool
    ) -> Dict[str, float]:
        """Hierarchical reward shaping."""
        # Combine multiple shaping methods
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # Potential-based component
        potential_shaping = self._potential_based_shaping(raw_reward, state_key, next_state_key)
        
        # Intrinsic motivation component
        intrinsic_shaping = self._intrinsic_motivation(raw_reward, state, next_state)
        
        # Combine
        shaped = 0.7 * potential_shaping["shaped"] + 0.3 * intrinsic_shaping["intrinsic"]
        
        return {
            "shaped": shaped,
            "intrinsic": intrinsic_shaping["intrinsic"],
            "total": shaped + raw_reward,
            "metadata": {
                "potential_component": potential_shaping["shaped"],
                "intrinsic_component": intrinsic_shaping["intrinsic"]
            }
        }
    
    def _state_to_key(self, state: NDArray[np.float64]) -> str:
        """Convert state to hashable key."""
        return ",".join(f"{v:.4f}" for v in state[:8])  # Use first 8 dimensions


# ============================================================================
# Quantum Memory Components
# ============================================================================

@dataclass
class QuantumExperience:
    """Experience stored in quantum memory."""
    state: QuantumState
    action: QuantumAction
    reward: float
    next_state: Optional[QuantumState] = None
    done: bool = False
    priority: float = 1.0
    timestamp: int = 0


class QuantumReplayBuffer:
    """Quantum-enhanced experience replay buffer."""
    
    def __init__(
        self,
        capacity: int = 10000,
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4
    ):
        self.capacity = capacity
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta
        
        self.buffer: Deque[QuantumExperience] = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity)
        self.position = 0
    
    def add(self, experience: QuantumExperience) -> None:
        """Add experience to buffer."""
        max_priority = self.priorities[:len(self.buffer)].max() if len(self.buffer) > 0 else 1.0
        priority = max_priority ** self.priority_alpha
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[QuantumExperience], NDArray[np.float64]]:
        """Sample batch with prioritization."""
        n = len(self.buffer)
        
        if n < batch_size:
            batch_size = n
        
        # Compute sampling probabilities
        priorities = self.priorities[:n]
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(n, batch_size, p=probabilities, replace=False)
        
        # Compute importance sampling weights
        weights = (n * probabilities[indices]) ** (-self.priority_beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [list(self.buffer)[i] for i in indices]
        
        return experiences, weights
    
    def update_priorities(self, indices: List[int], td_errors: NDArray[np.float64]) -> None:
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-6) ** self.priority_alpha
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.priorities = np.zeros(self.capacity)
        self.position = 0


class QuantumStateCache:
    """Cache for quantum state representations."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, QuantumState] = {}
        self.access_count: Dict[str, int] = {}
    
    def get(self, state_key: str) -> Optional[QuantumState]:
        """Get cached quantum state."""
        if state_key in self.cache:
            self.access_count[state_key] += 1
            return self.cache[state_key]
        return None
    
    def put(self, state_key: str, quantum_state: QuantumState) -> None:
        """Put quantum state in cache."""
        if len(self.cache) >= self.max_size:
            # Evict least accessed
            least_accessed = min(self.access_count, key=self.access_count.get)
            del self.cache[least_accessed]
            del self.access_count[least_accessed]
        
        self.cache[state_key] = quantum_state
        self.access_count[state_key] = 1
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_count.clear()


__all__ = [
    # State representation
    "QuantumState",
    "QuantumStateEncoder",
    "StateEncoding",
    
    # Action selection
    "QuantumAction",
    "QuantumActionSelector",
    "ActionSelectionMethod",
    
    # Reward processing
    "QuantumReward",
    "QuantumRewardProcessor",
    "RewardShapingMethod",
    
    # Memory components
    "QuantumExperience",
    "QuantumReplayBuffer",
    "QuantumStateCache"
]