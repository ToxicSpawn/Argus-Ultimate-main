"""
Quantum-Enhanced Reinforcement Learning for Trading.

A quantum policy network that maps trading state to action probabilities via
a parameterised quantum circuit.  The policy is trained using the REINFORCE
algorithm with the parameter shift rule for gradient computation.

Actions: BUY (0), HOLD (1), SELL (2)

This is a classical simulation of a quantum circuit — no quantum hardware
is used.  The quantum circuit provides a compact, expressive function
approximator for the policy, with entanglement enabling complex state-action
correlations.  For small state spaces this is tractable and exact.

Typical usage::

    from quantum.qml.quantum_rl import QuantumPolicyNetwork

    qpn = QuantumPolicyNetwork(n_state_features=8, n_actions=3)

    # Training loop
    result = qpn.train_episode(env_step_fn, max_steps=100)

    # Inference
    action = qpn.get_trading_action({
        'price': 60000, 'returns_5': 0.01, 'vol_10': 0.03,
        'rsi': 55, 'macd': 0.001, 'spread': 5.0,
        'volume': 1e6, 'regime': 'TRENDING'
    })
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Action constants
BUY = 0
HOLD = 1
SELL = 2
ACTION_NAMES = {BUY: "BUY", HOLD: "HOLD", SELL: "SELL"}

# Regime encoding
REGIME_MAP = {
    "TRENDING": 0.0,
    "MEAN_REVERTING": 0.25,
    "VOLATILE": 0.5,
    "CRISIS": 0.75,
    "FLAT": -0.25,
    "UNKNOWN": 0.0,
}


class QuantumPolicyNetwork:
    """
    Quantum policy network for RL trading agent.

    Maps state -> action probabilities via parameterised quantum circuit.
    Actions: BUY (0), HOLD (1), SELL (2)
    """

    def __init__(
        self,
        n_state_features: int = 8,
        n_actions: int = 3,
        n_layers: int = 2,
        n_qubits: int = 4,
        learning_rate: float = 0.01,
        epsilon: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        if n_state_features < 1:
            raise ValueError(f"n_state_features must be >= 1, got {n_state_features}")
        if n_actions < 2:
            raise ValueError(f"n_actions must be >= 2, got {n_actions}")
        if n_qubits < 1 or n_qubits > 12:
            raise ValueError(f"n_qubits must be in [1, 12], got {n_qubits}")

        self.n_state_features = n_state_features
        self.n_actions = n_actions
        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self._rng = np.random.RandomState(seed)

        # Circuit parameters: encoding + variational layers
        # Encoding: n_state_features scaling params
        # Variational: n_layers * n_qubits * 3 rotation angles
        self._n_params = n_state_features + n_layers * n_qubits * 3
        self.params = self._rng.randn(self._n_params) * 0.1

        # Training stats
        self._total_episodes = 0
        self._total_updates = 0
        self._episode_rewards: List[float] = []

    # ------------------------------------------------------------------
    # Quantum circuit forward pass
    # ------------------------------------------------------------------

    def _encode_state(self, state: np.ndarray) -> np.ndarray:
        """Encode trading state into quantum state.

        - Normalise features to [0, 2pi]
        - Apply Ry(feature * param) rotations
        - Apply entangling layers
        - Return statevector
        """
        n = self.n_qubits
        dim = 2 ** n

        state_vec = np.zeros(dim, dtype=np.complex128)
        state_vec[0] = 1.0 + 0j

        idx = 0

        # Feature encoding: Ry rotations on qubits (cycle through qubits)
        for f in range(self.n_state_features):
            q = f % n
            angle = float(state[f]) * float(self.params[idx])
            state_vec = self._apply_ry(state_vec, n, q, angle)
            idx += 1

        # Variational layers
        for layer in range(self.n_layers):
            for q in range(n):
                state_vec = self._apply_rx(state_vec, n, q, float(self.params[idx]))
                idx += 1
                state_vec = self._apply_ry(state_vec, n, q, float(self.params[idx]))
                idx += 1
                state_vec = self._apply_rz(state_vec, n, q, float(self.params[idx]))
                idx += 1

            # Entangling CNOT ring
            for q in range(n - 1):
                state_vec = self._apply_cnot(state_vec, n, q, q + 1)
            if n > 1:
                state_vec = self._apply_cnot(state_vec, n, n - 1, 0)

        return state_vec

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass: state -> action probabilities.

        Apply softmax to qubit Z-expectations to get action probabilities.
        """
        state_arr = np.asarray(state, dtype=np.float64).ravel()

        # Pad or truncate
        if len(state_arr) < self.n_state_features:
            padded = np.zeros(self.n_state_features)
            padded[: len(state_arr)] = state_arr
            state_arr = padded
        elif len(state_arr) > self.n_state_features:
            state_arr = state_arr[: self.n_state_features]

        # Handle NaN
        nan_mask = np.isnan(state_arr)
        if nan_mask.any():
            state_arr = state_arr.copy()
            state_arr[nan_mask] = 0.0

        sv = self._encode_state(state_arr)
        expectations = self._measure_z_expectations(sv)

        # Map expectations to action logits
        # Use first n_actions expectations, or combine if fewer qubits
        if self.n_qubits >= self.n_actions:
            logits = expectations[: self.n_actions]
        else:
            # Spread qubits across actions
            logits = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                q = a % self.n_qubits
                logits[a] = expectations[q] * (1 + 0.1 * a)

        # Softmax
        logits = logits - np.max(logits)  # numerical stability
        exp_logits = np.exp(logits)
        probs = exp_logits / (np.sum(exp_logits) + 1e-10)

        return probs

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action with epsilon-greedy exploration.

        Returns: (action, probability, entropy)
        """
        probs = self.forward(state)

        # Entropy
        safe_probs = np.clip(probs, 1e-10, 1.0)
        entropy = float(-np.sum(safe_probs * np.log(safe_probs)))

        # Epsilon-greedy
        if self._rng.rand() < self.epsilon:
            action = int(self._rng.randint(self.n_actions))
            exploration = True
        else:
            action = int(np.argmax(probs))
            exploration = False

        probability = float(probs[action])
        return action, probability, entropy

    # ------------------------------------------------------------------
    # Policy gradient update (REINFORCE)
    # ------------------------------------------------------------------

    def update(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        gamma: float = 0.99,
    ) -> Dict[str, float]:
        """REINFORCE policy gradient update.

        - Compute discounted returns
        - Compute log probabilities via parameter shift rule
        - Update params: theta += lr * sum(G_t * grad log pi(a|s))

        Returns: {loss, avg_reward, entropy}
        """
        if len(states) == 0:
            return {"loss": 0.0, "avg_reward": 0.0, "entropy": 0.0}

        # Compute discounted returns
        returns = np.zeros(len(rewards))
        g = 0.0
        for t in reversed(range(len(rewards))):
            g = rewards[t] + gamma * g
            returns[t] = g

        # Normalise returns
        if len(returns) > 1:
            ret_std = float(np.std(returns))
            if ret_std > 1e-8:
                returns = (returns - np.mean(returns)) / ret_std

        # Compute gradient via parameter shift rule
        shift = np.pi / 2  # standard parameter shift
        grad = np.zeros_like(self.params)
        total_entropy = 0.0
        total_loss = 0.0

        for t in range(len(states)):
            state = np.asarray(states[t], dtype=np.float64).ravel()
            action = actions[t]
            g_t = returns[t]

            # Current log probability
            probs = self.forward(state)
            safe_probs = np.clip(probs, 1e-10, 1.0)
            log_prob = np.log(safe_probs[action])
            total_loss -= float(g_t * log_prob)
            total_entropy += float(-np.sum(safe_probs * np.log(safe_probs)))

            # Parameter shift gradient for log pi(a|s)
            for p in range(len(self.params)):
                # Forward shift
                original = self.params[p]
                self.params[p] = original + shift
                probs_plus = self.forward(state)
                log_prob_plus = np.log(np.clip(probs_plus[action], 1e-10, 1.0))

                # Backward shift
                self.params[p] = original - shift
                probs_minus = self.forward(state)
                log_prob_minus = np.log(np.clip(probs_minus[action], 1e-10, 1.0))

                # Restore
                self.params[p] = original

                # Gradient of log pi(a|s) w.r.t. theta_p
                d_log_prob = (log_prob_plus - log_prob_minus) / (2 * np.sin(shift))
                grad[p] += g_t * d_log_prob

        # Average gradient
        grad /= len(states)

        # Update parameters (gradient ascent for reward maximisation)
        self.params += self.learning_rate * grad

        self._total_updates += 1

        return {
            "loss": round(total_loss / len(states), 6),
            "avg_reward": round(float(np.mean(rewards)), 6),
            "entropy": round(total_entropy / len(states), 6),
        }

    # ------------------------------------------------------------------
    # Episode training
    # ------------------------------------------------------------------

    def train_episode(
        self,
        env_step_fn: Callable[[int], Tuple[np.ndarray, float, bool]],
        initial_state: Optional[np.ndarray] = None,
        max_steps: int = 100,
    ) -> Dict[str, Any]:
        """Train on one episode.

        env_step_fn(action) -> (next_state, reward, done)

        Returns: {total_reward, steps, actions_taken, avg_entropy}
        """
        states: List[np.ndarray] = []
        actions: List[int] = []
        rewards: List[float] = []
        entropies: List[float] = []
        action_counts = {BUY: 0, HOLD: 0, SELL: 0}

        if initial_state is not None:
            state = np.asarray(initial_state, dtype=np.float64).ravel()
        else:
            # Dummy initial state
            state = np.zeros(self.n_state_features)

        for step in range(max_steps):
            action, prob, entropy = self.select_action(state)

            states.append(state.copy())
            actions.append(action)
            entropies.append(entropy)
            if action in action_counts:
                action_counts[action] += 1

            next_state, reward, done = env_step_fn(action)
            rewards.append(reward)
            state = np.asarray(next_state, dtype=np.float64).ravel()

            if done:
                break

        # Policy gradient update
        update_result = self.update(states, actions, rewards)

        self._total_episodes += 1
        total_reward = float(sum(rewards))
        self._episode_rewards.append(total_reward)

        return {
            "total_reward": round(total_reward, 6),
            "steps": len(states),
            "actions_taken": {ACTION_NAMES.get(k, str(k)): v for k, v in action_counts.items()},
            "avg_entropy": round(float(np.mean(entropies)) if entropies else 0.0, 4),
            "update_loss": update_result["loss"],
            "episode_number": self._total_episodes,
        }

    # ------------------------------------------------------------------
    # Trading action interface
    # ------------------------------------------------------------------

    def get_trading_action(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert market state dict to action recommendation.

        market_state: {price, returns_5, vol_10, rsi, macd, spread, volume, regime}
        Returns: {action: 'BUY'|'HOLD'|'SELL', confidence, entropy, exploration: bool}
        """
        # Extract features in canonical order
        features = np.zeros(self.n_state_features)
        feature_keys = [
            "price", "returns_5", "vol_10", "rsi", "macd",
            "spread", "volume", "regime",
        ]

        for i, key in enumerate(feature_keys):
            if i >= self.n_state_features:
                break
            val = market_state.get(key, 0.0)
            if key == "regime":
                val = REGIME_MAP.get(str(val).upper(), 0.0)
            elif isinstance(val, (int, float)):
                val = float(val)
            else:
                val = 0.0

            # Basic normalisation
            if key == "price":
                val = val / 100000.0  # scale prices
            elif key == "rsi":
                val = (val - 50.0) / 50.0  # centre RSI
            elif key == "volume":
                val = np.log1p(abs(val)) / 20.0  # log-scale volume

            features[i] = val

        probs = self.forward(features)
        action = int(np.argmax(probs))
        confidence = float(probs[action])

        safe_probs = np.clip(probs, 1e-10, 1.0)
        entropy = float(-np.sum(safe_probs * np.log(safe_probs)))

        # High entropy = uncertain -> flag as exploratory
        max_entropy = float(np.log(self.n_actions))
        exploration = entropy > 0.8 * max_entropy

        return {
            "action": ACTION_NAMES.get(action, "HOLD"),
            "confidence": round(confidence, 4),
            "entropy": round(entropy, 4),
            "exploration": exploration,
            "probabilities": {
                ACTION_NAMES.get(a, str(a)): round(float(probs[a]), 4)
                for a in range(self.n_actions)
            },
        }

    # ------------------------------------------------------------------
    # Gate primitives
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_ry(state: np.ndarray, n: int, qubit: int, angle: float) -> np.ndarray:
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        gate = np.array([[c, -s], [s, c]], dtype=np.complex128)
        return QuantumPolicyNetwork._apply_gate(state, n, qubit, gate)

    @staticmethod
    def _apply_rx(state: np.ndarray, n: int, qubit: int, angle: float) -> np.ndarray:
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        gate = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
        return QuantumPolicyNetwork._apply_gate(state, n, qubit, gate)

    @staticmethod
    def _apply_rz(state: np.ndarray, n: int, qubit: int, angle: float) -> np.ndarray:
        gate = np.array(
            [[np.exp(-1j * angle / 2), 0], [0, np.exp(1j * angle / 2)]],
            dtype=np.complex128,
        )
        return QuantumPolicyNetwork._apply_gate(state, n, qubit, gate)

    @staticmethod
    def _apply_gate(
        state: np.ndarray, n: int, qubit: int, gate: np.ndarray
    ) -> np.ndarray:
        shape = [2] * n
        psi = state.reshape(shape)
        psi = np.moveaxis(psi, qubit, -1)
        psi = np.einsum("ij,...j->...i", gate, psi)
        psi = np.moveaxis(psi, -1, qubit)
        return psi.reshape(2 ** n)

    @staticmethod
    def _apply_cnot(state: np.ndarray, n: int, control: int, target: int) -> np.ndarray:
        dim = 2 ** n
        new_state = np.zeros(dim, dtype=np.complex128)
        for i in range(dim):
            bits = [(i >> (n - 1 - q)) & 1 for q in range(n)]
            if bits[control] == 1:
                bits[target] ^= 1
                j = 0
                for q in range(n):
                    j = (j << 1) | bits[q]
                new_state[j] += state[i]
            else:
                new_state[i] += state[i]
        return new_state

    def _measure_z_expectations(self, state: np.ndarray) -> np.ndarray:
        """Measure <Z_i> for each qubit."""
        probs = np.abs(state) ** 2
        expectations = np.zeros(self.n_qubits)
        for q in range(self.n_qubits):
            for i in range(len(state)):
                bit = (i >> (self.n_qubits - 1 - q)) & 1
                expectations[q] += (1 - 2 * bit) * probs[i]
        return expectations

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return summary of the quantum policy network."""
        return {
            "n_state_features": self.n_state_features,
            "n_actions": self.n_actions,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_params": self._n_params,
            "total_episodes": self._total_episodes,
            "total_updates": self._total_updates,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "avg_reward_last_10": (
                round(float(np.mean(self._episode_rewards[-10:])), 4)
                if self._episode_rewards
                else None
            ),
            "method": "quantum_policy_reinforce",
        }
