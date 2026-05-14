"""
Quantum Deep Q-Network with VQC policy.

DQN-style RL agent where the Q-network is a parameterized quantum circuit.
Complementary to the existing ``quantum_rl.py``.

Reference
---------
Chen, Yang, Yoo, Ouyang, Hsieh, "Variational quantum circuits for deep
reinforcement learning," IEEE Access 8, 141007 (2020)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from quantum_simulator import QuantumCircuit, _simulate_statevector

logger = logging.getLogger(__name__)


class VQCQNetwork:
    """Variational Quantum Circuit Q-network."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        *,
        n_layers: int = 3,
    ) -> None:
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.n_qubits = max(
            int(np.ceil(np.log2(max(n_actions, 2)))),
            state_dim,
        )
        self.n_layers = int(n_layers)
        self.n_params = self.n_qubits * self.n_layers * 3
        rng = np.random.default_rng(42)
        self.params = rng.uniform(-np.pi, np.pi, self.n_params)

    def q_values(self, state: np.ndarray, params: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute Q-values via ⟨Z_q⟩ on action qubits."""
        if params is None:
            params = self.params
        qc = self._build_circuit(state, params)
        psi = _simulate_statevector(qc)
        probs = np.abs(psi) ** 2

        q_vals = np.zeros(self.n_actions)
        for a in range(min(self.n_actions, self.n_qubits)):
            z = 0.0
            for idx in range(len(probs)):
                bit = (idx >> a) & 1
                z += probs[idx] * (1.0 if bit == 0 else -1.0)
            q_vals[a] = float(z)
        return q_vals

    def _build_circuit(self, state: np.ndarray, params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for q in range(self.n_qubits):
            if q < self.state_dim:
                angle = float(np.clip(state[q], -np.pi, np.pi))
                qc.ry(angle, q)
        idx = 0
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                qc.rx(float(params[idx]), q)
                idx += 1
                qc.ry(float(params[idx]), q)
                idx += 1
                qc.rz(float(params[idx]), q)
                idx += 1
            for q in range(self.n_qubits):
                qc.cnot(q, (q + 1) % self.n_qubits)
        return qc


class QuantumDQN:
    """DQN agent with VQC Q-network."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        *,
        gamma: float = 0.95,
        learning_rate: float = 0.1,
    ) -> None:
        self.policy = VQCQNetwork(state_dim, n_actions)
        self.target = VQCQNetwork(state_dim, n_actions)
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.replay: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    def act(self, state: np.ndarray, *, epsilon: float = 0.0) -> int:
        if epsilon > 0 and np.random.random() < epsilon:
            return int(np.random.randint(self.policy.n_actions))
        return int(np.argmax(self.policy.q_values(state)))

    def remember(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s_next: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.append((s.copy(), int(a), float(r), s_next.copy(), bool(done)))
        if len(self.replay) > 5000:
            self.replay.pop(0)

    def train_step(self, batch_size: int = 32) -> Dict[str, Any]:
        from scipy.optimize import minimize as sp_minimize

        if len(self.replay) < batch_size:
            return {"loss": None, "samples": 0}

        rng = np.random.default_rng()
        indices = rng.choice(len(self.replay), size=batch_size, replace=False)
        batch = [self.replay[i] for i in indices]

        def loss_fn(params):
            total = 0.0
            for s, a, r, s_next, done in batch:
                q_values = self.policy.q_values(s, params)
                if done:
                    target_q = r
                else:
                    next_q = self.target.q_values(s_next)
                    target_q = r + self.gamma * float(np.max(next_q))
                total += (q_values[a] - target_q) ** 2
            return float(total / max(len(batch), 1))

        opt = sp_minimize(
            loss_fn,
            self.policy.params.copy(),
            method="COBYLA",
            options={"maxiter": 10, "rhobeg": self.learning_rate},
        )
        self.policy.params = np.asarray(opt.x, dtype=float)
        return {"loss": float(opt.fun), "samples": batch_size}

    def sync_target(self) -> None:
        self.target.params = self.policy.params.copy()
