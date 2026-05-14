"""
Quantum Convolutional Neural Network (QCNN).

A quantum analog of classical CNNs that uses translation-invariant 2-qubit
unitaries as "convolutional filters" and partial measurements as "pooling
layers". The result is a circuit with logarithmic depth and a small number
of trainable parameters.

Reference
---------
Cong, Choi, Lukin, "Quantum Convolutional Neural Networks,"
Nature Physics 15, 1273 (2019).

Architecture
------------
- Encode classical features via amplitude encoding (or angle encoding)
- For each layer:
  * **Conv layer**: parameterized 2-qubit unitary on every adjacent pair
  * **Pool layer**: measure every-other qubit and condition on outcome
- Final readout: measure the surviving qubits

For n input qubits, after log₂(n) layers we're left with 1 qubit, whose
expectation value is the network's output.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from quantum_simulator import QuantumCircuit, _simulate_statevector

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Quantum CNN
# ═════════════════════════════════════════════════════════════════════════════


class QuantumCNN:
    """
    Quantum Convolutional Neural Network.

    Parameters
    ----------
    n_qubits : int
        Number of input qubits. Should be a power of 2 for clean pooling.
    n_layers : int
        Number of (conv + pool) layers.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2) -> None:
        if n_qubits < 2:
            raise ValueError(f"n_qubits must be >= 2, got {n_qubits}")
        if n_qubits & (n_qubits - 1) != 0:
            logger.warning("n_qubits=%d is not a power of 2; pooling may not halve cleanly", n_qubits)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        # Each layer has 6 trainable parameters (3 for conv + 3 for pool)
        self.n_params = 6 * self.n_layers
        # Initialize parameters
        rng = np.random.default_rng(42)
        self.params = rng.uniform(-np.pi, np.pi, self.n_params)

    # ── Encoding ─────────────────────────────────────────────────────────────

    def encode(self, x: np.ndarray) -> QuantumCircuit:
        """
        Encode the feature vector x via angle encoding.

        Each feature x_i is mapped to an RY rotation on qubit i.
        """
        qc = QuantumCircuit(self.n_qubits)
        x_padded = np.zeros(self.n_qubits)
        x_padded[: len(x)] = x[: self.n_qubits]
        for q in range(self.n_qubits):
            qc.ry(float(x_padded[q]) * np.pi, q)
        return qc

    # ── Conv layer ───────────────────────────────────────────────────────────

    def _add_conv_layer(self, qc: QuantumCircuit, qubits: List[int], params: np.ndarray) -> None:
        """
        Translationally-invariant 2-qubit conv: same 3-parameter unitary on
        every adjacent pair of active qubits.
        """
        if len(qubits) < 2:
            return
        theta1, theta2, theta3 = params[0], params[1], params[2]
        for i in range(0, len(qubits) - 1, 2):
            q1 = qubits[i]
            q2 = qubits[i + 1]
            qc.ry(float(theta1), q1)
            qc.ry(float(theta2), q2)
            qc.cnot(q1, q2)
            qc.rz(float(theta3), q2)
        # Second pass on overlapping pairs (depth 2)
        for i in range(1, len(qubits) - 1, 2):
            q1 = qubits[i]
            q2 = qubits[i + 1]
            qc.ry(float(theta1), q1)
            qc.cnot(q1, q2)

    # ── Pool layer ───────────────────────────────────────────────────────────

    def _add_pool_layer(
        self, qc: QuantumCircuit, qubits: List[int], params: np.ndarray
    ) -> List[int]:
        """
        Pooling: pair (q_i, q_{i+1}) → keep q_i, "trace out" q_{i+1} via a
        controlled rotation followed by an X-basis measurement (we just keep
        the q_i side and conditionally rotate it).
        """
        theta1, theta2, theta3 = params[0], params[1], params[2]
        kept = []
        for i in range(0, len(qubits), 2):
            if i + 1 >= len(qubits):
                kept.append(qubits[i])
                continue
            q_keep = qubits[i]
            q_pool = qubits[i + 1]
            # Conditional rotation (the "pool" operator)
            qc.cry(float(theta1), q_pool, q_keep)
            qc.crz(float(theta2), q_pool, q_keep)
            qc.cnot(q_pool, q_keep)
            qc.ry(float(theta3), q_keep)
            kept.append(q_keep)
        return kept

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray, params: Optional[np.ndarray] = None) -> float:
        """
        Run the QCNN on input x. Returns ⟨Z⟩ on the final surviving qubit.
        """
        if params is None:
            params = self.params

        qc = self.encode(x)
        active_qubits = list(range(self.n_qubits))

        for layer in range(self.n_layers):
            conv_params = params[6 * layer : 6 * layer + 3]
            pool_params = params[6 * layer + 3 : 6 * layer + 6]
            self._add_conv_layer(qc, active_qubits, conv_params)
            active_qubits = self._add_pool_layer(qc, active_qubits, pool_params)
            if len(active_qubits) <= 1:
                break

        # Read out ⟨Z⟩ on the first surviving qubit
        state = _simulate_statevector(qc)
        readout_qubit = active_qubits[0] if active_qubits else 0
        z_exp = self._z_expectation(state, readout_qubit, self.n_qubits)
        return float(z_exp)

    def _z_expectation(self, state: np.ndarray, q: int, n: int) -> float:
        """⟨Z_q⟩ = P(q=0) - P(q=1)"""
        probs = np.abs(state) ** 2
        z = 0.0
        for idx in range(len(probs)):
            bit = (idx >> q) & 1
            z += probs[idx] * (1 if bit == 0 else -1)
        return float(z)

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 30,
        learning_rate: float = 0.05,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Train the QCNN parameters via gradient descent + finite differences.
        """
        from scipy.optimize import minimize as sp_minimize

        # Convert labels to ±1
        y_signed = np.where(y > 0, 1.0, -1.0)

        def cost(params):
            preds = np.array([self.forward(x, params) for x in X])
            return float(np.mean((preds - y_signed) ** 2))

        result = sp_minimize(
            cost,
            self.params.copy(),
            method="COBYLA",
            options={"maxiter": epochs, "rhobeg": 0.3},
        )

        self.params = result.x
        return {
            "final_cost": float(result.fun),
            "method": "qcnn_cobyla",
            "n_params": self.n_params,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels (0 or 1) for the input batch."""
        return np.array([1 if self.forward(x) > 0 else 0 for x in X])
