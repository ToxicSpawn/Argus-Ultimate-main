"""
Quantum Long Short-Term Memory (Q-LSTM) network.

A hybrid quantum-classical recurrent network that uses parameterized
quantum circuits as the gates of an LSTM cell:

    f_t = σ(VQC_f(h_{t-1}, x_t))
    i_t = σ(VQC_i(h_{t-1}, x_t))
    o_t = σ(VQC_o(h_{t-1}, x_t))
    c_tilde = tanh(VQC_c(h_{t-1}, x_t))
    c_t = f_t · c_{t-1} + i_t · c_tilde
    h_t = o_t · tanh(c_t)

Each VQC is a small variational quantum circuit (RY + CNOT + readout).

Reference
---------
Chen, Yang, Yoo, Liao, "Quantum Long Short-Term Memory,"
ICASSP 2022, arXiv:2009.01783

Trading use
-----------
Sequence model for return prediction; wire as alternative to LSTM in
ml/lstm_regime.py for hybrid quantum-classical regime detection.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from quantum_simulator import QuantumCircuit, _simulate_statevector

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Quantum LSTM cell
# ═════════════════════════════════════════════════════════════════════════════


class QuantumLSTMCell:
    """
    A single Q-LSTM cell with 4 VQC gates (forget, input, output, candidate).

    Parameters
    ----------
    input_size : int
        Number of input features per time step.
    hidden_size : int
        Number of hidden state units.
    n_qubits : int
        Number of qubits in each VQC. Should be >= max(input_size + hidden_size, 2).
    n_layers : int
        Variational ansatz depth per gate.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_qubits: int = 4,
        n_layers: int = 2,
    ) -> None:
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)

        # 4 VQC gates: forget, input, output, candidate
        # Each VQC has n_qubits * n_layers parameters (one RY per qubit per layer)
        self.n_params_per_gate = self.n_qubits * self.n_layers
        rng = np.random.default_rng(42)
        self.w_f = rng.uniform(-0.5, 0.5, self.n_params_per_gate)
        self.w_i = rng.uniform(-0.5, 0.5, self.n_params_per_gate)
        self.w_o = rng.uniform(-0.5, 0.5, self.n_params_per_gate)
        self.w_c = rng.uniform(-0.5, 0.5, self.n_params_per_gate)

        # Classical input/hidden mixing weights (small)
        self.W_in = rng.normal(0, 0.1, (n_qubits, input_size + hidden_size))
        self.b_in = np.zeros(n_qubits)

    def step(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single time-step forward pass.

        Parameters
        ----------
        x : np.ndarray
            Input vector of shape (input_size,).
        h_prev : np.ndarray
            Previous hidden state, shape (hidden_size,).
        c_prev : np.ndarray
            Previous cell state, shape (hidden_size,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (h_t, c_t) — new hidden and cell states.
        """
        # Encode classical inputs into qubit angles
        x_combined = np.concatenate([x, h_prev])
        angles = self.W_in @ x_combined + self.b_in
        # Clip angles to [-π, π]
        angles = np.clip(angles, -np.pi, np.pi)

        # Run 4 VQCs
        f = self._run_vqc(angles, self.w_f)
        i = self._run_vqc(angles, self.w_i)
        o = self._run_vqc(angles, self.w_o)
        c_tilde = self._run_vqc(angles, self.w_c, activation="tanh")

        # Sigmoid for f, i, o
        f = self._sigmoid(f)
        i_g = self._sigmoid(i)
        o_g = self._sigmoid(o)

        # Pad / repeat to match hidden_size
        if len(f) < self.hidden_size:
            f = np.tile(f, self.hidden_size // len(f) + 1)[: self.hidden_size]
            i_g = np.tile(i_g, self.hidden_size // len(i_g) + 1)[: self.hidden_size]
            o_g = np.tile(o_g, self.hidden_size // len(o_g) + 1)[: self.hidden_size]
            c_tilde = np.tile(c_tilde, self.hidden_size // len(c_tilde) + 1)[: self.hidden_size]
        else:
            f = f[: self.hidden_size]
            i_g = i_g[: self.hidden_size]
            o_g = o_g[: self.hidden_size]
            c_tilde = c_tilde[: self.hidden_size]

        c_t = f * c_prev + i_g * c_tilde
        h_t = o_g * np.tanh(c_t)
        return h_t, c_t

    def forward(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Forward pass over a full sequence.

        Parameters
        ----------
        X_seq : np.ndarray
            Sequence of shape (T, input_size).

        Returns
        -------
        np.ndarray
            Final hidden state of shape (hidden_size,).
        """
        T = X_seq.shape[0]
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        for t in range(T):
            h, c = self.step(X_seq[t], h, c)
        return h

    # ── VQC helpers ──────────────────────────────────────────────────────────

    def _run_vqc(
        self,
        encoding_angles: np.ndarray,
        ansatz_params: np.ndarray,
        activation: str = "linear",
    ) -> np.ndarray:
        """Run a small VQC and return per-qubit ⟨Z⟩ measurements."""
        qc = QuantumCircuit(self.n_qubits)

        # Encoding: RY of input angles
        for q in range(self.n_qubits):
            qc.ry(float(encoding_angles[q]), q)

        # Variational ansatz: RY + CNOT ladder per layer
        param_idx = 0
        for _ in range(self.n_layers):
            for q in range(self.n_qubits):
                qc.ry(float(ansatz_params[param_idx]), q)
                param_idx += 1
            for q in range(self.n_qubits - 1):
                qc.cnot(q, q + 1)

        # Measure ⟨Z_i⟩ for each qubit (analytical from statevector)
        state = _simulate_statevector(qc)
        z_exps = np.zeros(self.n_qubits)
        for q in range(self.n_qubits):
            z_exps[q] = self._z_expectation(state, q, self.n_qubits)

        if activation == "tanh":
            return np.tanh(z_exps)
        return z_exps

    def _z_expectation(self, state: np.ndarray, q: int, n: int) -> float:
        probs = np.abs(state) ** 2
        z = 0.0
        for idx in range(len(probs)):
            bit = (idx >> q) & 1
            z += probs[idx] * (1 if bit == 0 else -1)
        return float(z)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))


# ═════════════════════════════════════════════════════════════════════════════
# Quantum LSTM regression model
# ═════════════════════════════════════════════════════════════════════════════


class QuantumLSTMRegressor:
    """
    Sequence-to-scalar Q-LSTM regressor.

    Stack: input → QLSTM cell → linear readout → scalar output.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 4,
        n_qubits: int = 4,
        n_layers: int = 2,
    ) -> None:
        self.cell = QuantumLSTMCell(input_size, hidden_size, n_qubits, n_layers)
        self.readout = np.zeros(hidden_size)
        self.bias = 0.0

    def predict(self, X_seq: np.ndarray) -> float:
        """Forward pass; returns a single scalar prediction."""
        h_final = self.cell.forward(X_seq)
        return float(h_final @ self.readout + self.bias)

    def fit(
        self,
        X_seqs: List[np.ndarray],
        y: np.ndarray,
        *,
        epochs: int = 30,
        learning_rate: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Train the Q-LSTM via gradient descent on the linear readout layer.
        VQC parameters are kept fixed (reservoir-style training, à la Q-RC).
        """
        from scipy.optimize import minimize as sp_minimize

        # Compute features for each sequence
        H = np.zeros((len(X_seqs), self.cell.hidden_size))
        for i, x in enumerate(X_seqs):
            H[i] = self.cell.forward(x)

        # Solve readout via ridge regression
        ridge = 1e-3
        A = H.T @ H + ridge * np.eye(self.cell.hidden_size)
        b = H.T @ y
        self.readout = np.linalg.solve(A, b)
        self.bias = float(np.mean(y - H @ self.readout))

        # Training loss
        preds = np.array([self.predict(X_seqs[i]) for i in range(len(X_seqs))])
        mse = float(np.mean((preds - y) ** 2))

        return {
            "final_mse": mse,
            "method": "qlstm_ridge_readout",
            "n_params": self.cell.n_params_per_gate * 4,
        }
