"""
Quantum Transformer with quantum self-attention.

Implements a single-head quantum self-attention layer where the
query/key/value projections and the attention scores are computed via
parameterized quantum circuits.

Reference
---------
Cherrat et al., "Quantum Vision Transformers,"
arXiv:2209.08167 (2022)

Architecture
------------
For each input token x_t (a feature vector):
    Q_t = VQC_Q(x_t)
    K_t = VQC_K(x_t)
    V_t = VQC_V(x_t)
Then attention scores α_{t,s} = softmax(Q_t · K_s / √d_k) computed classically.
The output is Σ_s α_{t,s} V_s, projected via a final linear layer.

Trading use
-----------
Sequence-aware feature extractor for return time series. Wire as alternative
to classical Transformer in ml/ for regime-aware predictions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from quantum_simulator import QuantumCircuit, _simulate_statevector

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Quantum self-attention layer
# ═════════════════════════════════════════════════════════════════════════════


class QuantumSelfAttention:
    """
    Single-head quantum self-attention.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    n_qubits : int
        Qubit count for each Q/K/V circuit.
    n_layers : int
        Variational ansatz depth per circuit.
    """

    def __init__(
        self,
        d_model: int,
        n_qubits: int = 4,
        n_layers: int = 2,
    ) -> None:
        self.d_model = int(d_model)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        rng = np.random.default_rng(42)
        n_params = n_qubits * n_layers
        self.theta_Q = rng.uniform(-0.5, 0.5, n_params)
        self.theta_K = rng.uniform(-0.5, 0.5, n_params)
        self.theta_V = rng.uniform(-0.5, 0.5, n_params)
        # Linear projections from d_model → n_qubits
        self.W_in = rng.normal(0, 0.1, (n_qubits, d_model))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Apply quantum self-attention to a sequence.

        Parameters
        ----------
        X : np.ndarray
            Sequence of shape (T, d_model).

        Returns
        -------
        np.ndarray
            Output sequence of shape (T, n_qubits).
        """
        T = X.shape[0]
        Q = np.zeros((T, self.n_qubits))
        K = np.zeros((T, self.n_qubits))
        V = np.zeros((T, self.n_qubits))

        # Compute Q, K, V via VQCs
        for t in range(T):
            angles = self.W_in @ X[t]
            angles = np.clip(angles, -np.pi, np.pi)
            Q[t] = self._run_vqc(angles, self.theta_Q)
            K[t] = self._run_vqc(angles, self.theta_K)
            V[t] = self._run_vqc(angles, self.theta_V)

        # Classical attention: scores (T, T)
        scores = Q @ K.T / np.sqrt(self.n_qubits)
        # Softmax
        scores = scores - scores.max(axis=1, keepdims=True)
        attn = np.exp(scores)
        attn = attn / attn.sum(axis=1, keepdims=True)

        # Weighted sum of V
        out = attn @ V
        return out

    def _run_vqc(
        self,
        encoding_angles: np.ndarray,
        ansatz_params: np.ndarray,
    ) -> np.ndarray:
        """Run a VQC and return per-qubit ⟨Z⟩ expectations."""
        qc = QuantumCircuit(self.n_qubits)
        # Encoding
        for q in range(self.n_qubits):
            qc.ry(float(encoding_angles[q]), q)
        # Variational ansatz
        idx = 0
        for _ in range(self.n_layers):
            for q in range(self.n_qubits):
                qc.ry(float(ansatz_params[idx]), q)
                idx += 1
            for q in range(self.n_qubits - 1):
                qc.cnot(q, q + 1)

        state = _simulate_statevector(qc)
        z_exps = np.zeros(self.n_qubits)
        for q in range(self.n_qubits):
            probs = np.abs(state) ** 2
            z_exps[q] = sum(
                probs[i] * (1 if (i >> q) & 1 == 0 else -1)
                for i in range(len(probs))
            )
        return z_exps


# ═════════════════════════════════════════════════════════════════════════════
# Quantum Transformer block
# ═════════════════════════════════════════════════════════════════════════════


class QuantumTransformerBlock:
    """
    A single transformer block: quantum self-attention + classical
    feed-forward + residual connection.
    """

    def __init__(
        self,
        d_model: int,
        n_qubits: int = 4,
        n_attention_layers: int = 2,
    ) -> None:
        self.d_model = int(d_model)
        self.attention = QuantumSelfAttention(d_model, n_qubits, n_attention_layers)
        rng = np.random.default_rng(42)
        # Project attention output back to d_model
        self.W_out = rng.normal(0, 0.1, (d_model, n_qubits))
        # Feed-forward
        self.W_ff1 = rng.normal(0, 0.1, (d_model * 2, d_model))
        self.W_ff2 = rng.normal(0, 0.1, (d_model, d_model * 2))

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Quantum attention
        attn_out = self.attention.forward(X)  # (T, n_qubits)
        attn_proj = attn_out @ self.W_out.T  # (T, d_model)
        # Residual
        x = X + attn_proj
        # Feed-forward
        ff = np.tanh(x @ self.W_ff1.T)
        ff = ff @ self.W_ff2.T
        # Residual
        return x + ff


# ═════════════════════════════════════════════════════════════════════════════
# Quantum Transformer time-series predictor
# ═════════════════════════════════════════════════════════════════════════════


class QuantumTransformerPredictor:
    """
    Time-series predictor using a quantum transformer + classical readout.
    """

    def __init__(
        self,
        input_dim: int,
        n_blocks: int = 2,
        n_qubits: int = 4,
    ) -> None:
        self.input_dim = int(input_dim)
        self.blocks = [
            QuantumTransformerBlock(input_dim, n_qubits=n_qubits)
            for _ in range(n_blocks)
        ]
        self.readout = np.zeros(input_dim)
        self.bias = 0.0

    def predict(self, X_seq: np.ndarray) -> float:
        """X_seq: (T, input_dim) → scalar prediction."""
        x = X_seq.copy()
        for block in self.blocks:
            x = block.forward(x)
        # Pool: average across time, then linear readout
        pooled = x.mean(axis=0)
        return float(pooled @ self.readout + self.bias)

    def fit(
        self,
        X_seqs: List[np.ndarray],
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Fit the readout layer via ridge regression (frozen quantum layers)."""
        H = np.zeros((len(X_seqs), self.input_dim))
        for i, X in enumerate(X_seqs):
            x = X.copy()
            for block in self.blocks:
                x = block.forward(x)
            H[i] = x.mean(axis=0)
        ridge = 1e-3
        A = H.T @ H + ridge * np.eye(self.input_dim)
        b = H.T @ y
        self.readout = np.linalg.solve(A, b)
        self.bias = float(np.mean(y - H @ self.readout))
        preds = np.array([self.predict(X_seqs[i]) for i in range(len(X_seqs))])
        return {
            "final_mse": float(np.mean((preds - y) ** 2)),
            "method": "quantum_transformer_ridge",
        }
