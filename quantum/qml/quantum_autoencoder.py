"""
Quantum Auto-Encoder (QAE).

A quantum auto-encoder compresses an n-qubit state into a smaller latent
space of k < n qubits, discarding the remaining n - k "trash" qubits. The
encoder is a parameterized circuit U(θ) trained so that, after encoding,
the trash qubits are in |0⟩ (a known reference state).

Training minimizes the overlap of the trash register with the reference:
    L(θ) = 1 - ⟨0|_trash Tr_latent[U(θ) ρ U†(θ)] |0⟩_trash

Reference
---------
Romero, Olson, Aspuru-Guzik, "Quantum autoencoders for efficient
compression of quantum data," Quantum Sci. Technol. 2, 045001 (2017)

Trading use
-----------
Compress market-state many-body representations (QPCA complement) or
compress high-dim features into a low-dim quantum latent space for
downstream VQC classifiers.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from quantum_simulator import QuantumCircuit, _simulate_statevector

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Quantum Auto-Encoder
# ═════════════════════════════════════════════════════════════════════════════


class QuantumAutoEncoder:
    """
    Parameterized quantum auto-encoder.

    Parameters
    ----------
    n_qubits : int
        Total number of input qubits.
    n_latent : int
        Size of the latent register (must be < n_qubits).
    n_layers : int
        Depth of the encoder ansatz.
    """

    def __init__(
        self,
        n_qubits: int,
        n_latent: int,
        n_layers: int = 3,
    ) -> None:
        if n_latent >= n_qubits:
            raise ValueError("n_latent must be strictly less than n_qubits")
        self.n_qubits = int(n_qubits)
        self.n_latent = int(n_latent)
        self.n_trash = n_qubits - n_latent
        self.n_layers = int(n_layers)
        self.n_params = 3 * n_qubits * n_layers  # RX, RY, RZ per qubit per layer
        rng = np.random.default_rng(42)
        self.params = rng.uniform(-np.pi, np.pi, self.n_params)

    def encode_circuit(
        self,
        state_prep: Any,
        params: Optional[np.ndarray] = None,
    ) -> QuantumCircuit:
        """
        Build the full encode circuit: state preparation + encoder ansatz.
        """
        if params is None:
            params = self.params
        qc = QuantumCircuit(self.n_qubits)
        # State prep (either a callable or an existing circuit)
        if callable(state_prep):
            state_prep(qc)
        elif isinstance(state_prep, QuantumCircuit):
            for op in state_prep.operations:
                qc._ops.append(op)
        # Encoder ansatz
        idx = 0
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                qc.rx(float(params[idx]), q)
                idx += 1
                qc.ry(float(params[idx]), q)
                idx += 1
                qc.rz(float(params[idx]), q)
                idx += 1
            # Entangling layer
            for q in range(self.n_qubits - 1):
                qc.cnot(q, q + 1)
        return qc

    def cost(
        self,
        state_prep: Any,
        params: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute the cost L(θ) = 1 - ⟨0|_trash ρ_trash |0⟩_trash.

        Trash qubits are assumed to be the last n_trash qubits.
        """
        if params is None:
            params = self.params
        qc = self.encode_circuit(state_prep, params)
        state = _simulate_statevector(qc)

        # Compute the reduced density matrix on the trash qubits, then
        # evaluate ⟨0|_trash ρ_trash |0⟩_trash.
        n = self.n_qubits
        n_trash = self.n_trash
        n_latent = self.n_latent

        # P(trash = 0...0) = sum_latent |state[latent ⊗ 0_trash]|²
        # The trash qubits are the last n_trash qubits (in our convention,
        # qubit n-1 is the MSB). So indices where the top n_trash bits are 0.
        trash_mask = ((1 << n_trash) - 1) << n_latent  # bits n_latent..n-1
        p_zero = 0.0
        for idx in range(1 << n):
            if (idx & trash_mask) == 0:
                p_zero += float(np.abs(state[idx]) ** 2)

        return 1.0 - float(p_zero)

    def fit(
        self,
        state_prep: Any,
        *,
        n_iter: int = 100,
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """
        Train the auto-encoder to compress ``state_prep`` into the latent
        register.
        """
        from scipy.optimize import minimize as sp_minimize

        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        history: List[float] = []

        def cost_fn(params):
            c = self.cost(state_prep, params)
            history.append(c)
            return c

        best_cost = float("inf")
        best_params: Optional[np.ndarray] = None

        for trial in range(3):
            x0 = rng.uniform(-np.pi, np.pi, self.n_params)
            opt = sp_minimize(
                cost_fn,
                x0,
                method="COBYLA",
                options={"maxiter": n_iter, "rhobeg": 0.3},
            )
            if opt.fun < best_cost:
                best_cost = float(opt.fun)
                best_params = np.asarray(opt.x, dtype=float)

        if best_params is None:
            best_params = np.zeros(self.n_params)
        self.params = best_params

        return {
            "final_cost": best_cost,
            "compression_fidelity": 1.0 - best_cost,
            "history": history,
            "n_params": self.n_params,
            "method": "quantum_autoencoder",
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }
