"""
Density matrix quantum simulator.

Exact mixed-state simulation via the full density matrix ρ. Each gate is
applied as ρ → U ρ U†, and Kraus channels act as ρ → Σ K_i ρ K_i†.

Cost: O(d²) memory and O(d³) per gate, where d = 2^n. Feasible up to
n ≈ 10 qubits on typical hardware.

Use cases
---------
- Exact noise simulation with Kraus channels (no stochastic sampling)
- Exact fidelity / purity / entropy calculations
- Verification of stochastic trajectory results

Trading use
-----------
Reference simulator for verifying error-mitigation protocols in ARGUS's
quantum risk engine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from quantum_simulator import GateType, Operation, QuantumCircuit, _cached_matrix


# ═════════════════════════════════════════════════════════════════════════════
# Density matrix simulator
# ═════════════════════════════════════════════════════════════════════════════


class DensityMatrixSimulator:
    """
    Exact mixed-state simulator.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (keep n <= 10 for reasonable memory).
    """

    def __init__(self, n_qubits: int) -> None:
        if n_qubits < 1 or n_qubits > 12:
            raise ValueError(f"n_qubits must be 1..12, got {n_qubits}")
        self.n_qubits = int(n_qubits)
        d = 1 << self.n_qubits
        self.rho = np.zeros((d, d), dtype=np.complex128)
        self.rho[0, 0] = 1.0  # |0...0⟩⟨0...0|

    def reset(self) -> None:
        """Reset to |0...0⟩."""
        d = 1 << self.n_qubits
        self.rho = np.zeros((d, d), dtype=np.complex128)
        self.rho[0, 0] = 1.0

    def apply_unitary(self, U: np.ndarray) -> None:
        """Apply a full-system unitary: ρ → U ρ U†."""
        self.rho = U @ self.rho @ U.conj().T

    def apply_kraus(self, operators: List[np.ndarray]) -> None:
        """
        Apply a Kraus channel: ρ → Σ_i K_i ρ K_i†.

        Each K_i must be the same shape as ρ.
        """
        new_rho = np.zeros_like(self.rho)
        for K in operators:
            new_rho += K @ self.rho @ K.conj().T
        self.rho = new_rho

    def apply_circuit(self, circuit: QuantumCircuit) -> None:
        """Apply every gate in a QuantumCircuit."""
        n = self.n_qubits
        d = 1 << n
        for op in circuit.operations:
            if op.gate == GateType.MEASURE_ALL:
                break
            U = self._build_full_unitary(op, n)
            self.apply_unitary(U)

    def _build_full_unitary(self, op: Operation, n: int) -> np.ndarray:
        """Expand a single gate to an n-qubit unitary."""
        d = 1 << n
        if op.gate == GateType.CNOT:
            # Special CNOT construction
            return self._build_cnot(op.targets[0], op.targets[1], n)

        g = op.gate
        t = op.targets

        # Build local matrix
        _ONE_QUBIT = {
            GateType.H, GateType.X, GateType.Y, GateType.Z,
            GateType.S, GateType.SDG, GateType.T, GateType.TDG,
            GateType.RX, GateType.RY, GateType.RZ, GateType.U3, GateType.PHASE,
        }
        _TWO_QUBIT = {
            GateType.CZ, GateType.SWAP, GateType.ISWAP,
            GateType.CPHASE, GateType.CRX, GateType.CRY, GateType.CRZ,
            GateType.RXX, GateType.RYY, GateType.RZZ,
        }
        _THREE_QUBIT = {GateType.CCX, GateType.CCZ, GateType.CSWAP}

        if g in _ONE_QUBIT:
            U_local = _cached_matrix(g, op.params)
            return self._expand_1q(U_local, t[0], n)
        if g in _TWO_QUBIT:
            U_local = _cached_matrix(g, op.params)
            return self._expand_2q(U_local, t[0], t[1], n)
        if g in _THREE_QUBIT:
            U_local = _cached_matrix(g, op.params)
            return self._expand_3q(U_local, t[0], t[1], t[2], n)

        raise ValueError(f"Unsupported gate in density matrix sim: {g}")

    def _expand_1q(self, U: np.ndarray, q: int, n: int) -> np.ndarray:
        """Expand a 2x2 gate to an n-qubit unitary."""
        I2 = np.eye(2, dtype=np.complex128)
        result = np.array([[1]], dtype=np.complex128)
        for k in range(n):
            # qubit k=0 is LSB, so in the Kronecker product we put highest qubit first
            qubit_index = n - 1 - k
            if qubit_index == q:
                result = np.kron(result, U)
            else:
                result = np.kron(result, I2)
        return result

    def _expand_2q(self, U4: np.ndarray, q1: int, q2: int, n: int) -> np.ndarray:
        """Expand a 4x4 gate on (q1, q2) to an n-qubit unitary."""
        # Use the statevector-expansion trick: reshape identity state,
        # apply the gate via tensor axis permutation, then convert back.
        # For clarity we do full tensor construction.
        d = 1 << n
        result = np.eye(d, dtype=np.complex128)
        shape = (2,) * n
        M = result.reshape(shape + shape)
        # Compute ax positions (qubit 0 = LSB = axis n-1)
        ax1 = n - 1 - q1
        ax2 = n - 1 - q2
        # Build the full unitary by applying U4 to each column of the identity
        result_flat = np.zeros((d, d), dtype=np.complex128)
        U4_reshaped = U4.reshape(2, 2, 2, 2)
        for col in range(d):
            psi = np.zeros(d, dtype=np.complex128)
            psi[col] = 1.0
            tensor = psi.reshape(shape)
            perm = [ax1, ax2] + [a for a in range(n) if a not in (ax1, ax2)]
            inv_perm = [0] * n
            for i, p in enumerate(perm):
                inv_perm[p] = i
            tensor = np.transpose(tensor, axes=perm)
            rest = 1 << (n - 2)
            mat = tensor.reshape(4, rest)
            new_mat = U4 @ mat
            new_tensor = new_mat.reshape(2, 2, *([2] * (n - 2)))
            new_tensor = np.transpose(new_tensor, axes=inv_perm)
            result_flat[:, col] = new_tensor.reshape(d)
        return result_flat

    def _expand_3q(self, U8: np.ndarray, q1: int, q2: int, q3: int, n: int) -> np.ndarray:
        d = 1 << n
        shape = (2,) * n
        ax1, ax2, ax3 = n - 1 - q1, n - 1 - q2, n - 1 - q3
        result_flat = np.zeros((d, d), dtype=np.complex128)
        for col in range(d):
            psi = np.zeros(d, dtype=np.complex128)
            psi[col] = 1.0
            tensor = psi.reshape(shape)
            perm = [ax1, ax2, ax3] + [a for a in range(n) if a not in (ax1, ax2, ax3)]
            inv_perm = [0] * n
            for i, p in enumerate(perm):
                inv_perm[p] = i
            tensor = np.transpose(tensor, axes=perm)
            rest = 1 << (n - 3)
            mat = tensor.reshape(8, rest)
            new_mat = U8 @ mat
            new_tensor = new_mat.reshape(2, 2, 2, *([2] * (n - 3)))
            new_tensor = np.transpose(new_tensor, axes=inv_perm)
            result_flat[:, col] = new_tensor.reshape(d)
        return result_flat

    def _build_cnot(self, control: int, target: int, n: int) -> np.ndarray:
        """Build the full CNOT unitary as a dense matrix."""
        d = 1 << n
        U = np.eye(d, dtype=np.complex128)
        c_mask = 1 << control
        t_mask = 1 << target
        for idx in range(d):
            if idx & c_mask:
                j = idx ^ t_mask
                if idx < j:
                    U[idx, idx] = 0
                    U[j, j] = 0
                    U[idx, j] = 1
                    U[j, idx] = 1
        return U

    # ── Metrics ──────────────────────────────────────────────────────────────

    def purity(self) -> float:
        """Tr(ρ²) — 1 for pure states, 1/d for maximally mixed."""
        return float(np.real(np.trace(self.rho @ self.rho)))

    def von_neumann_entropy(self) -> float:
        """-Tr(ρ log ρ) in bits."""
        eigvals = np.linalg.eigvalsh(self.rho)
        eigvals = eigvals[eigvals > 1e-12]
        if len(eigvals) == 0:
            return 0.0
        return float(-np.sum(eigvals * np.log2(eigvals)))

    def fidelity_with_state(self, state: np.ndarray) -> float:
        """F(ρ, |ψ⟩) = ⟨ψ|ρ|ψ⟩ for pure target."""
        psi = np.asarray(state, dtype=np.complex128).ravel()
        return float(np.real(np.conj(psi) @ self.rho @ psi))

    def measurement_probs(self) -> np.ndarray:
        """Computational-basis measurement probabilities."""
        return np.real(np.diag(self.rho)).astype(float)
