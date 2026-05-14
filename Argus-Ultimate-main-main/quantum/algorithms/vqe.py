"""
Variational Quantum Eigensolver (VQE) on the ARGUS in-repo simulator.

VQE finds the lowest eigenvalue (ground state energy) of a Hamiltonian H by
minimizing ⟨ψ(θ)|H|ψ(θ)⟩ over a parameterized ansatz |ψ(θ)⟩. It is the
canonical quantum algorithm for problems that reduce to finding ground states:

- Ising Hamiltonians (combinatorial optimization, MaxCut, portfolio risk modes)
- Quantum chemistry (molecular ground states)
- Spin models, lattice problems

This implementation:
- Uses a hardware-efficient ansatz (RY + CNOT-ring layers)
- Routes through ``quantum_simulator`` (no external dependencies)
- COBYLA outer loop with multiple random restarts
- Returns ground energy, ground-state bitstring estimate, convergence history

Honest note: on a classical simulator, VQE has no quantum advantage over
exact diagonalization for small problems. Its value here is architectural
correctness — the same code path will run on real quantum hardware.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from quantum_simulator import QuantumCircuit, simulate

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# VQE solver
# ═════════════════════════════════════════════════════════════════════════════


class VQESolver:
    """
    Variational Quantum Eigensolver for Ising and Pauli-string Hamiltonians.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system.
    n_layers : int
        Number of layers in the variational ansatz.
    ansatz : str
        Ansatz type. Currently supported: ``"hardware_efficient"`` (RY + CNOT
        ring), ``"ry_linear"`` (RY + nearest-neighbor CNOT chain).
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 3,
        ansatz: str = "hardware_efficient",
    ) -> None:
        if n_qubits < 1 or n_qubits > 14:
            raise ValueError(f"n_qubits must be in [1, 14], got {n_qubits}")
        self.n_qubits = int(n_qubits)
        self.n_layers = max(1, int(n_layers))
        if ansatz not in ("hardware_efficient", "ry_linear"):
            raise ValueError(f"Unknown ansatz: {ansatz}")
        self.ansatz = ansatz

    # ── Public API ───────────────────────────────────────────────────────────

    def solve_ising(
        self,
        h: np.ndarray,
        J: np.ndarray,
        *,
        max_iter: int = 200,
        shots: int = 4096,
        n_restarts: int = 3,
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """
        Find the ground state of a transverse-field Ising Hamiltonian:

            H = Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j

        Parameters
        ----------
        h : np.ndarray
            Local field vector, shape ``(n_qubits,)``.
        J : np.ndarray
            Coupling matrix, shape ``(n_qubits, n_qubits)``. Only the upper
            triangle is used (symmetrized internally).

        Returns
        -------
        Dict[str, Any]
            ``{"ground_energy", "ground_state_bits", "convergence_history",
              "optimal_params", "method", "n_iterations", "elapsed_ms"}``
        """
        h = np.asarray(h, dtype=float).ravel()
        J = np.asarray(J, dtype=float)
        n = self.n_qubits
        if h.shape != (n,):
            raise ValueError(f"h shape {h.shape} != ({n},)")
        if J.shape != (n, n):
            raise ValueError(f"J shape {J.shape} != ({n}, {n})")
        # Collapse to upper-triangular: J_eff[i,j] = J[i,j] + J[j,i] for i<j.
        # This way the user can pass either upper-triangular or symmetric and
        # the energy formula sum_{i<j} J_eff[i,j] z_i z_j is correct in both cases.
        J_eff = np.zeros_like(J)
        for i in range(n):
            for j in range(i + 1, n):
                J_eff[i, j] = J[i, j] + J[j, i]

        return self._optimize_ising(h, J_eff, max_iter=max_iter, shots=shots,
                                    n_restarts=n_restarts, seed=seed)

    def solve_hamiltonian(
        self,
        pauli_terms: List[Tuple[str, float]],
        *,
        max_iter: int = 200,
        shots: int = 4096,
        n_restarts: int = 3,
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """
        Find the ground state of a generic Pauli-string Hamiltonian.

        Parameters
        ----------
        pauli_terms : List[Tuple[str, float]]
            List of (pauli_string, coefficient) pairs. Each ``pauli_string``
            is a string of ``"I"``, ``"X"``, ``"Y"``, ``"Z"`` of length
            n_qubits, with character 0 = qubit 0 (LSB).
            E.g., ``"ZZI"`` is Z⊗Z⊗I acting on qubits (0, 1, 2 respectively
            with qubit 2 first in string).

        Returns
        -------
        Same schema as ``solve_ising``.
        """
        for pauli_str, _ in pauli_terms:
            if len(pauli_str) != self.n_qubits:
                raise ValueError(
                    f"Pauli string '{pauli_str}' length {len(pauli_str)} "
                    f"!= n_qubits={self.n_qubits}"
                )
            for ch in pauli_str:
                if ch not in "IXYZ":
                    raise ValueError(f"Invalid Pauli char '{ch}' in '{pauli_str}'")

        return self._optimize_pauli(pauli_terms, max_iter=max_iter, shots=shots,
                                    n_restarts=n_restarts, seed=seed)

    # ── Ansatz ────────────────────────────────────────────────────────────────

    def n_params(self) -> int:
        """Number of variational parameters in the chosen ansatz."""
        if self.ansatz == "hardware_efficient":
            return self.n_qubits * self.n_layers
        elif self.ansatz == "ry_linear":
            return self.n_qubits * self.n_layers
        return 0

    def build_ansatz(self, params: np.ndarray) -> QuantumCircuit:
        """
        Build the variational ansatz circuit for the given parameter vector.
        """
        n = self.n_qubits
        qc = QuantumCircuit(n)

        if self.ansatz == "hardware_efficient":
            # Each layer: RY(θ_i) on every qubit, then CNOT ring
            idx = 0
            for layer in range(self.n_layers):
                for i in range(n):
                    qc.ry(float(params[idx]), i)
                    idx += 1
                # CNOT ring
                for i in range(n):
                    qc.cnot(i, (i + 1) % n)
        elif self.ansatz == "ry_linear":
            idx = 0
            for layer in range(self.n_layers):
                for i in range(n):
                    qc.ry(float(params[idx]), i)
                    idx += 1
                for i in range(n - 1):
                    qc.cnot(i, i + 1)

        return qc

    # ── Energy evaluation ────────────────────────────────────────────────────

    def _ising_energy(
        self,
        counts: Dict[str, int],
        h: np.ndarray,
        J: np.ndarray,
    ) -> float:
        """
        Compute the Ising energy from a measurement distribution.

            E = ⟨H⟩ = Σ counts[bitstring] · E(bitstring) / total_shots
            E(bits) = Σ_i h_i · z_i + Σ_{i<j} J_ij · z_i z_j

        where z_i = +1 if bit_i = 0, -1 if bit_i = 1.
        """
        n = self.n_qubits
        total = sum(counts.values())
        if total == 0:
            return 0.0
        acc = 0.0
        for bitstring, c in counts.items():
            # Convert to +1/-1 spins (qubit 0 = rightmost char)
            z = np.empty(n, dtype=float)
            for q in range(n):
                bit = bitstring[len(bitstring) - 1 - q] if q < len(bitstring) else "0"
                z[q] = 1.0 if bit == "0" else -1.0
            energy = float(np.sum(h * z))
            for i in range(n):
                for j in range(i + 1, n):
                    energy += J[i, j] * z[i] * z[j]
            acc += c * energy
        return acc / total

    def _pauli_energy(
        self,
        params: np.ndarray,
        pauli_terms: List[Tuple[str, float]],
        shots: int,
        seed: Optional[int],
    ) -> float:
        """
        Compute ⟨H⟩ for a Pauli-string Hamiltonian by measuring each Pauli
        string in its appropriate basis (basis-rotation trick).

        For each non-trivial Pauli term, append basis-rotation gates that map
        X → Z (H gate) or Y → Z (S† H), then measure, then compute the
        product of relevant qubits' z-eigenvalues weighted by the coefficient.
        """
        n = self.n_qubits
        total_energy = 0.0
        rng = np.random.default_rng(seed)

        for pauli_str, coeff in pauli_terms:
            if all(ch == "I" for ch in pauli_str):
                # Identity term contributes coeff directly
                total_energy += float(coeff)
                continue

            # Build a circuit with the ansatz + basis rotation
            qc = self.build_ansatz(params)
            for q in range(n):
                # Character at position (n - 1 - q) corresponds to qubit q? No —
                # for ergonomics, let position 0 of the pauli string be qubit 0.
                ch = pauli_str[q] if q < len(pauli_str) else "I"
                if ch == "X":
                    qc.h(q)
                elif ch == "Y":
                    qc.sdg(q)
                    qc.h(q)
                # Z and I require no rotation
            qc.measure_all()

            res = simulate(qc, shots=shots, seed=int(rng.integers(0, 2**31 - 1)))
            counts = res["counts"]

            # Compute expectation: product of (+1 for bit=0, -1 for bit=1) on
            # qubits where pauli_str[q] != 'I'
            total_shots = sum(counts.values())
            if total_shots == 0:
                continue
            expectation = 0.0
            for bitstring, c in counts.items():
                product = 1.0
                for q in range(n):
                    ch = pauli_str[q] if q < len(pauli_str) else "I"
                    if ch == "I":
                        continue
                    bit = (
                        bitstring[len(bitstring) - 1 - q]
                        if q < len(bitstring)
                        else "0"
                    )
                    sign = 1.0 if bit == "0" else -1.0
                    product *= sign
                expectation += c * product
            expectation /= total_shots
            total_energy += float(coeff) * expectation

        return total_energy

    # ── Optimization loops ───────────────────────────────────────────────────

    def _optimize_ising(
        self,
        h: np.ndarray,
        J: np.ndarray,
        *,
        max_iter: int,
        shots: int,
        n_restarts: int,
        seed: Optional[int],
    ) -> Dict[str, Any]:
        from scipy.optimize import minimize as sp_minimize

        t0 = time.perf_counter()
        n_params = self.n_params()
        rng = np.random.default_rng(seed)
        convergence: List[float] = []

        def cost_fn(params: np.ndarray) -> float:
            qc = self.build_ansatz(params)
            qc.measure_all()
            res = simulate(qc, shots=shots, seed=int(rng.integers(0, 2**31 - 1)))
            e = self._ising_energy(res["counts"], h, J)
            convergence.append(e)
            return e

        best_energy = float("inf")
        best_params: Optional[np.ndarray] = None

        for trial in range(n_restarts):
            x0 = rng.uniform(-np.pi, np.pi, n_params)
            opt = sp_minimize(
                cost_fn,
                x0,
                method="COBYLA",
                options={"maxiter": max_iter, "rhobeg": 0.5},
            )
            if opt.fun < best_energy:
                best_energy = float(opt.fun)
                best_params = np.asarray(opt.x, dtype=float)

        if best_params is None:
            best_params = np.zeros(n_params)

        # Final shot to find the most likely ground-state bitstring
        final_qc = self.build_ansatz(best_params)
        final_qc.measure_all()
        final_res = simulate(final_qc, shots=8192, seed=7)
        top_bitstring = max(final_res["counts"].items(), key=lambda kv: kv[1])[0]
        ground_state_bits = [
            int(top_bitstring[len(top_bitstring) - 1 - q]) if q < len(top_bitstring) else 0
            for q in range(self.n_qubits)
        ]

        # Energy of the discovered (most-likely) bitstring — this is the
        # variational lower bound concretely realized.
        z_top = np.array(
            [1.0 if b == 0 else -1.0 for b in ground_state_bits],
            dtype=float,
        )
        top_energy = float(np.sum(h * z_top))
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                top_energy += J[i, j] * z_top[i] * z_top[j]

        # Use the better of the two (most-likely-bitstring is exact for that bitstring,
        # variational best may be lower if convergence reached)
        reported_energy = min(best_energy, top_energy)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "ground_energy": float(reported_energy),
            "variational_energy": float(best_energy),
            "top_bitstring_energy": float(top_energy),
            "ground_state_bits": ground_state_bits,
            "convergence_history": convergence[-50:],
            "optimal_params": best_params.tolist(),
            "method": "vqe_in_repo_simulator",
            "n_iterations": len(convergence),
            "elapsed_ms": round(elapsed_ms, 2),
        }

    def _optimize_pauli(
        self,
        pauli_terms: List[Tuple[str, float]],
        *,
        max_iter: int,
        shots: int,
        n_restarts: int,
        seed: Optional[int],
    ) -> Dict[str, Any]:
        from scipy.optimize import minimize as sp_minimize

        t0 = time.perf_counter()
        n_params = self.n_params()
        rng = np.random.default_rng(seed)
        convergence: List[float] = []

        def cost_fn(params: np.ndarray) -> float:
            e = self._pauli_energy(params, pauli_terms, shots, seed=None)
            convergence.append(e)
            return e

        best_energy = float("inf")
        best_params: Optional[np.ndarray] = None

        for trial in range(n_restarts):
            x0 = rng.uniform(-np.pi, np.pi, n_params)
            opt = sp_minimize(
                cost_fn,
                x0,
                method="COBYLA",
                options={"maxiter": max_iter, "rhobeg": 0.5},
            )
            if opt.fun < best_energy:
                best_energy = float(opt.fun)
                best_params = np.asarray(opt.x, dtype=float)

        if best_params is None:
            best_params = np.zeros(n_params)

        final_qc = self.build_ansatz(best_params)
        final_qc.measure_all()
        final_res = simulate(final_qc, shots=8192, seed=7)
        top_bitstring = max(final_res["counts"].items(), key=lambda kv: kv[1])[0]
        ground_state_bits = [
            int(top_bitstring[len(top_bitstring) - 1 - q]) if q < len(top_bitstring) else 0
            for q in range(self.n_qubits)
        ]

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "ground_energy": float(best_energy),
            "ground_state_bits": ground_state_bits,
            "convergence_history": convergence[-50:],
            "optimal_params": best_params.tolist(),
            "method": "vqe_in_repo_simulator",
            "n_iterations": len(convergence),
            "elapsed_ms": round(elapsed_ms, 2),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Convenience: exact diagonalization for small Hamiltonians (testing)
# ═════════════════════════════════════════════════════════════════════════════


def exact_ising_ground_energy(h: np.ndarray, J: np.ndarray) -> Tuple[float, List[int]]:
    """
    Compute the exact ground energy and bitstring of an Ising Hamiltonian
    via brute-force enumeration. Used for unit-test reference.

    Returns
    -------
    Tuple[float, List[int]]
        (ground_energy, ground_state_bits)
    """
    h = np.asarray(h, dtype=float).ravel()
    J = np.asarray(J, dtype=float)
    n = len(h)
    best_e = float("inf")
    best_bits: List[int] = [0] * n
    for x in range(1 << n):
        z = np.empty(n, dtype=float)
        for q in range(n):
            z[q] = 1.0 if not ((x >> q) & 1) else -1.0
        e = float(np.sum(h * z))
        for i in range(n):
            for j in range(i + 1, n):
                e += J[i, j] * z[i] * z[j]
        if e < best_e:
            best_e = e
            best_bits = [int((x >> q) & 1) for q in range(n)]
    return best_e, best_bits
