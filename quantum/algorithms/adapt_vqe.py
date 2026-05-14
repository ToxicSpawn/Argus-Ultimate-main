"""
ADAPT-VQE: Adaptive Derivative-Assembled Pseudo-Trotter Variational Quantum Eigensolver.

ADAPT-VQE constructs the variational ansatz operator-by-operator from a
predefined pool of trial operators (Grimsley et al. 2019, Nature Comms).
At each layer, it picks the operator whose gradient is largest in absolute
value and adds it (Trotterized) to the circuit. Stops when no pool operator
has a gradient above the threshold.

This produces problem-specific shallow circuits — typically much shorter
than fixed hardware-efficient ansatzes for the same accuracy.

Reference
---------
Grimsley, Economou, Barnes, Mayhall, "An adaptive variational algorithm for
exact molecular simulations on a quantum computer," Nat. Commun. 10, 3007 (2019)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from quantum_simulator import QuantumCircuit, gradient, simulate

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Operator pool
# ═════════════════════════════════════════════════════════════════════════════


def build_qubit_excitation_pool(n_qubits: int) -> List[Dict[str, Any]]:
    """
    Build a pool of qubit-excitation operators on n qubits.

    Pool contains:
    - Single-qubit Pauli rotations: RY(θ) on each qubit
    - Two-qubit excitations: RXX, RYY, RZZ between adjacent qubits
    - Two-qubit excitations between every pair (long-range, more expressive)

    Each entry is a dict with:
        - "name": operator label
        - "apply": Callable(qc, theta) → None (appends Trotterized op to qc)
        - "qubits": tuple of involved qubit indices
    """
    pool: List[Dict[str, Any]] = []

    # 1q rotations
    for q in range(n_qubits):
        pool.append({
            "name": f"RY({q})",
            "apply": (lambda q_idx: lambda qc, t: qc.ry(t, q_idx))(q),
            "qubits": (q,),
        })

    # 2q nearest-neighbour excitations
    for i in range(n_qubits - 1):
        for gate_name, gate_fn in [
            ("RXX", lambda qc, t, q1, q2: qc.rxx(t, q1, q2)),
            ("RYY", lambda qc, t, q1, q2: qc.ryy(t, q1, q2)),
            ("RZZ", lambda qc, t, q1, q2: qc.rzz(t, q1, q2)),
        ]:
            j = i + 1
            pool.append({
                "name": f"{gate_name}({i},{j})",
                "apply": (lambda gn, q1, q2: lambda qc, t: gn(qc, t, q1, q2))(gate_fn, i, j),
                "qubits": (i, j),
            })

    return pool


# ═════════════════════════════════════════════════════════════════════════════
# ADAPT-VQE solver
# ═════════════════════════════════════════════════════════════════════════════


class AdaptVQE:
    """
    Adaptive VQE that grows the ansatz operator-by-operator from a pool.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system.
    operator_pool : str or List[Dict]
        Either a string pool name (currently only "qubit_excitations") or a
        custom operator pool built via ``build_qubit_excitation_pool``.
    """

    def __init__(
        self,
        n_qubits: int,
        operator_pool: Any = "qubit_excitations",
    ) -> None:
        if n_qubits < 1 or n_qubits > 10:
            raise ValueError(f"n_qubits must be in [1, 10], got {n_qubits}")
        self.n_qubits = int(n_qubits)
        if isinstance(operator_pool, str):
            if operator_pool == "qubit_excitations":
                self.pool = build_qubit_excitation_pool(self.n_qubits)
            else:
                raise ValueError(f"Unknown pool name: {operator_pool}")
        else:
            self.pool = list(operator_pool)

    def solve(
        self,
        hamiltonian: List[Tuple[str, float]],
        *,
        max_layers: int = 10,
        gradient_tol: float = 1e-3,
        shots: int = 2048,
        max_inner_iter: int = 100,
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """
        Find the ground state of a Pauli-string Hamiltonian via ADAPT-VQE.

        Parameters
        ----------
        hamiltonian : List[Tuple[str, float]]
            Pauli terms as (string, coefficient) pairs.
        max_layers : int
            Maximum number of pool operators to add.
        gradient_tol : float
            Stop if no pool operator has gradient larger than this.

        Returns
        -------
        Dict[str, Any]
            ``{"ground_energy", "selected_operators", "params", "n_layers",
              "convergence", "method", "elapsed_ms"}``
        """
        from scipy.optimize import minimize as sp_minimize

        t0 = time.perf_counter()

        # Validate Hamiltonian
        for ps, _ in hamiltonian:
            if len(ps) != self.n_qubits:
                raise ValueError(
                    f"Pauli string '{ps}' length != n_qubits={self.n_qubits}"
                )

        selected_ops: List[Dict[str, Any]] = []
        params: List[float] = []
        convergence: List[float] = []

        rng = np.random.default_rng(seed)

        last_added_name: Optional[str] = None
        for layer in range(max_layers):
            # Score each pool operator by |gradient at current state|
            scores = self._score_pool_gradients(
                selected_ops, params, hamiltonian, shots=shots, seed=seed,
            )
            # Sort by descending |gradient|, prefer ops we haven't just added
            order = np.argsort(np.abs(scores))[::-1]
            best_idx = int(order[0])
            # Prevent immediate duplicate (encourages ansatz diversity)
            if (
                last_added_name is not None
                and self.pool[best_idx]["name"] == last_added_name
                and len(order) > 1
            ):
                best_idx = int(order[1])
            best_grad = float(scores[best_idx])
            best_op = self.pool[best_idx]

            if abs(best_grad) < gradient_tol:
                logger.info(
                    "ADAPT-VQE converged at layer %d (best grad %.5f < tol %.5f)",
                    layer, best_grad, gradient_tol,
                )
                break

            # Add the best operator with a small initial parameter
            selected_ops.append(best_op)
            params.append(0.0)
            last_added_name = best_op["name"]

            # Re-optimize ALL parameters with COBYLA
            def cost_fn(p_vec):
                return self._compute_energy(selected_ops, p_vec, hamiltonian, shots)

            opt = sp_minimize(
                cost_fn,
                np.array(params, dtype=float),
                method="COBYLA",
                options={"maxiter": max_inner_iter, "rhobeg": 0.3},
            )
            params = list(opt.x)
            energy = float(opt.fun)
            convergence.append(energy)
            logger.debug("ADAPT-VQE layer %d: added %s, energy %.5f",
                         layer + 1, best_op["name"], energy)

        # Final energy
        if selected_ops:
            final_energy = self._compute_energy(
                selected_ops, np.array(params), hamiltonian, shots * 2,
            )
        else:
            final_energy = self._compute_energy([], np.array([]), hamiltonian, shots)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "ground_energy": float(final_energy),
            "selected_operators": [op["name"] for op in selected_ops],
            "params": list(params),
            "n_layers": len(selected_ops),
            "convergence": convergence,
            "method": "adapt_vqe",
            "elapsed_ms": elapsed_ms,
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def build_circuit(
        self,
        selected_ops: List[Dict[str, Any]],
        params: np.ndarray,
    ) -> QuantumCircuit:
        """Build the current ansatz circuit from the selected operators."""
        qc = QuantumCircuit(self.n_qubits)
        # Reference state: |0...0⟩
        for op, theta in zip(selected_ops, params):
            op["apply"](qc, float(theta))
        return qc

    def _compute_energy(
        self,
        selected_ops: List[Dict[str, Any]],
        params: np.ndarray,
        hamiltonian: List[Tuple[str, float]],
        shots: int,
    ) -> float:
        """Compute ⟨H⟩ for the current ansatz + parameters."""
        qc = self.build_circuit(selected_ops, params)
        return self._pauli_energy(qc, hamiltonian, shots)

    def _pauli_energy(
        self,
        qc: QuantumCircuit,
        hamiltonian: List[Tuple[str, float]],
        shots: int,
    ) -> float:
        """Compute ⟨H⟩ for a Pauli-string Hamiltonian via basis rotation."""
        n = self.n_qubits
        total = 0.0
        for pauli_str, coeff in hamiltonian:
            if all(ch == "I" for ch in pauli_str):
                total += float(coeff)
                continue
            # Build a rotated copy of qc
            rotated = QuantumCircuit(n)
            for op in qc.operations:
                rotated._ops.append(op)
            for q in range(n):
                ch = pauli_str[q] if q < len(pauli_str) else "I"
                if ch == "X":
                    rotated.h(q)
                elif ch == "Y":
                    rotated.sdg(q)
                    rotated.h(q)
            rotated.measure_all()
            res = simulate(rotated, shots=shots, seed=42)
            counts = res["counts"]
            tot_shots = sum(counts.values())
            if tot_shots == 0:
                continue
            exp = 0.0
            for bitstring, c in counts.items():
                product = 1.0
                for q in range(n):
                    ch = pauli_str[q] if q < len(pauli_str) else "I"
                    if ch == "I":
                        continue
                    bit = bitstring[len(bitstring) - 1 - q] if q < len(bitstring) else "0"
                    sign = 1.0 if bit == "0" else -1.0
                    product *= sign
                exp += c * product
            exp /= tot_shots
            total += float(coeff) * exp
        return total

    def _score_pool_gradients(
        self,
        selected_ops: List[Dict[str, Any]],
        params: List[float],
        hamiltonian: List[Tuple[str, float]],
        *,
        shots: int,
        seed: Optional[int],
    ) -> np.ndarray:
        """
        Score each pool operator by the energy *change* when added with a
        large parameter (not just gradient at zero, which is degenerate for
        many entangling operators acting on |0...0⟩).
        """
        n_pool = len(self.pool)
        scores = np.zeros(n_pool, dtype=float)
        # Baseline energy at current ansatz
        e_base = self._compute_energy(
            selected_ops, np.array(params, dtype=float), hamiltonian, shots // 2,
        )
        # Try each pool operator with multiple test angles to see which gives
        # the largest energy *decrease*. This is more robust than parameter-
        # shift at zero, which is exactly zero for many entangling operators.
        test_angles = [0.5, 1.0, np.pi / 2.0]
        for i, op in enumerate(self.pool):
            tentative_ops = selected_ops + [op]
            best_decrease = 0.0
            for theta in test_angles:
                params_test = list(params) + [theta]
                try:
                    e_test = self._compute_energy(
                        tentative_ops, np.array(params_test), hamiltonian, shots // 2,
                    )
                    decrease = e_base - e_test
                    if decrease > best_decrease:
                        best_decrease = decrease
                except Exception as exc:
                    logger.debug("ADAPT-VQE pool score %s failed: %s", op["name"], exc)
            scores[i] = best_decrease
        return scores
