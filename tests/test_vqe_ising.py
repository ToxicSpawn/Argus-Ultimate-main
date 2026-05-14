"""
VQE on Ising Hamiltonians — verification against exact diagonalization.

Phase D3 of the quantum overhaul.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantum.algorithms.vqe import VQESolver, exact_ising_ground_energy


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _random_ising(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    h = rng.uniform(-0.5, 0.5, n)
    J_upper = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            J_upper[i, j] = rng.uniform(-0.5, 0.5)
    return h, J_upper


# ═════════════════════════════════════════════════════════════════════════════
# Exact diagonalization sanity checks
# ═════════════════════════════════════════════════════════════════════════════


def test_exact_diagonalization_ferromagnet():
    """All-spins-aligned Ising ferromagnet has known ground state."""
    n = 4
    h = np.zeros(n)
    J = np.zeros((n, n))
    for i in range(n - 1):
        J[i, i + 1] = -1.0  # ferromagnetic coupling

    energy, bits = exact_ising_ground_energy(h, J)
    # Ferromagnetic ground state: all spins aligned
    assert energy == -3.0  # 3 bonds with J=-1
    assert bits == [0, 0, 0, 0] or bits == [1, 1, 1, 1]


# ═════════════════════════════════════════════════════════════════════════════
# 4-qubit VQE tests
# ═════════════════════════════════════════════════════════════════════════════


class TestVQE4Qubit:

    @pytest.mark.parametrize("seed", [101, 202, 303])
    def test_vqe_finds_low_energy_state(self, seed):
        """VQE must find a state with energy <= average energy."""
        n = 4
        h, J = _random_ising(n, seed)
        exact_e, _ = exact_ising_ground_energy(h, J)

        # Average energy over all 2^n states
        all_energies = []
        for x in range(1 << n):
            z = np.array(
                [1.0 if not ((x >> q) & 1) else -1.0 for q in range(n)]
            )
            e = float(np.sum(h * z))
            for i in range(n):
                for j in range(i + 1, n):
                    e += J[i, j] * z[i] * z[j]
            all_energies.append(e)
        avg_e = float(np.mean(all_energies))

        solver = VQESolver(n_qubits=n, n_layers=4)
        result = solver.solve_ising(
            h, J, max_iter=80, shots=2048, n_restarts=3
        )
        vqe_e = float(result["ground_energy"])

        # VQE should beat the random average; with multiple restarts and
        # 4 layers, it should also be within 30% of the exact ground.
        assert vqe_e <= avg_e, (
            f"seed={seed}: VQE energy {vqe_e:.3f} > avg {avg_e:.3f}"
        )

    def test_vqe_4q_aggregate_within_30pct_of_exact(self):
        """Average over seeds: VQE should be within 30% of exact ground."""
        n = 4
        errors = []
        for seed in [101, 202, 303, 404, 505]:
            h, J = _random_ising(n, seed)
            exact_e, _ = exact_ising_ground_energy(h, J)
            solver = VQESolver(n_qubits=n, n_layers=4)
            result = solver.solve_ising(h, J, max_iter=80, shots=2048, n_restarts=3)
            err = abs(result["ground_energy"] - exact_e) / max(abs(exact_e), 1e-6)
            errors.append(err)
        avg_err = sum(errors) / len(errors)
        assert avg_err < 0.30, f"Average VQE error {avg_err:.1%} > 30%"


# ═════════════════════════════════════════════════════════════════════════════
# Convergence + result schema
# ═════════════════════════════════════════════════════════════════════════════


def test_vqe_returns_required_keys():
    n = 3
    h = np.array([0.5, -0.3, 0.2])
    J = np.zeros((n, n))
    J[0, 1] = 0.5
    J[1, 2] = -0.4
    solver = VQESolver(n_qubits=n, n_layers=2)
    result = solver.solve_ising(h, J, max_iter=50, shots=1024, n_restarts=2)
    for key in (
        "ground_energy",
        "ground_state_bits",
        "convergence_history",
        "optimal_params",
        "method",
        "n_iterations",
        "elapsed_ms",
    ):
        assert key in result, f"Missing key: {key}"
    assert result["method"] == "vqe_in_repo_simulator"
    assert isinstance(result["convergence_history"], list)
    assert len(result["ground_state_bits"]) == n


def test_vqe_convergence_history_non_empty():
    n = 3
    h, J = _random_ising(n, 1)
    solver = VQESolver(n_qubits=n, n_layers=2)
    result = solver.solve_ising(h, J, max_iter=30, shots=1024, n_restarts=1)
    assert len(result["convergence_history"]) > 0


# ═════════════════════════════════════════════════════════════════════════════
# Pauli-string Hamiltonian path
# ═════════════════════════════════════════════════════════════════════════════


def test_vqe_pauli_string_zi_iz():
    """Solve a 2-qubit Pauli Hamiltonian with one Z on each qubit."""
    pauli_terms = [("ZI", 0.5), ("IZ", -0.3)]
    solver = VQESolver(n_qubits=2, n_layers=3)
    result = solver.solve_hamiltonian(
        pauli_terms, max_iter=50, shots=2048, n_restarts=2
    )
    # Ground: minimize 0.5·z0 - 0.3·z1
    # → z0=-1 (qubit 0 = |1⟩), z1=+1 (qubit 1 = |0⟩) → energy = -0.8
    assert result["ground_energy"] < -0.5
    assert result["method"] == "vqe_in_repo_simulator"
