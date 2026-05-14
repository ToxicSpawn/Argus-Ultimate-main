"""Trotter-Suzuki time evolution for Hamiltonian simulation.

Simulates U(t) = exp(-iHt) via first-order Trotter decomposition.

H = Σ_k c_k P_k  where P_k are tensor products of Pauli operators.

U(t) ≈ ∏_k exp(-i c_k P_k Δt) repeated n_steps times.

Uses the in-repo statevector simulator for gate application.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, field
from quantum_simulator import QuantumCircuit, simulate


@dataclass
class PauliTerm:
    """A single Pauli term in a Hamiltonian: c * (P_{q0} ⊗ P_{q1} ⊗ ...)."""

    coefficient: float
    paulis: Tuple[Tuple[int, str], ...]  # (qubit_index, "X"/"Y"/"Z")


@dataclass
class Hamiltonian:
    """Hamiltonian as a sum of Pauli terms."""

    terms: List[PauliTerm]
    name: str = "H"

    def __repr__(self) -> str:
        parts = []
        for term in self.terms:
            pauli_str = " ⊗ ".join(f"{p[1]}{p[0]}" for p in term.paulis)
            parts.append(f"{term.coefficient:+.4f} * {pauli_str}")
        return f"{self.name} = " + " + ".join(parts)


@dataclass(frozen=True)
class TrotterResult:
    """Result of Trotter time evolution."""

    circuit_name: str
    counts: dict[str, int]
    probabilities: dict[str, float]
    n_qubits: int
    time: float
    n_steps: int
    dt: float
    hamiltonian_name: str
    backend: str
    execution_mode: str
    expectation_z: dict[str, float]
    entanglement_score: float | None = None
    raw_statevector: list[complex] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "circuit_name": self.circuit_name,
            "counts": dict(self.counts),
            "probabilities": dict(self.probabilities),
            "n_qubits": self.n_qubits,
            "time": self.time,
            "n_steps": self.n_steps,
            "dt": self.dt,
            "hamiltonian_name": self.hamiltonian_name,
            "backend": self.backend,
            "execution_mode": self.execution_mode,
            "expectation_z": dict(self.expectation_z),
            "entanglement_score": self.entanglement_score,
        }


# ---------------------------------------------------------------------------
# Trotter evolution
# ---------------------------------------------------------------------------


def _apply_pauli_rotation(circuit: QuantumCircuit, qubit: int, pauli: str, angle: float) -> None:
    """Apply exp(-i * pauli * angle/2) via gate decomposition.

    exp(-i * Z * θ/2) = RZ(θ)
    exp(-i * X * θ/2) = H RZ(θ) H
    exp(-i * Y * θ/2) = S† H RZ(θ) H S
    """
    angle = float(angle)
    if pauli == "Z":
        circuit.rz(angle, qubit)
    elif pauli == "X":
        circuit.hadamard(qubit)
        circuit.rz(angle, qubit)
        circuit.hadamard(qubit)
    elif pauli == "Y":
        # exp(-i Y θ/2) = RY(-θ)
        circuit.ry(-angle, qubit)
    else:
        raise ValueError(f"Unknown Pauli: {pauli}")


def _apply_trotter_step(
    circuit: QuantumCircuit,
    hamiltonian: Hamiltonian,
    dt: float,
    n_qubits: int,
) -> None:
    """Apply one first-order Trotter step: ∏_k exp(-i c_k P_k dt)."""
    for term in hamiltonian.terms:
        angle = 2.0 * term.coefficient * dt
        # Single Pauli term: apply rotation directly
        if len(term.paulis) == 1:
            q, p = term.paulis[0]
            _apply_pauli_rotation(circuit, q, p, angle)
        # Two-body: use entangling decomposition
        elif len(term.paulis) == 2:
            (q0, p0), (q1, p1) = term.paulis
            # Basis change: rotate to Z basis
            _apply_basis_change(circuit, q0, p0, to_z=True)
            _apply_basis_change(circuit, q1, p1, to_z=True)
            # exp(-i Z⊗Z angle/2) = CNOT(q0,q1) RZ(q1, angle) CNOT(q0,q1)
            circuit.cnot(q0, q1)
            circuit.rz(angle, q1)
            circuit.cnot(q0, q1)
            # Undo basis change
            _apply_basis_change(circuit, q0, p0, to_z=False)
            _apply_basis_change(circuit, q1, p1, to_z=False)
        else:
            # Higher-body: use pair-wise decomposition (first-order Trotter)
            for i in range(len(term.paulis) - 1):
                q0, p0 = term.paulis[i]
                q1, p1 = term.paulis[i + 1]
                sub_angle = angle / (len(term.paulis) - 1)
                _apply_basis_change(circuit, q0, p0, to_z=True)
                _apply_basis_change(circuit, q1, p1, to_z=True)
                circuit.cnot(q0, q1)
                circuit.rz(sub_angle, q1)
                circuit.cnot(q0, q1)
                _apply_basis_change(circuit, q0, p0, to_z=False)
                _apply_basis_change(circuit, q1, p1, to_z=False)


def _apply_basis_change(circuit: QuantumCircuit, qubit: int, pauli: str, to_z: bool) -> None:
    """Change basis between Pauli eigenbasis and computational basis.

    If to_z=True: rotate from pauli basis to Z basis.
    If to_z=False: rotate from Z basis back to pauli basis.
    """
    if pauli == "Z":
        return  # Already in Z basis
    if pauli == "X":
        if to_z:
            circuit.hadamard(qubit)
        else:
            circuit.hadamard(qubit)
    elif pauli == "Y":
        if to_z:
            circuit.s(qubit)
            circuit.hadamard(qubit)
        else:
            circuit.hadamard(qubit)
            # S† = S³
            circuit.s(qubit)
            circuit.s(qubit)
            circuit.s(qubit)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def trotter_evolve(
    hamiltonian: Hamiltonian,
    *,
    n_qubits: int,
    time: float,
    n_steps: int = 10,
    initial_state: int = 0,
    shots: int = 1024,
    seed: int | None = None,
) -> dict[str, Any]:
    """Simulate U(t)|ψ⟩ via first-order Trotter.

    Args:
        hamiltonian: Hamiltonian as sum of Pauli terms.
        n_qubits: number of qubits.
        time: total evolution time.
        n_steps: number of Trotter steps.
        initial_state: initial computational basis state.
        shots: number of measurement shots.
        seed: random seed for measurements.

    Returns:
        dict with counts, probabilities, expectation values, and metadata.
    """
    circuit = QuantumCircuit(n_qubits)
    # Initialize in computational basis state
    if initial_state > 0:
        for i in range(n_qubits):
            if (initial_state >> i) & 1:
                circuit.x(i)

    dt = time / n_steps
    for _ in range(n_steps):
        _apply_trotter_step(circuit, hamiltonian, dt, n_qubits)

    circuit.measure_all()
    raw_result = simulate(circuit, shots=shots, seed=seed, backend="statevector")
    counts = raw_result.get("counts", {})
    probabilities = {k: v / shots for k, v in sorted(counts.items())}

    # Compute Z expectation for each qubit
    exp_z = {}
    for q in range(n_qubits):
        exp_z[f"Z{q}"] = 0.0
        for state_str, count in counts.items():
            bit = int(state_str[n_qubits - 1 - q])
            exp_z[f"Z{q}"] += (1 - 2 * bit) * count / shots

    entanglement = max(probabilities.values()) if probabilities else 0.0
    ent_score = max(0.0, min(1.0, entanglement * n_qubits - 1.0 / n_qubits)) if n_qubits > 1 else 0.0

    return TrotterResult(
        circuit_name="trotter_evolution",
        counts=counts,
        probabilities=probabilities,
        n_qubits=n_qubits,
        time=time,
        n_steps=n_steps,
        dt=dt,
        hamiltonian_name=hamiltonian.name,
        backend="statevector",
        execution_mode="classical_statevector_simulation",
        expectation_z=exp_z,
        entanglement_score=ent_score,
    ).to_dict()


# ---------------------------------------------------------------------------
# Common Hamiltonians
# ---------------------------------------------------------------------------


def ising_hamiltonian(n_qubits: int, j_zz: float = 1.0, h_x: float = 0.5) -> Hamiltonian:
    """Transverse-field Ising model: H = -J Σ Z_i Z_{i+1} - h Σ X_i."""
    terms: list[PauliTerm] = []
    for i in range(n_qubits - 1):
        terms.append(PauliTerm(coefficient=-j_zz, paulis=((i, "Z"), (i + 1, "Z"))))
    for i in range(n_qubits):
        terms.append(PauliTerm(coefficient=-h_x, paulis=((i, "X"),)))
    return Hamiltonian(terms=terms, name=f"TFIM({n_qubits}q)")


def heisenberg_hamiltonian(n_qubits: int, j: float = 1.0) -> Hamiltonian:
    """Heisenberg XXX model: H = J Σ (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})."""
    terms: list[PauliTerm] = []
    for i in range(n_qubits - 1):
        terms.append(PauliTerm(coefficient=j, paulis=((i, "X"), (i + 1, "X"))))
        terms.append(PauliTerm(coefficient=j, paulis=((i, "Y"), (i + 1, "Y"))))
        terms.append(PauliTerm(coefficient=j, paulis=((i, "Z"), (i + 1, "Z"))))
    return Hamiltonian(terms=terms, name=f"HeisenbergXXX({n_qubits}q)")


__all__ = [
    "PauliTerm",
    "Hamiltonian",
    "TrotterResult",
    "trotter_evolve",
    "ising_hamiltonian",
    "heisenberg_hamiltonian",
]
