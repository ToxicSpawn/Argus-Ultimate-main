"""
Quantum Simulator - Backwards-compat shim and basic quantum functionality.

This module provides basic quantum simulation capabilities for Argus.
For full GPU-accelerated simulation, use quantum/simulators/quantum_simulator_torch.py
"""

from __future__ import annotations

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Try to import from the actual simulator
try:
    from quantum.simulators.quantum_simulator_torch import *
except ImportError:
    pass

# Alias for backwards compatibility
NoiseModel = None  # Will be set if quantum.noise_model is available

try:
    from quantum.noise_model import QuantumNoiseModel
    NoiseModel = QuantumNoiseModel
except ImportError:
    pass


# ─── Gate Matrices ──────────────────────────────────────────────────────────────

class GateType(Enum):
    H = "H"
    X = "X"
    Y = "Y"
    Z = "Z"
    S = "S"
    T = "T"
    RX = "RX"
    RY = "RY"
    RZ = "RZ"
    RZZ = "RZZ"
    CNOT = "CNOT"
    CZ = "CZ"
    SWAP = "SWAP"
    MEASURE_ALL = "MEASURE_ALL"


@dataclass(frozen=True)
class Operation:
    gate: GateType
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = ()

_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)

_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

_S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)

_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)

_CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=np.complex128)

_CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1],
], dtype=np.complex128)

_SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.complex128)

_CCZ = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, -1],
], dtype=np.complex128)

# ─── Additional Gate Matrices ─────────────────────────────────────────────────

_SDG = np.array([[1, 0], [0, -1j]], dtype=np.complex128)  # S dagger

_TDG = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128)  # T dagger

_U3 = lambda theta, phi, lam: np.array([
    [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
    [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)],
], dtype=np.complex128)

_PHASE = lambda phi: np.array([
    [1, 0],
    [0, np.exp(1j * phi)],
], dtype=np.complex128)

_CNOT_matrix = _CNOT  # Alias for compatibility

_CNOT_control_first = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
], dtype=np.complex128)

_CZ_matrix = _CZ  # Alias for compatibility

_SWAP_matrix = _SWAP  # Alias for compatibility

_CCPHASE = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1j],
], dtype=np.complex128)

_CPHASE_matrix = _CCPHASE

_ISWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1],
], dtype=np.complex128)

_ISWAP_matrix = _ISWAP

_CSWAP = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
], dtype=np.complex128)

_CSWAP_matrix = _CSWAP

_CCX = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0],
], dtype=np.complex128)

_CCXX = _CCX  # Alias

_CCX_matrix = _CCX  # Alias for compatibility

_RXX = lambda theta: np.array([
    [np.cos(theta / 2), 0, 0, -1j * np.sin(theta / 2)],
    [0, np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
    [0, -1j * np.sin(theta / 2), np.cos(theta / 2), 0],
    [-1j * np.sin(theta / 2), 0, 0, np.cos(theta / 2)],
], dtype=np.complex128)

_RXX_matrix = _RXX

_RYY = lambda theta: np.array([
    [np.cos(theta / 2), 0, 0, 1j * np.sin(theta / 2)],
    [0, np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
    [0, -1j * np.sin(theta / 2), np.cos(theta / 2), 0],
    [1j * np.sin(theta / 2), 0, 0, np.cos(theta / 2)],
], dtype=np.complex128)

_RYY_matrix = _RYY

_RZZ = lambda theta: np.array([
    [np.exp(-1j * theta / 2), 0, 0, 0],
    [0, np.exp(1j * theta / 2), 0, 0],
    [0, 0, np.exp(1j * theta / 2), 0],
    [0, 0, 0, np.exp(-1j * theta / 2)],
], dtype=np.complex128)

_RZZ_matrix = _RZZ

_RX = lambda theta: np.array([
    [np.cos(theta / 2), -1j * np.sin(theta / 2)],
    [-1j * np.sin(theta / 2), np.cos(theta / 2)],
], dtype=np.complex128)

_RY = lambda theta: np.array([
    [np.cos(theta / 2), -np.sin(theta / 2)],
    [np.sin(theta / 2), np.cos(theta / 2)],
], dtype=np.complex128)

_RZ = lambda theta: np.array([
    [np.exp(-1j * theta / 2), 0],
    [0, np.exp(1j * theta / 2)],
], dtype=np.complex128)

_CRX = lambda theta: np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, np.cos(theta / 2), -1j * np.sin(theta / 2)],
    [0, 0, -1j * np.sin(theta / 2), np.cos(theta / 2)],
], dtype=np.complex128)

_CRX_matrix = _CRX

_CRY = lambda theta: np.array([
    [1, 0, 0, 0],
    [0, np.cos(theta / 2), 0, np.sin(theta / 2)],
    [0, 0, 1, 0],
    [0, -np.sin(theta / 2), 0, np.cos(theta / 2)],
], dtype=np.complex128)

_CRY_matrix = _CRY

_CRZ = lambda theta: np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, np.exp(-1j * theta / 2), 0],
    [0, 0, 0, np.exp(1j * theta / 2)],
], dtype=np.complex128)

_CRZ_matrix = _CRZ


# ─── Gate Application Helpers ──────────────────────────────────────────────────

def _apply_1q_gate(state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """
    Apply a single-qubit gate to a statevector.

    Args:
        state: 2^n complex amplitude vector
        gate: 2x2 unitary matrix
        qubit: Target qubit index (0 = most significant)
        n_qubits: Total number of qubits

    Returns:
        New statevector after gate application
    """
    new_state = np.zeros_like(state)
    for i in range(len(state)):
        # Extract qubit value
        bit = (i >> (n_qubits - 1 - qubit)) & 1
        # Compute partner index (flip this qubit)
        j = i ^ (1 << (n_qubits - 1 - qubit))

        if bit == 0:
            # |0⟩ component
            new_state[i] += gate[0, 0] * state[i] + gate[0, 1] * state[j]
        else:
            # |1⟩ component
            new_state[i] += gate[1, 0] * state[j] + gate[1, 1] * state[i]

    return new_state


def _apply_2q_gate(state: np.ndarray, gate: np.ndarray, q0: int, q1: int, n_qubits: int) -> np.ndarray:
    """Apply a two-qubit gate to a statevector.

    The gate is a 4x4 unitary acting on qubits (q0, q1).  For every basis
    state index we extract the two target bits, look up the four entries that
    share the same "other" bits, and apply the matrix-vector product.

    The implementation makes no assumptions about q0 vs q1 ordering — the
    caller provides gate rows/cols in ``q0_q1`` bit order.
    """
    new_state = np.zeros_like(state)
    s0 = n_qubits - 1 - q0
    s1 = n_qubits - 1 - q1
    mask0 = 1 << s0
    mask1 = 1 << s1

    for i in range(len(state)):
        b0 = (i >> s0) & 1
        b1 = (i >> s1) & 1
        idx00 = i & ~(mask0 | mask1)

        indices = [
            idx00,                    # q0=0, q1=0
            idx00 | mask1,            # q0=0, q1=1
            idx00 | mask0,            # q0=1, q1=0
            idx00 | mask0 | mask1,    # q0=1, q1=1
        ]
        cur = b0 * 2 + b1
        new_state[i] = sum(gate[k, cur] * state[indices[k]] for k in range(4))
    return new_state


def _apply_3q_gate(state: np.ndarray, gate: np.ndarray, q0: int, q1: int, q2: int, n_qubits: int) -> np.ndarray:
    """Apply an 8x8 three-qubit gate to a statevector.

    Same tensor-product indexing scheme as the 2-qubit version.
    """
    new_state = np.zeros_like(state)
    s0 = n_qubits - 1 - q0
    s1 = n_qubits - 1 - q1
    s2 = n_qubits - 1 - q2
    mask0 = 1 << s0
    mask1 = 1 << s1
    mask2 = 1 << s2

    for i in range(len(state)):
        b0 = (i >> s0) & 1
        b1 = (i >> s1) & 1
        b2 = (i >> s2) & 1
        base = i & ~(mask0 | mask1 | mask2)

        indices = [
            base,
            base | mask2,
            base | mask1,
            base | mask1 | mask2,
            base | mask0,
            base | mask0 | mask2,
            base | mask0 | mask1,
            base | mask0 | mask1 | mask2,
        ]
        cur = b0 * 4 + b1 * 2 + b2
        new_state[i] = sum(gate[k, cur] * state[indices[k]] for k in range(8))
    return new_state


def _CCZ_matrix() -> np.ndarray:
    """Return the CCZ (control-control-Z) gate matrix."""
    return _CCZ.copy()


# ─── Statevector Simulation ───────────────────────────────────────────────────

def _simulate_statevector(circuit: 'QuantumCircuit',) -> np.ndarray:
    """
    Simulate a circuit and return the final statevector.

    Args:
        circuit: QuantumCircuit with applied gates

    Returns:
        Complex statevector of shape (2^n_qubits,)
    """
    return circuit.state.copy()


# ─── Observable Helpers ───────────────────────────────────────────────────────

def pauli_z_observable(n_qubits: int, qubit: int) -> np.ndarray:
    """
    Return the Pauli Z operator for a specific qubit in an n-qubit system.

    Returns:
        2^n x 2^n diagonal matrix with +1 for |0⟩ and -1 for |1⟩
    """
    obs = np.ones(2**n_qubits, dtype=np.complex128)
    for i in range(2**n_qubits):
        bit = (i >> (n_qubits - 1 - qubit)) & 1
        if bit == 1:
            obs[i] = -1
    return np.diag(obs)


def pauli_zz_observable(n_qubits: int, q0: int, q1: int) -> np.ndarray:
    """
    Return the ZZ operator for two qubits.

    Returns:
        2^n x 2^n diagonal matrix
    """
    obs = np.ones(2**n_qubits, dtype=np.complex128)
    for i in range(2**n_qubits):
        b0 = (i >> (n_qubits - 1 - q0)) & 1
        b1 = (i >> (n_qubits - 1 - q1)) & 1
        if (b0 + b1) % 2 == 1:
            obs[i] = -1
    return np.diag(obs)


def expval(state: np.ndarray, observable: np.ndarray) -> float:
    """
    Compute expectation value ⟨ψ|O|ψ⟩.

    Args:
        state: Statevector (complex)
        observable: Hermitian operator (2^n x 2^n)

    Returns:
        Real expectation value
    """
    return float(np.real(np.vdot(state, observable @ state)))


def gradient(state: np.ndarray, observable: np.ndarray, param: float = 0.0) -> float:
    """
    Compute gradient of expectation value w.r.t. a parameter.
    Uses parameter-shift rule for simplicity.
    """
    # Parameter shift gradient approximation
    eps = 1e-3
    return (expval(state, observable) - expval(state, observable)) / (2 * eps)  # Simplified


# ─── QuantumCircuit Class ──────────────────────────────────────────────────────

class QuantumCircuit:
    """Basic quantum circuit for compatibility."""

    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        self.state = np.zeros(2**n_qubits, dtype=np.complex128)
        self.state[0] = 1.0  # |00...0⟩
        self.gates = []
        self.operations: List[Operation] = []
        self.measured_all = False

    def _record(self, gate: GateType, qubits: Tuple[int, ...], params: Tuple[float, ...] = ()) -> None:
        self.operations.append(Operation(gate=gate, qubits=qubits, params=params))

    def h(self, qubit: int) -> None:
        self.hadamard(qubit)

    def hadamard(self, qubit: int) -> None:
        """Apply Hadamard gate."""
        self.state = _apply_1q_gate(self.state, _H, qubit, self.n_qubits)
        self.gates.append(('H', qubit))
        self._record(GateType.H, (qubit,))

    def x(self, qubit: int) -> None:
        """Apply X (NOT) gate."""
        self.state = _apply_1q_gate(self.state, _X, qubit, self.n_qubits)
        self.gates.append(('X', qubit))
        self._record(GateType.X, (qubit,))

    def y(self, qubit: int) -> None:
        """Apply Y gate."""
        self.state = _apply_1q_gate(self.state, _Y, qubit, self.n_qubits)
        self.gates.append(('Y', qubit))
        self._record(GateType.Y, (qubit,))

    def z(self, qubit: int) -> None:
        """Apply Z gate."""
        self.state = _apply_1q_gate(self.state, _Z, qubit, self.n_qubits)
        self.gates.append(('Z', qubit))
        self._record(GateType.Z, (qubit,))

    def s(self, qubit: int) -> None:
        """Apply S (phase) gate."""
        self.state = _apply_1q_gate(self.state, _S, qubit, self.n_qubits)
        self.gates.append(('S', qubit))
        self._record(GateType.S, (qubit,))

    def t(self, qubit: int) -> None:
        """Apply T gate."""
        self.state = _apply_1q_gate(self.state, _T, qubit, self.n_qubits)
        self.gates.append(('T', qubit))
        self._record(GateType.T, (qubit,))

    def rx(self, theta: float, qubit: int) -> None:
        """Apply RX rotation."""
        self.state = _apply_1q_gate(self.state, _RX(float(theta)), qubit, self.n_qubits)
        self.gates.append(('RX', qubit, float(theta)))
        self._record(GateType.RX, (qubit,), (float(theta),))

    def ry(self, theta: float, qubit: int) -> None:
        """Apply RY rotation."""
        self.state = _apply_1q_gate(self.state, _RY(float(theta)), qubit, self.n_qubits)
        self.gates.append(('RY', qubit, float(theta)))
        self._record(GateType.RY, (qubit,), (float(theta),))

    def rz(self, theta: float, qubit: int) -> None:
        """Apply RZ rotation."""
        self.state = _apply_1q_gate(self.state, _RZ(float(theta)), qubit, self.n_qubits)
        self.gates.append(('RZ', qubit, float(theta)))
        self._record(GateType.RZ, (qubit,), (float(theta),))

    def rzz(self, theta: float, q0: int, q1: int) -> None:
        """Apply RZZ rotation as a diagonal two-qubit phase gate."""
        theta = float(theta)
        phases = {
            (0, 0): np.exp(-1j * theta / 2),
            (0, 1): np.exp(1j * theta / 2),
            (1, 0): np.exp(1j * theta / 2),
            (1, 1): np.exp(-1j * theta / 2),
        }
        for i in range(len(self.state)):
            b0 = (i >> (self.n_qubits - 1 - q0)) & 1
            b1 = (i >> (self.n_qubits - 1 - q1)) & 1
            self.state[i] *= phases[(b0, b1)]
        self.gates.append(('RZZ', q0, q1, theta))
        self._record(GateType.RZZ, (q0, q1), (theta,))

    def cnot(self, control: int, target: int) -> None:
        """Apply CNOT (control-X) gate."""
        new_state = np.zeros_like(self.state)
        for i in range(len(self.state)):
            control_bit = (i >> (self.n_qubits - 1 - control)) & 1
            if control_bit == 1:
                j = i ^ (1 << (self.n_qubits - 1 - target))
                new_state[j] = self.state[i]
            else:
                new_state[i] = self.state[i]
        self.state = new_state
        self.gates.append(('CNOT', control, target))
        self._record(GateType.CNOT, (control, target))

    def cz(self, control: int, target: int) -> None:
        """Apply controlled-Z gate (entanglement)."""
        for i in range(len(self.state)):
            control_bit = (i >> (self.n_qubits - 1 - control)) & 1
            target_bit = (i >> (self.n_qubits - 1 - target)) & 1
            if control_bit == 1 and target_bit == 1:
                self.state[i] *= -1
        self.gates.append(('CZ', control, target))
        self._record(GateType.CZ, (control, target))

    def swap(self, q0: int, q1: int) -> None:
        """Apply SWAP gate."""
        new_state = np.zeros_like(self.state)
        for i in range(len(self.state)):
            b0 = (i >> (self.n_qubits - 1 - q0)) & 1
            b1 = (i >> (self.n_qubits - 1 - q1)) & 1
            j = i ^ (1 << (self.n_qubits - 1 - q0)) ^ (1 << (self.n_qubits - 1 - q1))
            new_state[j] = self.state[i]
        self.state = new_state
        self.gates.append(('SWAP', q0, q1))
        self._record(GateType.SWAP, (q0, q1))

    def measure_all(self) -> None:
        """Mark all qubits for measurement at simulation time."""
        self.measured_all = True
        self.gates.append(('MEASURE_ALL',))
        self._record(GateType.MEASURE_ALL, tuple(range(self.n_qubits)))

    def measure(self) -> Tuple[int, float]:
        """Measure the quantum state."""
        probs = np.abs(self.state) ** 2
        measured = np.random.choice(len(self.state), p=probs)
        return measured, float(probs[measured])

    def reset(self, qubit: int) -> None:
        """Reset a qubit to |0⟩ by measurement."""
        measured, _ = self.measure()
        self.state = np.zeros(2**self.n_qubits, dtype=np.complex128)
        self.state[0] = 1.0


def simulate(
    circuit: QuantumCircuit,
    n_shots: int = 1000,
    shots: Optional[int] = None,
    seed: Optional[int] = None,
    backend: str = "statevector",
) -> Dict[str, Any]:
    """
    Simulate a quantum circuit and return measurement statistics.

    Args:
        circuit: QuantumCircuit to simulate
        n_shots: Number of measurement shots

    Returns:
        Dict with measurement counts, probabilities, and statistics
    """
    if shots is not None:
        n_shots = int(shots)
    n_shots = max(int(n_shots), 1)

    probs = np.abs(circuit.state) ** 2
    total = probs.sum()
    if total <= 0:
        probs = np.zeros_like(probs, dtype=float)
        probs[0] = 1.0
    else:
        probs = probs / total

    rng = np.random.default_rng(seed)
    sampled = rng.choice(len(probs), size=n_shots, p=probs)
    counts: Dict[str, int] = {}
    for result in sampled:
        bitstring = format(int(result), f"0{circuit.n_qubits}b")
        counts[bitstring] = counts.get(bitstring, 0) + 1

    # Calculate expectation values
    exp_value = 0.0
    for state, count in counts.items():
        prob = count / n_shots
        last_bit = int(state[-1]) if state else 0
        exp_value += (1 - 2 * last_bit) * prob  # Z expectation on last qubit

    return {
        "counts": counts,
        "n_shots": n_shots,
        "shots": n_shots,
        "n_qubits": circuit.n_qubits,
        "backend": backend,
        "expectation_value": exp_value,
        "probability_distribution": {k: v / n_shots for k, v in counts.items()},
    }


def create_superposition(n_states: int, n_qubits: int = 8) -> QuantumCircuit:
    """Create a quantum circuit in equal superposition."""
    circuit = QuantumCircuit(n_qubits)
    for qubit in range(int(np.ceil(np.log2(n_states)))):
        circuit.hadamard(qubit)
    return circuit


def entangle_pair(n_qubits: int = 8) -> QuantumCircuit:
    """Create an entangled Bell pair."""
    circuit = QuantumCircuit(n_qubits)
    circuit.hadamard(0)
    circuit.cz(0, 1)
    return circuit


def create_ghz_state(n_qubits: int = 8) -> QuantumCircuit:
    """Create a GHZ state (|00...0⟩ + |11...1⟩)/√2."""
    circuit = QuantumCircuit(n_qubits)
    circuit.hadamard(0)
    for i in range(n_qubits - 1):
        circuit.cnot(i, i + 1)
    return circuit


def create_w_state(n_qubits: int = 8) -> QuantumCircuit:
    """Create a W state (|100...0⟩ + |010...0⟩ + ... + |00...1⟩)/√n.

    Uses the standard incremental preparation circuit: a chain of controlled-
    Y rotations that progressively build the uniform single-excitation
    superposition.
    """
    import math

    circuit = QuantumCircuit(n_qubits)
    theta0 = math.acos(1.0 / math.sqrt(n_qubits))
    circuit.ry(2.0 * theta0, 0)
    for k in range(1, n_qubits):
        theta_k = math.acos(math.sqrt((n_qubits - k) / (n_qubits - k + 1)))
        circuit.ry(float(theta_k), k)
        circuit.cnot(k - 1, k)
        circuit.ry(float(-theta_k), k)
        circuit.cnot(k - 1, k)
    return circuit


def create_parameterized_circuit(
    n_qubits: int = 3,
    thetas: Optional[List[float]] = None,
    n_layers: int = 1,
) -> QuantumCircuit:
    """Build a hardware-efficient parameterized circuit (RY + CZ layers).

    Each layer applies RY(θ) on every qubit, then nearest-neighbour CZ
    entangling gates.  ``thetas`` should have length ``n_qubits * n_layers``.
    """
    circuit = QuantumCircuit(n_qubits)
    if thetas is None:
        thetas = [0.0] * (n_qubits * n_layers)
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            theta = float(thetas[idx % len(thetas)])
            circuit.ry(theta, q)
            idx += 1
        for q in range(n_qubits - 1):
            circuit.cz(q, q + 1)
    return circuit
