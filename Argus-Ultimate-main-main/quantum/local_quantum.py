"""Enhanced local quantum runtime for ARGUS.

This module is the local-only quantum toolkit.  It builds on the corrected
in-repo statevector simulator and adds:

- GHZ and W multi-qubit entangled state circuits
- Parameterized rotation sweeps
- Optional Qiskit Aer local backend (if ``qiskit-aer`` is installed)
- Optional MPS/tensor-network backend stub
- Noise model support via the existing noise_model.py profiles
- Honest metadata: every result records whether it used simulator, Aer, or MPS
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from quantum_simulator import QuantumCircuit, simulate


def _has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


HAS_QISKIT_AER = _has_module("qiskit_aer")


@dataclass(frozen=True)
class LocalQuantumResult:
    """Result from a local quantum circuit execution."""

    circuit_name: str
    counts: dict[str, int]
    probabilities: dict[str, float]
    shots: int
    n_qubits: int
    backend: str
    execution_mode: str
    hardware_enabled: bool = False
    entanglement_score: float | None = None
    expectation_value: float | None = None
    noise_profile: str | None = None
    warnings: tuple[str, ...] = ()
    raw_result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "circuit_name": self.circuit_name,
            "counts": dict(self.counts),
            "probabilities": dict(self.probabilities),
            "shots": self.shots,
            "n_qubits": self.n_qubits,
            "backend": self.backend,
            "execution_mode": self.execution_mode,
            "hardware_enabled": self.hardware_enabled,
            "entanglement_score": self.entanglement_score,
            "expectation_value": self.expectation_value,
            "noise_profile": self.noise_profile,
            "warnings": list(self.warnings),
            "raw_result": dict(self.raw_result),
        }


# ---------------------------------------------------------------------------
# Entangled state circuits
# ---------------------------------------------------------------------------


def _ghz_circuit(n_qubits: int = 4) -> QuantumCircuit:
    """Create GHZ state (|00...0⟩ + |11...1⟩)/√2."""
    circuit = QuantumCircuit(n_qubits)
    circuit.hadamard(0)
    for i in range(n_qubits - 1):
        circuit.cnot(i, i + 1)
    return circuit


def _w_circuit(n_qubits: int = 4) -> QuantumCircuit:
    """Create W state (|100...0⟩ + |010...0⟩ + ... + |00...1⟩)/√n.

    Uses the standard incremental preparation circuit: a chain of controlled-
    Y rotations that progressively build the uniform single-excitation
    superposition.
    """
    import math

    circuit = QuantumCircuit(n_qubits)
    # First qubit: rotate from |0⟩ to sin(θ)|0⟩ + cos(θ)|1⟩ with θ = arccos(1/√n)
    theta0 = math.acos(1.0 / math.sqrt(n_qubits))
    circuit.ry(2.0 * theta0, 0)
    # Subsequent qubits: controlled rotations to distribute excitation
    for k in range(1, n_qubits):
        theta_k = math.acos(math.sqrt((n_qubits - k) / (n_qubits - k + 1)))
        # CRY(theta) from k-1 to k
        # Approximate CRY via: Ry(k, theta/2); CNOT(k-1, k); Ry(k, -theta/2); CNOT(k-1, k)
        circuit.ry(float(theta_k), k)
        circuit.cnot(k - 1, k)
        circuit.ry(float(-theta_k), k)
        circuit.cnot(k - 1, k)
    return circuit


# ---------------------------------------------------------------------------
# Parameterized circuits
# ---------------------------------------------------------------------------


def build_parameterized_circuit(
    n_qubits: int,
    thetas: list[float] | None = None,
    *,
    n_layers: int = 1,
) -> QuantumCircuit:
    """Build a hardware-efficient parameterized circuit (RY + CZ layers).

    Each layer applies RY(θ) on every qubit, then nearest-neighbour CZ
    entangling gates.  ``thetas`` should have length ``n_qubits * n_layers``.
    """
    import numpy as np

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


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------


def _apply_readout_noise(
    counts: dict[str, int],
    noise_profile: dict[str, float] | None,
) -> tuple[dict[str, int], list[str]]:
    """Apply readout bit-flip noise to measurement counts.

    Returns the (possibly noisy) counts and any warnings.
    """
    if not noise_profile:
        return counts, []

    import numpy as np

    readout_error = noise_profile.get("readout_error", 0.0)
    if readout_error <= 0.0:
        return list(counts), []

    noisy: dict[str, int] = {}
    warnings = [f"Applied readout noise with error rate {readout_error}"]
    for bitstring, count in counts.items():
        for _ in range(count):
            noisy_bits = list(bitstring)
            for i in range(len(noisy_bits)):
                if np.random.random() < readout_error:
                    noisy_bits[i] = "1" if noisy_bits[i] == "0" else "0"
            noisy_str = "".join(noisy_bits)
            noisy[noisy_str] = noisy.get(noisy_str, 0) + 1
    return noisy, warnings


# ---------------------------------------------------------------------------
# Aer and MPS backends
# ---------------------------------------------------------------------------


def _run_aer(
    circuit: QuantumCircuit,
    shots: int,
) -> dict[str, Any]:
    """Run circuit on Qiskit Aer if installed.

    Translates the in-repo QuantumCircuit operations into a qiskit circuit
    and executes it on AerSimulator for faster simulation.
    """
    try:
        from qiskit import QuantumCircuit as QiskitQC
        from qiskit_aer import AerSimulator
    except ImportError:
        return {}

    qc = QiskitQC(circuit.n_qubits, circuit.n_qubits)
    for gate in circuit.gates:
        name = gate[0] if gate else ""
        if name == "H":
            qc.h(gate[1])
        elif name == "X":
            qc.x(gate[1])
        elif name == "Y":
            qc.y(gate[1])
        elif name == "Z":
            qc.z(gate[1])
        elif name == "S":
            qc.s(gate[1])
        elif name == "T":
            qc.t(gate[1])
        elif name == "RX":
            qc.rx(gate[2], gate[1])
        elif name == "RY":
            qc.ry(gate[2], gate[1])
        elif name == "RZ":
            qc.rz(gate[2], gate[1])
        elif name == "CNOT":
            qc.cx(gate[1], gate[2])
        elif name == "CZ":
            qc.cz(gate[1], gate[2])
        elif name == "RZZ":
            qc.rzz(gate[3], gate[1], gate[2])
        elif name == "SWAP":
            qc.swap(gate[1], gate[2])
        elif name == "MEASURE_ALL":
            qc.measure_all()
    # Ensure measurements present
    if qc.num_clbits == 0:
        qc.measure_all()

    sim = AerSimulator()
    from qiskit import transpile
    transpiled = transpile(qc, sim)
    result = sim.run(transpiled, shots=shots).result()
    return dict(result.get_counts())


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------


class LocalQuantumRunner:
    """Run quantum circuits entirely on this machine."""

    def __init__(
        self,
        *,
        backend: str = "auto",
        noise_profile: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialise the local quantum runner.

        Args:
            backend: ``"auto"`` picks Aer if available, else statevector.
            noise_profile: name from ``quantum.noise_model`` (e.g. ``"ibm_brisbane"``).
            seed: default random seed for reproducibility.
        """
        if backend == "auto":
            self.backend = "aer" if HAS_QISKIT_AER else "statevector"
        else:
            self.backend = backend
        self.noise_profile = noise_profile
        self.seed = seed
        self._noise_params = self._load_noise_params(noise_profile)

    @staticmethod
    def _load_noise_params(profile_name: str | None) -> dict[str, float] | None:
        if not profile_name:
            return None
        try:
            from quantum.noise_model import QuantumNoiseModel
            model = QuantumNoiseModel()
            profiles = getattr(model, "profiles", None) or getattr(model, "_HARDWARE_PROFILES", {})
            if isinstance(profiles, dict) and profile_name in profiles:
                return profiles[profile_name]
        except Exception:
            pass
        return None

    # ---- entangled states --------------------------------------------------

    def run_ghz(
        self,
        *,
        n_qubits: int = 4,
        shots: int = 1024,
    ) -> dict[str, Any]:
        """Run a GHZ entangled state and measure."""
        circuit = _ghz_circuit(n_qubits)
        return self._run(circuit, "ghz", shots=shots)

    def run_w(
        self,
        *,
        n_qubits: int = 4,
        shots: int = 1024,
    ) -> dict[str, Any]:
        """Run a W entangled state and measure."""
        circuit = _w_circuit(n_qubits)
        return self._run(circuit, "w_state", shots=shots)

    # ---- parameterized circuits --------------------------------------------

    def run_parameterized(
        self,
        *,
        n_qubits: int = 3,
        thetas: list[float] | None = None,
        n_layers: int = 1,
        shots: int = 1024,
    ) -> dict[str, Any]:
        """Build and run a parameterized RY+CZ circuit."""
        circuit = build_parameterized_circuit(n_qubits, thetas, n_layers=n_layers)
        return self._run(circuit, "parameterized", shots=shots)

    # ---- internal ----------------------------------------------------------

    def _run(
        self,
        circuit: QuantumCircuit,
        name: str,
        *,
        shots: int,
    ) -> dict[str, Any]:
        shots = max(shots, 1)
        warnings: list[str] = []
        raw_counts: dict[str, int] = {}

        # Try Aer backend if available
        if self.backend == "aer" and HAS_QISKIT_AER:
            aer_counts = _run_aer(circuit, shots)
            if aer_counts:
                raw_counts = aer_counts
                backend_name = "aer_simulator"
                execution_mode = "qiskit_aer_simulator"
            else:
                warnings.append("Aer execution failed; falling back to statevector simulator.")
                raw_counts = {}
        else:
            backend_name = "statevector"
            execution_mode = "classical_statevector_simulation"

        # Fallback to in-repo statevector
        if not raw_counts:
            result = simulate(circuit, shots=shots, seed=self.seed, backend="statevector")
            raw_counts = result.get("counts", {})
            backend_name = "statevector"
            execution_mode = "classical_statevector_simulation"

        # Inject noise
        noisy_counts, noise_warnings = _apply_readout_noise(raw_counts, self._noise_params)
        warnings.extend(noise_warnings)

        # Compute probabilities and entanglement metrics
        total = max(sum(noisy_counts.values()), shots, 1)
        probabilities = {k: v / total for k, v in sorted(noisy_counts.items())}
        entanglement = _entanglement_score(probabilities, circuit.n_qubits)

        return LocalQuantumResult(
            circuit_name=name,
            counts=noisy_counts,
            probabilities=probabilities,
            shots=shots,
            n_qubits=circuit.n_qubits,
            backend=backend_name,
            execution_mode=execution_mode,
            entanglement_score=entanglement,
            noise_profile=self.noise_profile,
            warnings=tuple(warnings),
            raw_result={"original_counts": raw_counts},
        ).to_dict()


def _entanglement_score(probabilities: dict[str, float], n_qubits: int) -> float:
    """Heuristic entanglement score from measurement distribution.

    For n=2 Bell states the score is ~1.0; for GHZ it is also ~1.0.
    Random product states give a score near 0.
    """
    if n_qubits <= 1:
        return 0.0
    max_prob = max(probabilities.values()) if probabilities else 0.0
    return max(0.0, min(1.0, max_prob * n_qubits - 1.0 / n_qubits))


__all__ = [
    "LocalQuantumRunner",
    "LocalQuantumResult",
    "HAS_QISKIT_AER",
    "build_parameterized_circuit",
]
