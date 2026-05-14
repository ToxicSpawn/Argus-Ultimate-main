"""Hardware-ready quantum execution for small validation circuits.

This is intentionally small and concrete: a Bell-state circuit that can run on
IBM Quantum via Qiskit Runtime when hardware is explicitly enabled and a token
is configured. Without credentials or optional SDKs it falls back to the
in-repo statevector simulator and says so in the result metadata.
"""

from __future__ import annotations

import os
import importlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActualQuantumResult:
    """Result from a real-or-simulated quantum circuit execution."""

    circuit_name: str
    counts: dict[str, int]
    probabilities: dict[str, float]
    shots: int
    provider: str
    backend: str
    execution_mode: str
    hardware_enabled: bool
    requested_hardware: bool
    entanglement_score: float
    status: str = "completed"
    quantum_advantage_claimed: bool = False
    job_id: str | None = None
    warnings: tuple[str, ...] = ()
    raw_result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "circuit_name": self.circuit_name,
            "counts": dict(self.counts),
            "probabilities": dict(self.probabilities),
            "shots": self.shots,
            "provider": self.provider,
            "backend": self.backend,
            "execution_mode": self.execution_mode,
            "hardware_enabled": self.hardware_enabled,
            "requested_hardware": self.requested_hardware,
            "entanglement_score": self.entanglement_score,
            "status": self.status,
            "quantum_advantage_claimed": self.quantum_advantage_claimed,
            "job_id": self.job_id,
            "warnings": list(self.warnings),
            "raw_result": dict(self.raw_result),
        }


class ActualQuantumRunner:
    """Run small quantum circuits on real hardware when explicitly enabled."""

    def __init__(
        self,
        *,
        hardware_enabled: bool = False,
        provider: str = "ibm",
        backend_name: str | None = None,
        local_only: bool = False,
    ) -> None:
        self.hardware_enabled = bool(hardware_enabled)
        self.provider = provider
        self.backend_name = backend_name
        self.local_only = bool(local_only)

    def run_bell_pair(
        self,
        *,
        shots: int = 1024,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Run a two-qubit Bell-state circuit.

        The circuit is ``H(0); CX(0, 1); measure_all``. With explicit hardware
        enabled and IBM credentials present, ARGUS attempts IBM Quantum. Any
        setup/runtime failure falls back to the in-repo simulator with a warning.
        """
        shots = max(int(shots), 1)
        warnings: list[str] = []

        if self.local_only:
            return self._run_in_repo_bell_pair(
                shots=shots,
                seed=seed,
                requested_hardware=False,
                warnings=warnings,
            ).to_dict()

        if self.hardware_enabled:
            hardware_result = self._try_ibm_hardware(shots=shots, warnings=warnings)
            if hardware_result is not None:
                return hardware_result.to_dict()
        elif self._ibm_token_present():
            warnings.append(
                "IBM_QUANTUM_TOKEN is configured, but hardware execution was not explicitly enabled."
            )

        return self._run_in_repo_bell_pair(
            shots=shots,
            seed=seed,
            requested_hardware=self.hardware_enabled,
            warnings=warnings,
        ).to_dict()

    def _try_ibm_hardware(
        self,
        *,
        shots: int,
        warnings: list[str],
    ) -> ActualQuantumResult | None:
        token = os.environ.get("IBM_QUANTUM_TOKEN") or os.environ.get("IBM_QUANTUM_API_KEY")
        if not token:
            warnings.append(
                "Hardware execution requested, but IBM_QUANTUM_TOKEN is not configured."
            )
            return None

        try:
            qiskit_circuit = self._build_qiskit_bell_pair()
            from .vendors.ibm_quantum import IBMQuantumBackend

            backend = IBMQuantumBackend(api_token=token, backend_name=self.backend_name)
            raw_result = backend.run_circuit(qiskit_circuit, shots=shots)
        except Exception as exc:
            warnings.append(f"IBM Quantum execution unavailable; using simulator fallback: {exc}")
            return None

        counts = _normalize_counts(raw_result.get("counts", {}), shots)
        method = str(raw_result.get("method", "unknown"))
        hardware_used = method == "ibm_hardware"
        if not hardware_used:
            warnings.append(
                f"IBM hardware was requested, but backend returned {method}; treating result as non-hardware."
            )

        probabilities = _counts_to_probabilities(counts, shots)
        return ActualQuantumResult(
            circuit_name="bell_pair",
            counts=counts,
            probabilities=probabilities,
            shots=shots,
            provider="ibm",
            backend=str(raw_result.get("backend", self.backend_name or "unknown")),
            execution_mode="remote_hardware_if_configured" if hardware_used else "qiskit_aer_simulator",
            hardware_enabled=hardware_used,
            requested_hardware=True,
            entanglement_score=_bell_entanglement_score(probabilities),
            warnings=tuple(warnings),
            raw_result=dict(raw_result),
        )

    def _run_in_repo_bell_pair(
        self,
        *,
        shots: int,
        seed: int | None,
        requested_hardware: bool,
        warnings: list[str],
    ) -> ActualQuantumResult:
        from quantum_simulator import QuantumCircuit, simulate

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        circuit.measure_all()
        raw_result = simulate(circuit, shots=shots, seed=seed, backend="statevector")
        counts = _normalize_counts(raw_result.get("counts", {}), shots)
        probabilities = _counts_to_probabilities(counts, shots)
        return ActualQuantumResult(
            circuit_name="bell_pair",
            counts=counts,
            probabilities=probabilities,
            shots=shots,
            provider="argus_in_repo",
            backend="statevector",
            execution_mode="classical_statevector_simulation",
            hardware_enabled=False,
            requested_hardware=requested_hardware,
            entanglement_score=_bell_entanglement_score(probabilities),
            warnings=tuple(warnings),
            raw_result=dict(raw_result),
        )

    @staticmethod
    def _build_qiskit_bell_pair() -> Any:
        qiskit = importlib.import_module("qiskit")
        QuantumCircuit = getattr(qiskit, "QuantumCircuit")

        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])
        return circuit

    @staticmethod
    def _ibm_token_present() -> bool:
        return bool(os.environ.get("IBM_QUANTUM_TOKEN") or os.environ.get("IBM_QUANTUM_API_KEY"))


def _normalize_counts(raw_counts: Any, shots: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    if isinstance(raw_counts, dict):
        for state, count in raw_counts.items():
            bitstring = str(state).replace(" ", "")
            if bitstring.startswith("0b"):
                bitstring = bitstring[2:]
            if bitstring:
                counts[bitstring] = counts.get(bitstring, 0) + int(count)
    if not counts:
        counts = {"00": shots}
    return counts


def _counts_to_probabilities(counts: dict[str, int], shots: int) -> dict[str, float]:
    total = max(sum(counts.values()), shots, 1)
    return {state: count / total for state, count in sorted(counts.items())}


def _bell_entanglement_score(probabilities: dict[str, float]) -> float:
    correlated = probabilities.get("00", 0.0) + probabilities.get("11", 0.0)
    anticorrelated = probabilities.get("01", 0.0) + probabilities.get("10", 0.0)
    return max(0.0, min(1.0, correlated - anticorrelated))


__all__ = ["ActualQuantumResult", "ActualQuantumRunner"]
