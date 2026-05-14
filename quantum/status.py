from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from typing import Any


@dataclass(frozen=True)
class QuantumCapability:
    name: str
    module: str
    status: str
    execution_mode: str
    requires: tuple[str, ...] = ()
    max_recommended_qubits: int | None = None
    honest_claim: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QuantumStatusReport:
    supported_capabilities: list[QuantumCapability]
    optional_dependencies: dict[str, bool]
    default_execution_mode: str
    hardware_enabled: bool
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported_capabilities": [cap.to_dict() for cap in self.supported_capabilities],
            "optional_dependencies": dict(self.optional_dependencies),
            "default_execution_mode": self.default_execution_mode,
            "hardware_enabled": self.hardware_enabled,
            "warnings": list(self.warnings),
        }


CANONICAL_CAPABILITIES: tuple[QuantumCapability, ...] = (
    QuantumCapability(
        name="actual_bell_pair_execution",
        module="quantum.actual_quantum",
        status="hardware_ready",
        execution_mode="remote_hardware_if_configured",
        requires=("qiskit", "qiskit_ibm_runtime", "IBM_QUANTUM_TOKEN"),
        max_recommended_qubits=2,
        honest_claim=(
            "Runs a real Bell-state circuit on IBM Quantum only when hardware is "
            "explicitly enabled and credentials are configured; otherwise uses simulator fallback."
        ),
    ),
    QuantumCapability(
        name="qaoa_portfolio_subset",
        module="quantum.algorithms.qaoa",
        status="supported",
        execution_mode="classical_statevector_simulation",
        max_recommended_qubits=14,
        honest_claim=(
            "QAOA is simulated classically and is useful as a discrete subset "
            "optimizer/research benchmark; classical optimizers usually win for small portfolios."
        ),
    ),
    QuantumCapability(
        name="qmc_var_cvar",
        module="quantum.algorithms.quantum_monte_carlo",
        status="supported",
        execution_mode="quantum_inspired_classical_sobol",
        requires=("scipy",),
        honest_claim=(
            "Sobol quasi-Monte Carlo can reduce sampling variance; this is not "
            "a hardware quantum advantage."
        ),
    ),
    QuantumCapability(
        name="mlqae_var",
        module="quantum.algorithms.quantum_amplitude_estimation",
        status="supported",
        execution_mode="classical_statevector_simulation",
        max_recommended_qubits=12,
        honest_claim=(
            "MLQAE is implemented for methodological correctness; on classical "
            "simulation it is slower than direct empirical VaR."
        ),
    ),
    QuantumCapability(
        name="quantum_kernel_classifier",
        module="quantum.qml.quantum_kernel",
        status="supported",
        execution_mode="classical_kernel_simulation",
        honest_claim="Quantum kernels are nonlinear feature maps with O(n^2) training cost.",
    ),
    QuantumCapability(
        name="quantum_reservoir_timeseries",
        module="quantum.qml.quantum_reservoir",
        status="supported",
        execution_mode="classical_statevector_simulation",
        max_recommended_qubits=10,
        honest_claim=(
            "Reservoir dynamics provide a nonlinear feature expansion; no speedup is claimed."
        ),
    ),
    QuantumCapability(
        name="dwave_annealing_provider",
        module="quantum.vendors.dwave_provider",
        status="optional_hardware_adapter",
        execution_mode="remote_hardware_if_configured",
        requires=("dwave", "DWAVE_API_KEY"),
        honest_claim="Requires account/API key; use only for research or explicit hardware experiments.",
    ),
    QuantumCapability(
        name="ibm_quantum_provider",
        module="quantum.vendors.ibm_provider",
        status="optional_hardware_adapter",
        execution_mode="remote_hardware_if_configured",
        requires=("qiskit", "IBM_QUANTUM_TOKEN"),
        honest_claim="Requires account/API token; queue latency usually dominates trading use cases.",
    ),
    QuantumCapability(
        name="local_ghz_state",
        module="quantum.local_quantum",
        status="supported",
        execution_mode="classical_statevector_simulation",
        max_recommended_qubits=20,
        honest_claim=(
            "GHZ entangled state simulated on this machine via local quantum runtime; "
            "no hardware quantum advantage is claimed."
        ),
    ),
    QuantumCapability(
        name="local_w_state",
        module="quantum.local_quantum",
        status="supported",
        execution_mode="classical_statevector_simulation",
        max_recommended_qubits=20,
        honest_claim=(
            "W entangled state simulated on this machine via local quantum runtime; "
            "no hardware quantum advantage is claimed."
        ),
    ),
    QuantumCapability(
        name="local_parameterized_circuit",
        module="quantum.local_quantum",
        status="supported",
        execution_mode="classical_statevector_simulation",
        max_recommended_qubits=20,
        honest_claim=(
            "Parameterized circuit simulated on this machine; "
            "useful for local model training, not for physical quantum speedup."
        ),
    ),
    QuantumCapability(
        name="local_quantum_walk",
        module="quantum.optimization.quantum_walk",
        status="supported",
        execution_mode="classical_statevector_simulation",
        honest_claim=(
            "Quantum walk simulated on this machine for asset centrality and clustering; "
            "no hardware quantum advantage is claimed."
        ),
    ),
    QuantumCapability(
        name="trotter_evolution",
        module="quantum.algorithms.trotter",
        status="supported",
        execution_mode="classical_statevector_simulation",
        max_recommended_qubits=14,
        honest_claim=(
            "Trotter-Suzuki time evolution simulated on this machine; "
            "useful for quantum-enhanced Markov chain and regime modeling."
        ),
    ),
    QuantumCapability(
        name="grover_search",
        module="quantum.algorithms.grover",
        status="supported",
        execution_mode="classical_statevector_simulation",
        max_recommended_qubits=16,
        honest_claim=(
            "Grover's search simulated on this machine; "
            "quantum query speedup is real but classical simulation is O(N) per query."
        ),
    ),
    QuantumCapability(
        name="vqe_ising",
        module="quantum.algorithms.vqe",
        status="supported",
        execution_mode="classical_statevector_simulation",
        max_recommended_qubits=12,
        honest_claim=(
            "VQE simulated on this machine; "
            "useful for finding ground states of Ising/Pauli Hamiltonians."
        ),
    ),
    QuantumCapability(
        name="mps_tensor_network",
        module="quantum.simulators.mps_backend",
        status="supported",
        execution_mode="mps_tensor_network",
        max_recommended_qubits=30,
        honest_claim=(
            "MPS tensor network simulated on this machine; "
            "enables larger qubit counts for low-entanglement circuits."
        ),
    ),
    QuantumCapability(
        name="statevector_observables",
        module="quantum.observables",
        status="supported",
        execution_mode="classical_statevector_simulation",
        honest_claim=(
            "Statevector inspection and observable expectation values; "
            "research/diagnostics tool, not physical quantum advantage."
        ),
    ),
)


def quantum_status_report(*, hardware_enabled: bool = False) -> QuantumStatusReport:
    """Return an honest capability report for the canonical quantum surface."""
    deps = {
        "numpy": _available("numpy"),
        "scipy": _available("scipy"),
        "qiskit": _available("qiskit"),
        "qiskit_aer": _available("qiskit_aer"),
        "qiskit_ibm_runtime": _available("qiskit_ibm_runtime"),
        "pennylane": _available("pennylane"),
        "dwave": _available("dwave"),
    }
    warnings = [
        "Default mode is classical simulation or quantum-inspired classical algorithms.",
        "Do not treat quantum outputs as live-trading authority; route through risk gates.",
        "No current module proves hardware quantum advantage for Argus trading decisions.",
    ]
    if hardware_enabled:
        warnings.append(
            "Hardware execution was requested; require explicit provider credentials and paper/backtest validation."
        )
    return QuantumStatusReport(
        supported_capabilities=list(CANONICAL_CAPABILITIES),
        optional_dependencies=deps,
        default_execution_mode="classical_simulation",
        hardware_enabled=bool(hardware_enabled),
        warnings=warnings,
    )


def recommended_max_statevector_qubits(memory_gb: float = 16.0, precision_bytes: int = 16) -> int:
    """Estimate safe dense statevector qubits for available memory.

    Uses a conservative 50% memory budget because simulators need temporary
    buffers in addition to the statevector itself.
    """
    if memory_gb <= 0:
        raise ValueError("memory_gb must be positive")
    if precision_bytes <= 0:
        raise ValueError("precision_bytes must be positive")
    usable_bytes = memory_gb * (1024 ** 3) * 0.5
    states = usable_bytes / precision_bytes
    qubits = 0
    while (1 << (qubits + 1)) <= states:
        qubits += 1
    return max(qubits, 1)


def _available(module_name: str) -> bool:
    return find_spec(module_name) is not None
