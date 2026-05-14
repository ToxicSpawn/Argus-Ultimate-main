"""
ARGUS Ultimate quantum module.

The canonical public surface is import-safe and honest by default: most
capabilities are classical simulation or quantum-inspired numerical methods,
with hardware adapters loaded only when explicitly configured.
"""

from .status import (
    CANONICAL_CAPABILITIES,
    QuantumCapability,
    QuantumStatusReport,
    quantum_status_report,
    recommended_max_statevector_qubits,
)
from .canonical import (
    ArgusQuantumFacade,
    QuantumExecutionMetadata,
    get_quantum_facade,
)
from .actual_quantum import ActualQuantumResult, ActualQuantumRunner
from .local_quantum import LocalQuantumRunner, LocalQuantumResult, HAS_QISKIT_AER
from .observables import (
    expectation_value,
    variance,
    von_neumann_entropy,
    purity,
    concurrence,
    fidelity,
    schmidt_coefficients,
    pauli_z,
    pauli_x,
    pauli_y,
    pauli_zz,
    probabilities_from_statevector,
    amplitudes_from_statevector,
    density_matrix,
)

# Core quantum components
try:
    from .cost_optimizer import CostOptimizer
    from .orchestrator import QuantumOrchestrator, VendorType
except Exception:
    CostOptimizer = None
    QuantumOrchestrator = None
    VendorType = None

# Production quantum simulator integration
try:
    from .production_quantum_simulator import (
        optimize_portfolio_with_quantum,
        discover_strategy_with_quantum,
        analyze_risk_with_quantum,
        get_quantum_simulator_status,
        ARGUSQuantumSimulator
    )
except Exception:
    # Fallback if production simulator not available
    optimize_portfolio_with_quantum = None
    discover_strategy_with_quantum = None
    analyze_risk_with_quantum = None
    get_quantum_simulator_status = None
    ARGUSQuantumSimulator = None

def _not_available(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise RuntimeError("Quantum module optional component not available in this environment.")

# Provide safe callable fallbacks rather than failing imports at import-time.
if optimize_portfolio_with_quantum is None:
    optimize_portfolio_with_quantum = _not_available
if discover_strategy_with_quantum is None:
    discover_strategy_with_quantum = _not_available
if analyze_risk_with_quantum is None:
    analyze_risk_with_quantum = _not_available
if get_quantum_simulator_status is None:
    get_quantum_simulator_status = _not_available

# Legacy quantum components are intentionally NOT auto-imported because many files
# in this tree are experimental and may be syntax-corrupted. Import explicitly if needed.

__all__ = [
    # Core components
    "CostOptimizer",
    "QuantumOrchestrator",
    "VendorType",

    # Production quantum simulator
    "optimize_portfolio_with_quantum",
    "discover_strategy_with_quantum",
    "analyze_risk_with_quantum",
    "get_quantum_simulator_status",
    "ARGUSQuantumSimulator",

    # Honest capability reporting
    "CANONICAL_CAPABILITIES",
    "QuantumCapability",
    "QuantumStatusReport",
    "quantum_status_report",
    "recommended_max_statevector_qubits",
    "ArgusQuantumFacade",
    "QuantumExecutionMetadata",
    "get_quantum_facade",
    "ActualQuantumResult",
    "ActualQuantumRunner",
    "LocalQuantumRunner",
    "LocalQuantumResult",
    "HAS_QISKIT_AER",
    "expectation_value",
    "variance",
    "von_neumann_entropy",
    "purity",
    "concurrence",
    "fidelity",
    "schmidt_coefficients",
    "pauli_z",
    "pauli_x",
    "pauli_y",
    "pauli_zz",
    "probabilities_from_statevector",
    "amplitudes_from_statevector",
    "density_matrix",
]
