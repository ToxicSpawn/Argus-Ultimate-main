"""
quantum/simulators/quantum_simulator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Re-exports from the canonical quantum_simulator.py at the repo root.

This file exists for package-qualified imports like:
    from quantum.simulators.quantum_simulator import QuantumCircuit

The canonical implementation lives at:
    quantum_simulator.py (repo root)
"""

# Import everything from the canonical root module
from quantum_simulator import (  # noqa: F401
    QuantumCircuit,
    simulate,
    _simulate_statevector,
    NoiseModel,
    # Gate matrices
    _H, _X, _Y, _Z, _S, _T, _SDG, _TDG,
    _U3, _PHASE,
    _CNOT_matrix, _CZ_matrix, _SWAP_matrix, _CCX_matrix,
    _CRX_matrix, _CRY_matrix, _CRZ_matrix, _CSWAP_matrix,
    _ISWAP_matrix, _RXX_matrix, _RYY_matrix, _RZZ_matrix,
    # Observables
    pauli_z_observable, pauli_zz_observable,
    expval, gradient,
)
