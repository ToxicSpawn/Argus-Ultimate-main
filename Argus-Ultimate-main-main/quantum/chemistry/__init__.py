"""
Quantum chemistry primitives.

Provides:
- Fermion-to-qubit mappings: Jordan-Wigner, Bravyi-Kitaev, parity
- Pauli grouping for measurement reduction
- Hubbard model and Heisenberg / TFIM Hamiltonian builders
- Active-space methods stub

Trading use: ARGUS uses these to encode many-body market models (e.g., a
crypto market with N assets becomes an N-fermion system; correlations are
Coulomb-like interactions).
"""

from .fermion_mapping import (
    jordan_wigner,
    bravyi_kitaev,
    parity_mapping,
    fermion_creation,
    fermion_annihilation,
)
from .hamiltonians import (
    hubbard_model,
    heisenberg_model,
    transverse_field_ising,
    pauli_grouping,
)

__all__ = [
    "jordan_wigner",
    "bravyi_kitaev",
    "parity_mapping",
    "fermion_creation",
    "fermion_annihilation",
    "hubbard_model",
    "heisenberg_model",
    "transverse_field_ising",
    "pauli_grouping",
]
