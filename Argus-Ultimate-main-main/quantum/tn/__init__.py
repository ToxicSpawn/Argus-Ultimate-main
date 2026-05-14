"""
Tensor network algorithms.

- ``dmrg``: Density Matrix Renormalization Group — variational MPS ground state
- ``tebd``: Time-Evolving Block Decimation — real-time MPS evolution

For 1D systems with low entanglement, MPS-based methods often beat both
statevector simulation and VQE.
"""

from .dmrg import DMRG, build_ising_mpo
from .tebd import TEBD

__all__ = ["DMRG", "build_ising_mpo", "TEBD"]
