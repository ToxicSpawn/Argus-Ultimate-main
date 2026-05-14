"""
Theoretical / research-grade quantum primitives.

- ``random_circuits``: Random circuit sampling (Google supremacy class)
- ``iqp``: Instantaneous Quantum Polynomial circuits
- ``haar_random``: Haar-random unitary sampling
- ``boson_sampling``: Boson Sampling (and Gaussian variant)
- ``unitary_design``: k-designs (approximate Haar)
"""

from .random_circuits import random_circuit, brick_wall_random_circuit
from .iqp import iqp_circuit, iqp_sample
from .haar_random import haar_random_unitary, haar_random_state
from .boson_sampling import boson_sample, gaussian_boson_sample
from .unitary_design import unitary_2_design

__all__ = [
    "random_circuit",
    "brick_wall_random_circuit",
    "iqp_circuit",
    "iqp_sample",
    "haar_random_unitary",
    "haar_random_state",
    "boson_sample",
    "gaussian_boson_sample",
    "unitary_2_design",
]
