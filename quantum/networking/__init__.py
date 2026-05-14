"""
Quantum networking primitives.

- ``teleportation``: quantum state teleportation between two parties via Bell pair
- ``entanglement_swapping``: extend entanglement across multiple parties
- ``quantum_repeater``: build long-distance entanglement via repeater chain
- ``secret_sharing``: quantum secret-sharing protocol
- ``superdense_coding``: send 2 classical bits via 1 qubit + Bell pair
"""

from .teleportation import quantum_teleport, build_bell_pair
from .entanglement_swap import entanglement_swapping, build_ghz
from .repeater import QuantumRepeater
from .secret_sharing import quantum_secret_share
from .superdense_coding import superdense_encode, superdense_decode

__all__ = [
    "quantum_teleport",
    "build_bell_pair",
    "entanglement_swapping",
    "build_ghz",
    "QuantumRepeater",
    "quantum_secret_share",
    "superdense_encode",
    "superdense_decode",
]
