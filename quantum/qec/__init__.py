"""
Quantum Error Correction toolkit.

- ``repetition_code``: 3-qubit bit-flip repetition code
- ``steane_code``: [[7,1,3]] Steane code (corrects single Pauli errors)
- ``surface_code``: distance-3 surface code (planar topological code)
- ``magic_state``: Bravyi-Kitaev 15-to-1 magic state distillation

All codes use only gates supported by the in-repo simulator.
"""

from .repetition_code import RepetitionCode
from .steane_code import SteaneCode
from .surface_code import SurfaceCode, MWPMDecoder
from .magic_state import (
    noisy_magic_state,
    distill_15_to_1,
    cascaded_distillation,
    MAGIC_STATE,
)

__all__ = [
    "RepetitionCode",
    "SteaneCode",
    "SurfaceCode",
    "MWPMDecoder",
    "MAGIC_STATE",
    "noisy_magic_state",
    "distill_15_to_1",
    "cascaded_distillation",
]
