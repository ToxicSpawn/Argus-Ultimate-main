"""
Hardware-aware features for quantum circuit compilation and control.

- ``routing``: SABRE / look-ahead routing for limited connectivity
- ``grape``: pulse-level optimal control (GRAPE)
- ``dynamical_decoupling``: XY-4 / CPMG / KDD sequences
- ``composite_pulses``: BB1 / SCROFULOUS / KDD error suppression
- ``native_gates``: vendor gate-set transpilation
"""

from .routing import SabreRouter, LookAheadRouter
from .dynamical_decoupling import (
    insert_xy4_dd,
    insert_cpmg_dd,
    insert_kdd_dd,
)
from .composite_pulses import (
    bb1_pulse,
    scrofulous_pulse,
    knill_pulse,
)
from .grape import GRAPEOptimizer
from .native_gates import (
    transpile_to_ibm_native,
    transpile_to_ionq_native,
    transpile_to_rigetti_native,
)

__all__ = [
    "SabreRouter",
    "LookAheadRouter",
    "insert_xy4_dd",
    "insert_cpmg_dd",
    "insert_kdd_dd",
    "bb1_pulse",
    "scrofulous_pulse",
    "knill_pulse",
    "GRAPEOptimizer",
    "transpile_to_ibm_native",
    "transpile_to_ionq_native",
    "transpile_to_rigetti_native",
]
