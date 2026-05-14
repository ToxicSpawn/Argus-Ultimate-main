"""
Native gate-set transpilation for vendor-specific hardware.

Each quantum hardware vendor exposes a different native gate set:

- **IBM**: {RZ, SX, X, CNOT, CZ}
- **IonQ**: {RX, RY, RZ, MS} (Mølmer-Sørensen entangling gate)
- **Rigetti**: {RX, RZ, CZ, XY}
- **Quantinuum**: {RX, RZ, ZZ}

This module provides transpilers that decompose arbitrary circuits into
each vendor's native gate set.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quantum_simulator import GateType, Operation, QuantumCircuit


# ═════════════════════════════════════════════════════════════════════════════
# IBM native: {RZ, SX, X, CNOT}
# ═════════════════════════════════════════════════════════════════════════════


def transpile_to_ibm_native(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Decompose into IBM native gate set:
        RZ, SX (= √X), X, CNOT
    """
    n = circuit.num_qubits
    out = QuantumCircuit(n)

    for op in circuit.operations:
        g = op.gate
        t = op.targets

        if g == GateType.MEASURE_ALL:
            out._ops.append(op)
            continue

        # 1q decompositions
        if g == GateType.H:
            # H = RZ(π) SX RZ(π/2) (up to global phase)
            out.rz(np.pi / 2.0, t[0])
            # SX is √X — approximate as RX(π/2)
            out.rx(np.pi / 2.0, t[0])
            out.rz(np.pi / 2.0, t[0])
        elif g == GateType.X:
            out.x(t[0])
        elif g == GateType.Y:
            # Y = X · Z up to phase, decompose to RZ + X
            out.rz(-np.pi, t[0])
            out.x(t[0])
        elif g == GateType.Z:
            out.rz(np.pi, t[0])
        elif g == GateType.RX:
            # RX(θ) = RZ(-π/2) RY(θ) RZ(π/2) ... or via Euler decomposition
            out.rz(-np.pi / 2.0, t[0])
            out.rx(op.params[0], t[0])  # native
            out.rz(np.pi / 2.0, t[0])
        elif g == GateType.RY:
            out.rx(np.pi / 2.0, t[0])
            out.rz(op.params[0], t[0])
            out.rx(-np.pi / 2.0, t[0])
        elif g == GateType.RZ:
            out.rz(op.params[0], t[0])
        elif g == GateType.S:
            out.rz(np.pi / 2.0, t[0])
        elif g == GateType.SDG:
            out.rz(-np.pi / 2.0, t[0])
        elif g == GateType.T:
            out.rz(np.pi / 4.0, t[0])
        elif g == GateType.TDG:
            out.rz(-np.pi / 4.0, t[0])

        # 2q decompositions
        elif g == GateType.CNOT:
            out.cnot(t[0], t[1])
        elif g == GateType.CZ:
            # CZ = H_target CNOT H_target
            out.rz(np.pi / 2.0, t[1])
            out.rx(np.pi / 2.0, t[1])
            out.rz(np.pi / 2.0, t[1])
            out.cnot(t[0], t[1])
            out.rz(np.pi / 2.0, t[1])
            out.rx(np.pi / 2.0, t[1])
            out.rz(np.pi / 2.0, t[1])
        else:
            # Unknown gate — keep as-is
            out._ops.append(op)

    return out


# ═════════════════════════════════════════════════════════════════════════════
# IonQ native: {RX, RY, RZ, MS}
# ═════════════════════════════════════════════════════════════════════════════


def transpile_to_ionq_native(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Decompose into IonQ native gate set:
        RX, RY, RZ, MS (Mølmer-Sørensen ≈ RXX(θ))

    IonQ has full all-to-all connectivity, so no SWAPs needed.
    """
    n = circuit.num_qubits
    out = QuantumCircuit(n)

    for op in circuit.operations:
        g = op.gate
        t = op.targets

        if g == GateType.MEASURE_ALL:
            out._ops.append(op)
            continue

        if g == GateType.H:
            # H = RY(π/2) X
            out.ry(np.pi / 2.0, t[0])
            out.rx(np.pi, t[0])
        elif g in (GateType.X, GateType.Y, GateType.Z):
            out._ops.append(op)
        elif g == GateType.RX:
            out.rx(op.params[0], t[0])
        elif g == GateType.RY:
            out.ry(op.params[0], t[0])
        elif g == GateType.RZ:
            out.rz(op.params[0], t[0])
        elif g == GateType.CNOT:
            # CNOT via MS gate: CNOT = (I⊗RY(-π/2)) RXX(π/2) (RX(-π/2)⊗RZ(π/2))
            out.ry(-np.pi / 2.0, t[1])
            out.rxx(np.pi / 2.0, t[0], t[1])
            out.rx(-np.pi / 2.0, t[0])
            out.rz(np.pi / 2.0, t[1])
            out.ry(np.pi / 2.0, t[1])
        elif g == GateType.RXX:
            out.rxx(op.params[0], t[0], t[1])
        else:
            out._ops.append(op)

    return out


# ═════════════════════════════════════════════════════════════════════════════
# Rigetti native: {RX(±π/2), RZ, CZ, XY}
# ═════════════════════════════════════════════════════════════════════════════


def transpile_to_rigetti_native(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Decompose into Rigetti native gate set:
        RX(±π/2), RZ(any), CZ
    """
    n = circuit.num_qubits
    out = QuantumCircuit(n)

    for op in circuit.operations:
        g = op.gate
        t = op.targets

        if g == GateType.MEASURE_ALL:
            out._ops.append(op)
            continue

        if g == GateType.H:
            out.rz(np.pi / 2.0, t[0])
            out.rx(np.pi / 2.0, t[0])
            out.rz(np.pi / 2.0, t[0])
        elif g == GateType.X:
            out.rx(np.pi / 2.0, t[0])
            out.rx(np.pi / 2.0, t[0])
        elif g == GateType.Z:
            out.rz(np.pi, t[0])
        elif g == GateType.RZ:
            out.rz(op.params[0], t[0])
        elif g == GateType.CZ:
            out.cz(t[0], t[1])
        elif g == GateType.CNOT:
            # CNOT = (H⊗I) CZ (H⊗I) on target qubit
            out.rz(np.pi / 2.0, t[1])
            out.rx(np.pi / 2.0, t[1])
            out.rz(np.pi / 2.0, t[1])
            out.cz(t[0], t[1])
            out.rz(np.pi / 2.0, t[1])
            out.rx(np.pi / 2.0, t[1])
            out.rz(np.pi / 2.0, t[1])
        else:
            out._ops.append(op)

    return out
