"""
OpenQASM 2.0 and 3.0 import / export for ARGUS QuantumCircuit.

OpenQASM is the standard textual format for quantum circuits, supported by
Qiskit, Cirq, and most quantum hardware vendors. Round-tripping enables
interop with the broader quantum software ecosystem.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

import numpy as np

from quantum_simulator import GateType, Operation, QuantumCircuit


# ═════════════════════════════════════════════════════════════════════════════
# Export to OpenQASM
# ═════════════════════════════════════════════════════════════════════════════


# Mapping from our GateType to OpenQASM 2.0 gate names
_QASM2_GATE_MAP = {
    GateType.H: "h",
    GateType.X: "x",
    GateType.Y: "y",
    GateType.Z: "z",
    GateType.S: "s",
    GateType.SDG: "sdg",
    GateType.T: "t",
    GateType.TDG: "tdg",
    GateType.RX: "rx",
    GateType.RY: "ry",
    GateType.RZ: "rz",
    GateType.U3: "u3",
    GateType.PHASE: "p",
    GateType.CNOT: "cx",
    GateType.CZ: "cz",
    GateType.SWAP: "swap",
    GateType.CCX: "ccx",
}


def to_qasm2(circuit: QuantumCircuit) -> str:
    """
    Export a QuantumCircuit to OpenQASM 2.0.

    Returns the QASM 2.0 source as a single string.
    """
    n = circuit.num_qubits
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{n}];",
        f"creg c[{n}];",
    ]

    for op in circuit.operations:
        if op.gate == GateType.MEASURE_ALL:
            for q in range(n):
                lines.append(f"measure q[{q}] -> c[{q}];")
            continue

        gate_name = _QASM2_GATE_MAP.get(op.gate)
        if gate_name is None:
            # Unsupported gate — emit as a comment
            lines.append(f"// unsupported: {op.gate}")
            continue

        # Format parameters
        if op.params:
            param_str = "(" + ",".join(f"{p}" for p in op.params) + ")"
        else:
            param_str = ""

        # Format qubit operands
        qubits_str = ",".join(f"q[{t}]" for t in op.targets)
        lines.append(f"{gate_name}{param_str} {qubits_str};")

    return "\n".join(lines)


def to_qasm3(circuit: QuantumCircuit) -> str:
    """
    Export a QuantumCircuit to OpenQASM 3.0.

    QASM 3 is similar to QASM 2 but uses different syntax for declarations.
    """
    n = circuit.num_qubits
    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"qubit[{n}] q;",
        f"bit[{n}] c;",
    ]

    for op in circuit.operations:
        if op.gate == GateType.MEASURE_ALL:
            lines.append("c = measure q;")
            continue

        gate_name = _QASM2_GATE_MAP.get(op.gate)
        if gate_name is None:
            lines.append(f"// unsupported: {op.gate}")
            continue

        if op.params:
            param_str = "(" + ",".join(f"{p}" for p in op.params) + ")"
        else:
            param_str = ""
        qubits_str = ",".join(f"q[{t}]" for t in op.targets)
        lines.append(f"{gate_name}{param_str} {qubits_str};")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Import from OpenQASM 2.0
# ═════════════════════════════════════════════════════════════════════════════


_QASM2_REVERSE_MAP = {v: k for k, v in _QASM2_GATE_MAP.items()}


def from_qasm2(source: str) -> QuantumCircuit:
    """
    Parse OpenQASM 2.0 source into a QuantumCircuit.

    Supports the gates in ``_QASM2_GATE_MAP`` plus measure statements.
    """
    n_qubits = 0
    operations: List[Operation] = []

    qreg_re = re.compile(r"qreg\s+\w+\s*\[\s*(\d+)\s*\]\s*;")
    gate_re = re.compile(
        r"(\w+)(?:\(([^)]*)\))?\s+(?:q\[(\d+)\])(?:\s*,\s*q\[(\d+)\])?(?:\s*,\s*q\[(\d+)\])?\s*;"
    )
    measure_re = re.compile(r"measure\s+q\[(\d+)\]\s*->\s*c\[(\d+)\]\s*;")

    for line in source.split("\n"):
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("OPENQASM") or line.startswith("include") or line.startswith("creg"):
            continue
        m = qreg_re.match(line)
        if m:
            n_qubits = int(m.group(1))
            continue
        m = measure_re.match(line)
        if m:
            # Track measure
            continue
        m = gate_re.match(line)
        if m:
            gate_name = m.group(1)
            params_str = m.group(2)
            q1 = int(m.group(3))
            q2 = int(m.group(4)) if m.group(4) else None
            q3 = int(m.group(5)) if m.group(5) else None

            gate_type = _QASM2_REVERSE_MAP.get(gate_name)
            if gate_type is None:
                continue

            params: tuple = ()
            if params_str:
                params = tuple(float(p.strip()) for p in params_str.split(","))

            targets = (q1,)
            if q2 is not None:
                targets = (q1, q2)
            if q3 is not None:
                targets = (q1, q2, q3)
            operations.append(Operation(gate_type, targets, params))

    qc = QuantumCircuit(max(1, n_qubits))
    qc._ops = operations
    return qc
