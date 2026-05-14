"""
Quantum circuit and statevector visualization.

- ``draw_circuit_text``: ASCII circuit diagram
- ``statevector_bar_chart``: text-mode amplitude bar chart
- ``bloch_sphere_data``: extract (x, y, z) for Bloch sphere plotting
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from quantum_simulator import GateType, QuantumCircuit


# ═════════════════════════════════════════════════════════════════════════════
# Text-mode circuit diagram
# ═════════════════════════════════════════════════════════════════════════════


def draw_circuit_text(circuit: QuantumCircuit) -> str:
    """
    Render a circuit as an ASCII diagram.

    Each row is a qubit; each column is a moment of execution.
    """
    n = circuit.num_qubits
    rows: List[List[str]] = [[] for _ in range(n)]

    for op in circuit.operations:
        if op.gate == GateType.MEASURE_ALL:
            for q in range(n):
                rows[q].append("[M]")
            continue

        # Gate display label
        if op.gate in (GateType.H, GateType.X, GateType.Y, GateType.Z):
            label = f"[{op.gate.value}]"
        elif op.gate in (GateType.RX, GateType.RY, GateType.RZ):
            label = f"[{op.gate.value}({op.params[0]:.2f})]"
        elif op.gate == GateType.CNOT:
            # Special: ● for control, ⊕ for target
            for q in range(n):
                if q == op.targets[0]:
                    rows[q].append("[●]")
                elif q == op.targets[1]:
                    rows[q].append("[⊕]")
                else:
                    rows[q].append("---")
            continue
        elif op.gate == GateType.SWAP:
            for q in range(n):
                if q in op.targets:
                    rows[q].append("[X]")
                else:
                    rows[q].append("---")
            continue
        else:
            label = f"[{op.gate.value}]"

        # Apply to all targets
        for q in range(n):
            if q in op.targets:
                rows[q].append(label)
            else:
                rows[q].append("---")

    # Pad rows to equal length and join
    max_len = max((len(r) for r in rows), default=0)
    for r in rows:
        while len(r) < max_len:
            r.append("---")

    lines = []
    for q, row in enumerate(rows):
        lines.append(f"q{q}: " + "".join(row))
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Statevector visualization
# ═════════════════════════════════════════════════════════════════════════════


def statevector_bar_chart(state: np.ndarray, *, max_states: int = 16) -> str:
    """
    Render a text-mode bar chart of the statevector probabilities.

    Limits to ``max_states`` to keep output manageable.
    """
    probs = np.abs(state) ** 2
    n_states = len(probs)
    n_qubits = int(np.log2(n_states))

    # Pick top max_states
    top_indices = np.argsort(probs)[::-1][:max_states]

    lines = []
    max_p = float(probs[top_indices[0]])
    for idx in top_indices:
        p = float(probs[idx])
        bitstring = format(int(idx), f"0{n_qubits}b")
        bar_len = int(round(p / max(max_p, 1e-12) * 30))
        bar = "█" * bar_len + "·" * (30 - bar_len)
        lines.append(f"|{bitstring}⟩  {bar}  {p:.4f}")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Bloch sphere extraction
# ═════════════════════════════════════════════════════════════════════════════


def bloch_sphere_data(state: np.ndarray) -> Dict[str, float]:
    """
    Extract Bloch sphere coordinates (x, y, z) for a single-qubit state.

    For a state |ψ⟩ = α|0⟩ + β|1⟩:
        x = 2 Re(α* β)
        y = 2 Im(α* β)
        z = |α|² - |β|²

    Returns
    -------
    Dict[str, float]
        ``{"x", "y", "z", "magnitude"}``
    """
    state = np.asarray(state, dtype=np.complex128)
    if state.size != 2:
        raise ValueError(f"Bloch sphere requires 1-qubit state (2 amplitudes), got {state.size}")
    alpha = state[0]
    beta = state[1]
    x = float(2 * np.real(np.conj(alpha) * beta))
    y = float(2 * np.imag(np.conj(alpha) * beta))
    z = float(abs(alpha) ** 2 - abs(beta) ** 2)
    mag = float(np.sqrt(x * x + y * y + z * z))
    return {"x": x, "y": y, "z": z, "magnitude": mag}
