"""
Quantum repeater for long-distance entanglement distribution.

A quantum repeater chain extends entanglement over distances longer than
the direct photon transmission distance by chaining short Bell pairs and
performing entanglement swapping at intermediate nodes.

Reference
---------
Briegel, Dür, Cirac, Zoller, "Quantum repeaters: the role of imperfect local
operations in quantum communication," PRL 81, 5932 (1998)
"""

from __future__ import annotations

from typing import Any, Dict, List

from quantum_simulator import QuantumCircuit


class QuantumRepeater:
    """
    A simple quantum repeater chain.

    Builds N short-distance Bell pairs in sequence, then performs
    entanglement swapping at each interior node.

    Parameters
    ----------
    n_segments : int
        Number of segments in the repeater chain. Total qubits = 2 * n.
    """

    def __init__(self, n_segments: int) -> None:
        if n_segments < 2:
            raise ValueError("n_segments must be >= 2")
        self.n_segments = int(n_segments)

    def build_chain(self) -> QuantumCircuit:
        """
        Build the entanglement-swap circuit.

        After this circuit, the end qubits (0 and 2*n-1) are entangled.
        """
        n_qubits = 2 * self.n_segments
        qc = QuantumCircuit(n_qubits)

        # Step 1: prepare a Bell pair on each segment
        for s in range(self.n_segments):
            q0 = 2 * s
            q1 = 2 * s + 1
            qc.h(q0)
            qc.cnot(q0, q1)

        # Step 2: entanglement swap at each interior node
        for s in range(self.n_segments - 1):
            # Bell measurement on (2s+1, 2(s+1)) = (2s+1, 2s+2)
            qb = 2 * s + 1
            qc_inner = 2 * s + 2
            qc.cnot(qb, qc_inner)
            qc.h(qb)

        return qc

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_segments": self.n_segments,
            "n_qubits": 2 * self.n_segments,
            "method": "briegel_repeater_chain",
        }
