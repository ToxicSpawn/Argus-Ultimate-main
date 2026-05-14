"""Quantum unified stubs — moved from root."""
from __future__ import annotations

from typing import Any


class QuantumCircuitStub:
    """Stub for quantum circuit operations."""
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits

    def run(self, shots: int = 1024) -> dict[str, Any]:
        return {"counts": {}, "n_qubits": self.n_qubits, "shots": shots}
