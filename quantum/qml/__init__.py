
"""
Quantum Machine Learning (import-safe facade).

Historically, the QML package was stubbed to satisfy compileall, and the old
`__init__.py` imported those stubs in a way that could crash at import-time.

This facade always exports dependency-light, runnable implementations from
`quantum.qml.models` (with optional sklearn acceleration if installed).
"""

from __future__ import annotations

from quantum.qml.models import (
    QuantumBoltzmannMachine,
    QuantumKernel,
    QuantumNeuralNetwork,
    QuantumSVM,
    VariationalQuantumClassifier,
)

__version__ = "1.1.0"

__all__ = [
    "QuantumBoltzmannMachine",
    "QuantumKernel",
    "QuantumNeuralNetwork",
    "QuantumSVM",
    "VariationalQuantumClassifier",
]
