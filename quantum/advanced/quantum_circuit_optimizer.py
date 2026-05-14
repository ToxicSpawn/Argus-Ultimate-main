# pyright: reportMissingImports=false
"""Compatibility wrapper for the advanced circuit optimization module."""

from .advanced_quantum_circuit_optimization import (
    AdvancedQuantumCircuitOptimizer,
    CircuitOptimizationStep,
    CircuitProfilingReport,
    OptimizationObjective,
    QuantumCircuitMetrics,
    QuantumCircuitProfile,
    QuantumGateOperation,
    QuantumHardwareType,
    QubitAllocation,
)

__all__ = [
    "AdvancedQuantumCircuitOptimizer",
    "CircuitOptimizationStep",
    "CircuitProfilingReport",
    "OptimizationObjective",
    "QuantumCircuitMetrics",
    "QuantumCircuitProfile",
    "QuantumGateOperation",
    "QuantumHardwareType",
    "QubitAllocation",
]
