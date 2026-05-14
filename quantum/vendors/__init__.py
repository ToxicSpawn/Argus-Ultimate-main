"""
Quantum vendor clients for ARGUS.

Provides real hardware integration with D-Wave and IBM Quantum,
plus classical fallbacks that work without any API keys or SDKs.

Vendor modules:
    dwave_solver        - D-Wave Ocean SDK (LeapHybrid, Neal SA, local annealing)
    ibm_quantum         - IBM Qiskit Runtime (hardware, Aer, classical)
    vendor_orchestrator - Routes to best backend with fallback chains
    base                - Base interfaces (QuantumVendor protocol, job request/result)
"""

from __future__ import annotations

from quantum.vendors.base import QuantumJobRequest, QuantumJobResult, QuantumVendor, SimulatorVendor

__all__ = [
    "QuantumJobRequest",
    "QuantumJobResult",
    "QuantumVendor",
    "SimulatorVendor",
]
