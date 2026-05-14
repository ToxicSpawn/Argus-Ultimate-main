"""
IBM provider compatibility shim.

This module redirects to quantum.vendors.ibm_quantum which contains
the real IBM Qiskit Runtime integration with graceful fallbacks.

Usage:
    from quantum.vendors.ibm_quantum import IBMQuantumBackend
"""

from __future__ import annotations

from quantum.vendors.ibm_quantum import IBMQuantumBackend

__all__ = ["IBMQuantumBackend"]
