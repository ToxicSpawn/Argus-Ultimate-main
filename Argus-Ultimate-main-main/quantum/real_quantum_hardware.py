"""
Real quantum hardware integration (safe stub).

The original version of this module in this repo contained unresolved merge
markers and auto-generated corruption. This implementation is intentionally
minimal and dependency-safe: it can be imported without Qiskit/Braket/etc.

If you want real hardware support, extend `RealQuantumHardwareManager` with the
provider SDKs and add the required dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


try:
    # Optional dependency
    from qiskit import QuantumCircuit  # type: ignore

    QISKIT_AVAILABLE = True
except Exception:
    QuantumCircuit = Any  # type: ignore
    QISKIT_AVAILABLE = False


@dataclass
class QuantumHardwareProvider:
    name: str
    provider_type: str  # e.g. "ibm", "rigetti", "ionq", "azure", "aws"
    api_key: Optional[str] = None
    region: str = "us-east-1"
    max_qubits: int = 32
    is_available: bool = True


@dataclass
class QuantumHardwareJob:
    job_id: str
    provider: str
    backend: str
    circuit: Any
    shots: int = 4096
    status: str = "queued"
    submitted_at: datetime = field(default_factory=datetime.now)
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class RealQuantumHardwareManager:
    """
    Baseline manager that tracks providers/jobs.

    This does not execute on real hardware; it provides a stable API surface so
    other modules can import and integrate without crashing.
    """

    def __init__(self) -> None:
        self.providers: Dict[str, QuantumHardwareProvider] = {}
        self.jobs: Dict[str, QuantumHardwareJob] = {}

    def register_provider(self, provider: QuantumHardwareProvider) -> None:
        self.providers[provider.name] = provider

    def submit_job(self, provider_name: str, backend: str, circuit: Any, shots: int = 4096) -> QuantumHardwareJob:
        if provider_name not in self.providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        job_id = f"job_{provider_name}_{int(datetime.now().timestamp())}"
        job = QuantumHardwareJob(job_id=job_id, provider=provider_name, backend=backend, circuit=circuit, shots=int(shots))
        self.jobs[job_id] = job
        # No real execution in this baseline version.
        job.status = "unsupported"
        job.error_message = "Real hardware execution not implemented in this baseline module."
        logger.warning(job.error_message)
        return job

