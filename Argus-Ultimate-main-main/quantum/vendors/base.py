"""
Quantum vendor base interfaces (restored, dependency-light).

The original vendor implementations were stubbed during repo repair. This file
defines a small, usable API so other modules can integrate against it without
requiring any specific quantum SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


@dataclass(frozen=True)
class QuantumJobRequest:
    vendor: str
    problem_type: str
    payload: Dict[str, Any]
    shots: int = 1000


@dataclass(frozen=True)
class QuantumJobResult:
    job_id: str
    vendor: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cost_estimate_usd: float = 0.0


class QuantumVendor(Protocol):
    """
    Minimal vendor protocol.
    """

    name: str

    def available(self) -> bool: ...

    def estimate_cost_usd(self, *, shots: int, problem_type: str) -> float: ...

    def submit(self, req: QuantumJobRequest) -> QuantumJobResult: ...


class SimulatorVendor:
    """
    Always-available "vendor" that pretends to run jobs locally.
    """

    name = "simulator"

    def available(self) -> bool:
        return True

    def estimate_cost_usd(self, *, shots: int, problem_type: str) -> float:
        _ = (shots, problem_type)
        return 0.0

    def submit(self, req: QuantumJobRequest) -> QuantumJobResult:
        return QuantumJobResult(
            job_id=f"sim_{id(req)}",
            vendor=self.name,
            status="completed",
            result={"note": "simulated vendor result", "problem_type": req.problem_type},
            cost_estimate_usd=0.0,
        )

