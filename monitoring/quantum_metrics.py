"""
Quantum-specific metrics collector.

Records:
- Per-cycle quantum gate counts
- Per-cycle quantum simulation wall-clock
- QAOA / VQE convergence trajectories
- VaR decisions (CVaR + size_factor) for audit trail
- Active quantum strategies (annealer mask)

Snapshot is exposed via the existing health server HTTP endpoint.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# QuantumMetricsCollector
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class _SimulationRecord:
    timestamp: float
    n_qubits: int
    n_gates: int
    wall_clock_ms: float
    backend: str


@dataclass
class _VarDecisionRecord:
    timestamp: float
    symbol: str
    cvar: float
    size_factor: float
    cap_applied: bool


class QuantumMetricsCollector:
    """
    Singleton metrics collector for quantum operations.

    Thread-safe; collects metrics from any code path that calls
    ``record_*``. Snapshot returns a JSON-serializable dict.
    """

    def __init__(self, max_history: int = 1000) -> None:
        self._lock = threading.Lock()
        self._max_history = int(max_history)

        self._simulations: Deque[_SimulationRecord] = deque(maxlen=max_history)
        self._gate_counts: Dict[str, int] = {}
        self._convergences: Dict[str, List[float]] = {}
        self._var_decisions: Deque[_VarDecisionRecord] = deque(maxlen=max_history)
        self._active_strategies: List[str] = []
        self._n_qaoa_runs = 0
        self._n_vqe_runs = 0
        self._n_grover_runs = 0
        self._n_hhl_runs = 0
        self._total_wall_clock_ms: float = 0.0
        self._start_time = time.time()

    # ── Recorders ────────────────────────────────────────────────────────────

    def record_simulation(
        self,
        n_qubits: int,
        n_gates: int,
        wall_clock_ms: float,
        backend: str = "local_simulator",
    ) -> None:
        with self._lock:
            self._simulations.append(_SimulationRecord(
                timestamp=time.time(),
                n_qubits=int(n_qubits),
                n_gates=int(n_gates),
                wall_clock_ms=float(wall_clock_ms),
                backend=str(backend),
            ))
            self._total_wall_clock_ms += float(wall_clock_ms)

    def record_gate(self, gate_type: str) -> None:
        with self._lock:
            self._gate_counts[gate_type] = self._gate_counts.get(gate_type, 0) + 1

    def record_convergence(self, problem: str, history: List[float]) -> None:
        with self._lock:
            self._convergences[str(problem)] = list(history[-50:])

    def record_qaoa_run(self) -> None:
        with self._lock:
            self._n_qaoa_runs += 1

    def record_vqe_run(self) -> None:
        with self._lock:
            self._n_vqe_runs += 1

    def record_grover_run(self) -> None:
        with self._lock:
            self._n_grover_runs += 1

    def record_hhl_run(self) -> None:
        with self._lock:
            self._n_hhl_runs += 1

    def record_var_decision(
        self,
        symbol: str,
        cvar: float,
        size_factor: float,
        cap_applied: bool = False,
    ) -> None:
        with self._lock:
            self._var_decisions.append(_VarDecisionRecord(
                timestamp=time.time(),
                symbol=str(symbol),
                cvar=float(cvar),
                size_factor=float(size_factor),
                cap_applied=bool(cap_applied),
            ))

    def record_active_strategies(self, strategies: List[str]) -> None:
        with self._lock:
            self._active_strategies = list(strategies)

    # ── Snapshot ─────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            n_sims = len(self._simulations)
            avg_wall_ms = (
                self._total_wall_clock_ms / n_sims if n_sims > 0 else 0.0
            )
            recent_var_caps = sum(
                1 for d in self._var_decisions if d.cap_applied
            )

            return {
                "uptime_seconds": time.time() - self._start_time,
                "n_simulations": n_sims,
                "n_qaoa_runs": self._n_qaoa_runs,
                "n_vqe_runs": self._n_vqe_runs,
                "n_grover_runs": self._n_grover_runs,
                "n_hhl_runs": self._n_hhl_runs,
                "total_wall_clock_ms": self._total_wall_clock_ms,
                "avg_simulation_wall_clock_ms": avg_wall_ms,
                "gate_counts": dict(self._gate_counts),
                "n_var_decisions": len(self._var_decisions),
                "n_var_caps_applied": recent_var_caps,
                "active_quantum_strategies": list(self._active_strategies),
                "convergences": {k: v for k, v in self._convergences.items()},
                "max_qubits_seen": (
                    max((r.n_qubits for r in self._simulations), default=0)
                ),
                "backends_used": list({r.backend for r in self._simulations}),
            }

    def reset(self) -> None:
        """Reset all counters. Used by tests."""
        with self._lock:
            self._simulations.clear()
            self._gate_counts.clear()
            self._convergences.clear()
            self._var_decisions.clear()
            self._active_strategies.clear()
            self._n_qaoa_runs = 0
            self._n_vqe_runs = 0
            self._n_grover_runs = 0
            self._n_hhl_runs = 0
            self._total_wall_clock_ms = 0.0
            self._start_time = time.time()


# ═════════════════════════════════════════════════════════════════════════════
# Singleton instance
# ═════════════════════════════════════════════════════════════════════════════


_INSTANCE: Optional[QuantumMetricsCollector] = None


def get_quantum_metrics() -> QuantumMetricsCollector:
    """Get the singleton metrics collector."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = QuantumMetricsCollector()
    return _INSTANCE
