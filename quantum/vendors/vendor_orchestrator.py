"""
Quantum Vendor Orchestrator for ARGUS.

Routes quantum computation to the best available backend with full
fallback chains. Tracks usage, cost, and improvement vs classical.

Vendor priority for optimization problems:
    portfolio_qubo / signal_selection -> D-Wave (annealing) -> classical
    vqe -> IBM Quantum (gate-based) -> classical scipy
    general circuit -> IBM Quantum -> Aer -> classical

All operations degrade gracefully to classical when no hardware is available.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QuantumJobRecord:
    """Record of a single quantum job execution."""
    timestamp: str
    vendor: str
    problem_type: str
    method_used: str
    hardware_used: bool
    timing_ms: float
    energy: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class QuantumVendorOrchestrator:
    """
    Routes quantum computation to best available backend with graceful degradation.

    Initializes all available vendors (D-Wave, IBM Quantum) and tracks usage.
    Everything works without any API keys via classical fallbacks.
    """

    def __init__(self):
        self._dwave = None
        self._ibm = None
        self._local_sim = True  # Always available
        self._job_history: List[QuantumJobRecord] = []
        self._cost_tracker: Dict[str, float] = {
            "dwave": 0.0,
            "ibm": 0.0,
            "classical": 0.0,
        }

        # Init vendors (never fail)
        self._init_dwave()
        self._init_ibm()

        vendors_up = []
        if self._dwave is not None:
            vendors_up.append("dwave")
        if self._ibm is not None:
            vendors_up.append("ibm")
        vendors_up.append("classical")
        logger.info("QuantumVendorOrchestrator initialized: vendors=%s", vendors_up)

    # ------------------------------------------------------------------
    # Vendor init (never raises)
    # ------------------------------------------------------------------

    def _init_dwave(self):
        try:
            from quantum.vendors.dwave_solver import DWaveSolver
            self._dwave = DWaveSolver()
        except Exception as exc:
            logger.debug("D-Wave vendor unavailable: %s", exc)

    def _init_ibm(self):
        try:
            from quantum.vendors.ibm_quantum import IBMQuantumBackend
            self._ibm = IBMQuantumBackend()
        except Exception as exc:
            logger.debug("IBM Quantum vendor unavailable: %s", exc)

    # ------------------------------------------------------------------
    # Record keeping
    # ------------------------------------------------------------------

    def _record(self, vendor: str, problem_type: str, result: Dict[str, Any]):
        method = result.get("method", "unknown")
        hw = result.get("hardware_used", False)
        timing = result.get("timing_ms", result.get("execution_time_ms", 0.0))
        energy = result.get("energy")

        record = QuantumJobRecord(
            timestamp=datetime.utcnow().isoformat(),
            vendor=vendor,
            problem_type=problem_type,
            method_used=method,
            hardware_used=hw,
            timing_ms=timing,
            energy=energy,
        )
        self._job_history.append(record)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve_optimization(self, problem_type: str, **kwargs) -> Dict[str, Any]:
        """
        Route optimization to best available backend.

        Problem types:
            portfolio_qubo: Portfolio selection via QUBO
                kwargs: expected_returns, cov_matrix, risk_aversion, budget
            signal_selection: Signal subset selection via QUBO
                kwargs: confidences, max_signals, diversity_penalty
            vqe: Variational Quantum Eigensolver for portfolio
                kwargs: expected_returns, cov_matrix, n_layers, risk_aversion
            circuit: Run a raw quantum circuit
                kwargs: circuit, shots

        Returns:
            result dict with: result data, vendor_used, fallback_chain, timing
        """
        fallback_chain: List[str] = []
        result: Optional[Dict[str, Any]] = None

        if problem_type == "portfolio_qubo":
            result, fallback_chain = self._solve_portfolio_qubo(**kwargs)
        elif problem_type == "signal_selection":
            result, fallback_chain = self._solve_signal_selection(**kwargs)
        elif problem_type == "vqe":
            result, fallback_chain = self._solve_vqe(**kwargs)
        elif problem_type == "circuit":
            result, fallback_chain = self._run_circuit(**kwargs)
        else:
            # Default: try D-Wave QUBO if possible, else classical
            result = {"error": f"Unknown problem type: {problem_type}"}
            fallback_chain = ["error"]

        if result is not None:
            result["vendor_used"] = fallback_chain[-1] if fallback_chain else "unknown"
            result["fallback_chain"] = fallback_chain

        return result

    def _solve_portfolio_qubo(self, **kwargs):
        """Portfolio optimization: D-Wave -> classical annealing."""
        chain = []

        # Try D-Wave
        if self._dwave is not None:
            try:
                chain.append("dwave")
                result = self._dwave.portfolio_optimize(
                    expected_returns=kwargs["expected_returns"],
                    cov_matrix=kwargs["cov_matrix"],
                    risk_aversion=kwargs.get("risk_aversion", 0.5),
                    budget=kwargs.get("budget"),
                )
                self._record("dwave", "portfolio_qubo", result)
                return result, chain
            except Exception as exc:
                logger.warning("D-Wave portfolio failed: %s", exc)

        # Classical fallback via local annealing in DWaveSolver (no SDK needed)
        try:
            from quantum.vendors.dwave_solver import DWaveSolver
            solver = DWaveSolver()  # Will use local fallback
            chain.append("classical_annealing")
            result = solver.portfolio_optimize(
                expected_returns=kwargs["expected_returns"],
                cov_matrix=kwargs["cov_matrix"],
                risk_aversion=kwargs.get("risk_aversion", 0.5),
                budget=kwargs.get("budget"),
            )
            self._record("classical", "portfolio_qubo", result)
            return result, chain
        except Exception as exc:
            logger.warning("Classical portfolio annealing failed: %s", exc)
            chain.append("error")
            return {"error": str(exc)}, chain

    def _solve_signal_selection(self, **kwargs):
        """Signal selection: D-Wave -> classical annealing."""
        chain = []

        if self._dwave is not None:
            try:
                chain.append("dwave")
                result = self._dwave.signal_select(
                    confidences=kwargs["confidences"],
                    max_signals=kwargs["max_signals"],
                    diversity_penalty=kwargs.get("diversity_penalty", 0.1),
                )
                self._record("dwave", "signal_selection", result)
                return result, chain
            except Exception as exc:
                logger.warning("D-Wave signal selection failed: %s", exc)

        # Classical fallback
        try:
            from quantum.vendors.dwave_solver import DWaveSolver
            solver = DWaveSolver()
            chain.append("classical_annealing")
            result = solver.signal_select(
                confidences=kwargs["confidences"],
                max_signals=kwargs["max_signals"],
                diversity_penalty=kwargs.get("diversity_penalty", 0.1),
            )
            self._record("classical", "signal_selection", result)
            return result, chain
        except Exception as exc:
            logger.warning("Classical signal selection failed: %s", exc)
            chain.append("error")
            return {"error": str(exc)}, chain

    def _solve_vqe(self, **kwargs):
        """VQE: IBM Quantum -> classical scipy."""
        chain = []

        if self._ibm is not None:
            try:
                chain.append("ibm")
                result = self._ibm.vqe_portfolio(
                    expected_returns=kwargs["expected_returns"],
                    cov_matrix=kwargs["cov_matrix"],
                    n_layers=kwargs.get("n_layers", 1),
                    risk_aversion=kwargs.get("risk_aversion", 0.5),
                )
                self._record("ibm", "vqe", result)
                return result, chain
            except Exception as exc:
                logger.warning("IBM VQE failed: %s", exc)

        # Classical fallback
        try:
            from quantum.vendors.ibm_quantum import IBMQuantumBackend
            backend = IBMQuantumBackend()  # Will use classical fallback
            chain.append("classical_scipy")
            result = backend.vqe_portfolio(
                expected_returns=kwargs["expected_returns"],
                cov_matrix=kwargs["cov_matrix"],
                n_layers=kwargs.get("n_layers", 1),
                risk_aversion=kwargs.get("risk_aversion", 0.5),
            )
            self._record("classical", "vqe", result)
            return result, chain
        except Exception as exc:
            logger.warning("Classical VQE failed: %s", exc)
            chain.append("error")
            return {"error": str(exc)}, chain

    def _run_circuit(self, **kwargs):
        """Raw circuit execution: IBM -> Aer -> classical."""
        chain = []
        circuit = kwargs.get("circuit")
        shots = kwargs.get("shots", 1000)

        if self._ibm is not None:
            try:
                chain.append("ibm")
                result = self._ibm.run_circuit(circuit, shots=shots)
                self._record("ibm", "circuit", result)
                return result, chain
            except Exception as exc:
                logger.warning("IBM circuit run failed: %s", exc)

        # Classical fallback
        try:
            from quantum.vendors.ibm_quantum import IBMQuantumBackend
            backend = IBMQuantumBackend()
            chain.append("classical")
            result = backend.run_circuit(circuit, shots=shots)
            self._record("classical", "circuit", result)
            return result, chain
        except Exception as exc:
            chain.append("error")
            return {"error": str(exc)}, chain

    def get_status(self) -> Dict[str, Any]:
        """
        Return status of all vendors.

        Returns dict with:
            vendor_name: {available, has_hardware, last_used, jobs_run}
        """
        status: Dict[str, Any] = {}

        # D-Wave
        if self._dwave is not None:
            info = self._dwave.get_solver_info()
            dwave_jobs = [j for j in self._job_history if j.vendor == "dwave"]
            status["dwave"] = {
                "available": True,
                "has_hardware": info.get("has_hardware", False),
                "has_sdk": info.get("has_sdk", False),
                "solver_name": info.get("solver_name", "unknown"),
                "jobs_run": info.get("jobs_run", 0),
                "last_used": dwave_jobs[-1].timestamp if dwave_jobs else None,
            }
        else:
            status["dwave"] = {"available": False, "has_hardware": False}

        # IBM
        if self._ibm is not None:
            info = self._ibm.get_backend_info()
            ibm_jobs = [j for j in self._job_history if j.vendor == "ibm"]
            status["ibm"] = {
                "available": True,
                "has_hardware": info.get("has_hardware", False),
                "has_qiskit": info.get("has_qiskit", False),
                "backend_name": info.get("backend_name", "unknown"),
                "jobs_run": info.get("jobs_run", 0),
                "last_used": ibm_jobs[-1].timestamp if ibm_jobs else None,
            }
        else:
            status["ibm"] = {"available": False, "has_hardware": False}

        # Classical (always available)
        classical_jobs = [j for j in self._job_history if j.vendor == "classical"]
        status["classical"] = {
            "available": True,
            "has_hardware": False,
            "jobs_run": len(classical_jobs),
            "last_used": classical_jobs[-1].timestamp if classical_jobs else None,
        }

        return status

    def get_usage_report(self) -> Dict[str, Any]:
        """
        Return usage report.

        Returns:
            total_jobs: int
            jobs_by_vendor: dict of vendor -> count
            jobs_by_problem: dict of problem_type -> count
            hardware_jobs: int
            avg_timing_ms: float
        """
        total = len(self._job_history)
        by_vendor: Dict[str, int] = {}
        by_problem: Dict[str, int] = {}
        hw_count = 0
        total_time = 0.0

        for job in self._job_history:
            by_vendor[job.vendor] = by_vendor.get(job.vendor, 0) + 1
            by_problem[job.problem_type] = by_problem.get(job.problem_type, 0) + 1
            if job.hardware_used:
                hw_count += 1
            total_time += job.timing_ms

        return {
            "total_jobs": total,
            "jobs_by_vendor": by_vendor,
            "jobs_by_problem": by_problem,
            "hardware_jobs": hw_count,
            "avg_timing_ms": total_time / max(total, 1),
        }

    def benchmark_all(self, test_problem: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run same problem on all available backends, compare results and timing.

        Args:
            test_problem: Optional dict with expected_returns and cov_matrix.
                         Generates a random 5-asset problem if not provided.

        Returns:
            results: dict of method_name -> {energy, timing_ms, selected_assets}
            best_method: str
            timing_comparison: dict of method -> ms
        """
        if test_problem is None:
            np.random.seed(42)
            n = 5
            test_problem = {
                "expected_returns": np.random.uniform(0.01, 0.1, n),
                "cov_matrix": np.eye(n) * 0.02 + np.random.uniform(0, 0.005, (n, n)),
            }
            # Make covariance symmetric positive definite
            cov = test_problem["cov_matrix"]
            test_problem["cov_matrix"] = (cov + cov.T) / 2 + np.eye(n) * 0.01

        mu = test_problem["expected_returns"]
        sigma = test_problem["cov_matrix"]

        results: Dict[str, Dict[str, Any]] = {}

        # D-Wave path
        if self._dwave is not None:
            try:
                t0 = time.perf_counter()
                r = self._dwave.portfolio_optimize(mu, sigma)
                elapsed = (time.perf_counter() - t0) * 1000.0
                results["dwave"] = {
                    "energy": r.get("energy", 0.0),
                    "timing_ms": elapsed,
                    "selected_assets": r.get("selected_assets", []),
                    "method": r.get("method", "unknown"),
                }
            except Exception as exc:
                results["dwave"] = {"error": str(exc)}

        # IBM VQE path
        if self._ibm is not None:
            try:
                t0 = time.perf_counter()
                r = self._ibm.vqe_portfolio(mu, sigma, max_iterations=20)
                elapsed = (time.perf_counter() - t0) * 1000.0
                results["ibm_vqe"] = {
                    "expected_return": r.get("expected_return", 0.0),
                    "risk": r.get("risk", 0.0),
                    "timing_ms": elapsed,
                    "method": r.get("method", "unknown"),
                }
            except Exception as exc:
                results["ibm_vqe"] = {"error": str(exc)}

        # Classical baseline (always run)
        try:
            from quantum.vendors.dwave_solver import DWaveSolver
            solver = DWaveSolver()
            t0 = time.perf_counter()
            r = solver.portfolio_optimize(mu, sigma)
            elapsed = (time.perf_counter() - t0) * 1000.0
            results["classical_annealing"] = {
                "energy": r.get("energy", 0.0),
                "timing_ms": elapsed,
                "selected_assets": r.get("selected_assets", []),
                "method": r.get("method", "unknown"),
            }
        except Exception as exc:
            results["classical_annealing"] = {"error": str(exc)}

        # Find best by energy (lower is better) among non-error results
        best_method = "none"
        best_energy = float("inf")
        timing_comparison: Dict[str, float] = {}

        for method, r in results.items():
            if "error" not in r:
                timing_comparison[method] = r.get("timing_ms", 0.0)
                e = r.get("energy", float("inf"))
                if e < best_energy:
                    best_energy = e
                    best_method = method

        return {
            "results": results,
            "best_method": best_method,
            "timing_comparison": timing_comparison,
        }
