"""
D-Wave quantum annealer integration for ARGUS.

Real D-Wave Ocean SDK integration with graceful fallback chain:
1. D-Wave LeapHybridSampler (real quantum hardware, free 1 min/month on Leap)
2. D-Wave SimulatedAnnealingSampler (classical, from Ocean SDK)
3. Local simulated quantum annealing (quantum/optimization/annealing.py)

Environment:
    DWAVE_API_TOKEN: D-Wave Leap API token (optional)
    DWAVE_SOLVER: Override solver name (optional)

Install:
    pip install dwave-ocean-sdk   (optional, falls back gracefully)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy SDK detection
# ---------------------------------------------------------------------------

_HAS_DWAVE = False
_dwave_sa = None
_dwave_hybrid = None
_dwave_dimod = None

try:
    import dimod as _dwave_dimod  # type: ignore[no-redef]
    _HAS_DWAVE = True
except ImportError:
    pass

try:
    from dwave.system import LeapHybridSampler as _LeapHybridSampler  # type: ignore
except ImportError:
    _LeapHybridSampler = None

try:
    import neal as _dwave_sa  # type: ignore[no-redef]
except ImportError:
    pass


class DWaveSolver:
    """
    Real D-Wave quantum annealer access.

    Uses DWAVE_API_TOKEN env var or passed token.
    Free tier: 1 minute/month on Leap hybrid solver.
    Falls back to simulated annealing if no token or dwave-ocean-sdk not installed.
    """

    def __init__(self, api_token: Optional[str] = None):
        self._token = api_token or os.environ.get("DWAVE_API_TOKEN")
        self._solver_name = os.environ.get("DWAVE_SOLVER")

        # Detect capabilities
        self._has_sdk = _HAS_DWAVE
        self._has_hybrid = _LeapHybridSampler is not None and bool(self._token)
        self._has_neal = _dwave_sa is not None

        # Lazy sampler instances
        self._hybrid_sampler = None
        self._sa_sampler = None

        # Track usage
        self._jobs_run = 0
        self._total_hw_ms = 0.0

        level = "hardware" if self._has_hybrid else ("sdk-sim" if self._has_neal else "local-sim")
        logger.info("DWaveSolver initialized: level=%s, sdk=%s, token=%s",
                     level, self._has_sdk, bool(self._token))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_hybrid_sampler(self):
        """Lazy-init the LeapHybridSampler."""
        if self._hybrid_sampler is None and self._has_hybrid:
            try:
                kwargs: dict = {}
                if self._token:
                    kwargs["token"] = self._token
                if self._solver_name:
                    kwargs["solver"] = self._solver_name
                self._hybrid_sampler = _LeapHybridSampler(**kwargs)
            except Exception as exc:
                logger.warning("LeapHybridSampler init failed: %s", exc)
                self._has_hybrid = False
        return self._hybrid_sampler

    def _get_sa_sampler(self):
        """Lazy-init the Neal SimulatedAnnealingSampler."""
        if self._sa_sampler is None and self._has_neal:
            self._sa_sampler = _dwave_sa.SimulatedAnnealingSampler()
        return self._sa_sampler

    @staticmethod
    def _qubo_dict_to_bqm(Q: Dict[Tuple[int, int], float]):
        """Convert a QUBO dict to a dimod BQM."""
        linear: Dict[int, float] = {}
        quadratic: Dict[Tuple[int, int], float] = {}
        for (i, j), w in Q.items():
            if i == j:
                linear[i] = linear.get(i, 0.0) + w
            else:
                key = (min(i, j), max(i, j))
                quadratic[key] = quadratic.get(key, 0.0) + w
        return _dwave_dimod.BinaryQuadraticModel(linear, quadratic, 0.0, _dwave_dimod.BINARY)

    @staticmethod
    def _sampleset_to_result(sampleset, method: str, timing_ms: float) -> Dict[str, Any]:
        """Convert a dimod SampleSet to our result dict."""
        best = sampleset.first
        solution = {int(k): int(v) for k, v in best.sample.items()}
        return {
            "solution": solution,
            "energy": float(best.energy),
            "timing_ms": timing_ms,
            "method": method,
            "num_reads": len(sampleset),
            "hardware_used": method.startswith("dwave_hybrid"),
        }

    def _solve_local(self, Q: Dict[Tuple[int, int], float],
                     num_reads: int) -> Dict[str, Any]:
        """Fallback to local simulated quantum annealing."""
        from quantum.optimization.annealing import solve_qubo
        t0 = time.perf_counter()
        result = solve_qubo(Q, num_reads=num_reads)
        elapsed = (time.perf_counter() - t0) * 1000.0
        result["timing_ms"] = elapsed
        result["method"] = "local_simulated_annealing"
        result["hardware_used"] = False
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve_qubo(self, Q: Dict[Tuple[int, int], float],
                   num_reads: int = 100,
                   label: str = "argus") -> Dict[str, Any]:
        """
        Solve QUBO problem on D-Wave hardware.

        Fallback chain:
        1. D-Wave LeapHybridSampler (real hardware, if token + SDK)
        2. Neal SimulatedAnnealingSampler (classical, if Ocean SDK)
        3. Local simulated quantum annealing (always available)

        Returns:
            solution: dict mapping variable -> 0/1
            energy: float, lowest energy found
            timing_ms: float, wall-clock milliseconds
            method: str, which solver was used
            num_reads: int
            hardware_used: bool
        """
        if not Q:
            return {"solution": {}, "energy": 0.0, "timing_ms": 0.0,
                    "method": "empty", "num_reads": 0, "hardware_used": False}

        self._jobs_run += 1

        # --- Try LeapHybridSampler (real hardware) ---
        if self._has_hybrid and self._has_sdk:
            sampler = self._get_hybrid_sampler()
            if sampler is not None:
                try:
                    bqm = self._qubo_dict_to_bqm(Q)
                    t0 = time.perf_counter()
                    sampleset = sampler.sample(bqm, label=label)
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    self._total_hw_ms += elapsed
                    logger.info("D-Wave hybrid solve: %.1f ms, energy=%.4f",
                                elapsed, sampleset.first.energy)
                    return self._sampleset_to_result(sampleset, "dwave_hybrid", elapsed)
                except Exception as exc:
                    logger.warning("D-Wave hybrid failed, falling back: %s", exc)

        # --- Try Neal SimulatedAnnealingSampler ---
        if self._has_neal and self._has_sdk:
            sampler = self._get_sa_sampler()
            if sampler is not None:
                try:
                    bqm = self._qubo_dict_to_bqm(Q)
                    t0 = time.perf_counter()
                    sampleset = sampler.sample(bqm, num_reads=num_reads)
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    return self._sampleset_to_result(sampleset, "dwave_neal_sa", elapsed)
                except Exception as exc:
                    logger.warning("Neal SA failed, falling back: %s", exc)

        # --- Local fallback ---
        return self._solve_local(Q, num_reads)

    def portfolio_optimize(
        self,
        expected_returns: Any,
        cov_matrix: Any,
        risk_aversion: float = 0.5,
        budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Formulate portfolio selection as QUBO and solve on D-Wave.

        QUBO encoding:
            min x^T (risk_aversion * Sigma) x - mu^T x
                + penalty * (sum(x) - budget)^2

        Args:
            expected_returns: 1D array of expected returns per asset.
            cov_matrix: 2D covariance matrix.
            risk_aversion: Return vs risk trade-off (0=return only, 1=risk only).
            budget: Number of assets to select. Defaults to n//2.

        Returns:
            selected_assets: list of selected asset indices
            weights: dict of asset index -> weight (equal weight among selected)
            expected_return: float
            risk: float
            method: str, solver used
        """
        mu = np.asarray(expected_returns, dtype=float).ravel()
        sigma = np.asarray(cov_matrix, dtype=float)
        n = len(mu)
        if budget is None:
            budget = max(1, n // 2)

        penalty = float(max(abs(mu).max(), abs(sigma).max(), 1.0)) * 2.0

        Q: Dict[Tuple[int, int], float] = {}

        # Return term (negate for minimization)
        for i in range(n):
            Q[(i, i)] = Q.get((i, i), 0.0) - float(mu[i])

        # Risk term
        for i in range(n):
            for j in range(i, n):
                key = (i, j)
                w = risk_aversion * float(sigma[i, j])
                if i == j:
                    Q[key] = Q.get(key, 0.0) + w
                else:
                    Q[key] = Q.get(key, 0.0) + 2.0 * w

        # Budget constraint: penalty * (sum(x) - budget)^2
        for i in range(n):
            Q[(i, i)] = Q.get((i, i), 0.0) + penalty * (1.0 - 2.0 * budget)
        for i in range(n):
            for j in range(i + 1, n):
                Q[(i, j)] = Q.get((i, j), 0.0) + 2.0 * penalty

        result = self.solve_qubo(Q, label="argus_portfolio")
        solution = result.get("solution", {})
        selected = sorted([i for i, v in solution.items() if v == 1])

        n_sel = len(selected) or 1
        weights = {i: 1.0 / n_sel for i in selected}

        # Compute portfolio metrics
        sel_mask = np.array([1.0 if i in selected else 0.0 for i in range(n)])
        exp_ret = float(sel_mask @ mu) / n_sel if n_sel else 0.0
        risk = float(sel_mask @ sigma @ sel_mask) / (n_sel ** 2) if n_sel else 0.0

        return {
            "selected_assets": selected,
            "weights": weights,
            "expected_return": exp_ret,
            "risk": risk,
            "energy": result.get("energy", 0.0),
            "method": result.get("method", "unknown"),
            "hardware_used": result.get("hardware_used", False),
        }

    def signal_select(
        self,
        confidences: List[float],
        max_signals: int,
        diversity_penalty: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Select optimal signal subset via QUBO on D-Wave.

        Maximize sum of confidences minus diversity penalty for selecting
        too many signals. Enforces max_signals cardinality constraint.

        Args:
            confidences: List of signal confidence values.
            max_signals: Maximum signals to select.
            diversity_penalty: Penalty weight for correlated selections.

        Returns:
            selected_indices: list of selected signal indices
            total_confidence: float, sum of selected confidences
            method: str
        """
        n = len(confidences)
        if n == 0:
            return {"selected_indices": [], "total_confidence": 0.0,
                    "method": "empty", "hardware_used": False}

        max_signals = min(max_signals, n)
        penalty = float(max(abs(c) for c in confidences) + 1.0) * 2.0

        Q: Dict[Tuple[int, int], float] = {}

        # Confidence term (negate for minimization)
        for i in range(n):
            Q[(i, i)] = Q.get((i, i), 0.0) - float(confidences[i])

        # Diversity penalty between all pairs
        if diversity_penalty > 0:
            for i in range(n):
                for j in range(i + 1, n):
                    Q[(i, j)] = Q.get((i, j), 0.0) + diversity_penalty

        # Cardinality constraint: (sum(x) - max_signals)^2
        if max_signals < n:
            K = max_signals
            for i in range(n):
                Q[(i, i)] = Q.get((i, i), 0.0) + penalty * (1.0 - 2.0 * K)
            for i in range(n):
                for j in range(i + 1, n):
                    Q[(i, j)] = Q.get((i, j), 0.0) + 2.0 * penalty

        result = self.solve_qubo(Q, label="argus_signal_select")
        solution = result.get("solution", {})
        selected = sorted([i for i, v in solution.items() if v == 1])
        total_conf = sum(confidences[i] for i in selected if i < n)

        return {
            "selected_indices": selected,
            "total_confidence": total_conf,
            "energy": result.get("energy", 0.0),
            "method": result.get("method", "unknown"),
            "hardware_used": result.get("hardware_used", False),
        }

    def get_solver_info(self) -> Dict[str, Any]:
        """
        Return available solver information.

        Returns:
            available: bool, whether any solver is usable
            solver_name: str
            num_qubits: int (0 for simulators)
            connectivity: str
            has_hardware: bool
            has_sdk: bool
            jobs_run: int
        """
        info: Dict[str, Any] = {
            "available": True,  # local fallback is always available
            "has_sdk": self._has_sdk,
            "has_hardware": self._has_hybrid,
            "jobs_run": self._jobs_run,
            "total_hardware_ms": self._total_hw_ms,
        }

        if self._has_hybrid:
            sampler = self._get_hybrid_sampler()
            if sampler is not None:
                props = getattr(sampler, "properties", {})
                info["solver_name"] = props.get("solver", "LeapHybridSampler")
                info["num_qubits"] = props.get("num_qubits", 0)
                info["connectivity"] = "pegasus"
            else:
                info["solver_name"] = "hybrid_unavailable"
                info["num_qubits"] = 0
                info["connectivity"] = "none"
        elif self._has_neal:
            info["solver_name"] = "neal_simulated_annealing"
            info["num_qubits"] = 0
            info["connectivity"] = "fully_connected_classical"
        else:
            info["solver_name"] = "local_simulated_annealing"
            info["num_qubits"] = 0
            info["connectivity"] = "fully_connected_classical"

        return info
