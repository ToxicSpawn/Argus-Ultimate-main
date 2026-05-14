"""
Quantum Approximate Optimization Algorithm (QAOA) for Portfolio Optimization.

This module implements a real QAOA solver that routes through the ARGUS
in-repo ``quantum_simulator``. The variational circuit uses the full QUBO
cost Hamiltonian (diagonal RZ + off-diagonal RZZ coupling), COBYLA outer-loop
optimization (gradient-free, robust to shot noise), and the parameter-shift
rule is available via ``quantum_simulator.gradient`` when gradient-based
optimization is preferred.

Implementation notes
--------------------
- **Circuit builder**: ``_build_qaoa_circuit(n, qubo, gammas, betas)`` applies
  an initial Hadamard layer, then ``n_layers`` alternating cost/mixer blocks
  where the cost block is ``exp(-iγC)`` with C = x^T Q x, expressed via RZ
  (diagonal) and RZZ (off-diagonal) gates, and the mixer is ``exp(-iβB)`` with
  B = Σ X_i, expressed via RX gates.
- **Cost evaluation**: For each COBYLA iteration, we run the circuit on the
  in-repo simulator to obtain a measurement distribution, then compute
  ``E[x^T Q x]`` as a sample-mean over measured bitstrings.
- **Portfolio decoding**: The measured bitstrings encode a subset selection;
  inverse-variance weights are computed within the selected set to produce
  a valid portfolio.

This version replaces the previous Qiskit/PennyLane/classical hierarchy. The
method string in the result is ``"qaoa_in_repo_simulator"`` (added to the
valid-methods set in ``tests/test_quantum_algorithms.py``).
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from quantum_simulator import QuantumCircuit, simulate

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# QAOA Portfolio Optimizer
# ═════════════════════════════════════════════════════════════════════════════


class QAOAPortfolioOptimizer:
    """
    Real QAOA implementation for the Markowitz portfolio problem.

    Decision variables are binary x_i (include asset i or not). After QAOA
    selects a subset, continuous weights are computed via inverse-variance
    within the selected subset.

    Parameters
    ----------
    n_layers : int
        Number of QAOA alternating layers (p). More layers = deeper circuit,
        generally better approximation but higher variational cost.
    max_assets : int
        Maximum subset size. Used as the QUBO budget constraint.
    use_hardware : bool
        Kept for backwards compatibility. No effect (we always use the
        in-repo simulator).
    """

    def __init__(
        self,
        n_layers: int = 2,
        max_assets: int = 12,
        use_hardware: bool = False,
    ) -> None:
        self.n_layers = max(1, int(n_layers))
        self.max_assets = max(2, int(max_assets))
        self.use_hardware = bool(use_hardware)

    # ── Public API ───────────────────────────────────────────────────────────

    def build_cost_hamiltonian(
        self,
        expected_returns: Any,
        covariance_matrix: Any,
        risk_aversion: float = 0.5,
    ) -> np.ndarray:
        """
        Build the symmetric QUBO matrix Q such that ``E(x) = x^T Q x`` encodes
        the Markowitz objective with a budget-equality penalty.

        The cost is
            -μ·x + λ·x^T Σ x + penalty · (Σ x_i - k)^2

        expanded into QUBO form.
        """
        mu = np.asarray(expected_returns, dtype=float).ravel()
        sigma = np.asarray(covariance_matrix, dtype=float)
        n = len(mu)
        lam = float(risk_aversion)

        mu_scale = float(np.max(np.abs(mu))) if n > 0 and np.max(np.abs(mu)) > 0 else 1.0

        Q = np.zeros((n, n), dtype=float)

        # Return term: -μ_i on diagonal (we're minimizing cost)
        for i in range(n):
            Q[i, i] -= mu[i] / mu_scale

        # Risk term: λ Σ (symmetrized)
        Q += lam * sigma / (mu_scale + 1e-12)

        # Budget penalty: P·(Σ x_i - k)^2 expanded as
        #   P·(k² - 2k Σ x_i + Σ x_i² + 2 Σ_{i<j} x_i x_j)
        # For binary x_i, x_i² = x_i.
        k = min(n, self.max_assets)
        penalty = 2.0 * max(float(np.abs(Q).max() or 1.0), 1.0)
        for i in range(n):
            Q[i, i] += penalty * (1.0 - 2.0 * k)
            for j in range(i + 1, n):
                Q[i, j] += 2.0 * penalty
                Q[j, i] += 2.0 * penalty

        return Q

    def optimize(
        self,
        expected_returns: Any,
        covariance_matrix: Any,
        risk_aversion: float = 0.5,
        budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run QAOA-based portfolio optimization.

        Returns a dict with keys:
        ``weights``, ``expected_return``, ``expected_risk``, ``sharpe``,
        ``method``, ``selected_assets``, ``n_iterations``, ``convergence_history``.
        """
        mu = np.asarray(expected_returns, dtype=float).ravel()
        sigma = np.asarray(covariance_matrix, dtype=float)
        n = len(mu)

        if n == 0:
            return self._empty_result("no_assets")

        if n == 1:
            return self._single_asset_result(mu)

        if budget is not None:
            old_max = self.max_assets
            self.max_assets = int(budget)
        else:
            old_max = self.max_assets

        qubo = self.build_cost_hamiltonian(mu, sigma, risk_aversion)

        result: Optional[Dict[str, Any]] = None

        # Try QAOA on the in-repo simulator for feasible sizes (n ≤ 14).
        if n <= 14:
            try:
                result = self._in_repo_simulator_qaoa(
                    qubo, n, mu, sigma, risk_aversion
                )
            except Exception as exc:
                logger.debug("In-repo simulator QAOA failed: %s", exc)

        if result is None:
            result = self._scipy_fallback(mu, sigma, risk_aversion)

        if budget is not None:
            self.max_assets = old_max

        return result

    def benchmark_vs_classical(
        self,
        expected_returns: Any,
        covariance_matrix: Any,
        risk_aversion: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Run QAOA (on in-repo simulator) and classical Markowitz side by side.
        Returns the standard benchmark dict used by the test suite.
        """
        mu = np.asarray(expected_returns, dtype=float).ravel()
        sigma = np.asarray(covariance_matrix, dtype=float)

        t0 = time.perf_counter()
        qaoa_result = self.optimize(mu, sigma, risk_aversion)
        qaoa_time = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        classical_result = self._scipy_fallback(mu, sigma, risk_aversion)
        classical_time = (time.perf_counter() - t0) * 1000

        qaoa_sharpe = float(qaoa_result.get("sharpe", 0.0) or 0.0)
        classical_sharpe = float(classical_result.get("sharpe", 0.0) or 0.0)

        if abs(classical_sharpe) > 1e-9:
            improvement = (qaoa_sharpe - classical_sharpe) / abs(classical_sharpe) * 100.0
        else:
            improvement = 0.0

        if qaoa_sharpe > classical_sharpe * 1.01:
            assessment = (
                "QAOA found a marginally better discrete-subset solution than "
                "continuous Markowitz. This is the expected behavior on classical "
                "simulation. True quantum advantage requires fault-tolerant hardware."
            )
        elif abs(qaoa_sharpe - classical_sharpe) < 0.01 * max(abs(classical_sharpe), 1e-6):
            assessment = (
                "QAOA and classical Markowitz found equivalent solutions. This is "
                "expected when simulating QAOA classically — both converge to similar "
                "optima."
            )
        else:
            assessment = (
                "Classical continuous Markowitz found a better solution than QAOA on "
                "this instance. Shallow-depth QAOA on classical simulation is "
                "suboptimal; real quantum advantage is expected only with >100 "
                "logical qubits at large problem sizes."
            )

        return {
            "qaoa_sharpe": qaoa_sharpe,
            "classical_sharpe": classical_sharpe,
            "qaoa_time_ms": qaoa_time,
            "classical_time_ms": classical_time,
            "improvement_pct": improvement,
            "honest_assessment": assessment,
            "qaoa_method": qaoa_result.get("method", "unknown"),
            "classical_method": classical_result.get("method", "unknown"),
        }

    # ── In-repo simulator path ───────────────────────────────────────────────

    def _in_repo_simulator_qaoa(
        self,
        qubo: np.ndarray,
        n: int,
        mu: np.ndarray,
        sigma: np.ndarray,
        risk_aversion: float,
    ) -> Dict[str, Any]:
        """
        QAOA optimizer running on ``quantum_simulator``. Uses COBYLA outer loop.
        """
        from scipy.optimize import minimize as sp_minimize

        p = self.n_layers
        convergence: List[float] = []
        shots = 1024
        rng = np.random.default_rng(42)

        def cost_of_bits(bits: np.ndarray) -> float:
            """E = bits^T Q bits (for a row vector of 0/1)."""
            return float(bits @ qubo @ bits)

        def expected_cost(counts: Dict[str, int]) -> float:
            """Sample-mean ⟨x^T Q x⟩ over measurement distribution."""
            total = sum(counts.values())
            if total == 0:
                return 0.0
            acc = 0.0
            for bitstring, c in counts.items():
                # bitstring is MSB-first; qubit 0 is the rightmost char.
                bits = np.array(
                    [
                        int(bitstring[len(bitstring) - 1 - q]) if q < len(bitstring) else 0
                        for q in range(n)
                    ],
                    dtype=float,
                )
                acc += c * cost_of_bits(bits)
            return acc / total

        def cost_fn(params: np.ndarray) -> float:
            gammas = params[:p]
            betas = params[p:]
            qc = self.build_variational_circuit(n, qubo, gammas, betas)
            qc.measure_all()
            res = simulate(qc, shots=shots, seed=int(rng.integers(0, 2**31 - 1)))
            ec = expected_cost(res["counts"])
            convergence.append(ec)
            return ec

        # COBYLA with a few random restarts
        best_cost = float("inf")
        best_params: Optional[np.ndarray] = None

        for trial in range(3):
            x0 = rng.uniform(0, 2.0 * np.pi, 2 * p)
            opt = sp_minimize(
                cost_fn,
                x0,
                method="COBYLA",
                options={"maxiter": 50, "rhobeg": 1.0},
            )
            if opt.fun < best_cost:
                best_cost = float(opt.fun)
                best_params = np.asarray(opt.x, dtype=float)

        if best_params is None:
            best_params = np.full(2 * p, np.pi / 2.0)

        # Final evaluation with more shots for a clean result
        final_qc = self.build_variational_circuit(n, qubo, best_params[:p], best_params[p:])
        final_qc.measure_all()
        final_res = simulate(final_qc, shots=4096, seed=7)
        final_counts = final_res["counts"]

        # Pick the most-frequent bitstring as the chosen subset
        top_bitstring = max(final_counts.items(), key=lambda kv: kv[1])[0]
        best_bits = np.array(
            [
                int(top_bitstring[len(top_bitstring) - 1 - q]) if q < len(top_bitstring) else 0
                for q in range(n)
            ],
            dtype=float,
        )

        return self._bits_to_result(
            best_bits,
            mu,
            sigma,
            risk_aversion,
            method="qaoa_in_repo_simulator",
            n_iterations=len(convergence),
            convergence_history=convergence[-20:],
        )

    # ── Circuit builders (used by both optimize and Phase C1 wiring) ─────────

    def build_variational_circuit(
        self,
        n: int,
        qubo: np.ndarray,
        gammas: Sequence[float],
        betas: Sequence[float],
    ) -> QuantumCircuit:
        """
        Build a p-layer QAOA circuit for the given QUBO using gates from
        ``quantum_simulator``.

        Cost layer: exp(-iγ C) with C = x^T Q x
            diagonal: RZ(2·γ·Q[i,i]) on qubit i
            off-diag: RZZ(2·γ·Q[i,j]) on (i,j) (symmetric, only i<j)
        Mixer layer: exp(-iβ B) with B = Σ X_i
            RX(2·β) on each qubit

        Does not append a final measurement; the caller should call
        ``qc.measure_all()`` if needed.
        """
        p = min(len(gammas), len(betas))
        qc = QuantumCircuit(int(n))

        # Initial superposition
        for q in range(n):
            qc.h(q)

        for layer in range(p):
            gamma = float(gammas[layer])
            beta = float(betas[layer])

            # Diagonal cost terms: RZ(2·γ·Q[i,i])
            for i in range(n):
                diag = float(qubo[i, i])
                if abs(diag) > 1e-12:
                    qc.rz(2.0 * gamma * diag, i)

            # Off-diagonal ZZ coupling: RZZ(2·γ·Q[i,j]) for i<j
            for i in range(n):
                for j in range(i + 1, n):
                    coup = float(qubo[i, j])
                    if abs(coup) > 1e-12:
                        qc.rzz(2.0 * gamma * coup, i, j)

            # Mixer: RX(2β) on each qubit
            for i in range(n):
                qc.rx(2.0 * beta, i)

        return qc

    def default_params(self, n_layers: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return default (gammas, betas) for a p-layer QAOA. Used by Phase C1
        when we want a circuit without running the optimization loop.
        """
        p = int(n_layers) if n_layers is not None else self.n_layers
        # Standard initial values used in the QAOA literature
        gammas = np.linspace(0.1, 0.5, p)
        betas = np.linspace(0.5, 0.1, p)
        return gammas, betas

    # ── Tensor-network MPS path (for n > 12 problems) ────────────────────────

    def _tensor_network_qaoa(
        self,
        qubo: np.ndarray,
        n: int,
        mu: np.ndarray,
        sigma: np.ndarray,
        risk_aversion: float,
    ) -> Dict[str, Any]:
        """
        QAOA via the in-repo simulator's MPS backend.

        Used for ``n > 12`` where statevector becomes expensive. The MPS
        backend handles the SWAP-network routing for non-adjacent RZZ gates
        automatically (Phase A2 fix). Convergence is via COBYLA on a small
        parameter set; we measure expected cost from sample distributions.
        """
        from scipy.optimize import minimize as sp_minimize

        p = self.n_layers
        convergence: List[float] = []
        shots = 512  # MPS sampling is moderately expensive
        rng = np.random.default_rng(42)

        def cost_of_bits(bits: np.ndarray) -> float:
            return float(bits @ qubo @ bits)

        def expected_cost(counts: Dict[str, int]) -> float:
            total = sum(counts.values())
            if total == 0:
                return 0.0
            acc = 0.0
            for bitstring, c in counts.items():
                bits = np.array(
                    [
                        int(bitstring[len(bitstring) - 1 - q])
                        if q < len(bitstring) else 0
                        for q in range(n)
                    ],
                    dtype=float,
                )
                acc += c * cost_of_bits(bits)
            return acc / total

        def cost_fn(params: np.ndarray) -> float:
            gammas = params[:p]
            betas = params[p:]
            qc = self.build_variational_circuit(n, qubo, gammas, betas)
            qc.measure_all()
            res = simulate(
                qc,
                shots=shots,
                seed=int(rng.integers(0, 2**31 - 1)),
                backend="mps",
            )
            ec = expected_cost(res["counts"])
            convergence.append(ec)
            return ec

        best_cost = float("inf")
        best_params: Optional[np.ndarray] = None

        for trial in range(2):
            x0 = rng.uniform(0, 2.0 * np.pi, 2 * p)
            try:
                opt = sp_minimize(
                    cost_fn,
                    x0,
                    method="COBYLA",
                    options={"maxiter": 30, "rhobeg": 1.0},
                )
                if opt.fun < best_cost:
                    best_cost = float(opt.fun)
                    best_params = np.asarray(opt.x, dtype=float)
            except Exception as exc:
                logger.debug("MPS QAOA trial failed: %s", exc)
                continue

        if best_params is None:
            best_params = np.full(2 * p, np.pi / 2.0)

        # Final shot to extract bitstring
        final_qc = self.build_variational_circuit(
            n, qubo, best_params[:p], best_params[p:]
        )
        final_qc.measure_all()
        try:
            final_res = simulate(final_qc, shots=2048, seed=7, backend="mps")
        except Exception:
            final_res = simulate(final_qc, shots=2048, seed=7)
        top_bitstring = max(final_res["counts"].items(), key=lambda kv: kv[1])[0]
        best_bits = np.array(
            [
                int(top_bitstring[len(top_bitstring) - 1 - q])
                if q < len(top_bitstring) else 0
                for q in range(n)
            ],
            dtype=float,
        )

        # If nothing selected (all zeros), pick top-k by mu
        if best_bits.sum() < 1:
            top_k = min(self.max_assets, n)
            top_indices = np.argsort(mu)[-top_k:]
            best_bits = np.zeros(n)
            best_bits[top_indices] = 1.0

        return self._bits_to_result(
            best_bits,
            mu,
            sigma,
            risk_aversion,
            method="qaoa_tensor_network_mps",
            n_iterations=len(convergence),
            convergence_history=convergence[-20:],
        )

    # ── Classical fallback ───────────────────────────────────────────────────

    def _scipy_fallback(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        risk_aversion: float,
    ) -> Dict[str, Any]:
        """Continuous-weight Markowitz via scipy SLSQP."""
        from scipy.optimize import minimize as sp_minimize

        n = len(mu)
        if n == 0:
            return self._empty_result("no_assets")
        if n == 1:
            return self._single_asset_result(mu)

        def neg_objective(w: np.ndarray) -> float:
            ret = float(w @ mu)
            risk = float(w @ sigma @ w)
            return -(ret - risk_aversion * risk)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n
        x0 = np.ones(n) / n

        opt = sp_minimize(
            neg_objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500},
        )

        w = np.maximum(opt.x, 0.0)
        w = w / (w.sum() or 1.0)

        ret_val = float(w @ mu)
        risk_val = float(np.sqrt(max(w @ sigma @ w, 0.0)))
        sharpe = ret_val / risk_val if risk_val > 1e-12 else 0.0
        selected = [i for i in range(n) if w[i] > 0.01]

        return {
            "weights": w.tolist(),
            "expected_return": ret_val,
            "expected_risk": risk_val,
            "sharpe": sharpe,
            "method": "classical_scipy_fallback",
            "selected_assets": selected,
            "n_iterations": int(getattr(opt, "nit", 0)),
            "convergence_history": [],
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _bits_to_result(
        self,
        bits: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        risk_aversion: float,
        method: str,
        n_iterations: int,
        convergence_history: List[float],
    ) -> Dict[str, Any]:
        """Convert a binary selection vector to a portfolio result dict."""
        n = len(mu)
        selected = [i for i in range(n) if bits[i] > 0.5]

        if not selected:
            # QAOA selected nothing — fall back to highest-return asset
            selected = [int(np.argmax(mu))]

        # Cap to budget if needed
        if len(selected) > self.max_assets:
            # Keep highest-return assets within the budget
            sorted_selected = sorted(selected, key=lambda i: -mu[i])
            selected = sorted_selected[: self.max_assets]

        sub_sigma = sigma[np.ix_(selected, selected)]
        diag = np.diag(sub_sigma)
        diag = np.maximum(diag, 1e-12)
        inv_var = 1.0 / diag
        raw_w = inv_var / inv_var.sum()

        weights = np.zeros(n)
        for idx, s in enumerate(selected):
            weights[s] = raw_w[idx]

        ret_val = float(weights @ mu)
        risk_val = float(np.sqrt(max(weights @ sigma @ weights, 0.0)))
        sharpe = ret_val / risk_val if risk_val > 1e-12 else 0.0

        return {
            "weights": weights.tolist(),
            "expected_return": ret_val,
            "expected_risk": risk_val,
            "sharpe": sharpe,
            "method": method,
            "selected_assets": selected,
            "n_iterations": int(n_iterations),
            "convergence_history": convergence_history,
        }

    @staticmethod
    def _empty_result(method: str) -> Dict[str, Any]:
        return {
            "weights": [],
            "expected_return": 0.0,
            "expected_risk": 0.0,
            "sharpe": 0.0,
            "method": method,
            "selected_assets": [],
            "n_iterations": 0,
            "convergence_history": [],
        }

    @staticmethod
    def _single_asset_result(mu: np.ndarray) -> Dict[str, Any]:
        return {
            "weights": [1.0],
            "expected_return": float(mu[0]),
            "expected_risk": 0.0,
            "sharpe": 0.0,
            "method": "single_asset",
            "selected_assets": [0],
            "n_iterations": 0,
            "convergence_history": [],
        }
