"""
Tests for quantum hardware vendor integrations.

All tests work without any API keys or quantum SDKs installed.
They exercise the classical fallback paths that are always available.

Run with: py -m pytest tests/test_quantum_vendors.py -v
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# D-Wave Solver Tests
# ---------------------------------------------------------------------------


class TestDWaveSolver:
    """Tests for DWaveSolver with classical fallback."""

    def _make_solver(self):
        from quantum.vendors.dwave_solver import DWaveSolver
        return DWaveSolver(api_token=None)  # Forces local fallback

    def test_init_no_token(self):
        solver = self._make_solver()
        assert solver is not None
        info = solver.get_solver_info()
        assert info["available"] is True

    def test_solve_empty_qubo(self):
        solver = self._make_solver()
        result = solver.solve_qubo({})
        assert result["solution"] == {}
        assert result["energy"] == 0.0
        assert result["hardware_used"] is False

    def test_solve_simple_qubo(self):
        solver = self._make_solver()
        # Minimize x0 + x1 - 3*x0*x1  => optimal: x0=1, x1=1, energy = 1+1-3 = -1
        Q = {(0, 0): 1.0, (1, 1): 1.0, (0, 1): -3.0}
        result = solver.solve_qubo(Q, num_reads=50)
        assert "solution" in result
        assert "energy" in result
        assert "timing_ms" in result
        assert "method" in result
        assert result["hardware_used"] is False
        # The solver should find a reasonable solution
        sol = result["solution"]
        assert set(sol.keys()) == {0, 1}
        assert all(v in (0, 1) for v in sol.values())

    def test_solve_qubo_optimal_solution(self):
        """Verify the solver finds the global optimum for a trivial QUBO."""
        solver = self._make_solver()
        # Only x0=1 gives energy = -5, everything else is worse
        Q = {(0, 0): -5.0, (1, 1): 10.0, (0, 1): 0.0}
        result = solver.solve_qubo(Q, num_reads=100)
        assert result["solution"][0] == 1
        assert result["solution"][1] == 0
        assert result["energy"] == pytest.approx(-5.0)

    def test_portfolio_optimize(self):
        solver = self._make_solver()
        mu = np.array([0.05, 0.10, 0.03, 0.07])
        sigma = np.eye(4) * 0.02
        result = solver.portfolio_optimize(mu, sigma, risk_aversion=0.5, budget=2)
        assert "selected_assets" in result
        assert "weights" in result
        assert "expected_return" in result
        assert "risk" in result
        assert "method" in result
        assert isinstance(result["selected_assets"], list)
        assert len(result["selected_assets"]) > 0

    def test_portfolio_optimize_respects_budget(self):
        solver = self._make_solver()
        mu = np.array([0.10, 0.08, 0.06, 0.04, 0.02])
        sigma = np.eye(5) * 0.01
        result = solver.portfolio_optimize(mu, sigma, risk_aversion=0.3, budget=2)
        # Budget constraint: should select approximately 2 assets
        n_selected = len(result["selected_assets"])
        assert 1 <= n_selected <= 3  # Allow small slack from annealing

    def test_portfolio_weights_sum_to_one(self):
        solver = self._make_solver()
        mu = np.array([0.05, 0.10, 0.03])
        sigma = np.eye(3) * 0.02
        result = solver.portfolio_optimize(mu, sigma)
        weights = result["weights"]
        if weights:
            total = sum(weights.values())
            assert total == pytest.approx(1.0, abs=0.01)

    def test_signal_select(self):
        solver = self._make_solver()
        confidences = [0.9, 0.7, 0.3, 0.8, 0.5]
        result = solver.signal_select(confidences, max_signals=2, diversity_penalty=0.1)
        assert "selected_indices" in result
        assert "total_confidence" in result
        assert "method" in result
        assert len(result["selected_indices"]) <= 3  # Allow some slack

    def test_signal_select_empty(self):
        solver = self._make_solver()
        result = solver.signal_select([], max_signals=2)
        assert result["selected_indices"] == []
        assert result["total_confidence"] == 0.0

    def test_signal_select_prefers_high_confidence(self):
        solver = self._make_solver()
        # One clearly dominant signal
        confidences = [0.01, 0.01, 0.99, 0.01]
        result = solver.signal_select(confidences, max_signals=1, diversity_penalty=0.0)
        assert 2 in result["selected_indices"]

    def test_get_solver_info(self):
        solver = self._make_solver()
        info = solver.get_solver_info()
        assert "available" in info
        assert "has_sdk" in info
        assert "solver_name" in info
        assert "jobs_run" in info
        assert info["available"] is True

    def test_qubo_solutions_are_binary(self):
        """All QUBO solutions must have binary (0/1) values."""
        solver = self._make_solver()
        Q = {(0, 0): -1.0, (1, 1): -2.0, (2, 2): -1.0, (0, 1): 0.5, (1, 2): 0.5}
        result = solver.solve_qubo(Q, num_reads=20)
        for v in result["solution"].values():
            assert v in (0, 1), f"Solution value {v} is not binary"


# ---------------------------------------------------------------------------
# IBM Quantum Backend Tests
# ---------------------------------------------------------------------------


class TestIBMQuantumBackend:
    """Tests for IBMQuantumBackend with classical fallback."""

    def _make_backend(self):
        from quantum.vendors.ibm_quantum import IBMQuantumBackend
        return IBMQuantumBackend(api_token=None)  # Forces classical fallback

    def test_init_no_token(self):
        backend = self._make_backend()
        assert backend is not None
        info = backend.get_backend_info()
        assert info["available"] is True

    def test_run_circuit_classical_fallback(self):
        backend = self._make_backend()
        result = backend.run_circuit(3, shots=500)
        assert "counts" in result
        assert "method" in result
        assert "execution_time_ms" in result
        assert result["shots"] == 500
        # Classical simulation produces bitstrings
        counts = result["counts"]
        assert isinstance(counts, dict)
        total = sum(counts.values())
        assert total == 500

    def test_run_circuit_bitstring_length(self):
        backend = self._make_backend()
        result = backend.run_circuit(4, shots=100)
        for bitstring in result["counts"]:
            assert len(bitstring) == 4
            assert all(c in "01" for c in bitstring)

    def test_vqe_portfolio_classical(self):
        backend = self._make_backend()
        mu = np.array([0.05, 0.10, 0.03])
        sigma = np.eye(3) * 0.02
        result = backend.vqe_portfolio(mu, sigma, max_iterations=20)
        assert "weights" in result
        assert "expected_return" in result
        assert "risk" in result
        assert "method" in result
        assert "n_iterations" in result
        # Weights should be reasonable
        if result["weights"]:
            total = sum(result["weights"].values())
            assert total > 0.5  # Most weight should be assigned

    def test_vqe_portfolio_returns_convergence(self):
        backend = self._make_backend()
        mu = np.array([0.08, 0.05])
        sigma = np.array([[0.02, 0.005], [0.005, 0.01]])
        result = backend.vqe_portfolio(mu, sigma, max_iterations=30)
        assert "convergence" in result
        assert isinstance(result["convergence"], list)

    def test_get_backend_info(self):
        backend = self._make_backend()
        info = backend.get_backend_info()
        assert "available" in info
        assert "has_qiskit" in info
        assert "has_aer" in info
        assert "backend_name" in info
        assert "jobs_run" in info
        assert info["available"] is True

    def test_classical_simulate_deterministic_shots(self):
        """Classical simulation should return exactly the requested number of shots."""
        from quantum.vendors.ibm_quantum import _classical_simulate_circuit
        counts = _classical_simulate_circuit(3, 1000)
        assert sum(counts.values()) == 1000


# ---------------------------------------------------------------------------
# Vendor Orchestrator Tests
# ---------------------------------------------------------------------------


class TestQuantumVendorOrchestrator:
    """Tests for QuantumVendorOrchestrator routing and fallbacks."""

    def _make_orchestrator(self):
        from quantum.vendors.vendor_orchestrator import QuantumVendorOrchestrator
        return QuantumVendorOrchestrator()

    def test_init(self):
        orch = self._make_orchestrator()
        assert orch is not None
        assert orch._local_sim is True

    def test_solve_portfolio_qubo(self):
        orch = self._make_orchestrator()
        mu = np.array([0.05, 0.10, 0.03, 0.07])
        sigma = np.eye(4) * 0.02
        result = orch.solve_optimization(
            "portfolio_qubo",
            expected_returns=mu,
            cov_matrix=sigma,
            risk_aversion=0.5,
        )
        assert "selected_assets" in result
        assert "vendor_used" in result
        assert "fallback_chain" in result
        assert isinstance(result["fallback_chain"], list)
        assert len(result["fallback_chain"]) > 0

    def test_solve_signal_selection(self):
        orch = self._make_orchestrator()
        result = orch.solve_optimization(
            "signal_selection",
            confidences=[0.9, 0.7, 0.3, 0.8],
            max_signals=2,
        )
        assert "selected_indices" in result
        assert "vendor_used" in result

    def test_solve_vqe(self):
        orch = self._make_orchestrator()
        mu = np.array([0.05, 0.10])
        sigma = np.array([[0.02, 0.005], [0.005, 0.01]])
        result = orch.solve_optimization(
            "vqe",
            expected_returns=mu,
            cov_matrix=sigma,
            n_layers=1,
        )
        assert "weights" in result
        assert "vendor_used" in result

    def test_solve_circuit(self):
        orch = self._make_orchestrator()
        result = orch.solve_optimization(
            "circuit",
            circuit=3,  # 3 qubits, classical fallback
            shots=500,
        )
        assert "counts" in result
        assert "vendor_used" in result

    def test_unknown_problem_type(self):
        orch = self._make_orchestrator()
        result = orch.solve_optimization("unknown_type")
        assert "error" in result

    def test_get_status(self):
        orch = self._make_orchestrator()
        status = orch.get_status()
        assert "classical" in status
        assert status["classical"]["available"] is True
        # D-Wave and IBM may or may not be available depending on imports
        assert "dwave" in status
        assert "ibm" in status

    def test_get_usage_report_empty(self):
        orch = self._make_orchestrator()
        report = orch.get_usage_report()
        assert report["total_jobs"] == 0
        assert report["hardware_jobs"] == 0

    def test_get_usage_report_after_jobs(self):
        orch = self._make_orchestrator()
        mu = np.array([0.05, 0.10])
        sigma = np.eye(2) * 0.02
        orch.solve_optimization("portfolio_qubo", expected_returns=mu, cov_matrix=sigma)
        orch.solve_optimization("signal_selection", confidences=[0.8, 0.3], max_signals=1)
        report = orch.get_usage_report()
        assert report["total_jobs"] == 2
        assert "jobs_by_problem" in report
        assert report["jobs_by_problem"].get("portfolio_qubo", 0) >= 1

    def test_benchmark_all(self):
        orch = self._make_orchestrator()
        result = orch.benchmark_all()
        assert "results" in result
        assert "best_method" in result
        assert "timing_comparison" in result
        # At minimum, classical_annealing should have run
        assert "classical_annealing" in result["results"]

    def test_benchmark_custom_problem(self):
        orch = self._make_orchestrator()
        mu = np.array([0.12, 0.08, 0.04])
        sigma = np.array([[0.03, 0.01, 0.005],
                          [0.01, 0.02, 0.003],
                          [0.005, 0.003, 0.01]])
        result = orch.benchmark_all({"expected_returns": mu, "cov_matrix": sigma})
        assert "results" in result
        # Should have at least classical_annealing
        assert any("error" not in v for v in result["results"].values())


# ---------------------------------------------------------------------------
# No-hardware fallback tests
# ---------------------------------------------------------------------------


class TestNoHardwareFallbacks:
    """Verify everything works without any API keys or quantum SDKs."""

    def test_dwave_solver_no_sdk(self):
        from quantum.vendors.dwave_solver import DWaveSolver
        solver = DWaveSolver(api_token=None)
        Q = {(0, 0): -1.0, (1, 1): -2.0, (0, 1): 0.5}
        result = solver.solve_qubo(Q)
        assert result["hardware_used"] is False
        assert "solution" in result

    def test_ibm_backend_no_sdk(self):
        from quantum.vendors.ibm_quantum import IBMQuantumBackend
        backend = IBMQuantumBackend(api_token=None)
        result = backend.run_circuit(2, shots=100)
        assert result["method"] == "classical_simulation"
        assert sum(result["counts"].values()) == 100

    def test_orchestrator_all_classical(self):
        from quantum.vendors.vendor_orchestrator import QuantumVendorOrchestrator
        orch = QuantumVendorOrchestrator()
        mu = np.array([0.05, 0.10])
        sigma = np.eye(2) * 0.02
        result = orch.solve_optimization("portfolio_qubo",
                                         expected_returns=mu, cov_matrix=sigma)
        # Should complete without error
        assert "error" not in result or result.get("selected_assets") is not None

    def test_redirect_imports_ibm(self):
        """ibm_provider.py should redirect to ibm_quantum."""
        from quantum.vendors.ibm_provider import IBMQuantumBackend as IB
        from quantum.vendors.ibm_quantum import IBMQuantumBackend
        assert IB is IBMQuantumBackend


# ---------------------------------------------------------------------------
# Solution quality tests
# ---------------------------------------------------------------------------


class TestSolutionQuality:
    """Verify QUBO solutions satisfy constraints."""

    def test_portfolio_qubo_valid_binary(self):
        from quantum.vendors.dwave_solver import DWaveSolver
        solver = DWaveSolver()
        mu = np.array([0.05, 0.10, 0.03, 0.07, 0.02])
        sigma = np.eye(5) * 0.02
        result = solver.portfolio_optimize(mu, sigma, budget=3)
        for idx in result["selected_assets"]:
            assert isinstance(idx, (int, np.integer))
            assert 0 <= idx < 5

    def test_signal_select_valid_indices(self):
        from quantum.vendors.dwave_solver import DWaveSolver
        solver = DWaveSolver()
        confidences = [0.9, 0.7, 0.3, 0.8, 0.5, 0.6]
        result = solver.signal_select(confidences, max_signals=3)
        for idx in result["selected_indices"]:
            assert isinstance(idx, (int, np.integer))
            assert 0 <= idx < 6

    def test_vqe_weights_nonnegative(self):
        from quantum.vendors.ibm_quantum import IBMQuantumBackend
        backend = IBMQuantumBackend()
        mu = np.array([0.05, 0.10, 0.03])
        sigma = np.eye(3) * 0.02
        result = backend.vqe_portfolio(mu, sigma, max_iterations=20)
        for w in result["weights"].values():
            assert w >= 0.0, f"Negative weight: {w}"
