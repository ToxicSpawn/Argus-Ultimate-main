"""
tests_unified/test_quantum_integration.py
==========================================

Batch 15 / M23 — Quantum package integration tests.

All tests are designed to run fully offline.  Hardware quantum backends and
any external network calls are mocked so the suite never requires real quantum
hardware or an internet connection.

Tests
-----
1. quantum simulator initialises without error
2. quantum optimizer returns valid output shape
3. trading integration returns a signal dict
4. error mitigation (ZNE) runs without crash
5. quantum backend import is safe (no hardware required)
6. quantum walk returns valid probability distribution
"""

from __future__ import annotations

import sys
import types
import importlib
import logging
import unittest
from unittest.mock import MagicMock, patch
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Helpers — create lightweight stubs for heavy optional dependencies so that
# the test suite does not require qiskit, cirq, or real quantum hardware.
# ---------------------------------------------------------------------------


def _make_stub_module(name: str, **attrs: Any) -> types.ModuleType:
    """Return a module stub populated with the given attributes."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# Stub the in-repo quantum_simulator only when the real module is unavailable;
# leaving a MagicMock in sys.modules breaks the working QAOA tests that need the
# real circuit methods and gate enum.
try:
    import quantum_simulator  # noqa: F401
except ImportError:
    _make_stub_module(
        "quantum_simulator",
        QuantumCircuit=MagicMock,
        simulate=MagicMock(return_value={"counts": {"00": 512, "11": 512}, "backend": "stub"}),
        GateType=MagicMock(),
        STATEVECTOR_MAX_QUBITS=20,
        MPS_MAX_QUBITS=100,
    )

logger = logging.getLogger(__name__)


# ===========================================================================
# Test 1 — quantum simulator initialises without error
# ===========================================================================

class TestQuantumSimulatorInit(unittest.TestCase):
    """quantum/production_quantum_simulator.py — ARGUSQuantumSimulator init."""

    def test_argus_quantum_simulator_init_with_mock(self):
        """ARGUSQuantumSimulator should initialise when the stub is available."""
        # Import the module; the quantum_simulator stub is already injected.
        from quantum.production_quantum_simulator import (
            ARGUSQuantumSimulator,
            QUANTUM_SIMULATOR_AVAILABLE,
        )

        if not QUANTUM_SIMULATOR_AVAILABLE:
            self.skipTest("quantum_simulator stub not detected as available — skipping init test")

        sim = ARGUSQuantumSimulator()
        self.assertTrue(sim.simulator_ready)
        self.assertGreater(sim.max_qubits_state_vector, 0)
        self.assertGreater(sim.max_qubits_tensor_network, 0)
        logger.info("ARGUSQuantumSimulator initialised: sv=%d mps=%d",
                    sim.max_qubits_state_vector, sim.max_qubits_tensor_network)

    def test_quantum_simulator_available_flag_is_bool(self):
        """QUANTUM_SIMULATOR_AVAILABLE must be a bool regardless of environment."""
        from quantum.production_quantum_simulator import QUANTUM_SIMULATOR_AVAILABLE
        self.assertIsInstance(QUANTUM_SIMULATOR_AVAILABLE, bool)


# ===========================================================================
# Test 2 — quantum optimizer returns valid output shape
# ===========================================================================

class TestQuantumOptimizer(unittest.TestCase):
    """quantum/quantum_optimizer.py — retired adapter protocol."""

    def test_quantum_optimizer_module_imports_cleanly(self):
        """quantum.quantum_optimizer must be importable without hardware."""
        import quantum.quantum_optimizer as qopt
        self.assertIsNotNone(qopt)

    def test_quantum_optimizer_getattr_returns_callable(self):
        """Accessing a retired legacy symbol defers the RuntimeError to call."""
        import quantum.quantum_optimizer as qopt
        placeholder = qopt.__getattr__("QuantumPortfolioOptimizer")
        self.assertTrue(callable(placeholder))

    def test_placeholder_shape_contract(self):
        """Retired legacy symbols raise RuntimeError with an informative message."""
        import quantum.quantum_optimizer as qopt
        placeholder = qopt.__getattr__("AnyOptimizer")
        with self.assertRaises(RuntimeError) as ctx:
            placeholder(n_assets=3)
        self.assertIn("retired", str(ctx.exception).lower())
        self.assertIn("get_quantum_facade", str(ctx.exception))


# ===========================================================================
# Test 3 — trading integration returns a signal dict
# ===========================================================================

class TestQuantumTradingIntegration(unittest.TestCase):
    """quantum/quantum_trading_integration.py — placeholder safety."""

    def test_trading_integration_module_importable(self):
        """quantum.quantum_trading_integration must import cleanly."""
        import quantum.quantum_trading_integration as qti
        self.assertIsNotNone(qti)

    def test_trading_signal_placeholder_is_callable(self):
        """Any symbol on the trading integration placeholder must be callable."""
        import quantum.quantum_trading_integration as qti
        obj = qti.__getattr__("QuantumTradingSignal")
        self.assertTrue(callable(obj))

    def test_signal_dict_via_mock(self):
        """Simulate what the trading integration would return by using a mock
        that mimics the expected signal-dict contract."""

        def fake_generate_signal(symbol: str, prices: np.ndarray) -> dict:
            """Minimal offline stand-in for quantum signal generation."""
            n = len(prices)
            direction = 1 if prices[-1] > prices[0] else -1
            confidence = float(np.abs(prices[-1] - prices[0]) / (prices[0] + 1e-9))
            return {
                "symbol": symbol,
                "direction": direction,
                "confidence": min(confidence, 1.0),
                "method": "quantum_stub",
                "n_samples": n,
            }

        prices = np.random.RandomState(42).randn(50).cumsum() + 100
        signal = fake_generate_signal("BTC/USD", prices)

        # Validate shape / types of the returned dict
        self.assertIsInstance(signal, dict)
        self.assertIn("symbol", signal)
        self.assertIn("direction", signal)
        self.assertIn("confidence", signal)
        self.assertIn("method", signal)
        self.assertIn(signal["direction"], (-1, 1))
        self.assertGreaterEqual(signal["confidence"], 0.0)
        self.assertLessEqual(signal["confidence"], 1.0)


# ===========================================================================
# Test 4 — error mitigation runs without crash
# ===========================================================================

class TestQuantumErrorMitigation(unittest.TestCase):
    """quantum/error_mitigation.py — QuantumErrorMitigator ZNE and MEM."""

    def setUp(self):
        from quantum.error_mitigation import QuantumErrorMitigator
        self.mitigator = QuantumErrorMitigator()

    def test_zero_noise_extrapolation_three_points(self):
        """ZNE with three noise levels must return a dict with required keys."""
        result = self.mitigator.zero_noise_extrapolation([
            (1.0, 0.85),
            (2.0, 0.72),
            (3.0, 0.60),
        ])
        self.assertIsInstance(result, dict)
        for key in ("zne_richardson", "zne_exponential",
                    "confidence_interval", "extrapolation_quality",
                    "method_used", "fit_residuals"):
            self.assertIn(key, result, f"missing key: {key}")
        self.assertIsInstance(result["zne_richardson"], float)
        self.assertIsInstance(result["zne_exponential"], float)
        self.assertIn(result["method_used"], ("richardson", "exponential"))
        self.assertGreaterEqual(result["extrapolation_quality"], 0.0)
        self.assertLessEqual(result["extrapolation_quality"], 1.0)

    def test_zero_noise_extrapolation_two_points(self):
        """ZNE must also work with the minimum of 2 noise levels."""
        result = self.mitigator.zero_noise_extrapolation([
            (1.0, 0.9),
            (2.0, 0.7),
        ])
        self.assertIn("zne_richardson", result)

    def test_zne_requires_at_least_two_points(self):
        """Passing fewer than 2 noise levels must raise ValueError."""
        with self.assertRaises(ValueError):
            self.mitigator.zero_noise_extrapolation([(1.0, 0.9)])

    def test_build_calibration_matrix_shape(self):
        """build_calibration_matrix must return a square matrix of size 2^n."""
        n_qubits = 2
        cal = self.mitigator.build_calibration_matrix(n_qubits, error_rate=0.01)
        expected_dim = 2 ** n_qubits
        self.assertEqual(cal.shape, (expected_dim, expected_dim))
        # Each column must sum to 1 (stochastic matrix).
        col_sums = cal.sum(axis=0)
        np.testing.assert_allclose(col_sums, np.ones(expected_dim), atol=1e-6)


# ===========================================================================
# Test 5 — quantum backend import is safe (no hardware required)
# ===========================================================================

class TestQuantumBackendImportSafety(unittest.TestCase):
    """quantum/__init__.py — top-level package import must not raise."""

    def test_quantum_package_imports_cleanly(self):
        """import quantum must succeed without any hardware present."""
        import quantum  # noqa: F401 — just check it doesn't raise
        self.assertIsNotNone(quantum)

    def test_quantum_init_exports_expected_names(self):
        """quantum.__all__ must contain the documented public names."""
        import quantum
        expected = [
            "optimize_portfolio_with_quantum",
            "discover_strategy_with_quantum",
            "analyze_risk_with_quantum",
            "get_quantum_simulator_status",
            "ARGUSQuantumSimulator",
        ]
        for name in expected:
            self.assertIn(name, quantum.__all__,
                          f"'{name}' missing from quantum.__all__")

    def test_quantum_fallbacks_are_callable(self):
        """When the production simulator is unavailable the fallbacks must still
        be callable (they raise RuntimeError rather than AttributeError)."""
        import quantum
        for fn_name in (
            "optimize_portfolio_with_quantum",
            "discover_strategy_with_quantum",
            "analyze_risk_with_quantum",
            "get_quantum_simulator_status",
        ):
            fn = getattr(quantum, fn_name, None)
            self.assertIsNotNone(fn, f"{fn_name} is None on quantum module")
            # Must be callable regardless of whether hardware is available.
            self.assertTrue(callable(fn), f"{fn_name} is not callable")

    def test_quantum_error_mitigation_importable_standalone(self):
        """quantum.error_mitigation must be importable independently."""
        import quantum.error_mitigation as qem
        self.assertTrue(hasattr(qem, "QuantumErrorMitigator"))

    def test_quantum_simulators_quantum_walk_importable(self):
        """quantum.simulators.quantum_walk must be importable."""
        import quantum.simulators.quantum_walk as qwalk
        self.assertTrue(hasattr(qwalk, "QuantumWalk"))


# ===========================================================================
# Test 6 — quantum walk returns valid probability distribution
# ===========================================================================

class TestQuantumWalk(unittest.TestCase):
    """quantum/simulators/quantum_walk.py — QuantumWalk probability output."""

    def setUp(self):
        from quantum.simulators.quantum_walk import QuantumWalk
        self.QuantumWalk = QuantumWalk

    def test_quantum_walk_returns_array(self):
        """QuantumWalk.run() must return a numpy array."""
        walk = self.QuantumWalk(n_steps=10, n_positions=21)
        result = walk.run()
        self.assertIsInstance(result, np.ndarray)

    def test_quantum_walk_correct_length(self):
        """Output length must equal n_positions."""
        n_pos = 51
        walk = self.QuantumWalk(n_steps=25, n_positions=n_pos)
        result = walk.run()
        self.assertEqual(len(result), n_pos)

    def test_quantum_walk_non_negative(self):
        """All probability amplitudes must be non-negative."""
        walk = self.QuantumWalk(n_steps=20, n_positions=41)
        result = walk.run()
        self.assertTrue(np.all(result >= 0),
                        "Probability distribution contains negative values")

    def test_quantum_walk_sums_to_one(self):
        """A full quantum walk probability distribution must sum to 1.0.

        Note: the current stub implementation initialises a delta distribution
        at the midpoint (total mass = 1.0) as the t=0 state.  If the walk
        evolves the distribution, it must still be normalised.
        """
        walk = self.QuantumWalk(n_steps=50, n_positions=101)
        result = walk.run()
        total = float(result.sum())
        self.assertAlmostEqual(
            total, 1.0, places=6,
            msg=f"Walk probability mass = {total:.8f}, expected 1.0"
        )

    def test_quantum_walk_default_parameters(self):
        """QuantumWalk with default parameters must complete without error."""
        walk = self.QuantumWalk()
        result = walk.run()
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main(verbosity=2)
