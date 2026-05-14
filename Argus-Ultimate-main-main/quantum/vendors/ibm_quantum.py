"""
IBM Quantum (Qiskit Runtime) integration for ARGUS.

Real IBM Quantum access with graceful fallback chain:
1. IBM Quantum hardware via Qiskit Runtime (free 10 min/month)
2. Qiskit Aer local simulator
3. Classical numpy simulation

Environment:
    IBM_QUANTUM_TOKEN: IBM Quantum API token (optional)

Install:
    pip install qiskit qiskit-ibm-runtime qiskit-aer   (optional, falls back gracefully)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy SDK detection
# ---------------------------------------------------------------------------

_HAS_QISKIT = False
_HAS_AER = False
_HAS_RUNTIME = False

try:
    from qiskit import QuantumCircuit  # type: ignore
    from qiskit.circuit.library import RealAmplitudes  # type: ignore
    _HAS_QISKIT = True
except ImportError:
    pass

try:
    from qiskit_aer import AerSimulator  # type: ignore
    _HAS_AER = True
except ImportError:
    pass

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2  # type: ignore
    _HAS_RUNTIME = True
except ImportError:
    pass


def _classical_simulate_circuit(n_qubits: int, shots: int) -> Dict[str, int]:
    """Minimal classical simulation: uniform random bitstrings."""
    counts: Dict[str, int] = {}
    for _ in range(shots):
        bits = "".join(str(np.random.randint(0, 2)) for _ in range(n_qubits))
        counts[bits] = counts.get(bits, 0) + 1
    return counts


class IBMQuantumBackend:
    """
    Real IBM Quantum access via Qiskit Runtime.

    Uses IBM_QUANTUM_TOKEN env var or passed token.
    Free tier: 10 minutes/month on real hardware.
    Falls back to Qiskit Aer simulator, then to classical simulation.
    """

    def __init__(self, api_token: Optional[str] = None,
                 backend_name: Optional[str] = None):
        self._token = api_token or os.environ.get("IBM_QUANTUM_TOKEN")
        self._backend_name = backend_name
        self._service = None
        self._backend = None

        self._has_qiskit = _HAS_QISKIT
        self._has_aer = _HAS_AER
        self._has_runtime = _HAS_RUNTIME and bool(self._token)

        # Track usage
        self._jobs_run = 0
        self._total_hw_ms = 0.0

        # Try to init runtime service
        if self._has_runtime:
            try:
                self._service = QiskitRuntimeService(
                    channel="ibm_quantum",
                    token=self._token,
                )
                if self._backend_name:
                    self._backend = self._service.backend(self._backend_name)
                else:
                    # Get least busy backend
                    backends = self._service.backends(
                        simulator=False, operational=True, min_num_qubits=5
                    )
                    if backends:
                        self._backend = backends[0]
                        self._backend_name = self._backend.name
                    else:
                        logger.warning("No operational IBM backends found")
                        self._has_runtime = False
            except Exception as exc:
                logger.warning("IBM Quantum Runtime init failed: %s", exc)
                self._has_runtime = False
                self._service = None

        level = ("hardware" if self._has_runtime
                 else ("aer-sim" if self._has_aer
                       else ("qiskit-classical" if self._has_qiskit
                             else "numpy-classical")))
        logger.info("IBMQuantumBackend initialized: level=%s, qiskit=%s, aer=%s, runtime=%s",
                     level, self._has_qiskit, self._has_aer, self._has_runtime)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_on_hardware(self, circuit, shots: int) -> Dict[str, Any]:
        """Run circuit on real IBM hardware via Qiskit Runtime."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # type: ignore

        pm = generate_preset_pass_manager(backend=self._backend, optimization_level=1)
        transpiled = pm.run(circuit)

        sampler = SamplerV2(backend=self._backend)
        t0 = time.perf_counter()
        job = sampler.run([transpiled], shots=shots)
        result = job.result()
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._total_hw_ms += elapsed

        # Extract counts from SamplerV2 result
        pub_result = result[0]
        counts = {}
        if hasattr(pub_result, "data"):
            # Qiskit Runtime v2 format
            for key in pub_result.data:
                creg = pub_result.data[key]
                if hasattr(creg, "get_counts"):
                    counts = creg.get_counts()
                    break

        return {
            "counts": counts,
            "method": "ibm_hardware",
            "backend": self._backend_name,
            "shots": shots,
            "execution_time_ms": elapsed,
        }

    def _run_on_aer(self, circuit, shots: int) -> Dict[str, Any]:
        """Run circuit on Qiskit Aer simulator."""
        sim = AerSimulator()
        from qiskit import transpile  # type: ignore
        transpiled = transpile(circuit, sim)
        t0 = time.perf_counter()
        result = sim.run(transpiled, shots=shots).result()
        elapsed = (time.perf_counter() - t0) * 1000.0
        counts = result.get_counts()
        return {
            "counts": dict(counts),
            "method": "qiskit_aer_simulator",
            "backend": "aer_simulator",
            "shots": shots,
            "execution_time_ms": elapsed,
        }

    def _run_classical(self, n_qubits: int, shots: int) -> Dict[str, Any]:
        """Fallback: classical random simulation."""
        t0 = time.perf_counter()
        counts = _classical_simulate_circuit(n_qubits, shots)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "counts": counts,
            "method": "classical_simulation",
            "backend": "numpy",
            "shots": shots,
            "execution_time_ms": elapsed,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_circuit(self, circuit: Any, shots: int = 1000,
                    mitigate_errors: bool = False) -> Dict[str, Any]:
        """
        Run a quantum circuit.

        Priority: IBM hardware -> Aer simulator -> classical simulation.
        Optionally applies measurement error mitigation to the results.

        Args:
            circuit: A qiskit QuantumCircuit, or an int (number of qubits for
                     classical fallback).
            shots: Number of measurement shots.
            mitigate_errors: If True, apply measurement error mitigation
                to the raw counts using a calibration matrix.

        Returns:
            counts: dict of bitstring -> count
            method: str, which backend was used
            backend: str, backend name
            shots: int
            execution_time_ms: float
            mitigated: bool, whether error mitigation was applied
            raw_counts: dict (only present if mitigated=True)
        """
        self._jobs_run += 1

        # Determine n_qubits for classical fallback
        n_qubits = getattr(circuit, "num_qubits", 2)
        if isinstance(circuit, int):
            n_qubits = circuit

        result = None

        # --- Try IBM hardware ---
        if self._has_runtime and self._backend is not None and self._has_qiskit:
            try:
                # Ensure circuit has measurements
                if hasattr(circuit, "num_clbits") and circuit.num_clbits == 0:
                    circuit = circuit.copy()
                    circuit.measure_all()
                result = self._run_on_hardware(circuit, shots)
            except Exception as exc:
                logger.warning("IBM hardware run failed, falling back: %s", exc)

        # --- Try Aer simulator ---
        if result is None and self._has_aer and self._has_qiskit and not isinstance(circuit, int):
            try:
                if hasattr(circuit, "num_clbits") and circuit.num_clbits == 0:
                    circuit = circuit.copy()
                    circuit.measure_all()
                result = self._run_on_aer(circuit, shots)
            except Exception as exc:
                logger.warning("Aer simulator failed, falling back: %s", exc)

        # --- Classical fallback ---
        if result is None:
            result = self._run_classical(n_qubits, shots)

        # --- Apply measurement error mitigation if requested ---
        if mitigate_errors and result.get("counts"):
            try:
                from quantum.error_mitigation import QuantumErrorMitigator
                mitigator = QuantumErrorMitigator()
                cal_matrix = mitigator.build_calibration_matrix(n_qubits, error_rate=0.01)
                mitigation = mitigator.measurement_error_mitigation(
                    result["counts"], cal_matrix
                )
                result["raw_counts"] = dict(result["counts"])
                result["counts"] = {
                    k: int(round(v))
                    for k, v in mitigation["mitigated_counts"].items()
                }
                result["mitigated"] = True
                result["fidelity_improvement"] = mitigation["fidelity_improvement"]
            except Exception as exc:
                logger.warning("Error mitigation failed: %s", exc)
                result["mitigated"] = False
        else:
            result["mitigated"] = False

        return result

    def vqe_portfolio(
        self,
        expected_returns: Any,
        cov_matrix: Any,
        n_layers: int = 1,
        max_iterations: int = 100,
        risk_aversion: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Run VQE for portfolio optimization.

        Build ansatz: RealAmplitudes circuit.
        Cost function: risk_aversion * portfolio_variance - (1-risk_aversion) * expected_return.
        Optimize: COBYLA (classical optimizer on parameterized circuit).

        Falls back to classical scipy optimization if Qiskit is unavailable.

        Args:
            expected_returns: 1D array of expected returns per asset.
            cov_matrix: 2D covariance matrix.
            n_layers: Number of ansatz layers (circuit depth).
            max_iterations: Maximum optimizer iterations.
            risk_aversion: Trade-off between return and risk.

        Returns:
            optimal_params: list of optimized parameters
            weights: dict of asset index -> weight
            expected_return: float
            risk: float
            n_iterations: int
            method: str
        """
        mu = np.asarray(expected_returns, dtype=float).ravel()
        sigma = np.asarray(cov_matrix, dtype=float)
        n = len(mu)

        if self._has_qiskit:
            return self._vqe_qiskit(mu, sigma, n_layers, max_iterations, risk_aversion)
        else:
            return self._vqe_classical(mu, sigma, max_iterations, risk_aversion)

    def _vqe_qiskit(
        self, mu: np.ndarray, sigma: np.ndarray,
        n_layers: int, max_iterations: int, risk_aversion: float,
    ) -> Dict[str, Any]:
        """VQE using Qiskit circuits."""
        from qiskit import QuantumCircuit  # type: ignore
        from qiskit.circuit.library import RealAmplitudes  # type: ignore

        n = len(mu)

        # Build ansatz
        ansatz = RealAmplitudes(n, reps=n_layers)
        n_params = ansatz.num_parameters

        iteration_count = [0]
        convergence: List[float] = []

        def cost_function(params):
            """Evaluate portfolio cost via circuit simulation."""
            iteration_count[0] += 1

            # Bind parameters and simulate
            bound = ansatz.assign_parameters(params)
            bound.measure_all()

            result = self.run_circuit(bound, shots=500)
            counts = result.get("counts", {})

            if not counts:
                return float("inf")

            total = sum(counts.values())
            # Compute expected portfolio weights from measurement probabilities
            weights = np.zeros(n)
            for bitstring, count in counts.items():
                bits = bitstring.replace(" ", "")
                for i, b in enumerate(bits[:n]):
                    if b == "1":
                        weights[i] += count
            weights /= max(total, 1)
            w_sum = weights.sum()
            if w_sum > 0:
                weights /= w_sum

            ret = float(weights @ mu)
            risk = float(weights @ sigma @ weights)
            cost = risk_aversion * risk - (1.0 - risk_aversion) * ret
            convergence.append(cost)
            return cost

        # Optimize with COBYLA
        from scipy.optimize import minimize as scipy_minimize
        x0 = np.random.uniform(-np.pi, np.pi, n_params)
        opt_result = scipy_minimize(
            cost_function, x0, method="COBYLA",
            options={"maxiter": max_iterations, "rhobeg": 0.5},
        )

        # Final weights from optimized parameters
        final_params = opt_result.x
        bound = ansatz.assign_parameters(final_params)
        bound.measure_all()
        final_result = self.run_circuit(bound, shots=2000)
        final_counts = final_result.get("counts", {})

        weights = np.zeros(n)
        total = sum(final_counts.values()) if final_counts else 1
        for bitstring, count in final_counts.items():
            bits = bitstring.replace(" ", "")
            for i, b in enumerate(bits[:n]):
                if b == "1":
                    weights[i] += count
        weights /= max(total, 1)
        w_sum = weights.sum()
        if w_sum > 0:
            weights /= w_sum

        exp_ret = float(weights @ mu)
        risk = float(weights @ sigma @ weights)

        return {
            "optimal_params": final_params.tolist(),
            "weights": {i: float(weights[i]) for i in range(n) if weights[i] > 0.01},
            "expected_return": exp_ret,
            "risk": risk,
            "n_iterations": iteration_count[0],
            "convergence": convergence[-10:],
            "method": final_result.get("method", "vqe_qiskit"),
        }

    def _vqe_classical(
        self, mu: np.ndarray, sigma: np.ndarray,
        max_iterations: int, risk_aversion: float,
    ) -> Dict[str, Any]:
        """Classical VQE fallback using scipy optimization."""
        from scipy.optimize import minimize as scipy_minimize

        n = len(mu)
        convergence: List[float] = []

        def cost_function(w):
            # Normalize weights to sum to 1
            weights = np.exp(w) / np.exp(w).sum()
            ret = float(weights @ mu)
            risk = float(weights @ sigma @ weights)
            cost = risk_aversion * risk - (1.0 - risk_aversion) * ret
            convergence.append(cost)
            return cost

        x0 = np.zeros(n)
        result = scipy_minimize(
            cost_function, x0, method="COBYLA",
            options={"maxiter": max_iterations},
        )

        # Final weights
        final_w = np.exp(result.x) / np.exp(result.x).sum()
        exp_ret = float(final_w @ mu)
        risk = float(final_w @ sigma @ final_w)

        return {
            "optimal_params": result.x.tolist(),
            "weights": {i: float(final_w[i]) for i in range(n) if final_w[i] > 0.01},
            "expected_return": exp_ret,
            "risk": risk,
            "n_iterations": len(convergence),
            "convergence": convergence[-10:],
            "method": "classical_scipy_vqe",
        }

    def get_backend_info(self) -> Dict[str, Any]:
        """
        Return backend availability information.

        Returns:
            available: bool
            backend_name: str
            n_qubits: int
            has_hardware: bool
            has_qiskit: bool
            has_aer: bool
            jobs_run: int
        """
        info: Dict[str, Any] = {
            "available": True,  # classical fallback always available
            "has_qiskit": self._has_qiskit,
            "has_aer": self._has_aer,
            "has_hardware": self._has_runtime,
            "jobs_run": self._jobs_run,
            "total_hardware_ms": self._total_hw_ms,
        }

        if self._has_runtime and self._backend is not None:
            info["backend_name"] = self._backend_name or "unknown"
            config = getattr(self._backend, "configuration", None)
            if config and callable(config):
                cfg = config()
                info["n_qubits"] = getattr(cfg, "n_qubits", 0)
            else:
                info["n_qubits"] = getattr(self._backend, "num_qubits", 0)
            info["status"] = "operational"
        elif self._has_aer:
            info["backend_name"] = "aer_simulator"
            info["n_qubits"] = 30  # Aer can simulate ~30 qubits
            info["status"] = "simulator"
        else:
            info["backend_name"] = "classical_numpy"
            info["n_qubits"] = 0
            info["status"] = "classical_fallback"

        return info
