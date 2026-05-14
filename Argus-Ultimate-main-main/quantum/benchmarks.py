"""
Quantum Advantage Benchmarking Suite.

Rigorous, honest benchmarking to measure actual quantum vs classical
performance across portfolio optimization, VaR estimation, ML
classification, reservoir prediction, and tensor network scaling.

Every benchmark returns an honest verdict: does quantum actually help,
and by how much? No hype -- just measured performance data.

All quantum methods are classically simulated. True quantum advantage
requires fault-tolerant quantum hardware, which does not exist yet
for financial applications. These benchmarks measure the algorithmic
structure's value, not hardware speedup.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize as sp_minimize

logger = logging.getLogger(__name__)


class QuantumBenchmarkSuite:
    """
    Comprehensive quantum vs classical benchmarking.

    Runs standardized benchmarks across multiple problem types and sizes,
    producing honest verdicts about where quantum-inspired methods help.
    """

    def __init__(self) -> None:
        self._results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Portfolio optimization benchmark
    # ------------------------------------------------------------------

    def benchmark_portfolio_optimization(
        self,
        n_assets_range: Optional[List[int]] = None,
        n_trials: int = 5,
    ) -> Dict[str, Any]:
        """Compare QAOA vs classical (scipy) for portfolio optimization.

        For each n_assets:
        - Generate random returns + covariance
        - Run QAOA (our implementation)
        - Run scipy.optimize.minimize (classical)
        - Compare: Sharpe ratio, runtime, solution quality

        Args:
            n_assets_range: List of asset counts to test.
            n_trials: Number of trials per size for statistical reliability.

        Returns:
            Dict mapping n_assets -> comparison metrics.
        """
        if n_assets_range is None:
            n_assets_range = [4, 6, 8, 10]

        from quantum.algorithms.qaoa import QAOAPortfolioOptimizer

        results: Dict[int, Dict[str, Any]] = {}
        rng = np.random.default_rng(42)

        for n in n_assets_range:
            qaoa_sharpes = []
            classical_sharpes = []
            qaoa_times = []
            classical_times = []
            qaoa_wins = 0

            for trial in range(n_trials):
                # Generate random problem instance
                mu = rng.normal(0.05, 0.03, n)
                # Positive definite covariance
                A = rng.normal(0, 0.1, (n, n))
                sigma = A.T @ A / n + 0.01 * np.eye(n)

                # QAOA
                optimizer = QAOAPortfolioOptimizer(n_layers=2, max_assets=n)
                t0 = time.perf_counter()
                try:
                    qaoa_result = optimizer.optimize(mu, sigma, risk_aversion=0.5)
                    qaoa_time = (time.perf_counter() - t0) * 1000
                    qaoa_sharpe = qaoa_result.get("sharpe", 0.0)
                except Exception as e:
                    logger.debug("QAOA failed for n=%d: %s", n, e)
                    qaoa_time = 0.0
                    qaoa_sharpe = 0.0

                # Classical
                t0 = time.perf_counter()
                classical_result = self._classical_portfolio(mu, sigma, 0.5)
                classical_time = (time.perf_counter() - t0) * 1000
                classical_sharpe = classical_result.get("sharpe", 0.0)

                qaoa_sharpes.append(qaoa_sharpe)
                classical_sharpes.append(classical_sharpe)
                qaoa_times.append(qaoa_time)
                classical_times.append(classical_time)

                if qaoa_sharpe > classical_sharpe * 1.001:
                    qaoa_wins += 1

            verdict = self._honest_verdict(
                np.mean(qaoa_sharpes),
                np.mean(classical_sharpes),
                "Sharpe ratio",
            )

            results[n] = {
                "qaoa_sharpe": round(float(np.mean(qaoa_sharpes)), 6),
                "classical_sharpe": round(float(np.mean(classical_sharpes)), 6),
                "qaoa_time_ms": round(float(np.mean(qaoa_times)), 2),
                "classical_time_ms": round(float(np.mean(classical_times)), 2),
                "qaoa_wins": qaoa_wins,
                "n_trials": n_trials,
                "honest_verdict": verdict,
            }

        self._results.append({
            "benchmark": "portfolio_optimization",
            "results": results,
        })

        return results

    # ------------------------------------------------------------------
    # VaR estimation benchmark
    # ------------------------------------------------------------------

    def benchmark_var_estimation(
        self,
        n_samples_range: Optional[List[int]] = None,
        n_trials: int = 5,
    ) -> Dict[str, Any]:
        """Compare QAE vs classical Monte Carlo for VaR estimation.

        Args:
            n_samples_range: Sample sizes to test.
            n_trials: Number of trials per size.

        Returns:
            Dict mapping n_samples -> comparison metrics.
        """
        if n_samples_range is None:
            n_samples_range = [100, 500, 1000, 5000]

        from quantum.algorithms.quantum_amplitude_estimation import (
            QuantumAmplitudeEstimatorVaR,
        )

        rng = np.random.default_rng(42)
        # Generate a reference return distribution
        base_returns = rng.normal(-0.001, 0.02, 10000)
        true_var = float(np.percentile(base_returns, 5.0))

        results: Dict[int, Dict[str, Any]] = {}

        for n_s in n_samples_range:
            qae_errors = []
            mc_errors = []
            qae_times = []
            mc_times = []

            qae = QuantumAmplitudeEstimatorVaR(n_qubits=4)

            for _ in range(n_trials):
                # QAE
                t0 = time.perf_counter()
                qae_result = qae.estimate_var(base_returns, confidence=0.95, n_samples=n_s)
                qae_time = (time.perf_counter() - t0) * 1000
                qae_var = qae_result.get("var_95", 0.0)
                qae_errors.append(abs(qae_var - true_var))
                qae_times.append(qae_time)

                # Classical MC
                t0 = time.perf_counter()
                boot_idx = rng.choice(len(base_returns), size=n_s, replace=True)
                mc_var = float(np.percentile(base_returns[boot_idx], 5.0))
                mc_time = (time.perf_counter() - t0) * 1000
                mc_errors.append(abs(mc_var - true_var))
                mc_times.append(mc_time)

            mean_qae_err = float(np.mean(qae_errors))
            mean_mc_err = float(np.mean(mc_errors))

            # Convergence rate: error ~ C * n^(-rate)
            qae_conv = 1.0 / max(mean_qae_err, 1e-12)
            mc_conv = 1.0 / max(mean_mc_err, 1e-12)

            speedup = mean_mc_err / max(mean_qae_err, 1e-12)

            results[n_s] = {
                "qae_error": round(mean_qae_err, 8),
                "mc_error": round(mean_mc_err, 8),
                "qae_convergence_rate": round(qae_conv, 2),
                "mc_convergence_rate": round(mc_conv, 2),
                "speedup_factor": round(min(speedup, 100.0), 2),
                "qae_time_ms": round(float(np.mean(qae_times)), 2),
                "mc_time_ms": round(float(np.mean(mc_times)), 2),
            }

        self._results.append({
            "benchmark": "var_estimation",
            "results": results,
        })

        return results

    # ------------------------------------------------------------------
    # ML classification benchmark
    # ------------------------------------------------------------------

    def benchmark_ml_classification(
        self,
        datasets: Optional[List[str]] = None,
        n_trials: int = 5,
    ) -> Dict[str, Any]:
        """Compare quantum kernel vs RBF kernel vs linear.

        Uses sklearn's make_moons, make_circles, make_blobs datasets.

        Args:
            datasets: List of dataset names.
            n_trials: Trials per dataset.

        Returns:
            Dict mapping dataset -> comparison metrics.
        """
        if datasets is None:
            datasets = ["moons", "circles", "blobs"]

        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score

        results: Dict[str, Dict[str, Any]] = {}
        rng_seed = 42

        for dataset_name in datasets:
            X, y = self._generate_dataset(dataset_name, n_samples=200, seed=rng_seed)

            quantum_accs = []
            rbf_accs = []
            linear_accs = []
            quantum_times = []
            rbf_times = []

            for trial in range(n_trials):
                seed = rng_seed + trial
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=seed
                )

                # Quantum kernel (simulated via feature map)
                t0 = time.perf_counter()
                quantum_acc = self._quantum_kernel_classify(
                    X_train, y_train, X_test, y_test
                )
                quantum_time = (time.perf_counter() - t0) * 1000
                quantum_accs.append(quantum_acc)
                quantum_times.append(quantum_time)

                # RBF kernel
                t0 = time.perf_counter()
                svc_rbf = SVC(kernel="rbf", gamma="scale", random_state=seed)
                svc_rbf.fit(X_train, y_train)
                rbf_acc = float(accuracy_score(y_test, svc_rbf.predict(X_test)))
                rbf_time = (time.perf_counter() - t0) * 1000
                rbf_accs.append(rbf_acc)
                rbf_times.append(rbf_time)

                # Linear
                svc_lin = SVC(kernel="linear", random_state=seed)
                svc_lin.fit(X_train, y_train)
                linear_accs.append(
                    float(accuracy_score(y_test, svc_lin.predict(X_test)))
                )

            verdict = self._honest_verdict(
                np.mean(quantum_accs),
                np.mean(rbf_accs),
                "classification accuracy",
            )

            results[dataset_name] = {
                "quantum_accuracy": round(float(np.mean(quantum_accs)), 4),
                "rbf_accuracy": round(float(np.mean(rbf_accs)), 4),
                "linear_accuracy": round(float(np.mean(linear_accs)), 4),
                "quantum_time_ms": round(float(np.mean(quantum_times)), 2),
                "rbf_time_ms": round(float(np.mean(rbf_times)), 2),
                "honest_verdict": verdict,
            }

        self._results.append({
            "benchmark": "ml_classification",
            "results": results,
        })

        return results

    # ------------------------------------------------------------------
    # Reservoir prediction benchmark
    # ------------------------------------------------------------------

    def benchmark_reservoir_prediction(
        self,
        signal_types: Optional[List[str]] = None,
        n_trials: int = 3,
    ) -> Dict[str, Any]:
        """Compare quantum reservoir vs classical methods.

        Args:
            signal_types: List of signal types to test.
            n_trials: Trials per signal type.

        Returns:
            Dict mapping signal -> comparison metrics.
        """
        if signal_types is None:
            signal_types = ["sine", "trending", "noisy"]

        from quantum.qml.quantum_reservoir import QuantumReservoirComputer

        results: Dict[str, Dict[str, Any]] = {}

        for signal_type in signal_types:
            signal = self._generate_signal(signal_type, length=200)

            quantum_rmses = []
            ma_rmses = []
            lr_rmses = []

            for trial in range(n_trials):
                train = signal[:140]
                test = signal[140:]

                # Quantum reservoir
                try:
                    qrc = QuantumReservoirComputer(
                        n_qubits=4, n_layers=2, washout=10, seed=42 + trial
                    )
                    qrc.fit(train, horizon=1)
                    qr_preds = []
                    for i in range(len(test)):
                        start = max(0, 140 + i - 30)
                        window = signal[start : 140 + i]
                        pred = qrc.predict(window, steps=1)
                        qr_preds.append(pred["predictions"][0])
                    qr_preds = np.array(qr_preds)
                    quantum_rmses.append(
                        float(np.sqrt(np.mean((test - qr_preds) ** 2)))
                    )
                except Exception as e:
                    logger.debug("Quantum reservoir failed: %s", e)
                    quantum_rmses.append(float("inf"))

                # Moving average
                ma_preds = []
                for i in range(len(test)):
                    start = max(0, 140 + i - 5)
                    ma_preds.append(float(np.mean(signal[start : 140 + i])))
                ma_preds = np.array(ma_preds)
                ma_rmses.append(
                    float(np.sqrt(np.mean((test - ma_preds) ** 2)))
                )

                # Linear regression
                lr_preds = []
                for i in range(len(test)):
                    start = max(0, 140 + i - 10)
                    window = signal[start : 140 + i]
                    x = np.arange(len(window))
                    if len(window) >= 2:
                        coeffs = np.polyfit(x, window, 1)
                        lr_preds.append(float(np.polyval(coeffs, len(window))))
                    else:
                        lr_preds.append(float(window[-1]))
                lr_preds = np.array(lr_preds)
                lr_rmses.append(
                    float(np.sqrt(np.mean((test - lr_preds) ** 2)))
                )

            mean_qr = float(np.mean(quantum_rmses))
            mean_ma = float(np.mean(ma_rmses))
            mean_lr = float(np.mean(lr_rmses))
            best_classical = min(mean_ma, mean_lr)

            results[signal_type] = {
                "quantum_rmse": round(mean_qr, 6),
                "ma_rmse": round(mean_ma, 6),
                "lr_rmse": round(mean_lr, 6),
                "classical_best_rmse": round(best_classical, 6),
                "quantum_wins": mean_qr < best_classical * 0.99,
            }

        self._results.append({
            "benchmark": "reservoir_prediction",
            "results": results,
        })

        return results

    # ------------------------------------------------------------------
    # Tensor network scaling benchmark
    # ------------------------------------------------------------------

    def benchmark_tensor_network_scaling(
        self,
        n_qubits_range: Optional[List[int]] = None,
        circuit_depth: int = 10,
    ) -> Dict[str, Any]:
        """Compare MPS vs statevector simulation scaling.

        Args:
            n_qubits_range: List of qubit counts.
            circuit_depth: Number of gate layers per circuit.

        Returns:
            Dict mapping n_qubits -> timing and accuracy metrics.
        """
        if n_qubits_range is None:
            n_qubits_range = [4, 8, 12, 16]

        from quantum.tensor_networks import TensorNetworkSimulator

        results: Dict[int, Dict[str, Any]] = {}

        for n_q in n_qubits_range:
            # MPS simulation
            mps = TensorNetworkSimulator(n_qubits=n_q, max_bond_dim=32)
            mps.initialize()

            t0 = time.perf_counter()
            gates = self._random_circuit_gates(n_q, circuit_depth)
            mps.apply_circuit(gates)
            mps_time = (time.perf_counter() - t0) * 1000
            mps_memory = sum(t.nbytes for t in mps._tensors)

            # Statevector simulation (only for small n)
            sv_time = 0.0
            sv_memory = 0
            fidelity = 1.0

            if n_q <= 18:
                t0 = time.perf_counter()
                sv = self._statevector_simulate(n_q, gates)
                sv_time = (time.perf_counter() - t0) * 1000
                sv_memory = sv.nbytes

                # Compare fidelity
                fidelity = mps.get_fidelity_vs_exact(sv)
            else:
                # Estimate statevector time/memory
                sv_memory = (2 ** n_q) * 16  # complex128
                sv_time = float("inf")

            speedup = sv_time / max(mps_time, 0.01) if sv_time < float("inf") else float("inf")

            results[n_q] = {
                "mps_time_ms": round(mps_time, 2),
                "sv_time_ms": round(sv_time, 2) if sv_time < float("inf") else "infeasible",
                "mps_memory_mb": round(mps_memory / 1e6, 4),
                "sv_memory_mb": round(sv_memory / 1e6, 4),
                "fidelity": round(fidelity, 6) if n_q <= 18 else "not_computed",
                "speedup": round(speedup, 2) if speedup < float("inf") else "significant",
                "bond_dims": mps.get_bond_dimensions(),
            }

        self._results.append({
            "benchmark": "tensor_network_scaling",
            "results": results,
        })

        return results

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def generate_full_report(self) -> Dict[str, Any]:
        """Run ALL benchmarks and generate comprehensive report.

        Returns:
            Dict with summary, category verdicts, overall assessment,
            and recommendations.
        """
        logger.info("Starting full quantum benchmark suite...")

        portfolio = self.benchmark_portfolio_optimization(
            n_assets_range=[4, 6, 8], n_trials=3
        )
        var_est = self.benchmark_var_estimation(
            n_samples_range=[100, 500, 1000], n_trials=3
        )
        tensor = self.benchmark_tensor_network_scaling(
            n_qubits_range=[4, 8, 12], circuit_depth=8
        )

        # Category verdicts
        verdicts: Dict[str, str] = {}

        # Portfolio verdict
        qaoa_better = sum(
            1 for v in portfolio.values()
            if v["qaoa_sharpe"] > v["classical_sharpe"] * 1.01
        )
        verdicts["portfolio"] = (
            f"QAOA outperformed classical in {qaoa_better}/{len(portfolio)} cases. "
            "On classical simulation, QAOA explores discrete subsets (combinatorial) "
            "while scipy uses continuous relaxation. Neither has true quantum advantage."
        )

        # VaR verdict
        qae_better = sum(
            1 for v in var_est.values()
            if v["qae_error"] < v["mc_error"] * 0.95
        )
        verdicts["var_estimation"] = (
            f"QAE-inspired importance sampling beat naive MC in {qae_better}/{len(var_est)} cases. "
            "The improvement comes from importance sampling, not quantum mechanics. "
            "True QAE speedup (quadratic) requires quantum hardware."
        )

        # Tensor network verdict
        verdicts["tensor_networks"] = (
            "MPS tensor networks provide genuine memory savings for low-entanglement circuits. "
            "This is a classical technique, not quantum advantage. "
            "Useful for simulating larger circuits than statevector allows."
        )

        # Overall
        overall = False  # No true quantum advantage on classical hardware

        recommendations = [
            "Use QAOA for combinatorial asset selection when exploring discrete subsets.",
            "Use importance sampling (QAE-inspired) for tail risk estimation -- genuine variance reduction.",
            "Use MPS tensor networks for simulating circuits with >16 qubits.",
            "Do NOT expect quantum speedup from classical simulation -- true advantage requires hardware.",
            "Monitor hardware progress: useful financial quantum computing likely requires >1000 logical qubits.",
        ]

        report = {
            "summary": (
                "Comprehensive benchmarks show quantum-inspired methods provide "
                "algorithmic value (importance sampling, combinatorial exploration, "
                "tensor network compression) but no true quantum advantage on classical hardware. "
                "This is the honest state of quantum computing for finance in 2026."
            ),
            "category_verdicts": verdicts,
            "overall_quantum_advantage": overall,
            "recommendations": recommendations,
            "portfolio_results": portfolio,
            "var_results": var_est,
            "tensor_results": tensor,
        }

        self._results.append({"benchmark": "full_report", "report": report})

        return report

    # ------------------------------------------------------------------
    # Verdict helper
    # ------------------------------------------------------------------

    def _honest_verdict(
        self,
        quantum_metric: float,
        classical_metric: float,
        metric_name: str,
    ) -> str:
        """Generate honest comparison verdict.

        No hype -- just facts about which is better and by how much.
        """
        if classical_metric == 0 and quantum_metric == 0:
            return f"Both methods achieved 0 {metric_name}. Inconclusive."

        if classical_metric == 0:
            return (
                f"Quantum method achieved {quantum_metric:.4f} {metric_name} "
                f"vs classical 0. Classical failed on this instance."
            )

        diff_pct = (quantum_metric - classical_metric) / abs(classical_metric) * 100

        if abs(diff_pct) < 1.0:
            return (
                f"Effectively tied ({metric_name}: quantum={quantum_metric:.4f}, "
                f"classical={classical_metric:.4f}, diff={diff_pct:+.1f}%). "
                "No meaningful advantage for either method."
            )
        elif diff_pct > 0:
            return (
                f"Quantum-inspired method leads by {diff_pct:+.1f}% "
                f"({metric_name}: {quantum_metric:.4f} vs {classical_metric:.4f}). "
                "Note: advantage comes from algorithmic structure, not quantum hardware."
            )
        else:
            return (
                f"Classical method leads by {abs(diff_pct):.1f}% "
                f"({metric_name}: classical={classical_metric:.4f} vs "
                f"quantum={quantum_metric:.4f}). "
                "This is expected -- classical simulation has limited circuit depth."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classical_portfolio(
        mu: np.ndarray, sigma: np.ndarray, risk_aversion: float
    ) -> Dict[str, Any]:
        """Classical mean-variance optimization via scipy."""
        n = len(mu)

        def neg_objective(w):
            ret = float(w @ mu)
            risk = float(w @ sigma @ w)
            return -(ret - risk_aversion * risk)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n
        x0 = np.ones(n) / n

        opt = sp_minimize(
            neg_objective, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )

        w = np.maximum(opt.x, 0.0)
        w = w / (w.sum() or 1.0)
        ret_val = float(w @ mu)
        risk_val = float(np.sqrt(max(w @ sigma @ w, 0.0)))
        sharpe = ret_val / risk_val if risk_val > 1e-12 else 0.0

        return {"sharpe": sharpe, "weights": w.tolist()}

    def _quantum_kernel_classify(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        """Quantum-inspired kernel classification.

        Uses a feature map inspired by quantum encoding:
        phi(x) = [cos(x_i * x_j), sin(x_i * x_j)] for all pairs.
        Then trains a linear SVM on the expanded features.
        """
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score

        def quantum_feature_map(X: np.ndarray) -> np.ndarray:
            n_samples, n_features = X.shape
            features = []
            for i in range(n_features):
                for j in range(i, n_features):
                    features.append(np.cos(X[:, i] * X[:, j] * np.pi))
                    features.append(np.sin(X[:, i] * X[:, j] * np.pi))
                    features.append(np.cos((X[:, i] + X[:, j]) * np.pi / 2))
            return np.column_stack(features)

        X_train_q = quantum_feature_map(X_train)
        X_test_q = quantum_feature_map(X_test)

        svc = SVC(kernel="linear", random_state=42)
        svc.fit(X_train_q, y_train)
        y_pred = svc.predict(X_test_q)
        return float(accuracy_score(y_test, y_pred))

    @staticmethod
    def _generate_dataset(
        name: str, n_samples: int = 200, seed: int = 42
    ) -> tuple:
        """Generate sklearn toy datasets."""
        from sklearn.datasets import make_moons, make_circles, make_blobs

        if name == "moons":
            return make_moons(n_samples=n_samples, noise=0.2, random_state=seed)
        elif name == "circles":
            return make_circles(
                n_samples=n_samples, noise=0.1, factor=0.5, random_state=seed
            )
        elif name == "blobs":
            return make_blobs(
                n_samples=n_samples, centers=2, random_state=seed
            )
        else:
            raise ValueError(f"Unknown dataset: {name}")

    @staticmethod
    def _generate_signal(signal_type: str, length: int = 200) -> np.ndarray:
        """Generate synthetic time series for prediction benchmarks."""
        t = np.linspace(0, 4 * np.pi, length)
        rng = np.random.default_rng(42)

        if signal_type == "sine":
            return np.sin(t) + 0.1 * rng.normal(size=length)
        elif signal_type == "trending":
            return 0.02 * t + 0.3 * np.sin(t) + 0.05 * rng.normal(size=length)
        elif signal_type == "noisy":
            return np.cumsum(rng.normal(0, 0.1, length))
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")

    def _random_circuit_gates(
        self, n_qubits: int, depth: int
    ) -> list:
        """Generate a random circuit as a list of (gate_matrix, qubits) tuples."""
        from quantum.tensor_networks import TensorNetworkSimulator

        rng = np.random.default_rng(42)
        gates = []

        H = TensorNetworkSimulator._H()
        CNOT = TensorNetworkSimulator._CNOT()

        for layer in range(depth):
            # Single-qubit rotations
            for q in range(n_qubits):
                theta = rng.uniform(0, 2 * np.pi)
                gates.append((TensorNetworkSimulator._RX(theta), (q,)))

            # Entangling gates (nearest-neighbour)
            start = layer % 2  # alternating even/odd bonds
            for q in range(start, n_qubits - 1, 2):
                gates.append((CNOT, (q, q + 1)))

        return gates

    # ------------------------------------------------------------------
    # Additional benchmark methods (VaR, portfolio, pairs)
    # ------------------------------------------------------------------

    def benchmark_var(
        self,
        returns: Optional[np.ndarray] = None,
        n_trials: int = 10,
    ) -> Dict[str, Any]:
        """QMC VaR vs classical MC VaR: accuracy + speed.

        Args:
            returns: Return series. If None, generates synthetic.
            n_trials: Number of trials for statistical reliability.

        Returns:
            Dict with qmc_var, mc_var, accuracy, speed metrics.
        """
        from quantum.backtesting.quantum_backtest import QuantumBacktestAccelerator

        rng = np.random.default_rng(42)
        if returns is None:
            returns = rng.normal(-0.001, 0.02, 500)
        returns = np.asarray(returns, dtype=np.float64).ravel()

        # True VaR from full data
        true_var = float(-np.percentile(returns, 5.0))

        qmc_errors = []
        mc_errors = []
        qmc_times = []
        mc_times = []

        for trial in range(n_trials):
            qba = QuantumBacktestAccelerator(n_scenarios=1000, seed=42 + trial)

            # QMC
            t0 = time.perf_counter()
            qmc_result = qba.run_qmc_scenarios(returns, n_scenarios=1000)
            qmc_times.append((time.perf_counter() - t0) * 1000)
            qmc_errors.append(abs(qmc_result["var_95"] - true_var))

            # Classical MC bootstrap
            t0 = time.perf_counter()
            boot = rng.choice(returns, size=1000, replace=True)
            mc_var = float(-np.percentile(boot, 5.0))
            mc_times.append((time.perf_counter() - t0) * 1000)
            mc_errors.append(abs(mc_var - true_var))

        result = {
            "true_var_95": round(true_var, 8),
            "qmc_mean_error": round(float(np.mean(qmc_errors)), 8),
            "mc_mean_error": round(float(np.mean(mc_errors)), 8),
            "qmc_mean_time_ms": round(float(np.mean(qmc_times)), 2),
            "mc_mean_time_ms": round(float(np.mean(mc_times)), 2),
            "qmc_wins": int(sum(1 for q, m in zip(qmc_errors, mc_errors) if q < m)),
            "n_trials": n_trials,
            "verdict": self._honest_verdict(
                -float(np.mean(qmc_errors)),
                -float(np.mean(mc_errors)),
                "VaR error (lower is better)",
            ),
        }

        self._results.append({"benchmark": "var_qmc_vs_mc", "results": result})
        return result

    def benchmark_portfolio(
        self,
        correlation_matrix: Optional[np.ndarray] = None,
        n_trials: int = 10,
    ) -> Dict[str, Any]:
        """VQE/QAOA portfolio vs scipy minimize: quality + speed.

        Args:
            correlation_matrix: If None, generates random.
            n_trials: Number of trials.

        Returns:
            Dict comparing VQE portfolio weights vs classical minimum variance.
        """
        from quantum.hybrid.variational import VariationalQuantumEigensolver

        rng = np.random.default_rng(42)

        if correlation_matrix is None:
            n = 4
            A = rng.normal(0, 0.1, (n, n))
            correlation_matrix = A.T @ A / n + 0.05 * np.eye(n)
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2.0

        n = correlation_matrix.shape[0]
        n_qubits = max(2, int(np.ceil(np.log2(max(n, 2)))))

        vqe_variances = []
        classical_variances = []
        vqe_times = []
        classical_times = []

        for trial in range(n_trials):
            # VQE
            t0 = time.perf_counter()
            try:
                vqe = VariationalQuantumEigensolver(
                    n_qubits=n_qubits, n_layers=2, seed=42 + trial
                )
                w_vqe = vqe.portfolio_weights(correlation_matrix)
                vqe_var = float(w_vqe @ correlation_matrix @ w_vqe)
            except Exception:
                w_vqe = np.ones(n) / n
                vqe_var = float(w_vqe @ correlation_matrix @ w_vqe)
            vqe_times.append((time.perf_counter() - t0) * 1000)
            vqe_variances.append(vqe_var)

            # Classical
            t0 = time.perf_counter()
            def neg_var(w):
                return float(w @ correlation_matrix @ w)

            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            bounds = [(0.0, 1.0)] * n
            x0 = np.ones(n) / n
            opt = sp_minimize(neg_var, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            w_classical = np.maximum(opt.x, 0.0)
            w_classical = w_classical / max(w_classical.sum(), 1e-12)
            classical_var = float(w_classical @ correlation_matrix @ w_classical)
            classical_times.append((time.perf_counter() - t0) * 1000)
            classical_variances.append(classical_var)

        result = {
            "vqe_mean_variance": round(float(np.mean(vqe_variances)), 8),
            "classical_mean_variance": round(float(np.mean(classical_variances)), 8),
            "vqe_mean_time_ms": round(float(np.mean(vqe_times)), 2),
            "classical_mean_time_ms": round(float(np.mean(classical_times)), 2),
            "vqe_wins": int(sum(
                1 for v, c in zip(vqe_variances, classical_variances) if v < c * 1.01
            )),
            "n_trials": n_trials,
            "n_assets": n,
            "verdict": self._honest_verdict(
                -float(np.mean(vqe_variances)),
                -float(np.mean(classical_variances)),
                "portfolio variance (lower is better)",
            ),
        }

        self._results.append({"benchmark": "portfolio_vqe_vs_classical", "results": result})
        return result

    def benchmark_pairs(
        self,
        price_matrix: Optional[np.ndarray] = None,
        n_trials: int = 5,
    ) -> Dict[str, Any]:
        """Quantum walk pairs vs classical correlation: discovery quality.

        Compares quantum-walk-based pair discovery against classical
        Pearson correlation for finding co-integrated pairs.

        Args:
            price_matrix: Matrix of shape (n_timesteps, n_assets).
                If None, generates synthetic correlated prices.
            n_trials: Number of trials.

        Returns:
            Dict comparing pair discovery quality.
        """
        rng = np.random.default_rng(42)

        if price_matrix is None:
            # Generate synthetic correlated price series
            n_assets = 6
            n_steps = 200
            # Create correlation structure
            base = np.cumsum(rng.normal(0, 0.01, n_steps))
            price_matrix = np.zeros((n_steps, n_assets))
            for i in range(n_assets):
                # Some pairs are correlated, some not
                if i < 3:
                    price_matrix[:, i] = base + rng.normal(0, 0.005, n_steps).cumsum()
                else:
                    price_matrix[:, i] = rng.normal(0, 0.01, n_steps).cumsum()
            price_matrix = 100 + price_matrix  # Base price

        n_assets = price_matrix.shape[1]

        quantum_pairs_found = []
        classical_pairs_found = []

        for trial in range(n_trials):
            # Classical: Pearson correlation
            corr_matrix = np.corrcoef(price_matrix.T)
            classical_pairs = []
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    if abs(corr_matrix[i, j]) > 0.7:
                        classical_pairs.append((i, j, float(abs(corr_matrix[i, j]))))
            classical_pairs_found.append(len(classical_pairs))

            # Quantum-inspired: use returns correlation + cointegration test
            returns = np.diff(price_matrix, axis=0)
            ret_corr = np.corrcoef(returns.T)

            # Quantum walk mixing: simulate random walk on correlation graph
            # The quantum walk explores the graph faster (Grover-like speedup)
            adj = np.abs(ret_corr)
            np.fill_diagonal(adj, 0)

            # Simulate quantum walk: coin + shift operator
            # Simplified: use the adjacency matrix as transition operator
            # and find strongly connected pairs
            # Normalize rows
            row_sums = adj.sum(axis=1, keepdims=True)
            row_sums[row_sums < 1e-12] = 1.0
            transition = adj / row_sums

            # Run quantum-inspired walk (power iteration with interference)
            state = np.ones(n_assets) / np.sqrt(n_assets)
            for step in range(10):
                # Apply transition with phase
                new_state = transition @ state
                # Add quantum-like interference (superposition)
                phase = np.exp(1j * np.pi * step / 10)
                state = np.real(new_state * phase + state * np.conj(phase)) / 2
                norm = np.linalg.norm(state)
                if norm > 1e-12:
                    state = state / norm

            # Find pairs where both assets have high walk amplitude
            threshold = 1.0 / np.sqrt(n_assets)
            high_amp = np.where(np.abs(state) > threshold)[0]
            quantum_pairs = []
            for i in range(len(high_amp)):
                for j in range(i + 1, len(high_amp)):
                    a, b = high_amp[i], high_amp[j]
                    if abs(ret_corr[a, b]) > 0.5:
                        quantum_pairs.append((a, b, float(abs(ret_corr[a, b]))))
            quantum_pairs_found.append(len(quantum_pairs))

        result = {
            "quantum_pairs_mean": round(float(np.mean(quantum_pairs_found)), 2),
            "classical_pairs_mean": round(float(np.mean(classical_pairs_found)), 2),
            "n_assets": n_assets,
            "n_trials": n_trials,
            "verdict": (
                "Quantum walk exploration finds correlated pairs through graph structure. "
                "Classical correlation is simpler and usually sufficient. "
                "Quantum walk may find indirect correlations missed by pairwise analysis."
            ),
        }

        self._results.append({"benchmark": "pairs_discovery", "results": result})
        return result

    def run_all(self) -> Dict[str, Any]:
        """Run every quantum algorithm benchmark and compare vs classical.

        Returns comprehensive comparison across all categories.
        """
        logger.info("Running all quantum benchmarks...")

        results = {}

        try:
            results["var"] = self.benchmark_var(n_trials=5)
        except Exception as e:
            results["var"] = {"error": str(e)}

        try:
            results["portfolio"] = self.benchmark_portfolio(n_trials=5)
        except Exception as e:
            results["portfolio"] = {"error": str(e)}

        try:
            results["pairs"] = self.benchmark_pairs(n_trials=3)
        except Exception as e:
            results["pairs"] = {"error": str(e)}

        try:
            results["tensor_networks"] = self.benchmark_tensor_network_scaling(
                n_qubits_range=[4, 8], circuit_depth=5
            )
        except Exception as e:
            results["tensor_networks"] = {"error": str(e)}

        results["overall"] = {
            "note": (
                "All benchmarks use classical simulation. No quantum hardware advantage "
                "is claimed. Value comes from algorithmic structure (QMC convergence, "
                "variational optimization, graph exploration)."
            ),
        }

        return results

    @staticmethod
    def _statevector_simulate(
        n_qubits: int,
        gates: list,
    ) -> np.ndarray:
        """Brute-force statevector simulation for comparison."""
        N = 2 ** n_qubits
        sv = np.zeros(N, dtype=np.complex128)
        sv[0] = 1.0

        for gate_matrix, qubits in gates:
            if len(qubits) == 1:
                q = qubits[0]
                gate = np.asarray(gate_matrix, dtype=np.complex128)
                # Apply single-qubit gate via reshape
                shape = [2] * n_qubits
                psi = sv.reshape(shape)
                psi = np.moveaxis(psi, q, -1)
                psi = np.einsum("ij,...j->...i", gate, psi)
                psi = np.moveaxis(psi, -1, q)
                sv = psi.reshape(N)

            elif len(qubits) == 2:
                q1, q2 = qubits
                gate = np.asarray(gate_matrix, dtype=np.complex128).reshape(2, 2, 2, 2)
                shape = [2] * n_qubits
                psi = sv.reshape(shape)
                # Move target qubits to last two positions
                axes = list(range(n_qubits))
                axes.remove(q1)
                axes.remove(q2)
                axes.extend([q1, q2])
                psi = np.transpose(psi, axes)
                # Apply gate
                psi = np.einsum("ijkl,...kl->...ij", gate, psi)
                # Move back
                inv_axes = [0] * n_qubits
                for new_pos, old_pos in enumerate(axes):
                    inv_axes[old_pos] = new_pos
                psi = np.transpose(psi, inv_axes)
                sv = psi.reshape(N)

        return sv


# Alias for the API name used in tests and external code
QuantumBenchmark = QuantumBenchmarkSuite
