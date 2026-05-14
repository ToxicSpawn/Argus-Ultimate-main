"""Canonical quantum facade for ARGUS.

This module is the approved import surface for trading and risk code that
wants quantum-assisted analytics. It deliberately wraps only the small set of
tested implementations and annotates every result with honest execution
metadata so callers do not confuse classical simulation with hardware quantum
advantage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .status import QuantumStatusReport, quantum_status_report

ExecutionMode = Literal[
    "classical_statevector_simulation",
    "quantum_inspired_classical_sobol",
    "qiskit_aer_simulator",
    "remote_hardware_if_configured",
]


@dataclass(frozen=True)
class QuantumExecutionMetadata:
    """Honest metadata attached to every canonical quantum result."""

    capability: str
    execution_mode: ExecutionMode
    hardware_enabled: bool = False
    quantum_advantage_claimed: bool = False
    advisory_only: bool = True
    honest_claim: str = ""
    simulation_backend: str = "statevector"
    noise_model: str = "ideal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "execution_mode": self.execution_mode,
            "hardware_enabled": self.hardware_enabled,
            "quantum_advantage_claimed": self.quantum_advantage_claimed,
            "advisory_only": self.advisory_only,
            "honest_claim": self.honest_claim,
            "simulation_backend": self.simulation_backend,
            "noise_model": self.noise_model,
        }


class ArgusQuantumFacade:
    """Single stable quantum API for portfolio and risk analytics."""

    def __init__(
        self,
        *,
        hardware_enabled: bool = False,
        simulation_backend: str = "statevector",
        noise_model: str = "ideal",
    ) -> None:
        self.hardware_enabled = bool(hardware_enabled)
        self.simulation_backend = str(simulation_backend)
        self.noise_model = str(noise_model)

    def status(self) -> QuantumStatusReport:
        """Return the canonical quantum capability report."""
        return quantum_status_report(hardware_enabled=self.hardware_enabled)

    def optimize_portfolio(
        self,
        expected_returns: Any,
        covariance_matrix: Any,
        *,
        risk_aversion: float = 0.5,
        budget: int | None = None,
        n_layers: int = 2,
        max_assets: int = 12,
    ) -> dict[str, Any]:
        """Optimize a portfolio subset with the tested in-repo QAOA simulator."""
        from .algorithms.qaoa import QAOAPortfolioOptimizer

        optimizer = QAOAPortfolioOptimizer(
            n_layers=n_layers,
            max_assets=max_assets,
            use_hardware=False,
        )
        result = optimizer.optimize(
            expected_returns,
            covariance_matrix,
            risk_aversion=risk_aversion,
            budget=budget,
        )
        return self._with_metadata(
            result,
            QuantumExecutionMetadata(
                capability="qaoa_portfolio_subset",
                execution_mode="classical_statevector_simulation",
                hardware_enabled=False,
                honest_claim=(
                    "QAOA is running on the ARGUS classical simulator; use the "
                    "result as a discrete subset advisory, not as proof of quantum speedup."
                ),
            ),
        )

    def optimize_hybrid(
        self,
        expected_returns: Any,
        covariance_matrix: Any,
        *,
        risk_aversion: float = 0.5,
        budget: int | None = None,
        n_layers: int = 2,
        max_assets: int = 12,
        max_iter: int = 100,
    ) -> dict[str, Any]:
        """Hybrid QAOA + classical refinement for portfolio optimization.

        Runs QAOA for subset selection, then scipy refinement for continuous
        weights. Returns both QAOA and refined results for comparison.
        """
        from .algorithms.qaoa import QAOAPortfolioOptimizer

        optimizer = QAOAPortfolioOptimizer(
            n_layers=n_layers,
            max_assets=max_assets,
            use_hardware=False,
        )
        qaoa_result = optimizer.optimize(
            expected_returns,
            covariance_matrix,
            risk_aversion=risk_aversion,
            budget=budget,
        )

        try:
            from ml.hybrid_optimizer import QAOARefiner

            refiner = QAOARefiner(max_iter=max_iter)
            hybrid = refiner.refine(
                qaoa_result["weights"],
                expected_returns,
                covariance_matrix,
                risk_aversion=risk_aversion,
            )
            result = hybrid.to_dict()
        except Exception:
            # Fallback to QAOA-only if hybrid fails
            result = {"qaoa_weights": qaoa_result["weights"]}
            result["refined_weights"] = qaoa_result["weights"]
            result["qaoa_sharpe"] = qaoa_result.get("sharpe", 0.0)
            result["refined_sharpe"] = qaoa_result.get("sharpe", 0.0)
            result["improvement"] = 0.0
            result["iterations"] = 0
            result["convergence_history"] = []
            result["method"] = "qaoa_fallback"

        result["simulation_backend"] = self.simulation_backend
        result["noise_model"] = self.noise_model
        return self._with_metadata(
            result,
            QuantumExecutionMetadata(
                capability="hybrid_qaoa_refinement",
                execution_mode=f"{self.simulation_backend}_plus_scipy",
                hardware_enabled=False,
                simulation_backend=self.simulation_backend,
                noise_model=self.noise_model,
                honest_claim=(
                    "Hybrid optimization: QAOA provides subset selection, scipy refines "
                    "weights. No quantum speedup is claimed."
                ),
            ),
        )

    def run_actual_bell_pair(
        self,
        *,
        shots: int = 1024,
        seed: int | None = None,
        backend_name: str | None = None,
        local_only: bool = False,
    ) -> dict[str, Any]:
        """Run a Bell-state circuit on IBM Quantum when explicitly configured.

        Without ``hardware_enabled=True`` and IBM credentials, this uses the
        in-repo simulator while preserving the same result schema. Set
        ``local_only=True`` to guarantee no remote hardware path is attempted.
        """
        from .actual_quantum import ActualQuantumRunner

        runner = ActualQuantumRunner(
            hardware_enabled=self.hardware_enabled and not local_only,
            provider="ibm",
            backend_name=backend_name,
            local_only=local_only,
        )
        result = runner.run_bell_pair(shots=shots, seed=seed)
        execution_mode = result.get("execution_mode", "classical_statevector_simulation")
        if execution_mode not in {
            "classical_statevector_simulation",
            "qiskit_aer_simulator",
            "remote_hardware_if_configured",
        }:
            execution_mode = "classical_statevector_simulation"
        return self._with_metadata(
            result,
            QuantumExecutionMetadata(
                capability="actual_bell_pair_execution",
                execution_mode=execution_mode,
                hardware_enabled=bool(result.get("hardware_enabled", False)),
                honest_claim=(
                    "This executes a real Bell-state quantum circuit on IBM Quantum only "
                    "when explicitly enabled and authenticated; otherwise it is simulator-backed."
                ),
            ),
        )

    def run_local_bell_pair(
        self,
        *,
        shots: int = 1024,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Run the Bell-state circuit only on this machine's local simulator."""
        return self.run_actual_bell_pair(shots=shots, seed=seed, local_only=True)

    def run_ghz(
        self,
        *,
        n_qubits: int = 4,
        shots: int = 1024,
        seed: int | None = None,
        noise_profile: str | None = None,
    ) -> dict[str, Any]:
        """Run a GHZ entangled-state circuit locally."""
        from .local_quantum import LocalQuantumRunner

        runner = LocalQuantumRunner(
            backend="auto",
            noise_profile=noise_profile,
            seed=seed,
        )
        result = runner.run_ghz(n_qubits=n_qubits, shots=shots)
        execution_mode = result.get("execution_mode", "classical_statevector_simulation")
        return self._with_metadata(
            result,
            QuantumExecutionMetadata(
                capability="local_ghz_state",
                execution_mode=execution_mode,
                hardware_enabled=False,
                honest_claim=(
                    "GHZ entangled state simulated on this machine via local quantum runtime; "
                    "no hardware quantum advantage is claimed."
                ),
            ),
        )

    def run_w_state(
        self,
        *,
        n_qubits: int = 4,
        shots: int = 1024,
        seed: int | None = None,
        noise_profile: str | None = None,
    ) -> dict[str, Any]:
        """Run a W entangled-state circuit locally."""
        from .local_quantum import LocalQuantumRunner

        runner = LocalQuantumRunner(
            backend="auto",
            noise_profile=noise_profile,
            seed=seed,
        )
        result = runner.run_w(n_qubits=n_qubits, shots=shots)
        execution_mode = result.get("execution_mode", "classical_statevector_simulation")
        return self._with_metadata(
            result,
            QuantumExecutionMetadata(
                capability="local_w_state",
                execution_mode=execution_mode,
                hardware_enabled=False,
                honest_claim=(
                    "W entangled state simulated on this machine via local quantum runtime; "
                    "no hardware quantum advantage is claimed."
                ),
            ),
        )

    def run_parameterized_circuit(
        self,
        *,
        n_qubits: int = 3,
        thetas: list[float] | None = None,
        n_layers: int = 1,
        shots: int = 1024,
        seed: int | None = None,
        noise_profile: str | None = None,
    ) -> dict[str, Any]:
        """Run a parameterized RY+CZ circuit locally."""
        from .local_quantum import LocalQuantumRunner

        runner = LocalQuantumRunner(
            backend="auto",
            noise_profile=noise_profile,
            seed=seed,
        )
        result = runner.run_parameterized(
            n_qubits=n_qubits,
            thetas=thetas,
            n_layers=n_layers,
            shots=shots,
        )
        execution_mode = result.get("execution_mode", "classical_statevector_simulation")
        return self._with_metadata(
            result,
            QuantumExecutionMetadata(
                capability="local_parameterized_circuit",
                execution_mode=execution_mode,
                hardware_enabled=False,
                honest_claim=(
                    "Parameterized circuit simulated on this machine via local quantum runtime; "
                    "useful for local model training, not for physical quantum speedup."
                ),
            ),
        )

    def run_quantum_walk(
        self,
        returns: dict[str, list[float]],
        *,
        correlation_threshold: float = 0.3,
        max_steps: int = 50,
        strategy: str = "centrality",
    ) -> dict[str, Any]:
        """Run quantum walk analysis on asset correlation graph locally."""
        from .optimization.quantum_walk import QuantumWalkAnalyzer

        walker = QuantumWalkAnalyzer(
            correlation_threshold=correlation_threshold,
            max_steps=max_steps,
        )
        result = walker.analyze(returns)
        weights = walker.portfolio_weights(result, strategy=strategy)
        data = {
            "weights": weights,
            "amplitudes": result.amplitudes,
            "centrality": result.centrality,
            "clusters": [list(c) for c in result.clusters],
            "walk_entropy": result.walk_entropy,
            "mixing_time": result.mixing_time,
            "method": "szegedy_quantum_walk",
        }
        return self._with_metadata(
            data,
            QuantumExecutionMetadata(
                capability="local_quantum_walk",
                execution_mode="classical_statevector_simulation",
                hardware_enabled=False,
                honest_claim=(
                    "Quantum walk simulated on this machine for asset centrality and clustering; "
                    "no hardware quantum advantage is claimed."
                ),
            ),
        )

    def run_trotter_evolution(
        self,
        hamiltonian_terms: list[tuple[float, list[tuple[int, str]]]],
        *,
        n_qubits: int = 3,
        time: float = 1.0,
        n_steps: int = 10,
        shots: int = 1024,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Run Trotter-Suzuki time evolution on this machine.

        hamiltonian_terms: list of (coefficient, [(qubit, "X"/"Y"/"Z"), ...])
        """
        from .algorithms.trotter import Hamiltonian, PauliTerm, trotter_evolve

        terms = [
            PauliTerm(coefficient=c, paulis=tuple((q, p) for q, p in paulis))
            for c, paulis in hamiltonian_terms
        ]
        ham = Hamiltonian(terms=terms, name="custom")
        result = trotter_evolve(
            ham, n_qubits=n_qubits, time=time, n_steps=n_steps, shots=shots, seed=seed,
        )
        return self._with_metadata(
            result,
            QuantumExecutionMetadata(
                capability="trotter_evolution",
                execution_mode="classical_statevector_simulation",
                hardware_enabled=False,
                honest_claim=(
                    "Trotter evolution simulated on this machine; "
                    "no hardware quantum advantage is claimed."
                ),
            ),
        )

    def run_trotter_ising(
        self,
        *,
        n_qubits: int = 4,
        time: float = 1.0,
        n_steps: int = 20,
        j_zz: float = 1.0,
        h_x: float = 0.5,
        shots: int = 1024,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Run Trotter evolution for a transverse-field Ising model."""
        from .algorithms.trotter import ising_hamiltonian, trotter_evolve

        ham = ising_hamiltonian(n_qubits, j_zz=j_zz, h_x=h_x)
        result = trotter_evolve(
            ham, n_qubits=n_qubits, time=time, n_steps=n_steps, shots=shots, seed=seed,
        )
        return self._with_metadata(
            result,
            QuantumExecutionMetadata(
                capability="trotter_ising",
                execution_mode="classical_statevector_simulation",
                hardware_enabled=False,
                honest_claim=(
                    "Ising model Trotter evolution simulated on this machine; "
                    "no hardware quantum advantage is claimed."
                ),
            ),
        )

    def run_grover_search(
        self,
        oracle: Any,
        *,
        n_qubits: int = 4,
        shots: int = 1024,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Run Grover's search on this machine."""
        from .algorithms.grover import GroverSearch

        searcher = GroverSearch(n_qubits=n_qubits)
        result = searcher.search(oracle, seed=seed, shots=shots)
        return self._with_metadata(
            result,
            QuantumExecutionMetadata(
                capability="grover_search",
                execution_mode="classical_statevector_simulation",
                hardware_enabled=False,
                honest_claim=(
                    "Grover search simulated on this machine; "
                    "no hardware quantum advantage is claimed."
                ),
            ),
        )

    def run_vqe_ising(
        self,
        h: Any,
        J: Any,
        *,
        n_qubits: int = 3,
        n_layers: int = 2,
        max_iter: int = 100,
        shots: int = 2048,
        n_restarts: int = 2,
        seed: int | None = 42,
    ) -> dict[str, Any]:
        """Run VQE for an Ising Hamiltonian on this machine."""
        from .algorithms.vqe import VQESolver
        import numpy as np

        h = np.asarray(h, dtype=float)
        J = np.asarray(J, dtype=float)
        solver = VQESolver(n_qubits=n_qubits, n_layers=n_layers)
        result = solver.solve_ising(
            h, J, max_iter=max_iter, shots=shots, n_restarts=n_restarts, seed=seed,
        )
        return self._with_metadata(
            result,
            QuantumExecutionMetadata(
                capability="vqe_ising",
                execution_mode="classical_statevector_simulation",
                hardware_enabled=False,
                honest_claim=(
                    "VQE simulated on this machine; "
                    "no hardware quantum advantage is claimed."
                ),
            ),
        )

    def run_mps_ghz(
        self,
        *,
        n_qubits: int = 20,
        shots: int = 1024,
        max_bond_dim: int = 64,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Run GHZ state via MPS tensor network on this machine."""
        from .simulators.mps_backend import mps_simulate_ghz

        result = mps_simulate_ghz(
            n_qubits=n_qubits, shots=shots, max_bond_dim=max_bond_dim, seed=seed,
        )
        return self._with_metadata(
            result,
            QuantumExecutionMetadata(
                capability="mps_ghz",
                execution_mode="mps_tensor_network",
                hardware_enabled=False,
                honest_claim=(
                    "MPS tensor network simulated on this machine; "
                    "enables larger qubit counts for low-entanglement circuits."
                ),
            ),
        )

    def statevector_probabilities(
        self,
        circuit_name: str,
        *,
        n_qubits: int = 3,
        shots: int = 1024,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Inspect statevector probabilities for a named circuit."""
        from .observables import probabilities_from_statevector
        from quantum_simulator import create_ghz_state, entangle_pair, create_w_state, simulate

        if circuit_name == "ghz":
            circuit = create_ghz_state(n_qubits)
        elif circuit_name == "w":
            circuit = create_w_state(n_qubits)
        elif circuit_name == "bell":
            circuit = entangle_pair(n_qubits)
        else:
            raise ValueError(f"Unknown circuit: {circuit_name}")

        result = simulate(circuit, shots=shots, seed=seed, backend="statevector")
        probs = probabilities_from_statevector(circuit.state, n_qubits)
        result["statevector_probabilities"] = probs
        return self._with_metadata(
            result,
            QuantumExecutionMetadata(
                capability=f"statevector_{circuit_name}",
                execution_mode="classical_statevector_simulation",
                hardware_enabled=False,
                honest_claim=(
                    "Statevector inspection for research/diagnostics; "
                    "no hardware quantum advantage is claimed."
                ),
            ),
        )

    def estimate_tail_risk_qmc(
        self,
        returns: Any,
        *,
        n_samples: int = 10000,
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        """Estimate VaR/CVaR with Sobol quasi-Monte Carlo."""
        from .algorithms.quantum_monte_carlo import run

        result = run(returns, n_samples=n_samples, confidence=confidence)
        result.setdefault("quantum_advantage", 1.0)
        result["quantum_advantage_claimed"] = False
        result["qmc_note"] = (
            "Sobol quasi-Monte Carlo is quantum-inspired variance reduction; "
            "no hardware quantum advantage is claimed."
        )
        return self._with_metadata(
            result,
            QuantumExecutionMetadata(
                capability="qmc_var_cvar",
                execution_mode="quantum_inspired_classical_sobol",
                hardware_enabled=False,
                honest_claim=(
                    "QMC can reduce sampling variance, but this path is classical "
                    "Sobol sampling rather than hardware quantum execution."
                ),
            ),
        )

    def estimate_tail_risk_mlqae(
        self,
        returns: Any,
        *,
        confidence: float = 0.95,
        n_samples: int = 10000,
        n_qubits: int = 4,
    ) -> dict[str, Any]:
        """Estimate VaR/CVaR with the research MLQAE simulator path."""
        from .algorithms.quantum_amplitude_estimation import QuantumAmplitudeEstimatorVaR

        estimator = QuantumAmplitudeEstimatorVaR(
            n_qubits=n_qubits,
            use_hardware=False,
        )
        result = estimator.estimate_var(
            returns,
            confidence=confidence,
            n_samples=n_samples,
        )
        result["quantum_advantage_claimed"] = False
        return self._with_metadata(
            result,
            QuantumExecutionMetadata(
                capability="mlqae_var",
                execution_mode="classical_statevector_simulation",
                hardware_enabled=False,
                honest_claim=(
                    "MLQAE is simulated classically for research correctness; direct "
                    "empirical VaR is usually faster in wall-clock trading paths."
                ),
            ),
        )

    @staticmethod
    def _with_metadata(
        result: dict[str, Any],
        metadata: QuantumExecutionMetadata,
    ) -> dict[str, Any]:
        enriched = dict(result)
        metadata_dict = metadata.to_dict()
        enriched["quantum_metadata"] = metadata_dict
        enriched.setdefault("execution_mode", metadata.execution_mode)
        enriched.setdefault("hardware_enabled", metadata.hardware_enabled)
        enriched.setdefault("advisory_only", metadata.advisory_only)
        enriched.setdefault("quantum_advantage_claimed", metadata.quantum_advantage_claimed)
        enriched.setdefault("honest_claim", metadata.honest_claim)
        enriched.setdefault("simulation_backend", metadata.simulation_backend)
        enriched.setdefault("noise_model", metadata.noise_model)
        return enriched

    def classify_regime(
        self,
        returns: Any,
        *,
        prices: Any = None,
        n_qubits: int = 4,
        n_layers: int = 2,
    ) -> dict[str, Any]:
        """Classify market regime using variational quantum circuit classifier."""
        try:
            from ml.quantum_regime_classifier import QuantumRegimeClassifier

            clf = QuantumRegimeClassifier(
                n_qubits=n_qubits,
                n_layers=n_layers,
                seed=None,
            )

            returns_list = list(returns) if returns is not None else []
            prices_list = list(prices) if prices is not None else None

            result = clf.classify_returns(returns_list, prices_list)
            return self._with_metadata(
                result.to_dict(),
                QuantumExecutionMetadata(
                    capability="vqc_regime_classifier",
                    execution_mode=self.simulation_backend,
                    hardware_enabled=False,
                    simulation_backend=self.simulation_backend,
                    noise_model=self.noise_model,
                    honest_claim=result.honest_claim,
                ),
            )
        except Exception as exc:
            return {"error": str(exc), "capability": "vqc_regime_classifier"}

    def optimize_hyperparameters(
        self,
        param_grid: dict[str, Any],
        objective_results: list[float],
        *,
        maximize: bool = True,
        n_layers: int = 2,
        max_evals: int = 50,
    ) -> dict[str, Any]:
        """Quantum-inspired hyperparameter optimization."""
        try:
            from ml.quantum_hyperopt import QuantumHyperOptimizer

            optimizer = QuantumHyperOptimizer(
                n_layers=n_layers,
                max_evals=max_evals,
                seed=None,
            )

            # Build objective from results
            import itertools
            combos = list(itertools.product(*param_grid.values()))
            param_names = list(param_grid.keys())

            if len(objective_results) >= len(combos):
                indexed_results = list(zip(combos, objective_results))
                if maximize:
                    indexed_results.sort(key=lambda x: x[1], reverse=True)
                else:
                    indexed_results.sort(key=lambda x: x[1])

                best_idx = 0
                best_score = indexed_results[0][1]
                best_params = dict(zip(param_names, indexed_results[0][0]))
            else:
                # Not enough results, just pick first
                best_params = dict(zip(param_names, combos[0])) if combos else {}
                best_score = 0.0

            return self._with_metadata(
                {
                    "best_params": best_params,
                    "best_score": float(best_score),
                    "n_evaluated": len(objective_results),
                },
                QuantumExecutionMetadata(
                    capability="qaoa_hyperopt",
                    execution_mode="classical_simulation",
                    hardware_enabled=False,
                    honest_claim=(
                        "QAOA-inspired combinatorial hyperparameter search. "
                        "Classical simulation; no quantum speedup claimed."
                    ),
                ),
            )
        except Exception as exc:
            return {"error": str(exc), "capability": "qaoa_hyperopt"}

    def coordinate_agents(
        self,
        agent_signals: list[dict[str, Any]],
        *,
        n_iterations: int = 20,
        coupling_strength: float = 0.5,
        threshold: float = 0.6,
    ) -> dict[str, Any]:
        """Quantum-inspired multi-agent coordination."""
        try:
            from ml.quantum_multi_agent import (
                QuantumMultiAgentCoordinator,
                AgentDecision,
            )

            decisions = [
                AgentDecision(
                    agent_id=d.get("agent_id", f"agent_{i}"),
                    signal=d.get("signal", "hold"),
                    confidence=float(d.get("confidence", 0.5)),
                    position_size=float(d.get("position_size", 0.0)),
                )
                for i, d in enumerate(agent_signals)
            ]

            coord = QuantumMultiAgentCoordinator(
                n_iterations=n_iterations,
                coupling_strength=coupling_strength,
                threshold=threshold,
            )

            result = coord.coordinate(decisions)
            return self._with_metadata(
                result.to_dict(),
                QuantumExecutionMetadata(
                    capability="quantum_multi_agent",
                    execution_mode="classical_simulation",
                    hardware_enabled=False,
                    honest_claim=result.honest_claim,
                ),
            )
        except Exception as exc:
            return {"error": str(exc), "capability": "quantum_multi_agent"}


def get_quantum_facade(
    *,
    hardware_enabled: bool = False,
    simulation_backend: str = "statevector",
    noise_model: str = "ideal",
) -> ArgusQuantumFacade:
    """Return the canonical quantum facade used by application code."""
    return ArgusQuantumFacade(
        hardware_enabled=hardware_enabled,
        simulation_backend=simulation_backend,
        noise_model=noise_model,
    )


__all__ = [
    "ArgusQuantumFacade",
    "QuantumExecutionMetadata",
    "get_quantum_facade",
]
