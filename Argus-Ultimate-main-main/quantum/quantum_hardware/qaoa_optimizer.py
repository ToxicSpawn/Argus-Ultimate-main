from __future__ import annotations

import importlib
import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

np = importlib.import_module("numpy")

from .classical_fallback import ClassicalFallbackOptimizer, ClassicalOptimizationResult
from .ibm_quantum import IBMQuantumClient, IBMQuantumJobRequest
from .qubo_builder import PortfolioQUBOBuilder, PortfolioQUBOConfig, PortfolioQUBOModel, PortfolioQUBOProblem

logger = logging.getLogger(__name__)

try:
    QuantumCircuit = getattr(importlib.import_module("qiskit"), "QuantumCircuit")
    _has_qiskit = True
except Exception:
    QuantumCircuit = None
    _has_qiskit = False


@dataclass(slots=True)
class QAOAConfig:
    depth: int = 1
    shots: int = 2048
    gamma_grid_size: int = 7
    beta_grid_size: int = 7
    use_hardware: bool = False
    backend_name: str | None = None
    qubo_config: PortfolioQUBOConfig = field(default_factory=PortfolioQUBOConfig)

    def __post_init__(self) -> None:
        self.depth = max(1, int(self.depth))
        self.shots = max(128, int(self.shots))
        self.gamma_grid_size = max(3, int(self.gamma_grid_size))
        self.beta_grid_size = max(3, int(self.beta_grid_size))


@dataclass(slots=True)
class QAOAOptimizationResult:
    weights: dict[str, float]
    best_bitstring: str
    objective_value: float
    expected_return: float
    portfolio_risk: float
    execution_mode: str
    backend_name: str
    shots: int
    solve_time_ms: float
    gamma: float
    beta: float
    quantum_job_id: str | None = None
    counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, float] = field(default_factory=dict)


class QAOAPortfolioOptimizer:
    """Portfolio QAOA wrapper that can target IBM runtime or simulator-style sampling."""

    def __init__(
        self,
        client: IBMQuantumClient | None = None,
        fallback_optimizer: ClassicalFallbackOptimizer | None = None,
        builder: PortfolioQUBOBuilder | None = None,
        config: QAOAConfig | None = None,
    ) -> None:
        self.client = client or IBMQuantumClient()
        self.fallback_optimizer = fallback_optimizer or ClassicalFallbackOptimizer()
        self.builder = builder or PortfolioQUBOBuilder()
        self.config = config or QAOAConfig()

    def optimize(self, symbols: list[str], expected_returns: Any, covariance_matrix: Any) -> QAOAOptimizationResult:
        started_at = time.perf_counter()
        problem = PortfolioQUBOProblem(
            symbols=symbols,
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            config=self.config.qubo_config,
        )
        qubo_model = self.builder.build(problem)

        if len(symbols) > 16:
            logger.info("Problem size %d is too large for exhaustive QAOA simulation, using classical fallback", len(symbols))
            return self._optimize_classically(problem, started_at)

        gamma, beta, probability_map = self._grid_search_probabilities(qubo_model)
        circuit = self._build_qaoa_circuit(qubo_model, gamma, beta)
        request = IBMQuantumJobRequest(
            circuit=circuit,
            shots=self.config.shots,
            backend_name=self.config.backend_name,
            use_hardware=self.config.use_hardware and circuit is not None,
            num_qubits=len(symbols),
            probability_map=None if circuit is not None else probability_map,
            metadata={"gamma": gamma, "beta": beta, "depth": self.config.depth},
        )
        job = self.client.submit_job(request)
        result = self.client.get_result(job.job_id)
        best_bitstring = self._pick_best_bitstring(qubo_model, result.counts)
        return self._build_result(problem, qubo_model, best_bitstring, result.counts, started_at, gamma, beta, result.backend_name, result.mode, job.job_id)

    def _optimize_classically(self, problem: PortfolioQUBOProblem, started_at: float) -> QAOAOptimizationResult:
        fallback = self.fallback_optimizer.optimize(problem)
        best_bitstring = self._weights_to_bitstring(problem.symbols, fallback)
        solve_time_ms = (time.perf_counter() - started_at) * 1000.0
        return QAOAOptimizationResult(
            weights=fallback.weights,
            best_bitstring=best_bitstring,
            objective_value=fallback.objective_value,
            expected_return=fallback.expected_return,
            portfolio_risk=fallback.portfolio_risk,
            execution_mode="classical_fallback",
            backend_name=fallback.solver_used,
            shots=0,
            solve_time_ms=solve_time_ms,
            gamma=0.0,
            beta=0.0,
            metadata={"fallback": 1.0},
        )

    def _grid_search_probabilities(self, qubo_model: PortfolioQUBOModel) -> tuple[float, float, dict[str, float]]:
        n_qubits = len(qubo_model.symbols)
        states = [format(index, f"0{n_qubits}b") for index in range(2 ** n_qubits)]
        state_vectors = np.asarray([[int(bit) for bit in state] for state in states], dtype=float)
        energies = np.asarray([qubo_model.energy(vector) for vector in state_vectors], dtype=float)
        energies = energies - np.min(energies)

        best_gamma = 0.0
        best_beta = 0.0
        best_expectation = float("inf")
        best_distribution: dict[str, float] = {}

        gamma_values = np.linspace(0.1, np.pi, self.config.gamma_grid_size)
        beta_values = np.linspace(0.1, np.pi / 2.0, self.config.beta_grid_size)

        for gamma, beta in itertools.product(gamma_values, beta_values):
            inverse_temperature = max(0.05, np.sin(gamma) ** 2 + np.cos(beta) ** 2)
            logits = -energies / inverse_temperature
            logits = logits - np.max(logits)
            probabilities = np.exp(logits)
            probabilities = probabilities / np.sum(probabilities)
            expectation = float(np.sum(probabilities * energies))
            if expectation < best_expectation:
                best_expectation = expectation
                best_gamma = float(gamma)
                best_beta = float(beta)
                best_distribution = {state: float(prob) for state, prob in zip(states, probabilities)}

        return best_gamma, best_beta, best_distribution

    def _build_qaoa_circuit(self, qubo_model: PortfolioQUBOModel, gamma: float, beta: float):
        if not _has_qiskit or QuantumCircuit is None:
            return None

        n_qubits = len(qubo_model.symbols)
        circuit = QuantumCircuit(n_qubits, n_qubits)
        circuit.h(range(n_qubits))

        for _ in range(self.config.depth):
            for index in range(n_qubits):
                diagonal = float(qubo_model.qubo_matrix[index, index] + qubo_model.linear_terms[index])
                circuit.rz(2.0 * gamma * diagonal, index)

            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    coupling = float(qubo_model.qubo_matrix[i, j] + qubo_model.qubo_matrix[j, i])
                    if abs(coupling) <= 1e-10:
                        continue
                    circuit.cx(i, j)
                    circuit.rz(2.0 * gamma * coupling, j)
                    circuit.cx(i, j)

            for index in range(n_qubits):
                circuit.rx(2.0 * beta, index)

        circuit.measure(range(n_qubits), range(n_qubits))
        return circuit

    def _pick_best_bitstring(self, qubo_model: PortfolioQUBOModel, counts: dict[str, int]) -> str:
        if not counts:
            return "0" * len(qubo_model.symbols)
        return min(counts.keys(), key=lambda state: (qubo_model.energy([int(bit) for bit in state]), -counts[state]))

    def _build_result(
        self,
        problem: PortfolioQUBOProblem,
        qubo_model: PortfolioQUBOModel,
        best_bitstring: str,
        counts: dict[str, int],
        started_at: float,
        gamma: float,
        beta: float,
        backend_name: str,
        execution_mode: str,
        job_id: str | None,
    ) -> QAOAOptimizationResult:
        vector = np.asarray([int(bit) for bit in best_bitstring], dtype=float)
        weights = qubo_model.bitstring_to_weights(vector)
        weight_vector = np.asarray([weights[symbol] for symbol in problem.symbols], dtype=float)
        covariance = np.asarray(problem.covariance_matrix, dtype=float)
        expected_returns = np.asarray(problem.expected_returns, dtype=float)
        expected_return = float(expected_returns @ weight_vector)
        portfolio_risk = float(np.sqrt(max(weight_vector @ covariance @ weight_vector, 0.0)))
        objective_value = float(problem.config.risk_aversion * weight_vector @ covariance @ weight_vector - expected_return)
        solve_time_ms = (time.perf_counter() - started_at) * 1000.0
        return QAOAOptimizationResult(
            weights=weights,
            best_bitstring=best_bitstring,
            objective_value=objective_value,
            expected_return=expected_return,
            portfolio_risk=portfolio_risk,
            execution_mode=execution_mode,
            backend_name=backend_name,
            shots=self.config.shots,
            solve_time_ms=solve_time_ms,
            gamma=gamma,
            beta=beta,
            quantum_job_id=job_id,
            counts=counts,
            metadata=dict(qubo_model.metadata),
        )

    @staticmethod
    def _weights_to_bitstring(symbols: Sequence[str], fallback: ClassicalOptimizationResult) -> str:
        if not symbols:
            return ""
        mean_weight = 1.0 / len(symbols)
        return "".join("1" if fallback.weights.get(symbol, 0.0) >= mean_weight else "0" for symbol in symbols)
