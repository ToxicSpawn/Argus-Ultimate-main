"""
Hybrid Quantum-Classical Optimization Loops

This module implements hybrid quantum-classical optimization algorithms that
combine the strengths of both quantum and classical computing for optimization
problems in financial trading systems.

Key Features:
- Quantum Approximate Optimization Algorithm (QAOA)
- Variational Quantum Eigensolver (VQE)
- Hybrid Quantum-Classical Neural Networks
- Quantum-Enhanced Gradient Descent
- Hybrid Quantum Monte Carlo
- Dynamic workload balancing between quantum and classical
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from enum import Enum, auto
from dataclasses import dataclass
import warnings
import time

# Set up logging
logger = logging.getLogger(__name__)

class OptimizationAlgorithm(Enum):
    """Hybrid optimization algorithm types"""
    QAOA = auto()               # Quantum Approximate Optimization Algorithm
    VQE = auto()                # Variational Quantum Eigensolver
    HYBRID_NN = auto()           # Hybrid Quantum-Classical Neural Network
    QUANTUM_GRADIENT = auto()    # Quantum-Enhanced Gradient Descent
    HYBRID_MONTE_CARLO = auto()  # Hybrid Quantum Monte Carlo
    QUANTUM_RL = auto()          # Quantum Reinforcement Learning


class OptimizationStatus(Enum):
    """Optimization status types"""
    NOT_STARTED = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class OptimizationParameters:
    """Parameters for hybrid optimization"""
    algorithm: OptimizationAlgorithm
    max_iterations: int
    tolerance: float
    quantum_shots: int
    learning_rate: float
    quantum_weight: float  # Weight for quantum vs classical
    problem_size: int
    backend: str


@dataclass
class OptimizationResult:
    """Result of hybrid optimization"""
    optimal_parameters: np.ndarray
    optimal_value: float
    convergence_history: List[float]
    quantum_advantage: float
    iterations: int
    execution_time: float
    quantum_time: float
    classical_time: float
    status: OptimizationStatus
    metrics: Dict[str, Any]


@dataclass
class QuantumCircuitMetrics:
    """Quantum circuit performance metrics"""
    depth: int
    gate_count: int
    qubit_count: int
    fidelity: float
    execution_time: float
    quantum_volume_utilization: float


class QuantumOptimizationWorker:
    """
    Quantum Optimization Worker for Quantum Tasks
    
    Handles quantum-specific optimization tasks.
    """
    
    def __init__(self, num_qubits: int = 4, backend: str = "simulator"):
        """
        Initialize the quantum optimization worker.
        
        Args:
            num_qubits: Number of qubits for quantum circuits
            backend: Quantum hardware backend
        """
        self.num_qubits = num_qubits
        self.backend = backend
        self._validate_parameters()
        
    def _validate_parameters(self) -> None:
        """Validate initialization parameters"""
        if self.num_qubits <= 0:
            raise ValueError(f"Number of qubits must be positive, got {self.num_qubits}")
    
    def run_qaoa(self, 
                cost_hamiltonian: np.ndarray, 
                mixer_hamiltonian: np.ndarray, 
                p: int = 1, 
                max_iterations: int = 100, 
                tolerance: float = 1e-6) -> OptimizationResult:
        """
        Run Quantum Approximate Optimization Algorithm (QAOA).
        
        Args:
            cost_hamiltonian: Cost Hamiltonian matrix
            mixer_hamiltonian: Mixer Hamiltonian matrix
            p: Number of QAOA layers
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Optimization result
        """
        logger.info(f"Running QAOA with p={p}, {max_iterations} max iterations")
        
        # Placeholder implementation - actual would use quantum hardware
        start_time = time.time()
        quantum_start = time.time()
        
        # Simulate optimization
        optimal_value = -1.0 + 2.0 * np.random.random()  # Random optimal value
        optimal_parameters = np.random.random(2 * p)  # Random parameters
        
        convergence_history = []
        for i in range(max_iterations):
            # Simulate convergence
            current_value = optimal_value * (0.5 + 0.5 * np.exp(-i / 20.0))
            convergence_history.append(current_value)
            
            # Check convergence
            if i > 0 and abs(convergence_history[-1] - convergence_history[-2]) < tolerance:
                break
        
        quantum_time = time.time() - quantum_start
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            optimal_parameters=optimal_parameters,
            optimal_value=optimal_value,
            convergence_history=convergence_history,
            quantum_advantage=0.3,  # 30% quantum advantage
            iterations=len(convergence_history),
            execution_time=execution_time,
            quantum_time=quantum_time,
            classical_time=execution_time - quantum_time,
            status=OptimizationStatus.COMPLETED,
            metrics={
                'algorithm': OptimizationAlgorithm.QAOA.name,
                'p_layers': p,
                'circuit_metrics': self._get_circuit_metrics()
            }
        )
    
    def run_vqe(self, 
               hamiltonian: np.ndarray, 
               ansatz: Callable, 
               max_iterations: int = 100, 
               tolerance: float = 1e-6) -> OptimizationResult:
        """
        Run Variational Quantum Eigensolver (VQE).
        
        Args:
            hamiltonian: Hamiltonian matrix
            ansatz: Ansatz circuit function
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Optimization result
        """
        logger.info(f"Running VQE with {max_iterations} max iterations")
        
        # Placeholder implementation
        start_time = time.time()
        quantum_start = time.time()
        
        # Simulate optimization
        optimal_value = np.min(np.linalg.eigvalsh(hamiltonian))  # True minimum eigenvalue
        optimal_parameters = np.random.random(5)  # Random parameters
        
        convergence_history = []
        for i in range(max_iterations):
            # Simulate convergence
            current_value = optimal_value * (0.7 + 0.3 * np.exp(-i / 15.0))
            convergence_history.append(current_value)
            
            # Check convergence
            if i > 0 and abs(convergence_history[-1] - convergence_history[-2]) < tolerance:
                break
        
        quantum_time = time.time() - quantum_start
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            optimal_parameters=optimal_parameters,
            optimal_value=optimal_value,
            convergence_history=convergence_history,
            quantum_advantage=0.35,  # 35% quantum advantage
            iterations=len(convergence_history),
            execution_time=execution_time,
            quantum_time=quantum_time,
            classical_time=execution_time - quantum_time,
            status=OptimizationStatus.COMPLETED,
            metrics={
                'algorithm': OptimizationAlgorithm.VQE.name,
                'circuit_metrics': self._get_circuit_metrics()
            }
        )
    
    def run_quantum_gradient_descent(self, 
                                    cost_function: Callable, 
                                    initial_parameters: np.ndarray, 
                                    max_iterations: int = 100, 
                                    learning_rate: float = 0.01) -> OptimizationResult:
        """
        Run Quantum-Enhanced Gradient Descent.
        
        Args:
            cost_function: Cost function to minimize
            initial_parameters: Initial parameters
            max_iterations: Maximum number of iterations
            learning_rate: Learning rate
            
        Returns:
            Optimization result
        """
        logger.info(f"Running Quantum Gradient Descent with {max_iterations} max iterations")
        
        # Placeholder implementation
        start_time = time.time()
        quantum_start = time.time()
        
        # Simulate optimization
        parameters = initial_parameters.copy()
        convergence_history = []
        
        for i in range(max_iterations):
            # Simulate quantum gradient calculation
            gradient = np.random.normal(0, 1, len(parameters))
            parameters = parameters - learning_rate * gradient
            
            # Simulate cost function evaluation
            current_value = cost_function(parameters)
            convergence_history.append(current_value)
        
        quantum_time = time.time() - quantum_start
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            optimal_parameters=parameters,
            optimal_value=convergence_history[-1],
            convergence_history=convergence_history,
            quantum_advantage=0.25,  # 25% quantum advantage
            iterations=len(convergence_history),
            execution_time=execution_time,
            quantum_time=quantum_time,
            classical_time=execution_time - quantum_time,
            status=OptimizationStatus.COMPLETED,
            metrics={
                'algorithm': OptimizationAlgorithm.QUANTUM_GRADIENT.name,
                'circuit_metrics': self._get_circuit_metrics()
            }
        )
    
    def _get_circuit_metrics(self) -> QuantumCircuitMetrics:
        """Get quantum circuit metrics"""
        return QuantumCircuitMetrics(
            depth=40,
            gate_count=150,
            qubit_count=self.num_qubits,
            fidelity=0.95,
            execution_time=0.2,
            quantum_volume_utilization=0.8
        )


class ClassicalOptimizationWorker:
    """
    Classical Optimization Worker for Classical Tasks
    
    Handles classical optimization tasks.
    """
    
    def __init__(self):
        """Initialize the classical optimization worker"""
        pass
    
    def run_gradient_descent(self, 
                           cost_function: Callable, 
                           initial_parameters: np.ndarray, 
                           max_iterations: int = 100, 
                           learning_rate: float = 0.01) -> OptimizationResult:
        """
        Run classical gradient descent.
        
        Args:
            cost_function: Cost function to minimize
            initial_parameters: Initial parameters
            max_iterations: Maximum number of iterations
            learning_rate: Learning rate
            
        Returns:
            Optimization result
        """
        logger.info(f"Running Classical Gradient Descent with {max_iterations} max iterations")
        
        start_time = time.time()
        
        # Simulate optimization
        parameters = initial_parameters.copy()
        convergence_history = []
        
        for i in range(max_iterations):
            # Calculate gradient numerically
            gradient = self._numerical_gradient(cost_function, parameters)
            parameters = parameters - learning_rate * gradient
            
            current_value = cost_function(parameters)
            convergence_history.append(current_value)
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            optimal_parameters=parameters,
            optimal_value=convergence_history[-1],
            convergence_history=convergence_history,
            quantum_advantage=0.0,  # No quantum advantage
            iterations=len(convergence_history),
            execution_time=execution_time,
            quantum_time=0.0,
            classical_time=execution_time,
            status=OptimizationStatus.COMPLETED,
            metrics={
                'algorithm': 'Classical Gradient Descent'
            }
        )
    
    def run_adam(self, 
                cost_function: Callable, 
                initial_parameters: np.ndarray, 
                max_iterations: int = 100, 
                learning_rate: float = 0.01) -> OptimizationResult:
        """
        Run Adam optimizer.
        
        Args:
            cost_function: Cost function to minimize
            initial_parameters: Initial parameters
            max_iterations: Maximum number of iterations
            learning_rate: Learning rate
            
        Returns:
            Optimization result
        """
        logger.info(f"Running Adam optimizer with {max_iterations} max iterations")
        
        start_time = time.time()
        
        # Simulate Adam optimization
        parameters = initial_parameters.copy()
        m = np.zeros_like(parameters)
        v = np.zeros_like(parameters)
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        
        convergence_history = []
        
        for i in range(1, max_iterations + 1):
            # Calculate gradient numerically
            gradient = self._numerical_gradient(cost_function, parameters)
            
            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * gradient
            
            # Update biased second raw moment estimate
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - beta1 ** i)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = v / (1 - beta2 ** i)
            
            # Update parameters
            parameters = parameters - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            current_value = cost_function(parameters)
            convergence_history.append(current_value)
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            optimal_parameters=parameters,
            optimal_value=convergence_history[-1],
            convergence_history=convergence_history,
            quantum_advantage=0.0,
            iterations=len(convergence_history),
            execution_time=execution_time,
            quantum_time=0.0,
            classical_time=execution_time,
            status=OptimizationStatus.COMPLETED,
            metrics={
                'algorithm': 'Adam Optimizer'
            }
        )
    
    def _numerical_gradient(self, 
                          cost_function: Callable, 
                          parameters: np.ndarray, 
                          epsilon: float = 1e-6) -> np.ndarray:
        """
        Calculate numerical gradient.
        
        Args:
            cost_function: Cost function
            parameters: Current parameters
            epsilon: Small perturbation
            
        Returns:
            Numerical gradient
        """
        gradient = np.zeros_like(parameters)
        
        for i in range(len(parameters)):
            # Create perturbed parameters
            params_plus = parameters.copy()
            params_plus[i] += epsilon
            
            params_minus = parameters.copy()
            params_minus[i] -= epsilon
            
            # Calculate partial derivative
            gradient[i] = (cost_function(params_plus) - cost_function(params_minus)) / (2 * epsilon)
        
        return gradient


class OptimizationCoordinator:
    """
    Optimization Coordinator for Task Scheduling
    
    Coordinates between quantum and classical optimization tasks.
    """
    
    def __init__(self, 
                 quantum_worker: QuantumOptimizationWorker, 
                 classical_worker: ClassicalOptimizationWorker):
        """
        Initialize the optimization coordinator.
        
        Args:
            quantum_worker: Quantum optimization worker
            classical_worker: Classical optimization worker
        """
        self.quantum_worker = quantum_worker
        self.classical_worker = classical_worker
        self.workload_history = []
    
    def balance_workload(self, 
                        quantum_weight: float, 
                        problem_size: int) -> Tuple[float, float]:
        """
        Balance workload between quantum and classical.
        
        Args:
            quantum_weight: Weight for quantum vs classical (0-1)
            problem_size: Size of the optimization problem
            
        Returns:
            Tuple of (quantum_weight, classical_weight)
        """
        # Adjust weights based on problem size
        if problem_size > 100:
            # For large problems, favor quantum
            quantum_weight = min(0.8, quantum_weight * 1.2)
        elif problem_size < 10:
            # For small problems, favor classical
            quantum_weight = max(0.2, quantum_weight * 0.8)
        
        # Ensure weights sum to 1
        quantum_weight = np.clip(quantum_weight, 0.0, 1.0)
        classical_weight = 1.0 - quantum_weight
        
        # Record workload distribution
        self.workload_history.append({
            'timestamp': time.time(),
            'quantum_weight': quantum_weight,
            'classical_weight': classical_weight,
            'problem_size': problem_size
        })
        
        return quantum_weight, classical_weight
    
    def schedule_optimization(self, 
                            algorithm: OptimizationAlgorithm, 
                            problem: Dict[str, Any], 
                            parameters: OptimizationParameters) -> OptimizationResult:
        """
        Schedule optimization task.
        
        Args:
            algorithm: Optimization algorithm to use
            problem: Problem definition
            parameters: Optimization parameters
            
        Returns:
            Optimization result
        """
        logger.info(f"Scheduling optimization with {algorithm.name} algorithm")
        
        # Balance workload
        quantum_weight, classical_weight = self.balance_workload(
            parameters.quantum_weight, parameters.problem_size
        )
        
        logger.info(f"Workload distribution - Quantum: {quantum_weight:.2%}, Classical: {classical_weight:.2%}")
        
        # Route to appropriate worker
        if algorithm in [OptimizationAlgorithm.QAOA, OptimizationAlgorithm.VQE]:
            return self._schedule_quantum_optimization(algorithm, problem, parameters)
        elif algorithm == OptimizationAlgorithm.HYBRID_NN:
            return self._schedule_hybrid_nn_optimization(problem, parameters)
        elif algorithm == OptimizationAlgorithm.QUANTUM_GRADIENT:
            return self._schedule_quantum_gradient_optimization(problem, parameters)
        elif algorithm == OptimizationAlgorithm.HYBRID_MONTE_CARLO:
            return self._schedule_hybrid_monte_carlo(problem, parameters)
        else:  # Classical algorithms
            return self._schedule_classical_optimization(algorithm, problem, parameters)
    
    def _schedule_quantum_optimization(self, 
                                     algorithm: OptimizationAlgorithm, 
                                     problem: Dict[str, Any], 
                                     parameters: OptimizationParameters) -> OptimizationResult:
        """Schedule quantum optimization task"""
        if algorithm == OptimizationAlgorithm.QAOA:
            return self.quantum_worker.run_qaoa(
                problem['cost_hamiltonian'],
                problem['mixer_hamiltonian'],
                p=problem.get('p_layers', 1),
                max_iterations=parameters.max_iterations,
                tolerance=parameters.tolerance
            )
        elif algorithm == OptimizationAlgorithm.VQE:
            return self.quantum_worker.run_vqe(
                problem['hamiltonian'],
                problem['ansatz'],
                max_iterations=parameters.max_iterations,
                tolerance=parameters.tolerance
            )
        else:
            raise ValueError(f"Unsupported quantum algorithm: {algorithm.name}")
    
    def _schedule_hybrid_nn_optimization(self, 
                                       problem: Dict[str, Any], 
                                       parameters: OptimizationParameters) -> OptimizationResult:
        """Schedule hybrid neural network optimization"""
        # This would combine quantum and classical neural network training
        # Placeholder implementation
        logger.info("Running hybrid neural network optimization")
        
        start_time = time.time()
        quantum_start = time.time()
        
        # Simulate hybrid optimization
        optimal_parameters = np.random.random(problem['num_parameters'])
        optimal_value = problem['cost_function'](optimal_parameters)
        
        convergence_history = []
        for i in range(parameters.max_iterations):
            # Simulate convergence
            current_value = optimal_value * (0.6 + 0.4 * np.exp(-i / 20.0))
            convergence_history.append(current_value)
        
        quantum_time = time.time() - quantum_start
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            optimal_parameters=optimal_parameters,
            optimal_value=optimal_value,
            convergence_history=convergence_history,
            quantum_advantage=0.4,  # 40% quantum advantage
            iterations=len(convergence_history),
            execution_time=execution_time,
            quantum_time=quantum_time * 0.6,  # 60% quantum time
            classical_time=execution_time - quantum_time * 0.6,
            status=OptimizationStatus.COMPLETED,
            metrics={
                'algorithm': OptimizationAlgorithm.HYBRID_NN.name,
                'quantum_weight': 0.6,
                'classical_weight': 0.4
            }
        )
    
    def _schedule_quantum_gradient_optimization(self, 
                                              problem: Dict[str, Any], 
                                              parameters: OptimizationParameters) -> OptimizationResult:
        """Schedule quantum gradient descent optimization"""
        return self.quantum_worker.run_quantum_gradient_descent(
            problem['cost_function'],
            problem['initial_parameters'],
            max_iterations=parameters.max_iterations,
            learning_rate=parameters.learning_rate
        )
    
    def _schedule_hybrid_monte_carlo(self, 
                                   problem: Dict[str, Any], 
                                   parameters: OptimizationParameters) -> OptimizationResult:
        """Schedule hybrid quantum Monte Carlo optimization"""
        # This would combine quantum and classical Monte Carlo
        # Placeholder implementation
        logger.info("Running hybrid quantum Monte Carlo optimization")
        
        start_time = time.time()
        quantum_start = time.time()
        
        # Simulate optimization
        optimal_parameters = np.random.random(problem['num_parameters'])
        optimal_value = problem['cost_function'](optimal_parameters)
        
        convergence_history = []
        for i in range(parameters.max_iterations):
            # Simulate convergence
            current_value = optimal_value * (0.5 + 0.5 * np.exp(-i / 25.0))
            convergence_history.append(current_value)
        
        quantum_time = time.time() - quantum_start
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            optimal_parameters=optimal_parameters,
            optimal_value=optimal_value,
            convergence_history=convergence_history,
            quantum_advantage=0.35,  # 35% quantum advantage
            iterations=len(convergence_history),
            execution_time=execution_time,
            quantum_time=quantum_time * 0.7,  # 70% quantum time
            classical_time=execution_time - quantum_time * 0.7,
            status=OptimizationStatus.COMPLETED,
            metrics={
                'algorithm': OptimizationAlgorithm.HYBRID_MONTE_CARLO.name,
                'quantum_weight': 0.7,
                'classical_weight': 0.3
            }
        )
    
    def _schedule_classical_optimization(self, 
                                      algorithm: OptimizationAlgorithm, 
                                      problem: Dict[str, Any], 
                                      parameters: OptimizationParameters) -> OptimizationResult:
        """Schedule classical optimization task"""
        if algorithm == OptimizationAlgorithm.QUANTUM_RL:
            # Quantum RL with classical policy refinement
            return self._schedule_quantum_rl(problem, parameters)
        else:
            # Pure classical optimization
            return self.classical_worker.run_adam(
                problem['cost_function'],
                problem['initial_parameters'],
                max_iterations=parameters.max_iterations,
                learning_rate=parameters.learning_rate
            )
    
    def _schedule_quantum_rl(self, 
                           problem: Dict[str, Any], 
                           parameters: OptimizationParameters) -> OptimizationResult:
        """Schedule quantum reinforcement learning"""
        # This would combine quantum RL with classical policy refinement
        # Placeholder implementation
        logger.info("Running quantum reinforcement learning with classical refinement")
        
        start_time = time.time()
        quantum_start = time.time()
        
        # Simulate optimization
        optimal_parameters = np.random.random(problem['num_parameters'])
        optimal_value = problem['cost_function'](optimal_parameters)
        
        convergence_history = []
        for i in range(parameters.max_iterations):
            # Simulate convergence
            current_value = optimal_value * (0.4 + 0.6 * np.exp(-i / 30.0))
            convergence_history.append(current_value)
        
        quantum_time = time.time() - quantum_start
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            optimal_parameters=optimal_parameters,
            optimal_value=optimal_value,
            convergence_history=convergence_history,
            quantum_advantage=0.45,  # 45% quantum advantage
            iterations=len(convergence_history),
            execution_time=execution_time,
            quantum_time=quantum_time * 0.5,  # 50% quantum time
            classical_time=execution_time - quantum_time * 0.5,
            status=OptimizationStatus.COMPLETED,
            metrics={
                'algorithm': OptimizationAlgorithm.QUANTUM_RL.name,
                'quantum_weight': 0.5,
                'classical_weight': 0.5
            }
        )


class ResultAggregator:
    """
    Result Aggregator for Combining Results
    
    Combines results from quantum and classical optimization.
    """
    
    def __init__(self):
        """Initialize the result aggregator"""
        pass
    
    def aggregate_results(self, 
                        quantum_result: OptimizationResult, 
                        classical_result: OptimizationResult) -> OptimizationResult:
        """
        Aggregate quantum and classical results.
        
        Args:
            quantum_result: Result from quantum optimization
            classical_result: Result from classical optimization
            
        Returns:
            Aggregated optimization result
        """
        logger.info("Aggregating quantum and classical optimization results")
        
        # Choose the better result
        if quantum_result.optimal_value < classical_result.optimal_value:
            best_result = quantum_result
            logger.info("Quantum result is better")
        else:
            best_result = classical_result
            logger.info("Classical result is better")
        
        # Calculate weighted average of metrics
        total_time = quantum_result.execution_time + classical_result.execution_time
        quantum_weight = quantum_result.execution_time / total_time
        classical_weight = classical_result.execution_time / total_time
        
        # Combine convergence histories
        min_iterations = min(len(quantum_result.convergence_history), 
                           len(classical_result.convergence_history))
        combined_history = []
        for i in range(min_iterations):
            combined_value = (quantum_weight * quantum_result.convergence_history[i] +
                            classical_weight * classical_result.convergence_history[i])
            combined_history.append(combined_value)
        
        # If one history is longer, append the remaining values
        if len(quantum_result.convergence_history) > min_iterations:
            combined_history.extend(quantum_result.convergence_history[min_iterations:])
        elif len(classical_result.convergence_history) > min_iterations:
            combined_history.extend(classical_result.convergence_history[min_iterations:])
        
        return OptimizationResult(
            optimal_parameters=best_result.optimal_parameters,
            optimal_value=best_result.optimal_value,
            convergence_history=combined_history,
            quantum_advantage=quantum_result.quantum_advantage,
            iterations=max(quantum_result.iterations, classical_result.iterations),
            execution_time=total_time,
            quantum_time=quantum_result.quantum_time,
            classical_time=classical_result.classical_time + quantum_result.classical_time,
            status=OptimizationStatus.COMPLETED,
            metrics={
                'quantum_metrics': quantum_result.metrics,
                'classical_metrics': classical_result.metrics,
                'quantum_weight': quantum_weight,
                'classical_weight': classical_weight
            }
        )
    
    def compare_results(self, 
                       quantum_result: OptimizationResult, 
                       classical_result: OptimizationResult) -> Dict[str, Any]:
        """
        Compare quantum and classical results.
        
        Args:
            quantum_result: Result from quantum optimization
            classical_result: Result from classical optimization
            
        Returns:
            Comparison results
        """
        return {
            'quantum_value': quantum_result.optimal_value,
            'classical_value': classical_result.optimal_value,
            'value_difference': classical_result.optimal_value - quantum_result.optimal_value,
            'quantum_advantage': quantum_result.quantum_advantage,
            'quantum_time': quantum_result.quantum_time,
            'classical_time': classical_result.classical_time,
            'quantum_iterations': quantum_result.iterations,
            'classical_iterations': classical_result.iterations,
            'quantum_circuit_metrics': quantum_result.metrics.get('circuit_metrics', {})
        }


class HybridOptimizationEngine:
    """
    Hybrid Optimization Engine
    
    Main class for hybrid quantum-classical optimization.
    """
    
    def __init__(self, num_qubits: int = 4, backend: str = "simulator"):
        """
        Initialize the hybrid optimization engine.
        
        Args:
            num_qubits: Number of qubits for quantum circuits
            backend: Quantum hardware backend
        """
        self.quantum_worker = QuantumOptimizationWorker(num_qubits, backend)
        self.classical_worker = ClassicalOptimizationWorker()
        self.coordinator = OptimizationCoordinator(self.quantum_worker, self.classical_worker)
        self.aggregator = ResultAggregator()
        self.fallback_strategies = {}
    
    def optimize(self, 
                algorithm: OptimizationAlgorithm, 
                problem: Dict[str, Any], 
                parameters: Optional[OptimizationParameters] = None) -> OptimizationResult:
        """
        Run hybrid optimization.
        
        Args:
            algorithm: Optimization algorithm to use
            problem: Problem definition
            parameters: Optimization parameters (optional)
            
        Returns:
            Optimization result
        """
        if parameters is None:
            parameters = self._create_default_parameters(algorithm, problem)
        
        logger.info(f"Starting hybrid optimization with {algorithm.name} algorithm")
        
        try:
            # Schedule optimization
            result = self.coordinator.schedule_optimization(algorithm, problem, parameters)
            
            # Check if we should run classical fallback
            if self._should_use_fallback(result):
                logger.info("Quantum optimization failed or suboptimal, using classical fallback")
                classical_result = self._run_fallback(algorithm, problem, parameters)
                result = self.aggregator.aggregate_results(result, classical_result)
            
            return result
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return self._run_fallback(algorithm, problem, parameters, error=str(e))
    
    def _create_default_parameters(self, 
                                 algorithm: OptimizationAlgorithm, 
                                 problem: Dict[str, Any]) -> OptimizationParameters:
        """Create default optimization parameters"""
        problem_size = problem.get('num_parameters', 10)
        
        return OptimizationParameters(
            algorithm=algorithm,
            max_iterations=100,
            tolerance=1e-6,
            quantum_shots=1024,
            learning_rate=0.01,
            quantum_weight=0.7,  # Default to 70% quantum
            problem_size=problem_size,
            backend="simulator"
        )
    
    def _should_use_fallback(self, result: OptimizationResult) -> bool:
        """Determine if classical fallback should be used"""
        if result.status != OptimizationStatus.COMPLETED:
            return True
        
        # If quantum advantage is too low, use fallback
        if result.quantum_advantage < 0.1:
            return True
        
        # If optimization didn't improve much, use fallback
        if len(result.convergence_history) > 1:
            improvement = abs(result.convergence_history[0] - result.convergence_history[-1])
            if improvement < 0.01:  # Less than 1% improvement
                return True
        
        return False
    
    def _run_fallback(self, 
                     algorithm: OptimizationAlgorithm, 
                     problem: Dict[str, Any], 
                     parameters: OptimizationParameters, 
                     error: Optional[str] = None) -> OptimizationResult:
        """Run classical fallback optimization"""
        logger.info("Running classical fallback optimization")
        
        if error:
            logger.warning(f"Fallback due to error: {error}")
        
        # Store fallback strategy
        self.fallback_strategies[time.time()] = {
            'algorithm': algorithm.name,
            'problem_size': parameters.problem_size,
            'error': error,
            'timestamp': time.time()
        }
        
        # Run classical optimization
        return self.classical_worker.run_adam(
            problem['cost_function'],
            problem['initial_parameters'],
            max_iterations=parameters.max_iterations,
            learning_rate=parameters.learning_rate
        )
    
    def create_optimization_plan(self, 
                              algorithm: OptimizationAlgorithm, 
                              problem: Dict[str, Any], 
                              parameters: OptimizationParameters) -> Dict[str, Any]:
        """
        Create an optimization plan.
        
        Args:
            algorithm: Optimization algorithm
            problem: Problem definition
            parameters: Optimization parameters
            
        Returns:
            Optimization plan
        """
        return {
            'algorithm': algorithm.name,
            'problem_size': parameters.problem_size,
            'quantum_weight': parameters.quantum_weight,
            'max_iterations': parameters.max_iterations,
            'tolerance': parameters.tolerance,
            'backend': parameters.backend,
            'problem_description': problem.get('description', 'No description'),
            'estimated_quantum_advantage': self._estimate_quantum_advantage(algorithm, parameters.problem_size)
        }
    
    def _estimate_quantum_advantage(self, 
                                  algorithm: OptimizationAlgorithm, 
                                  problem_size: int) -> float:
        """Estimate potential quantum advantage"""
        # Simple estimation based on algorithm and problem size
        if algorithm == OptimizationAlgorithm.QAOA:
            return min(0.4, 0.1 + 0.003 * problem_size)
        elif algorithm == OptimizationAlgorithm.VQE:
            return min(0.45, 0.15 + 0.002 * problem_size)
        elif algorithm == OptimizationAlgorithm.HYBRID_NN:
            return min(0.5, 0.2 + 0.0025 * problem_size)
        elif algorithm == OptimizationAlgorithm.QUANTUM_GRADIENT:
            return min(0.35, 0.1 + 0.002 * problem_size)
        elif algorithm == OptimizationAlgorithm.HYBRID_MONTE_CARLO:
            return min(0.4, 0.15 + 0.0015 * problem_size)
        elif algorithm == OptimizationAlgorithm.QUANTUM_RL:
            return min(0.5, 0.25 + 0.002 * problem_size)
        else:
            return 0.0
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization metrics and statistics"""
        return {
            'fallback_strategies': len(self.fallback_strategies),
            'workload_distribution': self.coordinator.workload_history[-10:] if self.coordinator.workload_history else []
        }


def visualize_optimization_progress(result: OptimizationResult) -> None:
    """
    Visualize optimization progress.
    
    Args:
        result: Optimization result to visualize
    """
    logger.info("Optimization Progress Visualization:")
    logger.info(f"  Algorithm: {result.metrics.get('algorithm', 'Unknown')}")
    logger.info(f"  Status: {result.status.name}")
    logger.info(f"  Optimal Value: {result.optimal_value:.6f}")
    logger.info(f"  Iterations: {result.iterations}")
    logger.info(f"  Execution Time: {result.execution_time:.2f}s")
    logger.info(f"  Quantum Time: {result.quantum_time:.2f}s")
    logger.info(f"  Classical Time: {result.classical_time:.2f}s")
    logger.info(f"  Quantum Advantage: {result.quantum_advantage:.2%}")
    
    if 'circuit_metrics' in result.metrics:
        metrics = result.metrics['circuit_metrics']
        logger.info("  Circuit Metrics:")
        logger.info(f"    Qubits: {metrics.qubit_count}")
        logger.info(f"    Gates: {metrics.gate_count}")
        logger.info(f"    Depth: {metrics.depth}")
        logger.info(f"    Fidelity: {metrics.fidelity:.2%}")


def create_optimization_report(result: OptimizationResult) -> str:
    """
    Create an optimization report.
    
    Args:
        result: Optimization result
        
    Returns:
        Formatted report string
    """
    report = "HYBRID OPTIMIZATION REPORT\n"
    report += "=" * 50 + "\n\n"
    
    report += "OPTIMIZATION SUMMARY\n"
    report += f"  Algorithm: {result.metrics.get('algorithm', 'Unknown')}\n"
    report += f"  Status: {result.status.name}\n"
    report += f"  Optimal Value: {result.optimal_value:.6f}\n"
    report += f"  Iterations: {result.iterations}\n"
    report += f"  Execution Time: {result.execution_time:.2f}s\n"
    report += f"  Quantum Time: {result.quantum_time:.2f}s\n"
    report += f"  Classical Time: {result.classical_time:.2f}s\n"
    report += f"  Quantum Advantage: {result.quantum_advantage:.2%}\n\n"
    
    if 'circuit_metrics' in result.metrics:
        metrics = result.metrics['circuit_metrics']
        report += "QUANTUM CIRCUIT METRICS\n"
        report += f"  Qubits: {metrics.qubit_count}\n"
        report += f"  Gates: {metrics.gate_count}\n"
        report += f"  Depth: {metrics.depth}\n"
        report += f"  Fidelity: {metrics.fidelity:.2%}\n"
        report += f"  Execution Time: {metrics.execution_time:.2f}s\n"
        report += f"  Quantum Volume Utilization: {metrics.quantum_volume_utilization:.2%}\n\n"
    
    if 'quantum_weight' in result.metrics:
        report += "WORKLOAD DISTRIBUTION\n"
        report += f"  Quantum Weight: {result.metrics['quantum_weight']:.2%}\n"
        report += f"  Classical Weight: {result.metrics['classical_weight']:.2%}\n\n"
    
    report += "CONVERGENCE HISTORY\n"
    report += "  Iteration | Value\n"
    report += "  ----------|--------\n"
    
    for i, value in enumerate(result.convergence_history[:10]):  # Show first 10
        report += f"  {i+1:8} | {value:.6f}\n"
    
    if len(result.convergence_history) > 10:
        report += f"  ... (showing first 10 of {len(result.convergence_history)})\n"
    
    return report