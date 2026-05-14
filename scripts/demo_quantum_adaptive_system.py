#!/usr/bin/env python3
"""
Quantum-Enhanced Adaptive Trading System Demo

This script demonstrates the complete quantum-enhanced adaptive trading system,
including all components working together with quantum optimizations.
"""

import sys
import os
import numpy as np
import time
from typing import Optional

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
import logging
import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Add imports for core components
import sys
import os
import logging
import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Import core components
from core.real_time_learning.orchestrator import RealTimeLearningOrchestrator
from core.real_time_learning.strategy_allocator import AdaptiveStrategyAllocator
from core.real_time_learning.correlation_matrix import DynamicCorrelationMatrix
from core.real_time_learning.order_router import SmartOrderRouter
from core.real_time_learning.regime_parameters import RegimeSpecificParameters
from core.real_time_learning.paper_validation import PaperValidationEngine
from core.real_time_learning.quantum_paper_validation import QuantumPaperValidationEngine

# Import quantum components
from quantum.advanced.quantum_circuit_optimizer import (
    AdvancedQuantumCircuitOptimizer,
    QuantumCircuitMetrics,
    QuantumHardwareType,
    OptimizationObjective
)
from quantum.advanced.quantum_neural_network import (
    QuantumNeuralNetwork,
    QNNArchitecture,
    QNNLayer,
    QNNLayerType,
    QuantumTrainingMode,
    QNNAdaptiveTrainer
)
from quantum.advanced.quantum_regime_detection import (
    MarketRegime,
    MarketDataFeatures,
    QuantumRegimeDetector,
    QuantumRegimeAdaptationSystem
)
from quantum.advanced.quantum_hardware_profiler import (
    QuantumHardwareProfiler,
    QuantumHardwareSelector,
    QuantumExecutionManager,
    QuantumResourceManager
)
from quantum.advanced.quantum_benchmarking import (
    QuantumBenchmarking,
    QuantumBenchmarkOrchestrator,
    QuantumPerformanceTracker,
    BenchmarkType,
    MetricType
)

# Import monitoring components
from monitoring.quantum_dashboard import QuantumDashboard
from monitoring.quantum_visualization import QuantumVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import core components
from core.real_time_learning.orchestrator import RealTimeLearningOrchestrator
from core.real_time_learning.strategy_allocator import AdaptiveStrategyAllocator
from core.real_time_learning.correlation_matrix import DynamicCorrelationMatrix
from core.real_time_learning.order_router import SmartOrderRouter
from core.real_time_learning.regime_parameters import RegimeSpecificParameters
from core.real_time_learning.paper_validation import PaperValidationEngine
from core.real_time_learning.quantum_paper_validation import QuantumPaperValidationEngine

# Import quantum components
from quantum.advanced.quantum_circuit_optimizer import (
    AdvancedQuantumCircuitOptimizer,
    QuantumCircuitMetrics,
    QuantumCircuitProfile,
    QuantumHardwareType,
    OptimizationObjective
)
from quantum.advanced.quantum_neural_network import (
    QuantumNeuralNetwork,
    QNNArchitecture,
    QNNLayer,
    QNNLayerType,
    QuantumTrainingMode,
    QNNAdaptiveTrainer
)
from quantum.advanced.quantum_regime_detection import (
    MarketRegime,
    MarketDataFeatures,
    QuantumRegimeDetector,
    QuantumRegimeAdaptationSystem
)
from quantum.advanced.quantum_hardware_profiler import (
    QuantumHardwareProfiler,
    QuantumHardwareSelector,
    QuantumExecutionManager,
    QuantumResourceManager
)
from quantum.advanced.quantum_benchmarking import (
    QuantumBenchmarking,
    QuantumBenchmarkOrchestrator,
    QuantumPerformanceTracker,
    BenchmarkType,
    MetricType
)

# Import monitoring components
from monitoring.quantum_dashboard import QuantumDashboard
from monitoring.quantum_visualization import QuantumVisualizer

class QuantumAdaptiveSystemDemo:
    """Comprehensive demo of the quantum-enhanced adaptive trading system"""
    
    def __init__(self):
        logger.info("Initializing Quantum-Enhanced Adaptive Trading System Demo")
        self._detector_trained: Optional[bool] = None
        
        # Initialize core adaptive components
        self.orchestrator = RealTimeLearningOrchestrator()
        self.strategy_allocator = AdaptiveStrategyAllocator()
        self.correlation_matrix = DynamicCorrelationMatrix()
        self.order_router = SmartOrderRouter()
        self.regime_parameters = RegimeSpecificParameters()
        
        # Initialize validation engines
        self.paper_validation = PaperValidationEngine()
        self.quantum_validation = QuantumPaperValidationEngine()
        
        # Initialize quantum components
        self.circuit_optimizer = AdvancedQuantumCircuitOptimizer()
        self.quantum_regime_detector = QuantumRegimeDetector(num_qubits=4)
        self.quantum_adaptation_system = QuantumRegimeAdaptationSystem(num_qubits=4)
        
        # Initialize quantum hardware management
        self.hardware_profiler = QuantumHardwareProfiler()
        self.hardware_selector = QuantumHardwareSelector(self.hardware_profiler)
        self.execution_manager = QuantumExecutionManager(self.hardware_profiler, self.hardware_selector)
        self.resource_manager = QuantumResourceManager(self.hardware_profiler)
        
        # Initialize benchmarking system
        self.benchmarking = QuantumBenchmarking()
        
        # Initialize monitoring
        self.quantum_dashboard = QuantumDashboard()
        self.quantum_visualizer = QuantumVisualizer()
        
        # Register components with orchestrator
        self._register_components()
        
        logger.info("Quantum-Enhanced Adaptive Trading System Demo initialized successfully")
    
    def _register_components(self):
        """Register all components with the orchestrator"""
        logger.info("Registering components with orchestrator...")
        
        # Register core adaptive components
        self.orchestrator.register_component(self.strategy_allocator)
        self.orchestrator.register_component(self.correlation_matrix)
        self.orchestrator.register_component(self.order_router)
        self.orchestrator.register_component(self.regime_parameters)
        
        # Register validation engines
        self.orchestrator.validator = self.paper_validation

        
        logger.info(f"Registered {len(self.orchestrator.components)} core components")
    
    def generate_synthetic_market_data(self, num_samples: int = 100) -> Tuple[List[MarketDataFeatures], List[MarketRegime]]:
        """Generate synthetic market data for demo purposes"""
        logger.info(f"Generating {num_samples} synthetic market data samples...")
        
        np.random.seed(42)
        features_list = []
        regimes = []
        
        for i in range(num_samples):
            # Generate random market features
            returns = np.random.normal(0, 0.01)
            volatility = np.random.uniform(0.01, 0.05)
            volume = np.random.uniform(0.8, 1.2)
            momentum = np.random.normal(0, 0.005)
            correlation = np.random.uniform(0.5, 0.9)
            sentiment = np.random.normal(0, 0.1)
            
            features = MarketDataFeatures(
                returns=np.array([returns]),
                volatility=np.array([volatility]),
                volume=np.array([volume]),
                momentum=np.array([momentum]),
                correlation=np.array([correlation]),
                sentiment=np.array([sentiment])
            )
            features_list.append(features)
            
            # Determine regime based on features
            if volatility > 0.03 and abs(returns) > 0.015:
                regimes.append(MarketRegime.VOLATILE)
            elif abs(momentum) > 0.007:
                regimes.append(MarketRegime.TRENDING)
            elif volatility < 0.02 and abs(returns) < 0.005:
                regimes.append(MarketRegime.STABLE)
            else:
                regimes.append(MarketRegime.RANGE)
        
        logger.info("Synthetic market data generated successfully")
        return features_list, regimes
    
def train_quantum_regime_detector(self, num_samples: int = 100, epochs: int = 20, min_accuracy: float = 0.95):
        """Train the quantum regime detector with termination conditions"""
        logger.info(f"Training quantum regime detector with {num_samples} samples for {epochs} epochs...")
        logger.info(f"Training will stop if accuracy reaches {min_accuracy:.0%} or after {epochs} epochs")

        # Generate training data
        features_list, regimes = self.generate_synthetic_market_data(num_samples)

        # Convert MarketDataFeatures to proper training data format
        training_data = {
            "features": [
                [
                    features.returns[0],
                    features.volatility[0],
                    features.volume[0],
                    features.momentum[0],
                    features.correlation,
                    features.sentiment[0]
                ] for features in features_list
            ],
            "regimes": [regime.value for regime in regimes]
        }

        # Generate training data
        features_list, regimes = self.generate_synthetic_market_data(num_samples)

        # Convert MarketDataFeatures to 2D array
        feature_array = np.array([
            [
                features.returns[0],
                features.volatility[0],
                features.volume[0],
                features.momentum[0],
                features.correlation,
                features.sentiment[0]
            ] for features in features_list
        ])

        # Convert regimes to array
        regime_array = np.array([regime.value for regime in regimes])

        # Train the detector with early stopping
        train_result = self.quantum_regime_detector.train_detector(
            training_data=training_data,
            epochs=epochs,
            learning_rate=0.01,
            early_stopping=True,
            min_accuracy=min_accuracy,
            patience=3  # Stop after 3 epochs without improvement
        )

        logger.info(f"Quantum regime detector training completed. "
                  f"Final accuracy: {train_result['final_accuracy']:.2%}, "
                  f"Quantum contribution: {train_result['avg_quantum_contribution']:.2%}, "
                  f"Epochs run: {train_result['epochs_run']}")

        return train_result
    
    def demonstrate_quantum_circuit_optimization(self):
        """Demonstrate quantum circuit optimization with proper typing"""
        logger.info("Demonstrating quantum circuit optimization...")

        # Create initial circuit metrics
        initial_metrics = QuantumCircuitMetrics(
            depth=150,
            gate_count=800,
            qubit_count=20,
            fidelity=0.92,
            estimated_latency_ms=80
        )

        # Create circuit profile with proper typing
        circuit_profile = self.circuit_optimizer.create_circuit_profile(
            circuit_id="portfolio_optimizer",
            hardware_type=QuantumHardwareType.IBM_QISKIT,
            initial_metrics=initial_metrics
        )

        # Optimize circuit
        optimized_profile = self.circuit_optimizer.optimize_circuit(circuit_profile)

        # Analyze results
        analysis = self.circuit_optimizer.analyze_optimization(optimized_profile)

        logger.info(f"Quantum circuit optimization results:")
        logger.info(f"  Original depth: {analysis['original_metrics']['depth']} → "
                  f"Optimized: {analysis['optimized_metrics']['depth']}")
        logger.info(f"  Original gates: {analysis['original_metrics']['gate_count']} → "
                  f"Optimized: {analysis['optimized_metrics']['gate_count']}")
        logger.info(f"  Original fidelity: {analysis['original_metrics']['fidelity']:.2%} → "
                  f"Optimized: {analysis['optimized_metrics']['fidelity']:.2%}")
        logger.info(f"  Overall improvement: {analysis['improvement_ratio']:.2%}")

        return optimized_profile
        
        # Create initial circuit metrics
        initial_metrics = QuantumCircuitMetrics(
            depth=150,
            gate_count=800,
            qubit_count=20,
            fidelity=0.92,
            estimated_latency_ms=80
        )
        
        # Create initial circuit metrics
        initial_metrics = QuantumCircuitMetrics(
            depth=150,
            gate_count=800,
            qubit_count=20,
            fidelity=0.92,
            estimated_latency_ms=80
        )

        # Create circuit profile with proper typing
        circuit_profile = self.circuit_optimizer.create_circuit_profile(
            circuit_id="portfolio_optimizer",
            hardware_type=QuantumHardwareType.IBM_QISKIT,
            initial_metrics=initial_metrics
        )
        
        # Optimize circuit
        optimized_profile = self.circuit_optimizer.optimize_circuit(circuit_profile)
        
        # Analyze results
        analysis = self.circuit_optimizer.analyze_optimization(optimized_profile)
        
        logger.info(f"Quantum circuit optimization results:")
        logger.info(f"  Original depth: {analysis['original_metrics']['depth']} → "
                  f"Optimized: {analysis['optimized_metrics']['depth']}")
        logger.info(f"  Original gates: {analysis['original_metrics']['gate_count']} → "
                  f"Optimized: {analysis['optimized_metrics']['gate_count']}")
        logger.info(f"  Original fidelity: {analysis['original_metrics']['fidelity']:.2%} → "
                  f"Optimized: {analysis['optimized_metrics']['fidelity']:.2%}")
        logger.info(f"  Overall improvement: {analysis['improvement_ratio']:.2%}")
        
        return optimized_profile
    
    def demonstrate_quantum_neural_network(self):
        """Demonstrate quantum neural network for adaptive learning"""
        logger.info("Demonstrating quantum neural network for adaptive learning...")
        
        # Create adaptive trainer
        trainer = QNNAdaptiveTrainer(input_dim=4, output_dim=1, hardware_backend="simulator")
        
        # Generate training data
        X = np.random.rand(50, 4)  # 50 samples, 4 features
        y = np.random.rand(50)    # Target values
        
        # Train adaptive model
        model = trainer.train_adaptive_model(X, y, epochs=15, learning_rate=0.01)
        
        # Get training summary
        summary = model.get_training_summary()
        
        logger.info(f"Quantum neural network training results:")
        logger.info(f"  Final accuracy: {summary['final_accuracy']:.2%}")
        logger.info(f"  Best accuracy: {summary['best_accuracy']:.2%}")
        logger.info(f"  Quantum advantage: {summary['final_quantum_advantage']:.2%}")
        logger.info(f"  Average fidelity: {summary['avg_fidelity']:.2%}")
        
        return model
    
    def demonstrate_quantum_regime_detection(self, num_test_samples: int = 5):
        """Demonstrate quantum-enhanced regime detection"""
        logger.info(f"Demonstrating quantum-enhanced regime detection with {num_test_samples} test samples...")
        
        # First train the detector if not already trained
        if not hasattr(self, '_detector_trained') or not self._detector_trained:
            self.train_quantum_regime_detector(num_samples=100, epochs=20)
            self._detector_trained = True
        
        # Generate test data
        test_features, test_regimes = self.generate_synthetic_market_data(num_test_samples)
        
        # Detect regimes
        detections = []
        for i, features in enumerate(test_features):
            detection = self.quantum_regime_detector.detect_regime(features)
            detections.append(detection)
            
            logger.info(f"Sample {i+1}:")
            logger.info(f"  True regime: {test_regimes[i].name}")
            logger.info(f"  Detected regime: {detection.regime.name} (confidence: {detection.confidence:.2%})")
            logger.info(f"  Quantum contribution: {detection.quantum_contribution:.2%}")
        
        # Calculate accuracy
        correct = sum(1 for i, detection in enumerate(detections)
                     if detection.regime == test_regimes[i])
        accuracy = correct / num_test_samples
        
        avg_quantum_contribution = np.mean([d.quantum_contribution for d in detections])
        
        logger.info(f"Regime detection results:")
        logger.info(f"  Accuracy: {accuracy:.2%}")
        logger.info(f"  Average quantum contribution: {avg_quantum_contribution:.2%}")
        
        return detections
    
    def demonstrate_quantum_hardware_profiling(self):
        """Demonstrate quantum hardware profiling and optimization with proper typing"""
        logger.info("Demonstrating quantum hardware profiling and optimization...")

        # Create a sample circuit with proper typing
        circuit = HardwareCircuitProfile(
            circuit_id="sample_circuit",
            num_qubits=10,
            depth=100,
            gate_count=500,
            gate_types={"rx": 100, "ry": 100, "cx": 300},
            connectivity=[(i, i+1) for i in range(9)]  # Linear connectivity
        )

        # Profile on different backends
        backends = ["simulator_statevector", "ibm_lagos", "ibm_nairobi"]
        comparison = self.hardware_profiler.compare_backends(circuit, backends)

        logger.info("Quantum hardware comparison results:")
        for backend_name, result in comparison['backends'].items():
            logger.info(f"  {backend_name}:")
            logger.info(f"    Compatibility: {result['compatibility']['overall_compatibility']:.2%}")
            logger.info(f"    Expected fidelity: {result['performance']['expected_fidelity']:.2%}")
            logger.info(f"    Expected execution time: {result['performance']['expected_execution_time_ms']:.1f}ms")
            logger.info(f"    Quantum volume utilization: {result['performance']['quantum_volume_utilization']:.2f}")

        # Select best backend
        best_backend = comparison['best_backend']
        logger.info(f"Best backend selected: {best_backend}")

        # Optimize for the best backend
        optimization_result = self.hardware_profiler.optimize_for_backend(best_backend, circuit)

        logger.info(f"Optimization results for {best_backend}:")
        logger.info(f"  Qubit reduction: {optimization_result.optimization_metrics['qubit_reduction']:.2%}")
        logger.info(f"  Depth reduction: {optimization_result.optimization_metrics['depth_reduction']:.2%}")
        logger.info(f"  Gate reduction: {optimization_result.optimization_metrics['gate_reduction']:.2%}")
        logger.info(f"  Fidelity improvement: {optimization_result.optimization_metrics['fidelity_improvement']:.2%}")

        return optimization_result
        
        # Create a sample circuit with proper typing
        circuit = HardwareCircuitProfile(
            circuit_id="sample_circuit",
            num_qubits=10,
            depth=100,
            gate_count=500,
            gate_types={"rx": 100, "ry": 100, "cx": 300},
            connectivity=[(i, i+1) for i in range(9)]  # Linear connectivity
        )
        
        # Profile on different backends
        backends = ["simulator_statevector", "ibm_lagos", "ibm_nairobi"]
        comparison = self.hardware_profiler.compare_backends(circuit, backends)
        
        logger.info("Quantum hardware comparison results:")
        for backend_name, result in comparison['backends'].items():
            logger.info(f"  {backend_name}:")
            logger.info(f"    Compatibility: {result['compatibility']['overall_compatibility']:.2%}")
            logger.info(f"    Expected fidelity: {result['performance']['expected_fidelity']:.2%}")
            logger.info(f"    Expected execution time: {result['performance']['expected_execution_time_ms']:.1f}ms")
            logger.info(f"    Quantum volume utilization: {result['performance']['quantum_volume_utilization']:.2f}")
        
        # Select best backend
        best_backend = comparison['best_backend']
        logger.info(f"Best backend selected: {best_backend}")
        
        # Optimize for the best backend
        optimization_result = self.hardware_profiler.optimize_for_backend(best_backend, circuit)
        
        logger.info(f"Optimization results for {best_backend}:")
        logger.info(f"  Qubit reduction: {optimization_result.optimization_metrics['qubit_reduction']:.2%}")
        logger.info(f"  Depth reduction: {optimization_result.optimization_metrics['depth_reduction']:.2%}")
        logger.info(f"  Gate reduction: {optimization_result.optimization_metrics['gate_reduction']:.2%}")
        logger.info(f"  Fidelity improvement: {optimization_result.optimization_metrics['fidelity_improvement']:.2%}")
        
        return optimization_result
    
    def demonstrate_quantum_benchmarking(self):
        """Demonstrate quantum benchmarking framework with proper typing"""
        logger.info("Demonstrating quantum benchmarking framework...")

        # Create a benchmark suite
        suite = self.benchmarking.create_benchmark_suite(
            suite_id="quantum_benchmark_demo",
            benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
            description="Demonstration of quantum portfolio optimization benchmarking"
        )

        # Create benchmark orchestrator
        orchestrator = QuantumBenchmarkOrchestrator(self.benchmarking)

        # Define benchmark functions
        def quantum_portfolio_optimization(params: Dict[str, Any]) -> Dict[MetricType, float]:
            """Simulate quantum portfolio optimization"""
            # Simulate quantum optimization with some random variation
            time.sleep(0.1)  # Simulate computation time

            return {
                MetricType.EXECUTION_TIME: 80 + np.random.normal(0, 5),
                MetricType.FIDELITY: 0.95 + np.random.normal(0, 0.01),
                MetricType.ACCURACY: 0.92 + np.random.normal(0, 0.01),
                MetricType.SHARPE_RATIO: 2.2 + np.random.normal(0, 0.05)
            }

        def classical_portfolio_optimization(params: Dict[str, Any]) -> Dict[MetricType, float]:
            """Simulate classical portfolio optimization"""
            # Simulate classical optimization with some random variation
            time.sleep(0.15)  # Simulate computation time

            return {
                MetricType.EXECUTION_TIME: 100 + np.random.normal(0, 5),
                MetricType.FIDELITY: 0.90 + np.random.normal(0, 0.01),
                MetricType.ACCURACY: 0.88 + np.random.normal(0, 0.01),
                MetricType.SHARPE_RATIO: 2.0 + np.random.normal(0, 0.05)
            }

        # Define parameter grid
        parameter_grid = [
            {"qubits": 4, "depth": 10, "iterations": 100},
            {"qubits": 5, "depth": 15, "iterations": 150},
            {"qubits": 6, "depth": 20, "iterations": 200}
        ]

        # Create benchmark plan
        plan = orchestrator.create_benchmark_plan(
            plan_id="portfolio_optimization_plan",
            benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
            description="Portfolio optimization benchmark plan",
            parameters_grid=parameter_grid,
            quantum_algorithm="QAOA",
            classical_algorithm="Mean-Variance",
            quantum_backend="simulator",
            classical_backend="cpu"
        )

        # Execute benchmark plan
        result = orchestrator.execute_benchmark_plan(
            plan_id="portfolio_optimization_plan",
            quantum_exec_func=quantum_portfolio_optimization,
            classical_exec_func=classical_portfolio_optimization
        )

        logger.info(f"Quantum benchmarking results:")
        logger.info(f"  Total benchmarks: {result['total_benchmarks']}")
        logger.info(f"  Successful benchmarks: {result['successful_benchmarks']}")
        logger.info(f"  Failed benchmarks: {result['failed_benchmarks']}")

        # Generate report
        report = suite.generate_report()
        logger.info(f"Benchmark report summary:")
        logger.info(f"  Average quantum advantage: {report['comparison_stats']['avg_quantum_advantage']:.2%}")
        logger.info(f"  Maximum quantum advantage: {report['comparison_stats']['max_quantum_advantage']:.2%}")
        logger.info(f"  Significant comparisons: {report['comparison_stats']['significant_comparisons']}")

        return result

            return {
                MetricType.EXECUTION_TIME: 80 + np.random.normal(0, 5),
                MetricType.FIDELITY: 0.95 + np.random.normal(0, 0.01),
                MetricType.ACCURACY: 0.92 + np.random.normal(0, 0.01),
                MetricType.SHARPE_RATIO: 2.2 + np.random.normal(0, 0.05)
            }

        def classical_portfolio_optimization(params: Dict[str, Any]) -> Dict[MetricType, float]:
            """Simulate classical portfolio optimization"""
            # Simulate classical optimization with some random variation
            time.sleep(0.15)  # Simulate computation time

            return {
                MetricType.EXECUTION_TIME: 100 + np.random.normal(0, 5),
                MetricType.FIDELITY: 0.90 + np.random.normal(0, 0.01),
                MetricType.ACCURACY: 0.88 + np.random.normal(0, 0.01),
                MetricType.SHARPE_RATIO: 2.0 + np.random.normal(0, 0.05)
            }

        # Define parameter grid
        parameter_grid = [
            {"qubits": 4, "depth": 10, "iterations": 100},
            {"qubits": 5, "depth": 15, "iterations": 150},
            {"qubits": 6, "depth": 20, "iterations": 200}
        ]

        # Create benchmark plan
        plan = orchestrator.create_benchmark_plan(
            plan_id="portfolio_optimization_plan",
            benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
            description="Portfolio optimization benchmark plan",
            parameters_grid=parameter_grid,
            quantum_algorithm="QAOA",
            classical_algorithm="Mean-Variance",
            quantum_backend="simulator",
            classical_backend="cpu"
        )

        # Execute benchmark plan
        result = orchestrator.execute_benchmark_plan(
            plan_id="portfolio_optimization_plan",
            quantum_exec_func=quantum_portfolio_optimization,
            classical_exec_func=classical_portfolio_optimization
        )
        
# Create a benchmark suite
        suite = self.benchmarking.create_benchmark_suite(
            suite_id="quantum_benchmark_demo",
            benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
            description="Demonstration of quantum portfolio optimization benchmarking"
        )

        # Create benchmark orchestrator
        orchestrator = QuantumBenchmarkOrchestrator()

        # Define benchmark functions
        def quantum_portfolio_optimization(params: Dict[str, Any]) -> Dict[MetricType, float]:
            """Simulate quantum portfolio optimization"""
            # Simulate quantum optimization with some random variation
            time.sleep(0.1)  # Simulate computation time
            
            return {
                MetricType.EXECUTION_TIME: 80 + np.random.normal(0, 5),
                MetricType.FIDELITY: 0.95 + np.random.normal(0, 0.01),
                MetricType.ACCURACY: 0.92 + np.random.normal(0, 0.01),
                MetricType.SHARPE_RATIO: 2.2 + np.random.normal(0, 0.05)
            }
        
        def classical_portfolio_optimization(params):
            """Simulate classical portfolio optimization"""
            # Simulate classical optimization with some random variation
            time.sleep(0.15)  # Simulate computation time
            
            return {
                MetricType.EXECUTION_TIME: 100 + np.random.normal(0, 5),
                MetricType.FIDELITY: 0.90 + np.random.normal(0, 0.01),
                MetricType.ACCURACY: 0.88 + np.random.normal(0, 0.01),
                MetricType.SHARPE_RATIO: 2.0 + np.random.normal(0, 0.05)
            }
        
        # Define parameter grid
        parameter_grid = [
            {"qubits": 4, "depth": 10, "iterations": 100},
            {"qubits": 5, "depth": 15, "iterations": 150},
            {"qubits": 6, "depth": 20, "iterations": 200}
        ]
        
        # Create benchmark plan
        plan = self.benchmarking.orchestrator.create_benchmark_plan(
            plan_id="portfolio_optimization_plan",
            benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
            description="Portfolio optimization benchmark plan",
            parameters_grid=parameter_grid,
            quantum_algorithm="QAOA",
            classical_algorithm="Mean-Variance",
            quantum_backend="simulator",
            classical_backend="cpu"
        )
        
        # Execute benchmark plan
        result = self.benchmarking.orchestrator.execute_benchmark_plan(
            plan_id="portfolio_optimization_plan",
            quantum_exec_func=quantum_portfolio_optimization,
            classical_exec_func=classical_portfolio_optimization
        )
        
        logger.info(f"Quantum benchmarking results:")
        logger.info(f"  Total benchmarks: {result['total_benchmarks']}")
        logger.info(f"  Successful benchmarks: {result['successful_benchmarks']}")
        logger.info(f"  Failed benchmarks: {result['failed_benchmarks']}")
        
        # Generate report
        report = suite.generate_report()
        logger.info(f"Benchmark report summary:")
        logger.info(f"  Average quantum advantage: {report['comparison_stats']['avg_quantum_advantage']:.2%}")
        logger.info(f"  Maximum quantum advantage: {report['comparison_stats']['max_quantum_advantage']:.2%}")
        logger.info(f"  Significant comparisons: {report['comparison_stats']['significant_comparisons']}")
        
        return result
    
def demonstrate_adaptive_system_integration(self, max_iterations: int = 3):
        """Demonstrate integration of quantum components with adaptive system with iteration limits"""
        logger.info("Demonstrating integration of quantum components with adaptive system...")
        logger.info(f"Running for maximum of {max_iterations} iterations")

        iteration = 0
        results = []

        # Initialize with default weights and parameters
        current_weights = {"momentum": 0.33, "mean_reversion": 0.33, "breakout": 0.33}
        current_params = {"max_leverage": 2.0, "position_size_pct": 0.05}

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"\n--- Iteration {iteration}/{max_iterations} ---")

            # Generate market data
            features_list, regimes = self.generate_synthetic_market_data(num_samples=1)
            current_features = features_list[0]

            # Detect regime using quantum detector
            regime_detection = self.quantum_regime_detector.detect_regime(current_features)

            logger.info(f"Detected market regime: {regime_detection.regime.name}")
            logger.info(f"Regime detection confidence: {regime_detection.confidence:.2%}")
            logger.info(f"Quantum contribution to detection: {regime_detection.quantum_contribution:.2%}")

            # Get adaptation from quantum adaptation system
            adaptation_result = self.quantum_adaptation_system.detect_and_adapt(current_features)

            logger.info("Quantum-enhanced adaptation results:")
            logger.info(f"  Strategy weights: {adaptation_result['adaptation']['strategy_weights']}")
            logger.info(f"  Risk parameters: {adaptation_result['adaptation']['risk_parameters']}")
            logger.info(f"  Execution parameters: {adaptation_result['adaptation']['execution_parameters']}")

            # Apply adaptation to core components
            strategy_weights = adaptation_result['adaptation']['strategy_weights']
            risk_params = adaptation_result['adaptation']['risk_parameters']

            # Update strategy allocator
            self.strategy_allocator.update_weights(strategy_weights)

            # Update regime parameters
            self.regime_parameters.update_parameters(
                regime_detection.regime,
                risk_params
            )

            logger.info("Adaptation applied to core components")

            # Get current component states
            current_weights = self.strategy_allocator.get_current_weights()
            current_params = self.regime_parameters.get_current_parameters()

            logger.info(f"Current strategy weights: {current_weights}")
            logger.info(f"Current regime parameters: {current_params}")

            # Validate changes with quantum validation engine
            proposed_changes = {
                "strategy_allocator": {
                    "old_weights": {"momentum": 0.33, "mean_reversion": 0.33, "breakout": 0.33},
                    "new_weights": current_weights
                },
                "regime_parameters": {
                    "old_params": {"max_leverage": 2.0, "position_size_pct": 0.05},
                    "new_params": {
                        "max_leverage": current_params["max_leverage"],
                        "position_size_pct": current_params["position_size_pct"]
                    }
                }
            }

            validation_result = self.quantum_validation.validate_with_quantum(
                proposed_changes=proposed_changes,
                classical_result={"valid": True, "sharpe_ratio": 2.0},
                hardware_backend="simulator",
                min_quantum_improvement=0.05
            )

            logger.info(f"Quantum validation results:")
            logger.info(f"  Classical validation: {validation_result['classical_valid']}")
            logger.info(f"  Quantum validation: {validation_result['quantum_valid']}")
            logger.info(f"  Quantum improvement: {validation_result['quantum_improvement']:.2%}")
            logger.info(f"  Quantum metadata: {validation_result['quantum_metadata']}")

            # Store iteration results
            results.append({
                "iteration": iteration,
                "regime_detection": regime_detection.to_dict(),
                "adaptation_result": adaptation_result,
                "validation_result": validation_result
            })

            # Check if we should stop early
            if validation_result['quantum_improvement'] < 0.01:
                logger.info("Quantum improvement below threshold (1%), stopping early")
                break

        logger.info(f"Completed {iteration} iterations of adaptive system integration")
        return {
            "iterations": iteration,
            "results": results,
            "final_state": {
                "strategy_weights": current_weights,
                "regime_parameters": current_params
            }
        }

        iteration = 0
        results = []

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"\n--- Iteration {iteration}/{max_iterations} ---")

            # Generate market data
            features_list, regimes = self.generate_synthetic_market_data(num_samples=1)
            current_features = features_list[0]

            # Detect regime using quantum detector
            regime_detection = self.quantum_regime_detector.detect_regime(current_features)

            logger.info(f"Detected market regime: {regime_detection.regime.name}")
            logger.info(f"Regime detection confidence: {regime_detection.confidence:.2%}")
            logger.info(f"Quantum contribution to detection: {regime_detection.quantum_contribution:.2%}")

            # Get adaptation from quantum adaptation system
            adaptation_result = self.quantum_adaptation_system.detect_and_adapt(current_features)

            logger.info("Quantum-enhanced adaptation results:")
            logger.info(f"  Strategy weights: {adaptation_result['adaptation']['strategy_weights']}")
            logger.info(f"  Risk parameters: {adaptation_result['adaptation']['risk_parameters']}")
            logger.info(f"  Execution parameters: {adaptation_result['adaptation']['execution_parameters']}")

            # Apply adaptation to core components
            strategy_weights = adaptation_result['adaptation']['strategy_weights']
            risk_params = adaptation_result['adaptation']['risk_parameters']

            # Update strategy allocator
            self.strategy_allocator.update_weights(strategy_weights)

            # Update regime parameters
            self.regime_parameters.update_parameters(
                regime_detection.regime,
                risk_params
            )

            logger.info("Adaptation applied to core components")

            # Get current component states
            current_weights = self.strategy_allocator.get_current_weights()
            current_params = self.regime_parameters.get_current_parameters()

            logger.info(f"Current strategy weights: {current_weights}")
            logger.info(f"Current regime parameters: {current_params}")

            # Validate changes with quantum validation engine
            proposed_changes = {
                "strategy_allocator": {
                    "old_weights": {"momentum": 0.33, "mean_reversion": 0.33, "breakout": 0.33},
                    "new_weights": current_weights
                },
                "regime_parameters": {
                    "old_params": {"max_leverage": 2.0, "position_size_pct": 0.05},
                    "new_params": {
                        "max_leverage": current_params["max_leverage"],
                        "position_size_pct": current_params["position_size_pct"]
                    }
                }
            }

            validation_result = self.quantum_validation.validate_with_quantum(
                proposed_changes=proposed_changes,
                classical_result={"valid": True, "sharpe_ratio": 2.0},
                hardware_backend="simulator",
                min_quantum_improvement=0.05
            )

            logger.info(f"Quantum validation results:")
            logger.info(f"  Classical validation: {validation_result['classical_valid']}")
            logger.info(f"  Quantum validation: {validation_result['quantum_valid']}")
            logger.info(f"  Quantum improvement: {validation_result['quantum_improvement']:.2%}")
            logger.info(f"  Quantum metadata: {validation_result['quantum_metadata']}")

            # Store iteration results
            results.append({
                "iteration": iteration,
                "regime_detection": regime_detection.to_dict(),
                "adaptation_result": adaptation_result,
                "validation_result": validation_result
            })

            # Check if we should stop early
            if validation_result['quantum_improvement'] < 0.01:
                logger.info("Quantum improvement below threshold (1%), stopping early")
                break

        logger.info(f"Completed {iteration} iterations of adaptive system integration")
        return {
            "iterations": iteration,
            "results": results,
            "final_state": {
                "strategy_weights": current_weights,
                "regime_parameters": current_params
            }
        }
    
    def demonstrate_quantum_dashboard(self):
        """Demonstrate quantum dashboard and visualization with proper method calls"""
        logger.info("Demonstrating quantum dashboard and visualization...")

        # Use proper method names based on the actual QuantumDashboard class
        self.quantum_dashboard.add_quantum_component(
            component_name="strategy_allocator",
            quantum_improvement=0.08,
            execution_mode="qaoa_simulator",
            metadata={
                "quantum_sharpe": 2.2,
                "classical_sharpe": 2.0,
                "quantum_kernel": "linear"
            }
        )

        self.quantum_dashboard.add_quantum_component(
            component_name="correlation_matrix",
            quantum_improvement=0.06,
            execution_mode="quantum_annealing",
            metadata={
                "quantum_diversification": 0.92,
                "classical_diversification": 0.88,
                "quantum_qubits": 8
            }
        )

        self.quantum_dashboard.add_quantum_component(
            component_name="order_router",
            quantum_improvement=0.10,
            execution_mode="quantum_search",
            metadata={
                "quantum_fill_rate": 0.95,
                "classical_fill_rate": 0.90,
                "quantum_latency": 15
            }
        )
        
        # Add some quantum components to dashboard
        self.quantum_dashboard.add_component(
            component_name="strategy_allocator",
            quantum_improvement=0.08,
            execution_mode="qaoa_simulator",
            metadata={
                "quantum_sharpe": 2.2,
                "classical_sharpe": 2.0,
                "quantum_kernel": "linear"
            }
        )
        
        self.quantum_dashboard.add_component(
            component_name="correlation_matrix",
            quantum_improvement=0.06,
            execution_mode="quantum_annealing",
            metadata={
                "quantum_diversification": 0.92,
                "classical_diversification": 0.88,
                "quantum_qubits": 8
            }
        )
        
        self.quantum_dashboard.add_component(
            component_name="order_router",
            quantum_improvement=0.10,
            execution_mode="quantum_search",
            metadata={
                "quantum_fill_rate": 0.95,
                "classical_fill_rate": 0.90,
                "quantum_latency": 15
            }
        )
        
        # Generate dashboard summary
        summary = self.quantum_dashboard.generate_global_summary()
        
        logger.info(f"Quantum dashboard summary:")
        logger.info(f"  Components tracked: {summary['components_tracked']}")
        logger.info(f"  Global quantum advantage: {summary['global_quantum_advantage']:.2%}")
        logger.info(f"  Last update: {summary['last_update']}")
        
        # Generate quantum advantage report
        report = self.quantum_dashboard.get_quantum_advantage_report()
        
        logger.info(f"Quantum advantage report:")
        logger.info(f"  Global quantum advantage: {report['global_quantum_advantage']:.2%}")
        logger.info(f"  Execution modes used: {report['execution_modes_used']}")
        
        # Get component breakdown
        breakdown = self.quantum_dashboard.generate_component_breakdown()
        
        logger.info(f"Component breakdown:")
        for component, data in breakdown.items():
            logger.info(f"  {component}:")
            logger.info(f"    Avg improvement: {data['avg_improvement']:.2%}")
            logger.info(f"    Execution mode: {data['execution_mode']}")
        
        return {
            "summary": summary,
            "report": report,
            "breakdown": breakdown
        }
    
def run_comprehensive_demo(self, max_iterations: int = 3):
        """Run a comprehensive demo of all quantum-enhanced components with iteration limits"""
        logger.info("=" * 60)
        logger.info("QUANTUM-ENHANCED ADAPTIVE TRADING SYSTEM - COMPREHENSIVE DEMO")
        logger.info("=" * 60)
        logger.info(f"Running comprehensive demo with maximum {max_iterations} iterations per section")

        # Initialize results dictionary
        results = {}

        # 1. Quantum Circuit Optimization
        logger.info("\n1. QUANTUM CIRCUIT OPTIMIZATION")
        logger.info("-" * 40)
        optimization_result = self.demonstrate_quantum_circuit_optimization()
        results['optimization_result'] = optimization_result

        # 2. Quantum Neural Network
        logger.info("\n2. QUANTUM NEURAL NETWORK")
        logger.info("-" * 40)
        qnn_model = self.demonstrate_quantum_neural_network()
        results['qnn_model'] = qnn_model

        # 3. Quantum Regime Detection
        logger.info("\n3. QUANTUM REGIME DETECTION")
        logger.info("-" * 40)
        detections = self.demonstrate_quantum_regime_detection(num_test_samples=3)
        results['detections'] = detections

        # 4. Quantum Hardware Profiling
        logger.info("\n4. QUANTUM HARDWARE PROFILING")
        logger.info("-" * 40)
        hardware_result = self.demonstrate_quantum_hardware_profiling()
        results['hardware_result'] = hardware_result

        # 5. Quantum Benchmarking
        logger.info("\n5. QUANTUM BENCHMARKING")
        logger.info("-" * 40)
        benchmark_result = self.demonstrate_quantum_benchmarking()
        results['benchmark_result'] = benchmark_result

        # 6. Adaptive System Integration
        logger.info("\n6. ADAPTIVE SYSTEM INTEGRATION")
        logger.info("-" * 40)
        integration_result = self.demonstrate_adaptive_system_integration(max_iterations=max_iterations)
        results['integration_result'] = integration_result

        # 7. Quantum Dashboard
        logger.info("\n7. QUANTUM DASHBOARD")
        logger.info("-" * 40)
        dashboard_result = self.demonstrate_quantum_dashboard()
        results['dashboard_result'] = dashboard_result

        # 8. System Status
        logger.info("\n8. SYSTEM STATUS SUMMARY")
        logger.info("-" * 40)

        # Get orchestrator status
        orchestrator_status = self.orchestrator.get_system_status()

        # Get quantum dashboard status
        quantum_status = self.quantum_dashboard.generate_global_summary()

        # Get benchmarking status
        benchmarking_status = self.benchmarking.generate_benchmark_report()

        logger.info(f"Orchestrator status:")
        logger.info(f"  Registered components: {len(orchestrator_status['components'])}")
        logger.info(f"  Recent changes: {orchestrator_status['recent_changes']}")
        logger.info(f"  Recent adaptations: {len(orchestrator_status['recent_adaptations'])}")

        logger.info(f"Quantum dashboard status:")
        logger.info(f"  Global quantum advantage: {quantum_status['global_quantum_advantage']:.2%}")
        logger.info(f"  Components tracked: {quantum_status['components_tracked']}")

        logger.info(f"Benchmarking status:")
        logger.info(f"  Total suites: {benchmarking_status['total_suites']}")
        logger.info(f"  Total comparisons: {benchmarking_status['total_comparisons']}")

        # Add system status to results
        results['system_status'] = {
            "orchestrator": orchestrator_status,
            "quantum": quantum_status,
            "benchmarking": benchmarking_status
        }

        logger.info("\n" + "=" * 60)
        logger.info("QUANTUM-ENHANCED ADAPTIVE TRADING SYSTEM DEMO COMPLETED")
        logger.info("=" * 60)

        return results

        # Initialize results dictionary
        results = {}

        # 1. Quantum Circuit Optimization
        logger.info("\n1. QUANTUM CIRCUIT OPTIMIZATION")
        logger.info("-" * 40)
        optimization_result = self.demonstrate_quantum_circuit_optimization()
        results['optimization_result'] = optimization_result

        # 2. Quantum Neural Network
        logger.info("\n2. QUANTUM NEURAL NETWORK")
        logger.info("-" * 40)
        qnn_model = self.demonstrate_quantum_neural_network()
        results['qnn_model'] = qnn_model

        # 3. Quantum Regime Detection
        logger.info("\n3. QUANTUM REGIME DETECTION")
        logger.info("-" * 40)
        detections = self.demonstrate_quantum_regime_detection(num_test_samples=3)
        results['detections'] = detections

        # 4. Quantum Hardware Profiling
        logger.info("\n4. QUANTUM HARDWARE PROFILING")
        logger.info("-" * 40)
        hardware_result = self.demonstrate_quantum_hardware_profiling()
        results['hardware_result'] = hardware_result

        # 5. Quantum Benchmarking
        logger.info("\n5. QUANTUM BENCHMARKING")
        logger.info("-" * 40)
        benchmark_result = self.demonstrate_quantum_benchmarking()
        results['benchmark_result'] = benchmark_result

        # 6. Adaptive System Integration
        logger.info("\n6. ADAPTIVE SYSTEM INTEGRATION")
        logger.info("-" * 40)
        integration_result = self.demonstrate_adaptive_system_integration(max_iterations=max_iterations)
        results['integration_result'] = integration_result

        # 7. Quantum Dashboard
        logger.info("\n7. QUANTUM DASHBOARD")
        logger.info("-" * 40)
        dashboard_result = self.demonstrate_quantum_dashboard()
        results['dashboard_result'] = dashboard_result

        # 8. System Status
        logger.info("\n8. SYSTEM STATUS SUMMARY")
        logger.info("-" * 40)

        # Get orchestrator status
        orchestrator_status = self.orchestrator.get_system_status()

        # Get quantum dashboard status
        quantum_status = self.quantum_dashboard.generate_global_summary()

        # Get benchmarking status
        benchmarking_status = self.benchmarking.generate_benchmark_report()

        logger.info(f"Orchestrator status:")
        logger.info(f"  Registered components: {len(orchestrator_status['components'])}")
        logger.info(f"  Recent changes: {orchestrator_status['recent_changes']}")
        logger.info(f"  Recent adaptations: {len(orchestrator_status['recent_adaptations'])}")

        logger.info(f"Quantum dashboard status:")
        logger.info(f"  Global quantum advantage: {quantum_status['global_quantum_advantage']:.2%}")
        logger.info(f"  Components tracked: {quantum_status['components_tracked']}")

        logger.info(f"Benchmarking status:")
        logger.info(f"  Total suites: {benchmarking_status['total_suites']}")
        logger.info(f"  Total comparisons: {benchmarking_status['total_comparisons']}")

        # Add system status to results
        results['system_status'] = {
            "orchestrator": orchestrator_status,
            "quantum": quantum_status,
            "benchmarking": benchmarking_status
        }

        logger.info("\n" + "=" * 60)
        logger.info("QUANTUM-ENHANCED ADAPTIVE TRADING SYSTEM DEMO COMPLETED")
        logger.info("=" * 60)

        return results

        # Initialize results dictionary
        results = {}

        # 1. Quantum Circuit Optimization
        logger.info("\n1. QUANTUM CIRCUIT OPTIMIZATION")
        logger.info("-" * 40)
        optimization_result = self.demonstrate_quantum_circuit_optimization()
        results['optimization_result'] = optimization_result

        # 2. Quantum Neural Network
        logger.info("\n2. QUANTUM NEURAL NETWORK")
        logger.info("-" * 40)
        qnn_model = self.demonstrate_quantum_neural_network()
        results['qnn_model'] = qnn_model

        # 3. Quantum Regime Detection
        logger.info("\n3. QUANTUM REGIME DETECTION")
        logger.info("-" * 40)
        detections = self.demonstrate_quantum_regime_detection(num_test_samples=3)
        results['detections'] = detections

        # 4. Quantum Hardware Profiling
        logger.info("\n4. QUANTUM HARDWARE PROFILING")
        logger.info("-" * 40)
        hardware_result = self.demonstrate_quantum_hardware_profiling()
        results['hardware_result'] = hardware_result

        # 5. Quantum Benchmarking
        logger.info("\n5. QUANTUM BENCHMARKING")
        logger.info("-" * 40)
        benchmark_result = self.demonstrate_quantum_benchmarking()
        results['benchmark_result'] = benchmark_result

        # 6. Adaptive System Integration
        logger.info("\n6. ADAPTIVE SYSTEM INTEGRATION")
        logger.info("-" * 40)
        integration_result = self.demonstrate_adaptive_system_integration(max_iterations=max_iterations)
        results['integration_result'] = integration_result

        # 7. Quantum Dashboard
        logger.info("\n7. QUANTUM DASHBOARD")
        logger.info("-" * 40)
        dashboard_result = self.demonstrate_quantum_dashboard()
        results['dashboard_result'] = dashboard_result

        # 8. System Status
        logger.info("\n8. SYSTEM STATUS SUMMARY")
        logger.info("-" * 40)

        # Get orchestrator status
        orchestrator_status = self.orchestrator.get_system_status()

        # Get quantum dashboard status
        quantum_status = self.quantum_dashboard.get_global_summary()

        # Get benchmarking status
        benchmarking_status = self.benchmarking.generate_benchmark_report()

        logger.info(f"Orchestrator status:")
        logger.info(f"  Registered components: {len(orchestrator_status['components'])}")
        logger.info(f"  Recent changes: {orchestrator_status['recent_changes']}")
        logger.info(f"  Recent adaptations: {len(orchestrator_status['recent_adaptations'])}")

        logger.info(f"Quantum dashboard status:")
        logger.info(f"  Global quantum advantage: {quantum_status['global_quantum_advantage']:.2%}")
        logger.info(f"  Components tracked: {quantum_status['components_tracked']}")

        logger.info(f"Benchmarking status:")
        logger.info(f"  Total suites: {benchmarking_status['total_suites']}")
        logger.info(f"  Total comparisons: {benchmarking_status['total_comparisons']}")

        # Add system status to results
        results['system_status'] = {
            "orchestrator": orchestrator_status,
            "quantum": quantum_status,
            "benchmarking": benchmarking_status
        }

        logger.info("\n" + "=" * 60)
        logger.info("QUANTUM-ENHANCED ADAPTIVE TRADING SYSTEM DEMO COMPLETED")
        logger.info("=" * 60)

        return results

if __name__ == "__main__":
    # Create and run the demo with iteration limits
    demo = QuantumAdaptiveSystemDemo()
    results = demo.run_comprehensive_demo(max_iterations=3)

    # Print summary
    print("\n\n=== DEMO COMPLETION SUMMARY ===")
    print(f"Quantum Circuit Optimization: {results['optimization_result'].improvement_ratio():.2%} improvement")
    print(f"Quantum Neural Network: {results['qnn_model'].get_training_summary()['final_quantum_advantage']:.2%} advantage")
    print(f"Quantum Regime Detection: {np.mean([d.quantum_contribution for d in results['detections']]):.2%} avg quantum contribution")
    print(f"Quantum Benchmarking: {results['benchmark_result']['comparison_stats']['avg_quantum_advantage']:.2%} avg advantage")
    print(f"System Integration: Ran {results['integration_result']['iterations']} iterations with final quantum improvement of {results['integration_result']['results'][-1]['validation_result']['quantum_improvement']:.2%}")
    print(f"Global Quantum Advantage: {results['dashboard_result']['summary']['global_quantum_advantage']:.2%}")
    print("Demo completed successfully without infinite loops")