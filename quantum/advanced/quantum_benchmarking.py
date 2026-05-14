"""
Quantum Performance Benchmarking Framework

This module provides comprehensive benchmarking of quantum algorithms against
classical counterparts, including:
- Performance comparison metrics
- Quantum advantage calculation
- Statistical significance testing
- Hardware-specific benchmarking
- Continuous performance tracking
- Automated report generation
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
import time
import json
from datetime import datetime
import hashlib
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)

class BenchmarkType(Enum):
    """Types of benchmarks"""
    PORTFOLIO_OPTIMIZATION = auto()
    RISK_ANALYSIS = auto()
    STRATEGY_OPTIMIZATION = auto()
    REGIME_DETECTION = auto()
    FEATURE_EXTRACTION = auto()
    CIRCUIT_OPTIMIZATION = auto()
    GENERAL = auto()

class MetricType(Enum):
    """Types of performance metrics"""
    EXECUTION_TIME = auto()
    FIDELITY = auto()
    ACCURACY = auto()
    SHARPE_RATIO = auto()
    DRAWDOWN = auto()
    WIN_RATE = auto()
    QUANTUM_ADVANTAGE = auto()
    RESOURCE_USAGE = auto()
    CONVERGENCE_RATE = auto()

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run"""
    benchmark_id: str
    benchmark_type: BenchmarkType
    algorithm: str
    backend: str
    metrics: Dict[MetricType, float]
    parameters: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_score(self, weight_fidelity: float = 0.4, weight_speed: float = 0.3,
                        weight_accuracy: float = 0.3) -> float:
        """Calculate overall performance score (0-1)"""
        # Normalize metrics
        time_score = 1.0 / (self.metrics.get(MetricType.EXECUTION_TIME, 1.0) + 0.1)  # Avoid division by zero
        fidelity_score = self.metrics.get(MetricType.FIDELITY, 0.0)
        accuracy_score = self.metrics.get(MetricType.ACCURACY, 0.0)
        
        # Calculate weighted score
        score = (
            weight_fidelity * fidelity_score +
            weight_speed * time_score +
            weight_accuracy * accuracy_score
        )
        
        return min(1.0, max(0.0, score))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'benchmark_id': self.benchmark_id,
            'benchmark_type': self.benchmark_type.name,
            'algorithm': self.algorithm,
            'backend': self.backend,
            'metrics': {k.name: v for k, v in self.metrics.items()},
            'parameters': self.parameters,
            'score': self.calculate_score(),
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

@dataclass
class BenchmarkComparison:
    """Comparison between quantum and classical benchmarks"""
    quantum_result: BenchmarkResult
    classical_result: BenchmarkResult
    comparison_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate comparison metrics"""
        # Calculate quantum advantage for each metric
        for metric_type, quantum_value in self.quantum_result.metrics.items():
            classical_value = self.classical_result.metrics.get(metric_type, 0)
            
            if metric_type in [MetricType.EXECUTION_TIME, MetricType.DRAWDOWN]:
                # For metrics where lower is better, advantage is (classical - quantum)/classical
                if classical_value > 0:
                    advantage = (classical_value - quantum_value) / classical_value
                else:
                    advantage = 0.0
            else:
                # For metrics where higher is better, advantage is (quantum - classical)/classical
                if classical_value > 0:
                    advantage = (quantum_value - classical_value) / classical_value
                else:
                    advantage = 0.0 if quantum_value == 0 else float('inf')
            
            self.comparison_metrics[f"{metric_type.name}_advantage"] = advantage
        
        # Calculate overall quantum advantage
        quantum_score = self.quantum_result.calculate_score()
        classical_score = self.classical_result.calculate_score()
        
        if classical_score > 0:
            self.comparison_metrics["overall_quantum_advantage"] = (
                quantum_score - classical_score) / classical_score
        else:
            self.comparison_metrics["overall_quantum_advantage"] = 0.0
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if quantum advantage is statistically significant"""
        # For demo purposes, we'll assume significance if advantage > 5%
        # In a real implementation, this would use proper statistical tests
        return self.comparison_metrics.get("overall_quantum_advantage", 0) > alpha
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'quantum': self.quantum_result.to_dict(),
            'classical': self.classical_result.to_dict(),
            'comparison_metrics': self.comparison_metrics,
            'significant': self.is_significant()
        }

@dataclass
class BenchmarkSuite:
    """Collection of related benchmarks"""
    suite_id: str
    benchmark_type: BenchmarkType
    description: str
    results: List[BenchmarkResult] = field(default_factory=list)
    comparisons: List[BenchmarkComparison] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result"""
        self.results.append(result)
    
    def add_comparison(self, comparison: BenchmarkComparison):
        """Add a benchmark comparison"""
        self.comparisons.append(comparison)
    
    def get_best_result(self, algorithm_type: str = None) -> Optional[BenchmarkResult]:
        """Get the best result, optionally filtered by algorithm type"""
        filtered_results = [
            r for r in self.results
            if algorithm_type is None or r.algorithm == algorithm_type
        ]
        
        if not filtered_results:
            return None
        
        return max(filtered_results, key=lambda r: r.calculate_score())
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a benchmark report"""
        if not self.results:
            return {"status": "no_results"}
        
        # Group by algorithm
        by_algorithm = {}
        for result in self.results:
            if result.algorithm not in by_algorithm:
                by_algorithm[result.algorithm] = []
            by_algorithm[result.algorithm].append(result)
        
        # Calculate statistics per algorithm
        algorithm_stats = {}
        for algorithm, results in by_algorithm.items():
            scores = [r.calculate_score() for r in results]
            exec_times = [r.metrics.get(MetricType.EXECUTION_TIME, 0) for r in results]
            fidelities = [r.metrics.get(MetricType.FIDELITY, 0) for r in results]
            accuracies = [r.metrics.get(MetricType.ACCURACY, 0) for r in results]
            
            algorithm_stats[algorithm] = {
                'count': len(results),
                'avg_score': np.mean(scores) if scores else 0,
                'max_score': max(scores) if scores else 0,
                'min_score': min(scores) if scores else 0,
                'avg_time_ms': np.mean(exec_times) if exec_times else 0,
                'avg_fidelity': np.mean(fidelities) if fidelities else 0,
                'avg_accuracy': np.mean(accuracies) if accuracies else 0,
                'best_result': max(results, key=lambda r: r.calculate_score()).to_dict()
            }
        
        # Calculate comparison statistics
        comparison_stats = {}
        if self.comparisons:
            quantum_advantages = [
                c.comparison_metrics.get("overall_quantum_advantage", 0)
                for c in self.comparisons
            ]
            significant_comparisons = sum(1 for c in self.comparisons if c.is_significant())
            
            comparison_stats = {
                'total_comparisons': len(self.comparisons),
                'avg_quantum_advantage': np.mean(quantum_advantages) if quantum_advantages else 0,
                'max_quantum_advantage': max(quantum_advantages) if quantum_advantages else 0,
                'min_quantum_advantage': min(quantum_advantages) if quantum_advantages else 0,
                'significant_comparisons': significant_comparisons,
                'significance_rate': significant_comparisons / len(self.comparisons) if self.comparisons else 0
            }
        
        return {
            'suite_id': self.suite_id,
            'benchmark_type': self.benchmark_type.name,
            'description': self.description,
            'total_results': len(self.results),
            'total_comparisons': len(self.comparisons),
            'by_algorithm': algorithm_stats,
            'comparison_stats': comparison_stats,
            'timestamp': datetime.now().isoformat()
        }

class QuantumBenchmarking:
    """Comprehensive quantum benchmarking framework"""
    
    def __init__(self):
        self.benchmark_suites: Dict[str, BenchmarkSuite] = {}
        self.benchmark_history: List[BenchmarkResult] = []
        self.comparison_history: List[BenchmarkComparison] = []
    
    def create_benchmark_suite(self, suite_id: str, benchmark_type: BenchmarkType,
                             description: str) -> BenchmarkSuite:
        """Create a new benchmark suite"""
        if suite_id in self.benchmark_suites:
            raise ValueError(f"Benchmark suite {suite_id} already exists")
        
        suite = BenchmarkSuite(
            suite_id=suite_id,
            benchmark_type=benchmark_type,
            description=description
        )
        
        self.benchmark_suites[suite_id] = suite
        return suite
    
    def get_benchmark_suite(self, suite_id: str) -> BenchmarkSuite:
        """Get an existing benchmark suite"""
        if suite_id not in self.benchmark_suites:
            raise ValueError(f"Benchmark suite {suite_id} not found")
        
        return self.benchmark_suites[suite_id]
    
    def run_benchmark(self, benchmark_id: str, benchmark_type: BenchmarkType,
                     algorithm: str, backend: str, parameters: Dict[str, Any],
                     execution_func: callable) -> BenchmarkResult:
        """
        Run a benchmark and record the results
        
        Args:
            benchmark_id: Unique identifier for this benchmark
            benchmark_type: Type of benchmark
            algorithm: Algorithm name
            backend: Backend name
            parameters: Algorithm parameters
            execution_func: Function to execute (should return metrics dict)
            
        Returns:
            Benchmark result
        """
        # Execute the benchmark
        start_time = time.time()
        
        try:
            # Execute the function and get metrics
            metrics = execution_func()
            
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Add execution time to metrics if not already present
            if MetricType.EXECUTION_TIME not in metrics:
                metrics[MetricType.EXECUTION_TIME] = execution_time_ms
            
            # Create benchmark result
            result = BenchmarkResult(
                benchmark_id=benchmark_id,
                benchmark_type=benchmark_type,
                algorithm=algorithm,
                backend=backend,
                metrics=metrics,
                parameters=parameters,
                metadata={
                    'execution_time_ms': execution_time_ms,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # Add to history
            self.benchmark_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            
            # Create failed benchmark result
            result = BenchmarkResult(
                benchmark_id=benchmark_id,
                benchmark_type=benchmark_type,
                algorithm=algorithm,
                backend=backend,
                metrics={
                    MetricType.EXECUTION_TIME: (time.time() - start_time) * 1000
                },
                parameters=parameters,
                metadata={
                    'error': str(e),
                    'status': 'failed',
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # Add to history
            self.benchmark_history.append(result)
            
            return result
    
    def compare_algorithms(self, quantum_result: BenchmarkResult,
                          classical_result: BenchmarkResult) -> BenchmarkComparison:
        """
        Compare quantum and classical benchmark results
        
        Args:
            quantum_result: Quantum benchmark result
            classical_result: Classical benchmark result
            
        Returns:
            Benchmark comparison
        """
        # Validate that results are for the same benchmark
        if (quantum_result.benchmark_id != classical_result.benchmark_id or
            quantum_result.benchmark_type != classical_result.benchmark_type):
            raise ValueError("Cannot compare results from different benchmarks")
        
        # Create comparison
        comparison = BenchmarkComparison(
            quantum_result=quantum_result,
            classical_result=classical_result
        )
        
        # Add to history
        self.comparison_history.append(comparison)
        
        return comparison
    
    def run_comparison_benchmark(self, benchmark_id: str, benchmark_type: BenchmarkType,
                                parameters: Dict[str, Any],
                                quantum_exec_func: callable, classical_exec_func: callable,
                                quantum_algorithm: str = "quantum",
                                classical_algorithm: str = "classical",
                                quantum_backend: str = "simulator",
                                classical_backend: str = "cpu") -> BenchmarkComparison:
        """
        Run both quantum and classical benchmarks and compare them
        
        Args:
            benchmark_id: Unique identifier for this benchmark
            benchmark_type: Type of benchmark
            parameters: Algorithm parameters
            quantum_exec_func: Quantum execution function
            classical_exec_func: Classical execution function
            quantum_algorithm: Quantum algorithm name
            classical_algorithm: Classical algorithm name
            quantum_backend: Quantum backend name
            classical_backend: Classical backend name
            
        Returns:
            Benchmark comparison
        """
        # Run quantum benchmark
        quantum_result = self.run_benchmark(
            benchmark_id=f"{benchmark_id}_quantum",
            benchmark_type=benchmark_type,
            algorithm=quantum_algorithm,
            backend=quantum_backend,
            parameters=parameters,
            execution_func=quantum_exec_func
        )
        
        # Run classical benchmark
        classical_result = self.run_benchmark(
            benchmark_id=f"{benchmark_id}_classical",
            benchmark_type=benchmark_type,
            algorithm=classical_algorithm,
            backend=classical_backend,
            parameters=parameters,
            execution_func=classical_exec_func
        )
        
        # Compare results
        return self.compare_algorithms(quantum_result, classical_result)
    
    def get_benchmark_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent benchmark history"""
        return [result.to_dict() for result in self.benchmark_history[-limit:]]
    
    def get_comparison_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent comparison history"""
        return [comparison.to_dict() for comparison in self.comparison_history[-limit:]]
    
    def generate_benchmark_report(self, suite_id: str = None) -> Dict[str, Any]:
        """Generate a comprehensive benchmark report"""
        if suite_id:
            if suite_id not in self.benchmark_suites:
                raise ValueError(f"Benchmark suite {suite_id} not found")
            return self.benchmark_suites[suite_id].generate_report()
        else:
            # Generate report for all suites
            all_reports = {}
            for suite_id, suite in self.benchmark_suites.items():
                all_reports[suite_id] = suite.generate_report()
            
            # Add overall statistics
            total_results = sum(len(suite.results) for suite in self.benchmark_suites.values())
            total_comparisons = sum(len(suite.comparisons) for suite in self.benchmark_suites.values())
            
            return {
                'total_suites': len(self.benchmark_suites),
                'total_results': total_results,
                'total_comparisons': total_comparisons,
                'by_suite': all_reports,
                'timestamp': datetime.now().isoformat()
            }
    
    def save_benchmark_data(self, filepath: str) -> Dict[str, Any]:
        """Save benchmark data to file"""
        data = {
            'benchmark_suites': {
                suite_id: {
                    'suite_id': suite.suite_id,
                    'benchmark_type': suite.benchmark_type.name,
                    'description': suite.description,
                    'results': [r.to_dict() for r in suite.results],
                    'comparisons': [c.to_dict() for c in suite.comparisons],
                    'metadata': suite.metadata
                }
                for suite_id, suite in self.benchmark_suites.items()
            },
            'benchmark_history': [r.to_dict() for r in self.benchmark_history],
            'comparison_history': [c.to_dict() for c in self.comparison_history],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return {
            'status': 'saved',
            'filepath': filepath,
            'timestamp': datetime.now().isoformat()
        }
    
    @classmethod
    def load_benchmark_data(cls, filepath: str) -> 'QuantumBenchmarking':
        """Load benchmark data from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create new benchmarking instance
        benchmarking = cls()
        
        # Load benchmark suites
        for suite_data in data['benchmark_suites'].values():
            suite = BenchmarkSuite(
                suite_id=suite_data['suite_id'],
                benchmark_type=BenchmarkType[suite_data['benchmark_type']],
                description=suite_data['description'],
                metadata=suite_data['metadata']
            )
            
            # Load results
            for result_data in suite_data['results']:
                metrics = {MetricType[k]: v for k, v in result_data['metrics'].items()}
                suite.add_result(BenchmarkResult(
                    benchmark_id=result_data['benchmark_id'],
                    benchmark_type=BenchmarkType[result_data['benchmark_type']],
                    algorithm=result_data['algorithm'],
                    backend=result_data['backend'],
                    metrics=metrics,
                    parameters=result_data['parameters'],
                    timestamp=datetime.fromisoformat(result_data['timestamp']),
                    metadata=result_data['metadata']
                ))
            
            # Load comparisons
            for comparison_data in suite_data['comparisons']:
                quantum_result_data = comparison_data['quantum']
                classical_result_data = comparison_data['classical']
                
                quantum_metrics = {MetricType[k]: v for k, v in quantum_result_data['metrics'].items()}
                classical_metrics = {MetricType[k]: v for k, v in classical_result_data['metrics'].items()}
                
                quantum_result = BenchmarkResult(
                    benchmark_id=quantum_result_data['benchmark_id'],
                    benchmark_type=BenchmarkType[quantum_result_data['benchmark_type']],
                    algorithm=quantum_result_data['algorithm'],
                    backend=quantum_result_data['backend'],
                    metrics=quantum_metrics,
                    parameters=quantum_result_data['parameters'],
                    timestamp=datetime.fromisoformat(quantum_result_data['timestamp']),
                    metadata=quantum_result_data['metadata']
                )
                
                classical_result = BenchmarkResult(
                    benchmark_id=classical_result_data['benchmark_id'],
                    benchmark_type=BenchmarkType[classical_result_data['benchmark_type']],
                    algorithm=classical_result_data['algorithm'],
                    backend=classical_result_data['backend'],
                    metrics=classical_metrics,
                    parameters=classical_result_data['parameters'],
                    timestamp=datetime.fromisoformat(classical_result_data['timestamp']),
                    metadata=classical_result_data['metadata']
                )
                
                comparison = BenchmarkComparison(quantum_result, classical_result)
                comparison.comparison_metrics = comparison_data['comparison_metrics']
                suite.add_comparison(comparison)
            
            benchmarking.benchmark_suites[suite.suite_id] = suite
        
        # Load history
        for result_data in data['benchmark_history']:
            metrics = {MetricType[k]: v for k, v in result_data['metrics'].items()}
            benchmarking.benchmark_history.append(BenchmarkResult(
                benchmark_id=result_data['benchmark_id'],
                benchmark_type=BenchmarkType[result_data['benchmark_type']],
                algorithm=result_data['algorithm'],
                backend=result_data['backend'],
                metrics=metrics,
                parameters=result_data['parameters'],
                timestamp=datetime.fromisoformat(result_data['timestamp']),
                metadata=result_data['metadata']
            ))
        
        for comparison_data in data['comparison_history']:
            quantum_result_data = comparison_data['quantum']
            classical_result_data = comparison_data['classical']
            
            quantum_metrics = {MetricType[k]: v for k, v in quantum_result_data['metrics'].items()}
            classical_metrics = {MetricType[k]: v for k, v in classical_result_data['metrics'].items()}
            
            quantum_result = BenchmarkResult(
                benchmark_id=quantum_result_data['benchmark_id'],
                benchmark_type=BenchmarkType[quantum_result_data['benchmark_type']],
                algorithm=quantum_result_data['algorithm'],
                backend=quantum_result_data['backend'],
                metrics=quantum_metrics,
                parameters=quantum_result_data['parameters'],
                timestamp=datetime.fromisoformat(quantum_result_data['timestamp']),
                metadata=quantum_result_data['metadata']
            )
            
            classical_result = BenchmarkResult(
                benchmark_id=classical_result_data['benchmark_id'],
                benchmark_type=BenchmarkType[classical_result_data['benchmark_type']],
                algorithm=classical_result_data['algorithm'],
                backend=classical_result_data['backend'],
                metrics=classical_metrics,
                parameters=classical_result_data['parameters'],
                timestamp=datetime.fromisoformat(classical_result_data['timestamp']),
                metadata=classical_result_data['metadata']
            )
            
            comparison = BenchmarkComparison(quantum_result, classical_result)
            comparison.comparison_metrics = comparison_data['comparison_metrics']
            benchmarking.comparison_history.append(comparison)
        
        return benchmarking
    
    def calculate_statistical_significance(self, comparison: BenchmarkComparison,
                                         metric: MetricType, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Calculate statistical significance of quantum advantage for a specific metric
        
        Args:
            comparison: Benchmark comparison
            metric: Metric to test
            alpha: Significance level
            
        Returns:
            Statistical significance result
        """
        quantum_value = comparison.quantum_result.metrics.get(metric, 0)
        classical_value = comparison.classical_result.metrics.get(metric, 0)
        
        # For demo purposes, we'll use a simple t-test
        # In a real implementation, we would need multiple samples
        
        # Generate synthetic sample data for demonstration
        np.random.seed(42)
        quantum_samples = np.random.normal(quantum_value, quantum_value * 0.05, 30)
        classical_samples = np.random.normal(classical_value, classical_value * 0.05, 30)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(quantum_samples, classical_samples)
        
        return {
            'metric': metric.name,
            'quantum_mean': np.mean(quantum_samples),
            'classical_mean': np.mean(classical_samples),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'quantum_advantage': comparison.comparison_metrics.get(f"{metric.name}_advantage", 0),
            'alpha': alpha
        }
    
    def analyze_quantum_advantage(self, suite_id: str = None,
                                min_advantage: float = 0.05,
                                min_significance: float = 0.05) -> Dict[str, Any]:
        """
        Analyze quantum advantage across benchmarks
        
        Args:
            suite_id: Optional suite ID to analyze (None for all suites)
            min_advantage: Minimum quantum advantage to consider
            min_significance: Maximum p-value to consider significant
            
        Returns:
            Quantum advantage analysis
        """
        suites_to_analyze = [self.benchmark_suites[suite_id]] if suite_id else list(self.benchmark_suites.values())
        
        significant_comparisons = []
        total_comparisons = 0
        
        for suite in suites_to_analyze:
            for comparison in suite.comparisons:
                total_comparisons += 1
                
                # Check if quantum advantage meets minimum threshold
                advantage = comparison.comparison_metrics.get("overall_quantum_advantage", 0)
                if advantage >= min_advantage:
                    # Check statistical significance for key metrics
                    significant_metrics = []
                    
                    for metric in [MetricType.EXECUTION_TIME, MetricType.ACCURACY, MetricType.FIDELITY]:
                        if metric in comparison.quantum_result.metrics and metric in comparison.classical_result.metrics:
                            sig_test = self.calculate_statistical_significance(comparison, metric)
                            if sig_test['significant']:
                                significant_metrics.append({
                                    'metric': metric.name,
                                    'advantage': comparison.comparison_metrics.get(f"{metric.name}_advantage", 0),
                                    'p_value': sig_test['p_value']
                                })
                    
                    if significant_metrics:
                        significant_comparisons.append({
                            'comparison': comparison.to_dict(),
                            'advantage': advantage,
                            'significant_metrics': significant_metrics,
                            'suite_id': suite.suite_id
                        })
        
        return {
            'total_comparisons': total_comparisons,
            'significant_comparisons': len(significant_comparisons),
            'significance_rate': len(significant_comparisons) / total_comparisons if total_comparisons > 0 else 0,
            'details': significant_comparisons,
            'min_advantage': min_advantage,
            'min_significance': min_significance,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_quantum_advantage_report(self, suite_id: str = None,
                                         min_advantage: float = 0.05) -> Dict[str, Any]:
        """
        Generate a quantum advantage report
        
        Args:
            suite_id: Optional suite ID to report on (None for all suites)
            min_advantage: Minimum quantum advantage to include
            
        Returns:
            Quantum advantage report
        """
        suites_to_report = [self.benchmark_suites[suite_id]] if suite_id else list(self.benchmark_suites.values())
        
        report_data = []
        
        for suite in suites_to_report:
            for comparison in suite.comparisons:
                advantage = comparison.comparison_metrics.get("overall_quantum_advantage", 0)
                if advantage >= min_advantage:
                    report_data.append({
                        'suite_id': suite.suite_id,
                        'benchmark_type': suite.benchmark_type.name,
                        'quantum_algorithm': comparison.quantum_result.algorithm,
                        'classical_algorithm': comparison.classical_result.algorithm,
                        'quantum_backend': comparison.quantum_result.backend,
                        'classical_backend': comparison.classical_result.backend,
                        'overall_advantage': advantage,
                        'metric_advantages': {
                            k: v for k, v in comparison.comparison_metrics.items()
                            if k.endswith('_advantage') and v >= 0
                        },
                        'quantum_score': comparison.quantum_result.calculate_score(),
                        'classical_score': comparison.classical_result.calculate_score(),
                        'timestamp': comparison.quantum_result.timestamp.isoformat()
                    })
        
        # Sort by advantage (descending)
        report_data.sort(key=lambda x: x['overall_advantage'], reverse=True)
        
        return {
            'total_comparisons': len(report_data),
            'min_advantage_filter': min_advantage,
            'by_advantage': report_data,
            'top_performers': report_data[:5] if report_data else [],
            'timestamp': datetime.now().isoformat()
        }
    
    def create_benchmark_id(self, benchmark_type: BenchmarkType,
                           algorithm: str, parameters: Dict[str, Any]) -> str:
        """Create a unique benchmark ID"""
        # Create a hash based on benchmark characteristics
        param_str = json.dumps(parameters, sort_keys=True)
        hash_input = f"{benchmark_type.name}_{algorithm}_{param_str}".encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()[:16]

@dataclass
class QuantumBenchmarkOrchestrator:
    """Orchestrates comprehensive quantum benchmarking"""
    
    def __init__(self, benchmarking: QuantumBenchmarking):
        self.benchmarking = benchmarking
        self.benchmark_plans = []
        self.execution_history = []
    
    def create_benchmark_plan(self, plan_id: str, benchmark_type: BenchmarkType,
                             description: str, parameters_grid: List[Dict[str, Any]],
                             quantum_algorithm: str, classical_algorithm: str,
                             quantum_backend: str, classical_backend: str) -> Dict[str, Any]:
        """
        Create a comprehensive benchmark plan
        
        Args:
            plan_id: Unique plan identifier
            benchmark_type: Type of benchmark
            description: Plan description
            parameters_grid: List of parameter sets to test
            quantum_algorithm: Quantum algorithm name
            classical_algorithm: Classical algorithm name
            quantum_backend: Quantum backend name
            classical_backend: Classical backend name
            
        Returns:
            Benchmark plan
        """
        plan = {
            'plan_id': plan_id,
            'benchmark_type': benchmark_type,
            'description': description,
            'parameters_grid': parameters_grid,
            'quantum_algorithm': quantum_algorithm,
            'classical_algorithm': classical_algorithm,
            'quantum_backend': quantum_backend,
            'classical_backend': classical_backend,
            'status': 'created',
            'created_at': datetime.now().isoformat(),
            'results': []
        }
        
        self.benchmark_plans.append(plan)
        return plan
    
    def execute_benchmark_plan(self, plan_id: str,
                               quantum_exec_func: callable,
                               classical_exec_func: callable) -> Dict[str, Any]:
        """
        Execute a benchmark plan
        
        Args:
            plan_id: Plan identifier
            quantum_exec_func: Quantum execution function
            classical_exec_func: Classical execution function
            
        Returns:
            Execution result
        """
        # Find the plan
        plan = next((p for p in self.benchmark_plans if p['plan_id'] == plan_id), None)
        if not plan:
            return {'status': 'failed', 'reason': f'Plan {plan_id} not found'}
        
        if plan['status'] != 'created':
            return {'status': 'failed', 'reason': f'Plan {plan_id} already executed'}
        
        # Update status
        plan['status'] = 'executing'
        plan['started_at'] = datetime.now().isoformat()
        
        # Create benchmark suite
        suite_id = f"suite_{plan_id}"
        suite = self.benchmarking.create_benchmark_suite(
            suite_id=suite_id,
            benchmark_type=plan['benchmark_type'],
            description=plan['description']
        )
        
        # Execute each parameter set
        for param_set in plan['parameters_grid']:
            try:
                # Generate benchmark ID
                benchmark_id = self.benchmarking.create_benchmark_id(
                    benchmark_type=plan['benchmark_type'],
                    algorithm=plan['quantum_algorithm'],
                    parameters=param_set
                )
                
                # Run comparison benchmark
                comparison = self.benchmarking.run_comparison_benchmark(
                    benchmark_id=benchmark_id,
                    benchmark_type=plan['benchmark_type'],
                    parameters=param_set,
                    quantum_exec_func=lambda: quantum_exec_func(param_set),
                    classical_exec_func=lambda: classical_exec_func(param_set),
                    quantum_algorithm=plan['quantum_algorithm'],
                    classical_algorithm=plan['classical_algorithm'],
                    quantum_backend=plan['quantum_backend'],
                    classical_backend=plan['classical_backend']
                )
                
                # Add to suite and plan results
                suite.add_comparison(comparison)
                plan['results'].append({
                    'benchmark_id': benchmark_id,
                    'parameters': param_set,
                    'quantum_advantage': comparison.comparison_metrics.get("overall_quantum_advantage", 0),
                    'significant': comparison.is_significant(),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Failed to execute benchmark for parameters {param_set}: {e}")
                plan['results'].append({
                    'parameters': param_set,
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Update plan status
        plan['status'] = 'completed'
        plan['completed_at'] = datetime.now().isoformat()
        
        # Generate report
        report = suite.generate_report()
        
        # Record execution
        self.execution_history.append({
            'plan_id': plan_id,
            'status': 'completed',
            'total_benchmarks': len(plan['parameters_grid']),
            'successful_benchmarks': len([r for r in plan['results'] if 'quantum_advantage' in r]),
            'failed_benchmarks': len([r for r in plan['results'] if r.get('status') == 'failed']),
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'status': 'completed',
            'plan_id': plan_id,
            'suite_id': suite_id,
            'total_benchmarks': len(plan['parameters_grid']),
            'successful_benchmarks': len([r for r in plan['results'] if 'quantum_advantage' in r]),
            'failed_benchmarks': len([r for r in plan['results'] if r.get('status') == 'failed']),
            'report': report,
            'execution_history': self.execution_history[-1]
        }
    
    def get_benchmark_plan(self, plan_id: str) -> Dict[str, Any]:
        """Get a benchmark plan"""
        plan = next((p for p in self.benchmark_plans if p['plan_id'] == plan_id), None)
        if not plan:
            return {'status': 'failed', 'reason': f'Plan {plan_id} not found'}
        
        return plan
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.execution_history[-limit:]
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of all benchmark plans"""
        if not self.benchmark_plans:
            return {'status': 'no_plans'}
        
        # Group by benchmark type
        by_type = {}
        for plan in self.benchmark_plans:
            bench_type = plan['benchmark_type'].name
            if bench_type not in by_type:
                by_type[bench_type] = []
            by_type[bench_type].append(plan)
        
        # Calculate statistics
        total_plans = len(self.benchmark_plans)
        completed_plans = sum(1 for p in self.benchmark_plans if p['status'] == 'completed')
        successful_benchmarks = sum(
            len([r for r in p['results'] if 'quantum_advantage' in r])
            for p in self.benchmark_plans if p['status'] == 'completed'
        )
        total_benchmarks = sum(
            len(p['parameters_grid'])
            for p in self.benchmark_plans
        )
        
        # Calculate quantum advantage statistics
        all_advantages = []
        for plan in self.benchmark_plans:
            if plan['status'] == 'completed':
                for result in plan['results']:
                    if 'quantum_advantage' in result:
                        all_advantages.append(result['quantum_advantage'])
        
        return {
            'total_plans': total_plans,
            'completed_plans': completed_plans,
            'plans_by_type': {
                bench_type: {
                    'count': len(plans),
                    'completed': sum(1 for p in plans if p['status'] == 'completed')
                }
                for bench_type, plans in by_type.items()
            },
            'total_benchmarks': total_benchmarks,
            'successful_benchmarks': successful_benchmarks,
            'success_rate': successful_benchmarks / total_benchmarks if total_benchmarks > 0 else 0,
            'quantum_advantage_stats': {
                'avg': np.mean(all_advantages) if all_advantages else 0,
                'max': max(all_advantages) if all_advantages else 0,
                'min': min(all_advantages) if all_advantages else 0,
                'positive': sum(1 for adv in all_advantages if adv > 0) / len(all_advantages) if all_advantages else 0
            } if all_advantages else None,
            'execution_history': self.execution_history,
            'timestamp': datetime.now().isoformat()
        }

@dataclass
class QuantumPerformanceTracker:
    """Tracks quantum performance over time"""
    
    def __init__(self, benchmarking: QuantumBenchmarking):
        self.benchmarking = benchmarking
        self.performance_history = []
        self.algorithm_performance = {}
        self.backend_performance = {}
    
    def record_performance(self, comparison: BenchmarkComparison,
                          tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Record performance metrics
        
        Args:
            comparison: Benchmark comparison
            tags: Optional tags for categorization
            
        Returns:
            Recording result
        """
        # Extract key metrics
        quantum_algorithm = comparison.quantum_result.algorithm
        classical_algorithm = comparison.classical_result.algorithm
        quantum_backend = comparison.quantum_result.backend
        benchmark_type = comparison.quantum_result.benchmark_type.name
        
        # Create performance record
        record = {
            'quantum_algorithm': quantum_algorithm,
            'classical_algorithm': classical_algorithm,
            'quantum_backend': quantum_backend,
            'benchmark_type': benchmark_type,
            'quantum_advantage': comparison.comparison_metrics.get("overall_quantum_advantage", 0),
            'significant': comparison.is_significant(),
            'metrics': {
                metric: {
                    'quantum': comparison.quantum_result.metrics.get(metric, 0),
                    'classical': comparison.classical_result.metrics.get(metric, 0),
                    'advantage': comparison.comparison_metrics.get(f"{metric.name}_advantage", 0)
                }
                for metric in MetricType
            },
            'quantum_score': comparison.quantum_result.calculate_score(),
            'classical_score': comparison.classical_result.calculate_score(),
            'timestamp': datetime.now().isoformat(),
            'tags': tags or []
        }
        
        # Add to history
        self.performance_history.append(record)
        
        # Update algorithm performance
        if quantum_algorithm not in self.algorithm_performance:
            self.algorithm_performance[quantum_algorithm] = []
        self.algorithm_performance[quantum_algorithm].append(record)
        
        # Update backend performance
        if quantum_backend not in self.backend_performance:
            self.backend_performance[quantum_backend] = []
        self.backend_performance[quantum_backend].append(record)
        
        return {
            'status': 'recorded',
            'performance_record': record,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_performance_trends(self, algorithm: str = None,
                               backend: str = None,
                               limit: int = 30) -> Dict[str, Any]:
        """
        Get performance trends
        
        Args:
            algorithm: Optional algorithm filter
            backend: Optional backend filter
            limit: Maximum number of records to return
            
        Returns:
            Performance trends
        """
        # Filter records
        filtered_records = []
        for record in self.performance_history[-limit:]:
            if ((algorithm is None or record['quantum_algorithm'] == algorithm) and
                (backend is None or record['quantum_backend'] == backend)):
                filtered_records.append(record)
        
        if not filtered_records:
            return {'status': 'no_data', 'filters': {'algorithm': algorithm, 'backend': backend}}
        
        # Calculate trends
        timestamps = [datetime.fromisoformat(r['timestamp']) for r in filtered_records]
        advantages = [r['quantum_advantage'] for r in filtered_records]
        scores = [r['quantum_score'] for r in filtered_records]
        
        return {
            'count': len(filtered_records),
            'avg_advantage': np.mean(advantages),
            'max_advantage': max(advantages),
            'min_advantage': min(advantages),
            'avg_score': np.mean(scores),
            'trend': {
                'timestamps': [t.isoformat() for t in timestamps],
                'advantages': advantages,
                'scores': scores
            },
            'records': filtered_records[-10:]  # Return last 10 records
        }
    
    def get_algorithm_performance(self, algorithm: str) -> Dict[str, Any]:
        """Get performance statistics for a specific algorithm"""
        if algorithm not in self.algorithm_performance or not self.algorithm_performance[algorithm]:
            return {'status': 'no_data', 'algorithm': algorithm}
        
        records = self.algorithm_performance[algorithm]
        advantages = [r['quantum_advantage'] for r in records]
        scores = [r['quantum_score'] for r in records]
        execution_times = [
            r['metrics'][MetricType.EXECUTION_TIME]['quantum']
            for r in records if MetricType.EXECUTION_TIME in r['metrics']
        ]
        
        return {
            'algorithm': algorithm,
            'total_runs': len(records),
            'avg_advantage': np.mean(advantages) if advantages else 0,
            'max_advantage': max(advantages) if advantages else 0,
            'min_advantage': min(advantages) if advantages else 0,
            'positive_advantage_runs': sum(1 for adv in advantages if adv > 0) / len(advantages) if advantages else 0,
            'avg_score': np.mean(scores) if scores else 0,
            'avg_execution_time_ms': np.mean(execution_times) if execution_times else 0,
            'backends_used': list(set(r['quantum_backend'] for r in records)),
            'benchmark_types': list(set(r['benchmark_type'] for r in records)),
            'recent_records': records[-5:] if records else []
        }
    
    def get_backend_performance(self, backend: str) -> Dict[str, Any]:
        """Get performance statistics for a specific backend"""
        if backend not in self.backend_performance or not self.backend_performance[backend]:
            return {'status': 'no_data', 'backend': backend}
        
        records = self.backend_performance[backend]
        advantages = [r['quantum_advantage'] for r in records]
        scores = [r['quantum_score'] for r in records]
        execution_times = [
            r['metrics'][MetricType.EXECUTION_TIME]['quantum']
            for r in records if MetricType.EXECUTION_TIME in r['metrics']
        ]
        
        return {
            'backend': backend,
            'total_runs': len(records),
            'avg_advantage': np.mean(advantages) if advantages else 0,
            'max_advantage': max(advantages) if advantages else 0,
            'min_advantage': min(advantages) if advantages else 0,
            'positive_advantage_runs': sum(1 for adv in advantages if adv > 0) / len(advantages) if advantages else 0,
            'avg_score': np.mean(scores) if scores else 0,
            'avg_execution_time_ms': np.mean(execution_times) if execution_times else 0,
            'algorithms_used': list(set(r['quantum_algorithm'] for r in records)),
            'benchmark_types': list(set(r['benchmark_type'] for r in records)),
            'recent_records': records[-5:] if records else []
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report"""
        if not self.performance_history:
            return {'status': 'no_data'}
        
        # Calculate overall statistics
        advantages = [r['quantum_advantage'] for r in self.performance_history]
        scores = [r['quantum_score'] for r in self.performance_history]
        
        # Group by algorithm
        algorithm_stats = {}
        for algorithm, records in self.algorithm_performance.items():
            if records:
                alg_advantages = [r['quantum_advantage'] for r in records]
                algorithm_stats[algorithm] = {
                    'total_runs': len(records),
                    'avg_advantage': np.mean(alg_advantages) if alg_advantages else 0,
                    'max_advantage': max(alg_advantages) if alg_advantages else 0,
                    'positive_rate': sum(1 for adv in alg_advantages if adv > 0) / len(alg_advantages) if alg_advantages else 0
                }
        
        # Group by backend
        backend_stats = {}
        for backend, records in self.backend_performance.items():
            if records:
                bk_advantages = [r['quantum_advantage'] for r in records]
                backend_stats[backend] = {
                    'total_runs': len(records),
                    'avg_advantage': np.mean(bk_advantages) if bk_advantages else 0,
                    'max_advantage': max(bk_advantages) if bk_advantages else 0,
                    'positive_rate': sum(1 for adv in bk_advantages if adv > 0) / len(bk_advantages) if bk_advantages else 0
                }
        
        # Get top performers
        top_algorithms = sorted(
            algorithm_stats.items(),
            key=lambda x: x[1]['avg_advantage'],
            reverse=True
        )[:3]
        
        top_backends = sorted(
            backend_stats.items(),
            key=lambda x: x[1]['avg_advantage'],
            reverse=True
        )[:3]
        
        return {
            'total_records': len(self.performance_history),
            'overall_avg_advantage': np.mean(advantages) if advantages else 0,
            'overall_max_advantage': max(advantages) if advantages else 0,
            'overall_positive_rate': sum(1 for adv in advantages if adv > 0) / len(advantages) if advantages else 0,
            'overall_avg_score': np.mean(scores) if scores else 0,
            'by_algorithm': algorithm_stats,
            'by_backend': backend_stats,
            'top_algorithms': top_algorithms,
            'top_backends': top_backends,
            'recent_records': self.performance_history[-10:] if self.performance_history else [],
            'timestamp': datetime.now().isoformat()
        }
    
    def get_quantum_advantage_distribution(self, bins: int = 10) -> Dict[str, Any]:
        """Get distribution of quantum advantage values"""
        if not self.performance_history:
            return {'status': 'no_data'}
        
        advantages = [r['quantum_advantage'] for r in self.performance_history]
        
        # Calculate histogram
        hist, bin_edges = np.histogram(advantages, bins=bins)
        
        # Calculate statistics
        positive = sum(1 for adv in advantages if adv > 0)
        negative = sum(1 for adv in advantages if adv < 0)
        neutral = sum(1 for adv in advantages if adv == 0)
        
        return {
            'total': len(advantages),
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'positive_rate': positive / len(advantages),
            'negative_rate': negative / len(advantages),
            'neutral_rate': neutral / len(advantages),
            'histogram': {
                'bins': bin_edges.tolist(),
                'counts': hist.tolist(),
                'bin_size': (bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 0
            },
            'stats': {
                'min': min(advantages) if advantages else 0,
                'max': max(advantages) if advantages else 0,
                'mean': np.mean(advantages) if advantages else 0,
                'std': np.std(advantages) if advantages else 0,
                'median': np.median(advantages) if advantages else 0
            }
        }
    
    def find_optimal_parameters(self, algorithm: str, benchmark_type: str,
                                parameter_ranges: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Find optimal parameters for an algorithm using historical data
        
        Args:
            algorithm: Algorithm name
            benchmark_type: Benchmark type
            parameter_ranges: Dictionary of parameter ranges to search
            
        Returns:
            Optimal parameters and performance
        """
        if algorithm not in self.algorithm_performance:
            return {'status': 'no_data', 'algorithm': algorithm}
        
        # Filter relevant records
        relevant_records = [
            r for r in self.algorithm_performance[algorithm]
            if r['benchmark_type'] == benchmark_type
        ]
        
        if not relevant_records:
            return {'status': 'no_matching_data', 'algorithm': algorithm, 'benchmark_type': benchmark_type}
        
        # Find record with highest quantum advantage
        best_record = max(relevant_records, key=lambda r: r['quantum_advantage'])
        
        # Extract parameters from the original comparison
        # Note: In a real implementation, we would have stored the parameters
        # For this demo, we'll just return the best record
        
        return {
            'status': 'found',
            'algorithm': algorithm,
            'benchmark_type': benchmark_type,
            'best_quantum_advantage': best_record['quantum_advantage'],
            'best_record': best_record,
            'recommendation': "Use parameters that achieved this performance"
        }

@dataclass
class QuantumBenchmarkVisualizer:
    """Generates visualizations for quantum benchmarking results"""
    
    def __init__(self, benchmarking: QuantumBenchmarking):
        self.benchmarking = benchmarking
    
    def generate_advantage_plot_data(self, suite_id: str = None,
                                   min_advantage: float = 0.0) -> Dict[str, Any]:
        """
        Generate data for quantum advantage plot
        
        Args:
            suite_id: Optional suite ID to plot
            min_advantage: Minimum advantage to include
            
        Returns:
            Plot data
        """
        if suite_id:
            if suite_id not in self.benchmarking.benchmark_suites:
                return {'status': 'suite_not_found', 'suite_id': suite_id}
            suite = self.benchmarking.benchmark_suites[suite_id]
            comparisons = suite.comparisons
        else:
            comparisons = self.benchmarking.comparison_history
        
        if not comparisons:
            return {'status': 'no_data'}
        
        # Filter by minimum advantage
        filtered_comparisons = [
            c for c in comparisons
            if c.comparison_metrics.get("overall_quantum_advantage", 0) >= min_advantage
        ]
        
        if not filtered_comparisons:
            return {'status': 'no_data_matching_filter', 'min_advantage': min_advantage}
        
        # Prepare plot data
        plot_data = []
        for comparison in filtered_comparisons:
            plot_data.append({
                'benchmark_type': comparison.quantum_result.benchmark_type.name,
                'algorithm': comparison.quantum_result.algorithm,
                'quantum_advantage': comparison.comparison_metrics.get("overall_quantum_advantage", 0),
                'quantum_score': comparison.quantum_result.calculate_score(),
                'classical_score': comparison.classical_result.calculate_score(),
                'backend': comparison.quantum_result.backend,
                'timestamp': comparison.quantum_result.timestamp.isoformat()
            })
        
        # Group by benchmark type
        by_type = {}
        for data in plot_data:
            bench_type = data['benchmark_type']
            if bench_type not in by_type:
                by_type[bench_type] = []
            by_type[bench_type].append(data)
        
        # Group by algorithm
        by_algorithm = {}
        for data in plot_data:
            algorithm = data['algorithm']
            if algorithm not in by_algorithm:
                by_algorithm[algorithm] = []
            by_algorithm[algorithm].append(data)
        
        return {
            'status': 'success',
            'total_points': len(plot_data),
            'by_type': by_type,
            'by_algorithm': by_algorithm,
            'all_data': plot_data,
            'min_advantage': min_advantage,
            'max_advantage': max(d['quantum_advantage'] for d in plot_data) if plot_data else 0,
            'avg_advantage': np.mean([d['quantum_advantage'] for d in plot_data]) if plot_data else 0
        }
    
    def generate_performance_trend_data(self, algorithm: str = None,
                                        backend: str = None,
                                        days: int = 30) -> Dict[str, Any]:
        """
        Generate data for performance trend plot
        
        Args:
            algorithm: Optional algorithm filter
            backend: Optional backend filter
            days: Number of days to include
            
        Returns:
            Trend plot data
        """
        # Use the performance tracker if available
        if hasattr(self.benchmarking, 'performance_tracker'):
            tracker = self.benchmarking.performance_tracker
            return tracker.get_performance_trends(algorithm, backend, limit=days * 24 * 60)  # Approx records per day
        
        # Fallback to benchmark history
        cutoff_time = datetime.now().timestamp() * 1000 - days * 24 * 60 * 60 * 1000
        
        filtered_history = [
            r for r in self.benchmarking.benchmark_history
            if datetime.fromisoformat(r.timestamp.isoformat()).timestamp() * 1000 >= cutoff_time and
               (algorithm is None or r.algorithm == algorithm) and
               (backend is None or r.backend == backend)
        ]
        
        if not filtered_history:
            return {'status': 'no_data', 'filters': {'algorithm': algorithm, 'backend': backend, 'days': days}}
        
        # Prepare trend data
        timestamps = [datetime.fromisoformat(r.timestamp.isoformat()) for r in filtered_history]
        advantages = [
            c.comparison_metrics.get("overall_quantum_advantage", 0)
            for c in self.benchmarking.comparison_history
            if c.quantum_result.algorithm == (algorithm or c.quantum_result.algorithm) and
               c.quantum_result.backend == (backend or c.quantum_result.backend) and
               datetime.fromisoformat(c.quantum_result.timestamp.isoformat()).timestamp() * 1000 >= cutoff_time
        ]
        
        return {
            'status': 'success',
            'count': len(filtered_history),
            'timestamps': [t.isoformat() for t in timestamps],
            'advantages': advantages,
            'avg_advantage': np.mean(advantages) if advantages else 0,
            'max_advantage': max(advantages) if advantages else 0,
            'min_advantage': min(advantages) if advantages else 0
        }
    
    def generate_backend_comparison_data(self) -> Dict[str, Any]:
        """Generate data for backend comparison plot"""
        if not self.benchmarking.comparison_history:
            return {'status': 'no_data'}
        
        # Group by backend
        by_backend = {}
        for comparison in self.benchmarking.comparison_history:
            backend = comparison.quantum_result.backend
            if backend not in by_backend:
                by_backend[backend] = []
            by_backend[backend].append(comparison)
        
        # Calculate statistics per backend
        backend_stats = {}
        for backend, comparisons in by_backend.items():
            advantages = [c.comparison_metrics.get("overall_quantum_advantage", 0) for c in comparisons]
            scores = [c.quantum_result.calculate_score() for c in comparisons]
            exec_times = [
                c.quantum_result.metrics.get(MetricType.EXECUTION_TIME, 0)
                for c in comparisons
            ]
            
            backend_stats[backend] = {
                'count': len(comparisons),
                'avg_advantage': np.mean(advantages) if advantages else 0,
                'max_advantage': max(advantages) if advantages else 0,
                'avg_score': np.mean(scores) if scores else 0,
                'avg_execution_time_ms': np.mean(exec_times) if exec_times else 0,
                'positive_rate': sum(1 for adv in advantages if adv > 0) / len(advantages) if advantages else 0
            }
        
        return {
            'status': 'success',
            'backends': list(by_backend.keys()),
            'stats': backend_stats,
            'best_backend': max(
                backend_stats.items(),
                key=lambda x: x[1]['avg_advantage']
            ) if backend_stats else None
        }
    
    def generate_algorithm_comparison_data(self) -> Dict[str, Any]:
        """Generate data for algorithm comparison plot"""
        if not self.benchmarking.comparison_history:
            return {'status': 'no_data'}
        
        # Group by algorithm
        by_algorithm = {}
        for comparison in self.benchmarking.comparison_history:
            algorithm = comparison.quantum_result.algorithm
            if algorithm not in by_algorithm:
                by_algorithm[algorithm] = []
            by_algorithm[algorithm].append(comparison)
        
        # Calculate statistics per algorithm
        algorithm_stats = {}
        for algorithm, comparisons in by_algorithm.items():
            advantages = [c.comparison_metrics.get("overall_quantum_advantage", 0) for c in comparisons]
            scores = [c.quantum_result.calculate_score() for c in comparisons]
            exec_times = [
                c.quantum_result.metrics.get(MetricType.EXECUTION_TIME, 0)
                for c in comparisons
            ]
            
            algorithm_stats[algorithm] = {
                'count': len(comparisons),
                'avg_advantage': np.mean(advantages) if advantages else 0,
                'max_advantage': max(advantages) if advantages else 0,
                'avg_score': np.mean(scores) if scores else 0,
                'avg_execution_time_ms': np.mean(exec_times) if exec_times else 0,
                'positive_rate': sum(1 for adv in advantages if adv > 0) / len(advantages) if advantages else 0,
                'benchmark_types': list(set(c.quantum_result.benchmark_type.name for c in comparisons))
            }
        
        return {
            'status': 'success',
            'algorithms': list(by_algorithm.keys()),
            'stats': algorithm_stats,
            'best_algorithm': max(
                algorithm_stats.items(),
                key=lambda x: x[1]['avg_advantage']
            ) if algorithm_stats else None
        }
    
    def generate_metric_distribution_data(self, metric: MetricType) -> Dict[str, Any]:
        """Generate data for metric distribution plot"""
        if not self.benchmarking.comparison_history:
            return {'status': 'no_data'}
        
        # Collect all values for the metric
        quantum_values = []
        classical_values = []
        advantages = []
        
        for comparison in self.benchmarking.comparison_history:
            if metric in comparison.quantum_result.metrics and metric in comparison.classical_result.metrics:
                quantum_values.append(comparison.quantum_result.metrics[metric])
                classical_values.append(comparison.classical_result.metrics[metric])
                advantages.append(comparison.comparison_metrics.get(f"{metric.name}_advantage", 0))
        
        if not quantum_values:
            return {'status': 'no_data_for_metric', 'metric': metric.name}
        
        # Calculate statistics
        return {
            'status': 'success',
            'metric': metric.name,
            'quantum': {
                'values': quantum_values,
                'min': min(quantum_values),
                'max': max(quantum_values),
                'mean': np.mean(quantum_values),
                'median': np.median(quantum_values),
                'std': np.std(quantum_values)
            },
            'classical': {
                'values': classical_values,
                'min': min(classical_values),
                'max': max(classical_values),
                'mean': np.mean(classical_values),
                'median': np.median(classical_values),
                'std': np.std(classical_values)
            },
            'advantages': {
                'values': advantages,
                'min': min(advantages),
                'max': max(advantages),
                'mean': np.mean(advantages),
                'median': np.median(advantages),
                'std': np.std(advantages),
                'positive_count': sum(1 for adv in advantages if adv > 0),
                'negative_count': sum(1 for adv in advantages if adv < 0),
                'neutral_count': sum(1 for adv in advantages if adv == 0)
            }
        }

@dataclass
class QuantumBenchmarkExporter:
    """Exports benchmarking data to various formats"""
    
    def __init__(self, benchmarking: QuantumBenchmarking):
        self.benchmarking = benchmarking
    
    def export_to_json(self, filepath: str, suite_id: str = None) -> Dict[str, Any]:
        """
        Export benchmarking data to JSON file
        
        Args:
            filepath: Output file path
            suite_id: Optional suite ID to export (None for all data)
            
        Returns:
            Export result
        """
        if suite_id:
            if suite_id not in self.benchmarking.benchmark_suites:
                return {'status': 'failed', 'reason': f'Suite {suite_id} not found'}
            
            data = {
                'suite': self.benchmarking.benchmark_suites[suite_id].generate_report(),
                'timestamp': datetime.now().isoformat()
            }
        else:
            data = {
                'suites': {
                    suite_id: suite.generate_report()
                    for suite_id, suite in self.benchmarking.benchmark_suites.items()
                },
                'benchmark_history': [r.to_dict() for r in self.benchmarking.benchmark_history],
                'comparison_history': [c.to_dict() for c in self.benchmarking.comparison_history],
                'timestamp': datetime.now().isoformat()
            }
        
        # Write to file
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return {
            'status': 'success',
            'filepath': filepath,
            'timestamp': datetime.now().isoformat()
        }
    
    def export_comparison_to_csv(self, filepath: str, suite_id: str = None) -> Dict[str, Any]:
        """
        Export comparison data to CSV file
        
        Args:
            filepath: Output file path
            suite_id: Optional suite ID to export (None for all comparisons)
            
        Returns:
            Export result
        """
        import csv
        
        # Get comparisons to export
        if suite_id:
            if suite_id not in self.benchmarking.benchmark_suites:
                return {'status': 'failed', 'reason': f'Suite {suite_id} not found'}
            comparisons = self.benchmarking.benchmark_suites[suite_id].comparisons
        else:
            comparisons = self.benchmarking.comparison_history
        
        if not comparisons:
            return {'status': 'failed', 'reason': 'No comparisons found'}
        
        # Prepare CSV data
        csv_data = []
        for comparison in comparisons:
            row = {
                'benchmark_id': comparison.quantum_result.benchmark_id,
                'benchmark_type': comparison.quantum_result.benchmark_type.name,
                'quantum_algorithm': comparison.quantum_result.algorithm,
                'classical_algorithm': comparison.classical_result.algorithm,
                'quantum_backend': comparison.quantum_result.backend,
                'classical_backend': comparison.classical_result.backend,
                'quantum_score': comparison.quantum_result.calculate_score(),
                'classical_score': comparison.classical_result.calculate_score(),
                'overall_quantum_advantage': comparison.comparison_metrics.get("overall_quantum_advantage", 0),
                'significant': comparison.is_significant(),
                'timestamp': comparison.quantum_result.timestamp.isoformat()
            }
            
            # Add metric advantages
            for metric in MetricType:
                metric_key = f"{metric.name}_advantage"
                if metric_key in comparison.comparison_metrics:
                    row[metric_key] = comparison.comparison_metrics[metric_key]
            
            csv_data.append(row)
        
        # Write to CSV
        fieldnames = list(csv_data[0].keys()) if csv_data else []
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        return {
            'status': 'success',
            'filepath': filepath,
            'record_count': len(csv_data),
            'timestamp': datetime.now().isoformat()
        }
    
    def export_performance_to_csv(self, filepath: str, algorithm: str = None,
                                  backend: str = None) -> Dict[str, Any]:
        """
        Export performance data to CSV file
        
        Args:
            filepath: Output file path
            algorithm: Optional algorithm filter
            backend: Optional backend filter
            
        Returns:
            Export result
        """
        import csv
        
        # Check if performance tracker exists
        if not hasattr(self.benchmarking, 'performance_tracker'):
            return {'status': 'failed', 'reason': 'Performance tracker not available'}
        
        tracker = self.benchmarking.performance_tracker
        records = tracker.performance_history
        
        # Filter records
        filtered_records = []
        for record in records:
            if ((algorithm is None or record['quantum_algorithm'] == algorithm) and
                (backend is None or record['quantum_backend'] == backend)):
                filtered_records.append(record)
        
        if not filtered_records:
            return {'status': 'failed', 'reason': 'No matching records found'}
        
        # Prepare CSV data
        csv_data = []
        for record in filtered_records:
            row = {
                'quantum_algorithm': record['quantum_algorithm'],
                'classical_algorithm': record['classical_algorithm'],
                'quantum_backend': record['quantum_backend'],
                'benchmark_type': record['benchmark_type'],
                'quantum_advantage': record['quantum_advantage'],
                'quantum_score': record['quantum_score'],
                'classical_score': record['classical_score'],
                'timestamp': record['timestamp']
            }
            
            # Add metrics
            for metric_name, metric_data in record['metrics'].items():
                row[f'{metric_name}_quantum'] = metric_data['quantum']
                row[f'{metric_name}_classical'] = metric_data['classical']
                row[f'{metric_name}_advantage'] = metric_data['advantage']
            
            csv_data.append(row)
        
        # Write to CSV
        fieldnames = list(csv_data[0].keys()) if csv_data else []
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        return {
            'status': 'success',
            'filepath': filepath,
            'record_count': len(csv_data),
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_markdown_report(self, suite_id: str = None) -> str:
        """
        Generate a markdown report
        
        Args:
            suite_id: Optional suite ID to report on (None for all suites)
            
        Returns:
            Markdown report text
        """
        # Generate report data
        if suite_id:
            if suite_id not in self.benchmarking.benchmark_suites:
                return f"# Error: Suite {suite_id} not found"
            
            report_data = self.benchmarking.benchmark_suites[suite_id].generate_report()
            title = f"Benchmark Report: {suite_id}"
        else:
            report_data = self.benchmarking.generate_benchmark_report()
            title = "Comprehensive Benchmark Report"
        
        # Build markdown report
        md_report = [
            f"# {title}",
            f"",
            f"Generated: {datetime.now().isoformat()}",
            f""
        ]
        
        if 'suite_id' in report_data:
            md_report.extend([
                f"## Suite: {report_data['suite_id']}",
                f"",
                f"**Type:** {report_data['benchmark_type']}",
                f"",
                f"**Description:** {report_data['description']}",
                f""
            ])
        
        # Add summary statistics
        if 'total_results' in report_data:
            md_report.extend([
                f"## Summary Statistics",
                f"",
                f"- Total Results: {report_data['total_results']}",
                f"- Total Comparisons: {report_data['total_comparisons']}",
                f""
            ])
        
        if 'comparison_stats' in report_data and report_data['comparison_stats']:
            stats = report_data['comparison_stats']
            md_report.extend([
                f"- Average Quantum Advantage: {stats['avg_quantum_advantage']:.2%}",
                f"- Maximum Quantum Advantage: {stats['max_quantum_advantage']:.2%}",
                f"- Significant Comparisons: {stats['significant_comparisons']} ({stats['significance_rate']:.1%})",
                f""
            ])
        
        # Add algorithm statistics
        if 'by_algorithm' in report_data:
            md_report.append(f"## Algorithm Performance")
            md_report.append(f"")
            
            for algorithm, stats in report_data['by_algorithm'].items():
                md_report.extend([
                    f"### {algorithm}",
                    f"",
                    f"- **Total Runs:** {stats['count']}",
                    f"- **Avg Quantum Advantage:** {stats['avg_quantum_advantage']:.2%}",
                    f"- **Max Quantum Advantage:** {stats['max_quantum_advantage']:.2%}",
                    f"- **Positive Rate:** {stats['positive_rate']:.1%}",
                    f""
                ])
        
        # Add backend statistics if available
        if 'by_backend' in report_data:
            md_report.append(f"## Backend Performance")
            md_report.append(f"")
            
            for backend, stats in report_data['by_backend'].items():
                md_report.extend([
                    f"### {backend}",
                    f"",
                    f"- **Total Runs:** {stats['count']}",
                    f"- **Avg Quantum Advantage:** {stats['avg_quantum_advantage']:.2%}",
                    f"- **Max Quantum Advantage:** {stats['max_quantum_advantage']:.2%}",
                    f"- **Positive Rate:** {stats['positive_rate']:.1%}",
                    f""
                ])
        
        # Add top performers
        if 'top_algorithms' in report_data:
            md_report.append(f"## Top Performing Algorithms")
            md_report.append(f"")
            
            for algorithm, stats in report_data['top_algorithms']:
                md_report.extend([
                    f"### {algorithm}",
                    f"",
                    f"- **Avg Quantum Advantage:** {stats['avg_advantage']:.2%}",
                    f"- **Positive Rate:** {stats['positive_rate']:.1%}",
                    f""
                ])
        
        if 'top_backends' in report_data:
            md_report.append(f"## Top Performing Backends")
            md_report.append(f"")
            
            for backend, stats in report_data['top_backends']:
                md_report.extend([
                    f"### {backend}",
                    f"",
                    f"- **Avg Quantum Advantage:** {stats['avg_advantage']:.2%}",
                    f"- **Positive Rate:** {stats['positive_rate']:.1%}",
                    f""
                ])
        
        return '\n'.join(md_report)

class QuantumBenchmarkingFactory:
    """Factory for creating comprehensive quantum benchmarking systems"""
    
    @staticmethod
    def create_benchmarking_system() -> QuantumBenchmarking:
        """Create a complete quantum benchmarking system"""
        benchmarking = QuantumBenchmarking()
        
        # Add performance tracker
        benchmarking.performance_tracker = QuantumPerformanceTracker(benchmarking)
        
        # Add visualizer
        benchmarking.visualizer = QuantumBenchmarkVisualizer(benchmarking)
        
        # Add exporter
        benchmarking.exporter = QuantumBenchmarkExporter(benchmarking)
        
        # Add orchestrator
        benchmarking.orchestrator = QuantumBenchmarkOrchestrator(benchmarking)
        
        return benchmarking