"""
Test cases for Quantum Benchmarking Framework
"""

import pytest
import numpy as np
from datetime import datetime
from quantum.advanced.quantum_benchmarking import (
    BenchmarkType,
    MetricType,
    BenchmarkResult,
    BenchmarkComparison,
    BenchmarkSuite,
    QuantumBenchmarking,
    QuantumBenchmarkOrchestrator,
    QuantumPerformanceTracker,
    QuantumBenchmarkVisualizer,
    QuantumBenchmarkExporter,
    QuantumBenchmarkingFactory
)


@pytest.fixture
def benchmarking_system():
    """Create a test benchmarking system"""
    return QuantumBenchmarkingFactory.create_benchmarking_system()


def test_benchmark_result():
    """Test BenchmarkResult class"""
    result = BenchmarkResult(
        benchmark_id="test_benchmark",
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        algorithm="quantum_qaoa",
        backend="simulator",
        metrics={
            MetricType.EXECUTION_TIME: 100.0,
            MetricType.FIDELITY: 0.95,
            MetricType.ACCURACY: 0.92
        },
        parameters={"qubits": 5, "depth": 20}
    )
    
    assert result.benchmark_id == "test_benchmark"
    assert result.benchmark_type == BenchmarkType.PORTFOLIO_OPTIMIZATION
    assert result.calculate_score() > 0
    assert "execution_time" in result.to_dict()["metrics"]


def test_benchmark_comparison():
    """Test BenchmarkComparison class"""
    quantum_result = BenchmarkResult(
        benchmark_id="test_benchmark",
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        algorithm="quantum_qaoa",
        backend="simulator",
        metrics={
            MetricType.EXECUTION_TIME: 80.0,
            MetricType.FIDELITY: 0.95,
            MetricType.ACCURACY: 0.92
        },
        parameters={"qubits": 5, "depth": 20}
    )
    
    classical_result = BenchmarkResult(
        benchmark_id="test_benchmark",
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        algorithm="classical_mvo",
        backend="cpu",
        metrics={
            MetricType.EXECUTION_TIME: 100.0,
            MetricType.FIDELITY: 0.90,
            MetricType.ACCURACY: 0.88
        },
        parameters={"qubits": 5, "depth": 20}
    )
    
    comparison = BenchmarkComparison(quantum_result, classical_result)
    
    assert comparison.comparison_metrics["EXECUTION_TIME_advantage"] > 0
    assert comparison.comparison_metrics["FIDELITY_advantage"] > 0
    assert comparison.comparison_metrics["ACCURACY_advantage"] > 0
    assert comparison.comparison_metrics["overall_quantum_advantage"] > 0
    assert comparison.is_significant()


def test_benchmark_suite():
    """Test BenchmarkSuite class"""
    suite = BenchmarkSuite(
        suite_id="test_suite",
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        description="Test portfolio optimization benchmarks"
    )
    
    # Add some results
    result1 = BenchmarkResult(
        benchmark_id="benchmark1",
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        algorithm="quantum_qaoa",
        backend="simulator",
        metrics={
            MetricType.EXECUTION_TIME: 80.0,
            MetricType.FIDELITY: 0.95
        },
        parameters={"qubits": 5}
    )
    
    result2 = BenchmarkResult(
        benchmark_id="benchmark2",
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        algorithm="classical_mvo",
        backend="cpu",
        metrics={
            MetricType.EXECUTION_TIME: 100.0,
            MetricType.FIDELITY: 0.90
        },
        parameters={"qubits": 5}
    )
    
    suite.add_result(result1)
    suite.add_result(result2)
    
    # Test suite methods
    assert suite.get_best_result() is not None
    assert len(suite.generate_report()["by_algorithm"]) > 0


def test_quantum_benchmarking_initialization(benchmarking_system):
    """Test QuantumBenchmarking initialization"""
    assert hasattr(benchmarking_system, 'benchmark_suites')
    assert hasattr(benchmarking_system, 'benchmark_history')
    assert hasattr(benchmarking_system, 'comparison_history')
    assert hasattr(benchmarking_system, 'performance_tracker')
    assert hasattr(benchmarking_system, 'visualizer')
    assert hasattr(benchmarking_system, 'exporter')
    assert hasattr(benchmarking_system, 'orchestrator')


def test_create_benchmark_suite(benchmarking_system):
    """Test creating a benchmark suite"""
    suite = benchmarking_system.create_benchmark_suite(
        suite_id="test_suite",
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        description="Test portfolio optimization benchmarks"
    )
    
    assert suite.suite_id == "test_suite"
    assert suite.benchmark_type == BenchmarkType.PORTFOLIO_OPTIMIZATION
    assert "test_suite" in benchmarking_system.benchmark_suites


def test_run_benchmark(benchmarking_system):
    """Test running a benchmark"""
    def dummy_exec_func():
        return {
            MetricType.EXECUTION_TIME: 50.0,
            MetricType.FIDELITY: 0.98,
            MetricType.ACCURACY: 0.95
        }
    
    result = benchmarking_system.run_benchmark(
        benchmark_id="test_benchmark",
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        algorithm="quantum_qaoa",
        backend="simulator",
        parameters={"qubits": 5, "depth": 20},
        execution_func=dummy_exec_func
    )
    
    assert result.benchmark_id == "test_benchmark"
    assert result.algorithm == "quantum_qaoa"
    assert len(benchmarking_system.benchmark_history) == 1


def test_compare_algorithms(benchmarking_system):
    """Test comparing algorithms"""
    # Create quantum result
    quantum_result = BenchmarkResult(
        benchmark_id="test_benchmark",
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        algorithm="quantum_qaoa",
        backend="simulator",
        metrics={
            MetricType.EXECUTION_TIME: 80.0,
            MetricType.FIDELITY: 0.95,
            MetricType.ACCURACY: 0.92
        },
        parameters={"qubits": 5, "depth": 20}
    )
    
    # Create classical result
    classical_result = BenchmarkResult(
        benchmark_id="test_benchmark",
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        algorithm="classical_mvo",
        backend="cpu",
        metrics={
            MetricType.EXECUTION_TIME: 100.0,
            MetricType.FIDELITY: 0.90,
            MetricType.ACCURACY: 0.88
        },
        parameters={"qubits": 5, "depth": 20}
    )
    
    comparison = benchmarking_system.compare_algorithms(quantum_result, classical_result)
    
    assert comparison.quantum_result == quantum_result
    assert comparison.classical_result == classical_result
    assert comparison.comparison_metrics["overall_quantum_advantage"] > 0
    assert len(benchmarking_system.comparison_history) == 1


def test_run_comparison_benchmark(benchmarking_system):
    """Test running a comparison benchmark"""
    def quantum_exec_func(params):
        return {
            MetricType.EXECUTION_TIME: 80.0,
            MetricType.FIDELITY: 0.95,
            MetricType.ACCURACY: 0.92
        }
    
    def classical_exec_func(params):
        return {
            MetricType.EXECUTION_TIME: 100.0,
            MetricType.FIDELITY: 0.90,
            MetricType.ACCURACY: 0.88
        }
    
    comparison = benchmarking_system.run_comparison_benchmark(
        benchmark_id="test_comparison",
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        parameters={"qubits": 5, "depth": 20},
        quantum_exec_func=quantum_exec_func,
        classical_exec_func=classical_exec_func
    )
    
    assert comparison.quantum_result.algorithm == "quantum"
    assert comparison.classical_result.algorithm == "classical"
    assert comparison.comparison_metrics["overall_quantum_advantage"] > 0


def test_benchmark_history(benchmarking_system):
    """Test benchmark history tracking"""
    def dummy_exec_func():
        return {MetricType.EXECUTION_TIME: 50.0}
    
    # Run multiple benchmarks
    for i in range(5):
        benchmarking_system.run_benchmark(
            benchmark_id=f"test_benchmark_{i}",
            benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
            algorithm="quantum_qaoa",
            backend="simulator",
            parameters={"qubits": 5, "depth": 20},
            execution_func=dummy_exec_func
        )
    
    history = benchmarking_system.get_benchmark_history(limit=10)
    assert len(history) == 5
    assert history[0]["benchmark_id"] == "test_benchmark_4"


def test_comparison_history(benchmarking_system):
    """Test comparison history tracking"""
    def quantum_exec_func(params):
        return {MetricType.EXECUTION_TIME: 80.0}
    
    def classical_exec_func(params):
        return {MetricType.EXECUTION_TIME: 100.0}
    
    # Run multiple comparisons
    for i in range(3):
        benchmarking_system.run_comparison_benchmark(
            benchmark_id=f"test_comparison_{i}",
            benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
            parameters={"qubits": 5, "depth": 20},
            quantum_exec_func=quantum_exec_func,
            classical_exec_func=classical_exec_func
        )
    
    history = benchmarking_system.get_comparison_history(limit=10)
    assert len(history) == 3
    assert history[0]["quantum"]["benchmark_id"] == "test_comparison_2_quantum"


def test_generate_benchmark_report(benchmarking_system):
    """Test generating benchmark reports"""
    # Create a suite with some results
    suite = benchmarking_system.create_benchmark_suite(
        suite_id="report_test",
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        description="Test suite for reporting"
    )
    
    # Add some results
    for i in range(3):
        result = BenchmarkResult(
            benchmark_id=f"report_benchmark_{i}",
            benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
            algorithm="quantum_qaoa" if i % 2 == 0 else "classical_mvo",
            backend="simulator" if i % 2 == 0 else "cpu",
            metrics={
                MetricType.EXECUTION_TIME: 80.0 if i % 2 == 0 else 100.0,
                MetricType.FIDELITY: 0.95 if i % 2 == 0 else 0.90
            },
            parameters={"qubits": 5}
        )
        suite.add_result(result)
    
    # Generate report
    report = suite.generate_report()
    assert report["suite_id"] == "report_test"
    assert len(report["by_algorithm"]) > 0


def test_statistical_significance(benchmarking_system):
    """Test statistical significance calculation"""
    # Create a comparison with quantum advantage
    quantum_result = BenchmarkResult(
        benchmark_id="test_benchmark",
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        algorithm="quantum_qaoa",
        backend="simulator",
        metrics={
            MetricType.EXECUTION_TIME: 80.0,
            MetricType.ACCURACY: 0.92
        },
        parameters={"qubits": 5}
    )
    
    classical_result = BenchmarkResult(
        benchmark_id="test_benchmark",
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        algorithm="classical_mvo",
        backend="cpu",
        metrics={
            MetricType.EXECUTION_TIME: 100.0,
            MetricType.ACCURACY: 0.88
        },
        parameters={"qubits": 5}
    )
    
    comparison = BenchmarkComparison(quantum_result, classical_result)
    
    # Test significance for accuracy
    significance = benchmarking_system.calculate_statistical_significance(
        comparison, MetricType.ACCURACY
    )
    
    assert "p_value" in significance
    assert "significant" in significance
    assert significance["metric"] == "ACCURACY"


def test_quantum_advantage_analysis(benchmarking_system):
    """Test quantum advantage analysis"""
    # Add some comparisons with varying advantages
    for i in range(5):
        quantum_advantage = 0.05 + i * 0.02  # Varying from 5% to 13%
        
        quantum_result = BenchmarkResult(
            benchmark_id=f"advantage_test_{i}",
            benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
            algorithm="quantum_qaoa",
            backend="simulator",
            metrics={
                MetricType.EXECUTION_TIME: 80.0 - i * 2,
                MetricType.ACCURACY: 0.90 + i * 0.01
            },
            parameters={"qubits": 5}
        )
        
        classical_result = BenchmarkResult(
            benchmark_id=f"advantage_test_{i}",
            benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
            algorithm="classical_mvo",
            backend="cpu",
            metrics={
                MetricType.EXECUTION_TIME: 100.0,
                MetricType.ACCURACY: 0.90
            },
            parameters={"qubits": 5}
        )
        
        comparison = BenchmarkComparison(quantum_result, classical_result)
        # Manually set the advantage we want
        comparison.comparison_metrics["overall_quantum_advantage"] = quantum_advantage
        benchmarking_system.comparison_history.append(comparison)
    
    # Analyze quantum advantage
    analysis = benchmarking_system.analyze_quantum_advantage(min_advantage=0.05)
    
    assert analysis["total_comparisons"] == 5
    assert analysis["significant_comparisons"] == 5
    assert analysis["significance_rate"] == 1.0


def test_quantum_advantage_report(benchmarking_system):
    """Test quantum advantage report generation"""
    # Add some comparisons with varying advantages
    for i in range(3):
        quantum_advantage = 0.05 + i * 0.05  # 5%, 10%, 15%
        
        quantum_result = BenchmarkResult(
            benchmark_id=f"report_test_{i}",
            benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
            algorithm=f"quantum_alg_{i}",
            backend="simulator",
            metrics={
                MetricType.EXECUTION_TIME: 80.0 - i * 5,
                MetricType.ACCURACY: 0.90 + i * 0.02
            },
            parameters={"qubits": 5}
        )
        
        classical_result = BenchmarkResult(
            benchmark_id=f"report_test_{i}",
            benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
            algorithm="classical_mvo",
            backend="cpu",
            metrics={
                MetricType.EXECUTION_TIME: 100.0,
                MetricType.ACCURACY: 0.90
            },
            parameters={"qubits": 5}
        )
        
        comparison = BenchmarkComparison(quantum_result, classical_result)
        comparison.comparison_metrics["overall_quantum_advantage"] = quantum_advantage
        benchmarking_system.comparison_history.append(comparison)
    
    # Generate report
    report = benchmarking_system.generate_quantum_advantage_report(min_advantage=0.05)
    
    assert report["total_comparisons"] == 3
    assert len(report["by_advantage"]) == 3
    assert len(report["top_performers"]) == 3


def test_benchmark_orchestrator(benchmarking_system):
    """Test benchmark orchestrator"""
    orchestrator = benchmarking_system.orchestrator
    
    # Create a benchmark plan
    plan = orchestrator.create_benchmark_plan(
        plan_id="test_plan",
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        description="Test portfolio optimization plan",
        parameters_grid=[
            {"qubits": 4, "depth": 10},
            {"qubits": 5, "depth": 15},
            {"qubits": 6, "depth": 20}
        ],
        quantum_algorithm="quantum_qaoa",
        classical_algorithm="classical_mvo",
        quantum_backend="simulator",
        classical_backend="cpu"
    )
    
    assert plan["status"] == "created"
    assert plan["plan_id"] == "test_plan"
    assert len(plan["parameters_grid"]) == 3


def test_performance_tracker(benchmarking_system):
    """Test performance tracker"""
    tracker = benchmarking_system.performance_tracker
n    # Create a comparison with quantum advantage
    quantum_result = BenchmarkResult(
        benchmark_id="tracker_test",
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        algorithm="quantum_qaoa",
        backend="simulator",
        metrics={
            MetricType.EXECUTION_TIME: 80.0,
            MetricType.ACCURACY: 0.92
        },
        parameters={"qubits": 5}
    )
    
    classical_result = BenchmarkResult(
        benchmark_id="tracker_test",
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        algorithm="classical_mvo",
        backend="cpu",
        metrics={
            MetricType.EXECUTION_TIME: 100.0,
            MetricType.ACCURACY: 0.88
        },
        parameters={"qubits": 5}
    )
    
    comparison = BenchmarkComparison(quantum_result, classical_result)
    
    # Record performance
    tracker.record_performance(comparison, tags=["portfolio", "optimization"])
    
    # Get performance trends
    trends = tracker.get_performance_trends(algorithm="quantum_qaoa")
    assert trends["status"] == "success"
    assert len(trends["trend"]["advantages"]) == 1


def test_benchmark_visualizer(benchmarking_system):
    """Test benchmark visualizer"""
    visualizer = benchmarking_system.visualizer
    
    # Generate advantage plot data
    plot_data = visualizer.generate_advantage_plot_data()
    assert plot_data["status"] in ["no_data", "success"]
    
    # Generate backend comparison data
    backend_data = visualizer.generate_backend_comparison_data()
    assert backend_data["status"] in ["no_data", "success"]


def test_benchmark_exporter(benchmarking_system, tmp_path):
    """Test benchmark exporter"""
    exporter = benchmarking_system.exporter
    
    # Create a test suite
    suite = benchmarking_system.create_benchmark_suite(
        suite_id="export_test",
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        description="Test suite for export"
    )
    
    # Add some results
    for i in range(2):
        result = BenchmarkResult(
            benchmark_id=f"export_benchmark_{i}",
            benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
            algorithm="quantum_qaoa" if i == 0 else "classical_mvo",
            backend="simulator" if i == 0 else "cpu",
            metrics={
                MetricType.EXECUTION_TIME: 80.0 if i == 0 else 100.0,
                MetricType.FIDELITY: 0.95 if i == 0 else 0.90
            },
            parameters={"qubits": 5}
        )
        suite.add_result(result)
    
    # Export to JSON
    json_file = tmp_path / "benchmark_export.json"
    export_result = exporter.export_to_json(str(json_file), suite_id="export_test")
    
    assert export_result["status"] == "success"
    assert json_file.exists()
    
    # Export comparison to CSV
    # First create a comparison
    quantum_result = BenchmarkResult(
        benchmark_id="csv_test",
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        algorithm="quantum_qaoa",
        backend="simulator",
        metrics={
            MetricType.EXECUTION_TIME: 80.0,
            MetricType.ACCURACY: 0.92
        },
        parameters={"qubits": 5}
    )
    
    classical_result = BenchmarkResult(
        benchmark_id="csv_test",
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        algorithm="classical_mvo",
        backend="cpu",
        metrics={
            MetricType.EXECUTION_TIME: 100.0,
            MetricType.ACCURACY: 0.88
        },
        parameters={"qubits": 5}
    )
    
    comparison = BenchmarkComparison(quantum_result, classical_result)
    benchmarking_system.comparison_history.append(comparison)
    
    csv_file = tmp_path / "comparison_export.csv"
    export_result = exporter.export_comparison_to_csv(str(csv_file))
    
    assert export_result["status"] == "success"
    assert csv_file.exists()


def test_benchmark_id_generation(benchmarking_system):
    """Test benchmark ID generation"""
    benchmark_id = benchmarking_system.create_benchmark_id(
        benchmark_type=BenchmarkType.PORTFOLIO_OPTIMIZATION,
        algorithm="quantum_qaoa",
        parameters={"qubits": 5, "depth": 20, "iterations": 100}
    )
    
    assert len(benchmark_id) == 16  # MD5 hash truncated to 16 chars
    assert isinstance(benchmark_id, str)


def test_benchmarking_factory():
    """Test benchmarking factory"""
    benchmarking = QuantumBenchmarkingFactory.create_benchmarking_system()
    
    assert hasattr(benchmarking, 'benchmark_suites')
    assert hasattr(benchmarking, 'performance_tracker')
    assert hasattr(benchmarking, 'visualizer')
    assert hasattr(benchmarking, 'exporter')
    assert hasattr(benchmarking, 'orchestrator')