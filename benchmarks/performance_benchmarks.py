"""
Performance Benchmarks for Argus Ultimate
==========================================

Comprehensive performance testing and benchmarking suite.
"""

import time
import asyncio
import statistics
import logging
from typing import Dict, List, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
import numpy as np

from unified_trading import UnifiedTradingOrchestrator
from unified_trading.order_management import OrderManager, Signal, OrderSide
from unified_trading.execution_engine import ExecutionEngine
from unified_trading.risk_integration import RiskIntegration
from unified_trading.portfolio_management import PortfolioManager
from unified_trading.signal_processing import SignalProcessor
from core.cache_manager import CacheManager, get_cache

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark execution result."""
    name: str
    iterations: int
    mean_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float
    throughput: float  # ops/sec
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'iterations': self.iterations,
            'mean_ms': self.mean_ms,
            'min_ms': self.min_ms,
            'max_ms': self.max_ms,
            'median_ms': self.median_ms,
            'p95_ms': self.p95_ms,
            'p99_ms': self.p99_ms,
            'std_ms': self.std_ms,
            'throughput': self.throughput,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self, name: str, iterations: int = 1000):
        self.name = name
        self.iterations = iterations
        self.results: List[float] = []
    
    async def setup(self):
        """Setup before benchmark. Override in subclass."""
        pass
    
    async def teardown(self):
        """Teardown after benchmark. Override in subclass."""
        pass
    
    async def run_iteration(self):
        """Single benchmark iteration. Override in subclass."""
        pass
    
    async def execute(self) -> BenchmarkResult:
        """Execute benchmark."""
        logger.info(f"Starting benchmark: {self.name} ({self.iterations} iterations)")
        
        await self.setup()
        
        # Warmup
        for _ in range(min(10, self.iterations // 10)):
            await self.run_iteration()
        
        # Actual benchmark
        self.results = []
        for i in range(self.iterations):
            start = time.perf_counter()
            await self.run_iteration()
            elapsed = (time.perf_counter() - start) * 1000
            self.results.append(elapsed)
        
        await self.teardown()
        
        # Calculate statistics
        self.results.sort()
        
        result = BenchmarkResult(
            name=self.name,
            iterations=self.iterations,
            mean_ms=statistics.mean(self.results),
            min_ms=min(self.results),
            max_ms=max(self.results),
            median_ms=statistics.median(self.results),
            p95_ms=np.percentile(self.results, 95),
            p99_ms=np.percentile(self.results, 99),
            std_ms=statistics.stdev(self.results) if len(self.results) > 1 else 0,
            throughput=self.iterations / (sum(self.results) / 1000)
        )
        
        logger.info(f"Benchmark complete: {result.mean_ms:.2f}ms mean, "
                   f"{result.throughput:.1f} ops/sec")
        
        return result


class OrderCreationBenchmark(PerformanceBenchmark):
    """Benchmark order creation performance."""
    
    def __init__(self, iterations: int = 10000):
        super().__init__("order_creation", iterations)
        self.order_manager = OrderManager()
    
    async def run_iteration(self):
        signal = Signal(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            confidence=0.8,
            strategy="momentum",
            suggested_qty=Decimal("0.1"),
            suggested_price=Decimal("45000")
        )
        await self.order_manager.create_order(signal)


class SignalProcessingBenchmark(PerformanceBenchmark):
    """Benchmark signal processing performance."""
    
    def __init__(self, iterations: int = 5000):
        super().__init__("signal_processing", iterations)
        self.processor = SignalProcessor()
    
    async def setup(self):
        await self.processor.initialize()
        # Register strategies
        await self.processor.register_strategy(
            "momentum",
            lambda s, p, **kw: Signal(
                symbol=s, side=OrderSide.BUY, confidence=0.7,
                strategy="momentum", suggested_qty=Decimal("0.1")
            )
        )
    
    async def run_iteration(self):
        await self.processor.generate_signals("BTC/USD", 45000.0)


class RiskCheckBenchmark(PerformanceBenchmark):
    """Benchmark risk check performance."""
    
    def __init__(self, iterations: int = 5000):
        super().__init__("risk_check", iterations)
        self.risk = RiskIntegration()
    
    async def setup(self):
        await self.risk.initialize({
            'max_position_size': 0.1,
            'max_drawdown': 0.2,
            'daily_loss_limit': 500
        })
    
    async def run_iteration(self):
        signal = Signal(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            confidence=0.8,
            strategy="test",
            suggested_qty=Decimal("0.1")
        )
        await self.risk.check_signal(signal)


class CacheBenchmark(PerformanceBenchmark):
    """Benchmark cache performance."""
    
    def __init__(self, iterations: int = 100000):
        super().__init__("cache_operations", iterations)
        self.cache = CacheManager()
        self.counter = 0
    
    async def run_iteration(self):
        key = f"key_{self.counter % 1000}"
        value = f"value_{self.counter}"
        
        # Set
        self.cache.set("benchmark", key, value)
        
        # Get
        result = self.cache.get("benchmark", key)
        
        self.counter += 1


class TickProcessingBenchmark(PerformanceBenchmark):
    """Benchmark complete tick processing."""
    
    def __init__(self, iterations: int = 1000):
        super().__init__("tick_processing", iterations)
        self.orchestrator = UnifiedTradingOrchestrator()
    
    async def setup(self):
        await self.orchestrator.initialize()
        await self.orchestrator.start()
    
    async def teardown(self):
        await self.orchestrator.stop()
    
    async def run_iteration(self):
        await self.orchestrator.process_tick("BTC/USD", 45000.0, volume=100)


class PortfolioUpdateBenchmark(PerformanceBenchmark):
    """Benchmark portfolio update performance."""
    
    def __init__(self, iterations: int = 5000):
        super().__init__("portfolio_update", iterations)
        self.portfolio = PortfolioManager()
        self.counter = 0
    
    async def setup(self):
        await self.portfolio.initialize(Decimal("10000"))
    
    async def run_iteration(self):
        from unified_trading.execution_engine import ExecutionResult, Fill
        
        execution = ExecutionResult(
            success=True,
            order_id=f"ORD-{self.counter}",
            status='filled',
            filled_qty=Decimal("0.01"),
            fills=[Fill(
                order_id=f"ORD-{self.counter}",
                symbol="BTC/USD",
                side="buy",
                filled_qty=Decimal("0.01"),
                price=Decimal("45000"),
                venue="binance",
                fees=Decimal("0.45")
            )]
        )
        
        await self.portfolio.update_position(execution)
        self.counter += 1


class BenchmarkSuite:
    """Suite of performance benchmarks."""
    
    def __init__(self):
        self.benchmarks: List[PerformanceBenchmark] = []
        self.results: List[BenchmarkResult] = []
    
    def add(self, benchmark: PerformanceBenchmark):
        """Add benchmark to suite."""
        self.benchmarks.append(benchmark)
    
    async def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        logger.info(f"Running {len(self.benchmarks)} benchmarks...")
        
        self.results = []
        for benchmark in self.benchmarks:
            try:
                result = await benchmark.execute()
                self.results.append(result)
            except Exception as e:
                logger.error(f"Benchmark {benchmark.name} failed: {e}")
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate benchmark report."""
        if not self.results:
            return "No benchmark results available"
        
        report = "# Performance Benchmark Report\n\n"
        report += f"Generated: {datetime.utcnow().isoformat()}\n\n"
        
        report += "## Summary\n\n"
        report += "| Benchmark | Mean (ms) | P95 (ms) | Throughput (ops/sec) |\n"
        report += "|-----------|-----------|----------|---------------------|\n"
        
        for result in self.results:
            report += (f"| {result.name} | {result.mean_ms:.2f} | "
                      f"{result.p95_ms:.2f} | {result.throughput:.1f} |\n")
        
        report += "\n## Detailed Results\n\n"
        
        for result in self.results:
            report += f"### {result.name}\n\n"
            report += f"- Iterations: {result.iterations:,}\n"
            report += f"- Mean: {result.mean_ms:.2f} ms\n"
            report += f"- Min: {result.min_ms:.2f} ms\n"
            report += f"- Max: {result.max_ms:.2f} ms\n"
            report += f"- Median: {result.median_ms:.2f} ms\n"
            report += f"- P95: {result.p95_ms:.2f} ms\n"
            report += f"- P99: {result.p99_ms:.2f} ms\n"
            report += f"- Std Dev: {result.std_ms:.2f} ms\n"
            report += f"- Throughput: {result.throughput:.1f} ops/sec\n\n"
        
        return report
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save results to file."""
        import json
        
        data = {
            'timestamp': datetime.utcnow().isoformat(),
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filename}")


# Predefined benchmark suites
async def run_standard_benchmarks():
    """Run standard benchmark suite."""
    suite = BenchmarkSuite()
    
    # Add core benchmarks
    suite.add(OrderCreationBenchmark(iterations=10000))
    suite.add(SignalProcessingBenchmark(iterations=5000))
    suite.add(RiskCheckBenchmark(iterations=5000))
    suite.add(CacheBenchmark(iterations=100000))
    suite.add(PortfolioUpdateBenchmark(iterations=5000))
    
    # Run
    results = await suite.run_all()
    
    # Generate report
    report = suite.generate_report()
    print(report)
    
    # Save results
    suite.save_results()
    
    return results


async def run_full_benchmarks():
    """Run full benchmark suite including heavy tests."""
    suite = BenchmarkSuite()
    
    # All benchmarks
    suite.add(OrderCreationBenchmark(iterations=50000))
    suite.add(SignalProcessingBenchmark(iterations=10000))
    suite.add(RiskCheckBenchmark(iterations=10000))
    suite.add(CacheBenchmark(iterations=500000))
    suite.add(TickProcessingBenchmark(iterations=1000))
    suite.add(PortfolioUpdateBenchmark(iterations=10000))
    
    results = await suite.run_all()
    
    report = suite.generate_report()
    print(report)
    
    suite.save_results("full_benchmark_results.json")
    
    return results


# Performance requirements
PERFORMANCE_REQUIREMENTS = {
    'order_creation': {'max_mean_ms': 10, 'min_throughput': 100},
    'signal_processing': {'max_mean_ms': 20, 'min_throughput': 50},
    'risk_check': {'max_mean_ms': 5, 'min_throughput': 200},
    'cache_operations': {'max_mean_ms': 0.1, 'min_throughput': 10000},
    'tick_processing': {'max_mean_ms': 100, 'min_throughput': 10},
    'portfolio_update': {'max_mean_ms': 5, 'min_throughput': 200}
}


def check_performance_requirements(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Check if benchmarks meet performance requirements."""
    failures = []
    
    for result in results:
        if result.name in PERFORMANCE_REQUIREMENTS:
            req = PERFORMANCE_REQUIREMENTS[result.name]
            
            if result.mean_ms > req['max_mean_ms']:
                failures.append({
                    'benchmark': result.name,
                    'metric': 'mean_ms',
                    'value': result.mean_ms,
                    'requirement': req['max_mean_ms']
                })
            
            if result.throughput < req['min_throughput']:
                failures.append({
                    'benchmark': result.name,
                    'metric': 'throughput',
                    'value': result.throughput,
                    'requirement': req['min_throughput']
                })
    
    return {
        'passed': len(failures) == 0,
        'failures': failures,
        'total_checks': len(results)
    }


if __name__ == '__main__':
    # Run benchmarks
    asyncio.run(run_standard_benchmarks())
