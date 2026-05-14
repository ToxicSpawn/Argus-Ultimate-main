"""
Argus Ultimate - Testing Framework
==================================

Comprehensive testing framework for all Argus components.
Provides base classes for unit, integration, and performance tests.
"""

import unittest
import asyncio
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test execution result."""
    test_name: str
    passed: bool
    duration_ms: float
    error_message: Optional[str] = None
    context: Dict[str, Any] = None


class ArgusTestCase(unittest.TestCase):
    """
    Base test case for all Argus tests.
    
    Provides:
    - Async test support
    - Performance measurement
    - Common assertions
    - Test data setup
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data = {}
        self.mocks = {}
        self._setup_test_data()
    
    def tearDown(self):
        """Clean up after test."""
        self._cleanup()
    
    def _setup_test_data(self):
        """Override to set up test-specific data."""
        pass
    
    def _setup_mocks(self):
        """Override to set up mock objects."""
        pass
    
    def _cleanup(self):
        """Override to clean up resources."""
        pass
    
    def run_async(self, coro):
        """Run async coroutine in test."""
        return asyncio.run(coro)
    
    def assert_performance(
        self,
        func: Callable,
        max_time_ms: float,
        *args,
        **kwargs
    ) -> Any:
        """
        Assert function executes within time limit.
        
        Args:
            func: Function to test
            max_time_ms: Maximum execution time in milliseconds
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            AssertionError: If execution time exceeds limit
        """
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        
        self.assertLess(
            elapsed,
            max_time_ms,
            f"Function took {elapsed:.2f}ms, exceeding limit of {max_time_ms}ms"
        )
        
        return result
    
    def assert_no_exceptions(self, func: Callable, *args, **kwargs):
        """
        Assert function raises no exceptions.
        
        Args:
            func: Function to test
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Raises:
            AssertionError: If function raises exception
        """
        try:
            func(*args, **kwargs)
        except Exception as e:
            self.fail(f"Function raised exception: {type(e).__name__}: {e}")
    
    def assert_async_no_exceptions(self, coro):
        """Assert async coroutine raises no exceptions."""
        try:
            self.run_async(coro)
        except Exception as e:
            self.fail(f"Coroutine raised exception: {type(e).__name__}: {e}")
    
    def assert_dict_has_keys(self, d: Dict, keys: List[str]):
        """Assert dictionary has all required keys."""
        for key in keys:
            self.assertIn(key, d, f"Missing required key: {key}")
    
    def assert_valid_numeric(
        self,
        value: Any,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_none: bool = False
    ):
        """
        Assert value is valid numeric within optional bounds.
        
        Args:
            value: Value to check
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_none: Whether None is allowed
        """
        if value is None:
            if allow_none:
                return
            self.fail("Value is None, but None not allowed")
        
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            self.fail(f"Value {value} is not numeric")
        
        if min_value is not None:
            self.assertGreaterEqual(
                num_value,
                min_value,
                f"Value {num_value} below minimum {min_value}"
            )
        
        if max_value is not None:
            self.assertLessEqual(
                num_value,
                max_value,
                f"Value {num_value} above maximum {max_value}"
            )


class IntegrationTest(ArgusTestCase):
    """
    Base class for integration tests.
    
    Provides:
    - Component integration testing
    - Database setup/teardown
    - External service mocking
    """
    
    def setUp(self):
        super().setUp()
        self._setup_integration()
    
    def tearDown(self):
        self._teardown_integration()
        super().tearDown()
    
    def _setup_integration(self):
        """Override to set up integration test environment."""
        pass
    
    def _teardown_integration(self):
        """Override to tear down integration test environment."""
        pass
    
    @contextmanager
    def temporary_database(self):
        """Context manager for temporary test database."""
        import tempfile
        import sqlite3
        
        db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        conn = sqlite3.connect(db_file.name)
        
        try:
            yield conn
        finally:
            conn.close()
            import os
            os.unlink(db_file.name)


class PerformanceTest(ArgusTestCase):
    """
    Base class for performance tests.
    
    Provides:
    - Benchmark execution
    - Performance regression detection
    - Resource usage tracking
    """
    
    def benchmark(
        self,
        func: Callable,
        iterations: int = 100,
        warmup: int = 10,
        *args,
        **kwargs
    ) -> Dict[str, float]:
        """
        Benchmark function performance.
        
        Args:
            func: Function to benchmark
            iterations: Number of iterations
            warmup: Number of warmup iterations
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Dictionary with benchmark statistics
        """
        # Warmup
        for _ in range(warmup):
            func(*args, **kwargs)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            times.append(time.perf_counter() - start)
        
        times.sort()
        
        return {
            'mean_ms': sum(times) / len(times) * 1000,
            'min_ms': times[0] * 1000,
            'max_ms': times[-1] * 1000,
            'p50_ms': times[int(len(times) * 0.50)] * 1000,
            'p95_ms': times[int(len(times) * 0.95)] * 1000,
            'p99_ms': times[int(len(times) * 0.99)] * 1000,
            'iterations': iterations
        }
    
    def assert_performance_regression(
        self,
        current: Dict[str, float],
        baseline: Dict[str, float],
        max_regression: float = 0.10
    ):
        """
        Assert no performance regression.
        
        Args:
            current: Current benchmark results
            baseline: Baseline benchmark results
            max_regression: Maximum allowed regression (0.10 = 10%)
            
        Raises:
            AssertionError: If performance regressed more than allowed
        """
        current_mean = current['mean_ms']
        baseline_mean = baseline['mean_ms']
        
        regression = (current_mean - baseline_mean) / baseline_mean
        
        self.assertLess(
            regression,
            max_regression,
            f"Performance regression: {regression:.1%} "
            f"(current: {current_mean:.2f}ms, baseline: {baseline_mean:.2f}ms)"
        )


class AsyncTestCase(unittest.IsolatedAsyncioTestCase):
    """
    Base class for async tests.
    
    Extends unittest.IsolatedAsyncioTestCase for proper async support.
    """
    
    async def asyncSetUp(self):
        """Set up async test fixtures."""
        pass
    
    async def asyncTearDown(self):
        """Clean up async test fixtures."""
        pass


class ComponentTest(ArgusTestCase):
    """
    Base class for component-level tests.
    
    Provides:
    - Component isolation
    - Dependency injection
    - State verification
    """
    
    def setUp(self):
        super().setUp()
        self.component = None
        self.dependencies = {}
    
    def inject_dependency(self, name: str, mock: Any):
        """Inject mock dependency."""
        self.dependencies[name] = mock
    
    def verify_state(self, expected_state: Dict[str, Any]):
        """Verify component state matches expected."""
        actual_state = self._get_component_state()
        
        for key, expected_value in expected_state.items():
            self.assertIn(key, actual_state, f"Missing state key: {key}")
            actual_value = actual_state[key]
            self.assertEqual(
                actual_value,
                expected_value,
                f"State mismatch for {key}: expected {expected_value}, got {actual_value}"
            )
    
    def _get_component_state(self) -> Dict[str, Any]:
        """Override to return component state as dictionary."""
        return {}


class E2ETest(IntegrationTest):
    """
    Base class for end-to-end tests.
    
    Provides:
    - Full system testing
    - Real-world scenario simulation
    - Multi-component integration
    """
    
    def setUp(self):
        super().setUp()
        self.system_under_test = None
    
    def simulate_market_scenario(
        self,
        scenario: str,
        duration_seconds: int = 60
    ) -> Dict[str, Any]:
        """
        Simulate market scenario for testing.
        
        Args:
            scenario: Scenario name (e.g., 'bull_market', 'high_volatility')
            duration_seconds: Simulation duration
            
        Returns:
            Simulation results
        """
        # Override with specific scenario implementation
        return {}
    
    def assert_system_health(self, health_checks: Dict[str, bool]):
        """Assert all system health checks pass."""
        for check_name, should_pass in health_checks.items():
            result = self._run_health_check(check_name)
            if should_pass:
                self.assertTrue(result, f"Health check failed: {check_name}")
            else:
                self.assertFalse(result, f"Health check should fail: {check_name}")
    
    def _run_health_check(self, check_name: str) -> bool:
        """Override to implement health checks."""
        return True


# Test utilities
def generate_test_data(count: int = 100) -> List[Dict]:
    """Generate sample test data."""
    import random
    
    data = []
    for i in range(count):
        data.append({
            'id': i,
            'symbol': random.choice(['BTC/USD', 'ETH/USD', 'SOL/USD']),
            'price': random.uniform(100, 50000),
            'volume': random.uniform(0.1, 100),
            'timestamp': time.time()
        })
    
    return data


def create_mock_order(**kwargs) -> Dict:
    """Create mock order for testing."""
    from decimal import Decimal
    
    return {
        'id': kwargs.get('id', 'ORD-TEST-001'),
        'symbol': kwargs.get('symbol', 'BTC/USD'),
        'side': kwargs.get('side', 'buy'),
        'quantity': Decimal(str(kwargs.get('quantity', 0.1))),
        'price': Decimal(str(kwargs.get('price', 45000))),
        'order_type': kwargs.get('order_type', 'limit'),
        'status': kwargs.get('status', 'pending')
    }


def create_mock_signal(**kwargs) -> Dict:
    """Create mock signal for testing."""
    return {
        'symbol': kwargs.get('symbol', 'BTC/USD'),
        'side': kwargs.get('side', 'buy'),
        'confidence': kwargs.get('confidence', 0.75),
        'strategy': kwargs.get('strategy', 'test_strategy'),
        'suggested_qty': kwargs.get('suggested_qty', 0.1),
        'suggested_price': kwargs.get('suggested_price', 45000),
        'timestamp': time.time()
    }
