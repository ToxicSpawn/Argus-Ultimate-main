# pyright: reportMissingImports=false
"""
Testing Framework for Quantum Reinforcement Learning.

This module provides:
- Unit tests for quantum components
- Integration tests for hybrid system
- Benchmarking tools
- Test utilities and fixtures
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test."""
    test_name: str
    passed: bool
    duration_ms: float
    message: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuite:
    """Collection of test results."""
    name: str
    results: List[TestResult] = field(default_factory=list)
    
    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)
    
    @property
    def total_count(self) -> int:
        return len(self.results)
    
    @property
    def pass_rate(self) -> float:
        return self.passed_count / max(self.total_count, 1)
    
    @property
    def total_duration_ms(self) -> float:
        return sum(r.duration_ms for r in self.results)


class QuantumRLTestRunner:
    """Runs tests for quantum RL components."""
    
    def __init__(self):
        self.test_suites: List[TestSuite] = []
    
    def run_test_suite(self, suite_name: str, tests: List[Callable]) -> TestSuite:
        """Run a suite of tests."""
        logger.info("Running test suite: %s", suite_name)
        suite = TestSuite(name=suite_name)
        
        for test in tests:
            result = self._run_single_test(test)
            suite.results.append(result)
            
            if result.passed:
                logger.info("  ✓ %s (%.2fms)", result.test_name, result.duration_ms)
            else:
                logger.error("  ✗ %s: %s", result.test_name, result.message)
        
        logger.info(
            "Suite %s: %d/%d passed (%.1f%%) in %.2fms",
            suite_name, suite.passed_count, suite.total_count,
            suite.pass_rate * 100, suite.total_duration_ms
        )
        
        self.test_suites.append(suite)
        return suite
    
    def _run_single_test(self, test: Callable) -> TestResult:
        """Run a single test function."""
        test_name = test.__name__
        
        start_time = time.time()
        try:
            test()
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                test_name=test_name,
                passed=True,
                duration_ms=duration_ms
            )
        except AssertionError as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                test_name=test_name,
                passed=False,
                duration_ms=duration_ms,
                message=str(e),
                error="AssertionError"
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                test_name=test_name,
                passed=False,
                duration_ms=duration_ms,
                message=str(e),
                error=type(e).__name__
            )
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate test report."""
        total_passed = sum(s.passed_count for s in self.test_suites)
        total_tests = sum(s.total_count for s in self.test_suites)
        total_duration = sum(s.total_duration_ms for s in self.test_suites)
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_tests - total_passed,
                "pass_rate": total_passed / max(total_tests, 1),
                "total_duration_ms": total_duration
            },
            "suites": [
                {
                    "name": s.name,
                    "passed": s.passed_count,
                    "failed": s.failed_count,
                    "duration_ms": s.total_duration_ms
                }
                for s in self.test_suites
            ]
        }


# ============================================================================
# Unit Tests for Quantum Components
# ============================================================================

class QuantumComponentTests:
    """Unit tests for quantum RL components."""
    
    @staticmethod
    def test_quantum_state_encoder():
        """Test quantum state encoding."""
        from quantum.reinforcement_learning.quantum_rl_utils import QuantumStateEncoder, StateEncoding
        
        encoder = QuantumStateEncoder(num_qubits=4, encoding=StateEncoding.ANGLE)
        state = np.array([0.5, 0.3, 0.8, 0.2, 0.1])
        
        quantum_state = encoder.encode(state)
        
        assert quantum_state.num_qubits == 4
        assert len(quantum_state.state_vector) == 16  # 2^4
        assert quantum_state.encoding == StateEncoding.ANGLE
        
        # Check normalization
        norm = np.linalg.norm(quantum_state.state_vector)
        assert abs(norm - 1.0) < 1e-6, f"State not normalized: {norm}"
    
    @staticmethod
    def test_quantum_action_selector():
        """Test quantum action selection."""
        from quantum.reinforcement_learning.quantum_rl_utils import (
            QuantumStateEncoder, QuantumActionSelector, ActionSelectionMethod, StateEncoding
        )
        
        encoder = QuantumStateEncoder(num_qubits=3, encoding=StateEncoding.ANGLE)
        selector = QuantumActionSelector(action_dim=4, method=ActionSelectionMethod.EPSILON_GREEDY, epsilon=0.1)
        
        state = np.array([0.5, 0.3, 0.8])
        quantum_state = encoder.encode(state)
        
        action = selector.select(quantum_state, training=False)
        
        assert 0 <= action.action_id < 4
        assert 0 <= action.quantum_probability <= 1.0
    
    @staticmethod
    def test_quantum_reward_processor():
        """Test quantum reward processing."""
        from quantum.reinforcement_learning.quantum_rl_utils import (
            QuantumRewardProcessor, RewardShapingMethod
        )
        
        processor = QuantumRewardProcessor(method=RewardShapingMethod.POTENTIAL_BASED)
        
        state = np.array([0.5, 0.3, 0.8])
        next_state = np.array([0.6, 0.4, 0.7])
        
        reward = processor.process(
            raw_reward=1.0,
            state=state,
            next_state=next_state,
            done=False
        )
        
        assert reward.raw_reward == 1.0
        assert isinstance(reward.total_reward, float)
    
    @staticmethod
    def test_quantum_replay_buffer():
        """Test quantum replay buffer."""
        from quantum.reinforcement_learning.quantum_rl_utils import (
            QuantumStateEncoder, QuantumActionSelector, QuantumExperience, QuantumReplayBuffer, StateEncoding
        )
        
        encoder = QuantumStateEncoder(num_qubits=3, encoding=StateEncoding.ANGLE)
        selector = QuantumActionSelector(action_dim=4)
        buffer = QuantumReplayBuffer(capacity=100)
        
        # Add some experiences
        for i in range(20):
            state = np.random.rand(5)
            quantum_state = encoder.encode(state)
            action = selector.select(quantum_state)
            
            experience = QuantumExperience(
                state=quantum_state,
                action=action,
                reward=np.random.randn(),
                done=False
            )
            buffer.add(experience)
        
        assert len(buffer) == 20
        
        # Sample batch
        batch, weights = buffer.sample(10)
        assert len(batch) == 10
        assert len(weights) == 10
    
    @staticmethod
    def test_market_state_encoder():
        """Test market state encoding."""
        from quantum.reinforcement_learning.trading_components import MarketStateEncoder
        
        encoder = MarketStateEncoder()
        
        market_data = {
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 102.0,
            "volume": 1000000.0
        }
        
        state = encoder.encode(market_data)
        
        assert isinstance(state, np.ndarray)
        assert len(state) > 0
    
    @staticmethod
    def test_action_decoder():
        """Test action decoding."""
        from quantum.reinforcement_learning.trading_components import ActionDecoder
        
        decoder = ActionDecoder()
        
        quantum_output = np.array([0.8, 0.1, 0.05, 0.05])
        action = decoder.decode(quantum_output, current_position=0.5)
        
        assert "action" in action
        assert action["action"] in ["BUY", "SELL", "HOLD", "CLOSE"]
    
    @staticmethod
    def test_risk_manager():
        """Test risk management."""
        from quantum.reinforcement_learning.trading_components import RiskManager, RiskLimits
        
        limits = RiskLimits(max_position_size=0.5, max_drawdown=0.10)
        manager = RiskManager(limits)
        
        action = {"position_change": 0.3}
        portfolio_info = {"value": 10000.0, "daily_pnl": 0.0}
        
        approved, result = manager.check_action(action, portfolio_info)
        
        assert isinstance(approved, bool)
        assert "violations" in result


# ============================================================================
# Integration Tests
# ============================================================================

class IntegrationTests:
    """Integration tests for hybrid quantum-classical system."""
    
    @staticmethod
    def test_qql_integration():
        """Test Quantum Q-Learning integration."""
        from quantum.reinforcement_learning.quantum_q_learning import QuantumQLearning, QQLConfig
        
        config = QQLConfig(num_qubits=4, num_layers=2)
        agent = QuantumQLearning(state_dim=8, action_dim=4, config=config)
        
        state = np.random.randn(8)
        action = agent.select_action(state, training=True)
        
        assert 0 <= action < 4
        
        # Test training step
        from quantum.reinforcement_learning.quantum_q_learning import Experience
        
        experience = Experience(
            state=state,
            action=action,
            reward=1.0,
            next_state=np.random.randn(8),
            done=False
        )
        agent.replay_buffer.add(experience)
        
        metrics = agent.train_step()
        assert "loss" in metrics
    
    @staticmethod
    def test_qdqn_integration():
        """Test Quantum DQN integration."""
        from quantum.reinforcement_learning.quantum_deep_q_network import (
            QuantumDeepQNetwork, QDQNConfig
        )
        
        config = QDQNConfig(num_qubits=4, num_quantum_layers=2)
        agent = QuantumDeepQNetwork(config)
        
        state = np.random.randn(8)
        action = agent.select_action(state, training=True)
        
        assert 0 <= action < 4
    
    @staticmethod
    def test_hybrid_rl_integration():
        """Test hybrid quantum-classical RL integration."""
        from quantum.reinforcement_learning.hybrid_quantum_classical_rl import (
            HybridQuantumClassicalRL, HybridRLConfig, HybridArchitecture
        )
        
        config = HybridRLConfig(
            architecture=HybridArchitecture.QUANTUM_FEATURE_EXTRACTION,
            quantum_num_qubits=4
        )
        agent = HybridQuantumClassicalRL(config)
        
        state = np.random.randn(8)
        action, log_prob, value = agent.select_action(state, training=True)
        
        assert 0 <= action < 4
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
    
    @staticmethod
    def test_quantum_backend_manager():
        """Test quantum backend manager."""
        from quantum.reinforcement_learning.quantum_backends import (
            QuantumBackendManager, QuantumBackendType, BackendConfig
        )
        
        manager = QuantumBackendManager()
        
        # Register local simulator
        config = BackendConfig(backend_type=QuantumBackendType.LOCAL_SIMULATOR)
        success = manager.register_backend(QuantumBackendType.LOCAL_SIMULATOR, config)
        
        assert success
        
        # Select backend
        backend = manager.select_backend()
        assert backend is not None
        
        # Execute circuit
        circuit = {
            "num_qubits": 4,
            "operations": [
                {"type": "h", "qubits": [0]},
                {"type": "h", "qubits": [1]},
                {"type": "cx", "qubits": [0, 1]}
            ]
        }
        
        result = backend.execute_circuit(circuit, num_shots=100)
        assert result.counts is not None
        assert len(result.counts) > 0


# ============================================================================
# Benchmarking Tools
# ============================================================================

class BenchmarkResult:
    """Result of a benchmark run."""
    
    def __init__(self, name: str, iterations: int):
        self.name = name
        self.iterations = iterations
        self.timings: List[float] = []
        self.results: List[Any] = []
    
    def add_timing(self, duration_ms: float, result: Any = None):
        """Add a timing measurement."""
        self.timings.append(duration_ms)
        if result is not None:
            self.results.append(result)
    
    @property
    def mean_time_ms(self) -> float:
        return np.mean(self.timings) if self.timings else 0.0
    
    @property
    def std_time_ms(self) -> float:
        return np.std(self.timings) if self.timings else 0.0
    
    @property
    def min_time_ms(self) -> float:
        return np.min(self.timings) if self.timings else 0.0
    
    @property
    def max_time_ms(self) -> float:
        return np.max(self.timings) if self.timings else 0.0
    
    @property
    def throughput(self) -> float:
        """Operations per second."""
        if self.mean_time_ms > 0:
            return 1000.0 / self.mean_time_ms
        return 0.0
    
    def summary(self) -> Dict[str, Any]:
        """Get benchmark summary."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "mean_time_ms": self.mean_time_ms,
            "std_time_ms": self.std_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "throughput_ops_per_sec": self.throughput
        }


class QuantumRLBenchmark:
    """Benchmarking tools for quantum RL components."""
    
    def __init__(self):
        self.benchmarks: List[BenchmarkResult] = []
    
    def benchmark_state_encoding(
        self,
        num_qubits: int = 4,
        num_iterations: int = 1000
    ) -> BenchmarkResult:
        """Benchmark state encoding performance."""
        from quantum.reinforcement_learning.quantum_rl_utils import QuantumStateEncoder, StateEncoding
        
        encoder = QuantumStateEncoder(num_qubits=num_qubits, encoding=StateEncoding.ANGLE)
        benchmark = BenchmarkResult(name="state_encoding", iterations=num_iterations)
        
        for _ in range(num_iterations):
            state = np.random.rand(2 ** num_qubits)
            
            start = time.time()
            quantum_state = encoder.encode(state)
            duration_ms = (time.time() - start) * 1000
            
            benchmark.add_timing(duration_ms, quantum_state)
        
        self.benchmarks.append(benchmark)
        return benchmark
    
    def benchmark_action_selection(
        self,
        action_dim: int = 4,
        num_iterations: int = 1000
    ) -> BenchmarkResult:
        """Benchmark action selection performance."""
        from quantum.reinforcement_learning.quantum_rl_utils import (
            QuantumStateEncoder, QuantumActionSelector, ActionSelectionMethod, StateEncoding
        )
        
        encoder = QuantumStateEncoder(num_qubits=4, encoding=StateEncoding.ANGLE)
        selector = QuantumActionSelector(action_dim=action_dim, method=ActionSelectionMethod.EPSILON_GREEDY)
        benchmark = BenchmarkResult(name="action_selection", iterations=num_iterations)
        
        for _ in range(num_iterations):
            state = np.random.rand(4)
            quantum_state = encoder.encode(state)
            
            start = time.time()
            action = selector.select(quantum_state, training=False)
            duration_ms = (time.time() - start) * 1000
            
            benchmark.add_timing(duration_ms, action)
        
        self.benchmarks.append(benchmark)
        return benchmark
    
    def benchmark_quantum_circuit(
        self,
        num_qubits: int = 4,
        num_layers: int = 3,
        num_iterations: int = 100
    ) -> BenchmarkResult:
        """Benchmark quantum circuit execution."""
        from quantum.reinforcement_learning.quantum_backends import (
            QuantumBackendManager, QuantumBackendType, BackendConfig
        )
        
        manager = QuantumBackendManager()
        config = BackendConfig(backend_type=QuantumBackendType.LOCAL_SIMULATOR)
        manager.register_backend(QuantumBackendType.LOCAL_SIMULATOR, config)
        backend = manager.select_backend()
        
        circuit = {
            "num_qubits": num_qubits,
            "operations": []
        }
        
        # Add layers of gates
        for layer in range(num_layers):
            for qubit in range(num_qubits):
                circuit["operations"].append({"type": "ry", "qubits": [qubit], "params": [np.random.rand() * 2 * np.pi]})
            for qubit in range(num_qubits - 1):
                circuit["operations"].append({"type": "cx", "qubits": [qubit, qubit + 1]})
        
        benchmark = BenchmarkResult(name="quantum_circuit", iterations=num_iterations)
        
        for _ in range(num_iterations):
            start = time.time()
            result = backend.execute_circuit(circuit, num_shots=100)
            duration_ms = (time.time() - start) * 1000
            
            benchmark.add_timing(duration_ms, result)
        
        self.benchmarks.append(benchmark)
        return benchmark
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks and return summary."""
        logger.info("Running quantum RL benchmarks...")
        
        results = {}
        
        # State encoding benchmark
        encoding_result = self.benchmark_state_encoding()
        results["state_encoding"] = encoding_result.summary()
        logger.info("State encoding: %.3fms mean", encoding_result.mean_time_ms)
        
        # Action selection benchmark
        action_result = self.benchmark_action_selection()
        results["action_selection"] = action_result.summary()
        logger.info("Action selection: %.3fms mean", action_result.mean_time_ms)
        
        # Quantum circuit benchmark
        circuit_result = self.benchmark_quantum_circuit()
        results["quantum_circuit"] = circuit_result.summary()
        logger.info("Quantum circuit: %.3fms mean", circuit_result.mean_time_ms)
        
        return results


# ============================================================================
# Test Utilities
# ============================================================================

class MockEnvironment:
    """Mock environment for testing."""
    
    def __init__(self, state_dim: int = 8, action_dim: int = 4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.step_count = 0
        self.max_steps = 100
    
    def reset(self) -> NDArray[np.float64]:
        """Reset environment."""
        self.step_count = 0
        return np.random.randn(self.state_dim)
    
    def step(self, action: int) -> Tuple[NDArray[np.float64], float, bool, Dict[str, Any]]:
        """Take a step."""
        self.step_count += 1
        
        next_state = np.random.randn(self.state_dim)
        reward = np.random.randn()
        done = self.step_count >= self.max_steps or np.random.rand() < 0.05
        
        return next_state, reward, done, {"step": self.step_count}


def run_all_tests() -> Dict[str, Any]:
    """Run all tests and return results."""
    runner = QuantumRLTestRunner()
    
    # Run unit tests
    unit_tests = [
        QuantumComponentTests.test_quantum_state_encoder,
        QuantumComponentTests.test_quantum_action_selector,
        QuantumComponentTests.test_quantum_reward_processor,
        QuantumComponentTests.test_quantum_replay_buffer,
        QuantumComponentTests.test_market_state_encoder,
        QuantumComponentTests.test_action_decoder,
        QuantumComponentTests.test_risk_manager
    ]
    unit_suite = runner.run_test_suite("unit_tests", unit_tests)
    
    # Run integration tests
    integration_tests = [
        IntegrationTests.test_qql_integration,
        IntegrationTests.test_qdqn_integration,
        IntegrationTests.test_hybrid_rl_integration,
        IntegrationTests.test_quantum_backend_manager
    ]
    integration_suite = runner.run_test_suite("integration_tests", integration_tests)
    
    return runner.generate_report()


__all__ = [
    # Test framework
    "QuantumRLTestRunner",
    "TestResult",
    "TestSuite",
    
    # Test classes
    "QuantumComponentTests",
    "IntegrationTests",
    
    # Benchmarking
    "QuantumRLBenchmark",
    "BenchmarkResult",
    
    # Utilities
    "MockEnvironment",
    "run_all_tests"
]