# pyright: reportMissingImports=false
"""
Quantum Reinforcement Learning Module for Argus Trading System.

This module provides a comprehensive quantum-enhanced reinforcement learning
framework for trading strategy optimization, including:

- Quantum Q-Learning (QQL)
- Quantum Deep Q-Networks (QDQN)
- Quantum Policy Gradient (QPG)
- Quantum Actor-Critic (QAC)
- Hybrid Quantum-Classical RL

Core Components:
- Quantum state representation and encoding
- Quantum action selection
- Quantum reward processing
- Quantum memory (experience replay)

Trading Integration:
- Market state encoder
- Action decoder for trading
- Reward function for trading
- Risk management integration
- Market data feed integration
- Execution engine integration
- Portfolio management integration

Infrastructure:
- Quantum backend support (Local, IBM, Rigetti, IonQ)
- Quantum advantage validation
- Performance metrics tracking
- Testing framework
- Benchmarking tools

Example Usage:
    from quantum.reinforcement_learning import (
        QuantumQLearning,
        QQLConfig,
        MarketStateEncoder,
        ActionDecoder,
        QuantumRLTradingOrchestrator
    )
    
    # Create agent
    config = QQLConfig(num_qubits=4, num_layers=3)
    agent = QuantumQLearning(state_dim=8, action_dim=4, config=config)
    
    # Create trading components
    state_encoder = MarketStateEncoder()
    action_decoder = ActionDecoder()
    
    # Create orchestrator
    orchestrator = QuantumRLTradingOrchestrator()
    orchestrator.set_agent(agent)
    orchestrator.set_components(state_encoder, action_decoder)
    
    # Start trading (async)
    # await orchestrator.start_trading("BTC/USDT")
"""

# Core RL Algorithms
from .quantum_q_learning import (
    QuantumQLearning,
    QQLConfig,
    QuantumState,
    QuantumStateEncoding,
    ExplorationStrategy,
    QQLMetrics,
    Experience
)

from .quantum_deep_q_network import (
    QuantumDeepQNetwork,
    QDQNConfig,
    QuantumNeuralNetwork,
    VariationalQuantumCircuit,
    PrioritizedReplayBuffer
)

from .quantum_policy_gradient import (
    QuantumPolicyGradient,
    QPGConfig,
    QuantumPolicyNetwork,
    QuantumValueNetwork,
    PolicyType,
    AdvantageEstimation,
    Episode as PGEpisode
)

from .quantum_actor_critic import (
    QuantumActorCritic,
    QACConfig,
    QuantumActorNetwork,
    QuantumCriticNetwork,
    ActorCriticType,
    Trajectory
)

from .hybrid_quantum_classical_rl import (
    HybridQuantumClassicalRL,
    HybridRLConfig,
    HybridArchitecture,
    ClassicalPolicyNetwork,
    ClassicalValueNetwork,
    QuantumFeatureExtractor,
    QuantumAdvantageEstimator
)

# Core Utilities
from .quantum_rl_utils import (
    QuantumState as CoreQuantumState,
    QuantumStateEncoder,
    StateEncoding,
    QuantumAction,
    QuantumActionSelector,
    ActionSelectionMethod,
    QuantumReward,
    QuantumRewardProcessor,
    RewardShapingMethod,
    QuantumExperience,
    QuantumReplayBuffer,
    QuantumStateCache
)

# Trading Components
from .trading_components import (
    MarketStateEncoder,
    MarketStateConfig,
    FeatureType,
    ActionDecoder,
    ActionConfig,
    TradingAction,
    TradingRewardFunction,
    RewardConfig,
    RewardType,
    RiskManager,
    RiskLimits
)

# Integration
from .trading_integration import (
    MarketDataFeed,
    MarketDataConfig,
    DataSource,
    ExecutionEngine,
    ExecutionConfig,
    Order,
    OrderType,
    OrderSide,
    PortfolioManager,
    PortfolioConfig,
    QuantumRLTradingOrchestrator
)

# Quantum Backends
from .quantum_backends import (
    QuantumBackendType,
    BackendConfig,
    QuantumCircuitResult,
    QuantumBackend,
    LocalSimulator,
    IBMQuantumBackend,
    RigettiBackend,
    IonQBackend,
    QuantumBackendManager
)

# Validation and Testing
from .quantum_validation import (
    QuantumAdvantageValidator,
    ValidationConfig,
    ValidationMetric,
    PerformanceMetrics,
    ComparisonResult,
    PerformanceTracker,
    ClassicalBaseline
)

from .testing_framework import (
    QuantumRLTestRunner,
    TestResult,
    TestSuite,
    QuantumComponentTests,
    IntegrationTests,
    QuantumRLBenchmark,
    BenchmarkResult,
    MockEnvironment,
    run_all_tests
)

# Classical-Quantum Interface
from .classical_quantum_interface import (
    ClassicalStatePreprocessor,
    PreprocessingConfig,
    NormalizationMethod,
    ClassicalActionPostprocessor,
    PostprocessingConfig,
    ActionMapping,
    HybridTrainingLoop,
    TrainingConfig,
    TrainingPhase,
    QuantumClassicalFallback,
    FallbackReason,
    QuantumErrorHandler
)

__all__ = [
    # Core RL Algorithms
    "QuantumQLearning",
    "QQLConfig",
    "QuantumDeepQNetwork",
    "QDQNConfig",
    "QuantumPolicyGradient",
    "QPGConfig",
    "QuantumActorCritic",
    "QACConfig",
    "HybridQuantumClassicalRL",
    "HybridRLConfig",
    
    # Core Utilities
    "QuantumStateEncoder",
    "StateEncoding",
    "QuantumActionSelector",
    "ActionSelectionMethod",
    "QuantumRewardProcessor",
    "QuantumReplayBuffer",
    
    # Trading Components
    "MarketStateEncoder",
    "ActionDecoder",
    "TradingRewardFunction",
    "RiskManager",
    
    # Integration
    "MarketDataFeed",
    "ExecutionEngine",
    "PortfolioManager",
    "QuantumRLTradingOrchestrator",
    
    # Quantum Backends
    "QuantumBackendType",
    "QuantumBackendManager",
    "LocalSimulator",
    
    # Validation
    "QuantumAdvantageValidator",
    "PerformanceMetrics",
    "PerformanceTracker",
    "ClassicalBaseline",
    
    # Testing
    "QuantumRLTestRunner",
    "QuantumRLBenchmark",
    "run_all_tests",
    
    # Interface
    "ClassicalStatePreprocessor",
    "ClassicalActionPostprocessor",
    "HybridTrainingLoop",
    "QuantumClassicalFallback"
]

__version__ = "1.0.0"
__author__ = "Argus Quantum RL Team"
