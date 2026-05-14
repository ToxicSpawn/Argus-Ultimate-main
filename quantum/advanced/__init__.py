# pyright: reportMissingImports=false
"""Advanced quantum components and utilities."""

from quantum.advanced.advanced_quantum_circuit_optimization import (
    AdvancedQuantumCircuitOptimizer,
    CircuitOptimizationStep,
    CircuitProfilingReport,
    OptimizationObjective,
    QuantumCircuitMetrics,
    QuantumCircuitProfile,
    QuantumGateOperation,
    QuantumHardwareType,
    QubitAllocation,
)

# Quantum Reinforcement Learning - lazy imports to avoid circular dependencies
try:
    from quantum.advanced.quantum_reinforcement_learning import (
        QuantumReinforcementLearning,
        QuantumRLAlgorithm,
        QuantumRLParameters,
        QuantumRLState,
        QuantumRLStateRepresentation,
        QuantumRLAction,
        QuantumRLExperience,
        QuantumRLPerformance,
        QuantumRLTrainingSession,
    )
except ImportError:
    pass

try:
    from quantum.advanced.quantum_q_learning import (
        QuantumQLearning,
        QQLParameters,
        QQLState,
        QQLExperience,
        QQLPerformance,
        QQLTrainingSession,
        QuantumQLearningBackend,
    )
except ImportError:
    pass

try:
    from quantum.advanced.quantum_deep_q_network import (
        QuantumDeepQNetwork,
        QDQNParameters,
        QDQNState,
        QDQNExperience,
        QDQNPerformance,
        QDQNTrainingSession,
        QDQNBackend,
        QDQNEncodingMethod,
        QuantumNeuralNetwork,
        QuantumDuelingNetwork,
    )
except ImportError:
    pass

try:
    from quantum.advanced.quantum_policy_gradient import (
        QuantumPolicyGradient,
        QPGParameters,
        QPGState,
        QPGTransition,
        QPGPerformance,
        QPGTrainingSession,
        QPGBackend,
        QPGMethod,
        QuantumPolicyNetwork,
        QuantumValueNetwork,
    )
except ImportError:
    pass

try:
    from quantum.advanced.quantum_actor_critic import (
        QuantumActorCritic,
        QACParameters,
        QACState,
        QACTransition,
        QACPerformance,
        QACTrainingSession,
        QACBackend,
        QuantumActorNetwork,
        QuantumCriticNetwork,
    )
except ImportError:
    pass

__all__ = [
    # Circuit Optimization
    "AdvancedQuantumCircuitOptimizer",
    "CircuitOptimizationStep",
    "CircuitProfilingReport",
    "OptimizationObjective",
    "QuantumCircuitMetrics",
    "QuantumCircuitProfile",
    "QuantumGateOperation",
    "QuantumHardwareType",
    "QubitAllocation",
    # Quantum Reinforcement Learning (main module)
    "QuantumReinforcementLearning",
    "QuantumRLAlgorithm",
    "QuantumRLParameters",
    "QuantumRLState",
    "QuantumRLStateRepresentation",
    "QuantumRLAction",
    "QuantumRLExperience",
    "QuantumRLPerformance",
    "QuantumRLTrainingSession",
    # Quantum Q-Learning
    "QuantumQLearning",
    "QQLParameters",
    "QQLState",
    "QQLExperience",
    "QQLPerformance",
    "QQLTrainingSession",
    "QuantumQLearningBackend",
    # Quantum Deep Q-Network
    "QuantumDeepQNetwork",
    "QDQNParameters",
    "QDQNState",
    "QDQNExperience",
    "QDQNPerformance",
    "QDQNTrainingSession",
    "QDQNBackend",
    "QDQNEncodingMethod",
    "QuantumNeuralNetwork",
    "QuantumDuelingNetwork",
    # Quantum Policy Gradient
    "QuantumPolicyGradient",
    "QPGParameters",
    "QPGState",
    "QPGTransition",
    "QPGPerformance",
    "QPGTrainingSession",
    "QPGBackend",
    "QPGMethod",
    "QuantumPolicyNetwork",
    "QuantumValueNetwork",
    # Quantum Actor-Critic
    "QuantumActorCritic",
    "QACParameters",
    "QACState",
    "QACTransition",
    "QACPerformance",
    "QACTrainingSession",
    "QACBackend",
    "QuantumActorNetwork",
    "QuantumCriticNetwork",
]