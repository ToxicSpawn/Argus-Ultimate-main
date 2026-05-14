# pyright: reportMissingImports=false
"""
Comprehensive tests for quantum reinforcement learning components.

This module provides tests for all quantum RL algorithms and integration tests
to ensure proper functionality and quantum advantage validation.
"""

from __future__ import annotations

import logging
import random
import unittest
from typing import Any, Dict, List

import numpy as np
import pytest

logger = logging.getLogger(__name__)


class TestQuantumQLearning(unittest.TestCase):
    """Tests for Quantum Q-Learning."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from quantum.advanced.quantum_q_learning import QuantumQLearning, QQLParameters
            self.qql_class = QuantumQLearning
            self.qql_params = QQLParameters
        except ImportError:
            self.skipTest("Quantum Q-Learning module not available")

    def test_initialization(self):
        """Test QQL initialization."""
        params = self.qql_params(
            state_dim=4,
            action_dim=2,
            qubits=4,
            episodes=10
        )
        qql = self.qql_class(params)
        self.assertIsNotNone(qql)
        self.assertEqual(qql.parameters.state_dim, 4)
        self.assertEqual(qql.parameters.action_dim, 2)

    def test_state_encoding(self):
        """Test state encoding."""
        params = self.qql_params(state_dim=4, action_dim=2, qubits=4, episodes=10)
        qql = self.qql_class(params)
        state = np.array([0.1, 0.5, 0.8, 0.3])
        encoded = qql.encode_state(state)
        self.assertIsInstance(encoded, (int, np.integer))
        self.assertGreaterEqual(encoded, 0)

    def test_action_selection(self):
        """Test action selection."""
        params = self.qql_params(state_dim=4, action_dim=2, qubits=4, episodes=10)
        qql = self.qql_class(params)
        state = np.array([0.1, 0.5, 0.8, 0.3])
        action, metadata = qql.select_action(state)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 2)
        self.assertIn("q_values", metadata)

    def test_experience_storage(self):
        """Test experience storage."""
        from quantum.advanced.quantum_q_learning import QQLExperience
        
        params = self.qql_params(state_dim=4, action_dim=2, qubits=4, episodes=10)
        qql = self.qql_class(params)
        
        experience = QQLExperience(
            state=np.array([0.1, 0.5, 0.8, 0.3]),
            action=0,
            reward=0.5,
            next_state=np.array([0.2, 0.6, 0.9, 0.4]),
            done=False
        )
        
        initial_size = len(qql.experience_replay)
        qql.store_experience(experience)
        self.assertEqual(len(qql.experience_replay), initial_size + 1)


class TestQuantumDeepQNetwork(unittest.TestCase):
    """Tests for Quantum Deep Q-Network."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from quantum.advanced.quantum_deep_q_network import QuantumDeepQNetwork, QDQNParameters
            self.qdqn_class = QuantumDeepQNetwork
            self.qdqn_params = QDQNParameters
        except ImportError:
            self.skipTest("Quantum Deep Q-Network module not available")

    def test_initialization(self):
        """Test QDQN initialization."""
        params = self.qdqn_params(
            state_dim=4,
            action_dim=2,
            qubits=4,
            episodes=10
        )
        qdqn = self.qdqn_class(params)
        self.assertIsNotNone(qdqn)
        self.assertEqual(qdqn.parameters.state_dim, 4)
        self.assertEqual(qdqn.parameters.action_dim, 2)

    def test_network_forward(self):
        """Test network forward pass."""
        from quantum.advanced.quantum_deep_q_network import QuantumNeuralNetwork
        
        qnn = QuantumNeuralNetwork(qubits=4, layers=2, action_dim=2)
        state = np.array([0.1, 0.5, 0.8, 0.3])
        q_values = qnn.forward(state)
        
        self.assertEqual(len(q_values), 2)
        self.assertTrue(np.all(np.isfinite(q_values)))

    def test_action_selection(self):
        """Test action selection."""
        params = self.qdqn_params(state_dim=4, action_dim=2, qubits=4, episodes=10)
        qdqn = self.qdqn_class(params)
        state = np.array([0.1, 0.5, 0.8, 0.3])
        action, metadata = qdqn.select_action(state)
        
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 2)
        self.assertIn("q_values", metadata)


class TestQuantumPolicyGradient(unittest.TestCase):
    """Tests for Quantum Policy Gradient."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from quantum.advanced.quantum_policy_gradient import QuantumPolicyGradient, QPGParameters
            self.qpg_class = QuantumPolicyGradient
            self.qpg_params = QPGParameters
        except ImportError:
            self.skipTest("Quantum Policy Gradient module not available")

    def test_initialization(self):
        """Test QPG initialization."""
        params = self.qpg_params(
            state_dim=4,
            action_dim=2,
            qubits=4,
            episodes=10
        )
        qpg = self.qpg_class(params)
        self.assertIsNotNone(qpg)
        self.assertEqual(qpg.parameters.state_dim, 4)
        self.assertEqual(qpg.parameters.action_dim, 2)

    def test_policy_network(self):
        """Test policy network forward pass."""
        from quantum.advanced.quantum_policy_gradient import QuantumPolicyNetwork
        
        policy = QuantumPolicyNetwork(qubits=4, layers=2, action_dim=2)
        state = np.array([0.1, 0.5, 0.8, 0.3])
        action_probs, entropy = policy.forward(state)
        
        self.assertEqual(len(action_probs), 2)
        self.assertAlmostEqual(np.sum(action_probs), 1.0, places=5)
        self.assertGreaterEqual(entropy, 0.0)

    def test_action_selection(self):
        """Test action selection."""
        from quantum.advanced.quantum_policy_gradient import QuantumPolicyNetwork
        
        policy = QuantumPolicyNetwork(qubits=4, layers=2, action_dim=2)
        state = np.array([0.1, 0.5, 0.8, 0.3])
        action, log_prob, metadata = policy.select_action(state)
        
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 2)
        self.assertLess(log_prob, 0)  # log probability should be negative


class TestQuantumActorCritic(unittest.TestCase):
    """Tests for Quantum Actor-Critic."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from quantum.advanced.quantum_actor_critic import QuantumActorCritic, QACParameters
            self.qac_class = QuantumActorCritic
            self.qac_params = QACParameters
        except ImportError:
            self.skipTest("Quantum Actor-Critic module not available")

    def test_initialization(self):
        """Test QAC initialization."""
        params = self.qac_params(
            state_dim=4,
            action_dim=2,
            actor_qubits=4,
            critic_qubits=4,
            episodes=10
        )
        qac = self.qac_class(params)
        self.assertIsNotNone(qac)
        self.assertEqual(qac.parameters.state_dim, 4)
        self.assertEqual(qac.parameters.action_dim, 2)

    def test_actor_network(self):
        """Test actor network forward pass."""
        from quantum.advanced.quantum_actor_critic import QuantumActorNetwork
        
        actor = QuantumActorNetwork(qubits=4, layers=2, action_dim=2)
        state = np.array([0.1, 0.5, 0.8, 0.3])
        action_probs, entropy = actor.forward(state)
        
        self.assertEqual(len(action_probs), 2)
        self.assertAlmostEqual(np.sum(action_probs), 1.0, places=5)
        self.assertGreaterEqual(entropy, 0.0)

    def test_critic_network(self):
        """Test critic network forward pass."""
        from quantum.advanced.quantum_actor_critic import QuantumCriticNetwork
        
        critic = QuantumCriticNetwork(qubits=4, layers=2)
        state = np.array([0.1, 0.5, 0.8, 0.3])
        value = critic.forward(state)
        
        self.assertIsInstance(value, (float, np.floating))


class TestQuantumAdvantageValidation(unittest.TestCase):
    """Tests for quantum advantage validation."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from quantum.advanced.quantum_q_learning import QuantumQLearning, QQLParameters
            self.qql_class = QuantumQLearning
            self.qql_params = QQLParameters
        except ImportError:
            self.skipTest("Quantum Q-Learning module not available")

    def test_quantum_advantage_threshold(self):
        """Test quantum advantage validation with threshold."""
        params = self.qql_params(
            state_dim=4,
            action_dim=2,
            qubits=4,
            episodes=10,
            quantum_advantage_threshold=0.05
        )
        qql = self.qql_class(params)
        
        # Simulate some performance history
        from quantum.advanced.quantum_q_learning import QQLPerformance
        for i in range(25):
            performance = QQLPerformance(
                episode=i,
                step=10,
                reward=random.uniform(-1, 1),
                cumulative_reward=random.uniform(-5, 5),
                exploration_rate=0.1,
                q_value_avg=random.uniform(0, 1)
            )
            qql.session.performance_history.append(performance)
        
        advantage = qql.validate_quantum_advantage()
        self.assertIsInstance(advantage, float)


class TestIntegration(unittest.TestCase):
    """Integration tests for quantum RL system."""

    def test_multiple_algorithms(self):
        """Test that multiple algorithms can be initialized and run."""
        algorithms = []
        
        try:
            from quantum.advanced.quantum_q_learning import QuantumQLearning, QQLParameters
            params = QQLParameters(state_dim=4, action_dim=2, qubits=4, episodes=5)
            algorithms.append(("QQL", QuantumQLearning(params)))
        except ImportError:
            pass
        
        try:
            from quantum.advanced.quantum_deep_q_network import QuantumDeepQNetwork, QDQNParameters
            params = QDQNParameters(state_dim=4, action_dim=2, qubits=4, episodes=5)
            algorithms.append(("QDQN", QuantumDeepQNetwork(params)))
        except ImportError:
            pass
        
        try:
            from quantum.advanced.quantum_policy_gradient import QuantumPolicyGradient, QPGParameters
            params = QPGParameters(state_dim=4, action_dim=2, qubits=4, episodes=5)
            algorithms.append(("QPG", QuantumPolicyGradient(params)))
        except ImportError:
            pass
        
        try:
            from quantum.advanced.quantum_actor_critic import QuantumActorCritic, QACParameters
            params = QACParameters(state_dim=4, action_dim=2, actor_qubits=4, critic_qubits=4, episodes=5)
            algorithms.append(("QAC", QuantumActorCritic(params)))
        except ImportError:
            pass
        
        self.assertGreater(len(algorithms), 0, "At least one algorithm should be available")
        
        for name, algorithm in algorithms:
            self.assertIsNotNone(algorithm)
            logger.info(f"Successfully initialized {name}")

    def test_visualization_data(self):
        """Test visualization data generation."""
        try:
            from quantum.advanced.quantum_q_learning import QuantumQLearning, QQLParameters
            params = QQLParameters(state_dim=4, action_dim=2, qubits=4, episodes=5)
            qql = QuantumQLearning(params)
            
            # Add some performance data
            from quantum.advanced.quantum_q_learning import QQLPerformance
            for i in range(5):
                performance = QQLPerformance(
                    episode=i,
                    step=10,
                    reward=random.uniform(-1, 1),
                    cumulative_reward=random.uniform(-5, 5),
                    exploration_rate=0.1,
                    q_value_avg=random.uniform(0, 1)
                )
                qql.session.performance_history.append(performance)
            
            viz_data = qql.get_visualization_data()
            self.assertIn("status", viz_data)
            self.assertIn("performance", viz_data)
            
        except ImportError:
            self.skipTest("Quantum Q-Learning module not available")


def run_tests():
    """Run all tests."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(unittest.makeSuite(TestQuantumQLearning))
    suite.addTest(unittest.makeSuite(TestQuantumDeepQNetwork))
    suite.addTest(unittest.makeSuite(TestQuantumPolicyGradient))
    suite.addTest(unittest.makeSuite(TestQuantumActorCritic))
    suite.addTest(unittest.makeSuite(TestQuantumAdvantageValidation))
    suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests when executed directly
    success = run_tests()
    exit(0 if success else 1)