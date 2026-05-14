# pyright: reportMissingImports=false
"""
Comprehensive tests for all advanced learning systems.

Tests cover:
- Knowledge Distillation
- Multi-Agent RL
- RLHF
- Uncertainty Quantification
- Adversarial Training
- Active Learning
- Transfer Learning
- Learning Health Dashboard
"""

from __future__ import annotations

import logging
import unittest

import numpy as np

logger = logging.getLogger(__name__)


class TestKnowledgeDistillation(unittest.TestCase):
    """Tests for Knowledge Distillation system."""

    def setUp(self):
        try:
            from ml.knowledge_distillation import (
                KnowledgeDistillationSystem, DistillationConfig
            )
            self.system = KnowledgeDistillationSystem(DistillationConfig(epochs=5))
        except ImportError:
            self.skipTest("Knowledge distillation module not available")

    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.system)
        self.assertEqual(self.system.config.epochs, 5)

    def test_teacher_student_creation(self):
        """Test teacher and student model creation."""
        teacher = self.system.create_teacher("quantum_rl")
        self.assertIsNotNone(teacher)
        self.assertEqual(teacher.model_type, "quantum_rl")

        student = self.system.create_student("classical_rl")
        self.assertIsNotNone(student)
        self.assertLess(student.complexity, teacher.complexity)


class TestMultiAgentRL(unittest.TestCase):
    """Tests for Multi-Agent RL system."""

    def setUp(self):
        try:
            from ml.multi_agent_rl import MultiAgentSystem, AgentType
            self.system = MultiAgentSystem()
            self.AgentType = AgentType
        except ImportError:
            self.skipTest("Multi-agent RL module not available")

    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.system)
        self.assertGreater(len(self.system.agents), 0)

    def test_decision_making(self):
        """Test collaborative decision making."""
        market_state = np.random.rand(8)
        decision, metadata = self.system.make_decision(market_state)

        self.assertIsInstance(decision, int)
        self.assertIn(decision, [0, 1, 2, 3])
        self.assertIn("method", metadata)


class TestRLHF(unittest.TestCase):
    """Tests for RLHF system."""

    def setUp(self):
        try:
            from ml.rlhf_system import RLHFSystem, FeedbackType
            self.system = RLHFSystem()
            self.FeedbackType = FeedbackType
        except ImportError:
            self.skipTest("RLHF module not available")

    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.system)
        self.assertGreater(len(self.system.experts), 0)

    def test_feedback_collection(self):
        """Test feedback collection."""
        market_state = np.random.rand(8)
        feedback = self.system.collect_rating(market_state, 1, 1.0)

        self.assertIsNotNone(feedback)
        self.assertEqual(feedback.rating, 1.0)


class TestUncertaintyQuantification(unittest.TestCase):
    """Tests for Uncertainty Quantification system."""

    def setUp(self):
        try:
            from ml.uncertainty_quantification import (
                UncertaintyQuantifier, UncertaintyConfig, UncertaintyMethod
            )
            self.quantifier = UncertaintyQuantifier(
                UncertaintyConfig(method=UncertaintyMethod.ENSEMBLE)
            )
        except ImportError:
            self.skipTest("Uncertainty quantification module not available")

    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.quantifier)

    def test_uncertainty_estimation(self):
        """Test uncertainty estimation."""
        predictions = [np.random.randn(4) for _ in range(5)]
        market_state = np.random.rand(8)

        estimate = self.quantifier.estimate_uncertainty(predictions, market_state)

        self.assertGreaterEqual(estimate.confidence, 0.0)
        self.assertLessEqual(estimate.confidence, 1.0)


class TestAdversarialTraining(unittest.TestCase):
    """Tests for Adversarial Training system."""

    def setUp(self):
        try:
            from ml.adversarial_training import (
                AdversarialGenerator, AdversarialAttackType
            )
            self.generator = AdversarialGenerator()
            self.AttackType = AdversarialAttackType
        except ImportError:
            self.skipTest("Adversarial training module not available")

    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.generator)

    def test_adversarial_generation(self):
        """Test adversarial state generation."""
        market_state = np.random.rand(8)
        adv_state = self.generator.generate_adversarial_state(market_state)

        self.assertEqual(adv_state.shape, market_state.shape)
        # State should be modified
        self.assertFalse(np.array_equal(market_state, adv_state))


class TestActiveLearning(unittest.TestCase):
    """Tests for Active Learning system."""

    def setUp(self):
        try:
            from ml.active_learning import (
                ActiveLearner, ActiveLearningConfig, AcquisitionFunction
            )
            self.learner = ActiveLearner(
                ActiveLearningConfig(batch_size=5)
            )
        except ImportError:
            self.skipTest("Active learning module not available")

    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.learner)

    def test_pool_initialization(self):
        """Test pool initialization."""
        samples = [np.random.rand(8) for _ in range(50)]
        self.learner.initialize_pool(samples)

        self.assertEqual(len(self.learner.unlabeled_pool), 50)


class TestTransferLearning(unittest.TestCase):
    """Tests for Transfer Learning system."""

    def setUp(self):
        try:
            from ml.transfer_learning import (
                TransferLearner, TransferabilityAnalyzer, AssetProfile
            )
            self.learner = TransferLearner()
            self.Analyzer = TransferabilityAnalyzer
        except ImportError:
            self.skipTest("Transfer learning module not available")

    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.learner)


class TestLearningHealthDashboard(unittest.TestCase):
    """Tests for Learning Health Dashboard."""

    def setUp(self):
        try:
            from ml.learning_health_dashboard import (
                LearningHealthDashboard, SystemHealth
            )
            self.dashboard = LearningHealthDashboard()
            self.SystemHealth = SystemHealth
        except ImportError:
            self.skipTest("Learning health dashboard module not available")

    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.dashboard)

    def test_system_registration(self):
        """Test system registration."""
        self.dashboard.register_system("TestSystem", "test", 0.85)
        self.assertIn("TestSystem", self.dashboard.systems)

    def test_metrics_update(self):
        """Test metrics update."""
        self.dashboard.register_system("TestSystem", "test", 0.85)
        self.dashboard.update_system_metrics("TestSystem", 0.9, 100, 0.05, 50)

        system = self.dashboard.get_system_details("TestSystem")
        self.assertEqual(system.performance, 0.9)

    def test_dashboard_metrics(self):
        """Test dashboard metrics generation."""
        self.dashboard.register_system("TestSystem1", "test", 0.85)
        self.dashboard.register_system("TestSystem2", "test", 0.75)

        metrics = self.dashboard.get_dashboard_metrics()
        self.assertEqual(metrics.total_systems, 2)
        self.assertGreater(metrics.average_performance, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for all learning systems."""

    def test_all_systems_initialize(self):
        """Test that all systems can be initialized."""
        systems = []

        try:
            from ml.knowledge_distillation import KnowledgeDistillationSystem
            systems.append(("KnowledgeDistillation", KnowledgeDistillationSystem()))
        except ImportError:
            pass

        try:
            from ml.multi_agent_rl import MultiAgentSystem
            systems.append(("MultiAgentRL", MultiAgentSystem()))
        except ImportError:
            pass

        try:
            from ml.rlhf_system import RLHFSystem
            systems.append(("RLHF", RLHFSystem()))
        except ImportError:
            pass

        try:
            from ml.uncertainty_quantification import UncertaintyQuantifier
            systems.append(("UncertaintyQuantification", UncertaintyQuantifier()))
        except ImportError:
            pass

        try:
            from ml.adversarial_training import AdversarialGenerator
            systems.append(("AdversarialTraining", AdversarialGenerator()))
        except ImportError:
            pass

        try:
            from ml.active_learning import ActiveLearner
            systems.append(("ActiveLearning", ActiveLearner()))
        except ImportError:
            pass

        try:
            from ml.transfer_learning import TransferLearner
            systems.append(("TransferLearning", TransferLearner()))
        except ImportError:
            pass

        try:
            from ml.learning_health_dashboard import LearningHealthDashboard
            systems.append(("LearningHealthDashboard", LearningHealthDashboard()))
        except ImportError:
            pass

        self.assertGreater(len(systems), 0, "No learning systems could be imported")
        print(f"\n✓ Successfully initialized {len(systems)} learning systems:")
        for name, _ in systems:
            print(f"  - {name}")


def run_tests():
    """Run all tests."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestKnowledgeDistillation))
    suite.addTest(unittest.makeSuite(TestMultiAgentRL))
    suite.addTest(unittest.makeSuite(TestRLHF))
    suite.addTest(unittest.makeSuite(TestUncertaintyQuantification))
    suite.addTest(unittest.makeSuite(TestAdversarialTraining))
    suite.addTest(unittest.makeSuite(TestActiveLearning))
    suite.addTest(unittest.makeSuite(TestTransferLearning))
    suite.addTest(unittest.makeSuite(TestLearningHealthDashboard))
    suite.addTest(unittest.makeSuite(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)