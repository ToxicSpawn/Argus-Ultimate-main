"""
Test cases for the Real-Time Learning Orchestrator
"""

import unittest
from unittest.mock import MagicMock
from datetime import datetime, timezone, timedelta
from core.real_time_learning.orchestrator import (
    RealTimeLearningOrchestrator,
    LearningComponent,
    SafetyValidator,
    LearningAudit
)


class TestLearningComponent(LearningComponent):
    """Test component implementation"""

    def __init__(self, name: str):
        self.name = name
        self.params = {"test_param": 1.0}
        self.update_count = 0

    def learn(self, data: dict) -> dict:
        self.update_count += 1
        self.params["test_param"] += 0.1
        self.last_updated = datetime.now(timezone.utc)
        return self.params

    def get_params(self) -> dict:
        return self.params

    def rollback(self) -> None:
        self.params["test_param"] = max(1.0, self.params["test_param"] - 0.1)

    def validate(self, new_params: dict) -> bool:
        return new_params["test_param"] <= 2.0  # Don't allow > 2.0


class TestRealTimeLearning(unittest.TestCase):
    """Test the real-time learning orchestrator"""

    def setUp(self):
        self.orchestrator = RealTimeLearningOrchestrator()
        self.component1 = TestLearningComponent("test1")
        self.component2 = TestLearningComponent("test2")
        self.orchestrator.register_component(self.component1)
        self.orchestrator.register_component(self.component2)

    def test_component_registration(self):
        """Test component registration"""
        self.assertEqual(len(self.orchestrator.components), 2)
        self.assertIn("test1", self.orchestrator.components)
        self.assertIn("test2", self.orchestrator.components)

    def test_learning_cycle(self):
        """Test basic learning cycle"""
        # Set different update frequencies
        self.component1.update_frequency = 1  # Update every cycle
        self.component2.update_frequency = 2  # Update every 2 cycles

        # First cycle - only component1 updates
        results = self.orchestrator.on_market_data({"test": "data"})
        self.assertEqual(len(results), 1)
        self.assertIn("test1", results)
        self.assertEqual(self.component1.update_count, 1)
        self.assertEqual(self.component2.update_count, 0)

        # Second cycle - both update
        results = self.orchestrator.on_market_data({"test": "data"})
        self.assertEqual(len(results), 2)
        self.assertIn("test1", results)
        self.assertIn("test2", results)
        self.assertEqual(self.component1.update_count, 2)
        self.assertEqual(self.component2.update_count, 1)

    def test_safety_validation(self):
        """Test safety validation blocks invalid changes"""
        # Set a validation rule that blocks changes > 2.0
        self.component1.params["test_param"] = 1.9  # Just below limit

        # This update should be allowed (2.0)
        results = self.orchestrator.on_market_data({"test": "data"})
        self.assertEqual(results["test1"]["status"], "success")
        self.assertEqual(self.component1.params["test_param"], 2.0)

        # Next update would exceed limit (2.1) - should be blocked
        results = self.orchestrator.on_market_data({"test": "data"})
        self.assertEqual(results["test1"]["status"], "success")  # Rollback happens
        self.assertEqual(self.component1.params["test_param"], 2.0)  # Should stay at 2.0

    def test_rollback_functionality(self):
        """Test manual rollback"""
        # Make a change
        self.orchestrator.on_market_data({"test": "data"})
        original_value = self.component1.params["test_param"]

        # Manually rollback
        self.orchestrator.rollback_component("test1")
        self.assertLess(self.component1.params["test_param"], original_value)

    def test_pause_resume(self):
        """Test pausing and resuming learning"""
        # Pause a component
        self.orchestrator.pause_learning("test1")
        self.assertIn("test1", self.orchestrator.paused_components)

        # Verify it doesn't update
        results = self.orchestrator.on_market_data({"test": "data"})
        self.assertNotIn("test1", results)

        # Resume
        self.orchestrator.resume_learning("test1")
        self.assertNotIn("test1", self.orchestrator.paused_components)

        # Verify it updates again
        results = self.orchestrator.on_market_data({"test": "data"})
        self.assertIn("test1", results)

    def test_audit_trail(self):
        """Test audit trail logging"""
        # Make a change
        self.orchestrator.on_market_data({"test": "data"})

        # Check audit trail
        changes = self.orchestrator.audit.get_recent_changes(hours=1)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["component"], "test1")
        self.assertEqual(changes[0]["changes"]["test_param"], 1.1)

    def test_emergency_shutdown(self):
        """Test emergency shutdown"""
        # Make some changes
        self.orchestrator.on_market_data({"test": "data"})
        original_params = self.component1.get_params()

        # Trigger emergency shutdown
        self.orchestrator.emergency_shutdown()

        # Verify all components paused
        self.assertEqual(len(self.orchestrator.paused_components), 2)

        # Verify rollback happened
        new_params = self.component1.get_params()
        self.assertLess(new_params["test_param"], original_params["test_param"])

    def test_state_save_load(self):
        """Test saving and loading state"""
        import tempfile
        import json

        # Make some changes
        self.orchestrator.on_market_data({"test": "data"})
        original_params = self.component1.get_params()
        original_cycle = self.orchestrator.cycle_count

        # Save state
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            test_file = f.name

        self.orchestrator.save_state(test_file)

        # Verify file exists and has content
        with open(test_file, 'r') as f:
            state = json.load(f)

        self.assertEqual(state["cycle_count"], original_cycle)
        self.assertEqual(state["components"]["test1"]["params"], original_params)

        # Modify current state
        self.orchestrator.on_market_data({"test": "data"})
        new_cycle = self.orchestrator.cycle_count
        new_params = self.component1.get_params()

        # Load state
        self.orchestrator.load_state(test_file)

        # Verify state restored
        self.assertEqual(self.orchestrator.cycle_count, original_cycle)
        self.assertEqual(self.component1.get_params(), original_params)


if __name__ == "__main__":
    unittest.main()