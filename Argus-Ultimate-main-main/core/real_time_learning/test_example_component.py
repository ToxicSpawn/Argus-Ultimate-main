"""
Test cases for the AdaptiveVolatilityCluster example component
"""

import unittest
from datetime import datetime, timezone, timedelta
from core.real_time_learning.example_component import AdaptiveVolatilityCluster


class TestAdaptiveVolatilityCluster(unittest.TestCase):
    """Test the adaptive volatility clustering component"""

    def setUp(self):
        self.component = AdaptiveVolatilityCluster()

    def test_initial_state(self):
        """Test initial component state"""
        params = self.component.get_params()
        self.assertEqual(params["low_vol_threshold"], 0.01)
        self.assertEqual(params["high_vol_threshold"], 0.03)
        self.assertEqual(params["cluster_centers"], [0.01, 0.02, 0.04])
        self.assertEqual(params["current_regime"], "medium")

    def test_volatility_clustering(self):
        """Test volatility clustering updates"""
        # Feed some low volatility data
        for _ in range(10):
            self.component.learn({
                "realized_volatility": 0.005,  # Very low vol
                "price_jumps": [],
                "volume": 1.0,
                "timestamp": datetime.now(timezone.utc)
            })

        # Check that low vol threshold decreased
        params = self.component.get_params()
        self.assertLess(params["low_vol_threshold"], 0.01)

        # Feed some high volatility data
        for _ in range(10):
            self.component.learn({
                "realized_volatility": 0.05,  # High vol
                "price_jumps": [0.01, -0.005, 0.015],
                "volume": 2.0,
                "timestamp": datetime.now(timezone.utc)
            })

        # Check that high vol threshold increased
        params = self.component.get_params()
        self.assertGreater(params["high_vol_threshold"], 0.03)

    def test_regime_detection(self):
        """Test regime detection"""
        # Test low volatility
        params = self.component.learn({
            "realized_volatility": 0.005,
            "price_jumps": [],
            "volume": 1.0,
            "timestamp": datetime.now(timezone.utc)
        })
        self.assertEqual(params["current_regime"], "low")

        # Test high volatility
        params = self.component.learn({
            "realized_volatility": 0.045,
            "price_jumps": [0.02, -0.01, 0.03],
            "volume": 3.0,
            "timestamp": datetime.now(timezone.utc)
        })
        self.assertEqual(params["current_regime"], "high")

    def test_transition_matrix(self):
        """Test transition matrix updates"""
        # Create a sequence of regime changes
        test_data = [
            {"realized_volatility": 0.005, "timestamp": datetime.now(timezone.utc)},  # low
            {"realized_volatility": 0.006, "timestamp": datetime.now(timezone.utc)},  # low
            {"realized_volatility": 0.025, "timestamp": datetime.now(timezone.utc)},  # medium
            {"realized_volatility": 0.035, "timestamp": datetime.now(timezone.utc)},  # high
            {"realized_volatility": 0.025, "timestamp": datetime.now(timezone.utc)},  # medium
        ]

        # Feed the data
        for data in test_data:
            self.component.learn(data)

        # Check that transition probabilities updated
        params = self.component.get_params()
        transitions = params["transition_matrix"]

        # Should see increased probability of low->medium transition
        self.assertGreater(transitions["low"]["medium"], 0.25)

    def test_validation(self):
        """Test parameter validation"""
        # Test valid changes
        valid_params = {
            "low_vol_threshold": 0.008,
            "high_vol_threshold": 0.035,
            "cluster_centers": [0.008, 0.02, 0.045],
            "transition_matrix": {
                "low": {"low": 0.7, "medium": 0.25, "high": 0.05},
                "medium": {"low": 0.15, "medium": 0.7, "high": 0.15},
                "high": {"low": 0.05, "medium": 0.25, "high": 0.7}
            }
        }
        self.assertTrue(self.component.validate(valid_params))

        # Test invalid threshold separation
        invalid_params = {
            **valid_params,
            "high_vol_threshold": 0.009  # Too close to low threshold
        }
        self.assertFalse(self.component.validate(invalid_params))

        # Test invalid cluster centers
        invalid_params = {
            **valid_params,
            "cluster_centers": [0.008, 0.009, 0.045]  # Centers too close
        }
        self.assertFalse(self.component.validate(invalid_params))

    def test_rollback(self):
        """Test rollback functionality"""
        # Make some changes
        original_params = self.component.get_params()
        for _ in range(5):
            self.component.learn({
                "realized_volatility": 0.05,
                "price_jumps": [0.01, -0.005],
                "volume": 2.0,
                "timestamp": datetime.now(timezone.utc)
            })

        # Verify changes were made
        changed_params = self.component.get_params()
        self.assertNotEqual(original_params["high_vol_threshold"], changed_params["high_vol_threshold"])

        # Rollback
        self.component.rollback()
        rolled_back_params = self.component.get_params()

        # Verify rollback worked
        self.assertEqual(original_params["low_vol_threshold"], rolled_back_params["low_vol_threshold"])
        self.assertEqual(original_params["high_vol_threshold"], rolled_back_params["high_vol_threshold"])

    def test_state_restore(self):
        """Test state restoration"""
        # Make some changes
        for _ in range(3):
            self.component.learn({
                "realized_volatility": 0.03,
                "price_jumps": [0.005],
                "volume": 1.5,
                "timestamp": datetime.now(timezone.utc)
            })

        # Save state
        state = {
            "params": self.component.get_params(),
            "current_regime": self.component.current_regime,
            "last_updated": self.component.last_updated.isoformat()
        }

        # Modify component
        self.component.learn({
            "realized_volatility": 0.06,
            "price_jumps": [0.02],
            "volume": 3.0,
            "timestamp": datetime.now(timezone.utc)
        })

        # Restore state
        self.component._restore_state(state)
        restored_params = self.component.get_params()

        # Verify restoration
        self.assertEqual(state["params"]["low_vol_threshold"], restored_params["low_vol_threshold"])
        self.assertEqual(state["params"]["high_vol_threshold"], restored_params["high_vol_threshold"])
        self.assertEqual(state["current_regime"], restored_params["current_regime"])


if __name__ == "__main__":
    unittest.main()