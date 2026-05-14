"""
Test cases for the SmartExecutionEngine component
"""

import unittest
from datetime import datetime, timezone, timedelta
from core.real_time_learning.execution_engine import SmartExecutionEngine


class TestSmartExecutionEngine(unittest.TestCase):
    """Test the smart execution engine component"""

    def setUp(self):
        self.engine = SmartExecutionEngine()

    def test_initial_state(self):
        """Test initial component state"""
        params = self.engine.get_params()
        self.assertEqual(params["base_participation"], 0.20)
        self.assertEqual(params["aggressiveness"], 0.5)
        self.assertEqual(params["latency_buffer_ms"], 10)
        self.assertEqual(params["current_exchange"], "binance")
        self.assertAlmostEqual(params["avg_slippage"], 0.0005)
        self.assertAlmostEqual(params["fill_ratio"], 0.95)

    def test_liquidity_adaptation(self):
        """Test liquidity-based adaptations"""
        # Test low liquidity
        self.engine.learn({
            "order_book": {"bids": [[100, 10], [99, 5]], "asks": [[101, 10], [102, 5]]},
            "exchange": "binance",
            "volatility": 0.01,
            "volume": 200000,  # Below low threshold (500k)
            "timestamp": datetime.now(timezone.utc)
        })

        params = self.engine.get_params()
        self.assertLess(params["base_participation"], 0.20)  # Should decrease
        self.assertLess(params["aggressiveness"], 0.5)       # Should decrease
        self.assertLess(params["min_slice_size"], 0.05)     # Should decrease

        # Test high liquidity
        for _ in range(5):  # Need multiple updates to overcome learning rate
            self.engine.learn({
                "order_book": {"bids": [[100, 1000], [99, 500]], "asks": [[101, 1000], [102, 500]]},
                "exchange": "binance",
                "volatility": 0.01,
                "volume": 6000000,  # Above high threshold (5M)
                "timestamp": datetime.now(timezone.utc)
            })

        params = self.engine.get_params()
        self.assertGreater(params["base_participation"], 0.20)  # Should increase
        self.assertGreater(params["aggressiveness"], 0.5)       # Should increase

    def test_volatility_adaptation(self):
        """Test volatility-based adaptations"""
        # Test high volatility
        self.engine.learn({
            "order_book": {"bids": [[100, 100], [99, 50]], "asks": [[101, 100], [102, 50]]},
            "exchange": "binance",
            "volatility": 0.05,  # High volatility
            "volume": 2000000,
            "timestamp": datetime.now(timezone.utc)
        })

        params = self.engine.get_params()
        self.assertLess(params["base_participation"], 0.20)  # Should decrease
        self.assertGreater(params["aggressiveness"], 0.5)    # Should increase (to get fills)
        self.assertGreater(params["algorithm_weights"]["POV"], 0.2)  # Should increase POV

    def test_performance_adaptation(self):
        """Test performance-based adaptations"""
        # Simulate high slippage
        self.engine.avg_slippage = 0.002  # 0.2% slippage
        self.engine.learn({
            "order_book": {"bids": [[100, 100], [99, 50]], "asks": [[101, 100], [102, 50]]},
            "exchange": "binance",
            "volatility": 0.01,
            "volume": 2000000,
            "timestamp": datetime.now(timezone.utc)
        })

        params = self.engine.get_params()
        self.assertLess(params["aggressiveness"], 0.5)       # Should decrease to reduce slippage
        self.assertLess(params["base_participation"], 0.20)  # Should decrease

    def test_exchange_latency_adaptation(self):
        """Test exchange-specific latency adaptations"""
        # Simulate high latency exchange
        self.engine.current_exchange = "bybit"
        self.engine.params["exchange_latency"]["bybit"] = 20  # High latency

        self.engine.learn({
            "order_book": {"bids": [[100, 100], [99, 50]], "asks": [[101, 100], [102, 50]]},
            "exchange": "bybit",
            "volatility": 0.01,
            "volume": 2000000,
            "timestamp": datetime.now(timezone.utc)
        })

        params = self.engine.get_params()
        self.assertGreater(params["latency_buffer_ms"], 10)  # Should increase
        self.assertLess(params["aggressiveness"], 0.5)      # Should decrease

    def test_algorithm_adaptation(self):
        """Test execution algorithm adaptations"""
        # Start with default weights
        initial_weights = self.engine.params["algorithm_weights"].copy()

        # Simulate high volatility (should increase POV)
        self.engine.learn({
            "order_book": {"bids": [[100, 100], [99, 50]], "asks": [[101, 100], [102, 50]]},
            "exchange": "binance",
            "volatility": 0.04,  # High volatility
            "volume": 2000000,
            "timestamp": datetime.now(timezone.utc)
        })

        new_weights = self.engine.params["algorithm_weights"]
        self.assertGreater(new_weights["POV"], initial_weights["POV"])
        self.assertLess(new_weights["TWAP"], initial_weights["TWAP"])

    def test_validation(self):
        """Test parameter validation"""
        # Test valid parameters
        valid_params = {
            "base_participation": 0.25,
            "aggressiveness": 0.6,
            "min_slice_size": 0.06,
            "max_slice_size": 0.25,
            "latency_buffer_ms": 15,
            "algorithm_weights": {
                "TWAP": 0.35,
                "VWAP": 0.35,
                "POV": 0.2,
                "Iceberg": 0.1
            },
            "exchange_latency": {
                "binance": 8,
                "bybit": 12,
                "okx": 10
            },
            "liquidity_thresholds": {
                "low": 500000,
                "medium": 2000000,
                "high": 5000000
            }
        }
        self.assertTrue(self.engine.validate(valid_params))

        # Test invalid participation rate
        invalid_params = valid_params.copy()
        invalid_params["base_participation"] = 0.01  # Too low
        self.assertFalse(self.engine.validate(invalid_params))

        invalid_params["base_participation"] = 0.6  # Too high
        self.assertFalse(self.engine.validate(invalid_params))

        # Test invalid aggressiveness
        invalid_params = valid_params.copy()
        invalid_params["aggressiveness"] = 0.05  # Too low
        self.assertFalse(self.engine.validate(invalid_params))

        invalid_params["aggressiveness"] = 1.1  # Too high
        self.assertFalse(self.engine.validate(invalid_params))

        # Test invalid algorithm weights
        invalid_params = valid_params.copy()
        invalid_params["algorithm_weights"] = {
            "TWAP": 0.9,  # Too high, others too low
            "VWAP": 0.05,
            "POV": 0.03,
            "Iceberg": 0.02
        }
        self.assertFalse(self.engine.validate(invalid_params))

    def test_rollback(self):
        """Test rollback functionality"""
        # Make some changes
        original_params = self.engine.get_params()
        self.engine.learn({
            "order_book": {"bids": [[100, 100], [99, 50]], "asks": [[101, 100], [102, 50]]},
            "exchange": "binance",
            "volatility": 0.03,
            "volume": 2000000,
            "timestamp": datetime.now(timezone.utc)
        })

        # Verify changes were made
        changed_params = self.engine.get_params()
        self.assertNotEqual(original_params["base_participation"], changed_params["base_participation"])

        # Rollback
        self.engine.rollback()
        rolled_back_params = self.engine.get_params()

        # Verify rollback worked
        self.assertEqual(original_params["base_participation"], rolled_back_params["base_participation"])
        self.assertEqual(original_params["aggressiveness"], rolled_back_params["aggressiveness"])
        self.assertEqual(original_params["latency_buffer_ms"], rolled_back_params["latency_buffer_ms"])

    def test_trade_learning(self):
        """Test learning from individual trades"""
        # Initial state
        initial_slippage = self.engine.avg_slippage
        initial_fill = self.engine.fill_ratio

        # Simulate a problematic trade
        self.engine.learn_from_trade({
            "symbol": "BTC/USDT",
            "size": 1.0,
            "slippage": 0.003,      # High slippage (0.3%)
            "fill_ratio": 0.7,     # Poor fill ratio (70%)
            "execution_time_ms": 20,
            "algorithm": "TWAP",
            "exchange": "binance",
            "timestamp": datetime.now(timezone.utc)
        })

        # Verify performance metrics updated
        self.assertGreater(self.engine.avg_slippage, initial_slippage)
        self.assertLess(self.engine.fill_ratio, initial_fill)

        # Verify parameters adapted to reduce slippage
        params = self.engine.get_params()
        self.assertLess(params["aggressiveness"], 0.5)  # Should be more passive
        self.assertLess(params["base_participation"], 0.20)  # Should reduce participation

    def test_state_restore(self):
        """Test state restoration"""
        # Make some changes
        self.engine.learn({
            "order_book": {"bids": [[100, 100], [99, 50]], "asks": [[101, 100], [102, 50]]},
            "exchange": "binance",
            "volatility": 0.02,
            "volume": 3000000,
            "timestamp": datetime.now(timezone.utc)
        })

        # Save state
        state = {
            "params": self.engine.get_params(),
            "current_exchange": self.engine.current_exchange,
            "avg_slippage": self.engine.avg_slippage,
            "fill_ratio": self.engine.fill_ratio,
            "last_updated": self.engine.last_updated
        }

        # Modify engine
        self.engine.learn({
            "order_book": {"bids": [[100, 200], [99, 100]], "asks": [[101, 200], [102, 100]]},
            "exchange": "bybit",
            "volatility": 0.04,
            "volume": 1000000,
            "timestamp": datetime.now(timezone.utc)
        })

        # Restore state
        self.engine._restore_state(state)
        restored_params = self.engine.get_params()

        # Verify restoration
        self.assertEqual(state["params"]["base_participation"], restored_params["base_participation"])
        self.assertEqual(state["params"]["aggressiveness"], restored_params["aggressiveness"])
        self.assertEqual(state["current_exchange"], restored_params["current_exchange"])
        self.assertAlmostEqual(state["avg_slippage"], restored_params["avg_slippage"])
        self.assertAlmostEqual(state["fill_ratio"], restored_params["fill_ratio"])


if __name__ == "__main__":
    unittest.main()