# pyright: reportMissingImports=false
"""
Tests for Maximum Edge Systems.

Tests cover:
- Maximum Edge Orchestrator
- DeFi Yield Optimizer
- Latency-Based Stops
- Dynamic Latency Sizing
"""

from __future__ import annotations

import logging
import unittest
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)


class TestMaximumEdgeOrchestrator(unittest.TestCase):
    """Tests for Maximum Edge Orchestrator."""

    def setUp(self):
        try:
            from edge.maximum_edge_orchestrator import (
                MaximumEdgeOrchestrator, MarketRegime
            )
            self.orchestrator = MaximumEdgeOrchestrator()
            self.MarketRegime = MarketRegime
        except ImportError:
            self.skipTest("Maximum Edge Orchestrator not available")

    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.orchestrator)
        self.assertGreater(len(self.orchestrator.allocations), 0)

    def test_strategy_allocations(self):
        """Test strategy allocations sum to 100%."""
        total = sum(a.allocated_pct for a in self.orchestrator.allocations.values())
        self.assertAlmostEqual(total, 100.0, delta=1.0)

    def test_update_market_regime(self):
        """Test market regime updates."""
        regime = self.MarketRegime(
            regime="crisis",
            volatility=0.10,
            trend_strength=-0.5,
            funding_environment="positive",
            liquidity_score=0.6
        )
        actions = self.orchestrator.update_market_regime(regime)
        self.assertIsInstance(actions, list)
        self.assertIsNotNone(self.orchestrator.current_regime)

    def test_record_trade_result(self):
        """Test recording trade results."""
        initial_trades = self.orchestrator.total_trades
        self.orchestrator.record_trade_result(
            strategy_name="funding_rate_arb",
            pnl=10.0,
            edge_bps=15.0
        )
        self.assertEqual(self.orchestrator.total_trades, initial_trades + 1)

    def test_get_edge_report(self):
        """Test edge report generation."""
        report = self.orchestrator.get_edge_report()
        self.assertIsNotNone(report)
        self.assertGreater(report.total_expected_edge_bps, 0)
        self.assertIsInstance(report.recommendations, list)


class TestDeFiYieldOptimizer(unittest.TestCase):
    """Tests for DeFi Yield Optimizer."""

    def setUp(self):
        try:
            from defi.defi_yield_optimizer import (
                DeFiYieldOptimizer, ProtocolAPY, ProtocolType, RiskTier
            )
            self.optimizer = DeFiYieldOptimizer()
            self.ProtocolAPY = ProtocolAPY
            self.ProtocolType = ProtocolType
            self.RiskTier = RiskTier
        except ImportError:
            self.skipTest("DeFi Yield Optimizer not available")

    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.optimizer)

    def test_find_best_yield(self):
        """Test finding best yield."""
        # Add test data
        test_data = [
            self.ProtocolAPY(
                protocol="aave_v3",
                protocol_type=self.ProtocolType.LENDING,
                asset="USDC",
                apy=0.05,
                apy_base=0.03,
                apy_reward=0.02,
                tvl_usd=1_000_000_000,
                risk_tier=self.RiskTier.BLUE_CHIP,
                min_deposit=1.0,
                withdrawal_fee=0.0
            ),
            self.ProtocolAPY(
                protocol="compound_v3",
                protocol_type=self.ProtocolType.LENDING,
                asset="USDC",
                apy=0.04,
                apy_base=0.04,
                apy_reward=0.0,
                tvl_usd=500_000_000,
                risk_tier=self.RiskTier.BLUE_CHIP,
                min_deposit=1.0,
                withdrawal_fee=0.0
            ),
        ]
        self.optimizer.update_apy_data(test_data)
        
        result = self.optimizer.find_best_yield("USDC", 1000.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.protocol, "aave_v3")

    def test_get_portfolio_summary(self):
        """Test portfolio summary."""
        summary = self.optimizer.get_portfolio_summary()
        self.assertIn("total_value_usd", summary)
        self.assertIn("weighted_avg_apy", summary)


class TestLatencyBasedStops(unittest.TestCase):
    """Tests for Latency-Based Stops."""

    def setUp(self):
        try:
            from risk.latency_based_stops import (
                LatencyBasedStops, LatencyTier
            )
            self.stops = LatencyBasedStops(base_stop_pct=0.02)
            self.LatencyTier = LatencyTier
        except ImportError:
            self.skipTest("Latency-Based Stops not available")

    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.stops)
        self.assertEqual(self.stops.base_stop_pct, 0.02)

    def test_calculate_stop_low_latency(self):
        """Test stop calculation with low latency."""
        # Simulate low latency
        self.stops.current_latency_ms = 10.0
        self.stops.current_tier = self.LatencyTier.LOW
        
        adjustment = self.stops.calculate_stop(
            entry_price=50000.0,
            side="long"
        )
        
        self.assertEqual(adjustment.adjustment_factor, 1.0)
        self.assertLess(adjustment.adjusted_stop_pct, 0.03)

    def test_calculate_stop_high_latency(self):
        """Test stop calculation with high latency."""
        # Simulate high latency
        self.stops.current_latency_ms = 150.0
        self.stops.current_tier = self.LatencyTier.HIGH
        
        adjustment = self.stops.calculate_stop(
            entry_price=50000.0,
            side="long"
        )
        
        self.assertGreater(adjustment.adjustment_factor, 1.0)
        self.assertGreater(adjustment.adjusted_stop_pct, 0.02)

    def test_get_status(self):
        """Test status retrieval."""
        status = self.stops.get_status()
        self.assertIn("base_stop_pct", status)
        self.assertIn("current_tier", status)


class TestDynamicLatencySizing(unittest.TestCase):
    """Tests for Dynamic Latency Sizing."""

    def setUp(self):
        try:
            from execution.dynamic_latency_sizing import (
                DynamicLatencySizing, SizingMode, FillQuality
            )
            self.sizing = DynamicLatencySizing(
                mode=SizingMode.BALANCED,
                base_position_usd=1000.0
            )
            self.SizingMode = SizingMode
            self.FillQuality = FillQuality
        except ImportError:
            self.skipTest("Dynamic Latency Sizing not available")

    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.sizing)
        self.assertEqual(self.sizing.base_position_usd, 1000.0)

    def test_calculate_size_low_latency(self):
        """Test size calculation with low latency."""
        self.sizing.current_latency_ms = 10.0
        self.sizing.current_volatility = 0.02
        self.sizing.current_depth_score = 0.8
        
        adjustment = self.sizing.calculate_size("BTC/USD", "binance")
        
        # Size should be reduced but not too much
        self.assertGreater(adjustment.adjustment_factor, 0.3)
        self.assertGreater(adjustment.adjusted_size_usd, 300.0)

    def test_calculate_size_high_latency(self):
        """Test size calculation with high latency."""
        self.sizing.current_latency_ms = 150.0
        self.sizing.current_volatility = 0.02
        self.sizing.current_depth_score = 0.8
        
        adjustment = self.sizing.calculate_size("BTC/USD", "binance")
        
        self.assertLess(adjustment.adjusted_size_usd, 1000.0)

    def test_should_trade(self):
        """Test trade decision."""
        # Good conditions
        self.sizing.current_latency_ms = 10.0
        self.sizing.current_volatility = 0.02
        should_trade, reason = self.sizing.should_trade("BTC/USD", "binance")
        self.assertTrue(should_trade)
        
        # Bad conditions
        self.sizing.current_latency_ms = 600.0
        should_trade, reason = self.sizing.should_trade("BTC/USD", "binance")
        self.assertFalse(should_trade)

    def test_get_status(self):
        """Test status retrieval."""
        status = self.sizing.get_status()
        self.assertIn("mode", status)
        self.assertIn("current_conditions", status)
        self.assertIn("factors", status)


class TestComponentRegistryIntegration(unittest.TestCase):
    """Tests for Component Registry integration."""

    def test_new_components_registered(self):
        """Test that new components are registered."""
        from unittest.mock import MagicMock
        from core.component_registry import ComponentRegistry
        
        mock_system = MagicMock()
        mock_system.config = MagicMock()
        
        registry = ComponentRegistry(system=mock_system)
        
        # Check that properties exist
        self.assertTrue(hasattr(registry, "maximum_edge_orchestrator"))
        self.assertTrue(hasattr(registry, "defi_yield_optimizer"))
        self.assertTrue(hasattr(registry, "latency_based_stops"))
        self.assertTrue(hasattr(registry, "dynamic_latency_sizing"))


def run_tests():
    """Run all tests."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMaximumEdgeOrchestrator))
    suite.addTest(unittest.makeSuite(TestDeFiYieldOptimizer))
    suite.addTest(unittest.makeSuite(TestLatencyBasedStops))
    suite.addTest(unittest.makeSuite(TestDynamicLatencySizing))
    suite.addTest(unittest.makeSuite(TestComponentRegistryIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
