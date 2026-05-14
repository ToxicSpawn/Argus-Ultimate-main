"""
Tests for Unified Trading System
================================

Comprehensive tests for all unified_trading modules.
"""

import unittest
import asyncio
from decimal import Decimal
from datetime import datetime

from tests.framework.test_base import ArgusTestCase, AsyncTestCase, create_mock_order

from unified_trading import UnifiedTradingOrchestrator
from unified_trading.order_management import OrderManager, Order, OrderSide, OrderType, Signal
from unified_trading.execution_engine import ExecutionEngine, ExecutionResult, Fill
from unified_trading.risk_integration import RiskIntegration, RiskLimits
from unified_trading.portfolio_management import PortfolioManager, Position
from unified_trading.signal_processing import SignalProcessor, ProcessedSignal, SignalStrength


class TestOrderManager(ArgusTestCase):
    """Tests for OrderManager."""
    
    def setUp(self):
        super().setUp()
        self.order_manager = OrderManager()
    
    def test_create_order_from_signal(self):
        """Test creating order from signal."""
        signal = Signal(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            confidence=0.8,
            strategy="momentum",
            suggested_qty=Decimal("0.1"),
            suggested_price=Decimal("45000")
        )
        
        # Run async test
        order = self.run_async(self.order_manager.create_order(signal))
        
        self.assertIsNotNone(order)
        self.assertEqual(order.symbol, "BTC/USD")
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.quantity, Decimal("0.1"))
        self.assertIsNotNone(order.id)
    
    def test_order_validation(self):
        """Test order validation."""
        # Invalid: negative quantity
        with self.assertRaises(Exception):
            Order(
                id="TEST-001",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("-0.1")
            )
    
    def test_get_active_orders(self):
        """Test getting active orders."""
        # Create some orders
        signal = Signal(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            confidence=0.8,
            strategy="momentum",
            suggested_qty=Decimal("0.1")
        )
        
        self.run_async(self.order_manager.create_order(signal))
        
        active_orders = self.run_async(self.order_manager.get_active_orders())
        self.assertEqual(len(active_orders), 1)


class TestExecutionEngine(ArgusTestCase):
    """Tests for ExecutionEngine."""
    
    def setUp(self):
        super().setUp()
        self.execution_engine = ExecutionEngine()
    
    def test_venue_selection(self):
        """Test venue selection logic."""
        # Mock order
        order = Order(
            id="TEST-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1")
        )
        
        # Should select a venue
        venue = self.execution_engine._select_venue(order)
        self.assertIsNotNone(venue)
        self.assertIn(venue, ['binance', 'kraken', 'coinbase'])
    
    def test_calculate_fees(self):
        """Test fee calculation."""
        # Get a venue adapter
        adapter = self.execution_engine._venues.get('binance')
        self.assertIsNotNone(adapter)
        
        # Calculate fees
        qty = Decimal("1.0")
        price = Decimal("50000")
        fees = adapter._calculate_fees(qty, price)
        
        # Should be 0.1% of notional
        expected = qty * price * Decimal("0.001")
        self.assertEqual(fees, expected)


class TestRiskIntegration(ArgusTestCase):
    """Tests for RiskIntegration."""
    
    def setUp(self):
        super().setUp()
        self.risk = RiskIntegration()
        self.run_async(self.risk.initialize({
            'max_position_size': 0.1,
            'max_drawdown': 0.2,
            'daily_loss_limit': 500
        }))
    
    def test_risk_limits_configuration(self):
        """Test risk limits configuration."""
        limits = RiskLimits({
            'max_position_size': 0.1,
            'max_drawdown': 0.2
        })
        
        errors = limits.validate_config()
        self.assertEqual(len(errors), 0)
    
    def test_invalid_risk_limits(self):
        """Test invalid risk limits detection."""
        limits = RiskLimits({
            'max_position_size': 1.5  # Invalid: > 1
        })
        
        errors = limits.validate_config()
        self.assertGreater(len(errors), 0)
    
    def test_check_signal_confidence(self):
        """Test signal confidence check."""
        # Low confidence signal should be rejected
        low_conf_signal = Signal(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            confidence=0.3,
            strategy="test",
            suggested_qty=Decimal("0.1")
        )
        
        result = self.run_async(self.risk.check_signal(low_conf_signal))
        self.assertFalse(result.allowed)


class TestPortfolioManager(ArgusTestCase):
    """Tests for PortfolioManager."""
    
    def setUp(self):
        super().setUp()
        self.portfolio = PortfolioManager()
        self.run_async(self.portfolio.initialize(Decimal("10000")))
    
    def test_initialization(self):
        """Test portfolio initialization."""
        summary = self.run_async(self.portfolio.get_summary())
        
        self.assertEqual(summary.total_value, Decimal("10000"))
        self.assertEqual(summary.cash_balance, Decimal("10000"))
        self.assertEqual(summary.num_positions, 0)
    
    def test_position_creation(self):
        """Test position creation from fill."""
        # Create a fill
        fill = Fill(
            order_id="ORD-001",
            symbol="BTC/USD",
            side="buy",
            filled_qty=Decimal("0.1"),
            price=Decimal("45000"),
            venue="binance",
            fees=Decimal("4.5")
        )
        
        # Create execution result
        execution = ExecutionResult(
            success=True,
            order_id="ORD-001",
            status='filled',
            filled_qty=Decimal("0.1"),
            avg_price=Decimal("45000"),
            fills=[fill]
        )
        
        # Update portfolio
        self.run_async(self.portfolio.update_position(execution))
        
        # Check position created
        position = self.run_async(self.portfolio.get_position("BTC/USD"))
        self.assertIsNotNone(position)
        self.assertEqual(position.quantity, Decimal("0.1"))
        self.assertEqual(position.symbol, "BTC/USD")


class TestSignalProcessor(ArgusTestCase):
    """Tests for SignalProcessor."""
    
    def setUp(self):
        super().setUp()
        self.processor = SignalProcessor()
        self.run_async(self.processor.initialize())
    
    def test_signal_aggregation(self):
        """Test signal aggregation."""
        # Add multiple signals
        signal1 = Signal(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            confidence=0.8,
            strategy="momentum",
            suggested_qty=Decimal("0.1")
        )
        
        signal2 = Signal(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            confidence=0.7,
            strategy="mean_reversion",
            suggested_qty=Decimal("0.15")
        )
        
        self.run_async(self.processor._aggregator.add_signal(signal1))
        self.run_async(self.processor._aggregator.add_signal(signal2))
        
        # Aggregate
        aggregated = self.run_async(self.processor._aggregator.aggregate("BTC/USD"))
        
        self.assertIsNotNone(aggregated)
        self.assertEqual(aggregated.side, OrderSide.BUY)
        self.assertGreater(aggregated.confidence, 0.5)
    
    def test_signal_filter(self):
        """Test signal filtering."""
        # Create a weak signal
        weak_signal = ProcessedSignal(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            confidence=0.4,
            strength=SignalStrength.WEAK,
            suggested_qty=Decimal("0.1"),
            suggested_price=Decimal("45000"),
            strategies=["test"]
        )
        
        # Should be filtered out
        is_valid = self.processor._apply_filter(weak_signal)
        self.assertFalse(is_valid)


class TestUnifiedOrchestrator(ArgusTestCase):
    """Integration tests for UnifiedTradingOrchestrator."""
    
    def setUp(self):
        super().setUp()
        self.orchestrator = UnifiedTradingOrchestrator()
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        result = self.run_async(self.orchestrator.initialize())
        self.assertTrue(result)
        self.assertTrue(self.orchestrator.state.is_initialized)
    
    def test_process_tick(self):
        """Test processing market tick."""
        # Initialize first
        self.run_async(self.orchestrator.initialize())
        self.run_async(self.orchestrator.start())
        
        # Process tick
        result = self.run_async(
            self.orchestrator.process_tick("BTC/USD", 45000.0, volume=100)
        )
        
        self.assertIsNotNone(result)
        self.assertIn('success', result)
    
    def test_get_status(self):
        """Test getting system status."""
        # Initialize
        self.run_async(self.orchestrator.initialize())
        
        # Get status
        status = self.run_async(self.orchestrator.get_status())
        
        self.assertIn('state', status)
        self.assertIn('portfolio', status)
        self.assertIn('risk', status)


class TestConfiguration(ArgusTestCase):
    """Tests for unified configuration."""
    
    def test_config_loading(self):
        """Test configuration loading."""
        from core.unified_config import config
        
        # Should be able to get values
        mode = config.get('trading.mode')
        self.assertIn(mode, ['paper', 'live', 'hybrid'])
        
        # Should have nested access
        max_position = config.get('risk.max_position_size')
        self.assert_valid_numeric(max_position, min_value=0, max_value=1)
    
    def test_config_validation(self):
        """Test configuration validation."""
        from core.unified_config import config
        
        errors = config.validate()
        # Should have no validation errors
        self.assertEqual(len(errors), 0)
    
    def test_config_types(self):
        """Test configuration type conversion."""
        from core.unified_config import config
        
        # Get as specific types
        balance = config.get_float('trading.initial_balance')
        self.assertIsInstance(balance, float)
        self.assertGreater(balance, 0)
        
        max_pos = config.get_float('risk.max_position_size')
        self.assertIsInstance(max_pos, float)


class TestExceptionManager(ArgusTestCase):
    """Tests for exception management."""
    
    def test_exception_hierarchy(self):
        """Test exception class hierarchy."""
        from core.exception_manager import (
            ArgusException,
            TradingException,
            OrderProcessingError
        )
        
        # Create exception
        exc = OrderProcessingError(
            "Test error",
            order_id="TEST-001",
            symbol="BTC/USD"
        )
        
        # Check inheritance
        self.assertIsInstance(exc, TradingException)
        self.assertIsInstance(exc, ArgusException)
        
        # Check attributes
        self.assertEqual(exc.message, "Test error")
        self.assertEqual(exc.code, "ORDER_ERROR")
    
    def test_exception_to_dict(self):
        """Test exception serialization."""
        from core.exception_manager import RiskViolationError
        
        exc = RiskViolationError(
            "Position size too large",
            violation_type="position_size",
            current_value=0.15,
            limit=0.1
        )
        
        exc_dict = exc.to_dict()
        
        self.assertIn('type', exc_dict)
        self.assertIn('message', exc_dict)
        self.assertIn('code', exc_dict)
        self.assertIn('details', exc_dict)


if __name__ == '__main__':
    unittest.main()
