"""
Core Orchestrator - Unified Trading System
==========================================

Main orchestration module that coordinates all trading system components.
Refactored from unified_trading_system.py to improve maintainability.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal

from unified_trading.order_management import OrderManager
from unified_trading.execution_engine import ExecutionEngine
from unified_trading.risk_integration import RiskIntegration
from unified_trading.portfolio_management import PortfolioManager
from unified_trading.signal_processing import SignalProcessor
from unified_trading.data_management import DataManager
from unified_trading.monitoring import SystemMonitor
from unified_trading.persistence import StateManager
from unified_trading.logging import AuditLogger
from unified_trading.api import APILayer

from core.exception_manager import (
    exception_manager,
    handle_errors,
    OrderProcessingError,
    RiskViolationError,
    ExecutionError
)

logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Current state of the trading system."""
    is_running: bool = False
    is_initialized: bool = False
    start_time: Optional[datetime] = None
    last_tick_time: Optional[datetime] = None
    active_orders: int = 0
    error_count: int = 0
    warning_count: int = 0


@dataclass
class SystemConfig:
    """Configuration for the unified trading system."""
    mode: str = "paper"  # paper, live, hybrid
    initial_balance: Decimal = Decimal("10000")
    base_currency: str = "USD"
    max_positions: int = 10
    risk_limits: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.risk_limits is None:
            self.risk_limits = {
                "max_position_size": 0.1,  # 10% of portfolio
                "max_drawdown": 0.2,  # 20%
                "daily_loss_limit": 500,
                "var_confidence": 0.95
            }


class UnifiedTradingOrchestrator:
    """
    Main orchestrator for the unified trading system.
    
    Coordinates all components: order management, execution, risk,
    portfolio, signals, data, monitoring, persistence, and API.
    
    This replaces the monolithic unified_trading_system.py with
    a modular, maintainable architecture.
    """
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.state = SystemState()
        
        # Initialize all subsystems
        self.logger = AuditLogger()
        self.monitor = SystemMonitor()
        self.state_manager = StateManager()
        self.data_manager = DataManager()
        self.signal_processor = SignalProcessor()
        self.portfolio_manager = PortfolioManager()
        self.risk_integration = RiskIntegration()
        self.execution_engine = ExecutionEngine()
        self.order_manager = OrderManager()
        self.api_layer = APILayer()
        
        # Event bus for component communication
        self._event_handlers: Dict[str, List[callable]] = {}
        
        logger.info("UnifiedTradingOrchestrator initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing unified trading system...")
            
            # Initialize in dependency order
            await self.state_manager.initialize()
            await self.data_manager.initialize()
            await self.signal_processor.initialize()
            await self.portfolio_manager.initialize(self.config.initial_balance)
            await self.risk_integration.initialize(self.config.risk_limits)
            await self.execution_engine.initialize()
            await self.order_manager.initialize()
            await self.monitor.initialize()
            await self.api_layer.initialize()
            
            # Load previous state if exists
            saved_state = await self.state_manager.load_state()
            if saved_state:
                await self._restore_state(saved_state)
            
            self.state.is_initialized = True
            self.state.start_time = datetime.utcnow()
            
            logger.info("System initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}", exc_info=True)
            exception_manager.handle_exception(e, {"phase": "initialization"})
            return False
    
    async def start(self) -> bool:
        """
        Start the trading system.
        
        Returns:
            bool: True if started successfully
        """
        if not self.state.is_initialized:
            if not await self.initialize():
                return False
        
        try:
            logger.info("Starting trading system...")
            
            self.state.is_running = True
            
            # Start all components
            await self.data_manager.start()
            await self.signal_processor.start()
            await self.monitor.start()
            await self.api_layer.start()
            
            # Start main loop
            asyncio.create_task(self._main_loop())
            
            logger.info("Trading system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}", exc_info=True)
            exception_manager.handle_exception(e, {"phase": "start"})
            return False
    
    async def stop(self) -> bool:
        """
        Stop the trading system gracefully.
        
        Returns:
            bool: True if stopped successfully
        """
        try:
            logger.info("Stopping trading system...")
            
            self.state.is_running = False
            
            # Stop all components in reverse order
            await self.api_layer.stop()
            await self.monitor.stop()
            await self.signal_processor.stop()
            await self.data_manager.stop()
            
            # Save state
            await self._save_state()
            
            logger.info("Trading system stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}", exc_info=True)
            exception_manager.handle_exception(e, {"phase": "stop"})
            return False
    
    @handle_errors(logger_name=__name__, reraise=False)
    async def process_tick(self, symbol: str, price: float, **kwargs) -> Dict[str, Any]:
        """
        Process a market data tick.
        
        Args:
            symbol: Trading symbol
            price: Current price
            **kwargs: Additional market data (volume, high, low, etc.)
            
        Returns:
            Dict with processing results
        """
        if not self.state.is_running:
            return {"success": False, "error": "System not running"}
        
        self.state.last_tick_time = datetime.utcnow()
        
        # Update data manager
        await self.data_manager.update_market_data(symbol, price, **kwargs)
        
        # Process signals
        signals = await self.signal_processor.generate_signals(symbol, price)
        
        results = []
        for signal in signals:
            # Check risk limits
            risk_check = await self.risk_integration.check_signal(signal)
            if not risk_check.allowed:
                await self.logger.log_risk_violation(signal, risk_check.reason)
                continue
            
            # Create and submit order
            try:
                order = await self.order_manager.create_order(signal)
                result = await self.execution_engine.execute(order)
                results.append(result)
                
                # Update portfolio
                await self.portfolio_manager.update_position(result)
                
            except (OrderProcessingError, RiskViolationError, ExecutionError) as e:
                logger.error(f"Order failed: {e}")
                results.append({"success": False, "error": str(e)})
        
        # Update monitoring
        await self.monitor.record_tick(symbol, price, len(signals))
        
        return {
            "success": True,
            "symbol": symbol,
            "price": price,
            "signals": len(signals),
            "orders": len(results),
            "results": results
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "state": {
                "running": self.state.is_running,
                "initialized": self.state.is_initialized,
                "uptime": self._get_uptime(),
                "active_orders": self.state.active_orders
            },
            "portfolio": await self.portfolio_manager.get_summary(),
            "risk": await self.risk_integration.get_status(),
            "data": await self.data_manager.get_status(),
            "monitoring": await self.monitor.get_metrics(),
            "exceptions": exception_manager.get_exception_stats()
        }
    
    async def _main_loop(self):
        """Main system loop."""
        while self.state.is_running:
            try:
                # Perform periodic tasks
                await self._periodic_tasks()
                
                # Save state periodically
                if self._should_save_state():
                    await self._save_state()
                
                await asyncio.sleep(1)  # 1 second loop
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                exception_manager.handle_exception(e, {"phase": "main_loop"})
    
    async def _periodic_tasks(self):
        """Execute periodic maintenance tasks."""
        # Check portfolio health
        await self.portfolio_manager.check_health()
        
        # Update risk metrics
        await self.risk_integration.update_metrics()
        
        # Clean up old data
        await self.data_manager.cleanup_old_data()
        
        # Check for stuck orders
        await self.order_manager.check_stuck_orders()
    
    async def _save_state(self):
        """Save current system state."""
        state_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "config": self.config,
            "portfolio": await self.portfolio_manager.get_state(),
            "positions": await self.portfolio_manager.get_positions(),
            "orders": await self.order_manager.get_active_orders(),
            "statistics": {
                "error_count": self.state.error_count,
                "warning_count": self.state.warning_count
            }
        }
        await self.state_manager.save_state(state_data)
    
    async def _restore_state(self, state_data: Dict):
        """Restore system from saved state."""
        try:
            if "portfolio" in state_data:
                await self.portfolio_manager.restore_state(state_data["portfolio"])
            
            if "positions" in state_data:
                await self.portfolio_manager.restore_positions(state_data["positions"])
            
            if "orders" in state_data:
                await self.order_manager.restore_orders(state_data["orders"])
            
            logger.info("System state restored successfully")
            
        except Exception as e:
            logger.error(f"Failed to restore state: {e}")
            exception_manager.handle_exception(e, {"phase": "restore_state"})
    
    def _get_uptime(self) -> float:
        """Get system uptime in seconds."""
        if self.state.start_time:
            return (datetime.utcnow() - self.state.start_time).total_seconds()
        return 0.0
    
    def _should_save_state(self) -> bool:
        """Determine if state should be saved."""
        # Save every 60 seconds
        if not hasattr(self, '_last_save_time'):
            self._last_save_time = datetime.utcnow()
            return True
        
        elapsed = (datetime.utcnow() - self._last_save_time).total_seconds()
        if elapsed >= 60:
            self._last_save_time = datetime.utcnow()
            return True
        
        return False
    
    # Event handling
    def on(self, event: str, handler: callable):
        """Register event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
    
    async def emit(self, event: str, data: Dict[str, Any]):
        """Emit event to all registered handlers."""
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Event handler error for {event}: {e}")
