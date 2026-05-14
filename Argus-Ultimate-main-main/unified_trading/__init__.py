"""
Unified Trading System - Refactored Architecture
================================================

This package replaces the monolithic unified_trading_system.py (734KB)
with a modular, maintainable architecture.

Modules:
- core_orchestrator: Main system orchestration
- order_management: Order lifecycle management
- execution_engine: Trade execution and routing
- risk_integration: Risk management integration
- portfolio_management: Portfolio tracking and management
- signal_processing: Trading signal processing
- data_management: Data ingestion and management
- monitoring: System monitoring and alerts
- persistence: State persistence and recovery
- logging: Comprehensive logging and audit
- api: API integration layer

Version: 2.0.0 (Refactored)
"""

from unified_trading.core_orchestrator import UnifiedTradingOrchestrator
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

__version__ = "2.0.0"
__all__ = [
    "UnifiedTradingOrchestrator",
    "OrderManager",
    "ExecutionEngine",
    "RiskIntegration",
    "PortfolioManager",
    "SignalProcessor",
    "DataManager",
    "SystemMonitor",
    "StateManager",
    "AuditLogger",
    "APILayer",
]
