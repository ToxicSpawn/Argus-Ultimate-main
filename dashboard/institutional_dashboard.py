# pyright: reportMissingImports=false
"""
Institutional-Grade Trading Dashboard
======================================
Real-time web dashboard with full control capabilities.

FEATURES:
1. Live Metrics Display (PnL, positions, signals, latency)
2. Human Override Controls (emergency stop, pause, resume)
3. Strategy Management (enable/disable, weight adjustment)
4. Position Management (close all, modify, hedge)
5. Risk Controls (drawdown limits, exposure limits)
6. Learning Status (parameter updates, learning cycles)
7. Alert History (all alerts with acknowledgment)
8. Performance Analytics (Sharpe, drawdown, win rate)

This is the COMMAND CENTER for Argus.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from threading import Lock

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System operational states."""
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    DEGRADED = "degraded"
    STARTING = "starting"
    STOPPING = "stopping"


class AlertLevel(Enum):
    """Alert levels for dashboard."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Position:
    """Current position information."""
    symbol: str
    side: str  # "long", "short"
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    leverage: float
    stop_loss: float
    take_profit: float
    age_seconds: float


@dataclass
class TradingMetrics:
    """Live trading metrics."""
    timestamp: datetime
    total_pnl: float
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    total_trades: int
    open_positions: int
    total_exposure: float
    learning_cycles: int
    parameters_updated: int
    quantum_signals: int
    avg_latency_ms: float


@dataclass
class DashboardAlert:
    """Dashboard alert."""
    id: str
    timestamp: datetime
    level: AlertLevel
    message: str
    acknowledged: bool = False
    source: str = ""


class InstitutionalDashboard:
    """
    Institutional-grade trading dashboard with full control.
    
    Provides real-time visibility and human override capabilities.
    """
    
    def __init__(self, orchestrator=None, learning_hub=None):
        self.orchestrator = orchestrator
        self.learning_hub = learning_hub
        
        # System state
        self.system_state: SystemState = SystemState.RUNNING
        self.state_lock: Lock = Lock()
        
        # Live data
        self.current_positions: Dict[str, Position] = {}
        self.metrics_history: deque = deque(maxlen=1000)
        self.alerts: deque = deque(maxlen=500)
        self.trade_history: deque = deque(maxlen=1000)
        
        # Control state
        self.emergency_stop_active: bool = False
        self.trading_paused: bool = False
        self.max_position_size: float = 10000.0
        self.max_daily_loss: float = 1000.0
        self.max_drawdown_pct: float = 0.20
        
        # Callbacks for control actions
        self._control_callbacks: Dict[str, Callable] = {}
        
        # Performance tracking
        self.start_time: datetime = datetime.now()
        self.daily_start_pnl: float = 0.0
        
        logger.info("InstitutionalDashboard initialized")
    
    def register_callback(self, action: str, callback: Callable) -> None:
        """Register a control action callback."""
        self._control_callbacks[action] = callback
    
    # ========================================================================
    # CONTROL ACTIONS
    # ========================================================================
    
    def emergency_stop(self, reason: str = "Manual trigger") -> Dict[str, Any]:
        """
        EMERGENCY STOP - Immediately halt all trading.
        
        This is the nuclear button. Closes all positions and stops trading.
        """
        with self.state_lock:
            self.emergency_stop_active = True
            self.trading_paused = True
            self.system_state = SystemState.EMERGENCY_STOP
        
        alert = DashboardAlert(
            id=f"estop_{int(time.time())}",
            timestamp=datetime.now(),
            level=AlertLevel.CRITICAL,
            message=f"🚨 EMERGENCY STOP: {reason}",
            source="human"
        )
        self.alerts.append(alert)
        
        logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
        
        # Execute callback if registered
        if "emergency_stop" in self._control_callbacks:
            self._control_callbacks["emergency_stop"](reason)
        
        return {
            "status": "emergency_stop_activated",
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "positions_to_close": len(self.current_positions),
        }
    
    def resume_trading(self, reason: str = "Manual resume") -> Dict[str, Any]:
        """Resume trading after pause or emergency stop."""
        with self.state_lock:
            self.emergency_stop_active = False
            self.trading_paused = False
            self.system_state = SystemState.RUNNING
        
        alert = DashboardAlert(
            id=f"resume_{int(time.time())}",
            timestamp=datetime.now(),
            level=AlertLevel.INFO,
            message=f"✅ Trading resumed: {reason}",
            source="human"
        )
        self.alerts.append(alert)
        
        logger.info(f"Trading resumed: {reason}")
        
        if "resume" in self._control_callbacks:
            self._control_callbacks["resume"](reason)
        
        return {"status": "resumed", "timestamp": datetime.now().isoformat()}
    
    def pause_trading(self, reason: str = "Manual pause") -> Dict[str, Any]:
        """Pause trading without closing positions."""
        with self.state_lock:
            self.trading_paused = True
            self.system_state = SystemState.PAUSED
        
        alert = DashboardAlert(
            id=f"pause_{int(time.time())}",
            timestamp=datetime.now(),
            level=AlertLevel.WARNING,
            message=f"⏸️ Trading paused: {reason}",
            source="human"
        )
        self.alerts.append(alert)
        
        logger.warning(f"Trading paused: {reason}")
        
        if "pause" in self._control_callbacks:
            self._control_callbacks["pause"](reason)
        
        return {"status": "paused", "timestamp": datetime.now().isoformat()}
    
    def close_position(self, symbol: str, reason: str = "Manual close") -> Dict[str, Any]:
        """Manually close a specific position."""
        alert = DashboardAlert(
            id=f"close_{symbol}_{int(time.time())}",
            timestamp=datetime.now(),
            level=AlertLevel.INFO,
            message=f"📊 Closing position: {symbol} - {reason}",
            source="human"
        )
        self.alerts.append(alert)
        
        logger.info(f"Manual close requested: {symbol} - {reason}")
        
        if "close_position" in self._control_callbacks:
            self._control_callbacks["close_position"](symbol, reason)
        
        return {"status": "closing", "symbol": symbol, "reason": reason}
    
    def close_all_positions(self, reason: str = "Manual close all") -> Dict[str, Any]:
        """Close all open positions."""
        symbols = list(self.current_positions.keys())
        
        alert = DashboardAlert(
            id=f"close_all_{int(time.time())}",
            timestamp=datetime.now(),
            level=AlertLevel.WARNING,
            message=f"📊 Closing ALL {len(symbols)} positions: {reason}",
            source="human"
        )
        self.alerts.append(alert)
        
        logger.warning(f"Close all positions requested: {reason}")
        
        if "close_all" in self._control_callbacks:
            self._control_callbacks["close_all"](reason)
        
        return {"status": "closing_all", "positions": symbols, "reason": reason}
    
    def set_max_position_size(self, size_usd: float) -> Dict[str, Any]:
        """Update maximum position size."""
        old_value = self.max_position_size
        self.max_position_size = size_usd
        
        alert = DashboardAlert(
            id=f"max_pos_{int(time.time())}",
            timestamp=datetime.now(),
            level=AlertLevel.INFO,
            message=f"📏 Max position size: ${old_value:,.0f} → ${size_usd:,.0f}",
            source="human"
        )
        self.alerts.append(alert)
        
        return {"status": "updated", "old_value": old_value, "new_value": size_usd}
    
    def set_max_daily_loss(self, loss_usd: float) -> Dict[str, Any]:
        """Update maximum daily loss limit."""
        old_value = self.max_daily_loss
        self.max_daily_loss = loss_usd
        
        alert = DashboardAlert(
            id=f"max_loss_{int(time.time())}",
            timestamp=datetime.now(),
            level=AlertLevel.INFO,
            message=f"📉 Max daily loss: ${old_value:,.0f} → ${loss_usd:,.0f}",
            source="human"
        )
        self.alerts.append(alert)
        
        return {"status": "updated", "old_value": old_value, "new_value": loss_usd}
    
    def enable_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """Enable a specific strategy."""
        alert = DashboardAlert(
            id=f"enable_{strategy_name}_{int(time.time())}",
            timestamp=datetime.now(),
            level=AlertLevel.INFO,
            message=f"✅ Strategy enabled: {strategy_name}",
            source="human"
        )
        self.alerts.append(alert)
        
        if "enable_strategy" in self._control_callbacks:
            self._control_callbacks["enable_strategy"](strategy_name)
        
        return {"status": "enabled", "strategy": strategy_name}
    
    def disable_strategy(self, strategy_name: str, reason: str = "") -> Dict[str, Any]:
        """Disable a specific strategy."""
        alert = DashboardAlert(
            id=f"disable_{strategy_name}_{int(time.time())}",
            timestamp=datetime.now(),
            level=AlertLevel.WARNING,
            message=f"⛔ Strategy disabled: {strategy_name} - {reason}",
            source="human"
        )
        self.alerts.append(alert)
        
        if "disable_strategy" in self._control_callbacks:
            self._control_callbacks["disable_strategy"](strategy_name, reason)
        
        return {"status": "disabled", "strategy": strategy_name, "reason": reason}
    
    def acknowledge_alert(self, alert_id: str) -> Dict[str, Any]:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return {"status": "acknowledged", "alert_id": alert_id}
        
        return {"status": "not_found", "alert_id": alert_id}
    
    # ========================================================================
    # DATA COLLECTION
    # ========================================================================
    
    def update_metrics(self, metrics: TradingMetrics) -> None:
        """Update current metrics."""
        self.metrics_history.append(metrics)
    
    def update_position(self, symbol: str, position: Position) -> None:
        """Update a position."""
        self.current_positions[symbol] = position
    
    def remove_position(self, symbol: str) -> None:
        """Remove a closed position."""
        self.current_positions.pop(symbol, None)
    
    def add_alert(self, level: AlertLevel, message: str, source: str = "system") -> None:
        """Add an alert."""
        alert = DashboardAlert(
            id=f"alert_{int(time.time())}_{hash(message) % 1000}",
            timestamp=datetime.now(),
            level=level,
            message=message,
            source=source
        )
        self.alerts.append(alert)
    
    # ========================================================================
    # DATA RETRIEVAL
    # ========================================================================
    
    def get_dashboard_state(self) -> Dict[str, Any]:
        """Get complete dashboard state for display."""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            "system": {
                "state": self.system_state.value,
                "emergency_stop": self.emergency_stop_active,
                "trading_paused": self.trading_paused,
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            },
            "metrics": self._metrics_to_dict(latest_metrics) if latest_metrics else None,
            "positions": {
                symbol: self._position_to_dict(pos)
                for symbol, pos in self.current_positions.items()
            },
            "limits": {
                "max_position_size": self.max_position_size,
                "max_daily_loss": self.max_daily_loss,
                "max_drawdown_pct": self.max_drawdown_pct,
            },
            "alerts": [
                self._alert_to_dict(a) for a in list(self.alerts)[-20:]  # Last 20
            ],
            "learning": self._get_learning_status(),
            "quantum": self._get_quantum_status(),
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_metrics_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical metrics."""
        metrics = list(self.metrics_history)[-limit:]
        return [self._metrics_to_dict(m) for m in metrics]
    
    def get_alerts(self, unacknowledged_only: bool = False) -> List[Dict[str, Any]]:
        """Get alerts."""
        alerts = list(self.alerts)
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
        return [self._alert_to_dict(a) for a in alerts[-50:]]
    
    def get_control_state(self) -> Dict[str, Any]:
        """Get current control state."""
        return {
            "emergency_stop_active": self.emergency_stop_active,
            "trading_paused": self.trading_paused,
            "system_state": self.system_state.value,
            "available_actions": [
                "emergency_stop",
                "resume_trading",
                "pause_trading",
                "close_position",
                "close_all_positions",
                "set_max_position_size",
                "set_max_daily_loss",
                "enable_strategy",
                "disable_strategy",
            ],
        }
    
    def _get_learning_status(self) -> Dict[str, Any]:
        """Get learning system status."""
        if self.learning_hub:
            status = self.learning_hub.get_status()
            return {
                "learning_cycles": status.get("total_learning_cycles", 0),
                "ensemble_models": len(status.get("ensemble", {}).get("models", {})),
                "strategy_params": status.get("strategy_learner", {}).get("tracked_params", 0),
            }
        return {"status": "not_connected"}
    
    def _get_quantum_status(self) -> Dict[str, Any]:
        """Get quantum system status."""
        if self.learning_hub and hasattr(self.learning_hub, 'ensemble'):
            return {
                "enabled": True,
                "quantum_weight": self.learning_hub.ensemble.models.get("quantum", {}).learned_weight if hasattr(self.learning_hub.ensemble, 'models') else 0,
            }
        return {"enabled": False}
    
    def _metrics_to_dict(self, metrics: TradingMetrics) -> Dict[str, Any]:
        """Convert metrics to dict."""
        return {
            "timestamp": metrics.timestamp.isoformat(),
            "total_pnl": metrics.total_pnl,
            "daily_pnl": metrics.daily_pnl,
            "weekly_pnl": metrics.weekly_pnl,
            "monthly_pnl": metrics.monthly_pnl,
            "win_rate": metrics.win_rate,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "current_drawdown": metrics.current_drawdown,
            "total_trades": metrics.total_trades,
            "open_positions": metrics.open_positions,
            "total_exposure": metrics.total_exposure,
            "learning_cycles": metrics.learning_cycles,
            "parameters_updated": metrics.parameters_updated,
            "quantum_signals": metrics.quantum_signals,
            "avg_latency_ms": metrics.avg_latency_ms,
        }
    
    def _position_to_dict(self, pos: Position) -> Dict[str, Any]:
        """Convert position to dict."""
        return {
            "symbol": pos.symbol,
            "side": pos.side,
            "size": pos.size,
            "entry_price": pos.entry_price,
            "current_price": pos.current_price,
            "unrealized_pnl": pos.unrealized_pnl,
            "leverage": pos.leverage,
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
            "age_seconds": pos.age_seconds,
        }
    
    def _alert_to_dict(self, alert: DashboardAlert) -> Dict[str, Any]:
        """Convert alert to dict."""
        return {
            "id": alert.id,
            "timestamp": alert.timestamp.isoformat(),
            "level": alert.level.value,
            "message": alert.message,
            "acknowledged": alert.acknowledged,
            "source": alert.source,
        }


# Global singleton
_dashboard: Optional[InstitutionalDashboard] = None


def get_dashboard(
    orchestrator=None,
    learning_hub=None
) -> InstitutionalDashboard:
    """Get or create the institutional dashboard."""
    global _dashboard
    if _dashboard is None:
        _dashboard = InstitutionalDashboard(orchestrator, learning_hub)
    return _dashboard


__all__ = [
    "InstitutionalDashboard",
    "SystemState",
    "AlertLevel",
    "Position",
    "TradingMetrics",
    "DashboardAlert",
    "get_dashboard",
]
