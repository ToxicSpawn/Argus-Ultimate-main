"""
Enhanced Monitoring Dashboard for Quantum Modules
==================================================
Real-time monitoring of:
- Quantum module performance
- Alpha signal aggregation
- Strategy P&L tracking
- Risk metrics
- System health
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class ModuleStatus:
    """Status of a single module."""
    name: str
    category: str
    is_active: bool = True
    last_update: float = field(default_factory=time.time)
    performance_score: float = 0.0  # 0-100
    error_count: int = 0
    warning_count: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantumMetrics:
    """Quantum-specific metrics."""
    qubit_utilization: float = 0.0  # 0-100%
    entanglement_fidelity: float = 0.0  # 0-1
    coherence_time_us: float = 0.0  # microseconds
    quantum_advantage: float = 0.0  # multiplier vs classical
    circuit_depth: int = 0
    gate_errors: float = 0.0  # error rate


@dataclass
class TradingMetrics:
    """Real-time trading metrics."""
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    total_pnl: float = 0.0
    open_positions: int = 0
    win_rate_24h: float = 0.0
    sharpe_24h: float = 0.0
    max_drawdown_today: float = 0.0
    trades_24h: int = 0
    alpha_signals_received: int = 0
    alpha_signals_acted: int = 0


class QuantumModuleMonitor:
    """
    Quantum Module Monitor
    ======================
    Tracks performance and health of all quantum modules.
    """
    
    def __init__(self):
        self.modules: Dict[str, ModuleStatus] = {}
        self.quantum_metrics = QuantumMetrics()
        self.trading_metrics = TradingMetrics()
        self.alerts: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.max_history = 10000
        
    def register_module(self, name: str, category: str) -> None:
        """Register a module for monitoring."""
        self.modules[name] = ModuleStatus(
            name=name,
            category=category,
            last_update=time.time()
        )
        logger.info(f"Registered module: {name} ({category})")
    
    def update_module(
        self,
        name: str,
        performance_score: float,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update module metrics."""
        if name not in self.modules:
            self.register_module(name, "unknown")
        
        module = self.modules[name]
        module.performance_score = performance_score
        module.last_update = time.time()
        
        if metrics:
            module.metrics.update(metrics)
        
        # Check for alerts
        if performance_score < 50:
            self._add_alert("warning", f"Low performance: {name} ({performance_score:.1f})")
        if module.error_count > 10:
            self._add_alert("error", f"High error count: {name} ({module.error_count})")
    
    def record_error(self, module_name: str, error: str) -> None:
        """Record an error for a module."""
        if module_name in self.modules:
            self.modules[module_name].error_count += 1
        self._add_alert("error", f"Error in {module_name}: {error}")
    
    def record_warning(self, module_name: str, warning: str) -> None:
        """Record a warning for a module."""
        if module_name in self.modules:
            self.modules[module_name].warning_count += 1
        self._add_alert("warning", f"Warning in {module_name}: {warning}")
    
    def update_quantum_metrics(
        self,
        qubit_utilization: float = 0,
        entanglement_fidelity: float = 0,
        coherence_time_us: float = 0,
        quantum_advantage: float = 0,
        circuit_depth: int = 0,
        gate_errors: float = 0
    ) -> None:
        """Update quantum-specific metrics."""
        self.quantum_metrics.qubit_utilization = qubit_utilization
        self.quantum_metrics.entanglement_fidelity = entanglement_fidelity
        self.quantum_metrics.coherence_time_us = coherence_time_us
        self.quantum_metrics.quantum_advantage = quantum_advantage
        self.quantum_metrics.circuit_depth = circuit_depth
        self.quantum_metrics.gate_errors = gate_errors
    
    def update_trading_metrics(
        self,
        daily_pnl: float = 0,
        daily_pnl_pct: float = 0,
        total_pnl: float = 0,
        open_positions: int = 0,
        win_rate_24h: float = 0,
        sharpe_24h: float = 0,
        max_drawdown_today: float = 0,
        trades_24h: int = 0,
        alpha_signals_received: int = 0,
        alpha_signals_acted: int = 0
    ) -> None:
        """Update trading metrics."""
        self.trading_metrics.daily_pnl = daily_pnl
        self.trading_metrics.daily_pnl_pct = daily_pnl_pct
        self.trading_metrics.total_pnl = total_pnl
        self.trading_metrics.open_positions = open_positions
        self.trading_metrics.win_rate_24h = win_rate_24h
        self.trading_metrics.sharpe_24h = sharpe_24h
        self.trading_metrics.max_drawdown_today = max_drawdown_today
        self.trading_metrics.trades_24h = trades_24h
        self.trading_metrics.alpha_signals_received = alpha_signals_received
        self.trading_metrics.alpha_signals_acted = alpha_signals_acted
    
    def _add_alert(self, level: str, message: str) -> None:
        """Add an alert."""
        alert = {
            "timestamp": time.time(),
            "level": level,
            "message": message
        }
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        logger.log(
            getattr(logging, level.upper(), logging.INFO),
            message
        )
    
    def get_dashboard(self) -> Dict[str, Any]:
        """Get current dashboard state."""
        # Calculate aggregate metrics
        active_modules = sum(1 for m in self.modules.values() if m.is_active)
        total_modules = len(self.modules)
        avg_performance = (
            sum(m.performance_score for m in self.modules.values()) / total_modules
            if total_modules > 0 else 0
        )
        total_errors = sum(m.error_count for m in self.modules.values())
        total_warnings = sum(m.warning_count for m in self.modules.values())
        
        # Categorize modules
        modules_by_category: Dict[str, List[Dict]] = defaultdict(list)
        for module in self.modules.values():
            modules_by_category[module.category].append({
                "name": module.name,
                "active": module.is_active,
                "performance": module.performance_score,
                "errors": module.error_count,
                "warnings": module.warning_count,
                "last_update": module.last_update
            })
        
        return {
            "timestamp": time.time(),
            "system": {
                "active_modules": active_modules,
                "total_modules": total_modules,
                "avg_performance": round(avg_performance, 2),
                "total_errors": total_errors,
                "total_warnings": total_warnings
            },
            "quantum": {
                "qubit_utilization": self.quantum_metrics.qubit_utilization,
                "entanglement_fidelity": self.quantum_metrics.entanglement_fidelity,
                "coherence_time_us": self.quantum_metrics.coherence_time_us,
                "quantum_advantage": self.quantum_metrics.quantum_advantage,
                "circuit_depth": self.quantum_metrics.circuit_depth,
                "gate_errors": self.quantum_metrics.gate_errors
            },
            "trading": {
                "daily_pnl": self.trading_metrics.daily_pnl,
                "daily_pnl_pct": self.trading_metrics.daily_pnl_pct,
                "total_pnl": self.trading_metrics.total_pnl,
                "open_positions": self.trading_metrics.open_positions,
                "win_rate_24h": self.trading_metrics.win_rate_24h,
                "sharpe_24h": self.trading_metrics.sharpe_24h,
                "max_drawdown_today": self.trading_metrics.max_drawdown_today,
                "trades_24h": self.trading_metrics.trades_24h,
                "alpha_signals_received": self.trading_metrics.alpha_signals_received,
                "alpha_signals_acted": self.trading_metrics.alpha_signals_acted
            },
            "modules": dict(modules_by_category),
            "recent_alerts": self.alerts[-10:]
        }
    
    def get_health_score(self) -> Dict[str, Any]:
        """Calculate overall system health score."""
        scores = []
        
        # Module health
        if self.modules:
            module_health = sum(m.performance_score for m in self.modules.values()) / len(self.modules)
            scores.append(("modules", module_health, 0.3))
        
        # Error health
        total_ops = sum(m.error_count + m.warning_count for m in self.modules.values())
        error_health = max(0, 100 - total_ops)
        scores.append(("errors", error_health, 0.2))
        
        # Quantum health
        quantum_health = (
            self.quantum_metrics.entanglement_fidelity * 100 * 0.4 +
            max(0, 100 - self.quantum_metrics.gate_errors * 1000) * 0.3 +
            min(self.quantum_metrics.quantum_advantage * 20, 100) * 0.3
        )
        scores.append(("quantum", quantum_health, 0.2))
        
        # Trading health
        trading_health = min(100, max(0, 50 + self.trading_metrics.daily_pnl_pct * 1000))
        scores.append(("trading", trading_health, 0.3))
        
        # Weighted average
        total_weight = sum(w for _, _, w in scores)
        health_score = sum(s * w for _, s, w in scores) / total_weight
        
        return {
            "overall": round(health_score, 2),
            "components": {name: round(score, 2) for name, score, _ in scores},
            "status": "healthy" if health_score > 70 else "degraded" if health_score > 40 else "critical"
        }
    
    def format_dashboard_text(self) -> str:
        """Format dashboard as readable text."""
        dashboard = self.get_dashboard()
        health = self.get_health_score()
        
        text = f"""
╔══════════════════════════════════════════════════════════════╗
║           ARGUS QUANTUM MONITORING DASHBOARD                ║
║           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                        ║
╚══════════════════════════════════════════════════════════════╝

SYSTEM HEALTH: {health['status'].upper()} ({health['overall']:.1f}/100)
├─ Modules: {health['components'].get('modules', 0):.1f}/100
├─ Errors: {health['components'].get('errors', 0):.1f}/100
├─ Quantum: {health['components'].get('quantum', 0):.1f}/100
└─ Trading: {health['components'].get('trading', 0):.1f}/100

MODULES
├─ Active: {dashboard['system']['active_modules']}/{dashboard['system']['total_modules']}
├─ Avg Performance: {dashboard['system']['avg_performance']:.1f}/100
├─ Total Errors: {dashboard['system']['total_errors']}
└─ Total Warnings: {dashboard['system']['total_warnings']}

QUANTUM METRICS
├─ Qubit Utilization: {dashboard['quantum']['qubit_utilization']:.1f}%
├─ Entanglement Fidelity: {dashboard['quantum']['entanglement_fidelity']:.3f}
├─ Coherence Time: {dashboard['quantum']['coherence_time_us']:.1f}µs
├─ Quantum Advantage: {dashboard['quantum']['quantum_advantage']:.2f}x
├─ Circuit Depth: {dashboard['quantum']['circuit_depth']}
└─ Gate Errors: {dashboard['quantum']['gate_errors']:.4f}

TRADING METRICS
├─ Daily P&L: ${dashboard['trading']['daily_pnl']:.2f} ({dashboard['trading']['daily_pnl_pct']:.2f}%)
├─ Total P&L: ${dashboard['trading']['total_pnl']:.2f}
├─ Open Positions: {dashboard['trading']['open_positions']}
├─ Win Rate (24h): {dashboard['trading']['win_rate_24h']:.1f}%
├─ Sharpe (24h): {dashboard['trading']['sharpe_24h']:.2f}
├─ Max Drawdown Today: {dashboard['trading']['max_drawdown_today']:.2f}%
├─ Trades (24h): {dashboard['trading']['trades_24h']}
├─ Alpha Signals Received: {dashboard['trading']['alpha_signals_received']}
└─ Alpha Signals Acted: {dashboard['trading']['alpha_signals_acted']}

RECENT ALERTS
"""
        for alert in dashboard['recent_alerts'][-5:]:
            ts = datetime.fromtimestamp(alert['timestamp']).strftime('%H:%M:%S')
            text += f"├─ [{ts}] {alert['level'].upper()}: {alert['message']}\n"
        
        text += "╚══════════════════════════════════════════════════════════════╝"
        return text


class PerformanceTracker:
    """
    Performance Tracker
    ===================
    Tracks and analyzes strategy performance over time.
    """
    
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.daily_stats: Dict[str, Dict[str, Any]] = {}
        self.equity_curve: List[float] = []
        
    def record_trade(self, trade: Dict[str, Any]) -> None:
        """Record a completed trade."""
        self.trades.append({
            **trade,
            "timestamp": time.time()
        })
        
        # Update daily stats
        day = datetime.now().strftime('%Y-%m-%d')
        if day not in self.daily_stats:
            self.daily_stats[day] = {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "pnl": 0.0,
                "volume": 0.0
            }
        
        stats = self.daily_stats[day]
        stats["trades"] += 1
        stats["pnl"] += trade.get("pnl", 0)
        stats["volume"] += trade.get("size", 0) * trade.get("price", 0)
        
        if trade.get("pnl", 0) > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
    
    def update_equity(self, equity: float) -> None:
        """Update equity curve."""
        self.equity_curve.append(equity)
    
    def get_daily_report(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get daily performance report."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        return self.daily_stats.get(date, {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "pnl": 0.0,
            "volume": 0.0
        })
    
    def get_weekly_report(self) -> Dict[str, Any]:
        """Get weekly performance summary."""
        today = datetime.now()
        week_ago = today.timestamp() - 7 * 24 * 3600
        
        week_trades = [t for t in self.trades if t["timestamp"] > week_ago]
        
        if not week_trades:
            return {"trades": 0, "pnl": 0, "win_rate": 0}
        
        wins = sum(1 for t in week_trades if t.get("pnl", 0) > 0)
        total_pnl = sum(t.get("pnl", 0) for t in week_trades)
        
        return {
            "trades": len(week_trades),
            "wins": wins,
            "losses": len(week_trades) - wins,
            "win_rate": wins / len(week_trades) * 100,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(week_trades)
        }


# Singleton instance
_monitor: Optional[QuantumModuleMonitor] = None
_tracker: Optional[PerformanceTracker] = None


def get_monitor() -> QuantumModuleMonitor:
    """Get or create singleton monitor."""
    global _monitor
    if _monitor is None:
        _monitor = QuantumModuleMonitor()
    return _monitor


def get_tracker() -> PerformanceTracker:
    """Get or create singleton tracker."""
    global _tracker
    if _tracker is None:
        _tracker = PerformanceTracker()
    return _tracker


# Export
__all__ = [
    "ModuleStatus",
    "QuantumMetrics",
    "TradingMetrics",
    "QuantumModuleMonitor",
    "PerformanceTracker",
    "get_monitor",
    "get_tracker"
]
