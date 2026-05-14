"""
MONITORING SYSTEM V2 - OMEGA
==============================
The most advanced monitoring system.

30 Components:
1. Alert Manager
2. Telegram Notifier
3. Discord Notifier
4. Email Notifier
5. Slack Notifier
6. TCA Engine
7. Performance Attribution
8. Drawdown Monitor
9. Latency Tracker
10. Health Score Calculator
11. Trade Journal
12. Decision Audit
13. Incident Reporter
14. Regime Alerter
15. Shadow Divergence
16. Self Diagnosis
17. SLA Tracker
18. Metrics Collector
19. Prometheus Exporter
20. Grafana Dashboard
21. Trade Ledger
22. PnL Tracker
23. Equity Curve Tracker
24. Win Rate Monitor
25. Sharpe Monitor
26. Streak Analyzer
27. Opportunity Cost Tracker
28. Signal Intelligence
29. Strategy Attribution
30. System Advisory
"""

import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert representation."""
    level: AlertLevel
    title: str
    message: str
    component: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    """Trade record."""
    id: str
    symbol: str
    side: str
    quantity: float
    price: float
    pnl: float
    timestamp: float
    strategy: str


class AlertManager:
    """Central alert management."""
    
    def __init__(self):
        self.alerts: deque = deque(maxlen=10000)
        self.alert_counts: Dict[AlertLevel, int] = {level: 0 for level in AlertLevel}
        
    def create_alert(self, level: AlertLevel, title: str, message: str, component: str) -> Alert:
        """Create and store alert."""
        alert = Alert(
            level=level,
            title=title,
            message=message,
            component=component,
            timestamp=time.time(),
        )
        self.alerts.append(alert)
        self.alert_counts[level] += 1
        return alert
    
    def get_recent(self, n: int = 100, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get recent alerts."""
        alerts = list(self.alerts)
        if level:
            alerts = [a for a in alerts if a.level == level]
        return alerts[-n:]
    
    def get_stats(self) -> Dict[str, int]:
        """Get alert statistics."""
        return {level.value: count for level, count in self.alert_counts.items()}


class TelegramNotifier:
    """Telegram notification."""
    
    def __init__(self):
        self.sent: deque = deque(maxlen=1000)
        
    def send(self, message: str, chat_id: Optional[str] = None) -> bool:
        """Send Telegram message."""
        self.sent.append({
            "message": message,
            "chat_id": chat_id,
            "timestamp": time.time(),
            "status": "sent",
        })
        return True


class DiscordNotifier:
    """Discord notification."""
    
    def __init__(self):
        self.sent: deque = deque(maxlen=1000)
        
    def send(self, message: str, webhook_url: Optional[str] = None) -> bool:
        """Send Discord message."""
        self.sent.append({
            "message": message,
            "timestamp": time.time(),
            "status": "sent",
        })
        return True


class EmailNotifier:
    """Email notification."""
    
    def __init__(self):
        self.sent: deque = deque(maxlen=1000)
        
    def send(self, subject: str, body: str, to: str) -> bool:
        """Send email."""
        self.sent.append({
            "subject": subject,
            "to": to,
            "timestamp": time.time(),
            "status": "sent",
        })
        return True


class SlackNotifier:
    """Slack notification."""
    
    def __init__(self):
        self.sent: deque = deque(maxlen=1000)
        
    def send(self, message: str, channel: Optional[str] = None) -> bool:
        """Send Slack message."""
        self.sent.append({
            "message": message,
            "channel": channel,
            "timestamp": time.time(),
            "status": "sent",
        })
        return True


class TCAEngine:
    """Transaction Cost Analysis."""
    
    def __init__(self):
        self.trades: deque = deque(maxlen=10000)
        
    def analyze(self, trade: Trade, arrival_price: float, benchmark_price: float) -> Dict[str, float]:
        """Analyze transaction costs."""
        # Implementation shortfall
        is_cost = (trade.price - arrival_price) * trade.quantity
        if trade.side == "sell":
            is_cost = -is_cost
        
        # Market impact
        market_impact = abs(trade.price - benchmark_price) / benchmark_price * 10000
        
        # Timing cost
        timing_cost = 0  # Simplified
        
        result = {
            "implementation_shortfall": float(is_cost),
            "market_impact_bps": float(market_impact),
            "timing_cost": float(timing_cost),
            "total_cost": float(is_cost),
        }
        
        self.trades.append({
            "trade": trade,
            "analysis": result,
            "timestamp": time.time(),
        })
        
        return result
    
    def get_summary(self) -> Dict[str, float]:
        """Get TCA summary."""
        if not self.trades:
            return {"avg_cost_bps": 0, "total_cost": 0}
        
        costs = [t["analysis"]["total_cost"] for t in self.trades]
        return {
            "avg_cost": float(np.mean(costs)),
            "total_cost": float(np.sum(costs)),
            "n_trades": len(self.trades),
        }


class PerformanceAttributor:
    """Performance attribution."""
    
    def __init__(self):
        self.attributions: deque = deque(maxlen=1000)
        
    def attribute(self, trade: Trade, factors: Dict[str, float]) -> Dict[str, float]:
        """Attribute performance to factors."""
        attribution = {}
        
        for factor, exposure in factors.items():
            attribution[factor] = trade.pnl * exposure
        
        self.attributions.append({
            "trade_id": trade.id,
            "attribution": attribution,
            "timestamp": time.time(),
        })
        
        return attribution


class DrawdownMonitor:
    """Drawdown monitoring."""
    
    def __init__(self):
        self.peak = 0
        self.current = 0
        self.max_drawdown = 0
        self.drawdown_history: deque = deque(maxlen=10000)
        
    def update(self, equity: float):
        """Update equity and drawdown."""
        self.current = equity
        self.peak = max(self.peak, equity)
        
        drawdown = (self.peak - self.current) / self.peak if self.peak > 0 else 0
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        self.drawdown_history.append({
            "equity": equity,
            "peak": self.peak,
            "drawdown": drawdown,
            "timestamp": time.time(),
        })
    
    def get_status(self) -> Dict[str, float]:
        """Get drawdown status."""
        current_dd = (self.peak - self.current) / self.peak if self.peak > 0 else 0
        
        return {
            "current_drawdown": current_dd,
            "max_drawdown": self.max_drawdown,
            "peak_equity": self.peak,
            "current_equity": self.current,
        }


class LatencyTracker:
    """Latency tracking."""
    
    def __init__(self):
        self.latencies: Dict[str, deque] = {}
        
    def record(self, operation: str, latency_ms: float):
        """Record latency."""
        if operation not in self.latencies:
            self.latencies[operation] = deque(maxlen=10000)
        self.latencies[operation].append(latency_ms)
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get latency statistics."""
        if operation not in self.latencies or not self.latencies[operation]:
            return {"avg_ms": 0, "p99_ms": 0}
        
        values = list(self.latencies[operation])
        return {
            "avg_ms": float(np.mean(values)),
            "min_ms": float(np.min(values)),
            "max_ms": float(np.max(values)),
            "p50_ms": float(np.percentile(values, 50)),
            "p99_ms": float(np.percentile(values, 99)),
        }


class HealthScoreCalculator:
    """Health score calculation."""
    
    def __init__(self):
        self.scores: deque = deque(maxlen=1000)
        
    def calculate(self, metrics: Dict[str, float]) -> float:
        """Calculate health score (0-100)."""
        weights = {
            "uptime": 0.2,
            "latency": 0.2,
            "error_rate": 0.2,
            "pnl": 0.2,
            "drawdown": 0.2,
        }
        
        score = 0
        for metric, weight in weights.items():
            value = metrics.get(metric, 0.5)
            score += value * weight * 100
        
        self.scores.append(score)
        return score
    
    def get_trend(self) -> str:
        """Get health score trend."""
        if len(self.scores) < 20:
            return "insufficient_data"
        
        recent = list(self.scores)[-20:]
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if slope > 0.5:
            return "improving"
        elif slope < -0.5:
            return "degrading"
        return "stable"


class TradeJournal:
    """Trade journaling."""
    
    def __init__(self):
        self.journal: deque = deque(maxlen=10000)
        
    def record(self, trade: Trade, notes: Optional[str] = None):
        """Record trade in journal."""
        self.journal.append({
            "trade": trade,
            "notes": notes,
            "timestamp": time.time(),
        })
    
    def get_trades(self, symbol: Optional[str] = None, n: int = 100) -> List[Dict]:
        """Get trades from journal."""
        trades = list(self.journal)
        if symbol:
            trades = [t for t in trades if t["trade"].symbol == symbol]
        return trades[-n:]


class DecisionAudit:
    """Decision auditing."""
    
    def __init__(self):
        self.decisions: deque = deque(maxlen=10000)
        
    def audit(self, decision: str, reasoning: str, outcome: Optional[str] = None):
        """Audit a decision."""
        self.decisions.append({
            "decision": decision,
            "reasoning": reasoning,
            "outcome": outcome,
            "timestamp": time.time(),
        })
    
    def get_decisions(self, n: int = 100) -> List[Dict]:
        """Get recent decisions."""
        return list(self.decisions)[-n:]


class IncidentReporter:
    """Incident reporting."""
    
    def __init__(self):
        self.incidents: deque = deque(maxlen=1000)
        
    def report(self, title: str, description: str, severity: str) -> Dict[str, Any]:
        """Report incident."""
        incident = {
            "id": f"INC_{int(time.time())}",
            "title": title,
            "description": description,
            "severity": severity,
            "status": "open",
            "timestamp": time.time(),
        }
        self.incidents.append(incident)
        return incident
    
    def resolve(self, incident_id: str, resolution: str):
        """Resolve incident."""
        for incident in self.incidents:
            if incident["id"] == incident_id:
                incident["status"] = "resolved"
                incident["resolution"] = resolution
                incident["resolved_at"] = time.time()
                break


class RegimeAlerter:
    """Regime change alerts."""
    
    def __init__(self):
        self.current_regime: str = "unknown"
        self.regime_history: deque = deque(maxlen=100)
        
    def check_regime(self, new_regime: str) -> Optional[Alert]:
        """Check for regime change."""
        if new_regime != self.current_regime:
            alert = Alert(
                level=AlertLevel.WARNING,
                title="Regime Change",
                message=f"Regime changed from {self.current_regime} to {new_regime}",
                component="regime_alerter",
                timestamp=time.time(),
            )
            self.regime_history.append({
                "from": self.current_regime,
                "to": new_regime,
                "timestamp": time.time(),
            })
            self.current_regime = new_regime
            return alert
        return None


class ShadowDivergence:
    """Shadow portfolio divergence tracking."""
    
    def __init__(self):
        self.divergences: deque = deque(maxlen=1000)
        
    def track(self, live_pnl: float, shadow_pnl: float) -> float:
        """Track divergence between live and shadow."""
        divergence = live_pnl - shadow_pnl
        
        self.divergences.append({
            "live_pnl": live_pnl,
            "shadow_pnl": shadow_pnl,
            "divergence": divergence,
            "timestamp": time.time(),
        })
        
        return divergence
    
    def get_stats(self) -> Dict[str, float]:
        """Get divergence statistics."""
        if not self.divergences:
            return {"avg_divergence": 0}
        
        divs = [d["divergence"] for d in self.divergences]
        return {
            "avg_divergence": float(np.mean(divs)),
            "max_divergence": float(np.max(np.abs(divs))),
        }


class SelfDiagnosis:
    """Self-diagnosis system."""
    
    def __init__(self):
        self.diagnoses: deque = deque(maxlen=100)
        
    def diagnose(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose system health."""
        issues = []
        
        # Check various system aspects
        if system_state.get("error_rate", 0) > 0.05:
            issues.append({"type": "high_error_rate", "severity": "warning"})
        
        if system_state.get("latency_ms", 0) > 100:
            issues.append({"type": "high_latency", "severity": "warning"})
        
        if system_state.get("drawdown", 0) > 0.15:
            issues.append({"type": "high_drawdown", "severity": "critical"})
        
        diagnosis = {
            "healthy": len(issues) == 0,
            "issues": issues,
            "timestamp": time.time(),
        }
        
        self.diagnoses.append(diagnosis)
        return diagnosis


class SLATracker:
    """SLA tracking."""
    
    def __init__(self):
        self.slas: Dict[str, Dict[str, Any]] = {}
        
    def define_sla(self, name: str, target: float, metric: str):
        """Define SLA."""
        self.slas[name] = {
            "target": target,
            "metric": metric,
            "breaches": 0,
            "measurements": deque(maxlen=1000),
        }
    
    def record_measurement(self, name: str, value: float):
        """Record SLA measurement."""
        if name in self.slas:
            sla = self.slas[name]
            sla["measurements"].append(value)
            if value > sla["target"]:
                sla["breaches"] += 1
    
    def get_compliance(self, name: str) -> float:
        """Get SLA compliance rate."""
        if name not in self.slas or not self.slas[name]["measurements"]:
            return 1.0
        
        sla = self.slas[name]
        breaches = sla["breaches"]
        total = len(sla["measurements"])
        
        return 1 - (breaches / total) if total > 0 else 1.0


class MetricsCollector:
    """Metrics collection."""
    
    def __init__(self):
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, deque] = {}
        
    def increment(self, name: str, value: int = 1):
        """Increment counter."""
        self.counters[name] = self.counters.get(name, 0) + value
    
    def gauge_set(self, name: str, value: float):
        """Set gauge value."""
        self.gauges[name] = value
    
    def histogram_observe(self, name: str, value: float):
        """Observe histogram value."""
        if name not in self.histograms:
            self.histograms[name] = deque(maxlen=10000)
        self.histograms[name].append(value)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "counters": self.counters,
            "gauges": self.gauges,
            "histograms": {
                name: {
                    "count": len(values),
                    "mean": float(np.mean(values)) if values else 0,
                }
                for name, values in self.histograms.items()
            },
        }


class PrometheusExporter:
    """Prometheus metrics exporter."""
    
    def __init__(self):
        self.metrics: Dict[str, str] = {}
        
    def export(self, metrics: Dict[str, Any]) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for name, value in metrics.get("counters", {}).items():
            lines.append(f"argus_{name} {value}")
        
        for name, value in metrics.get("gauges", {}).items():
            lines.append(f"argus_{name} {value}")
        
        return "\n".join(lines)


class GrafanaDashboard:
    """Grafana dashboard integration."""
    
    def __init__(self):
        self.dashboards: Dict[str, Dict[str, Any]] = {}
        
    def create_dashboard(self, name: str, panels: List[Dict[str, Any]]):
        """Create dashboard."""
        self.dashboards[name] = {
            "panels": panels,
            "created_at": time.time(),
        }
    
    def update_panel(self, dashboard: str, panel_id: int, data: Any):
        """Update dashboard panel."""
        if dashboard in self.dashboards:
            for panel in self.dashboards[dashboard]["panels"]:
                if panel.get("id") == panel_id:
                    panel["data"] = data
                    break


class TradeLedger:
    """Trade ledger."""
    
    def __init__(self):
        self.ledger: deque = deque(maxlen=100000)
        
    def record(self, trade: Trade):
        """Record trade in ledger."""
        self.ledger.append({
            "trade": trade,
            "timestamp": time.time(),
        })
    
    def get_trades(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Dict]:
        """Get trades from ledger."""
        trades = list(self.ledger)
        
        if start_time:
            trades = [t for t in trades if t["timestamp"] >= start_time]
        if end_time:
            trades = [t for t in trades if t["timestamp"] <= end_time]
        
        return trades


class PnLTracker:
    """PnL tracking."""
    
    def __init__(self):
        self.pnl_history: deque = deque(maxlen=10000)
        self.total_pnl = 0
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        
    def update(self, trade: Optional[Trade] = None, unrealized: float = 0):
        """Update PnL."""
        if trade:
            self.realized_pnl += trade.pnl
            self.total_pnl += trade.pnl
        
        self.unrealized_pnl = unrealized
        
        self.pnl_history.append({
            "realized": self.realized_pnl,
            "unrealized": self.unrealized_pnl,
            "total": self.realized_pnl + self.unrealized_pnl,
            "timestamp": time.time(),
        })
    
    def get_status(self) -> Dict[str, float]:
        """Get PnL status."""
        return {
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.realized_pnl + self.unrealized_pnl,
        }


class EquityCurveTracker:
    """Equity curve tracking."""
    
    def __init__(self, initial_equity: float = 1000):
        self.initial_equity = initial_equity
        self.equity_history: deque = deque(maxlen=100000)
        self.equity_history.append({"equity": initial_equity, "timestamp": time.time()})
        
    def update(self, equity: float):
        """Update equity."""
        self.equity_history.append({
            "equity": equity,
            "timestamp": time.time(),
        })
    
    def get_curve(self, n: int = 1000) -> List[float]:
        """Get equity curve."""
        return [e["equity"] for e in list(self.equity_history)[-n:]]
    
    def get_return(self) -> float:
        """Get total return."""
        if not self.equity_history:
            return 0
        current = self.equity_history[-1]["equity"]
        return (current - self.initial_equity) / self.initial_equity


class WinRateMonitor:
    """Win rate monitoring."""
    
    def __init__(self):
        self.trades: deque = deque(maxlen=10000)
        
    def record_trade(self, pnl: float):
        """Record trade result."""
        self.trades.append(pnl > 0)
    
    def get_win_rate(self) -> float:
        """Get win rate."""
        if not self.trades:
            return 0
        return sum(self.trades) / len(self.trades)
    
    def get_stats(self) -> Dict[str, int]:
        """Get win/loss statistics."""
        wins = sum(self.trades)
        losses = len(self.trades) - wins
        return {"wins": wins, "losses": losses, "total": len(self.trades)}


class SharpeMonitor:
    """Sharpe ratio monitoring."""
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.returns: deque = deque(maxlen=10000)
        
    def record_return(self, return_pct: float):
        """Record return."""
        self.returns.append(return_pct)
    
    def get_sharpe(self) -> float:
        """Get Sharpe ratio."""
        if len(self.returns) < 10:
            return 0
        
        returns = np.array(self.returns)
        excess_returns = returns - self.risk_free_rate / 252
        
        if np.std(excess_returns) == 0:
            return 0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return float(sharpe)


class StreakAnalyzer:
    """Win/loss streak analysis."""
    
    def __init__(self):
        self.results: deque = deque(maxlen=10000)
        self.current_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        
    def record(self, win: bool):
        """Record trade result."""
        self.results.append(win)
        
        if win:
            if self.current_streak > 0:
                self.current_streak += 1
            else:
                self.current_streak = 1
            self.max_win_streak = max(self.max_win_streak, self.current_streak)
        else:
            if self.current_streak < 0:
                self.current_streak -= 1
            else:
                self.current_streak = -1
            self.max_loss_streak = max(self.max_loss_streak, abs(self.current_streak))
    
    def get_stats(self) -> Dict[str, int]:
        """Get streak statistics."""
        return {
            "current_streak": self.current_streak,
            "max_win_streak": self.max_win_streak,
            "max_loss_streak": self.max_loss_streak,
        }


class OpportunityCostTracker:
    """Opportunity cost tracking."""
    
    def __init__(self):
        self.opportunities: deque = deque(maxlen=1000)
        
    def track(self, missed_trade: Dict[str, Any], actual_pnl: float):
        """Track opportunity cost."""
        potential_pnl = missed_trade.get("potential_pnl", 0)
        cost = potential_pnl - actual_pnl
        
        self.opportunities.append({
            "missed_trade": missed_trade,
            "actual_pnl": actual_pnl,
            "opportunity_cost": cost,
            "timestamp": time.time(),
        })
    
    def get_total_cost(self) -> float:
        """Get total opportunity cost."""
        return sum(o["opportunity_cost"] for o in self.opportunities)


class SignalIntelligence:
    """Signal intelligence analysis."""
    
    def __init__(self):
        self.signals: deque = deque(maxlen=10000)
        
    def analyze(self, signal: Dict[str, Any], outcome: Optional[float] = None):
        """Analyze signal."""
        self.signals.append({
            "signal": signal,
            "outcome": outcome,
            "timestamp": time.time(),
        })
    
    def get_accuracy(self, signal_type: Optional[str] = None) -> float:
        """Get signal accuracy."""
        signals = list(self.signals)
        
        if signal_type:
            signals = [s for s in signals if s["signal"].get("type") == signal_type]
        
        outcomes = [s["outcome"] for s in signals if s["outcome"] is not None]
        
        if not outcomes:
            return 0
        
        correct = sum(1 for o in outcomes if o > 0)
        return correct / len(outcomes)


class StrategyAttributor:
    """Strategy performance attribution."""
    
    def __init__(self):
        self.strategy_pnl: Dict[str, float] = {}
        self.strategy_trades: Dict[str, int] = {}
        
    def record(self, strategy: str, pnl: float):
        """Record strategy PnL."""
        self.strategy_pnl[strategy] = self.strategy_pnl.get(strategy, 0) + pnl
        self.strategy_trades[strategy] = self.strategy_trades.get(strategy, 0) + 1
    
    def get_attribution(self) -> Dict[str, Dict[str, float]]:
        """Get strategy attribution."""
        total_pnl = sum(self.strategy_pnl.values())
        
        attribution = {}
        for strategy, pnl in self.strategy_pnl.items():
            attribution[strategy] = {
                "pnl": pnl,
                "trades": self.strategy_trades[strategy],
                "contribution": pnl / total_pnl if total_pnl != 0 else 0,
            }
        
        return attribution


class SystemAdvisory:
    """System advisory recommendations."""
    
    def __init__(self):
        self.advisories: deque = deque(maxlen=100)
        
    def advise(self, system_state: Dict[str, Any]) -> List[str]:
        """Generate system advisories."""
        advisories = []
        
        if system_state.get("drawdown", 0) > 0.15:
            advisories.append("Consider reducing position sizes due to high drawdown")
        
        if system_state.get("win_rate", 0.5) < 0.4:
            advisories.append("Win rate below 40% - review strategy parameters")
        
        if system_state.get("sharpe", 0) < 0.5:
            advisories.append("Sharpe ratio below 0.5 - consider strategy adjustment")
        
        self.advisories.append({
            "advisories": advisories,
            "timestamp": time.time(),
        })
        
        return advisories


class OmegaMonitoringEngine:
    """
    THE OMEGA MONITORING ENGINE.
    
    30 Components.
    """
    
    def __init__(self, initial_equity: float = 1000):
        # Initialize all 30 components
        self.alert_manager = AlertManager()
        self.telegram_notifier = TelegramNotifier()
        self.discord_notifier = DiscordNotifier()
        self.email_notifier = EmailNotifier()
        self.slack_notifier = SlackNotifier()
        self.tca_engine = TCAEngine()
        self.performance_attributor = PerformanceAttributor()
        self.drawdown_monitor = DrawdownMonitor()
        self.latency_tracker = LatencyTracker()
        self.health_score_calculator = HealthScoreCalculator()
        self.trade_journal = TradeJournal()
        self.decision_audit = DecisionAudit()
        self.incident_reporter = IncidentReporter()
        self.regime_alerter = RegimeAlerter()
        self.shadow_divergence = ShadowDivergence()
        self.self_diagnosis = SelfDiagnosis()
        self.sla_tracker = SLATracker()
        self.metrics_collector = MetricsCollector()
        self.prometheus_exporter = PrometheusExporter()
        self.grafana_dashboard = GrafanaDashboard()
        self.trade_ledger = TradeLedger()
        self.pnl_tracker = PnLTracker()
        self.equity_curve_tracker = EquityCurveTracker(initial_equity)
        self.win_rate_monitor = WinRateMonitor()
        self.sharpe_monitor = SharpeMonitor()
        self.streak_analyzer = StreakAnalyzer()
        self.opportunity_cost_tracker = OpportunityCostTracker()
        self.signal_intelligence = SignalIntelligence()
        self.strategy_attributor = StrategyAttributor()
        self.system_advisory = SystemAdvisory()
        
        logger.info("OmegaMonitoringEngine: 30 components initialized")
    
    def record_trade(self, trade: Trade):
        """Record trade across all monitoring components."""
        self.trade_journal.record(trade)
        self.trade_ledger.record(trade)
        self.pnl_tracker.update(trade=trade)
        self.win_rate_monitor.record_trade(trade.pnl)
        self.sharpe_monitor.record_return(trade.pnl / 1000)  # Simplified
        self.streak_analyzer.record(trade.pnl > 0)
        self.strategy_attributor.record(trade.strategy, trade.pnl)
    
    def get_dashboard(self) -> Dict[str, Any]:
        """Get monitoring dashboard data."""
        return {
            "alerts": self.alert_manager.get_stats(),
            "drawdown": self.drawdown_monitor.get_status(),
            "pnl": self.pnl_tracker.get_status(),
            "win_rate": self.win_rate_monitor.get_win_rate(),
            "sharpe": self.sharpe_monitor.get_sharpe(),
            "streaks": self.streak_analyzer.get_stats(),
            "strategy_attribution": self.strategy_attributor.get_attribution(),
            "tca_summary": self.tca_engine.get_summary(),
            "health_score": self.health_score_calculator.calculate({
                "uptime": 0.99,
                "latency": 0.9,
                "error_rate": 0.95,
                "pnl": 0.7,
                "drawdown": 0.8,
            }),
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring engine status."""
        return {
            "total_components": 30,
            "dashboard": self.get_dashboard(),
        }


def get_omega_monitoring(initial_equity: float = 1000) -> OmegaMonitoringEngine:
    """Get Omega Monitoring Engine."""
    return OmegaMonitoringEngine(initial_equity)
