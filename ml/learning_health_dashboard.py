# pyright: reportMissingImports=false
"""
Learning Health Dashboard for Argus Trading.

This module provides a comprehensive dashboard to monitor all learning systems,
track performance, and ensure continuous improvement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SystemHealth(Enum):
    """Health status of a learning system."""
    HEALTHY = auto()
    DEGRADED = auto()
    WARNING = auto()
    CRITICAL = auto()
    OFFLINE = auto()


@dataclass
class LearningSystemMetrics:
    """Metrics for a single learning system."""
    system_name: str
    system_type: str
    health: SystemHealth
    performance: float  # 0-1
    last_updated: datetime
    training_samples: int
    error_rate: float
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardMetrics:
    """Overall dashboard metrics."""
    timestamp: datetime
    total_systems: int
    healthy_systems: int
    degraded_systems: int
    warning_systems: int
    critical_systems: int
    overall_health: SystemHealth
    average_performance: float
    active_learning_rate: float  # Samples per hour
    total_training_samples: int


class LearningHealthDashboard:
    """Dashboard for monitoring all learning systems."""

    def __init__(self):
        """Initialize the learning health dashboard."""
        self.systems: Dict[str, LearningSystemMetrics] = {}
        self.health_history: List[DashboardMetrics] = []
        self.alerts: List[Dict[str, Any]] = []
        
    def register_system(self,
                       system_name: str,
                       system_type: str,
                       performance: float = 0.0) -> None:
        """Register a learning system for monitoring."""
        self.systems[system_name] = LearningSystemMetrics(
            system_name=system_name,
            system_type=system_type,
            health=SystemHealth.HEALTHY,
            performance=performance,
            last_updated=datetime.now(),
            training_samples=0,
            error_rate=0.0,
            latency_ms=0.0
        )
        logger.info(f"Registered learning system: {system_name}")
    
    def update_system_metrics(self,
                             system_name: str,
                             performance: float,
                             training_samples: int,
                             error_rate: float,
                             latency_ms: float) -> None:
        """Update metrics for a learning system."""
        if system_name not in self.systems:
            logger.warning(f"Unknown system: {system_name}")
            return
        
        system = self.systems[system_name]
        system.performance = performance
        system.training_samples = training_samples
        system.error_rate = error_rate
        system.latency_ms = latency_ms
        system.last_updated = datetime.now()
        
        # Update health status
        system.health = self._calculate_health(performance, error_rate, latency_ms)
        
        # Check for alerts
        self._check_alerts(system)
    
    def _calculate_health(self, performance: float, error_rate: float, latency_ms: float) -> SystemHealth:
        """Calculate health status based on metrics."""
        if performance < 0.3 or error_rate > 0.3:
            return SystemHealth.CRITICAL
        elif performance < 0.5 or error_rate > 0.15:
            return SystemHealth.WARNING
        elif performance < 0.7 or latency_ms > 1000:
            return SystemHealth.DEGRADED
        else:
            return SystemHealth.HEALTHY
    
    def _check_alerts(self, system: LearningSystemMetrics) -> None:
        """Check for alert conditions."""
        # Performance drop alert
        if system.performance < 0.4:
            self.alerts.append({
                "type": "performance_drop",
                "system": system.system_name,
                "severity": "high",
                "message": f"Low performance: {system.performance:.2%}",
                "timestamp": datetime.now()
            })
        
        # High error rate alert
        if system.error_rate > 0.2:
            self.alerts.append({
                "type": "high_error_rate",
                "system": system.system_name,
                "severity": "high",
                "message": f"High error rate: {system.error_rate:.2%}",
                "timestamp": datetime.now()
            })
        
        # High latency alert
        if system.latency_ms > 500:
            self.alerts.append({
                "type": "high_latency",
                "system": system.system_name,
                "severity": "medium",
                "message": f"High latency: {system.latency_ms:.0f}ms",
                "timestamp": datetime.now()
            })
    
    def get_dashboard_metrics(self) -> DashboardMetrics:
        """Get overall dashboard metrics."""
        if not self.systems:
            return DashboardMetrics(
                timestamp=datetime.now(),
                total_systems=0,
                healthy_systems=0,
                degraded_systems=0,
                warning_systems=0,
                critical_systems=0,
                overall_health=SystemHealth.OFFLINE,
                average_performance=0.0,
                active_learning_rate=0.0,
                total_training_samples=0
            )
        
        # Count systems by health
        health_counts = {h: 0 for h in SystemHealth}
        total_performance = 0.0
        total_samples = 0
        
        for system in self.systems.values():
            health_counts[system.health] += 1
            total_performance += system.performance
            total_samples += system.training_samples
        
        # Determine overall health
        if health_counts[SystemHealth.CRITICAL] > 0:
            overall_health = SystemHealth.CRITICAL
        elif health_counts[SystemHealth.WARNING] > 0:
            overall_health = SystemHealth.WARNING
        elif health_counts[SystemHealth.DEGRADED] > 0:
            overall_health = SystemHealth.DEGRADED
        else:
            overall_health = SystemHealth.HEALTHY
        
        avg_performance = total_performance / len(self.systems)
        
        metrics = DashboardMetrics(
            timestamp=datetime.now(),
            total_systems=len(self.systems),
            healthy_systems=health_counts[SystemHealth.HEALTHY],
            degraded_systems=health_counts[SystemHealth.DEGRADED],
            warning_systems=health_counts[SystemHealth.WARNING],
            critical_systems=health_counts[SystemHealth.CRITICAL],
            overall_health=overall_health,
            average_performance=avg_performance,
            active_learning_rate=0.0,  # Would be calculated from actual data
            total_training_samples=total_samples
        )
        
        self.health_history.append(metrics)
        
        # Keep only recent history
        if len(self.health_history) > 1000:
            self.health_history = self.health_history[-1000:]
        
        return metrics
    
    def get_system_details(self, system_name: str) -> Optional[LearningSystemMetrics]:
        """Get detailed metrics for a specific system."""
        return self.systems.get(system_name)
    
    def get_alerts(self, 
                  severity: Optional[str] = None,
                  limit: int = 50) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered by severity."""
        alerts = self.alerts
        
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        
        # Sort by timestamp (most recent first)
        alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return alerts[:limit]
    
    def get_performance_trend(self, system_name: str, window: int = 100) -> List[float]:
        """Get performance trend for a system."""
        # This would track historical performance
        # For now, return simulated data
        return [random.uniform(0.6, 0.9) for _ in range(min(window, 50))]
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive health report."""
        metrics = self.get_dashboard_metrics()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall": {
                "health": metrics.overall_health.name,
                "performance": f"{metrics.average_performance:.2%}",
                "systems_count": metrics.total_systems,
                "healthy_ratio": f"{metrics.healthy_systems / max(1, metrics.total_systems):.2%}"
            },
            "systems": {
                name: {
                    "type": system.system_type,
                    "health": system.health.name,
                    "performance": f"{system.performance:.2%}",
                    "samples": system.training_samples,
                    "error_rate": f"{system.error_rate:.2%}",
                    "latency": f"{system.latency_ms:.0f}ms"
                }
                for name, system in self.systems.items()
            },
            "alerts": {
                "total": len(self.alerts),
                "critical": sum(1 for a in self.alerts if a["severity"] == "high"),
                "warning": sum(1 for a in self.alerts if a["severity"] == "medium"),
                "recent": self.get_alerts(limit=5)
            },
            "recommendations": self._generate_recommendations(metrics)
        }
        
        return report
    
    def _generate_recommendations(self, metrics: DashboardMetrics) -> List[str]:
        """Generate recommendations based on current state."""
        recommendations = []
        
        if metrics.critical_systems > 0:
            recommendations.append("CRITICAL: Address critical system health issues immediately")
        
        if metrics.warning_systems > 0:
            recommendations.append("WARNING: Investigate systems with degraded performance")
        
        if metrics.average_performance < 0.6:
            recommendations.append("Consider increasing training frequency or model complexity")
        
        if metrics.total_training_samples < 1000:
            recommendations.append("Collect more training data for better model performance")
        
        critical_alerts = self.get_alerts(severity="high", limit=10)
        if critical_alerts:
            recommendations.append(f"Address {len(critical_alerts)} critical alerts")
        
        return recommendations
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a quick health summary."""
        metrics = self.get_dashboard_metrics()
        
        return {
            "overall_health": metrics.overall_health.name,
            "performance": f"{metrics.average_performance:.2%}",
            "systems": f"{metrics.healthy_systems}/{metrics.total_systems} healthy",
            "total_samples": metrics.total_training_samples,
            "active_alerts": len(self.alerts)
        }


class SystemIntegrator:
    """Integrates all learning systems with the dashboard."""

    def __init__(self, dashboard: Optional[LearningHealthDashboard] = None):
        self.dashboard = dashboard or LearningHealthDashboard()
        self._register_default_systems()
    
    def _register_default_systems(self) -> None:
        """Register all default learning systems."""
        default_systems = [
            ("MetaLearner", "meta_learning", 0.75),
            ("EnsembleSignalHub", "ensemble", 0.80),
            ("OnlineLearner", "online_learning", 0.72),
            ("QuantumRL", "quantum_reinforcement_learning", 0.68),
            ("CausalInference", "causal", 0.70),
            ("KnowledgeDistillation", "distillation", 0.85),
            ("MultiAgentRL", "multi_agent", 0.73),
            ("RLHF", "human_feedback", 0.65),
            ("UncertaintyQuantifier", "uncertainty", 0.78),
            ("AdversarialTrainer", "adversarial", 0.66),
            ("ActiveLearner", "active_learning", 0.71),
            ("TransferLearner", "transfer_learning", 0.69),
        ]
        
        for name, type_, perf in default_systems:
            self.dashboard.register_system(name, type_, perf)
        
        logger.info(f"Registered {len(default_systems)} learning systems")
    
    def update_all_systems(self) -> Dict[str, Any]:
        """Update metrics for all systems (simulated)."""
        import random
        
        updates = {}
        for name, system in self.dashboard.systems.items():
            # Simulate metric updates
            performance = max(0, min(1, system.performance + random.uniform(-0.05, 0.05)))
            samples = system.training_samples + random.randint(0, 100)
            error_rate = max(0, min(1, system.error_rate + random.uniform(-0.02, 0.02)))
            latency = max(10, system.latency_ms + random.uniform(-50, 50))
            
            self.dashboard.update_system_metrics(
                name, performance, samples, error_rate, latency
            )
            
            updates[name] = {
                "performance": performance,
                "health": system.health.name
            }
        
        return updates
    
    def get_full_report(self) -> Dict[str, Any]:
        """Get comprehensive report of all systems."""
        return self.dashboard.generate_report()


import random


__all__ = [
    "LearningHealthDashboard",
    "SystemIntegrator",
    "LearningSystemMetrics",
    "DashboardMetrics",
    "SystemHealth"
]