"""
Monitoring Module
=================

System monitoring, metrics collection, and alerting.
Refactored from unified_trading_system.py.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """System metric."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert."""
    level: str  # info, warning, error, critical
    message: str
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SystemMonitor:
    """
    Monitors system health and performance.
    """
    
    def __init__(self):
        self._metrics: Dict[str, List[Metric]] = {}
        self._alerts: deque = deque(maxlen=1000)
        self._tick_count = 0
        self._last_tick_time: Optional[datetime] = None
        self._running = False
        
        logger.info("SystemMonitor initialized")
    
    async def initialize(self):
        """Initialize monitor."""
        logger.info("System monitor initialized")
    
    async def record_tick(
        self,
        symbol: str,
        price: float,
        signal_count: int
    ):
        """Record a market data tick."""
        self._tick_count += 1
        self._last_tick_time = datetime.utcnow()
        
        # Record tick metric
        await self.record_metric(
            "ticks_total",
            float(self._tick_count),
            labels={"symbol": symbol}
        )
        
        await self.record_metric(
            "signals_generated",
            float(signal_count),
            labels={"symbol": symbol}
        )
    
    async def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a metric."""
        if name not in self._metrics:
            self._metrics[name] = deque(maxlen=10000)
        
        metric = Metric(
            name=name,
            value=value,
            labels=labels or {}
        )
        
        self._metrics[name].append(metric)
    
    async def record_alert(
        self,
        level: str,
        message: str,
        source: str,
        **metadata
    ):
        """Record an alert."""
        alert = Alert(
            level=level,
            message=message,
            source=source,
            metadata=metadata
        )
        
        self._alerts.append(alert)
        
        # Log based on level
        if level == "critical":
            logger.critical(f"[{source}] {message}")
        elif level == "error":
            logger.error(f"[{source}] {message}")
        elif level == "warning":
            logger.warning(f"[{source}] {message}")
        else:
            logger.info(f"[{source}] {message}")
    
    async def get_metrics(
        self,
        name: Optional[str] = None,
        limit: int = 100
    ) -> List[Metric]:
        """Get recorded metrics."""
        if name:
            return list(self._metrics.get(name, []))[-limit:]
        
        all_metrics = []
        for metrics in self._metrics.values():
            all_metrics.extend(metrics)
        
        return sorted(all_metrics, key=lambda x: x.timestamp)[-limit:]
    
    async def get_alerts(
        self,
        level: Optional[str] = None,
        limit: int = 100
    ) -> List[Alert]:
        """Get recorded alerts."""
        alerts = list(self._alerts)
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return alerts[-limit:]
    
    async def start(self):
        """Start monitoring."""
        self._running = True
        logger.info("System monitor started")
    
    async def stop(self):
        """Stop monitoring."""
        self._running = False
        logger.info("System monitor stopped")
