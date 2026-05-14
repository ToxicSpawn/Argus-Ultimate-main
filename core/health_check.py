"""
Health Check System
===================

Comprehensive health checking and diagnostics for Argus Ultimate.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from unified_trading import UnifiedTradingOrchestrator
from core.unified_config import config
from core.cache_manager import get_cache

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    response_time_ms: float = 0.0


@dataclass
class SystemHealth:
    """Overall system health."""
    status: HealthStatus
    checks: List[HealthCheckResult]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    uptime_seconds: float = 0.0
    version: str = "15.0.0"


class HealthChecker:
    """
    Comprehensive health checker for all system components.
    """
    
    def __init__(self, orchestrator: Optional[UnifiedTradingOrchestrator] = None):
        self.orchestrator = orchestrator
        self.checks: Dict[str, Callable] = {}
        self._last_check: Optional[SystemHealth] = None
        self._start_time = time.time()
        
        # Register default checks
        self._register_default_checks()
        
        logger.info("HealthChecker initialized")
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check("config", self._check_config)
        self.register_check("memory", self._check_memory)
        self.register_check("cache", self._check_cache)
        self.register_check("trading", self._check_trading)
        self.register_check("risk", self._check_risk)
    
    def register_check(self, name: str, check_func: Callable):
        """Register a health check."""
        self.checks[name] = check_func
        logger.debug(f"Health check registered: {name}")
    
    async def run_all_checks(self) -> SystemHealth:
        """Run all registered health checks."""
        results = []
        
        for name, check_func in self.checks.items():
            start = time.time()
            try:
                result = await check_func()
                result.response_time_ms = (time.time() - start) * 1000
                results.append(result)
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results.append(HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {e}",
                    response_time_ms=(time.time() - start) * 1000
                ))
        
        # Determine overall status
        statuses = [r.status for r in results]
        
        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        health = SystemHealth(
            status=overall_status,
            checks=results,
            uptime_seconds=time.time() - self._start_time,
            version="15.0.0"
        )
        
        self._last_check = health
        return health
    
    async def _check_config(self) -> HealthCheckResult:
        """Check configuration health."""
        start = time.time()
        
        errors = config.validate()
        
        if errors:
            return HealthCheckResult(
                name="config",
                status=HealthStatus.UNHEALTHY,
                message=f"Configuration invalid: {', '.join(errors)}",
                response_time_ms=(time.time() - start) * 1000
            )
        
        return HealthCheckResult(
            name="config",
            status=HealthStatus.HEALTHY,
            message="Configuration valid",
            details={
                "trading_mode": config.get_str('trading.mode'),
                "symbols_count": len(config.get_list('trading.symbols'))
            },
            response_time_ms=(time.time() - start) * 1000
        )
    
    async def _check_memory(self) -> HealthCheckResult:
        """Check memory usage."""
        import psutil
        
        start = time.time()
        
        memory = psutil.virtual_memory()
        used_percent = memory.percent
        
        status = HealthStatus.HEALTHY
        message = f"Memory usage: {used_percent:.1f}%"
        
        if used_percent > 90:
            status = HealthStatus.UNHEALTHY
            message = f"Critical memory usage: {used_percent:.1f}%"
        elif used_percent > 75:
            status = HealthStatus.DEGRADED
            message = f"High memory usage: {used_percent:.1f}%"
        
        return HealthCheckResult(
            name="memory",
            status=status,
            message=message,
            details={
                "total_gb": memory.total / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent": used_percent
            },
            response_time_ms=(time.time() - start) * 1000
        )
    
    async def _check_cache(self) -> HealthCheckResult:
        """Check cache health."""
        start = time.time()
        
        cache = get_cache()
        stats = cache.get_stats()
        
        total_hits = sum(s.get('hit', 0) for s in stats.values())
        total_misses = sum(s.get('miss', 0) for s in stats.values())
        
        hit_rate = 0.0
        if total_hits + total_misses > 0:
            hit_rate = total_hits / (total_hits + total_misses)
        
        return HealthCheckResult(
            name="cache",
            status=HealthStatus.HEALTHY,
            message=f"Cache hit rate: {hit_rate:.1%}",
            details={
                "namespaces": len(stats),
                "hit_rate": hit_rate,
                "stats": stats
            },
            response_time_ms=(time.time() - start) * 1000
        )
    
    async def _check_trading(self) -> HealthCheckResult:
        """Check trading system health."""
        start = time.time()
        
        if not self.orchestrator:
            return HealthCheckResult(
                name="trading",
                status=HealthStatus.UNKNOWN,
                message="Orchestrator not available",
                response_time_ms=(time.time() - start) * 1000
            )
        
        if not self.orchestrator.state.is_running:
            return HealthCheckResult(
                name="trading",
                status=HealthStatus.UNHEALTHY,
                message="Trading system not running",
                response_time_ms=(time.time() - start) * 1000
            )
        
        active_orders = len(
            await self.orchestrator.order_manager.get_active_orders()
        )
        
        return HealthCheckResult(
            name="trading",
            status=HealthStatus.HEALTHY,
            message="Trading system operational",
            details={
                "running": True,
                "active_orders": active_orders,
                "uptime": self.orchestrator._get_uptime()
            },
            response_time_ms=(time.time() - start) * 1000
        )
    
    async def _check_risk(self) -> HealthCheckResult:
        """Check risk management health."""
        start = time.time()
        
        if not self.orchestrator:
            return HealthCheckResult(
                name="risk",
                status=HealthStatus.UNKNOWN,
                message="Orchestrator not available",
                response_time_ms=(time.time() - start) * 1000
            )
        
        # Get risk status
        risk_status = await self.orchestrator.risk_integration.check_health()
        
        if not risk_status['healthy']:
            return HealthCheckResult(
                name="risk",
                status=HealthStatus.DEGRADED,
                message=f"Risk issues: {', '.join(risk_status['issues'])}",
                details=risk_status,
                response_time_ms=(time.time() - start) * 1000
            )
        
        return HealthCheckResult(
            name="risk",
            status=HealthStatus.HEALTHY,
            message="Risk management healthy",
            details=risk_status,
            response_time_ms=(time.time() - start) * 1000
        )
    
    def get_last_check(self) -> Optional[SystemHealth]:
        """Get results of last health check."""
        return self._last_check
    
    def is_healthy(self) -> bool:
        """Quick check if system is healthy."""
        if not self._last_check:
            return False
        return self._last_check.status == HealthStatus.HEALTHY


# Diagnostic tools
class SystemDiagnostics:
    """System diagnostic tools."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information."""
        import platform
        import psutil
        
        return {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor()
            },
            "cpu": {
                "count": psutil.cpu_count(),
                "percent": psutil.cpu_percent(interval=1),
                "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            "memory": psutil.virtual_memory()._asdict(),
            "disk": psutil.disk_usage('/')._asdict(),
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
        }
    
    @staticmethod
    def get_process_info() -> Dict[str, Any]:
        """Get current process information."""
        import os
        import psutil
        
        process = psutil.Process(os.getpid())
        
        return {
            "pid": process.pid,
            "name": process.name(),
            "status": process.status(),
            "created": datetime.fromtimestamp(process.create_time()).isoformat(),
            "cpu_percent": process.cpu_percent(),
            "memory_info": process.memory_info()._asdict(),
            "num_threads": process.num_threads(),
            "num_fds": process.num_fds() if hasattr(process, 'num_fds') else None
        }
    
    @staticmethod
    async def run_diagnostics() -> Dict[str, Any]:
        """Run full system diagnostics."""
        return {
            "system": SystemDiagnostics.get_system_info(),
            "process": SystemDiagnostics.get_process_info(),
            "timestamp": datetime.utcnow().isoformat()
        }
