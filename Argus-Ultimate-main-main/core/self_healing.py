"""Self-Healing and Auto-Recovery Mechanisms.

Features:
- Health monitoring and diagnostics
- Automatic component recovery
- Circuit breakers
- State rollback and checkpointing
- Deadlock detection and resolution
- Resource leak detection
- Graceful degradation
- Alert escalation
"""

from __future__ import annotations

import logging
import time
import traceback
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Type
from enum import Enum
from collections import deque
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    RECOVERING = "recovering"


class ComponentState(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class HealthCheck:
    component: str
    status: HealthStatus
    message: str
    timestamp: float
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    component: str
    action_type: str
    attempted_at: float
    succeeded: bool
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class ComponentHealth:
    name: str
    state: ComponentState
    status: HealthStatus
    last_check: float
    consecutive_failures: int
    total_failures: int
    restart_count: int
    metrics: Dict[str, Any] = field(default_factory=dict)


class HealthMonitor:
    def __init__(self):
        self._components: Dict[str, ComponentHealth] = {}
        self._check_interval = 30
        self._max_consecutive_failures = 3
        self._health_history: deque = deque(maxlen=1000)

    def register_component(
        self,
        name: str,
        health_check_fn: Optional[Callable] = None,
    ) -> None:
        self._components[name] = ComponentHealth(
            name=name,
            state=ComponentState.INITIALIZING,
            status=HealthStatus.HEALTHY,
            last_check=time.time(),
            consecutive_failures=0,
            total_failures=0,
            restart_count=0,
        )
        
        if health_check_fn:
            self._components[name].health_check_fn = health_check_fn

    def update_state(
        self,
        name: str,
        state: ComponentState,
        metrics: Optional[Dict] = None,
    ) -> None:
        if name not in self._components:
            self.register_component(name)
        
        comp = self._components[name]
        comp.state = state
        comp.last_check = time.time()
        
        if metrics:
            comp.metrics.update(metrics)

    def record_failure(self, name: str) -> None:
        if name not in self._components:
            self.register_component(name)
        
        comp = self._components[name]
        comp.consecutive_failures += 1
        comp.total_failures += 1
        
        if comp.consecutive_failures >= self._max_consecutive_failures:
            comp.status = HealthStatus.CRITICAL
        elif comp.consecutive_failures > 0:
            comp.status = HealthStatus.UNHEALTHY

    def record_success(self, name: str) -> None:
        if name not in self._components:
            return
        
        comp = self._components[name]
        comp.consecutive_failures = 0
        
        if comp.status != HealthStatus.HEALTHY:
            comp.status = HealthStatus.HEALTHY

    def get_overall_status(self) -> HealthStatus:
        if not self._components:
            return HealthStatus.HEALTHY
        
        statuses = [c.status for c in self._components.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY

    def get_component_status(self, name: str) -> Optional[ComponentHealth]:
        return self._components.get(name)

    def get_all_statuses(self) -> Dict[str, ComponentHealth]:
        return self._components.copy()


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_attempts: int = 1,
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_attempts = half_open_attempts
        
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = "closed"
        self._half_open_attempts_left = self._half_open_attempts

    @property
    def is_open(self) -> bool:
        if self._state == "open":
            if time.time() - self._last_failure_time >= self._recovery_timeout:
                self._state = "half_open"
                self._half_open_attempts_left = self._half_open_attempts
                return False
            return True
        return False

    def record_success(self) -> None:
        if self._state == "half_open":
            self._half_open_attempts_left -= 1
            if self._half_open_attempts_left <= 0:
                self._state = "closed"
                self._failure_count = 0
        elif self._state == "closed":
            self._failure_count = 0

    def record_failure(self) -> bool:
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._state == "half_open":
            self._state = "open"
            return True
        elif self._failure_count >= self._failure_threshold:
            self._state = "open"
            return True
        
        return False

    def get_state(self) -> str:
        return self._state


class CheckpointManager:
    def __init__(self, max_checkpoints: int = 10):
        self._max_checkpoints = max_checkpoints
        self._checkpoints: deque = deque(maxlen=max_checkpoints)
        self._current_state: Dict[str, Any] = {}

    def save_checkpoint(
        self,
        name: str,
        state: Dict[str, Any],
    ) -> str:
        checkpoint_id = f"{name}_{int(time.time() * 1000)}"
        
        checkpoint = {
            "id": checkpoint_id,
            "name": name,
            "state": state.copy(),
            "timestamp": time.time(),
        }
        
        self._checkpoints.append(checkpoint)
        self._current_state = state.copy()
        
        logger.info(f"Saved checkpoint: {checkpoint_id}")
        return checkpoint_id

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        for cp in self._checkpoints:
            if cp["id"] == checkpoint_id:
                self._current_state = cp["state"].copy()
                return self._current_state.copy()
        
        return None

    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        if self._checkpoints:
            latest = self._checkpoints[-1]
            return latest["state"].copy()
        return None

    def rollback(self, steps: int = 1) -> Optional[Dict[str, Any]]:
        if len(self._checkpoints) >= steps:
            for _ in range(steps):
                self._checkpoints.pop()
            
            if self._checkpoints:
                latest = self._checkpoints[-1]
                self._current_state = latest["state"].copy()
                return self._current_state.copy()
        
        return None


class DeadlockDetector:
    def __init__(self, timeout_seconds: float = 30.0):
        self._timeout = timeout_seconds
        self._lock_timestamps: Dict[str, float] = {}
        self._lock_holders: Dict[str, str] = {}

    @contextmanager
    def track_lock(self, lock_id: str, holder_id: str):
        acquired = False
        try:
            if lock_id in self._lock_holders:
                wait_time = time.time() - self._lock_timestamps.get(lock_id, 0)
                if wait_time > self._timeout:
                    logger.warning(
                        f"Potential deadlock detected for lock {lock_id}, "
                        f"held by {self._lock_holders[lock_id]} for {wait_time:.1f}s"
                    )
            
            self._lock_timestamps[lock_id] = time.time()
            self._lock_holders[lock_id] = holder_id
            acquired = True
            
            yield
            
        finally:
            if acquired:
                self._lock_holders.pop(lock_id, None)
                self._lock_timestamps.pop(lock_id, None)


class ResourceMonitor:
    def __init__(self):
        self._resource_usage: Dict[str, List[float]] = {}
        self._limits: Dict[str, float] = {}
        self._alerts: deque = deque(maxlen=100)

    def set_limit(self, resource: str, limit: float) -> None:
        self._limits[resource] = limit

    def record_usage(self, resource: str, value: float) -> None:
        if resource not in self._resource_usage:
            self._resource_usage[resource] = []
        
        self._resource_usage[resource].append(value)
        
        if len(self._resource_usage[resource]) > 1000:
            self._resource_usage[resource] = self._resource_usage[resource][-1000:]
        
        if resource in self._limits and value > self._limits[resource]:
            self._alerts.append({
                "resource": resource,
                "value": value,
                "limit": self._limits[resource],
                "timestamp": time.time(),
            })

    def get_usage(self, resource: str) -> Dict[str, float]:
        if resource not in self._resource_usage:
            return {"current": 0.0, "avg": 0.0, "max": 0.0}
        
        values = self._resource_usage[resource]
        return {
            "current": values[-1] if values else 0.0,
            "avg": sum(values) / len(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }

    def get_alerts(self) -> List[Dict[str, Any]]:
        return list(self._alerts)


class AutoRecoveryManager:
    def __init__(self, health_monitor: HealthMonitor):
        self._health_monitor = health_monitor
        self._recovery_actions: List[RecoveryAction] = []
        self._recovery_handlers: Dict[str, Callable] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._checkpoint_manager = CheckpointManager()

    def register_recovery_handler(
        self,
        component: str,
        handler: Callable,
    ) -> None:
        self._recovery_handlers[component] = handler
        self._circuit_breakers[component] = CircuitBreaker()

    async def attempt_recovery(
        self,
        component: str,
        max_attempts: int = 3,
    ) -> bool:
        if component not in self._recovery_handlers:
            logger.warning(f"No recovery handler for component: {component}")
            return False
        
        breaker = self._circuit_breakers.get(component)
        if breaker and breaker.is_open:
            logger.warning(f"Circuit breaker open for: {component}")
            return False
        
        handler = self._recovery_handlers[component]
        
        for attempt in range(max_attempts):
            t0 = time.time()
            
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
                
                duration_ms = (time.time() - t0) * 1000
                
                self._recovery_actions.append(RecoveryAction(
                    component=component,
                    action_type="restart",
                    attempted_at=t0,
                    succeeded=True,
                    duration_ms=duration_ms,
                ))
                
                if breaker:
                    breaker.record_success()
                
                self._health_monitor.record_success(component)
                logger.info(f"Successfully recovered component: {component}")
                return True
                
            except Exception as e:
                duration_ms = (time.time() - t0) * 1000
                
                self._recovery_actions.append(RecoveryAction(
                    component=component,
                    action_type="restart",
                    attempted_at=t0,
                    succeeded=False,
                    error=str(e),
                    duration_ms=duration_ms,
                ))
                
                logger.error(
                    f"Recovery attempt {attempt + 1} failed for {component}: {e}"
                )
        
        if breaker:
            breaker.record_failure()
        
        self._health_monitor.record_failure(component)
        
        return False

    def save_state(
        self,
        name: str,
        state: Dict[str, Any],
    ) -> str:
        return self._checkpoint_manager.save_checkpoint(name, state)

    def restore_state(
        self,
        checkpoint_id: str,
    ) -> Optional[Dict[str, Any]]:
        return self._checkpoint_manager.load_checkpoint(checkpoint_id)

    def get_recovery_history(self) -> List[RecoveryAction]:
        return self._recovery_actions


class SelfHealingSystem:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        self._health_monitor = HealthMonitor()
        self._recovery_manager = AutoRecoveryManager(self._health_monitor)
        self._resource_monitor = ResourceMonitor()
        self._deadlock_detector = DeadlockDetector(
            timeout_seconds=self.config.get("deadlock_timeout", 30.0),
        )
        
        self._running = False
        self._monitor_task: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Self-healing system started")

    def stop(self) -> None:
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Self-healing system stopped")

    def _monitor_loop(self) -> None:
        while self._running:
            try:
                self._check_health()
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            time.sleep(30)

    def _check_health(self) -> None:
        status = self._health_monitor.get_overall_status()
        
        if status == HealthStatus.CRITICAL:
            logger.critical("System health is CRITICAL")
            self._trigger_emergency_recovery()

    def _trigger_emergency_recovery(self) -> None:
        components = self._health_monitor.get_all_statuses()
        
        for name, health in components.items():
            if health.status == HealthStatus.CRITICAL:
                logger.warning(f"Attempting emergency recovery for: {name}")
                asyncio.create_task(self._recovery_manager.attempt_recovery(name))

    def register_component(
        self,
        name: str,
        health_check_fn: Optional[Callable] = None,
        recovery_fn: Optional[Callable] = None,
    ) -> None:
        self._health_monitor.register_component(name, health_check_fn)
        
        if recovery_fn:
            self._recovery_manager.register_recovery_handler(name, recovery_fn)

    def record_component_state(
        self,
        name: str,
        state: ComponentState,
        metrics: Optional[Dict] = None,
    ) -> None:
        self._health_monitor.update_state(name, state, metrics)

    def record_resource_usage(
        self,
        resource: str,
        value: float,
    ) -> None:
        self._resource_monitor.record_usage(resource, value)

    def get_health_report(self) -> Dict[str, Any]:
        return {
            "overall_status": self._health_monitor.get_overall_status().value,
            "components": {
                name: {
                    "state": h.state.value,
                    "status": h.status.value,
                    "consecutive_failures": h.consecutive_failures,
                    "total_failures": h.total_failures,
                    "restart_count": h.restart_count,
                }
                for name, h in self._health_monitor.get_all_statuses().items()
            },
            "resource_alerts": self._resource_monitor.get_alerts()[-10:],
        }

    @contextmanager
    def safe_execution(self, component: str):
        try:
            yield
            self._health_monitor.record_success(component)
        except Exception as e:
            self._health_monitor.record_failure(component)
            raise

    def get_lock_context(self, lock_id: str, holder_id: str):
        return self._deadlock_detector.track_lock(lock_id, holder_id)


import asyncio
