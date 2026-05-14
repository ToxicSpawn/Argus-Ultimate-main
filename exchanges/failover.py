"""
Argus Trading System - Exchange Failover Manager
================================================

Automated failover between multiple exchanges for high availability.

Features:
- Health monitoring for each exchange
- Automatic failover on connection loss
- Latency-based routing
- Manual failover trigger
- Failback with configurable delay
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Awaitable

logger = logging.getLogger(__name__)


class ExchangeStatus(Enum):
    """Exchange connection status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ExchangeHealth:
    """Health metrics for an exchange."""
    name: str
    status: ExchangeStatus = ExchangeStatus.UNKNOWN
    latency_ms: float = 0.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    error_message: Optional[str] = None


@dataclass
class FailoverConfig:
    """Configuration for failover behavior."""
    # Health check settings
    health_check_interval: float = 5.0      # Seconds between health checks
    health_check_timeout: float = 3.0       # Timeout for health check

    # Failover thresholds
    failure_threshold: int = 3              # Failures before marking unhealthy
    recovery_threshold: int = 5             # Successes before marking healthy
    max_latency_ms: float = 1000.0          # Max acceptable latency

    # Failback settings
    failback_delay: float = 60.0            # Seconds to wait before failback
    auto_failback: bool = True              # Automatically return to primary

    # Priority (lower = higher priority)
    exchange_priority: Dict[str, int] = field(default_factory=lambda: {
        "kraken": 1,
        "coinbase": 2,
    })


class ExchangeFailoverManager:
    """
    Manages failover between multiple exchanges.

    Monitors exchange health and automatically routes
    requests to the best available exchange.
    """

    def __init__(
        self,
        config: Optional[FailoverConfig] = None,
    ) -> None:
        self.config = config or FailoverConfig()
        self._exchanges: Dict[str, Any] = {}
        self._health: Dict[str, ExchangeHealth] = {}
        self._active_exchange: Optional[str] = None
        self._primary_exchange: Optional[str] = None
        self._failover_time: Optional[datetime] = None
        self._running = False
        self._health_task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_failover: List[Callable[[str, str], Awaitable[None]]] = []
        self._on_failback: List[Callable[[str], Awaitable[None]]] = []
        self._on_status_change: List[Callable[[str, ExchangeStatus], Awaitable[None]]] = []

    def register_exchange(self, name: str, exchange: Any, primary: bool = False) -> None:
        """
        Register an exchange for failover management.

        Args:
            name: Exchange identifier
            exchange: Exchange client instance
            primary: Whether this is the primary (preferred) exchange
        """
        self._exchanges[name] = exchange
        self._health[name] = ExchangeHealth(name=name)

        if primary or self._primary_exchange is None:
            self._primary_exchange = name

        if self._active_exchange is None:
            self._active_exchange = name

        logger.info("Registered exchange: %s (primary=%s)", name, primary)

    def on_failover(self, callback: Callable[[str, str], Awaitable[None]]) -> None:
        """Register callback for failover events. Args: (from_exchange, to_exchange)"""
        self._on_failover.append(callback)

    def on_failback(self, callback: Callable[[str], Awaitable[None]]) -> None:
        """Register callback for failback events. Args: (to_exchange)"""
        self._on_failback.append(callback)

    def on_status_change(self, callback: Callable[[str, ExchangeStatus], Awaitable[None]]) -> None:
        """Register callback for status changes. Args: (exchange, new_status)"""
        self._on_status_change.append(callback)

    @property
    def active_exchange(self) -> Optional[str]:
        """Get currently active exchange name."""
        return self._active_exchange

    @property
    def active(self) -> Optional[Any]:
        """Get currently active exchange client."""
        if self._active_exchange:
            return self._exchanges.get(self._active_exchange)
        return None

    def get_exchange(self, name: str) -> Optional[Any]:
        """Get exchange by name."""
        return self._exchanges.get(name)

    def get_health(self, name: Optional[str] = None) -> Dict[str, ExchangeHealth]:
        """Get health status for one or all exchanges."""
        if name:
            return {name: self._health[name]} if name in self._health else {}
        return dict(self._health)

    async def start(self) -> None:
        """Start health monitoring."""
        if self._running:
            return

        self._running = True
        self._health_task = asyncio.create_task(self._health_check_loop())
        logger.info("Failover manager started. Active: %s", self._active_exchange)

    async def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        logger.info("Failover manager stopped.")

    async def _health_check_loop(self) -> None:
        """Main health check loop."""
        while self._running:
            try:
                await self._check_all_exchanges()
                await self._evaluate_failover()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check error: %s", e)
                await asyncio.sleep(1)

    async def _check_all_exchanges(self) -> None:
        """Check health of all registered exchanges."""
        tasks = [
            self._check_exchange_health(name, exchange)
            for name, exchange in self._exchanges.items()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_exchange_health(self, name: str, exchange: Any) -> None:
        """Check health of a single exchange."""
        health = self._health[name]
        old_status = health.status

        try:
            start = time.perf_counter()

            # Try to ping or fetch ticker
            if hasattr(exchange, 'ping'):
                await asyncio.wait_for(
                    exchange.ping(),
                    timeout=self.config.health_check_timeout,
                )
            elif hasattr(exchange, 'fetch_ticker'):
                await asyncio.wait_for(
                    exchange.fetch_ticker("BTC/AUD"),
                    timeout=self.config.health_check_timeout,
                )
            else:
                # Fallback: check if connected
                if hasattr(exchange, 'is_connected') and not exchange.is_connected:
                    raise ConnectionError("Exchange not connected")

            latency_ms = (time.perf_counter() - start) * 1000

            # Update health metrics
            health.latency_ms = latency_ms
            health.last_success = datetime.now(timezone.utc)
            health.consecutive_failures = 0
            health.consecutive_successes += 1
            health.error_message = None

            # Determine status
            if latency_ms > self.config.max_latency_ms:
                health.status = ExchangeStatus.DEGRADED
            elif health.consecutive_successes >= self.config.recovery_threshold:
                health.status = ExchangeStatus.HEALTHY
            else:
                health.status = ExchangeStatus.DEGRADED

        except asyncio.TimeoutError:
            health.consecutive_failures += 1
            health.consecutive_successes = 0
            health.last_failure = datetime.now(timezone.utc)
            health.error_message = "Health check timeout"

            if health.consecutive_failures >= self.config.failure_threshold:
                health.status = ExchangeStatus.UNHEALTHY
            else:
                health.status = ExchangeStatus.DEGRADED

        except Exception as e:
            health.consecutive_failures += 1
            health.consecutive_successes = 0
            health.last_failure = datetime.now(timezone.utc)
            health.error_message = str(e)

            if health.consecutive_failures >= self.config.failure_threshold:
                health.status = ExchangeStatus.UNHEALTHY
            else:
                health.status = ExchangeStatus.DEGRADED

        # Notify on status change
        if health.status != old_status:
            logger.info("Exchange %s status: %s -> %s", name, old_status.value, health.status.value)
            await self._notify_status_change(name, health.status)

    async def _evaluate_failover(self) -> None:
        """Evaluate if failover or failback is needed."""
        if not self._active_exchange:
            return

        active_health = self._health.get(self._active_exchange)

        # Check if active exchange is unhealthy
        if active_health and active_health.status == ExchangeStatus.UNHEALTHY:
            await self._trigger_failover()

        # Check for failback opportunity
        elif self.config.auto_failback and self._active_exchange != self._primary_exchange:
            await self._evaluate_failback()

    async def _trigger_failover(self) -> None:
        """Trigger failover to next best exchange."""
        current = self._active_exchange
        next_exchange = self._find_best_exchange(exclude=current)

        if next_exchange:
            logger.warning(
                "FAILOVER: %s -> %s (reason: %s)",
                current,
                next_exchange,
                self._health[current].error_message if current else "unknown",
            )

            old_exchange = self._active_exchange
            self._active_exchange = next_exchange
            self._failover_time = datetime.now(timezone.utc)

            # Notify callbacks
            for callback in self._on_failover:
                try:
                    await callback(old_exchange, next_exchange)
                except Exception as e:
                    logger.error("Failover callback error: %s", e)
        else:
            logger.error("FAILOVER FAILED: No healthy exchanges available!")

    async def _evaluate_failback(self) -> None:
        """Evaluate if we should failback to primary."""
        if not self._primary_exchange or not self._failover_time:
            return

        primary_health = self._health.get(self._primary_exchange)
        if not primary_health or primary_health.status != ExchangeStatus.HEALTHY:
            return

        # Check if enough time has passed
        elapsed = (datetime.now(timezone.utc) - self._failover_time).total_seconds()
        if elapsed < self.config.failback_delay:
            return

        logger.info("FAILBACK: %s -> %s", self._active_exchange, self._primary_exchange)

        self._active_exchange = self._primary_exchange
        self._failover_time = None

        # Notify callbacks
        for callback in self._on_failback:
            try:
                await callback(self._primary_exchange)
            except Exception as e:
                logger.error("Failback callback error: %s", e)

    def _find_best_exchange(self, exclude: Optional[str] = None) -> Optional[str]:
        """Find the best healthy exchange."""
        candidates = []

        for name, health in self._health.items():
            if name == exclude:
                continue
            if health.status in (ExchangeStatus.HEALTHY, ExchangeStatus.DEGRADED):
                priority = self.config.exchange_priority.get(name, 999)
                candidates.append((priority, health.latency_ms, name))

        if not candidates:
            return None

        # Sort by priority, then latency
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2]

    async def _notify_status_change(self, name: str, status: ExchangeStatus) -> None:
        """Notify status change callbacks."""
        for callback in self._on_status_change:
            try:
                await callback(name, status)
            except Exception as e:
                logger.error("Status change callback error: %s", e)

    async def force_failover(self, to_exchange: str) -> bool:
        """
        Manually trigger failover to a specific exchange.

        Args:
            to_exchange: Target exchange name

        Returns:
            True if failover successful
        """
        if to_exchange not in self._exchanges:
            logger.error("Cannot failover to unknown exchange: %s", to_exchange)
            return False

        health = self._health.get(to_exchange)
        if health and health.status == ExchangeStatus.UNHEALTHY:
            logger.warning("Forcing failover to unhealthy exchange: %s", to_exchange)

        old_exchange = self._active_exchange
        self._active_exchange = to_exchange
        self._failover_time = datetime.now(timezone.utc)

        logger.info("Manual failover: %s -> %s", old_exchange, to_exchange)

        for callback in self._on_failover:
            try:
                await callback(old_exchange, to_exchange)
            except Exception as e:
                logger.error("Failover callback error: %s", e)

        return True

    def status_summary(self) -> Dict[str, Any]:
        """Get a summary of failover status."""
        return {
            "active_exchange": self._active_exchange,
            "primary_exchange": self._primary_exchange,
            "failover_active": self._active_exchange != self._primary_exchange,
            "failover_time": self._failover_time.isoformat() if self._failover_time else None,
            "exchanges": {
                name: {
                    "status": health.status.value,
                    "latency_ms": health.latency_ms,
                    "consecutive_failures": health.consecutive_failures,
                    "error": health.error_message,
                }
                for name, health in self._health.items()
            },
        }


# Convenience function
def create_failover_manager(
    exchanges: Dict[str, Any],
    primary: Optional[str] = None,
    **config_kwargs,
) -> ExchangeFailoverManager:
    """
    Create a failover manager with exchanges pre-registered.

    Args:
        exchanges: Dict of {name: exchange_client}
        primary: Name of primary exchange
        **config_kwargs: FailoverConfig parameters

    Returns:
        Configured ExchangeFailoverManager
    """
    config = FailoverConfig(**config_kwargs)
    manager = ExchangeFailoverManager(config)

    for name, exchange in exchanges.items():
        is_primary = (name == primary) if primary else False
        manager.register_exchange(name, exchange, primary=is_primary)

    return manager
