#!/usr/bin/env python3
"""
Service Circuit Breaker Pattern — per-service fault isolation.

Prevents cascading failures by tracking error rates and temporarily
short-circuiting calls to unhealthy downstream services (exchanges, APIs,
databases).

This is distinct from ``core/circuit_breaker.py`` which is a *trading*
circuit breaker (drawdown / loss-streak based).  This module implements the
classic software-engineering circuit breaker pattern for service calls.

State machine::

    CLOSED ──(failure_threshold reached)──► OPEN
       ▲                                      │
       │                              (recovery_timeout_s)
       │                                      ▼
       └───(success_threshold reached)──── HALF_OPEN

Usage::

    breaker = ServiceCircuitBreaker(failure_threshold=5, recovery_timeout_s=60)

    try:
        result = breaker.call("kraken-api", exchange.fetch_ticker, "BTC/USD")
    except CircuitOpenError:
        # Service is unhealthy — use fallback
        pass

Thread-safe: each service gets its own lock.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# States and exceptions
# ---------------------------------------------------------------------------


class ServiceState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is OPEN.

    Attributes
    ----------
    service_name : str
        The service whose circuit is open.
    retry_after : float
        Seconds remaining before the circuit transitions to HALF_OPEN.
    """

    def __init__(self, service_name: str, retry_after: float) -> None:
        self.service_name = service_name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit OPEN for '{service_name}' — retry after {retry_after:.1f}s"
        )


# ---------------------------------------------------------------------------
# Per-service state
# ---------------------------------------------------------------------------


@dataclass
class _ServiceBreaker:
    """Internal mutable state for a single service."""

    state: ServiceState = ServiceState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    total_calls: int = 0
    total_failures: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


# ---------------------------------------------------------------------------
# Service Circuit Breaker
# ---------------------------------------------------------------------------


class ServiceCircuitBreaker:
    """Thread-safe, per-service circuit breaker for downstream service calls.

    Parameters
    ----------
    failure_threshold:
        Number of consecutive failures before opening the circuit.
    recovery_timeout_s:
        Seconds the circuit stays OPEN before transitioning to HALF_OPEN.
    success_threshold:
        Number of consecutive successes in HALF_OPEN required to close.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 60.0,
        success_threshold: int = 3,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.success_threshold = success_threshold

        self._breakers: Dict[str, _ServiceBreaker] = {}
        self._global_lock = threading.Lock()

        logger.info(
            "ServiceCircuitBreaker initialised — failure_threshold=%d recovery=%ds "
            "success_threshold=%d",
            failure_threshold,
            int(recovery_timeout_s),
            success_threshold,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_breaker(self, service_name: str) -> _ServiceBreaker:
        """Return (or create) the breaker state for *service_name*."""
        with self._global_lock:
            if service_name not in self._breakers:
                self._breakers[service_name] = _ServiceBreaker()
                logger.debug(
                    "ServiceCircuitBreaker created breaker for '%s'", service_name
                )
            return self._breakers[service_name]

    def _record_success(
        self, breaker: _ServiceBreaker, service_name: str
    ) -> None:
        """Record a successful call and transition state if appropriate."""
        breaker.total_calls += 1

        if breaker.state == ServiceState.HALF_OPEN:
            breaker.success_count += 1
            if breaker.success_count >= self.success_threshold:
                breaker.state = ServiceState.CLOSED
                breaker.failure_count = 0
                breaker.success_count = 0
                logger.info(
                    "ServiceCircuitBreaker '%s' HALF_OPEN → CLOSED (recovered)",
                    service_name,
                )
        elif breaker.state == ServiceState.CLOSED:
            # Reset failure counter on success
            breaker.failure_count = 0

    def _record_failure(
        self, breaker: _ServiceBreaker, service_name: str
    ) -> None:
        """Record a failed call and transition state if appropriate."""
        breaker.failure_count += 1
        breaker.total_calls += 1
        breaker.total_failures += 1
        breaker.last_failure_time = time.time()

        if breaker.state == ServiceState.HALF_OPEN:
            # Any failure in half-open goes straight back to open
            breaker.state = ServiceState.OPEN
            breaker.success_count = 0
            logger.warning(
                "ServiceCircuitBreaker '%s' HALF_OPEN → OPEN (failure during probe)",
                service_name,
            )
        elif breaker.state == ServiceState.CLOSED:
            if breaker.failure_count >= self.failure_threshold:
                breaker.state = ServiceState.OPEN
                logger.warning(
                    "ServiceCircuitBreaker '%s' CLOSED → OPEN "
                    "(failures=%d >= threshold=%d)",
                    service_name,
                    breaker.failure_count,
                    self.failure_threshold,
                )

    def _maybe_transition_to_half_open(
        self, breaker: _ServiceBreaker, service_name: str
    ) -> None:
        """Transition from OPEN to HALF_OPEN if recovery timeout has elapsed."""
        if breaker.state == ServiceState.OPEN:
            elapsed = time.time() - breaker.last_failure_time
            if elapsed >= self.recovery_timeout_s:
                breaker.state = ServiceState.HALF_OPEN
                breaker.success_count = 0
                logger.info(
                    "ServiceCircuitBreaker '%s' OPEN → HALF_OPEN (%.1fs elapsed)",
                    service_name,
                    elapsed,
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(
        self,
        service_name: str,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute *fn* through the circuit breaker for *service_name*.

        Parameters
        ----------
        service_name:
            Logical name of the downstream service (e.g. ``"kraken-api"``).
        fn:
            The callable to invoke.
        *args, **kwargs:
            Forwarded to *fn*.

        Returns
        -------
        Any
            The return value of *fn* on success.

        Raises
        ------
        CircuitOpenError
            If the circuit is OPEN and the recovery timeout has not elapsed.
        Exception
            Any exception raised by *fn* is re-raised after recording the
            failure.
        """
        breaker = self._get_breaker(service_name)

        with breaker.lock:
            self._maybe_transition_to_half_open(breaker, service_name)

            if breaker.state == ServiceState.OPEN:
                retry_after = self.recovery_timeout_s - (
                    time.time() - breaker.last_failure_time
                )
                raise CircuitOpenError(service_name, max(retry_after, 0.0))

        # Execute outside the lock so other services aren't blocked
        try:
            result = fn(*args, **kwargs)
        except Exception:
            with breaker.lock:
                self._record_failure(breaker, service_name)
            raise

        with breaker.lock:
            self._record_success(breaker, service_name)

        return result

    def get_state(self, service_name: str) -> str:
        """Return the current state for *service_name* as a string.

        Returns ``"closed"``, ``"open"``, or ``"half_open"``.
        If no breaker exists for the service, returns ``"closed"``.
        """
        with self._global_lock:
            breaker = self._breakers.get(service_name)
        if breaker is None:
            return ServiceState.CLOSED.value

        with breaker.lock:
            self._maybe_transition_to_half_open(breaker, service_name)
            return breaker.state.value

    def get_all_states(self) -> Dict[str, str]:
        """Return a snapshot of ``{service_name: state_string}`` for all services."""
        with self._global_lock:
            names = list(self._breakers.keys())

        result: Dict[str, str] = {}
        for name in names:
            result[name] = self.get_state(name)
        return result

    def get_stats(self, service_name: str) -> Dict[str, Any]:
        """Return detailed statistics for *service_name*.

        Returns
        -------
        dict
            Keys: ``state``, ``failure_count``, ``total_calls``,
            ``total_failures``, ``error_rate``.
        """
        breaker = self._get_breaker(service_name)
        with breaker.lock:
            self._maybe_transition_to_half_open(breaker, service_name)
            total = breaker.total_calls
            failures = breaker.total_failures
            return {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "total_calls": total,
                "total_failures": failures,
                "error_rate": failures / total if total > 0 else 0.0,
            }

    def reset(self, service_name: str) -> None:
        """Force-reset *service_name* to CLOSED, clearing all counters.

        Useful for manual recovery after an operator confirms the service
        is healthy.
        """
        breaker = self._get_breaker(service_name)
        with breaker.lock:
            old_state = breaker.state
            breaker.state = ServiceState.CLOSED
            breaker.failure_count = 0
            breaker.success_count = 0
            breaker.last_failure_time = 0.0

        logger.info(
            "ServiceCircuitBreaker '%s' force-reset %s → CLOSED",
            service_name,
            old_state.value,
        )
