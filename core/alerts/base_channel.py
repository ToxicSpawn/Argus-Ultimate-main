"""AbstractAlertChannel ABC + RateLimiter — Push 60."""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque

from core.alerts.alert_models import AlertEvent, AlertLevel

logger = logging.getLogger(__name__)


class RateLimiter:
    """Sliding-window rate limiter.

    Tracks send timestamps in a deque; returns True if under limit.
    """

    def __init__(self, max_per_minute: int = 30) -> None:
        self._max = max_per_minute
        self._timestamps: Deque[float] = deque()

    def allow(self) -> bool:
        now = time.monotonic()
        # Evict timestamps older than 60s
        while self._timestamps and now - self._timestamps[0] > 60.0:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._max:
            return False
        self._timestamps.append(now)
        return True

    @property
    def current_count(self) -> int:
        now = time.monotonic()
        return sum(1 for t in self._timestamps if now - t <= 60.0)


class AbstractAlertChannel(ABC):
    """Base class for all Argus alert delivery channels.

    Parameters
    ----------
    name : str
        Human-readable channel identifier.
    min_level : AlertLevel
        Minimum alert level this channel will deliver.
    rate_limit_per_min : int
        Maximum messages per 60-second window.
    enabled : bool
        Master enable/disable toggle.
    """

    def __init__(
        self,
        name: str,
        min_level: AlertLevel = AlertLevel.INFO,
        rate_limit_per_min: int = 30,
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.min_level = min_level
        self.enabled = enabled
        self._limiter = RateLimiter(rate_limit_per_min)
        self._sent = 0
        self._dropped = 0
        self._errors = 0

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abstractmethod
    async def _deliver(self, event: AlertEvent) -> bool:
        """Deliver a single alert. Return True on success."""
        ...

    # ------------------------------------------------------------------
    # Public send (with level filter + rate limit)
    # ------------------------------------------------------------------

    async def send(self, event: AlertEvent) -> bool:
        """Filter, rate-limit, then deliver an alert."""
        if not self.enabled:
            return False
        if event.level < self.min_level:
            return False
        if not self._limiter.allow():
            self._dropped += 1
            logger.debug("%s: rate-limited, dropping alert", self.name)
            return False
        try:
            success = await self._deliver(event)
            if success:
                self._sent += 1
            else:
                self._errors += 1
            return success
        except Exception as exc:  # noqa: BLE001
            self._errors += 1
            logger.error("%s: delivery error: %s", self.name, exc)
            return False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def sent(self) -> int:
        return self._sent

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def errors(self) -> int:
        return self._errors

    def status(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "min_level": self.min_level.name,
            "sent": self._sent,
            "dropped": self._dropped,
            "errors": self._errors,
        }
