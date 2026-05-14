"""
Circuit breaker for data/execution layers.

After failure_threshold consecutive failures, opens for cooldown_s.
Then half-open: one trial; success -> closed, failure -> open again.

Usage: if not cb.allow(): return fallback; try: r = await fn(); cb.record_success(); return r
       except Exception: cb.record_failure(); return fallback
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        cooldown_s: float = 60.0,
        name: str = "default",
    ) -> None:
        self.failure_threshold = int(max(1, failure_threshold))
        self.cooldown_s = float(max(1.0, cooldown_s))
        self.name = str(name or "default")
        self._failures = 0
        self._last_failure_time: float = 0.0
        self._state: str = "closed"  # closed | open | half_open

    def _now(self) -> float:
        return time.time()

    @property
    def state(self) -> str:
        if self._state == "open":
            if (self._now() - self._last_failure_time) >= self.cooldown_s:
                self._state = "half_open"
        return self._state

    def record_success(self) -> None:
        self._failures = 0
        self._state = "closed"

    def record_failure(self) -> None:
        self._failures += 1
        self._last_failure_time = self._now()
        if self._state == "half_open":
            self._state = "open"
        elif self._failures >= self.failure_threshold:
            self._state = "open"
            logger.warning("CircuitBreaker %s opened after %d failures", self.name, self._failures)

    def allow(self) -> bool:
        """Return True if call is allowed (closed or half_open)."""
        s = self.state
        return s in ("closed", "half_open")
