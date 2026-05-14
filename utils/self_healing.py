"""
Self-healing: auto restart/reinit of failed components (e.g. market data, execution).

On N consecutive failures, trigger reinit or restart; integrate with k8s when available.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class SelfHealingController:
    """
    Track failures per component; on threshold, call heal_fn(component_name).
    heal_fn can reinit exchange, reload config, or signal k8s restart.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_s: float = 60.0,
        heal_fn: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.failure_threshold = int(max(1, failure_threshold))
        self.cooldown_s = float(max(1.0, cooldown_s))
        self.heal_fn = heal_fn
        self._failures: dict = {}
        self._last_heal: dict = {}

    def record_failure(self, component: str) -> None:
        """Call when component fails; may trigger heal."""
        key = str(component)
        self._failures[key] = int(self._failures.get(key, 0)) + 1
        if self._failures[key] >= self.failure_threshold:
            self._trigger_heal(key)

    def record_success(self, component: str) -> None:
        """Reset failure count on success."""
        self._failures[str(component)] = 0

    def _trigger_heal(self, component: str) -> None:
        now = time.time()
        last = self._last_heal.get(component, 0)
        if now - last < self.cooldown_s:
            return
        self._last_heal[component] = now
        self._failures[component] = 0
        if self.heal_fn:
            try:
                result = self.heal_fn(component)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
                logger.warning("Self-healing triggered for %s", component)
            except Exception as e:
                logger.warning("Self-heal failed for %s: %s", component, e)
