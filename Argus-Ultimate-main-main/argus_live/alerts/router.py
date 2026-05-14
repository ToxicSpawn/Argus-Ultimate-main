from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from argus_live.alerts.models import AlertEvent
from argus_live.alerts.severity import AlertSeverity

logger = logging.getLogger(__name__)


class AlertRouter:
    """Routes alerts to a log file and invokes severity-specific hooks."""

    def __init__(
        self,
        log_path: Path | str,
        *,
        critical_hook: Callable[[AlertEvent], None] | None = None,
        major_hook: Callable[[AlertEvent], None] | None = None,
        warning_hook: Callable[[AlertEvent], None] | None = None,
    ) -> None:
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._critical_hook = critical_hook
        self._major_hook = major_hook
        self._warning_hook = warning_hook

    @property
    def log_path(self) -> Path:
        return self._log_path

    def route(self, alert: AlertEvent) -> None:
        """Write *alert* to the log file and fire the appropriate severity hook."""
        line = f"[{alert.severity.value}] {alert.title}: {alert.message}"
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        logger.info("Alert routed: %s (%s)", alert.alert_type, alert.severity.value)

        if alert.severity is AlertSeverity.CRITICAL and self._critical_hook:
            self._critical_hook(alert)
        elif alert.severity is AlertSeverity.MAJOR and self._major_hook:
            self._major_hook(alert)
        elif alert.severity is AlertSeverity.WARNING and self._warning_hook:
            self._warning_hook(alert)
