from __future__ import annotations

import json
import logging
from pathlib import Path

from argus_live.alerts.models import AlertEvent

logger = logging.getLogger(__name__)


class AlertStore:
    """Append-only JSONL persistence for alert events."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, alert: AlertEvent) -> None:
        """Serialise *alert* as a single JSON line and append to the store file."""
        line = json.dumps(alert.to_dict(), default=str)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        logger.debug("Alert stored: %s", alert.alert_type)
