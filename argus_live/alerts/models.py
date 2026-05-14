from __future__ import annotations

import dataclasses
import datetime
from typing import Any

from argus_live.alerts.severity import AlertSeverity


@dataclasses.dataclass(frozen=True)
class AlertEvent:
    """Immutable record of a single alert firing."""

    alert_type: str
    severity: AlertSeverity
    title: str
    message: str
    payload: dict[str, Any] = dataclasses.field(default_factory=dict)
    created_at_utc: datetime.datetime = dataclasses.field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON encoding."""
        return {
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "payload": self.payload,
            "created_at_utc": self.created_at_utc.isoformat(),
        }
