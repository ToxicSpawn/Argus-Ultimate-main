from __future__ import annotations

import enum


class AlertSeverity(enum.Enum):
    """Severity levels for alerts, ordered from least to most severe."""

    INFO = "INFO"
    WARNING = "WARNING"
    MAJOR = "MAJOR"
    CRITICAL = "CRITICAL"
