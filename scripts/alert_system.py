"""Alert routing for risk, execution, and system events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    severity: str
    title: str
    message: str
    timestamp: str


class AlertSystem:
    def __init__(self, min_severity: str = "info"):
        self.levels = {"info": 0, "warning": 1, "error": 2, "critical": 3}
        self.min_severity = min_severity
        self.history: list[Alert] = []

    def send(self, severity: str, title: str, message: str) -> Alert | None:
        if self.levels[severity] < self.levels[self.min_severity]:
            return None
        alert = Alert(severity, title, message, datetime.now(timezone.utc).isoformat())
        self.history.append(alert)
        getattr(logger, "error" if severity in {"error", "critical"} else "warning" if severity == "warning" else "info")(
            "%s: %s - %s", severity.upper(), title, message
        )
        return alert

    def check_risk(self, drawdown: float, daily_loss: float) -> list[Alert]:
        alerts: list[Alert] = []
        if drawdown > 0.15:
            alert = self.send("critical", "Max drawdown breached", f"Drawdown is {drawdown:.1%}")
            if alert is not None:
                alerts.append(alert)
        elif drawdown > 0.10:
            alert = self.send("warning", "Drawdown elevated", f"Drawdown is {drawdown:.1%}")
            if alert is not None:
                alerts.append(alert)
        if daily_loss > 0.05:
            alert = self.send("critical", "Daily loss breached", f"Daily loss is {daily_loss:.1%}")
            if alert is not None:
                alerts.append(alert)
        return alerts


def _demo() -> None:
    logging.basicConfig(level=logging.INFO)
    alerts = AlertSystem()
    alerts.check_risk(0.12, 0.02)
    print("Alert system ready")
    print(alerts.history[-1])


if __name__ == "__main__":
    _demo()
