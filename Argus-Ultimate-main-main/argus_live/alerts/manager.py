from __future__ import annotations

import logging
from pathlib import Path

from argus_live.alerts.incident_artifacts import write_incident_artifact
from argus_live.alerts.models import AlertEvent
from argus_live.alerts.router import AlertRouter
from argus_live.alerts.severity import AlertSeverity
from argus_live.alerts.store import AlertStore

logger = logging.getLogger(__name__)


class AlertingManager:
    """Top-level facade: store, route, and generate incident artifacts."""

    def __init__(
        self,
        store: AlertStore,
        router: AlertRouter,
        incident_dir: Path | str = "reports/incidents",
    ) -> None:
        self._store = store
        self._router = router
        self._incident_dir = Path(incident_dir)

    def handle(self, alert: AlertEvent | None) -> None:
        """Process *alert* end-to-end.  ``None`` is silently ignored."""
        if alert is None:
            return

        self._store.append(alert)
        self._router.route(alert)

        if alert.severity in (AlertSeverity.CRITICAL, AlertSeverity.MAJOR):
            write_incident_artifact(alert, self._incident_dir)
            logger.warning(
                "Incident artifact generated for %s (%s)",
                alert.alert_type,
                alert.severity.value,
            )
