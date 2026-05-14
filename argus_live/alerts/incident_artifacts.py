from __future__ import annotations

import json
import logging
from pathlib import Path

from argus_live.alerts.models import AlertEvent

logger = logging.getLogger(__name__)


def write_incident_artifact(
    alert: AlertEvent,
    output_dir: Path | str = "reports/incidents",
) -> Path:
    """Write a JSON incident artifact for *alert* and return the file path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = alert.created_at_utc.strftime("%Y%m%dT%H%M%S")
    filename = f"{ts}_{alert.alert_type}.json"
    path = out / filename

    data = alert.to_dict()
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("Incident artifact written: %s", path)
    return path
