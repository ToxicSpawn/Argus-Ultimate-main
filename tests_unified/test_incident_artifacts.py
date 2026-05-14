from __future__ import annotations

import json
from pathlib import Path

from argus_live.alerts.incident_artifacts import write_incident_artifact
from argus_live.alerts.models import AlertEvent
from argus_live.alerts.severity import AlertSeverity


def test_artifact_written_to_tmp_path(tmp_path: Path) -> None:
    alert = AlertEvent(
        alert_type="DRAWDOWN_BREACH",
        severity=AlertSeverity.CRITICAL,
        title="Drawdown breach",
        message="Drawdown 15% > max 10%",
        payload={"current_drawdown_pct": 15.0},
    )
    path = write_incident_artifact(alert, output_dir=tmp_path)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["alert_type"] == "DRAWDOWN_BREACH"
    assert data["severity"] == "CRITICAL"
