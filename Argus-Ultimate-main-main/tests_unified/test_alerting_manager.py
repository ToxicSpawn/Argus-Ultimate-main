from __future__ import annotations

from pathlib import Path

from argus_live.alerts.manager import AlertingManager
from argus_live.alerts.models import AlertEvent
from argus_live.alerts.router import AlertRouter
from argus_live.alerts.severity import AlertSeverity
from argus_live.alerts.store import AlertStore


def test_handle_none_does_not_crash(tmp_path: Path) -> None:
    store = AlertStore(tmp_path / "alerts.jsonl")
    router = AlertRouter(tmp_path / "routed.log")
    manager = AlertingManager(store, router, incident_dir=tmp_path / "incidents")
    manager.handle(None)
    # No files should have been created
    assert not (tmp_path / "alerts.jsonl").exists()


def test_handle_critical_creates_incident(tmp_path: Path) -> None:
    store = AlertStore(tmp_path / "alerts.jsonl")
    router = AlertRouter(tmp_path / "routed.log")
    incident_dir = tmp_path / "incidents"
    manager = AlertingManager(store, router, incident_dir=incident_dir)

    alert = AlertEvent(
        alert_type="DRAWDOWN_BREACH",
        severity=AlertSeverity.CRITICAL,
        title="Drawdown breach",
        message="Critical drawdown",
    )
    manager.handle(alert)

    assert (tmp_path / "alerts.jsonl").exists()
    assert (tmp_path / "routed.log").exists()
    assert incident_dir.exists()
    artifacts = list(incident_dir.glob("*.json"))
    assert len(artifacts) == 1
