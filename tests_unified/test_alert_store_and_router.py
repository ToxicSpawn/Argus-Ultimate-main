from __future__ import annotations

from pathlib import Path

from argus_live.alerts.models import AlertEvent
from argus_live.alerts.router import AlertRouter
from argus_live.alerts.severity import AlertSeverity
from argus_live.alerts.store import AlertStore


def _make_alert() -> AlertEvent:
    return AlertEvent(
        alert_type="TEST_ALERT",
        severity=AlertSeverity.WARNING,
        title="Test title",
        message="Test message",
    )


def test_store_creates_jsonl(tmp_path: Path) -> None:
    store = AlertStore(tmp_path / "alerts.jsonl")
    alert = _make_alert()
    store.append(alert)
    assert store.path.exists()
    lines = store.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert "TEST_ALERT" in lines[0]


def test_router_creates_log_file(tmp_path: Path) -> None:
    log_file = tmp_path / "routed.log"
    router = AlertRouter(log_file)
    alert = _make_alert()
    router.route(alert)
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "WARNING" in content
    assert "Test title" in content
