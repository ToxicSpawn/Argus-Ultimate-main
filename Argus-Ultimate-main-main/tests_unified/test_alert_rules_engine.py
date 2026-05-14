from __future__ import annotations

from argus_live.alerts.rules_engine import AlertRulesEngine
from argus_live.alerts.severity import AlertSeverity


def test_drawdown_breach_emits_alert() -> None:
    engine = AlertRulesEngine()
    alert = engine.evaluate_drawdown(current_drawdown_pct=15.0, max_drawdown_pct=10.0)
    assert alert is not None
    assert alert.alert_type == "DRAWDOWN_BREACH"
    assert alert.severity in (AlertSeverity.MAJOR, AlertSeverity.CRITICAL)
    assert "15.00%" in alert.message
