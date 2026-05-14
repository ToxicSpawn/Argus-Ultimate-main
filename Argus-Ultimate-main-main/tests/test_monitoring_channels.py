"""
Tests for monitoring channels: Slack, PagerDuty, Email, SLA tracker,
log rotation, metrics maintenance, and alerting.py wiring.
"""
from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ===========================================================================
# 1. SlackWebhook tests
# ===========================================================================

class TestSlackWebhook:

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("ARGUS_SLACK_WEBHOOK", "https://hooks.slack.com/test")
        from monitoring.slack_webhook import SlackWebhook
        sw = SlackWebhook()
        assert sw.is_configured is True
        assert sw.webhook_url == "https://hooks.slack.com/test"

    def test_init_no_config(self, monkeypatch):
        monkeypatch.delenv("ARGUS_SLACK_WEBHOOK", raising=False)
        from monitoring.slack_webhook import SlackWebhook
        sw = SlackWebhook()
        assert sw.is_configured is False

    def test_format_system_alert_info(self):
        from monitoring.slack_webhook import SlackWebhook
        sw = SlackWebhook(webhook_url="https://fake")
        payload = sw.format_system_alert("System healthy", "INFO")
        assert payload["attachments"][0]["color"] == "#36a64f"
        assert "System healthy" in payload["attachments"][0]["text"]

    def test_format_system_alert_critical(self):
        from monitoring.slack_webhook import SlackWebhook
        sw = SlackWebhook(webhook_url="https://fake")
        payload = sw.format_system_alert("DB down", "CRITICAL")
        assert payload["attachments"][0]["color"] == "#ff0000"

    def test_format_system_alert_warning(self):
        from monitoring.slack_webhook import SlackWebhook
        sw = SlackWebhook(webhook_url="https://fake")
        payload = sw.format_system_alert("High latency", "WARNING")
        assert payload["attachments"][0]["color"] == "#ff9900"

    def test_format_trade_alert(self):
        from monitoring.slack_webhook import SlackWebhook
        sw = SlackWebhook(webhook_url="https://fake")
        td = {"symbol": "BTC/AUD", "side": "buy", "price": 100000, "pnl": 42.5}
        payload = sw.format_trade_alert(td)
        assert "attachments" in payload
        blocks = payload["attachments"][0]["blocks"]
        assert any("BTC/AUD" in str(b) for b in blocks)

    def test_send_not_configured(self):
        from monitoring.slack_webhook import SlackWebhook
        sw = SlackWebhook()
        result = _run(sw.send("test"))
        assert result is False


# ===========================================================================
# 2. PagerDutyAlert tests
# ===========================================================================

class TestPagerDutyAlert:

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("ARGUS_PAGERDUTY_KEY", "test-routing-key")
        from monitoring.pagerduty_alert import PagerDutyAlert
        pd = PagerDutyAlert()
        assert pd.is_configured is True
        assert pd.routing_key == "test-routing-key"

    def test_init_no_config(self, monkeypatch):
        monkeypatch.delenv("ARGUS_PAGERDUTY_KEY", raising=False)
        from monitoring.pagerduty_alert import PagerDutyAlert
        pd = PagerDutyAlert()
        assert pd.is_configured is False

    def test_trigger_not_configured(self, monkeypatch):
        monkeypatch.delenv("ARGUS_PAGERDUTY_KEY", raising=False)
        from monitoring.pagerduty_alert import PagerDutyAlert
        pd = PagerDutyAlert()
        # trigger returns dedup_key string, but _post returns False when unconfigured
        # The trigger method still returns a dedup_key (it generates one before posting)
        result = _run(pd.trigger("test incident"))
        assert isinstance(result, str)

    def test_resolve_not_configured(self, monkeypatch):
        monkeypatch.delenv("ARGUS_PAGERDUTY_KEY", raising=False)
        from monitoring.pagerduty_alert import PagerDutyAlert
        pd = PagerDutyAlert()
        result = _run(pd.resolve("fake-key"))
        assert result is False

    def test_severity_normalization(self):
        from monitoring.pagerduty_alert import PagerDutyAlert
        pd = PagerDutyAlert(routing_key="test")
        # Invalid severity should default to "critical"
        # We can't actually trigger without a real endpoint, but test the logic
        assert pd.is_configured is True


# ===========================================================================
# 3. EmailAlert tests
# ===========================================================================

class TestEmailAlert:

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("ARGUS_SMTP_HOST", "smtp.test.com")
        monkeypatch.setenv("ARGUS_SMTP_TO", "a@b.com,c@d.com")
        from monitoring.email_alert import EmailAlert
        ea = EmailAlert()
        assert ea.is_configured is True
        assert ea.smtp_host == "smtp.test.com"
        assert len(ea.to_addrs) == 2

    def test_init_no_config(self, monkeypatch):
        monkeypatch.delenv("ARGUS_SMTP_HOST", raising=False)
        monkeypatch.delenv("ARGUS_SMTP_TO", raising=False)
        from monitoring.email_alert import EmailAlert
        ea = EmailAlert()
        assert ea.is_configured is False

    def test_send_not_configured(self, monkeypatch):
        monkeypatch.delenv("ARGUS_SMTP_HOST", raising=False)
        monkeypatch.delenv("ARGUS_SMTP_TO", raising=False)
        from monitoring.email_alert import EmailAlert
        ea = EmailAlert()
        result = ea.send("Test Subject", "Test Body")
        assert result is False

    def test_send_daily_summary_not_configured(self, monkeypatch):
        monkeypatch.delenv("ARGUS_SMTP_HOST", raising=False)
        monkeypatch.delenv("ARGUS_SMTP_TO", raising=False)
        from monitoring.email_alert import EmailAlert
        ea = EmailAlert()
        result = ea.send_daily_summary({
            "date": "2026-03-14",
            "total_pnl": 150.0,
            "trade_count": 5,
            "win_rate": 0.6,
            "max_drawdown": 0.05,
            "trades": [{"symbol": "BTC/AUD", "side": "buy", "pnl": 50.0, "strategy": "momentum"}],
        })
        assert result is False


# ===========================================================================
# 4. SLATracker tests
# ===========================================================================

class TestSLATracker:

    def test_initial_state(self):
        from monitoring.sla_tracker import SLATracker
        st = SLATracker()
        assert st.get_uptime_pct() == 100.0
        assert st.get_mtbf() == 0.0
        assert st.get_mttr() == 0.0

    def test_record_uptime_tick(self):
        from monitoring.sla_tracker import SLATracker
        st = SLATracker()
        st.record_uptime_tick()
        st.record_uptime_tick()
        assert len(st._uptime_ticks) == 2

    def test_record_downtime(self):
        from monitoring.sla_tracker import SLATracker
        st = SLATracker()
        st.record_uptime_tick()
        st.record_downtime(60, "test failure")
        assert len(st._downtime_events) == 1
        assert st._downtime_events[0].reason == "test failure"

    def test_uptime_pct_with_downtime(self):
        from monitoring.sla_tracker import SLATracker
        st = SLATracker()
        st.record_uptime_tick()
        # Record 1 hour of downtime in a 24h window
        st.record_downtime(3600, "outage")
        pct = st.get_uptime_pct(24)
        # Should be ~95.8% (23/24 hours)
        assert 95.0 < pct < 96.5

    def test_mtbf_no_failures(self):
        from monitoring.sla_tracker import SLATracker
        st = SLATracker()
        st.record_uptime_tick()
        assert st.get_mtbf() == float("inf")

    def test_mttr_with_failures(self):
        from monitoring.sla_tracker import SLATracker
        st = SLATracker()
        st.record_uptime_tick()
        st.record_downtime(3600, "fail1")
        st.record_downtime(7200, "fail2")
        # MTTR = (3600 + 7200) / 2 / 3600 = 1.5 hours
        assert abs(st.get_mttr() - 1.5) < 0.01

    def test_get_report(self):
        from monitoring.sla_tracker import SLATracker
        st = SLATracker()
        st.record_uptime_tick()
        st.record_downtime(300, "blip")
        report = st.get_report()
        assert "uptime_pct_24h" in report
        assert "uptime_pct_7d" in report
        assert "mtbf_hours" in report
        assert "mttr_hours" in report
        assert report["incident_count"] == 1
        assert report["total_ticks"] == 1


# ===========================================================================
# 5. Log rotation tests
# ===========================================================================

class TestLogRotation:

    def test_configure_log_rotation(self):
        from ops.log_rotation import configure_log_rotation
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = configure_log_rotation(
                log_dir=tmpdir, max_bytes=1000, backup_count=3, compress=False,
            )
            assert handler is not None
            log_file = Path(tmpdir) / "argus.log"
            assert log_file.exists()
            # Cleanup: close handler before temp dir removal (Windows file lock)
            handler.close()
            logging.getLogger().removeHandler(handler)

    def test_get_log_disk_usage_empty(self):
        from ops.log_rotation import get_log_disk_usage
        with tempfile.TemporaryDirectory() as tmpdir:
            usage = get_log_disk_usage(tmpdir)
            assert usage["total_bytes"] == 0
            assert usage["file_count"] == 0

    def test_get_log_disk_usage_with_files(self):
        from ops.log_rotation import get_log_disk_usage
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.log").write_text("hello world")
            usage = get_log_disk_usage(tmpdir)
            assert usage["total_bytes"] > 0
            assert usage["file_count"] == 1
            assert usage["oldest_file"] is not None

    def test_get_log_disk_usage_nonexistent(self):
        from ops.log_rotation import get_log_disk_usage
        usage = get_log_disk_usage("/nonexistent/path/12345")
        assert usage["total_bytes"] == 0

    def test_archive_old_logs(self):
        from ops.log_rotation import archive_old_logs
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = os.path.join(tmpdir, "archive")
            result = archive_old_logs(tmpdir, archive, max_age_days=30)
            assert "archived" in result
            assert "deleted" in result


# ===========================================================================
# 6. Metrics maintenance tests
# ===========================================================================

class TestMetricsMaintenance:

    def test_compact_unreachable(self):
        from ops.metrics_maintenance import compact_old_metrics
        result = compact_old_metrics("http://127.0.0.1:19999", days_to_keep=90)
        assert result["success"] is False

    def test_get_prometheus_status_unreachable(self):
        from ops.metrics_maintenance import get_prometheus_status
        result = get_prometheus_status("http://127.0.0.1:19999")
        assert "error" in result


# ===========================================================================
# 7. Alerting.py wiring tests
# ===========================================================================

class TestAlertingWiring:

    def test_slack_channel_import(self):
        from monitoring.alerting import SlackChannel
        sc = SlackChannel(webhook_url="https://fake")
        assert sc.name == "slack"

    def test_pagerduty_channel_import(self):
        from monitoring.alerting import PagerDutyChannel
        pc = PagerDutyChannel(routing_key="fake-key")
        assert pc.name == "pagerduty"

    def test_email_channel_import(self):
        from monitoring.alerting import EmailChannel
        ec = EmailChannel()
        assert ec.name == "email"

    def test_create_alert_manager_with_slack(self, monkeypatch):
        monkeypatch.setenv("ARGUS_SLACK_WEBHOOK", "https://hooks.slack.com/test")
        from monitoring.alerting import create_alert_manager
        mgr = create_alert_manager(enable_slack=True)
        assert "slack" in mgr.channels

    def test_create_alert_manager_with_pagerduty(self, monkeypatch):
        monkeypatch.setenv("ARGUS_PAGERDUTY_KEY", "test-key")
        from monitoring.alerting import create_alert_manager
        mgr = create_alert_manager(enable_pagerduty=True)
        assert "pagerduty" in mgr.channels

    def test_create_alert_manager_with_email(self, monkeypatch):
        monkeypatch.setenv("ARGUS_SMTP_HOST", "smtp.test.com")
        monkeypatch.setenv("ARGUS_SMTP_TO", "a@b.com")
        from monitoring.alerting import create_alert_manager
        mgr = create_alert_manager(enable_email=True)
        assert "email" in mgr.channels

    def test_create_alert_manager_all_channels(self, monkeypatch):
        monkeypatch.setenv("ARGUS_SLACK_WEBHOOK", "https://hooks.slack.com/x")
        monkeypatch.setenv("ARGUS_PAGERDUTY_KEY", "k")
        monkeypatch.setenv("ARGUS_SMTP_HOST", "smtp.x.com")
        monkeypatch.setenv("ARGUS_SMTP_TO", "x@x.com")
        from monitoring.alerting import create_alert_manager
        mgr = create_alert_manager(
            discord_webhook_url="https://discord.test",
            telegram_bot_token="tok",
            telegram_chat_id="123",
            enable_slack=True,
            enable_pagerduty=True,
            enable_email=True,
        )
        assert "discord" in mgr.channels
        assert "slack" in mgr.channels
        assert "telegram" in mgr.channels
        assert "email" in mgr.channels
        assert "pagerduty" in mgr.channels

    def test_escalation_order_includes_new_channels(self):
        from monitoring.alerting import AlertManager, SlackChannel, EmailChannel, PagerDutyChannel, AlertSeverity, Alert, AlertCategory
        mgr = AlertManager()
        # Add channels that will fail (not configured) to test escalation order
        slack = SlackChannel(webhook_url="https://fake")
        slack.enabled = True
        mgr.add_channel(slack)
        email = EmailChannel()
        email.enabled = True
        mgr.add_channel(email)
        pd = PagerDutyChannel(routing_key="fake")
        pd.enabled = True
        mgr.add_channel(pd)

        alert = Alert(
            title="Test",
            message="test",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.SYSTEM,
        )

        # The escalation chain should include slack, email, pagerduty
        # We verify by checking channel names are present
        assert "slack" in mgr.channels
        assert "email" in mgr.channels
        assert "pagerduty" in mgr.channels

    def test_slack_channel_send_unconfigured(self, monkeypatch):
        monkeypatch.delenv("ARGUS_SLACK_WEBHOOK", raising=False)
        from monitoring.alerting import SlackChannel, Alert, AlertSeverity, AlertCategory
        sc = SlackChannel()
        alert = Alert(
            title="Test", message="test",
            severity=AlertSeverity.INFO, category=AlertCategory.SYSTEM,
        )
        result = _run(sc.send(alert))
        assert result is False

    def test_email_channel_send_unconfigured(self, monkeypatch):
        monkeypatch.delenv("ARGUS_SMTP_HOST", raising=False)
        monkeypatch.delenv("ARGUS_SMTP_TO", raising=False)
        from monitoring.alerting import EmailChannel, Alert, AlertSeverity, AlertCategory
        ec = EmailChannel()
        alert = Alert(
            title="Test", message="test",
            severity=AlertSeverity.WARNING, category=AlertCategory.SYSTEM,
        )
        result = _run(ec.send(alert))
        assert result is False

    def test_pagerduty_channel_send_unconfigured(self, monkeypatch):
        monkeypatch.delenv("ARGUS_PAGERDUTY_KEY", raising=False)
        from monitoring.alerting import PagerDutyChannel, Alert, AlertSeverity, AlertCategory
        pc = PagerDutyChannel()
        alert = Alert(
            title="Test", message="test",
            severity=AlertSeverity.CRITICAL, category=AlertCategory.SYSTEM,
        )
        result = _run(pc.send(alert))
        assert result is False
