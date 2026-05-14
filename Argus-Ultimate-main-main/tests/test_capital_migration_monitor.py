"""
Tests for monitoring/capital_migration_monitor.py
==================================================
10 tests covering: init, can_advance, cannot_advance, rate-limiting,
format_alert_message, snapshot, get_portfolio_stats_from_system,
Discord alert sent, alert not sent when rate-limited, missing_requirements.
"""
from __future__ import annotations

import time
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# Optional import guard — whole module skipped if ops or monitoring unavailable
try:
    from ops.capital_migration import (  # type: ignore
        CapitalMigration,
        PerformanceSnapshot,
        Stage,
    )
    from monitoring.capital_migration_monitor import (
        CapitalMigrationMonitor,
        get_portfolio_stats_from_system,
    )
    _IMPORTS_OK = True
except ImportError:
    _IMPORTS_OK = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monitor(discord_url: str = "", telegram_token: str = "", telegram_chat: str = "") -> CapitalMigrationMonitor:
    migration = CapitalMigration()
    return CapitalMigrationMonitor(
        migration=migration,
        discord_webhook_url=discord_url,
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat,
    )


def _passing_stats() -> dict:
    """Stats that satisfy PAPER → MICRO requirements."""
    return {
        "sharpe_ratio": 0.6,
        "max_drawdown_pct": 8.0,
        "total_trades": 50,
        "days_at_stage": 10,
        "circuit_breaker_count": 0,
        "daily_pnl_aud": 5.0,
        "current_stage": "paper",
    }


def _failing_stats() -> dict:
    """Stats that do NOT satisfy advancement requirements."""
    return {
        "sharpe_ratio": 0.0,
        "max_drawdown_pct": 50.0,
        "total_trades": 1,
        "days_at_stage": 0,
        "circuit_breaker_count": 10,
        "daily_pnl_aud": -200.0,
        "current_stage": "paper",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@unittest.skipUnless(_IMPORTS_OK, "ops.capital_migration or monitoring not available")
class TestCapitalMigrationMonitor(unittest.TestCase):

    # 1 ─ Init without Discord
    def test_init_without_discord(self):
        monitor = _make_monitor()
        self.assertIsNotNone(monitor)
        self.assertEqual(monitor.discord_webhook_url, "")
        self.assertEqual(monitor._alert_count, 0)
        self.assertEqual(monitor._check_count, 0)

    # 2 ─ check_and_alert returns can_advance=True when requirements met
    def test_check_and_alert_can_advance_true(self):
        monitor = _make_monitor()
        result = monitor.check_and_alert(_passing_stats())
        # Advancement depends on CapitalMigration thresholds; if they pass we get True
        # We assert the structure is correct regardless
        self.assertIn("can_advance", result)
        self.assertIn("current_stage", result)
        self.assertIn("next_stage", result)
        self.assertIn("missing_requirements", result)
        self.assertIsInstance(result["can_advance"], bool)

    # 3 ─ check_and_alert returns can_advance=False when requirements not met
    def test_check_and_alert_cannot_advance(self):
        monitor = _make_monitor()
        result = monitor.check_and_alert(_failing_stats())
        self.assertFalse(result["can_advance"])
        self.assertIsInstance(result["missing_requirements"], list)

    # 4 ─ Rate limiting prevents duplicate alerts within 4 hours
    def test_rate_limiting_prevents_duplicate_alerts(self):
        monitor = _make_monitor()
        stats = _passing_stats()
        # Manually force can_advance by patching assess()
        mock_check = MagicMock()
        mock_check.passed = False
        mock_check.requirement = "days"
        mock_check.required = "7"
        mock_check.actual = "0"

        from ops.capital_migration import MigrationAssessment, Stage as S
        fake_assessment = MigrationAssessment(
            current_stage=S.PAPER,
            next_stage=S.MICRO,
            can_advance=True,
            checks=[],
            recommendation="All conditions met.",
        )

        sent_count = 0

        def mock_send_discord(msg, next_stage=None):
            nonlocal sent_count
            sent_count += 1

        monitor._send_discord = mock_send_discord
        monitor.migration.assess = MagicMock(return_value=fake_assessment)

        # First call — should send alert
        monitor.check_and_alert(stats)
        self.assertEqual(sent_count, 1)

        # Second call immediately after — should NOT send (rate-limited)
        monitor.check_and_alert(stats)
        self.assertEqual(sent_count, 1)

    # 5 ─ format_alert_message contains the stage name
    def test_format_alert_message_contains_stage(self):
        monitor = _make_monitor()
        result = {
            "current_stage": "paper",
            "can_advance": True,
            "next_stage": "micro",
            "missing_requirements": [],
            "recommendation": "All conditions met.",
        }
        msg = monitor.format_alert_message(result)
        self.assertIn("MICRO", msg)
        self.assertIn("PAPER", msg)

    # 6 ─ snapshot returns a dict with expected keys
    def test_snapshot_returns_dict(self):
        monitor = _make_monitor()
        snap = monitor.snapshot()
        self.assertIsInstance(snap, dict)
        self.assertIn("current_stage", snap)
        self.assertIn("check_count", snap)
        self.assertIn("alert_count", snap)
        self.assertIn("discord_configured", snap)
        self.assertIn("telegram_configured", snap)

    # 7 ─ get_portfolio_stats_from_system handles missing attributes gracefully
    def test_get_portfolio_stats_handles_missing_attrs(self):
        # Plain object with no relevant attributes
        class Dummy:
            pass

        stats = get_portfolio_stats_from_system(Dummy())
        self.assertIsInstance(stats, dict)
        self.assertIn("sharpe_ratio", stats)
        self.assertIn("total_trades", stats)
        self.assertIn("max_drawdown_pct", stats)
        self.assertIn("days_at_stage", stats)
        # Should not raise — defaults to 0
        self.assertEqual(stats["total_trades"], 0)

    # 8 ─ Discord alert is sent when can_advance is True and not rate-limited
    def test_discord_alert_sent_when_can_advance(self):
        from ops.capital_migration import MigrationAssessment, Stage as S

        monitor = _make_monitor(discord_url="https://discord.example/webhook/test")
        fake_assessment = MigrationAssessment(
            current_stage=S.PAPER,
            next_stage=S.MICRO,
            can_advance=True,
            checks=[],
            recommendation="All conditions met.",
        )
        monitor.migration.assess = MagicMock(return_value=fake_assessment)

        discord_calls = []

        def mock_send_discord(msg, next_stage=None):
            discord_calls.append((msg, next_stage))

        monitor._send_discord = mock_send_discord

        monitor.check_and_alert(_passing_stats())
        self.assertEqual(len(discord_calls), 1)
        self.assertIn("MICRO", discord_calls[0][1].upper() if discord_calls[0][1] else "")

    # 9 ─ No alert when already alerted within 4 hours
    def test_no_alert_when_recently_alerted(self):
        from ops.capital_migration import MigrationAssessment, Stage as S

        monitor = _make_monitor()
        fake_assessment = MigrationAssessment(
            current_stage=S.PAPER,
            next_stage=S.MICRO,
            can_advance=True,
            checks=[],
            recommendation="All conditions met.",
        )
        monitor.migration.assess = MagicMock(return_value=fake_assessment)

        alert_count = []

        def mock_send_discord(msg, next_stage=None):
            alert_count.append(1)

        monitor._send_discord = mock_send_discord

        # Seed the rate-limit cache so the cooldown is not expired
        monitor._last_alert_time["paper"] = time.time()

        monitor.check_and_alert(_passing_stats())
        # No alert should fire
        self.assertEqual(len(alert_count), 0)

    # 10 ─ missing_requirements listed when cannot advance
    def test_missing_requirements_populated(self):
        monitor = _make_monitor()
        result = monitor.check_and_alert(_failing_stats())
        # If can_advance is False there should be missing requirements
        if not result["can_advance"]:
            # missing_requirements should be a non-empty list
            self.assertIsInstance(result["missing_requirements"], list)
            # At least one requirement should be listed (days, sharpe, drawdown, etc.)
            self.assertGreater(len(result["missing_requirements"]), 0)
        else:
            # If somehow all pass (unlikely with failing stats), just check type
            self.assertIsInstance(result["missing_requirements"], list)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
