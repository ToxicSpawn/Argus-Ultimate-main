"""
Tests for the alerting module.
"""

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime

from monitoring.alerting import (
    Alert,
    AlertSeverity,
    AlertCategory,
    AlertManager,
    RateLimitConfig,
    LogChannel,
    CallbackChannel,
    create_alert_manager,
    get_alert_manager,
)


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation(self):
        """Test basic alert creation."""
        alert = Alert(
            title="Test Alert",
            message="This is a test",
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
        )

        assert alert.title == "Test Alert"
        assert alert.message == "This is a test"
        assert alert.severity == AlertSeverity.INFO
        assert alert.category == AlertCategory.SYSTEM
        assert len(alert.alert_id) == 16  # SHA256 prefix

    def test_alert_signature(self):
        """Test alert signature for deduplication."""
        alert = Alert(
            title="Test Alert",
            message="This is a test",
            severity=AlertSeverity.INFO,
            category=AlertCategory.TRADE,
        )
        assert alert.signature == "trade:Test Alert"

    def test_alert_to_dict(self):
        """Test alert serialization."""
        alert = Alert(
            title="Test Alert",
            message="This is a test",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RISK,
            metadata={"key": "value"},
        )
        d = alert.to_dict()

        assert d["title"] == "Test Alert"
        assert d["severity"] == "warning"
        assert d["category"] == "risk"
        assert d["metadata"]["key"] == "value"

    def test_alert_discord_format(self):
        """Test Discord webhook formatting."""
        alert = Alert(
            title="Test Alert",
            message="This is a test",
            severity=AlertSeverity.ERROR,
            category=AlertCategory.SYSTEM,
        )
        payload = alert.format_for_discord()

        assert "embeds" in payload
        assert len(payload["embeds"]) == 1
        assert "Test Alert" in payload["embeds"][0]["title"]

    def test_alert_telegram_format(self):
        """Test Telegram formatting."""
        alert = Alert(
            title="Test Alert",
            message="This is a test",
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
        )
        text = alert.format_for_telegram()

        assert "*Test Alert*" in text
        assert "This is a test" in text
        assert "risk" in text.lower()


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_severity_levels(self):
        """Test severity level ordering."""
        assert AlertSeverity.DEBUG.level < AlertSeverity.INFO.level
        assert AlertSeverity.INFO.level < AlertSeverity.WARNING.level
        assert AlertSeverity.WARNING.level < AlertSeverity.ERROR.level
        assert AlertSeverity.ERROR.level < AlertSeverity.CRITICAL.level

    def test_severity_emoji(self):
        """Test severity emojis."""
        assert AlertSeverity.WARNING.emoji == "⚠️"
        assert AlertSeverity.ERROR.emoji == "❌"
        assert AlertSeverity.CRITICAL.emoji == "🚨"


class TestLogChannel:
    """Tests for LogChannel."""

    @pytest.mark.asyncio
    async def test_log_channel_send(self):
        """Test sending alert through log channel."""
        channel = LogChannel()

        alert = Alert(
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
        )

        result = await channel.send(alert)
        assert result is True
        assert channel._delivery_count == 1

    def test_log_channel_should_send(self):
        """Test severity filtering."""
        channel = LogChannel(min_severity=AlertSeverity.WARNING)

        info_alert = Alert(
            title="Info",
            message="Info message",
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
        )
        assert channel.should_send(info_alert) is False

        warn_alert = Alert(
            title="Warning",
            message="Warning message",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.SYSTEM,
        )
        assert channel.should_send(warn_alert) is True


class TestCallbackChannel:
    """Tests for CallbackChannel."""

    @pytest.mark.asyncio
    async def test_callback_channel(self):
        """Test callback-based channel."""
        received_alerts = []

        def callback(alert: Alert) -> bool:
            received_alerts.append(alert)
            return True

        channel = CallbackChannel(callback=callback, name="test_callback")

        alert = Alert(
            title="Test",
            message="Test",
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
        )

        result = await channel.send(alert)
        assert result is True
        assert len(received_alerts) == 1
        assert received_alerts[0].title == "Test"


class TestAlertManager:
    """Tests for AlertManager."""

    @pytest.mark.asyncio
    async def test_send_alert(self):
        """Test basic alert sending."""
        manager = AlertManager()

        alert = await manager.send_alert(
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
        )

        assert alert is not None
        assert alert.title == "Test Alert"
        assert "log" in alert.delivered_channels

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test alert rate limiting."""
        config = RateLimitConfig(
            max_alerts_per_minute=2,
            cooldown_seconds=0.1,
        )
        manager = AlertManager(rate_limit_config=config)

        # First two should succeed
        alert1 = await manager.send_alert(
            title="Alert 1",
            message="Test",
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
        )
        assert alert1 is not None

        alert2 = await manager.send_alert(
            title="Alert 2",
            message="Test",
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
        )
        assert alert2 is not None

        # Third should be rate limited
        alert3 = await manager.send_alert(
            title="Alert 3",
            message="Test",
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
        )
        assert alert3 is None

    @pytest.mark.asyncio
    async def test_bypass_rate_limit(self):
        """Test bypassing rate limit."""
        config = RateLimitConfig(max_alerts_per_minute=1)
        manager = AlertManager(rate_limit_config=config)

        # First alert uses up the limit
        await manager.send_alert(
            title="Alert 1",
            message="Test",
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
        )

        # Second alert should succeed with bypass
        alert2 = await manager.send_alert(
            title="Alert 2",
            message="Test",
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
            bypass_rate_limit=True,
        )
        assert alert2 is not None

    @pytest.mark.asyncio
    async def test_convenience_methods(self):
        """Test convenience alert methods."""
        manager = AlertManager()

        # Trade alert
        trade_alert = await manager.trade_alert(
            title="Trade Executed",
            message="Bought BTC",
            symbol="BTC/AUD",
        )
        assert trade_alert is not None
        assert trade_alert.category == AlertCategory.TRADE

        # Risk alert
        risk_alert = await manager.risk_alert(
            title="High Risk",
            message="Drawdown warning",
        )
        assert risk_alert is not None
        assert risk_alert.category == AlertCategory.RISK

        # System alert
        system_alert = await manager.system_alert(
            title="System Status",
            message="All systems go",
        )
        assert system_alert is not None
        assert system_alert.category == AlertCategory.SYSTEM

    @pytest.mark.asyncio
    async def test_drawdown_alert(self):
        """Test drawdown threshold alert."""
        manager = AlertManager()

        # Below threshold - no alert
        alert1 = await manager.drawdown_alert(
            current_drawdown=0.05,
            max_drawdown=0.05,
            threshold=0.10,
        )
        assert alert1 is None

        # Above threshold - should alert
        alert2 = await manager.drawdown_alert(
            current_drawdown=0.12,
            max_drawdown=0.12,
            threshold=0.10,
        )
        assert alert2 is not None
        assert alert2.severity == AlertSeverity.WARNING

        # Critical level
        alert3 = await manager.drawdown_alert(
            current_drawdown=0.22,
            max_drawdown=0.22,
            threshold=0.10,
        )
        assert alert3 is not None
        assert alert3.severity == AlertSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_circuit_breaker_alert(self):
        """Test circuit breaker alert."""
        manager = AlertManager()

        alert = await manager.circuit_breaker_alert(
            reason="Daily loss limit exceeded",
            duration_minutes=60,
        )

        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL
        assert "Circuit Breaker" in alert.title

    @pytest.mark.asyncio
    async def test_get_history(self):
        """Test alert history retrieval."""
        manager = AlertManager(history_size=100)

        for i in range(5):
            await manager.send_alert(
                title=f"Alert {i}",
                message="Test",
                severity=AlertSeverity.INFO,
                category=AlertCategory.SYSTEM,
            )

        history = manager.get_history(limit=10)
        assert len(history) == 5

    def test_get_stats(self):
        """Test statistics retrieval."""
        manager = AlertManager()

        stats = manager.get_stats()
        assert "total_alerts" in stats
        assert "rate_limited" in stats
        assert "channels" in stats
        assert "log" in stats["channels"]


class TestCreateAlertManager:
    """Tests for factory function."""

    def test_create_with_defaults(self):
        """Test creating manager with defaults."""
        manager = create_alert_manager()

        assert "log" in manager.channels
        assert len(manager.channels) == 1

    def test_create_with_discord(self):
        """Test creating manager with Discord webhook."""
        manager = create_alert_manager(
            discord_webhook_url="https://discord.com/api/webhooks/123/abc"
        )

        assert "log" in manager.channels
        assert "discord" in manager.channels


class TestGetAlertManager:
    """Tests for global alert manager."""

    def test_singleton_pattern(self):
        """Test that get_alert_manager returns same instance."""
        manager1 = get_alert_manager()
        manager2 = get_alert_manager()

        assert manager1 is manager2
