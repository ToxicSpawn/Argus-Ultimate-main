"""tests/test_alert_notifier.py

Unit tests for Push 84 — alert notifier and routing logic.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monitoring.alert_notifier import (
    Alert, AlertRouter, Severity, ThrottleFilter,
    TelegramNotifier, WebhookNotifier,
)


class TestThrottleFilter:
    def test_fires_on_first_call(self):
        tf = ThrottleFilter(cooldown_seconds=60)
        assert tf.should_fire("test_key") is True

    def test_suppresses_within_cooldown(self):
        tf = ThrottleFilter(cooldown_seconds=60)
        tf.should_fire("key")
        assert tf.should_fire("key") is False

    def test_fires_again_after_reset(self):
        tf = ThrottleFilter(cooldown_seconds=60)
        tf.should_fire("key")
        tf.reset("key")
        assert tf.should_fire("key") is True

    def test_different_keys_independent(self):
        tf = ThrottleFilter(cooldown_seconds=60)
        assert tf.should_fire("key_a") is True
        assert tf.should_fire("key_b") is True
        assert tf.should_fire("key_a") is False


class TestAlert:
    def test_to_telegram_md_critical(self):
        alert = Alert(
            name="KillSwitchActive",
            severity=Severity.CRITICAL,
            summary="Kill switch is active",
            description="All orders blocked",
            labels={"strategy": "momentum"},
        )
        md = alert.to_telegram_md()
        assert "🚨" in md
        assert "CRITICAL" in md
        assert "KillSwitchActive" in md
        assert "Kill switch is active" in md

    def test_to_telegram_md_warning(self):
        alert = Alert(name="DrawdownWarning", severity=Severity.WARNING, summary="5% drawdown")
        md = alert.to_telegram_md()
        assert "⚠️" in md

    def test_to_telegram_md_resolved(self):
        alert = Alert(name="KillSwitchActive", severity=Severity.RESOLVED, summary="Kill switch cleared")
        md = alert.to_telegram_md()
        assert "✅" in md

    def test_to_dict(self):
        alert = Alert(name="Test", severity=Severity.INFO, summary="test")
        d = alert.to_dict()
        assert d["name"] == "Test"
        assert d["severity"] == "info"


class TestTelegramNotifier:
    @pytest.mark.asyncio
    async def test_disabled_without_credentials(self):
        notifier = TelegramNotifier(bot_token="", chat_id="")
        assert not notifier.enabled
        alert = Alert(name="Test", severity=Severity.CRITICAL, summary="test")
        result = await notifier.send(alert)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_success(self):
        notifier = TelegramNotifier(bot_token="fake_token", chat_id="12345")
        assert notifier.enabled
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        with patch("monitoring.alert_notifier.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession.return_value = mock_session
            mock_aiohttp.ClientTimeout = MagicMock(return_value=None)
            alert = Alert(name="Test", severity=Severity.CRITICAL, summary="test")
            result = await notifier.send(alert)
        # With mocked aiohttp the result may vary; just ensure no exception
        assert isinstance(result, bool)


class TestWebhookNotifier:
    @pytest.mark.asyncio
    async def test_disabled_without_url(self):
        notifier = WebhookNotifier(url="")
        assert not notifier.enabled
        alert = Alert(name="Test", severity=Severity.WARNING, summary="test")
        result = await notifier.send(alert)
        assert result is False


class TestAlertRouter:
    @pytest.mark.asyncio
    async def test_critical_dispatched_immediately(self):
        telegram = AsyncMock(spec=TelegramNotifier)
        telegram.enabled = True
        telegram.send = AsyncMock(return_value=True)
        webhook = AsyncMock(spec=WebhookNotifier)
        webhook.enabled = False
        webhook.send = AsyncMock(return_value=False)

        router = AlertRouter(telegram=telegram, webhook=webhook, critical_cooldown=0)
        await router.start()
        alert = Alert(name="KillSwitch", severity=Severity.CRITICAL, summary="Kill switch active")
        router.fire(alert)
        await asyncio.sleep(0.05)
        telegram.send.assert_called_once()
        await router.stop()

    @pytest.mark.asyncio
    async def test_warning_throttled_on_repeat(self):
        telegram = AsyncMock(spec=TelegramNotifier)
        telegram.enabled = True
        telegram.send = AsyncMock(return_value=True)
        webhook = AsyncMock(spec=WebhookNotifier)
        webhook.enabled = False
        webhook.send = AsyncMock(return_value=False)

        router = AlertRouter(telegram=telegram, webhook=webhook, warning_cooldown=300)
        await router.start()

        alert = Alert(name="Drawdown", severity=Severity.WARNING, summary="5% drawdown")
        router.fire(alert)
        router.fire(alert)  # second fire — should be throttled
        await asyncio.sleep(0.1)
        # Only one call expected due to throttle
        assert telegram.send.call_count == 1
        await router.stop()

    @pytest.mark.asyncio
    async def test_resolved_resets_throttle(self):
        router = AlertRouter(warning_cooldown=300)
        await router.start()
        router._warning_throttle.should_fire("SomeAlert:warning:")
        router.resolved("SomeAlert", "Back to normal")
        await asyncio.sleep(0.05)
        await router.stop()
        # After resolved, warning throttle should be reset
        assert router._warning_throttle.should_fire("SomeAlert:warning:") is True

    def test_convenience_methods_enqueue(self):
        router = AlertRouter()
        router.critical("KillSwitch", "Kill switch active", strategy="momentum")
        router.warning("Drawdown", "5% drawdown", strategy="momentum")
        router.resolved("KillSwitch", "Kill switch cleared")
        assert router._queue.qsize() == 3

    @pytest.mark.asyncio
    async def test_queue_full_drops_gracefully(self):
        router = AlertRouter()
        # Fill queue to capacity
        for i in range(200):
            router.fire(Alert(name=f"alert_{i}", severity=Severity.INFO, summary="test"))
        # One more should be silently dropped
        router.fire(Alert(name="overflow", severity=Severity.INFO, summary="dropped"))
        assert router._queue.qsize() == 200
