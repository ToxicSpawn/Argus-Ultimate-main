"""Push 60 — Alert Manager: 26 tests."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# AlertLevel + AlertEvent tests (6)
# ---------------------------------------------------------------------------
from core.alerts.alert_models import AlertEvent, AlertLevel


class TestAlertLevel:
    def test_ordering(self):
        assert AlertLevel.DEBUG < AlertLevel.INFO < AlertLevel.WARNING
        assert AlertLevel.ERROR < AlertLevel.CRITICAL

    def test_emoji(self):
        assert AlertLevel.CRITICAL.emoji == "🚨"
        assert AlertLevel.INFO.emoji == "ℹ️"

    def test_label(self):
        assert AlertLevel.WARNING.label == "Warning"


class TestAlertEvent:
    def test_to_dict_keys(self):
        e = AlertEvent.info("Test", "body")
        d = e.to_dict()
        assert "level" in d and "title" in d and "ts" in d

    def test_formatted_text_contains_title(self):
        e = AlertEvent.warning("Price drop", symbol="BTCUSDT")
        text = e.formatted_text()
        assert "Price drop" in text
        assert "BTCUSDT" in text

    def test_factory_methods(self):
        assert AlertEvent.info("x").level == AlertLevel.INFO
        assert AlertEvent.error("x").level == AlertLevel.ERROR
        assert AlertEvent.critical("x").level == AlertLevel.CRITICAL


# ---------------------------------------------------------------------------
# RateLimiter tests (4)
# ---------------------------------------------------------------------------
from core.alerts.base_channel import RateLimiter


class TestRateLimiter:
    def test_allows_under_limit(self):
        rl = RateLimiter(max_per_minute=5)
        for _ in range(5):
            assert rl.allow() is True

    def test_blocks_over_limit(self):
        rl = RateLimiter(max_per_minute=3)
        for _ in range(3):
            rl.allow()
        assert rl.allow() is False

    def test_current_count(self):
        rl = RateLimiter(max_per_minute=10)
        rl.allow()
        rl.allow()
        assert rl.current_count == 2

    def test_zero_limit_always_blocks(self):
        rl = RateLimiter(max_per_minute=0)
        assert rl.allow() is False


# ---------------------------------------------------------------------------
# AbstractAlertChannel tests (4) via mock subclass
# ---------------------------------------------------------------------------
from core.alerts.base_channel import AbstractAlertChannel


class _MockChannel(AbstractAlertChannel):
    def __init__(self, should_succeed=True, **kwargs):
        super().__init__("mock", **kwargs)
        self._should_succeed = should_succeed

    async def _deliver(self, event):
        return self._should_succeed


class TestAbstractChannel:
    def test_send_increments_sent(self):
        ch = _MockChannel()
        asyncio.get_event_loop().run_until_complete(
            ch.send(AlertEvent.info("test"))
        )
        assert ch.sent == 1

    def test_disabled_channel_skips(self):
        ch = _MockChannel(enabled=False)
        result = asyncio.get_event_loop().run_until_complete(
            ch.send(AlertEvent.info("test"))
        )
        assert result is False
        assert ch.sent == 0

    def test_level_filter(self):
        ch = _MockChannel(min_level=AlertLevel.ERROR)
        result = asyncio.get_event_loop().run_until_complete(
            ch.send(AlertEvent.info("too low"))
        )
        assert result is False

    def test_status_dict(self):
        ch = _MockChannel()
        s = ch.status()
        assert "name" in s and "sent" in s and "enabled" in s


# ---------------------------------------------------------------------------
# TelegramChannel tests (3)
# ---------------------------------------------------------------------------
from core.alerts.telegram_channel import TelegramChannel


class TestTelegramChannel:
    def test_from_env_creates_instance(self):
        ch = TelegramChannel.from_env()
        assert isinstance(ch, TelegramChannel)

    def test_missing_token_returns_false(self):
        ch = TelegramChannel(bot_token="", chat_id="")
        result = asyncio.get_event_loop().run_until_complete(
            ch._deliver(AlertEvent.info("test"))
        )
        assert result is False

    def test_format_escapes_special_chars(self):
        ch = TelegramChannel("token", "chat")
        text = ch._format(AlertEvent.info("Hello. World!"))
        assert "\\." in text or "\\!" in text  # MD2 escaping


# ---------------------------------------------------------------------------
# DiscordChannel tests (2)
# ---------------------------------------------------------------------------
from core.alerts.discord_channel import DiscordChannel


class TestDiscordChannel:
    def test_from_env_creates_instance(self):
        ch = DiscordChannel.from_env()
        assert isinstance(ch, DiscordChannel)

    def test_build_payload_has_embeds(self):
        ch = DiscordChannel(webhook_url="http://example.com")
        payload = ch._build_payload(AlertEvent.warning("MDD", "max drawdown hit", symbol="BTCUSDT"))
        assert "embeds" in payload
        assert len(payload["embeds"]) == 1


# ---------------------------------------------------------------------------
# WebhookChannel tests (2)
# ---------------------------------------------------------------------------
from core.alerts.webhook_channel import WebhookChannel


class TestWebhookChannel:
    def test_from_env_creates_instance(self):
        ch = WebhookChannel.from_env()
        assert isinstance(ch, WebhookChannel)

    def test_missing_url_returns_false(self):
        ch = WebhookChannel(url="")
        result = asyncio.get_event_loop().run_until_complete(
            ch._deliver(AlertEvent.info("test"))
        )
        assert result is False


# ---------------------------------------------------------------------------
# AlertManager tests (5)
# ---------------------------------------------------------------------------
from core.alerts.alert_manager import AlertManager


class TestAlertManager:
    def test_register_channel(self):
        mgr = AlertManager()
        mgr.register_channel(_MockChannel())
        assert "mock" in mgr.channels

    def test_unregister_channel(self):
        mgr = AlertManager()
        mgr.register_channel(_MockChannel())
        mgr.unregister_channel("mock")
        assert "mock" not in mgr.channels

    def test_send_returns_count(self):
        mgr = AlertManager()
        mgr.register_channel(_MockChannel())
        sent = asyncio.get_event_loop().run_until_complete(
            mgr.send(AlertEvent.info("hello"))
        )
        assert sent == 1

    def test_global_level_filter(self):
        mgr = AlertManager(min_level=AlertLevel.ERROR)
        mgr.register_channel(_MockChannel())
        sent = asyncio.get_event_loop().run_until_complete(
            mgr.send(AlertEvent.info("too low"))
        )
        assert sent == 0

    def test_status_dict(self):
        mgr = AlertManager()
        mgr.register_channel(_MockChannel())
        s = mgr.status()
        assert "channels" in s and "total_sent" in s
