"""TelegramChannel — sends alerts via Telegram Bot API — Push 60.

Configuration via environment variables::

    ARGUS_TG_BOT_TOKEN    Telegram bot token (required)
    ARGUS_TG_CHAT_ID      Target chat / channel ID (required)
    ARGUS_TG_MIN_LEVEL    Minimum level (default WARNING)
    ARGUS_TG_ENABLED      1/true/yes to enable (default true)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from core.alerts.alert_models import AlertEvent, AlertLevel
from core.alerts.base_channel import AbstractAlertChannel

logger = logging.getLogger(__name__)

_TG_API = "https://api.telegram.org/bot{token}/sendMessage"

# MarkdownV2 chars that must be escaped
_MD2_ESCAPE = r"_*[]()~`>#+-=|{}.!"


def _escape_md2(text: str) -> str:
    for ch in _MD2_ESCAPE:
        text = text.replace(ch, f"\\{ch}")
    return text


class TelegramChannel(AbstractAlertChannel):
    """Delivers alerts to a Telegram chat via Bot API."""

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        min_level: AlertLevel = AlertLevel.WARNING,
        rate_limit_per_min: int = 20,
        enabled: bool = True,
        timeout: float = 10.0,
    ) -> None:
        super().__init__("telegram", min_level, rate_limit_per_min, enabled)
        self._token = bot_token
        self._chat_id = chat_id
        self._timeout = timeout

    @classmethod
    def from_env(cls) -> "TelegramChannel":
        token = os.getenv("ARGUS_TG_BOT_TOKEN", "")
        chat_id = os.getenv("ARGUS_TG_CHAT_ID", "")
        min_level_str = os.getenv("ARGUS_TG_MIN_LEVEL", "WARNING").upper()
        enabled = os.getenv("ARGUS_TG_ENABLED", "true").lower() in {"1", "true", "yes"}
        min_level = AlertLevel[min_level_str] if min_level_str in AlertLevel.__members__ else AlertLevel.WARNING
        return cls(bot_token=token, chat_id=chat_id, min_level=min_level, enabled=enabled)

    async def _deliver(self, event: AlertEvent) -> bool:
        if not self._token or not self._chat_id:
            logger.warning("TelegramChannel: missing bot_token or chat_id")
            return False
        try:
            import aiohttp
        except ImportError:
            logger.warning("TelegramChannel: aiohttp not installed")
            return False

        text = self._format(event)
        url = _TG_API.format(token=self._token)
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "MarkdownV2",
        }
        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=self._timeout)) as resp:
                        if resp.status == 200:
                            return True
                        body = await resp.text()
                        logger.warning("TelegramChannel: HTTP %d: %s", resp.status, body[:200])
                        return False
            except Exception as exc:
                logger.warning("TelegramChannel: attempt %d error: %s", attempt + 1, exc)
        return False

    def _format(self, event: AlertEvent) -> str:
        title = _escape_md2(f"{event.level.emoji} [{event.level.label}] {event.title}")
        lines = [f"*{title}*"]
        if event.symbol:
            lines.append(_escape_md2(f"Symbol: {event.symbol}"))
        if event.body:
            lines.append(_escape_md2(event.body))
        return "\n".join(lines)
