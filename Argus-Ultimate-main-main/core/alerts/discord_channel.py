"""DiscordChannel — sends rich embeds to Discord webhook — Push 60.

Configuration via environment variables::

    ARGUS_DISCORD_WEBHOOK_URL   Discord webhook URL (required)
    ARGUS_DISCORD_MIN_LEVEL     Minimum level (default WARNING)
    ARGUS_DISCORD_ENABLED       1/true/yes (default true)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from core.alerts.alert_models import AlertEvent, AlertLevel
from core.alerts.base_channel import AbstractAlertChannel

logger = logging.getLogger(__name__)

_LEVEL_COLOURS = {
    AlertLevel.DEBUG:    0x95A5A6,   # grey
    AlertLevel.INFO:     0x3498DB,   # blue
    AlertLevel.WARNING:  0xF39C12,   # orange
    AlertLevel.ERROR:    0xE74C3C,   # red
    AlertLevel.CRITICAL: 0x8E44AD,   # purple
}


class DiscordChannel(AbstractAlertChannel):
    """Delivers alerts as Discord embed messages via webhook."""

    def __init__(
        self,
        webhook_url: str,
        min_level: AlertLevel = AlertLevel.WARNING,
        rate_limit_per_min: int = 20,
        enabled: bool = True,
        timeout: float = 10.0,
        username: str = "Argus",
    ) -> None:
        super().__init__("discord", min_level, rate_limit_per_min, enabled)
        self._webhook_url = webhook_url
        self._timeout = timeout
        self._username = username

    @classmethod
    def from_env(cls) -> "DiscordChannel":
        url = os.getenv("ARGUS_DISCORD_WEBHOOK_URL", "")
        min_level_str = os.getenv("ARGUS_DISCORD_MIN_LEVEL", "WARNING").upper()
        enabled = os.getenv("ARGUS_DISCORD_ENABLED", "true").lower() in {"1", "true", "yes"}
        min_level = AlertLevel[min_level_str] if min_level_str in AlertLevel.__members__ else AlertLevel.WARNING
        return cls(webhook_url=url, min_level=min_level, enabled=enabled)

    async def _deliver(self, event: AlertEvent) -> bool:
        if not self._webhook_url:
            logger.warning("DiscordChannel: missing webhook URL")
            return False
        try:
            import aiohttp
        except ImportError:
            logger.warning("DiscordChannel: aiohttp not installed")
            return False

        payload = self._build_payload(event)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as resp:
                    if resp.status in {200, 204}:
                        return True
                    body = await resp.text()
                    logger.warning("DiscordChannel: HTTP %d: %s", resp.status, body[:200])
                    return False
        except Exception as exc:
            logger.error("DiscordChannel: error: %s", exc)
            return False

    def _build_payload(self, event: AlertEvent) -> dict:
        desc_parts = []
        if event.symbol:
            desc_parts.append(f"**Symbol:** {event.symbol}")
        if event.body:
            desc_parts.append(event.body)
        embed = {
            "title": f"{event.level.emoji} {event.title}",
            "description": "\n".join(desc_parts) or "No details.",
            "color": _LEVEL_COLOURS.get(event.level, 0xFFFFFF),
            "footer": {"text": f"Argus | {event.source}"},
            "fields": [
                {"name": "Level", "value": event.level.label, "inline": True},
            ],
        }
        if event.tags:
            embed["fields"].append(
                {"name": "Tags", "value": ", ".join(event.tags), "inline": True}
            )
        return {"username": self._username, "embeds": [embed]}
