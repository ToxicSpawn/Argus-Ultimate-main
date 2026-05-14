"""Push 68 — Async Telegram bot notifier for Argus alerts.

Sends formatted alert messages to a Telegram chat via Bot API.
Features:
  - Async aiohttp HTTP client (non-blocking)
  - Per-severity message formatting with emoji
  - Rate limiting: max N messages per minute
  - Exponential backoff retry (3 attempts)
  - Dry-run mode for testing
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from core.monitoring.alert_rules import AlertEvent, Severity


_SEVERITY_EMOJI = {
    Severity.INFO:      "ℹ️",
    Severity.WARN:      "⚠️",
    Severity.CRITICAL:  "🔴",
    Severity.EMERGENCY: "🚨",
}


@dataclass
class TelegramConfig:
    bot_token: str = ""
    chat_id: str = ""
    max_per_minute: int = 10
    dry_run: bool = False
    timeout_secs: float = 10.0


class TelegramNotifier:
    """Async Telegram alert notifier.

    Args:
        config: TelegramConfig with bot_token and chat_id
    """

    BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, config: TelegramConfig):
        self.cfg = config
        self._sent_times: Deque[float] = deque(maxlen=config.max_per_minute)
        self._total_sent: int = 0
        self._total_failed: int = 0

    async def send_alert(self, event: AlertEvent) -> bool:
        """Send alert event to Telegram. Returns True on success."""
        if not self._rate_ok():
            return False
        msg = self._format_message(event)
        return await self._send(msg)

    async def send_text(self, text: str) -> bool:
        """Send arbitrary text message."""
        if not self._rate_ok():
            return False
        return await self._send(text)

    def _format_message(self, event: AlertEvent) -> str:
        emoji = _SEVERITY_EMOJI.get(event.severity, "")
        ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(event.fired_at))
        return (
            f"{emoji} *Argus Alert* [{event.severity.value}]\n"
            f"Rule: `{event.rule_name}`\n"
            f"Message: {event.message}\n"
            f"Value: `{event.value:.6f}` | Threshold: `{event.threshold}`\n"
            f"Time: `{ts}`"
        )

    async def _send(self, text: str, attempt: int = 0) -> bool:
        if self.cfg.dry_run or not self.cfg.bot_token:
            self._total_sent += 1
            self._sent_times.append(time.time())
            return True

        try:
            import aiohttp
        except ImportError:
            return False

        url = self.BASE_URL.format(token=self.cfg.bot_token)
        payload = {
            "chat_id": self.cfg.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.cfg.timeout_secs)
                ) as resp:
                    if resp.status == 200:
                        self._total_sent += 1
                        self._sent_times.append(time.time())
                        return True
                    if resp.status == 429 and attempt < 3:
                        await asyncio.sleep(2 ** attempt)
                        return await self._send(text, attempt + 1)
                    self._total_failed += 1
                    return False
        except Exception:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                return await self._send(text, attempt + 1)
            self._total_failed += 1
            return False

    def _rate_ok(self) -> bool:
        now = time.time()
        while self._sent_times and now - self._sent_times[0] > 60.0:
            self._sent_times.popleft()
        return len(self._sent_times) < self.cfg.max_per_minute

    @property
    def total_sent(self) -> int:
        return self._total_sent

    @property
    def total_failed(self) -> int:
        return self._total_failed
