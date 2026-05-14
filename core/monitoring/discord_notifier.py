"""Push 68 — Async Discord webhook notifier for Argus alerts.

Sends colour-coded embed messages to a Discord channel
via an incoming webhook URL.

Embed colours:
  INFO      — 0x3498db (blue)
  WARN      — 0xf39c12 (orange)
  CRITICAL  — 0xe74c3c (red)
  EMERGENCY — 0x8e44ad (purple)
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

from core.monitoring.alert_rules import AlertEvent, Severity


_SEVERITY_COLOR = {
    Severity.INFO:      0x3498DB,
    Severity.WARN:      0xF39C12,
    Severity.CRITICAL:  0xE74C3C,
    Severity.EMERGENCY: 0x8E44AD,
}

_SEVERITY_EMOJI = {
    Severity.INFO:      "ℹ️",
    Severity.WARN:      "⚠️",
    Severity.CRITICAL:  "🔴",
    Severity.EMERGENCY: "🚨",
}


@dataclass
class DiscordConfig:
    webhook_url: str = ""
    username: str = "Argus"
    avatar_url: str = ""
    max_per_minute: int = 10
    dry_run: bool = False
    timeout_secs: float = 10.0


class DiscordNotifier:
    """Async Discord webhook alert notifier."""

    def __init__(self, config: DiscordConfig):
        self.cfg = config
        self._sent_times: Deque[float] = deque(maxlen=config.max_per_minute)
        self._total_sent: int = 0
        self._total_failed: int = 0

    async def send_alert(self, event: AlertEvent) -> bool:
        """Send alert as a Discord embed. Returns True on success."""
        if not self._rate_ok():
            return False
        payload = self._build_embed(event)
        return await self._post(payload)

    async def send_text(self, text: str) -> bool:
        if not self._rate_ok():
            return False
        return await self._post({"content": text,
                                  "username": self.cfg.username})

    def _build_embed(self, event: AlertEvent) -> dict:
        color = _SEVERITY_COLOR.get(event.severity, 0xFFFFFF)
        emoji = _SEVERITY_EMOJI.get(event.severity, "")
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(event.fired_at))
        payload = {
            "username": self.cfg.username,
            "embeds": [{
                "title": f"{emoji} Argus Alert — {event.severity.value}",
                "description": event.message,
                "color": color,
                "fields": [
                    {"name": "Rule",      "value": f"`{event.rule_name}`",  "inline": True},
                    {"name": "Value",     "value": f"`{event.value:.6f}`", "inline": True},
                    {"name": "Threshold", "value": f"`{event.threshold}`", "inline": True},
                ],
                "timestamp": ts,
                "footer": {"text": "Argus Ultimate"},
            }]
        }
        if self.cfg.avatar_url:
            payload["avatar_url"] = self.cfg.avatar_url
        return payload

    async def _post(self, payload: dict, attempt: int = 0) -> bool:
        if self.cfg.dry_run or not self.cfg.webhook_url:
            self._total_sent += 1
            self._sent_times.append(time.time())
            return True

        try:
            import aiohttp
        except ImportError:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.cfg.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.cfg.timeout_secs)
                ) as resp:
                    if resp.status in (200, 204):
                        self._total_sent += 1
                        self._sent_times.append(time.time())
                        return True
                    if resp.status == 429 and attempt < 3:
                        await asyncio.sleep(2 ** attempt)
                        return await self._post(payload, attempt + 1)
                    self._total_failed += 1
                    return False
        except Exception:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                return await self._post(payload, attempt + 1)
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
