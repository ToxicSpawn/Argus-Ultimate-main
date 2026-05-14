"""monitoring/alert_notifier.py

Argus Ultimate — In-process alert notifier (Push 84).

Lightweight Python-side alerting that fires BEFORE Prometheus/Alertmanager
has had time to evaluate rules. Used for:
  - Instant kill-switch notification
  - Pre-trade risk gate breach
  - WS feed total loss

Provides:
  - TelegramNotifier  — send messages via Bot API
  - WebhookNotifier   — generic HTTP POST (Discord, Slack, custom)
  - AlertRouter       — severity-based routing to multiple notifiers
  - ThrottleFilter    — prevents alert storms (per-key cooldown)

All network calls are async and non-blocking.
Fails silently — never crashes the trading loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    RESOLVED = "resolved"


SEVERITY_EMOJI = {
    Severity.INFO: "ℹ️",
    Severity.WARNING: "⚠️",
    Severity.CRITICAL: "🚨",
    Severity.RESOLVED: "✅",
}


@dataclass
class Alert:
    name: str
    severity: Severity
    summary: str
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    runbook: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_telegram_md(self) -> str:
        emoji = SEVERITY_EMOJI.get(self.severity, "")
        lines = [
            f"{emoji} *ARGUS {self.severity.value.upper()}*",
            f"*{self.name}*",
            f"{self.summary}",
        ]
        if self.description:
            lines.append(self.description)
        if self.labels:
            label_str = " | ".join(f"{k}={v}" for k, v in self.labels.items())
            lines.append(f"`{label_str}`")
        if self.runbook:
            lines.append(f"📖 [Runbook]({self.runbook})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "severity": self.severity.value,
            "summary": self.summary,
            "description": self.description,
            "labels": self.labels,
            "timestamp": self.timestamp,
        }


class ThrottleFilter:
    """Prevents repeated alerts for the same key within a cooldown window."""

    def __init__(self, cooldown_seconds: float = 300.0) -> None:
        self.cooldown = cooldown_seconds
        self._last_fired: Dict[str, float] = {}

    def should_fire(self, key: str) -> bool:
        now = time.time()
        last = self._last_fired.get(key, 0.0)
        if now - last >= self.cooldown:
            self._last_fired[key] = now
            return True
        return False

    def reset(self, key: str) -> None:
        self._last_fired.pop(key, None)


class TelegramNotifier:
    """Sends alerts to a Telegram chat via Bot API."""

    TELEGRAM_API = "https://api.telegram.org"

    def __init__(
        self,
        bot_token: str = "",
        chat_id: str = "",
    ) -> None:
        self.bot_token = bot_token or os.environ.get("ARGUS_TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("ARGUS_TELEGRAM_CHAT_ID", "")
        self._session = None

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    async def send(self, alert: Alert) -> bool:
        """Send alert message. Returns True on success."""
        if not self.enabled:
            log.debug("TelegramNotifier disabled (no token/chat_id)")
            return False
        try:
            import aiohttp
            url = f"{self.TELEGRAM_API}/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": alert.to_telegram_md(),
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        log.info("Telegram alert sent: %s [%s]", alert.name, alert.severity.value)
                        return True
                    body = await resp.text()
                    log.warning("Telegram API error %d: %s", resp.status, body[:200])
                    return False
        except Exception as e:
            log.error("TelegramNotifier error: %s", e)
            return False


class WebhookNotifier:
    """Sends alert JSON payload to a generic HTTP webhook."""

    def __init__(self, url: str = "") -> None:
        self.url = url or os.environ.get("ARGUS_WEBHOOK_URL", "")

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    async def send(self, alert: Alert) -> bool:
        if not self.enabled:
            return False
        try:
            import aiohttp
            payload = alert.to_dict()
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5),
                    headers={"Content-Type": "application/json", "X-Argus-Alert": alert.severity.value},
                ) as resp:
                    ok = 200 <= resp.status < 300
                    if ok:
                        log.info("Webhook alert sent: %s [%s]", alert.name, alert.severity.value)
                    else:
                        log.warning("Webhook error %d for %s", resp.status, alert.name)
                    return ok
        except Exception as e:
            log.error("WebhookNotifier error: %s", e)
            return False


class AlertRouter:
    """
    Routes alerts to notifiers based on severity.
    Critical   → all notifiers, immediate
    Warning    → throttled (default 5m cooldown)
    Info       → webhook only
    Resolved   → all notifiers
    """

    def __init__(
        self,
        telegram: Optional[TelegramNotifier] = None,
        webhook: Optional[WebhookNotifier] = None,
        warning_cooldown: float = 300.0,
        critical_cooldown: float = 60.0,
    ) -> None:
        self.telegram = telegram or TelegramNotifier()
        self.webhook = webhook or WebhookNotifier()
        self._warning_throttle = ThrottleFilter(cooldown_seconds=warning_cooldown)
        self._critical_throttle = ThrottleFilter(cooldown_seconds=critical_cooldown)
        self._queue: asyncio.Queue[Alert] = asyncio.Queue(maxsize=200)
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start background alert dispatch worker."""
        self._worker_task = asyncio.create_task(self._worker(), name="alert-router")
        log.info("AlertRouter started (telegram=%s webhook=%s)",
                 self.telegram.enabled, self.webhook.enabled)

    async def stop(self) -> None:
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    def fire(self, alert: Alert) -> None:
        """Non-blocking: enqueue alert for dispatch. Safe to call from hot path."""
        try:
            self._queue.put_nowait(alert)
        except asyncio.QueueFull:
            log.warning("AlertRouter queue full, dropping alert: %s", alert.name)

    def critical(self, name: str, summary: str, description: str = "", **labels) -> None:
        """Convenience: fire a critical alert."""
        self.fire(Alert(
            name=name,
            severity=Severity.CRITICAL,
            summary=summary,
            description=description,
            labels=labels,
        ))

    def warning(self, name: str, summary: str, description: str = "", **labels) -> None:
        """Convenience: fire a warning alert."""
        self.fire(Alert(
            name=name,
            severity=Severity.WARNING,
            summary=summary,
            description=description,
            labels=labels,
        ))

    def resolved(self, name: str, summary: str, **labels) -> None:
        """Convenience: fire a resolved notification."""
        self.fire(Alert(
            name=name,
            severity=Severity.RESOLVED,
            summary=summary,
            labels=labels,
        ))

    async def _worker(self) -> None:
        while True:
            try:
                alert = await self._queue.get()
                await self._dispatch(alert)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("AlertRouter worker error: %s", e)

    async def _dispatch(self, alert: Alert) -> None:
        key = f"{alert.name}:{alert.severity.value}:{':'.join(sorted(alert.labels.values()))}"

        if alert.severity == Severity.CRITICAL:
            if not self._critical_throttle.should_fire(key):
                return
            await asyncio.gather(
                self.telegram.send(alert),
                self.webhook.send(alert),
                return_exceptions=True,
            )

        elif alert.severity == Severity.WARNING:
            if not self._warning_throttle.should_fire(key):
                return
            await asyncio.gather(
                self.telegram.send(alert),
                self.webhook.send(alert),
                return_exceptions=True,
            )

        elif alert.severity == Severity.RESOLVED:
            self._critical_throttle.reset(key.replace(":critical:", ":resolved:"))
            self._warning_throttle.reset(key.replace(":resolved:", ":warning:"))
            await asyncio.gather(
                self.telegram.send(alert),
                self.webhook.send(alert),
                return_exceptions=True,
            )

        else:  # INFO
            await self.webhook.send(alert)


# Module-level default router (uses env vars automatically)
ROUTER = AlertRouter()
