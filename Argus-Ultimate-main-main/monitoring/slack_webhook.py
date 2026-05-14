"""
Slack Webhook Alerting -- sends trade and system alerts to Slack channels.

Uses Slack Block Kit formatting for rich message display.
Config: ARGUS_SLACK_WEBHOOK env var, or pass webhook_url directly.

Gracefully disabled if no webhook URL configured.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Slack attachment colour constants
_COLOR_GREEN = "#36a64f"
_COLOR_ORANGE = "#ff9900"
_COLOR_RED = "#ff0000"
_COLOR_BLUE = "#3498db"

_SEVERITY_COLORS = {
    "INFO": _COLOR_GREEN,
    "WARNING": _COLOR_ORANGE,
    "CRITICAL": _COLOR_RED,
    "ERROR": _COLOR_RED,
    "DEBUG": _COLOR_BLUE,
}


class SlackWebhook:
    """Async Slack webhook client for ARGUS trading alerts."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        username: str = "ARGUS Bot",
    ) -> None:
        self.webhook_url = webhook_url or os.environ.get("ARGUS_SLACK_WEBHOOK")
        self.username = username

    @property
    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send(
        self,
        message: str,
        channel: Optional[str] = None,
        severity: str = "INFO",
    ) -> bool:
        """Send a message to Slack via incoming webhook.

        Args:
            message:  Plain-text message body.
            channel:  Override channel (e.g. ``#alerts``).  ``None`` uses
                      the channel configured on the webhook itself.
            severity: One of ``INFO``, ``WARNING``, ``CRITICAL``, ``ERROR``,
                      ``DEBUG``.  Controls attachment colour.

        Returns:
            ``True`` on successful delivery, ``False`` otherwise.
        """
        payload = self.format_system_alert(message, severity)
        if channel:
            payload["channel"] = channel
        return await self._post(payload)

    def format_trade_alert(self, trade_data: Dict[str, Any]) -> dict:
        """Build a Slack Block Kit payload for a trade notification.

        ``trade_data`` should contain keys such as ``symbol``, ``side``,
        ``price``, ``quantity``, ``pnl``, ``strategy``.  Missing keys
        are silently omitted.
        """
        fields = []
        field_keys = [
            ("symbol", "Symbol"),
            ("side", "Side"),
            ("price", "Price"),
            ("quantity", "Quantity"),
            ("pnl", "P&L"),
            ("strategy", "Strategy"),
            ("regime", "Regime"),
            ("confidence", "Confidence"),
        ]
        for key, label in field_keys:
            value = trade_data.get(key)
            if value is not None:
                fields.append({
                    "type": "mrkdwn",
                    "text": f"*{label}:*\n{value}",
                })

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Trade Alert -- {trade_data.get('symbol', 'N/A')}",
                },
            },
            {
                "type": "section",
                "fields": fields[:10],  # Slack limit
            },
        ]

        pnl = trade_data.get("pnl")
        if pnl is not None:
            color = _COLOR_GREEN if float(pnl) >= 0 else _COLOR_RED
        else:
            color = _COLOR_BLUE

        return {
            "username": self.username,
            "attachments": [{"color": color, "blocks": blocks}],
        }

    def format_system_alert(self, message: str, severity: str = "INFO") -> dict:
        """Build a simple Slack payload for a system alert.

        Returns a dict ready to POST as JSON.
        """
        color = _SEVERITY_COLORS.get(severity.upper(), _COLOR_GREEN)
        return {
            "username": self.username,
            "attachments": [
                {
                    "color": color,
                    "text": message,
                    "footer": f"ARGUS | {severity.upper()}",
                }
            ],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _post(self, payload: dict) -> bool:
        if not self.is_configured:
            return False
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, self._post_sync, payload)
        except Exception:
            logger.debug("SlackWebhook._post error", exc_info=True)
            return False

    def _post_sync(self, payload: dict) -> bool:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                return resp.status == 200
        except urllib.error.HTTPError as exc:
            logger.warning("SlackWebhook HTTP %d: %s", exc.code, exc.reason)
            return False
        except Exception:
            logger.debug("SlackWebhook._post_sync failed", exc_info=True)
            return False


# Module-level singleton
_instance: Optional[SlackWebhook] = None


def get_slack() -> SlackWebhook:
    global _instance
    if _instance is None:
        _instance = SlackWebhook()
    return _instance
