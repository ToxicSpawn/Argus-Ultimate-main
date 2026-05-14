"""
Discord Webhook Alerting — sends trade alerts to Discord channel.

Sends rich embeds for:
  - New trade entries (blue embed)
  - Trade exits with P&L (green=profit, red=loss)
  - Risk alerts (orange)
  - System errors (red with @here mention)
  - Daily summary (purple)

Config:
  DISCORD_WEBHOOK_URL env var

Gracefully disabled if no webhook URL configured.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

# Discord colour constants
_COLOR_GREEN  = 3_066_993
_COLOR_RED    = 15_158_332
_COLOR_BLUE   = 3_447_003
_COLOR_ORANGE = 15_844_367
_COLOR_PURPLE = 10_181_046
_COLOR_GREY   = 9_807_270


class DiscordWebhook:
    """Async Discord webhook client for ARGUS trading alerts."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        username: str = "ARGUS Bot",
        avatar_url: Optional[str] = None,
    ) -> None:
        self.webhook_url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")
        self.username = username
        self.avatar_url = avatar_url

    @property
    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    # ------------------------------------------------------------------
    # Public send methods
    # ------------------------------------------------------------------

    async def send_trade_entry(
        self,
        symbol: str,
        strategy: str,
        qty_usd: float,
        price: float,
        signal_confidence: float,
        regime: str,
    ) -> bool:
        embed = {
            "title": f"📈 Trade Entry — {symbol}",
            "color": _COLOR_BLUE,
            "fields": [
                {"name": "Strategy", "value": strategy, "inline": True},
                {"name": "Quantity", "value": f"${qty_usd:,.0f}", "inline": True},
                {"name": "Price", "value": f"${price:,.4f}", "inline": True},
                {"name": "Confidence", "value": f"{signal_confidence*100:.0f}%", "inline": True},
                {"name": "Regime", "value": regime, "inline": True},
            ],
            "timestamp": _iso_now(),
        }
        return await self._send({"embeds": [embed]})

    async def send_trade_exit(
        self,
        symbol: str,
        pnl_usd: float,
        pnl_pct: float,
        entry_price: float,
        exit_price: float,
        exit_reason: str,
    ) -> bool:
        profit = pnl_usd >= 0
        color = _COLOR_GREEN if profit else _COLOR_RED
        emoji = "✅" if profit else "❌"
        embed = {
            "title": f"{emoji} Trade Exit — {symbol}",
            "color": color,
            "fields": [
                {"name": "P&L", "value": f"${pnl_usd:+.2f} ({pnl_pct*100:+.2f}%)", "inline": True},
                {"name": "Entry", "value": f"${entry_price:,.4f}", "inline": True},
                {"name": "Exit", "value": f"${exit_price:,.4f}", "inline": True},
                {"name": "Reason", "value": exit_reason, "inline": True},
            ],
            "timestamp": _iso_now(),
        }
        return await self._send({"embeds": [embed]})

    async def send_risk_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "WARNING",
    ) -> bool:
        color = _COLOR_RED if severity == "CRITICAL" else _COLOR_ORANGE
        emoji = "🚨" if severity == "CRITICAL" else "⚠️"
        embed = {
            "title": f"{emoji} Risk Alert — {alert_type}",
            "description": message,
            "color": color,
            "fields": [
                {"name": "Severity", "value": severity, "inline": True},
            ],
            "timestamp": _iso_now(),
        }
        return await self._send({"embeds": [embed]})

    async def send_system_error(self, error_message: str, mention_here: bool = False) -> bool:
        content = "@here " if mention_here else ""
        embed = {
            "title": "🔴 System Error",
            "description": error_message[:2000],
            "color": _COLOR_RED,
            "timestamp": _iso_now(),
        }
        return await self._send({"content": content, "embeds": [embed]})

    async def send_daily_summary(
        self,
        date: str,
        total_pnl_usd: float,
        win_rate: float,
        trade_count: int,
        best_trade: float,
        worst_trade: float,
    ) -> bool:
        profit = total_pnl_usd >= 0
        color = _COLOR_GREEN if profit else _COLOR_RED
        embed = {
            "title": f"📊 Daily Summary — {date}",
            "color": color,
            "fields": [
                {"name": "Total P&L", "value": f"${total_pnl_usd:+.2f}", "inline": True},
                {"name": "Win Rate", "value": f"{win_rate*100:.1f}%", "inline": True},
                {"name": "Trades", "value": str(trade_count), "inline": True},
                {"name": "Best Trade", "value": f"${best_trade:+.2f}", "inline": True},
                {"name": "Worst Trade", "value": f"${worst_trade:+.2f}", "inline": True},
            ],
            "timestamp": _iso_now(),
        }
        return await self._send({"embeds": [embed]})

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _send(self, payload: dict) -> bool:
        """POST payload to Discord webhook. Returns True on success."""
        if not self.is_configured:
            return False

        payload.setdefault("username", self.username)
        if self.avatar_url:
            payload.setdefault("avatar_url", self.avatar_url)

        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, self._post_sync, payload)
        except Exception:
            logger.debug("DiscordWebhook._send error", exc_info=True)
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
                return resp.status in (200, 204)
        except urllib.error.HTTPError as e:
            logger.warning("DiscordWebhook HTTP %d: %s", e.code, e.reason)
            return False
        except Exception:
            logger.debug("DiscordWebhook._post_sync failed", exc_info=True)
            return False


# Module-level singleton
_bot: Optional[DiscordWebhook] = None


def get_discord() -> DiscordWebhook:
    global _bot
    if _bot is None:
        _bot = DiscordWebhook()
    return _bot


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc).isoformat()
