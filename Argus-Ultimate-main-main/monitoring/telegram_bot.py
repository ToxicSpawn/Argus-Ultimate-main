"""
Telegram Alert Bot — real-time P&L and risk notifications to mobile.

Sends alerts for:
  - Trade opens/closes with P&L
  - Circuit breaker trips
  - Drawdown warnings (> 5%)
  - Funding rate harvest opportunities
  - Daily P&L summary (sent at midnight AEST)
  - System startup/shutdown

Setup:
  1. Create a bot via @BotFather: /newbot → get TELEGRAM_BOT_TOKEN
  2. Get your chat ID: message @userinfobot → get TELEGRAM_CHAT_ID
  3. Set env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

Usage:
    bot = TelegramBot()
    await bot.send_trade_alert("BTC/USD", "BUY", price=65000, size=0.01,
                                strategy="trend_follow")
    await bot.send_daily_summary(pnl=125.50, win_rate=0.62, trades=8)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


_LEVEL_EMOJI = {
    AlertLevel.INFO: "ℹ️",
    AlertLevel.WARNING: "⚠️",
    AlertLevel.CRITICAL: "🚨",
}


class TelegramBot:
    """
    Async Telegram bot for ARGUS trading alerts.

    Uses the Telegram Bot API (https://api.telegram.org) via aiohttp.
    Gracefully disabled if TELEGRAM_BOT_TOKEN not set.
    """

    API_BASE = "https://api.telegram.org/bot{token}"
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0  # seconds
    RATE_LIMIT_SECONDS = 1.0  # min time between messages

    def __init__(
        self,
        token: str = None,
        chat_id: str = None,
        enabled: bool = None,
        parse_mode: str = "HTML",
    ):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        # enabled defaults to True if credentials present, False otherwise
        self.enabled = (
            enabled if enabled is not None else bool(self.token and self.chat_id)
        )
        self.parse_mode = parse_mode
        self._last_sent: float = 0.0
        self._session = None  # aiohttp.ClientSession, lazy init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send(self, text: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        """
        Send a raw text message. Prepends level emoji.
        Returns True on success, False on failure (never raises).
        """
        if not self.enabled:
            return False

        emoji = _LEVEL_EMOJI[level]
        full_text = f"{emoji} {text}"

        # Rate limit: sleep if last message too recent
        now = time.monotonic()
        elapsed = now - self._last_sent
        if elapsed < self.RATE_LIMIT_SECONDS:
            await asyncio.sleep(self.RATE_LIMIT_SECONDS - elapsed)

        # Retry loop with exponential backoff
        for attempt in range(1, self.MAX_RETRIES + 1):
            success = await self._post(full_text)
            if success:
                self._last_sent = time.monotonic()
                return True
            if attempt < self.MAX_RETRIES:
                delay = self.RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Telegram send attempt %d/%d failed; retrying in %.1fs",
                    attempt,
                    self.MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)

        logger.error("Telegram: all %d send attempts failed.", self.MAX_RETRIES)
        return False

    async def send_trade_alert(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        strategy: str,
        pnl_usd: float = None,
        exit_price: float = None,
    ) -> bool:
        """
        Format and send a trade open or close alert.

        Open format:
        🟢 TRADE OPEN — BTC/USD
        Side: BUY | Strategy: trend_follow
        Price: $65,000.00 | Qty: 0.001000 BTC
        Value: $65.00

        Close format (when exit_price and pnl_usd provided):
        🔴 TRADE CLOSE — BTC/USD
        Entry: $65,000 → Exit: $66,000
        P&L: +$10.00 (+1.54%) ✓
        Strategy: trend_follow
        """
        base_currency = symbol.split("/")[0] if "/" in symbol else symbol

        if exit_price is not None and pnl_usd is not None:
            # Close alert
            pnl_sign = "+" if pnl_usd >= 0 else ""
            pnl_pct = (
                ((exit_price - price) / price * 100)
                if price > 0
                else 0.0
            )
            if side.upper() in ("SELL", "SHORT"):
                pnl_pct = -pnl_pct
            pct_sign = "+" if pnl_pct >= 0 else ""
            outcome_icon = "✓" if pnl_usd >= 0 else "✗"
            lines = [
                f"🔴 <b>TRADE CLOSE — {symbol}</b>",
                f"Entry: ${price:,.2f} → Exit: ${exit_price:,.2f}",
                f"P&amp;L: {pnl_sign}${pnl_usd:.2f} ({pct_sign}{pnl_pct:.2f}%) {outcome_icon}",
                f"Strategy: {strategy}",
            ]
            level = AlertLevel.INFO if pnl_usd >= 0 else AlertLevel.WARNING
        else:
            # Open alert
            value = price * quantity
            lines = [
                f"🟢 <b>TRADE OPEN — {symbol}</b>",
                f"Side: {side.upper()} | Strategy: {strategy}",
                f"Price: ${price:,.2f} | Qty: {quantity:.6f} {base_currency}",
                f"Value: ${value:.2f}",
            ]
            level = AlertLevel.INFO

        text = "\n".join(lines)
        return await self.send(text, level)

    async def send_risk_alert(
        self,
        alert_type: str,
        value: float,
        threshold: float,
        message: str = "",
    ) -> bool:
        """
        Send a risk/circuit breaker alert.

        Format:
        🚨 RISK ALERT — DRAWDOWN
        Current: -6.5% | Threshold: -5.0%
        Action: Position scaling reduced to 50%
        """
        lines = [
            f"<b>RISK ALERT — {alert_type.upper()}</b>",
            f"Current: {value:.2f}% | Threshold: {threshold:.2f}%",
        ]
        if message:
            lines.append(f"Action: {message}")

        text = "\n".join(lines)
        return await self.send(text, AlertLevel.CRITICAL)

    async def send_daily_summary(
        self,
        pnl_usd: float,
        pnl_pct: float,
        win_rate: float,
        num_trades: int,
        capital: float,
        drawdown_pct: float = 0.0,
    ) -> bool:
        """
        Send the daily P&L summary.

        Format:
        📊 ARGUS DAILY SUMMARY
        Date: 2026-03-10 (AEST)

        P&L: +$125.50 (+1.26%) ✅
        Win Rate: 62.5% (8 trades)
        Capital: $1,125.50
        Max DD: -2.1%

        Keep stacking 💪
        """
        try:
            import pytz

            aest = pytz.timezone("Australia/Sydney")
            date_str = datetime.now(aest).strftime("%Y-%m-%d")
        except Exception:
            date_str = datetime.utcnow().strftime("%Y-%m-%d")

        pnl_sign = "+" if pnl_usd >= 0 else ""
        pct_sign = "+" if pnl_pct >= 0 else ""
        outcome = "✅" if pnl_usd >= 0 else "❌"
        win_rate_pct = win_rate * 100 if win_rate <= 1.0 else win_rate
        closing = "Keep stacking 💪" if pnl_usd >= 0 else "Tomorrow is another day 🎯"

        lines = [
            "📊 <b>ARGUS DAILY SUMMARY</b>",
            f"Date: {date_str} (AEST)",
            "",
            f"P&amp;L: {pnl_sign}${pnl_usd:.2f} ({pct_sign}{pnl_pct:.2f}%) {outcome}",
            f"Win Rate: {win_rate_pct:.1f}% ({num_trades} trades)",
            f"Capital: ${capital:,.2f}",
            f"Max DD: -{abs(drawdown_pct):.1f}%",
            "",
            closing,
        ]

        text = "\n".join(lines)
        level = AlertLevel.INFO if pnl_usd >= 0 else AlertLevel.WARNING
        return await self.send(text, level)

    async def send_funding_alert(
        self, symbol: str, venue: str, rate_pct_8h: float
    ) -> bool:
        """Send funding rate harvest opportunity alert."""
        annualized = rate_pct_8h * 3 * 365
        lines = [
            f"<b>FUNDING HARVEST OPPORTUNITY</b>",
            f"Symbol: {symbol} on {venue}",
            f"Rate: {rate_pct_8h:+.4f}% (8h) | Ann: {annualized:.1f}%",
            f"Action: Consider delta-neutral funding harvest",
        ]
        text = "\n".join(lines)
        return await self.send(text, AlertLevel.INFO)

    async def send_startup_message(
        self, mode: str, capital: float, symbols: List[str]
    ) -> bool:
        """Send system startup notification."""
        symbol_list = ", ".join(symbols) if symbols else "none"
        lines = [
            "🚀 <b>ARGUS SYSTEM STARTED</b>",
            f"Mode: {mode.upper()}",
            f"Capital: ${capital:,.2f}",
            f"Symbols: {symbol_list}",
            f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        ]
        text = "\n".join(lines)
        return await self.send(text, AlertLevel.INFO)

    async def send_shutdown_message(self, reason: str, session_pnl: float) -> bool:
        """Send system shutdown notification."""
        pnl_sign = "+" if session_pnl >= 0 else ""
        lines = [
            "🛑 <b>ARGUS SYSTEM SHUTDOWN</b>",
            f"Reason: {reason}",
            f"Session P&amp;L: {pnl_sign}${session_pnl:.2f}",
            f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        ]
        text = "\n".join(lines)
        level = AlertLevel.WARNING if "error" in reason.lower() else AlertLevel.INFO
        return await self.send(text, level)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post(self, text: str) -> bool:
        """Low-level POST to Telegram sendMessage API. Uses aiohttp if available, else urllib."""
        url = self.API_BASE.format(token=self.token) + "/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": self.parse_mode,
            "disable_web_page_preview": True,
        }

        # Try aiohttp first (preferred async path)
        try:
            import aiohttp

            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()

            async with self._session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    return True
                body = await resp.text()
                logger.warning("Telegram API error %d: %s", resp.status, body[:200])
                return False

        except ImportError:
            pass
        except Exception as exc:
            logger.warning("Telegram aiohttp POST failed: %s", exc)
            return False

        # Fallback: urllib (sync, run in executor to avoid blocking event loop)
        try:
            import json as _json

            data = _json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, _urllib_send, req)
            return result
        except Exception as exc:
            logger.warning("Telegram urllib POST failed: %s", exc)
            return False

    async def close(self):
        """Close the aiohttp session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled (no credentials)"
        return f"TelegramBot({status}, chat_id={self.chat_id or 'not set'})"


def _urllib_send(req: urllib.request.Request) -> bool:
    """Synchronous urllib send; called via run_in_executor."""
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:
        logger.warning("Telegram urllib send error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_bot: Optional[TelegramBot] = None


def get_bot(**kwargs) -> TelegramBot:
    """Get or create the default global Telegram bot."""
    global _default_bot
    if _default_bot is None:
        _default_bot = TelegramBot(**kwargs)
    return _default_bot
