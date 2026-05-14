"""
Telegram Dashboard — real-time remote monitoring and control.

Closes the UX gap vs Cryptohopper/3Commas cloud dashboards.
No platform fees, no data leaving your server (the bot talks directly
to Telegram's API from your Hetzner VPS).

Features:
  • Instant trade alerts (buy/sell/stop with P&L)
  • Circuit breaker and drawdown notifications
  • Live strategy leaderboard on demand
  • Commands: /status /pause /resume /top /balance /force_close
  • Rate-limited to avoid Telegram flood limits

Setup:
  1. Create a bot via @BotFather, copy the token.
  2. Get your chat_id from @userinfobot.
  3. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env

Dependency: python-telegram-bot>=20.0  (pip install python-telegram-bot)
Fallback: logs to console if telegram is not installed.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

MIN_ALERT_INTERVAL = 2.0   # seconds between alerts (flood protection)


class TelegramDashboard:
    """
    Argus Telegram dashboard controller.

    Usage:
        dashboard = TelegramDashboard(
            token=os.environ['TELEGRAM_BOT_TOKEN'],
            chat_id=os.environ['TELEGRAM_CHAT_ID'],
        )
        asyncio.run(dashboard.start())   # starts polling in background
        dashboard.send_trade_alert(...)  # call from strategy on fill
    """

    COMMANDS = ["/status", "/pause", "/resume", "/top", "/balance", "/force_close"]

    def __init__(
        self,
        token: str = "",
        chat_id: str = "",
        on_pause: Optional[Callable] = None,
        on_resume: Optional[Callable] = None,
        on_force_close: Optional[Callable] = None,
        registry=None,      # StrategyRegistry instance
        portfolio=None,     # portfolio object with .summary() method
    ) -> None:
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._on_pause = on_pause
        self._on_resume = on_resume
        self._on_force_close = on_force_close
        self._registry = registry
        self._portfolio = portfolio
        self._paused = False
        self._last_alert_time: float = 0.0
        self._app = None

    # ------------------------------------------------------------------
    def send_trade_alert(
        self,
        action: str,
        symbol: str,
        qty: float,
        price: float,
        pnl_pct: Optional[float] = None,
        reason: str = "",
    ) -> None:
        """
        Send a trade fill alert. Rate-limited to MIN_ALERT_INTERVAL.
        """
        now = time.time()
        if now - self._last_alert_time < MIN_ALERT_INTERVAL:
            return
        self._last_alert_time = now

        emoji = "\U0001f7e2" if action == "buy" else "\U0001f534"
        pnl_str = f" | P&L: {pnl_pct:+.2%}" if pnl_pct is not None else ""
        msg = (
            f"{emoji} <b>{action.upper()}</b> {symbol}\n"
            f"Qty: {qty:.6f} @ ${price:,.2f}{pnl_str}\n"
            f"Reason: {reason}"
        )
        self._send(msg)

    def send_circuit_breaker_alert(self, drawdown_pct: float, action: str) -> None:
        msg = (
            f"\u26a0\ufe0f <b>CIRCUIT BREAKER</b>\n"
            f"Portfolio drawdown: {drawdown_pct:.1%}\n"
            f"Action taken: {action}"
        )
        self._send(msg)

    def send_strategy_suspended(self, strategy_id: str, reason: str) -> None:
        msg = f"\u26d4 Strategy <b>{strategy_id}</b> auto-suspended\nReason: {reason}"
        self._send(msg)

    def send_daily_summary(self, pnl_pct: float, trades: int,
                           win_rate: float, best_strategy: str) -> None:
        emoji = "\U0001f4c8" if pnl_pct >= 0 else "\U0001f4c9"
        msg = (
            f"{emoji} <b>Daily Summary</b>\n"
            f"P&L: {pnl_pct:+.2%} | Trades: {trades}\n"
            f"Win rate: {win_rate:.1%} | Best: {best_strategy}"
        )
        self._send(msg)

    # ------------------------------------------------------------------
    async def start(self) -> None:
        """
        Start Telegram polling. Call once at bot startup.
        Requires python-telegram-bot>=20 installed.
        """
        try:
            from telegram import Update
            from telegram.ext import Application, CommandHandler, ContextTypes

            app = Application.builder().token(self.token).build()

            async def status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
                text = "\U0001f916 Argus is PAUSED" if self._paused else "\U0001f916 Argus is RUNNING"
                if self._portfolio:
                    text += f"\n{self._portfolio.summary()}"
                await update.message.reply_html(text)

            async def pause_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
                self._paused = True
                if self._on_pause:
                    self._on_pause()
                await update.message.reply_text("\u23f8 Argus PAUSED — no new orders will be placed.")

            async def resume_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
                self._paused = False
                if self._on_resume:
                    self._on_resume()
                await update.message.reply_text("\u25b6\ufe0f Argus RESUMED.")

            async def top_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
                if self._registry:
                    await update.message.reply_text(
                        self._registry.leaderboard_str(), parse_mode=None
                    )
                else:
                    await update.message.reply_text("Registry not connected.")

            async def force_close_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
                if self._on_force_close:
                    self._on_force_close()
                await update.message.reply_text("\U0001f6d1 Force close triggered on all open positions.")

            app.add_handler(CommandHandler("status", status_cmd))
            app.add_handler(CommandHandler("pause", pause_cmd))
            app.add_handler(CommandHandler("resume", resume_cmd))
            app.add_handler(CommandHandler("top", top_cmd))
            app.add_handler(CommandHandler("force_close", force_close_cmd))

            self._app = app
            logger.info("Telegram dashboard started")
            await app.run_polling()

        except ImportError:
            logger.warning(
                "python-telegram-bot not installed. "
                "Run: pip install python-telegram-bot"
            )

    def _send(self, text: str) -> None:
        """
        Fire-and-forget send. Falls back to logger if telegram unavailable.
        """
        if not self.token or not self.chat_id:
            logger.info("[TelegramDashboard] %s", text)
            return
        try:
            import httpx
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
            }
            import threading
            def _post():
                try:
                    httpx.post(url, json=payload, timeout=5.0)
                except Exception as exc:
                    logger.warning("Telegram send failed: %s", exc)
            threading.Thread(target=_post, daemon=True).start()
        except ImportError:
            logger.info("[TelegramDashboard] %s", text)
