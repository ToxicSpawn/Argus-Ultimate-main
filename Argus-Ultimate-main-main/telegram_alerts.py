"""
Telegram Alerts System
======================
Sends trade alerts, warnings, and system status to Telegram.

Features:
- Trade execution alerts
- P&L updates
- Drawdown warnings
- System health notifications
- Daily performance summary

Setup:
1. Create a Telegram bot via @BotFather
2. Get your chat ID via @userinfobot
3. Set environment variables:
   - TELEGRAM_BOT_TOKEN=your_bot_token
   - TELEGRAM_CHAT_ID=your_chat_id
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger("telegram_alerts")


class AlertLevel(Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertConfig:
    """Telegram alert configuration."""
    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = True
    min_level: AlertLevel = AlertLevel.INFO
    rate_limit_seconds: float = 1.0  # Min time between alerts


class TelegramAlerts:
    """Send alerts to Telegram."""
    
    BASE_URL = "https://api.telegram.org/bot"
    
    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()
        self._last_alert_time: float = 0
        self._alert_count: int = 0
        self._session = None
    
    @classmethod
    def from_env(cls) -> "TelegramAlerts":
        """Create from environment variables."""
        config = AlertConfig(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            enabled=bool(os.getenv("TELEGRAM_BOT_TOKEN"))
        )
        return cls(config)
    
    async def connect(self) -> bool:
        """Verify bot connection."""
        if not self.config.bot_token or not self.config.chat_id:
            logger.warning("Telegram not configured - alerts disabled")
            return False
        
        try:
            import aiohttp
            self._session = aiohttp.ClientSession()
            
            url = f"{self.BASE_URL}{self.config.bot_token}/getMe"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    bot_name = data.get("result", {}).get("username", "unknown")
                    logger.info(f"Telegram connected: @{bot_name}")
                    return True
                else:
                    logger.error(f"Telegram connect failed: HTTP {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"Telegram connect error: {e}")
            return False
    
    async def send_alert(self, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        """Send an alert to Telegram.
        
        Args:
            message: Alert message text
            level: Alert level (affects emoji and filtering)
        """
        if not self.config.enabled:
            return False
        
        if not self._session:
            await self.connect()
        
        # Rate limiting
        now = time.time()
        if now - self._last_alert_time < self.config.rate_limit_seconds:
            return False
        
        # Level filtering
        level_order = [AlertLevel.INFO, AlertLevel.SUCCESS, AlertLevel.WARNING, 
                      AlertLevel.ERROR, AlertLevel.CRITICAL]
        if level_order.index(level) < level_order.index(self.config.min_level):
            return False
        
        # Add emoji prefix
        emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.SUCCESS: "✅",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }
        
        formatted_msg = f"{emoji.get(level, '')} {message}"
        
        try:
            url = f"{self.BASE_URL}{self.config.bot_token}/sendMessage"
            payload = {
                "chat_id": self.config.chat_id,
                "text": formatted_msg,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            }
            
            async with self._session.post(url, json=payload) as resp:
                if resp.status == 200:
                    self._last_alert_time = now
                    self._alert_count += 1
                    return True
                else:
                    logger.error(f"Telegram send failed: HTTP {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
    
    async def send_trade_alert(self, symbol: str, side: str, price: float, 
                                quantity: float, pnl: float = None, pnl_pct: float = None):
        """Send trade execution alert."""
        side_emoji = "🟢 BUY" if side.lower() in ("buy", "long") else "🔴 SELL"
        
        msg = f"<b>Trade Executed</b>\n"
        msg += f"{side_emoji} {symbol}\n"
        msg += f"Price: ${price:,.2f}\n"
        msg += f"Quantity: {quantity:.6f}\n"
        
        if pnl is not None:
            pnl_emoji = "📈" if pnl >= 0 else "📉"
            msg += f"P&L: {pnl_emoji} ${pnl:,.2f}"
            if pnl_pct is not None:
                msg += f" ({pnl_pct:+.2f}%)"
        
        await self.send_alert(msg, AlertLevel.SUCCESS if (pnl or 0) >= 0 else AlertLevel.WARNING)
    
    async def send_pnl_update(self, total_pnl: float, pnl_pct: float, 
                               win_rate: float, trade_count: int):
        """Send P&L summary update."""
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        
        msg = f"<b>P&L Update</b>\n"
        msg += f"Total P&L: {pnl_emoji} ${total_pnl:,.2f} ({pnl_pct:+.2f}%)\n"
        msg += f"Win Rate: {win_rate:.1f}%\n"
        msg += f"Trades: {trade_count}"
        
        level = AlertLevel.SUCCESS if total_pnl >= 0 else AlertLevel.WARNING
        await self.send_alert(msg, level)
    
    async def send_drawdown_warning(self, current_drawdown: float, max_allowed: float):
        """Send drawdown warning."""
        msg = f"<b>⚠️ Drawdown Warning</b>\n"
        msg += f"Current: {current_drawdown:.2f}%\n"
        msg += f"Max Allowed: {max_allowed:.2f}%"
        
        level = AlertLevel.CRITICAL if current_drawdown > max_allowed * 0.9 else AlertLevel.WARNING
        await self.send_alert(msg, level)
    
    async def send_circuit_breaker(self, reason: str):
        """Send circuit breaker alert."""
        msg = f"<b>🛑 CIRCUIT BREAKER TRIGGERED</b>\n"
        msg += f"Reason: {reason}\n"
        msg += f"Trading paused"
        
        await self.send_alert(msg, AlertLevel.CRITICAL)
    
    async def send_daily_summary(self, stats: dict):
        """Send daily performance summary."""
        pnl = stats.get("total_pnl", 0)
        pnl_pct = stats.get("pnl_pct", 0)
        win_rate = stats.get("win_rate", 0)
        trades = stats.get("trades", 0)
        best_trade = stats.get("best_trade", 0)
        worst_trade = stats.get("worst_trade", 0)
        
        pnl_emoji = "📈" if pnl >= 0 else "📉"
        
        msg = f"<b>📊 Daily Summary</b>\n"
        msg += f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
        msg += f"─────────────────\n"
        msg += f"P&L: {pnl_emoji} ${pnl:,.2f} ({pnl_pct:+.2f}%)\n"
        msg += f"Win Rate: {win_rate:.1f}%\n"
        msg += f"Trades: {trades}\n"
        msg += f"Best: +${best_trade:,.2f}\n"
        msg += f"Worst: -${abs(worst_trade):,.2f}"
        
        await self.send_alert(msg, AlertLevel.SUCCESS if pnl >= 0 else AlertLevel.WARNING)
    
    async def send_system_status(self, status: str, details: dict = None):
        """Send system status update."""
        msg = f"<b>🤖 System Status</b>\n"
        msg += f"Status: {status}\n"
        
        if details:
            for key, value in details.items():
                msg += f"{key}: {value}\n"
        
        await self.send_alert(msg, AlertLevel.INFO)
    
    async def send_price_alert(self, symbol: str, price: float, 
                                direction: str, threshold: float):
        """Send price threshold alert."""
        msg = f"<b>💰 Price Alert: {symbol}</b>\n"
        msg += f"Price: ${price:,.2f}\n"
        msg += f"Direction: {direction}\n"
        msg += f"Threshold: ${threshold:,.2f}"
        
        await self.send_alert(msg, AlertLevel.INFO)
    
    async def close(self):
        """Close the session."""
        if self._session:
            await self._session.close()


# ── Demo ─────────────────────────────────────────────────────────────────────

async def demo():
    """Demo Telegram alerts (requires valid token/chat_id)."""
    alerts = TelegramAlerts.from_env()
    
    if not alerts.config.bot_token:
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables")
        print("Example:")
        print('  $env:TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."')
        print('  $env:TELEGRAM_CHAT_ID="123456789"')
        return
    
    connected = await alerts.connect()
    if not connected:
        print("Failed to connect to Telegram")
        return
    
    # Send test alerts
    await alerts.send_system_status("Starting", {"Mode": "Paper", "Pairs": "XBT, ETH, SOL"})
    await asyncio.sleep(1)
    
    await alerts.send_trade_alert("XBT/AUD", "buy", 107500.0, 0.01)
    await asyncio.sleep(1)
    
    await alerts.send_pnl_update(1275.54, 27.54, 66.7, 12)
    await asyncio.sleep(1)
    
    await alerts.send_daily_summary({
        "total_pnl": 1275.54,
        "pnl_pct": 27.54,
        "win_rate": 66.7,
        "trades": 24,
        "best_trade": 350.00,
        "worst_trade": -125.00
    })
    
    print("Demo alerts sent!")
    await alerts.close()


if __name__ == "__main__":
    asyncio.run(demo())
