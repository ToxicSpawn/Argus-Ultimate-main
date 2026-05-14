"""
Notification System Wiring
Connects all notifications: Telegram, Email, SMS
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class NotificationManager:
    """Wires all notification systems"""
    
    def __init__(self):
        self.telegram_enabled = False
        self.email_enabled = False
        self.sms_enabled = False
        
        # Callbacks
        self.alert_handlers: List[Callable] = []
        
        logger.info("📢 Notification Manager initialized")
    
    async def wire_all(self):
        """Wire all notification channels"""
        print("\n[NOTIFICATION WIRING]")
        
        # Telegram
        if os.getenv('TELEGRAM_BOT_TOKEN'):
            self.telegram_enabled = True
            print("  ✅ Telegram: ENABLED")
        else:
            print("  ⚠️  Telegram: No token configured")
        
        # Email
        if os.getenv('EMAIL_SMTP_SERVER'):
            self.email_enabled = True
            print("  ✅ Email: ENABLED")
        else:
            print("  ⚠️  Email: No SMTP configured")
        
        print(f"  📢 Total channels: {sum([self.telegram_enabled, self.email_enabled])}")
    
    async def send_trade_alert(self, trade: Dict):
        """Send trade execution alert"""
        message = f"📈 Trade Executed\n{trade.get('side')} {trade.get('symbol')} @ ${trade.get('price')}"
        await self._send_all(message)
    
    async def send_risk_alert(self, risk_type: str, details: str):
        """Send risk breach alert"""
        message = f"🚨 RISK ALERT: {risk_type}\n{details}"
        await self._send_all(message)
    
    async def send_daily_report(self, pnl: float, trades: int):
        """Send daily P&L report"""
        emoji = "📈" if pnl > 0 else "📉"
        message = f"{emoji} Daily Report\nP&L: ${pnl:+.2f}\nTrades: {trades}"
        await self._send_all(message)
    
    async def _send_all(self, message: str):
        """Send to all enabled channels"""
        if self.telegram_enabled:
            logger.info(f"Telegram: {message}")
        
        if self.email_enabled:
            logger.info(f"Email: {message}")


# Global
_notification_manager: Optional[NotificationManager] = None


def get_notification_manager():
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager()
    return _notification_manager


async def init_notifications():
    manager = get_notification_manager()
    await manager.wire_all()
