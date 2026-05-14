"""
Argus Trading System - Alerting Module
======================================

Multi-channel alert system for real-time trading notifications.
Supports Discord, Telegram, and extensible webhook integrations.

Features:
- Rate limiting to prevent alert storms
- Alert severity levels and filtering
- Alert aggregation for similar events
- Alert history and acknowledgment tracking
- Async delivery with retries
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# Alert Types
# =============================================================================

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def emoji(self) -> str:
        """Get emoji for severity."""
        return {
            AlertSeverity.DEBUG: "🔍",
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨",
        }.get(self, "📢")

    @property
    def level(self) -> int:
        """Get numeric level for comparison."""
        return {
            AlertSeverity.DEBUG: 0,
            AlertSeverity.INFO: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.ERROR: 3,
            AlertSeverity.CRITICAL: 4,
        }.get(self, 1)


class AlertCategory(str, Enum):
    """Alert categories for filtering and routing."""
    TRADE = "trade"           # Trade execution alerts
    RISK = "risk"             # Risk management alerts
    SYSTEM = "system"         # System health alerts
    SIGNAL = "signal"         # Strategy signal alerts
    PERFORMANCE = "performance"  # Performance alerts
    CAPITAL = "capital"       # Capital/balance alerts


@dataclass
class Alert:
    """
    Alert message to be sent through notification channels.
    """
    title: str
    message: str
    severity: AlertSeverity
    category: AlertCategory
    timestamp: datetime = field(default_factory=datetime.utcnow)
    alert_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    delivery_attempts: int = 0
    delivered_channels: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.alert_id:
            # Generate unique alert ID
            content = f"{self.title}{self.message}{self.timestamp.isoformat()}"
            self.alert_id = hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def signature(self) -> str:
        """Get signature for deduplication (title + category)."""
        return f"{self.category.value}:{self.title}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "acknowledged": self.acknowledged,
        }

    def format_for_discord(self) -> Dict[str, Any]:
        """Format alert for Discord webhook."""
        color_map = {
            AlertSeverity.DEBUG: 0x808080,    # Gray
            AlertSeverity.INFO: 0x3498DB,     # Blue
            AlertSeverity.WARNING: 0xF39C12,  # Orange
            AlertSeverity.ERROR: 0xE74C3C,    # Red
            AlertSeverity.CRITICAL: 0x9B59B6, # Purple
        }

        embed = {
            "title": f"{self.severity.emoji} {self.title}",
            "description": self.message,
            "color": color_map.get(self.severity, 0x3498DB),
            "timestamp": self.timestamp.isoformat(),
            "footer": {"text": f"Argus | {self.category.value.upper()}"},
            "fields": [],
        }

        # Add metadata fields
        for key, value in list(self.metadata.items())[:5]:  # Limit to 5 fields
            embed["fields"].append({
                "name": key.replace("_", " ").title(),
                "value": str(value)[:100],  # Truncate long values
                "inline": True,
            })

        return {"embeds": [embed]}

    def format_for_telegram(self) -> str:
        """Format alert for Telegram."""
        lines = [
            f"{self.severity.emoji} *{self.title}*",
            "",
            self.message,
            "",
            f"📁 Category: `{self.category.value}`",
            f"⏰ Time: `{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC`",
        ]

        if self.metadata:
            lines.append("\n📊 *Details:*")
            for key, value in list(self.metadata.items())[:5]:
                lines.append(f"  • {key}: `{value}`")

        return "\n".join(lines)


# =============================================================================
# Alert Channels
# =============================================================================

class AlertChannel(ABC):
    """Abstract base class for alert delivery channels."""

    def __init__(self, name: str, min_severity: AlertSeverity = AlertSeverity.INFO):
        self.name = name
        self.min_severity = min_severity
        self.enabled = True
        self._delivery_count = 0
        self._failure_count = 0

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Send alert through this channel. Returns True if successful."""
        pass

    def should_send(self, alert: Alert) -> bool:
        """Check if alert should be sent through this channel."""
        return self.enabled and alert.severity.level >= self.min_severity.level

    @property
    def stats(self) -> Dict[str, Any]:
        """Get channel statistics."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "min_severity": self.min_severity.value,
            "delivered": self._delivery_count,
            "failures": self._failure_count,
        }


class DiscordWebhookChannel(AlertChannel):
    """Discord webhook alert channel."""

    def __init__(
        self,
        webhook_url: str,
        name: str = "discord",
        min_severity: AlertSeverity = AlertSeverity.INFO,
    ):
        super().__init__(name, min_severity)
        self.webhook_url = webhook_url

    async def send(self, alert: Alert) -> bool:
        """Send alert to Discord webhook."""
        if not self.webhook_url:
            logger.warning("Discord webhook URL not configured")
            return False

        try:
            import aiohttp

            payload = alert.format_for_discord()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status in (200, 204):
                        self._delivery_count += 1
                        logger.debug("Alert sent to Discord: %s", alert.title)
                        return True
                    else:
                        self._failure_count += 1
                        logger.warning(
                            "Discord webhook failed: %s - %s",
                            response.status,
                            await response.text(),
                        )
                        return False

        except ImportError:
            logger.warning("aiohttp not installed, Discord alerts disabled")
            return False
        except Exception as e:
            self._failure_count += 1
            logger.error("Discord alert failed: %s", e)
            return False


class TelegramChannel(AlertChannel):
    """Telegram bot alert channel."""

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        name: str = "telegram",
        min_severity: AlertSeverity = AlertSeverity.INFO,
    ):
        super().__init__(name, min_severity)
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    async def send(self, alert: Alert) -> bool:
        """Send alert to Telegram."""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram not configured")
            return False

        try:
            import aiohttp

            payload = {
                "chat_id": self.chat_id,
                "text": alert.format_for_telegram(),
                "parse_mode": "Markdown",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._api_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 200:
                        self._delivery_count += 1
                        logger.debug("Alert sent to Telegram: %s", alert.title)
                        return True
                    else:
                        self._failure_count += 1
                        logger.warning(
                            "Telegram API failed: %s - %s",
                            response.status,
                            await response.text(),
                        )
                        return False

        except ImportError:
            logger.warning("aiohttp not installed, Telegram alerts disabled")
            return False
        except Exception as e:
            self._failure_count += 1
            logger.error("Telegram alert failed: %s", e)
            return False


class TwilioSMSChannel(AlertChannel):
    """
    Twilio SMS alert channel backed by monitoring.twilio_alert.TwilioAlert.

    Wraps the synchronous TwilioAlert.send_sms() call inside an executor so it
    does not block the async event loop.  Credentials are read from environment
    variables by TwilioAlert itself (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
    TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER).  When credentials are absent the
    channel silently returns False and the caller can escalate further.
    """

    def __init__(
        self,
        name: str = "twilio_sms",
        min_severity: AlertSeverity = AlertSeverity.WARNING,
    ):
        super().__init__(name, min_severity)
        try:
            from monitoring.twilio_alert import TwilioAlert
            self._twilio = TwilioAlert()
            if not self._twilio.is_configured():
                logger.debug("TwilioSMSChannel: credentials not set — channel disabled")
                self.enabled = False
        except Exception as exc:
            logger.debug("TwilioSMSChannel init failed: %s", exc)
            self._twilio = None
            self.enabled = False

    async def send(self, alert: Alert) -> bool:
        """Send an SMS via Twilio (run in thread pool to avoid blocking the loop)."""
        if self._twilio is None or not self.enabled:
            return False
        try:
            from monitoring.twilio_alert import AlertLevel as TwilioLevel
            level_map = {
                AlertSeverity.DEBUG: TwilioLevel.INFO,
                AlertSeverity.INFO: TwilioLevel.INFO,
                AlertSeverity.WARNING: TwilioLevel.WARNING,
                AlertSeverity.ERROR: TwilioLevel.CRITICAL,
                AlertSeverity.CRITICAL: TwilioLevel.EMERGENCY,
            }
            twilio_level = level_map.get(alert.severity, TwilioLevel.WARNING)
            message = f"{alert.title}: {alert.message}"
            loop = asyncio.get_running_loop()
            success = await loop.run_in_executor(
                None, lambda: self._twilio.send_sms(message, twilio_level)
            )
            if success:
                self._delivery_count += 1
                logger.debug("Alert sent via Twilio SMS: %s", alert.title)
            else:
                self._failure_count += 1
            return bool(success)
        except Exception as exc:
            self._failure_count += 1
            logger.error("TwilioSMSChannel send failed: %s", exc)
            return False


class SlackChannel(AlertChannel):
    """Slack webhook alert channel backed by monitoring.slack_webhook.SlackWebhook."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        name: str = "slack",
        min_severity: AlertSeverity = AlertSeverity.INFO,
    ):
        super().__init__(name, min_severity)
        self.webhook_url = webhook_url
        try:
            from monitoring.slack_webhook import SlackWebhook
            self._slack = SlackWebhook(webhook_url=webhook_url)
            if not self._slack.is_configured:
                logger.debug("SlackChannel: webhook URL not set -- channel disabled")
                self.enabled = False
        except Exception as exc:
            logger.debug("SlackChannel init failed: %s", exc)
            self._slack = None
            self.enabled = False

    async def send(self, alert: Alert) -> bool:
        if self._slack is None or not self.enabled:
            return False
        try:
            severity_map = {
                AlertSeverity.DEBUG: "DEBUG",
                AlertSeverity.INFO: "INFO",
                AlertSeverity.WARNING: "WARNING",
                AlertSeverity.ERROR: "ERROR",
                AlertSeverity.CRITICAL: "CRITICAL",
            }
            sev = severity_map.get(alert.severity, "INFO")
            msg = f"*{alert.title}*\n{alert.message}"
            success = await self._slack.send(msg, severity=sev)
            if success:
                self._delivery_count += 1
            else:
                self._failure_count += 1
            return success
        except Exception as exc:
            self._failure_count += 1
            logger.error("SlackChannel send failed: %s", exc)
            return False


class PagerDutyChannel(AlertChannel):
    """PagerDuty alert channel backed by monitoring.pagerduty_alert.PagerDutyAlert."""

    def __init__(
        self,
        routing_key: Optional[str] = None,
        name: str = "pagerduty",
        min_severity: AlertSeverity = AlertSeverity.ERROR,
    ):
        super().__init__(name, min_severity)
        self.routing_key = routing_key
        try:
            from monitoring.pagerduty_alert import PagerDutyAlert
            self._pd = PagerDutyAlert(routing_key=routing_key)
            if not self._pd.is_configured:
                logger.debug("PagerDutyChannel: routing key not set -- channel disabled")
                self.enabled = False
        except Exception as exc:
            logger.debug("PagerDutyChannel init failed: %s", exc)
            self._pd = None
            self.enabled = False

    async def send(self, alert: Alert) -> bool:
        if self._pd is None or not self.enabled:
            return False
        try:
            severity_map = {
                AlertSeverity.DEBUG: "info",
                AlertSeverity.INFO: "info",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "error",
                AlertSeverity.CRITICAL: "critical",
            }
            sev = severity_map.get(alert.severity, "critical")
            summary = f"{alert.title}: {alert.message}"
            dedup_key = await self._pd.trigger(
                summary=summary,
                severity=sev,
                component=alert.category.value,
            )
            if dedup_key:
                self._delivery_count += 1
                return True
            else:
                self._failure_count += 1
                return False
        except Exception as exc:
            self._failure_count += 1
            logger.error("PagerDutyChannel send failed: %s", exc)
            return False


class EmailChannel(AlertChannel):
    """Email alert channel backed by monitoring.email_alert.EmailAlert."""

    def __init__(
        self,
        name: str = "email",
        min_severity: AlertSeverity = AlertSeverity.WARNING,
        **smtp_kwargs,
    ):
        super().__init__(name, min_severity)
        self._smtp_kwargs = smtp_kwargs
        try:
            from monitoring.email_alert import EmailAlert
            self._email = EmailAlert(**smtp_kwargs)
            if not self._email.is_configured:
                logger.debug("EmailChannel: SMTP not configured -- channel disabled")
                self.enabled = False
        except Exception as exc:
            logger.debug("EmailChannel init failed: %s", exc)
            self._email = None
            self.enabled = False

    async def send(self, alert: Alert) -> bool:
        if self._email is None or not self.enabled:
            return False
        try:
            subject = f"[ARGUS {alert.severity.value.upper()}] {alert.title}"
            body = f"{alert.message}\n\nCategory: {alert.category.value}\nTime: {alert.timestamp.isoformat()}"
            loop = asyncio.get_running_loop()
            success = await loop.run_in_executor(
                None, lambda: self._email.send(subject, body)
            )
            if success:
                self._delivery_count += 1
            else:
                self._failure_count += 1
            return bool(success)
        except Exception as exc:
            self._failure_count += 1
            logger.error("EmailChannel send failed: %s", exc)
            return False


class LogChannel(AlertChannel):
    """Log-based alert channel (always available fallback)."""

    def __init__(
        self,
        name: str = "log",
        min_severity: AlertSeverity = AlertSeverity.DEBUG,
    ):
        super().__init__(name, min_severity)

    async def send(self, alert: Alert) -> bool:
        """Log alert to Python logger."""
        log_level = {
            AlertSeverity.DEBUG: logging.DEBUG,
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }.get(alert.severity, logging.INFO)

        logger.log(
            log_level,
            "[ALERT][%s] %s: %s",
            alert.category.value.upper(),
            alert.title,
            alert.message,
        )
        self._delivery_count += 1
        return True


class CallbackChannel(AlertChannel):
    """Custom callback-based alert channel."""

    def __init__(
        self,
        callback: Callable[[Alert], bool],
        name: str = "callback",
        min_severity: AlertSeverity = AlertSeverity.INFO,
    ):
        super().__init__(name, min_severity)
        self.callback = callback

    async def send(self, alert: Alert) -> bool:
        """Execute callback with alert."""
        try:
            result = self.callback(alert)
            if result:
                self._delivery_count += 1
            else:
                self._failure_count += 1
            return result
        except Exception as e:
            self._failure_count += 1
            logger.error("Callback alert failed: %s", e)
            return False


# =============================================================================
# Alert Manager
# =============================================================================

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    max_alerts_per_minute: int = 10
    max_alerts_per_hour: int = 60
    cooldown_seconds: float = 5.0  # Min time between same alert type
    aggregate_window_seconds: float = 60.0  # Window for aggregating similar alerts


class AlertManager:
    """
    Central alert management system.

    Features:
    - Multi-channel delivery
    - Rate limiting
    - Alert deduplication/aggregation
    - History tracking
    - Severity filtering
    """

    def __init__(
        self,
        rate_limit_config: Optional[RateLimitConfig] = None,
        history_size: int = 1000,
    ):
        self.channels: Dict[str, AlertChannel] = {}
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.history_size = history_size

        # Alert tracking
        self._alert_history: deque = deque(maxlen=history_size)
        self._recent_timestamps: deque = deque(maxlen=100)
        self._last_alert_time: Dict[str, float] = {}  # signature -> timestamp
        self._pending_aggregation: Dict[str, List[Alert]] = {}

        # CRITICAL alert dedup: don't fire same CRITICAL alert more than once per 5 minutes
        self._critical_dedup_window_s: float = 300.0  # 5 minutes
        self._last_critical_time: Dict[str, float] = {}  # signature -> timestamp

        # Paper mode flag: set to True to suppress CRITICAL reconciliation drift alerts
        self._paper_mode: bool = False

        # Statistics
        self._total_alerts = 0
        self._rate_limited = 0
        self._aggregated = 0

        # Add default log channel
        self.add_channel(LogChannel())

    def add_channel(self, channel: AlertChannel) -> None:
        """Add an alert delivery channel."""
        self.channels[channel.name] = channel
        logger.info("Added alert channel: %s", channel.name)

    def remove_channel(self, name: str) -> None:
        """Remove an alert channel."""
        if name in self.channels:
            del self.channels[name]
            logger.info("Removed alert channel: %s", name)

    def _is_rate_limited(self, alert: Alert) -> bool:
        """Check if alert should be rate limited."""
        now = time.time()

        # Check per-minute limit
        minute_ago = now - 60
        recent_minute = sum(1 for ts in self._recent_timestamps if ts > minute_ago)
        if recent_minute >= self.rate_limit_config.max_alerts_per_minute:
            return True

        # Check per-hour limit
        hour_ago = now - 3600
        recent_hour = sum(1 for ts in self._recent_timestamps if ts > hour_ago)
        if recent_hour >= self.rate_limit_config.max_alerts_per_hour:
            return True

        # Check cooldown for same alert type
        signature = alert.signature
        last_time = self._last_alert_time.get(signature)
        if last_time and (now - last_time) < self.rate_limit_config.cooldown_seconds:
            return True

        return False

    def _should_aggregate(self, alert: Alert) -> bool:
        """Check if alert should be aggregated with pending similar alerts."""
        signature = alert.signature
        if signature not in self._pending_aggregation:
            return False

        pending = self._pending_aggregation[signature]
        if not pending:
            return False

        oldest = pending[0].timestamp
        window = timedelta(seconds=self.rate_limit_config.aggregate_window_seconds)
        return datetime.utcnow() - oldest < window

    async def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        category: AlertCategory = AlertCategory.SYSTEM,
        metadata: Optional[Dict[str, Any]] = None,
        bypass_rate_limit: bool = False,
    ) -> Optional[Alert]:
        """
        Send an alert through all configured channels.

        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity level
            category: Alert category
            metadata: Additional alert metadata
            bypass_rate_limit: Skip rate limiting (for critical alerts)

        Returns:
            Alert object if sent, None if rate limited
        """
        alert = Alert(
            title=title,
            message=message,
            severity=severity,
            category=category,
            metadata=metadata or {},
        )

        # Paper mode: suppress CRITICAL alerts for reconciliation drift
        if self._paper_mode and severity == AlertSeverity.CRITICAL:
            _is_recon = any(k in title.lower() for k in ("reconcil", "drift", "divergen"))
            if _is_recon:
                logger.debug("Paper mode: suppressing CRITICAL reconciliation alert: %s", title)
                return None

        # CRITICAL alert dedup: don't fire same CRITICAL alert more than once per 5 minutes
        if severity == AlertSeverity.CRITICAL and not bypass_rate_limit:
            sig = alert.signature
            now = time.time()
            last_critical = self._last_critical_time.get(sig, 0.0)
            if (now - last_critical) < self._critical_dedup_window_s:
                logger.debug("CRITICAL alert dedup: suppressing duplicate %s (last fired %.0fs ago)", title, now - last_critical)
                return None
            self._last_critical_time[sig] = now

        # Check rate limiting
        if not bypass_rate_limit and self._is_rate_limited(alert):
            self._rate_limited += 1
            logger.debug("Alert rate limited: %s", alert.title)

            # Try to aggregate
            signature = alert.signature
            if signature not in self._pending_aggregation:
                self._pending_aggregation[signature] = []
            self._pending_aggregation[signature].append(alert)
            self._aggregated += 1
            return None

        # Record timestamp
        now = time.time()
        self._recent_timestamps.append(now)
        self._last_alert_time[alert.signature] = now
        self._total_alerts += 1

        # Check for aggregated alerts
        signature = alert.signature
        aggregated_count = 0
        if signature in self._pending_aggregation:
            aggregated_count = len(self._pending_aggregation[signature])
            if aggregated_count > 0:
                alert.message = f"{alert.message}\n\n(+{aggregated_count} similar alerts aggregated)"
                alert.metadata["aggregated_count"] = aggregated_count
            self._pending_aggregation[signature] = []

        # Send through all channels
        for channel in self.channels.values():
            if channel.should_send(alert):
                try:
                    alert.delivery_attempts += 1
                    success = await channel.send(alert)
                    if success:
                        alert.delivered_channels.append(channel.name)
                except Exception as e:
                    logger.error("Channel %s failed: %s", channel.name, e)

        # Store in history
        self._alert_history.append(alert)

        return alert

    async def send_with_escalation(
        self,
        message: str,
        severity: AlertSeverity,
        title: str = "ARGUS Alert",
        category: AlertCategory = AlertCategory.SYSTEM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Alert]:
        """
        Send an alert with automatic channel escalation.

        Tries channels in priority order: Discord → Telegram → Twilio SMS.
        Escalation only occurs for severity >= WARNING.  For INFO / DEBUG the
        method behaves like a normal ``send_alert`` call (all eligible channels).

        Escalation logic
        ----------------
        1. Try Discord (``DiscordWebhookChannel``).  If delivered → done.
        2. If Discord fails AND severity >= WARNING → try Telegram
           (``TelegramChannel``).  If delivered → done.
        3. If Telegram also fails AND severity >= WARNING → try Twilio SMS
           (``TwilioSMSChannel``).

        For non-escalated severities (DEBUG / INFO) every eligible channel is
        attempted in parallel (same behaviour as ``send_alert``).

        Returns the Alert object regardless of delivery outcome so callers can
        inspect ``alert.delivered_channels``.
        """
        alert = Alert(
            title=title,
            message=message,
            severity=severity,
            category=category,
            metadata=metadata or {},
        )

        _WARNING_LEVEL = AlertSeverity.WARNING.level
        should_escalate = severity.level >= _WARNING_LEVEL

        if not should_escalate:
            # INFO / DEBUG — use normal multi-channel delivery
            return await self.send_alert(
                title=title,
                message=message,
                severity=severity,
                category=category,
                metadata=metadata,
            )

        # Ordered escalation chain for WARNING and above
        # Discord -> Slack -> Telegram -> Email -> PagerDuty -> Twilio SMS
        escalation_order = ["discord", "slack", "telegram", "email", "pagerduty", "twilio_sms"]

        for channel_name in escalation_order:
            channel = self.channels.get(channel_name)
            if channel is None:
                # Channel not registered — skip to next
                logger.debug(
                    "send_with_escalation: channel '%s' not registered, skipping",
                    channel_name,
                )
                continue
            if not channel.should_send(alert):
                logger.debug(
                    "send_with_escalation: channel '%s' filtered out alert (min_severity mismatch)",
                    channel_name,
                )
                continue
            try:
                alert.delivery_attempts += 1
                success = await channel.send(alert)
            except Exception as exc:
                logger.error(
                    "send_with_escalation: channel '%s' raised: %s", channel_name, exc
                )
                success = False

            if success:
                alert.delivered_channels.append(channel_name)
                logger.debug(
                    "send_with_escalation: delivered via '%s' — stopping escalation",
                    channel_name,
                )
                break
            else:
                logger.warning(
                    "send_with_escalation: channel '%s' failed — escalating", channel_name
                )

        # Always log via the log channel as an audit trail
        log_channel = self.channels.get("log")
        if log_channel is not None and "log" not in alert.delivered_channels:
            try:
                await log_channel.send(alert)
                alert.delivered_channels.append("log")
            except Exception as _e:
                logger.debug("alerting error: %s", _e)

        self._alert_history.append(alert)
        self._total_alerts += 1
        return alert

    # Convenience methods for common alerts
    async def trade_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        **metadata: Any,
    ) -> Optional[Alert]:
        """Send a trade-related alert."""
        return await self.send_alert(
            title=title,
            message=message,
            severity=severity,
            category=AlertCategory.TRADE,
            metadata=metadata,
        )

    async def risk_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        **metadata: Any,
    ) -> Optional[Alert]:
        """Send a risk-related alert."""
        return await self.send_alert(
            title=title,
            message=message,
            severity=severity,
            category=AlertCategory.RISK,
            metadata=metadata,
            # Risk alerts bypass rate limiting
            bypass_rate_limit=severity.level >= AlertSeverity.ERROR.level,
        )

    async def system_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        **metadata: Any,
    ) -> Optional[Alert]:
        """Send a system-related alert."""
        return await self.send_alert(
            title=title,
            message=message,
            severity=severity,
            category=AlertCategory.SYSTEM,
            metadata=metadata,
        )

    async def drawdown_alert(
        self,
        current_drawdown: float,
        max_drawdown: float,
        threshold: float = 0.10,
    ) -> Optional[Alert]:
        """Send a drawdown threshold breach alert."""
        if current_drawdown < threshold:
            return None

        severity = AlertSeverity.WARNING
        if current_drawdown >= 0.15:
            severity = AlertSeverity.ERROR
        if current_drawdown >= 0.20:
            severity = AlertSeverity.CRITICAL

        return await self.risk_alert(
            title="Drawdown Threshold Breach",
            message=f"Current drawdown of {current_drawdown:.2%} exceeds threshold of {threshold:.2%}",
            severity=severity,
            current_drawdown=f"{current_drawdown:.2%}",
            max_drawdown=f"{max_drawdown:.2%}",
            threshold=f"{threshold:.2%}",
        )

    async def daily_loss_alert(
        self,
        daily_loss: float,
        daily_limit: float,
    ) -> Optional[Alert]:
        """Send a daily loss limit approach/breach alert."""
        pct_of_limit = daily_loss / daily_limit if daily_limit > 0 else 0

        if pct_of_limit < 0.8:
            return None

        if pct_of_limit >= 1.0:
            severity = AlertSeverity.CRITICAL
            title = "Daily Loss Limit Breached"
        else:
            severity = AlertSeverity.WARNING
            title = "Approaching Daily Loss Limit"

        return await self.risk_alert(
            title=title,
            message=f"Daily loss of ${daily_loss:.2f} AUD is {pct_of_limit:.0%} of daily limit",
            severity=severity,
            daily_loss=f"${daily_loss:.2f}",
            daily_limit=f"${daily_limit:.2f}",
            pct_of_limit=f"{pct_of_limit:.0%}",
        )

    async def circuit_breaker_alert(
        self,
        reason: str,
        duration_minutes: int = 60,
    ) -> Optional[Alert]:
        """Send circuit breaker activation alert."""
        return await self.risk_alert(
            title="Circuit Breaker Activated",
            message=f"Trading halted: {reason}. Auto-resume in {duration_minutes} minutes.",
            severity=AlertSeverity.CRITICAL,
            reason=reason,
            duration_minutes=duration_minutes,
        )

    async def consecutive_losses_alert(
        self,
        consecutive_losses: int,
        threshold: int,
    ) -> Optional[Alert]:
        """Send alert when consecutive losses hit or exceed threshold."""
        return await self.risk_alert(
            title="Consecutive Losses Alert",
            message=f"Consecutive losses: {consecutive_losses} (threshold: {threshold}). Consider reducing size or pausing.",
            severity=AlertSeverity.WARNING if consecutive_losses >= threshold else AlertSeverity.INFO,
            consecutive_losses=consecutive_losses,
            threshold=threshold,
        )

    async def error_rate_alert(
        self,
        error_rate: float,
        threshold: float,
    ) -> Optional[Alert]:
        """Send alert when error rate exceeds threshold."""
        return await self.risk_alert(
            title="Error Rate Alert",
            message=f"Error rate {error_rate:.1%} exceeds threshold {threshold:.1%}. Check logs and stability.",
            severity=AlertSeverity.CRITICAL if error_rate >= threshold else AlertSeverity.WARNING,
            error_rate=f"{error_rate:.1%}",
            threshold=f"{threshold:.1%}",
        )

    async def position_opened(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        strategy: str = "",
    ) -> Optional[Alert]:
        """Send position opened alert."""
        value = quantity * price
        return await self.trade_alert(
            title=f"Position Opened: {symbol}",
            message=f"{side.upper()} {quantity:.6f} {symbol.split('/')[0]} @ ${price:.2f} (${value:.2f} AUD)",
            severity=AlertSeverity.INFO,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=f"${price:.2f}",
            value=f"${value:.2f}",
            strategy=strategy,
        )

    async def position_closed(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
    ) -> Optional[Alert]:
        """Send position closed alert."""
        emoji = "🟢" if pnl >= 0 else "🔴"
        severity = AlertSeverity.INFO if pnl >= 0 else AlertSeverity.WARNING

        return await self.trade_alert(
            title=f"{emoji} Position Closed: {symbol}",
            message=f"Closed {side.upper()} {symbol}: P&L ${pnl:.2f} ({pnl_pct:+.2%})",
            severity=severity,
            symbol=symbol,
            entry_price=f"${entry_price:.2f}",
            exit_price=f"${exit_price:.2f}",
            pnl=f"${pnl:.2f}",
            pnl_pct=f"{pnl_pct:+.2%}",
        )

    def get_history(
        self,
        limit: int = 100,
        category: Optional[AlertCategory] = None,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """Get alert history with optional filtering."""
        alerts = list(self._alert_history)

        if category:
            alerts = [a for a in alerts if a.category == category]
        if severity:
            alerts = [a for a in alerts if a.severity.level >= severity.level]

        return alerts[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get alert manager statistics."""
        return {
            "total_alerts": self._total_alerts,
            "rate_limited": self._rate_limited,
            "aggregated": self._aggregated,
            "history_size": len(self._alert_history),
            "channels": {name: ch.stats for name, ch in self.channels.items()},
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_alert_manager(
    discord_webhook_url: Optional[str] = None,
    telegram_bot_token: Optional[str] = None,
    telegram_chat_id: Optional[str] = None,
    rate_limit_config: Optional[RateLimitConfig] = None,
    enable_twilio: bool = False,
    enable_slack: bool = False,
    enable_pagerduty: bool = False,
    enable_email: bool = False,
    slack_webhook_url: Optional[str] = None,
    pagerduty_routing_key: Optional[str] = None,
    paper_mode: bool = False,
) -> AlertManager:
    """
    Create an AlertManager with configured channels.

    Registers channels in escalation order:
    Discord -> Slack -> Telegram -> Email -> PagerDuty -> Twilio SMS.

    Channels self-disable when their env-vars / credentials are not set.

    Args:
        discord_webhook_url:   Discord webhook URL for alerts.
        telegram_bot_token:    Telegram bot token.
        telegram_chat_id:      Telegram chat ID.
        rate_limit_config:     Rate limiting configuration.
        enable_twilio:         Register Twilio SMS channel.
        enable_slack:          Register Slack webhook channel.
        enable_pagerduty:      Register PagerDuty channel.
        enable_email:          Register email (SMTP) channel.
        slack_webhook_url:     Slack webhook URL (or use ARGUS_SLACK_WEBHOOK env).
        pagerduty_routing_key: PagerDuty routing key (or use ARGUS_PAGERDUTY_KEY env).

    Returns:
        Configured AlertManager instance.
    """
    manager = AlertManager(rate_limit_config=rate_limit_config)
    manager._paper_mode = bool(paper_mode)

    if discord_webhook_url:
        manager.add_channel(DiscordWebhookChannel(discord_webhook_url))

    if enable_slack:
        manager.add_channel(SlackChannel(webhook_url=slack_webhook_url))

    if telegram_bot_token and telegram_chat_id:
        manager.add_channel(TelegramChannel(telegram_bot_token, telegram_chat_id))

    if enable_email:
        manager.add_channel(EmailChannel())

    if enable_pagerduty:
        manager.add_channel(PagerDutyChannel(routing_key=pagerduty_routing_key))

    if enable_twilio:
        manager.add_channel(TwilioSMSChannel())

    return manager


# =============================================================================
# Backward Compatibility
# =============================================================================

class Alerting:
    """Legacy compatibility wrapper."""

    def __init__(self) -> None:
        self._manager = AlertManager()

    def monitor(self) -> Dict[str, Any]:
        """Legacy monitor method."""
        return {"status": "healthy", "stats": self._manager.get_stats()}


# Global instance for simple usage
_default_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get or create the default alert manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = AlertManager()
    return _default_manager


async def send_alert(
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.INFO,
    category: AlertCategory = AlertCategory.SYSTEM,
    **metadata: Any,
) -> Optional[Alert]:
    """Convenience function to send alert via default manager."""
    return await get_alert_manager().send_alert(
        title=title,
        message=message,
        severity=severity,
        category=category,
        metadata=metadata,
    )
