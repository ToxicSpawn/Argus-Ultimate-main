"""EmailChannel — sends alerts via SMTP — Push 60.

Configuration via environment variables::

    ARGUS_EMAIL_HOST        SMTP host (default smtp.gmail.com)
    ARGUS_EMAIL_PORT        SMTP port (default 465)
    ARGUS_EMAIL_USER        SMTP username / from address
    ARGUS_EMAIL_PASSWORD    SMTP password / app password
    ARGUS_EMAIL_TO          Recipient address
    ARGUS_EMAIL_MIN_LEVEL   Minimum level (default ERROR)
    ARGUS_EMAIL_ENABLED     1/true/yes (default true)
"""
from __future__ import annotations

import asyncio
import logging
import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from core.alerts.alert_models import AlertEvent, AlertLevel
from core.alerts.base_channel import AbstractAlertChannel

logger = logging.getLogger(__name__)


class EmailChannel(AbstractAlertChannel):
    """Delivers alerts via SMTP (SSL by default)."""

    def __init__(
        self,
        host: str = "smtp.gmail.com",
        port: int = 465,
        username: str = "",
        password: str = "",
        to_address: str = "",
        min_level: AlertLevel = AlertLevel.ERROR,
        rate_limit_per_min: int = 5,
        enabled: bool = True,
        use_ssl: bool = True,
    ) -> None:
        super().__init__("email", min_level, rate_limit_per_min, enabled)
        self._host = host
        self._port = port
        self._user = username
        self._password = password
        self._to = to_address
        self._use_ssl = use_ssl

    @classmethod
    def from_env(cls) -> "EmailChannel":
        return cls(
            host=os.getenv("ARGUS_EMAIL_HOST", "smtp.gmail.com"),
            port=int(os.getenv("ARGUS_EMAIL_PORT", "465")),
            username=os.getenv("ARGUS_EMAIL_USER", ""),
            password=os.getenv("ARGUS_EMAIL_PASSWORD", ""),
            to_address=os.getenv("ARGUS_EMAIL_TO", ""),
            min_level=AlertLevel[os.getenv("ARGUS_EMAIL_MIN_LEVEL", "ERROR").upper()],
            enabled=os.getenv("ARGUS_EMAIL_ENABLED", "true").lower() in {"1", "true", "yes"},
        )

    async def _deliver(self, event: AlertEvent) -> bool:
        if not self._user or not self._to:
            logger.warning("EmailChannel: missing credentials or recipient")
            return False
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._send_sync, event)

    def _send_sync(self, event: AlertEvent) -> bool:
        subject = f"[Argus {event.level.label}] {event.title}"
        html = self._build_html(event)
        plain = event.formatted_text(include_emoji=False)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self._user
        msg["To"] = self._to
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html, "html"))

        try:
            ctx = ssl.create_default_context() if self._use_ssl else None
            if self._use_ssl:
                with smtplib.SMTP_SSL(self._host, self._port, context=ctx) as server:
                    server.login(self._user, self._password)
                    server.sendmail(self._user, self._to, msg.as_string())
            else:
                with smtplib.SMTP(self._host, self._port) as server:
                    server.starttls()
                    server.login(self._user, self._password)
                    server.sendmail(self._user, self._to, msg.as_string())
            return True
        except Exception as exc:
            logger.error("EmailChannel: SMTP error: %s", exc)
            return False

    def _build_html(self, event: AlertEvent) -> str:
        colour = {30: "#F39C12", 40: "#E74C3C", 50: "#8E44AD"}.get(event.level.value, "#3498DB")
        return f"""
        <html><body>
        <div style="font-family:monospace;border-left:4px solid {colour};padding:8px">
          <h3>{event.level.emoji} [{event.level.label}] {event.title}</h3>
          {'<p><b>Symbol:</b> ' + event.symbol + '</p>' if event.symbol else ''}
          <p>{event.body}</p>
        </div>
        </body></html>
        """
