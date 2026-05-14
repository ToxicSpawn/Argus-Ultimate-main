"""
IncidentReporter — Discord webhook + email alerting on emergency stop.

Triggers on:
  - Emergency stop events
  - Circuit-breaker trips
  - Any custom severity=CRITICAL incident

Channels
--------
  1. Discord webhook  (DISCORD_WEBHOOK_URL env var or constructor param)
  2. SMTP email       (SMTP_* env vars or constructor params)

Usage
-----
    reporter = IncidentReporter(
        discord_webhook_url="https://discord.com/api/webhooks/...",
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        smtp_user="bot@example.com",
        smtp_password="secret",
        email_recipients=["trader@example.com"],
    )
    asyncio.run(reporter.emergency_stop("Max drawdown breached", context={"dd": -0.12}))
"""

from __future__ import annotations

import asyncio
import logging
import os
import smtplib
import time
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


# Discord embed colours
_COLOUR: Dict[Severity, int] = {
    Severity.INFO:     0x3498DB,   # blue
    Severity.WARNING:  0xF39C12,   # orange
    Severity.CRITICAL: 0xE74C3C,   # red
}


@dataclass
class Incident:
    title: str
    message: str
    severity: Severity = Severity.CRITICAL
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def context_str(self) -> str:
        if not self.context:
            return ""
        return "\n".join(f"  {k}: {v}" for k, v in self.context.items())

    def discord_embed(self) -> dict:
        fields = [
            {"name": k, "value": str(v), "inline": True}
            for k, v in self.context.items()
        ]
        return {
            "embeds": [{
                "title": f"[{self.severity.value}] {self.title}",
                "description": self.message,
                "color": _COLOUR[self.severity],
                "fields": fields,
                "footer": {"text": f"Argus Ultimate • {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(self.timestamp))}"},
            }]
        }

    def email_body(self) -> str:
        ctx = self.context_str()
        body = (
            f"Severity : {self.severity.value}\n"
            f"Title    : {self.title}\n"
            f"Time     : {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(self.timestamp))}\n\n"
            f"{self.message}\n"
        )
        if ctx:
            body += f"\nContext:\n{ctx}\n"
        return body


class IncidentReporter:
    """
    Sends incident alerts to Discord and/or email.

    All constructor params fall back to environment variables if not supplied.

    Env vars
    --------
    DISCORD_WEBHOOK_URL
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD
    ALERT_EMAIL_RECIPIENTS  (comma-separated)
    ALERT_EMAIL_FROM
    """

    def __init__(
        self,
        discord_webhook_url: Optional[str] = None,
        smtp_host: Optional[str] = None,
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        email_recipients: Optional[List[str]] = None,
        email_from: Optional[str] = None,
        timeout_sec: int = 10,
    ) -> None:
        self._webhook = discord_webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        self._smtp_host = smtp_host or os.getenv("SMTP_HOST")
        self._smtp_port = int(os.getenv("SMTP_PORT", smtp_port))
        self._smtp_user = smtp_user or os.getenv("SMTP_USER")
        self._smtp_pass = smtp_password or os.getenv("SMTP_PASSWORD")
        self._recipients = email_recipients or [
            r.strip() for r in os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(",") if r.strip()
        ]
        self._from = email_from or os.getenv("ALERT_EMAIL_FROM", self._smtp_user or "argus@localhost")
        self._timeout = timeout_sec
        self._history: List[Incident] = []

    # ------------------------------------------------------------------
    # High-level convenience methods
    # ------------------------------------------------------------------

    async def emergency_stop(self, reason: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Fire a CRITICAL incident for an emergency stop event."""
        incident = Incident(
            title="🚨 EMERGENCY STOP TRIGGERED",
            message=reason,
            severity=Severity.CRITICAL,
            context=context or {},
        )
        await self.report(incident)

    async def circuit_breaker(self, reason: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Fire a CRITICAL incident for a circuit-breaker trip."""
        incident = Incident(
            title="⛔ CIRCUIT BREAKER TRIPPED",
            message=reason,
            severity=Severity.CRITICAL,
            context=context or {},
        )
        await self.report(incident)

    async def warn(self, title: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        incident = Incident(
            title=title,
            message=message,
            severity=Severity.WARNING,
            context=context or {},
        )
        await self.report(incident)

    # ------------------------------------------------------------------
    # Core dispatcher
    # ------------------------------------------------------------------

    async def report(self, incident: Incident) -> None:
        """Dispatch incident to all configured channels concurrently."""
        self._history.append(incident)
        logger.warning(
            "[IncidentReporter] %s | %s | %s",
            incident.severity.value, incident.title, incident.message,
        )
        tasks = []
        if self._webhook:
            tasks.append(self._send_discord(incident))
        if self._smtp_host and self._recipients:
            tasks.append(self._send_email(incident))
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.error("IncidentReporter channel error: %s", r)
        else:
            logger.warning("IncidentReporter: no channels configured — incident logged only")

    # ------------------------------------------------------------------
    # Discord
    # ------------------------------------------------------------------

    async def _send_discord(self, incident: Incident) -> None:
        if not _AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not installed — Discord alert skipped")
            return
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self._webhook,  # type: ignore[arg-type]
                json=incident.discord_embed(),
            ) as resp:
                if resp.status not in (200, 204):
                    text = await resp.text()
                    raise RuntimeError(f"Discord webhook {resp.status}: {text}")
        logger.info("IncidentReporter: Discord alert sent [%s]", incident.title)

    # ------------------------------------------------------------------
    # Email (SMTP / STARTTLS)
    # ------------------------------------------------------------------

    async def _send_email(self, incident: Incident) -> None:
        """Send via SMTP in a thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._send_email_sync, incident)

    def _send_email_sync(self, incident: Incident) -> None:
        subject = f"[{incident.severity.value}] Argus Ultimate — {incident.title}"
        body = incident.email_body()
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = self._from
        msg["To"] = ", ".join(self._recipients)

        try:
            with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=self._timeout) as smtp:  # type: ignore[arg-type]
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
                if self._smtp_user and self._smtp_pass:
                    smtp.login(self._smtp_user, self._smtp_pass)
                smtp.sendmail(self._from, self._recipients, msg.as_string())
            logger.info("IncidentReporter: email sent to %s", self._recipients)
        except smtplib.SMTPException as exc:
            raise RuntimeError(f"SMTP error: {exc}") from exc

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def recent_incidents(self, n: int = 20) -> List[Incident]:
        return self._history[-n:]

    def critical_count(self) -> int:
        return sum(1 for i in self._history if i.severity == Severity.CRITICAL)
