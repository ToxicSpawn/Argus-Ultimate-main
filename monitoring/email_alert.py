"""
Email Alerting -- sends trade and system alerts via SMTP.

Config env vars:
  ARGUS_SMTP_HOST, ARGUS_SMTP_PORT, ARGUS_SMTP_USER, ARGUS_SMTP_PASS,
  ARGUS_SMTP_FROM, ARGUS_SMTP_TO (comma-separated list)

Gracefully disabled if SMTP host is not configured.
"""
from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EmailAlert:
    """SMTP email alert sender for ARGUS trading system."""

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        from_addr: Optional[str] = None,
        to_addrs: Optional[List[str]] = None,
    ) -> None:
        self.smtp_host = smtp_host or os.environ.get("ARGUS_SMTP_HOST")
        self.smtp_port = int(os.environ.get("ARGUS_SMTP_PORT", smtp_port))
        self.username = username or os.environ.get("ARGUS_SMTP_USER")
        self.password = password or os.environ.get("ARGUS_SMTP_PASS")
        self.from_addr = from_addr or os.environ.get("ARGUS_SMTP_FROM", "argus@localhost")
        to_env = os.environ.get("ARGUS_SMTP_TO", "")
        self.to_addrs = to_addrs or [a.strip() for a in to_env.split(",") if a.strip()]

    @property
    def is_configured(self) -> bool:
        return bool(self.smtp_host and self.to_addrs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, subject: str, body: str, html: bool = False) -> bool:
        """Send an email via SMTP with STARTTLS.

        Args:
            subject: Email subject line.
            body:    Email body (plain text or HTML).
            html:    If ``True``, body is sent as ``text/html``.

        Returns:
            ``True`` on successful delivery.
        """
        if not self.is_configured:
            logger.debug("EmailAlert: SMTP not configured")
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)
            msg["Subject"] = subject

            mime_type = "html" if html else "plain"
            msg.attach(MIMEText(body, mime_type))

            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

            logger.debug("Email sent: %s", subject)
            return True

        except Exception as exc:
            logger.error("EmailAlert.send failed: %s", exc)
            return False

    def send_daily_summary(self, summary_data: Dict[str, Any]) -> bool:
        """Send a formatted HTML daily summary email.

        ``summary_data`` keys:
          - ``date`` (str): trading date
          - ``total_pnl`` (float): total P&L in AUD
          - ``trade_count`` (int): number of trades
          - ``win_rate`` (float): win rate 0-1
          - ``max_drawdown`` (float): max drawdown 0-1
          - ``trades`` (list[dict]): individual trade rows
        """
        date = summary_data.get("date", "N/A")
        pnl = summary_data.get("total_pnl", 0.0)
        trade_count = summary_data.get("trade_count", 0)
        win_rate = summary_data.get("win_rate", 0.0)
        max_drawdown = summary_data.get("max_drawdown", 0.0)
        trades = summary_data.get("trades", [])

        pnl_color = "#36a64f" if pnl >= 0 else "#ff0000"

        trade_rows = ""
        for t in trades[:50]:  # limit rows
            t_pnl = t.get("pnl", 0)
            row_color = "#36a64f" if t_pnl >= 0 else "#ff0000"
            trade_rows += (
                f"<tr>"
                f"<td>{t.get('symbol', '')}</td>"
                f"<td>{t.get('side', '')}</td>"
                f"<td style=\"color:{row_color}\">${t_pnl:+.2f}</td>"
                f"<td>{t.get('strategy', '')}</td>"
                f"</tr>"
            )

        html = f"""
        <html><body style="font-family:Arial,sans-serif;max-width:600px;margin:auto;">
        <h2>ARGUS Daily Summary &mdash; {date}</h2>
        <table style="width:100%;border-collapse:collapse;">
          <tr><td><b>Total P&amp;L</b></td>
              <td style="color:{pnl_color};font-size:1.2em;"><b>${pnl:+.2f} AUD</b></td></tr>
          <tr><td><b>Trades</b></td><td>{trade_count}</td></tr>
          <tr><td><b>Win Rate</b></td><td>{win_rate:.1%}</td></tr>
          <tr><td><b>Max Drawdown</b></td><td>{max_drawdown:.2%}</td></tr>
        </table>
        <h3>Trade Details</h3>
        <table style="width:100%;border-collapse:collapse;border:1px solid #ddd;">
          <tr style="background:#f0f0f0;">
            <th style="padding:4px;text-align:left;">Symbol</th>
            <th style="padding:4px;text-align:left;">Side</th>
            <th style="padding:4px;text-align:left;">P&amp;L</th>
            <th style="padding:4px;text-align:left;">Strategy</th>
          </tr>
          {trade_rows}
        </table>
        <p style="color:#888;font-size:0.8em;">Generated by ARGUS Trading System</p>
        </body></html>
        """

        subject = f"ARGUS Daily Summary {date} | P&L ${pnl:+.2f}"
        return self.send(subject, html, html=True)


# Module-level singleton
_instance: Optional[EmailAlert] = None


def get_email_alert() -> EmailAlert:
    global _instance
    if _instance is None:
        _instance = EmailAlert()
    return _instance
