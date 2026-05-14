"""
Twilio Alert — SMS and phone call alerts for critical trading events.

Used for: circuit breaker trips, critical liquidation risk, daily loss limit breach,
system crash detection. More reliable than Discord/Telegram for critical alerts.

Requires: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER
Disabled automatically when credentials not set — no crash.

Rate limiting: max 1 SMS per 60s per level to avoid spam.
Uses urllib only — no requests or twilio SDK dependency.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Twilio REST API base
_TWILIO_API_BASE = "https://api.twilio.com/2010-04-01/Accounts/{account_sid}"


class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


@dataclass
class AlertRecord:
    """Record of a single sent alert."""
    level: AlertLevel
    message: str
    sent_ts: float
    delivery_type: str  # "SMS" | "CALL" | "BOTH"
    success: bool
    error: str = ""


class TwilioAlert:
    """
    Twilio-backed SMS and voice-call alert system for ARGUS critical events.

    Reads credentials from environment variables:
        TWILIO_ACCOUNT_SID   — Twilio Account SID
        TWILIO_AUTH_TOKEN    — Twilio Auth Token
        TWILIO_FROM_NUMBER   — Twilio phone number (E.164, e.g. +12025551234)
        TWILIO_TO_NUMBER     — Destination phone number (E.164)

    When credentials are absent the class silently no-ops on every call.
    Rate limiting: at most one alert per level per 60 seconds.
    """

    _RATE_LIMIT_SECONDS = 60

    def __init__(self) -> None:
        self._account_sid: Optional[str] = os.environ.get("TWILIO_ACCOUNT_SID")
        self._auth_token: Optional[str] = os.environ.get("TWILIO_AUTH_TOKEN")
        self._from_number: Optional[str] = os.environ.get("TWILIO_FROM_NUMBER")
        self._to_number: Optional[str] = os.environ.get("TWILIO_TO_NUMBER")

        self._history: List[AlertRecord] = []
        self._lock = threading.Lock()
        # last_sent: level → timestamp of last successful send
        self._last_sent: Dict[AlertLevel, float] = {}

        if not self.is_configured():
            logger.debug("Twilio not configured — alerts disabled")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """Return True if all four Twilio credentials are present."""
        return all([
            self._account_sid,
            self._auth_token,
            self._from_number,
            self._to_number,
        ])

    def send_sms(self, message: str, level: AlertLevel = AlertLevel.WARNING) -> bool:
        """
        Send an SMS via Twilio Messages API.

        Rate-limited to one message per level per 60 seconds.
        Returns True on success, False on failure or when not configured.
        """
        if not self.is_configured():
            logger.debug("Twilio not configured — skipping SMS")
            return False

        if self._is_rate_limited(level):
            logger.debug(
                "Twilio SMS rate-limited for level %s — skipping", level.value
            )
            return False

        truncated = message[:1600]  # Twilio SMS limit ~1600 chars
        body_text = f"[ARGUS {level.value}] {truncated}"

        data = {
            "From": self._from_number,
            "To": self._to_number,
            "Body": body_text,
        }
        endpoint = f"/Messages.json"
        success = self._twilio_request(endpoint, data)

        record = AlertRecord(
            level=level,
            message=message,
            sent_ts=time.time(),
            delivery_type="SMS",
            success=success,
            error="" if success else "request failed",
        )
        with self._lock:
            self._history.append(record)
            if success:
                self._last_sent[level] = time.time()

        if success:
            logger.info("Twilio SMS sent | level=%s", level.value)
        else:
            logger.warning("Twilio SMS failed | level=%s", level.value)

        return success

    def make_call(self, message: str) -> bool:
        """
        Initiate a Twilio voice call that reads out `message` via TTS.

        Uses TwiML <Say> verb hosted inline via the twiml parameter.
        Returns True on success, False on failure or when not configured.
        """
        if not self.is_configured():
            logger.debug("Twilio not configured — skipping call")
            return False

        # TwiML that speaks the message once, then hangs up
        safe_msg = message.replace("&", "and").replace("<", "").replace(">", "")[:500]
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            f"<Say voice=\"alice\">[ARGUS EMERGENCY] {safe_msg}</Say>"
            "<Say voice=\"alice\">This is an automated alert from ARGUS trading system.</Say>"
            "</Response>"
        )

        data = {
            "From": self._from_number,
            "To": self._to_number,
            "Twiml": twiml,
        }
        endpoint = "/Calls.json"
        success = self._twilio_request(endpoint, data)

        record = AlertRecord(
            level=AlertLevel.EMERGENCY,
            message=message,
            sent_ts=time.time(),
            delivery_type="CALL",
            success=success,
            error="" if success else "request failed",
        )
        with self._lock:
            self._history.append(record)

        if success:
            logger.info("Twilio voice call initiated")
        else:
            logger.warning("Twilio voice call failed")

        return success

    def alert(self, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        """
        Route an alert based on severity:
            EMERGENCY → voice call + SMS (BOTH)
            CRITICAL  → SMS
            WARNING   → SMS
            INFO      → SMS (if configured, otherwise silent)

        Returns True if at least one delivery succeeded.
        """
        if not self.is_configured():
            logger.debug("Twilio not configured — skipping alert level=%s", level.value)
            return False

        if level == AlertLevel.EMERGENCY:
            sms_ok = self.send_sms(message, level)
            call_ok = self.make_call(message)
            success = sms_ok or call_ok

            # Update the last record to show BOTH
            with self._lock:
                if self._history:
                    last = self._history[-1]
                    # Mark the most recent record with BOTH if both were attempted
                    last.delivery_type = "BOTH"
                    last.success = success

            return success

        elif level == AlertLevel.CRITICAL:
            return self.send_sms(message, level)

        else:  # WARNING, INFO
            return self.send_sms(message, level)

    def get_history(self) -> List[AlertRecord]:
        """Return a copy of the alert history."""
        with self._lock:
            return list(self._history)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _twilio_request(self, endpoint: str, data: Dict) -> bool:
        """
        POST to the Twilio REST API using urllib (no SDK required).

        Authentication: HTTP Basic Auth with base64(account_sid:auth_token).
        Returns True on HTTP 2xx, False otherwise.
        """
        if not self._account_sid or not self._auth_token:
            return False

        base_url = _TWILIO_API_BASE.format(account_sid=self._account_sid)
        url = base_url + endpoint

        payload = urllib.parse.urlencode(data).encode("utf-8")

        # Build Basic Auth header
        credentials = f"{self._account_sid}:{self._auth_token}"
        encoded = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
        headers = {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
                if 200 <= status < 300:
                    return True
                body = resp.read().decode("utf-8", errors="replace")
                logger.warning(
                    "Twilio API returned HTTP %d: %.200s", status, body
                )
                return False
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")
                err_data = json.loads(body)
                logger.warning(
                    "Twilio HTTP %d | code=%s msg=%s",
                    exc.code,
                    err_data.get("code", "?"),
                    err_data.get("message", "?"),
                )
            except Exception:
                logger.warning("Twilio HTTP error %d", exc.code)
            return False
        except OSError as exc:
            logger.warning("Twilio network error: %s", exc)
            return False

    def _is_rate_limited(self, level: AlertLevel) -> bool:
        """Return True if the last send for this level was within the rate limit window."""
        with self._lock:
            last = self._last_sent.get(level)
        if last is None:
            return False
        return (time.time() - last) < self._RATE_LIMIT_SECONDS
