"""WebhookChannel — generic HTTP POST JSON alert delivery — Push 60.

Configuration via environment variables::

    ARGUS_WEBHOOK_URL         Target URL (required)
    ARGUS_WEBHOOK_SECRET      HMAC-SHA256 signing secret (optional)
    ARGUS_WEBHOOK_MIN_LEVEL   Minimum level (default WARNING)
    ARGUS_WEBHOOK_ENABLED     1/true/yes (default true)

The request includes header::

    X-Argus-Signature: sha256=<hmac_hex>

if a secret is configured.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from typing import Optional

from core.alerts.alert_models import AlertEvent, AlertLevel
from core.alerts.base_channel import AbstractAlertChannel

logger = logging.getLogger(__name__)


class WebhookChannel(AbstractAlertChannel):
    """Delivers alerts as signed JSON POST requests."""

    def __init__(
        self,
        url: str,
        secret: str = "",
        min_level: AlertLevel = AlertLevel.WARNING,
        rate_limit_per_min: int = 60,
        enabled: bool = True,
        timeout: float = 10.0,
    ) -> None:
        super().__init__("webhook", min_level, rate_limit_per_min, enabled)
        self._url = url
        self._secret = secret
        self._timeout = timeout

    @classmethod
    def from_env(cls) -> "WebhookChannel":
        return cls(
            url=os.getenv("ARGUS_WEBHOOK_URL", ""),
            secret=os.getenv("ARGUS_WEBHOOK_SECRET", ""),
            min_level=AlertLevel[os.getenv("ARGUS_WEBHOOK_MIN_LEVEL", "WARNING").upper()],
            enabled=os.getenv("ARGUS_WEBHOOK_ENABLED", "true").lower() in {"1", "true", "yes"},
        )

    async def _deliver(self, event: AlertEvent) -> bool:
        if not self._url:
            logger.warning("WebhookChannel: missing URL")
            return False
        try:
            import aiohttp
        except ImportError:
            logger.warning("WebhookChannel: aiohttp not installed")
            return False

        payload = event.to_dict()
        body = json.dumps(payload)
        headers = {"Content-Type": "application/json"}
        if self._secret:
            sig = hmac.new(
                self._secret.encode(),
                body.encode(),
                hashlib.sha256,
            ).hexdigest()
            headers["X-Argus-Signature"] = f"sha256={sig}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._url,
                    data=body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as resp:
                    if resp.status < 400:
                        return True
                    logger.warning("WebhookChannel: HTTP %d", resp.status)
                    return False
        except Exception as exc:
            logger.error("WebhookChannel: error: %s", exc)
            return False
