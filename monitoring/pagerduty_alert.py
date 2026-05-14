"""
PagerDuty Integration -- triggers and resolves incidents via Events API v2.

Config: ARGUS_PAGERDUTY_KEY env var (routing/integration key), or pass directly.
Gracefully disabled if no routing key configured.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

_EVENTS_URL = "https://events.pagerduty.com/v2/enqueue"

_VALID_SEVERITIES = {"critical", "error", "warning", "info"}


class PagerDutyAlert:
    """Async PagerDuty Events API v2 client for ARGUS alerts."""

    def __init__(self, routing_key: Optional[str] = None) -> None:
        self.routing_key = routing_key or os.environ.get("ARGUS_PAGERDUTY_KEY")

    @property
    def is_configured(self) -> bool:
        return bool(self.routing_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def trigger(
        self,
        summary: str,
        severity: str = "critical",
        source: str = "argus",
        component: Optional[str] = None,
    ) -> str:
        """Trigger a PagerDuty incident.

        Args:
            summary:   Human-readable incident description.
            severity:  One of ``critical``, ``error``, ``warning``, ``info``.
            source:    Affected system / hostname.
            component: Sub-component (e.g. ``risk_manager``).

        Returns:
            The ``dedup_key`` used — callers can pass it to :meth:`resolve`.
        """
        severity = severity.lower()
        if severity not in _VALID_SEVERITIES:
            severity = "critical"

        dedup_key = uuid.uuid4().hex[:32]

        payload = {
            "routing_key": self.routing_key,
            "event_action": "trigger",
            "dedup_key": dedup_key,
            "payload": {
                "summary": summary[:1024],
                "severity": severity,
                "source": source,
                "component": component or "argus-trading",
                "custom_details": {
                    "system": "ARGUS Crypto Trading",
                },
            },
        }

        await self._post(payload)
        return dedup_key

    async def resolve(self, dedup_key: str) -> bool:
        """Resolve a previously triggered incident.

        Args:
            dedup_key: The key returned by :meth:`trigger`.

        Returns:
            ``True`` if the resolve request was accepted.
        """
        payload = {
            "routing_key": self.routing_key,
            "event_action": "resolve",
            "dedup_key": dedup_key,
        }
        return await self._post(payload)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _post(self, payload: dict) -> bool:
        if not self.is_configured:
            logger.debug("PagerDutyAlert: routing key not set")
            return False
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, self._post_sync, payload)
        except Exception:
            logger.debug("PagerDutyAlert._post error", exc_info=True)
            return False

    def _post_sync(self, payload: dict) -> bool:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _EVENTS_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 202
        except urllib.error.HTTPError as exc:
            logger.warning("PagerDuty HTTP %d: %s", exc.code, exc.reason)
            return False
        except Exception:
            logger.debug("PagerDutyAlert._post_sync failed", exc_info=True)
            return False


# Module-level singleton
_instance: Optional[PagerDutyAlert] = None


def get_pagerduty() -> PagerDutyAlert:
    global _instance
    if _instance is None:
        _instance = PagerDutyAlert()
    return _instance
