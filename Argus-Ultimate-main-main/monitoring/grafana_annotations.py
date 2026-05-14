"""
Grafana Annotations — push trade and event annotations to Grafana dashboards.

Uses the Grafana HTTP API to create annotations that overlay on time-series
panels, giving visual context for trades, regime changes, and system events.

No external dependencies — uses urllib from the standard library.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GrafanaAnnotator:
    """Create annotations on Grafana dashboards via the HTTP API.

    Parameters
    ----------
    grafana_url : str, optional
        Base URL of the Grafana instance (e.g. ``http://r740:3000``).
        Falls back to ``ARGUS_GRAFANA_URL`` environment variable.
    api_key : str, optional
        Grafana API key or service account token.
        Falls back to ``ARGUS_GRAFANA_API_KEY`` environment variable.
    """

    def __init__(
        self,
        grafana_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.grafana_url = (
            grafana_url
            or os.environ.get("ARGUS_GRAFANA_URL", "")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("ARGUS_GRAFANA_API_KEY", "")

    def annotate_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Create a Grafana annotation for a trade event.

        Parameters
        ----------
        trade_data : dict
            Must contain at minimum ``symbol`` and ``side``.
            Optional: ``pnl``, ``price``, ``quantity``, ``strategy``.

        Returns
        -------
        bool
            True if the annotation was created successfully.
        """
        symbol = trade_data.get("symbol", "unknown")
        side = trade_data.get("side", "unknown")
        pnl = trade_data.get("pnl", 0.0)
        price = trade_data.get("price", 0.0)

        text = f"Trade: {side} {symbol} @ {price:.2f}"
        if pnl:
            text += f" | PnL: ${pnl:.2f}"

        tags = ["trade", side.lower(), symbol.replace("/", "_").lower()]
        if pnl > 0:
            tags.append("profit")
        elif pnl < 0:
            tags.append("loss")

        return self.annotate_event(text=text, tags=tags)

    def annotate_event(
        self,
        text: str,
        tags: Optional[List[str]] = None,
        dashboard_id: Optional[int] = None,
        panel_id: Optional[int] = None,
        epoch_ms: Optional[int] = None,
    ) -> bool:
        """Create a generic Grafana annotation.

        Parameters
        ----------
        text : str
            Annotation text (supports HTML).
        tags : list of str, optional
            Tags for filtering annotations.
        dashboard_id : int, optional
            Scope annotation to a specific dashboard.
        panel_id : int, optional
            Scope annotation to a specific panel.
        epoch_ms : int, optional
            Timestamp in milliseconds.  Defaults to now.

        Returns
        -------
        bool
            True if the annotation was created successfully.
        """
        if not self.grafana_url:
            logger.debug("GrafanaAnnotator: no grafana_url configured, skipping")
            return False

        if epoch_ms is None:
            epoch_ms = int(time.time() * 1000)

        payload: Dict[str, Any] = {
            "text": text,
            "time": epoch_ms,
            "tags": tags or ["argus"],
        }
        if dashboard_id is not None:
            payload["dashboardId"] = dashboard_id
        if panel_id is not None:
            payload["panelId"] = panel_id

        url = f"{self.grafana_url}/api/annotations"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
                if status in (200, 201):
                    logger.debug("Grafana annotation created: %s", text[:80])
                    return True
                logger.warning("Grafana annotation returned status %d", status)
                return False
        except urllib.error.URLError as exc:
            logger.debug("Grafana annotation failed: %s", exc)
            return False
        except Exception as exc:
            logger.debug("Grafana annotation error: %s", exc)
            return False
