"""
Metrics Maintenance -- Prometheus TSDB compaction and admin operations.

Requires Prometheus ``--web.enable-admin-api`` flag (already set in
docker-compose.r740.yml).

Usage:
    from ops.metrics_maintenance import compact_old_metrics
    result = compact_old_metrics("http://r740:9090")
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def compact_old_metrics(
    prometheus_url: str = "http://localhost:9090",
    days_to_keep: int = 90,
) -> Dict[str, Any]:
    """Trigger Prometheus TSDB compaction via the admin API.

    This calls ``/api/v1/admin/tsdb/clean_tombstones`` to compact
    deleted series, and ``/api/v1/admin/tsdb/snapshot`` is *not* called
    (snapshot is a separate concern).

    The ``days_to_keep`` parameter is informational — actual retention is
    controlled by ``--storage.tsdb.retention.time`` on the Prometheus
    server.  This function simply triggers compaction of tombstoned data.

    Args:
        prometheus_url: Base URL of the Prometheus server.
        days_to_keep:   Documented retention period (for logging only).

    Returns:
        Dict with ``success`` (bool), ``status_code``, and optional ``error``.
    """
    url = f"{prometheus_url.rstrip('/')}/api/v1/admin/tsdb/clean_tombstones"

    logger.info(
        "Triggering Prometheus TSDB compaction at %s (retention=%dd)",
        prometheus_url, days_to_keep,
    )

    try:
        req = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.status
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        logger.error("Prometheus compaction HTTP %d: %s", exc.code, exc.reason)
        return {"success": False, "status_code": exc.code, "error": exc.reason}
    except Exception as exc:
        logger.error("Prometheus compaction failed: %s", exc)
        return {"success": False, "status_code": 0, "error": str(exc)}

    success = 200 <= status < 300
    if success:
        logger.info("Prometheus TSDB compaction triggered (HTTP %d)", status)
    else:
        logger.warning("Prometheus compaction unexpected status: %d", status)

    return {"success": success, "status_code": status, "response": body}


def get_prometheus_status(
    prometheus_url: str = "http://localhost:9090",
) -> Dict[str, Any]:
    """Fetch Prometheus TSDB status for monitoring.

    Returns head stats, min/max time, series count, etc.
    """
    url = f"{prometheus_url.rstrip('/')}/api/v1/status/tsdb"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("data", {})
    except Exception as exc:
        logger.error("Failed to fetch Prometheus TSDB status: %s", exc)
        return {"error": str(exc)}
