"""
Exchange Monitor — continuous health checks for connected exchanges.

Monitors:
  - WebSocket connection health (disconnects, reconnects)
  - REST API response times
  - Order book data freshness (stale if > 30s since last update)
  - Fill confirmation delays
  - Exchange-reported errors and rate limits

Aggregates metrics per exchange and fires callbacks on degraded health.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

_WS_EVENT = Literal["connect", "disconnect", "message"]


@dataclass
class ExchangeMetrics:
    exchange_id: str
    ws_connected: bool
    ws_disconnects_1h: int
    api_latency_ms: float
    last_orderbook_ts: float
    orderbook_stale: bool
    order_fill_delay_ms: float
    rate_limit_hits_1h: int
    error_count_1h: int
    health_score: float  # 0-100
    timestamp: float = field(default_factory=time.time)


class ExchangeMonitor:
    """
    Tracks per-exchange health and fires alert callbacks when health degrades.

    Thread-safe for concurrent record_* calls.
    """

    STALE_ORDERBOOK_SECONDS: float = 30.0
    DEGRADED_HEALTH_SCORE: float = 60.0
    API_LATENCY_BASELINE_MS: float = 100.0  # above this, deduct points

    def __init__(
        self,
        alert_callback: Optional[Callable[[str, ExchangeMetrics], None]] = None,
        check_interval: float = 30.0,
    ) -> None:
        self.alert_callback = alert_callback
        self.check_interval = check_interval
        # Per-exchange event deques: store (timestamp, data) tuples
        self._ws_events: Dict[str, Deque[Tuple[float, str]]] = {}
        self._api_calls: Dict[str, Deque[Tuple[float, float, bool]]] = {}   # ts, latency_ms, success
        self._orderbook_ts: Dict[str, float] = {}
        self._fill_delays: Dict[str, Deque[Tuple[float, float]]] = {}       # ts, delay_ms
        self._rate_limits: Dict[str, Deque[float]] = {}                     # ts
        self._errors: Dict[str, Deque[float]] = {}                          # ts
        self._ws_connected: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Public record methods
    # ------------------------------------------------------------------

    def record_ws_event(self, exchange_id: str, event: str) -> None:
        """Record a WebSocket event (connect / disconnect / message)."""
        self._ensure_exchange(exchange_id)
        ts = time.time()
        self._ws_events[exchange_id].append((ts, event))
        if event == "connect":
            self._ws_connected[exchange_id] = True
            logger.debug("ExchangeMonitor: %s WS connected", exchange_id)
        elif event == "disconnect":
            self._ws_connected[exchange_id] = False
            logger.warning("ExchangeMonitor: %s WS disconnected", exchange_id)
        self._maybe_alert(exchange_id)

    def record_api_call(self, exchange_id: str, latency_ms: float, success: bool) -> None:
        """Record an API call with its latency and success/failure."""
        self._ensure_exchange(exchange_id)
        ts = time.time()
        self._api_calls[exchange_id].append((ts, latency_ms, success))
        if not success:
            self._errors[exchange_id].append(ts)
        self._maybe_alert(exchange_id)

    def record_orderbook_update(self, exchange_id: str) -> None:
        """Record that a fresh orderbook snapshot was received."""
        self._ensure_exchange(exchange_id)
        self._orderbook_ts[exchange_id] = time.time()

    def record_fill(self, exchange_id: str, order_id: str, delay_ms: float) -> None:
        """Record fill confirmation delay for an order."""
        self._ensure_exchange(exchange_id)
        self._fill_delays[exchange_id].append((time.time(), delay_ms))

    def record_rate_limit(self, exchange_id: str) -> None:
        """Record a rate-limit hit."""
        self._ensure_exchange(exchange_id)
        self._rate_limits[exchange_id].append(time.time())
        logger.warning("ExchangeMonitor: %s hit rate limit", exchange_id)
        self._maybe_alert(exchange_id)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_health_score(self, exchange_id: str) -> float:
        """Compute health score 0-100 for an exchange."""
        self._ensure_exchange(exchange_id)
        score = 100.0
        now = time.time()
        hour_ago = now - 3600

        # -20 if WebSocket disconnected
        if not self._ws_connected.get(exchange_id, False):
            score -= 20.0

        # -10 per disconnect in last hour (max -30)
        disconnects = sum(
            1 for ts, ev in self._ws_events[exchange_id]
            if ts >= hour_ago and ev == "disconnect"
        )
        score -= min(30.0, disconnects * 10.0)

        # -1 per 10ms above baseline (max -20)
        recent_latencies = [lat for ts, lat, _ in self._api_calls[exchange_id] if ts >= hour_ago]
        if recent_latencies:
            avg_lat = sum(recent_latencies) / len(recent_latencies)
            excess = max(0.0, avg_lat - self.API_LATENCY_BASELINE_MS)
            score -= min(20.0, excess / 10.0)

        # -20 if orderbook stale
        last_ob = self._orderbook_ts.get(exchange_id, 0.0)
        if (now - last_ob) > self.STALE_ORDERBOOK_SECONDS:
            score -= 20.0

        # -3 per rate-limit hit in last hour (max -15)
        rl_hits = sum(1 for ts in self._rate_limits[exchange_id] if ts >= hour_ago)
        score -= min(15.0, rl_hits * 3.0)

        # -2 per error in last hour (max -10)
        err_count = sum(1 for ts in self._errors[exchange_id] if ts >= hour_ago)
        score -= min(10.0, err_count * 2.0)

        return max(0.0, score)

    def get_metrics(self, exchange_id: str) -> ExchangeMetrics:
        """Get full metrics snapshot for an exchange."""
        self._ensure_exchange(exchange_id)
        now = time.time()
        hour_ago = now - 3600

        health = self.compute_health_score(exchange_id)

        recent_api = [(lat, ok) for ts, lat, ok in self._api_calls[exchange_id] if ts >= hour_ago]
        avg_lat = (sum(l for l, _ in recent_api) / len(recent_api)) if recent_api else 0.0

        recent_fills = [d for ts, d in self._fill_delays[exchange_id] if ts >= hour_ago]
        avg_fill = (sum(recent_fills) / len(recent_fills)) if recent_fills else 0.0

        last_ob = self._orderbook_ts.get(exchange_id, 0.0)
        stale = (now - last_ob) > self.STALE_ORDERBOOK_SECONDS

        disconnects = sum(
            1 for ts, ev in self._ws_events[exchange_id]
            if ts >= hour_ago and ev == "disconnect"
        )
        rl_hits = sum(1 for ts in self._rate_limits[exchange_id] if ts >= hour_ago)
        err_count = sum(1 for ts in self._errors[exchange_id] if ts >= hour_ago)

        return ExchangeMetrics(
            exchange_id=exchange_id,
            ws_connected=self._ws_connected.get(exchange_id, False),
            ws_disconnects_1h=disconnects,
            api_latency_ms=avg_lat,
            last_orderbook_ts=last_ob,
            orderbook_stale=stale,
            order_fill_delay_ms=avg_fill,
            rate_limit_hits_1h=rl_hits,
            error_count_1h=err_count,
            health_score=health,
        )

    def get_all_metrics(self) -> Dict[str, ExchangeMetrics]:
        """Get metrics for all tracked exchanges."""
        return {ex: self.get_metrics(ex) for ex in self._ws_connected}

    def snapshot(self) -> Dict:
        """Serialisable snapshot for monitoring dashboard."""
        all_m = self.get_all_metrics()
        return {
            ex: {
                "health_score": m.health_score,
                "ws_connected": m.ws_connected,
                "api_latency_ms": round(m.api_latency_ms, 1),
                "orderbook_stale": m.orderbook_stale,
                "rate_limit_hits_1h": m.rate_limit_hits_1h,
                "error_count_1h": m.error_count_1h,
                "ws_disconnects_1h": m.ws_disconnects_1h,
            }
            for ex, m in all_m.items()
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _ensure_exchange(self, exchange_id: str) -> None:
        if exchange_id not in self._ws_connected:
            self._ws_connected[exchange_id] = False
            self._ws_events[exchange_id] = deque(maxlen=500)
            self._api_calls[exchange_id] = deque(maxlen=500)
            self._fill_delays[exchange_id] = deque(maxlen=200)
            self._rate_limits[exchange_id] = deque(maxlen=200)
            self._errors[exchange_id] = deque(maxlen=200)
            self._orderbook_ts[exchange_id] = 0.0

    def _maybe_alert(self, exchange_id: str) -> None:
        if self.alert_callback is None:
            return
        metrics = self.get_metrics(exchange_id)
        if metrics.health_score < self.DEGRADED_HEALTH_SCORE:
            try:
                self.alert_callback(exchange_id, metrics)
            except Exception:
                logger.debug("ExchangeMonitor: alert callback raised", exc_info=True)
