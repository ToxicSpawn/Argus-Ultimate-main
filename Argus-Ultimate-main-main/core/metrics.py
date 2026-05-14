"""Package: core.metrics

Prometheus counters, gauges and histograms exported from the core/ package.

All metrics use the ``argus_`` prefix for easy dashboarding in Grafana.

Usage::

    from core.metrics import (
        TRADE_COUNTER, CYCLE_COUNTER, DRAWDOWN_GAUGE,
        CYCLE_LATENCY_HISTOGRAM, record_cycle,
    )

    TRADE_COUNTER.labels(side="buy", symbol="BTC/AUD").inc()
    record_cycle(latency_ms=42.5, error=False)

Prometheus client is optional — if not installed every call is a no-op so
the rest of the system starts without it.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Null-object fallback when prometheus_client is not installed
# ---------------------------------------------------------------------------

class _NullMetric:
    """Drop-in no-op replacement for any prometheus_client metric type."""
    def labels(self, **_kwargs) -> "_NullMetric":
        return self
    def inc(self, amount: float = 1) -> None:
        pass
    def dec(self, amount: float = 1) -> None:
        pass
    def set(self, value: float) -> None:  # noqa: A003
        pass
    def observe(self, value: float) -> None:
        pass
    def time(self):
        import contextlib
        return contextlib.nullcontext()


_NULL = _NullMetric()

_PROM_AVAILABLE = False
try:
    from prometheus_client import Counter, Gauge, Histogram, REGISTRY  # type: ignore[import]
    _PROM_AVAILABLE = True
except ImportError:
    logger.debug("prometheus_client not installed — metrics are no-ops.")
    Counter = Gauge = Histogram = None  # type: ignore[assignment,misc]


def _make_counter(name: str, doc: str, labels=()) -> object:
    if not _PROM_AVAILABLE:
        return _NULL
    try:
        return Counter(name, doc, list(labels))
    except Exception:
        return _NULL


def _make_gauge(name: str, doc: str, labels=()) -> object:
    if not _PROM_AVAILABLE:
        return _NULL
    try:
        return Gauge(name, doc, list(labels))
    except Exception:
        return _NULL


def _make_histogram(name: str, doc: str, labels=(), buckets=None) -> object:
    if not _PROM_AVAILABLE:
        return _NULL
    kwargs = {"labelnames": list(labels)}
    if buckets is not None:
        kwargs["buckets"] = buckets
    try:
        return Histogram(name, doc, **kwargs)
    except Exception:
        return _NULL


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

TRADE_COUNTER = _make_counter(
    "argus_trades_total",
    "Total number of trades executed",
    labels=["side", "symbol"],
)

CYCLE_COUNTER = _make_counter(
    "argus_cycles_total",
    "Total number of completed trading cycles",
)

ERROR_COUNTER = _make_counter(
    "argus_errors_total",
    "Total number of errors across all subsystems",
    labels=["subsystem"],
)

RATE_LIMIT_COUNTER = _make_counter(
    "argus_rate_limit_hits_total",
    "Total number of exchange rate-limit hits",
    labels=["exchange"],
)

DRAWDOWN_GAUGE = _make_gauge(
    "argus_drawdown_pct",
    "Current portfolio drawdown percentage",
)

CAPITAL_GAUGE = _make_gauge(
    "argus_capital_aud",
    "Current portfolio capital in AUD",
)

OPEN_POSITIONS_GAUGE = _make_gauge(
    "argus_open_positions",
    "Number of currently open positions",
)

CYCLE_LATENCY_HISTOGRAM = _make_histogram(
    "argus_cycle_latency_ms",
    "Trading cycle wall-clock latency in milliseconds",
    buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
)

ORDER_LATENCY_HISTOGRAM = _make_histogram(
    "argus_order_latency_ms",
    "Order placement latency in milliseconds",
    labels=["exchange"],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def record_cycle(latency_ms: float, *, error: bool = False) -> None:
    """Increment cycle counter, record latency histogram, optionally bump error counter."""
    CYCLE_COUNTER.inc()
    CYCLE_LATENCY_HISTOGRAM.observe(latency_ms)
    if error:
        ERROR_COUNTER.labels(subsystem="trading_cycle").inc()


def get_prometheus_text() -> str:
    """Return Prometheus text-format string of all registered metrics."""
    if not _PROM_AVAILABLE:
        return "# prometheus_client not installed\n"
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST  # noqa: F401
        return generate_latest(REGISTRY).decode("utf-8")
    except Exception as exc:
        return f"# ERROR: {exc}\n"
