"""M10 — Prometheus metrics exported from core/.

Provides a thin, dependency-optional wrapper.  When ``prometheus_client`` is
installed the real counters/histograms are created; otherwise a no-op shim is
used so the rest of the codebase can import unconditionally.

Usage::

    from core.metrics_exporter import METRICS

    METRICS.orders_placed.inc()
    with METRICS.execution_latency.time():
        ...
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── prometheus_client availability ───────────────────────────────────────────
try:
    from prometheus_client import (  # type: ignore[import-untyped]
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False


class _Noop:
    """No-op metric that silently accepts all calls."""

    def inc(self, amount: float = 1) -> None:
        pass

    def dec(self, amount: float = 1) -> None:
        pass

    def set(self, value: float) -> None:
        pass

    def observe(self, value: float) -> None:
        pass

    def labels(self, **kwargs: Any) -> "_Noop":
        return self

    def time(self) -> "_NoopCtx":
        return _NoopCtx()


class _NoopCtx:
    def __enter__(self) -> "_NoopCtx":
        return self

    def __exit__(self, *_: Any) -> None:
        pass


def _counter(name: str, doc: str, labelnames: list[str] | None = None) -> Any:
    if _PROM_AVAILABLE:
        return Counter(name, doc, labelnames or [])
    return _Noop()


def _gauge(name: str, doc: str) -> Any:
    if _PROM_AVAILABLE:
        return Gauge(name, doc)
    return _Noop()


def _histogram(name: str, doc: str, buckets: list[float] | None = None) -> Any:
    if _PROM_AVAILABLE:
        kwargs: dict[str, Any] = {}
        if buckets:
            kwargs["buckets"] = buckets
        return Histogram(name, doc, **kwargs)
    return _Noop()


@dataclass
class ArgusMetrics:
    """Central registry of all Argus Prometheus metrics."""

    # ── Trading ───────────────────────────────────────────────────────────────
    orders_placed: Any = field(
        default_factory=lambda: _counter(
            "argus_orders_placed_total",
            "Total orders placed",
            ["symbol", "side"],
        )
    )
    orders_filled: Any = field(
        default_factory=lambda: _counter(
            "argus_orders_filled_total",
            "Total orders filled",
            ["symbol"],
        )
    )
    orders_rejected: Any = field(
        default_factory=lambda: _counter(
            "argus_orders_rejected_total",
            "Orders rejected by risk checks",
            ["reason"],
        )
    )

    # ── Execution latency ─────────────────────────────────────────────────────
    execution_latency: Any = field(
        default_factory=lambda: _histogram(
            "argus_execution_latency_seconds",
            "Order execution round-trip latency",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
        )
    )
    signal_processing_latency: Any = field(
        default_factory=lambda: _histogram(
            "argus_signal_processing_latency_seconds",
            "Signal pipeline processing latency",
        )
    )

    # ── Risk ──────────────────────────────────────────────────────────────────
    risk_blocks: Any = field(
        default_factory=lambda: _counter(
            "argus_risk_blocks_total",
            "Signals blocked by risk manager",
            ["reason"],
        )
    )
    drawdown_current: Any = field(
        default_factory=lambda: _gauge(
            "argus_drawdown_current",
            "Current drawdown fraction",
        )
    )

    # ── System ────────────────────────────────────────────────────────────────
    cycle_errors: Any = field(
        default_factory=lambda: _counter(
            "argus_cycle_errors_total",
            "Unhandled exceptions in trading cycles",
            ["component"],
        )
    )
    active_positions: Any = field(
        default_factory=lambda: _gauge(
            "argus_active_positions",
            "Number of currently open positions",
        )
    )


# Module-level singleton
METRICS: ArgusMetrics = ArgusMetrics()


def start_metrics_server(port: int = 8001) -> None:
    """Start the Prometheus HTTP metrics server on *port*.

    No-op when ``prometheus_client`` is not installed.
    """
    if not _PROM_AVAILABLE:
        logger.warning("prometheus_client not installed — metrics server not started")
        return
    start_http_server(port)
    logger.info("Prometheus metrics server started on :%d", port)


def register_latency_telemetry(telemetry: Any) -> None:  # noqa: ANN401
    """
    Wire HFT latency telemetry Prometheus metrics into the METRICS singleton.

    When ``prometheus_client`` is available this function creates Gauge metrics
    for every LatencyStage p50/p95/p99/p999 percentile and a Counter for
    completed journeys, then registers a periodic callback that refreshes the
    gauge values on every ``get_stats()`` call.

    When ``prometheus_client`` is not installed the function is a no-op (the
    telemetry object keeps working — it simply won't be scraped by Prometheus).

    Parameters
    ----------
    telemetry:
        A :class:`hft_engine.latency_telemetry.LatencyTelemetry` instance
        (typed as ``Any`` to avoid a hard import dependency on the HFT module).

    Example::

        from hft_engine.latency_telemetry import LatencyTelemetry
        from core.metrics_exporter import register_latency_telemetry

        tel = LatencyTelemetry.get_instance()
        register_latency_telemetry(tel)
    """
    if not _PROM_AVAILABLE:
        logger.warning(
            "prometheus_client not installed — latency telemetry metrics not registered"
        )
        return

    try:
        from prometheus_client import Gauge as _Gauge  # type: ignore[import-untyped]

        # ── per-stage latency gauges ─────────────────────────────────────────
        _stage_latency_gauge = _Gauge(
            "argus_hft_stage_latency_us",
            "HFT pipeline stage latency in microseconds",
            ["stage", "quantile"],
        )

        # ── aggregate latency gauges ─────────────────────────────────────────
        _aggregate_latency_gauge = _Gauge(
            "argus_hft_aggregate_latency_us",
            "HFT aggregate tick-to-order/trade latency in microseconds",
            ["stage", "quantile"],
        )

        # ── completed journeys counter (gauge so it can be set directly) ────
        _journeys_gauge = _Gauge(
            "argus_hft_completed_journeys_total",
            "Total completed HFT trade journeys (cumulative)",
        )

        def _refresh() -> None:
            """Pull latest stats from telemetry and push to Prometheus gauges."""
            try:
                stats = telemetry.get_stats()
                quantile_map = {
                    "p50_us": "0.5",
                    "p95_us": "0.95",
                    "p99_us": "0.99",
                    "p999_us": "0.999",
                }
                aggregate_keys = {"TICK_TO_ORDER", "TICK_TO_TRADE"}

                for key, stage_data in stats.items():
                    if not isinstance(stage_data, dict):
                        continue
                    for pkey, qlabel in quantile_map.items():
                        val = stage_data.get(pkey, 0.0)
                        if key in aggregate_keys:
                            _aggregate_latency_gauge.labels(
                                stage=key, quantile=qlabel
                            ).set(val)
                        else:
                            _stage_latency_gauge.labels(
                                stage=key, quantile=qlabel
                            ).set(val)

                _journeys_gauge.set(stats.get("completed_journeys", 0))
            except Exception as exc:  # noqa: BLE001
                logger.warning("latency telemetry prometheus refresh failed: %s", exc)

        # Attach the refresh callable as an attribute on the telemetry object
        # so callers can invoke it manually (e.g. in a scrape endpoint hook).
        telemetry._prometheus_refresh = _refresh  # type: ignore[attr-defined]

        # Perform an initial population so metrics are non-zero from the start.
        _refresh()

        logger.info(
            "Latency telemetry Prometheus metrics registered "
            "(argus_hft_stage_latency_us, argus_hft_aggregate_latency_us, "
            "argus_hft_completed_journeys_total)"
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to register latency telemetry metrics: %s", exc, exc_info=True)
