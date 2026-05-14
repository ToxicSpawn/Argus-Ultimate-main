"""Push 79 — PrometheusRegistry: pure-stdlib Prometheus text format.

Implements a minimal Prometheus exposition format without
dependency on prometheus_client.

Metrics:
  Gauges:
    argus_equity                  Current portfolio equity
    argus_portfolio_heat          Portfolio heat ratio [0,1]
    argus_open_positions          Number of open positions
    argus_kill_switch             1 if kill switch active
    argus_margin_ratio            Current margin utilisation

  Counters:
    argus_signals_total           Total signals emitted
    argus_orders_total            Total orders submitted
    argus_fills_total             Total fills processed
    argus_risk_events_total{type} Risk events by type

  Histograms:
    argus_order_latency_us        Order round-trip latency µs
    argus_signal_to_order_us      Signal → order latency µs

text_exposition() returns full Prometheus text format string.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class _Gauge:
    name: str
    help: str
    value: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)

    def set(self, v: float) -> None:
        self.value = v

    def inc(self, v: float = 1.0) -> None:
        self.value += v


@dataclass
class _Counter:
    name: str
    help: str
    _values: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def inc(self, labels: Optional[Dict[str, str]] = None, v: float = 1.0) -> None:
        key = _label_str(labels or {})
        self._values[key] += v

    def total(self, labels: Optional[Dict[str, str]] = None) -> float:
        return self._values[_label_str(labels or {})]


@dataclass
class _Histogram:
    name: str
    help: str
    buckets: List[float] = field(
        default_factory=lambda: [10, 50, 100, 250, 500, 1000, 2500, 5000, float("inf")]
    )
    _counts: List[int] = field(default_factory=list)
    _sum: float = 0.0
    _total: int = 0

    def __post_init__(self):
        self._counts = [0] * len(self.buckets)

    def observe(self, value: float) -> None:
        self._sum   += value
        self._total += 1
        for i, b in enumerate(self.buckets):
            if value <= b:
                self._counts[i] += 1


def _label_str(labels: Dict[str, str]) -> str:
    if not labels:
        return ""
    parts = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return "{" + parts + "}"


class PrometheusRegistry:
    """Minimal Prometheus metrics registry."""

    def __init__(self):
        self._start = time.time()

        # Gauges
        self.equity          = _Gauge("argus_equity",         "Current portfolio equity USD")
        self.heat            = _Gauge("argus_portfolio_heat",  "Portfolio heat ratio")
        self.open_positions  = _Gauge("argus_open_positions",  "Number of open positions")
        self.kill_switch     = _Gauge("argus_kill_switch",     "1 if kill switch active")
        self.margin_ratio    = _Gauge("argus_margin_ratio",    "Margin utilisation ratio")

        # Counters
        self.signals_total   = _Counter("argus_signals_total",      "Total signals emitted")
        self.orders_total    = _Counter("argus_orders_total",        "Total orders submitted")
        self.fills_total     = _Counter("argus_fills_total",         "Total fills processed")
        self.risk_events     = _Counter("argus_risk_events_total",   "Risk events by type")

        # Histograms
        self.order_latency   = _Histogram("argus_order_latency_us",    "Order latency µs")
        self.signal_latency  = _Histogram("argus_signal_to_order_us",  "Signal→order latency µs")

    # ------------------------------------------------------------------
    # Update helpers (called from engine / risk manager)
    # ------------------------------------------------------------------

    def update_from_engine(self, engine_stats: dict) -> None:
        self.signals_total.inc(v=engine_stats.get("signals_received", 0))
        self.orders_total.inc(v=engine_stats.get("orders_submitted", 0))
        lat = engine_stats.get("avg_latency_us", 0)
        if lat > 0:
            self.order_latency.observe(lat)

    def update_from_risk(self, risk_stats: dict) -> None:
        self.equity.set(risk_stats.get("equity", 0))
        self.heat.set(risk_stats.get("portfolio_heat", 0))
        self.kill_switch.set(1.0 if risk_stats.get("kill_switch") else 0.0)

    def update_from_om(self, om_stats: dict) -> None:
        self.open_positions.set(len(om_stats.get("positions", {})))
        self.fills_total.inc(v=om_stats.get("total_fills", 0))

    def record_risk_event(self, event_type: str) -> None:
        self.risk_events.inc(labels={"type": event_type})

    # ------------------------------------------------------------------
    # Exposition
    # ------------------------------------------------------------------

    def text_exposition(self) -> str:
        """Return Prometheus text format string."""
        lines: List[str] = []
        now = time.time()

        def _gauge(g: _Gauge) -> None:
            lines.append(f"# HELP {g.name} {g.help}")
            lines.append(f"# TYPE {g.name} gauge")
            lines.append(f"{g.name} {g.value}")

        def _counter(c: _Counter) -> None:
            lines.append(f"# HELP {c.name} {c.help}")
            lines.append(f"# TYPE {c.name} counter")
            for label_str, val in c._values.items():
                lines.append(f"{c.name}{label_str} {val}")
            if not c._values:
                lines.append(f"{c.name} 0")

        def _histogram(h: _Histogram) -> None:
            lines.append(f"# HELP {h.name} {h.help}")
            lines.append(f"# TYPE {h.name} histogram")
            cumulative = 0
            for i, b in enumerate(h.buckets):
                cumulative += h._counts[i]
                le = "+Inf" if math.isinf(b) else str(b)
                lines.append(f'{h.name}_bucket{{le="{le}"}} {cumulative}')
            lines.append(f"{h.name}_sum {h._sum}")
            lines.append(f"{h.name}_count {h._total}")

        for g in (self.equity, self.heat, self.open_positions,
                  self.kill_switch, self.margin_ratio):
            _gauge(g)

        for c in (self.signals_total, self.orders_total,
                  self.fills_total, self.risk_events):
            _counter(c)

        for h in (self.order_latency, self.signal_latency):
            _histogram(h)

        # Uptime
        lines.append("# HELP argus_uptime_secs Process uptime seconds")
        lines.append("# TYPE argus_uptime_secs gauge")
        lines.append(f"argus_uptime_secs {now - self._start:.1f}")

        return "\n".join(lines) + "\n"
