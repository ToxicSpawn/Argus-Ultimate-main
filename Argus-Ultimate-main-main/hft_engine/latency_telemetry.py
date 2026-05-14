"""
hft_engine/latency_telemetry.py
================================
Institutional-grade tick-to-trade latency telemetry system.

Provides:
  - LatencyStage enum (MARKET_DATA_RX → FILL_RX)
  - TradeJourney dataclass with per-stage nanosecond timestamps
  - LatencyTelemetry singleton: ring-buffer storage, percentile tracking,
    Prometheus exposition-format export, threshold alerting
  - HotPathProfiler: context-manager + decorator based wall-clock profiling,
    GIL-contention detection, GC pressure scoring
  - JitterMonitor: inter-tick regularity tracking per symbol
  - LatencyReport dataclass + generate_report() factory

All timestamps are in nanoseconds (time.time_ns()).
Latency properties are expressed in microseconds (µs).
"""

from __future__ import annotations

import asyncio
import gc
import logging
import statistics
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LatencyStage
# ─────────────────────────────────────────────────────────────────────────────

class LatencyStage(Enum):
    """Ordered stages of a tick-to-trade journey."""
    MARKET_DATA_RX = auto()   # Market data received from exchange feed
    SIGNAL_COMPUTE = auto()   # Alpha / signal computation
    RISK_CHECK     = auto()   # Risk / pre-trade checks
    ORDER_SUBMIT   = auto()   # Order sent to exchange
    ACK_RX         = auto()   # Exchange acknowledgement received
    FILL_RX        = auto()   # Fill / execution report received


# ─────────────────────────────────────────────────────────────────────────────
# TradeJourney
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeJourney:
    """Full lifecycle of a single tick-to-trade event."""

    journey_id: str                        # UUID
    symbol: str
    timestamps: Dict[LatencyStage, int] = field(default_factory=dict)
    start_ns: int = field(default_factory=time.time_ns)

    # ── convenience helpers ──────────────────────────────────────────────────

    def _delta_us(self, start: LatencyStage, end: LatencyStage) -> float:
        """Return microseconds between two stages; NaN if either is missing."""
        t0 = self.timestamps.get(start)
        t1 = self.timestamps.get(end)
        if t0 is None or t1 is None:
            return float("nan")
        return (t1 - t0) / 1_000.0

    def _stage_duration_us(self, stage: LatencyStage) -> float:
        """Duration *within* a stage (from its ts to the next stage's ts)."""
        stages = list(LatencyStage)
        idx = stages.index(stage)
        if idx + 1 >= len(stages):
            return float("nan")
        return self._delta_us(stage, stages[idx + 1])

    # ── public properties ────────────────────────────────────────────────────

    @property
    def tick_to_order_us(self) -> float:
        """Microseconds from MARKET_DATA_RX to ORDER_SUBMIT."""
        return self._delta_us(LatencyStage.MARKET_DATA_RX, LatencyStage.ORDER_SUBMIT)

    @property
    def tick_to_trade_us(self) -> float:
        """Microseconds from MARKET_DATA_RX to FILL_RX."""
        return self._delta_us(LatencyStage.MARKET_DATA_RX, LatencyStage.FILL_RX)

    @property
    def order_to_ack_us(self) -> float:
        """Microseconds from ORDER_SUBMIT to ACK_RX."""
        return self._delta_us(LatencyStage.ORDER_SUBMIT, LatencyStage.ACK_RX)

    @property
    def signal_latency_us(self) -> float:
        """Microseconds for SIGNAL_COMPUTE stage."""
        return self._stage_duration_us(LatencyStage.SIGNAL_COMPUTE)

    @property
    def risk_latency_us(self) -> float:
        """Microseconds for RISK_CHECK stage."""
        return self._stage_duration_us(LatencyStage.RISK_CHECK)


# ─────────────────────────────────────────────────────────────────────────────
# LatencyTelemetry (singleton)
# ─────────────────────────────────────────────────────────────────────────────

class LatencyTelemetry:
    """
    Thread-safe singleton that manages TradeJourney lifecycle and
    aggregates per-stage latency percentiles.

    Usage::

        tel = LatencyTelemetry.get_instance()
        jid = tel.start_journey("AAPL")
        tel.mark(jid, LatencyStage.SIGNAL_COMPUTE)
        ...
        tel.complete_journey(jid)
        stats = tel.get_stats()
    """

    _instance: Optional["LatencyTelemetry"] = None
    _lock: threading.Lock = threading.Lock()
    RING_BUFFER_SIZE = 10_000

    # Alert thresholds in microseconds
    WARNING_THRESHOLD_US  = 5_000    #  5 ms
    CRITICAL_THRESHOLD_US = 50_000   # 50 ms

    def __init__(self) -> None:
        self._journeys: Dict[str, TradeJourney] = {}
        self._ring: deque = deque(maxlen=self.RING_BUFFER_SIZE)
        # Per-stage samples (in µs); key = LatencyStage
        self._samples: Dict[LatencyStage, deque] = {
            stage: deque(maxlen=self.RING_BUFFER_SIZE) for stage in LatencyStage
        }
        # tick_to_order and tick_to_trade are stored separately
        self._t2o_samples: deque = deque(maxlen=self.RING_BUFFER_SIZE)
        self._t2t_samples: deque = deque(maxlen=self.RING_BUFFER_SIZE)
        self._rw_lock = threading.RLock()

    # ── singleton ────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "LatencyTelemetry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── journey lifecycle ────────────────────────────────────────────────────

    def start_journey(self, symbol: str) -> str:
        """Create a new TradeJourney, stamp MARKET_DATA_RX, return journey_id."""
        journey_id = str(uuid.uuid4())
        now_ns = time.time_ns()
        journey = TradeJourney(
            journey_id=journey_id,
            symbol=symbol,
            start_ns=now_ns,
        )
        journey.timestamps[LatencyStage.MARKET_DATA_RX] = now_ns
        with self._rw_lock:
            self._journeys[journey_id] = journey
        return journey_id

    def mark(self, journey_id: str, stage: LatencyStage) -> None:
        """Record nanosecond timestamp for *stage* on the given journey."""
        now_ns = time.time_ns()
        with self._rw_lock:
            journey = self._journeys.get(journey_id)
            if journey is None:
                logger.warning("mark() called for unknown journey %s", journey_id)
                return
            journey.timestamps[stage] = now_ns

    def complete_journey(self, journey_id: str) -> Optional[TradeJourney]:
        """
        Finalise journey: compute per-stage durations, push to ring buffer,
        accumulate percentile samples.
        """
        with self._rw_lock:
            journey = self._journeys.pop(journey_id, None)
            if journey is None:
                logger.warning("complete_journey() called for unknown journey %s", journey_id)
                return None

            self._ring.append(journey)

            # Accumulate stage-level samples
            stages = list(LatencyStage)
            for i, stage in enumerate(stages[:-1]):
                next_stage = stages[i + 1]
                t0 = journey.timestamps.get(stage)
                t1 = journey.timestamps.get(next_stage)
                if t0 is not None and t1 is not None and t1 >= t0:
                    self._samples[stage].append((t1 - t0) / 1_000.0)

            # tick-to-order and tick-to-trade aggregate metrics
            t2o = journey.tick_to_order_us
            t2t = journey.tick_to_trade_us
            if not (t2o != t2o):  # nan check
                self._t2o_samples.append(t2o)
            if not (t2t != t2t):
                self._t2t_samples.append(t2t)

            return journey

    # ── percentile stats ─────────────────────────────────────────────────────

    @staticmethod
    def _percentiles(data: deque) -> Dict[str, float]:
        """Return p50/p95/p99/p999 for a sequence of µs values."""
        if not data:
            return {"count": 0, "p50_us": 0.0, "p95_us": 0.0, "p99_us": 0.0, "p999_us": 0.0}
        arr = np.array(data, dtype=np.float64)
        return {
            "count": len(arr),
            "p50_us":  float(np.percentile(arr, 50)),
            "p95_us":  float(np.percentile(arr, 95)),
            "p99_us":  float(np.percentile(arr, 99)),
            "p999_us": float(np.percentile(arr, 99.9)),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return all percentiles for each LatencyStage plus aggregate metrics."""
        with self._rw_lock:
            result: Dict[str, Any] = {}
            for stage in LatencyStage:
                result[stage.name] = self._percentiles(self._samples[stage])
            result["TICK_TO_ORDER"] = self._percentiles(self._t2o_samples)
            result["TICK_TO_TRADE"] = self._percentiles(self._t2t_samples)
            result["completed_journeys"] = len(self._ring)
            result["in_flight_journeys"]  = len(self._journeys)
        return result

    # ── alert thresholds ─────────────────────────────────────────────────────

    def get_alert_thresholds(self) -> Dict[str, Any]:
        """
        Return threshold breaches.
        Any p99 > WARNING_THRESHOLD_US  → 'warning'
        Any p99 > CRITICAL_THRESHOLD_US → 'critical'
        """
        stats = self.get_stats()
        alerts: List[Dict[str, Any]] = []
        for key, s in stats.items():
            if not isinstance(s, dict):
                continue
            p99 = s.get("p99_us", 0.0)
            if p99 > self.CRITICAL_THRESHOLD_US:
                alerts.append({"stage": key, "p99_us": p99, "level": "critical"})
            elif p99 > self.WARNING_THRESHOLD_US:
                alerts.append({"stage": key, "p99_us": p99, "level": "warning"})
        return {
            "alerts": alerts,
            "has_critical": any(a["level"] == "critical" for a in alerts),
            "has_warning":  any(a["level"] == "warning"  for a in alerts),
        }

    # ── Prometheus exposition format ─────────────────────────────────────────

    def export_prometheus_metrics(self) -> str:
        """
        Return a string in Prometheus text exposition format covering all
        per-stage p50/p95/p99/p999 gauges and completed-journey counter.
        """
        stats = self.get_stats()
        lines: List[str] = []

        def _gauge_block(metric_name: str, help_text: str, values: Dict[str, float],
                         stage_label: str) -> None:
            lines.append(f"# HELP {metric_name} {help_text}")
            lines.append(f"# TYPE {metric_name} gauge")
            for quantile_key, quantile_val in [
                ("p50_us", "0.5"), ("p95_us", "0.95"),
                ("p99_us", "0.99"), ("p999_us", "0.999"),
            ]:
                val = values.get(quantile_key, 0.0)
                lines.append(
                    f'{metric_name}{{stage="{stage_label}",quantile="{quantile_val}"}} {val:.3f}'
                )

        for stage in LatencyStage:
            s = stats.get(stage.name, {})
            _gauge_block(
                "argus_hft_stage_latency_us",
                "HFT pipeline stage latency in microseconds",
                s,
                stage.name,
            )

        # Aggregate
        for agg_key in ("TICK_TO_ORDER", "TICK_TO_TRADE"):
            s = stats.get(agg_key, {})
            _gauge_block(
                "argus_hft_aggregate_latency_us",
                "HFT aggregate tick-to-order/trade latency in microseconds",
                s,
                agg_key,
            )

        # Journey counter
        lines.append("# HELP argus_hft_completed_journeys_total Total completed trade journeys")
        lines.append("# TYPE argus_hft_completed_journeys_total counter")
        lines.append(f"argus_hft_completed_journeys_total {stats.get('completed_journeys', 0)}")

        return "\n".join(lines) + "\n"

    # ── reset ────────────────────────────────────────────────────────────────

    def reset_stats(self) -> None:
        """Clear all accumulated data (for testing / roll-over)."""
        with self._rw_lock:
            self._journeys.clear()
            self._ring.clear()
            for stage in LatencyStage:
                self._samples[stage].clear()
            self._t2o_samples.clear()
            self._t2t_samples.clear()


# ─────────────────────────────────────────────────────────────────────────────
# HotPathProfiler
# ─────────────────────────────────────────────────────────────────────────────

class HotPathProfiler:
    """
    Lightweight instrumentation for async/sync hot paths.

    Context manager::

        with profiler.section("signal_compute"):
            ...

    Decorator::

        @profiler.instrument
        async def my_async_fn():
            ...

    Report::

        profiler.report()  # -> dict[section -> {calls, mean_us, p99_us, total_us}]
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # section_name -> list of elapsed_ns
        self._data: Dict[str, List[int]] = defaultdict(list)
        self._process_start_ns: int = time.time_ns()
        # GC pressure tracking
        self._gc_pause_total_ns: int = 0
        self._install_gc_callbacks()

    # ── GC callbacks ─────────────────────────────────────────────────────────

    def _install_gc_callbacks(self) -> None:
        self._gc_start_ns: Dict[int, int] = {}

        def _gc_before(phase: str, info: dict) -> None:  # type: ignore[type-arg]
            self._gc_start_ns[id(info)] = time.time_ns()

        def _gc_after(phase: str, info: dict) -> None:  # type: ignore[type-arg]
            start = self._gc_start_ns.pop(id(info), None)
            if start is not None:
                self._gc_pause_total_ns += time.time_ns() - start

        try:
            gc.callbacks.append(_gc_before)
            gc.callbacks.append(_gc_after)
        except Exception:  # noqa: BLE001
            pass

    # ── context manager ───────────────────────────────────────────────────────

    @contextmanager
    def section(self, name: str) -> Generator[None, None, None]:
        """Context manager that records wall-clock nanoseconds for *name*."""
        t0 = time.time_ns()
        try:
            yield
        finally:
            elapsed = time.time_ns() - t0
            with self._lock:
                self._data[name].append(elapsed)

    # ── decorator ────────────────────────────────────────────────────────────

    def instrument(self, fn: Callable) -> Callable:
        """
        Decorator for async functions.  Records wall-clock time for each call
        under the function's qualified name.
        """
        name = getattr(fn, "__qualname__", fn.__name__)

        @wraps(fn)
        async def _wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.time_ns()
            try:
                return await fn(*args, **kwargs)
            finally:
                elapsed = time.time_ns() - t0
                with self._lock:
                    self._data[name].append(elapsed)

        return _wrapper

    # ── report ────────────────────────────────────────────────────────────────

    def report(self) -> Dict[str, Dict[str, float]]:
        """
        Return per-section stats.

        Returns:
            dict mapping section name to
            ``{calls, mean_us, p99_us, total_us}``
        """
        result: Dict[str, Dict[str, float]] = {}
        with self._lock:
            for name, samples in self._data.items():
                if not samples:
                    continue
                arr = np.array(samples, dtype=np.float64) / 1_000.0  # → µs
                result[name] = {
                    "calls":    float(len(arr)),
                    "mean_us":  float(arr.mean()),
                    "p99_us":   float(np.percentile(arr, 99)),
                    "total_us": float(arr.sum()),
                }
        return result

    # ── GIL contention detection ─────────────────────────────────────────────

    def detect_gil_contention(self, iterations: int = 100) -> float:
        """
        Estimate asyncio task-switch overhead in microseconds.

        Runs a tight busy-wait loop via asyncio and measures the discrepancy
        between expected and actual wall-clock time.
        Returns overhead in µs per iteration.
        """
        async def _probe() -> float:
            samples: List[float] = []
            for _ in range(iterations):
                t0 = time.perf_counter_ns()
                await asyncio.sleep(0)  # voluntary yield → measures switch cost
                elapsed = (time.perf_counter_ns() - t0) / 1_000.0
                samples.append(elapsed)
            return float(statistics.median(samples))

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Cannot block; return cached value or 0
                return 0.0
            return loop.run_until_complete(_probe())
        except RuntimeError:
            return 0.0

    # ── GC pressure ──────────────────────────────────────────────────────────

    def gc_pressure_score(self) -> float:
        """
        Return GC pause time as a percentage of total profiler runtime.

        Triggers a manual GC collection first to capture any pending pauses,
        then computes: gc_pause_total / elapsed_wall_clock * 100.
        """
        gc.collect()
        elapsed_ns = time.time_ns() - self._process_start_ns
        if elapsed_ns <= 0:
            return 0.0
        return (self._gc_pause_total_ns / elapsed_ns) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# JitterMonitor
# ─────────────────────────────────────────────────────────────────────────────

class JitterMonitor:
    """
    Tracks arrival-time regularity of market-data ticks per symbol.

    Usage::

        jm = JitterMonitor()
        jm.record_tick("AAPL")
        ...
        print(jm.jitter_us("AAPL"))
    """

    MAX_SAMPLES = 10_000

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # symbol -> deque of nanosecond arrival timestamps
        self._arrivals: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.MAX_SAMPLES))

    # ── recording ─────────────────────────────────────────────────────────────

    def record_tick(self, symbol: str) -> None:
        """Record current wall-clock nanosecond as a tick arrival for *symbol*."""
        now_ns = time.time_ns()
        with self._lock:
            self._arrivals[symbol].append(now_ns)

    # ── metrics ───────────────────────────────────────────────────────────────

    def _inter_tick_intervals_us(self, symbol: str) -> List[float]:
        """Return list of inter-tick intervals in µs for *symbol*."""
        with self._lock:
            arrivals = list(self._arrivals[symbol])
        if len(arrivals) < 2:
            return []
        return [(arrivals[i] - arrivals[i - 1]) / 1_000.0 for i in range(1, len(arrivals))]

    def jitter_us(self, symbol: str) -> float:
        """Standard deviation of inter-tick intervals in microseconds."""
        intervals = self._inter_tick_intervals_us(symbol)
        if len(intervals) < 2:
            return 0.0
        return float(statistics.stdev(intervals))

    def late_ticks(self, symbol: str, threshold_us: float = 1_000.0) -> int:
        """Count of ticks arriving more than *threshold_us* µs after the previous tick."""
        intervals = self._inter_tick_intervals_us(symbol)
        if not intervals:
            return 0
        median_interval = statistics.median(intervals)
        return sum(1 for iv in intervals if iv > median_interval + threshold_us)

    def gap_detected(self, symbol: str, gap_threshold_us: float = 5_000.0) -> bool:
        """
        True if the last recorded tick for *symbol* is more than
        *gap_threshold_us* µs ago — i.e. the feed has gone silent.
        """
        with self._lock:
            arrivals = self._arrivals.get(symbol)
            if not arrivals:
                return False
            last_ns = arrivals[-1]
        elapsed_us = (time.time_ns() - last_ns) / 1_000.0
        return elapsed_us > gap_threshold_us


# ─────────────────────────────────────────────────────────────────────────────
# LatencyReport
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LatencyReport:
    """Consolidated snapshot of all latency telemetry."""

    timestamp_ns: int
    stage_stats:  Dict[str, Dict[str, float]]
    alert_info:   Dict[str, Any]
    profiler_report: Dict[str, Dict[str, float]]
    gc_pressure_pct: float
    jitter_by_symbol: Dict[str, float]     # symbol -> jitter_us
    late_tick_counts: Dict[str, int]       # symbol -> late_ticks count
    gap_symbols:      List[str]            # symbols with detected gaps

    @property
    def is_healthy(self) -> bool:
        """
        True if:
          - p99 tick-to-order latency < 10 ms (10,000 µs), AND
          - No active data gaps on any symbol
        """
        t2o_stats = self.stage_stats.get("TICK_TO_ORDER", {})
        p99_us = t2o_stats.get("p99_us", 0.0)
        if p99_us >= 10_000.0:
            return False
        if self.gap_symbols:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "timestamp_ns":      self.timestamp_ns,
            "is_healthy":        self.is_healthy,
            "stage_stats":       self.stage_stats,
            "alert_info":        self.alert_info,
            "profiler_report":   self.profiler_report,
            "gc_pressure_pct":   self.gc_pressure_pct,
            "jitter_by_symbol":  self.jitter_by_symbol,
            "late_tick_counts":  self.late_tick_counts,
            "gap_symbols":       self.gap_symbols,
        }


def generate_report(
    telemetry: LatencyTelemetry,
    profiler: HotPathProfiler,
    jitter: JitterMonitor,
    symbols: Optional[List[str]] = None,
    gap_threshold_us: float = 5_000.0,
    late_threshold_us: float = 1_000.0,
) -> LatencyReport:
    """
    Build a LatencyReport from the three monitoring objects.

    Parameters
    ----------
    telemetry:
        LatencyTelemetry singleton.
    profiler:
        HotPathProfiler instance.
    jitter:
        JitterMonitor instance.
    symbols:
        Explicit list of symbols to check jitter/gaps for.
        Defaults to all symbols seen by the JitterMonitor.
    gap_threshold_us:
        Threshold for gap_detected() in µs.
    late_threshold_us:
        Threshold for late_ticks() in µs.
    """
    # Determine symbols to check
    if symbols is None:
        with jitter._lock:
            symbols = list(jitter._arrivals.keys())

    stage_stats   = telemetry.get_stats()
    alert_info    = telemetry.get_alert_thresholds()
    prof_report   = profiler.report()
    gc_pct        = profiler.gc_pressure_score()

    jitter_map:  Dict[str, float] = {}
    late_counts: Dict[str, int]   = {}
    gap_syms:    List[str]        = []

    for sym in symbols:
        jitter_map[sym]  = jitter.jitter_us(sym)
        late_counts[sym] = jitter.late_ticks(sym, threshold_us=late_threshold_us)
        if jitter.gap_detected(sym, gap_threshold_us=gap_threshold_us):
            gap_syms.append(sym)

    return LatencyReport(
        timestamp_ns=time.time_ns(),
        stage_stats=stage_stats,
        alert_info=alert_info,
        profiler_report=prof_report,
        gc_pressure_pct=gc_pct,
        jitter_by_symbol=jitter_map,
        late_tick_counts=late_counts,
        gap_symbols=gap_syms,
    )
