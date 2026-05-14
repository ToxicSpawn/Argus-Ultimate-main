from __future__ import annotations

import asyncio
import logging
import os
import time
import tracemalloc
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Tuple

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

_WARNING_TIMESTAMPS: Deque[float] = deque(maxlen=200_000)
_ERROR_TIMESTAMPS: Deque[float] = deque(maxlen=200_000)


class _SystemHealthLogCounterHandler(logging.Handler):
    """Counts warning/error log records for rolling one-hour health metrics."""

    def emit(self, record: logging.LogRecord) -> None:
        now = time.time()
        if int(record.levelno) >= int(logging.ERROR):
            _ERROR_TIMESTAMPS.append(now)
        elif int(record.levelno) >= int(logging.WARNING):
            _WARNING_TIMESTAMPS.append(now)


def _ensure_log_counter_handler() -> None:
    root = logging.getLogger()
    for handler in root.handlers:
        if bool(getattr(handler, "_argus_system_health_counter", False)):
            return
    handler = _SystemHealthLogCounterHandler(level=logging.WARNING)
    setattr(handler, "_argus_system_health_counter", True)
    root.addHandler(handler)


def _prune_old(ts_queue: Deque[float], cutoff_ts: float) -> None:
    while ts_queue and float(ts_queue[0]) < cutoff_ts:
        ts_queue.popleft()


@dataclass(slots=True)
class SystemHealthSnapshot:
    timestamp: str
    cycles_completed: int
    avg_latency_ms: float
    errors_last_hour: int
    warnings_last_hour: int
    event_loop_delay_ms: float
    engine_uptime_seconds: float
    memory_rss_mb: float
    memory_python_mb: float


class SystemHealthMetricsCollector:
    """Tracks runtime health metrics and builds periodic snapshot records."""

    def __init__(self, *, enabled: bool = True, snapshot_interval_cycles: int = 10) -> None:
        self.enabled = bool(enabled)
        self.snapshot_interval_cycles = max(1, int(snapshot_interval_cycles or 10))
        self._start_monotonic = time.perf_counter()
        self._cycles_completed = 0
        self._cycle_latencies_ms: Deque[float] = deque(maxlen=50_000)
        self._last_event_loop_delay_ms = 0.0
        self._ps_process = None
        if psutil is not None:
            try:
                self._ps_process = psutil.Process(os.getpid())
            except Exception:
                self._ps_process = None
        if not tracemalloc.is_tracing():
            try:
                tracemalloc.start()
            except Exception:
                pass
        if self.enabled:
            _ensure_log_counter_handler()

    @property
    def cycles_completed(self) -> int:
        return int(self._cycles_completed)

    @property
    def uptime_seconds(self) -> float:
        return float(max(0.0, time.perf_counter() - self._start_monotonic))

    async def sample_event_loop_delay_ms(self) -> float:
        """Estimate scheduler delay by yielding one tick on the current loop."""
        if not self.enabled:
            return 0.0
        loop = asyncio.get_running_loop()
        t0 = float(loop.time())
        waiter: asyncio.Future[None] = loop.create_future()
        loop.call_soon(waiter.set_result, None)
        await waiter
        delay_ms = max(0.0, (float(loop.time()) - t0) * 1000.0)
        self._last_event_loop_delay_ms = float(delay_ms)
        return float(delay_ms)

    def record_cycle(self, *, cycle_latency_ms: float, event_loop_delay_ms: float | None = None) -> None:
        if not self.enabled:
            return
        self._cycles_completed += 1
        self._cycle_latencies_ms.append(max(0.0, float(cycle_latency_ms)))
        if event_loop_delay_ms is not None:
            self._last_event_loop_delay_ms = max(0.0, float(event_loop_delay_ms))

    def should_snapshot(self, *, cycles_completed: int | None = None) -> bool:
        if not self.enabled:
            return False
        c = int(self._cycles_completed if cycles_completed is None else cycles_completed)
        return c > 0 and (c % int(self.snapshot_interval_cycles) == 0)

    def _counts_last_hour(self) -> Tuple[int, int]:
        cutoff = float(time.time()) - 3600.0
        _prune_old(_WARNING_TIMESTAMPS, cutoff)
        _prune_old(_ERROR_TIMESTAMPS, cutoff)
        return int(len(_ERROR_TIMESTAMPS)), int(len(_WARNING_TIMESTAMPS))

    def _sample_memory_mb(self) -> Tuple[float, float]:
        rss_mb = 0.0
        py_mb = 0.0
        try:
            if self._ps_process is not None:
                rss_mb = float(self._ps_process.memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            rss_mb = 0.0
        try:
            if tracemalloc.is_tracing():
                current, _peak = tracemalloc.get_traced_memory()
                py_mb = float(current) / (1024.0 * 1024.0)
        except Exception:
            py_mb = 0.0
        return max(0.0, rss_mb), max(0.0, py_mb)

    def build_snapshot(self, *, cycles_completed: int | None = None) -> SystemHealthSnapshot:
        errors_last_hour, warnings_last_hour = self._counts_last_hour()
        c = int(self._cycles_completed if cycles_completed is None else cycles_completed)
        if self._cycle_latencies_ms:
            avg_latency_ms = float(sum(self._cycle_latencies_ms) / len(self._cycle_latencies_ms))
        else:
            avg_latency_ms = 0.0
        rss_mb, py_mb = self._sample_memory_mb()
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        return SystemHealthSnapshot(
            timestamp=ts,
            cycles_completed=max(0, c),
            avg_latency_ms=max(0.0, avg_latency_ms),
            errors_last_hour=max(0, errors_last_hour),
            warnings_last_hour=max(0, warnings_last_hour),
            event_loop_delay_ms=max(0.0, float(self._last_event_loop_delay_ms)),
            engine_uptime_seconds=self.uptime_seconds,
            memory_rss_mb=float(rss_mb),
            memory_python_mb=float(py_mb),
        )
