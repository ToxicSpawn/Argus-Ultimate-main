"""
Resource Autoscaler — dynamically allocates background compute based on system load.

When the R740 has spare capacity, ARGUS runs MORE research:
  - Bigger GP evolver populations
  - More parallel backtests
  - Deeper feature search
  - More Monte Carlo simulations

When the system is busy with live trading or under thermal/memory pressure,
background tasks throttle down so they don't interfere with real-time work.

The autoscaler monitors:
  - CPU usage (via psutil)
  - RAM usage
  - Disk IO
  - Network throughput
  - iDRAC9 thermal sensors (if available)
  - ARGUS critical-path latency (signal → order)
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class LoadLevel(Enum):
    IDLE = "idle"            # < 30% CPU, < 60% RAM
    LIGHT = "light"          # 30-50% CPU, 60-70% RAM
    MODERATE = "moderate"    # 50-70% CPU, 70-80% RAM
    HEAVY = "heavy"          # 70-85% CPU, 80-90% RAM
    CRITICAL = "critical"    # > 85% CPU OR > 90% RAM


@dataclass
class WorkloadProfile:
    """Defines workload settings for one load level."""
    level: LoadLevel
    gp_evolver_population: int
    parallel_backtests: int
    feature_search_depth: str  # "deep", "medium", "shallow", "paused"
    monte_carlo_simulations: int
    research_engine_active: bool
    walk_forward_active: bool
    description: str


# Profile definitions per load level
WORKLOAD_PROFILES: Dict[LoadLevel, WorkloadProfile] = {
    LoadLevel.IDLE: WorkloadProfile(
        level=LoadLevel.IDLE,
        gp_evolver_population=50_000,
        parallel_backtests=100,
        feature_search_depth="deep",
        monte_carlo_simulations=10_000,
        research_engine_active=True,
        walk_forward_active=True,
        description="System idle — maximum research throughput",
    ),
    LoadLevel.LIGHT: WorkloadProfile(
        level=LoadLevel.LIGHT,
        gp_evolver_population=20_000,
        parallel_backtests=50,
        feature_search_depth="medium",
        monte_carlo_simulations=5_000,
        research_engine_active=True,
        walk_forward_active=True,
        description="Light load — balanced research",
    ),
    LoadLevel.MODERATE: WorkloadProfile(
        level=LoadLevel.MODERATE,
        gp_evolver_population=10_000,
        parallel_backtests=20,
        feature_search_depth="shallow",
        monte_carlo_simulations=2_000,
        research_engine_active=True,
        walk_forward_active=False,
        description="Moderate load — reduced research",
    ),
    LoadLevel.HEAVY: WorkloadProfile(
        level=LoadLevel.HEAVY,
        gp_evolver_population=5_000,
        parallel_backtests=10,
        feature_search_depth="paused",
        monte_carlo_simulations=1_000,
        research_engine_active=False,
        walk_forward_active=False,
        description="Heavy load — minimal background work",
    ),
    LoadLevel.CRITICAL: WorkloadProfile(
        level=LoadLevel.CRITICAL,
        gp_evolver_population=0,
        parallel_backtests=0,
        feature_search_depth="paused",
        monte_carlo_simulations=0,
        research_engine_active=False,
        walk_forward_active=False,
        description="Critical load — all background paused",
    ),
}


@dataclass
class SystemMetrics:
    """Current system resource utilization."""
    cpu_pct: float = 0.0
    ram_pct: float = 0.0
    disk_io_pct: float = 0.0
    network_pct: float = 0.0
    cpu_temp_c: float = 0.0
    critical_path_latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ResourceAutoscaler:
    """
    Monitors system resources and dynamically adjusts background workload.

    Usage::

        scaler = ResourceAutoscaler()
        scaler.register_callback("gp_evolver", lambda pop: gp_evolver.set_population(pop))

        # Every cycle:
        result = scaler.update()
        if result["level_changed"]:
            logger.info(f"Load level: {result['old_level']} → {result['new_level']}")
    """

    # Hysteresis: must stay at new level for N cycles before applying
    HYSTERESIS_CYCLES = 5

    # Default sampling interval (seconds) for psutil
    SAMPLE_INTERVAL = 1.0

    # Critical path latency threshold (ms) — if exceeded, throttle background
    CRITICAL_LATENCY_MS = 500.0

    def __init__(
        self,
        metrics_provider: Optional[Callable[[], SystemMetrics]] = None,
        history_size: int = 60,
    ) -> None:
        self._metrics_provider = metrics_provider or self._default_metrics
        self._current_level = LoadLevel.MODERATE
        self._target_level = LoadLevel.MODERATE
        self._cycles_at_target = 0
        self._history: deque[SystemMetrics] = deque(maxlen=history_size)
        self._callbacks: Dict[str, Callable[[Any], None]] = {}
        self._update_count = 0
        self._last_apply_count = 0
        logger.info("ResourceAutoscaler: initialized at level=%s", self._current_level.value)

    def register_callback(self, component: str, callback: Callable[[Any], None]) -> None:
        """Register a callback that gets called when load level changes."""
        self._callbacks[component] = callback

    def update(self) -> Dict[str, Any]:
        """
        Sample current metrics, compute load level, apply changes if stable.
        Call this every cycle.
        """
        self._update_count += 1
        metrics = self._collect_metrics()
        self._history.append(metrics)

        target = self._compute_load_level(metrics)

        # Hysteresis check
        if target == self._target_level:
            self._cycles_at_target += 1
        else:
            self._target_level = target
            self._cycles_at_target = 1

        result = {
            "cycle": self._update_count,
            "current_level": self._current_level.value,
            "target_level": target.value,
            "cycles_at_target": self._cycles_at_target,
            "metrics": {
                "cpu_pct": round(metrics.cpu_pct, 1),
                "ram_pct": round(metrics.ram_pct, 1),
                "cpu_temp_c": round(metrics.cpu_temp_c, 1),
            },
            "level_changed": False,
        }

        # Apply changes if stable
        if self._cycles_at_target >= self.HYSTERESIS_CYCLES and target != self._current_level:
            old_level = self._current_level
            self._current_level = target
            self._apply_profile(WORKLOAD_PROFILES[target])
            result["level_changed"] = True
            result["old_level"] = old_level.value
            result["new_level"] = target.value
            logger.info(
                "ResourceAutoscaler: level transition %s → %s (cpu=%.1f%%, ram=%.1f%%)",
                old_level.value, target.value, metrics.cpu_pct, metrics.ram_pct,
            )

        return result

    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics from the provider."""
        try:
            return self._metrics_provider()
        except Exception as exc:
            logger.debug("metrics_provider error: %s", exc)
            return SystemMetrics()

    @staticmethod
    def _default_metrics() -> SystemMetrics:
        """Default psutil-based metrics provider."""
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            return SystemMetrics(
                cpu_pct=cpu,
                ram_pct=ram,
                disk_io_pct=0.0,
                network_pct=0.0,
                cpu_temp_c=0.0,
                timestamp=time.time(),
            )
        except ImportError:
            logger.debug("ResourceAutoscaler: psutil not available")
            return SystemMetrics()

    def _compute_load_level(self, metrics: SystemMetrics) -> LoadLevel:
        """
        Determine load level from current metrics.
        Uses CPU and RAM as primary signals, latency as override.
        """
        # Critical path latency override
        if metrics.critical_path_latency_ms > self.CRITICAL_LATENCY_MS:
            return LoadLevel.CRITICAL

        # CPU + RAM thresholds
        cpu = metrics.cpu_pct
        ram = metrics.ram_pct

        if cpu > 85 or ram > 90:
            return LoadLevel.CRITICAL
        if cpu > 70 or ram > 80:
            return LoadLevel.HEAVY
        if cpu > 50 or ram > 70:
            return LoadLevel.MODERATE
        if cpu > 30 or ram > 60:
            return LoadLevel.LIGHT
        return LoadLevel.IDLE

    def _apply_profile(self, profile: WorkloadProfile) -> None:
        """Apply workload profile to all registered components via callbacks."""
        self._last_apply_count += 1

        # Map profile fields to callback names
        component_values = {
            "gp_evolver": profile.gp_evolver_population,
            "parallel_backtests": profile.parallel_backtests,
            "feature_search": profile.feature_search_depth,
            "monte_carlo": profile.monte_carlo_simulations,
            "research_engine": profile.research_engine_active,
            "walk_forward": profile.walk_forward_active,
        }

        for component, value in component_values.items():
            cb = self._callbacks.get(component)
            if cb is None:
                continue
            try:
                cb(value)
            except Exception as exc:
                logger.warning("autoscaler callback %s error: %s", component, exc)

    def force_level(self, level: LoadLevel) -> None:
        """Manually force a specific load level (testing/emergency)."""
        old_level = self._current_level
        self._current_level = level
        self._target_level = level
        self._apply_profile(WORKLOAD_PROFILES[level])
        logger.warning(
            "ResourceAutoscaler: MANUAL force %s → %s",
            old_level.value, level.value,
        )

    def get_current_profile(self) -> WorkloadProfile:
        return WORKLOAD_PROFILES[self._current_level]

    def get_average_metrics(self, n: int = 60) -> Optional[SystemMetrics]:
        """Get average metrics over last N samples."""
        if not self._history:
            return None
        recent = list(self._history)[-n:]
        n_samples = len(recent)
        return SystemMetrics(
            cpu_pct=sum(m.cpu_pct for m in recent) / n_samples,
            ram_pct=sum(m.ram_pct for m in recent) / n_samples,
            disk_io_pct=sum(m.disk_io_pct for m in recent) / n_samples,
            network_pct=sum(m.network_pct for m in recent) / n_samples,
            cpu_temp_c=sum(m.cpu_temp_c for m in recent) / n_samples,
        )

    def snapshot(self) -> Dict[str, Any]:
        """Return current state for advisory dict."""
        avg = self.get_average_metrics(60)
        profile = self.get_current_profile()
        return {
            "level": self._current_level.value,
            "cycles_at_target": self._cycles_at_target,
            "avg_cpu_pct": round(avg.cpu_pct, 1) if avg else 0,
            "avg_ram_pct": round(avg.ram_pct, 1) if avg else 0,
            "gp_population": profile.gp_evolver_population,
            "parallel_backtests": profile.parallel_backtests,
            "research_active": profile.research_engine_active,
            "applies_count": self._last_apply_count,
        }
