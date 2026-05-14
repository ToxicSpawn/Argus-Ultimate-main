"""
Self-Debugging Component Monitor — detects anomalous behaviour in system
components and can auto-disable persistently failing ones.

Checks performed on each component's output history:
    * NaN/None output (invalid data)
    * Output outside 3-sigma range (statistical outlier)
    * Constant output (stuck / frozen component)
    * Output frequency anomaly (component producing too fast or too slow)

In-memory rolling window — no SQLite needed (fast, stateless restart).

Usage::

    dbg = SelfDebugger()
    dbg.record_output("alpha_model", 0.42)
    dbg.record_output("alpha_model", 0.41)
    alert = dbg.detect_anomaly("alpha_model")
    logger.info(alert)
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DebugAlert:
    """Result of an anomaly check on a component."""

    component: str
    anomalous: bool
    reason: str
    severity: str  # "info", "warning", "critical"
    recommendation: str

    def __repr__(self) -> str:
        return (
            f"DebugAlert(component={self.component!r}, anomalous={self.anomalous}, "
            f"severity={self.severity!r}, reason={self.reason!r})"
        )


@dataclass
class _ComponentState:
    """Internal state tracker for a single component."""

    values: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    consecutive_anomalies: int = 0
    disabled: bool = False
    nan_count: int = 0
    total_count: int = 0


class SelfDebugger:
    """Monitor component outputs and detect anomalous behaviour.

    Parameters
    ----------
    window_size : int
        Maximum number of recent outputs to retain per component.
    sigma_threshold : float
        Number of standard deviations for outlier detection.
    stuck_threshold : int
        Number of identical consecutive outputs to flag as "stuck".
    auto_disable_threshold : int
        Number of consecutive anomalous readings before auto-disabling.
    """

    def __init__(
        self,
        window_size: int = 200,
        sigma_threshold: float = 3.0,
        stuck_threshold: int = 20,
        auto_disable_threshold: int = 5,
    ) -> None:
        self._window_size = window_size
        self._sigma_threshold = sigma_threshold
        self._stuck_threshold = stuck_threshold
        self._auto_disable_threshold = auto_disable_threshold
        self._states: Dict[str, _ComponentState] = defaultdict(
            lambda: _ComponentState(
                values=deque(maxlen=window_size),
                timestamps=deque(maxlen=window_size),
            )
        )
        self._lock = threading.Lock()
        logger.info(
            "SelfDebugger initialised (window=%d, sigma=%.1f, stuck=%d, auto_disable=%d)",
            window_size, sigma_threshold, stuck_threshold, auto_disable_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_output(
        self,
        component: str,
        value: Any,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a component's output value.

        Parameters
        ----------
        component : str
            Component name.
        value : Any
            Output value.  Non-numeric values are recorded as NaN.
        timestamp : datetime | None
            UTC timestamp.  Defaults to now.
        """
        ts = (timestamp or datetime.now(timezone.utc)).timestamp()

        with self._lock:
            state = self._states[component]
            state.total_count += 1

            if value is None:
                state.values.append(float("nan"))
                state.nan_count += 1
            elif isinstance(value, (int, float)):
                if math.isnan(value) or math.isinf(value):
                    state.values.append(float("nan"))
                    state.nan_count += 1
                else:
                    state.values.append(float(value))
            else:
                # Non-numeric: record as NaN
                state.values.append(float("nan"))
                state.nan_count += 1

            state.timestamps.append(ts)

    def detect_anomaly(self, component: str) -> DebugAlert:
        """Run all anomaly checks on a component.

        Parameters
        ----------
        component : str
            Component name to check.

        Returns
        -------
        DebugAlert
            Anomaly detection result.
        """
        with self._lock:
            state = self._states.get(component)
            if state is None or len(state.values) == 0:
                return DebugAlert(
                    component=component,
                    anomalous=False,
                    reason="No data recorded yet",
                    severity="info",
                    recommendation="Await first output",
                )

            if state.disabled:
                return DebugAlert(
                    component=component,
                    anomalous=True,
                    reason="Component has been auto-disabled",
                    severity="critical",
                    recommendation="Investigate and re-enable manually",
                )

            # Check 1: NaN/None output (latest value)
            alert = self._check_nan(component, state)
            if alert and alert.anomalous:
                state.consecutive_anomalies += 1
                return alert

            # Check 2: Output outside 3-sigma
            alert = self._check_sigma(component, state)
            if alert and alert.anomalous:
                state.consecutive_anomalies += 1
                return alert

            # Check 3: Stuck (constant output)
            alert = self._check_stuck(component, state)
            if alert and alert.anomalous:
                state.consecutive_anomalies += 1
                return alert

            # Check 4: Frequency anomaly
            alert = self._check_frequency(component, state)
            if alert and alert.anomalous:
                state.consecutive_anomalies += 1
                return alert

            # All clear
            state.consecutive_anomalies = 0
            return DebugAlert(
                component=component,
                anomalous=False,
                reason="All checks passed",
                severity="info",
                recommendation="No action needed",
            )

    def auto_disable(self, component: str) -> bool:
        """Disable a component if it has been anomalous for too many consecutive readings.

        Parameters
        ----------
        component : str
            Component name.

        Returns
        -------
        bool
            ``True`` if the component was disabled, ``False`` otherwise.
        """
        with self._lock:
            state = self._states.get(component)
            if state is None:
                return False

            if state.disabled:
                return False  # Already disabled

            if state.consecutive_anomalies >= self._auto_disable_threshold:
                state.disabled = True
                logger.warning(
                    "AUTO-DISABLED component %s after %d consecutive anomalies",
                    component, state.consecutive_anomalies,
                )
                return True

            return False

    def get_component_health(self) -> Dict[str, str]:
        """Return health status for all tracked components.

        Returns
        -------
        dict[str, str]
            Mapping of component name → ``"healthy"``, ``"warning"``, or ``"disabled"``.
        """
        with self._lock:
            result: Dict[str, str] = {}
            for name, state in self._states.items():
                if state.disabled:
                    result[name] = "disabled"
                elif state.consecutive_anomalies > 0:
                    result[name] = "warning"
                else:
                    result[name] = "healthy"
            return result

    def re_enable(self, component: str) -> bool:
        """Re-enable a previously disabled component.

        Parameters
        ----------
        component : str
            Component name.

        Returns
        -------
        bool
            ``True`` if the component was re-enabled, ``False`` if it was not disabled.
        """
        with self._lock:
            state = self._states.get(component)
            if state is None or not state.disabled:
                return False
            state.disabled = False
            state.consecutive_anomalies = 0
            logger.info("Re-enabled component %s", component)
            return True

    # ------------------------------------------------------------------
    # Anomaly check implementations
    # ------------------------------------------------------------------

    def _check_nan(self, component: str, state: _ComponentState) -> Optional[DebugAlert]:
        """Check if the most recent output is NaN/None."""
        if len(state.values) == 0:
            return None

        latest = state.values[-1]
        if math.isnan(latest):
            nan_ratio = state.nan_count / max(state.total_count, 1)
            return DebugAlert(
                component=component,
                anomalous=True,
                reason=f"Latest output is NaN/None (NaN ratio: {nan_ratio:.1%})",
                severity="warning" if nan_ratio < 0.5 else "critical",
                recommendation="Check input data pipeline and upstream dependencies",
            )
        return None

    def _check_sigma(self, component: str, state: _ComponentState) -> Optional[DebugAlert]:
        """Check if the latest value is outside 3-sigma range."""
        valid = [v for v in state.values if not math.isnan(v)]
        if len(valid) < 10:
            return None  # Not enough data

        mean = sum(valid) / len(valid)
        variance = sum((v - mean) ** 2 for v in valid) / len(valid)
        std = math.sqrt(variance) if variance > 0 else 0.0

        if std < 1e-12:
            return None  # Zero variance — handled by stuck check

        latest = state.values[-1]
        if math.isnan(latest):
            return None  # Handled by NaN check

        z_score = abs(latest - mean) / std
        if z_score > self._sigma_threshold:
            return DebugAlert(
                component=component,
                anomalous=True,
                reason=f"Output {latest:.4f} is {z_score:.1f} sigma from mean {mean:.4f} (std={std:.4f})",
                severity="warning" if z_score < 5.0 else "critical",
                recommendation="Check for data spikes, model instability, or input corruption",
            )
        return None

    def _check_stuck(self, component: str, state: _ComponentState) -> Optional[DebugAlert]:
        """Check if the component is producing constant output (stuck)."""
        valid = [v for v in state.values if not math.isnan(v)]
        if len(valid) < self._stuck_threshold:
            return None

        recent = list(valid)[-self._stuck_threshold:]
        if all(abs(v - recent[0]) < 1e-12 for v in recent):
            return DebugAlert(
                component=component,
                anomalous=True,
                reason=f"Output stuck at {recent[0]:.6f} for {self._stuck_threshold} consecutive readings",
                severity="warning",
                recommendation="Component may be frozen — check for deadlocks or stale cache",
            )
        return None

    def _check_frequency(self, component: str, state: _ComponentState) -> Optional[DebugAlert]:
        """Check for output frequency anomalies (too fast or too slow)."""
        if len(state.timestamps) < 10:
            return None

        intervals = []
        ts_list = list(state.timestamps)
        for i in range(1, len(ts_list)):
            intervals.append(ts_list[i] - ts_list[i - 1])

        if not intervals:
            return None

        mean_interval = sum(intervals) / len(intervals)
        if mean_interval < 1e-6:
            return DebugAlert(
                component=component,
                anomalous=True,
                reason="Output frequency extremely high (near-zero interval between outputs)",
                severity="warning",
                recommendation="Possible tight loop or duplicate output — check event source",
            )

        # Check if latest interval is >5x the mean (too slow)
        latest_interval = intervals[-1]
        if latest_interval > mean_interval * 5.0 and mean_interval > 0.1:
            return DebugAlert(
                component=component,
                anomalous=True,
                reason=f"Output gap {latest_interval:.1f}s is {latest_interval/mean_interval:.1f}x the mean interval ({mean_interval:.1f}s)",
                severity="warning",
                recommendation="Component may be stalled — check for blocking operations",
            )

        return None
