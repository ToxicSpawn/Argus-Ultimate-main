"""
Error tracker — collects every runtime failure into a single report.

Tracks:
- Component failures (which component, how many times, last error)
- Recurring errors (same error 3+ times = flagged)
- Auto-disabled components (failed 5+ times consecutively)
- Error timeline (when errors started, frequency)

Report saved to data/error_report.json every 50 cycles.
Read it with: py -c "import json; print(json.dumps(json.load(open('data/error_report.json')), indent=2))"
Or just tell Claude Code: "fix ARGUS errors"
"""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ComponentError:
    component: str
    error_type: str
    error_msg: str
    traceback_short: str
    timestamp: float
    cycle: int


class ErrorTracker:
    """
    Tracks all runtime errors across ARGUS components.

    Usage:
        tracker = ErrorTracker()
        tracker.record("vol_forecaster", exception, cycle=42)
        tracker.maybe_disable("vol_forecaster")  # disables after 5 consecutive failures
        tracker.save_report()  # writes to data/error_report.json
    """

    def __init__(
        self,
        report_path: str = "data/error_report.json",
        auto_disable_threshold: int = 5,
        save_interval_cycles: int = 50,
    ) -> None:
        self.report_path = report_path
        self.auto_disable_threshold = auto_disable_threshold
        self.save_interval_cycles = save_interval_cycles

        # Error tracking state
        self._errors: List[ComponentError] = []
        self._consecutive_failures: Dict[str, int] = defaultdict(int)
        self._total_failures: Dict[str, int] = defaultdict(int)
        self._last_error: Dict[str, str] = {}
        self._disabled_components: Dict[str, str] = {}  # name → reason
        self._last_success: Dict[str, float] = {}
        self._first_error_ts: Dict[str, float] = {}
        self._cycle_count = 0

        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        logger.info("ErrorTracker: initialized (report=%s, auto_disable_threshold=%d)",
                     report_path, auto_disable_threshold)

    def record(self, component: str, exc: Exception, cycle: int = 0) -> None:
        """Record a component failure."""
        self._cycle_count = max(self._cycle_count, cycle)
        error_type = type(exc).__name__
        error_msg = str(exc)[:200]
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        tb_short = "".join(tb[-3:])[:500] if tb else ""

        entry = ComponentError(
            component=component,
            error_type=error_type,
            error_msg=error_msg,
            traceback_short=tb_short,
            timestamp=time.time(),
            cycle=cycle,
        )
        self._errors.append(entry)
        if len(self._errors) > 1000:
            self._errors = self._errors[-500:]

        self._consecutive_failures[component] += 1
        self._total_failures[component] += 1
        self._last_error[component] = f"{error_type}: {error_msg}"

        if component not in self._first_error_ts:
            self._first_error_ts[component] = time.time()

    def record_success(self, component: str) -> None:
        """Record a successful component call — resets consecutive failure count."""
        self._consecutive_failures[component] = 0
        self._last_success[component] = time.time()

    def should_disable(self, component: str) -> bool:
        """Check if a component should be auto-disabled."""
        return self._consecutive_failures.get(component, 0) >= self.auto_disable_threshold

    def disable(self, component: str, reason: str = "") -> None:
        """Mark a component as disabled."""
        if component not in self._disabled_components:
            self._disabled_components[component] = reason or self._last_error.get(component, "unknown")
            logger.warning(
                "ErrorTracker: AUTO-DISABLED %s after %d consecutive failures: %s",
                component, self._consecutive_failures.get(component, 0),
                self._disabled_components[component][:100],
            )

    def is_disabled(self, component: str) -> bool:
        """Check if a component is disabled."""
        return component in self._disabled_components

    def maybe_save(self, cycle: int) -> None:
        """Save report if interval elapsed."""
        self._cycle_count = cycle
        if cycle % self.save_interval_cycles == 0:
            self.save_report()

    def save_report(self) -> None:
        """Save error report to JSON file."""
        # Build recurring errors (same component + same error type 3+ times)
        recurring = {}
        error_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for e in self._errors:
            error_counts[e.component][e.error_type] += 1
        for comp, types in error_counts.items():
            for etype, count in types.items():
                if count >= 3:
                    recurring[f"{comp}:{etype}"] = {
                        "count": count,
                        "last_msg": self._last_error.get(comp, ""),
                    }

        report = {
            "generated_at": time.time(),
            "cycle": self._cycle_count,
            "summary": {
                "total_errors": len(self._errors),
                "components_with_errors": len(self._total_failures),
                "auto_disabled": len(self._disabled_components),
                "recurring_errors": len(recurring),
            },
            "disabled_components": dict(self._disabled_components),
            "recurring_errors": recurring,
            "top_failing_components": sorted(
                [{"component": k, "total_failures": v, "consecutive": self._consecutive_failures.get(k, 0),
                  "last_error": self._last_error.get(k, "")}
                 for k, v in self._total_failures.items()],
                key=lambda x: -x["total_failures"],
            )[:20],
            "recent_errors": [
                {"component": e.component, "type": e.error_type, "msg": e.error_msg,
                 "cycle": e.cycle, "timestamp": e.timestamp}
                for e in self._errors[-50:]
            ],
        }

        try:
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
        except Exception as exc:
            logger.debug("ErrorTracker: save failed: %s", exc)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_errors": len(self._errors),
            "disabled_components": list(self._disabled_components.keys()),
            "top_failures": sorted(
                [(k, v) for k, v in self._total_failures.items()],
                key=lambda x: -x[1],
            )[:5],
        }
