"""
TimingIntelligence — Three-layer trade timing gate.

Layer 1 — Hour-of-day stats:
    Uses perf_attribution.compute_by_hour() to find hours with negative
    sharpe_like performance. bad_hour → score -= 20, action ≥ DEFER.
    Stats refreshed every refresh_interval_cycles.

Layer 2 — Macro events:
    advisory["session_effect"]["macro_event_imminent"] → BLOCK

Layer 3 — System state:
    advisory["latency"]["p99_ms"] > latency_block_ms → DEFER
    advisory["market_anomaly"]["severity"] > anomaly_block_threshold → BLOCK

Score 0–100. Actions: IDEAL / OK / DEFER / BLOCK

Output: advisory["timing_intelligence"]
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums / dataclasses
# ---------------------------------------------------------------------------

class TimingAction(str, Enum):
    IDEAL = "ideal"   # 80–100 score
    OK    = "ok"      # 50–79
    DEFER = "defer"   # 20–49 — log but continue
    BLOCK = "block"   # 0–19  — skip this signal


@dataclass
class TimingReport:
    action: TimingAction
    score: float              # 0–100
    reasons: List[str]
    good_hours: List[int]     # UTC hours with positive sharpe_like
    bad_hours: List[int]      # UTC hours with negative sharpe_like
    current_hour: int
    ts: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# TimingIntelligence
# ---------------------------------------------------------------------------

class TimingIntelligence:
    """
    Three-layer timing gate.

    Parameters
    ----------
    perf_attribution            : PerformanceAttribution instance (optional)
    bad_hour_sharpe_threshold   : sharpe_like below this → bad hour (default -0.1)
    block_bad_hours             : True = BLOCK in bad hours, False = DEFER only
    latency_defer_ms            : p99 latency above this → DEFER (default 500)
    latency_block_ms            : p99 latency above this → BLOCK (default 1000)
    anomaly_block_threshold     : anomaly severity above this → BLOCK (default 0.6)
    refresh_interval_cycles     : how often to refresh hour stats (default 200)
    config                      : optional config object
    """

    def __init__(
        self,
        perf_attribution: Optional[Any] = None,
        bad_hour_sharpe_threshold: float = -0.10,
        block_bad_hours: bool = True,
        latency_defer_ms: float = 500.0,
        latency_block_ms: float = 1000.0,
        anomaly_block_threshold: float = 0.60,
        refresh_interval_cycles: int = 200,
        config: Optional[Any] = None,
    ) -> None:
        self.perf_attribution          = perf_attribution
        self.bad_hour_sharpe_threshold = float(bad_hour_sharpe_threshold)
        self.block_bad_hours           = bool(block_bad_hours)
        self.latency_defer_ms          = float(latency_defer_ms)
        self.latency_block_ms          = float(latency_block_ms)
        self.anomaly_block_threshold   = float(anomaly_block_threshold)
        self.refresh_interval_cycles   = max(1, int(refresh_interval_cycles))
        self.config                    = config

        # Hour stats: {hour_int: sharpe_like_float}
        self._hour_stats: Dict[int, float] = {}
        self._bad_hours:  Set[int] = set()
        self._good_hours: Set[int] = set()
        self._last_refresh_cycle: int = -999999

        self._last_report: Optional[TimingReport] = None

    # ── Public API ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        advisory: Dict[str, Any],
        current_hour: int,
        cycle: int = 0,
    ) -> TimingReport:
        """
        Evaluate whether now is a good time to trade.

        Parameters
        ----------
        advisory     : full advisory dict
        current_hour : current UTC hour (0–23)
        cycle        : current cycle number
        """
        score: float = 100.0
        reasons: List[str] = []
        block = False
        defer = False

        # ── Auto-refresh hour stats if stale ──────────────────────────────
        if cycle - self._last_refresh_cycle >= self.refresh_interval_cycles:
            self.refresh_hour_stats(cycle)

        # ── Layer 1: hour-of-day stats ─────────────────────────────────────
        hour = int(current_hour) % 24
        if hour in self._bad_hours:
            score -= 25.0
            if self.block_bad_hours:
                block = True
                reasons.append(f"bad_hour={hour} (block_bad_hours=True)")
            else:
                defer = True
                reasons.append(f"bad_hour={hour} (defer only)")
        elif hour in self._good_hours:
            score = min(100.0, score + 5.0)

        # ── Layer 2: macro events ──────────────────────────────────────────
        _se = advisory.get("session_effect") or {}
        macro_imminent = bool(_se.get("macro_event_imminent", False))
        if macro_imminent:
            block = True
            score -= 40.0
            reasons.append("macro_event_imminent")

        # ── Layer 3: system state ──────────────────────────────────────────
        _lat = advisory.get("latency") or {}
        p99_ms = float(_lat.get("p99_ms", 0.0) or 0.0)
        if p99_ms > self.latency_block_ms:
            defer = True
            score -= 30.0
            reasons.append(f"latency_critical p99={p99_ms:.0f}ms")
        elif p99_ms > self.latency_defer_ms:
            defer = True
            score -= 15.0
            reasons.append(f"latency_elevated p99={p99_ms:.0f}ms")

        _ma = advisory.get("market_anomaly") or {}
        anomaly_severity = float(_ma.get("severity", 0.0) or 0.0)
        if anomaly_severity > self.anomaly_block_threshold:
            block = True
            score -= 35.0
            reasons.append(f"anomaly={anomaly_severity:.2f}")

        # ── Determine action ──────────────────────────────────────────────
        score = max(0.0, min(100.0, score))
        if block or score < 20.0:
            action = TimingAction.BLOCK
        elif defer or score < 50.0:
            action = TimingAction.DEFER
        elif score >= 80.0:
            action = TimingAction.IDEAL
        else:
            action = TimingAction.OK

        report = TimingReport(
            action       = action,
            score        = round(score, 2),
            reasons      = reasons,
            good_hours   = sorted(self._good_hours),
            bad_hours    = sorted(self._bad_hours),
            current_hour = hour,
        )
        self._last_report = report
        return report

    def refresh_hour_stats(self, cycle: int = 0) -> None:
        """
        Pull updated hour-of-day performance stats from perf_attribution.
        Called automatically every refresh_interval_cycles.
        """
        self._last_refresh_cycle = cycle
        if self.perf_attribution is None:
            return
        try:
            stats = self.perf_attribution.compute_by_hour()
            if not stats:
                return
            self._hour_stats = {}
            self._bad_hours  = set()
            self._good_hours = set()
            for hour_key, metrics in stats.items():
                try:
                    h = int(hour_key)
                    sharpe_like = float(
                        metrics.get("sharpe_like", 0.0)
                        or metrics.get("sharpe", 0.0)
                        or 0.0
                    )
                    self._hour_stats[h] = sharpe_like
                    if sharpe_like < self.bad_hour_sharpe_threshold:
                        self._bad_hours.add(h)
                    elif sharpe_like > abs(self.bad_hour_sharpe_threshold):
                        self._good_hours.add(h)
                except (TypeError, ValueError):
                    pass
            logger.debug(
                "TimingIntelligence: refreshed hour stats — %d bad, %d good",
                len(self._bad_hours), len(self._good_hours),
            )
        except Exception as exc:
            logger.debug("TimingIntelligence: refresh_hour_stats failed: %s", exc)

    def snapshot(self) -> Dict[str, Any]:
        r = self._last_report
        if r is None:
            return {
                "action": TimingAction.OK.value,
                "score": 100.0,
                "reasons": [],
                "good_hours": [],
                "bad_hours": [],
                "current_hour": -1,
            }
        return {
            "action":       r.action.value,
            "score":        r.score,
            "reasons":      r.reasons,
            "good_hours":   r.good_hours,
            "bad_hours":    r.bad_hours,
            "current_hour": r.current_hour,
            "ts":           r.ts,
        }
