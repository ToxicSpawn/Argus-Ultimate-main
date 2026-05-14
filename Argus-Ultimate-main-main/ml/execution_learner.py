"""
Execution Learning Agent for ARGUS.

Learns optimal order timing and sizing from past fill data. Tracks
slippage by hour-of-day, day-of-week, volatility, and spread conditions
to recommend when to trade and when to wait.

Usage:
    learner = ExecutionLearner()
    learner.record_fill("BTC/USD", "buy", slippage_bps=2.5,
                        hour_utc=14, day_of_week=2, volatility=0.03, spread_bps=5.0)
    window = learner.get_optimal_execution_window("BTC/USD")
    delay = learner.should_delay_execution("BTC/USD", current_hour_utc=3, current_spread_bps=15.0)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FillRecord:
    """Single execution fill record for learning."""
    symbol: str
    side: str
    slippage_bps: float
    hour_utc: int
    day_of_week: int  # 0=Monday, 6=Sunday
    volatility: float
    spread_bps: float
    size_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# ExecutionLearner
# ---------------------------------------------------------------------------


class ExecutionLearner:
    """
    Learns optimal execution timing and sizing from historical fills.

    Parameters
    ----------
    max_history : int
        Maximum fill records to retain per symbol (default 5000).
    """

    def __init__(self, max_history: int = 5000) -> None:
        self._max_history = max(100, int(max_history))
        self._fill_history: Dict[str, List[FillRecord]] = defaultdict(list)

    # ── Recording ─────────────────────────────────────────────────────────

    def record_fill(
        self,
        symbol: str,
        side: str,
        slippage_bps: float,
        hour_utc: int,
        day_of_week: int,
        volatility: float,
        spread_bps: float,
        size_usd: float = 0.0,
    ) -> None:
        """Record execution quality metrics for a fill."""
        rec = FillRecord(
            symbol=str(symbol),
            side=str(side).lower(),
            slippage_bps=float(slippage_bps),
            hour_utc=int(hour_utc) % 24,
            day_of_week=int(day_of_week) % 7,
            volatility=float(volatility),
            spread_bps=float(spread_bps),
            size_usd=float(size_usd),
        )
        fills = self._fill_history[symbol]
        fills.append(rec)
        if len(fills) > self._max_history:
            self._fill_history[symbol] = fills[-self._max_history:]

    # ── Optimal window ────────────────────────────────────────────────────

    def get_optimal_execution_window(self, symbol: str) -> dict:
        """
        Analyze fills to find best/worst execution times.

        Returns
        -------
        dict with keys:
            best_hours, worst_hours, best_days, worst_days,
            avg_slippage_by_hour, avg_slippage_by_day
        """
        fills = self._fill_history.get(symbol, [])
        if len(fills) < 5:
            return {
                "best_hours": [],
                "worst_hours": [],
                "best_days": [],
                "worst_days": [],
                "avg_slippage_by_hour": {},
                "avg_slippage_by_day": {},
                "total_fills": len(fills),
            }

        # Slippage by hour
        hour_slippage: Dict[int, List[float]] = defaultdict(list)
        for f in fills:
            hour_slippage[f.hour_utc].append(f.slippage_bps)

        avg_by_hour = {
            h: round(float(np.mean(slips)), 4)
            for h, slips in hour_slippage.items()
            if slips
        }

        # Slippage by day
        day_slippage: Dict[int, List[float]] = defaultdict(list)
        for f in fills:
            day_slippage[f.day_of_week].append(f.slippage_bps)

        avg_by_day = {
            d: round(float(np.mean(slips)), 4)
            for d, slips in day_slippage.items()
            if slips
        }

        # Sort hours by average slippage (lower = better)
        sorted_hours = sorted(avg_by_hour.items(), key=lambda x: x[1])
        best_hours = [h for h, _ in sorted_hours[:3]] if sorted_hours else []
        worst_hours = [h for h, _ in sorted_hours[-3:]] if sorted_hours else []

        # Sort days by average slippage
        sorted_days = sorted(avg_by_day.items(), key=lambda x: x[1])
        best_days = [d for d, _ in sorted_days[:3]] if sorted_days else []
        worst_days = [d for d, _ in sorted_days[-3:]] if len(sorted_days) > 3 else []

        return {
            "best_hours": best_hours,
            "worst_hours": worst_hours,
            "best_days": best_days,
            "worst_days": worst_days,
            "avg_slippage_by_hour": avg_by_hour,
            "avg_slippage_by_day": avg_by_day,
            "total_fills": len(fills),
        }

    # ── Optimal order size ────────────────────────────────────────────────

    def get_optimal_order_size(self, symbol: str, current_spread_bps: float) -> dict:
        """
        Based on historical fills, recommend order size that minimizes slippage.

        Returns
        -------
        dict with keys:
            recommended_max_usd, reason, historical_slippage_at_size
        """
        fills = self._fill_history.get(symbol, [])
        sized_fills = [f for f in fills if f.size_usd > 0]

        if len(sized_fills) < 5:
            return {
                "recommended_max_usd": 500.0,
                "reason": "insufficient data — using conservative default",
                "historical_slippage_at_size": [],
            }

        # Bucket by size
        sizes = np.array([f.size_usd for f in sized_fills])
        slips = np.array([f.slippage_bps for f in sized_fills])

        # Find size threshold where slippage starts increasing
        # Split into quartiles
        quartiles = np.percentile(sizes, [25, 50, 75])
        buckets = []
        for q_lo, q_hi in zip([0] + list(quartiles), list(quartiles) + [float("inf")]):
            mask = (sizes >= q_lo) & (sizes < q_hi)
            if mask.any():
                avg_slip = float(np.mean(slips[mask]))
                avg_size = float(np.mean(sizes[mask]))
                buckets.append({
                    "avg_size_usd": round(avg_size, 2),
                    "avg_slippage_bps": round(avg_slip, 4),
                    "count": int(mask.sum()),
                })

        # Recommend: largest bucket where slippage < 2x median slippage
        median_slip = float(np.median(slips))
        recommended = float(np.median(sizes))  # default

        for bucket in buckets:
            if bucket["avg_slippage_bps"] <= median_slip * 2.0:
                recommended = max(recommended, bucket["avg_size_usd"])

        # Adjust for current spread
        spread_factor = max(0.5, 1.0 - (current_spread_bps - 5.0) / 20.0)
        recommended *= spread_factor

        reason = (
            f"based on {len(sized_fills)} fills; median slip={median_slip:.2f}bps; "
            f"spread_adj={spread_factor:.2f}"
        )

        return {
            "recommended_max_usd": round(recommended, 2),
            "reason": reason,
            "historical_slippage_at_size": buckets,
        }

    # ── Delay recommendation ──────────────────────────────────────────────

    def should_delay_execution(
        self,
        symbol: str,
        current_hour_utc: int,
        current_spread_bps: float,
    ) -> dict:
        """
        Return whether execution should be delayed based on historical patterns.

        Returns
        -------
        dict with keys:
            delay (bool), reason (str), suggested_wait_minutes (int)
        """
        fills = self._fill_history.get(symbol, [])
        if len(fills) < 10:
            return {
                "delay": False,
                "reason": "insufficient data for recommendation",
                "suggested_wait_minutes": 0,
            }

        hour = int(current_hour_utc) % 24

        # Check if current hour is a historically bad hour
        hour_slippage: Dict[int, List[float]] = defaultdict(list)
        for f in fills:
            hour_slippage[f.hour_utc].append(f.slippage_bps)

        current_hour_avg = float(np.mean(hour_slippage.get(hour, [0.0]))) if hour in hour_slippage else 0.0
        overall_avg = float(np.mean([f.slippage_bps for f in fills]))

        reasons = []
        delay = False
        wait_minutes = 0

        # Rule 1: Current hour has >2x average slippage
        if current_hour_avg > overall_avg * 2.0 and current_hour_avg > 3.0:
            delay = True
            reasons.append(
                f"hour {hour} avg slippage {current_hour_avg:.1f}bps "
                f"is {current_hour_avg/max(overall_avg, 0.01):.1f}x average"
            )
            # Find next good hour
            for offset in range(1, 6):
                next_hour = (hour + offset) % 24
                next_avg = float(np.mean(hour_slippage.get(next_hour, [overall_avg])))
                if next_avg <= overall_avg * 1.5:
                    wait_minutes = offset * 60
                    break
            if wait_minutes == 0:
                wait_minutes = 60

        # Rule 2: Current spread is very wide relative to historical
        spread_fills = [f.spread_bps for f in fills]
        median_spread = float(np.median(spread_fills))
        if current_spread_bps > median_spread * 3.0 and current_spread_bps > 10.0:
            delay = True
            reasons.append(
                f"spread {current_spread_bps:.1f}bps is "
                f"{current_spread_bps/max(median_spread, 0.01):.1f}x median ({median_spread:.1f}bps)"
            )
            wait_minutes = max(wait_minutes, 15)

        return {
            "delay": delay,
            "reason": "; ".join(reasons) if reasons else "conditions acceptable",
            "suggested_wait_minutes": wait_minutes,
        }

    # ── Snapshot ──────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Return current state for dashboard/logging."""
        return {
            "symbols_tracked": len(self._fill_history),
            "total_fills": sum(len(v) for v in self._fill_history.values()),
            "per_symbol": {
                sym: len(fills) for sym, fills in self._fill_history.items()
            },
        }
