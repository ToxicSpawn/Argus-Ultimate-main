"""
Adaptive TWAP/VWAP Execution Engine.

Creates time-sliced execution plans and dynamically adjusts slice sizes
based on real-time volume participation.  Supports two styles:

* **TWAP** — equal-sized slices at fixed intervals.
* **VWAP** — slices weighted by a volume profile (crypto U-shaped: higher
  volume at the start and end of each hour).

Usage::

    twap = AdaptiveTWAP()
    plan = twap.create_plan("BTC/AUD", total_size=0.5, duration_s=300, style="vwap")
    slice_ = twap.get_next_slice(plan.plan_id, current_volume=120_000)
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TWAPSlice:
    """A single child-order slice within a TWAP/VWAP plan."""

    time_offset_s: float          # seconds from plan start
    size: float                   # base-currency quantity for this slice
    price_limit: Optional[float] = None   # optional limit price
    executed: bool = False
    actual_fill_price: Optional[float] = None
    actual_fill_size: Optional[float] = None


@dataclass
class TWAPPlan:
    """A complete TWAP/VWAP execution plan."""

    plan_id: str
    symbol: str
    slices: List[TWAPSlice]
    total_size: float
    style: str                    # "twap" or "vwap"
    duration_s: float
    created_at: float
    completed: bool = False
    _next_idx: int = 0           # internal: next slice to dispatch


# Crypto intra-hour volume profile weights (60 buckets, one per minute).
# U-shaped: higher at minute 0 and minute 59, lower in the middle.
def _build_volume_profile(n_buckets: int = 60) -> List[float]:
    """Build a U-shaped volume distribution over *n_buckets* intervals."""
    mid = (n_buckets - 1) / 2.0
    raw = []
    for i in range(n_buckets):
        dist = abs(i - mid) / mid   # 0 at centre, 1 at edges
        raw.append(0.5 + 0.5 * dist)
    total = sum(raw)
    return [w / total for w in raw]


_VOLUME_PROFILE = _build_volume_profile()


class AdaptiveTWAP:
    """Creates and manages adaptive TWAP/VWAP execution plans.

    Parameters
    ----------
    default_slices : int
        Default number of slices when not specified.
    max_participation_rate : float
        Maximum fraction of current market volume a single slice should
        represent (used for VWAP volume adaptation).
    """

    def __init__(
        self,
        default_slices: int = 10,
        max_participation_rate: float = 0.10,
    ) -> None:
        self._default_slices = max(2, default_slices)
        self._max_participation = max_participation_rate
        self._lock = threading.Lock()
        self._plans: Dict[str, TWAPPlan] = {}
        logger.info("AdaptiveTWAP initialised — default_slices=%d max_participation=%.2f",
                     self._default_slices, self._max_participation)

    # ------------------------------------------------------------------
    # Plan creation
    # ------------------------------------------------------------------

    def create_plan(
        self,
        symbol: str,
        total_size: float,
        duration_s: float,
        style: str = "twap",
        num_slices: Optional[int] = None,
        price_limit: Optional[float] = None,
    ) -> TWAPPlan:
        """Create a new TWAP or VWAP execution plan.

        Parameters
        ----------
        symbol : str
            Trading pair.
        total_size : float
            Total base-currency quantity to execute.
        duration_s : float
            Execution window in seconds.
        style : str
            ``"twap"`` for equal slices or ``"vwap"`` for volume-weighted.
        num_slices : int | None
            Number of child-order slices.  Defaults to ``default_slices``.
        price_limit : float | None
            Optional worst-case price limit applied to every slice.

        Returns
        -------
        TWAPPlan
        """
        if style not in ("twap", "vwap"):
            raise ValueError(f"Unknown style: {style!r}. Use 'twap' or 'vwap'.")
        if total_size <= 0:
            raise ValueError("total_size must be positive")
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")

        n = num_slices or self._default_slices
        n = max(1, n)
        interval = duration_s / n

        slices: List[TWAPSlice] = []
        if style == "twap":
            slice_size = total_size / n
            for i in range(n):
                slices.append(TWAPSlice(
                    time_offset_s=i * interval,
                    size=slice_size,
                    price_limit=price_limit,
                ))
        else:
            # VWAP: use the volume profile to weight slices.
            weights = self._vwap_weights(n)
            for i in range(n):
                slices.append(TWAPSlice(
                    time_offset_s=i * interval,
                    size=total_size * weights[i],
                    price_limit=price_limit,
                ))

        plan_id = uuid.uuid4().hex[:12]
        plan = TWAPPlan(
            plan_id=plan_id,
            symbol=symbol,
            slices=slices,
            total_size=total_size,
            style=style,
            duration_s=duration_s,
            created_at=time.time(),
        )

        with self._lock:
            self._plans[plan_id] = plan

        logger.info("Created %s plan %s: %s size=%.6f duration=%ds slices=%d",
                     style.upper(), plan_id, symbol, total_size, int(duration_s), n)
        return plan

    # ------------------------------------------------------------------
    # Slice dispatch
    # ------------------------------------------------------------------

    def get_next_slice(
        self,
        plan_id: str,
        current_volume: Optional[float] = None,
    ) -> Optional[TWAPSlice]:
        """Return the next un-executed slice for *plan_id*.

        If *current_volume* is provided and the plan is VWAP, the slice size
        is scaled to stay within the participation-rate limit.

        Returns ``None`` when all slices have been dispatched.
        """
        with self._lock:
            plan = self._plans.get(plan_id)
            if plan is None:
                logger.warning("Unknown plan_id: %s", plan_id)
                return None
            if plan.completed:
                return None

            idx = plan._next_idx
            if idx >= len(plan.slices):
                plan.completed = True
                return None

            sl = plan.slices[idx]

            # Volume-adaptive sizing for VWAP.
            if plan.style == "vwap" and current_volume is not None and current_volume > 0:
                max_size = current_volume * self._max_participation
                if sl.size > max_size:
                    logger.debug("Plan %s slice %d capped: %.6f -> %.6f (participation limit)",
                                 plan_id, idx, sl.size, max_size)
                    sl.size = max_size

            plan._next_idx = idx + 1
            if plan._next_idx >= len(plan.slices):
                plan.completed = True

        return sl

    def mark_executed(
        self,
        plan_id: str,
        slice_idx: int,
        fill_price: float,
        fill_size: float,
    ) -> None:
        """Mark a specific slice as executed with fill details."""
        with self._lock:
            plan = self._plans.get(plan_id)
            if plan is None:
                return
            if 0 <= slice_idx < len(plan.slices):
                sl = plan.slices[slice_idx]
                sl.executed = True
                sl.actual_fill_price = fill_price
                sl.actual_fill_size = fill_size

    # ------------------------------------------------------------------
    # Volume adaptation
    # ------------------------------------------------------------------

    def adjust_for_volume(
        self,
        plan_id: str,
        current_volume: float,
        avg_volume: float,
    ) -> None:
        """Redistribute remaining slices based on volume participation.

        If *current_volume* is below *avg_volume*, remaining slices are shrunk
        (slow down).  If above, they are expanded (speed up) up to the
        participation-rate cap.
        """
        if avg_volume <= 0:
            return

        ratio = current_volume / avg_volume
        with self._lock:
            plan = self._plans.get(plan_id)
            if plan is None or plan.completed:
                return

            remaining = [s for s in plan.slices[plan._next_idx:] if not s.executed]
            if not remaining:
                return

            # Compute remaining quantity.
            executed_qty = sum(
                s.actual_fill_size or s.size
                for s in plan.slices[:plan._next_idx]
                if s.executed
            )
            remaining_qty = max(0.0, plan.total_size - executed_qty)
            if remaining_qty <= 0:
                plan.completed = True
                return

            # Scale factor clamped to [0.5, 2.0].
            scale = max(0.5, min(2.0, ratio))
            base_size = remaining_qty / len(remaining)
            for sl in remaining:
                sl.size = base_size * scale

        logger.debug("Plan %s volume adjustment: ratio=%.2f scale=%.2f remaining_qty=%.6f",
                      plan_id, ratio, scale, remaining_qty)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_plan(self, plan_id: str) -> Optional[TWAPPlan]:
        """Return a plan by ID."""
        with self._lock:
            return self._plans.get(plan_id)

    def active_plans(self) -> List[TWAPPlan]:
        """Return all non-completed plans."""
        with self._lock:
            return [p for p in self._plans.values() if not p.completed]

    def cancel_plan(self, plan_id: str) -> bool:
        """Cancel a plan (mark completed, do not dispatch further slices)."""
        with self._lock:
            plan = self._plans.get(plan_id)
            if plan is None:
                return False
            plan.completed = True
        logger.info("Plan %s cancelled", plan_id)
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _vwap_weights(n: int) -> List[float]:
        """Generate *n* VWAP weights from the volume profile."""
        if n <= 0:
            return []
        if n == 1:
            return [1.0]

        profile = _VOLUME_PROFILE
        # Resample the 60-bucket profile to *n* buckets.
        weights: List[float] = []
        for i in range(n):
            # Map slice index to profile index.
            frac = i / (n - 1) if n > 1 else 0.5
            pidx = frac * (len(profile) - 1)
            lo = int(pidx)
            hi = min(lo + 1, len(profile) - 1)
            t = pidx - lo
            weights.append(profile[lo] * (1 - t) + profile[hi] * t)

        w_sum = sum(weights)
        return [w / w_sum for w in weights] if w_sum > 0 else [1.0 / n] * n
