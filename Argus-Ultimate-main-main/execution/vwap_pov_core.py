"""
VWAP and POV (percentage of volume) execution algorithms.

Time-slice orders to match historical volume curve (VWAP) or target participation rate (POV).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def vwap_schedule(
    total_size: float,
    start_ts: float,
    end_ts: float,
    volume_curve: Optional[List[float]] = None,
    num_slices: int = 20,
) -> List[Dict[str, Any]]:
    """
    Produce (timestamp, size) slices to approximate VWAP.
    volume_curve: cumulative fraction of daily volume per slice (length num_slices); default linear.
    """
    if volume_curve is None or len(volume_curve) != num_slices:
        volume_curve = np.linspace(0, 1, num_slices + 1)[1:].tolist()
    prev = 0.0
    slices = []
    for i, frac in enumerate(volume_curve):
        t = start_ts + (end_ts - start_ts) * (i + 1) / num_slices
        slice_pct = frac - prev
        prev = frac
        sz = max(0.0, total_size * slice_pct)
        if sz > 0:
            slices.append({"ts": t, "size": sz})
    return slices


def pov_schedule(
    total_size: float,
    participation_rate: float,
    interval_sec: float,
    estimated_volume_per_interval: float,
    num_slices: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    POV: each slice = participation_rate * estimated_volume in that interval.
    total_size is capped by sum of slices; slices may be adjusted to not exceed total_size.
    """
    if num_slices is None:
        num_slices = max(1, int(3600 / interval_sec))
    slice_vol = max(1e-9, estimated_volume_per_interval)
    slice_size = participation_rate * slice_vol
    n = max(1, int(total_size / slice_size))
    n = min(n, num_slices)
    base_slice = total_size / n
    return [{"ts": 0.0, "size": base_slice}] * n


def run_vwap_slicer(
    total_size: float,
    duration_sec: float,
    execute_slice: Callable[[float, float], Any],
    volume_curve: Optional[List[float]] = None,
    num_slices: int = 20,
    fee_rate: float = 0.001,
) -> Dict[str, Any]:
    """
    Run VWAP by calling execute_slice(ts, size) for each slice.
    Returns summary: filled_total, avg_price, slippage_estimate.
    """
    start = time.perf_counter()
    end_ts = start + duration_sec
    schedule = vwap_schedule(total_size, start, end_ts, volume_curve=volume_curve, num_slices=num_slices)
    filled_total = 0.0
    cost_total = 0.0
    for s in schedule:
        result = execute_slice(s["ts"], s["size"])
        if isinstance(result, dict):
            filled_total += result.get("filled", 0.0)
            cost_total += result.get("filled", 0.0) * result.get("avg_price", 0.0)
        else:
            filled_total += s["size"]
            cost_total += s["size"] * s.get("price", 0.0) * fee_rate
    avg_price = cost_total / filled_total if filled_total > 0 else 0.0
    return {"filled_total": filled_total, "avg_price": avg_price, "slices": len(schedule)}
