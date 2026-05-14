"""
TWAP (Time-Weighted Average Price) order slicer.

Splits a single order into N equal-sized slices over a duration, with equal time spacing.
Use for large orders to reduce market impact (reference: Hummingbot TWAP executor).
"""

from __future__ import annotations

from typing import Any, Dict, List


def twap_schedule(
    total_quantity: float,
    duration_seconds: float,
    num_slices: int,
) -> List[Dict[str, Any]]:
    """
    Return a list of { "delay_seconds": float, "size": float } for TWAP execution.
    Slices are equal size and evenly spaced in time.
    """
    if total_quantity <= 0 or duration_seconds <= 0 or num_slices < 1:
        return []
    n = max(1, int(num_slices))
    size_per_slice = total_quantity / n
    interval = duration_seconds / n
    schedule = []
    for i in range(n):
        schedule.append({
            "delay_seconds": i * interval,
            "size": size_per_slice,
        })
    return schedule
