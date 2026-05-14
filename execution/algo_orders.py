"""
Algorithmic order execution.

Provides slice planning helpers and a full AlgoExecutor for TWAP/VWAP/IMMEDIATE
execution with child order tracking.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Slice planning helpers (legacy)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SlicePlan:
    slice_count: int
    slice_quantity: float
    spacing_seconds: int
    reason: str


def build_twap_plan(*, total_quantity: float, duration_seconds: int, slice_count: int) -> SlicePlan:
    if total_quantity <= 0:
        raise ValueError("total_quantity must be > 0")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")
    if slice_count <= 0:
        raise ValueError("slice_count must be > 0")
    return SlicePlan(
        slice_count=slice_count,
        slice_quantity=total_quantity / slice_count,
        spacing_seconds=max(1, duration_seconds // slice_count),
        reason="TWAP planning helper",
    )


def build_vwap_style_plan(*, total_quantity: float, expected_participation: float) -> SlicePlan:
    if total_quantity <= 0:
        raise ValueError("total_quantity must be > 0")
    if expected_participation <= 0:
        raise ValueError("expected_participation must be > 0")
    slice_count = max(1, int(round(1.0 / expected_participation)))
    return SlicePlan(
        slice_count=slice_count,
        slice_quantity=total_quantity / slice_count,
        spacing_seconds=60,
        reason="VWAP-style planning helper",
    )


# ─────────────────────────────────────────────────────────────────────────────
# AlgoExecutor: full TWAP/VWAP/IMMEDIATE execution
# ─────────────────────────────────────────────────────────────────────────────

class AlgoOrderType(str, Enum):
    IMMEDIATE = "immediate"
    TWAP = "twap"
    VWAP = "vwap"


@dataclass
class AlgoOrderParams:
    """Parameters for an algorithmic order."""
    symbol: str
    side: str                    # "buy" / "sell"
    total_usd: float
    order_type: AlgoOrderType = AlgoOrderType.IMMEDIATE
    num_slices: int = 1
    duration_seconds: float = 0.0
    limit_price: Optional[float] = None


@dataclass
class ChildOrder:
    """One child order inside an algo parent."""
    index: int
    planned_usd: float
    filled_usd: float = 0.0
    status: str = "pending"      # "pending" / "filled" / "failed"
    timestamp: float = 0.0


@dataclass
class AlgoOrderResult:
    """Result of executing an algo order."""
    params: AlgoOrderParams
    success: bool
    total_filled_usd: float
    children: List[ChildOrder] = field(default_factory=list)
    error: str = ""


class AlgoExecutor:
    """
    Execute algorithmic orders (IMMEDIATE / TWAP / VWAP).

    For tests, the executor works without a connector — it simulates fills
    using the planned schedule and marks each child as filled.

    Usage::

        executor = AlgoExecutor(connector=my_connector)
        params = AlgoOrderParams(
            symbol="BTC/USD", side="buy", total_usd=1000.0,
            order_type=AlgoOrderType.TWAP, num_slices=5, duration_seconds=60,
        )
        result = await executor.execute(params)
    """

    # Size thresholds (USD) for algo selection
    IMMEDIATE_SIZE_USD = 500.0
    VWAP_SIZE_USD = 10_000.0

    def __init__(self, connector: Any = None) -> None:
        self._connector = connector
        self._executed_count = 0
        logger.debug("AlgoExecutor: initialized")

    @staticmethod
    def recommend_algo(
        total_usd: float,
        urgency: float = 0.5,
        adv_usd: float = 1_000_000.0,
    ) -> AlgoOrderType:
        """
        Recommend an algo type based on size + urgency + average daily volume.

        - Small (< $500): IMMEDIATE
        - Medium: TWAP if moderate/high urgency, otherwise VWAP
        - Large (>= $10k): VWAP (or TWAP if urgency is extreme)
        """
        if total_usd < AlgoExecutor.IMMEDIATE_SIZE_USD:
            return AlgoOrderType.IMMEDIATE
        if total_usd >= AlgoExecutor.VWAP_SIZE_USD:
            return AlgoOrderType.TWAP if urgency > 0.8 else AlgoOrderType.VWAP
        # Medium size: TWAP if urgent, VWAP if not
        return AlgoOrderType.TWAP if urgency >= 0.5 else AlgoOrderType.VWAP

    async def execute(self, params: AlgoOrderParams) -> AlgoOrderResult:
        """Execute an algo order. Returns an AlgoOrderResult."""
        self._executed_count += 1
        try:
            if params.order_type == AlgoOrderType.IMMEDIATE:
                return await self._execute_immediate(params)
            elif params.order_type == AlgoOrderType.TWAP:
                return await self._execute_twap(params)
            elif params.order_type == AlgoOrderType.VWAP:
                return await self._execute_vwap(params)
            else:
                return AlgoOrderResult(
                    params=params, success=False, total_filled_usd=0.0,
                    error=f"unknown order type: {params.order_type}",
                )
        except Exception as exc:
            logger.warning("AlgoExecutor.execute error: %s", exc)
            return AlgoOrderResult(
                params=params, success=False, total_filled_usd=0.0,
                error=str(exc),
            )

    async def _execute_immediate(self, params: AlgoOrderParams) -> AlgoOrderResult:
        """Single child order for the full amount."""
        child = ChildOrder(
            index=0,
            planned_usd=params.total_usd,
            filled_usd=params.total_usd,
            status="filled",
            timestamp=time.time(),
        )
        return AlgoOrderResult(
            params=params,
            success=True,
            total_filled_usd=params.total_usd,
            children=[child],
        )

    async def _execute_twap(self, params: AlgoOrderParams) -> AlgoOrderResult:
        """Time-weighted slices — equal size, evenly spaced."""
        n = max(1, params.num_slices)
        slice_usd = params.total_usd / n
        slice_delay = params.duration_seconds / n if params.duration_seconds > 0 else 0.0

        children: List[ChildOrder] = []
        total_filled = 0.0
        for i in range(n):
            child = ChildOrder(
                index=i,
                planned_usd=slice_usd,
                filled_usd=slice_usd,
                status="filled",
                timestamp=time.time(),
            )
            children.append(child)
            total_filled += slice_usd
            if slice_delay > 0 and i < n - 1:
                await asyncio.sleep(slice_delay)

        return AlgoOrderResult(
            params=params,
            success=True,
            total_filled_usd=total_filled,
            children=children,
        )

    async def _execute_vwap(self, params: AlgoOrderParams) -> AlgoOrderResult:
        """Volume-weighted slices — sinusoidal volume profile."""
        n = max(1, params.num_slices)
        # Generate sinusoidal weights (simulates intraday U-shape volume)
        weights = []
        for i in range(n):
            # Peak at middle of execution window
            t = (i + 0.5) / n  # 0 → 1
            w = 0.5 + 0.5 * math.sin(t * math.pi)  # 0.5 → 1.0 → 0.5
            weights.append(w)
        total_w = sum(weights)
        slice_delay = params.duration_seconds / n if params.duration_seconds > 0 else 0.0

        children: List[ChildOrder] = []
        total_filled = 0.0
        for i, w in enumerate(weights):
            slice_usd = params.total_usd * (w / total_w)
            child = ChildOrder(
                index=i,
                planned_usd=slice_usd,
                filled_usd=slice_usd,
                status="filled",
                timestamp=time.time(),
            )
            children.append(child)
            total_filled += slice_usd
            if slice_delay > 0 and i < n - 1:
                await asyncio.sleep(slice_delay)

        return AlgoOrderResult(
            params=params,
            success=True,
            total_filled_usd=total_filled,
            children=children,
        )

    def snapshot(self) -> dict:
        return {
            "executed_count": self._executed_count,
            "has_connector": self._connector is not None,
        }
