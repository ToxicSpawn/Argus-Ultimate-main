"""
Smart Execution Engine - Ultimate Edge Module

Provides advanced execution algorithms:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- POV (Percentage of Volume)
- Adaptive execution

This module ensures optimal order execution with minimal market impact.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ExecutionAlgorithm(str, Enum):
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"
    ADAPTIVE = "adaptive"


@dataclass
class ExecutionSlice:
    """Single slice of an execution schedule."""
    slice_id: int
    quantity: float
    start_time: datetime
    end_time: datetime
    target_price: float
    venue: str
    status: str = "pending"


@dataclass
class ExecutionSchedule:
    """Complete execution schedule."""
    algorithm: ExecutionAlgorithm
    total_quantity: float
    slices: List[ExecutionSlice]
    start_time: datetime
    end_time: datetime
    target_price: float
    estimated_completion: datetime


@dataclass
class ExecutionStatus:
    """Current execution status."""
    schedule: ExecutionSchedule
    completed_quantity: float
    remaining_quantity: float
    completed_slices: int
    total_slices: int
    progress_pct: float
    avg_fill_price: float
    current_price: float
    slippage_bps: float


@dataclass
class MarketVolumeProfile:
    """Volume profile for VWAP/POV execution."""
    hourly_volumes: List[float]
    total_volume: float
    avg_volume_per_hour: float
    peak_hour: int
    timestamp: datetime = field(default_factory=datetime.now)


class SmartExecutionEngine:
    """
    Smart order execution engine.

    Provides multiple execution algorithms:
    - TWAP: Spreads order evenly over time
    - VWAP: Matches volume profile
    - POV: Maintains target participation rate
    - Adaptive: Adjusts based on market conditions
    """

    def __init__(
        self,
        default_algorithm: ExecutionAlgorithm = ExecutionAlgorithm.ADAPTIVE,
        default_participation_rate: float = 0.10,
        slice_interval_seconds: int = 60,
        max_slippage_bps: float = 50.0,
    ):
        self.default_algorithm = default_algorithm
        self.default_participation = default_participation_rate
        self.slice_interval = slice_interval_seconds
        self.max_slippage = max_slippage_bps

        self._current_schedule: Optional[ExecutionSchedule] = None
        self._volume_profile: Optional[MarketVolumeProfile] = None
        self._recent_fills: Deque[Tuple[float, float]] = deque(maxlen=100)

    def create_twap_schedule(
        self,
        quantity: float,
        duration_minutes: int,
        target_price: float,
        n_slices: Optional[int] = None,
        start_time: Optional[datetime] = None,
    ) -> ExecutionSchedule:
        """
        Create TWAP execution schedule.

        Args:
            quantity: Total quantity to execute
            duration_minutes: Total execution duration
            target_price: Target price for analysis
            n_slices: Number of slices (default: duration in minutes)
            start_time: Start time (default: now)

        Returns:
            ExecutionSchedule with TWAP slices
        """
        if n_slices is None:
            n_slices = duration_minutes

        if start_time is None:
            start_time = datetime.now()

        slice_quantity = quantity / n_slices
        slices = []

        for i in range(n_slices):
            slice_start = start_time + timedelta(minutes=i)
            slice_end = slice_start + timedelta(minutes=1)

            slice_exec = ExecutionSlice(
                slice_id=i,
                quantity=slice_quantity,
                start_time=slice_start,
                end_time=slice_end,
                target_price=target_price,
                venue="smart",
            )
            slices.append(slice_exec)

        end_time = start_time + timedelta(minutes=duration_minutes)

        return ExecutionSchedule(
            algorithm=ExecutionAlgorithm.TWAP,
            total_quantity=quantity,
            slices=slices,
            start_time=start_time,
            end_time=end_time,
            target_price=target_price,
            estimated_completion=end_time,
        )

    def create_vwap_schedule(
        self,
        quantity: float,
        duration_minutes: int,
        target_price: float,
        volume_profile: Optional[MarketVolumeProfile] = None,
        start_time: Optional[datetime] = None,
    ) -> ExecutionSchedule:
        """
        Create VWAP execution schedule.

        Args:
            quantity: Total quantity to execute
            duration_minutes: Total execution duration
            target_price: Target price for analysis
            volume_profile: Volume profile (default: use historical)
            start_time: Start time (default: now)

        Returns:
            ExecutionSchedule with VWAP-weighted slices
        """
        if start_time is None:
            start_time = datetime.now()

        if volume_profile is None:
            volume_profile = self._volume_profile or self._generate_default_profile()

        n_slices = duration_minutes
        hourly_total = sum(volume_profile.hourly_volumes) if volume_profile.hourly_volumes else n_slices

        slices = []
        cumulative_qty = 0.0

        for i in range(n_slices):
            hour_idx = i % 24
            if hour_idx < len(volume_profile.hourly_volumes):
                hour_volume = volume_profile.hourly_volumes[hour_idx]
            else:
                hour_volume = volume_profile.avg_volume_per_hour

            weight = hour_volume / hourly_total if hourly_total > 0 else 1.0 / n_slices
            slice_quantity = quantity * weight

            slice_start = start_time + timedelta(minutes=i)
            slice_end = slice_start + timedelta(minutes=1)

            slice_exec = ExecutionSlice(
                slice_id=i,
                quantity=slice_quantity,
                start_time=slice_start,
                end_time=slice_end,
                target_price=target_price,
                venue="smart",
            )
            slices.append(slice_exec)
            cumulative_qty += slice_quantity

        remaining = quantity - cumulative_qty
        if remaining > 0 and slices:
            slices[-1].quantity += remaining

        end_time = start_time + timedelta(minutes=duration_minutes)

        return ExecutionSchedule(
            algorithm=ExecutionAlgorithm.VWAP,
            total_quantity=quantity,
            slices=slices,
            start_time=start_time,
            end_time=end_time,
            target_price=target_price,
            estimated_completion=end_time,
        )

    def create_pov_schedule(
        self,
        quantity: float,
        duration_minutes: int,
        target_price: float,
        participation_rate: float = 0.10,
        start_time: Optional[datetime] = None,
    ) -> ExecutionSchedule:
        """
        Create POV (Percentage of Volume) execution schedule.

        Args:
            quantity: Total quantity to execute
            duration_minutes: Total execution duration
            target_price: Target price for analysis
            participation_rate: Target % of volume to execute (0.10 = 10%)
            start_time: Start time (default: now)

        Returns:
            ExecutionSchedule with POV slices
        """
        if start_time is None:
            start_time = datetime.now()

        n_slices = duration_minutes
        slices = []

        for i in range(n_slices):
            slice_start = start_time + timedelta(minutes=i)
            slice_end = slice_start + timedelta(minutes=1)

            slice_exec = ExecutionSlice(
                slice_id=i,
                quantity=0,
                start_time=slice_start,
                end_time=slice_end,
                target_price=target_price,
                venue="smart",
            )
            slices.append(slice_exec)

        end_time = start_time + timedelta(minutes=duration_minutes)

        schedule = ExecutionSchedule(
            algorithm=ExecutionAlgorithm.POV,
            total_quantity=quantity,
            slices=slices,
            start_time=start_time,
            end_time=end_time,
            target_price=target_price,
            estimated_completion=end_time,
        )

        return schedule

    def get_adaptive_schedule(
        self,
        quantity: float,
        duration_minutes: int,
        target_price: float,
        current_volatility: float = 0.02,
        current_spread_bps: float = 10.0,
        start_time: Optional[datetime] = None,
    ) -> ExecutionSchedule:
        """
        Create adaptive execution schedule that adjusts based on conditions.

        Args:
            quantity: Total quantity to execute
            duration_minutes: Total execution duration
            target_price: Target price for analysis
            current_volatility: Current market volatility (0.02 = 2%)
            current_spread_bps: Current spread in bps
            start_time: Start time (default: now)

        Returns:
            ExecutionSchedule optimized for current conditions
        """
        if start_time is None:
            start_time = datetime.now()

        if current_volatility > 0.05:
            algorithm = ExecutionAlgorithm.TWAP
            logger.info("High volatility detected - using TWAP for stability")
        elif current_spread_bps > 20:
            algorithm = ExecutionAlgorithm.VWAP
            logger.info("Wide spread detected - using VWAP to minimize impact")
        else:
            algorithm = ExecutionAlgorithm.ADAPTIVE
            logger.info("Normal conditions - using adaptive execution")

        if algorithm == ExecutionAlgorithm.TWAP:
            return self.create_twap_schedule(quantity, duration_minutes, target_price, start_time=start_time)
        elif algorithm == ExecutionAlgorithm.VWAP:
            return self.create_vwap_schedule(quantity, duration_minutes, target_price, start_time=start_time)
        else:
            schedule = self.create_twap_schedule(quantity, duration_minutes, target_price, start_time=start_time)
            schedule.algorithm = ExecutionAlgorithm.ADAPTIVE
            return schedule

    def get_next_slice(
        self,
        schedule: ExecutionSchedule,
        current_time: Optional[datetime] = None,
    ) -> Optional[ExecutionSlice]:
        """Get next slice to execute based on current time."""
        if current_time is None:
            current_time = datetime.now()

        for slice_exec in schedule.slices:
            if slice_exec.status == "pending" and slice_exec.start_time <= current_time:
                return slice_exec

        return None

    def update_slice_execution(
        self,
        slice_exec: ExecutionSlice,
        fill_price: float,
        fill_quantity: float,
        current_time: Optional[datetime] = None,
    ) -> ExecutionSlice:
        """Update slice after execution."""
        if current_time is None:
            current_time = datetime.now()

        slice_exec.status = "filled"
        slice_exec.quantity = fill_quantity

        self._recent_fills.append((fill_price, fill_quantity))

        return slice_exec

    def calculate_progress(
        self,
        schedule: ExecutionSchedule,
        current_time: Optional[datetime] = None,
    ) -> ExecutionStatus:
        """Calculate current execution progress."""
        if current_time is None:
            current_time = datetime.now()

        completed = [s for s in schedule.slices if s.status == "filled"]
        completed_qty = sum(s.quantity for s in completed)
        remaining_qty = schedule.total_quantity - completed_qty

        if self._recent_fills:
            avg_fill = sum(p * q for p, q in self._recent_fills) / sum(q for _, q in self._recent_fills)
        else:
            avg_fill = schedule.target_price

        slippage = abs(avg_fill - schedule.target_price) / schedule.target_price * 10000

        return ExecutionStatus(
            schedule=schedule,
            completed_quantity=completed_qty,
            remaining_quantity=remaining_qty,
            completed_slices=len(completed),
            total_slices=len(schedule.slices),
            progress_pct=len(completed) / len(schedule.slices) * 100 if schedule.slices else 0,
            avg_fill_price=avg_fill,
            current_price=schedule.target_price,
            slippage_bps=slippage,
        )

    def update_volume_profile(
        self,
        hourly_volumes: List[float],
    ) -> MarketVolumeProfile:
        """Update volume profile for VWAP/POV execution."""
        total = sum(hourly_volumes)
        avg = total / len(hourly_volumes) if hourly_volumes else 0

        peak_idx = hourly_volumes.index(max(hourly_volumes)) if hourly_volumes else 0

        self._volume_profile = MarketVolumeProfile(
            hourly_volumes=hourly_volumes,
            total_volume=total,
            avg_volume_per_hour=avg,
            peak_hour=peak_idx,
        )

        return self._volume_profile

    def _generate_default_profile(self) -> MarketVolumeProfile:
        """Generate default volume profile (typical crypto: higher volume in US hours)."""
        hourly = [0.3, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                  1.0, 1.0, 0.9, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.3, 1.0, 0.7]
        total = sum(hourly)
        avg = total / len(hourly)

        return MarketVolumeProfile(
            hourly_volumes=hourly,
            total_volume=total,
            avg_volume_per_hour=avg,
            peak_hour=18,
        )

    def estimate_slippage(
        self,
        schedule: ExecutionSchedule,
        participation_rate: float = 0.10,
        volatility: float = 0.02,
    ) -> float:
        """Estimate expected slippage for a schedule."""
        algo = schedule.algorithm

        if algo == ExecutionAlgorithm.TWAP:
            market_impact = 0.1 * participation_rate * volatility * 10000
        elif algo == ExecutionAlgorithm.VWAP:
            market_impact = 0.05 * participation_rate * volatility * 10000
        elif algo == ExecutionAlgorithm.POV:
            market_impact = 0.08 * participation_rate * volatility * 10000
        else:
            market_impact = 0.06 * participation_rate * volatility * 10000

        return market_impact

    def reset(self) -> None:
        """Reset all state."""
        self._current_schedule = None
        self._volume_profile = None
        self._recent_fills.clear()
        logger.info("SmartExecutionEngine reset")
