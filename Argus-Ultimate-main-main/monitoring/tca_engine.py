"""
Transaction Cost Analysis (TCA) Engine - Ultimate Edge Module

Provides comprehensive cost analysis:
- Implementation shortfall
- Execution quality metrics
- Venue analysis
- Market impact estimation
- Slippage analysis

This module minimizes trading costs through detailed analysis.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VenueType(str, Enum):
    EXCHANGE = "exchange"
    MARKET_MAKER = "market_maker"
    DARK_POOL = "dark_pool"
    BROKER = "broker"


@dataclass
class TradeExecution:
    """Single trade execution record."""
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    venue: str
    timestamp: datetime
    arrival_price: float = 0.0
    commission: float = 0.0
    slippage_bps: float = 0.0


@dataclass
class ExecutionQuality:
    """Execution quality metrics."""
    implementation_shortfall_bps: float = 0.0
    realized_slippage_bps: float = 0.0
    market_impact_bps: float = 0.0
    timing_cost_bps: float = 0.0
    venue_cost_bps: float = 0.0
    total_cost_bps: float = 0.0
    arrival_price: float = 0.0
    fill_price: float = 0.0
    participation_rate: float = 0.0


@dataclass
class VenueMetrics:
    """Performance metrics for a venue."""
    venue: str
    avg_slippage_bps: float = 0.0
    avg_market_impact_bps: float = 0.0
    fill_rate: float = 0.0
    avg_latency_ms: float = 0.0
    total_trades: int = 0
    total_volume: float = 0.0


@dataclass
class TCAReport:
    """Full TCA report."""
    symbol: str
    period_start: datetime
    period_end: datetime
    total_trades: int
    total_volume: float
    avg_slippage_bps: float
    avg_implementation_shortfall_bps: float
    worst_trade: ExecutionQuality
    best_trade: ExecutionQuality
    venue_rankings: List[VenueMetrics]
    cost_breakdown: Dict[str, float]


class TCAEngine:
    """
    Transaction Cost Analysis engine.

    Provides:
    - Real-time execution quality tracking
    - Implementation shortfall calculation
    - Venue performance analysis
    - Market impact estimation
    - Cost attribution
    """

    def __init__(
        self,
        symbol: str = "BTC/USD",
        lookback_trades: int = 100,
    ):
        self.symbol = symbol
        self.lookback = lookback_trades

        self._executions: Deque[TradeExecution] = deque(maxlen=lookback_trades)
        self._venue_metrics: Dict[str, VenueMetrics] = {}
        self._total_costs_bps: Deque[float] = deque(maxlen=lookback_trades)

    def record_execution(
        self,
        order_id: str,
        side: str,
        quantity: float,
        fill_price: float,
        venue: str,
        arrival_price: float,
        commission: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> ExecutionQuality:
        """
        Record a trade execution and calculate quality metrics.

        Args:
            order_id: Order identifier
            side: BUY or SELL
            quantity: Fill quantity
            fill_price: Actual fill price
            venue: Execution venue
            arrival_price: Price when order was placed
            commission: Commission paid
            timestamp: Execution timestamp

        Returns:
            ExecutionQuality with all cost metrics
        """
        if timestamp is None:
            timestamp = datetime.now()

        slippage_bps = ((fill_price - arrival_price) / arrival_price) * 10000
        if side.upper() == "SELL":
            slippage_bps = -slippage_bps

        execution = TradeExecution(
            order_id=order_id,
            symbol=self.symbol,
            side=side,
            quantity=quantity,
            price=fill_price,
            venue=venue,
            timestamp=timestamp,
            arrival_price=arrival_price,
            commission=commission,
            slippage_bps=slippage_bps,
        )
        self._executions.append(execution)

        self._update_venue_metrics(venue, slippage_bps, commission, quantity)
        self._total_costs_bps.append(abs(slippage_bps))

        quality = self._calculate_execution_quality(execution)
        return quality

    def _update_venue_metrics(
        self,
        venue: str,
        slippage_bps: float,
        commission: float,
        quantity: float,
    ) -> None:
        """Update metrics for a specific venue."""
        if venue not in self._venue_metrics:
            self._venue_metrics[venue] = VenueMetrics(venue=venue)

        m = self._venue_metrics[venue]
        n = m.total_trades

        new_slippage = (m.avg_slippage_bps * n + abs(slippage_bps)) / (n + 1)
        m.avg_slippage_bps = new_slippage
        m.total_trades += 1
        m.total_volume += quantity

    def _calculate_execution_quality(self, execution: TradeExecution) -> ExecutionQuality:
        """Calculate execution quality metrics."""
        slippage = abs(execution.slippage_bps)

        implementation_shortfall = abs(execution.slippage_bps)

        market_impact = slippage * 0.3
        timing_cost = slippage * 0.4
        venue_cost = slippage * 0.3

        return ExecutionQuality(
            implementation_shortfall_bps=implementation_shortfall,
            realized_slippage_bps=slippage,
            market_impact_bps=market_impact,
            timing_cost_bps=timing_cost,
            venue_cost_bps=venue_cost,
            total_cost_bps=slippage,
            arrival_price=execution.arrival_price,
            fill_price=execution.price,
            participation_rate=0.0,
        )

    def analyze_trade(
        self,
        side: str,
        quantity: float,
        arrival_price: float,
        fill_price: float,
        venue: str,
        commission: float = 0.0,
    ) -> ExecutionQuality:
        """
        Analyze a trade and return execution quality.

        Args:
            side: BUY or SELL
            quantity: Order quantity
            arrival_price: Price when decision was made
            fill_price: Actual execution price
            venue: Venue used
            commission: Commission paid

        Returns:
            ExecutionQuality metrics
        """
        slippage_bps = ((fill_price - arrival_price) / arrival_price) * 10000
        if side.upper() == "SELL":
            slippage_bps = -slippage_bps

        slippage = abs(slippage_bps)
        implementation_shortfall = abs(slippage_bps)

        market_impact = slippage * 0.3
        timing_cost = slippage * 0.4
        venue_cost = slippage * 0.3

        return ExecutionQuality(
            implementation_shortfall_bps=implementation_shortfall,
            realized_slippage_bps=slippage,
            market_impact_bps=market_impact,
            timing_cost_bps=timing_cost,
            venue_cost_bps=venue_cost,
            total_cost_bps=slippage,
            arrival_price=arrival_price,
            fill_price=fill_price,
            participation_rate=0.0,
        )

    def get_venue_rankings(self) -> List[VenueMetrics]:
        """Get venue performance rankings (best to worst)."""
        venues = list(self._venue_metrics.values())
        return sorted(venues, key=lambda x: x.avg_slippage_bps)

    def get_best_venue(self) -> Optional[str]:
        """Get the best performing venue by slippage."""
        rankings = self.get_venue_rankings()
        return rankings[0].venue if rankings else None

    def get_worst_venue(self) -> Optional[str]:
        """Get the worst performing venue by slippage."""
        rankings = self.get_venue_rankings()
        return rankings[-1].venue if rankings else None

    def get_avg_cost_bps(self) -> float:
        """Get average total cost in bps."""
        if not self._total_costs_bps:
            return 0.0
        return np.mean(list(self._total_costs_bps))

    def generate_report(
        self,
        period_hours: int = 24,
    ) -> TCAReport:
        """
        Generate comprehensive TCA report.

        Args:
            period_hours: Report period in hours

        Returns:
            TCAReport with full analysis
        """
        now = datetime.now()
        cutoff = now.timestamp() - (period_hours * 3600)

        recent_executions = [
            e for e in self._executions
            if e.timestamp.timestamp() >= cutoff
        ]

        if not recent_executions:
            return TCAReport(
                symbol=self.symbol,
                period_start=now,
                period_end=now,
                total_trades=0,
                total_volume=0.0,
                avg_slippage_bps=0.0,
                avg_implementation_shortfall_bps=0.0,
                worst_trade=ExecutionQuality(),
                best_trade=ExecutionQuality(),
                venue_rankings=[],
                cost_breakdown={},
            )

        total_volume = sum(e.quantity for e in recent_executions)
        slippages = [abs(e.slippage_bps) for e in recent_executions]
        avg_slippage = np.mean(slippages) if slippages else 0.0

        qualities = [self._calculate_execution_quality(e) for e in recent_executions]
        avg_shortfall = np.mean([q.implementation_shortfall_bps for q in qualities]) if qualities else 0.0

        worst_idx = np.argmax(slippages)
        best_idx = np.argmin(slippages)

        venue_rankings = self.get_venue_rankings()

        cost_breakdown = {
            "market_impact": avg_slippage * 0.3,
            "timing": avg_slippage * 0.4,
            "venue": avg_slippage * 0.3,
            "commission": sum(e.commission for e in recent_executions) / total_volume if total_volume > 0 else 0.0,
        }

        return TCAReport(
            symbol=self.symbol,
            period_start=recent_executions[0].timestamp,
            period_end=recent_executions[-1].timestamp,
            total_trades=len(recent_executions),
            total_volume=total_volume,
            avg_slippage_bps=avg_slippage,
            avg_implementation_shortfall_bps=avg_shortfall,
            worst_trade=qualities[worst_idx],
            best_trade=qualities[best_idx],
            venue_rankings=venue_rankings,
            cost_breakdown=cost_breakdown,
        )

    def get_cost_warning(self) -> Optional[str]:
        """Get warning if costs are too high."""
        avg_cost = self.get_avg_cost_bps()

        if avg_cost > 50:
            return f"CRITICAL: Average slippage {avg_cost:.1f} bps exceeds 50 bps threshold"
        elif avg_cost > 25:
            return f"WARNING: Average slippage {avg_cost:.1f} bps exceeds 25 bps threshold"
        elif avg_cost > 15:
            return f"CAUTION: Average slippage {avg_cost:.1f} bps exceeds 15 bps threshold"

        return None

    def reset(self) -> None:
        """Reset all state."""
        self._executions.clear()
        self._venue_metrics.clear()
        self._total_costs_bps.clear()
        logger.info("TCAEngine reset")
