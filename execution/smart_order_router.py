"""
Smart Order Router — intelligent order routing across venues and strategies.

Routes orders to optimal execution venues based on:
- Liquidity depth across venues
- Fee structures
- Latency considerations
- Historical fill rates
- Market impact estimation

Features:
- Multi-venue order splitting
- TWAP/VWAP execution algorithms
- Iceberg order simulation
- Venue health monitoring
- Cost analysis and optimization

Example::

    router = SmartOrderRouter()
    router.register_venue("binance", fee=0.001, latency_ms=50, fill_rate=0.95)
    router.register_venue("kraken", fee=0.0016, latency_ms=80, fill_rate=0.90)
    
    plan = router.create_execution_plan("BTC/USD", side="buy", amount=100, quote=5000000)
    print(plan.venues, plan.estimated_cost)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VenueConfig:
    """Exchange/venue configuration."""
    venue_id: str
    fee_maker: float  # Maker fee (e.g., 0.001 = 0.1%)
    fee_taker: float  # Taker fee
    latency_ms: float  # Average latency
    fill_rate: float  # Historical fill rate (0-1)
    max_order_size_usd: float
    supports_iceberg: bool
    supports_twap: bool
    health_score: float  # 0-1, venue health
    enabled: bool = True


@dataclass
class VenueLiquidity:
    """Venue liquidity snapshot."""
    venue_id: str
    symbol: str
    bid_depth_usd: float
    ask_depth_usd: float
    spread_bps: float
    timestamp: float


@dataclass
class OrderSlice:
    """Single order slice in execution plan."""
    venue_id: str
    symbol: str
    side: str
    amount: float
    price_limit: Optional[float]
    order_type: str  # "limit", "market", "iceberg"
    time_in_force: str  # "GTC", "IOC", "FOK"
    estimated_fee: float
    estimated_fill_time_ms: float


@dataclass
class ExecutionPlan:
    """Complete order execution plan."""
    symbol: str
    side: str
    total_amount: float
    target_price: float
    strategy: str  # "aggressive", "balanced", "passive", "twap", "iceberg"
    slices: List[OrderSlice]
    venues: List[str]
    estimated_total_fee: float
    estimated_fill_time_ms: float
    estimated_slippage_pct: float
    market_impact_pct: float
    reasoning: List[str] = field(default_factory=list)


@dataclass
class _VenueState:
    liquidity: Deque[VenueLiquidity] = field(
        default_factory=lambda: deque(maxlen=1000)
    )
    recent_fills: Deque[bool] = field(
        default_factory=lambda: deque(maxlen=100)
    )
    last_health_check: float = 0.0


class SmartOrderRouter:
    """
    Intelligent order routing across multiple venues.

    Parameters
    ----------
    default_strategy : str
        Default execution strategy (default "balanced").
    max_venues_per_order : int
        Maximum venues to split across (default 3).
    min_slice_size_usd : float
        Minimum slice size to avoid dust (default $100).
    health_check_interval : int
        Seconds between venue health checks (default 60).
    """

    def __init__(
        self,
        default_strategy: str = "balanced",
        max_venues_per_order: int = 3,
        min_slice_size_usd: float = 100.0,
        health_check_interval: int = 60,
    ) -> None:
        self._default_strategy = default_strategy
        self._max_venues = max_venues_per_order
        self._min_slice = min_slice_size_usd
        self._health_interval = health_check_interval
        
        self._venues: Dict[str, VenueConfig] = {}
        self._venue_states: Dict[str, _VenueState] = {}
        self._liquidity_data: Dict[str, Dict[str, VenueLiquidity]] = {}  # symbol -> venue -> liquidity

        logger.info(
            "SmartOrderRouter initialized: strategy=%s max_venues=%d",
            default_strategy, max_venues_per_order,
        )

    def register_venue(
        self,
        venue_id: str,
        fee_maker: float,
        fee_taker: float,
        latency_ms: float,
        fill_rate: float = 0.95,
        max_order_size_usd: float = 1_000_000,
        supports_iceberg: bool = False,
        supports_twap: bool = False,
    ) -> None:
        """Register a trading venue."""
        self._venues[venue_id] = VenueConfig(
            venue_id=venue_id,
            fee_maker=fee_maker,
            fee_taker=fee_taker,
            latency_ms=latency_ms,
            fill_rate=fill_rate,
            max_order_size_usd=max_order_size_usd,
            supports_iceberg=supports_iceberg,
            supports_twap=supports_twap,
            health_score=1.0,
        )
        self._venue_states[venue_id] = _VenueState()
        logger.info("Registered venue: %s (maker=%.3f%% taker=%.3f%% latency=%.0fms)",
                    venue_id, fee_maker * 100, fee_taker * 100, latency_ms)

    def update_liquidity(
        self,
        venue_id: str,
        symbol: str,
        bid_depth_usd: float,
        ask_depth_usd: float,
        spread_bps: float,
    ) -> None:
        """Update venue liquidity data."""
        if venue_id not in self._venue_states:
            return

        state = self._venue_states[venue_id]
        liquidity = VenueLiquidity(
            venue_id=venue_id,
            symbol=symbol,
            bid_depth_usd=bid_depth_usd,
            ask_depth_usd=ask_depth_usd,
            spread_bps=spread_bps,
            timestamp=time.time(),
        )
        state.liquidity.append(liquidity)

        if symbol not in self._liquidity_data:
            self._liquidity_data[symbol] = {}
        self._liquidity_data[symbol][venue_id] = liquidity

    def update_fill_result(self, venue_id: str, filled: bool) -> None:
        """Update venue fill result for health tracking."""
        if venue_id in self._venue_states:
            self._venue_states[venue_id].recent_fills.append(filled)
            self._update_venue_health(venue_id)

    def _update_venue_health(self, venue_id: str) -> None:
        """Update venue health score based on recent fills."""
        state = self._venue_states[venue_id]
        fills = list(state.recent_fills)
        
        if not fills:
            return

        # Calculate fill rate
        fill_rate = sum(fills) / len(fills)
        
        # Update venue config
        if venue_id in self._venues:
            venue = self._venues[venue_id]
            # Smooth update
            venue.health_score = venue.health_score * 0.8 + fill_rate * 0.2

    def _get_venue_score(
        self,
        venue_id: str,
        symbol: str,
        side: str,
        amount_usd: float,
    ) -> float:
        """Calculate venue score for order routing."""
        if venue_id not in self._venues:
            return 0.0

        venue = self._venues[venue_id]
        if not venue.enabled:
            return 0.0

        # Check size limit
        if amount_usd > venue.max_order_size_usd:
            return 0.0

        score = 0.0

        # Fee score (lower is better)
        fee = venue.fee_taker  # Assume taker for simplicity
        fee_score = max(0, 1 - fee * 100)  # Normalize
        score += fee_score * 0.3

        # Latency score (lower is better)
        latency_score = max(0, 1 - venue.latency_ms / 500)
        score += latency_score * 0.2

        # Fill rate score
        score += venue.fill_rate * 0.2

        # Health score
        score += venue.health_score * 0.2

        # Liquidity score
        liquidity = self._liquidity_data.get(symbol, {}).get(venue_id)
        if liquidity:
            depth = liquidity.ask_depth_usd if side == "buy" else liquidity.bid_depth_usd
            liquidity_score = min(1.0, depth / amount_usd)
            score += liquidity_score * 0.1

        return score

    def create_execution_plan(
        self,
        symbol: str,
        side: str,
        amount_usd: float,
        target_price: Optional[float] = None,
        strategy: Optional[str] = None,
        urgency: str = "normal",  # "low", "normal", "high", "urgent"
    ) -> Optional[ExecutionPlan]:
        """
        Create optimal execution plan for order.

        Parameters
        ----------
        symbol : str
            Trading pair.
        side : str
            "buy" or "sell".
        amount_usd : float
            Order value in USD.
        target_price : float, optional
            Target execution price.
        strategy : str, optional
            Execution strategy override.
        urgency : str
            Order urgency level.

        Returns
        -------
        ExecutionPlan or None if no venues available.
        """
        strategy = strategy or self._default_strategy
        
        # Adjust strategy based on urgency
        if urgency == "urgent":
            strategy = "aggressive"
        elif urgency == "low":
            strategy = "passive"

        # Get venue scores
        venue_scores = []
        for venue_id in self._venues:
            score = self._get_venue_score(venue_id, symbol, side, amount_usd)
            if score > 0:
                venue_scores.append((venue_id, score))

        if not venue_scores:
            logger.warning("No venues available for %s %s order", symbol, side)
            return None

        # Sort by score
        venue_scores.sort(key=lambda x: x[1], reverse=True)

        # Select venues based on strategy
        if strategy == "aggressive":
            # Use best single venue for speed
            selected_venues = venue_scores[:1]
        elif strategy == "balanced":
            # Split across top venues
            n_venues = min(self._max_venues, len(venue_scores))
            selected_venues = venue_scores[:n_venues]
        elif strategy == "passive":
            # Use multiple venues, prioritize low fees
            selected_venues = venue_scores[:min(3, len(venue_scores))]
        elif strategy == "twap":
            # Time-weighted across venues
            selected_venues = venue_scores[:min(2, len(venue_scores))]
        else:
            selected_venues = venue_scores[:1]

        # Create order slices
        slices = []
        total_fee = 0.0
        total_time = 0.0

        if strategy == "twap":
            # TWAP: split into time-based slices
            n_slices = 10
            slice_amount = amount_usd / n_slices
            venue_id = selected_venues[0][0]
            venue = self._venues[venue_id]
            
            for i in range(n_slices):
                slice_obj = OrderSlice(
                    venue_id=venue_id,
                    symbol=symbol,
                    side=side,
                    amount=slice_amount,
                    price_limit=target_price,
                    order_type="limit",
                    time_in_force="GTC",
                    estimated_fee=slice_amount * venue.fee_taker,
                    estimated_fill_time_ms=venue.latency_ms + 60000,  # 1 min between slices
                )
                slices.append(slice_obj)
                total_fee += slice_obj.estimated_fee
                total_time += 60000  # 1 minute between slices
        else:
            # Split across venues
            venue_amount = amount_usd / len(selected_venues)
            
            for venue_id, score in selected_venues:
                venue = self._venues[venue_id]
                
                # Determine order type based on strategy
                if strategy == "aggressive":
                    order_type = "market"
                    time_in_force = "IOC"
                elif strategy == "passive":
                    order_type = "limit"
                    time_in_force = "GTC"
                else:
                    order_type = "limit"
                    time_in_force = "IOC"

                slice_obj = OrderSlice(
                    venue_id=venue_id,
                    symbol=symbol,
                    side=side,
                    amount=venue_amount,
                    price_limit=target_price,
                    order_type=order_type,
                    time_in_force=time_in_force,
                    estimated_fee=venue_amount * venue.fee_taker,
                    estimated_fill_time_ms=venue.latency_ms * 2,
                )
                slices.append(slice_obj)
                total_fee += slice_obj.estimated_fee
                total_time = max(total_time, slice_obj.estimated_fill_time_ms)

        # Estimate slippage
        avg_spread = self._estimate_avg_spread(symbol)
        if strategy == "aggressive":
            slippage = avg_spread * 0.5 + (amount_usd / 1_000_000) * 0.001
        elif strategy == "passive":
            slippage = 0.0
        else:
            slippage = avg_spread * 0.3

        # Estimate market impact
        market_impact = (amount_usd / 10_000_000) * 0.001  # 0.1% per $10M

        # Build reasoning
        reasoning = []
        reasoning.append(f"Using {len(slices)} slice(s) across {len(selected_venues)} venue(s)")
        reasoning.append(f"Strategy: {strategy} (urgency: {urgency})")
        reasoning.append(f"Estimated fee: ${total_fee:.2f} ({total_fee/amount_usd*100:.3f}%)")
        if selected_venues:
            reasoning.append(f"Primary venue: {selected_venues[0][0]} (score: {selected_venues[0][1]:.2f})")

        plan = ExecutionPlan(
            symbol=symbol,
            side=side,
            total_amount=amount_usd,
            target_price=target_price or 0.0,
            strategy=strategy,
            slices=slices,
            venues=[v[0] for v in selected_venues],
            estimated_total_fee=total_fee,
            estimated_fill_time_ms=total_time,
            estimated_slippage_pct=slippage * 100,
            market_impact_pct=market_impact * 100,
            reasoning=reasoning,
        )

        return plan

    def _estimate_avg_spread(self, symbol: str) -> float:
        """Estimate average spread for symbol."""
        if symbol not in self._liquidity_data:
            return 0.001  # Default 0.1%

        spreads = [
            l.spread_bps / 10000
            for l in self._liquidity_data[symbol].values()
        ]
        return np.mean(spreads) if spreads else 0.001

    def get_venue_health(self) -> Dict[str, float]:
        """Get health scores for all venues."""
        return {
            venue_id: venue.health_score
            for venue_id, venue in self._venues.items()
        }

    def get_best_venue(
        self,
        symbol: str,
        side: str,
        amount_usd: float,
    ) -> Optional[str]:
        """Get best venue for a specific order."""
        scores = []
        for venue_id in self._venues:
            score = self._get_venue_score(venue_id, symbol, side, amount_usd)
            if score > 0:
                scores.append((venue_id, score))
        
        if scores:
            return max(scores, key=lambda x: x[1])[0]
        return None

    def get_all_venues(self) -> List[str]:
        """Get all registered venues."""
        return sorted(self._venues.keys())


__all__ = ["SmartOrderRouter", "ExecutionPlan", "VenueConfig", "OrderSlice"]
