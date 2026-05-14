"""Stablecoin flow tracker for on-chain analysis.

Tracks stablecoin (USDT, USDC, DAI) flows to/from exchanges
to detect accumulation or distribution patterns.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StablecoinFlow:
    """Single stablecoin flow event.
    
    Attributes
    ----------
    token : str
        Stablecoin symbol (e.g., "USDT", "USDC")
    amount : float
        Flow amount in USD
    direction : str
        "inflow" (to exchange) or "outflow" (from exchange)
    timestamp : float
        Unix timestamp
    tx_hash : str
        Transaction hash
    """
    token: str = ""
    amount: float = 0.0
    direction: str = "inflow"
    timestamp: float = field(default_factory=time.time)
    tx_hash: str = ""


@dataclass
class AggregateFlow:
    """Aggregated stablecoin flows over a period.
    
    Attributes
    ----------
    period_start : float
        Start of aggregation period
    period_end : float
        End of aggregation period
    total_inflow : float
        Total inflow in USD
    total_outflow : float
        Total outflow in USD
    net_flow : float
        Net flow (inflow - outflow)
    flow_count : int
        Number of flow events
    """
    period_start: float = 0.0
    period_end: float = 0.0
    total_inflow: float = 0.0
    total_outflow: float = 0.0
    net_flow: float = 0.0
    flow_count: int = 0
    
    @property
    def direction(self) -> str:
        """Return 'inflow', 'outflow', or 'neutral'."""
        if self.net_flow > 1_000_000:
            return "inflow"
        elif self.net_flow < -1_000_000:
            return "outflow"
        return "neutral"


class StablecoinFlowTracker:
    """Tracker for stablecoin flows to/from exchanges.
    
    Parameters
    ----------
    cache_ttl_s : float
        Cache time-to-live in seconds (default 300)
    tokens : list[str]
        Stablecoin symbols to track (default ["USDT", "USDC", "DAI"])
    """
    
    def __init__(
        self,
        cache_ttl_s: float = 300.0,
        tokens: Optional[List[str]] = None,
    ) -> None:
        self._cache_ttl_s = cache_ttl_s
        self._tokens = tokens or ["USDT", "USDC", "DAI"]
        self._flows: List[StablecoinFlow] = []
        self._last_fetch_time: float = 0.0
    
    async def get_recent_flows(
        self,
        hours: float = 24.0,
    ) -> List[StablecoinFlow]:
        """Get recent stablecoin flows."""
        now = time.time()
        cutoff = now - (hours * 3600)
        return [f for f in self._flows if f.timestamp >= cutoff]
    
    async def get_aggregate_flow(
        self,
        hours: float = 24.0,
    ) -> AggregateFlow:
        """Get aggregated flows over a period."""
        flows = await self.get_recent_flows(hours)
        now = time.time()
        
        if not flows:
            return AggregateFlow(
                period_start=now - (hours * 3600),
                period_end=now,
            )
        
        inflows = [f.amount for f in flows if f.direction == "inflow"]
        outflows = [f.amount for f in flows if f.direction == "outflow"]
        
        total_inflow = sum(inflows)
        total_outflow = sum(outflows)
        
        return AggregateFlow(
            period_start=min(f.timestamp for f in flows),
            period_end=max(f.timestamp for f in flows),
            total_inflow=total_inflow,
            total_outflow=total_outflow,
            net_flow=total_inflow - total_outflow,
            flow_count=len(flows),
        )
    
    def add_flow(self, flow: StablecoinFlow) -> None:
        """Add a flow event to the tracker."""
        self._flows.append(flow)
        # Keep only last 10000 flows
        if len(self._flows) > 10000:
            self._flows = self._flows[-10000:]
