"""Chain metrics provider for on-chain data analysis.

Provides MVRV Z-Score, SOPR, exchange flow metrics, and signal bias
for Bitcoin market analysis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import logging

logger = logging.getLogger(__name__)


@dataclass
class ChainMetricSnapshot:
    """Snapshot of on-chain metrics at a point in time."""
    
    mvrv_zscore: float = 0.0
    sopr: float = 1.0
    net_exchange_flow_btc: float = 0.0
    signal_bias: float = 0.0
    timestamp: float = field(default_factory=time.time)
    hash_rate: float = 0.0
    difficulty: float = 0.0
    n_tx: int = 0
    miners_revenue_usd: float = 0.0
    miners_revenue_btc: float = 0.0
    market_price_usd: float = 0.0


class ChainMetricsProvider:
    """Provider for on-chain metrics from blockchain data sources."""
    
    def __init__(
        self,
        cache_ttl_s: float = 900.0,
        api_url: Optional[str] = None,
    ) -> None:
        self._cache_ttl_s = cache_ttl_s
        self._api_url = api_url or "https://api.blockchain.info/charts"
        self._cached_snapshot: Optional[ChainMetricSnapshot] = None
        self._last_fetch_time: float = 0.0
    
    def get_signal_bias(self) -> float:
        """Return the current signal bias from cached snapshot, or 0.0 if no data."""
        if self._cached_snapshot is None:
            return 0.0
        return self._cached_snapshot.signal_bias
    
    async def get_metrics(self) -> ChainMetricSnapshot:
        """Fetch current on-chain metrics.
        
        Returns cached snapshot if fresh, otherwise fetches new data.
        """
        now = time.time()
        
        # Return cached if fresh
        if (
            self._cached_snapshot is not None
            and (now - self._last_fetch_time) < self._cache_ttl_s
        ):
            return self._cached_snapshot
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                # Fetch market price
                async with session.get(f"{self._api_url}/market-price") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        price = data.get("values", [{}])[-1].get("y", 0.0) if data.get("values") else 0.0
                    else:
                        price = 0.0
                
                # Create snapshot with fetched data
                snapshot = ChainMetricSnapshot(
                    market_price_usd=price,
                    timestamp=now,
                )
                
                self._cached_snapshot = snapshot
                self._last_fetch_time = now
                return snapshot
                
        except Exception as e:
            logger.warning("Failed to fetch chain metrics: %s", e)
            # Return cached or empty snapshot
            if self._cached_snapshot is not None:
                return self._cached_snapshot
            return ChainMetricSnapshot(timestamp=now)
    
    def compute_signal_bias(self, snapshot: ChainMetricSnapshot) -> float:
        """Compute signal bias from on-chain metrics.
        
        Returns value in [-1, +1] range:
        - Positive: bullish (MVRV > 1, SOPR > 1, outflows > inflows)
        - Negative: bearish (MVRV < 1, SOPR < 1, inflows > outflows)
        """
        bias = 0.0
        
        # MVRV Z-Score contribution
        if snapshot.mvrv_zscore > 2.0:
            bias -= 0.3  # Overvalued
        elif snapshot.mvrv_zscore < 0:
            bias += 0.3  # Undervalued
        
        # SOPR contribution
        if snapshot.sopr > 1.0:
            bias += 0.2  # Profit-taking
        elif snapshot.sopr < 0.95:
            bias -= 0.2  # Capitulation
        
        # Exchange flow contribution
        if snapshot.net_exchange_flow_btc < -1000:
            bias += 0.3  # Net outflows (accumulation)
        elif snapshot.net_exchange_flow_btc > 1000:
            bias -= 0.3  # Net inflows (distribution)
        
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, bias))
