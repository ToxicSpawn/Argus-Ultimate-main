"""DEX price aggregator for DeFi arbitrage detection.

Aggregates prices from multiple DEXes (Uniswap, Sushiswap, etc.)
and detects CEX-DEX arbitrage opportunities.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DexPrice:
    """Price from a DEX.
    
    Attributes
    ----------
    symbol : str
        Trading pair symbol (e.g., "ETH/USD")
    dex_name : str
        DEX name (e.g., "uniswap_v3", "sushiswap")
    price : float
        Price from the DEX
    liquidity_usd : float
        Available liquidity in USD
    timestamp : float
        Unix timestamp
    """
    symbol: str = ""
    dex_name: str = ""
    price: float = 0.0
    liquidity_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ArbOpportunity:
    """CEX-DEX arbitrage opportunity.
    
    Attributes
    ----------
    symbol : str
        Trading pair symbol
    cex_price : float
        CEX price
    dex_price : float
        DEX price
    spread_bps : float
        Spread in basis points
    dex_name : str
        DEX with the opportunity
    profitable : bool
        Whether the opportunity is profitable after fees
    estimated_profit_usd : float
        Estimated profit in USD
    timestamp : float
        Unix timestamp
    """
    symbol: str = ""
    cex_price: float = 0.0
    dex_price: float = 0.0
    spread_bps: float = 0.0
    dex_name: str = ""
    profitable: bool = False
    estimated_profit_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)


class DexPriceAggregator:
    """Aggregator for DEX prices and arbitrage detection.
    
    Parameters
    ----------
    config : Any
        Configuration object with dex_aggregator_* settings
    """
    
    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.ethereum_rpc: Optional[str] = None
        self.solana_rpc: Optional[str] = None
        self._price_cache: Dict[str, DexPrice] = {}
        self._cache_ts: float = 0.0
        
        # Get settings from config
        if config:
            self._cache_ttl_s = getattr(config, "dex_aggregator_cache_ttl_s", 30)
            self._min_spread_bps = getattr(config, "dex_aggregator_min_spread_bps", 20)
        else:
            self._cache_ttl_s = 30
            self._min_spread_bps = 20
    
    async def get_dex_prices(
        self,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, DexPrice]:
        """Get current DEX prices for symbols.
        
        Returns cached prices if fresh, otherwise fetches new data.
        Returns empty dict if no RPC clients are configured.
        """
        # Check cache
        now = time.time()
        if (
            len(self._price_cache) > 0
            and (now - self._cache_ts) < self._cache_ttl_s
        ):
            if symbols:
                return {s: self._price_cache[s] for s in symbols if s in self._price_cache}
            return self._price_cache.copy()
        
        # No RPC clients - return empty or cached
        if not self.ethereum_rpc and not self.solana_rpc:
            return {}
        
        # Fetch prices (placeholder - would connect to actual DEX contracts)
        # For now, return empty
        return {}
    
    async def get_cex_dex_spreads(
        self,
        cex_prices: Dict[str, float],
    ) -> List[ArbOpportunity]:
        """Calculate CEX-DEX spreads and find arbitrage opportunities.
        
        Parameters
        ----------
        cex_prices : dict
            Dictionary of symbol -> CEX price
            
        Returns
        -------
        list[ArbOpportunity]
            List of arbitrage opportunities sorted by spread
        """
        dex_prices = await self.get_dex_prices()
        opportunities: List[ArbOpportunity] = []
        
        for symbol, cex_price in cex_prices.items():
            if symbol not in dex_prices:
                continue
            
            dex = dex_prices[symbol]
            if dex.price <= 0:
                continue
            
            # Calculate spread
            spread_pct = abs(cex_price - dex.price) / cex_price * 100
            spread_bps = spread_pct * 100
            
            if spread_bps < self._min_spread_bps:
                continue
            
            # Determine direction
            profitable = spread_bps > self._min_spread_bps * 1.5  # Account for fees
            
            opp = ArbOpportunity(
                symbol=symbol,
                cex_price=cex_price,
                dex_price=dex.price,
                spread_bps=spread_bps,
                dex_name=dex.dex_name,
                profitable=profitable,
                estimated_profit_usd=dex.liquidity_usd * (spread_pct / 100) * 0.1,
            )
            opportunities.append(opp)
        
        # Sort by spread descending
        opportunities.sort(key=lambda x: x.spread_bps, reverse=True)
        return opportunities
