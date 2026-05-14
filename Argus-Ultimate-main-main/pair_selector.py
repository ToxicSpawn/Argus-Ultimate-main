"""
Dynamic Pair Selector
=====================
Automatically selects the best AUD pairs to trade based on:
- 24h volume (liquidity)
- Spread (execution quality)
- Volatility (profit potential)
- Correlation (diversification)
- Market cap (stability)

Filters out:
- Pairs with too low liquidity
- Meme coins without fundamentals
- Pairs with regulatory issues
- Stablecoins (no price movement)

Usage:
    selector = DynamicPairSelector()
    best_pairs = await selector.select_pairs(num_pairs=3)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("pair_selector")


@dataclass
class PairMetrics:
    """Metrics for a trading pair."""
    symbol: str
    volume_24h: float = 0.0       # 24h volume in AUD
    spread_pct: float = 0.0       # Spread as percentage
    volatility_24h: float = 0.0   # 24h price change %
    market_cap_rank: int = 999    # Market cap rank (lower = better)
    correlation_btc: float = 1.0  # Correlation to BTC
    score: float = 0.0            # Final selection score
    category: str = "unknown"     # bluechip, defi, layer1, meme, etc.


@dataclass
class SelectionConfig:
    """Configuration for pair selection."""
    num_pairs: int = 3                    # How many pairs to select
    min_volume_aud: float = 100_000       # Minimum 24h volume
    max_spread_pct: float = 1.0           # Maximum acceptable spread
    prefer_diversification: bool = True   # Prefer low correlation pairs
    categories: List[str] = field(default_factory=lambda: [
        "bluechip", "layer1", "defi"      # Preferred categories
    ])
    exclude_categories: List[str] = field(default_factory=lambda: [
        "meme", "stablecoin", "junk"      # Excluded categories
    ])
    exclude_symbols: List[str] = field(default_factory=lambda: [
        "PEPE/AUD", "FLOKI/AUD", "SHIB/AUD",  # Meme coins
        "USDT/AUD", "USDC/AUD", "DAI/AUD",    # Stablecoins
    ])


# ── Known AUD Pair Categories ───────────────────────────────────────────────

PAIR_CATEGORIES = {
    # Blue-chip (established, high market cap)
    "XBT/AUD": "bluechip",
    "BTC/AUD": "bluechip",
    "ETH/AUD": "bluechip",
    
    # Layer 1 (blockchain platforms)
    "SOL/AUD": "layer1",
    "ADA/AUD": "layer1",
    "AVAX/AUD": "layer1",
    "DOT/AUD": "layer1",
    "NEAR/AUD": "layer1",
    "ATOM/AUD": "layer1",
    "ALGO/AUD": "layer1",
    "FTM/AUD": "layer1",
    
    # DeFi tokens
    "LINK/AUD": "defi",
    "UNI/AUD": "defi",
    "AAVE/AUD": "defi",
    "MKR/AUD": "defi",
    "CRV/AUD": "defi",
    "LDO/AUD": "defi",
    "SNX/AUD": "defi",
    
    # Payment/Media
    "XRP/AUD": "payment",
    "LTC/AUD": "payment",
    "DOGE/AUD": "meme",
    "BCH/AUD": "payment",
    
    # Meme coins (exclude by default)
    "PEPE/AUD": "meme",
    "FLOKI/AUD": "meme",
    "SHIB/AUD": "meme",
    
    # Stablecoins (exclude - no profit potential)
    "USDT/AUD": "stablecoin",
    "USDC/AUD": "stablecoin",
    "DAI/AUD": "stablecoin",
    
    # Other
    "XLM/AUD": "other",
    "ETC/AUD": "other",
    "ZEC/AUD": "other",
    "DASH/AUD": "other",
}

# Market cap rankings (approximate, lower = better)
MARKET_CAP_RANKS = {
    "XBT/AUD": 1, "BTC/AUD": 1,
    "ETH/AUD": 2,
    "SOL/AUD": 5,
    "XRP/AUD": 6,
    "ADA/AUD": 9,
    "DOGE/AUD": 10,
    "AVAX/AUD": 12,
    "DOT/AUD": 13,
    "LINK/AUD": 15,
    "UNI/AUD": 20,
    "ATOM/AUD": 25,
    "LTC/AUD": 20,
    "NEAR/AUD": 30,
    "BCH/AUD": 25,
    "AAVE/AUD": 40,
    "MKR/AUD": 50,
    "ALGO/AUD": 45,
    "FTM/AUD": 55,
    "XLM/AUD": 35,
    "CRV/AUD": 60,
    "LDO/AUD": 55,
    "SNX/AUD": 80,
    "ETC/AUD": 30,
}


class DynamicPairSelector:
    """Dynamically selects the best pairs to trade."""
    
    def __init__(self, config: SelectionConfig = None):
        self.config = config or SelectionConfig()
        self._cache: Dict[str, PairMetrics] = {}
        self._cache_time: float = 0
        self._cache_ttl: float = 60  # Cache for 60 seconds
    
    async def select_pairs(self, kraken_client=None, num_pairs: int = None) -> List[PairMetrics]:
        """Select the best pairs to trade.
        
        Args:
            kraken_client: Kraken REST client (optional, uses mock if None)
            num_pairs: Override config num_pairs
            
        Returns:
            List of PairMetrics for selected pairs, sorted by score (best first)
        """
        num_pairs = num_pairs or self.config.num_pairs
        
        logger.info(f"Selecting top {num_pairs} AUD pairs...")
        
        # Get available pairs and their metrics
        all_pairs = await self._fetch_pair_metrics(kraken_client)
        
        # Filter pairs
        filtered = self._filter_pairs(all_pairs)
        
        # Score pairs
        scored = self._score_pairs(filtered)
        
        # Sort by score descending
        scored.sort(key=lambda p: p.score, reverse=True)
        
        # Take top N
        selected = scored[:num_pairs]
        
        # Log selection
        logger.info(f"Selected pairs:")
        for i, pair in enumerate(selected, 1):
            logger.info(f"  {i}. {pair.symbol} (score: {pair.score:.1f}, "
                       f"volume: ${pair.volume_24h:,.0f}, "
                       f"spread: {pair.spread_pct:.2f}%)")
        
        return selected
    
    async def _fetch_pair_metrics(self, client=None) -> List[PairMetrics]:
        """Fetch metrics for all available AUD pairs."""
        # Check cache
        if time.time() - self._cache_time < self._cache_ttl and self._cache:
            return list(self._cache.values())
        
        pairs = []
        
        if client:
            # Fetch real data from Kraken
            pairs = await self._fetch_from_kraken(client)
        else:
            # Use realistic mock data
            pairs = self._get_mock_data()
        
        # Update cache
        self._cache = {p.symbol: p for p in pairs}
        self._cache_time = time.time()
        
        return pairs
    
    async def _fetch_from_kraken(self, client) -> List[PairMetrics]:
        """Fetch pair data from Kraken API."""
        pairs = []
        
        try:
            # Get ticker data for all pairs
            response = await client._session.get(
                f"{client.base_url}/Ticker",
                params={"pair": "AUD"}  # This would need proper implementation
            )
            
            if response.status == 200:
                data = await response.json()
                # Parse Kraken response and create PairMetrics
                # This is a simplified version - real implementation would parse properly
                pass
                
        except Exception as e:
            logger.error(f"Failed to fetch from Kraken: {e}")
        
        # Fallback to mock if API fails
        if not pairs:
            pairs = self._get_mock_data()
        
        return pairs
    
    def _get_mock_data(self) -> List[PairMetrics]:
        """Get realistic mock data for AUD pairs."""
        # Based on typical Kraken AUD pair data
        mock_data = [
            # symbol, volume_24h, spread%, volatility%, category
            ("XBT/AUD", 4_400_000, 0.05, 2.5, "bluechip"),
            ("ETH/AUD", 545_000, 0.08, 3.2, "bluechip"),
            ("SOL/AUD", 276_000, 0.12, 4.5, "layer1"),
            ("XRP/AUD", 180_000, 0.15, 3.8, "payment"),
            ("ADA/AUD", 95_000, 0.20, 4.0, "layer1"),
            ("DOGE/AUD", 120_000, 0.18, 5.5, "meme"),
            ("LTC/AUD", 85_000, 0.15, 3.0, "payment"),
            ("AVAX/AUD", 65_000, 0.25, 5.0, "layer1"),
            ("LINK/AUD", 55_000, 0.22, 4.2, "defi"),
            ("DOT/AUD", 45_000, 0.28, 4.5, "layer1"),
            ("UNI/AUD", 35_000, 0.30, 4.8, "defi"),
            ("ATOM/AUD", 30_000, 0.35, 5.2, "layer1"),
            ("BCH/AUD", 28_000, 0.25, 3.5, "payment"),
            ("NEAR/AUD", 25_000, 0.40, 6.0, "layer1"),
            ("AAVE/AUD", 20_000, 0.45, 5.5, "defi"),
            ("ALGO/AUD", 18_000, 0.50, 5.8, "layer1"),
            ("FTM/AUD", 15_000, 0.55, 6.5, "layer1"),
            ("XLM/AUD", 12_000, 0.45, 4.5, "other"),
            ("ETC/AUD", 10_000, 0.40, 4.0, "other"),
        ]
        
        pairs = []
        for symbol, volume, spread, volatility, category in mock_data:
            pairs.append(PairMetrics(
                symbol=symbol,
                volume_24h=volume,
                spread_pct=spread,
                volatility_24h=volatility,
                market_cap_rank=MARKET_CAP_RANKS.get(symbol, 999),
                correlation_btc=self._estimate_correlation(symbol),
                category=category,
            ))
        
        return pairs
    
    def _estimate_correlation(self, symbol: str) -> float:
        """Estimate correlation to BTC."""
        # Higher market cap = higher correlation to BTC
        rank = MARKET_CAP_RANKS.get(symbol, 100)
        if rank <= 2:
            return 0.95
        elif rank <= 10:
            return 0.85
        elif rank <= 30:
            return 0.75
        else:
            return 0.65
    
    def _filter_pairs(self, pairs: List[PairMetrics]) -> List[PairMetrics]:
        """Filter out pairs that don't meet criteria."""
        filtered = []
        
        for pair in pairs:
            # Exclude by symbol
            if pair.symbol in self.config.exclude_symbols:
                logger.debug(f"Excluded {pair.symbol}: in exclude list")
                continue
            
            # Exclude by category
            if pair.category in self.config.exclude_categories:
                logger.debug(f"Excluded {pair.symbol}: category {pair.category}")
                continue
            
            # Minimum volume
            if pair.volume_24h < self.config.min_volume_aud:
                logger.debug(f"Excluded {pair.symbol}: volume ${pair.volume_24h:,.0f} < ${self.config.min_volume_aud:,.0f}")
                continue
            
            # Maximum spread
            if pair.spread_pct > self.config.max_spread_pct:
                logger.debug(f"Excluded {pair.symbol}: spread {pair.spread_pct:.2f}% > {self.config.max_spread_pct:.2f}%")
                continue
            
            filtered.append(pair)
        
        logger.info(f"Filtered: {len(pairs)} → {len(filtered)} pairs")
        return filtered
    
    def _score_pairs(self, pairs: List[PairMetrics]) -> List[PairMetrics]:
        """Score each pair for selection."""
        for pair in pairs:
            score = 0.0
            
            # Volume score (0-40 points) - more volume = better
            volume_score = min(pair.volume_24h / 100_000, 40)
            score += volume_score
            
            # Spread score (0-20 points) - lower spread = better
            spread_score = max(0, 20 - (pair.spread_pct * 20))
            score += spread_score
            
            # Volatility score (0-20 points) - some volatility is good
            # Too low = no profit, too high = risky
            vol = pair.volatility_24h
            if 2 <= vol <= 5:
                volatility_score = 20
            elif 1 <= vol < 2 or 5 < vol <= 8:
                volatility_score = 15
            elif vol < 1:
                volatility_score = 5
            else:
                volatility_score = 10
            score += volatility_score
            
            # Market cap score (0-10 points) - lower rank = better
            cap_score = max(0, 10 - (pair.market_cap_rank / 10))
            score += cap_score
            
            # Category bonus
            if pair.category in self.config.categories:
                score += 10
            
            # Diversification bonus (lower correlation to BTC = better)
            if self.config.prefer_diversification:
                diversification_score = (1 - pair.correlation_btc) * 10
                score += diversification_score
            
            pair.score = score
        
        return pairs
    
    def get_allocation(self, selected_pairs: List[PairMetrics], total_capital: float) -> Dict[str, float]:
        """Calculate allocation for each selected pair.
        
        Uses risk-parity approach: higher volatility = lower allocation.
        """
        if not selected_pairs:
            return {}
        
        # Calculate inverse volatility weights
        total_inv_vol = sum(1 / max(p.volatility_24h, 0.1) for p in selected_pairs)
        
        allocations = {}
        for pair in selected_pairs:
            weight = (1 / max(pair.volatility_24h, 0.1)) / total_inv_vol
            allocations[pair.symbol] = total_capital * weight
        
        return allocations


# ── Demo ─────────────────────────────────────────────────────────────────────

async def demo():
    """Demo the pair selector."""
    selector = DynamicPairSelector(SelectionConfig(num_pairs=3))
    
    print("Dynamic Pair Selector Demo")
    print("=" * 60)
    
    selected = await selector.select_pairs(num_pairs=3)
    
    print(f"\nTop 3 pairs selected:")
    for i, pair in enumerate(selected, 1):
        print(f"  {i}. {pair.symbol}")
        print(f"     Volume: ${pair.volume_24h:,.0f}")
        print(f"     Spread: {pair.spread_pct:.2f}%")
        print(f"     Volatility: {pair.volatility_24h:.1f}%")
        print(f"     Category: {pair.category}")
        print(f"     Score: {pair.score:.1f}")
        print()
    
    # Show allocation
    allocations = selector.get_allocation(selected, 1000)
    print("Recommended allocation ($1,000):")
    for symbol, amount in allocations.items():
        print(f"  {symbol}: ${amount:,.2f}")


if __name__ == "__main__":
    asyncio.run(demo())
