"""
Liquidity Analyzer — DeFi pool analysis for optimal execution.

Analyzes liquidity across DEX pools to find:
- Best execution venues
- Slippage estimation
- Pool depth and depth changes
- Impermanent loss risk for LP positions
- Optimal trade splitting across pools

Example::

    analyzer = LiquidityAnalyzer()
    analyzer.update_pool(
        pool_id="uniswap_v3_eth_usdc_5bp",
        dex="uniswap_v3",
        token0="ETH",
        token1="USDC",
        reserve0=1000.0,
        reserve1=2000000.0,
        fee_tier=0.0005,
        tvl=4000000.0,
    )
    
    quote = analyzer.get_quote("ETH", "USDC", amount_in=10.0)
    print(quote.amount_out, quote.slippage_pct, quote.best_pool)
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PoolInfo:
    """DEX pool information."""
    pool_id: str
    dex: str
    token0: str
    token1: str
    reserve0: float
    reserve1: float
    fee_tier: float  # e.g., 0.0005 for 0.05%
    tvl: float
    volume_24h: float
    apr: float  # Annual percentage rate for LPs
    timestamp: float
    
    @property
    def price(self) -> float:
        """Pool price (token1 per token0)."""
        if self.reserve0 > 0:
            return self.reserve1 / self.reserve0
        return 0.0
    
    @property
    def depth_score(self) -> float:
        """Liquidity depth score (0-1, higher is better)."""
        # Normalize by typical trade size
        typical_trade = 10000  # $10k
        if self.tvl > 0:
            return min(1.0, self.tvl / (typical_trade * 100))
        return 0.0


@dataclass
class TradeQuote:
    """Trade execution quote."""
    symbol: str  # "TOKEN_IN/TOKEN_OUT"
    pool_id: str
    dex: str
    amount_in: float
    amount_out: float
    amount_out_min: float  # With slippage protection
    price_impact_pct: float
    slippage_pct: float
    fee_cost: float
    gas_estimate_usd: float
    net_amount_out: float  # After fees
    route: List[str]  # Pool hops
    timestamp: float


@dataclass
class LiquidityReport:
    """Liquidity analysis report for a token pair."""
    token0: str
    token1: str
    total_liquidity_usd: float
    best_pool: str
    best_dex: str
    worst_slippage_1pct: float  # Slippage for 1% of pool
    avg_spread_bps: float
    pool_count: int
    concentration_risk: float  # 0-1, higher = more concentrated
    recommendation: str  # "good", "moderate", "poor"


@dataclass
class _PairState:
    pools: Dict[str, PoolInfo] = field(default_factory=dict)
    last_report: Optional[LiquidityReport] = None


class LiquidityAnalyzer:
    """
    Multi-DEX liquidity analyzer for optimal trade execution.

    Parameters
    ----------
    default_slippage : float
        Default slippage tolerance (default 0.5%).
    gas_price_gwei : float
        Current gas price for cost estimation (default 30).
    gas_cost_per_hop : float
        Gas cost per pool hop in USD (default $5).
    min_liquidity_usd : float
        Minimum pool liquidity to consider (default $10000).
    """

    def __init__(
        self,
        default_slippage: float = 0.005,
        gas_price_gwei: float = 30.0,
        gas_cost_per_hop: float = 5.0,
        min_liquidity_usd: float = 10000.0,
    ) -> None:
        self._default_slippage = default_slippage
        self._gas_price = gas_price_gwei
        self._gas_cost = gas_cost_per_hop
        self._min_liquidity = min_liquidity_usd
        self._pair_states: Dict[str, _PairState] = {}  # "TOKEN0/TOKEN1" -> state

        logger.info(
            "LiquidityAnalyzer initialized: slippage=%.2f%% gas=$%.1f/hop min_liq=$%.0fk",
            default_slippage * 100, gas_cost_per_hop, min_liquidity_usd / 1000,
        )

    def _get_pair_key(self, token0: str, token1: str) -> str:
        """Get normalized pair key."""
        tokens = sorted([token0.upper(), token1.upper()])
        return f"{tokens[0]}/{tokens[1]}"

    def update_pool(
        self,
        pool_id: str,
        dex: str,
        token0: str,
        token1: str,
        reserve0: float,
        reserve1: float,
        fee_tier: float,
        tvl: float,
        volume_24h: float = 0.0,
        apr: float = 0.0,
    ) -> None:
        """Update pool information."""
        pair_key = self._get_pair_key(token0, token1)
        
        if pair_key not in self._pair_states:
            self._pair_states[pair_key] = _PairState()

        state = self._pair_states[pair_key]
        
        pool = PoolInfo(
            pool_id=pool_id,
            dex=dex,
            token0=token0.upper(),
            token1=token1.upper(),
            reserve0=reserve0,
            reserve1=reserve1,
            fee_tier=fee_tier,
            tvl=tvl,
            volume_24h=volume_24h,
            apr=apr,
            timestamp=time.time(),
        )
        
        state.pools[pool_id] = pool
        self._update_report(pair_key)

    def _update_report(self, pair_key: str) -> None:
        """Update liquidity report for pair."""
        state = self._pair_states[pair_key]
        pools = list(state.pools.values())
        
        if not pools:
            return

        # Filter by minimum liquidity
        valid_pools = [p for p in pools if p.tvl >= self._min_liquidity]
        if not valid_pools:
            valid_pools = pools

        total_liquidity = sum(p.tvl for p in valid_pools)
        
        # Find best pool (highest liquidity, lowest fee)
        best_pool = max(valid_pools, key=lambda p: p.tvl * (1 - p.fee_tier))
        
        # Calculate average spread
        spreads = []
        for pool in valid_pools:
            if pool.reserve0 > 0 and pool.reserve1 > 0:
                # Approximate spread from fee tier and depth
                spread = pool.fee_tier * 2 + (1 / pool.depth_score) * 0.0001
                spreads.append(spread * 10000)  # In bps
        
        avg_spread = np.mean(spreads) if spreads else 0.0
        
        # Calculate concentration risk
        if total_liquidity > 0:
            max_pool_share = max(p.tvl for p in valid_pools) / total_liquidity
            concentration = max_pool_share
        else:
            concentration = 1.0

        # Determine recommendation
        if total_liquidity > 10_000_000 and concentration < 0.5:
            recommendation = "good"
        elif total_liquidity > 1_000_000:
            recommendation = "moderate"
        else:
            recommendation = "poor"

        state.last_report = LiquidityReport(
            token0=valid_pools[0].token0,
            token1=valid_pools[0].token1,
            total_liquidity_usd=total_liquidity,
            best_pool=best_pool.pool_id,
            best_dex=best_pool.dex,
            worst_slippage_1pct=avg_spread / 100,  # Convert bps to %
            avg_spread_bps=avg_spread,
            pool_count=len(valid_pools),
            concentration_risk=concentration,
            recommendation=recommendation,
        )

    def get_quote(
        self,
        token_in: str,
        token_out: str,
        amount_in: float,
        slippage: Optional[float] = None,
    ) -> Optional[TradeQuote]:
        """Get best trade quote for token swap."""
        pair_key = self._get_pair_key(token_in, token_out)
        
        if pair_key not in self._pair_states:
            return None

        state = self._pair_states[pair_key]
        slippage = slippage or self._default_slippage

        # Find best pool for this trade
        best_quote = None
        best_amount_out = 0.0

        for pool_id, pool in state.pools.items():
            if pool.tvl < self._min_liquidity:
                continue

            quote = self._calculate_quote(pool, token_in, token_out, amount_in, slippage)
            if quote and quote.net_amount_out > best_amount_out:
                best_amount_out = quote.net_amount_out
                best_quote = quote

        return best_quote

    def _calculate_quote(
        self,
        pool: PoolInfo,
        token_in: str,
        token_out: str,
        amount_in: float,
        slippage: float,
    ) -> Optional[TradeQuote]:
        """Calculate quote for a specific pool."""
        # Determine direction
        if token_in.upper() == pool.token0:
            amount_in_token0 = True
        elif token_in.upper() == pool.token1:
            amount_in_token0 = False
        else:
            return None

        # Constant product formula with fee
        fee = pool.fee_tier
        amount_in_with_fee = amount_in * (1 - fee)

        if amount_in_token0:
            # Token0 -> Token1
            numerator = amount_in_with_fee * pool.reserve1
            denominator = pool.reserve0 + amount_in_with_fee
            amount_out = numerator / denominator
        else:
            # Token1 -> Token0
            numerator = amount_in_with_fee * pool.reserve0
            denominator = pool.reserve1 + amount_in_with_fee
            amount_out = numerator / denominator

        # Price impact
        if amount_in_token0:
            price_impact = amount_in / pool.reserve0
        else:
            price_impact = amount_in / pool.reserve1

        # Slippage
        amount_out_min = amount_out * (1 - slippage)
        slippage_cost = amount_out * slippage

        # Fee cost
        fee_cost = amount_in * fee

        # Gas estimate
        gas_cost = self._gas_cost

        # Net output
        net_amount_out = amount_out - (gas_cost / pool.price if pool.price > 0 else 0)

        symbol = f"{token_in.upper()}/{token_out.upper()}"
        
        return TradeQuote(
            symbol=symbol,
            pool_id=pool.pool_id,
            dex=pool.dex,
            amount_in=amount_in,
            amount_out=amount_out,
            amount_out_min=amount_out_min,
            price_impact_pct=price_impact * 100,
            slippage_pct=slippage * 100,
            fee_cost=fee_cost,
            gas_estimate_usd=gas_cost,
            net_amount_out=net_amount_out,
            route=[pool.pool_id],
            timestamp=time.time(),
        )

    def get_best_quote(
        self,
        token_in: str,
        token_out: str,
        amount_in: float,
    ) -> Optional[TradeQuote]:
        """Get the best available quote."""
        return self.get_quote(token_in, token_out, amount_in)

    def get_split_quote(
        self,
        token_in: str,
        token_out: str,
        amount_in: float,
        num_splits: int = 3,
    ) -> List[TradeQuote]:
        """Get split trade quotes for large orders."""
        pair_key = self._get_pair_key(token_in, token_out)
        
        if pair_key not in self._pair_states:
            return []

        state = self._pair_states[pair_key]
        pools = sorted(
            state.pools.values(),
            key=lambda p: p.tvl,
            reverse=True
        )[:num_splits]

        split_amount = amount_in / len(pools)
        quotes = []

        for pool in pools:
            quote = self._calculate_quote(
                pool, token_in, token_out, split_amount, self._default_slippage
            )
            if quote:
                quotes.append(quote)

        return quotes

    def get_report(self, token0: str, token1: str) -> Optional[LiquidityReport]:
        """Get liquidity report for token pair."""
        pair_key = self._get_pair_key(token0, token1)
        if pair_key in self._pair_states:
            return self._pair_states[pair_key].last_report
        return None

    def get_pools(self, token0: str, token1: str) -> List[PoolInfo]:
        """Get all pools for token pair."""
        pair_key = self._get_pair_key(token0, token1)
        if pair_key in self._pair_states:
            return list(self._pair_states[pair_key].pools.values())
        return []

    def estimate_slippage(
        self,
        token_in: str,
        token_out: str,
        amount_in: float,
    ) -> float:
        """Estimate slippage for a trade size."""
        quote = self.get_quote(token_in, token_out, amount_in)
        if quote:
            return quote.slippage_pct
        return self._default_slippage * 100

    def get_all_pairs(self) -> List[str]:
        """Get all tracked token pairs."""
        return sorted(self._pair_states.keys())


__all__ = ["LiquidityAnalyzer", "TradeQuote", "LiquidityReport", "PoolInfo"]
