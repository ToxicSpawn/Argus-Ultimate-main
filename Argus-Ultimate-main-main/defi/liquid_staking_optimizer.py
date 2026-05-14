"""
Liquid Staking Optimizer
========================
Optimizes liquid staking positions:
- Lido (stETH)
- Rocket Pool (rETH)
- Frax (sfrxETH)
- Coinbase (cbETH)
- Mantle (mETH)

Features:
- APY comparison
- Depeg risk analysis
- Optimal allocation
- Restaking opportunities
- MEV rewards tracking
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class LiquidStakingProvider(Enum):
    """Liquid staking providers."""
    LIDO = "lido"
    ROCKET_POOL = "rocket_pool"
    FRAX = "frax"
    COINBASE = "coinbase"
    MANTLE = "mantle"
    SWELL = "swell"
    BINANCE = "binance"  # BETH


@dataclass
class StakingPool:
    """Liquid staking pool."""
    provider: LiquidStakingProvider
    token: str  # e.g., "stETH"
    underlying: str  # e.g., "ETH"
    apy: float
    tvl_eth: float
    total_staked_eth: float
    validators: int
    fee_pct: float
    exchange_rate: float  # stETH/ETH
    depeg_risk: float  # 0-1
    liquidity_depth: float  # ETH available at 1% slippage
    mev_rewards_pct: float  # % of rewards from MEV
    last_updated: float = field(default_factory=time.time)


@dataclass
class StakingPosition:
    """User staking position."""
    provider: LiquidStakingProvider
    token: str
    amount_tokens: float
    amount_eth: float
    entry_rate: float
    current_rate: float
    rewards_earned: float
    apr: float
    unstaking_available: bool = True
    withdrawal_queue: int = 0  # Days to unstake


@dataclass
class StakingOpportunity:
    """Staking opportunity."""
    provider: LiquidStakingProvider
    token: str
    apy: float
    tvl_eth: float
    depeg_risk: float
    liquidity_score: float
    composite_score: float
    recommended_allocation_pct: float
    reasons: List[str] = field(default_factory=list)


class LiquidStakingAnalyzer:
    """
    Liquid Staking Analyzer
    =======================
    Analyzes liquid staking providers.
    """
    
    def __init__(self):
        self.pools: Dict[LiquidStakingProvider, StakingPool] = {}
        self.apy_history: Dict[LiquidStakingProvider, List[float]] = {}
        self.exchange_rate_history: Dict[LiquidStakingProvider, List[float]] = {}
        
        self._init_default_pools()
    
    def _init_default_pools(self) -> None:
        """Initialize default pool data."""
        default_pools = [
            StakingPool(
                provider=LiquidStakingProvider.LIDO,
                token="stETH",
                underlying="ETH",
                apy=0.038,  # 3.8%
                tvl_eth=9_500_000,
                total_staked_eth=9_500_000,
                validators=295_000,
                fee_pct=0.10,  # 10%
                exchange_rate=1.001,
                depeg_risk=0.05,
                liquidity_depth=50_000,
                mev_rewards_pct=0.05
            ),
            StakingPool(
                provider=LiquidStakingProvider.ROCKET_POOL,
                token="rETH",
                underlying="ETH",
                apy=0.035,
                tvl_eth=1_200_000,
                total_staked_eth=1_200_000,
                validators=37_500,
                fee_pct=0.14,  # 14% commission to node operators
                exchange_rate=1.002,
                depeg_risk=0.08,
                liquidity_depth=10_000,
                mev_rewards_pct=0.03
            ),
            StakingPool(
                provider=LiquidStakingProvider.FRAX,
                token="sfrxETH",
                underlying="ETH",
                apy=0.042,
                tvl_eth=250_000,
                total_staked_eth=250_000,
                validators=7_800,
                fee_pct=0.10,
                exchange_rate=1.003,
                depeg_risk=0.12,
                liquidity_depth=3_000,
                mev_rewards_pct=0.08
            ),
            StakingPool(
                provider=LiquidStakingProvider.COINBASE,
                token="cbETH",
                underlying="ETH",
                apy=0.033,
                tvl_eth=3_500_000,
                total_staked_eth=3_500_000,
                validators=109_000,
                fee_pct=0.25,  # 25% fee
                exchange_rate=0.999,
                depeg_risk=0.03,
                liquidity_depth=20_000,
                mev_rewards_pct=0.02
            ),
            StakingPool(
                provider=LiquidStakingProvider.SWELL,
                token="swETH",
                underlying="ETH",
                apy=0.040,
                tvl_eth=600_000,
                total_staked_eth=600_000,
                validators=18_750,
                fee_pct=0.10,
                exchange_rate=1.001,
                depeg_risk=0.10,
                liquidity_depth=5_000,
                mev_rewards_pct=0.06
            )
        ]
        
        for pool in default_pools:
            self.pools[pool.provider] = pool
    
    def update_pool(self, pool: StakingPool) -> None:
        """Update pool data."""
        self.pools[pool.provider] = pool
        
        # Update history
        if pool.provider not in self.apy_history:
            self.apy_history[pool.provider] = []
        if pool.provider not in self.exchange_rate_history:
            self.exchange_rate_history[pool.provider] = []
        
        self.apy_history[pool.provider].append(pool.apy)
        self.exchange_rate_history[pool.provider].append(pool.exchange_rate)
    
    def calculate_depeg_risk(self, provider: LiquidStakingProvider) -> Dict[str, Any]:
        """Calculate depeg risk for a provider."""
        if provider not in self.pools:
            return {"risk": 0.5, "factors": []}
        
        pool = self.pools[provider]
        risk_factors = []
        risk_score = 0.0
        
        # Exchange rate deviation
        rate_deviation = abs(pool.exchange_rate - 1.0)
        if rate_deviation > 0.02:
            risk_score += 0.3
            risk_factors.append(f"Exchange rate deviation: {rate_deviation*100:.2f}%")
        
        # TVL concentration
        total_tvl = sum(p.tvl_eth for p in self.pools.values())
        concentration = pool.tvl_eth / total_tvl if total_tvl > 0 else 0
        if concentration > 0.3:
            risk_score += 0.2
            risk_factors.append(f"High concentration: {concentration*100:.1f}%")
        
        # Liquidity depth
        liquidity_ratio = pool.liquidity_depth / pool.tvl_eth if pool.tvl_eth > 0 else 0
        if liquidity_ratio < 0.001:
            risk_score += 0.2
            risk_factors.append(f"Low liquidity: {liquidity_ratio*100:.3f}% of TVL")
        
        # APY volatility
        if provider in self.apy_history and len(self.apy_history[provider]) > 10:
            apy_vol = np.std(self.apy_history[provider][-30:])
            if apy_vol > 0.01:
                risk_score += 0.1
                risk_factors.append(f"High APY volatility: {apy_vol*100:.2f}%")
        
        return {
            "risk": min(risk_score, 1.0),
            "risk_factors": risk_factors,
            "exchange_rate": pool.exchange_rate,
            "liquidity_ratio": liquidity_ratio
        }
    
    def calculate_composite_score(self, pool: StakingPool) -> float:
        """Calculate composite score for a pool."""
        # APY score (higher is better)
        apy_score = min(pool.apy / 0.05, 1.0)  # 5% = max score
        
        # Depeg risk score (lower is better)
        depeg_score = 1 - pool.depeg_risk
        
        # Liquidity score (higher is better)
        liquidity_score = min(pool.liquidity_depth / 50_000, 1.0)
        
        # TVL score (higher = more established)
        tvl_score = min(pool.tvl_eth / 5_000_000, 1.0)
        
        # Fee score (lower is better)
        fee_score = 1 - pool.fee_pct
        
        # Weighted composite
        composite = (
            apy_score * 0.30 +
            depeg_score * 0.25 +
            liquidity_score * 0.20 +
            tvl_score * 0.15 +
            fee_score * 0.10
        )
        
        return composite
    
    def find_best_pools(
        self,
        min_liquidity: float = 1000,
        max_depeg_risk: float = 0.2
    ) -> List[StakingOpportunity]:
        """Find best staking pools."""
        opportunities = []
        
        for provider, pool in self.pools.items():
            if pool.liquidity_depth < min_liquidity:
                continue
            if pool.depeg_risk > max_depeg_risk:
                continue
            
            score = self.calculate_composite_score(pool)
            
            opp = StakingOpportunity(
                provider=provider,
                token=pool.token,
                apy=pool.apy,
                tvl_eth=pool.tvl_eth,
                depeg_risk=pool.depeg_risk,
                liquidity_score=min(pool.liquidity_depth / 50_000, 1.0),
                composite_score=score,
                recommended_allocation_pct=0  # Will be calculated
            )
            
            opportunities.append(opp)
        
        # Sort by composite score
        opportunities.sort(key=lambda o: o.composite_score, reverse=True)
        
        # Calculate allocations (proportional to score)
        total_score = sum(o.composite_score for o in opportunities)
        if total_score > 0:
            for opp in opportunities:
                opp.recommended_allocation_pct = (opp.composite_score / total_score) * 100
        
        return opportunities


class StakingOptimizer:
    """
    Staking Optimizer
    =================
    Optimizes staking allocation.
    """
    
    def __init__(self, total_eth: float = 100.0):
        self.total_eth = total_eth
        self.analyzer = LiquidStakingAnalyzer()
        self.positions: Dict[LiquidStakingProvider, StakingPosition] = {}
    
    def calculate_optimal_allocation(
        self,
        risk_tolerance: float = 0.5  # 0 = conservative, 1 = aggressive
    ) -> Dict[str, Any]:
        """Calculate optimal staking allocation."""
        opportunities = self.analyzer.find_best_pools()
        
        if not opportunities:
            return {"allocations": [], "expected_apy": 0}
        
        # Adjust for risk tolerance
        if risk_tolerance < 0.3:
            # Conservative: prefer low depeg risk
            for opp in opportunities:
                opp.composite_score *= (1 - opp.depeg_risk)
        elif risk_tolerance > 0.7:
            # Aggressive: prefer high APY
            for opp in opportunities:
                opp.composite_score *= (1 + opp.apy)
        
        # Recalculate allocations
        total_score = sum(o.composite_score for o in opportunities)
        
        allocations = []
        weighted_apy = 0
        
        for opp in opportunities:
            if total_score > 0:
                allocation_pct = (opp.composite_score / total_score) * 100
            else:
                allocation_pct = 100 / len(opportunities)
            
            allocation_eth = self.total_eth * (allocation_pct / 100)
            
            allocations.append({
                "provider": opp.provider.value,
                "token": opp.token,
                "allocation_pct": allocation_pct,
                "allocation_eth": allocation_eth,
                "apy": opp.apy * 100,
                "depeg_risk": opp.depeg_risk * 100
            })
            
            weighted_apy += (allocation_pct / 100) * opp.apy
        
        return {
            "total_eth": self.total_eth,
            "risk_tolerance": risk_tolerance,
            "allocations": allocations,
            "expected_apy_pct": weighted_apy * 100,
            "expected_annual_rewards_eth": self.total_eth * weighted_apy
        }
    
    def calculate_restaking_opportunity(
        self,
        staked_token: str,
        restaking_protocol: str = "eigenlayer"
    ) -> Dict[str, Any]:
        """Calculate restaking opportunity."""
        # EigenLayer restaking adds additional yield
        base_restaking_apy = 0.03  # 3% additional
        
        # Get base staking APY
        base_apy = 0.038  # Default Lido rate
        
        for pool in self.analyzer.pools.values():
            if pool.token == staked_token:
                base_apy = pool.apy
                break
        
        total_apy = base_apy + base_restaking_apy
        
        return {
            "staked_token": staked_token,
            "restaking_protocol": restaking_protocol,
            "base_apy_pct": base_apy * 100,
            "restaking_apy_pct": base_restaking_apy * 100,
            "total_apy_pct": total_apy * 100,
            "additional_yield_pct": (base_restaking_apy / base_apy * 100) if base_apy > 0 else 0,
            "risks": [
                "Smart contract risk",
                "Slashing risk (doubled)",
                "Liquidity risk",
                "Protocol risk"
            ]
        }
    
    def calculate_unstaking_strategy(
        self,
        position: StakingPosition,
        target_price_eth: float
    ) -> Dict[str, Any]:
        """Calculate optimal unstaking strategy."""
        # Check if should unstake
        current_value = position.amount_tokens * position.current_rate
        target_value = position.amount_tokens * target_price_eth
        
        should_unstake = target_value > current_value * 1.1  # 10% better
        
        # Calculate costs
        unstaking_fee = 0.001  # 0.1%
        gas_cost_eth = 0.005
        
        net_value = target_value * (1 - unstaking_fee) - gas_cost_eth
        
        return {
            "should_unstake": should_unstake,
            "current_value_eth": current_value,
            "target_value_eth": target_value,
            "unstaking_fee_eth": target_value * unstaking_fee,
            "gas_cost_eth": gas_cost_eth,
            "net_value_eth": net_value,
            "profit_eth": net_value - current_value,
            "days_to_unstake": position.withdrawal_queue
        }


class YieldAggregator:
    """
    Yield Aggregator
    ================
    Aggregates yield opportunities across protocols.
    """
    
    def __init__(self):
        self.staking_analyzer = LiquidStakingAnalyzer()
        self.yield_sources: Dict[str, Dict[str, Any]] = {}
    
    def add_yield_source(
        self,
        source_name: str,
        protocol: str,
        apy: float,
        risk_score: float,
        min_deposit: float = 0
    ) -> None:
        """Add yield source."""
        self.yield_sources[source_name] = {
            "protocol": protocol,
            "apy": apy,
            "risk_score": risk_score,
            "min_deposit": min_deposit,
            "type": "yield"
        }
    
    def get_best_yields(
        self,
        max_risk: float = 0.3,
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get best yield opportunities."""
        filtered = [
            (name, data) for name, data in self.yield_sources.items()
            if data["risk_score"] <= max_risk
        ]
        
        # Sort by APY
        sorted_sources = sorted(filtered, key=lambda x: x[1]["apy"], reverse=True)
        
        return [
            {"source": name, **data}
            for name, data in sorted_sources[:n]
        ]


# Export
__all__ = [
    "LiquidStakingProvider",
    "StakingPool",
    "StakingPosition",
    "StakingOpportunity",
    "LiquidStakingAnalyzer",
    "StakingOptimizer",
    "YieldAggregator"
]
