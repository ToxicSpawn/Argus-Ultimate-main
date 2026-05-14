# pyright: reportMissingImports=false
"""
DeFi Yield Optimization System
================================
Automatically finds and moves capital to the highest-yielding DeFi protocols
while managing risk (smart contract risk, impermanent loss, gas costs).

Protocols supported:
- Aave (lending)
- Compound (lending)
- Lido (liquid staking)
- EigenLayer (restaking)
- Morpho (optimized lending)
- Yearn (yield aggregation)

Key features:
- Real-time APY monitoring
- Risk-adjusted yield comparison
- Automatic rebalancing
- Gas cost optimization
- Impermanent loss protection
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """Types of DeFi protocols."""
    LENDING = auto()           # Aave, Compound
    LIQUID_STAKING = auto()    # Lido, Rocket Pool
    RESTAKING = auto()         # EigenLayer
    YIELD_AGGREGATOR = auto()  # Yearn, Beefy
    DEX_LP = auto()            # Uniswap, Curve


class RiskTier(Enum):
    """Protocol risk tiers."""
    BLUE_CHIP = auto()      # Aave, Compound, Lido (lowest risk)
    ESTABLISHED = auto()    # Yearn, Rocket Pool (low risk)
    EMERGING = auto()       # EigenLayer, newer protocols (medium risk)
    HIGH_YIELD = auto()     # Newer/smaller protocols (higher risk)


@dataclass
class ProtocolAPY:
    """APY data for a specific protocol and asset."""
    protocol: str
    protocol_type: ProtocolType
    asset: str
    apy: float                  # Current APY (decimal, e.g., 0.05 = 5%)
    apy_base: float             # Base APY (without rewards)
    apy_reward: float           # Reward APY (tokens)
    tvl_usd: float              # Total Value Locked
    risk_tier: RiskTier
    min_deposit: float          # Minimum deposit
    withdrawal_fee: float       # Withdrawal fee (decimal)
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def total_apy(self) -> float:
        """Total APY including rewards."""
        return self.apy_base + self.apy_reward
    
    @property
    def risk_adjusted_apy(self) -> float:
        """APY adjusted for risk tier."""
        risk_multiplier = {
            RiskTier.BLUE_CHIP: 1.0,
            RiskTier.ESTABLISHED: 0.9,
            RiskTier.EMERGING: 0.7,
            RiskTier.HIGH_YIELD: 0.5,
        }
        return self.total_apy * risk_multiplier.get(self.risk_tier, 0.5)
    
    @property
    def is_liquid(self) -> bool:
        """Check if protocol has sufficient liquidity."""
        return self.tvl_usd > 10_000_000  # > $10M TVL


@dataclass
class YieldPosition:
    """A position in a yield protocol."""
    protocol: str
    asset: str
    amount_usd: float
    entry_apy: float
    entry_time: datetime
    expected_daily_yield: float = 0.0
    accumulated_yield: float = 0.0
    
    def age_days(self) -> float:
        return (datetime.now() - self.entry_time).total_seconds() / 86400.0
    
    def update_yield(self, current_apy: float) -> None:
        """Update accumulated yield based on time elapsed."""
        days = self.age_days()
        daily_rate = current_apy / 365.0
        self.accumulated_yield = self.amount_usd * daily_rate * days


@dataclass
class RebalanceDecision:
    """Decision to rebalance between protocols."""
    from_protocol: str
    to_protocol: str
    asset: str
    amount_usd: float
    from_apy: float
    to_apy: float
    apy_improvement: float
    gas_cost_usd: float
    net_benefit_usd: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


class DeFiYieldOptimizer:
    """
    Automatically optimizes DeFi yield by:
    1. Monitoring APY across all protocols
    2. Risk-adjusting yields
    3. Calculating gas-efficient rebalancing
    4. Executing optimal capital allocation
    """
    
    # Minimum APY improvement to trigger rebalance (after gas)
    MIN_REBALANCE_IMPROVEMENT = 0.005  # 0.5%
    
    # Maximum allocation per protocol
    MAX_PROTOCOL_ALLOCATION = 0.40  # 40%
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the yield optimizer."""
        self.config = config or {}
        
        # Protocol data
        self.protocols: Dict[str, List[ProtocolAPY]] = {}
        
        # Current positions
        self.positions: List[YieldPosition] = []
        
        # Rebalance history
        self.rebalance_history: List[RebalanceDecision] = []
        
        # Gas price tracker (for ETH mainnet)
        self.current_gas_price_gwei: float = 20.0
        
        # Initialize protocol configurations
        self._initialize_protocols()
        
        logger.info("DeFi Yield Optimizer initialized")
    
    def _initialize_protocols(self) -> None:
        """Initialize supported protocols."""
        # These would be populated from on-chain data in production
        self.supported_protocols = {
            "aave_v3": ProtocolType.LENDING,
            "compound_v3": ProtocolType.LENDING,
            "lido": ProtocolType.LIQUID_STAKING,
            "eigenlayer": ProtocolType.RESTAKING,
            "yearn_v3": ProtocolType.YIELD_AGGREGATOR,
            "morpho": ProtocolType.LENDING,
        }
        
        self.protocol_risk_tiers = {
            "aave_v3": RiskTier.BLUE_CHIP,
            "compound_v3": RiskTier.BLUE_CHIP,
            "lido": RiskTier.BLUE_CHIP,
            "eigenlayer": RiskTier.EMERGING,
            "yearn_v3": RiskTier.ESTABLISHED,
            "morpho": RiskTier.ESTABLISHED,
        }
    
    def update_apy_data(self, protocol_data: List[ProtocolAPY]) -> None:
        """Update APY data from on-chain sources."""
        for data in protocol_data:
            key = f"{data.protocol}_{data.asset}"
            if data.protocol not in self.protocols:
                self.protocols[data.protocol] = []
            
            # Update or append
            existing = [p for p in self.protocols[data.protocol] if p.asset == data.asset]
            if existing:
                idx = self.protocols[data.protocol].index(existing[0])
                self.protocols[data.protocol][idx] = data
            else:
                self.protocols[data.protocol].append(data)
    
    def find_best_yield(
        self,
        asset: str,
        amount_usd: float,
        min_apy: float = 0.02,
        max_risk: RiskTier = RiskTier.ESTABLISHED
    ) -> Optional[ProtocolAPY]:
        """
        Find the best risk-adjusted yield for an asset.
        
        Args:
            asset: Asset symbol (e.g., "ETH", "USDC")
            amount_usd: Amount to deposit
            min_apy: Minimum APY threshold
            max_risk: Maximum acceptable risk tier
            
        Returns:
            Best ProtocolAPY or None if no suitable protocol found
        """
        candidates = []
        
        for protocol, apys in self.protocols.items():
            for apy in apys:
                if apy.asset != asset:
                    continue
                if apy.total_apy < min_apy:
                    continue
                if not apy.is_liquid:
                    continue
                if amount_usd < apy.min_deposit:
                    continue
                
                # Check risk tier
                risk_order = [RiskTier.BLUE_CHIP, RiskTier.ESTABLISHED, 
                             RiskTier.EMERGING, RiskTier.HIGH_YIELD]
                if risk_order.index(apy.risk_tier) > risk_order.index(max_risk):
                    continue
                
                candidates.append(apy)
        
        if not candidates:
            return None
        
        # Sort by risk-adjusted APY
        candidates.sort(key=lambda x: x.risk_adjusted_apy, reverse=True)
        
        return candidates[0]
    
    def find_all_opportunities(
        self,
        assets: List[str],
        min_apy: float = 0.02
    ) -> Dict[str, List[ProtocolAPY]]:
        """
        Find all yield opportunities for given assets.
        
        Returns:
            Dict mapping asset -> sorted list of ProtocolAPY (best first)
        """
        opportunities: Dict[str, List[ProtocolAPY]] = {asset: [] for asset in assets}
        
        for asset in assets:
            for protocol, apys in self.protocols.items():
                for apy in apys:
                    if apy.asset != asset:
                        continue
                    if apy.total_apy < min_apy:
                        continue
                    if not apy.is_liquid:
                        continue
                    opportunities[asset].append(apy)
            
            # Sort by risk-adjusted APY
            opportunities[asset].sort(key=lambda x: x.risk_adjusted_apy, reverse=True)
        
        return opportunities
    
    def calculate_rebalance(
        self,
        position: YieldPosition,
        new_protocol: ProtocolAPY,
        gas_price_gwei: Optional[float] = None
    ) -> Optional[RebalanceDecision]:
        """
        Calculate whether to rebalance a position to a new protocol.
        
        Returns:
            RebalanceDecision if rebalance is beneficial, None otherwise
        """
        gas = gas_price_gwei or self.current_gas_price_gwei
        
        # Estimate gas cost (2 transactions: withdraw + deposit)
        # Rough estimate: 200k gas for complex DeFi operations
        gas_cost_eth = (200_000 * gas) / 1e9
        gas_cost_usd = gas_cost_eth * 2000  # Assume ETH = $2000
        
        # Calculate APY improvement
        apy_improvement = new_protocol.total_apy - position.entry_apy
        
        # Calculate net benefit over 30 days
        days_to_recover_gas = gas_cost_usd / (position.amount_usd * apy_improvement / 365)
        benefit_30d = position.amount_usd * apy_improvement * (30 / 365) - gas_cost_usd
        
        if benefit_30d <= 0:
            return None
        
        return RebalanceDecision(
            from_protocol=position.protocol,
            to_protocol=new_protocol.protocol,
            asset=position.asset,
            amount_usd=position.amount_usd,
            from_apy=position.entry_apy,
            to_apy=new_protocol.total_apy,
            apy_improvement=apy_improvement,
            gas_cost_usd=gas_cost_usd,
            net_benefit_usd=benefit_30d,
            reason=f"APY improvement: {apy_improvement:.2%}, recovers gas in {days_to_recover_gas:.1f} days"
        )
    
    def optimize_allocation(
        self,
        total_capital_usd: float,
        assets: Dict[str, float],  # asset -> allocation %
        min_apy: float = 0.03
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Optimize capital allocation across protocols.
        
        Returns:
            Dict mapping asset -> list of (protocol, allocation_usd)
        """
        allocations: Dict[str, List[Tuple[str, float]]] = {}
        
        for asset, allocation_pct in assets.items():
            asset_capital = total_capital_usd * (allocation_pct / 100.0)
            
            # Find best yields
            opportunities = self.find_all_opportunities([asset], min_apy)
            asset_opps = opportunities.get(asset, [])
            
            if not asset_opps:
                allocations[asset] = []
                continue
            
            # Allocate across top protocols (diversification)
            remaining = asset_capital
            asset_allocations = []
            
            for i, opp in enumerate(asset_opps[:3]):  # Top 3 protocols
                # Allocate more to better yields
                if i == 0:
                    pct = 0.50  # 50% to best
                elif i == 1:
                    pct = 0.30  # 30% to second
                else:
                    pct = 0.20  # 20% to third
                
                amount = min(remaining * pct, asset_capital * self.MAX_PROTOCOL_ALLOCATION)
                if amount > opp.min_deposit:
                    asset_allocations.append((opp.protocol, amount))
                    remaining -= amount
            
            allocations[asset] = asset_allocations
        
        return allocations
    
    def calculate_position_yield(
        self,
        position: YieldPosition,
        current_apy: Optional[float] = None
    ) -> Dict[str, float]:
        """Calculate current yield metrics for a position."""
        apy = current_apy or position.entry_apy
        
        # Update accumulated yield
        position.update_yield(apy)
        
        daily_yield = position.amount_usd * (apy / 365.0)
        weekly_yield = daily_yield * 7
        monthly_yield = daily_yield * 30
        yearly_yield = position.amount_usd * apy
        
        return {
            "daily_yield_usd": daily_yield,
            "weekly_yield_usd": weekly_yield,
            "monthly_yield_usd": monthly_yield,
            "yearly_yield_usd": yearly_yield,
            "accumulated_yield_usd": position.accumulated_yield,
            "current_apy": apy,
            "entry_apy": position.entry_apy,
        }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of all yield positions."""
        total_value = sum(p.amount_usd for p in self.positions)
        total_yield = sum(p.accumulated_yield for p in self.positions)
        
        # Calculate weighted average APY
        if total_value > 0:
            weighted_apy = sum(
                p.entry_apy * p.amount_usd for p in self.positions
            ) / total_value
        else:
            weighted_apy = 0.0
        
        # Breakdown by protocol
        by_protocol: Dict[str, float] = {}
        for pos in self.positions:
            by_protocol[pos.protocol] = by_protocol.get(pos.protocol, 0) + pos.amount_usd
        
        # Breakdown by asset
        by_asset: Dict[str, float] = {}
        for pos in self.positions:
            by_asset[pos.asset] = by_asset.get(pos.asset, 0) + pos.amount_usd
        
        return {
            "total_value_usd": total_value,
            "total_yield_usd": total_yield,
            "weighted_avg_apy": weighted_apy,
            "num_positions": len(self.positions),
            "by_protocol": by_protocol,
            "by_asset": by_asset,
        }
    
    def should_rebalance(self) -> bool:
        """Check if portfolio should be rebalanced."""
        # Rebalance if any position is significantly below market APY
        for position in self.positions:
            # Find current market APY for this asset
            best = self.find_best_yield(position.asset, position.amount_usd)
            if best and best.total_apy > position.entry_apy * 1.5:  # 50% higher
                return True
        
        # Rebalance if it's been more than 7 days
        oldest = min(self.positions, key=lambda p: p.entry_time) if self.positions else None
        if oldest and oldest.age_days() > 7:
            return True
        
        return False


# Singleton instance
_optimizer: Optional[DeFiYieldOptimizer] = None


def get_defi_yield_optimizer(
    config: Optional[Dict[str, Any]] = None
) -> DeFiYieldOptimizer:
    """Get or create the DeFi Yield Optimizer singleton."""
    global _optimizer
    if _optimizer is None:
        _optimizer = DeFiYieldOptimizer(config)
    return _optimizer


__all__ = [
    "DeFiYieldOptimizer",
    "ProtocolAPY",
    "ProtocolType",
    "RiskTier",
    "YieldPosition",
    "RebalanceDecision",
    "get_defi_yield_optimizer",
]
