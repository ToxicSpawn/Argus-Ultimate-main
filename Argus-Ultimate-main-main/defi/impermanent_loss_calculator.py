"""
Impermanent Loss Calculator
============================
Calculates and manages impermanent loss for LP positions:
- IL calculation for any price change
- Fee income estimation
- Break-even analysis
- Optimal range selection (for concentrated liquidity)
- IL hedging strategies
- Historical IL simulation

Formula: IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LPPosition:
    """Liquidity provider position."""
    pool_name: str
    token0: str
    token1: str
    initial_amount0: float
    initial_amount1: float
    initial_price: float  # Price of token0 in terms of token1
    current_price: float
    fee_rate: float  # Pool fee rate (e.g., 0.003 for 0.3%)
    tvl: float = 0.0
    volume_24h: float = 0.0
    is_concentrated: bool = False
    lower_price: Optional[float] = None
    upper_price: Optional[float] = None


@dataclass
class ILResult:
    """Impermanent loss calculation result."""
    initial_value: float
    hold_value: float  # Value if just held
    lp_value: float  # Value as LP
    impermanent_loss: float
    impermanent_loss_pct: float
    fee_income: float
    net_value: float
    net_profit_loss: float
    net_profit_loss_pct: float
    break_even_days: Optional[float] = None


@dataclass
class ILScenario:
    """IL scenario analysis."""
    price_change_pct: float
    new_price: float
    il_pct: float
    fees_needed_pct: float
    days_to_breakeven: Optional[float] = None


class ImpermanentLossCalculator:
    """
    Impermanent Loss Calculator
    ===========================
    Calculates IL for standard and concentrated liquidity positions.
    """
    
    @staticmethod
    def calculate_il(
        initial_price: float,
        current_price: float
    ) -> float:
        """
        Calculate impermanent loss percentage.
        
        Formula: IL = 2 * sqrt(r) / (1 + r) - 1
        where r = current_price / initial_price
        """
        if initial_price <= 0 or current_price <= 0:
            return 0.0
        
        price_ratio = current_price / initial_price
        
        if price_ratio <= 0:
            return 0.0
        
        il = (2 * math.sqrt(price_ratio) / (1 + price_ratio)) - 1
        
        return il
    
    @staticmethod
    def calculate_il_amount(
        initial_value: float,
        initial_price: float,
        current_price: float
    ) -> Dict[str, float]:
        """Calculate IL in absolute amounts."""
        price_ratio = current_price / initial_price
        
        # Value if held
        hold_value = initial_value
        
        # Value as LP
        il_pct = ImpermanentLossCalculator.calculate_il(initial_price, current_price)
        lp_value = initial_value * (1 + il_pct)
        
        # Impermanent loss amount
        il_amount = hold_value - lp_value
        
        return {
            "initial_value": initial_value,
            "hold_value": hold_value,
            "lp_value": lp_value,
            "impermanent_loss": il_amount,
            "impermanent_loss_pct": il_pct * 100
        }
    
    @staticmethod
    def calculate_concentrated_il(
        initial_price: float,
        current_price: float,
        lower_price: float,
        upper_price: float
    ) -> Dict[str, float]:
        """Calculate IL for concentrated liquidity (Uniswap V3 style)."""
        if current_price <= lower_price:
            # All in token0
            price_ratio = lower_price / initial_price
            il = (2 * math.sqrt(price_ratio) / (1 + price_ratio)) - 1
            out_of_range = True
        elif current_price >= upper_price:
            # All in token1
            price_ratio = upper_price / initial_price
            il = (2 * math.sqrt(price_ratio) / (1 + price_ratio)) - 1
            out_of_range = True
        else:
            # In range - standard IL calculation
            il = ImpermanentLossCalculator.calculate_il(initial_price, current_price)
            out_of_range = False
        
        # Concentrated liquidity amplifies IL
        range_width = upper_price - lower_price
        range_center = (upper_price + lower_price) / 2
        concentration_factor = initial_price / range_width if range_width > 0 else 1
        
        # Adjusted IL for concentration
        adjusted_il = il * min(concentration_factor, 10)  # Cap at 10x
        
        return {
            "impermanent_loss_pct": il * 100,
            "adjusted_il_pct": adjusted_il * 100,
            "out_of_range": out_of_range,
            "concentration_factor": concentration_factor
        }
    
    @staticmethod
    def calculate_fee_income(
        volume_24h: float,
        fee_rate: float,
        pool_share: float,
        days: int = 1
    ) -> float:
        """Calculate fee income over a period."""
        daily_fees = volume_24h * fee_rate
        user_fees = daily_fees * pool_share
        
        return user_fees * days
    
    @staticmethod
    def calculate_break_even(
        impermanent_loss_pct: float,
        daily_fee_yield_pct: float
    ) -> Optional[float]:
        """Calculate days to break even on IL."""
        if daily_fee_yield_pct <= 0:
            return None
        
        # Need fees to cover IL
        days = abs(impermanent_loss_pct) / daily_fee_yield_pct
        
        return days
    
    def analyze_position(self, position: LPPosition, days: int = 30) -> ILResult:
        """Complete analysis of an LP position."""
        # Calculate initial value
        initial_value = position.initial_amount0 * position.initial_price + position.initial_amount1
        
        # Calculate IL
        il_result = self.calculate_il_amount(
            initial_value,
            position.initial_price,
            position.current_price
        )
        
        # Estimate fee income
        pool_share = position.tvl / (position.tvl + position.volume_24h) if position.tvl > 0 else 0.001
        fee_income = self.calculate_fee_income(
            position.volume_24h,
            position.fee_rate,
            pool_share,
            days
        )
        
        # Calculate daily fee yield
        daily_fee_yield = fee_income / days / initial_value * 100 if initial_value > 0 else 0
        
        # Break even
        break_even = self.calculate_break_even(
            il_result["impermanent_loss_pct"],
            daily_fee_yield
        )
        
        # Net result
        net_value = il_result["lp_value"] + fee_income
        net_profit = net_value - initial_value
        
        return ILResult(
            initial_value=initial_value,
            hold_value=il_result["hold_value"],
            lp_value=il_result["lp_value"],
            impermanent_loss=il_result["impermanent_loss"],
            impermanent_loss_pct=il_result["impermanent_loss_pct"],
            fee_income=fee_income,
            net_value=net_value,
            net_profit_loss=net_profit,
            net_profit_loss_pct=(net_profit / initial_value * 100) if initial_price > 0 else 0,
            break_even_days=break_even
        )
    
    def scenario_analysis(
        self,
        position: LPPosition,
        price_changes: List[float] = None
    ) -> List[ILScenario]:
        """Analyze IL across multiple price scenarios."""
        if price_changes is None:
            price_changes = [-50, -30, -20, -10, -5, 0, 5, 10, 20, 30, 50, 100]
        
        scenarios = []
        
        for change_pct in price_changes:
            new_price = position.initial_price * (1 + change_pct / 100)
            
            # Calculate IL
            il_pct = self.calculate_il(position.initial_price, new_price) * 100
            
            # Calculate fees needed to break even
            pool_share = position.tvl / (position.tvl + position.volume_24h) if position.tvl > 0 else 0.001
            daily_fee_yield = position.volume_24h * position.fee_rate * pool_share / position.tvl * 100 if position.tvl > 0 else 0
            
            fees_needed = abs(il_pct) if il_pct < 0 else 0
            days_to_breakeven = self.calculate_break_even(il_pct, daily_fee_yield)
            
            scenarios.append(ILScenario(
                price_change_pct=change_pct,
                new_price=new_price,
                il_pct=il_pct,
                fees_needed_pct=fees_needed,
                days_to_breakeven=days_to_breakeven
            ))
        
        return scenarios


class ILHedgingStrategy:
    """
    IL Hedging Strategy
    ===================
    Strategies to hedge impermanent loss.
    """
    
    @staticmethod
    def calculate_delta_hedge(
        position_value: float,
        price_change_pct: float,
        hedge_ratio: float = 0.5
    ) -> Dict[str, Any]:
        """Calculate delta hedging strategy."""
        # Delta hedge: short perpetual to offset IL
        il_pct = ImpermanentLossCalculator.calculate_il(1.0, 1 + price_change_pct / 100)
        
        hedge_size = position_value * hedge_ratio
        hedge_pnl = -hedge_size * (price_change_pct / 100)  # Short profits when price drops
        
        il_loss = position_value * il_pct
        hedged_il = il_loss + hedge_pnl
        
        return {
            "hedge_size": hedge_size,
            "hedge_ratio": hedge_ratio,
            "il_without_hedge": il_loss,
            "hedge_pnl": hedge_pnl,
            "net_il": hedged_il,
            "il_reduction_pct": (1 - abs(hedged_il) / abs(il_loss)) * 100 if il_loss != 0 else 0
        }
    
    @staticmethod
    def calculate_range_optimal(
        current_price: float,
        volatility: float,
        target_il_pct: float = 5.0
    ) -> Dict[str, float]:
        """Calculate optimal range for concentrated liquidity."""
        # Based on volatility, calculate range that keeps IL under target
        # Using normal distribution assumption
        
        # 1 standard deviation = ~68% of price moves
        std_dev = current_price * volatility
        
        # Calculate range based on target IL
        # IL at ±X% price move
        target_price_change = target_il_pct / 100 * 2  # Approximate
        
        lower_price = current_price * (1 - target_price_change)
        upper_price = current_price * (1 + target_price_change)
        
        # Adjust for volatility
        volatility_range = std_dev * 2  # 2 std devs = 95%
        
        return {
            "current_price": current_price,
            "lower_price": max(lower_price, current_price * 0.5),
            "upper_price": upper_price,
            "range_width_pct": (upper_price - lower_price) / current_price * 100,
            "expected_il_pct": target_il_pct,
            "volatility_adjusted_range": volatility_range / current_price * 100
        }
    
    @staticmethod
    def compare_strategies(
        initial_value: float,
        price_change_pct: float,
        days: int = 30,
        daily_volume: float = 1000000,
        fee_rate: float = 0.003
    ) -> Dict[str, Any]:
        """Compare different strategies."""
        # Strategy 1: Hold
        hold_value = initial_value * (1 + price_change_pct / 100)
        
        # Strategy 2: Standard LP
        il_pct = ImpermanentLossCalculator.calculate_il(1.0, 1 + price_change_pct / 100)
        lp_value = initial_value * (1 + il_pct)
        
        # Estimate fees
        pool_share = initial_value / (initial_value + daily_volume)
        daily_fees = daily_volume * fee_rate * pool_share
        total_fees = daily_fees * days
        
        lp_with_fees = lp_value + total_fees
        
        # Strategy 3: Hedged LP
        hedge = ILHedgingStrategy.calculate_delta_hedge(initial_value, price_change_pct, 0.5)
        hedged_lp = lp_with_fees + hedge["hedge_pnl"]
        
        return {
            "hold": {
                "value": hold_value,
                "pnl": hold_value - initial_value,
                "pnl_pct": price_change_pct
            },
            "lp": {
                "value": lp_with_fees,
                "pnl": lp_with_fees - initial_value,
                "pnl_pct": (lp_with_fees - initial_value) / initial_value * 100,
                "il_pct": il_pct * 100,
                "fees": total_fees
            },
            "hedged_lp": {
                "value": hedged_lp,
                "pnl": hedged_lp - initial_value,
                "pnl_pct": (hedged_lp - initial_value) / initial_value * 100,
                "hedge_pnl": hedge["hedge_pnl"]
            },
            "recommendation": "hold" if hold_value > max(lp_with_fees, hedged_lp) else
                             "lp" if lp_with_fees > hedged_lp else "hedged_lp"
        }


class ILVisualizer:
    """
    IL Visualizer
    =============
    Generates data for IL visualization.
    """
    
    @staticmethod
    def generate_il_curve(
        initial_price: float = 1.0,
        price_range: Tuple[float, float] = (0.1, 10.0),
        points: int = 100
    ) -> List[Dict[str, float]]:
        """Generate IL curve data points."""
        prices = np.logspace(
            np.log10(price_range[0]),
            np.log10(price_range[1]),
            points
        )
        
        curve_data = []
        for price in prices:
            il = ImpermanentLossCalculator.calculate_il(initial_price, price)
            curve_data.append({
                "price_ratio": price / initial_price,
                "price_change_pct": (price / initial_price - 1) * 100,
                "il_pct": il * 100,
                "il_amount": 100 * il  # Per $100 invested
            })
        
        return curve_data
    
    @staticmethod
    def generate_fee_analysis(
        il_pct: float,
        fee_rates: List[float] = None,
        volumes: List[float] = None,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Generate fee analysis data."""
        if fee_rates is None:
            fee_rates = [0.0001, 0.0005, 0.001, 0.003, 0.005, 0.01]
        if volumes is None:
            volumes = [100000, 500000, 1000000, 5000000, 10000000]
        
        analysis = []
        
        for fee_rate in fee_rates:
            for volume in volumes:
                # Assume $1M TVL position
                tvl = 1000000
                pool_share = tvl / (tvl + volume)
                daily_fees = volume * fee_rate * pool_share
                total_fees = daily_fees * days
                fee_yield = total_fees / tvl * 100
                
                profitable = fee_yield > abs(il_pct)
                
                analysis.append({
                    "fee_rate": fee_rate * 100,
                    "volume_24h": volume,
                    "daily_fees": daily_fees,
                    "total_fees": total_fees,
                    "fee_yield_pct": fee_yield,
                    "il_pct": abs(il_pct),
                    "net_yield_pct": fee_yield - abs(il_pct),
                    "profitable": profitable
                })
        
        return analysis


# Export
__all__ = [
    "LPPosition",
    "ILResult",
    "ILScenario",
    "ImpermanentLossCalculator",
    "ILHedgingStrategy",
    "ILVisualizer"
]
