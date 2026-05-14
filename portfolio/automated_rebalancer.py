"""
Automated Portfolio Rebalancer
===============================
Features:
- Target allocation management
- Drift detection and correction
- Tax-efficient rebalancing
- Risk-parity allocation
- Momentum-based tilting
- Automatic threshold-based triggers
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class RebalanceMethod(Enum):
    """Rebalancing methods."""
    THRESHOLD = "threshold"  # Rebalance when drift exceeds threshold
    SCHEDULED = "scheduled"  # Rebalance on schedule
    MOMENTUM = "momentum"  # Momentum-based tilting
    RISK_PARITY = "risk_parity"  # Risk parity allocation
    ADAPTIVE = "adaptive"  # Adaptive based on conditions


class AllocationMethod(Enum):
    """Allocation methods."""
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP = "market_cap"
    RISK_PARITY = "risk_parity"
    MAX_SHARPE = "max_sharpe"
    MIN_VAR = "min_variance"
    CUSTOM = "custom"


@dataclass
class AssetAllocation:
    """Target allocation for an asset."""
    symbol: str
    target_pct: float  # Target percentage
    min_pct: float = 0.0  # Minimum percentage
    max_pct: float = 1.0  # Maximum percentage
    current_pct: float = 0.0  # Current percentage
    drift_pct: float = 0.0  # Drift from target
    enabled: bool = True


@dataclass
class RebalanceConfig:
    """Rebalancing configuration."""
    method: RebalanceMethod = RebalanceMethod.THRESHOLD
    allocation_method: AllocationMethod = AllocationMethod.CUSTOM
    drift_threshold_pct: float = 5.0  # Rebalance when drift > 5%
    min_trade_size_usd: float = 10.0  # Minimum trade size
    max_trade_size_pct: float = 0.25  # Max 25% of portfolio per trade
    rebalance_fee_pct: float = 0.1  # Estimated fee per trade
    tax_efficient: bool = False  # Tax-efficient mode
    schedule_hours: int = 24  # For scheduled rebalancing
    momentum_window_days: int = 30  # For momentum-based


@dataclass
class RebalanceTrade:
    """A single rebalancing trade."""
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    current_price: float
    target_value_usd: float
    current_value_usd: float
    drift_pct: float
    estimated_fee: float
    priority: int = 0  # Higher = execute first


@dataclass
class RebalanceResult:
    """Result of rebalancing."""
    timestamp: float
    method_used: str
    trades: List[RebalanceTrade]
    total_trades: int
    total_value_traded_usd: float
    estimated_fees_usd: float
    portfolio_value_usd: float
    allocations_before: Dict[str, float]
    allocations_after: Dict[str, float]
    success: bool = True
    error: Optional[str] = None


class PortfolioRebalancer:
    """
    Automated Portfolio Rebalancer
    ==============================
    Manages and rebalances portfolio allocations.
    """
    
    def __init__(self, config: Optional[RebalanceConfig] = None):
        self.config = config or RebalanceConfig()
        self.allocations: Dict[str, AssetAllocation] = {}
        self.portfolio_value_usd: float = 0.0
        self.prices: Dict[str, float] = {}
        self.rebalance_history: List[RebalanceResult] = []
        
        # Momentum data
        self.price_history: Dict[str, List[float]] = {}
        
        logger.info("PortfolioRebalancer initialized")
    
    def set_allocations(self, allocations: List[AssetAllocation]) -> None:
        """Set target allocations."""
        for alloc in allocations:
            self.allocations[alloc.symbol] = alloc
        logger.info(f"Set {len(allocations)} target allocations")
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices."""
        self.prices.update(prices)
        
        # Update price history for momentum
        for symbol, price in prices.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(price)
            # Keep only recent history
            if len(self.price_history[symbol]) > 1000:
                self.price_history[symbol] = self.price_history[symbol][-1000:]
    
    def update_portfolio(self, holdings: Dict[str, float], portfolio_value: float) -> None:
        """Update current portfolio state."""
        self.portfolio_value_usd = portfolio_value
        
        # Calculate current percentages
        for symbol, alloc in self.allocations.items():
            if symbol in holdings:
                current_value = holdings[symbol] * self.prices.get(symbol, 0)
                alloc.current_pct = current_value / portfolio_value if portfolio_value > 0 else 0
                alloc.drift_pct = alloc.current_pct - alloc.target_pct
            else:
                alloc.current_pct = 0
                alloc.drift_pct = -alloc.target_pct
    
    def check_rebalance_needed(self) -> Tuple[bool, List[str]]:
        """Check if rebalancing is needed."""
        if self.config.method == RebalanceMethod.THRESHOLD:
            drifted = []
            for symbol, alloc in self.allocations.items():
                if alloc.enabled and abs(alloc.drift_pct * 100) > self.config.drift_threshold_pct:
                    drifted.append(symbol)
            return len(drifted) > 0, drifted
        
        elif self.config.method == RebalanceMethod.SCHEDULED:
            if self.rebalance_history:
                last_rebalance = self.rebalance_history[-1].timestamp
                hours_since = (time.time() - last_rebalance) / 3600
                return hours_since >= self.config.schedule_hours, []
            return True, []
        
        return False, []
    
    def calculate_trades_threshold(self) -> List[RebalanceTrade]:
        """Calculate trades for threshold-based rebalancing."""
        trades = []
        
        for symbol, alloc in self.allocations.items():
            if not alloc.enabled:
                continue
            
            drift = alloc.drift_pct
            if abs(drift * 100) < self.config.drift_threshold_pct:
                continue
            
            # Calculate target and current values
            target_value = self.portfolio_value_usd * alloc.target_pct
            current_value = self.portfolio_value_usd * alloc.current_pct
            diff_value = target_value - current_value
            
            if abs(diff_value) < self.config.min_trade_size_usd:
                continue
            
            price = self.prices.get(symbol, 0)
            if price <= 0:
                continue
            
            quantity = abs(diff_value) / price
            
            # Limit trade size
            max_trade_value = self.portfolio_value_usd * self.config.max_trade_size_pct
            if abs(diff_value) > max_trade_value:
                quantity = max_trade_value / price
                diff_value = max_trade_value * np.sign(diff_value)
            
            trade = RebalanceTrade(
                symbol=symbol,
                side="buy" if diff_value > 0 else "sell",
                quantity=quantity,
                current_price=price,
                target_value_usd=target_value,
                current_value_usd=current_value,
                drift_pct=drift * 100,
                estimated_fee=abs(diff_value) * self.config.rebalance_fee_pct / 100,
                priority=int(abs(drift) * 100)  # Higher drift = higher priority
            )
            
            trades.append(trade)
        
        # Sort by priority (highest drift first)
        trades.sort(key=lambda t: t.priority, reverse=True)
        
        return trades
    
    def calculate_trades_risk_parity(self) -> List[RebalanceTrade]:
        """Calculate trades for risk parity allocation."""
        # Calculate volatility for each asset
        volatilities = {}
        for symbol, alloc in self.allocations.items():
            if symbol in self.price_history and len(self.price_history[symbol]) > 20:
                prices = np.array(self.price_history[symbol][-20:])
                returns = np.diff(np.log(prices))
                volatilities[symbol] = np.std(returns) * np.sqrt(365)  # Annualized
            else:
                volatilities[symbol] = 0.5  # Default 50% volatility
        
        # Risk parity: weight inversely proportional to volatility
        total_inv_vol = sum(1 / v for v in volatilities.values() if v > 0)
        
        if total_inv_vol > 0:
            for symbol, alloc in self.allocations.items():
                if alloc.enabled and volatilities.get(symbol, 0) > 0:
                    inv_vol = 1 / volatilities[symbol]
                    alloc.target_pct = inv_vol / total_inv_vol
        
        # Now calculate trades for new targets
        return self.calculate_trades_threshold()
    
    def calculate_trades_momentum(self) -> List[RebalanceTrade]:
        """Calculate trades with momentum tilting."""
        # Calculate momentum for each asset
        momentums = {}
        for symbol, alloc in self.allocations.items():
            if symbol in self.price_history and len(self.price_history[symbol]) >= self.config.momentum_window_days:
                prices = self.price_history[symbol]
                recent = np.mean(prices[-7:])  # Last week
                older = np.mean(prices[-self.config.momentum_window_days:-self.config.momentum_window_days+7])
                momentums[symbol] = (recent - older) / older if older > 0 else 0
            else:
                momentums[symbol] = 0
        
        # Tilt allocation based on momentum
        momentum_factor = 0.2  # 20% tilt
        total_momentum = sum(max(0, m) for m in momentums.values())
        
        if total_momentum > 0:
            for symbol, alloc in self.allocations.items():
                if alloc.enabled:
                    momentum = momentums.get(symbol, 0)
                    if momentum > 0:
                        # Increase allocation for positive momentum
                        tilt = (momentum / total_momentum) * momentum_factor
                        alloc.target_pct = min(alloc.max_pct, alloc.target_pct + tilt)
        
        # Normalize to 100%
        total = sum(a.target_pct for a in self.allocations.values() if a.enabled)
        if total > 0:
            for alloc in self.allocations.values():
                if alloc.enabled:
                    alloc.target_pct /= total
        
        return self.calculate_trades_threshold()
    
    def calculate_trades(self) -> List[RebalanceTrade]:
        """Calculate rebalancing trades based on method."""
        if self.config.method == RebalanceMethod.RISK_PARITY:
            return self.calculate_trades_risk_parity()
        elif self.config.method == RebalanceMethod.MOMENTUM:
            return self.calculate_trades_momentum()
        else:
            return self.calculate_trades_threshold()
    
    def execute_rebalance(self) -> RebalanceResult:
        """Execute rebalancing."""
        logger.info("Executing portfolio rebalance")
        
        # Record allocations before
        before = {s: a.current_pct for s, a in self.allocations.items()}
        
        # Calculate trades
        trades = self.calculate_trades()
        
        if not trades:
            logger.info("No trades needed")
            return RebalanceResult(
                timestamp=time.time(),
                method_used=self.config.method.value,
                trades=[],
                total_trades=0,
                total_value_traded_usd=0,
                estimated_fees_usd=0,
                portfolio_value_usd=self.portfolio_value_usd,
                allocations_before=before,
                allocations_after=before
            )
        
        # Execute trades (simulated)
        total_value = 0
        total_fees = 0
        
        for trade in trades:
            value = trade.quantity * trade.current_price
            total_value += value
            total_fees += trade.estimated_fee
            
            logger.info(
                f"Trade: {trade.side.upper()} {trade.quantity:.4f} {trade.symbol} "
                f"@ ${trade.current_price:.2f} (drift: {trade.drift_pct:.1f}%)"
            )
        
        # Update allocations after
        for trade in trades:
            if trade.symbol in self.allocations:
                alloc = self.allocations[trade.symbol]
                if trade.side == "buy":
                    alloc.current_pct += (trade.quantity * trade.current_price) / self.portfolio_value_usd
                else:
                    alloc.current_pct -= (trade.quantity * trade.current_price) / self.portfolio_value_usd
                alloc.drift_pct = alloc.current_pct - alloc.target_pct
        
        after = {s: a.current_pct for s, a in self.allocations.items()}
        
        result = RebalanceResult(
            timestamp=time.time(),
            method_used=self.config.method.value,
            trades=trades,
            total_trades=len(trades),
            total_value_traded_usd=total_value,
            estimated_fees_usd=total_fees,
            portfolio_value_usd=self.portfolio_value_usd,
            allocations_before=before,
            allocations_after=after
        )
        
        self.rebalance_history.append(result)
        
        logger.info(f"Rebalance complete: {len(trades)} trades, ${total_value:.2f} traded")
        
        return result
    
    async def auto_rebalance(self) -> Optional[RebalanceResult]:
        """Automatically check and rebalance if needed."""
        needed, drifted = self.check_rebalance_needed()
        
        if needed:
            logger.info(f"Rebalance needed for: {drifted}")
            return self.execute_rebalance()
        
        return None
    
    def get_allocation_report(self) -> Dict[str, Any]:
        """Get allocation report."""
        report = {
            "portfolio_value_usd": self.portfolio_value_usd,
            "total_rebalances": len(self.rebalance_history),
            "allocations": {}
        }
        
        for symbol, alloc in self.allocations.items():
            report["allocations"][symbol] = {
                "target_pct": alloc.target_pct * 100,
                "current_pct": alloc.current_pct * 100,
                "drift_pct": alloc.drift_pct * 100,
                "needs_rebalance": abs(alloc.drift_pct * 100) > self.config.drift_threshold_pct
            }
        
        return report


class SmartRebalancer:
    """
    Smart Rebalancer
    ================
    Combines multiple rebalancing strategies.
    """
    
    def __init__(self):
        self.threshold_rebalancer = PortfolioRebalancer(RebalanceConfig(
            method=RebalanceMethod.THRESHOLD,
            drift_threshold_pct=5.0
        ))
        self.risk_parity_rebalancer = PortfolioRebalancer(RebalanceConfig(
            method=RebalanceMethod.RISK_PARITY
        ))
        self.momentum_rebalancer = PortfolioRebalancer(RebalanceConfig(
            method=RebalanceMethod.MOMENTUM,
            momentum_window_days=30
        ))
        
        self.active_method = "threshold"
    
    def select_method(self, market_conditions: Dict[str, Any]) -> str:
        """Select best rebalancing method."""
        volatility = market_conditions.get("volatility", 0.02)
        trend_strength = market_conditions.get("trend_strength", 0)
        
        if volatility > 0.04:
            # High volatility - use risk parity
            return "risk_parity"
        elif abs(trend_strength) > 0.5:
            # Strong trend - use momentum
            return "momentum"
        else:
            # Normal conditions - threshold
            return "threshold"
    
    async def smart_rebalance(
        self,
        holdings: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> RebalanceResult:
        """Execute smart rebalancing."""
        if market_conditions is None:
            market_conditions = {"volatility": 0.02, "trend_strength": 0}
        
        # Select method
        method = self.select_method(market_conditions)
        self.active_method = method
        
        # Get appropriate rebalancer
        if method == "risk_parity":
            rebalancer = self.risk_parity_rebalancer
        elif method == "momentum":
            rebalancer = self.momentum_rebalancer
        else:
            rebalancer = self.threshold_rebalancer
        
        # Update and execute
        rebalancer.update_prices(prices)
        rebalancer.update_portfolio(holdings, portfolio_value)
        
        result = rebalancer.execute_rebalance()
        result.method_used = f"SMART({method})"
        
        return result


# Export
__all__ = [
    "RebalanceMethod",
    "AllocationMethod",
    "AssetAllocation",
    "RebalanceConfig",
    "RebalanceTrade",
    "RebalanceResult",
    "PortfolioRebalancer",
    "SmartRebalancer"
]
