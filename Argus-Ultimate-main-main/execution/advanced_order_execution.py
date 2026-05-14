"""
Advanced Order Execution Optimizer
===================================
Minimizes market impact and slippage through:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- Iceberg orders
- Adaptive execution
- Smart routing
- Anti-gaming logic

Based on institutional execution algorithms.
"""

import asyncio
import logging
import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"
    ADAPTIVE = "adaptive"


class ExecutionStrategy(Enum):
    """Execution strategies."""
    AGGRESSIVE = "aggressive"  # Execute quickly, higher cost
    BALANCED = "balanced"  # Balance speed and cost
    PASSIVE = "passive"  # Minimize cost, slower
    ADAPTIVE = "adaptive"  # Adjust based on conditions


class TimeInForce(Enum):
    """Time in force options."""
    GTC = "gtc"  # Good till cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    DAY = "day"  # Day order


@dataclass
class OrderParams:
    """Order parameters."""
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    strategy: ExecutionStrategy = ExecutionStrategy.BALANCED
    max_slippage_pct: float = 0.5
    deadline_seconds: float = 60
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderSlice:
    """A slice of a larger order."""
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: float = field(default_factory=time.time)
    filled: bool = False
    fill_price: Optional[float] = None
    fill_quantity: float = 0.0
    fee: float = 0.0


@dataclass
class ExecutionResult:
    """Result of order execution."""
    symbol: str
    side: str
    requested_quantity: float
    filled_quantity: float
    avg_fill_price: float
    total_fees: float
    slippage_pct: float
    execution_time_ms: float
    slices: List[OrderSlice]
    strategy_used: str
    success: bool = True
    error: Optional[str] = None


class TWAPExecutor:
    """
    TWAP (Time-Weighted Average Price) Executor
    ============================================
    Slices large orders over time to minimize market impact.
    """
    
    def __init__(self, num_slices: int = 10, slice_interval_ms: float = 1000):
        self.num_slices = num_slices
        self.slice_interval_ms = slice_interval_ms
    
    def calculate_slices(self, params: OrderParams) -> List[Dict[str, Any]]:
        """Calculate order slices."""
        slice_qty = params.quantity / self.num_slices
        
        # Add randomness to avoid detection
        slices = []
        for i in range(self.num_slices):
            # Vary quantity by ±10%
            qty = slice_qty * np.random.uniform(0.9, 1.1)
            
            # Vary timing by ±20%
            delay = self.slice_interval_ms * i * np.random.uniform(0.8, 1.2)
            
            slices.append({
                "slice_num": i + 1,
                "quantity": qty,
                "delay_ms": delay,
                "order_type": OrderType.LIMIT if params.price else OrderType.MARKET
            })
        
        return slices
    
    async def execute(self, params: OrderParams) -> ExecutionResult:
        """Execute TWAP strategy."""
        logger.info(f"Executing TWAP: {params.symbol} {params.side} {params.quantity}")
        
        slices = self.calculate_slices(params)
        executed_slices = []
        
        total_filled = 0
        total_cost = 0
        total_fees = 0
        
        for slice_info in slices:
            # Wait for slice interval
            await asyncio.sleep(slice_info["delay_ms"] / 1000)
            
            # Simulate execution
            fill_price = params.price or 50000  # Simulated
            fill_price *= np.random.uniform(0.999, 1.001)  # Small variation
            
            slice_order = OrderSlice(
                order_id=f"twap_{int(time.time())}_{slice_info['slice_num']}",
                symbol=params.symbol,
                side=params.side,
                quantity=slice_info["quantity"],
                price=fill_price,
                filled=True,
                fill_price=fill_price,
                fill_quantity=slice_info["quantity"],
                fee=fill_price * slice_info["quantity"] * 0.001
            )
            
            executed_slices.append(slice_order)
            total_filled += slice_info["quantity"]
            total_cost += fill_price * slice_info["quantity"]
            total_fees += slice_order.fee
        
        avg_price = total_cost / total_filled if total_filled > 0 else 0
        
        return ExecutionResult(
            symbol=params.symbol,
            side=params.side,
            requested_quantity=params.quantity,
            filled_quantity=total_filled,
            avg_fill_price=avg_price,
            total_fees=total_fees,
            slippage_pct=self._calculate_slippage(params, avg_price),
            execution_time_ms=len(slices) * self.slice_interval_ms,
            slices=executed_slices,
            strategy_used="TWAP"
        )
    
    def _calculate_slippage(self, params: OrderParams, avg_price: float) -> float:
        """Calculate slippage from expected price."""
        if params.price:
            return abs(avg_price - params.price) / params.price * 100
        return 0


class VWAPExecutor:
    """
    VWAP (Volume-Weighted Average Price) Executor
    ==============================================
    Executes based on historical volume profile.
    """
    
    def __init__(self):
        # Typical crypto volume profile (hourly weights)
        self.volume_profile = [
            0.03, 0.025, 0.02, 0.02, 0.02, 0.025,  # 0-5 UTC (low)
            0.03, 0.04, 0.05, 0.06, 0.07, 0.08,    # 6-11 UTC (rising)
            0.09, 0.08, 0.07, 0.06, 0.06, 0.05,    # 12-17 UTC (US hours)
            0.05, 0.04, 0.04, 0.04, 0.035, 0.03     # 18-23 UTC (evening)
        ]
    
    def get_current_weight(self) -> float:
        """Get current hour's volume weight."""
        hour = int(time.time() / 3600) % 24
        return self.volume_profile[hour]
    
    def calculate_slices(self, params: OrderParams, duration_hours: float = 4) -> List[Dict[str, Any]]:
        """Calculate VWAP slices based on volume profile."""
        slices = []
        remaining = params.quantity
        
        for hour_offset in range(int(duration_hours)):
            hour = (int(time.time() / 3600) + hour_offset) % 24
            weight = self.volume_profile[hour]
            
            # Allocate quantity based on volume weight
            qty = params.quantity * weight
            qty = min(qty, remaining)
            remaining -= qty
            
            if qty > 0:
                slices.append({
                    "hour": hour,
                    "quantity": qty,
                    "weight": weight
                })
        
        return slices
    
    async def execute(self, params: OrderParams) -> ExecutionResult:
        """Execute VWAP strategy."""
        logger.info(f"Executing VWAP: {params.symbol} {params.side} {params.quantity}")
        
        slices = self.calculate_slices(params)
        executed_slices = []
        
        total_filled = 0
        total_cost = 0
        total_fees = 0
        
        for i, slice_info in enumerate(slices):
            # Simulate waiting for the right hour
            await asyncio.sleep(0.1)  # Compressed for simulation
            
            fill_price = 50000 * np.random.uniform(0.999, 1.001)
            
            slice_order = OrderSlice(
                order_id=f"vwap_{int(time.time())}_{i}",
                symbol=params.symbol,
                side=params.side,
                quantity=slice_info["quantity"],
                price=fill_price,
                filled=True,
                fill_price=fill_price,
                fill_quantity=slice_info["quantity"],
                fee=fill_price * slice_info["quantity"] * 0.001
            )
            
            executed_slices.append(slice_order)
            total_filled += slice_info["quantity"]
            total_cost += fill_price * slice_info["quantity"]
            total_fees += slice_order.fee
        
        avg_price = total_cost / total_filled if total_filled > 0 else 0
        
        return ExecutionResult(
            symbol=params.symbol,
            side=params.side,
            requested_quantity=params.quantity,
            filled_quantity=total_filled,
            avg_fill_price=avg_price,
            total_fees=total_fees,
            slippage_pct=0,
            execution_time_ms=len(slices) * 3600000,  # Hours in ms
            slices=executed_slices,
            strategy_used="VWAP"
        )


class IcebergExecutor:
    """
    Iceberg Order Executor
    ======================
    Shows only a small portion of the order at a time.
    """
    
    def __init__(self, visible_ratio: float = 0.1):
        self.visible_ratio = visible_ratio
    
    def calculate_slices(self, params: OrderParams) -> List[Dict[str, Any]]:
        """Calculate iceberg slices."""
        visible_qty = params.quantity * self.visible_ratio
        hidden_qty = params.quantity - visible_qty
        
        slices = []
        remaining = params.quantity
        
        while remaining > 0:
            slice_qty = min(visible_qty, remaining)
            slices.append({
                "visible": slice_qty,
                "hidden": remaining - slice_qty
            })
            remaining -= slice_qty
        
        return slices
    
    async def execute(self, params: OrderParams) -> ExecutionResult:
        """Execute iceberg order."""
        logger.info(f"Executing Iceberg: {params.symbol} {params.side} {params.quantity}")
        
        slices = self.calculate_slices(params)
        executed_slices = []
        
        total_filled = 0
        total_cost = 0
        total_fees = 0
        
        for i, slice_info in enumerate(slices):
            await asyncio.sleep(0.05)  # Simulate time between refills
            
            fill_price = params.price or 50000
            fill_price *= np.random.uniform(0.9995, 1.0005)
            
            slice_order = OrderSlice(
                order_id=f"iceberg_{int(time.time())}_{i}",
                symbol=params.symbol,
                side=params.side,
                quantity=slice_info["visible"],
                price=fill_price,
                filled=True,
                fill_price=fill_price,
                fill_quantity=slice_info["visible"],
                fee=fill_price * slice_info["visible"] * 0.001
            )
            
            executed_slices.append(slice_order)
            total_filled += slice_info["visible"]
            total_cost += fill_price * slice_info["visible"]
            total_fees += slice_order.fee
        
        avg_price = total_cost / total_filled if total_filled > 0 else 0
        
        return ExecutionResult(
            symbol=params.symbol,
            side=params.side,
            requested_quantity=params.quantity,
            filled_quantity=total_filled,
            avg_fill_price=avg_price,
            total_fees=total_fees,
            slippage_pct=0,
            execution_time_ms=len(slices) * 50,
            slices=executed_slices,
            strategy_used="ICEBERG"
        )


class AdaptiveExecutor:
    """
    Adaptive Execution
    ==================
    Adjusts execution based on real-time market conditions.
    """
    
    def __init__(self):
        self.market_state = "normal"
        self.volatility_threshold = 0.02
        
    def analyze_market(self, params: OrderParams) -> Dict[str, Any]:
        """Analyze current market conditions."""
        # In production: analyze order book, volatility, volume
        return {
            "volatility": np.random.uniform(0.01, 0.03),
            "volume": np.random.uniform(1000000, 10000000),
            "spread": np.random.uniform(0.0001, 0.001),
            "order_book_imbalance": np.random.uniform(-0.5, 0.5)
        }
    
    def select_strategy(self, params: OrderParams, market: Dict[str, Any]) -> str:
        """Select best execution strategy based on conditions."""
        volatility = market["volatility"]
        spread = market["spread"]
        
        if volatility > self.volatility_threshold:
            # High volatility - use TWAP to spread risk
            return "TWAP"
        elif spread > 0.0005:
            # Wide spread - use limit orders
            return "ICEBERG"
        elif params.quantity > 10000:
            # Large order - use VWAP
            return "VWAP"
        else:
            # Normal conditions - balanced approach
            return "BALANCED"
    
    async def execute(self, params: OrderParams) -> ExecutionResult:
        """Execute with adaptive strategy."""
        logger.info(f"Executing Adaptive: {params.symbol} {params.side} {params.quantity}")
        
        # Analyze market
        market = self.analyze_market(params)
        
        # Select strategy
        strategy = self.select_strategy(params, market)
        logger.info(f"Selected strategy: {strategy} (vol={market['volatility']:.3f})")
        
        # Execute with selected strategy
        if strategy == "TWAP":
            executor = TWAPExecutor(num_slices=5)
        elif strategy == "VWAP":
            executor = VWAPExecutor()
        elif strategy == "ICEBERG":
            executor = IcebergExecutor()
        else:
            # Balanced - use TWAP with fewer slices
            executor = TWAPExecutor(num_slices=3)
        
        result = await executor.execute(params)
        result.strategy_used = f"ADAPTIVE({strategy})"
        
        return result


class SmartOrderRouter:
    """
    Smart Order Router
    ==================
    Routes orders to best exchange for execution.
    """
    
    def __init__(self):
        self.exchange_scores: Dict[str, Dict[str, float]] = {}
        
    def update_exchange_score(
        self,
        exchange: str,
        fill_rate: float,
        avg_slippage: float,
        latency_ms: float
    ) -> None:
        """Update exchange execution score."""
        # Higher fill rate = better, lower slippage = better, lower latency = better
        score = fill_rate * 0.4 + (1 - avg_slippage) * 0.4 + (1 - min(latency_ms / 100, 1)) * 0.2
        
        self.exchange_scores[exchange] = {
            "score": score,
            "fill_rate": fill_rate,
            "slippage": avg_slippage,
            "latency": latency_ms
        }
    
    def get_best_exchange(self, symbol: str) -> str:
        """Get best exchange for execution."""
        if not self.exchange_scores:
            return "binance"  # Default
        
        # Sort by score
        sorted_exchanges = sorted(
            self.exchange_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        return sorted_exchanges[0][0]
    
    async def route_order(self, params: OrderParams) -> ExecutionResult:
        """Route order to best exchange."""
        exchange = self.get_best_exchange(params.symbol)
        logger.info(f"Routing order to {exchange}")
        
        # Execute on selected exchange
        executor = AdaptiveExecutor()
        result = await executor.execute(params)
        result.strategy_used = f"ROUTED({exchange})"
        
        return result


class ExecutionOptimizer:
    """
    Execution Optimizer
    ===================
    Main entry point for optimized execution.
    """
    
    def __init__(self):
        self.twap = TWAPExecutor()
        self.vwap = VWAPExecutor()
        self.iceberg = IcebergExecutor()
        self.adaptive = AdaptiveExecutor()
        self.router = SmartOrderRouter()
        
        self.execution_history: List[ExecutionResult] = []
    
    async def execute(self, params: OrderParams) -> ExecutionResult:
        """Execute order with optimal strategy."""
        
        # Select executor based on order type and strategy
        if params.order_type == OrderType.TWAP:
            executor = self.twap
        elif params.order_type == OrderType.VWAP:
            executor = self.vwap
        elif params.order_type == OrderType.ICEBERG:
            executor = self.iceberg
        elif params.strategy == ExecutionStrategy.ADAPTIVE:
            executor = self.adaptive
        else:
            # Default to adaptive
            executor = self.adaptive
        
        # Execute
        result = await executor.execute(params)
        
        # Store in history
        self.execution_history.append(result)
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.execution_history:
            return {"total_orders": 0}
        
        total_orders = len(self.execution_history)
        successful = sum(1 for r in self.execution_history if r.success)
        avg_slippage = np.mean([r.slippage_pct for r in self.execution_history])
        avg_time = np.mean([r.execution_time_ms for r in self.execution_history])
        total_fees = sum(r.total_fees for r in self.execution_history)
        
        return {
            "total_orders": total_orders,
            "success_rate": successful / total_orders * 100,
            "avg_slippage_pct": avg_slippage,
            "avg_execution_time_ms": avg_time,
            "total_fees": total_fees,
            "strategies_used": list(set(r.strategy_used for r in self.execution_history))
        }


# Export
__all__ = [
    "OrderType",
    "ExecutionStrategy",
    "TimeInForce",
    "OrderParams",
    "OrderSlice",
    "ExecutionResult",
    "TWAPExecutor",
    "VWAPExecutor",
    "IcebergExecutor",
    "AdaptiveExecutor",
    "SmartOrderRouter",
    "ExecutionOptimizer"
]
