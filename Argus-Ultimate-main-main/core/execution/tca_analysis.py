#!/usr/bin/env python3
"""
Transaction Cost Analysis System

Analyzes execution quality by comparing actual execution
against various benchmarks (VWAP, TWAP, arrival price)
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import statistics
import time


@dataclass
class ExecutionResult:
    """Result of an order execution"""

    order_id: str
    symbol: str
    side: str
    quantity: float
    executed_quantity: float
    avg_price: float
    slippage: float
    commissions: float
    execution_time: float
    fill_prices: List[float]
    fill_sizes: List[float]
    timestamp: str

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "executed_quantity": self.executed_quantity,
            "avg_price": self.avg_price,
            "slippage": self.slippage,
            "commissions": self.commissions,
            "execution_time": self.execution_time,
            "fill_prices": self.fill_prices,
            "fill_sizes": self.fill_sizes,
            "timestamp": self.timestamp,
        }


@dataclass
class TCAMetrics:
    """TCA metrics for execution analysis"""

    order_id: str
    symbol: str
    side: str
    quantity: float

    arrival_price: float  # Price when order was received
    vwap: float  # Volume-weighted average price during execution
    twap: float  # Time-weighted average price during execution

    actual_avg_price: float  # Actual execution price
    slippage: float  # Price impact (actual - arrival)
    market_impact: float  # Price movement during execution
    timing_risk: float  # Cost of not executing immediately

    implementation_shortfall: float  # Cost vs. ideal execution
    vwap_performance: float  # Performance vs. VWAP
    twap_performance: float  # Performance vs. TWAP

    execution_duration: float  # Time to complete (seconds)
    timestamp: str

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "arrival_price": self.arrival_price,
            "vwap": self.vwap,
            "twap": self.twap,
            "actual_avg_price": self.actual_avg_price,
            "slippage": self.slippage,
            "market_impact": self.market_impact,
            "timing_risk": self.timing_risk,
            "implementation_shortfall": self.implementation_shortfall,
            "vwap_performance": self.vwap_performance,
            "twap_performance": self.twap_performance,
            "execution_duration": self.execution_duration,
            "timestamp": self.timestamp,
        }


class TCAAnalysis:
    """
    Transaction Cost Analysis System

    Analyzes execution quality by comparing actual execution
    against various benchmarks (VWAP, TWAP, arrival price)
    """

    def __init__(self):
        self.execution_history: List[ExecutionResult] = []
        self.tca_metrics: List[TCAMetrics] = []

    def analyze_execution(self, execution: ExecutionResult, market_data: Dict[str, Any]) -> TCAMetrics:
        """
        Analyze execution quality against benchmarks

        Args:
            execution: Execution result to analyze
            market_data: Market data during execution period

        Returns:
            TCA metrics for the execution
        """
        arrival_price = market_data.get("arrival_price", execution.avg_price)
        vwap = self._calculate_vwap(market_data)
        twap = self._calculate_twap(market_data)

        slippage = execution.avg_price - arrival_price
        market_impact = self._calculate_market_impact(market_data)

        # Implementation shortfall (simplified)
        ideal_price = arrival_price
        implementation_shortfall = execution.avg_price - ideal_price

        # Performance vs. benchmarks
        vwap_performance = execution.avg_price - vwap
        twap_performance = execution.avg_price - twap

        # Timing risk (simplified)
        timing_risk = abs(vwap - arrival_price)

        metrics = TCAMetrics(
            order_id=execution.order_id,
            symbol=execution.symbol,
            side=execution.side,
            quantity=execution.quantity,
            arrival_price=arrival_price,
            vwap=vwap,
            twap=twap,
            actual_avg_price=execution.avg_price,
            slippage=slippage,
            market_impact=market_impact,
            timing_risk=timing_risk,
            implementation_shortfall=implementation_shortfall,
            vwap_performance=vwap_performance,
            twap_performance=twap_performance,
            execution_duration=execution.execution_time,
            timestamp=execution.timestamp,
        )

        self.tca_metrics.append(metrics)
        return metrics

    def _calculate_vwap(self, market_data: Dict[str, Any]) -> float:
        """Calculate VWAP from market data"""
        prices = market_data.get("prices", [])
        volumes = market_data.get("volumes", [])

        if not prices or not volumes:
            return market_data.get("mid_price", 0)

        price_volume_sum = sum(p * v for p, v in zip(prices, volumes))
        volume_sum = sum(volumes)

        return price_volume_sum / volume_sum if volume_sum > 0 else prices[-1]

    def _calculate_twap(self, market_data: Dict[str, Any]) -> float:
        """Calculate TWAP from market data"""
        prices = market_data.get("prices", [])

        if not prices:
            return market_data.get("mid_price", 0)

        return statistics.mean(prices)

    def _calculate_market_impact(self, market_data: Dict[str, Any]) -> float:
        """Calculate market impact (simplified)"""
        prices = market_data.get("prices", [])

        if len(prices) < 2:
            return 0

        return abs(prices[-1] - prices[0]) / prices[0]

    def get_summary_report(self) -> Dict[str, Any]:
        """Generate TCA summary report"""
        if not self.tca_metrics:
            return {"error": "No TCA metrics available"}

        total_slippage = sum(m.slippage for m in self.tca_metrics)
        avg_slippage = total_slippage / len(self.tca_metrics)

        total_market_impact = sum(m.market_impact for m in self.tca_metrics)
        avg_market_impact = total_market_impact / len(self.tca_metrics)

        total_implementation_shortfall = sum(m.implementation_shortfall for m in self.tca_metrics)
        avg_implementation_shortfall = total_implementation_shortfall / len(self.tca_metrics)

        return {
            "total_executions": len(self.tca_metrics),
            "average_slippage": avg_slippage,
            "average_market_impact": avg_market_impact,
            "average_implementation_shortfall": avg_implementation_shortfall,
            "total_trading_cost": total_slippage + total_market_impact,
        }
