"""
Enhanced Fee Optimizer v2.0
=============================
Advanced fee optimization with batch trading and exchange routing.

Extends the existing FeeOptimizer with:
- Multi-exchange fee comparison
- Batch order optimization
- Fee-aware order routing
- Savings tracking and reporting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BatchOrder:
    """Order for batching."""
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    urgency: float
    order_type: str  # "limit" or "market"
    timestamp: datetime
    fee_estimate: float


@dataclass
class ExchangeQuote:
    """Fee quote from an exchange."""
    exchange: str
    maker_fee_bps: float
    taker_fee_bps: float
    spread_bps: float
    liquidity_score: float  # 0-1
    latency_ms: float
    volume_24h: float


class MultiExchangeRouter:
    """
    Routes orders to optimal exchange for lowest fees.
    """
    
    def __init__(self) -> None:
        """Initialize multi-exchange router."""
        self._exchanges: Dict[str, ExchangeQuote] = {}
        self._order_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def register_exchange(self, quote: ExchangeQuote) -> None:
        """Register exchange with current fee quote."""
        self._exchanges[quote.exchange] = quote
    
    def get_best_exchange(
        self,
        order_type: str,
        urgency: float = 0.5
    ) -> Optional[ExchangeQuote]:
        """
        Get best exchange for order.
        
        Args:
            order_type: "limit" or "market"
            urgency: Order urgency (0-1)
            
        Returns:
            Best exchange quote
        """
        if not self._exchanges:
            return None
        
        best = None
        best_score = float('inf')
        
        for quote in self._exchanges.values():
            # Calculate effective fee
            if order_type == "limit":
                fee = quote.maker_fee_bps
            else:
                fee = quote.taker_fee_bps
            
            # Adjust for spread (wider spread = worse execution)
            effective_cost = fee + quote.spread_bps * 0.5
            
            # Adjust for latency (more important for urgent orders)
            latency_penalty = quote.latency_ms * urgency * 0.1
            
            # Adjust for liquidity (lower liquidity = worse)
            liquidity_penalty = (1 - quote.liquidity_score) * 5
            
            total_score = effective_cost + latency_penalty + liquidity_penalty
            
            if total_score < best_score:
                best_score = total_score
                best = quote
        
        return best
    
    def compare_exchanges(self, order_type: str = "limit") -> List[Dict[str, Any]]:
        """Compare all exchanges by fees."""
        comparisons = []
        
        for exchange, quote in self._exchanges.items():
            fee = quote.maker_fee_bps if order_type == "limit" else quote.taker_fee_bps
            
            comparisons.append({
                "exchange": exchange,
                "fee_bps": fee,
                "spread_bps": quote.spread_bps,
                "liquidity_score": quote.liquidity_score,
                "latency_ms": quote.latency_ms,
                "effective_cost": fee + quote.spread_bps * 0.5
            })
        
        # Sort by effective cost
        comparisons.sort(key=lambda x: x["effective_cost"])
        
        return comparisons


class BatchOrderOptimizer:
    """
    Batches orders to reduce total fees.
    """
    
    def __init__(
        self,
        batch_window_seconds: int = 60,
        min_batch_size: int = 2,
        max_batch_size: int = 20,
        same_symbol_only: bool = True
    ) -> None:
        """
        Initialize batch order optimizer.
        
        Args:
            batch_window_seconds: Max time to wait for batch
            min_batch_size: Minimum orders to form a batch
            max_batch_size: Maximum orders per batch
            same_symbol_only: Only batch same symbols together
        """
        self.batch_window = batch_window_seconds
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.same_symbol_only = same_symbol_only
        
        self._pending: Dict[str, List[BatchOrder]] = {}  # By symbol
        self._total_batched: int = 0
        self._total_saved: float = 0.0
    
    def add_order(self, order: BatchOrder) -> Optional[List[BatchOrder]]:
        """
        Add order to batch.
        
        Returns batch if ready to execute.
        """
        symbol = order.symbol if self.same_symbol_only else "_all"
        
        if symbol not in self._pending:
            self._pending[symbol] = []
        
        self._pending[symbol].append(order)
        
        # Check if batch is ready
        if len(self._pending[symbol]) >= self.max_batch_size:
            return self._flush_batch(symbol)
        
        return None
    
    def check_and_flush(self) -> List[List[BatchOrder]]:
        """Check all batches and flush ready ones."""
        batches = []
        
        for symbol in list(self._pending.keys()):
            orders = self._pending[symbol]
            
            if not orders:
                continue
            
            # Check time window
            oldest = orders[0].timestamp
            elapsed = (datetime.now() - oldest).total_seconds()
            
            if elapsed >= self.batch_window and len(orders) >= self.min_batch_size:
                batch = self._flush_batch(symbol)
                if batch:
                    batches.append(batch)
        
        return batches
    
    def _flush_batch(self, symbol: str) -> Optional[List[BatchOrder]]:
        """Flush batch for symbol."""
        if symbol not in self._pending:
            return None
        
        orders = self._pending[symbol]
        if not orders:
            return None
        
        # Calculate savings
        individual_fees = sum(o.fee_estimate for o in orders)
        # Batched fee (single order fee for combined quantity)
        total_quantity = sum(o.quantity for o in orders)
        avg_price = np.mean([o.price for o in orders])
        batch_fee = total_quantity * avg_price * 0.0002  # Simplified
        
        savings = individual_fees - batch_fee
        self._total_saved += savings
        self._total_batched += len(orders)
        
        # Clear batch
        self._pending[symbol] = []
        
        logger.info(
            "Batch executed: %s - %d orders, saved $%.2f",
            symbol, len(orders), savings
        )
        
        return orders
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch optimizer statistics."""
        pending_count = sum(len(orders) for orders in self._pending.values())
        
        return {
            "total_batched": self._total_batched,
            "total_saved": self._total_saved,
            "pending_orders": pending_count,
            "pending_symbols": len([s for s, o in self._pending.items() if o])
        }


class FeeSavingsTracker:
    """
    Tracks fee savings from optimization.
    """
    
    def __init__(self) -> None:
        """Initialize savings tracker."""
        self._savings_history: List[Dict[str, Any]] = []
        self._total_saved: float = 0.0
        self._total_fees_paid: float = 0.0
        self._total_volume: float = 0.0
    
    def record_trade(
        self,
        symbol: str,
        volume: float,
        fee_paid: float,
        fee_without_optimization: float
    ) -> None:
        """Record a trade and savings."""
        savings = fee_without_optimization - fee_paid
        
        self._savings_history.append({
            "symbol": symbol,
            "volume": volume,
            "fee_paid": fee_paid,
            "fee_without_opt": fee_without_optimization,
            "savings": savings,
            "timestamp": datetime.now()
        })
        
        self._total_saved += savings
        self._total_fees_paid += fee_paid
        self._total_volume += volume
    
    def get_savings_report(self, days: int = 30) -> Dict[str, Any]:
        """Get savings report."""
        cutoff = datetime.now() - timedelta(days=days)
        recent = [
            s for s in self._savings_history
            if s["timestamp"] > cutoff
        ]
        
        recent_savings = sum(s["savings"] for s in recent)
        recent_fees = sum(s["fee_paid"] for s in recent)
        
        return {
            "period_days": days,
            "total_saved": self._total_saved,
            "recent_saved": recent_savings,
            "recent_fees": recent_fees,
            "savings_rate": recent_savings / (recent_savings + recent_fees) * 100
            if (recent_savings + recent_fees) > 0 else 0,
            "n_trades_optimized": len(recent),
            "annualized_savings": recent_savings * (365 / max(1, days))
        }


class EnhancedFeeOptimizer:
    """
    Enhanced fee optimizer with multi-exchange routing and batching.
    """
    
    def __init__(
        self,
        enable_batching: bool = True,
        enable_exchange_routing: bool = True,
        target_fee_bps: float = 3.0
    ) -> None:
        """
        Initialize enhanced fee optimizer.
        
        Args:
            enable_batching: Enable order batching
            enable_exchange_routing: Enable multi-exchange routing
            target_fee_bps: Target average fee in basis points
        """
        self.enable_batching = enable_batching
        self.enable_exchange_routing = enable_exchange_routing
        self.target_fee_bps = target_fee_bps
        
        self.exchange_router = MultiExchangeRouter()
        self.batch_optimizer = BatchOrderOptimizer()
        self.savings_tracker = FeeSavingsTracker()
        
        self._order_counter: int = 0
    
    def create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        urgency: float = 0.5
    ) -> Dict[str, Any]:
        """
        Create optimized order.
        
        Returns order with optimal exchange and type.
        """
        self._order_counter += 1
        order_value = quantity * price
        
        # Determine order type
        if urgency > 0.8:
            order_type = "market"
        else:
            order_type = "limit"
        
        # Get best exchange
        best_exchange = None
        if self.enable_exchange_routing:
            best_exchange = self.exchange_router.get_best_exchange(
                order_type=order_type,
                urgency=urgency
            )
        
        # Estimate fee
        if best_exchange:
            fee_bps = best_exchange.maker_fee_bps if order_type == "limit" else best_exchange.taker_fee_bps
            exchange = best_exchange.exchange
        else:
            fee_bps = 5.0 if order_type == "limit" else 10.0
            exchange = "default"
        
        fee_amount = order_value * fee_bps / 10000
        
        # Create batch order
        batch_order = BatchOrder(
            order_id=f"order_{self._order_counter}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            urgency=urgency,
            order_type=order_type,
            timestamp=datetime.now(),
            fee_estimate=fee_amount
        )
        
        # Try to batch
        batch = None
        if self.enable_batching and urgency < 0.7:
            batch = self.batch_optimizer.add_order(batch_order)
        
        return {
            "order_id": batch_order.order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "order_type": order_type,
            "exchange": exchange,
            "fee_bps": fee_bps,
            "fee_amount": fee_amount,
            "in_batch": batch is not None,
            "batch_ready": batch is not None
        }
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        batch_stats = self.batch_optimizer.get_stats()
        savings_report = self.savings_tracker.get_savings_report()
        
        return {
            "batching": batch_stats,
            "savings": savings_report,
            "exchange_routing": self.enable_exchange_routing,
            "target_fee_bps": self.target_fee_bps,
            "total_orders_processed": self._order_counter
        }
    
    def calculate_optimal_strategy(
        self,
        monthly_volume: float,
        trades_per_month: int,
        avg_trade_size: float
    ) -> Dict[str, Any]:
        """
        Calculate optimal fee strategy.
        
        Returns recommendations for minimizing fees.
        """
        # Current cost estimate
        current_fee_bps = 10.0  # Assume taker fee
        current_monthly_cost = monthly_volume * current_fee_bps / 10000
        
        # Optimized cost (limit orders + batching)
        optimized_fee_bps = 3.0  # Maker fee with batching
        optimized_monthly_cost = monthly_volume * optimized_fee_bps / 10000
        
        savings = current_monthly_cost - optimized_monthly_cost
        
        return {
            "monthly_volume": monthly_volume,
            "current_monthly_cost": current_monthly_cost,
            "optimized_monthly_cost": optimized_monthly_cost,
            "monthly_savings": savings,
            "annual_savings": savings * 12,
            "recommendations": [
                f"Use limit orders for {100 - int(optimized_fee_bps/current_fee_bps*100)}% fee reduction",
                f"Batch orders to reduce fee events by ~{(1 - optimized_fee_bps/current_fee_bps) * 100:.0f}%",
                "Route to lowest-fee exchange for each order type",
                "Increase 30-day volume to reach higher fee tiers"
            ]
        }
