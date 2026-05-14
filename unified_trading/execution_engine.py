"""
Execution Engine Module
=======================

Handles trade execution, venue routing, and fill processing.
Refactored from unified_trading_system.py.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal
from enum import Enum

from unified_trading.order_management import Order, OrderStatus
from core.exception_manager import (
    ExecutionError,
    VenueUnavailableError,
    handle_errors,
    retry_on_error
)

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Execution status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass
class Fill:
    """Represents an order fill."""
    order_id: str
    symbol: str
    side: str
    filled_qty: Decimal
    price: Decimal
    venue: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    fees: Decimal = field(default_factory=lambda: Decimal("0"))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of order execution."""
    success: bool
    order_id: str
    status: ExecutionStatus
    filled_qty: Decimal
    avg_price: Optional[Decimal] = None
    remaining_qty: Decimal = field(default_factory=lambda: Decimal("0"))
    venue: Optional[str] = None
    fills: List[Fill] = field(default_factory=list)
    error: Optional[str] = None
    latency_ms: float = 0.0
    fees: Decimal = field(default_factory=lambda: Decimal("0"))
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VenueConfig:
    """Configuration for trading venue."""
    name: str
    api_endpoint: str
    fee_maker: Decimal
    fee_taker: Decimal
    latency_ms: float
    supports_limit: bool = True
    supports_market: bool = True
    supports_stop: bool = False
    max_order_size: Decimal = field(default_factory=lambda: Decimal("1000000"))
    is_active: bool = True


class VenueAdapter:
    """Adapter for trading venue."""
    
    def __init__(self, config: VenueConfig):
        self.config = config
        self._client = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to venue."""
        try:
            # Simulate connection
            self._connected = True
            logger.info(f"Connected to {self.config.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.config.name}: {e}")
            raise VenueUnavailableError(
                f"Connection failed: {e}",
                venue=self.config.name
            ) from e
    
    async def disconnect(self):
        """Disconnect from venue."""
        self._connected = False
        logger.info(f"Disconnected from {self.config.name}")
    
    @retry_on_error(max_retries=3, exceptions=(ExecutionError,), delay=0.5)
    async def submit_order(self, order: Order) -> ExecutionResult:
        """Submit order to venue."""
        if not self._connected:
            raise VenueUnavailableError("Not connected", venue=self.config.name)
        
        start_time = datetime.utcnow()
        
        try:
            # Simulate order submission
            await asyncio.sleep(0.01)  # Simulate network latency
            
            # Simulate fill (for demo purposes)
            fill_qty = order.quantity
            fill_price = order.price or Decimal("50000")  # Default price for demo
            
            fill = Fill(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side.value,
                filled_qty=fill_qty,
                price=fill_price,
                venue=self.config.name,
                fees=self._calculate_fees(fill_qty, fill_price)
            )
            
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return ExecutionResult(
                success=True,
                order_id=order.id,
                status=ExecutionStatus.FILLED,
                filled_qty=fill_qty,
                avg_price=fill_price,
                remaining_qty=Decimal("0"),
                venue=self.config.name,
                fills=[fill],
                latency_ms=latency,
                fees=fill.fees
            )
            
        except Exception as e:
            logger.error(f"Order submission failed on {self.config.name}: {e}")
            raise ExecutionError(
                f"Execution failed: {e}",
                venue=self.config.name,
                retry_possible=True
            ) from e
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on venue."""
        try:
            await asyncio.sleep(0.01)  # Simulate network latency
            logger.info(f"Order {order_id} cancelled on {self.config.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[ExecutionResult]:
        """Get order status from venue."""
        try:
            # Simulate API call
            await asyncio.sleep(0.005)
            return None  # Would return actual status
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return None
    
    def _calculate_fees(self, qty: Decimal, price: Decimal) -> Decimal:
        """Calculate trading fees."""
        notional = qty * price
        # Assume taker fee for simplicity
        return notional * self.config.fee_taker


class ExecutionEngine:
    """
    Main execution engine for order routing and execution.
    """
    
    def __init__(self):
        self._venues: Dict[str, VenueAdapter] = {}
        self._execution_history: List[ExecutionResult] = []
        self._fill_history: List[Fill] = []
        self._lock = asyncio.Lock()
        
        # Default venues
        self._default_venues = [
            VenueConfig(
                name="binance",
                api_endpoint="https://api.binance.com",
                fee_maker=Decimal("0.001"),
                fee_taker=Decimal("0.001"),
                latency_ms=50.0
            ),
            VenueConfig(
                name="kraken",
                api_endpoint="https://api.kraken.com",
                fee_maker=Decimal("0.0016"),
                fee_taker=Decimal("0.0026"),
                latency_ms=80.0
            ),
            VenueConfig(
                name="coinbase",
                api_endpoint="https://api.coinbase.com",
                fee_maker=Decimal("0.004"),
                fee_taker=Decimal("0.006"),
                latency_ms=100.0
            )
        ]
        
        logger.info("ExecutionEngine initialized")
    
    async def initialize(self):
        """Initialize execution engine and connect to venues."""
        logger.info("Initializing execution engine...")
        
        # Create venue adapters
        for config in self._default_venues:
            adapter = VenueAdapter(config)
            self._venues[config.name] = adapter
        
        # Connect to all venues
        connected = []
        for name, adapter in self._venues.items():
            try:
                if await adapter.connect():
                    connected.append(name)
            except VenueUnavailableError as e:
                logger.warning(f"Could not connect to {name}: {e}")
        
        logger.info(f"Connected to {len(connected)} venues: {connected}")
    
    async def execute(self, order: Order) -> ExecutionResult:
        """
        Execute an order on the best available venue.
        
        Args:
            order: Order to execute
            
        Returns:
            ExecutionResult with execution details
        """
        # Select best venue
        venue = self._select_venue(order)
        if not venue:
            return ExecutionResult(
                success=False,
                order_id=order.id,
                status=ExecutionStatus.ERROR,
                filled_qty=Decimal("0"),
                error="No venue available"
            )
        
        # Execute on selected venue
        try:
            adapter = self._venues[venue]
            result = await adapter.submit_order(order)
            
            # Record execution
            async with self._lock:
                self._execution_history.append(result)
                self._fill_history.extend(result.fills)
            
            # Update order
            order.venue = venue
            if result.success:
                order.status = OrderStatus.FILLED
                order.filled_qty = result.filled_qty
                order.avg_price = result.avg_price
            else:
                order.status = OrderStatus.REJECTED
            
            return result
            
        except ExecutionError as e:
            logger.error(f"Execution failed for order {order.id}: {e}")
            return ExecutionResult(
                success=False,
                order_id=order.id,
                status=ExecutionStatus.ERROR,
                filled_qty=Decimal("0"),
                error=str(e)
            )
    
    async def cancel(self, order_id: str, venue: str) -> bool:
        """Cancel order on specific venue."""
        if venue not in self._venues:
            logger.error(f"Unknown venue: {venue}")
            return False
        
        adapter = self._venues[venue]
        return await adapter.cancel_order(order_id)
    
    def _select_venue(self, order: Order) -> Optional[str]:
        """
        Select best venue for order execution.
        
        Considers:
        - Venue availability
        - Trading fees
        - Latency
        - Order size limits
        """
        available_venues = []
        
        for name, adapter in self._venues.items():
            config = adapter.config
            
            # Check if venue supports order type
            if order.order_type.value == "limit" and not config.supports_limit:
                continue
            if order.order_type.value == "market" and not config.supports_market:
                continue
            
            # Check order size limit
            if order.quantity > config.max_order_size:
                continue
            
            # Check if venue is active
            if not config.is_active:
                continue
            
            # Calculate venue score (lower is better)
            fee_score = float(config.fee_taker)
            latency_score = config.latency_ms / 1000.0  # Normalize
            
            score = fee_score + latency_score
            
            available_venues.append((name, score))
        
        if not available_venues:
            return None
        
        # Select venue with lowest score
        available_venues.sort(key=lambda x: x[1])
        return available_venues[0][0]
    
    async def get_execution_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[ExecutionResult]:
        """Get execution history."""
        async with self._lock:
            history = self._execution_history[-limit:]
        
        if symbol:
            # Filter by symbol (need to join with order data)
            pass
        
        return history
    
    async def get_fill_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Fill]:
        """Get fill history."""
        async with self._lock:
            fills = self._fill_history[-limit:]
        
        if symbol:
            fills = [f for f in fills if f.symbol == symbol]
        
        return fills
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        async with self._lock:
            total_executions = len(self._execution_history)
            total_fills = len(self._fill_history)
            
            if self._execution_history:
                success_count = sum(1 for e in self._execution_history if e.success)
                success_rate = success_count / total_executions
                
                avg_latency = sum(e.latency_ms for e in self._execution_history) / total_executions
                
                total_fees = sum(e.fees for e in self._execution_history)
            else:
                success_rate = 0.0
                avg_latency = 0.0
                total_fees = Decimal("0")
        
        return {
            "total_executions": total_executions,
            "total_fills": total_fills,
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency,
            "total_fees": float(total_fees),
            "active_venues": len([v for v in self._venues.values() if v._connected])
        }
    
    async def get_venue_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all venues."""
        status = {}
        for name, adapter in self._venues.items():
            status[name] = {
                "connected": adapter._connected,
                "config": {
                    "fee_maker": float(adapter.config.fee_maker),
                    "fee_taker": float(adapter.config.fee_taker),
                    "latency_ms": adapter.config.latency_ms
                }
            }
        return status
