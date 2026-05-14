"""
Order Management Module
=====================

Handles order lifecycle from creation through execution to settlement.
Refactored from unified_trading_system.py for better maintainability.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal
from enum import Enum, auto

from core.exception_manager import (
    OrderProcessingError,
    handle_errors,
    validate_required
)

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order lifecycle status."""
    PENDING = auto()
    SUBMITTED = auto()
    PARTIAL = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    ERROR = auto()


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a trading order."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_price: Optional[Decimal] = None
    venue: Optional[str] = None
    strategy: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        validate_required(self.symbol, "symbol", str)
        validate_required(self.quantity, "quantity", Decimal)
        
        if self.quantity <= 0:
            raise OrderProcessingError(
                "Order quantity must be positive",
                order_id=self.id,
                symbol=self.symbol
            )
    
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL
        ]
    
    def update_fill(self, filled_qty: Decimal, avg_price: Decimal):
        """Update order with fill information."""
        self.filled_qty += filled_qty
        self.avg_price = avg_price
        
        if self.filled_qty >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL
        
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.order_type.value,
            "quantity": str(self.quantity),
            "price": str(self.price) if self.price else None,
            "status": self.status.name,
            "filled_qty": str(self.filled_qty),
            "avg_price": str(self.avg_price) if self.avg_price else None,
            "venue": self.venue,
            "strategy": self.strategy,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


@dataclass
class Signal:
    """Trading signal from strategy."""
    symbol: str
    side: OrderSide
    confidence: float
    strategy: str
    suggested_qty: Decimal
    suggested_price: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class OrderManager:
    """
    Manages order lifecycle from creation through execution.
    """
    
    def __init__(self):
        self._orders: Dict[str, Order] = {}
        self._active_orders: Dict[str, Order] = {}
        self._order_history: List[Order] = []
        self._lock = asyncio.Lock()
        
        logger.info("OrderManager initialized")
    
    async def initialize(self):
        """Initialize order manager."""
        logger.info("Order manager initialized")
    
    async def create_order(self, signal: Signal) -> Order:
        """
        Create order from trading signal.
        
        Args:
            signal: Trading signal from strategy
            
        Returns:
            Order: Created order
            
        Raises:
            OrderProcessingError: If order creation fails
        """
        try:
            # Validate signal
            if signal.confidence < 0.5:
                raise OrderProcessingError(
                    f"Signal confidence too low: {signal.confidence}",
                    symbol=signal.symbol
                )
            
            # Generate order ID
            order_id = self._generate_order_id()
            
            # Create order
            order = Order(
                id=order_id,
                symbol=signal.symbol,
                side=signal.side,
                order_type=OrderType.MARKET,  # Default to market
                quantity=signal.suggested_qty,
                price=signal.suggested_price,
                strategy=signal.strategy,
                metadata=signal.metadata
            )
            
            # Store order
            async with self._lock:
                self._orders[order_id] = order
                self._active_orders[order_id] = order
            
            logger.info(f"Order created: {order_id} {signal.symbol} {signal.side.value}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            raise OrderProcessingError(
                f"Order creation failed: {e}",
                symbol=signal.symbol
            ) from e
    
    @handle_errors(logger_name=__name__, reraise=True)
    async def submit_order(self, order: Order, venue: str) -> Order:
        """
        Submit order to venue for execution.
        
        Args:
            order: Order to submit
            venue: Target venue
            
        Returns:
            Order: Updated order
        """
        order.venue = venue
        order.status = OrderStatus.SUBMITTED
        order.updated_at = datetime.utcnow()
        
        logger.info(f"Order {order.id} submitted to {venue}")
        return order
    
    @handle_errors(logger_name=__name__, reraise=False)
    async def update_order(self, order_id: str, **kwargs) -> Optional[Order]:
        """
        Update order with new information.
        
        Args:
            order_id: Order ID
            **kwargs: Fields to update
            
        Returns:
            Order: Updated order or None if not found
        """
        async with self._lock:
            order = self._orders.get(order_id)
            if not order:
                logger.warning(f"Order not found: {order_id}")
                return None
            
            # Update fields
            for key, value in kwargs.items():
                if hasattr(order, key):
                    setattr(order, key, value)
            
            order.updated_at = datetime.utcnow()
            
            # Update active orders tracking
            if not order.is_active() and order_id in self._active_orders:
                del self._active_orders[order_id]
                self._order_history.append(order)
            
            return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            bool: True if cancelled successfully
        """
        async with self._lock:
            order = self._active_orders.get(order_id)
            if not order:
                logger.warning(f"Cannot cancel - order not active: {order_id}")
                return False
            
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.utcnow()
            
            del self._active_orders[order_id]
            self._order_history.append(order)
            
            logger.info(f"Order cancelled: {order_id}")
            return True
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    async def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        return list(self._active_orders.values())
    
    async def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol."""
        return [
            order for order in self._orders.values()
            if order.symbol == symbol
        ]
    
    async def check_stuck_orders(self) -> List[Order]:
        """
        Check for stuck orders (submitted but not filled for long time).
        
        Returns:
            List of stuck orders
        """
        stuck_orders = []
        timeout_seconds = 300  # 5 minutes
        
        for order in self._active_orders.values():
            if order.status == OrderStatus.SUBMITTED:
                elapsed = (datetime.utcnow() - order.created_at).total_seconds()
                if elapsed > timeout_seconds:
                    stuck_orders.append(order)
        
        if stuck_orders:
            logger.warning(f"Found {len(stuck_orders)} stuck orders")
        
        return stuck_orders
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get order statistics."""
        total_orders = len(self._orders)
        active_orders = len(self._active_orders)
        filled_orders = len([o for o in self._order_history if o.status == OrderStatus.FILLED])
        cancelled_orders = len([o for o in self._order_history if o.status == OrderStatus.CANCELLED])
        
        return {
            "total_orders": total_orders,
            "active_orders": active_orders,
            "filled_orders": filled_orders,
            "cancelled_orders": cancelled_orders,
            "fill_rate": filled_orders / (filled_orders + cancelled_orders) if (filled_orders + cancelled_orders) > 0 else 0
        }
    
    async def restore_orders(self, orders_data: List[Dict]):
        """Restore orders from saved state."""
        async with self._lock:
            for order_data in orders_data:
                try:
                    order = Order(
                        id=order_data["id"],
                        symbol=order_data["symbol"],
                        side=OrderSide(order_data["side"]),
                        order_type=OrderType(order_data["type"]),
                        quantity=Decimal(order_data["quantity"]),
                        price=Decimal(order_data["price"]) if order_data.get("price") else None,
                        status=OrderStatus[order_data["status"]],
                        filled_qty=Decimal(order_data.get("filled_qty", "0")),
                        venue=order_data.get("venue"),
                        strategy=order_data.get("strategy")
                    )
                    
                    self._orders[order.id] = order
                    
                    if order.is_active():
                        self._active_orders[order.id] = order
                    else:
                        self._order_history.append(order)
                
                except Exception as e:
                    logger.error(f"Failed to restore order: {e}")
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        import uuid
        return f"ORD-{uuid.uuid4().hex[:12].upper()}"
