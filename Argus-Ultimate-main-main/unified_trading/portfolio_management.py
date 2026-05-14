"""
Portfolio Management Module
===========================

Portfolio tracking, P&L calculation, and position management.
Refactored from unified_trading_system.py.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal
from collections import defaultdict

from unified_trading.execution_engine import ExecutionResult, Fill
from core.exception_manager import handle_errors

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Portfolio position."""
    symbol: str
    quantity: Decimal
    avg_entry_price: Decimal
    side: str  # "long" or "short"
    market_price: Optional[Decimal] = None
    unrealized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))
    opened_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    trade_count: int = 0
    max_size: Decimal = field(default_factory=lambda: Decimal("0"))


@dataclass
class PortfolioSummary:
    """Portfolio summary statistics."""
    total_value: Decimal
    cash_balance: Decimal
    positions_value: Decimal
    total_pnl: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    day_pnl: Decimal
    total_fees: Decimal
    margin_used: Decimal
    margin_available: Decimal
    num_positions: int
    max_drawdown: float
    win_rate: float
    sharpe_ratio: float


@dataclass
class TradeRecord:
    """Individual trade record."""
    id: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    fees: Decimal
    pnl: Optional[Decimal]
    timestamp: datetime
    venue: str
    strategy: Optional[str] = None


class PortfolioManager:
    """
    Manages portfolio positions, P&L, and performance tracking.
    """
    
    def __init__(self):
        self._positions: Dict[str, Position] = {}
        self._cash_balance: Decimal = Decimal("0")
        self._trade_history: List[TradeRecord] = []
        self._daily_pnl: Decimal = Decimal("0")
        self._total_fees: Decimal = Decimal("0")
        self._lock = asyncio.Lock()
        
        # Performance tracking
        self._equity_curve: List[Dict] = []
        self._win_count = 0
        self._loss_count = 0
        self._max_equity = Decimal("0")
        self._max_drawdown = 0.0
        
        logger.info("PortfolioManager initialized")
    
    async def initialize(self, initial_balance: Decimal):
        """Initialize portfolio with starting balance."""
        self._cash_balance = initial_balance
        self._max_equity = initial_balance
        
        logger.info(f"Portfolio initialized with {initial_balance}")
    
    @handle_errors(logger_name=__name__, reraise=False)
    async def update_position(self, execution: ExecutionResult):
        """
        Update position from execution result.
        
        Args:
            execution: Execution result from trade
        """
        if not execution.success:
            logger.warning(f"Not updating position for failed execution: {execution.order_id}")
            return
        
        for fill in execution.fills:
            await self._process_fill(fill)
    
    async def _process_fill(self, fill: Fill):
        """Process a fill and update positions."""
        async with self._lock:
            symbol = fill.symbol
            
            # Calculate trade value
            trade_value = fill.filled_qty * fill.price
            fees = fill.fees
            
            # Update cash balance
            if fill.side == "buy":
                self._cash_balance -= trade_value + fees
            else:
                self._cash_balance += trade_value - fees
            
            # Update or create position
            if symbol in self._positions:
                position = self._positions[symbol]
                
                if fill.side == "buy":
                    # Increase long position
                    total_cost = (position.quantity * position.avg_entry_price) + trade_value
                    total_qty = position.quantity + fill.filled_qty
                    position.avg_entry_price = total_cost / total_qty
                    position.quantity = total_qty
                else:
                    # Decrease long position or increase short
                    if position.side == "long" and position.quantity > fill.filled_qty:
                        # Partial close
                        realized = (fill.price - position.avg_entry_price) * fill.filled_qty
                        position.realized_pnl += realized
                        self._daily_pnl += realized
                        position.quantity -= fill.filled_qty
                    else:
                        # Flip to short or close all
                        realized = (fill.price - position.avg_entry_price) * min(
                            position.quantity, fill.filled_qty
                        )
                        position.realized_pnl += realized
                        self._daily_pnl += realized
                        
                        if fill.filled_qty > position.quantity:
                            # Flip to short
                            position.side = "short"
                            position.quantity = fill.filled_qty - position.quantity
                            position.avg_entry_price = fill.price
                        else:
                            position.quantity = Decimal("0")
                
                position.total_fees += fees
                position.trade_count += 1
                position.max_size = max(position.max_size, position.quantity)
                position.updated_at = datetime.utcnow()
                
            else:
                # Create new position
                side = "long" if fill.side == "buy" else "short"
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=fill.filled_qty,
                    avg_entry_price=fill.price,
                    side=side,
                    market_price=fill.price,
                    total_fees=fees,
                    trade_count=1,
                    max_size=fill.filled_qty
                )
            
            # Record trade
            trade = TradeRecord(
                id=f"TRD-{fill.order_id}",
                symbol=symbol,
                side=fill.side,
                quantity=fill.filled_qty,
                price=fill.price,
                fees=fees,
                pnl=None,  # Calculate if closing
                timestamp=fill.timestamp,
                venue=fill.venue
            )
            self._trade_history.append(trade)
            
            # Update fees
            self._total_fees += fees
            
            # Update win/loss tracking
            if trade.pnl:
                if trade.pnl > 0:
                    self._win_count += 1
                else:
                    self._loss_count += 1
            
            # Update equity curve
            portfolio_value = await self._calculate_total_value()
            self._equity_curve.append({
                "timestamp": datetime.utcnow().isoformat(),
                "value": float(portfolio_value)
            })
            
            # Update max equity and drawdown
            if portfolio_value > self._max_equity:
                self._max_equity = portfolio_value
            
            drawdown = (self._max_equity - portfolio_value) / self._max_equity
            self._max_drawdown = max(self._max_drawdown, float(drawdown))
            
            logger.info(f"Fill processed: {fill.symbol} {fill.side} {fill.filled_qty} @ {fill.price}")
    
    async def close_position(self, symbol: str, price: Decimal) -> Optional[Decimal]:
        """
        Close position and calculate P&L.
        
        Args:
            symbol: Symbol to close
            price: Exit price
            
        Returns:
            Realized P&L or None if no position
        """
        async with self._lock:
            if symbol not in self._positions:
                return None
            
            position = self._positions[symbol]
            
            # Calculate P&L
            if position.side == "long":
                pnl = (price - position.avg_entry_price) * position.quantity
            else:
                pnl = (position.avg_entry_price - price) * position.quantity
            
            # Update cash
            position_value = position.quantity * price
            if position.side == "long":
                self._cash_balance += position_value
            else:
                self._cash_balance -= position_value
            
            # Update realized P&L
            position.realized_pnl += pnl
            self._daily_pnl += pnl
            
            # Update win/loss
            if pnl > 0:
                self._win_count += 1
            else:
                self._loss_count += 1
            
            # Remove position
            del self._positions[symbol]
            
            logger.info(f"Position closed: {symbol} P&L: {pnl}")
            return pnl
    
    async def update_market_prices(self, prices: Dict[str, Decimal]):
        """Update market prices for all positions."""
        async with self._lock:
            for symbol, price in prices.items():
                if symbol in self._positions:
                    position = self._positions[symbol]
                    position.market_price = price
                    
                    # Calculate unrealized P&L
                    if position.side == "long":
                        position.unrealized_pnl = (price - position.avg_entry_price) * position.quantity
                    else:
                        position.unrealized_pnl = (position.avg_entry_price - price) * position.quantity
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self._positions.get(symbol)
    
    async def get_positions(self) -> List[Position]:
        """Get all positions."""
        return list(self._positions.values())
    
    async def get_summary(self) -> PortfolioSummary:
        """Get portfolio summary."""
        async with self._lock:
            positions_value = sum(
                p.quantity * (p.market_price or p.avg_entry_price)
                for p in self._positions.values()
            )
            
            total_unrealized = sum(p.unrealized_pnl for p in self._positions.values())
            total_realized = sum(p.realized_pnl for p in self._positions.values())
            
            total_value = self._cash_balance + positions_value
            
            # Calculate win rate
            total_trades = self._win_count + self._loss_count
            win_rate = self._win_count / total_trades if total_trades > 0 else 0.0
            
            return PortfolioSummary(
                total_value=total_value,
                cash_balance=self._cash_balance,
                positions_value=positions_value,
                total_pnl=total_unrealized + total_realized,
                unrealized_pnl=total_unrealized,
                realized_pnl=total_realized,
                day_pnl=self._daily_pnl,
                total_fees=self._total_fees,
                margin_used=positions_value * Decimal("0.1"),  # 10% margin
                margin_available=self._cash_balance,
                num_positions=len(self._positions),
                max_drawdown=self._max_drawdown,
                win_rate=win_rate,
                sharpe_ratio=0.0  # Would calculate
            )
    
    async def get_state(self) -> Dict[str, Any]:
        """Get portfolio state for persistence."""
        return {
            "cash_balance": float(self._cash_balance),
            "total_fees": float(self._total_fees),
            "daily_pnl": float(self._daily_pnl),
            "win_count": self._win_count,
            "loss_count": self._loss_count,
            "max_equity": float(self._max_equity),
            "max_drawdown": self._max_drawdown
        }
    
    async def restore_state(self, state: Dict[str, Any]):
        """Restore portfolio from state."""
        self._cash_balance = Decimal(str(state.get("cash_balance", 0)))
        self._total_fees = Decimal(str(state.get("total_fees", 0)))
        self._daily_pnl = Decimal(str(state.get("daily_pnl", 0)))
        self._win_count = state.get("win_count", 0)
        self._loss_count = state.get("loss_count", 0)
        self._max_equity = Decimal(str(state.get("max_equity", self._cash_balance)))
        self._max_drawdown = state.get("max_drawdown", 0.0)
    
    async def restore_positions(self, positions_data: List[Dict]):
        """Restore positions from saved state."""
        for pos_data in positions_data:
            try:
                position = Position(
                    symbol=pos_data["symbol"],
                    quantity=Decimal(str(pos_data["quantity"])),
                    avg_entry_price=Decimal(str(pos_data["avg_entry_price"])),
                    side=pos_data["side"],
                    realized_pnl=Decimal(str(pos_data.get("realized_pnl", 0))),
                    total_fees=Decimal(str(pos_data.get("total_fees", 0))),
                    trade_count=pos_data.get("trade_count", 0),
                    max_size=Decimal(str(pos_data.get("max_size", 0)))
                )
                self._positions[position.symbol] = position
            except Exception as e:
                logger.error(f"Failed to restore position: {e}")
    
    async def check_health(self) -> Dict[str, Any]:
        """Check portfolio health."""
        issues = []
        
        if self._cash_balance < 0:
            issues.append("Negative cash balance")
        
        if self._max_drawdown > 0.2:  # 20%
            issues.append(f"High drawdown: {self._max_drawdown:.1%}")
        
        total_trades = self._win_count + self._loss_count
        if total_trades > 10:
            win_rate = self._win_count / total_trades
            if win_rate < 0.3:
                issues.append(f"Low win rate: {win_rate:.1%}")
        
        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "positions": len(self._positions),
            "cash_balance": float(self._cash_balance)
        }
    
    async def reset_daily_pnl(self):
        """Reset daily P&L (call at start of trading day)."""
        self._daily_pnl = Decimal("0")
        logger.info("Daily P&L reset")
    
    async def _calculate_total_value(self) -> Decimal:
        """Calculate total portfolio value."""
        positions_value = sum(
            p.quantity * (p.market_price or p.avg_entry_price)
            for p in self._positions.values()
        )
        return self._cash_balance + positions_value
