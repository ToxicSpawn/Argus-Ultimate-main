"""
Real-Time Position Tracker
Tracks live positions, P&L, and risk metrics
Wires exchange positions to risk management
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import numpy as np

from wiring.exchange_connector import get_exchange_manager, LivePosition, LiveOrder

logger = logging.getLogger(__name__)


@dataclass
class PositionSnapshot:
    """Real-time position snapshot"""
    symbol: str
    exchange: str
    amount: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float
    side: str  # 'long' or 'short'
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class PortfolioSnapshot:
    """Complete portfolio snapshot"""
    timestamp: datetime
    total_value: float
    cash_balance: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    total_exposure: float
    margin_used: float
    available_margin: float
    positions: List[PositionSnapshot] = field(default_factory=list)


class RealtimePositionTracker:
    """
    Real-time position tracking across all exchanges
    Calculates live P&L and risk metrics
    """
    
    def __init__(self):
        self.positions: Dict[str, PositionSnapshot] = {}  # symbol+exchange -> position
        self.cash_balances: Dict[str, float] = defaultdict(float)  # exchange -> cash
        self.order_history: List[LiveOrder] = []
        self.trade_history: List[Dict] = []
        
        # Performance tracking
        self.daily_starting_value: float = 0.0
        self.peak_portfolio_value: float = 0.0
        self.max_drawdown: float = 0.0
        
        # Update loop
        self.update_task: Optional[asyncio.Task] = None
        self.is_running = False
        self.update_interval_seconds: float = 1.0
        
        # Callbacks
        self.position_callbacks: List[Callable] = []
        self.risk_callbacks: List[Callable] = []
        
        logger.info("📊 Real-time position tracker initialized")
    
    async def start(self):
        """Start position tracking loop"""
        self.is_running = True
        self.update_task = asyncio.create_task(self._update_loop())
        
        # Set daily starting value
        portfolio = await self.get_portfolio_snapshot()
        self.daily_starting_value = portfolio.total_value
        
        logger.info(f"✅ Position tracker started (daily value: ${self.daily_starting_value:,.2f})")
    
    async def stop(self):
        """Stop position tracking"""
        self.is_running = False
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        logger.info("⏹️ Position tracker stopped")
    
    async def _update_loop(self):
        """Continuous update loop"""
        while self.is_running:
            try:
                await self.sync_positions()
                await self.calculate_live_pnl()
                await self._check_risk_limits()
                
            except Exception as e:
                logger.error(f"Position update error: {e}")
            
            await asyncio.sleep(self.update_interval_seconds)
    
    async def sync_positions(self):
        """Sync positions from all exchanges"""
        manager = get_exchange_manager()
        
        # Get positions from exchanges
        exchange_positions = await manager.get_all_positions()
        
        # Update local position cache
        for pos in exchange_positions:
            key = f"{pos.symbol}_{pos.exchange}"
            
            # Get current price
            current_price = await self._get_current_price(pos.symbol, pos.exchange)
            
            if current_price > 0:
                market_value = pos.amount * current_price
                unrealized_pnl = pos.amount * (current_price - pos.entry_price)
                unrealized_pnl_pct = (current_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0
                
                snapshot = PositionSnapshot(
                    symbol=pos.symbol,
                    exchange=pos.exchange,
                    amount=pos.amount,
                    entry_price=pos.entry_price,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    realized_pnl=pos.realized_pnl,
                    side='long' if pos.amount > 0 else 'short'
                )
                
                old_snapshot = self.positions.get(key)
                self.positions[key] = snapshot
                
                # Notify if significant change
                if old_snapshot:
                    pnl_change = abs(snapshot.unrealized_pnl - old_snapshot.unrealized_pnl)
                    if pnl_change > 10:  # $10 change
                        await self._notify_position_change(snapshot)
    
    async def calculate_live_pnl(self) -> PortfolioSnapshot:
        """Calculate live portfolio P&L"""
        total_positions_value = sum(p.market_value for p in self.positions.values())
        total_unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        total_realized_pnl = sum(p.realized_pnl for p in self.positions.values())
        
        total_cash = sum(self.cash_balances.values())
        total_value = total_positions_value + total_cash
        
        # Calculate daily P&L
        daily_pnl = total_value - self.daily_starting_value if self.daily_starting_value > 0 else 0
        
        # Calculate exposure
        gross_exposure = sum(abs(p.market_value) for p in self.positions.values())
        
        # Update peak and drawdown
        if total_value > self.peak_portfolio_value:
            self.peak_portfolio_value = total_value
        
        current_drawdown = (self.peak_portfolio_value - total_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        portfolio = PortfolioSnapshot(
            timestamp=datetime.now(),
            total_value=total_value,
            cash_balance=total_cash,
            positions_value=total_positions_value,
            unrealized_pnl=total_unrealized_pnl,
            realized_pnl=total_realized_pnl,
            daily_pnl=daily_pnl,
            total_exposure=gross_exposure,
            margin_used=gross_exposure * 0.1,  # Assume 10% margin
            available_margin=total_cash * 10,  # 10x leverage available
            positions=list(self.positions.values())
        )
        
        return portfolio
    
    async def process_order_fill(self, order: LiveOrder):
        """Process order fill and update positions"""
        if order.status.value == "filled" and order.filled_amount > 0:
            # Calculate realized P&L
            key = f"{order.symbol}_{order.exchange}"
            existing_position = self.positions.get(key)
            
            if existing_position:
                # Calculate realized P&L
                if order.side.value == "sell":
                    realized = order.filled_amount * (order.average_price - existing_position.entry_price)
                    existing_position.realized_pnl += realized
                    existing_position.amount -= order.filled_amount
                else:  # buy
                    # Adjust average entry price
                    total_cost = (existing_position.amount * existing_position.entry_price + 
                                 order.filled_amount * order.average_price)
                    existing_position.amount += order.filled_amount
                    existing_position.entry_price = total_cost / existing_position.amount
            else:
                # New position
                self.positions[key] = PositionSnapshot(
                    symbol=order.symbol,
                    exchange=order.exchange,
                    amount=order.filled_amount if order.side.value == "buy" else -order.filled_amount,
                    entry_price=order.average_price,
                    current_price=order.average_price,
                    market_value=order.filled_amount * order.average_price,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                    realized_pnl=0.0,
                    side='long' if order.side.value == "buy" else 'short'
                )
            
            # Log trade
            self.trade_history.append({
                'timestamp': datetime.now(),
                'order': order,
                'realized_pnl': existing_position.realized_pnl if existing_position else 0
            })
            
            logger.info(f"💰 Fill: {order.side.value.upper()} {order.filled_amount} {order.symbol} "
                       f"@ ${order.average_price:,.2f} (PnL: ${existing_position.realized_pnl if existing_position else 0:+.2f})")
    
    async def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        """Get current portfolio snapshot"""
        return await self.calculate_live_pnl()
    
    async def get_position(self, symbol: str, exchange: str) -> Optional[PositionSnapshot]:
        """Get specific position"""
        key = f"{symbol}_{exchange}"
        return self.positions.get(key)
    
    async def get_exposure_by_symbol(self) -> Dict[str, float]:
        """Get exposure aggregated by symbol across exchanges"""
        exposure = defaultdict(float)
        for pos in self.positions.values():
            exposure[pos.symbol] += pos.market_value
        return dict(exposure)
    
    async def get_exposure_by_exchange(self) -> Dict[str, float]:
        """Get exposure by exchange"""
        exposure = defaultdict(float)
        for pos in self.positions.values():
            exposure[pos.exchange] += pos.market_value
        return dict(exposure)
    
    async def flatten_all_positions(self):
        """Close all positions (emergency)"""
        logger.warning("🚨 EMERGENCY: Flattening all positions!")
        
        manager = get_exchange_manager()
        
        for key, pos in list(self.positions.items()):
            try:
                if pos.amount > 0:
                    # Sell long position
                    await manager.submit_order(
                        exchange=pos.exchange,
                        symbol=pos.symbol,
                        side="sell",
                        amount=abs(pos.amount),
                        order_type="market"
                    )
                elif pos.amount < 0:
                    # Cover short position
                    await manager.submit_order(
                        exchange=pos.exchange,
                        symbol=pos.symbol,
                        side="buy",
                        amount=abs(pos.amount),
                        order_type="market"
                    )
                
                logger.info(f"  Closed: {pos.symbol} on {pos.exchange}")
                
            except Exception as e:
                logger.error(f"  Failed to close {pos.symbol}: {e}")
    
    async def _check_risk_limits(self):
        """Check risk limits and trigger callbacks if breached"""
        portfolio = await self.get_portfolio_snapshot()
        
        # Check daily loss limit
        if self.daily_starting_value > 0:
            daily_loss_pct = abs(portfolio.daily_pnl) / self.daily_starting_value
            
            if daily_loss_pct > 0.05:  # 5% daily loss
                logger.warning(f"🛑 Daily loss limit reached: {daily_loss_pct*100:.1f}%")
                for callback in self.risk_callbacks:
                    await callback("daily_loss_limit", portfolio)
        
        # Check max drawdown
        if self.max_drawdown > 0.10:  # 10% drawdown
            logger.warning(f"🚨 Max drawdown reached: {self.max_drawdown*100:.1f}%")
            for callback in self.risk_callbacks:
                await callback("max_drawdown", portfolio)
    
    async def _get_current_price(self, symbol: str, exchange: str) -> float:
        """Get current market price"""
        manager = get_exchange_manager()
        connector = manager.connectors.get(exchange)
        
        if connector:
            ticker = await connector.get_ticker(symbol)
            return ticker.get("last", 0.0)
        
        return 0.0
    
    async def _notify_position_change(self, position: PositionSnapshot):
        """Notify position change callbacks"""
        for callback in self.position_callbacks:
            await callback(position)
    
    def register_position_callback(self, callback: Callable):
        """Register position update callback"""
        self.position_callbacks.append(callback)
    
    def register_risk_callback(self, callback: Callable):
        """Register risk breach callback"""
        self.risk_callbacks.append(callback)
    
    def get_performance_stats(self) -> Dict:
        """Get trading performance statistics"""
        if not self.trade_history:
            return {}
        
        realized_pnls = [t['realized_pnl'] for t in self.trade_history]
        
        return {
            'total_trades': len(self.trade_history),
            'winning_trades': sum(1 for p in realized_pnls if p > 0),
            'losing_trades': sum(1 for p in realized_pnls if p < 0),
            'total_realized_pnl': sum(realized_pnls),
            'avg_trade_pnl': np.mean(realized_pnls) if realized_pnls else 0,
            'max_drawdown': self.max_drawdown,
            'current_positions': len(self.positions),
            'total_exposure': sum(abs(p.market_value) for p in self.positions.values())
        }


# Global instance
_position_tracker: Optional[RealtimePositionTracker] = None


def get_position_tracker() -> RealtimePositionTracker:
    """Get singleton position tracker"""
    global _position_tracker
    if _position_tracker is None:
        _position_tracker = RealtimePositionTracker()
    return _position_tracker


async def init_position_tracking():
    """Initialize position tracking"""
    tracker = get_position_tracker()
    await tracker.start()
    
    # Register order fill callback
    manager = get_exchange_manager()
    manager.register_order_callback(tracker.process_order_fill)
