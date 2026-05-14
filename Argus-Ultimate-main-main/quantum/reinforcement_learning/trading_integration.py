# pyright: reportMissingImports=false
"""
Integration Layer for Quantum RL Trading System.

This module provides:
- Market data feed integration
- Execution engine integration
- Portfolio management integration
- Live trading orchestration
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Available market data sources."""
    BINANCE = auto()
    BYBIT = auto()
    COINBASE = auto()
    KRAKEN = auto()
    SIMULATED = auto()
    HISTORICAL = auto()


class OrderType(Enum):
    """Order types for execution."""
    MARKET = auto()
    LIMIT = auto()
    STOP_LOSS = auto()
    TAKE_PROFIT = auto()
    TRAILING_STOP = auto()


class OrderSide(Enum):
    """Order sides."""
    BUY = auto()
    SELL = auto()


@dataclass
class Order:
    """Trading order representation."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    order_id: Optional[str] = None
    status: str = "pending"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketDataConfig:
    """Configuration for market data feed."""
    data_source: DataSource = DataSource.SIMULATED
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT"])
    timeframe: str = "1h"
    lookback_period: int = 1000
    update_interval_ms: int = 1000
    enable_order_book: bool = False
    enable_trades: bool = False


class MarketDataFeed:
    """Provides market data to the quantum RL system."""
    
    def __init__(self, config: Optional[MarketDataConfig] = None):
        self.config = config or MarketDataConfig()
        self.latest_data: Dict[str, Dict[str, Any]] = {}
        self.historical_data: Dict[str, List[Dict[str, Any]]] = {s: [] for s in self.config.symbols}
        self.is_running = False
        self._callbacks: List[Callable] = []
    
    async def start(self) -> None:
        """Start the market data feed."""
        self.is_running = True
        logger.info("Starting market data feed for %s", self.config.symbols)
        
        # Initialize historical data
        for symbol in self.config.symbols:
            self.historical_data[symbol] = await self._fetch_historical_data(symbol)
    
    async def stop(self) -> None:
        """Stop the market data feed."""
        self.is_running = False
        logger.info("Stopping market data feed")
    
    def subscribe(self, callback: Callable) -> None:
        """Subscribe to market data updates."""
        self._callbacks.append(callback)
    
    async def get_latest(self, symbol: str) -> Dict[str, Any]:
        """Get latest market data for a symbol."""
        if symbol in self.latest_data:
            return self.latest_data[symbol]
        
        # Fetch fresh data
        data = await self._fetch_latest_data(symbol)
        self.latest_data[symbol] = data
        return data
    
    async def get_historical(
        self,
        symbol: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get historical data for a symbol."""
        data = self.historical_data.get(symbol, [])
        if limit:
            return data[-limit:]
        return data
    
    async def _fetch_historical_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch historical data from source."""
        # In production, this would connect to actual exchange APIs
        # For now, return simulated data
        historical = []
        
        base_price = 50000.0  # Simulated BTC price
        
        for i in range(self.config.lookback_period):
            # Generate realistic OHLCV data
            price_change = np.random.randn() * 0.02  # 2% volatility
            close = base_price * (1 + price_change)
            high = close * (1 + abs(np.random.randn() * 0.01))
            low = close * (1 - abs(np.random.randn() * 0.01))
            open_price = close * (1 + np.random.randn() * 0.005)
            volume = np.random.exponential(100)
            
            candle = {
                "timestamp": datetime.now().isoformat(),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "quote_volume": volume * close
            }
            
            historical.append(candle)
            base_price = close
        
        return historical
    
    async def _fetch_latest_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch latest market data."""
        # Simulated latest data
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "price": 50000.0 + np.random.randn() * 100,
            "volume": np.random.exponential(100),
            "bid": 49999.0,
            "ask": 50001.0,
            "order_book": {
                "best_bid": 49999.0,
                "best_ask": 50001.0,
                "bid_volume": 10.0,
                "ask_volume": 8.0
            }
        }


@dataclass
class ExecutionConfig:
    """Configuration for execution engine."""
    paper_trading: bool = True
    slippage_model: str = "fixed"  # fixed, percentage, volume_impact
    slippage_percentage: float = 0.001
    max_slippage: float = 0.01
    commission_rate: float = 0.001
    min_order_size: float = 0.001
    max_order_size: float = 100.0
    enable_stop_loss: bool = True
    enable_take_profit: bool = True


class ExecutionEngine:
    """Handles order execution for quantum RL trading."""
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.order_history: List[Order] = []
        self.open_orders: Dict[str, Order] = {}
        self.positions: Dict[str, float] = {}
        self.avg_entry_prices: Dict[str, float] = {}
        
        # Performance tracking
        self.total_commission: float = 0.0
        self.total_slippage: float = 0.0
    
    async def execute_order(self, order: Order) -> Order:
        """Execute a trading order."""
        # Validate order
        if not self._validate_order(order):
            order.status = "rejected"
            return order
        
        # Apply slippage
        executed_price = self._apply_slippage(order)
        
        # Calculate commission
        commission = self._calculate_commission(order, executed_price)
        
        # Update order
        order.price = executed_price
        order.status = "filled"
        order.order_id = f"order_{len(self.order_history) + 1}"
        
        # Update positions
        self._update_position(order, executed_price, commission)
        
        # Track history
        self.order_history.append(order)
        
        logger.info(
            "Order executed: %s %s %s @ %.2f (commission: %.4f)",
            order.side.name, order.quantity, order.symbol, executed_price, commission
        )
        
        return order
    
    async def execute_signal(
        self,
        signal: Dict[str, Any],
        symbol: str
    ) -> Order:
        """Execute a trading signal from quantum RL."""
        action = signal.get("action", "HOLD")
        position_change = signal.get("position_change", 0.0)
        
        if action == "HOLD" or abs(position_change) < 0.01:
            # No action needed
            return Order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.0,
                status="no_action"
            )
        
        # Determine order side and quantity
        side = OrderSide.BUY if position_change > 0 else OrderSide.SELL
        quantity = abs(position_change)
        
        # Get current price
        current_price = 50000.0  # Would fetch from market data in production
        
        # Create order
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=current_price
        )
        
        return await self.execute_order(order)
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order parameters."""
        if order.quantity < self.config.min_order_size:
            logger.warning("Order quantity too small: %s", order.quantity)
            return False
        
        if order.quantity > self.config.max_order_size:
            logger.warning("Order quantity too large: %s", order.quantity)
            return False
        
        return True
    
    def _apply_slippage(self, order: Order) -> float:
        """Apply slippage to order price."""
        base_price = order.price or 50000.0
        
        if self.config.slippage_model == "fixed":
            slippage = self.config.slippage_percentage
        elif self.config.slippage_model == "percentage":
            slippage = self.config.slippage_percentage * order.quantity
        else:  # volume_impact
            slippage = self.config.slippage_percentage * np.sqrt(order.quantity)
        
        slippage = min(slippage, self.config.max_slippage)
        
        if order.side == OrderSide.BUY:
            executed_price = base_price * (1 + slippage)
        else:
            executed_price = base_price * (1 - slippage)
        
        self.total_slippage += abs(executed_price - base_price) * order.quantity
        
        return executed_price
    
    def _calculate_commission(self, order: Order, price: float) -> float:
        """Calculate trading commission."""
        commission = order.quantity * price * self.config.commission_rate
        self.total_commission += commission
        return commission
    
    def _update_position(self, order: Order, price: float, commission: float) -> None:
        """Update position after order execution."""
        symbol = order.symbol
        
        current_position = self.positions.get(symbol, 0.0)
        current_avg_price = self.avg_entry_prices.get(symbol, price)
        
        if order.side == OrderSide.BUY:
            # Adding to long position
            new_position = current_position + order.quantity
            
            if current_position >= 0:
                # Averaging up
                total_value = current_position * current_avg_price + order.quantity * price
                new_avg_price = total_value / new_position if new_position > 0 else price
            else:
                # Reducing short / flipping to long
                if order.quantity >= abs(current_position):
                    new_avg_price = price  # New long position
                else:
                    new_avg_price = current_avg_price  # Still short
        else:
            # Adding to short position (or reducing long)
            new_position = current_position - order.quantity
            
            if current_position <= 0:
                # Averaging down short
                total_value = abs(current_position) * current_avg_price + order.quantity * price
                new_avg_price = total_value / abs(new_position) if new_position < 0 else price
            else:
                # Reducing long / flipping to short
                if order.quantity >= current_position:
                    new_avg_price = price  # New short position
                else:
                    new_avg_price = current_avg_price  # Still long
        
        self.positions[symbol] = new_position
        self.avg_entry_prices[symbol] = new_avg_price
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get current position for a symbol."""
        return {
            "symbol": symbol,
            "size": self.positions.get(symbol, 0.0),
            "avg_entry_price": self.avg_entry_prices.get(symbol, 0.0),
            "unrealized_pnl": self._calculate_unrealized_pnl(symbol)
        }
    
    def _calculate_unrealized_pnl(self, symbol: str) -> float:
        """Calculate unrealized PnL for a position."""
        position = self.positions.get(symbol, 0.0)
        avg_price = self.avg_entry_prices.get(symbol, 0.0)
        
        if position == 0:
            return 0.0
        
        # Would use current market price in production
        current_price = 50000.0
        
        if position > 0:
            return position * (current_price - avg_price)
        else:
            return abs(position) * (avg_price - current_price)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get execution performance statistics."""
        return {
            "total_orders": len(self.order_history),
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "open_positions": len([p for p in self.positions.values() if p != 0])
        }


@dataclass
class PortfolioConfig:
    """Configuration for portfolio management."""
    initial_capital: float = 10000.0
    base_currency: str = "USDT"
    max_leverage: float = 1.0
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_positions: int = 5
    rebalance_threshold: float = 0.1


class PortfolioManager:
    """Manages portfolio state and allocation."""
    
    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig()
        
        self.cash: float = config.initial_capital if config else 10000.0
        self.positions: Dict[str, float] = {}
        self.avg_entry_prices: Dict[str, float] = {}
        self.portfolio_value_history: List[float] = [self.cash]
        
        # Performance tracking
        self.daily_pnl: float = 0.0
        self.total_realized_pnl: float = 0.0
    
    def update_from_execution(self, execution_engine: ExecutionEngine) -> None:
        """Update portfolio state from execution engine."""
        self.positions = execution_engine.positions.copy()
        self.avg_entry_prices = execution_engine.avg_entry_prices.copy()
    
    def get_portfolio_value(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """Calculate total portfolio value."""
        positions_value = 0.0
        
        for symbol, position in self.positions.items():
            if position != 0:
                price = current_prices.get(symbol, 50000.0) if current_prices else 50000.0
                positions_value += position * price
        
        total_value = self.cash + positions_value
        self.portfolio_value_history.append(total_value)
        
        return total_value
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: Optional[float] = None
    ) -> float:
        """Calculate position size based on risk parameters."""
        portfolio_value = self.get_portfolio_value()
        
        # Risk-based position sizing
        risk_amount = portfolio_value * self.config.risk_per_trade
        
        if stop_loss_price:
            # Calculate position size based on stop loss
            risk_per_unit = abs(entry_price - stop_loss_price)
            if risk_per_unit > 0:
                position_size = risk_amount / risk_per_unit
            else:
                position_size = 0.0
        else:
            # Default to fixed percentage of portfolio
            position_size = (portfolio_value * self.config.risk_per_trade) / entry_price
        
        # Apply limits
        position_size = min(position_size, portfolio_value * 0.2 / entry_price)  # Max 20% per position
        
        return position_size
    
    def get_allocation(self, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Get current portfolio allocation."""
        portfolio_value = self.get_portfolio_value(current_prices)
        
        if portfolio_value == 0:
            return {}
        
        allocation = {"cash": self.cash / portfolio_value}
        
        for symbol, position in self.positions.items():
            if position != 0:
                price = current_prices.get(symbol, 50000.0) if current_prices else 50000.0
                position_value = position * price
                allocation[symbol] = position_value / portfolio_value
        
        return allocation
    
    def calculate_returns(self) -> Dict[str, float]:
        """Calculate various return metrics."""
        if len(self.portfolio_value_history) < 2:
            return {"total_return": 0.0, "daily_return": 0.0}
        
        current_value = self.portfolio_value_history[-1]
        initial_value = self.portfolio_value_history[0]
        
        total_return = (current_value - initial_value) / initial_value
        
        return {
            "total_return": total_return,
            "current_value": current_value,
            "initial_value": initial_value,
            "daily_pnl": self.daily_pnl
        }
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """Calculate portfolio risk metrics."""
        if len(self.portfolio_value_history) < 2:
            return {"volatility": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}
        
        values = np.array(self.portfolio_value_history)
        returns = np.diff(values) / values[:-1]
        
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
        
        # Sharpe ratio (assuming risk-free rate of 0)
        mean_return = np.mean(returns)
        std_return = np.std(returns) + 1e-8
        sharpe = (mean_return / std_return) * np.sqrt(252) if len(returns) > 1 else 0.0
        
        # Max drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        return {
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown
        }


class QuantumRLTradingOrchestrator:
    """Orchestrates quantum RL trading with all components."""
    
    def __init__(
        self,
        market_data_config: Optional[MarketDataConfig] = None,
        execution_config: Optional[ExecutionConfig] = None,
        portfolio_config: Optional[PortfolioConfig] = None
    ):
        self.market_data = MarketDataFeed(market_data_config)
        self.execution = ExecutionEngine(execution_config)
        self.portfolio = PortfolioManager(portfolio_config)
        
        # RL components (to be set)
        self.agent: Optional[Any] = None
        self.state_encoder: Optional[Any] = None
        self.action_decoder: Optional[Any] = None
        self.risk_manager: Optional[Any] = None
        
        # Trading state
        self.is_trading = False
        self.trade_count = 0
        
    def set_agent(self, agent: Any) -> None:
        """Set the RL agent."""
        self.agent = agent
    
    def set_components(
        self,
        state_encoder: Any,
        action_decoder: Any,
        risk_manager: Optional[Any] = None
    ) -> None:
        """Set trading components."""
        self.state_encoder = state_encoder
        self.action_decoder = action_decoder
        self.risk_manager = risk_manager
    
    async def start_trading(self, symbol: str) -> None:
        """Start live/paper trading."""
        if self.agent is None:
            raise ValueError("RL agent not set")
        
        self.is_trading = True
        logger.info("Starting trading for %s", symbol)
        
        await self.market_data.start()
        
        while self.is_trading:
            try:
                # Get market data
                market_data = await self.market_data.get_latest(symbol)
                
                # Get portfolio state
                portfolio_info = {
                    "value": self.portfolio.get_portfolio_value(),
                    "cash": self.portfolio.cash,
                    "positions": self.portfolio.positions,
                    "daily_pnl": self.portfolio.daily_pnl
                }
                
                # Get position info
                position_info = self.execution.get_position(symbol)
                
                # Encode state
                if self.state_encoder:
                    state = self.state_encoder.encode(market_data, position_info, portfolio_info)
                else:
                    state = np.zeros(8)  # Default state
                
                # Get action from agent
                action, _ = self.agent.select_action(state, training=False)
                
                # Decode action
                if self.action_decoder:
                    decoded_action = self.action_decoder.decode(action, position_info["size"])
                else:
                    decoded_action = {"action": "HOLD", "position_change": 0.0}
                
                # Risk check
                if self.risk_manager:
                    approved, risk_result = self.risk_manager.check_action(decoded_action, portfolio_info)
                    if not approved:
                        logger.warning("Action rejected by risk manager: %s", risk_result["violations"])
                        decoded_action = risk_result["adjusted_action"]
                
                # Execute
                order = await self.execution.execute_signal(decoded_action, symbol)
                
                # Update portfolio
                self.portfolio.update_from_execution(self.execution)
                
                self.trade_count += 1
                
                if self.trade_count % 10 == 0:
                    logger.info(
                        "Trade #%d | Portfolio: %.2f | Position: %.4f",
                        self.trade_count,
                        self.portfolio.get_portfolio_value(),
                        self.portfolio.positions.get(symbol, 0.0)
                    )
                
                # Wait for next update
                await asyncio.sleep(self.market_data.config.update_interval_ms / 1000)
                
            except Exception as e:
                logger.error("Error in trading loop: %s", e)
                await asyncio.sleep(1)
    
    async def stop_trading(self) -> None:
        """Stop trading."""
        self.is_trading = False
        await self.market_data.stop()
        logger.info("Trading stopped")
    
    def get_trading_stats(self) -> Dict[str, Any]:
        """Get trading statistics."""
        portfolio_returns = self.portfolio.calculate_returns()
        portfolio_risk = self.portfolio.get_risk_metrics()
        execution_stats = self.execution.get_performance_stats()
        
        return {
            "trades": self.trade_count,
            "portfolio": portfolio_returns,
            "risk": portfolio_risk,
            "execution": execution_stats,
            "is_trading": self.is_trading
        }


__all__ = [
    # Market data
    "MarketDataFeed",
    "MarketDataConfig",
    "DataSource",
    
    # Execution
    "ExecutionEngine",
    "ExecutionConfig",
    "Order",
    "OrderType",
    "OrderSide",
    
    # Portfolio
    "PortfolioManager",
    "PortfolioConfig",
    
    # Orchestrator
    "QuantumRLTradingOrchestrator"
]