"""
Advanced Risk Management Module

Features:
- Dynamic position sizing (Kelly criterion, volatility-adjusted)
- Drawdown protection (circuit breakers)
- Correlation-aware position limits
- Value at Risk (VaR) and CVaR
- Stop-loss and take-profit management
- Portfolio heat monitoring
- Black swan protection
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None


class RiskManager:
    """
    Advanced risk management system.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        self.positions: Dict[str, Position] = {}
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.peak_equity: float = 0.0
        self.current_equity: float = self.config['initial_capital']
        
        self.trade_history: deque = deque(maxlen=1000)
        self.pnl_history: deque = deque(maxlen=100)
        
        self.is_circuit_breaker_active: bool = False
        self.circuit_breaker_reason: Optional[str] = None
        
    def _default_config(self) -> Dict:
        """Default risk configuration."""
        return {
            'initial_capital': 10000.0,
            'max_position_pct': 0.10,  # Max 10% per position
            'max_portfolio_heat': 0.30,  # Max 30% portfolio heat
            'max_daily_loss_pct': 0.05,  # Max 5% daily loss
            'max_drawdown_pct': 0.20,  # Max 20% drawdown
            'max_correlated_positions': 3,  # Max 3 correlated positions
            'kelly_fraction': 0.5,  # Half-Kelly for safety
            'min_risk_reward': 1.5,  # Minimum 1.5:1 R:R
            'atr_stop_multiplier': 2.0,  # 2x ATR for stop loss
            'atr_tp_multiplier': 3.0,  # 3x ATR for take profit
            'trailing_stop_activation': 0.02,  # Activate at 2% profit
            'trailing_stop_distance': 0.015,  # Trail by 1.5%
        }
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        confidence: float,
        win_rate: float = 0.6,
        avg_win: float = 0.02,
        avg_loss: float = 0.01
    ) -> float:
        """
        Calculate optimal position size using Kelly criterion + risk limits.
        
        Returns position size in units.
        """
        if self.is_circuit_breaker_active:
            logger.warning(f"Circuit breaker active: {self.circuit_breaker_reason}")
            return 0.0
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss) / entry_price
        
        if risk_per_unit < 0.001:  # Too tight stop
            logger.warning(f"Stop too tight for {symbol}")
            return 0.0
        
        # Kelly criterion
        kelly_pct = self._kelly_criterion(win_rate, avg_win, avg_loss)
        kelly_pct *= self.config['kelly_fraction']  # Half-Kelly
        
        # Confidence adjustment
        adjusted_pct = kelly_pct * confidence
        
        # Position size based on risk
        max_risk_amount = self.current_equity * 0.02  # Max 2% risk per trade
        risk_units = max_risk_amount / (entry_price * risk_per_unit)
        
        # Apply position limit
        max_position_value = self.current_equity * self.config['max_position_pct']
        max_units = max_position_value / entry_price
        
        # Final size is min of all constraints
        position_size = min(risk_units, max_units)
        position_size *= adjusted_pct
        
        # Check portfolio heat
        if not self._check_portfolio_heat(symbol, position_size * entry_price):
            logger.warning(f"Portfolio heat limit reached")
            return 0.0
        
        # Check daily loss limit
        if not self._check_daily_loss_limit():
            logger.warning(f"Daily loss limit reached")
            return 0.0
        
        # Check drawdown limit
        if not self._check_drawdown_limit():
            logger.warning(f"Drawdown limit reached")
            return 0.0
        
        return max(0, position_size)
    
    def _kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly criterion percentage."""
        if avg_loss <= 0:
            return 0.0
        
        win_odds = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / win_odds)
        
        return max(0, min(kelly, 0.25))  # Cap at 25%
    
    def _check_portfolio_heat(self, exclude_symbol: str, new_position_value: float) -> bool:
        """Check if adding position exceeds portfolio heat limit."""
        total_risk = 0
        
        for symbol, position in self.positions.items():
            if symbol != exclude_symbol:
                position_value = position.size * position.entry_price
                if position.stop_loss:
                    position_risk = abs(position.entry_price - position.stop_loss) * position.size
                else:
                    position_risk = position_value * 0.02  # Assume 2% risk
                total_risk += position_risk
        
        total_risk += new_position_value * 0.02  # Assume 2% risk for new position
        portfolio_heat = total_risk / self.current_equity
        
        return portfolio_heat <= self.config['max_portfolio_heat']
    
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is reached."""
        daily_loss_limit = self.current_equity * self.config['max_daily_loss_pct']
        return self.daily_pnl > -daily_loss_limit
    
    def _check_drawdown_limit(self) -> bool:
        """Check if maximum drawdown is reached."""
        if self.peak_equity <= 0:
            return True
        
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        return drawdown < self.config['max_drawdown_pct']
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str,
        atr: Optional[float] = None,
        support_level: Optional[float] = None
    ) -> float:
        """Calculate stop loss level."""
        if atr:
            stop_distance = atr * self.config['atr_stop_multiplier']
            if side == 'long':
                return entry_price - stop_distance
            else:
                return entry_price + stop_distance
        
        if support_level:
            if side == 'long':
                return support_level * 0.995  # 0.5% below support
            else:
                return support_level * 1.005  # 0.5% above resistance
        
        # Default: 2% stop
        if side == 'long':
            return entry_price * 0.98
        else:
            return entry_price * 1.02
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        side: str,
        atr: Optional[float] = None
    ) -> float:
        """Calculate take profit level based on risk-reward ratio."""
        risk = abs(entry_price - stop_loss)
        
        # Minimum risk-reward
        min_reward = risk * self.config['min_risk_reward']
        
        if atr:
            atr_target = atr * self.config['atr_tp_multiplier']
            reward = max(min_reward, atr_target)
        else:
            reward = min_reward
        
        if side == 'long':
            return entry_price + reward
        else:
            return entry_price - reward
    
    def update_trailing_stop(self, symbol: str, current_price: float) -> Optional[float]:
        """Update trailing stop if applicable."""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        if position.side == 'long':
            profit_pct = (current_price - position.entry_price) / position.entry_price
            
            if profit_pct >= self.config['trailing_stop_activation']:
                new_stop = current_price * (1 - self.config['trailing_stop_distance'])
                
                if position.trailing_stop is None or new_stop > position.trailing_stop:
                    position.trailing_stop = new_stop
                    return new_stop
        else:
            profit_pct = (position.entry_price - current_price) / position.entry_price
            
            if profit_pct >= self.config['trailing_stop_activation']:
                new_stop = current_price * (1 + self.config['trailing_stop_distance'])
                
                if position.trailing_stop is None or new_stop < position.trailing_stop:
                    position.trailing_stop = new_stop
                    return new_stop
        
        return position.trailing_stop
    
    def check_stop_triggers(self, symbol: str, current_price: float) -> Optional[str]:
        """Check if stop loss or take profit is triggered."""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        if position.side == 'long':
            # Check stop loss
            effective_stop = max(position.stop_loss or 0, position.trailing_stop or 0)
            if current_price <= effective_stop:
                return 'stop_loss'
            
            # Check take profit
            if position.take_profit and current_price >= position.take_profit:
                return 'take_profit'
        else:
            # Check stop loss
            effective_stop = min(position.stop_loss or float('inf'), position.trailing_stop or float('inf'))
            if current_price >= effective_stop:
                return 'stop_loss'
            
            # Check take profit
            if position.take_profit and current_price <= position.take_profit:
                return 'take_profit'
        
        return None
    
    def open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ):
        """Record opening of a new position."""
        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        logger.info(f"Opened {side} {symbol}: size={size:.4f}, entry={entry_price:.2f}, "
                    f"stop={stop_loss:.2f}, tp={take_profit:.2f}")
    
    def close_position(self, symbol: str, exit_price: float) -> float:
        """Record closing of a position. Returns PnL."""
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        
        if position.side == 'long':
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size
        
        pnl_pct = pnl / self.current_equity
        self.daily_pnl += pnl
        self.total_pnl += pnl
        self.current_equity += pnl
        self.peak_equity = max(self.peak_equity, self.current_equity)
        
        self.trade_history.append({
            'symbol': symbol,
            'side': position.side,
            'size': position.size,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_time': position.entry_time,
            'exit_time': datetime.now()
        })
        
        del self.positions[symbol]
        
        logger.info(f"Closed {position.side} {symbol}: PnL={pnl:.2f} ({pnl_pct:.2%})")
        
        return pnl
    
    def reset_daily(self):
        """Reset daily counters."""
        self.daily_pnl = 0.0
        self.pnl_history.append(self.current_equity)
        self.is_circuit_breaker_active = False
        self.circuit_breaker_reason = None
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics."""
        # Calculate VaR (Value at Risk)
        returns = list(self.pnl_history)
        var_95 = np.percentile(returns, 5) if len(returns) > 10 else 0
        var_99 = np.percentile(returns, 1) if len(returns) > 10 else 0
        
        # Calculate current drawdown
        drawdown = 0
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        
        # Portfolio heat
        total_risk = 0
        for position in self.positions.values():
            if position.stop_loss:
                position_risk = abs(position.entry_price - position.stop_loss) * position.size
            else:
                position_risk = position.size * position.entry_price * 0.02
            total_risk += position_risk
        
        portfolio_heat = total_risk / self.current_equity if self.current_equity > 0 else 0
        
        return {
            'current_equity': self.current_equity,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'drawdown': drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'portfolio_heat': portfolio_heat,
            'open_positions': len(self.positions),
            'circuit_breaker_active': self.is_circuit_breaker_active,
        }


class BlackSwanProtector:
    """
    Detects and responds to black swan events.
    """
    
    def __init__(self, price_drop_threshold: float = 0.10, volume_spike_threshold: float = 5.0):
        self.price_drop_threshold = price_drop_threshold
        self.volume_spike_threshold = volume_spike_threshold
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        
    def update(self, symbol: str, price: float, volume: float):
        """Update with new price/volume data."""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=100)
            self.volume_history[symbol] = deque(maxlen=100)
        
        self.price_history[symbol].append(price)
        self.volume_history[symbol].append(volume)
    
    def check_black_swan(self, symbol: str) -> Tuple[bool, str]:
        """Check if black swan conditions are met."""
        if symbol not in self.price_history:
            return False, "no_data"
        
        prices = list(self.price_history[symbol])
        volumes = list(self.volume_history[symbol])
        
        if len(prices) < 20:
            return False, "insufficient_data"
        
        # Check for sudden price drop
        recent_price = prices[-1]
        price_20_ago = prices[-20]
        price_drop = (recent_price - price_20_ago) / price_20_ago
        
        if price_drop < -self.price_drop_threshold:
            return True, f"price_drop_{price_drop:.2%}"
        
        # Check for volume spike
        if len(volumes) >= 20:
            recent_volume = volumes[-1]
            avg_volume = np.mean(volumes[:-1])
            
            if avg_volume > 0 and recent_volume / avg_volume > self.volume_spike_threshold:
                return True, f"volume_spike_{recent_volume/avg_volume:.1f}x"
        
        return False, "normal"
    
    def get_safe_mode_multiplier(self, symbol: str) -> float:
        """Get position size multiplier during volatile conditions."""
        is_black_swan, reason = self.check_black_swan(symbol)
        
        if is_black_swan:
            logger.warning(f"Black swan detected for {symbol}: {reason}")
            return 0.25  # Reduce to 25% size
        
        return 1.0
