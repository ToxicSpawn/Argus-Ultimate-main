"""
Aggressive Risk Management for Leveraged Trading
=================================================
Protects capital while maximizing returns with 3x leverage.

Features:
- Dynamic position sizing based on volatility
- Automatic stop-loss management
- Drawdown protection circuit breakers
- Leverage scaling based on confidence
- Real-time risk monitoring
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limits for aggressive trading."""
    max_position_pct: float = 0.10      # 10% max per position
    max_leverage: float = 3.0           # 3x max leverage
    max_risk_per_trade: float = 0.05    # 5% risk per trade
    max_daily_loss: float = 0.15        # 15% daily loss limit
    max_drawdown: float = 0.25          # 25% max drawdown
    max_correlation: float = 0.7        # Max correlation between positions
    min_liquidity_ratio: float = 0.1    # Min 10% of order book


@dataclass
class Position:
    """Active position tracking."""
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    size: float
    leverage: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    confidence: float
    
    @property
    def current_pnl_pct(self) -> float:
        """Calculate current P&L percentage."""
        # Would need current price - placeholder
        return 0.0
    
    @property
    def risk_amount(self) -> float:
        """Calculate risk amount."""
        return abs(self.entry_price - self.stop_loss) / self.entry_price * self.size


class AggressiveRiskManager:
    """
    Risk management optimized for aggressive leveraged trading.
    
    Balances protection with profit maximization.
    """
    
    def __init__(
        self,
        initial_capital: float,
        limits: Optional[RiskLimits] = None
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.limits = limits or RiskLimits()
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.position_history: deque = deque(maxlen=1000)
        
        # Risk metrics
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.consecutive_losses: int = 0
        self.last_reset: datetime = datetime.now()
        
        # Circuit breakers
        self.circuit_breakers = {
            "daily_loss": False,
            "drawdown": False,
            "consecutive_losses": False,
            "volatility_spike": False
        }
        
        logger.info(f"AggressiveRiskManager initialized: ${initial_capital}, max leverage {self.limits.max_leverage}x")
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        confidence: float,
        volatility: float
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size using modified Kelly criterion.
        
        Factors in:
        - Account risk limits
        - Volatility adjustment
        - Confidence scaling
        - Current exposure
        """
        # Check circuit breakers
        if any(self.circuit_breakers.values()):
            return {
                "allowed": False,
                "reason": "Circuit breaker active",
                "size": 0,
                "leverage": 1.0
            }
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss) / entry_price
        
        if risk_per_unit <= 0:
            return {
                "allowed": False,
                "reason": "Invalid stop loss",
                "size": 0,
                "leverage": 1.0
            }
        
        # Maximum risk amount
        max_risk = self.current_capital * self.limits.max_risk_per_trade
        
        # Adjust for consecutive losses (reduce size)
        if self.consecutive_losses >= 3:
            max_risk *= 0.5
        elif self.consecutive_losses >= 5:
            max_risk *= 0.25
        
        # Base position size
        base_size = max_risk / risk_per_unit
        
        # Apply confidence scaling (higher confidence = larger position)
        confidence_multiplier = 0.5 + confidence * 0.5  # 0.5x to 1.0x
        adjusted_size = base_size * confidence_multiplier
        
        # Apply volatility scaling (higher vol = smaller position)
        vol_multiplier = min(1.0, 0.02 / volatility) if volatility > 0 else 1.0
        adjusted_size *= vol_multiplier
        
        # Calculate optimal leverage
        optimal_leverage = self._calculate_optimal_leverage(
            confidence, volatility, risk_per_unit
        )
        
        # Apply leverage to size
        leveraged_size = adjusted_size * optimal_leverage
        
        # Cap at maximum position size
        max_position = self.current_capital * self.limits.max_position_pct * self.limits.max_leverage
        final_size = min(leveraged_size, max_position)
        
        # Check total exposure
        current_exposure = self._calculate_total_exposure()
        if current_exposure + final_size > self.current_capital * self.limits.max_leverage:
            final_size = max(0, self.current_capital * self.limits.max_leverage - current_exposure)
        
        return {
            "allowed": final_size > 10,  # Minimum $10 position
            "size": final_size,
            "leverage": optimal_leverage,
            "risk_amount": final_size * risk_per_unit,
            "risk_pct": (final_size * risk_per_unit) / self.current_capital * 100,
            "max_loss": final_size * risk_per_unit,
            "reason": "Approved" if final_size > 10 else "Position too small"
        }
    
    def _calculate_optimal_leverage(
        self,
        confidence: float,
        volatility: float,
        risk_per_unit: float
    ) -> float:
        """Calculate optimal leverage based on conditions."""
        # Base leverage from confidence
        base_leverage = 1.0 + confidence * (self.limits.max_leverage - 1.0)
        
        # Reduce for high volatility
        if volatility > 0.05:  # >5% volatility
            vol_reduction = min(0.5, (volatility - 0.05) * 5)
            base_leverage *= (1 - vol_reduction)
        
        # Reduce for high risk per unit
        if risk_per_unit > 0.03:  # >3% risk
            risk_reduction = min(0.3, (risk_per_unit - 0.03) * 3)
            base_leverage *= (1 - risk_reduction)
        
        return max(1.0, min(self.limits.max_leverage, base_leverage))
    
    def _calculate_total_exposure(self) -> float:
        """Calculate total current exposure."""
        return sum(p.size * p.leverage for p in self.positions.values())
    
    def can_open_position(self, symbol: str) -> Tuple[bool, str]:
        """Check if a new position can be opened."""
        # Check circuit breakers
        if any(self.circuit_breakers.values()):
            active = [k for k, v in self.circuit_breakers.items() if v]
            return False, f"Circuit breakers active: {active}"
        
        # Check daily loss limit
        if self.daily_pnl < -self.current_capital * self.limits.max_daily_loss:
            self.circuit_breakers["daily_loss"] = True
            return False, "Daily loss limit reached"
        
        # Check drawdown
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if drawdown > self.limits.max_drawdown:
            self.circuit_breakers["drawdown"] = True
            return False, f"Max drawdown reached: {drawdown:.1%}"
        
        # Check consecutive losses
        if self.consecutive_losses >= 7:
            self.circuit_breakers["consecutive_losses"] = True
            return False, "Too many consecutive losses"
        
        # Check max positions
        if len(self.positions) >= 10:
            return False, "Maximum positions reached"
        
        # Check if already have position in symbol
        if symbol in self.positions:
            return False, f"Already have position in {symbol}"
        
        return True, "Approved"
    
    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        leverage: float,
        stop_loss: float,
        take_profit: float,
        confidence: float
    ) -> bool:
        """Record a new position."""
        can_open, reason = self.can_open_position(symbol)
        if not can_open:
            logger.warning(f"Cannot open position: {reason}")
            return False
        
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size=size,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now(),
            confidence=confidence
        )
        
        self.positions[symbol] = position
        self.daily_trades += 1
        
        logger.info(f"Position opened: {side} {symbol} at ${entry_price}, size=${size:.2f}, leverage={leverage}x")
        return True
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "manual"
    ) -> Dict[str, Any]:
        """Close a position and record P&L."""
        if symbol not in self.positions:
            return {"success": False, "reason": "Position not found"}
        
        position = self.positions[symbol]
        
        # Calculate P&L
        if position.side == "long":
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - exit_price) / position.entry_price
        
        pnl_pct *= position.leverage  # Apply leverage
        pnl_amount = position.size * pnl_pct
        
        # Update capital
        self.current_capital += pnl_amount
        self.daily_pnl += pnl_amount
        
        # Update peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Track consecutive losses
        if pnl_amount < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Record trade
        trade_record = {
            "symbol": symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "size": position.size,
            "leverage": position.leverage,
            "pnl_pct": pnl_pct * 100,
            "pnl_amount": pnl_amount,
            "reason": reason,
            "duration": (datetime.now() - position.entry_time).total_seconds() / 60
        }
        self.position_history.append(trade_record)
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Position closed: {symbol}, P&L: ${pnl_amount:+.2f} ({pnl_pct*100:+.2f}%)")
        
        return {
            "success": True,
            **trade_record
        }
    
    def check_stops(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check and execute stop losses and take profits."""
        triggered = []
        
        for symbol, position in list(self.positions.items()):
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            
            # Check stop loss
            if position.side == "long":
                if current_price <= position.stop_loss:
                    result = self.close_position(symbol, current_price, "stop_loss")
                    triggered.append(result)
                elif current_price >= position.take_profit:
                    result = self.close_position(symbol, current_price, "take_profit")
                    triggered.append(result)
            else:  # short
                if current_price >= position.stop_loss:
                    result = self.close_position(symbol, current_price, "stop_loss")
                    triggered.append(result)
                elif current_price <= position.take_profit:
                    result = self.close_position(symbol, current_price, "take_profit")
                    triggered.append(result)
        
        return triggered
    
    def reset_daily(self):
        """Reset daily counters."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.circuit_breakers["daily_loss"] = False
        self.last_reset = datetime.now()
        logger.info("Daily risk counters reset")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Get current risk report."""
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0
        daily_loss_pct = -self.daily_pnl / self.current_capital if self.daily_pnl < 0 and self.current_capital > 0 else 0
        
        return {
            "capital": {
                "initial": self.initial_capital,
                "current": self.current_capital,
                "peak": self.peak_capital,
                "pnl": self.current_capital - self.initial_capital,
                "return_pct": (self.current_capital - self.initial_capital) / self.initial_capital * 100
            },
            "risk_metrics": {
                "drawdown": drawdown * 100,
                "daily_pnl": self.daily_pnl,
                "daily_loss_pct": daily_loss_pct * 100,
                "consecutive_losses": self.consecutive_losses,
                "active_positions": len(self.positions),
                "daily_trades": self.daily_trades
            },
            "exposure": {
                "total": self._calculate_total_exposure(),
                "leverage_ratio": self._calculate_total_exposure() / self.current_capital if self.current_capital > 0 else 0,
                "positions": {
                    sym: {
                        "side": p.side,
                        "size": p.size,
                        "leverage": p.leverage,
                        "entry": p.entry_price
                    }
                    for sym, p in self.positions.items()
                }
            },
            "circuit_breakers": self.circuit_breakers,
            "limits": {
                "max_daily_loss": f"{self.limits.max_daily_loss * 100}%",
                "max_drawdown": f"{self.limits.max_drawdown * 100}%",
                "max_leverage": f"{self.limits.max_leverage}x",
                "max_risk_per_trade": f"{self.limits.max_risk_per_trade * 100}%"
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get trading statistics."""
        if not self.position_history:
            return {"trades": 0}
        
        trades = list(self.position_history)
        wins = [t for t in trades if t["pnl_amount"] > 0]
        losses = [t for t in trades if t["pnl_amount"] < 0]
        
        total_pnl = sum(t["pnl_amount"] for t in trades)
        avg_win = sum(t["pnl_amount"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl_amount"] for t in losses) / len(losses) if losses else 0
        
        return {
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(sum(t["pnl_amount"] for t in wins) / sum(t["pnl_amount"] for t in losses)) if losses and sum(t["pnl_amount"] for t in losses) != 0 else 0,
            "avg_duration_minutes": sum(t["duration"] for t in trades) / len(trades) if trades else 0
        }


# ============================================================================
# Quick Start
# ============================================================================

def create_risk_manager(
    capital: float = 1000.0,
    max_leverage: float = 3.0
) -> AggressiveRiskManager:
    """Create an aggressive risk manager."""
    limits = RiskLimits(
        max_position_pct=0.10,
        max_leverage=max_leverage,
        max_risk_per_trade=0.05,
        max_daily_loss=0.15,
        max_drawdown=0.25
    )
    return AggressiveRiskManager(capital, limits)


if __name__ == "__main__":
    # Demo
    risk_mgr = create_risk_manager(capital=1000, max_leverage=3)
    
    # Calculate position size
    result = risk_mgr.calculate_position_size(
        symbol="BTCUSDT",
        entry_price=85000,
        stop_loss=83500,
        confidence=0.8,
        volatility=0.03
    )
    
    print("Position Sizing Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    print("\nRisk Report:")
    report = risk_mgr.get_risk_report()
    print(f"  Capital: ${report['capital']['current']:.2f}")
    print(f"  Max Daily Loss: ${report['capital']['current'] * 0.15:.2f}")
    print(f"  Max Drawdown: ${report['capital']['current'] * 0.25:.2f}")
