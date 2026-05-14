"""
Circuit Breaker System - Advanced Safety & Risk Controls
=========================================================

Emergency stop mechanisms and risk controls to protect capital.
Automatically intervenes during dangerous market conditions.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """States of circuit breaker."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Trading halted
    HALF_OPEN = "half_open"  # Testing if safe to resume


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    # Consecutive losses
    max_consecutive_losses: int = 5
    
    # Drawdown limits
    max_daily_drawdown_pct: float = 0.15  # 15%
    max_position_drawdown_pct: float = 0.10  # 10% per position
    
    # Volatility limits
    max_volatility_spike: float = 2.0  # 2x normal volatility
    
    # Error limits
    max_api_errors_per_minute: int = 10
    max_slippage_bps: float = 50.0  # 50 bps max slippage
    
    # Cooldown periods
    open_duration_seconds: float = 300.0  # 5 minutes
    half_open_test_trades: int = 3
    
    # Auto-recovery
    auto_reset_after_volatility_normalizes: bool = True


@dataclass
class CircuitBreakerEvent:
    """Record of circuit breaker activation."""
    timestamp: datetime
    state: CircuitBreakerState
    reason: str
    details: Dict
    recovered_at: Optional[datetime] = None


class CircuitBreaker:
    """
    Advanced circuit breaker for trading safety.
    
    Monitors multiple risk factors and halts trading when necessary.
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        
        # Risk tracking
        self.consecutive_losses = 0
        self.daily_drawdown = 0.0
        self.peak_portfolio_value = 0.0
        self.api_errors = deque(maxlen=100)
        self.slippage_readings = deque(maxlen=50)
        
        # Event history
        self.events: List[CircuitBreakerEvent] = []
        self.last_open_time: Optional[float] = None
        self.half_open_trade_count = 0
        
        # Callbacks
        self.on_open: Optional[Callable] = None
        self.on_close: Optional[Callable] = None
        
        logger.info("Circuit Breaker initialized")
    
    def check(self, portfolio_value: float, pnl: float = 0,
              volatility: float = 0.0, api_error: bool = False,
              slippage_bps: float = 0.0) -> bool:
        """
        Check if trading should be allowed.
        
        Returns:
            True if trading allowed, False if halted
        """
        current_time = time.time()
        
        # Update tracking
        self._update_tracking(portfolio_value, pnl, api_error, slippage_bps)
        
        # State machine
        if self.state == CircuitBreakerState.OPEN:
            return self._handle_open_state(current_time)
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return self._handle_half_open_state(pnl)
        
        else:  # CLOSED
            return self._check_triggers(current_time, portfolio_value, 
                                        pnl, volatility, slippage_bps)
    
    def _update_tracking(self, portfolio_value: float, pnl: float,
                        api_error: bool, slippage_bps: float):
        """Update risk tracking metrics."""
        # Track peak value
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
        
        # Track losses
        if pnl < 0:
            self.consecutive_losses += 1
            self.daily_drawdown += abs(pnl) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0
        else:
            self.consecutive_losses = 0
        
        # Track API errors
        if api_error:
            self.api_errors.append(time.time())
        
        # Track slippage
        if slippage_bps > 0:
            self.slippage_readings.append(slippage_bps)
    
    def _check_triggers(self, current_time: float, portfolio_value: float,
                       pnl: float, volatility: float, 
                       slippage_bps: float) -> bool:
        """Check if any circuit breaker should trigger."""
        
        # 1. Consecutive losses
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            self._trip("Consecutive losses exceeded",
                      {'losses': self.consecutive_losses})
            return False
        
        # 2. Daily drawdown
        if self.daily_drawdown >= self.config.max_daily_drawdown_pct:
            self._trip("Daily drawdown limit reached",
                      {'drawdown': f"{self.daily_drawdown:.1%}"})
            return False
        
        # 3. API errors
        recent_errors = sum(1 for t in self.api_errors 
                          if current_time - t < 60)
        if recent_errors >= self.config.max_api_errors_per_minute:
            self._trip("Too many API errors",
                      {'errors_last_minute': recent_errors})
            return False
        
        # 4. Slippage
        if slippage_bps > self.config.max_slippage_bps:
            self._trip("Excessive slippage detected",
                      {'slippage_bps': slippage_bps})
            return False
        
        return True
    
    def _handle_open_state(self, current_time: float) -> bool:
        """Handle when circuit is open."""
        if self.last_open_time is None:
            return False
        
        # Check if cooldown period passed
        elapsed = current_time - self.last_open_time
        
        if elapsed >= self.config.open_duration_seconds:
            # Move to half-open to test
            self.state = CircuitBreakerState.HALF_OPEN
            self.half_open_trade_count = 0
            logger.info("Circuit breaker entering HALF_OPEN state - testing...")
            return True
        
        return False
    
    def _handle_half_open_state(self, pnl: float) -> bool:
        """Handle half-open state (testing)."""
        self.half_open_trade_count += 1
        
        # If profitable in half-open, close circuit
        if pnl > 0:
            self._close("Successful test trades")
            return True
        
        # If too many test trades, re-open
        if self.half_open_trade_count >= self.config.half_open_test_trades:
            if pnl <= 0:
                self._trip("Failed test trades in half-open", {})
                return False
        
        return True
    
    def _trip(self, reason: str, details: Dict):
        """Trip the circuit breaker (open)."""
        self.state = CircuitBreakerState.OPEN
        self.last_open_time = time.time()
        
        event = CircuitBreakerEvent(
            timestamp=datetime.utcnow(),
            state=CircuitBreakerState.OPEN,
            reason=reason,
            details=details
        )
        self.events.append(event)
        
        logger.warning(f"🚨 CIRCUIT BREAKER TRIPPED: {reason}")
        logger.warning(f"Details: {details}")
        
        # Execute callback
        if self.on_open:
            try:
                self.on_open(reason, details)
            except Exception as e:
                logger.error(f"Circuit breaker open callback error: {e}")
    
    def _close(self, reason: str):
        """Close the circuit breaker (resume normal operation)."""
        self.state = CircuitBreakerState.CLOSED
        self.consecutive_losses = 0
        self.daily_drawdown = 0.0
        self.half_open_trade_count = 0
        
        # Update last event
        if self.events:
            self.events[-1].recovered_at = datetime.utcnow()
        
        logger.info(f"✅ CIRCUIT BREAKER CLOSED: {reason}")
        
        # Execute callback
        if self.on_close:
            try:
                self.on_close(reason)
            except Exception as e:
                logger.error(f"Circuit breaker close callback error: {e}")
    
    def manual_reset(self):
        """Manually reset circuit breaker."""
        self._close("Manual reset")
    
    def get_status(self) -> Dict:
        """Get current circuit breaker status."""
        return {
            'state': self.state.value,
            'consecutive_losses': self.consecutive_losses,
            'daily_drawdown_pct': self.daily_drawdown,
            'peak_value': self.peak_portfolio_value,
            'api_errors_last_minute': len([t for t in self.api_errors 
                                          if time.time() - t < 60]),
            'avg_slippage_bps': sum(self.slippage_readings) / len(self.slippage_readings) 
                              if self.slippage_readings else 0,
            'event_count': len(self.events),
            'last_event': self.events[-1] if self.events else None
        }


class KillSwitch:
    """
    Emergency kill switch for immediate trading halt.
    Can be triggered manually or automatically.
    """
    
    def __init__(self):
        self.armed = True
        self.triggered = False
        self.trigger_time: Optional[float] = None
        self.trigger_reason: Optional[str] = None
        self.on_trigger: Optional[Callable] = None
        
        logger.info("Kill Switch armed and ready")
    
    def trigger(self, reason: str = "Emergency stop"):
        """Trigger kill switch."""
        if not self.armed:
            logger.warning("Kill switch trigger ignored - not armed")
            return
        
        self.triggered = True
        self.trigger_time = time.time()
        self.trigger_reason = reason
        
        logger.critical(f"💀 KILL SWITCH TRIGGERED: {reason}")
        logger.critical("All trading halted immediately!")
        
        # Execute callback
        if self.on_trigger:
            try:
                self.on_trigger(reason)
            except Exception as e:
                logger.error(f"Kill switch callback error: {e}")
    
    def disarm(self):
        """Disarm kill switch (use with caution)."""
        self.armed = False
        logger.warning("Kill switch DISARMED")
    
    def rearm(self):
        """Rearm kill switch."""
        self.armed = True
        self.triggered = False
        logger.info("Kill switch REARMED")
    
    def check(self) -> bool:
        """Check if trading should be allowed."""
        if self.triggered and self.armed:
            return False
        return True
    
    def get_status(self) -> Dict:
        """Get kill switch status."""
        return {
            'armed': self.armed,
            'triggered': self.triggered,
            'trigger_time': datetime.fromtimestamp(self.trigger_time) if self.trigger_time else None,
            'trigger_reason': self.trigger_reason
        }


class PositionGuard:
    """
    Guards individual positions against excessive losses.
    Per-position circuit breaker.
    """
    
    def __init__(self, max_drawdown_pct: float = 0.10,
                 max_position_size_pct: float = 0.20):
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_size_pct = max_position_size_pct
        self.positions: Dict[str, Dict] = {}
        
        logger.info(f"Position Guard initialized (max_dd={max_drawdown_pct:.1%})")
    
    def register_position(self, symbol: str, entry_price: float,
                         size: float, side: str):
        """Register new position for monitoring."""
        self.positions[symbol] = {
            'entry_price': entry_price,
            'size': size,
            'side': side,
            'peak_value': entry_price * size,
            'current_value': entry_price * size,
            'drawdown': 0.0
        }
    
    def update_position(self, symbol: str, current_price: float):
        """Update position with current price."""
        if symbol not in self.positions:
            return {'action': 'none'}
        
        pos = self.positions[symbol]
        current_value = current_price * pos['size']
        
        # Track peak
        if current_value > pos['peak_value']:
            pos['peak_value'] = current_value
        
        # Calculate drawdown
        pos['current_value'] = current_value
        pos['drawdown'] = (pos['peak_value'] - current_value) / pos['peak_value']
        
        # Check limits
        if pos['drawdown'] >= self.max_drawdown_pct:
            logger.warning(f"Position {symbol} hit max drawdown: {pos['drawdown']:.1%}")
            return {
                'action': 'liquidate',
                'reason': 'max_drawdown',
                'drawdown': pos['drawdown']
            }
        
        elif pos['drawdown'] >= self.max_drawdown_pct * 0.7:
            return {
                'action': 'warning',
                'reason': 'approaching_max_drawdown',
                'drawdown': pos['drawdown']
            }
        
        return {'action': 'none'}
    
    def close_position(self, symbol: str):
        """Close position and remove from monitoring."""
        if symbol in self.positions:
            del self.positions[symbol]


# Global safety system
class SafetySystem:
    """Master safety controller integrating all protections."""
    
    def __init__(self):
        self.circuit_breaker = CircuitBreaker()
        self.kill_switch = KillSwitch()
        self.position_guard = PositionGuard()
        
        # Set up callbacks
        self.circuit_breaker.on_open = self._on_circuit_open
        self.circuit_breaker.on_close = self._on_circuit_close
        self.kill_switch.on_trigger = self._on_kill_trigger
        
        logger.info("Safety System initialized with full protections")
    
    def check_all(self, portfolio_value: float = 0, pnl: float = 0,
                  symbol: Optional[str] = None, 
                  current_price: Optional[float] = None) -> Dict:
        """
        Check all safety systems.
        
        Returns:
            Dict with 'allow_trading' and 'actions'
        """
        result = {
            'allow_trading': True,
            'actions': [],
            'warnings': []
        }
        
        # Check kill switch
        if not self.kill_switch.check():
            result['allow_trading'] = False
            result['actions'].append('kill_switch_active')
            return result
        
        # Check circuit breaker
        if not self.circuit_breaker.check(portfolio_value, pnl):
            result['allow_trading'] = False
            result['actions'].append('circuit_breaker_open')
            return result
        
        # Check position guard
        if symbol and current_price:
            guard_result = self.position_guard.update_position(symbol, current_price)
            if guard_result['action'] == 'liquidate':
                result['actions'].append(f'liquidate_{symbol}')
                result['warnings'].append(f"Position {symbol} drawdown: {guard_result['drawdown']:.1%}")
            elif guard_result['action'] == 'warning':
                result['warnings'].append(f"Position {symbol} approaching max drawdown")
        
        return result
    
    def _on_circuit_open(self, reason: str, details: Dict):
        """Callback when circuit opens."""
        logger.critical(f"Trading halted: {reason}")
        # Could send SMS/email here
    
    def _on_circuit_close(self, reason: str):
        """Callback when circuit closes."""
        logger.info(f"Trading resumed: {reason}")
    
    def _on_kill_trigger(self, reason: str):
        """Callback when kill switch triggered."""
        logger.critical("EMERGENCY STOP - All positions must be closed!")
        # Could force close all positions here
    
    def emergency_stop(self, reason: str = "Manual emergency stop"):
        """Manual emergency stop."""
        self.kill_switch.trigger(reason)
    
    def get_full_status(self) -> Dict:
        """Get status of all safety systems."""
        return {
            'circuit_breaker': self.circuit_breaker.get_status(),
            'kill_switch': self.kill_switch.get_status(),
            'position_guard': {
                'monitored_positions': len(self.position_guard.positions),
                'max_drawdown_setting': self.position_guard.max_drawdown_pct
            }
        }


# Singleton
_safety_system: Optional[SafetySystem] = None


def get_safety_system() -> SafetySystem:
    """Get global safety system."""
    global _safety_system
    if _safety_system is None:
        _safety_system = SafetySystem()
    return _safety_system
