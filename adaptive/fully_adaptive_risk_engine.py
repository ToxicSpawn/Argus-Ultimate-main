"""
Fully Adaptive Risk Engine for Argus Ultimate.

Integrates all adaptive risk components into a unified system:
- Volatility-adjusted position sizing (existing AdaptivePositionSizer)
- Correlation-aware portfolio sizing (existing CorrelationMonitor)
- Dynamic trailing stops (NEW)
- Performance-responsive risk limits (NEW)
- Time-based trading pauses after losses (NEW)
- Regime-adaptive risk budgets (NEW)

This engine orchestrates all adaptive components and provides a single
interface for the trading system to get fully adaptive risk decisions.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Dataclasses
# ---------------------------------------------------------------------------

class RiskState(Enum):
    """Current risk state of the system."""
    NORMAL = "normal"
    CAUTIOUS = "cautious"        # After 2+ losses
    DEFENSIVE = "defensive"      # After 3+ losses or 5%+ drawdown
    PAUSED = "paused"            # Trading paused after severe losses
    RECOVERY = "recovery"        # Gradually returning to normal


class TrailingStopType(Enum):
    """Types of trailing stops."""
    ATR_BASED = "atr_based"      # Based on ATR multiples
    PERCENTAGE = "percentage"    # Fixed percentage trail
    STRUCTURE = "structure"      # Based on market structure (swing highs/lows)
    CHANDELIER = "chandelier"    # Chandelier exit (highest high - ATR)


@dataclass
class PerformanceMetrics:
    """Rolling performance metrics for adaptive adjustments."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    total_pnl_pct: float = 0.0
    rolling_pnl_10: deque = field(default_factory=lambda: deque(maxlen=10))
    rolling_pnl_30: deque = field(default_factory=lambda: deque(maxlen=30))
    rolling_pnl_100: deque = field(default_factory=lambda: deque(maxlen=100))
    last_trade_time: float = 0.0
    last_win_time: float = 0.0
    last_loss_time: float = 0.0
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.5
        return self.winning_trades / self.total_trades
    
    @property
    def avg_pnl_10(self) -> float:
        if len(self.rolling_pnl_10) == 0:
            return 0.0
        return float(np.mean(self.rolling_pnl_10))
    
    @property
    def avg_pnl_30(self) -> float:
        if len(self.rolling_pnl_30) == 0:
            return 0.0
        return float(np.mean(self.rolling_pnl_30))
    
    @property
    def sharpe_30(self) -> float:
        if len(self.rolling_pnl_30) < 2:
            return 0.0
        arr = np.array(self.rolling_pnl_30)
        std = np.std(arr)
        if std <= 0:
            return 0.0
        return float(np.mean(arr) / std)


@dataclass
class TrailingStopState:
    """State of a trailing stop for a position."""
    symbol: str
    entry_price: float
    current_stop: float
    highest_price: float      # For longs
    lowest_price: float       # For shorts
    stop_type: TrailingStopType
    atr_value: float = 0.0
    trail_distance: float = 0.0
    breakeven_triggered: bool = False
    partial_taken: bool = False
    last_updated: float = 0.0


@dataclass
class AdaptiveRiskConfig:
    """Configuration for the fully adaptive risk engine."""
    # Base position sizing
    base_position_pct: float = 0.10           # 10% base position
    min_position_pct: float = 0.02            # 2% minimum
    max_position_pct: float = 0.20            # 20% maximum
    
    # Volatility adjustment
    volatility_target: float = 0.15           # 15% annualized vol target
    volatility_lookback: int = 20             # 20 bars for vol calculation
    
    # Performance-responsive limits
    max_daily_loss_pct: float = 0.10          # 10% max daily loss
    cautious_after_losses: int = 2            # Become cautious after 2 losses
    defensive_after_losses: int = 3           # Become defensive after 3 losses
    pause_after_losses: int = 5               # Pause after 5 consecutive losses
    pause_duration_minutes: int = 60          # Pause for 1 hour
    
    # Drawdown response
    drawdown_cautious_pct: float = 0.05       # 5% DD → cautious
    drawdown_defensive_pct: float = 0.10      # 10% DD → defensive
    drawdown_pause_pct: float = 0.15          # 15% DD → pause
    
    # Recovery
    recovery_trades_needed: int = 3           # 3 wins to recover from pause
    recovery_position_scale: float = 0.5      # Start at 50% size in recovery
    
    # Trailing stops
    trailing_stop_enabled: bool = True
    trailing_stop_type: TrailingStopType = TrailingStopType.ATR_BASED
    trailing_atr_multiplier: float = 2.5      # Trail at 2.5x ATR
    breakeven_trigger_pct: float = 0.02       # Move to breakeven at 2% profit
    partial_tp_enabled: bool = True
    partial_tp_pct: float = 0.50              # Take 50% at 2x risk
    partial_tp_at_2r: bool = True             # Take partial at 2R
    
    # Correlation
    correlation_reduce_threshold: float = 0.75  # Reduce when avg corr > 0.75
    correlation_crisis_threshold: float = 0.90  # Crisis mode at 0.90
    correlation_position_reduction: float = 0.5  # Reduce by 50% when correlated


# ---------------------------------------------------------------------------
# Performance Tracker
# ---------------------------------------------------------------------------

class PerformanceTracker:
    """Tracks trading performance for adaptive adjustments."""
    
    def __init__(self, config: AdaptiveRiskConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
        self._trade_history: List[Dict] = []
    
    def record_trade(self, pnl_pct: float, symbol: str) -> None:
        """Record a completed trade."""
        now = time.time()
        self.metrics.total_trades += 1
        self.metrics.last_trade_time = now
        
        if pnl_pct >= 0:
            self.metrics.winning_trades += 1
            self.metrics.consecutive_wins += 1
            self.metrics.consecutive_losses = 0
            self.metrics.last_win_time = now
        else:
            self.metrics.losing_trades += 1
            self.metrics.consecutive_losses += 1
            self.metrics.consecutive_wins = 0
            self.metrics.last_loss_time = now
        
        self.metrics.total_pnl_pct += pnl_pct
        self.metrics.rolling_pnl_10.append(pnl_pct)
        self.metrics.rolling_pnl_30.append(pnl_pct)
        self.metrics.rolling_pnl_100.append(pnl_pct)
        
        self._trade_history.append({
            "timestamp": now,
            "symbol": symbol,
            "pnl_pct": pnl_pct,
            "consecutive_losses": self.metrics.consecutive_losses,
            "consecutive_wins": self.metrics.consecutive_wins,
        })
    
    def get_risk_state(self) -> RiskState:
        """Determine current risk state based on performance."""
        cl = self.metrics.consecutive_losses
        
        # Check pause conditions first
        if cl >= self.config.pause_after_losses:
            return RiskState.PAUSED
        
        # Check if we're in recovery (recently paused)
        if cl == 0 and self.metrics.consecutive_wins >= 1:
            # Check if we were recently in a bad state
            if len(self._trade_history) > 0:
                recent_losses = sum(1 for t in self._trade_history[-10:] if t["pnl_pct"] < 0)
                if recent_losses >= 3:
                    return RiskState.RECOVERY
        
        # Check defensive conditions
        if cl >= self.config.defensive_after_losses:
            return RiskState.DEFENSIVE
        
        # Check cautious conditions
        if cl >= self.config.cautious_after_losses:
            return RiskState.CAUTIOUS
        
        # Check recent performance
        if len(self.metrics.rolling_pnl_10) >= 5:
            if self.metrics.avg_pnl_10 < -1.0:  # Losing >1% avg over last 10
                return RiskState.CAUTIOUS
        
        return RiskState.NORMAL
    
    def get_risk_multiplier(self) -> float:
        """Get position size multiplier based on risk state."""
        state = self.get_risk_state()
        multipliers = {
            RiskState.NORMAL: 1.0,
            RiskState.CAUTIOUS: 0.7,
            RiskState.DEFENSIVE: 0.4,
            RiskState.PAUSED: 0.0,
            RiskState.RECOVERY: 0.5,
        }
        return multipliers.get(state, 1.0)
    
    def get_confidence_multiplier(self) -> float:
        """Adjust confidence based on recent performance."""
        sharpe = self.metrics.sharpe_30
        if sharpe > 1.0:
            return 1.15      # Hot streak - boost confidence
        elif sharpe > 0.5:
            return 1.05      # Good performance
        elif sharpe > 0:
            return 1.0       # Neutral
        elif sharpe > -0.5:
            return 0.9       # Slight underperformance
        else:
            return 0.75      # Poor performance - reduce


# ---------------------------------------------------------------------------
# Dynamic Trailing Stop Manager
# ---------------------------------------------------------------------------

class TrailingStopManager:
    """Manages dynamic trailing stops for all positions."""
    
    def __init__(self, config: AdaptiveRiskConfig):
        self.config = config
        self._positions: Dict[str, TrailingStopState] = {}
        self._atr_cache: Dict[str, float] = {}
    
    def open_position(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        atr: float = 0.0,
    ) -> TrailingStopState:
        """Initialize trailing stop for a new position."""
        if side.lower() == "long":
            initial_stop = entry_price - (atr * self.config.trailing_atr_multiplier) if atr > 0 else entry_price * 0.97
            state = TrailingStopState(
                symbol=symbol,
                entry_price=entry_price,
                current_stop=initial_stop,
                highest_price=entry_price,
                lowest_price=entry_price,
                stop_type=self.config.trailing_stop_type,
                atr_value=atr,
                trail_distance=atr * self.config.trailing_atr_multiplier if atr > 0 else entry_price * 0.03,
                last_updated=time.time(),
            )
        else:  # short
            initial_stop = entry_price + (atr * self.config.trailing_atr_multiplier) if atr > 0 else entry_price * 1.03
            state = TrailingStopState(
                symbol=symbol,
                entry_price=entry_price,
                current_stop=initial_stop,
                highest_price=entry_price,
                lowest_price=entry_price,
                stop_type=self.config.trailing_stop_type,
                atr_value=atr,
                trail_distance=atr * self.config.trailing_atr_multiplier if atr > 0 else entry_price * 0.03,
                last_updated=time.time(),
            )
        
        self._positions[symbol] = state
        logger.info(
            "Trailing stop opened: %s entry=%.2f stop=%.2f atr=%.2f",
            symbol, entry_price, state.current_stop, atr,
        )
        return state
    
    def update_price(self, symbol: str, price: float, atr: float = 0.0) -> Optional[Dict[str, Any]]:
        """Update position with new price, return action if stop hit."""
        if symbol not in self._positions:
            return None
        
        state = self._positions[symbol]
        actions = {"close": False, "partial_close": False, "move_to_breakeven": False, "new_stop": None}
        
        # Update ATR if provided
        if atr > 0:
            state.atr_value = atr
            state.trail_distance = atr * self.config.trailing_atr_multiplier
        
        # Determine if long or short based on entry vs current
        is_long = state.current_stop < state.entry_price
        
        if is_long:
            # Update highest price
            if price > state.highest_price:
                state.highest_price = price
            
            # Calculate new stop based on type
            if state.stop_type == TrailingStopType.ATR_BASED:
                new_stop = state.highest_price - state.trail_distance
            elif state.stop_type == TrailingStopType.PERCENTAGE:
                new_stop = state.highest_price * 0.97  # 3% trail
            elif state.stop_type == TrailingStopType.CHANDELIER:
                new_stop = state.highest_price - (state.atr_value * 3.0)
            else:
                new_stop = state.highest_price - state.trail_distance
            
            # Only move stop UP, never down
            if new_stop > state.current_stop:
                state.current_stop = new_stop
                actions["new_stop"] = new_stop
            
            # Check breakeven trigger
            profit_pct = (price - state.entry_price) / state.entry_price
            if not state.breakeven_triggered and profit_pct >= self.config.breakeven_trigger_pct:
                if state.current_stop < state.entry_price:
                    state.current_stop = state.entry_price
                    state.breakeven_triggered = True
                    actions["move_to_breakeven"] = True
                    logger.info("%s: Moved stop to breakeven at %.2f%% profit", symbol, profit_pct * 100)
            
            # Check partial take profit
            if self.config.partial_tp_enabled and not state.partial_taken:
                risk = state.entry_price - (self._positions[symbol].entry_price - state.trail_distance)
                reward = price - state.entry_price
                if risk > 0 and reward >= risk * 2.0:  # 2R achieved
                    state.partial_taken = True
                    actions["partial_close"] = True
                    logger.info("%s: Partial TP triggered at 2R (price=%.2f)", symbol, price)
            
            # Check stop hit
            if price <= state.current_stop:
                actions["close"] = True
                logger.info("%s: Trailing stop hit at %.2f (highest=%.2f)", symbol, price, state.highest_price)
        
        else:  # short
            if price < state.lowest_price:
                state.lowest_price = price
            
            if state.stop_type == TrailingStopType.ATR_BASED:
                new_stop = state.lowest_price + state.trail_distance
            elif state.stop_type == TrailingStopType.PERCENTAGE:
                new_stop = state.lowest_price * 1.03
            else:
                new_stop = state.lowest_price + state.trail_distance
            
            if new_stop < state.current_stop:
                state.current_stop = new_stop
                actions["new_stop"] = new_stop
            
            profit_pct = (state.entry_price - price) / state.entry_price
            if not state.breakeven_triggered and profit_pct >= self.config.breakeven_trigger_pct:
                if state.current_stop > state.entry_price:
                    state.current_stop = state.entry_price
                    state.breakeven_triggered = True
                    actions["move_to_breakeven"] = True
            
            if price >= state.current_stop:
                actions["close"] = True
        
        state.last_updated = time.time()
        return actions
    
    def close_position(self, symbol: str) -> None:
        """Remove trailing stop for closed position."""
        if symbol in self._positions:
            del self._positions[symbol]
    
    def get_stop_price(self, symbol: str) -> Optional[float]:
        """Get current stop price for a symbol."""
        if symbol in self._positions:
            return self._positions[symbol].current_stop
        return None
    
    def get_all_stops(self) -> Dict[str, float]:
        """Get all current stop prices."""
        return {sym: state.current_stop for sym, state in self._positions.items()}


# ---------------------------------------------------------------------------
# Fully Adaptive Risk Engine
# ---------------------------------------------------------------------------

class FullyAdaptiveRiskEngine:
    """
    Unified adaptive risk engine that orchestrates all risk components.
    
    This is the main interface for the trading system to get fully adaptive
    risk decisions including position sizing, stops, and risk limits.
    """
    
    def __init__(self, config: Optional[AdaptiveRiskConfig] = None):
        self.config = config or AdaptiveRiskConfig()
        
        # Sub-components
        self.performance_tracker = PerformanceTracker(self.config)
        self.trailing_stop_manager = TrailingStopManager(self.config)
        
        # State
        self.risk_state = RiskState.NORMAL
        self.daily_pnl_pct = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.current_drawdown_pct = 0.0
        
        # Correlation tracking
        self._correlation_values: Dict[str, float] = {}
        self._avg_correlation: float = 0.0
        
        # Pause tracking
        self._pause_end_time: float = 0.0
        self._recovery_wins_needed: int = 0
        
        # Volatility tracking
        self._volatility_cache: Dict[str, float] = {}
        
        logger.info(
            "FullyAdaptiveRiskEngine initialized: base_pos=%.0f%%, max_dd=%.0f%%, "
            "trailing_stops=%s, performance_responsive=%s",
            self.config.base_position_pct * 100,
            self.config.drawdown_pause_pct * 100,
            self.config.trailing_stop_enabled,
            True,
        )
    
    def update_equity(self, equity: float) -> None:
        """Update current equity and drawdown."""
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        if self.peak_equity > 0:
            self.current_drawdown_pct = (self.peak_equity - equity) / self.peak_equity
    
    def update_daily_pnl(self, pnl_pct: float) -> None:
        """Update daily PnL percentage."""
        self.daily_pnl_pct = pnl_pct
    
    def update_correlation(self, symbol: str, correlation: float) -> None:
        """Update correlation value for a symbol."""
        self._correlation_values[symbol] = correlation
        if self._correlation_values:
            self._avg_correlation = float(np.mean(list(self._correlation_values.values())))
    
    def update_volatility(self, symbol: str, volatility: float) -> None:
        """Update volatility for a symbol."""
        self._volatility_cache[symbol] = volatility
    
    def record_trade(self, pnl_pct: float, symbol: str) -> None:
        """Record a completed trade for performance tracking."""
        self.performance_tracker.record_trade(pnl_pct, symbol)
        
        # Update risk state
        self.risk_state = self.performance_tracker.get_risk_state()
        
        # Handle pause state
        if self.risk_state == RiskState.PAUSED:
            self._pause_end_time = time.time() + (self.config.pause_duration_minutes * 60)
            self._recovery_wins_needed = self.config.recovery_trades_needed
            logger.warning(
                "TRADING PAUSED: %d consecutive losses. Resuming at %s",
                self.performance_tracker.metrics.consecutive_losses,
                datetime.fromtimestamp(self._pause_end_time).strftime("%H:%M:%S"),
            )
        
        # Handle recovery
        if self.risk_state == RiskState.RECOVERY:
            self._recovery_wins_needed -= 1
            if self._recovery_wins_needed <= 0:
                self.risk_state = RiskState.NORMAL
                logger.info("Recovery complete - returning to NORMAL risk state")
    
    def is_trading_allowed(self) -> Tuple[bool, str]:
        """Check if trading is currently allowed."""
        now = time.time()
        
        # Check pause
        if self.risk_state == RiskState.PAUSED:
            if now < self._pause_end_time:
                remaining = int(self._pause_end_time - now)
                return False, f"Trading paused: {remaining}s remaining (consecutive losses)"
            else:
                # Pause expired, move to recovery
                self.risk_state = RiskState.RECOVERY
                logger.info("Pause expired - entering RECOVERY mode")
        
        # Check daily loss limit
        if self.daily_pnl_pct <= -self.config.max_daily_loss_pct:
            return False, f"Daily loss limit hit: {self.daily_pnl_pct:.2f}%"
        
        # Check drawdown pause
        if self.current_drawdown_pct >= self.config.drawdown_pause_pct:
            self.risk_state = RiskState.PAUSED
            self._pause_end_time = now + (self.config.pause_duration_minutes * 60)
            return False, f"Drawdown pause: {self.current_drawdown_pct:.2f}% drawdown"
        
        return True, f"Trading allowed (state={self.risk_state.value})"
    
    def compute_position_size(
        self,
        symbol: str,
        base_size_pct: float,
        confidence: float,
        entry_price: float,
    ) -> Dict[str, Any]:
        """
        Compute fully adaptive position size.
        
        Returns dict with:
        - size_pct: Final position size as percentage
        - size_aud: Position size in AUD
        - multipliers: Dict of all multipliers applied
        - reason: Explanation of sizing
        """
        # Get base configuration
        base_pct = base_size_pct or self.config.base_position_pct
        
        # Check if trading allowed
        allowed, reason = self.is_trading_allowed()
        if not allowed:
            return {
                "size_pct": 0.0,
                "size_aud": 0.0,
                "multipliers": {"trading_allowed": 0.0},
                "reason": reason,
            }
        
        multipliers = {}
        
        # 1. Risk state multiplier
        risk_mult = self.performance_tracker.get_risk_multiplier()
        multipliers["risk_state"] = risk_mult
        
        # 2. Volatility adjustment
        vol_mult = self._get_volatility_multiplier(symbol)
        multipliers["volatility"] = vol_mult
        
        # 3. Correlation adjustment
        corr_mult = self._get_correlation_multiplier()
        multipliers["correlation"] = corr_mult
        
        # 4. Drawdown adjustment
        dd_mult = self._get_drawdown_multiplier()
        multipliers["drawdown"] = dd_mult
        
        # 5. Confidence adjustment
        conf_mult = self.performance_tracker.get_confidence_multiplier() * (0.5 + confidence * 0.5)
        multipliers["confidence"] = conf_mult
        
        # 6. Recovery scaling
        recovery_mult = self.config.recovery_position_scale if self.risk_state == RiskState.RECOVERY else 1.0
        multipliers["recovery"] = recovery_mult
        
        # Calculate final multiplier
        final_mult = risk_mult * vol_mult * corr_mult * dd_mult * conf_mult * recovery_mult
        multipliers["final"] = final_mult
        
        # Calculate final size
        final_pct = base_pct * final_mult
        
        # Apply min/max bounds
        final_pct = max(self.config.min_position_pct, min(self.config.max_position_pct, final_pct))
        
        # Calculate AUD size (assuming equity)
        size_aud = self.current_equity * final_pct if self.current_equity > 0 else 1000 * final_pct
        
        return {
            "size_pct": final_pct,
            "size_aud": size_aud,
            "multipliers": multipliers,
            "reason": f"state={self.risk_state.value}, mult={final_mult:.3f}",
        }
    
    def _get_volatility_multiplier(self, symbol: str) -> float:
        """Get volatility-based position size multiplier."""
        vol = self._volatility_cache.get(symbol, 0.15)  # Default 15%
        target_vol = self.config.volatility_target
        
        if vol <= 0 or target_vol <= 0:
            return 1.0
        
        # Inverse relationship: high vol → smaller position
        ratio = target_vol / vol
        return float(np.clip(ratio, 0.25, 2.0))
    
    def _get_correlation_multiplier(self) -> float:
        """Get correlation-based position size multiplier."""
        avg_corr = self._avg_correlation
        
        if avg_corr >= self.config.correlation_crisis_threshold:
            return 0.1  # Crisis mode - 90% reduction
        elif avg_corr >= self.config.correlation_reduce_threshold:
            return self.config.correlation_position_reduction  # 50% reduction
        elif avg_corr >= 0.5:
            return 0.75  # Moderate correlation - 25% reduction
        else:
            return 1.0  # Low correlation - no reduction
    
    def _get_drawdown_multiplier(self) -> float:
        """Get drawdown-based position size multiplier."""
        dd = self.current_drawdown_pct
        
        if dd >= self.config.drawdown_defensive_pct:
            # Linear reduction from defensive to pause threshold
            range_size = self.config.drawdown_pause_pct - self.config.drawdown_defensive_pct
            if range_size > 0:
                severity = (dd - self.config.drawdown_defensive_pct) / range_size
                return float(1.0 - severity * 0.7)  # Reduce to 30% at pause threshold
            return 0.3
        elif dd >= self.config.drawdown_cautious_pct:
            return 0.7  # 30% reduction
        else:
            return 1.0  # No reduction
    
    def open_position(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        atr: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Open a new position with adaptive trailing stop.
        
        Returns dict with stop configuration.
        """
        stop_state = None
        if self.config.trailing_stop_enabled and atr > 0:
            stop_state = self.trailing_stop_manager.open_position(
                symbol, entry_price, side, atr,
            )
        
        return {
            "symbol": symbol,
            "entry_price": entry_price,
            "initial_stop": stop_state.current_stop if stop_state else entry_price * 0.97,
            "trail_distance": stop_state.trail_distance if stop_state else 0,
            "stop_type": self.config.trailing_stop_type.value,
        }
    
    def update_position(self, symbol: str, price: float, atr: float = 0.0) -> Dict[str, Any]:
        """
        Update position with new price data.
        
        Returns actions to take (close, partial close, etc).
        """
        return self.trailing_stop_manager.update_price(symbol, price, atr) or {}
    
    def close_position(self, symbol: str) -> None:
        """Close a position and remove trailing stop."""
        self.trailing_stop_manager.close_position(symbol)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the adaptive risk engine."""
        pm = self.performance_tracker.metrics
        return {
            "risk_state": self.risk_state.value,
            "daily_pnl_pct": self.daily_pnl_pct,
            "current_drawdown_pct": self.current_drawdown_pct,
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "performance": {
                "total_trades": pm.total_trades,
                "win_rate": pm.win_rate,
                "consecutive_wins": pm.consecutive_wins,
                "consecutive_losses": pm.consecutive_losses,
                "avg_pnl_10": pm.avg_pnl_10,
                "avg_pnl_30": pm.avg_pnl_30,
                "sharpe_30": pm.sharpe_30,
            },
            "correlation": {
                "avg_correlation": self._avg_correlation,
                "regime": "crisis" if self._avg_correlation > 0.9 else "high" if self._avg_correlation > 0.75 else "normal",
            },
            "active_trailing_stops": len(self.trailing_stop_manager._positions),
            "trailing_stops": self.trailing_stop_manager.get_all_stops(),
            "multipliers": {
                "risk_state": self.performance_tracker.get_risk_multiplier(),
                "confidence": self.performance_tracker.get_confidence_multiplier(),
                "volatility": self._get_volatility_multiplier("BTC/USD"),
                "correlation": self._get_correlation_multiplier(),
                "drawdown": self._get_drawdown_multiplier(),
            },
        }
