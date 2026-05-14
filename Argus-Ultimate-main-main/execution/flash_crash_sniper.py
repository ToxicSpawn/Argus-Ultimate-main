"""
Flash Crash Sniper v2.0
========================
Automatically buys 5%+ dips in Argus Ultimate.

Provides:
- Real-time price monitoring
- Flash crash detection
- Auto-buy on significant dips
- Recovery detection
- Risk-adjusted entry sizing
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CrashSeverity(Enum):
    """Flash crash severity levels."""
    MINOR = "minor"          # 3-5% drop
    MODERATE = "moderate"    # 5-10% drop
    SEVERE = "severe"        # 10-20% drop
    EXTREME = "extreme"      # 20%+ drop


@dataclass
class FlashCrashEvent:
    """Detected flash crash event."""
    symbol: str
    timestamp: datetime
    peak_price: float
    trough_price: float
    drop_pct: float
    severity: CrashSeverity
    volume_spike: float  # Volume multiplier vs average
    recovery_time_seconds: Optional[float]
    buy_executed: bool
    buy_price: Optional[float] = None
    buy_amount: Optional[float] = None
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None


@dataclass
class SniperConfig:
    """Flash crash sniper configuration."""
    min_drop_pct: float = 5.0           # Minimum drop to trigger
    max_drop_pct: float = 50.0          # Maximum drop (might be real crash)
    volume_confirmation: float = 2.0    # Volume spike multiplier
    recovery_threshold: float = 0.3     # 30% recovery to confirm bottom
    max_position_pct: float = 0.10      # Max 10% of portfolio per snipe
    cooldown_minutes: int = 30          # Cooldown between snipes per symbol
    enable_long_only: bool = True       # Only buy dips, not short crashes


class PriceTracker:
    """
    Tracks price history for crash detection.
    """
    
    def __init__(self, window_size: int = 100) -> None:
        """
        Initialize price tracker.
        
        Args:
            window_size: Number of prices to track
        """
        self.window_size = window_size
        self._prices: Deque[float] = deque(maxlen=window_size)
        self._volumes: Deque[float] = deque(maxlen=window_size)
        self._timestamps: Deque[datetime] = deque(maxlen=window_size)
    
    def update(self, price: float, volume: float = 0.0) -> None:
        """Update with new price/volume."""
        self._prices.append(price)
        self._volumes.append(volume)
        self._timestamps.append(datetime.now())
    
    @property
    def current_price(self) -> Optional[float]:
        """Get current price."""
        return self._prices[-1] if self._prices else None
    
    @property
    def highest_price(self) -> Optional[float]:
        """Get highest price in window."""
        return max(self._prices) if self._prices else None
    
    @property
    def lowest_price(self) -> Optional[float]:
        """Get lowest price in window."""
        return min(self._prices) if self._prices else None
    
    @property
    def avg_volume(self) -> float:
        """Get average volume."""
        if not self._volumes:
            return 0.0
        return float(np.mean(self._volumes))
    
    @property
    def current_volume(self) -> float:
        """Get current volume."""
        return self._volumes[-1] if self._volumes else 0.0
    
    def get_price_change_from_peak(self) -> float:
        """Get percentage change from peak."""
        if not self._prices or len(self._prices) < 2:
            return 0.0
        
        peak = max(self._prices)
        current = self._prices[-1]
        
        return (current - peak) / peak * 100
    
    def get_recent_returns(self, n: int = 20) -> np.ndarray:
        """Get recent returns."""
        if len(self._prices) < n + 1:
            return np.array([])
        
        prices = np.array(list(self._prices)[-n-1:])
        returns = np.diff(prices) / prices[:-1]
        return returns
    
    def get_volatility(self, n: int = 20) -> float:
        """Get recent volatility."""
        returns = self.get_recent_returns(n)
        if len(returns) < 2:
            return 0.0
        return float(np.std(returns) * np.sqrt(365 * 24))  # Annualized


class CrashDetector:
    """
    Detects flash crashes in price data.
    """
    
    def __init__(self, config: SniperConfig) -> None:
        """
        Initialize crash detector.
        
        Args:
            config: Sniper configuration
        """
        self.config = config
        self._trackers: Dict[str, PriceTracker] = {}
        self._last_snipe: Dict[str, datetime] = {}
    
    def update(
        self,
        symbol: str,
        price: float,
        volume: float = 0.0
    ) -> Optional[FlashCrashEvent]:
        """
        Update with new price and check for crash.
        
        Returns FlashCrashEvent if crash detected.
        """
        # Initialize tracker if needed
        if symbol not in self._trackers:
            self._trackers[symbol] = PriceTracker(window_size=100)
        
        tracker = self._trackers[symbol]
        
        # Check for crash BEFORE updating price
        event = self._check_for_crash(symbol, tracker, price, volume)
        
        # Update tracker
        tracker.update(price, volume)
        
        return event
    
    def _check_for_crash(
        self,
        symbol: str,
        tracker: PriceTracker,
        current_price: float,
        current_volume: float
    ) -> Optional[FlashCrashEvent]:
        """Check if current price represents a flash crash."""
        if tracker.highest_price is None:
            return None
        
        # Check cooldown
        if symbol in self._last_snipe:
            cooldown_end = self._last_snipe[symbol] + timedelta(
                minutes=self.config.cooldown_minutes
            )
            if datetime.now() < cooldown_end:
                return None
        
        # Calculate drop from peak
        peak_price = tracker.highest_price
        drop_pct = (peak_price - current_price) / peak_price * 100
        
        # Check if drop is significant enough
        if drop_pct < self.config.min_drop_pct:
            return None
        
        # Check if drop is too large (might be real crash, not flash)
        if drop_pct > self.config.max_drop_pct:
            logger.warning(
                "%s: Drop %.1f%% exceeds max - might be real crash, skipping",
                symbol, drop_pct
            )
            return None
        
        # Check volume confirmation
        avg_volume = tracker.avg_volume
        volume_spike = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        if volume_spike < self.config.volume_confirmation:
            logger.debug(
                "%s: Volume spike %.2fx below threshold %.2fx",
                symbol, volume_spike, self.config.volume_confirmation
            )
            return None
        
        # Determine severity
        severity = self._classify_severity(drop_pct)
        
        # Create event
        event = FlashCrashEvent(
            symbol=symbol,
            timestamp=datetime.now(),
            peak_price=peak_price,
            trough_price=current_price,
            drop_pct=drop_pct,
            severity=severity,
            volume_spike=volume_spike,
            recovery_time_seconds=None,
            buy_executed=False
        )
        
        logger.warning(
            "Flash crash detected: %s - %.1f%% drop (%s severity), volume %.1fx",
            symbol, drop_pct, severity.value, volume_spike
        )
        
        return event
    
    def _classify_severity(self, drop_pct: float) -> CrashSeverity:
        """Classify crash severity."""
        if drop_pct >= 20:
            return CrashSeverity.EXTREME
        elif drop_pct >= 10:
            return CrashSeverity.SEVERE
        elif drop_pct >= 5:
            return CrashSeverity.MODERATE
        else:
            return CrashSeverity.MINOR
    
    def record_snipe(self, symbol: str) -> None:
        """Record that a snipe was executed."""
        self._last_snipe[symbol] = datetime.now()
    
    def get_tracker(self, symbol: str) -> Optional[PriceTracker]:
        """Get price tracker for symbol."""
        return self._trackers.get(symbol)


class RecoveryDetector:
    """
    Detects price recovery after crash for optimal exit.
    """
    
    def __init__(
        self,
        target_recovery_pct: float = 50.0,  # Target 50% recovery
        max_hold_hours: int = 24
    ) -> None:
        """
        Initialize recovery detector.
        
        Args:
            target_recovery_pct: Target recovery percentage
            max_hold_hours: Maximum hold time
        """
        self.target_recovery_pct = target_recovery_pct
        self.max_hold_hours = max_hold_hours
        
        self._entries: Dict[str, Dict[str, Any]] = {}
    
    def record_entry(
        self,
        symbol: str,
        entry_price: float,
        crash_low: float
    ) -> None:
        """Record entry price."""
        self._entries[symbol] = {
            "entry_price": entry_price,
            "crash_low": crash_low,
            "entry_time": datetime.now(),
            "highest_since_entry": entry_price
        }
    
    def update_and_check_exit(
        self,
        symbol: str,
        current_price: float
    ) -> Tuple[bool, str]:
        """
        Update price and check if should exit.
        
        Returns (should_exit, reason).
        """
        if symbol not in self._entries:
            return False, "no_entry"
        
        entry = self._entries[symbol]
        
        # Update highest price since entry
        if current_price > entry["highest_since_entry"]:
            entry["highest_since_entry"] = current_price
        
        # Calculate recovery from crash low
        recovery_from_low = (current_price - entry["crash_low"]) / entry["crash_low"] * 100
        
        # Calculate profit from entry
        profit_pct = (current_price - entry["entry_price"]) / entry["entry_price"] * 100
        
        # Check hold time
        hold_time = datetime.now() - entry["entry_time"]
        
        # Exit conditions
        
        # 1. Target recovery reached
        if recovery_from_low >= self.target_recovery_pct:
            return True, f"target_recovery_{recovery_from_low:.1f}%"
        
        # 2. Good profit achieved
        if profit_pct >= 3.0:
            return True, f"profit_target_{profit_pct:.1f}%"
        
        # 3. Maximum hold time exceeded
        if hold_time >= timedelta(hours=self.max_hold_hours):
            return True, f"max_hold_time_{hold_time.total_seconds()/3600:.1f}h"
        
        # 4. Price starts dropping again (trailing stop)
        if entry["highest_since_entry"] > entry["entry_price"]:
            drawdown_from_high = (entry["highest_since_entry"] - current_price) / entry["highest_since_entry"] * 100
            if drawdown_from_high >= 5.0:
                return True, f"trailing_stop_{drawdown_from_high:.1f}%"
        
        return False, "holding"
    
    def remove_entry(self, symbol: str) -> None:
        """Remove entry tracking."""
        if symbol in self._entries:
            del self._entries[symbol]


class FlashCrashSniper:
    """
    Main flash crash sniper for Argus.
    
    Automatically buys significant dips with risk management.
    """
    
    def __init__(
        self,
        config: Optional[SniperConfig] = None,
        portfolio_value: float = 10000.0
    ) -> None:
        """
        Initialize flash crash sniper.
        
        Args:
            config: Sniper configuration
            portfolio_value: Current portfolio value
        """
        self.config = config or SniperConfig()
        self.portfolio_value = portfolio_value
        
        self.detector = CrashDetector(self.config)
        self.recovery_detector = RecoveryDetector()
        
        self._active_snipes: Dict[str, FlashCrashEvent] = {}
        self._snipe_history: List[FlashCrashEvent] = []
        self._total_pnl: float = 0.0
        
        logger.info(
            "FlashCrashSniper initialized: min_drop=%.1f%%, max_pos=%.1f%%",
            self.config.min_drop_pct, self.config.max_position_pct * 100
        )
    
    def update(
        self,
        symbol: str,
        price: float,
        volume: float = 0.0
    ) -> Optional[Dict[str, Any]]:
        """
        Update with new price data.
        
        Returns buy signal if flash crash detected.
        """
        # Check for crash
        crash_event = self.detector.update(symbol, price, volume)
        
        if crash_event:
            # Calculate buy amount
            buy_amount = self._calculate_buy_amount(crash_event)
            
            return {
                "action": "BUY",
                "symbol": symbol,
                "price": price,
                "amount_usd": buy_amount,
                "drop_pct": crash_event.drop_pct,
                "severity": crash_event.severity.value,
                "volume_spike": crash_event.volume_spike,
                "reason": f"Flash crash: {crash_event.drop_pct:.1f}% drop"
            }
        
        # Check if in active snipe - monitor for exit
        if symbol in self._active_snipes:
            should_exit, reason = self.recovery_detector.update_and_check_exit(
                symbol, price
            )
            
            if should_exit:
                return {
                    "action": "SELL",
                    "symbol": symbol,
                    "price": price,
                    "reason": reason
                }
        
        return None
    
    def _calculate_buy_amount(self, event: FlashCrashEvent) -> float:
        """Calculate buy amount based on crash severity."""
        # Base position size
        base_pct = self.config.max_position_pct
        
        # Adjust based on severity
        severity_multiplier = {
            CrashSeverity.MINOR: 0.5,
            CrashSeverity.MODERATE: 0.75,
            CrashSeverity.SEVERE: 1.0,
            CrashSeverity.EXTREME: 0.5  # Smaller for extreme (might not recover)
        }
        
        adjusted_pct = base_pct * severity_multiplier.get(event.severity, 0.5)
        
        return self.portfolio_value * adjusted_pct
    
    def execute_buy(
        self,
        symbol: str,
        price: float,
        amount_usd: float
    ) -> FlashCrashEvent:
        """
        Execute a flash crash buy.
        
        Args:
            symbol: Symbol to buy
            price: Buy price
            amount_usd: Amount in USD
            
        Returns:
            FlashCrashEvent with buy details
        """
        # Get crash event from detector
        tracker = self.detector.get_tracker(symbol)
        if tracker:
            peak = tracker.highest_price or price
            drop_pct = (peak - price) / peak * 100
        else:
            peak = price
            drop_pct = 0.0
        
        # Create/Update event
        event = FlashCrashEvent(
            symbol=symbol,
            timestamp=datetime.now(),
            peak_price=peak,
            trough_price=price,
            drop_pct=drop_pct,
            severity=self.detector._classify_severity(drop_pct),
            volume_spike=1.0,
            recovery_time_seconds=None,
            buy_executed=True,
            buy_price=price,
            buy_amount=amount_usd
        )
        
        # Record in active snipes
        self._active_snipes[symbol] = event
        self._snipe_history.append(event)
        
        # Record for cooldown
        self.detector.record_snipe(symbol)
        
        # Record entry for recovery tracking
        self.recovery_detector.record_entry(symbol, price, price)
        
        logger.info(
            "Flash crash buy executed: %s @ %.2f, amount=$%.2f",
            symbol, price, amount_usd
        )
        
        return event
    
    def execute_sell(
        self,
        symbol: str,
        price: float,
        reason: str
    ) -> Dict[str, Any]:
        """
        Execute sell after recovery.
        
        Returns trade summary.
        """
        if symbol not in self._active_snipes:
            return {"error": "No active snipe"}
        
        event = self._active_snipes[symbol]
        
        if event.buy_price is None:
            return {"error": "No buy price recorded"}
        
        # Calculate PnL
        buy_price = event.buy_price
        buy_amount = event.buy_amount or 0
        
        # Calculate quantity
        if buy_price > 0:
            quantity = buy_amount / buy_price
            sell_amount = quantity * price
            pnl = sell_amount - buy_amount
            pnl_pct = (price - buy_price) / buy_price * 100
        else:
            pnl = 0
            pnl_pct = 0
        
        # Update event
        event.current_price = price
        event.unrealized_pnl = pnl
        
        # Update totals
        self._total_pnl += pnl
        
        # Remove from active
        del self._active_snipes[symbol]
        self.recovery_detector.remove_entry(symbol)
        
        summary = {
            "symbol": symbol,
            "buy_price": buy_price,
            "sell_price": price,
            "amount_usd": buy_amount,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "total_pnl": self._total_pnl
        }
        
        logger.info(
            "Flash crash sell: %s @ %.2f, PnL=$%.2f (%.1f%%)",
            symbol, price, pnl, pnl_pct
        )
        
        return summary
    
    def update_portfolio_value(self, value: float) -> None:
        """Update portfolio value."""
        self.portfolio_value = value
    
    def get_active_snipes(self) -> Dict[str, FlashCrashEvent]:
        """Get all active snipes."""
        return self._active_snipes.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sniper statistics."""
        completed_snipes = len(self._snipe_history)
        active_snipes = len(self._active_snipes)
        
        return {
            "total_snipes": completed_snipes,
            "active_snipes": active_snipes,
            "total_pnl": self._total_pnl,
            "portfolio_value": self.portfolio_value,
            "config": {
                "min_drop_pct": self.config.min_drop_pct,
                "max_position_pct": self.config.max_position_pct * 100,
                "cooldown_minutes": self.config.cooldown_minutes
            }
        }
