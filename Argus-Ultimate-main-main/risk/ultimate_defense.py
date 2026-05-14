"""
Ultimate Defense System - Ultimate Edge Module

Multi-layered protection system:
- Multi-level circuit breakers
- Emergency kill switches
- Panic mode
- Black swan detection
- Cascading loss prevention
- Recovery protocols

This module provides institutional-grade capital protection.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DefenseLevel(str, Enum):
    """Defense activation levels."""
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"
    EMERGENCY = "emergency"


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker."""
    level: DefenseLevel
    trigger_count: int
    last_trigger: Optional[datetime]
    consecutive_triggers: int
    is_tripped: bool


@dataclass
class DefenseEvent:
    """Defense system event."""
    timestamp: datetime
    level: DefenseLevel
    trigger: str
    details: Dict
    action_taken: str


@dataclass
class KillSwitchConfig:
    """Kill switch configuration."""
    max_daily_loss_pct: float = 0.10
    max_drawdown_pct: float = 0.15
    max_position_size_pct: float = 0.25
    max_leverage: float = 5.0
    max_correlation: float = 0.90


@dataclass
class DefenseStatus:
    """Overall defense system status."""
    current_level: DefenseLevel
    circuit_breakers: Dict[str, CircuitBreakerState]
    kill_switch_active: bool
    panic_mode_active: bool
    events: List[DefenseEvent]
    timestamp: datetime = field(default_factory=datetime.now)


class CircuitBreaker:
    """Individual circuit breaker with multiple levels."""

    def __init__(
        self,
        name: str,
        l1_threshold: float,
        l2_threshold: float,
        l3_threshold: float,
        l1_cooldown_seconds: int = 60,
        l2_cooldown_seconds: int = 300,
        l3_cooldown_seconds: int = 900,
    ):
        self.name = name
        self.l1_threshold = l1_threshold
        self.l2_threshold = l2_threshold
        self.l3_threshold = l3_threshold
        self.l1_cooldown = l1_cooldown_seconds
        self.l2_cooldown = l2_cooldown_seconds
        self.l3_cooldown = l3_cooldown_seconds

        self._state = CircuitBreakerState(
            level=DefenseLevel.GREEN,
            trigger_count=0,
            last_trigger=None,
            consecutive_triggers=0,
            is_tripped=False,
        )

    def check(self, value: float) -> DefenseLevel:
        """Check if value triggers circuit breaker."""
        now = datetime.now()

        if value >= self.l3_threshold:
            if not self._should_cooldown(self.l3_cooldown):
                return self._state.level

            self._state.level = DefenseLevel.RED
            self._state.trigger_count += 1
            self._state.last_trigger = now
            self._state.consecutive_triggers += 1
            self._state.is_tripped = True
            return DefenseLevel.RED

        elif value >= self.l2_threshold:
            if not self._should_cooldown(self.l2_cooldown):
                return self._state.level

            self._state.level = DefenseLevel.ORANGE
            self._state.trigger_count += 1
            self._state.last_trigger = now
            self._state.consecutive_triggers += 1
            return DefenseLevel.ORANGE

        elif value >= self.l1_threshold:
            if not self._should_cooldown(self.l1_cooldown):
                return self._state.level

            self._state.level = DefenseLevel.YELLOW
            self._state.trigger_count += 1
            self._state.last_trigger = now
            self._state.consecutive_triggers = 1
            return DefenseLevel.YELLOW

        self._state.level = DefenseLevel.GREEN
        self._state.consecutive_triggers = 0
        self._state.is_tripped = False
        return DefenseLevel.GREEN

    def _should_cooldown(self, seconds: int) -> bool:
        """Check if cooldown period has elapsed."""
        if self._state.last_trigger is None:
            return True
        elapsed = (datetime.now() - self._state.last_trigger).total_seconds()
        return elapsed >= seconds

    def reset(self) -> None:
        """Manually reset circuit breaker."""
        self._state = CircuitBreakerState(
            level=DefenseLevel.GREEN,
            trigger_count=0,
            last_trigger=None,
            consecutive_triggers=0,
            is_tripped=False,
        )

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state


class UltimateDefense:
    """
    Multi-layered defense system for capital protection.

    Features:
    - 3-level circuit breakers (loss, drawdown, correlation)
    - Automatic kill switch
    - Panic mode with gradual exit
    - Black swan detection
    - Manual override
    """

    def __init__(
        self,
        config: Optional[KillSwitchConfig] = None,
        enable_panic_mode: bool = True,
    ):
        self.config = config or KillSwitchConfig()
        self.enable_panic_mode = enable_panic_mode

        self._loss_breaker = CircuitBreaker(
            name="loss",
            l1_threshold=0.02,
            l2_threshold=0.05,
            l3_threshold=0.10,
            l1_cooldown_seconds=60,
            l2_cooldown_seconds=300,
            l3_cooldown_seconds=900,
        )

        self._drawdown_breaker = CircuitBreaker(
            name="drawdown",
            l1_threshold=0.05,
            l2_threshold=0.10,
            l3_threshold=0.15,
            l1_cooldown_seconds=120,
            l2_cooldown_seconds=600,
            l3_cooldown_seconds=1800,
        )

        self._correlation_breaker = CircuitBreaker(
            name="correlation",
            l1_threshold=0.60,
            l2_threshold=0.75,
            l3_threshold=0.90,
            l1_cooldown_seconds=300,
            l2_cooldown_seconds=600,
            l3_cooldown_seconds=3600,
        )

        self._volatility_breaker = CircuitBreaker(
            name="volatility",
            l1_threshold=0.03,
            l2_threshold=0.05,
            l3_threshold=0.10,
            l1_cooldown_seconds=60,
            l2_cooldown_seconds=180,
            l3_cooldown_seconds=600,
        )

        self._breakers = {
            "loss": self._loss_breaker,
            "drawdown": self._drawdown_breaker,
            "correlation": self._correlation_breaker,
            "volatility": self._volatility_breaker,
        }

        self._kill_switch_active = False
        self._panic_mode_active = False
        self._manual_override = False
        self._events: List[DefenseEvent] = []
        self._peak_value = 0.0
        self._starting_capital = 0.0

    def initialize(self, starting_capital: float) -> None:
        """Initialize defense system with starting capital."""
        self._starting_capital = starting_capital
        self._peak_value = starting_capital
        self._kill_switch_active = False
        self._panic_mode_active = False
        logger.info(
            "UltimateDefense initialized: capital=%.2f, max_daily_loss=%.1f%%, max_drawdown=%.1f%%",
            starting_capital,
            self.config.max_daily_loss_pct * 100,
            self.config.max_drawdown_pct * 100,
        )

    def update_peak(self, current_value: float) -> None:
        """Update peak equity value."""
        if current_value > self._peak_value:
            self._peak_value = current_value

    def check(
        self,
        current_value: float,
        daily_pnl: float,
        portfolio_value: float,
        volatility: float = 0.0,
        correlation: float = 0.0,
    ) -> DefenseStatus:
        """
        Check all defense systems.

        Args:
            current_value: Current portfolio value
            daily_pnl: Today's PnL (negative for loss)
            portfolio_value: Total portfolio value
            volatility: Current volatility
            correlation: Current correlation

        Returns:
            DefenseStatus with current protection level
        """
        self.update_peak(current_value)

        daily_loss_pct = abs(daily_pnl) / portfolio_value if portfolio_value > 0 else 0.0
        if daily_pnl < 0:
            loss_level = self._loss_breaker.check(daily_loss_pct)
        else:
            loss_level = DefenseLevel.GREEN

        drawdown = (self._peak_value - current_value) / self._peak_value if self._peak_value > 0 else 0.0
        dd_level = self._drawdown_breaker.check(drawdown)

        corr_level = self._correlation_breaker.check(correlation)
        vol_level = self._volatility_breaker.check(volatility)

        levels = [loss_level, dd_level, corr_level, vol_level]

        if DefenseLevel.EMERGENCY in levels or DefenseLevel.RED in levels:
            current_level = DefenseLevel.RED
        elif DefenseLevel.ORANGE in levels:
            current_level = DefenseLevel.ORANGE
        elif DefenseLevel.YELLOW in levels:
            current_level = DefenseLevel.YELLOW
        else:
            current_level = DefenseLevel.GREEN

        if current_level == DefenseLevel.RED and not self._kill_switch_active:
            self._activate_kill_switch("RED level triggered across breakers")

        if self.enable_panic_mode and current_level == DefenseLevel.ORANGE and not self._panic_mode_active:
            self._activate_panic_mode("ORANGE level - preparing for potential exit")

        states = {name: cb.get_state() for name, cb in self._breakers.items()}

        return DefenseStatus(
            current_level=current_level,
            circuit_breakers=states,
            kill_switch_active=self._kill_switch_active,
            panic_mode_active=self._panic_mode_active,
            events=self._events[-10:],
        )

    def _activate_kill_switch(self, reason: str) -> None:
        """Activate emergency kill switch."""
        self._kill_switch_active = True
        event = DefenseEvent(
            timestamp=datetime.now(),
            level=DefenseLevel.EMERGENCY,
            trigger="kill_switch",
            details={"reason": reason},
            action_taken="All trading halted",
        )
        self._events.append(event)
        logger.critical("KILL SWITCH ACTIVATED: %s", reason)

    def _activate_panic_mode(self, reason: str) -> None:
        """Activate panic mode with gradual exit."""
        self._panic_mode_active = True
        event = DefenseEvent(
            timestamp=datetime.now(),
            level=DefenseLevel.ORANGE,
            trigger="panic_mode",
            details={"reason": reason},
            action_taken="Gradual position exit initiated",
        )
        self._events.append(event)
        logger.warning("PANIC MODE ACTIVATED: %s", reason)

    def should_block_trade(
        self,
        trade_value: float,
        portfolio_value: float,
    ) -> tuple[bool, str]:
        """
        Check if a trade should be blocked.

        Returns:
            (should_block, reason)
        """
        if self._kill_switch_active:
            return True, "Kill switch is active"

        if self._manual_override:
            return True, "Manual override is active"

        trade_pct = trade_value / portfolio_value if portfolio_value > 0 else 0.0
        if trade_pct > self.config.max_position_size_pct:
            return True, f"Trade size {trade_pct*100:.1f}% exceeds max {self.config.max_position_size_pct*100:.1f}%"

        return False, ""

    def should_reduce_leverage(self, current_leverage: float) -> tuple[bool, float]:
        """
        Check if leverage should be reduced.

        Returns:
            (should_reduce, recommended_leverage)
        """
        if self._kill_switch_active or self._panic_mode_active:
            return True, 1.0

        if current_leverage > self.config.max_leverage:
            return True, self.config.max_leverage

        for cb in self._breakers.values():
            state = cb.get_state()
            if state.level in (DefenseLevel.ORANGE, DefenseLevel.RED):
                return True, max(1.0, current_leverage * 0.5)

        return False, current_leverage

    def get_exit_recommendation(self) -> tuple[bool, str, float]:
        """
        Get position exit recommendation.

        Returns:
            (should_exit, reason, percentage_to_exit)
        """
        if self._kill_switch_active:
            return True, "Kill switch active - full exit", 1.0

        if self._panic_mode_active:
            return True, "Panic mode - gradual exit", 0.5

        for name, cb in self._breakers.items():
            state = cb.get_state()
            if state.level == DefenseLevel.RED:
                return True, f"Circuit breaker {name} at RED level", 0.75
            elif state.level == DefenseLevel.ORANGE:
                return True, f"Circuit breaker {name} at ORANGE level", 0.25

        return False, "", 0.0

    def detect_black_swan(
        self,
        returns: List[float],
        window: int = 20,
    ) -> tuple[bool, float]:
        """
        Detect black swan events (extreme moves).

        Args:
            returns: Historical returns
            window: Lookback window

        Returns:
            (is_black_swan, severity)
        """
        if len(returns) < window:
            return False, 0.0

        recent = returns[-window:]
        mean = np.mean(recent)
        std = np.std(recent)

        if std == 0:
            return False, 0.0

        z_score = abs(recent[-1] - mean) / std

        if z_score > 5:
            return True, 1.0
        elif z_score > 4:
            return True, 0.75
        elif z_score > 3:
            return True, 0.5

        return False, 0.0

    def manual_activate_kill_switch(self, reason: str) -> None:
        """Manually activate kill switch."""
        self._kill_switch_active = True
        event = DefenseEvent(
            timestamp=datetime.now(),
            level=DefenseLevel.EMERGENCY,
            trigger="manual_kill_switch",
            details={"reason": reason, "manual": True},
            action_taken="Manual activation by operator",
        )
        self._events.append(event)
        logger.critical("MANUAL KILL SWITCH ACTIVATED: %s", reason)

    def manual_deactivate(self) -> None:
        """Manually deactivate kill switch / panic mode."""
        self._kill_switch_active = False
        self._panic_mode_active = False
        self._manual_override = True
        for cb in self._breakers.values():
            cb.reset()
        event = DefenseEvent(
            timestamp=datetime.now(),
            level=DefenseLevel.GREEN,
            trigger="manual_deactivation",
            details={},
            action_taken="Manual deactivation",
        )
        self._events.append(event)
        logger.info("Manual defense deactivation")

    def reset_daily(self) -> None:
        """Reset daily counters (call at start of trading day)."""
        self._starting_capital = self._peak_value
        for cb in self._breakers.values():
            cb.reset()
        logger.info("Defense system daily reset")

    def get_status(self) -> DefenseStatus:
        """Get current defense system status."""
        states = {name: cb.get_state() for name, cb in self._breakers.items()}
        return DefenseStatus(
            current_level=DefenseLevel.GREEN,
            circuit_breakers=states,
            kill_switch_active=self._kill_switch_active,
            panic_mode_active=self._panic_mode_active,
            events=self._events[-10:],
        )

    def get_recent_events(self, limit: int = 20) -> List[DefenseEvent]:
        """Get recent defense events."""
        return self._events[-limit:]
