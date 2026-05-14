#!/usr/bin/env python3
"""
Circuit Breaker - Automated Risk Management System
===================================================

Intelligent circuit breaker that halts trading during extreme market conditions
to prevent catastrophic losses.
"""

import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""

    max_drawdown_threshold: float = 0.15  # 15% max drawdown
    volatility_threshold: float = 0.05  # 5% volatility threshold
    consecutive_loss_threshold: int = 5  # Max consecutive losses
    time_window_minutes: int = 60  # Time window for checks
    cooldown_period_minutes: int = 30  # Cooldown after trigger


class CircuitBreaker:
    """
    Circuit Breaker - Automated risk management system

    Monitors trading performance and automatically halts trading
    when risk thresholds are exceeded.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()

        # State tracking
        self.is_active = False
        self.triggered_at: Optional[datetime] = None
        self.last_check_time = datetime.now()

        # Performance tracking
        self.portfolio_value = 0.0
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.consecutive_losses = 0
        self.recent_trades: List[Dict[str, Any]] = []

        # Risk metrics
        self.volatility = 0.0
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None

        logger.info("Circuit Breaker initialized")

    def check_trading_allowed(
        self, current_portfolio_value: float, recent_trades: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check if trading is currently allowed.

        Args:
            current_portfolio_value: Current portfolio value
            recent_trades: Recent trades for analysis

        Returns:
            Dictionary with decision and reasoning
        """
        self.portfolio_value = current_portfolio_value

        # Update peak value
        if current_portfolio_value > self.peak_value:
            self.peak_value = current_portfolio_value

        # Calculate current drawdown
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - current_portfolio_value) / self.peak_value
        else:
            self.current_drawdown = 0.0

        # Update trade history
        if recent_trades:
            self.recent_trades.extend(recent_trades)
            # Keep only recent trades
            cutoff_time = datetime.now() - timedelta(minutes=self.config.time_window_minutes)
            self.recent_trades = [
                trade for trade in self.recent_trades if trade.get("timestamp", datetime.min) > cutoff_time
            ]

        # Check if we're in cooldown period
        if self.is_active:
            if self._check_cooldown_expired():
                self._deactivate_circuit_breaker()
                return {
                    "allowed": True,
                    "reason": "Cooldown period expired, circuit breaker deactivated",
                }
            else:
                return {
                    "allowed": False,
                    "reason": f"Circuit breaker active until {self.triggered_at + timedelta(minutes=self.config.cooldown_period_minutes)}",
                }

        # Perform risk checks
        risk_checks = self._perform_risk_checks()

        if risk_checks["should_trigger"]:
            self._activate_circuit_breaker(risk_checks["trigger_reason"])
            return {
                "allowed": False,
                "reason": f"Circuit breaker triggered: {risk_checks['trigger_reason']}",
                "risk_metrics": risk_checks,
            }

        return {"allowed": True, "reason": "All risk checks passed", "risk_metrics": risk_checks}

    def _perform_risk_checks(self) -> Dict[str, Any]:
        """Perform comprehensive risk checks"""
        checks = {
            "drawdown_check": self.current_drawdown > self.config.max_drawdown_threshold,
            "volatility_check": self.volatility > self.config.volatility_threshold,
            "consecutive_loss_check": self.consecutive_losses >= self.config.consecutive_loss_threshold,
            "time_window_check": self._check_time_window_risks(),
        }

        should_trigger = any(checks.values())

        trigger_reason = None
        if checks["drawdown_check"]:
            trigger_reason = ".2%"
        elif checks["volatility_check"]:
            trigger_reason = ".2%"
        elif checks["consecutive_loss_check"]:
            trigger_reason = f"{self.consecutive_losses} consecutive losses"
        elif checks["time_window_check"]:
            trigger_reason = "Time window risk limits exceeded"

        return {
            "should_trigger": should_trigger,
            "trigger_reason": trigger_reason,
            "current_drawdown": self.current_drawdown,
            "volatility": self.volatility,
            "consecutive_losses": self.consecutive_losses,
            "individual_checks": checks,
        }

    def _check_time_window_risks(self) -> bool:
        """Check risks within the time window"""
        if not self.recent_trades:
            return False

        # Calculate loss rate in time window
        losses = [t for t in self.recent_trades if t.get("pnl", 0) < 0]
        loss_rate = len(losses) / len(self.recent_trades) if self.recent_trades else 0

        # Trigger if loss rate > 70% in time window
        return loss_rate > 0.7

    def _activate_circuit_breaker(self, reason: str) -> None:
        """Activate the circuit breaker"""
        self.is_active = True
        self.triggered_at = datetime.now()
        self.failure_count += 1
        self.last_failure_time = self.triggered_at

        logger.warning(f"CIRCUIT BREAKER ACTIVATED: {reason}")
        logger.warning(f"Trading halted for {self.config.cooldown_period_minutes} minutes")

    def _deactivate_circuit_breaker(self) -> None:
        """Deactivate the circuit breaker"""
        self.is_active = False
        self.triggered_at = None

        logger.info("Circuit breaker deactivated - trading resumed")

    def _check_cooldown_expired(self) -> bool:
        """Check if cooldown period has expired"""
        if not self.triggered_at:
            return True

        cooldown_end = self.triggered_at + timedelta(minutes=self.config.cooldown_period_minutes)
        return datetime.now() >= cooldown_end

    def update_trade_result(self, pnl: float) -> None:
        """
        Update with trade result to track consecutive losses.

        Args:
            pnl: Profit/loss from the trade
        """
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def update_volatility(self, volatility: float) -> None:
        """
        Update current market volatility.

        Args:
            volatility: Current volatility measure
        """
        self.volatility = volatility

    def manual_override(self, activate: bool, reason: str = "Manual override") -> None:
        """
        Manually override circuit breaker state.

        Args:
            activate: True to activate, False to deactivate
            reason: Reason for override
        """
        if activate and not self.is_active:
            self._activate_circuit_breaker(reason)
        elif not activate and self.is_active:
            self._deactivate_circuit_breaker()
            logger.info(f"Manual circuit breaker deactivation: {reason}")

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        return {
            "is_active": self.is_active,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "cooldown_remaining_minutes": self._get_cooldown_remaining(),
            "current_drawdown": self.current_drawdown,
            "volatility": self.volatility,
            "consecutive_losses": self.consecutive_losses,
            "failure_count": self.failure_count,
            "config": {
                "max_drawdown_threshold": self.config.max_drawdown_threshold,
                "volatility_threshold": self.config.volatility_threshold,
                "consecutive_loss_threshold": self.config.consecutive_loss_threshold,
                "cooldown_period_minutes": self.config.cooldown_period_minutes,
            },
        }

    def _get_cooldown_remaining(self) -> float:
        """Get remaining cooldown time in minutes"""
        if not self.is_active or not self.triggered_at:
            return 0.0

        cooldown_end = self.triggered_at + timedelta(minutes=self.config.cooldown_period_minutes)
        remaining = cooldown_end - datetime.now()

        return max(0.0, remaining.total_seconds() / 60)

    def reset(self) -> None:
        """Reset circuit breaker state"""
        self.is_active = False
        self.triggered_at = None
        self.peak_value = self.portfolio_value
        self.current_drawdown = 0.0
        self.consecutive_losses = 0
        self.recent_trades.clear()
        self.volatility = 0.0

        logger.info("Circuit breaker reset")

    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        return {
            "circuit_breaker_status": self.get_status(),
            "risk_assessment": {
                "drawdown_level": (
                    "high" if self.current_drawdown > 0.1 else "moderate" if self.current_drawdown > 0.05 else "low"
                ),
                "volatility_level": (
                    "high" if self.volatility > 0.03 else "moderate" if self.volatility > 0.015 else "low"
                ),
                "loss_streak_level": (
                    "high" if self.consecutive_losses >= 3 else "moderate" if self.consecutive_losses >= 2 else "low"
                ),
            },
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []

        if self.current_drawdown > self.config.max_drawdown_threshold * 0.8:
            recommendations.append("Consider reducing position sizes")
        elif self.current_drawdown > self.config.max_drawdown_threshold * 0.5:
            recommendations.append("Monitor drawdown closely")

        if self.volatility > self.config.volatility_threshold * 0.8:
            recommendations.append("High volatility detected - consider wider stops")

        if self.consecutive_losses >= self.config.consecutive_loss_threshold - 1:
            recommendations.append("Multiple consecutive losses - review strategy")

        if not recommendations:
            recommendations.append("Risk levels within acceptable ranges")

        return recommendations


_DEFAULT_CIRCUIT_BREAKER: Optional[CircuitBreaker] = None


def get_circuit_breaker(config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """
    Backwards-compat factory used by some legacy modules.
    Returns a module-level singleton when `config` is None.
    """

    global _DEFAULT_CIRCUIT_BREAKER
    if config is None:
        if _DEFAULT_CIRCUIT_BREAKER is None:
            _DEFAULT_CIRCUIT_BREAKER = CircuitBreaker()
        return _DEFAULT_CIRCUIT_BREAKER
    return CircuitBreaker(config=config)
