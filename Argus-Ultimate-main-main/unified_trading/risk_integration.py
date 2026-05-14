"""
Risk Integration Module
=======================

Integrates risk management with trading system.
Refactored from unified_trading_system.py.
"""

import asyncio
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal
from collections import deque

from unified_trading.order_management import Order, Signal
from core.exception_manager import (
    RiskViolationError,
    handle_errors
)

logger = logging.getLogger(__name__)


@dataclass
class RiskCheck:
    """Result of risk check."""
    allowed: bool
    reason: Optional[str] = None
    risk_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Position information."""
    symbol: str
    quantity: Decimal
    avg_entry_price: Decimal
    side: str  # "long" or "short"
    unrealized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    max_exposure: Decimal = field(default_factory=lambda: Decimal("0"))
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PortfolioRisk:
    """Portfolio risk metrics."""
    total_value: Decimal
    cash_balance: Decimal
    total_exposure: Decimal
    var_95: Decimal
    cvar_95: Decimal
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    beta: float
    correlation_matrix: Optional[np.ndarray] = None


class RiskLimits:
    """Risk limits configuration."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Default limits
        self.max_position_size = self.config.get("max_position_size", 0.1)  # 10%
        self.max_drawdown = self.config.get("max_drawdown", 0.2)  # 20%
        self.daily_loss_limit = self.config.get("daily_loss_limit", 500)
        self.var_confidence = self.config.get("var_confidence", 0.95)
        self.max_leverage = self.config.get("max_leverage", 3.0)
        self.max_correlation = self.config.get("max_correlation", 0.8)
        self.max_sector_exposure = self.config.get("max_sector_exposure", 0.3)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any errors."""
        errors = []
        
        if self.max_position_size <= 0 or self.max_position_size > 1:
            errors.append("max_position_size must be between 0 and 1")
        
        if self.max_drawdown <= 0 or self.max_drawdown > 1:
            errors.append("max_drawdown must be between 0 and 1")
        
        if self.daily_loss_limit <= 0:
            errors.append("daily_loss_limit must be positive")
        
        if self.var_confidence <= 0 or self.var_confidence >= 1:
            errors.append("var_confidence must be between 0 and 1")
        
        if self.max_leverage <= 0:
            errors.append("max_leverage must be positive")
        
        return errors


class RiskIntegration:
    """
    Integrates risk management throughout trading system.
    """
    
    def __init__(self):
        self.limits: Optional[RiskLimits] = None
        self._positions: Dict[str, Position] = {}
        self._daily_pnl = Decimal("0")
        self._daily_trades = 0
        self._equity_curve: deque = deque(maxlen=252)  # 1 year
        self._returns_history: deque = deque(maxlen=252)
        self._last_reset = datetime.utcnow()
        self._lock = asyncio.Lock()
        
        # Risk statistics
        self._violation_count = 0
        self._blocked_orders = 0
        
        logger.info("RiskIntegration initialized")
    
    async def initialize(self, risk_config: Dict[str, Any]):
        """Initialize risk management with configuration."""
        logger.info("Initializing risk management...")
        
        self.limits = RiskLimits(risk_config)
        
        # Validate configuration
        errors = self.limits.validate_config()
        if errors:
            raise ValueError(f"Invalid risk configuration: {errors}")
        
        logger.info(f"Risk limits: max_position={self.limits.max_position_size}, "
                   f"max_drawdown={self.limits.max_drawdown}, "
                   f"daily_loss={self.limits.daily_loss_limit}")
    
    @handle_errors(logger_name=__name__, reraise=False)
    async def check_signal(self, signal: Signal) -> RiskCheck:
        """
        Check if trading signal passes risk checks.
        
        Args:
            signal: Trading signal to check
            
        Returns:
            RiskCheck with result and details
        """
        # Check daily loss limit
        if self._daily_pnl <= -self.limits.daily_loss_limit:
            self._blocked_orders += 1
            return RiskCheck(
                allowed=False,
                reason=f"Daily loss limit reached: {self._daily_pnl}",
                details={"daily_pnl": float(self._daily_pnl)}
            )
        
        # Check if position already exists
        if signal.symbol in self._positions:
            position = self._positions[signal.symbol]
            
            # Check for position size increase
            new_qty = position.quantity + signal.suggested_qty
            portfolio_value = self._get_portfolio_value()
            
            if portfolio_value > 0:
                position_pct = float(new_qty * signal.suggested_price) / float(portfolio_value)
                
                if position_pct > self.limits.max_position_size:
                    self._blocked_orders += 1
                    return RiskCheck(
                        allowed=False,
                        reason=f"Position size exceeds limit: {position_pct:.1%}",
                        risk_score=position_pct,
                        details={
                            "position_pct": position_pct,
                            "max_allowed": self.limits.max_position_size
                        }
                    )
        
        # Check signal confidence
        if signal.confidence < 0.5:
            return RiskCheck(
                allowed=False,
                reason=f"Signal confidence too low: {signal.confidence:.2f}",
                risk_score=1.0 - signal.confidence
            )
        
        # All checks passed
        return RiskCheck(
            allowed=True,
            risk_score=0.1,  # Low risk
            details={"confidence": signal.confidence}
        )
    
    @handle_errors(logger_name=__name__, reraise=False)
    async def check_order(self, order: Order, portfolio_value: Decimal) -> RiskCheck:
        """
        Check if order passes risk checks.
        
        Args:
            order: Order to check
            portfolio_value: Current portfolio value
            
        Returns:
            RiskCheck with result
        """
        # Calculate order notional value
        order_notional = order.quantity * (order.price or Decimal("0"))
        
        # Check position size limit
        if portfolio_value > 0:
            position_pct = float(order_notional) / float(portfolio_value)
            
            if position_pct > self.limits.max_position_size:
                self._violation_count += 1
                raise RiskViolationError(
                    f"Order size {position_pct:.1%} exceeds limit {self.limits.max_position_size:.1%}",
                    violation_type="position_size",
                    current_value=position_pct,
                    limit=self.limits.max_position_size
                )
        
        # Check daily loss limit
        if self._daily_pnl <= -self.limits.daily_loss_limit:
            self._violation_count += 1
            raise RiskViolationError(
                f"Daily loss limit reached: {self._daily_pnl}",
                violation_type="daily_loss",
                current_value=float(self._daily_pnl),
                limit=self.limits.daily_loss_limit
            )
        
        return RiskCheck(allowed=True, risk_score=0.1)
    
    async def update_position(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        side: str,
        realized_pnl: Decimal = None
    ):
        """
        Update position information.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Current price
            side: "long" or "short"
            realized_pnl: Realized P&L from trade
        """
        async with self._lock:
            if symbol in self._positions:
                # Update existing position
                position = self._positions[symbol]
                position.quantity = quantity
                position.avg_entry_price = price
                position.updated_at = datetime.utcnow()
                
                if realized_pnl:
                    position.realized_pnl += realized_pnl
                    self._daily_pnl += realized_pnl
            else:
                # Create new position
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_entry_price=price,
                    side=side,
                    max_exposure=quantity * price
                )
            
            # Update equity curve
            portfolio_value = self._calculate_portfolio_value(price)
            self._equity_curve.append({
                "timestamp": datetime.utcnow().isoformat(),
                "value": float(portfolio_value)
            })
    
    async def remove_position(self, symbol: str, realized_pnl: Decimal = None):
        """Remove position from tracking."""
        async with self._lock:
            if symbol in self._positions:
                position = self._positions[symbol]
                
                if realized_pnl:
                    self._daily_pnl += realized_pnl
                
                del self._positions[symbol]
                logger.info(f"Position removed: {symbol}")
    
    async def calculate_var(self, confidence: float = 0.95) -> Decimal:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            confidence: Confidence level (default 0.95)
            
        Returns:
            VaR as Decimal
        """
        if len(self._returns_history) < 30:
            return Decimal("0")
        
        returns = np.array([r for r in self._returns_history])
        var = np.percentile(returns, (1 - confidence) * 100)
        portfolio_value = self._get_portfolio_value()
        
        return Decimal(str(var)) * portfolio_value
    
    async def calculate_cvar(self, confidence: float = 0.95) -> Decimal:
        """Calculate Conditional VaR (Expected Shortfall)."""
        if len(self._returns_history) < 30:
            return Decimal("0")
        
        returns = np.array([r for r in self._returns_history])
        var = np.percentile(returns, (1 - confidence) * 100)
        cvar = returns[returns <= var].mean()
        portfolio_value = self._get_portfolio_value()
        
        return Decimal(str(cvar)) * portfolio_value
    
    async def get_portfolio_risk(self) -> PortfolioRisk:
        """Calculate comprehensive portfolio risk metrics."""
        portfolio_value = self._get_portfolio_value()
        
        # Calculate exposures
        total_exposure = sum(
            p.quantity * p.avg_entry_price
            for p in self._positions.values()
        )
        
        # Calculate VaR and CVaR
        var_95 = await self.calculate_var(0.95)
        cvar_95 = await self.calculate_cvar(0.95)
        
        # Calculate drawdown
        max_drawdown, current_drawdown = self._calculate_drawdown()
        
        # Calculate Sharpe ratio
        sharpe = self._calculate_sharpe()
        
        return PortfolioRisk(
            total_value=portfolio_value,
            cash_balance=self._get_cash_balance(),
            total_exposure=total_exposure,
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe,
            beta=0.0  # Would calculate against benchmark
        )
    
    async def get_positions(self) -> List[Position]:
        """Get all tracked positions."""
        return list(self._positions.values())
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        return self._positions.get(symbol)
    
    async def reset_daily_limits(self):
        """Reset daily limits (call at start of trading day)."""
        self._daily_pnl = Decimal("0")
        self._daily_trades = 0
        self._last_reset = datetime.utcnow()
        logger.info("Daily risk limits reset")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get risk management status."""
        return {
            "limits": {
                "max_position_size": self.limits.max_position_size if self.limits else None,
                "max_drawdown": self.limits.max_drawdown if self.limits else None,
                "daily_loss_limit": self.limits.daily_loss_limit if self.limits else None,
                "var_confidence": self.limits.var_confidence if self.limits else None
            },
            "daily_stats": {
                "daily_pnl": float(self._daily_pnl),
                "daily_trades": self._daily_trades,
                "last_reset": self._last_reset.isoformat()
            },
            "positions": len(self._positions),
            "violations": self._violation_count,
            "blocked_orders": self._blocked_orders
        }
    
    async def update_metrics(self):
        """Update risk metrics (call periodically)."""
        # This would update any ongoing calculations
        pass
    
    async def check_health(self) -> Dict[str, Any]:
        """Check risk system health."""
        issues = []
        
        if self._violation_count > 10:
            issues.append(f"High violation count: {self._violation_count}")
        
        if self._daily_pnl < -self.limits.daily_loss_limit * 0.8:
            issues.append(f"Approaching daily loss limit: {self._daily_pnl}")
        
        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "violation_count": self._violation_count,
            "blocked_orders": self._blocked_orders
        }
    
    def _get_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value."""
        positions_value = sum(
            p.quantity * p.avg_entry_price
            for p in self._positions.values()
        )
        return positions_value + self._get_cash_balance()
    
    def _get_cash_balance(self) -> Decimal:
        """Get cash balance."""
        # Would get from actual balance
        return Decimal("10000")  # Default
    
    def _calculate_portfolio_value(self, current_price: Decimal) -> Decimal:
        """Calculate portfolio value with current prices."""
        positions_value = sum(
            p.quantity * current_price
            for p in self._positions.values()
        )
        return positions_value + self._get_cash_balance()
    
    def _calculate_drawdown(self) -> tuple:
        """Calculate max and current drawdown."""
        if len(self._equity_curve) < 2:
            return 0.0, 0.0
        
        values = [e["value"] for e in self._equity_curve]
        peak = max(values)
        current = values[-1]
        
        max_dd = 0.0
        running_peak = values[0]
        
        for v in values:
            if v > running_peak:
                running_peak = v
            dd = (running_peak - v) / running_peak
            if dd > max_dd:
                max_dd = dd
        
        current_dd = (peak - current) / peak if peak > 0 else 0.0
        
        return max_dd, current_dd
    
    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio."""
        if len(self._returns_history) < 30:
            return 0.0
        
        returns = np.array([r for r in self._returns_history])
        
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        return (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
