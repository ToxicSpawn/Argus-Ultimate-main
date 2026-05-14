"""
Leverage Manager v2.0
======================
Safe leverage management for small accounts in Argus Ultimate.

Provides:
- Dynamic leverage adjustment
- Liquidation price monitoring
- Cross-margin optimization
- Position sizing with leverage
- Auto-deleveraging on risk
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MarginType(Enum):
    """Margin type."""
    ISOLATED = "isolated"
    CROSS = "cross"


class LeverageAction(Enum):
    """Leverage adjustment actions."""
    INCREASE = "increase"
    DECREASE = "decrease"
    MAINTAIN = "maintain"
    CLOSE = "close"
    DELEVERAGE = "deleverage"


@dataclass
class LeveragePosition:
    """Leveraged position details."""
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    current_price: float
    size: float  # Position size in base asset
    leverage: float
    margin_type: MarginType
    margin_used: float
    liquidation_price: float
    unrealized_pnl: float
    margin_ratio: float  # Current margin ratio
    timestamp: datetime


@dataclass
class LeverageRecommendation:
    """Leverage adjustment recommendation."""
    symbol: str
    action: LeverageAction
    current_leverage: float
    target_leverage: float
    reason: str
    risk_score: float  # 0-1, higher = more risk
    estimated_impact: Dict[str, float]
    timestamp: datetime


@dataclass
class LiquidationWarning:
    """Liquidation warning."""
    symbol: str
    current_price: float
    liquidation_price: float
    distance_pct: float
    margin_ratio: float
    time_to_liquidation_estimate: Optional[float]  # minutes
    severity: str  # "warning", "critical", "imminent"
    recommended_action: str


class LeverageCalculator:
    """
    Calculates leverage-related metrics.
    """
    
    @staticmethod
    def calculate_liquidation_price(
        entry_price: float,
        leverage: float,
        side: str,
        maintenance_margin_rate: float = 0.005
    ) -> float:
        """
        Calculate liquidation price.
        
        For longs: Liquidation when price drops below entry * (1 - 1/leverage + MMR)
        For shorts: Liquidation when price rises above entry * (1 + 1/leverage - MMR)
        """
        if side == "long":
            return entry_price * (1 - 1/leverage + maintenance_margin_rate)
        else:
            return entry_price * (1 + 1/leverage - maintenance_margin_rate)
    
    @staticmethod
    def calculate_margin_ratio(
        position_value: float,
        margin: float,
        unrealized_pnl: float
    ) -> float:
        """
        Calculate margin ratio (margin / position_value).
        
        Lower ratio = closer to liquidation.
        """
        if position_value == 0:
            return 1.0
        
        effective_margin = margin + unrealized_pnl
        return effective_margin / position_value
    
    @staticmethod
    def calculate_safe_leverage(
        volatility: float,
        max_drawdown: float = 0.2,
        win_rate: float = 0.5,
        avg_win_loss_ratio: float = 1.5
    ) -> float:
        """
        Calculate safe leverage based on volatility and strategy metrics.
        
        Uses modified Kelly criterion for leverage.
        """
        # Base leverage from volatility
        # Higher volatility = lower leverage
        if volatility > 0.1:
            base_leverage = 2.0
        elif volatility > 0.05:
            base_leverage = 3.0
        elif volatility > 0.02:
            base_leverage = 5.0
        else:
            base_leverage = 10.0
        
        # Kelly criterion adjustment
        kelly = win_rate - (1 - win_rate) / avg_win_loss_ratio
        kelly_multiplier = max(0.25, min(1.0, kelly * 2))
        
        # Max drawdown constraint
        max_leverage_from_dd = 1.0 / max_drawdown
        
        safe_leverage = min(
            base_leverage * kelly_multiplier,
            max_leverage_from_dd,
            20.0  # Hard cap
        )
        
        return max(1.0, safe_leverage)
    
    @staticmethod
    def calculate_position_size(
        account_balance: float,
        leverage: float,
        risk_per_trade: float,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """
        Calculate position size based on risk.
        
        Returns position size in base asset.
        """
        # Risk amount
        risk_amount = account_balance * risk_per_trade
        
        # Price risk per unit
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return 0.0
        
        # Position size (limited by leverage)
        max_position_value = account_balance * leverage
        risk_based_size = risk_amount / price_risk
        max_size = max_position_value / entry_price
        
        return min(risk_based_size, max_size)


class LiquidationMonitor:
    """
    Monitors positions for liquidation risk.
    """
    
    def __init__(
        self,
        warning_threshold: float = 0.15,  # 15% from liquidation
        critical_threshold: float = 0.05,  # 5% from liquidation
        check_interval_seconds: int = 5
    ) -> None:
        """
        Initialize liquidation monitor.
        
        Args:
            warning_threshold: Distance to liquidation for warning
            critical_threshold: Distance to liquidation for critical alert
            check_interval_seconds: How often to check
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval_seconds
        
        self._positions: Dict[str, LeveragePosition] = {}
        self._warnings: List[LiquidationWarning] = []
    
    def update_position(self, position: LeveragePosition) -> None:
        """Update position data."""
        self._positions[position.symbol] = position
    
    def remove_position(self, symbol: str) -> None:
        """Remove position from monitoring."""
        if symbol in self._positions:
            del self._positions[symbol]
    
    def check_all_positions(self) -> List[LiquidationWarning]:
        """
        Check all positions for liquidation risk.
        
        Returns list of warnings.
        """
        warnings = []
        
        for symbol, position in self._positions.items():
            warning = self._check_position(position)
            if warning:
                warnings.append(warning)
                self._warnings.append(warning)
        
        return warnings
    
    def _check_position(self, position: LeveragePosition) -> Optional[LiquidationWarning]:
        """Check single position for liquidation risk."""
        # Calculate distance to liquidation
        if position.side == "long":
            distance = (position.current_price - position.liquidation_price) / position.current_price
        else:
            distance = (position.liquidation_price - position.current_price) / position.current_price
        
        # Determine severity
        if distance <= self.critical_threshold:
            severity = "imminent"
            action = "CLOSE POSITION IMMEDIATELY"
        elif distance <= self.warning_threshold:
            severity = "critical"
            action = "Reduce position or add margin"
        elif distance <= self.warning_threshold * 2:
            severity = "warning"
            action = "Monitor closely"
        else:
            return None
        
        # Estimate time to liquidation (simplified)
        # Based on historical volatility
        time_estimate = self._estimate_time_to_liquidation(position, distance)
        
        warning = LiquidationWarning(
            symbol=position.symbol,
            current_price=position.current_price,
            liquidation_price=position.liquidation_price,
            distance_pct=distance * 100,
            margin_ratio=position.margin_ratio,
            time_to_liquidation_estimate=time_estimate,
            severity=severity,
            recommended_action=action
        )
        
        logger.warning(
            "Liquidation %s: %s - %.2f%% away, margin ratio: %.4f",
            severity, position.symbol, distance * 100, position.margin_ratio
        )
        
        return warning
    
    def _estimate_time_to_liquidation(
        self,
        position: LeveragePosition,
        distance: float
    ) -> Optional[float]:
        """
        Estimate time to liquidation in minutes.
        
        Simplified estimation based on price velocity.
        """
        # Would need price history for accurate estimation
        # Return None for now
        return None
    
    def get_riskiest_positions(self, n: int = 5) -> List[LeveragePosition]:
        """Get positions closest to liquidation."""
        positions = list(self._positions.values())
        
        def distance_to_liq(p: LeveragePosition) -> float:
            if p.side == "long":
                return (p.current_price - p.liquidation_price) / p.current_price
            else:
                return (p.liquidation_price - p.current_price) / p.current_price
        
        positions.sort(key=distance_to_liq)
        return positions[:n]


class LeverageManager:
    """
    Main leverage manager for Argus.
    
    Manages leverage positions safely with automatic risk management.
    """
    
    def __init__(
        self,
        max_leverage: float = 10.0,
        risk_per_trade: float = 0.02,  # 2% risk per trade
        max_portfolio_leverage: float = 3.0,
        auto_deleverage: bool = True
    ) -> None:
        """
        Initialize leverage manager.
        
        Args:
            max_leverage: Maximum allowed leverage per position
            risk_per_trade: Risk per trade as fraction of account
            max_portfolio_leverage: Maximum portfolio-wide leverage
            auto_deleverage: Whether to auto-deleverage on risk
        """
        self.max_leverage = max_leverage
        self.risk_per_trade = risk_per_trade
        self.max_portfolio_leverage = max_portfolio_leverage
        self.auto_deleverage = auto_deleverage
        
        self.calculator = LeverageCalculator()
        self.monitor = LiquidationMonitor()
        
        self._positions: Dict[str, LeveragePosition] = {}
        self._account_balance: float = 0.0
        self._total_margin_used: float = 0.0
        
        logger.info(
            "LeverageManager initialized: max_lev=%.1fx, risk=%.1f%%",
            max_leverage, risk_per_trade * 100
        )
    
    def update_balance(self, balance: float) -> None:
        """Update account balance."""
        self._account_balance = balance
    
    def calculate_entry(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss_price: float,
        account_balance: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate leveraged entry parameters.
        
        Returns position size, leverage, liquidation price, etc.
        """
        balance = account_balance or self._account_balance
        
        if balance <= 0:
            return {"error": "Invalid balance"}
        
        # Calculate safe leverage
        # For now, use max leverage (would use volatility in production)
        leverage = min(self.max_leverage, 10.0)
        
        # Calculate position size based on risk
        size = self.calculator.calculate_position_size(
            account_balance=balance,
            leverage=leverage,
            risk_per_trade=self.risk_per_trade,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price
        )
        
        # Calculate liquidation price
        liq_price = self.calculator.calculate_liquidation_price(
            entry_price=entry_price,
            leverage=leverage,
            side=side
        )
        
        # Calculate margin required
        position_value = size * entry_price
        margin_required = position_value / leverage
        
        # Check if within portfolio limits
        projected_leverage = (self._total_margin_used + margin_required) / balance
        if projected_leverage > self.max_portfolio_leverage:
            # Reduce size to stay within limits
            max_margin = balance * self.max_portfolio_leverage - self._total_margin_used
            if max_margin <= 0:
                return {"error": "Portfolio leverage limit reached"}
            
            position_value = max_margin * leverage
            size = position_value / entry_price
            margin_required = max_margin
        
        return {
            "symbol": symbol,
            "side": side,
            "leverage": leverage,
            "size": size,
            "position_value": position_value,
            "margin_required": margin_required,
            "liquidation_price": liq_price,
            "risk_amount": balance * self.risk_per_trade,
            "risk_pct": self.risk_per_trade * 100,
            "distance_to_liq_pct": abs(entry_price - liq_price) / entry_price * 100
        }
    
    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        leverage: float
    ) -> LeveragePosition:
        """
        Open a leveraged position.
        
        Args:
            symbol: Trading pair
            side: "long" or "short"
            entry_price: Entry price
            size: Position size
            leverage: Leverage multiplier
            
        Returns:
            LeveragePosition
        """
        # Calculate liquidation price
        liq_price = self.calculator.calculate_liquidation_price(
            entry_price=entry_price,
            leverage=leverage,
            side=side
        )
        
        # Calculate margin
        position_value = size * entry_price
        margin = position_value / leverage
        
        position = LeveragePosition(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_price=entry_price,
            size=size,
            leverage=leverage,
            margin_type=MarginType.ISOLATED,
            margin_used=margin,
            liquidation_price=liq_price,
            unrealized_pnl=0.0,
            margin_ratio=1.0,
            timestamp=datetime.now()
        )
        
        self._positions[symbol] = position
        self.monitor.update_position(position)
        self._total_margin_used += margin
        
        logger.info(
            "Position opened: %s %s %.4f @ %.2f with %.1fx leverage",
            symbol, side, size, entry_price, leverage
        )
        
        return position
    
    def update_price(self, symbol: str, current_price: float) -> Optional[LiquidationWarning]:
        """
        Update position with current price.
        
        Returns liquidation warning if risk detected.
        """
        if symbol not in self._positions:
            return None
        
        position = self._positions[symbol]
        position.current_price = current_price
        
        # Update unrealized PnL
        if position.side == "long":
            position.unrealized_pnl = (current_price - position.entry_price) * position.size
        else:
            position.unrealized_pnl = (position.entry_price - current_price) * position.size
        
        # Update margin ratio
        position_value = position.size * current_price
        position.margin_ratio = self.calculator.calculate_margin_ratio(
            position_value, position.margin_used, position.unrealized_pnl
        )
        
        # Update monitor
        self.monitor.update_position(position)
        
        # Check for liquidation risk
        warnings = self.monitor.check_all_positions()
        
        # Auto-deleverage if enabled
        if self.auto_deleverage and warnings:
            for warning in warnings:
                if warning.severity in ("critical", "imminent"):
                    self._auto_deleverage(warning)
        
        return warnings[0] if warnings else None
    
    def _auto_deleverage(self, warning: LiquidationWarning) -> None:
        """Automatically reduce position on liquidation risk."""
        symbol = warning.symbol
        
        if symbol not in self._positions:
            return
        
        position = self._positions[symbol]
        
        # Reduce position by 50%
        reduce_size = position.size * 0.5
        
        logger.warning(
            "Auto-deleveraging %s: reducing position by %.4f",
            symbol, reduce_size
        )
        
        # Update position
        position.size -= reduce_size
        position.margin_used = position.size * position.current_price / position.leverage
        
        # Update liquidation price
        position.liquidation_price = self.calculator.calculate_liquidation_price(
            position.entry_price,
            position.leverage,
            position.side
        )
    
    def close_position(self, symbol: str, exit_price: float) -> Dict[str, Any]:
        """
        Close a leveraged position.
        
        Returns trade summary.
        """
        if symbol not in self._positions:
            return {"error": "Position not found"}
        
        position = self._positions[symbol]
        
        # Calculate PnL
        if position.side == "long":
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size
        
        pnl_pct = pnl / position.margin_used * 100 if position.margin_used > 0 else 0
        
        # Remove position
        del self._positions[symbol]
        self.monitor.remove_position(symbol)
        self._total_margin_used -= position.margin_used
        
        summary = {
            "symbol": symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "size": position.size,
            "leverage": position.leverage,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "margin_used": position.margin_used,
            "roi": pnl / position.margin_used if position.margin_used > 0 else 0
        }
        
        logger.info(
            "Position closed: %s %s PnL=%.2f (%.1f%%)",
            symbol, position.side, pnl, pnl_pct
        )
        
        return summary
    
    def get_portfolio_leverage(self) -> Dict[str, float]:
        """Get portfolio leverage metrics."""
        if self._account_balance <= 0:
            return {"effective_leverage": 0, "margin_used": 0, "free_margin": 0}
        
        effective_leverage = self._total_margin_used / self._account_balance
        free_margin = self._account_balance - self._total_margin_used
        
        return {
            "effective_leverage": effective_leverage,
            "margin_used": self._total_margin_used,
            "free_margin": free_margin,
            "balance": self._account_balance,
            "n_positions": len(self._positions)
        }
    
    def get_all_positions(self) -> List[LeveragePosition]:
        """Get all open positions."""
        return list(self._positions.values())
    
    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized PnL across all positions."""
        return sum(p.unrealized_pnl for p in self._positions.values())
    
    def get_recommendations(self) -> List[LeverageRecommendation]:
        """Get leverage adjustment recommendations."""
        recommendations = []
        
        portfolio = self.get_portfolio_leverage()
        
        # Check if portfolio leverage is too high
        if portfolio["effective_leverage"] > self.max_portfolio_leverage * 0.8:
            recommendations.append(LeverageRecommendation(
                symbol="PORTFOLIO",
                action=LeverageAction.DELEVERAGE,
                current_leverage=portfolio["effective_leverage"],
                target_leverage=self.max_portfolio_leverage * 0.6,
                reason="Portfolio leverage approaching limit",
                risk_score=0.8,
                estimated_impact={"risk_reduction": 0.2},
                timestamp=datetime.now()
            ))
        
        # Check individual positions
        for symbol, position in self._positions.items():
            margin_ratio = position.margin_ratio
            
            if margin_ratio < 0.2:
                recommendations.append(LeverageRecommendation(
                    symbol=symbol,
                    action=LeverageAction.CLOSE,
                    current_leverage=position.leverage,
                    target_leverage=0,
                    reason=f"Margin ratio critical: {margin_ratio:.2%}",
                    risk_score=0.95,
                    estimated_impact={"risk_reduction": 1.0},
                    timestamp=datetime.now()
                ))
            elif margin_ratio < 0.4:
                recommendations.append(LeverageRecommendation(
                    symbol=symbol,
                    action=LeverageAction.DECREASE,
                    current_leverage=position.leverage,
                    target_leverage=position.leverage * 0.5,
                    reason=f"Margin ratio low: {margin_ratio:.2%}",
                    risk_score=0.7,
                    estimated_impact={"risk_reduction": 0.3},
                    timestamp=datetime.now()
                ))
        
        return recommendations
