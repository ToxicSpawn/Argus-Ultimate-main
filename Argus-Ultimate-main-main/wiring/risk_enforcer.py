"""
Risk Enforcer - Live Risk Management
Wires risk rules to position management for automatic enforcement
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from wiring.realtime_position_tracker import get_position_tracker, PortfolioSnapshot
from wiring.exchange_connector import get_exchange_manager

logger = logging.getLogger(__name__)


@dataclass
class RiskRule:
    """Risk rule definition"""
    name: str
    type: str  # 'limit', 'stop', 'alert'
    threshold: float
    action: str  # 'notify', 'reduce', 'close_all', 'pause'
    is_active: bool = True


class RiskEnforcer:
    """
    Real-time risk rule enforcement
    Automatically manages positions when risk limits breached
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Risk rules
        self.rules: List[RiskRule] = [
            RiskRule("daily_loss", "limit", 0.05, "close_all"),      # 5% daily loss
            RiskRule("max_drawdown", "limit", 0.10, "pause"),         # 10% drawdown
            RiskRule("position_concentration", "limit", 0.15, "reduce"), # 15% single position
            RiskRule("total_exposure", "limit", 0.50, "reduce"),        # 50% total exposure
            RiskRule("volatility_spike", "alert", 3.0, "notify"),      # 3σ volatility
        ]
        
        # State
        self.is_enforcing = False
        self.trading_paused = False
        self.pause_reason: Optional[str] = None
        
        # Callbacks
        self.alert_callbacks: List[Callable] = []
        self.action_callbacks: List[Callable] = []
        
        # Statistics
        self.violations_count = 0
        self.last_violation: Optional[str] = None
        
        logger.info("🛡️ Risk enforcer initialized")
    
    async def start(self):
        """Start risk enforcement"""
        self.is_enforcing = True
        
        # Register with position tracker
        tracker = get_position_tracker()
        tracker.register_risk_callback(self._on_risk_event)
        
        # Start enforcement loop
        asyncio.create_task(self._enforcement_loop())
        
        logger.info("✅ Risk enforcement active")
        logger.info(f"   Rules: {len(self.rules)} configured")
        for rule in self.rules:
            logger.info(f"     - {rule.name}: {rule.threshold*100:.1f}% → {rule.action}")
    
    async def stop(self):
        """Stop risk enforcement"""
        self.is_enforcing = False
        logger.info("⏹️ Risk enforcement stopped")
    
    async def _enforcement_loop(self):
        """Continuous risk monitoring"""
        while self.is_enforcing:
            try:
                # Get current portfolio
                tracker = get_position_tracker()
                portfolio = await tracker.get_portfolio_snapshot()
                
                # Check all rules
                for rule in self.rules:
                    if not rule.is_active:
                        continue
                    
                    violated = await self._check_rule(rule, portfolio)
                    
                    if violated:
                        await self._enforce_rule(rule, portfolio)
                
            except Exception as e:
                logger.error(f"Risk enforcement error: {e}")
            
            await asyncio.sleep(1)  # Check every second
    
    async def _check_rule(self, rule: RiskRule, portfolio: PortfolioSnapshot) -> bool:
        """Check if rule is violated"""
        if rule.name == "daily_loss":
            if portfolio.daily_pnl < 0:
                loss_pct = abs(portfolio.daily_pnl) / (portfolio.total_value + abs(portfolio.daily_pnl))
                return loss_pct > rule.threshold
        
        elif rule.name == "max_drawdown":
            # Get from tracker
            tracker = get_position_tracker()
            if tracker.max_drawdown > rule.threshold:
                return True
        
        elif rule.name == "position_concentration":
            for pos in portfolio.positions:
                if pos.market_value / portfolio.total_value > rule.threshold:
                    return True
        
        elif rule.name == "total_exposure":
            if portfolio.total_exposure / portfolio.total_value > rule.threshold:
                return True
        
        elif rule.name == "volatility_spike":
            # Would need volatility calculation
            pass
        
        return False
    
    async def _enforce_rule(self, rule: RiskRule, portfolio: PortfolioSnapshot):
        """Enforce risk rule action"""
        self.violations_count += 1
        self.last_violation = rule.name
        
        logger.warning(f"🚨 RISK VIOLATION: {rule.name} (threshold: {rule.threshold*100:.1f}%)")
        
        # Send alerts
        for callback in self.alert_callbacks:
            await callback(rule, portfolio)
        
        # Execute action
        if rule.action == "close_all":
            await self._action_close_all(portfolio)
        
        elif rule.action == "pause":
            await self._action_pause_trading(rule.name)
        
        elif rule.action == "reduce":
            await self._action_reduce_positions(portfolio)
        
        elif rule.action == "notify":
            logger.info(f"📢 Risk alert: {rule.name}")
        
        # Notify action taken
        for callback in self.action_callbacks:
            await callback(rule, rule.action, portfolio)
    
    async def _action_close_all(self, portfolio: PortfolioSnapshot):
        """Emergency close all positions"""
        logger.error("🚨 EMERGENCY: Closing ALL positions!")
        
        tracker = get_position_tracker()
        await tracker.flatten_all_positions()
        
        self.trading_paused = True
        self.pause_reason = "emergency_close_all"
    
    async def _action_pause_trading(self, reason: str):
        """Pause trading"""
        if not self.trading_paused:
            logger.warning(f"⏸️ Trading PAUSED: {reason}")
            self.trading_paused = True
            self.pause_reason = reason
    
    async def _action_reduce_positions(self, portfolio: PortfolioSnapshot):
        """Reduce position sizes"""
        logger.info("📉 Reducing positions by 50%...")
        
        manager = get_exchange_manager()
        tracker = get_position_tracker()
        
        for pos in portfolio.positions:
            try:
                reduce_amount = pos.amount * 0.5
                
                if pos.side == "long":
                    await manager.submit_order(
                        exchange=pos.exchange,
                        symbol=pos.symbol,
                        side="sell",
                        amount=reduce_amount,
                        order_type="market"
                    )
                else:  # short
                    await manager.submit_order(
                        exchange=pos.exchange,
                        symbol=pos.symbol,
                        side="buy",
                        amount=reduce_amount,
                        order_type="market"
                    )
                
                logger.info(f"  Reduced {pos.symbol} by 50%")
                
            except Exception as e:
                logger.error(f"  Failed to reduce {pos.symbol}: {e}")
    
    async def _on_risk_event(self, event_type: str, portfolio: PortfolioSnapshot):
        """Handle risk events from position tracker"""
        if event_type == "daily_loss_limit":
            await self._action_close_all(portfolio)
        
        elif event_type == "max_drawdown":
            await self._action_pause_trading("max_drawdown")
    
    def resume_trading(self):
        """Manually resume trading after pause"""
        if self.trading_paused:
            logger.info(f"▶️ Trading RESUMED (was paused due to: {self.pause_reason})")
            self.trading_paused = False
            self.pause_reason = None
    
    def add_rule(self, rule: RiskRule):
        """Add custom risk rule"""
        self.rules.append(rule)
        logger.info(f"✅ Added risk rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove risk rule"""
        self.rules = [r for r in self.rules if r.name != rule_name]
    
    def get_status(self) -> Dict:
        """Get enforcer status"""
        return {
            "is_enforcing": self.is_enforcing,
            "trading_paused": self.trading_paused,
            "pause_reason": self.pause_reason,
            "active_rules": len([r for r in self.rules if r.is_active]),
            "total_violations": self.violations_count,
            "last_violation": self.last_violation
        }
    
    def register_alert_callback(self, callback: Callable):
        """Register alert callback"""
        self.alert_callbacks.append(callback)
    
    def register_action_callback(self, callback: Callable):
        """Register action callback"""
        self.action_callbacks.append(callback)


# Global instance
_risk_enforcer: Optional[RiskEnforcer] = None


def get_risk_enforcer(config: Dict = None) -> RiskEnforcer:
    """Get singleton risk enforcer"""
    global _risk_enforcer
    if _risk_enforcer is None:
        _risk_enforcer = RiskEnforcer(config)
    return _risk_enforcer


async def init_risk_enforcement(config: Dict = None):
    """Initialize risk enforcement"""
    enforcer = get_risk_enforcer(config)
    await enforcer.start()
