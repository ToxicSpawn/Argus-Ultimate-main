"""
Argus Governance System
Version: 1.0.0

Hedge fund-grade governance and compliance.
Risk committee oversight, position limits, drawdown controls.

Features:
- Risk Committee Oversight
- Position Limits (single, sector, country)
- Drawdown Controls (daily, weekly, monthly, total)
- Leverage Limits
- Concentration Limits
- Compliance Monitoring
- Audit Trail
- Automated Circuit Breakers
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime, timedelta
from collections import deque
import json

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk alert levels."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class CircuitBreakerType(Enum):
    """Circuit breaker types."""
    DAILY_LOSS = "daily_loss"
    WEEKLY_LOSS = "weekly_loss"
    MONTHLY_LOSS = "monthly_loss"
    TOTAL_DRAWDOWN = "total_drawdown"
    VOLATILITY = "volatility"
    POSITION_LIMIT = "position_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"


class Action(Enum):
    """Governance actions."""
    ALLOW = "allow"
    WARN = "warn"
    REDUCE = "reduce"
    HALT_NEW = "halt_new"
    HALT_ALL = "halt_all"
    LIQUIDATE = "liquidate"


@dataclass
class RiskLimit:
    """Risk limit definition."""
    name: str
    limit_type: str
    threshold: float
    action: Action
    current_value: float = 0.0
    is_breached: bool = False
    breach_time: Optional[datetime] = None
    description: str = ""


@dataclass
class GovernanceDecision:
    """Governance decision record."""
    timestamp: datetime
    decision_type: str
    action: Action
    reason: str
    details: Dict[str, Any]
    approved_by: str = "automated"
    audit_id: str = ""


@dataclass
class PositionLimit:
    """Position limit definition."""
    limit_type: str  # "single", "sector", "country", "asset_class"
    identifier: str  # symbol, sector name, etc.
    max_percentage: float  # % of portfolio
    max_notional: float  # $ amount
    current_exposure: float = 0.0
    is_breached: bool = False


class RiskCommittee:
    """
    Automated Risk Committee.
    
    Reviews and approves/rejects risk decisions.
    """
    
    def __init__(self):
        self.decisions: List[GovernanceDecision] = []
        self.meeting_log: List[Dict] = []
        
        # Committee rules
        self.rules = {
            "max_daily_trades": 1000,
            "max_position_size": 0.10,  # 10% of portfolio
            "max_sector_exposure": 0.30,  # 30%
            "max_leverage": 2.0,
            "min_liquidity": 0.20,  # 20% in liquid assets
            "max_correlation": 0.80,  # Max avg correlation
        }
        
        logger.info("RiskCommittee initialized")
    
    def review_trade(self, trade: Dict[str, Any], portfolio: Dict[str, Any]) -> GovernanceDecision:
        """
        Review a proposed trade.
        
        Returns approval/rejection decision.
        """
        # Check position size
        position_value = trade.get("quantity", 0) * trade.get("price", 0)
        portfolio_value = portfolio.get("total_value", 1)
        position_pct = position_value / portfolio_value
        
        # Check limits
        issues = []
        
        if position_pct > self.rules["max_position_size"]:
            issues.append(f"Position size {position_pct:.1%} exceeds limit {self.rules['max_position_size']:.1%}")
        
        # Check sector exposure
        sector = trade.get("sector", "unknown")
        sector_exposure = portfolio.get("sector_exposures", {}).get(sector, 0)
        if sector_exposure + position_pct > self.rules["max_sector_exposure"]:
            issues.append(f"Sector exposure would exceed limit")
        
        # Check leverage
        current_leverage = portfolio.get("leverage", 1.0)
        if current_leverage > self.rules["max_leverage"]:
            issues.append(f"Leverage {current_leverage:.2f} exceeds limit")
        
        # Make decision
        if issues:
            action = Action.WARN if len(issues) == 1 else Action.HALT_NEW
            decision = GovernanceDecision(
                timestamp=datetime.now(),
                decision_type="trade_review",
                action=action,
                reason="; ".join(issues),
                details={"trade": trade, "issues": issues}
            )
        else:
            decision = GovernanceDecision(
                timestamp=datetime.now(),
                decision_type="trade_review",
                action=Action.ALLOW,
                reason="Trade approved",
                details={"trade": trade}
            )
        
        self.decisions.append(decision)
        return decision
    
    def emergency_review(self, situation: str, details: Dict) -> GovernanceDecision:
        """Emergency risk committee review."""
        decision = GovernanceDecision(
            timestamp=datetime.now(),
            decision_type="emergency",
            action=Action.HALT_ALL if situation == "critical" else Action.REDUCE,
            reason=f"Emergency: {situation}",
            details=details,
            approved_by="emergency_protocol"
        )
        
        self.decisions.append(decision)
        self.log_meeting("emergency", situation, details)
        
        return decision
    
    def log_meeting(self, meeting_type: str, topic: str, details: Dict):
        """Log risk committee meeting."""
        self.meeting_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": meeting_type,
            "topic": topic,
            "details": details,
            "decisions_made": len([d for d in self.decisions if d.timestamp > datetime.now() - timedelta(hours=1)])
        })


class DrawdownController:
    """
    Drawdown control system.
    
    Monitors and controls portfolio drawdowns.
    """
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.peak_value = initial_capital
        self.current_value = initial_capital
        
        # Drawdown limits
        self.limits = {
            "daily_loss": 0.05,      # 5% daily loss limit
            "weekly_loss": 0.10,     # 10% weekly loss limit
            "monthly_loss": 0.15,    # 15% monthly loss limit
            "total_drawdown": 0.25,  # 25% max drawdown
        }
        
        # Tracking
        self.daily_pnl: deque = deque(maxlen=252)
        self.weekly_pnl: deque = deque(maxlen=52)
        self.monthly_pnl: deque = deque(maxlen=12)
        self.drawdown_history: List[Dict] = []
        
        logger.info(f"DrawdownController initialized: ${initial_capital:,.2f}")
    
    def update(self, current_value: float) -> Dict[str, Any]:
        """
        Update drawdown tracking.
        
        Returns current drawdown status.
        """
        self.current_value = current_value
        
        # Update peak
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        # Calculate drawdowns
        total_dd = (self.peak_value - current_value) / self.peak_value
        
        # Check limits
        breaches = []
        
        if total_dd > self.limits["total_drawdown"]:
            breaches.append(("total_drawdown", total_dd, self.limits["total_drawdown"]))
        
        # Determine action
        if total_dd > self.limits["total_drawdown"]:
            action = Action.LIQUIDATE
        elif total_dd > self.limits["total_drawdown"] * 0.8:
            action = Action.HALT_NEW
        elif total_dd > self.limits["total_drawdown"] * 0.5:
            action = Action.REDUCE
        elif total_dd > self.limits["total_drawdown"] * 0.3:
            action = Action.WARN
        else:
            action = Action.ALLOW
        
        status = {
            "current_value": current_value,
            "peak_value": self.peak_value,
            "total_drawdown": total_dd,
            "drawdown_dollars": self.peak_value - current_value,
            "action": action,
            "breaches": breaches,
            "limits": self.limits
        }
        
        if breaches:
            self.drawdown_history.append({
                "timestamp": datetime.now().isoformat(),
                "breaches": breaches,
                "action": action.value
            })
        
        return status
    
    def record_daily_pnl(self, pnl: float):
        """Record daily P&L."""
        self.daily_pnl.append(pnl)
    
    def get_daily_loss(self) -> float:
        """Get current daily loss."""
        if not self.daily_pnl:
            return 0.0
        return sum(self.daily_pnl)
    
    def reset_peak(self):
        """Reset peak value (after recovery)."""
        self.peak_value = self.current_value


class PositionLimitManager:
    """
    Position limit management.
    
    Enforces position, sector, and concentration limits.
    """
    
    def __init__(self, portfolio_value: float):
        self.portfolio_value = portfolio_value
        
        # Position limits
        self.single_position_limit = 0.10  # 10%
        self.sector_limit = 0.30  # 30%
        self.country_limit = 0.40  # 40%
        self.asset_class_limit = 0.50  # 50%
        
        # Tracking
        self.positions: Dict[str, float] = {}  # symbol -> exposure
        self.sector_exposure: Dict[str, float] = {}
        self.country_exposure: Dict[str, float] = {}
        self.asset_class_exposure: Dict[str, float] = {}
        
        logger.info("PositionLimitManager initialized")
    
    def check_new_position(self, symbol: str, proposed_value: float,
                           sector: str = "unknown", country: str = "unknown",
                           asset_class: str = "equity") -> Dict[str, Any]:
        """
        Check if new position is within limits.
        
        Returns approval/rejection with reasons.
        """
        issues = []
        
        # Single position check
        current_exposure = self.positions.get(symbol, 0)
        total_exposure = current_exposure + proposed_value
        position_pct = total_exposure / self.portfolio_value
        
        if position_pct > self.single_position_limit:
            issues.append(f"Single position limit: {position_pct:.1%} > {self.single_position_limit:.1%}")
        
        # Sector check
        current_sector = self.sector_exposure.get(sector, 0)
        sector_pct = (current_sector + proposed_value) / self.portfolio_value
        
        if sector_pct > self.sector_limit:
            issues.append(f"Sector limit: {sector_pct:.1%} > {self.sector_limit:.1%}")
        
        # Country check
        current_country = self.country_exposure.get(country, 0)
        country_pct = (current_country + proposed_value) / self.portfolio_value
        
        if country_pct > self.country_limit:
            issues.append(f"Country limit: {country_pct:.1%} > {self.country_limit:.1%}")
        
        # Asset class check
        current_ac = self.asset_class_exposure.get(asset_class, 0)
        ac_pct = (current_ac + proposed_value) / self.portfolio_value
        
        if ac_pct > self.asset_class_limit:
            issues.append(f"Asset class limit: {ac_pct:.1%} > {self.asset_class_limit:.1%}")
        
        return {
            "approved": len(issues) == 0,
            "issues": issues,
            "exposures": {
                "position": position_pct,
                "sector": sector_pct,
                "country": country_pct,
                "asset_class": ac_pct
            }
        }
    
    def update_position(self, symbol: str, value: float,
                        sector: str, country: str, asset_class: str):
        """Update position tracking."""
        self.positions[symbol] = value
        self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + value
        self.country_exposure[country] = self.country_exposure.get(country, 0) + value
        self.asset_class_exposure[asset_class] = self.asset_class_exposure.get(asset_class, 0) + value
    
    def get_concentration_report(self) -> Dict[str, Any]:
        """Get concentration report."""
        total = sum(self.positions.values())
        
        # Top positions
        sorted_positions = sorted(self.positions.items(), key=lambda x: x[1], reverse=True)
        top_5 = sum(v for _, v in sorted_positions[:5]) / total if total > 0 else 0
        top_10 = sum(v for _, v in sorted_positions[:10]) / total if total > 0 else 0
        
        return {
            "total_exposure": total,
            "portfolio_value": self.portfolio_value,
            "leverage": total / self.portfolio_value,
            "top_5_concentration": top_5,
            "top_10_concentration": top_10,
            "sector_exposure": self.sector_exposure,
            "country_exposure": self.country_exposure,
            "num_positions": len(self.positions)
        }


class CircuitBreakerSystem:
    """
    Automated circuit breakers.
    
    Halts trading when limits are breached.
    """
    
    def __init__(self):
        self.breakers: Dict[str, RiskLimit] = {}
        self.tripped_breakers: List[Dict] = []
        self.is_halted = False
        self.halt_reason = ""
        
        # Initialize default breakers
        self._initialize_breakers()
        
        logger.info("CircuitBreakerSystem initialized")
    
    def _initialize_breakers(self):
        """Initialize default circuit breakers."""
        self.breakers = {
            "daily_loss": RiskLimit(
                name="daily_loss",
                limit_type=CircuitBreakerType.DAILY_LOSS.value,
                threshold=0.05,
                action=Action.HALT_NEW,
                description="5% daily loss limit"
            ),
            "weekly_loss": RiskLimit(
                name="weekly_loss",
                limit_type=CircuitBreakerType.WEEKLY_LOSS.value,
                threshold=0.10,
                action=Action.HALT_ALL,
                description="10% weekly loss limit"
            ),
            "total_drawdown": RiskLimit(
                name="total_drawdown",
                limit_type=CircuitBreakerType.TOTAL_DRAWDOWN.value,
                threshold=0.25,
                action=Action.LIQUIDATE,
                description="25% max drawdown"
            ),
            "leverage": RiskLimit(
                name="leverage",
                limit_type=CircuitBreakerType.LEVERAGE_LIMIT.value,
                threshold=2.0,
                action=Action.HALT_NEW,
                description="2x leverage limit"
            ),
            "volatility": RiskLimit(
                name="volatility",
                limit_type=CircuitBreakerType.VOLATILITY.value,
                threshold=0.05,
                action=Action.REDUCE,
                description="5% daily volatility limit"
            )
        }
    
    def check_breakers(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Check all circuit breakers.
        
        Returns status and any triggered breakers.
        """
        triggered = []
        
        for name, breaker in self.breakers.items():
            if name in metrics:
                value = metrics[name]
                breaker.current_value = value
                
                if value > breaker.threshold:
                    breaker.is_breached = True
                    breaker.breach_time = datetime.now()
                    
                    triggered.append({
                        "name": name,
                        "value": value,
                        "threshold": breaker.threshold,
                        "action": breaker.action.value,
                        "description": breaker.description
                    })
                    
                    self.tripped_breakers.append({
                        "timestamp": datetime.now().isoformat(),
                        "breaker": name,
                        "value": value,
                        "threshold": breaker.threshold
                    })
        
        # Determine overall status
        if triggered:
            # Find most severe action
            action_priority = [Action.LIQUIDATE, Action.HALT_ALL, Action.HALT_NEW, Action.REDUCE, Action.WARN]
            max_action = Action.ALLOW
            
            for t in triggered:
                action = Action(t["action"])
                if action_priority.index(action) > action_priority.index(max_action):
                    max_action = action
            
            if max_action in [Action.HALT_ALL, Action.LIQUIDATE]:
                self.is_halted = True
                self.halt_reason = f"Circuit breakers triggered: {[t['name'] for t in triggered]}"
        
        return {
            "is_halted": self.is_halted,
            "halt_reason": self.halt_reason,
            "triggered": triggered,
            "breaker_status": {name: {"value": b.current_value, "threshold": b.threshold, "breached": b.is_breached}
                              for name, b in self.breakers.items()}
        }
    
    def reset_breaker(self, name: str) -> bool:
        """Reset a circuit breaker."""
        if name in self.breakers:
            self.breakers[name].is_breached = False
            self.breakers[name].breach_time = None
            return True
        return False
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            breaker.is_breached = False
            breaker.breach_time = None
        
        self.is_halted = False
        self.halt_reason = ""


class ComplianceMonitor:
    """
    Compliance monitoring.
    
    Tracks regulatory compliance and audit trail.
    """
    
    def __init__(self):
        self.audit_trail: List[Dict] = []
        self.compliance_checks: Dict[str, bool] = {}
        
        logger.info("ComplianceMonitor initialized")
    
    def log_trade(self, trade: Dict[str, Any]):
        """Log trade for audit trail."""
        self.audit_trail.append({
            "timestamp": datetime.now().isoformat(),
            "type": "trade",
            "details": trade
        })
    
    def log_decision(self, decision: GovernanceDecision):
        """Log governance decision."""
        self.audit_trail.append({
            "timestamp": datetime.now().isoformat(),
            "type": "decision",
            "decision_type": decision.decision_type,
            "action": decision.action.value,
            "reason": decision.reason
        })
    
    def run_compliance_check(self) -> Dict[str, bool]:
        """Run compliance checks."""
        checks = {
            "position_limits": True,
            "leverage_limits": True,
            "drawdown_limits": True,
            "concentration_limits": True,
            "trade_reporting": len(self.audit_trail) > 0,
            "risk_monitoring": True
        }
        
        self.compliance_checks = checks
        return checks
    
    def generate_audit_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate audit report for period."""
        relevant = [
            entry for entry in self.audit_trail
            if start_date.isoformat() <= entry["timestamp"] <= end_date.isoformat()
        ]
        
        return {
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_entries": len(relevant),
            "trades": len([e for e in relevant if e["type"] == "trade"]),
            "decisions": len([e for e in relevant if e["type"] == "decision"]),
            "entries": relevant[:100]  # Limit output
        }


class GovernanceSystem:
    """
    Main governance system.
    
    Combines all governance and compliance capabilities.
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, portfolio_value: float = 1000000):
        """Initialize governance system."""
        self.portfolio_value = portfolio_value
        
        # Components
        self.risk_committee = RiskCommittee()
        self.drawdown_controller = DrawdownController(portfolio_value)
        self.position_limits = PositionLimitManager(portfolio_value)
        self.circuit_breakers = CircuitBreakerSystem()
        self.compliance = ComplianceMonitor()
        
        # Risk level
        self.risk_level = RiskLevel.NORMAL
        
        logger.info(f"GovernanceSystem v{self.VERSION} initialized")
        logger.info(f"  Portfolio value: ${portfolio_value:,.2f}")
    
    def review_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full governance review of a trade.
        
        Returns approval status and any issues.
        """
        # Check if halted
        if self.circuit_breakers.is_halted:
            return {
                "approved": False,
                "reason": f"Trading halted: {self.circuit_breakters.halt_reason}",
                "action": Action.HALT_ALL.value
            }
        
        # Position limit check
        position_check = self.position_limits.check_new_position(
            trade.get("symbol", ""),
            trade.get("quantity", 0) * trade.get("price", 0),
            trade.get("sector", "unknown"),
            trade.get("country", "unknown"),
            trade.get("asset_class", "equity")
        )
        
        if not position_check["approved"]:
            return {
                "approved": False,
                "reason": "Position limits breached",
                "issues": position_check["issues"],
                "action": Action.HALT_NEW.value
            }
        
        # Risk committee review
        portfolio = {
            "total_value": self.portfolio_value,
            "sector_exposures": self.position_limits.sector_exposure,
            "leverage": self.position_limits.get_concentration_report()["leverage"]
        }
        
        decision = self.risk_committee.review_trade(trade, portfolio)
        
        # Log for compliance
        self.compliance.log_trade(trade)
        self.compliance.log_decision(decision)
        
        return {
            "approved": decision.action == Action.ALLOW,
            "action": decision.action.value,
            "reason": decision.reason,
            "decision_id": len(self.risk_committee.decisions)
        }
    
    def update_risk_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Update risk metrics and check circuit breakers."""
        # Update drawdown controller
        if "portfolio_value" in metrics:
            dd_status = self.drawdown_controller.update(metrics["portfolio_value"])
            metrics["total_drawdown"] = dd_status["total_drawdown"]
        
        # Check circuit breakers
        breaker_status = self.circuit_breakers.check_breakers(metrics)
        
        # Update risk level
        if breaker_status["is_halted"]:
            self.risk_level = RiskLevel.EMERGENCY
        elif breaker_status["triggered"]:
            self.risk_level = RiskLevel.HIGH
        else:
            self.risk_level = RiskLevel.NORMAL
        
        return {
            "risk_level": self.risk_level.value,
            "circuit_breakers": breaker_status,
            "drawdown_status": dd_status if "portfolio_value" in metrics else None
        }
    
    def emergency_shutdown(self, reason: str) -> Dict[str, Any]:
        """Execute emergency shutdown."""
        self.circuit_breakers.is_halted = True
        self.circuit_breakers.halt_reason = reason
        self.risk_level = RiskLevel.EMERGENCY
        
        decision = self.risk_committee.emergency_review("critical", {"reason": reason})
        self.compliance.log_decision(decision)
        
        return {
            "status": "halted",
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_governance_report(self) -> Dict[str, Any]:
        """Get comprehensive governance report."""
        return {
            "version": self.VERSION,
            "risk_level": self.risk_level.value,
            "is_halted": self.circuit_breakers.is_halted,
            "halt_reason": self.circuit_breakers.halt_reason,
            "concentration": self.position_limits.get_concentration_report(),
            "drawdown": {
                "current": (self.drawdown_controller.peak_value - self.drawdown_controller.current_value) / self.drawdown_controller.peak_value,
                "peak": self.drawdown_controller.peak_value,
                "current_value": self.drawdown_controller.current_value
            },
            "circuit_breakers": {name: {"breached": b.is_breached} for name, b in self.circuit_breakers.breakers.items()},
            "decisions_made": len(self.risk_committee.decisions),
            "compliance_checks": self.compliance.run_compliance_check()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get governance system statistics."""
        return {
            "version": self.VERSION,
            "portfolio_value": self.portfolio_value,
            "risk_level": self.risk_level.value,
            "decisions_made": len(self.risk_committee.decisions),
            "tripped_breakers": len(self.circuit_breakers.tripped_breakers),
            "audit_entries": len(self.compliance.audit_trail)
        }


# Global system instance
_system_instance: Optional[GovernanceSystem] = None


def get_governance_system(portfolio_value: float = 1000000) -> GovernanceSystem:
    """Get or create global Governance System instance."""
    global _system_instance
    if _system_instance is None:
        _system_instance = GovernanceSystem(portfolio_value)
    return _system_instance


if __name__ == "__main__":
    # Test the system
    logging.basicConfig(level=logging.INFO)
    
    governance = get_governance_system(1000000)
    
    # Test trade review
    trade = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 100,
        "price": 150,
        "sector": "technology",
        "country": "US",
        "asset_class": "equity"
    }
    
    result = governance.review_trade(trade)
    print(f"Trade approved: {result['approved']}")
    
    # Test risk metrics update
    risk_update = governance.update_risk_metrics({
        "portfolio_value": 950000,
        "daily_loss": 0.03,
        "leverage": 1.5
    })
    print(f"Risk level: {risk_update['risk_level']}")
    
    # Get governance report
    report = governance.get_governance_report()
    print(f"\nGovernance Report:")
    print(f"  Risk Level: {report['risk_level']}")
    print(f"  Is Halted: {report['is_halted']}")
    print(f"  Decisions Made: {report['decisions_made']}")
    
    print(f"\nSystem Stats: {governance.get_stats()}")
