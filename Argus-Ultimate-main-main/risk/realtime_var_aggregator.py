"""
Real-time VaR Aggregator v2.0
==============================
Streaming Value at Risk calculation for Argus Ultimate.

Provides:
- Real-time VaR calculation on every tick
- Portfolio-level VaR aggregation
- Correlation-aware risk
- Multiple confidence levels (95%, 99%, 99.9%)
- Risk budget tracking
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PositionRisk:
    """Risk metrics for a single position."""
    symbol: str
    position_value: float
    var_95: float
    var_99: float
    var_99_9: float
    expected_shortfall_99: float
    marginal_var: float
    component_var: float
    contribution_pct: float


@dataclass
class PortfolioRisk:
    """Aggregated portfolio risk metrics."""
    timestamp: datetime
    total_var_95: float
    total_var_99: float
    total_var_99_9: float
    expected_shortfall_99: float
    portfolio_value: float
    risk_budget_used_pct: float
    max_drawdown_estimate: float
    sharpe_estimate: float
    position_risks: Dict[str, PositionRisk]
    correlation_regime: str


@dataclass
class RiskAlert:
    """Risk alert."""
    timestamp: datetime
    alert_type: str
    severity: str  # "info", "warning", "critical"
    message: str
    metric_name: str
    metric_value: float
    threshold: float


class StreamingVaRCalculator:
    """
    Streaming VaR calculator that updates on every tick.
    
    Uses exponentially weighted moving statistics for real-time updates.
    """
    
    def __init__(
        self,
        confidence_levels: List[float] = None,
        lookback_window: int = 1000,
        decay_factor: float = 0.94
    ) -> None:
        """
        Initialize streaming VaR calculator.
        
        Args:
            confidence_levels: VaR confidence levels
            lookback_window: Historical lookback window
            decay_factor: EWMA decay factor
        """
        self.confidence_levels = confidence_levels or [0.95, 0.99, 0.999]
        self.lookback_window = lookback_window
        self.decay_factor = decay_factor
        
        # Per-symbol statistics
        self._returns: Dict[str, Deque[float]] = {}
        self._means: Dict[str, float] = {}
        self._vars: Dict[str, float] = {}
        self._last_update: Dict[str, datetime] = {}
    
    def update(self, symbol: str, return_value: float) -> Dict[str, float]:
        """
        Update with new return observation.
        
        Args:
            symbol: Asset symbol
            return_value: Return observation
            
        Returns:
            Dictionary of VaR values
        """
        # Initialize if new symbol
        if symbol not in self._returns:
            self._returns[symbol] = deque(maxlen=self.lookback_window)
            self._means[symbol] = 0.0
            self._vars[symbol] = 0.01  # Initial variance estimate
        
        # Add return
        self._returns[symbol].append(return_value)
        
        # Update EWMA statistics
        old_mean = self._means[symbol]
        old_var = self._vars[symbol]
        
        n = len(self._returns[symbol])
        if n >= 2:
            # EWMA update
            self._means[symbol] = (
                self.decay_factor * old_mean + (1 - self.decay_factor) * return_value
            )
            
            var_update = (return_value - old_mean) ** 2
            self._vars[symbol] = (
                self.decay_factor * old_var + (1 - self.decay_factor) * var_update
            )
        
        self._last_update[symbol] = datetime.now()
        
        # Calculate VaR at each confidence level
        return self.calculate_var(symbol)
    
    def calculate_var(self, symbol: str) -> Dict[str, float]:
        """
        Calculate VaR for a symbol at all confidence levels.
        
        Returns:
            Dictionary with VaR values
        """
        if symbol not in self._vars:
            return {f"var_{int(cl*100)}": 0.0 for cl in self.confidence_levels}
        
        mean = self._means[symbol]
        std = np.sqrt(self._vars[symbol])
        
        # Z-scores for confidence levels
        from scipy import stats as scipy_stats
        
        result = {}
        for cl in self.confidence_levels:
            z = scipy_stats.norm.ppf(1 - cl)
            var = -(mean + z * std)
            result[f"var_{int(cl*100)}"] = float(var)
        
        # Expected Shortfall (CVaR) at 99%
        z_99 = scipy_stats.norm.ppf(0.99)
        es_99 = -(mean - std * scipy_stats.norm.pdf(z_99) / 0.01)
        result["expected_shortfall_99"] = float(es_99)
        
        return result
    
    def get_volatility(self, symbol: str) -> float:
        """Get current volatility estimate."""
        if symbol not in self._vars:
            return 0.0
        return float(np.sqrt(self._vars[symbol]))
    
    def get_all_var(self) -> Dict[str, Dict[str, float]]:
        """Get VaR for all tracked symbols."""
        return {symbol: self.calculate_var(symbol) for symbol in self._returns}


class CorrelationTracker:
    """
    Tracks and updates correlation matrix in real-time.
    """
    
    def __init__(self, decay_factor: float = 0.95) -> None:
        """
        Initialize correlation tracker.
        
        Args:
            decay_factor: EWMA decay factor for correlation updates
        """
        self.decay_factor = decay_factor
        self._symbols: List[str] = []
        self._returns: Dict[str, Deque[float]] = {}
        self._correlation_matrix: Optional[np.ndarray] = None
    
    def add_symbol(self, symbol: str) -> None:
        """Add symbol to tracking."""
        if symbol not in self._symbols:
            self._symbols.append(symbol)
            self._returns[symbol] = deque(maxlen=500)
    
    def update(self, returns: Dict[str, float]) -> np.ndarray:
        """
        Update correlations with new returns.
        
        Args:
            returns: Dictionary of symbol -> return
            
        Returns:
            Updated correlation matrix
        """
        # Add returns
        for symbol, ret in returns.items():
            if symbol in self._returns:
                self._returns[symbol].append(ret)
        
        # Calculate correlation matrix if we have enough data
        n = len(self._symbols)
        if n < 2:
            self._correlation_matrix = np.eye(n)
            return self._correlation_matrix
        
        # Get aligned returns
        min_len = min(len(r) for r in self._returns.values() if len(r) > 0)
        if min_len < 10:
            self._correlation_matrix = np.eye(n)
            return self._correlation_matrix
        
        # Build returns matrix
        returns_matrix = np.column_stack([
            list(self._returns[s])[-min_len:]
            for s in self._symbols
        ])
        
        # Calculate correlation
        self._correlation_matrix = np.corrcoef(returns_matrix.T)
        
        # Handle NaN
        self._correlation_matrix = np.nan_to_num(
            self._correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0
        )
        
        # Ensure diagonal is 1
        np.fill_diagonal(self._correlation_matrix, 1.0)
        
        return self._correlation_matrix
    
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols."""
        if self._correlation_matrix is None:
            return 0.0
        
        if symbol1 not in self._symbols or symbol2 not in self._symbols:
            return 0.0
        
        i = self._symbols.index(symbol1)
        j = self._symbols.index(symbol2)
        
        return float(self._correlation_matrix[i, j])
    
    def get_matrix(self) -> Optional[np.ndarray]:
        """Get current correlation matrix."""
        return self._correlation_matrix


class RiskBudgetTracker:
    """
    Tracks risk budget utilization.
    """
    
    def __init__(self, total_risk_budget: float = 0.1) -> None:
        """
        Initialize risk budget tracker.
        
        Args:
            total_risk_budget: Total portfolio VaR budget (as fraction of portfolio)
        """
        self.total_risk_budget = total_risk_budget
        self._allocations: Dict[str, float] = {}
        self._current_risk: Dict[str, float] = {}
    
    def allocate_budget(self, symbol: str, allocation: float) -> None:
        """Allocate risk budget to symbol."""
        self._allocations[symbol] = allocation
    
    def update_risk(self, symbol: str, var: float) -> None:
        """Update current risk for symbol."""
        self._current_risk[symbol] = var
    
    def get_utilization(self) -> Dict[str, Any]:
        """Get risk budget utilization."""
        total_allocated = sum(self._allocations.values())
        total_used = sum(self._current_risk.values())
        
        utilization_by_symbol = {}
        for symbol in self._allocations:
            allocated = self._allocations.get(symbol, 0)
            used = self._current_risk.get(symbol, 0)
            
            if allocated > 0:
                utilization_by_symbol[symbol] = {
                    "allocated": allocated,
                    "used": used,
                    "utilization_pct": used / allocated * 100,
                }
        
        return {
            "total_budget": self.total_risk_budget,
            "total_allocated": total_allocated,
            "total_used": total_used,
            "overall_utilization_pct": (total_used / self.total_risk_budget * 100)
            if self.total_risk_budget > 0 else 0,
            "by_symbol": utilization_by_symbol,
        }
    
    def check_limits(self) -> List[RiskAlert]:
        """Check for risk limit breaches."""
        alerts = []
        utilization = self.get_utilization()
        
        # Check overall utilization
        if utilization["overall_utilization_pct"] > 90:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type="risk_limit",
                severity="critical",
                message=f"Risk budget utilization at {utilization['overall_utilization_pct']:.1f}%",
                metric_name="overall_utilization",
                metric_value=utilization["overall_utilization_pct"],
                threshold=90.0
            ))
        elif utilization["overall_utilization_pct"] > 75:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type="risk_limit",
                severity="warning",
                message=f"Risk budget utilization at {utilization['overall_utilization_pct']:.1f}%",
                metric_name="overall_utilization",
                metric_value=utilization["overall_utilization_pct"],
                threshold=75.0
            ))
        
        # Check individual symbol limits
        for symbol, data in utilization["by_symbol"].items():
            if data["utilization_pct"] > 100:
                alerts.append(RiskAlert(
                    timestamp=datetime.now(),
                    alert_type="position_limit",
                    severity="critical",
                    message=f"{symbol} risk budget exceeded: {data['utilization_pct']:.1f}%",
                    metric_name=f"{symbol}_utilization",
                    metric_value=data["utilization_pct"],
                    threshold=100.0
                ))
        
        return alerts


class RealtimeVARAggregator:
    """
    Main real-time VaR aggregator for Argus.
    
    Combines streaming VaR, correlation tracking, and risk budgeting.
    """
    
    def __init__(
        self,
        portfolio_value: float,
        risk_budget_pct: float = 0.1
    ) -> None:
        """
        Initialize real-time VaR aggregator.
        
        Args:
            portfolio_value: Current portfolio value
            risk_budget_pct: Risk budget as percentage of portfolio
        """
        self.portfolio_value = portfolio_value
        self.risk_budget_pct = risk_budget_pct
        
        self.var_calculator = StreamingVaRCalculator()
        self.correlation_tracker = CorrelationTracker()
        self.risk_budget = RiskBudgetTracker(total_risk_budget=risk_budget_pct)
        
        self._position_values: Dict[str, float] = {}
        self._risk_history: Deque[PortfolioRisk] = deque(maxlen=1000)
        self._alerts: List[RiskAlert] = []
        
        logger.info(
            "RealtimeVARAggregator initialized: portfolio=%.2f, budget=%.1f%%",
            portfolio_value, risk_budget_pct * 100
        )
    
    def update_position(self, symbol: str, value: float) -> None:
        """Update position value."""
        self._position_values[symbol] = value
        self.correlation_tracker.add_symbol(symbol)
    
    def process_tick(self, symbol: str, return_value: float) -> PortfolioRisk:
        """
        Process a new tick and update all risk metrics.
        
        Args:
            symbol: Asset symbol
            return_value: Return observation
            
        Returns:
            Updated portfolio risk metrics
        """
        # Update VaR for this symbol
        var_results = self.var_calculator.update(symbol, return_value)
        
        # Update correlation tracker
        self.correlation_tracker.update({symbol: return_value})
        
        # Update risk budget
        position_value = self._position_values.get(symbol, 0.0)
        if position_value > 0:
            var_pct = var_results.get("var_99", 0.0)
            self.risk_budget.update_risk(symbol, abs(var_pct) * position_value)
        
        # Calculate portfolio VaR
        portfolio_risk = self._calculate_portfolio_risk()
        
        self._risk_history.append(portfolio_risk)
        
        # Check for alerts
        new_alerts = self.risk_budget.check_limits()
        self._alerts.extend(new_alerts)
        
        return portfolio_risk
    
    def _calculate_portfolio_risk(self) -> PortfolioRisk:
        """Calculate aggregated portfolio risk."""
        symbols = list(self._position_values.keys())
        n = len(symbols)
        
        if n == 0:
            return PortfolioRisk(
                timestamp=datetime.now(),
                total_var_95=0.0,
                total_var_99=0.0,
                total_var_99_9=0.0,
                expected_shortfall_99=0.0,
                portfolio_value=self.portfolio_value,
                risk_budget_used_pct=0.0,
                max_drawdown_estimate=0.0,
                sharpe_estimate=0.0,
                position_risks={},
                correlation_regime="normal"
            )
        
        # Get individual VaRs
        position_risks = {}
        total_var_95 = 0.0
        total_var_99 = 0.0
        total_var_99_9 = 0.0
        total_es_99 = 0.0
        
        for symbol in symbols:
            position_value = self._position_values[symbol]
            var_results = self.var_calculator.calculate_var(symbol)
            
            var_95 = var_results.get("var_95", 0.0) * abs(position_value)
            var_99 = var_results.get("var_99", 0.0) * abs(position_value)
            var_99_9 = var_results.get("var_99_9", 0.0) * abs(position_value)
            es_99 = var_results.get("expected_shortfall_99", 0.0) * abs(position_value)
            
            position_risks[symbol] = PositionRisk(
                symbol=symbol,
                position_value=position_value,
                var_95=var_95,
                var_99=var_99,
                var_99_9=var_99_9,
                expected_shortfall_99=es_99,
                marginal_var=0.0,  # Would need full portfolio calculation
                component_var=0.0,
                contribution_pct=0.0
            )
            
            total_var_95 += var_95
            total_var_99 += var_99
            total_var_99_9 += var_99_9
            total_es_99 += es_99
        
        # Simplified portfolio VaR (would use correlation matrix in production)
        correlation_regime = self._assess_correlation_regime()
        
        # Risk budget utilization
        budget_utilization = self.risk_budget.get_utilization()
        
        return PortfolioRisk(
            timestamp=datetime.now(),
            total_var_95=total_var_95,
            total_var_99=total_var_99,
            total_var_99_9=total_var_99_9,
            expected_shortfall_99=total_es_99,
            portfolio_value=self.portfolio_value,
            risk_budget_used_pct=budget_utilization["overall_utilization_pct"],
            max_drawdown_estimate=total_var_99_9,  # Simplified
            sharpe_estimate=0.0,  # Would need return history
            position_risks=position_risks,
            correlation_regime=correlation_regime
        )
    
    def _assess_correlation_regime(self) -> str:
        """Assess current correlation regime."""
        corr_matrix = self.correlation_tracker.get_matrix()
        
        if corr_matrix is None or corr_matrix.size == 0:
            return "normal"
        
        # Get off-diagonal correlations
        n = corr_matrix.shape[0]
        if n < 2:
            return "normal"
        
        off_diag = []
        for i in range(n):
            for j in range(i + 1, n):
                off_diag.append(corr_matrix[i, j])
        
        avg_corr = np.mean(off_diag)
        
        if avg_corr > 0.7:
            return "crisis"
        elif avg_corr > 0.5:
            return "high"
        elif avg_corr > 0.3:
            return "elevated"
        elif avg_corr < 0.1:
            return "low"
        else:
            return "normal"
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk summary."""
        if not self._risk_history:
            return {"status": "no_data"}
        
        latest = self._risk_history[-1]
        
        return {
            "timestamp": latest.timestamp.isoformat(),
            "portfolio_value": latest.portfolio_value,
            "var_95": latest.total_var_95,
            "var_99": latest.total_var_99,
            "var_99_9": latest.total_var_99_9,
            "expected_shortfall_99": latest.expected_shortfall_99,
            "risk_budget_used_pct": latest.risk_budget_used_pct,
            "correlation_regime": latest.correlation_regime,
            "n_positions": len(latest.position_risks),
            "recent_alerts": len([a for a in self._alerts[-10:]]),
        }
    
    def get_recent_alerts(self, n: int = 10) -> List[RiskAlert]:
        """Get recent risk alerts."""
        return self._alerts[-n:]
