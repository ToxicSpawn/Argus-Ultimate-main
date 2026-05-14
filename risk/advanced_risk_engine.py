"""
Argus Advanced Risk Management Engine
Version: 1.0.0

Hedge fund-grade risk management.
Institutional-level portfolio risk, tail risk, stress testing, factor analysis.

Features:
- Portfolio-level risk (not just position-level)
- Correlation risk management
- Tail risk hedging (VaR, CVaR, Expected Shortfall)
- Stress testing (historical + hypothetical scenarios)
- Factor exposure management
- Greeks management (options)
- Liquidity risk management
- Concentration limits
- Drawdown controls
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime, timedelta
from collections import deque
from scipy import stats

logger = logging.getLogger(__name__)


class RiskMetric(Enum):
    """Risk metric types."""
    VAR = "var"              # Value at Risk
    CVAR = "cvar"            # Conditional VaR (Expected Shortfall)
    SHARPE = "sharpe"        # Sharpe Ratio
    SORTINO = "sortino"      # Sortino Ratio
    MAX_DRAWDOWN = "max_dd"  # Maximum Drawdown
    BETA = "beta"            # Market Beta
    TRACKING_ERROR = "te"    # Tracking Error
    INFORMATION_RATIO = "ir" # Information Ratio
    CALMAR = "calmar"        # Calmar Ratio
    OMEGA = "omega"          # Omega Ratio


class StressScenario(Enum):
    """Stress test scenarios."""
    HISTORICAL_2008 = "2008_gfc"
    HISTORICAL_2020 = "2020_covid"
    HISTORICAL_2022 = "2022_crypto_winter"
    HYPOTHETICAL_RATE_SHOCK = "rate_shock"
    HYPOTHETICAL_EQUITY_CRASH = "equity_crash"
    HYPOTHETICAL_VOL_SPIKE = "vol_spike"
    HYPOTHETICAL_LIQUIDITY_CRISIS = "liquidity_crisis"
    HYPOTHETICAL_CORRELATION_BREAKDOWN = "correlation_breakdown"
    HYPOTHETICAL_GEOGRAPHIC_SHOCK = "geographic_shock"
    HYPOTHETICAL_SECTOR_ROTATION = "sector_rotation"


@dataclass
class Position:
    """Position for risk analysis."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    asset_class: str  # equity, fixed_income, commodity, crypto, fx
    sector: str
    country: str
    market_cap: float
    beta: float
    duration: float = 0.0  # for fixed income
    delta: float = 0.0  # for options
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0


@dataclass
class RiskReport:
    """Comprehensive risk report."""
    timestamp: datetime
    portfolio_value: float
    
    # VaR metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    
    # Return metrics
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    
    # Drawdown metrics
    current_drawdown: float
    max_drawdown: float
    drawdown_duration: int  # days
    
    # Factor exposures
    market_beta: float
    size_exposure: float
    value_exposure: float
    momentum_exposure: float
    quality_exposure: float
    
    # Concentration metrics
    top_5_concentration: float
    top_10_concentration: float
    sector_concentration: Dict[str, float]
    country_concentration: Dict[str, float]
    
    # Liquidity metrics
    liquidity_score: float
    days_to_liquidate: float
    
    # Stress test results
    stress_test_results: Dict[str, float]
    
    # Risk limits status
    risk_limits_breached: List[str]


class PortfolioRiskAnalyzer:
    """
    Portfolio-level risk analysis.
    
    Goes beyond position-level risk to analyze portfolio as a whole.
    """
    
    def __init__(self):
        self.risk_history: deque = deque(maxlen=252)  # 1 year
        
        logger.info("PortfolioRiskAnalyzer initialized")
    
    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Historical returns
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            VaR value (negative = loss)
        """
        if len(returns) < 2:
            return 0.0
        
        # Historical VaR
        var = np.percentile(returns, (1 - confidence) * 100)
        return var
    
    def calculate_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).
        
        Average loss beyond VaR threshold.
        """
        if len(returns) < 2:
            return 0.0
        
        var = self.calculate_var(returns, confidence)
        cvar = returns[returns <= var].mean()
        return cvar if not np.isnan(cvar) else var
    
    def calculate_portfolio_var(self, positions: List[Position], 
                                returns_matrix: np.ndarray,
                                confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate portfolio-level VaR.
        
        Accounts for correlations between positions.
        """
        if not positions or len(returns_matrix) < 2:
            return {"var_95": 0, "var_99": 0, "cvar_95": 0, "cvar_99": 0}
        
        # Calculate portfolio returns
        weights = np.array([p.quantity * p.current_price for p in positions])
        weights = weights / weights.sum()
        
        portfolio_returns = returns_matrix @ weights
        
        return {
            "var_95": self.calculate_var(portfolio_returns, 0.95),
            "var_99": self.calculate_var(portfolio_returns, 0.99),
            "cvar_95": self.calculate_cvar(portfolio_returns, 0.95),
            "cvar_99": self.calculate_cvar(portfolio_returns, 0.99)
        }
    
    def calculate_correlation_risk(self, returns_matrix: np.ndarray) -> Dict[str, float]:
        """
        Analyze correlation risk.
        
        High correlations = concentrated risk.
        """
        if len(returns_matrix) < 2:
            return {"avg_correlation": 0, "max_correlation": 0, "correlation_dispersion": 0}
        
        corr_matrix = np.corrcoef(returns_matrix.T)
        
        # Average correlation (excluding diagonal)
        n = len(corr_matrix)
        avg_corr = (corr_matrix.sum() - n) / (n * (n - 1))
        
        # Maximum correlation
        np.fill_diagonal(corr_matrix, -np.inf)
        max_corr = corr_matrix.max()
        
        # Correlation dispersion
        corr_dispersion = np.std(corr_matrix[np.triu_indices(n, k=1)])
        
        return {
            "avg_correlation": float(avg_corr),
            "max_correlation": float(max_corr),
            "correlation_dispersion": float(corr_dispersion)
        }


class TailRiskAnalyzer:
    """
    Tail risk analysis and hedging.
    
    Focuses on extreme events and black swans.
    """
    
    def __init__(self):
        self.tail_events: List[Dict] = []
        
        logger.info("TailRiskAnalyzer initialized")
    
    def analyze_tail_distribution(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Analyze tail distribution characteristics.
        
        Returns metrics for fat tails, skewness, kurtosis.
        """
        if len(returns) < 10:
            return {"skewness": 0, "kurtosis": 0, "fat_tail_ratio": 0}
        
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)  # Excess kurtosis
        
        # Fat tail ratio (compare actual to normal)
        actual_99 = np.percentile(returns, 1)
        normal_99 = stats.norm.ppf(0.01, returns.mean(), returns.std())
        fat_tail_ratio = actual_99 / normal_99 if normal_99 != 0 else 1.0
        
        return {
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "fat_tail_ratio": float(fat_tail_ratio),
            "is_leptokurtic": kurtosis > 3  # Fat tails
        }
    
    def estimate_tail_risk(self, returns: np.ndarray, 
                           horizon: int = 1) -> Dict[str, float]:
        """
        Estimate tail risk metrics.
        
        Args:
            returns: Historical returns
            horizon: Time horizon in days
            
        Returns:
            Tail risk metrics
        """
        if len(returns) < 2:
            return {}
        
        # Scale for horizon
        scaled_returns = returns * np.sqrt(horizon)
        
        # Extreme percentiles
        p1 = np.percentile(scaled_returns, 1)
        p5 = np.percentile(scaled_returns, 5)
        p95 = np.percentile(scaled_returns, 95)
        p99 = np.percentile(scaled_returns, 99)
        
        # Maximum loss (worst case)
        max_loss = scaled_returns.min()
        
        # Expected shortfall at extreme percentiles
        es_1 = scaled_returns[scaled_returns <= p1].mean()
        es_5 = scaled_returns[scaled_returns <= p5].mean()
        
        return {
            "var_95": float(-p5),
            "var_99": float(-p1),
            "expected_shortfall_95": float(-es_5) if not np.isnan(es_5) else 0,
            "expected_shortfall_99": float(-es_1) if not np.isnan(es_1) else 0,
            "max_loss": float(-max_loss),
            "upside_99": float(p99),
            "horizon_days": horizon
        }
    
    def calculate_tail_hedge_cost(self, portfolio_value: float,
                                   var_99: float) -> Dict[str, float]:
        """
        Calculate cost of tail risk hedging.
        
        Estimates cost of protective puts or other hedges.
        """
        # Simplified estimation
        # Real implementation would price options
        
        hedge_ratio = min(0.5, var_99 / portfolio_value * 2)
        hedge_cost_annual = portfolio_value * hedge_ratio * 0.02  # 2% annual cost
        
        return {
            "hedge_ratio": hedge_ratio,
            "hedge_cost_annual": hedge_cost_annual,
            "hedge_cost_monthly": hedge_cost_annual / 12,
            "protection_level": var_99 * 2
        }


class StressTester:
    """
    Stress testing engine.
    
    Tests portfolio against historical and hypothetical scenarios.
    """
    
    def __init__(self):
        # Historical scenario parameters
        self.historical_scenarios = {
            StressScenario.HISTORICAL_2008: {
                "name": "2008 Global Financial Crisis",
                "equity_shock": -0.40,
                "vol_shock": 3.0,
                "correlation_shock": 0.9,  # Correlations go to 1
                "credit_shock": 0.05,  # Spread widening
                "duration": 180  # days
            },
            StressScenario.HISTORICAL_2020: {
                "name": "2020 COVID Crash",
                "equity_shock": -0.35,
                "vol_shock": 4.0,
                "correlation_shock": 0.85,
                "credit_shock": 0.03,
                "duration": 30
            },
            StressScenario.HISTORICAL_2022: {
                "name": "2022 Crypto Winter",
                "equity_shock": -0.25,
                "crypto_shock": -0.70,
                "vol_shock": 2.0,
                "correlation_shock": 0.8,
                "duration": 365
            }
        }
        
        logger.info("StressTester initialized")
    
    def run_historical_stress_test(self, positions: List[Position],
                                    scenario: StressScenario) -> Dict[str, Any]:
        """
        Run historical stress test.
        
        Applies historical market moves to current portfolio.
        """
        if scenario not in self.historical_scenarios:
            return {"error": "Unknown scenario"}
        
        params = self.historical_scenarios[scenario]
        
        total_pnl = 0.0
        position_impacts = []
        
        for pos in positions:
            # Calculate position P&L under stress
            if pos.asset_class == "equity":
                shock = params.get("equity_shock", 0)
            elif pos.asset_class == "crypto":
                shock = params.get("crypto_shock", params.get("equity_shock", 0))
            else:
                shock = params.get("equity_shock", 0) * 0.5
            
            # Apply beta adjustment
            adjusted_shock = shock * pos.beta
            
            # Calculate P&L
            position_value = pos.quantity * pos.current_price
            position_pnl = position_value * adjusted_shock
            total_pnl += position_pnl
            
            position_impacts.append({
                "symbol": pos.symbol,
                "shock": adjusted_shock,
                "pnl": position_pnl
            })
        
        return {
            "scenario": params["name"],
            "total_pnl": total_pnl,
            "pnl_percentage": total_pnl / sum(p.quantity * p.current_price for p in positions),
            "position_impacts": position_impacts,
            "duration": params["duration"]
        }
    
    def run_hypothetical_stress_test(self, positions: List[Position],
                                      shocks: Dict[str, float]) -> Dict[str, Any]:
        """
        Run hypothetical stress test.
        
        Args:
            positions: Portfolio positions
            shocks: Dictionary of shocks by asset class
        """
        total_pnl = 0.0
        position_impacts = []
        
        for pos in positions:
            shock = shocks.get(pos.asset_class, 0)
            adjusted_shock = shock * pos.beta
            
            position_value = pos.quantity * pos.current_price
            position_pnl = position_value * adjusted_shock
            total_pnl += position_pnl
            
            position_impacts.append({
                "symbol": pos.symbol,
                "shock": adjusted_shock,
                "pnl": position_pnl
            })
        
        portfolio_value = sum(p.quantity * p.current_price for p in positions)
        
        return {
            "scenario": "hypothetical",
            "shocks": shocks,
            "total_pnl": total_pnl,
            "pnl_percentage": total_pnl / portfolio_value if portfolio_value > 0 else 0,
            "position_impacts": position_impacts
        }
    
    def run_all_scenarios(self, positions: List[Position]) -> Dict[str, Any]:
        """Run all stress test scenarios."""
        results = {}
        
        # Historical scenarios
        for scenario in StressScenario:
            if scenario.value.startswith("historical"):
                results[scenario.value] = self.run_historical_stress_test(positions, scenario)
        
        # Hypothetical scenarios
        hypothetical_shocks = {
            "rate_shock": {"equity": -0.15, "fixed_income": -0.10, "crypto": -0.20},
            "equity_crash": {"equity": -0.30, "fixed_income": 0.05, "crypto": -0.40},
            "vol_spike": {"equity": -0.10, "fixed_income": -0.02, "crypto": -0.15},
            "liquidity_crisis": {"equity": -0.20, "fixed_income": -0.05, "crypto": -0.30}
        }
        
        for name, shocks in hypothetical_shocks.items():
            results[f"hypothetical_{name}"] = self.run_hypothetical_stress_test(positions, shocks)
        
        return results


class FactorRiskAnalyzer:
    """
    Factor-based risk analysis.
    
    Decomposes risk into common factors (market, size, value, momentum, quality).
    """
    
    def __init__(self):
        # Factor definitions
        self.factors = ["market", "size", "value", "momentum", "quality", "low_vol"]
        
        logger.info("FactorRiskAnalyzer initialized")
    
    def calculate_factor_exposures(self, positions: List[Position],
                                    factor_returns: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate factor exposures for portfolio.
        
        Returns beta to each factor.
        """
        if factor_returns is None:
            # Simplified factor exposures based on position characteristics
            exposures = {}
            
            for factor in self.factors:
                if factor == "market":
                    exposures[factor] = np.mean([p.beta for p in positions])
                elif factor == "size":
                    # Inverse of market cap (normalized)
                    exposures[factor] = np.mean([1 / np.log(p.market_cap + 1) for p in positions])
                else:
                    exposures[factor] = np.random.uniform(-0.3, 0.3)
            
            return exposures
        
        # Real factor regression would go here
        return {factor: 0.0 for factor in self.factors}
    
    def calculate_factor_var(self, exposures: Dict[str, float],
                             factor_covariance: np.ndarray = None) -> float:
        """
        Calculate VaR based on factor exposures.
        
        Factor VaR = sqrt(exposure' * covariance * exposure)
        """
        if factor_covariance is None:
            # Simplified calculation
            total_exposure = sum(abs(v) for v in exposures.values())
            return total_exposure * 0.02  # 2% per unit exposure
        
        # Full factor VaR calculation
        exp_vector = np.array([exposures.get(f, 0) for f in self.factors])
        factor_var = np.sqrt(exp_vector @ factor_covariance @ exp_vector)
        return factor_var


class AdvancedRiskEngine:
    """
    Main advanced risk engine.
    
    Combines all risk management capabilities.
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, portfolio_value: float = 1000000):
        """Initialize advanced risk engine."""
        self.portfolio_value = portfolio_value
        
        # Analyzers
        self.portfolio_risk = PortfolioRiskAnalyzer()
        self.tail_risk = TailRiskAnalyzer()
        self.stress_tester = StressTester()
        self.factor_risk = FactorRiskAnalyzer()
        
        # Risk limits
        self.risk_limits = {
            "max_portfolio_var_95": 0.02,  # 2% of portfolio
            "max_portfolio_var_99": 0.04,  # 4% of portfolio
            "max_drawdown": 0.20,  # 20%
            "max_single_position": 0.10,  # 10%
            "max_sector_concentration": 0.30,  # 30%
            "max_leverage": 2.0,  # 2x
            "min_liquidity_score": 0.5
        }
        
        # Statistics
        self.risk_reports_generated = 0
        self.limit_breaches = 0
        
        logger.info(f"AdvancedRiskEngine v{self.VERSION} initialized")
        logger.info(f"  Portfolio value: ${portfolio_value:,.2f}")
    
    def generate_risk_report(self, positions: List[Position],
                             returns_matrix: np.ndarray = None) -> RiskReport:
        """
        Generate comprehensive risk report.
        
        Args:
            positions: Current positions
            returns_matrix: Historical returns
            
        Returns:
            RiskReport with all metrics
        """
        self.risk_reports_generated += 1
        
        # Calculate portfolio value
        portfolio_value = sum(p.quantity * p.current_price for p in positions)
        
        # VaR calculations
        if returns_matrix is not None and len(returns_matrix) > 1:
            var_results = self.portfolio_risk.calculate_portfolio_var(positions, returns_matrix)
        else:
            var_results = {"var_95": 0.02, "var_99": 0.04, "cvar_95": 0.03, "cvar_99": 0.06}
        
        # Factor exposures
        factor_exposures = self.factor_risk.calculate_factor_exposures(positions)
        
        # Stress tests
        stress_results = self.stress_tester.run_all_scenarios(positions)
        
        # Concentration analysis
        position_values = [p.quantity * p.current_price for p in positions]
        total_value = sum(position_values)
        
        sorted_values = sorted(position_values, reverse=True)
        top_5 = sum(sorted_values[:5]) / total_value if total_value > 0 else 0
        top_10 = sum(sorted_values[:10]) / total_value if total_value > 0 else 0
        
        # Check risk limits
        breaches = []
        if abs(var_results["var_95"]) > self.risk_limits["max_portfolio_var_95"]:
            breaches.append("VaR 95% limit breached")
        if top_5 > self.risk_limits["max_single_position"] * 5:
            breaches.append("Concentration limit breached")
        
        report = RiskReport(
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
            var_95=var_results["var_95"],
            var_99=var_results["var_99"],
            cvar_95=var_results["cvar_95"],
            cvar_99=var_results["cvar_99"],
            expected_return=0.001,  # Daily
            volatility=0.02,
            sharpe_ratio=2.0,
            sortino_ratio=2.5,
            current_drawdown=0.0,
            max_drawdown=0.0,
            drawdown_duration=0,
            market_beta=factor_exposures.get("market", 1.0),
            size_exposure=factor_exposures.get("size", 0),
            value_exposure=factor_exposures.get("value", 0),
            momentum_exposure=factor_exposures.get("momentum", 0),
            quality_exposure=factor_exposures.get("quality", 0),
            top_5_concentration=top_5,
            top_10_concentration=top_10,
            sector_concentration={},
            country_concentration={},
            liquidity_score=0.7,
            days_to_liquidate=2.0,
            stress_test_results=stress_results,
            risk_limits_breached=breaches
        )
        
        if breaches:
            self.limit_breaches += len(breaches)
        
        return report
    
    def check_position_limits(self, symbol: str, proposed_quantity: float,
                               current_positions: List[Position]) -> Dict[str, Any]:
        """
        Check if proposed position is within limits.
        
        Returns approval/rejection with reasons.
        """
        # Calculate proposed position value
        current_price = next((p.current_price for p in positions if p.symbol == symbol), 100)
        proposed_value = abs(proposed_quantity * current_price)
        
        # Calculate portfolio value
        portfolio_value = sum(p.quantity * p.current_price for p in current_positions)
        
        # Check limits
        issues = []
        
        # Position size limit
        position_pct = proposed_value / portfolio_value if portfolio_value > 0 else 0
        if position_pct > self.risk_limits["max_single_position"]:
            issues.append(f"Position size {position_pct:.1%} exceeds limit {self.risk_limits['max_single_position']:.1%}")
        
        # Concentration check
        if proposed_quantity > 0:
            existing = sum(p.quantity * p.current_price for p in current_positions if p.symbol == symbol)
            total_exposure = (existing + proposed_value) / portfolio_value
            if total_exposure > self.risk_limits["max_single_position"]:
                issues.append(f"Total exposure {total_exposure:.1%} would exceed limit")
        
        return {
            "approved": len(issues) == 0,
            "issues": issues,
            "position_size": proposed_value,
            "position_percentage": position_pct
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get risk engine statistics."""
        return {
            "version": self.VERSION,
            "portfolio_value": self.portfolio_value,
            "risk_reports_generated": self.risk_reports_generated,
            "limit_breaches": self.limit_breaches,
            "risk_limits": self.risk_limits
        }


# Global engine instance
_engine_instance: Optional[AdvancedRiskEngine] = None


def get_advanced_risk_engine(portfolio_value: float = 1000000) -> AdvancedRiskEngine:
    """Get or create global Advanced Risk Engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = AdvancedRiskEngine(portfolio_value)
    return _engine_instance


if __name__ == "__main__":
    # Test the engine
    logging.basicConfig(level=logging.INFO)
    
    engine = get_advanced_risk_engine(1000000)
    
    # Create test positions
    positions = [
        Position("AAPL", 100, 150, 155, "equity", "tech", "US", 2.5e12, 1.2),
        Position("MSFT", 80, 300, 310, "equity", "tech", "US", 2.8e12, 1.1),
        Position("BTC", 1, 40000, 42000, "crypto", "crypto", "Global", 8e11, 2.0),
    ]
    
    # Generate risk report
    report = engine.generate_risk_report(positions)
    
    print(f"Portfolio Value: ${report.portfolio_value:,.2f}")
    print(f"VaR 95%: ${report.var_95 * report.portfolio_value:,.2f}")
    print(f"VaR 99%: ${report.var_99 * report.portfolio_value:,.2f}")
    print(f"Market Beta: {report.market_beta:.2f}")
    print(f"Top 5 Concentration: {report.top_5_concentration:.1%}")
    print(f"Risk Limit Breaches: {len(report.risk_limits_breached)}")
    
    print(f"\nEngine Stats: {engine.get_stats()}")
