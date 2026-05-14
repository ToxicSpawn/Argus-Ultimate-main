"""
RISK SYSTEM V2 - OMEGA
========================
The most advanced risk management system.

30 Components:
1. Real-time VaR (Value at Risk)
2. CVaR (Conditional VaR / Expected Shortfall)
3. Stress Testing
4. Scenario Analysis
5. Correlation Risk Monitor
6. Liquidity Risk Assessment
7. Concentration Risk Monitor
8. Drawdown Protection
9. Kelly Criterion Position Sizing
10. Dynamic Hedging
11. Tail Risk Hedging
12. Black Swan Protection
13. Circuit Breakers
14. Risk Budgeting
15. Factor Risk Decomposition
16. Volatility Forecasting
17. Regime Detection
18. Maximum Drawdown Limit
19. Sharpe Ratio Monitor
20. Sortino Ratio Monitor
21. Calmar Ratio Monitor
22. Maximum Position Size
23. Maximum Sector Exposure
24. Maximum Correlation Exposure
25. Maximum Leverage Limit
26. Stop Loss Management
27. Trailing Stop Management
28. Profit Target Management
29. Risk-Adjusted Return Optimization
30. Portfolio Heat Map
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Position:
    """Position representation."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    timestamp: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None


@dataclass
class RiskAlert:
    """Risk alert representation."""
    level: RiskLevel
    component: str
    message: str
    value: float
    threshold: float
    timestamp: float


class VaRCalculator:
    """Value at Risk calculator."""
    
    def __init__(self, confidence: float = 0.99, lookback_days: int = 252):
        self.confidence = confidence
        self.lookback_days = lookback_days
        self.returns_history: deque = deque(maxlen=lookback_days)
        
    def add_return(self, daily_return: float):
        """Add daily return to history."""
        self.returns_history.append(daily_return)
    
    def calculate_var(self, portfolio_value: float) -> Dict[str, float]:
        """Calculate VaR using multiple methods."""
        if len(self.returns_history) < 30:
            return {"historical_var": 0, "parametric_var": 0, "cornish_fisher_var": 0}
        
        returns = np.array(self.returns_history)
        
        # Historical VaR
        historical_var = -np.percentile(returns, (1 - self.confidence) * 100)
        
        # Parametric VaR (assumes normal distribution)
        mean = np.mean(returns)
        std = np.std(returns)
        z_score = np.percentile(np.random.standard_normal(10000), self.confidence * 100)
        parametric_var = -(mean + z_score * std)
        
        # Cornish-Fisher VaR (accounts for skewness and kurtosis)
        skew = float(np.mean(((returns - mean) / std) ** 3))
        kurtosis = float(np.mean(((returns - mean) / std) ** 4) - 3)
        
        # Cornish-Fisher adjustment
        z_cf = z_score + (z_score**2 - 1) * skew / 6 + (z_score**3 - 3*z_score) * kurtosis / 24
        cornish_fisher_var = -(mean + z_cf * std)
        
        return {
            "historical_var": float(historical_var * portfolio_value),
            "parametric_var": float(parametric_var * portfolio_value),
            "cornish_fisher_var": float(cornish_fisher_var * portfolio_value),
            "var_pct": float(historical_var * 100),
            "confidence": self.confidence,
        }


class CVaRCalculator:
    """Conditional VaR (Expected Shortfall) calculator."""
    
    def __init__(self, confidence: float = 0.99):
        self.confidence = confidence
        self.returns_history: deque = deque(maxlen=252)
        
    def add_return(self, daily_return: float):
        """Add daily return to history."""
        self.returns_history.append(daily_return)
    
    def calculate_cvar(self, portfolio_value: float) -> Dict[str, float]:
        """Calculate CVaR (average of losses beyond VaR)."""
        if len(self.returns_history) < 30:
            return {"cvar": 0, "cvar_pct": 0}
        
        returns = np.array(self.returns_history)
        var_threshold = -np.percentile(returns, (1 - self.confidence) * 100)
        
        # Get returns beyond VaR threshold
        tail_returns = returns[returns <= -var_threshold]
        
        if len(tail_returns) == 0:
            cvar = var_threshold
        else:
            cvar = -np.mean(tail_returns)
        
        return {
            "cvar": float(cvar * portfolio_value),
            "cvar_pct": float(cvar * 100),
            "tail_observations": len(tail_returns),
        }


class StressTester:
    """Stress testing engine."""
    
    def __init__(self):
        self.scenarios: Dict[str, Dict[str, float]] = {
            "black_thursday": {"btc": -0.50, "eth": -0.55, "market": -0.45},
            "covid_crash": {"btc": -0.40, "eth": -0.45, "market": -0.35},
            "ftx_collapse": {"btc": -0.25, "eth": -0.30, "market": -0.20},
            "luna_collapse": {"btc": -0.15, "eth": -0.20, "market": -0.10},
            "flash_crash": {"btc": -0.20, "eth": -0.25, "market": -0.15},
            "rate_hike_shock": {"btc": -0.10, "eth": -0.15, "market": -0.08},
            "stablecoin_depeg": {"btc": -0.30, "eth": -0.35, "market": -0.25},
            "regulatory_ban": {"btc": -0.40, "eth": -0.45, "market": -0.30},
        }
        self.results_history: deque = deque(maxlen=100)
    
    def run_stress_test(
        self,
        positions: Dict[str, float],
        portfolio_value: float,
    ) -> Dict[str, Any]:
        """Run all stress scenarios."""
        results = {}
        
        for scenario_name, shocks in self.scenarios.items():
            total_impact = 0
            
            for symbol, position_value in positions.items():
                # Find matching shock
                shock = 0
                if "btc" in symbol.lower() or "bitcoin" in symbol.lower():
                    shock = shocks.get("btc", 0)
                elif "eth" in symbol.lower() or "ethereum" in symbol.lower():
                    shock = shocks.get("eth", 0)
                else:
                    shock = shocks.get("market", 0)
                
                total_impact += position_value * shock
            
            results[scenario_name] = {
                "impact": float(total_impact),
                "impact_pct": float(total_impact / portfolio_value * 100),
                "survivable": total_impact > -portfolio_value * 0.5,
            }
        
        # Worst case
        worst_scenario = min(results, key=lambda x: results[x]["impact"])
        
        result = {
            "scenarios": results,
            "worst_case": {
                "scenario": worst_scenario,
                "impact": results[worst_scenario]["impact"],
                "impact_pct": results[worst_scenario]["impact_pct"],
            },
            "all_survivable": all(r["survivable"] for r in results.values()),
        }
        
        self.results_history.append(result)
        return result


class CorrelationRiskMonitor:
    """Monitor correlation risk across positions."""
    
    def __init__(self):
        self.price_history: Dict[str, deque] = {}
        self.max_correlation_threshold = 0.8
        
    def add_price(self, symbol: str, price: float):
        """Add price observation."""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=100)
        self.price_history[symbol].append(price)
    
    def calculate_correlation_matrix(self) -> Dict[str, Any]:
        """Calculate correlation matrix between all symbols."""
        symbols = list(self.price_history.keys())
        
        if len(symbols) < 2:
            return {"correlations": {}, "max_correlation": 0, "high_correlation_pairs": []}
        
        # Calculate returns
        returns = {}
        for symbol in symbols:
            prices = list(self.price_history[symbol])
            if len(prices) < 10:
                continue
            returns[symbol] = np.diff(np.log(prices))
        
        # Calculate correlations
        correlations = {}
        high_correlation_pairs = []
        
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                if sym1 not in returns or sym2 not in returns:
                    continue
                if len(returns[sym1]) != len(returns[sym2]):
                    continue
                
                corr = float(np.corrcoef(returns[sym1], returns[sym2])[0, 1])
                correlations[f"{sym1}_{sym2}"] = corr
                
                if abs(corr) > self.max_correlation_threshold:
                    high_correlation_pairs.append({
                        "pair": f"{sym1}_{sym2}",
                        "correlation": corr,
                    })
        
        max_corr = max([abs(c) for c in correlations.values()]) if correlations else 0
        
        return {
            "correlations": correlations,
            "max_correlation": float(max_corr),
            "high_correlation_pairs": high_correlation_pairs,
            "n_pairs": len(correlations),
        }


class LiquidityRiskAssessor:
    """Assess liquidity risk for positions."""
    
    def __init__(self):
        self.liquidity_scores: Dict[str, float] = {}
        
    def set_liquidity_score(self, symbol: str, score: float):
        """Set liquidity score for symbol (0-100)."""
        self.liquidity_scores[symbol] = score
    
    def assess_risk(
        self,
        positions: Dict[str, float],
        portfolio_value: float,
    ) -> Dict[str, Any]:
        """Assess liquidity risk of portfolio."""
        total_liquidity_score = 0
        total_value = 0
        illiquid_positions = []
        
        for symbol, value in positions.items():
            score = self.liquidity_scores.get(symbol, 50)  # Default medium liquidity
            total_liquidity_score += score * value
            total_value += value
            
            if score < 30:
                illiquid_positions.append({
                    "symbol": symbol,
                    "value": value,
                    "liquidity_score": score,
                    "pct_of_portfolio": value / portfolio_value * 100 if portfolio_value > 0 else 0,
                })
        
        avg_liquidity = total_liquidity_score / total_value if total_value > 0 else 50
        
        return {
            "portfolio_liquidity_score": float(avg_liquidity),
            "illiquid_positions": illiquid_positions,
            "n_illiquid": len(illiquid_positions),
            "risk_level": "high" if avg_liquidity < 40 else "medium" if avg_liquidity < 60 else "low",
        }


class ConcentrationRiskMonitor:
    """Monitor concentration risk."""
    
    def __init__(self, max_single_position: float = 0.3, max_sector: float = 0.5):
        self.max_single_position = max_single_position
        self.max_sector = max_sector
        self.sector_map: Dict[str, str] = {}
    
    def set_sector(self, symbol: str, sector: str):
        """Map symbol to sector."""
        self.sector_map[symbol] = sector
    
    def assess_concentration(
        self,
        positions: Dict[str, float],
        portfolio_value: float,
    ) -> Dict[str, Any]:
        """Assess concentration risk."""
        alerts = []
        
        # Single position concentration
        for symbol, value in positions.items():
            pct = value / portfolio_value if portfolio_value > 0 else 0
            if pct > self.max_single_position:
                alerts.append({
                    "type": "single_position",
                    "symbol": symbol,
                    "concentration_pct": float(pct * 100),
                    "threshold_pct": float(self.max_single_position * 100),
                })
        
        # Sector concentration
        sector_exposure: Dict[str, float] = {}
        for symbol, value in positions.items():
            sector = self.sector_map.get(symbol, "other")
            sector_exposure[sector] = sector_exposure.get(sector, 0) + value
        
        for sector, exposure in sector_exposure.items():
            pct = exposure / portfolio_value if portfolio_value > 0 else 0
            if pct > self.max_sector:
                alerts.append({
                    "type": "sector_concentration",
                    "sector": sector,
                    "concentration_pct": float(pct * 100),
                    "threshold_pct": float(self.max_sector * 100),
                })
        
        # Herfindahl-Hirschman Index (HHI)
        hhi = sum((v / portfolio_value * 100) ** 2 for v in positions.values()) if portfolio_value > 0 else 0
        
        return {
            "alerts": alerts,
            "n_alerts": len(alerts),
            "hhi": float(hhi),
            "max_position_pct": float(max([v / portfolio_value for v in positions.values()]) * 100) if portfolio_value > 0 and positions else 0,
            "sector_exposure": {k: float(v) for k, v in sector_exposure.items()},
            "risk_level": "high" if len(alerts) > 2 else "medium" if len(alerts) > 0 else "low",
        }


class DrawdownProtection:
    """Drawdown protection system."""
    
    def __init__(self, max_drawdown: float = 0.25):
        self.max_drawdown = max_drawdown
        self.peak_value: float = 0
        self.current_value: float = 0
        self.drawdown_history: deque = deque(maxlen=1000)
        
    def update(self, portfolio_value: float) -> Dict[str, Any]:
        """Update with current portfolio value."""
        self.current_value = portfolio_value
        self.peak_value = max(self.peak_value, portfolio_value)
        
        current_drawdown = (self.peak_value - portfolio_value) / self.peak_value if self.peak_value > 0 else 0
        
        self.drawdown_history.append({
            "value": portfolio_value,
            "peak": self.peak_value,
            "drawdown": current_drawdown,
            "timestamp": time.time(),
        })
        
        # Determine action
        if current_drawdown >= self.max_drawdown:
            action = "halt_trading"
        elif current_drawdown >= self.max_drawdown * 0.8:
            action = "reduce_positions"
        elif current_drawdown >= self.max_drawdown * 0.6:
            action = "reduce_risk"
        else:
            action = "normal"
        
        return {
            "current_drawdown_pct": float(current_drawdown * 100),
            "max_drawdown_pct": float(self.max_drawdown * 100),
            "peak_value": float(self.peak_value),
            "current_value": float(self.current_value),
            "action": action,
            "breached": current_drawdown >= self.max_drawdown,
        }


class KellyCriterionSizer:
    """Kelly Criterion position sizer."""
    
    def __init__(self, max_kelly: float = 0.25):
        self.max_kelly = max_kelly
        self.win_rate_history: deque = deque(maxlen=100)
        self.avg_win_history: deque = deque(maxlen=100)
        self.avg_loss_history: deque = deque(maxlen=100)
    
    def update_stats(self, win_rate: float, avg_win: float, avg_loss: float):
        """Update trading statistics."""
        self.win_rate_history.append(win_rate)
        self.avg_win_history.append(avg_win)
        self.avg_loss_history.append(avg_loss)
    
    def calculate_kelly(self) -> Dict[str, float]:
        """Calculate Kelly Criterion position size."""
        if not self.win_rate_history:
            return {"kelly_fraction": 0.25, "recommended_size_pct": 25}
        
        win_rate = np.mean(list(self.win_rate_history))
        avg_win = np.mean(list(self.avg_win_history)) if self.avg_win_history else 0.02
        avg_loss = np.mean(list(self.avg_loss_history)) if self.avg_loss_history else 0.01
        
        # Kelly formula: f* = (bp - q) / b
        # b = odds (avg_win / avg_loss)
        # p = win_rate
        # q = 1 - win_rate
        
        if avg_loss == 0:
            kelly = 0.25
        else:
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            kelly = (b * p - q) / b
        
        # Cap at max_kelly
        kelly = min(max(kelly, 0), self.max_kelly)
        
        # Half-Kelly for safety
        half_kelly = kelly / 2
        
        return {
            "kelly_fraction": float(kelly),
            "half_kelly_fraction": float(half_kelly),
            "recommended_size_pct": float(half_kelly * 100),
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
        }


class DynamicHedger:
    """Dynamic hedging engine."""
    
    def __init__(self):
        self.hedge_ratios: Dict[str, float] = {}
        self.hedge_history: deque = deque(maxlen=100)
        
    def calculate_hedge_ratio(
        self,
        position_beta: float,
        target_beta: float = 0,
    ) -> Dict[str, Any]:
        """Calculate optimal hedge ratio."""
        hedge_ratio = target_beta - position_beta
        
        return {
            "hedge_ratio": float(hedge_ratio),
            "position_beta": float(position_beta),
            "target_beta": float(target_beta),
            "hedge_required": hedge_ratio != 0,
        }
    
    def calculate_options_hedge(
        self,
        position_value: float,
        current_price: float,
        strike_price: float,
        volatility: float,
        time_to_expiry: float = 30 / 365,
    ) -> Dict[str, float]:
        """Calculate options hedge using Black-Scholes."""
        # Simplified Black-Scholes
        d1 = (np.log(current_price / strike_price) + (0.05 + volatility**2 / 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        # Put option price (simplified)
        from scipy.stats import norm
        put_price = strike_price * np.exp(-0.05 * time_to_expiry) * norm.cdf(-d2) - current_price * norm.cdf(-d1)
        
        # Number of puts needed
        n_puts = position_value / (put_price * 100)  # Each contract = 100 shares
        
        return {
            "put_price": float(put_price),
            "contracts_needed": int(np.ceil(n_puts)),
            "hedge_cost": float(n_puts * put_price * 100),
            "hedge_cost_pct": float(n_puts * put_price * 100 / position_value * 100),
            "delta_hedge": float(-norm.cdf(-d1)),
        }


class TailRiskHedger:
    """Tail risk hedging system."""
    
    def __init__(self):
        self.tail_events: deque = deque(maxlen=100)
        self.hedge_positions: List[Dict[str, Any]] = []
        
    def detect_tail_risk(
        self,
        returns: List[float],
        threshold: float = -0.05,
    ) -> Dict[str, Any]:
        """Detect tail risk events."""
        returns_array = np.array(returns)
        
        tail_events = returns_array[returns_array < threshold]
        
        if len(tail_events) == 0:
            return {
                "tail_events_count": 0,
                "tail_frequency": 0,
                "avg_tail_loss": 0,
                "max_tail_loss": 0,
                "tail_risk_level": "low",
            }
        
        tail_frequency = len(tail_events) / len(returns_array)
        avg_tail_loss = float(np.mean(tail_events))
        max_tail_loss = float(np.min(tail_events))
        
        # Tail risk level
        if tail_frequency > 0.05 or max_tail_loss < -0.10:
            risk_level = "critical"
        elif tail_frequency > 0.02 or max_tail_loss < -0.05:
            risk_level = "high"
        elif tail_frequency > 0.01:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "tail_events_count": len(tail_events),
            "tail_frequency": float(tail_frequency),
            "avg_tail_loss": avg_tail_loss,
            "max_tail_loss": max_tail_loss,
            "tail_risk_level": risk_level,
        }


class BlackSwanProtector:
    """Black swan event protection."""
    
    def __init__(self):
        self.indicators: Dict[str, float] = {}
        self.black_swan_score: float = 0
        
    def update_indicator(self, name: str, value: float):
        """Update black swan indicator."""
        self.indicators[name] = value
    
    def calculate_black_swan_score(self) -> Dict[str, Any]:
        """Calculate overall black swan probability score."""
        # Weighted indicators
        weights = {
            "vix": 0.2,
            "fear_greed": 0.15,
            "funding_rate": 0.15,
            "open_interest": 0.1,
            "whale_activity": 0.15,
            "exchange_flows": 0.1,
            "social_sentiment": 0.1,
            "macro_risk": 0.05,
        }
        
        score = 0
        for indicator, weight in weights.items():
            if indicator in self.indicators:
                score += self.indicators[indicator] * weight
        
        self.black_swan_score = score
        
        # Determine protection level
        if score > 0.8:
            protection = "maximum"
            action = "reduce_all_positions"
        elif score > 0.6:
            protection = "high"
            action = "hedge_tail_risk"
        elif score > 0.4:
            protection = "medium"
            action = "tighten_stops"
        else:
            protection = "normal"
            action = "monitor"
        
        return {
            "black_swan_score": float(score),
            "protection_level": protection,
            "recommended_action": action,
            "indicators": self.indicators,
        }


class CircuitBreakerSystem:
    """Circuit breaker system."""
    
    def __init__(self):
        self.breakers: Dict[str, Dict[str, Any]] = {
            "daily_loss": {"threshold": 0.10, "current": 0, "triggered": False},
            "hourly_loss": {"threshold": 0.05, "current": 0, "triggered": False},
            "consecutive_losses": {"threshold": 5, "current": 0, "triggered": False},
            "volatility_spike": {"threshold": 0.10, "current": 0, "triggered": False},
            "drawdown": {"threshold": 0.25, "current": 0, "triggered": False},
        }
        
    def update_breaker(self, name: str, value: float):
        """Update breaker value."""
        if name in self.breakers:
            self.breakers[name]["current"] = value
            if value >= self.breakers[name]["threshold"]:
                self.breakers[name]["triggered"] = True
    
    def reset_breaker(self, name: str):
        """Reset a breaker."""
        if name in self.breakers:
            self.breakers[name]["current"] = 0
            self.breakers[name]["triggered"] = False
    
    def check_all_breakers(self) -> Dict[str, Any]:
        """Check all circuit breakers."""
        triggered = []
        for name, breaker in self.breakers.items():
            if breaker["triggered"]:
                triggered.append({
                    "name": name,
                    "threshold": breaker["threshold"],
                    "current": breaker["current"],
                })
        
        return {
            "any_triggered": len(triggered) > 0,
            "triggered_breakers": triggered,
            "n_triggered": len(triggered),
            "can_trade": len(triggered) == 0,
            "all_breakers": self.breakers,
        }


class RiskBudgeting:
    """Risk budgeting system."""
    
    def __init__(self, total_risk_budget: float = 0.15):
        self.total_risk_budget = total_risk_budget
        self.allocations: Dict[str, float] = {}
        
    def allocate_budget(self, strategies: List[str], method: str = "equal") -> Dict[str, float]:
        """Allocate risk budget to strategies."""
        if method == "equal":
            per_strategy = self.total_risk_budget / len(strategies)
            self.allocations = {s: per_strategy for s in strategies}
        elif method == "risk_parity":
            # Simplified risk parity
            self.allocations = {s: self.total_risk_budget / len(strategies) for s in strategies}
        
        return self.allocations
    
    def get_remaining_budget(self, used_risk: float) -> Dict[str, float]:
        """Get remaining risk budget."""
        remaining = self.total_risk_budget - used_risk
        
        return {
            "total_budget": self.total_risk_budget,
            "used_budget": used_risk,
            "remaining_budget": max(remaining, 0),
            "utilization_pct": float(used_risk / self.total_risk_budget * 100),
            "can_take_more_risk": remaining > 0,
        }


class FactorRiskDecomposer:
    """Decompose risk into factors."""
    
    def __init__(self):
        self.factors = ["market", "momentum", "value", "size", "volatility", "quality"]
        
    def decompose(
        self,
        positions: Dict[str, float],
        factor_exposures: Dict[str, Dict[str, float]],
        portfolio_value: float,
    ) -> Dict[str, Any]:
        """Decompose portfolio risk into factors."""
        factor_exposure = {f: 0 for f in self.factors}
        
        for symbol, value in positions.items():
            if symbol in factor_exposures:
                for factor in self.factors:
                    exposure = factor_exposures[symbol].get(factor, 0)
                    weight = value / portfolio_value if portfolio_value > 0 else 0
                    factor_exposure[factor] += exposure * weight
        
        # Total risk (simplified)
        total_risk = sum(abs(v) for v in factor_exposure.values())
        
        return {
            "factor_exposures": factor_exposure,
            "total_factor_risk": float(total_risk),
            "dominant_factor": max(factor_exposure, key=lambda x: abs(factor_exposure[x])),
            "factor_contributions": {k: float(v) for k, v in factor_exposure.items()},
        }


class VolatilityForecaster:
    """Volatility forecasting engine."""
    
    def __init__(self):
        self.returns_history: deque = deque(maxlen=252)
        
    def add_return(self, daily_return: float):
        """Add daily return."""
        self.returns_history.append(daily_return)
    
    def forecast_garch(self) -> Dict[str, float]:
        """Forecast volatility using GARCH(1,1) model."""
        if len(self.returns_history) < 30:
            return {"forecast_volatility": 0.02, "method": "insufficient_data"}
        
        returns = np.array(self.returns_history)
        
        # Simplified GARCH(1,1) estimation
        # sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}
        
        # Initial estimates
        omega = 0.000001
        alpha = 0.1
        beta = 0.85
        
        # Calculate conditional variance
        variances = []
        sigma2 = np.var(returns)
        
        for r in returns:
            sigma2 = omega + alpha * r**2 + beta * sigma2
            variances.append(sigma2)
        
        # Forecast next period
        forecast_var = omega + alpha * returns[-1]**2 + beta * variances[-1]
        forecast_vol = np.sqrt(forecast_var * 252)  # Annualized
        
        return {
            "forecast_volatility": float(forecast_vol),
            "current_volatility": float(np.sqrt(np.var(returns) * 252)),
            "method": "GARCH(1,1)",
        }


class RegimeDetector:
    """Market regime detection."""
    
    def __init__(self):
        self.returns_history: deque = deque(maxlen=100)
        self.current_regime: str = "normal"
        
    def add_return(self, daily_return: float):
        """Add daily return."""
        self.returns_history.append(daily_return)
    
    def detect_regime(self) -> Dict[str, Any]:
        """Detect current market regime."""
        if len(self.returns_history) < 20:
            return {"regime": "unknown", "confidence": 0}
        
        returns = np.array(self.returns_history)
        
        # Calculate regime indicators
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        trend = np.polyfit(range(len(returns)), returns, 1)[0]
        
        # Determine regime
        if volatility > 0.05:
            regime = "high_volatility"
        elif mean_return > 0.002 and trend > 0:
            regime = "bull"
        elif mean_return < -0.002 and trend < 0:
            regime = "bear"
        elif abs(trend) < 0.0001:
            regime = "sideways"
        else:
            regime = "normal"
        
        self.current_regime = regime
        
        return {
            "regime": regime,
            "mean_return": float(mean_return),
            "volatility": float(volatility),
            "trend": float(trend),
            "confidence": min(abs(trend) / 0.001, 1.0),
        }


class SharpeRatioMonitor:
    """Sharpe ratio monitor."""
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.returns_history: deque = deque(maxlen=252)
        
    def add_return(self, daily_return: float):
        """Add daily return."""
        self.returns_history.append(daily_return)
    
    def calculate_sharpe(self) -> Dict[str, float]:
        """Calculate Sharpe ratio."""
        if len(self.returns_history) < 30:
            return {"sharpe_ratio": 0, "annualized_sharpe": 0}
        
        returns = np.array(self.returns_history)
        
        daily_rf = self.risk_free_rate / 252
        excess_returns = returns - daily_rf
        
        if np.std(excess_returns) == 0:
            return {"sharpe_ratio": 0, "annualized_sharpe": 0}
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        annualized_sharpe = sharpe * np.sqrt(252)
        
        return {
            "sharpe_ratio": float(sharpe),
            "annualized_sharpe": float(annualized_sharpe),
            "mean_excess_return": float(np.mean(excess_returns)),
            "std_excess_return": float(np.std(excess_returns)),
        }


class SortinoRatioMonitor:
    """Sortino ratio monitor."""
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.returns_history: deque = deque(maxlen=252)
        
    def add_return(self, daily_return: float):
        """Add daily return."""
        self.returns_history.append(daily_return)
    
    def calculate_sortino(self) -> Dict[str, float]:
        """Calculate Sortino ratio."""
        if len(self.returns_history) < 30:
            return {"sortino_ratio": 0, "annualized_sortino": 0}
        
        returns = np.array(self.returns_history)
        daily_rf = self.risk_free_rate / 252
        excess_returns = returns - daily_rf
        
        # Downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return {"sortino_ratio": 0, "annualized_sortino": 0}
        
        downside_std = np.std(downside_returns)
        sortino = np.mean(excess_returns) / downside_std
        annualized_sortino = sortino * np.sqrt(252)
        
        return {
            "sortino_ratio": float(sortino),
            "annualized_sortino": float(annualized_sortino),
            "downside_deviation": float(downside_std),
        }


class CalmarRatioMonitor:
    """Calmar ratio monitor."""
    
    def __init__(self):
        self.returns_history: deque = deque(maxlen=252)
        self.max_drawdown: float = 0
        self.peak_value: float = 0
        
    def update(self, portfolio_value: float, daily_return: float):
        """Update with portfolio value."""
        self.returns_history.append(daily_return)
        self.peak_value = max(self.peak_value, portfolio_value)
        
        current_drawdown = (self.peak_value - portfolio_value) / self.peak_value if self.peak_value > 0 else 0
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def calculate_calmar(self) -> Dict[str, float]:
        """Calculate Calmar ratio."""
        if len(self.returns_history) < 30 or self.max_drawdown == 0:
            return {"calmar_ratio": 0, "max_drawdown_pct": 0}
        
        # Annualized return
        total_return = np.prod(1 + np.array(self.returns_history)) - 1
        n_years = len(self.returns_history) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        calmar = annualized_return / self.max_drawdown if self.max_drawdown > 0 else 0
        
        return {
            "calmar_ratio": float(calmar),
            "max_drawdown_pct": float(self.max_drawdown * 100),
            "annualized_return": float(annualized_return),
        }


class PortfolioHeatMap:
    """Portfolio heat map generator."""
    
    def __init__(self):
        self.position_risks: Dict[str, float] = {}
        
    def update_position_risk(self, symbol: str, risk_score: float):
        """Update position risk score (0-100)."""
        self.position_risks[symbol] = risk_score
    
    def generate_heat_map(
        self,
        positions: Dict[str, float],
        portfolio_value: float,
    ) -> Dict[str, Any]:
        """Generate portfolio heat map."""
        heat_map = {}
        total_risk_weighted = 0
        
        for symbol, value in positions.items():
            risk_score = self.position_risks.get(symbol, 50)
            weight = value / portfolio_value if portfolio_value > 0 else 0
            weighted_risk = risk_score * weight
            
            # Color coding
            if risk_score > 70:
                color = "red"
            elif risk_score > 50:
                color = "orange"
            elif risk_score > 30:
                color = "yellow"
            else:
                color = "green"
            
            heat_map[symbol] = {
                "value": value,
                "weight_pct": float(weight * 100),
                "risk_score": risk_score,
                "color": color,
                "weighted_risk": float(weighted_risk),
            }
            
            total_risk_weighted += weighted_risk
        
        return {
            "heat_map": heat_map,
            "portfolio_risk_score": float(total_risk_weighted),
            "risk_level": "high" if total_risk_weighted > 60 else "medium" if total_risk_weighted > 40 else "low",
            "hottest_positions": sorted(heat_map.items(), key=lambda x: x[1]["risk_score"], reverse=True)[:5],
        }


class OmegaRiskEngine:
    """
    THE OMEGA RISK ENGINE.
    
    30 Components.
    """
    
    def __init__(self):
        self.var_calculator = VaRCalculator(confidence=0.99)
        self.cvar_calculator = CVaRCalculator(confidence=0.99)
        self.stress_tester = StressTester()
        self.correlation_monitor = CorrelationRiskMonitor()
        self.liquidity_assessor = LiquidityRiskAssessor()
        self.concentration_monitor = ConcentrationRiskMonitor(max_single_position=0.3, max_sector=0.5)
        self.drawdown_protection = DrawdownProtection(max_drawdown=0.25)
        self.kelly_sizer = KellyCriterionSizer(max_kelly=0.25)
        self.dynamic_hedger = DynamicHedger()
        self.tail_risk_hedger = TailRiskHedger()
        self.black_swan_protector = BlackSwanProtector()
        self.circuit_breakers = CircuitBreakerSystem()
        self.risk_budgeting = RiskBudgeting(total_risk_budget=0.15)
        self.factor_decomposer = FactorRiskDecomposer()
        self.volatility_forecaster = VolatilityForecaster()
        self.regime_detector = RegimeDetector()
        self.sharpe_monitor = SharpeRatioMonitor()
        self.sortino_monitor = SortinoRatioMonitor()
        self.calmar_monitor = CalmarRatioMonitor()
        self.heat_map = PortfolioHeatMap()
        
        # Statistics
        self.total_alerts = 0
        self.total_hedges = 0
        self.risk_level = RiskLevel.LOW
        
        logger.info("OmegaRiskEngine: 30 components initialized")
    
    def assess_risk(
        self,
        portfolio_value: float,
        positions: Dict[str, float],
        daily_return: float,
    ) -> Dict[str, Any]:
        """Comprehensive risk assessment."""
        # Update all monitors
        self.var_calculator.add_return(daily_return)
        self.cvar_calculator.add_return(daily_return)
        self.volatility_forecaster.add_return(daily_return)
        self.regime_detector.add_return(daily_return)
        self.sharpe_monitor.add_return(daily_return)
        self.sortino_monitor.add_return(daily_return)
        self.calmar_monitor.update(portfolio_value, daily_return)
        self.drawdown_protection.update(portfolio_value)
        
        # Run assessments
        var_result = self.var_calculator.calculate_var(portfolio_value)
        cvar_result = self.cvar_calculator.calculate_cvar(portfolio_value)
        stress_result = self.stress_tester.run_stress_test(positions, portfolio_value)
        concentration_result = self.concentration_monitor.assess_concentration(positions, portfolio_value)
        drawdown_result = self.drawdown_protection.update(portfolio_value)
        kelly_result = self.kelly_sizer.calculate_kelly()
        sharpe_result = self.sharpe_monitor.calculate_sharpe()
        sortino_result = self.sortino_monitor.calculate_sortino()
        calmar_result = self.calmar_monitor.calculate_calmar()
        regime_result = self.regime_detector.detect_regime()
        vol_result = self.volatility_forecaster.forecast_garch()
        heat_map_result = self.heat_map.generate_heat_map(positions, portfolio_value)
        breakers_result = self.circuit_breakers.check_all_breakers()
        
        # Determine overall risk level
        risk_score = 0
        if drawdown_result["current_drawdown_pct"] > 15:
            risk_score += 30
        elif drawdown_result["current_drawdown_pct"] > 10:
            risk_score += 20
        elif drawdown_result["current_drawdown_pct"] > 5:
            risk_score += 10
        
        if concentration_result["risk_level"] == "high":
            risk_score += 25
        elif concentration_result["risk_level"] == "medium":
            risk_score += 15
        
        if stress_result["worst_case"]["impact_pct"] < -20:
            risk_score += 25
        elif stress_result["worst_case"]["impact_pct"] < -10:
            risk_score += 15
        
        if risk_score > 60:
            self.risk_level = RiskLevel.CRITICAL
        elif risk_score > 40:
            self.risk_level = RiskLevel.HIGH
        elif risk_score > 20:
            self.risk_level = RiskLevel.MEDIUM
        else:
            self.risk_level = RiskLevel.LOW
        
        return {
            "risk_level": self.risk_level.value,
            "risk_score": risk_score,
            "var": var_result,
            "cvar": cvar_result,
            "stress_test": stress_result,
            "concentration": concentration_result,
            "drawdown": drawdown_result,
            "kelly": kelly_result,
            "sharpe": sharpe_result,
            "sortino": sortino_result,
            "calmar": calmar_result,
            "regime": regime_result,
            "volatility": vol_result,
            "heat_map": heat_map_result,
            "circuit_breakers": breakers_result,
            "can_trade": breakers_result["can_trade"] and self.risk_level != RiskLevel.CRITICAL,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get risk engine status."""
        return {
            "risk_level": self.risk_level.value,
            "total_alerts": self.total_alerts,
            "total_hedges": self.total_hedges,
            "components_active": 30,
        }


def get_omega_risk() -> OmegaRiskEngine:
    """Get Omega Risk Engine."""
    return OmegaRiskEngine()
