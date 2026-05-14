"""
Advanced Risk Management
========================
Every known risk management technique implemented.

Risk Techniques:
1. Kelly Criterion - Optimal position sizing
2. Value at Risk (VaR) - Downside risk quantification
3. Conditional VaR (CVaR) - Expected tail loss
4. Maximum Drawdown Control - Limit drawdowns
5. Volatility Targeting - Scale to target volatility
6. Correlation-Based Risk - Account for correlations
7. Tail Risk Hedging - Protect against black swans
8. Dynamic Stop Losses - Adaptive stops
9. Risk Parity - Equal risk contribution
10. Monte Carlo Simulation - Scenario analysis
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 1. KELLY CRITERION
# ============================================================================

class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing.
    
    Formula: f* = (bp - q) / b
    Where:
        f* = fraction of capital to bet
        b = odds received (win/loss ratio)
        p = probability of winning
        q = probability of losing (1-p)
    """
    
    def __init__(self, 
                 min_fraction: float = 0.01,
                 max_fraction: float = 0.25,
                 fractional_kelly: float = 0.5):  # Half-Kelly for safety
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.fractional_kelly = fractional_kelly
        
        # Tracking
        self.win_history: Deque[bool] = deque(maxlen=100)
        self.win_amounts: Deque[float] = deque(maxlen=100)
        self.loss_amounts: Deque[float] = deque(maxlen=100)
    
    def record_trade(self, won: bool, win_amount: float = 0.0, loss_amount: float = 0.0) -> None:
        """Record trade outcome for Kelly calculation."""
        self.win_history.append(won)
        if won:
            self.win_amounts.append(win_amount)
        else:
            self.loss_amounts.append(loss_amount)
    
    def calculate(self, 
                  win_rate: Optional[float] = None,
                  win_loss_ratio: Optional[float] = None) -> float:
        """Calculate Kelly fraction."""
        # Use provided values or calculate from history
        if win_rate is None:
            if not self.win_history:
                return self.min_fraction
            win_rate = sum(self.win_history) / len(self.win_history)
        
        if win_loss_ratio is None:
            if self.win_amounts and self.loss_amounts:
                avg_win = np.mean(self.win_amounts)
                avg_loss = np.mean(self.loss_amounts)
                if avg_loss > 0:
                    win_loss_ratio = avg_win / avg_loss
                else:
                    win_loss_ratio = 1.5
            else:
                win_loss_ratio = 1.5
        
        # Kelly formula
        p = win_rate
        q = 1 - win_rate
        b = win_loss_ratio
        
        kelly_fraction = (b * p - q) / b
        
        # Apply fractional Kelly for safety
        kelly_fraction *= self.fractional_kelly
        
        # Clamp to bounds
        kelly_fraction = np.clip(kelly_fraction, self.min_fraction, self.max_fraction)
        
        return max(0.0, kelly_fraction)
    
    def get_position_size(self, capital: float, confidence: float = 1.0) -> float:
        """Get position size based on Kelly and confidence."""
        kelly = self.calculate()
        adjusted_kelly = kelly * confidence
        return capital * adjusted_kelly


# ============================================================================
# 2. VALUE AT RISK (VaR)
# ============================================================================

class ValueAtRisk:
    """
    Value at Risk calculation using multiple methods.
    
    Methods:
    1. Historical VaR - Based on historical returns
    2. Parametric VaR - Assuming normal distribution
    3. Cornish-Fisher VaR - Adjusts for skewness/kurtosis
    """
    
    def __init__(self, confidence: float = 0.95, lookback: int = 100):
        self.confidence = confidence
        self.lookback = lookback
        self.returns: Deque[float] = deque(maxlen=1000)
    
    def update(self, returns: List[float]) -> None:
        """Update with new returns."""
        for r in returns:
            self.returns.append(r)
    
    def add_return(self, return_pct: float) -> None:
        """Add a single return."""
        self.returns.append(return_pct)
    
    def calculate_historical(self) -> float:
        """Calculate Historical VaR."""
        if len(self.returns) < 20:
            return 0.02  # Default 2%
        
        returns_array = np.array(list(self.returns)[-self.lookback:])
        var = np.percentile(returns_array, (1 - self.confidence) * 100)
        return abs(var)
    
    def calculate_parametric(self) -> float:
        """Calculate Parametric VaR (assumes normal distribution)."""
        if len(self.returns) < 20:
            return 0.02
        
        returns_array = np.array(list(self.returns)[-self.lookback:])
        mean = np.mean(returns_array)
        std = np.std(returns_array)
        
        # Z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf(1 - self.confidence)
        
        var = mean + z_score * std
        return abs(var)
    
    def calculate_cornish_fisher(self) -> float:
        """Calculate Cornish-Fisher VaR (adjusts for non-normality)."""
        if len(self.returns) < 30:
            return self.calculate_parametric()
        
        returns_array = np.array(list(self.returns)[-self.lookback:])
        mean = np.mean(returns_array)
        std = np.std(returns_array)
        skew = float(np.mean(((returns_array - mean) / std) ** 3))
        kurtosis = float(np.mean(((returns_array - mean) / std) ** 4) - 3)
        
        from scipy import stats
        z = stats.norm.ppf(1 - self.confidence)
        
        # Cornish-Fisher adjustment
        z_cf = (z + 
                (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurtosis / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)
        
        var = mean + z_cf * std
        return abs(var)
    
    def get_var(self, method: str = "historical") -> float:
        """Get VaR using specified method."""
        if method == "historical":
            return self.calculate_historical()
        elif method == "parametric":
            return self.calculate_parametric()
        elif method == "cornish_fisher":
            return self.calculate_cornish_fisher()
        else:
            return self.calculate_historical()


# ============================================================================
# 3. CONDITIONAL VaR (CVaR) / Expected Shortfall
# ============================================================================

class ConditionalVaR:
    """
    Conditional VaR (CVaR) - Expected loss given that loss exceeds VaR.
    Also known as Expected Shortfall.
    """
    
    def __init__(self, confidence: float = 0.95, lookback: int = 100):
        self.confidence = confidence
        self.lookback = lookback
        self.returns: Deque[float] = deque(maxlen=1000)
    
    def add_return(self, return_pct: float) -> None:
        """Add a single return."""
        self.returns.append(return_pct)
    
    def calculate(self) -> float:
        """Calculate CVaR."""
        if len(self.returns) < 20:
            return 0.03  # Default 3%
        
        returns_array = np.array(list(self.returns)[-self.lookback:])
        
        # Find VaR threshold
        var_threshold = np.percentile(returns_array, (1 - self.confidence) * 100)
        
        # Average of returns below VaR
        tail_returns = returns_array[returns_array <= var_threshold]
        
        if len(tail_returns) == 0:
            return abs(var_threshold)
        
        cvar = np.mean(tail_returns)
        return abs(cvar)


# ============================================================================
# 4. MAXIMUM DRAWDOWN CONTROL
# ============================================================================

class MaxDrawdownControl:
    """
    Limits position sizes based on current drawdown.
    As drawdown increases, position sizes decrease.
    """
    
    def __init__(self, 
                 max_drawdown: float = 0.15,
                 scaling_start: float = 0.05,
                 scaling_factor: float = 2.0):
        self.max_drawdown = max_drawdown
        self.scaling_start = scaling_start
        self.scaling_factor = scaling_factor
        
        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0
    
    def update_equity(self, equity: float) -> None:
        """Update current equity."""
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity
    
    def get_current_drawdown(self) -> float:
        """Get current drawdown percentage."""
        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity
    
    def get_position_multiplier(self) -> float:
        """Get multiplier to apply to position sizes."""
        dd = self.get_current_drawdown()
        
        if dd < self.scaling_start:
            return 1.0  # Full position size
        
        # Scale down linearly
        scaled_dd = (dd - self.scaling_start) / (self.max_drawdown - self.scaling_start)
        scaled_dd = min(1.0, scaled_dd)
        
        multiplier = 1.0 - scaled_dd * self.scaling_factor
        return max(0.1, multiplier)  # Never go to zero
    
    def should_stop_trading(self) -> bool:
        """Check if we should stop trading due to drawdown."""
        return self.get_current_drawdown() >= self.max_drawdown


# ============================================================================
# 5. VOLATILITY TARGETING
# ============================================================================

class VolatilityTargeting:
    """
    Scales positions to target a specific volatility level.
    
    If market volatility doubles, position sizes halve.
    """
    
    def __init__(self, 
                 target_volatility: float = 0.15,  # 15% annualized
                 lookback: int = 20,
                 min_multiplier: float = 0.25,
                 max_multiplier: float = 2.0):
        self.target_vol = target_volatility
        self.lookback = lookback
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        
        self.returns: Deque[float] = deque(maxlen=1000)
    
    def add_return(self, return_pct: float) -> None:
        """Add a return observation."""
        self.returns.append(return_pct)
    
    def calculate_current_volatility(self) -> float:
        """Calculate current realized volatility (annualized)."""
        if len(self.returns) < self.lookback:
            return self.target_vol
        
        recent = list(self.returns)[-self.lookback:]
        daily_vol = np.std(recent)
        annualized_vol = daily_vol * np.sqrt(252)  # Trading days per year
        
        return annualized_vol
    
    def get_volatility_multiplier(self) -> float:
        """Get multiplier to scale positions."""
        current_vol = self.calculate_current_volatility()
        
        if current_vol == 0:
            return 1.0
        
        # Target / Current = multiplier
        multiplier = self.target_vol / current_vol
        
        # Clamp
        multiplier = np.clip(multiplier, self.min_multiplier, self.max_multiplier)
        
        return multiplier
    
    def get_volatility_scaled_position(self, base_position: float) -> float:
        """Get position size scaled for volatility."""
        multiplier = self.get_volatility_multiplier()
        return base_position * multiplier


# ============================================================================
# 6. CORRELATION-BASED RISK
# ============================================================================

class CorrelationRisk:
    """
    Adjusts risk based on portfolio correlations.
    When assets are highly correlated, reduce exposure.
    """
    
    def __init__(self, correlation_threshold: float = 0.7):
        self.correlation_threshold = correlation_threshold
        self.asset_returns: Dict[str, Deque[float]] = {}
    
    def update_asset_returns(self, asset: str, returns: List[float]) -> None:
        """Update returns for an asset."""
        if asset not in self.asset_returns:
            self.asset_returns[asset] = deque(maxlen=200)
        
        for r in returns:
            self.asset_returns[asset].append(r)
    
    def calculate_correlation(self, asset1: str, asset2: str) -> float:
        """Calculate correlation between two assets."""
        if asset1 not in self.asset_returns or asset2 not in self.asset_returns:
            return 0.0
        
        r1 = list(self.asset_returns[asset1])
        r2 = list(self.asset_returns[asset2])
        
        min_len = min(len(r1), len(r2))
        if min_len < 20:
            return 0.0
        
        return float(np.corrcoef(r1[-min_len:], r2[-min_len:])[0, 1])
    
    def get_average_correlation(self, assets: List[str]) -> float:
        """Get average pairwise correlation."""
        correlations = []
        
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                corr = self.calculate_correlation(assets[i], assets[j])
                correlations.append(corr)
        
        if not correlations:
            return 0.0
        
        return np.mean(correlations)
    
    def get_correlation_adjustment(self, assets: List[str]) -> float:
        """Get position size adjustment based on correlations."""
        avg_corr = self.get_average_correlation(assets)
        
        if avg_corr < self.correlation_threshold:
            return 1.0  # No adjustment
        
        # Scale down as correlation increases
        excess_corr = avg_corr - self.correlation_threshold
        adjustment = 1.0 - excess_corr * 0.5
        
        return max(0.3, adjustment)


# ============================================================================
# 7. TAIL RISK HEDGING
# ============================================================================

class TailRiskHedging:
    """
    Detects and hedges against tail risk events.
    """
    
    def __init__(self, 
                 var_threshold: float = 0.02,
                 tail_detection_window: int = 20):
        self.var_threshold = var_threshold
        self.window = tail_detection_window
        
        self.returns: Deque[float] = deque(maxlen=500)
        self.tail_events: List[Dict] = []
    
    def add_return(self, return_pct: float) -> None:
        """Add a return observation."""
        self.returns.append(return_pct)
        
        # Check for tail event
        if abs(return_pct) > self.var_threshold * 3:  # 3x VaR
            self.tail_events.append({
                "return": return_pct,
                "timestamp": time.time(),
                "severity": abs(return_pct) / self.var_threshold,
            })
    
    def get_tail_risk_level(self) -> float:
        """Get current tail risk level (0-1)."""
        if len(self.returns) < self.window:
            return 0.5
        
        recent = list(self.returns)[-self.window:]
        
        # Kurtosis as measure of tail heaviness
        if np.std(recent) == 0:
            return 0.5
        
        kurtosis = float(np.mean(((recent - np.mean(recent)) / np.std(recent)) ** 4) - 3)
        
        # Map to 0-1
        risk_level = 0.5 + np.tanh(kurtosis / 2) * 0.5
        return np.clip(risk_level, 0.0, 1.0)
    
    def should_hedge(self) -> bool:
        """Check if tail hedging should be activated."""
        risk_level = self.get_tail_risk_level()
        recent_events = len([e for e in self.tail_events if time.time() - e["timestamp"] < 3600])
        
        return risk_level > 0.7 or recent_events > 2
    
    def get_hedge_size(self, position_value: float) -> float:
        """Get hedge position size."""
        risk_level = self.get_tail_risk_level()
        hedge_ratio = risk_level * 0.3  # Hedge up to 30% of position
        
        return position_value * hedge_ratio


# ============================================================================
# 8. DYNAMIC STOP LOSSES
# ============================================================================

class DynamicStopLoss:
    """
    Adaptive stop losses that adjust based on volatility and performance.
    """
    
    def __init__(self,
                 base_atr_multiplier: float = 2.0,
                 min_stop_pct: float = 0.005,
                 max_stop_pct: float = 0.05,
                 breakeven_threshold: float = 0.02):
        self.base_atr_mult = base_atr_multiplier
        self.min_stop = min_stop_pct
        self.max_stop = max_stop_pct
        self.breakeven_threshold = breakeven_threshold
        
        self.atr_history: Deque[float] = deque(maxlen=100)
    
    def update_atr(self, atr: float) -> None:
        """Update ATR value."""
        self.atr_history.append(atr)
    
    def calculate_stop_distance(self, 
                                 entry_price: float,
                                 current_price: float,
                                 side: str) -> float:
        """Calculate stop loss distance."""
        if not self.atr_history:
            # Default to percentage
            return entry_price * 0.02
        
        current_atr = self.atr_history[-1]
        stop_distance = current_atr * self.base_atr_mult
        
        # Check if we should tighten stop (in profit)
        profit_pct = (current_price - entry_price) / entry_price
        if side == "sell":
            profit_pct = -profit_pct
        
        if profit_pct > self.breakeven_threshold:
            # Move stop to breakeven + small profit
            return entry_price * 0.005
        
        # Clamp
        stop_pct = stop_distance / entry_price
        stop_pct = np.clip(stop_pct, self.min_stop, self.max_stop)
        
        return entry_price * stop_pct
    
    def get_stop_price(self,
                       entry_price: float,
                       current_price: float,
                       side: str) -> float:
        """Get stop loss price."""
        stop_distance = self.calculate_stop_distance(entry_price, current_price, side)
        
        if side == "buy":
            return current_price - stop_distance
        else:
            return current_price + stop_distance


# ============================================================================
# 9. RISK PARITY
# ============================================================================

class RiskParity:
    """
    Allocates capital so each position contributes equal risk.
    """
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.asset_volatilities: Dict[str, Deque[float]] = {}
        self.asset_correlations: Dict[Tuple[str, str], float] = {}
    
    def update_volatility(self, asset: str, volatility: float) -> None:
        """Update volatility for an asset."""
        if asset not in self.asset_volatilities:
            self.asset_volatilities[asset] = deque(maxlen=self.lookback)
        self.asset_volatilities[asset].append(volatility)
    
    def update_correlation(self, asset1: str, asset2: str, correlation: float) -> None:
        """Update correlation between assets."""
        key = (asset1, asset2)
        self.asset_correlations[key] = correlation
    
    def calculate_equal_risk_weights(self,
                                      volatilities: Dict[str, float],
                                      target_risk: float = 0.1) -> Dict[str, float]:
        """Calculate weights for equal risk contribution."""
        assets = list(volatilities.keys())
        n = len(assets)
        
        if n == 0:
            return {}
        
        # Simple inverse volatility weighting
        inv_vol = {a: 1.0 / max(v, 0.01) for a, v in volatilities.items()}
        total_inv_vol = sum(inv_vol.values())
        
        weights = {a: (inv_vol[a] / total_inv_vol) * (target_risk * n) for a in assets}
        
        return weights


# ============================================================================
# 10. MONTE CARLO SIMULATION
# ============================================================================

class MonteCarloRisk:
    """
    Monte Carlo simulation for risk analysis.
    """
    
    def __init__(self, num_simulations: int = 1000, horizon: int = 20):
        self.num_simulations = num_simulations
        self.horizon = horizon
        self.returns: Deque[float] = deque(maxlen=500)
    
    def add_return(self, return_pct: float) -> None:
        """Add a return observation."""
        self.returns.append(return_pct)
    
    def simulate_paths(self,
                       initial_value: float,
                       num_days: Optional[int] = None) -> np.ndarray:
        """Simulate future price paths."""
        if len(self.returns) < 20:
            # Default simulation
            mean_return = 0.0001
            volatility = 0.02
        else:
            returns_array = np.array(self.returns)
            mean_return = np.mean(returns_array)
            volatility = np.std(returns_array)
        
        horizon = num_days or self.horizon
        
        # Generate random paths
        daily_returns = np.random.normal(
            mean_return, 
            volatility, 
            (self.num_simulations, horizon)
        )
        
        # Cumulative returns
        price_paths = initial_value * np.exp(np.cumsum(daily_returns, axis=1))
        
        return price_paths
    
    def calculate_var(self, 
                      initial_value: float,
                      confidence: float = 0.95,
                      horizon: int = None) -> float:
        """Calculate VaR using Monte Carlo."""
        paths = self.simulate_paths(initial_value, horizon)
        
        # Final values
        final_values = paths[:, -1]
        
        # Calculate losses
        losses = initial_value - final_values
        
        # VaR at confidence level
        var = np.percentile(losses, confidence * 100)
        
        return max(0.0, var)
    
    def get_worst_case(self, initial_value: float, horizon: int = None) -> float:
        """Get worst case scenario."""
        paths = self.simulate_paths(initial_value, horizon)
        return float(np.min(paths[:, -1]))
    
    def get_best_case(self, initial_value: float, horizon: int = None) -> float:
        """Get best case scenario."""
        paths = self.simulate_paths(initial_value, horizon)
        return float(np.max(paths[:, -1]))


# ============================================================================
# UNIFIED RISK MANAGER
# ============================================================================

class UnifiedRiskManager:
    """
    Combines all risk management techniques into a unified system.
    """
    
    def __init__(self, capital: float = 10000.0):
        self.capital = capital
        self.peak_capital = capital
        
        # Initialize all risk modules
        self.kelly = KellyCriterion()
        self.var = ValueAtRisk(confidence=0.95)
        self.cvar = ConditionalVaR(confidence=0.95)
        self.drawdown_control = MaxDrawdownControl(max_drawdown=0.15)
        self.vol_targeting = VolatilityTargeting(target_volatility=0.15)
        self.correlation_risk = CorrelationRisk()
        self.tail_hedging = TailRiskHedging()
        self.dynamic_stops = DynamicStopLoss()
        self.risk_parity = RiskParity()
        self.monte_carlo = MonteCarloRisk()
        
        # Current state
        self.current_positions: Dict[str, Dict] = {}
        self.daily_pnl: float = 0.0
        self.consecutive_losses: int = 0
    
    def update_capital(self, new_capital: float) -> None:
        """Update current capital."""
        self.capital = new_capital
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
        
        self.drawdown_control.update_equity(new_capital)
    
    def calculate_position_size(self,
                                 signal_confidence: float,
                                 signal_strength: float,
                                 assets: List[str] = None) -> float:
        """
        Calculate optimal position size using all risk modules.
        """
        # Base Kelly calculation
        kelly_size = self.kelly.get_position_size(self.capital, signal_confidence)
        
        # Volatility adjustment
        vol_multiplier = self.vol_targeting.get_volatility_multiplier()
        
        # Drawdown adjustment
        dd_multiplier = self.drawdown_control.get_position_multiplier()
        
        # Correlation adjustment (if multiple assets)
        if assets and len(assets) > 1:
            corr_adjustment = self.correlation_risk.get_correlation_adjustment(assets)
        else:
            corr_adjustment = 1.0
        
        # Tail risk adjustment
        if self.tail_hedging.should_hedge():
            tail_adjustment = 0.7  # Reduce by 30%
        else:
            tail_adjustment = 1.0
        
        # Combine all adjustments
        position_size = kelly_size * vol_multiplier * dd_multiplier * corr_adjustment * tail_adjustment
        
        # Signal strength scaling
        position_size *= signal_strength
        
        # Final bounds
        max_position = self.capital * 0.25  # Never more than 25%
        position_size = min(position_size, max_position)
        
        return max(0.0, position_size)
    
    def should_take_trade(self) -> Tuple[bool, str]:
        """Check if we should take a trade based on risk rules."""
        # Check drawdown limit
        if self.drawdown_control.should_stop_trading():
            return False, f"Max drawdown reached ({self.drawdown_control.get_current_drawdown():.1%})"
        
        # Check consecutive losses
        if self.consecutive_losses >= 5:
            return False, f"Too many consecutive losses ({self.consecutive_losses})"
        
        # Check daily loss limit
        daily_loss_limit = -self.capital * 0.02  # 2% daily loss limit
        if self.daily_pnl < daily_loss_limit:
            return False, f"Daily loss limit reached ({self.daily_pnl:.2f})"
        
        return True, "OK"
    
    def record_trade(self,
                     pnl: float,
                     entry_price: float,
                     exit_price: float,
                     side: str,
                     position_value: float) -> None:
        """Record a completed trade."""
        self.update_capital(self.capital + pnl)
        self.daily_pnl += pnl
        
        # Update Kelly
        self.kelly.record_trade(
            won=pnl > 0,
            win_amount=abs(pnl) if pnl > 0 else 0,
            loss_amount=abs(pnl) if pnl < 0 else 0
        )
        
        # Update VaR
        return_pct = pnl / position_value if position_value > 0 else 0
        self.var.add_return(return_pct)
        self.cvar.add_return(return_pct)
        self.monte_carlo.add_return(return_pct)
        
        # Track consecutive losses
        if pnl > 0:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Get comprehensive risk report."""
        return {
            "capital": self.capital,
            "peak_capital": self.peak_capital,
            "current_drawdown": self.drawdown_control.get_current_drawdown(),
            "daily_pnl": self.daily_pnl,
            "consecutive_losses": self.consecutive_losses,
            "kelly_fraction": self.kelly.calculate(),
            "var_95": self.var.get_var("historical"),
            "cvar_95": self.cvar.calculate(),
            "vol_multiplier": self.vol_targeting.get_volatility_multiplier(),
            "dd_multiplier": self.drawdown_control.get_position_multiplier(),
            "tail_risk_level": self.tail_hedging.get_tail_risk_level(),
            "should_hedge": self.tail_hedging.should_hedge(),
            "can_trade": self.should_take_trade()[0],
        }


# Singleton
_risk_manager: Optional[UnifiedRiskManager] = None


def get_risk_manager(capital: float = 10000.0) -> UnifiedRiskManager:
    """Get or create singleton risk manager."""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = UnifiedRiskManager(capital)
    return _risk_manager


def reset_risk_manager() -> None:
    """Reset singleton."""
    global _risk_manager
    _risk_manager = None
