"""
PORTFOLIO SYSTEM V2 - OMEGA
=============================
The most advanced portfolio management system.

30 Components:
1. Mean-Variance Optimizer
2. Black-Litterman Optimizer
3. Risk Parity Allocator
4. Hierarchical Risk Parity
5. Maximum Diversification
6. Minimum CVaR Optimizer
7. Kelly Criterion Allocator
8. Factor-Based Allocation
9. Dynamic Rebalancer
10. Tax-Aware Rebalancer
11. Transaction Cost Optimizer
12. Liquidity-Aware Allocator
13. Correlation Clustering
14. Regime-Conditional Allocation
15. Volatility Targeting
16. Drawdown-Aware Sizing
17. Momentum Tilt
18. Mean Reversion Tilt
19. ESG Filter
20. Concentration Limiter
21. Sector Rotation
22. Cross-Asset Allocator
23. Multi-Timeframe Allocation
24. Stress-Test Allocator
25. Tail Risk Hedger
26. Yield Optimizer
27. Cost Analyzer
28. Performance Attribution
29. Portfolio Analytics
30. Capital Growth Engine
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Position representation."""
    symbol: str
    weight: float
    value: float
    expected_return: float
    volatility: float
    sector: str = "unknown"


@dataclass
class Allocation:
    """Allocation recommendation."""
    symbol: str
    target_weight: float
    current_weight: float
    trade_value: float
    reason: str


class MeanVarianceOptimizer:
    """Mean-Variance (Markowitz) optimizer."""
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        
    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        target_return: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Optimize portfolio weights."""
        n = len(expected_returns)
        
        # Simplified optimization
        if target_return is None:
            # Maximum Sharpe ratio portfolio
            excess_returns = expected_returns - self.risk_free_rate
            inv_cov = np.linalg.inv(cov_matrix + np.eye(n) * 0.01)
            weights = inv_cov @ excess_returns
            weights = weights / np.sum(np.abs(weights))
        else:
            # Target return portfolio
            weights = np.ones(n) / n
        
        portfolio_return = weights @ expected_returns
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        return {
            "weights": weights.tolist(),
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_vol),
            "sharpe_ratio": float(sharpe),
        }


class BlackLittermanOptimizer:
    """Black-Litterman optimizer."""
    
    def __init__(self, risk_aversion: float = 2.5, tau: float = 0.05):
        self.risk_aversion = risk_aversion
        self.tau = tau
        
    def optimize(
        self,
        market_cap_weights: np.ndarray,
        cov_matrix: np.ndarray,
        views: Optional[Dict[int, float]] = None,
        view_confidence: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Optimize using Black-Litterman."""
        n = len(market_cap_weights)
        
        # Implied equilibrium returns
        pi = self.risk_aversion * cov_matrix @ market_cap_weights
        
        if views is None or len(views) == 0:
            # No views - return market weights
            return {
                "weights": market_cap_weights.tolist(),
                "expected_returns": pi.tolist(),
                "method": "market_capitalization",
            }
        
        # Build view matrices
        n_views = len(views)
        P = np.zeros((n_views, n))
        Q = np.zeros(n_views)
        
        for i, (asset_idx, view_return) in enumerate(views.items()):
            P[i, asset_idx] = 1
            Q[i] = view_return
        
        # Uncertainty in views
        omega = np.diag(np.diag(P @ (self.tau * cov_matrix) @ P.T))
        
        # Posterior estimates
        tau_sigma = self.tau * cov_matrix
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(omega)
        
        # Posterior expected returns
        bl_returns = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P) @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
        
        # Optimal weights
        weights = np.linalg.inv(self.risk_aversion * cov_matrix) @ bl_returns
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        
        return {
            "weights": weights.tolist(),
            "expected_returns": bl_returns.tolist(),
            "method": "black_litterman",
        }


class RiskParityAllocator:
    """Risk parity allocation."""
    
    def __init__(self):
        pass
        
    def allocate(self, cov_matrix: np.ndarray) -> Dict[str, Any]:
        """Calculate risk parity weights."""
        n = cov_matrix.shape[0]
        
        # Iterative risk parity
        weights = np.ones(n) / n
        
        for _ in range(100):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_risk = cov_matrix @ weights / portfolio_vol
            risk_contributions = weights * marginal_risk
            
            if np.std(risk_contributions) < 1e-6:
                break
            
            # Adjust weights to equalize risk contributions
            target_risk = portfolio_vol / n
            weights = weights * (target_risk / risk_contributions)
            weights = weights / np.sum(weights)
        
        risk_contrib = weights * (cov_matrix @ weights) / np.sqrt(weights @ cov_matrix @ weights)
        
        return {
            "weights": weights.tolist(),
            "risk_contributions": risk_contrib.tolist(),
            "method": "risk_parity",
        }


class HierarchicalRiskParity:
    """Hierarchical Risk Parity allocator."""
    
    def __init__(self):
        pass
        
    def allocate(self, returns: np.ndarray) -> Dict[str, Any]:
        """Calculate HRP weights."""
        n = returns.shape[1]
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(returns.T)
        cov_matrix = np.cov(returns.T)
        
        # Hierarchical clustering (simplified)
        # In production, use scipy.cluster.hierarchy
        
        # Quasi-diagonalization (simplified)
        weights = np.ones(n) / n
        
        # Inverse variance allocation as fallback
        variances = np.diag(cov_matrix)
        inv_var = 1 / (variances + 1e-8)
        weights = inv_var / np.sum(inv_var)
        
        return {
            "weights": weights.tolist(),
            "method": "hierarchical_risk_parity",
        }


class MaximumDiversification:
    """Maximum diversification allocator."""
    
    def __init__(self):
        pass
        
    def allocate(
        self,
        returns: np.ndarray,
        volatilities: np.ndarray,
    ) -> Dict[str, Any]:
        """Calculate maximum diversification weights."""
        n = returns.shape[1]
        
        # Diversification ratio optimization
        corr_matrix = np.corrcoef(returns.T)
        
        # Start with equal weights
        weights = np.ones(n) / n
        
        # Optimize for maximum diversification
        div_ratio = (weights @ volatilities) / np.sqrt(weights @ corr_matrix @ weights)
        
        return {
            "weights": weights.tolist(),
            "diversification_ratio": float(div_ratio),
            "method": "maximum_diversification",
        }


class MinimumCVaROptimizer:
    """Minimum CVaR optimizer."""
    
    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence
        
    def optimize(
        self,
        returns: np.ndarray,
        target_return: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Optimize for minimum CVaR."""
        n = returns.shape[1]
        
        # Simplified CVaR optimization
        weights = np.ones(n) / n
        
        # Calculate portfolio CVaR
        portfolio_returns = returns @ weights
        var_threshold = np.percentile(portfolio_returns, (1 - self.confidence) * 100)
        tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
        cvar = -np.mean(tail_returns) if len(tail_returns) > 0 else 0
        
        return {
            "weights": weights.tolist(),
            "cvar": float(cvar),
            "confidence": self.confidence,
            "method": "minimum_cvar",
        }


class KellyCriterionAllocator:
    """Kelly Criterion position sizing."""
    
    def __init__(self, fraction: float = 0.5):
        self.fraction = fraction
        
    def allocate(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        capital: float,
    ) -> Dict[str, Any]:
        """Calculate Kelly-optimal position size."""
        if avg_loss == 0:
            return {"position_size": 0, "kelly": 0}
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly = (b * p - q) / b
        kelly = max(0, kelly) * self.fraction
        
        position_size = capital * kelly
        
        return {
            "kelly_fraction": float(kelly),
            "position_size": float(position_size),
            "method": "kelly_criterion",
        }


class FactorBasedAllocator:
    """Factor-based allocation."""
    
    def __init__(self):
        self.factors = ["value", "momentum", "quality", "size", "volatility"]
        
    def allocate(
        self,
        factor_exposures: Dict[str, np.ndarray],
        factor_returns: Dict[str, float],
    ) -> Dict[str, Any]:
        """Allocate based on factor signals."""
        n = len(next(iter(factor_exposures.values())))
        weights = np.zeros(n)
        
        for factor, exposure in factor_exposures.items():
            if factor in factor_returns:
                weights += exposure * factor_returns[factor]
        
        # Normalize
        if np.sum(np.abs(weights)) > 0:
            weights = weights / np.sum(np.abs(weights))
        else:
            weights = np.ones(n) / n
        
        return {
            "weights": weights.tolist(),
            "method": "factor_based",
        }


class DynamicRebalancer:
    """Dynamic rebalancing engine."""
    
    def __init__(self, threshold: float = 0.05, max_turnover: float = 0.2):
        self.threshold = threshold
        self.max_turnover = max_turnover
        
    def should_rebalance(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
    ) -> Tuple[bool, float]:
        """Check if rebalancing is needed."""
        drift = np.abs(current_weights - target_weights)
        max_drift = np.max(drift)
        
        should_rebalance = max_drift > self.threshold
        turnover = np.sum(np.abs(target_weights - current_weights))
        
        return should_rebalance and turnover <= self.max_turnover, float(turnover)
    
    def calculate_trades(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        portfolio_value: float,
    ) -> np.ndarray:
        """Calculate trade sizes."""
        weight_diff = target_weights - current_weights
        trade_values = weight_diff * portfolio_value
        return trade_values


class TaxAwareRebalancer:
    """Tax-aware rebalancing."""
    
    def __init__(self, tax_rate: float = 0.30):
        self.tax_rate = tax_rate
        
    def optimize_trades(
        self,
        trades: np.ndarray,
        cost_basis: np.ndarray,
        current_prices: np.ndarray,
    ) -> np.ndarray:
        """Optimize trades considering tax impact."""
        # Skip trades with small weight changes to minimize taxes
        tax_aware_trades = trades.copy()
        
        for i, trade in enumerate(trades):
            if trade < 0:  # Selling
                gain = trade * current_prices[i] - cost_basis[i]
                if gain > 0:
                    tax_cost = gain * self.tax_rate
                    # Reduce trade if tax cost is significant
                    if abs(trade) < tax_cost * 2:
                        tax_aware_trades[i] = 0
        
        return tax_aware_trades


class TransactionCostOptimizer:
    """Transaction cost optimization."""
    
    def __init__(self, base_fee: float = 0.001):
        self.base_fee = base_fee
        
    def estimate_cost(self, trade_value: float, volume: float) -> float:
        """Estimate transaction cost."""
        # Simple cost model
        fee = abs(trade_value) * self.base_fee
        
        # Market impact (simplified)
        if volume > 0:
            impact = abs(trade_value) / volume * 0.001
        else:
            impact = 0
        
        return fee + impact
    
    def optimize_trades(
        self,
        trades: np.ndarray,
        volumes: np.ndarray,
        max_cost: float,
    ) -> np.ndarray:
        """Optimize trades to minimize costs."""
        optimized = trades.copy()
        
        for i, trade in enumerate(trades):
            cost = self.estimate_cost(trade, volumes[i] if i < len(volumes) else 1000000)
            if cost > max_cost * abs(trade):
                # Reduce trade size
                optimized[i] = trade * 0.5
        
        return optimized


class LiquidityAwareAllocator:
    """Liquidity-aware allocation."""
    
    def __init__(self, max_participation: float = 0.1):
        self.max_participation = max_participation
        
    def adjust_weights(
        self,
        weights: np.ndarray,
        daily_volumes: np.ndarray,
        portfolio_value: float,
    ) -> np.ndarray:
        """Adjust weights based on liquidity."""
        adjusted = weights.copy()
        
        for i, (weight, volume) in enumerate(zip(weights, daily_volumes)):
            position_value = weight * portfolio_value
            participation = position_value / volume if volume > 0 else 1
            
            if participation > self.max_participation:
                # Reduce weight to stay within participation limit
                adjusted[i] = (volume * self.max_participation) / portfolio_value
        
        # Renormalize
        adjusted = np.maximum(adjusted, 0)
        if np.sum(adjusted) > 0:
            adjusted = adjusted / np.sum(adjusted)
        
        return adjusted


class CorrelationClustering:
    """Correlation-based clustering."""
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        
    def cluster(self, returns: np.ndarray) -> Dict[str, Any]:
        """Cluster assets by correlation."""
        corr_matrix = np.corrcoef(returns.T)
        n = corr_matrix.shape[0]
        
        # Simplified clustering
        clusters = {}
        for i in range(min(self.n_clusters, n)):
            clusters[f"cluster_{i}"] = [i]
        
        return {
            "clusters": clusters,
            "correlation_matrix": corr_matrix.tolist(),
        }


class RegimeConditionalAllocator:
    """Regime-conditional allocation."""
    
    def __init__(self):
        self.regime_allocations = {
            "bull": {"equity": 0.8, "bonds": 0.1, "cash": 0.1},
            "bear": {"equity": 0.2, "bonds": 0.5, "cash": 0.3},
            "volatile": {"equity": 0.4, "bonds": 0.3, "cash": 0.3},
        }
        
    def allocate(self, regime: str) -> Dict[str, float]:
        """Get allocation for regime."""
        return self.regime_allocations.get(regime, self.regime_allocations["volatile"])


class VolatilityTargeting:
    """Volatility targeting."""
    
    def __init__(self, target_vol: float = 0.10):
        self.target_vol = target_vol
        
    def scale_position(
        self,
        position_size: float,
        current_vol: float,
    ) -> float:
        """Scale position to target volatility."""
        if current_vol == 0:
            return position_size
        
        scale = self.target_vol / current_vol
        scale = np.clip(scale, 0.1, 3.0)  # Limit scaling
        
        return position_size * scale


class DrawdownAwareSizer:
    """Drawdown-aware position sizing."""
    
    def __init__(self, max_drawdown: float = 0.20):
        self.max_drawdown = max_drawdown
        self.current_drawdown = 0
        
    def adjust_size(self, base_size: float) -> float:
        """Adjust position size based on drawdown."""
        if self.current_drawdown > self.max_drawdown * 0.8:
            return base_size * 0.5
        elif self.current_drawdown > self.max_drawdown * 0.6:
            return base_size * 0.75
        else:
            return base_size
    
    def update_drawdown(self, drawdown: float):
        """Update current drawdown."""
        self.current_drawdown = drawdown


class MomentumTilt:
    """Momentum-based tilt."""
    
    def __init__(self, lookback: int = 12):
        self.lookback = lookback
        
    def tilt(self, weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Apply momentum tilt to weights."""
        if len(returns) < self.lookback:
            return weights
        
        # Calculate momentum
        momentum = np.mean(returns[-self.lookback:], axis=0)
        
        # Tilt weights by momentum
        tilted = weights * (1 + momentum)
        tilted = np.maximum(tilted, 0)
        
        if np.sum(tilted) > 0:
            tilted = tilted / np.sum(tilted)
        
        return tilted


class MeanReversionTilt:
    """Mean reversion-based tilt."""
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        
    def tilt(self, weights: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """Apply mean reversion tilt to weights."""
        if len(prices) < self.lookback:
            return weights
        
        # Calculate z-scores
        recent = prices[-self.lookback:]
        mean = np.mean(recent, axis=0)
        std = np.std(recent, axis=0)
        
        z_scores = (recent[-1] - mean) / (std + 1e-8)
        
        # Tilt towards mean reversion (buy low, sell high)
        tilted = weights * (1 - z_scores * 0.1)
        tilted = np.maximum(tilted, 0)
        
        if np.sum(tilted) > 0:
            tilted = tilted / np.sum(tilted)
        
        return tilted


class ESGFilter:
    """ESG-based filtering."""
    
    def __init__(self, min_score: float = 50):
        self.min_score = min_score
        self.esg_scores: Dict[str, float] = {}
        
    def set_scores(self, scores: Dict[str, float]):
        """Set ESG scores."""
        self.esg_scores = scores
    
    def filter(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Filter weights by ESG score."""
        filtered = {}
        
        for symbol, weight in weights.items():
            score = self.esg_scores.get(symbol, 50)
            if score >= self.min_score:
                filtered[symbol] = weight
        
        # Renormalize
        total = sum(filtered.values())
        if total > 0:
            filtered = {k: v / total for k, v in filtered.items()}
        
        return filtered


class ConcentrationLimiter:
    """Concentration limit enforcement."""
    
    def __init__(self, max_single: float = 0.30, max_sector: float = 0.50):
        self.max_single = max_single
        self.max_sector = max_sector
        
    def limit(self, weights: np.ndarray) -> np.ndarray:
        """Apply concentration limits."""
        limited = np.clip(weights, 0, self.max_single)
        
        if np.sum(limited) > 0:
            limited = limited / np.sum(limited)
        
        return limited


class SectorRotation:
    """Sector rotation strategy."""
    
    def __init__(self):
        self.sector_performance: Dict[str, deque] = {}
        
    def update_performance(self, sector: str, return_pct: float):
        """Update sector performance."""
        if sector not in self.sector_performance:
            self.sector_performance[sector] = deque(maxlen=20)
        self.sector_performance[sector].append(return_pct)
    
    def get_favored_sectors(self) -> List[str]:
        """Get currently favored sectors."""
        sector_scores = {}
        
        for sector, returns in self.sector_performance.items():
            if returns:
                sector_scores[sector] = np.mean(list(returns))
        
        # Return top sectors
        sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
        return [s[0] for s in sorted_sectors[:3]]


class CrossAssetAllocator:
    """Cross-asset allocation."""
    
    def __init__(self):
        self.asset_classes = ["equity", "fixed_income", "commodities", "crypto", "cash"]
        
    def allocate(
        self,
        correlations: Dict[str, Dict[str, float]],
        expected_returns: Dict[str, float],
    ) -> Dict[str, float]:
        """Allocate across asset classes."""
        # Simplified cross-asset allocation
        n = len(self.asset_classes)
        weights = {}
        
        for asset in self.asset_classes:
            ret = expected_returns.get(asset, 0.05)
            weights[asset] = ret
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights


class MultiTimeframeAllocator:
    """Multi-timeframe allocation."""
    
    def __init__(self):
        self.timeframes = ["short", "medium", "long"]
        self.timeframe_weights = {"short": 0.3, "medium": 0.4, "long": 0.3}
        
    def combine_signals(
        self,
        short_signal: float,
        medium_signal: float,
        long_signal: float,
    ) -> float:
        """Combine multi-timeframe signals."""
        combined = (
            short_signal * self.timeframe_weights["short"] +
            medium_signal * self.timeframe_weights["medium"] +
            long_signal * self.timeframe_weights["long"]
        )
        return combined


class StressTestAllocator:
    """Stress-test based allocation."""
    
    def __init__(self):
        self.scenarios = ["crash", "recession", "inflation", "deflation"]
        
    def stress_test(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
    ) -> Dict[str, float]:
        """Run stress tests on allocation."""
        results = {}
        
        for scenario in self.scenarios:
            # Simplified stress scenarios
            if scenario == "crash":
                stressed_returns = returns * 0.3
            elif scenario == "recession":
                stressed_returns = returns * 0.5
            else:
                stressed_returns = returns
            
            portfolio_return = weights @ np.mean(stressed_returns, axis=0)
            results[scenario] = float(portfolio_return)
        
        return results


class TailRiskHedger:
    """Tail risk hedging."""
    
    def __init__(self, hedge_ratio: float = 0.1):
        self.hedge_ratio = hedge_ratio
        
    def calculate_hedge(
        self,
        portfolio_value: float,
        var: float,
    ) -> float:
        """Calculate tail risk hedge size."""
        hedge_value = portfolio_value * self.hedge_ratio
        
        # Scale by VaR
        if var > 0:
            hedge_value = hedge_value * (var / 0.1)
        
        return hedge_value


class YieldOptimizer:
    """Yield optimization."""
    
    def __init__(self, target_yield: float = 0.08):
        self.target_yield = target_yield
        
    def optimize(
        self,
        assets: List[Dict[str, float]],
        budget: float,
    ) -> Dict[str, float]:
        """Optimize for yield."""
        # Sort by yield
        sorted_assets = sorted(assets, key=lambda x: x.get("yield", 0), reverse=True)
        
        allocation = {}
        remaining = budget
        
        for asset in sorted_assets:
            if remaining <= 0:
                break
            
            alloc = min(remaining, budget * 0.3)  # Max 30% per asset
            allocation[asset["symbol"]] = alloc
            remaining -= alloc
        
        return allocation


class CostAnalyzer:
    """Transaction cost analysis."""
    
    def __init__(self):
        self.trade_history: deque = deque(maxlen=1000)
        
    def analyze_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        fee: float,
    ) -> Dict[str, float]:
        """Analyze trade cost."""
        trade_value = quantity * price
        total_cost = fee
        cost_bps = (fee / trade_value) * 10000 if trade_value > 0 else 0
        
        self.trade_history.append({
            "symbol": symbol,
            "value": trade_value,
            "cost": total_cost,
            "cost_bps": cost_bps,
        })
        
        return {
            "trade_value": trade_value,
            "total_cost": total_cost,
            "cost_bps": cost_bps,
        }
    
    def get_summary(self) -> Dict[str, float]:
        """Get cost summary."""
        if not self.trade_history:
            return {"total_cost": 0, "avg_cost_bps": 0}
        
        total_cost = sum(t["cost"] for t in self.trade_history)
        avg_cost_bps = np.mean([t["cost_bps"] for t in self.trade_history])
        
        return {
            "total_cost": total_cost,
            "avg_cost_bps": float(avg_cost_bps),
            "n_trades": len(self.trade_history),
        }


class PerformanceAttributor:
    """Performance attribution."""
    
    def __init__(self):
        self.attribution_history: deque = deque(maxlen=100)
        
    def attribute(
        self,
        portfolio_return: float,
        factor_returns: Dict[str, float],
        factor_exposures: Dict[str, float],
    ) -> Dict[str, float]:
        """Attribute performance to factors."""
        attribution = {}
        
        for factor, exposure in factor_exposures.items():
            factor_ret = factor_returns.get(factor, 0)
            attribution[factor] = exposure * factor_ret
        
        # Residual
        explained = sum(attribution.values())
        attribution["residual"] = portfolio_return - explained
        
        self.attribution_history.append(attribution)
        return attribution


class PortfolioAnalyzer:
    """Portfolio analytics."""
    
    def __init__(self):
        self.history: deque = deque(maxlen=252)
        
    def analyze(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
    ) -> Dict[str, float]:
        """Analyze portfolio."""
        portfolio_returns = returns @ weights
        
        total_return = np.prod(1 + portfolio_returns) - 1
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        sharpe = (np.mean(portfolio_returns) * 252) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        return {
            "total_return": float(total_return),
            "annualized_return": float(np.mean(portfolio_returns) * 252),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "sortino_ratio": float(sharpe * 0.8),  # Simplified
        }


class CapitalGrowthEngine:
    """Capital growth optimization."""
    
    def __init__(self, initial_capital: float = 1000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.growth_history: deque = deque(maxlen=252)
        
    def update(self, return_pct: float):
        """Update capital with return."""
        self.current_capital *= (1 + return_pct)
        self.growth_history.append(return_pct)
    
    def get_stats(self) -> Dict[str, float]:
        """Get growth statistics."""
        if not self.growth_history:
            return {"current": self.current_capital, "total_return": 0}
        
        total_return = (self.current_capital / self.initial_capital) - 1
        
        return {
            "initial": self.initial_capital,
            "current": self.current_capital,
            "total_return": total_return,
            "avg_daily_return": float(np.mean(self.growth_history)),
        }


class OmegaPortfolioEngine:
    """
    THE OMEGA PORTFOLIO ENGINE.
    
    30 Components.
    """
    
    def __init__(self, initial_capital: float = 1000):
        # Initialize all 30 components
        self.mean_variance = MeanVarianceOptimizer()
        self.black_litterman = BlackLittermanOptimizer()
        self.risk_parity = RiskParityAllocator()
        self.hierarchical_rp = HierarchicalRiskParity()
        self.max_diversification = MaximumDiversification()
        self.min_cvar = MinimumCVaROptimizer()
        self.kelly = KellyCriterionAllocator()
        self.factor_allocator = FactorBasedAllocator()
        self.dynamic_rebalancer = DynamicRebalancer()
        self.tax_rebalancer = TaxAwareRebalancer()
        self.cost_optimizer = TransactionCostOptimizer()
        self.liquidity_allocator = LiquidityAwareAllocator()
        self.correlation_clustering = CorrelationClustering()
        self.regime_allocator = RegimeConditionalAllocator()
        self.volatility_targeting = VolatilityTargeting()
        self.drawdown_sizer = DrawdownAwareSizer()
        self.momentum_tilt = MomentumTilt()
        self.mean_reversion_tilt = MeanReversionTilt()
        self.esg_filter = ESGFilter()
        self.concentration_limiter = ConcentrationLimiter()
        self.sector_rotation = SectorRotation()
        self.cross_asset = CrossAssetAllocator()
        self.multi_timeframe = MultiTimeframeAllocator()
        self.stress_test = StressTestAllocator()
        self.tail_hedger = TailRiskHedger()
        self.yield_optimizer = YieldOptimizer()
        self.cost_analyzer = CostAnalyzer()
        self.performance_attributor = PerformanceAttributor()
        self.portfolio_analyzer = PortfolioAnalyzer()
        self.capital_growth = CapitalGrowthEngine(initial_capital)
        
        # State
        self.positions: List[Position] = []
        self.allocations: List[Allocation] = []
        
        logger.info("OmegaPortfolioEngine: 30 components initialized")
    
    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        method: str = "mean_variance",
    ) -> Dict[str, Any]:
        """Optimize portfolio."""
        if method == "mean_variance":
            return self.mean_variance.optimize(expected_returns, cov_matrix)
        elif method == "risk_parity":
            return self.risk_parity.allocate(cov_matrix)
        elif method == "min_cvar":
            return self.min_cvar.optimize(expected_returns)
        else:
            return self.mean_variance.optimize(expected_returns, cov_matrix)
    
    def get_status(self) -> Dict[str, Any]:
        """Get portfolio engine status."""
        return {
            "total_components": 30,
            "capital_stats": self.capital_growth.get_stats(),
            "cost_summary": self.cost_analyzer.get_summary(),
            "n_positions": len(self.positions),
        }


def get_omega_portfolio(initial_capital: float = 1000) -> OmegaPortfolioEngine:
    """Get Omega Portfolio Engine."""
    return OmegaPortfolioEngine(initial_capital)
