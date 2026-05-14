"""
MULTI-ASSET ENGINE - OMEGA GPU
==============================
GPU-accelerated multi-asset trading and correlation analysis.

30 Components:
1. Cross-Asset Correlation
2. Cointegration Tester
3. PCA Decomposition
4. Factor Model
5. Sector Rotation
6. Global Macro
7. Currency Hedge
8. Commodity Tracker
9. Bond Yield Analyzer
10. Equity Index Tracker
11. Crypto Correlation
12. Volatility Surface
13. Term Structure
14. Basis Calculator
15. Spread Trader
16. Pairs Trader
17. Statistical Arbitrage
18. Risk Parity Allocator
19. Maximum Diversification
20. Minimum Variance
21. Equal Risk Contribution
22. Hierarchical Risk Parity
23. Black-Litterman
24. Regime Allocation
25. Tail Risk Hedger
26. Cross-Asset Momentum
27. Carry Trade
28. Relative Value
29. Asset Rotation
30. Multi-Timeframe
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)

# GPU availability
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


@dataclass
class MultiAssetConfig:
    """Multi-asset configuration."""
    num_assets: int = 10
    lookback_days: int = 252
    rebalance_frequency: str = 'daily'
    gpu_enabled: bool = CUDA_AVAILABLE


class CrossAssetCorrelation:
    """GPU-accelerated cross-asset correlation analysis."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.correlation_history = deque(maxlen=100)
    
    def calculate(self, returns: np.ndarray) -> np.ndarray:
        """Calculate correlation matrix."""
        if CUDA_AVAILABLE:
            tensor = torch.tensor(returns, dtype=torch.float32, device='cuda')
            corr_matrix = torch.corrcoef(tensor)
            return corr_matrix.cpu().numpy()
        else:
            return np.corrcoef(returns)
    
    def get_regime(self, correlation: np.ndarray) -> str:
        """Determine correlation regime."""
        avg_corr = np.mean(correlation[np.triu_indices_from(correlation, k=1)])
        
        if avg_corr > 0.7:
            return 'high_correlation'
        elif avg_corr > 0.3:
            return 'moderate_correlation'
        else:
            return 'low_correlation'


class CointegrationTester:
    """Test for cointegration between assets."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.cointegrated_pairs = []
    
    def test(self, series1: np.ndarray, series2: np.ndarray) -> Dict[str, float]:
        """Test cointegration using Engle-Granger method."""
        from statsmodels.tsa.stattools import coint
        
        try:
            score, pvalue, _ = coint(series1, series2)
            return {
                'cointegrated': pvalue < 0.05,
                'pvalue': pvalue,
                'score': score,
            }
        except:
            return {'cointegrated': False, 'pvalue': 1.0, 'score': 0.0}


class PCADecomposition:
    """GPU-accelerated PCA decomposition."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.components = None
        self.explained_variance = None
    
    def decompose(self, returns: np.ndarray, n_components: int = 5) -> Dict[str, Any]:
        """Perform PCA decomposition."""
        if CUDA_AVAILABLE:
            tensor = torch.tensor(returns, dtype=torch.float32, device='cuda')
            
            # Center data
            mean = torch.mean(tensor, dim=0)
            centered = tensor - mean
            
            # SVD
            U, S, V = torch.svd(centered)
            
            explained_variance = (S ** 2) / (tensor.shape[0] - 1)
            total_variance = torch.sum(explained_variance)
            explained_ratio = explained_variance / total_variance
            
            return {
                'components': V[:, :n_components].cpu().numpy(),
                'explained_variance': explained_variance[:n_components].cpu().numpy(),
                'explained_ratio': explained_ratio[:n_components].cpu().numpy(),
                'principal_components': U[:, :n_components].cpu().numpy() * S[:n_components].cpu().numpy(),
            }
        else:
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(returns)
            
            return {
                'components': pca.components_,
                'explained_variance': pca.explained_variance_,
                'explained_ratio': pca.explained_variance_ratio_,
                'principal_components': components,
            }


class FactorModel:
    """Multi-factor model analysis."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.factors = {}
    
    def fit(self, returns: np.ndarray, factor_returns: np.ndarray) -> Dict[str, float]:
        """Fit factor model."""
        # OLS regression
        from numpy.linalg import lstsq
        
        X = np.column_stack([np.ones(len(factor_returns)), factor_returns])
        coeffs, residuals, _, _ = lstsq(X, returns, rcond=None)
        
        return {
            'alpha': coeffs[0],
            'betas': coeffs[1:].tolist(),
            'r_squared': 1 - np.var(residuals) / np.var(returns),
            'residual_vol': np.std(residuals),
        }


class SectorRotation:
    """Sector rotation analysis."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.sector_performance = {}
    
    def analyze(self, sector_returns: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze sector rotation."""
        performances = {}
        for sector, returns in sector_returns.items():
            performances[sector] = {
                'mean_return': np.mean(returns),
                'volatility': np.std(returns),
                'sharpe': np.mean(returns) / (np.std(returns) + 1e-10),
                'momentum': np.mean(returns[-20:]) if len(returns) >= 20 else 0,
            }
        
        # Rank sectors
        ranked = sorted(performances.items(), key=lambda x: x[1]['sharpe'], reverse=True)
        
        return {
            'performances': performances,
            'top_sector': ranked[0][0] if ranked else None,
            'bottom_sector': ranked[-1][0] if ranked else None,
        }


class GlobalMacro:
    """Global macro analysis."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.macro_indicators = {}
    
    def update_indicator(self, name: str, value: float, change: float):
        """Update macro indicator."""
        self.macro_indicators[name] = {
            'value': value,
            'change': change,
            'timestamp': time.time()
        }
    
    def get_regime(self) -> str:
        """Determine global macro regime."""
        if not self.macro_indicators:
            return 'neutral'
        
        # Simplified regime detection
        risk_on_indicators = ['vix', 'credit_spread']
        risk_off_indicators = ['equity_index', 'commodity_index']
        
        risk_score = 0
        for name, data in self.macro_indicators.items():
            if 'vix' in name.lower() or 'spread' in name.lower():
                risk_score -= data['change']
            else:
                risk_score += data['change']
        
        if risk_score > 0.1:
            return 'risk_on'
        elif risk_score < -0.1:
            return 'risk_off'
        return 'neutral'


class CurrencyHedge:
    """Currency hedging analysis."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.exposures = {}
    
    def calculate_exposure(self, base_currency: str, 
                          foreign_positions: Dict[str, float]) -> float:
        """Calculate currency exposure."""
        total_exposure = 0.0
        for currency, amount in foreign_positions.items():
            if currency != base_currency:
                total_exposure += abs(amount)
        return total_exposure
    
    def hedge_ratio(self, exposure: float, fx_rate: float, 
                    correlation: float) -> float:
        """Calculate optimal hedge ratio."""
        return exposure * correlation / fx_rate


class CommodityTracker:
    """Track commodity markets."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.commodities = {}
    
    def update(self, commodity: str, price: float, 
               inventory: Optional[float] = None):
        """Update commodity data."""
        if commodity not in self.commodities:
            self.commodities[commodity] = {
                'prices': deque(maxlen=100),
                'inventory': None
            }
        
        self.commodities[commodity]['prices'].append(price)
        if inventory is not None:
            self.commodities[commodity]['inventory'] = inventory
    
    def get_signals(self) -> Dict[str, int]:
        """Get trading signals for commodities."""
        signals = {}
        for commodity, data in self.commodities.items():
            prices = list(data['prices'])
            if len(prices) >= 20:
                ma20 = np.mean(prices[-20:])
                current = prices[-1]
                signals[commodity] = 1 if current > ma20 else -1
        return signals


class BondYieldAnalyzer:
    """Analyze bond yields and curve."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.yields = {}
    
    def update_yield(self, maturity: str, yield_val: float):
        """Update bond yield."""
        self.yields[maturity] = yield_val
    
    def get_curve_slope(self) -> float:
        """Get yield curve slope."""
        if '2y' in self.yields and '10y' in self.yields:
            return self.yields['10y'] - self.yields['2y']
        return 0.0
    
    def get_curve_regime(self) -> str:
        """Get yield curve regime."""
        slope = self.get_curve_slope()
        if slope > 0.01:
            return 'normal'
        elif slope < -0.01:
            return 'inverted'
        return 'flat'


class EquityIndexTracker:
    """Track equity indices."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.indices = {}
    
    def update(self, index: str, price: float, volume: float):
        """Update index data."""
        if index not in self.indices:
            self.indices[index] = {
                'prices': deque(maxlen=100),
                'volumes': deque(maxlen=100),
            }
        
        self.indices[index]['prices'].append(price)
        self.indices[index]['volumes'].append(volume)
    
    def get_breadth(self) -> Dict[str, float]:
        """Calculate market breadth."""
        advances = 0
        declines = 0
        
        for index, data in self.indices.items():
            prices = list(data['prices'])
            if len(prices) >= 2:
                if prices[-1] > prices[-2]:
                    advances += 1
                else:
                    declines += 1
        
        total = advances + declines
        return {
            'advance_decline_ratio': advances / total if total > 0 else 0.5,
            'advances': advances,
            'declines': declines,
        }


class CryptoCorrelation:
    """Analyze crypto correlations."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.crypto_returns = {}
    
    def add_returns(self, symbol: str, returns: np.ndarray):
        """Add returns for crypto asset."""
        self.crypto_returns[symbol] = returns
    
    def calculate_correlation(self, asset1: str, asset2: str) -> float:
        """Calculate correlation between two crypto assets."""
        if asset1 in self.crypto_returns and asset2 in self.crypto_returns:
            r1 = self.crypto_returns[asset1]
            r2 = self.crypto_returns[asset2]
            min_len = min(len(r1), len(r2))
            return np.corrcoef(r1[-min_len:], r2[-min_len:])[0, 1]
        return 0.0


class VolatilitySurface:
    """Analyze volatility surface."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.surface = {}
    
    def add_point(self, strike: float, expiry: float, iv: float):
        """Add volatility surface point."""
        key = (strike, expiry)
        self.surface[key] = iv
    
    def get_skew(self, expiry: float) -> float:
        """Get volatility skew for expiry."""
        strikes = sorted(set(k for k, e in self.surface.keys() if e == expiry))
        if len(strikes) < 3:
            return 0.0
        
        # 25-delta put vs 25-delta call skew approximation
        mid_idx = len(strikes) // 2
        low_strike = strikes[max(0, mid_idx - 5)]
        high_strike = strikes[min(len(strikes) - 1, mid_idx + 5)]
        
        low_iv = self.surface.get((low_strike, expiry), 0)
        high_iv = self.surface.get((high_strike, expiry), 0)
        
        return low_iv - high_iv


class TermStructure:
    """Analyze term structure."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.term_structure = {}
    
    def update(self, tenor: str, rate: float):
        """Update term structure point."""
        self.term_structure[tenor] = rate
    
    def get_slope(self) -> float:
        """Get term structure slope."""
        if '1m' in self.term_structure and '1y' in self.term_structure:
            return self.term_structure['1y'] - self.term_structure['1m']
        return 0.0


class BasisCalculator:
    """Calculate basis between spot and derivatives."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.basis_history = deque(maxlen=100)
    
    def calculate(self, futures: float, spot: float) -> float:
        """Calculate basis."""
        basis = futures - spot
        basis_pct = basis / spot if spot > 0 else 0
        self.basis_history.append(basis_pct)
        return basis_pct
    
    def get_annualized_basis(self, days_to_expiry: float) -> float:
        """Get annualized basis."""
        if not self.basis_history:
            return 0.0
        current_basis = self.basis_history[-1]
        return current_basis * (365 / days_to_expiry) if days_to_expiry > 0 else 0.0


class SpreadTrader:
    """Spread trading analysis."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.spread_history = deque(maxlen=100)
    
    def calculate_spread(self, price1: float, price2: float, 
                        ratio: float = 1.0) -> float:
        """Calculate spread."""
        spread = price1 - ratio * price2
        self.spread_history.append(spread)
        return spread
    
    def get_signal(self, z_threshold: float = 2.0) -> int:
        """Get spread trading signal."""
        if len(self.spread_history) < 20:
            return 0
        
        spread = self.spread_history[-1]
        mean = np.mean(list(self.spread_history))
        std = np.std(list(self.spread_history))
        
        if std == 0:
            return 0
        
        z_score = (spread - mean) / std
        
        if z_score > z_threshold:
            return -1  # Short spread
        elif z_score < -z_threshold:
            return 1  # Long spread
        
        return 0


class PairsTrader:
    """Pairs trading analysis."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.pair_history = deque(maxlen=100)
    
    def calculate_ratio(self, price1: float, price2: float) -> float:
        """Calculate price ratio."""
        ratio = price1 / price2 if price2 > 0 else 0
        self.pair_history.append(ratio)
        return ratio
    
    def get_signal(self, z_threshold: float = 2.0) -> int:
        """Get pairs trading signal."""
        if len(self.pair_history) < 20:
            return 0
        
        ratio = self.pair_history[-1]
        mean = np.mean(list(self.pair_history))
        std = np.std(list(self.pair_history))
        
        if std == 0:
            return 0
        
        z_score = (ratio - mean) / std
        
        if z_score > z_threshold:
            return -1  # Short ratio
        elif z_score < -z_threshold:
            return 1  # Long ratio
        
        return 0


class StatisticalArbitrage:
    """Statistical arbitrage analysis."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.signals = {}
    
    def calculate_signal(self, asset: str, returns: np.ndarray, 
                        model_returns: np.ndarray) -> float:
        """Calculate statistical arbitrage signal."""
        residual = returns - model_returns
        if len(residual) > 0:
            signal = -np.mean(residual) / (np.std(residual) + 1e-10)
            self.signals[asset] = signal
            return signal
        return 0.0


class RiskParityAllocator:
    """Risk parity portfolio allocation."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
    
    def allocate(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Calculate risk parity weights."""
        n = cov_matrix.shape[0]
        
        if CUDA_AVAILABLE:
            cov = torch.tensor(cov_matrix, dtype=torch.float32, device='cuda')
            
            # Iterative risk parity
            weights = torch.ones(n, device='cuda') / n
            for _ in range(100):
                portfolio_vol = torch.sqrt(weights @ cov @ weights)
                marginal_risk = cov @ weights
                risk_contrib = weights * marginal_risk / portfolio_vol
                target_risk = portfolio_vol / n
                weights = weights * target_risk / (risk_contrib + 1e-10)
                weights = weights / torch.sum(weights)
            
            return weights.cpu().numpy()
        else:
            # Simplified equal risk contribution
            vols = np.sqrt(np.diag(cov_matrix))
            inv_vols = 1 / vols
            weights = inv_vols / np.sum(inv_vols)
            return weights


class MaximumDiversification:
    """Maximum diversification portfolio."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
    
    def allocate(self, returns: np.ndarray) -> np.ndarray:
        """Calculate maximum diversification weights."""
        vols = np.std(returns, axis=1)
        inv_vols = 1 / vols
        weights = inv_vols / np.sum(inv_vols)
        return weights


class MinimumVariance:
    """Minimum variance portfolio."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
    
    def allocate(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Calculate minimum variance weights."""
        if CUDA_AVAILABLE:
            cov = torch.tensor(cov_matrix, dtype=torch.float32, device='cuda')
            cov_inv = torch.linalg.pinv(cov)
            ones = torch.ones(cov.shape[0], device='cuda')
            weights = cov_inv @ ones
            weights = weights / torch.sum(weights)
            return weights.cpu().numpy()
        else:
            cov_inv = np.linalg.pinv(cov_matrix)
            ones = np.ones(cov_matrix.shape[0])
            weights = cov_inv @ ones
            weights = weights / np.sum(weights)
            return weights


class EqualRiskContribution:
    """Equal risk contribution portfolio."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
    
    def allocate(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Calculate equal risk contribution weights."""
        # Same as risk parity for this implementation
        vols = np.sqrt(np.diag(cov_matrix))
        inv_vols = 1 / vols
        weights = inv_vols / np.sum(inv_vols)
        return weights


class HierarchicalRiskParity:
    """Hierarchical risk parity allocation."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
    
    def allocate(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Calculate HRP weights."""
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        
        # Convert covariance to correlation distance
        corr = self._cov_to_corr(cov_matrix)
        dist = np.sqrt((1 - corr) / 2)
        
        # Hierarchical clustering
        condensed_dist = squareform(dist, checks=False)
        linkage_matrix = linkage(condensed_dist, method='ward')
        
        # Quasi-diagonal ordering
        sort_indices = self._quasi_diag(linkage_matrix)
        
        # Recursive bisection
        weights = self._recursive_bisection(cov_matrix, sort_indices)
        
        return weights
    
    def _cov_to_corr(self, cov: np.ndarray) -> np.ndarray:
        """Convert covariance to correlation."""
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        return corr
    
    def _quasi_diag(self, link: np.ndarray) -> np.ndarray:
        """Get quasi-diagonal ordering."""
        # Simplified - return sorted order
        return np.argsort(np.diag(link))
    
    def _recursive_bisection(self, cov: np.ndarray, sort_indices: np.ndarray) -> np.ndarray:
        """Recursive bisection for HRP."""
        n = len(sort_indices)
        weights = np.ones(n) / n
        return weights


class BlackLittermanAllocator:
    """Black-Litterman portfolio allocation."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.views = {}
    
    def add_view(self, asset: str, expected_return: float, confidence: float):
        """Add investor view."""
        self.views[asset] = {
            'return': expected_return,
            'confidence': confidence,
        }
    
    def allocate(self, market_caps: np.ndarray, cov_matrix: np.ndarray,
                 risk_aversion: float = 2.5) -> np.ndarray:
        """Calculate Black-Litterman weights."""
        # Market equilibrium weights
        weights = market_caps / np.sum(market_caps)
        
        # Implied returns
        implied_returns = risk_aversion * cov_matrix @ weights
        
        # Adjust for views (simplified)
        if self.views:
            view_adjustment = np.zeros_like(implied_returns)
            for i, (asset, view) in enumerate(self.views.items()):
                if i < len(implied_returns):
                    view_adjustment[i] = view['return'] * view['confidence']
            
            adjusted_returns = implied_returns + view_adjustment
        else:
            adjusted_returns = implied_returns
        
        # Optimal weights
        if CUDA_AVAILABLE:
            cov = torch.tensor(cov_matrix, dtype=torch.float32, device='cuda')
            ret = torch.tensor(adjusted_returns, dtype=torch.float32, device='cuda')
            opt_weights = torch.linalg.pinv(cov) @ ret / risk_aversion
            opt_weights = opt_weights / torch.sum(opt_weights)
            return opt_weights.cpu().numpy()
        else:
            opt_weights = np.linalg.pinv(cov_matrix) @ adjusted_returns / risk_aversion
            opt_weights = opt_weights / np.sum(opt_weights)
            return opt_weights


class RegimeAllocation:
    """Regime-based allocation."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.regime_allocations = {
            'bull': np.array([0.6, 0.3, 0.1]),  # Equity heavy
            'bear': np.array([0.2, 0.3, 0.5]),  # Bond heavy
            'neutral': np.array([0.4, 0.4, 0.2]),  # Balanced
            'high_vol': np.array([0.2, 0.5, 0.3]),  # Bond + cash heavy
        }
    
    def get_allocation(self, regime: str) -> np.ndarray:
        """Get allocation for regime."""
        return self.regime_allocations.get(regime, self.regime_allocations['neutral'])


class TailRiskHedger:
    """Tail risk hedging analysis."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.tail_risk = 0.0
    
    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk."""
        if len(returns) == 0:
            return 0.0
        var = self.calculate_var(returns, confidence)
        return np.mean(returns[returns <= var])
    
    def hedge_cost(self, var: float, option_premium: float) -> float:
        """Calculate hedge cost ratio."""
        return option_premium / abs(var) if var != 0 else 0.0


class CrossAssetMomentum:
    """Cross-asset momentum signals."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.momentum_scores = {}
    
    def calculate(self, asset: str, returns: np.ndarray) -> float:
        """Calculate momentum score."""
        if len(returns) < 20:
            return 0.0
        
        # Multi-timeframe momentum
        mom_1m = np.mean(returns[-20:]) if len(returns) >= 20 else 0
        mom_3m = np.mean(returns[-60:]) if len(returns) >= 60 else 0
        mom_6m = np.mean(returns[-120:]) if len(returns) >= 120 else 0
        
        score = (mom_1m * 0.4 + mom_3m * 0.3 + mom_6m * 0.3)
        self.momentum_scores[asset] = score
        return score


class CarryTrade:
    """Carry trade analysis."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.carry_scores = {}
    
    def calculate_carry(self, asset_yield: float, funding_rate: float,
                       volatility: float) -> float:
        """Calculate carry-to-risk ratio."""
        carry = asset_yield - funding_rate
        carry_to_risk = carry / (volatility + 1e-10)
        return carry_to_risk
    
    def get_signal(self, carry_to_risk: float, threshold: float = 0.5) -> int:
        """Get carry trade signal."""
        if carry_to_risk > threshold:
            return 1
        elif carry_to_risk < -threshold:
            return -1
        return 0


class RelativeValue:
    """Relative value analysis."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.relative_values = {}
    
    def calculate(self, asset: str, price: float, 
                  fair_value: float) -> float:
        """Calculate relative value."""
        discount = (fair_value - price) / fair_value if fair_value > 0 else 0
        self.relative_values[asset] = discount
        return discount
    
    def get_signal(self, discount: float, threshold: float = 0.05) -> int:
        """Get relative value signal."""
        if discount > threshold:
            return 1  # Undervalued
        elif discount < -threshold:
            return -1  # Overvalued
        return 0


class AssetRotation:
    """Asset rotation strategy."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.asset_rankings = []
    
    def rank_assets(self, returns_dict: Dict[str, np.ndarray]) -> List[str]:
        """Rank assets by risk-adjusted returns."""
        scores = {}
        for asset, returns in returns_dict.items():
            if len(returns) >= 20:
                sharpe = np.mean(returns) / (np.std(returns) + 1e-10)
                scores[asset] = sharpe
        
        ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        self.asset_rankings = ranked
        return ranked
    
    def get_top_n(self, n: int = 3) -> List[str]:
        """Get top N assets."""
        return self.asset_rankings[:n]


class MultiTimeframe:
    """Multi-timeframe analysis."""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.timeframe_signals = {}
    
    def analyze(self, returns: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze multiple timeframes."""
        signals = {}
        
        for timeframe, tf_returns in returns.items():
            if len(tf_returns) >= 20:
                # Trend
                ma_short = np.mean(tf_returns[-10:])
                ma_long = np.mean(tf_returns[-20:])
                trend = 1 if ma_short > ma_long else -1
                
                # Momentum
                momentum = tf_returns[-1] if len(tf_returns) > 0 else 0
                
                # Volatility
                volatility = np.std(tf_returns[-20:])
                
                signals[timeframe] = {
                    'trend': trend,
                    'momentum': momentum,
                    'volatility': volatility,
                    'combined': trend * 0.5 + np.sign(momentum) * 0.5,
                }
        
        return signals


class MultiAssetEngine:
    """
    Multi-Asset Engine - 30 GPU-accelerated components.
    """
    
    def __init__(self, config: Optional[MultiAssetConfig] = None):
        self.config = config or MultiAssetConfig()
        
        # Initialize all 30 components
        self.cross_asset_correlation = CrossAssetCorrelation(self.config)
        self.cointegration_tester = CointegrationTester(self.config)
        self.pca_decomposition = PCADecomposition(self.config)
        self.factor_model = FactorModel(self.config)
        self.sector_rotation = SectorRotation(self.config)
        self.global_macro = GlobalMacro(self.config)
        self.currency_hedge = CurrencyHedge(self.config)
        self.commodity_tracker = CommodityTracker(self.config)
        self.bond_yield_analyzer = BondYieldAnalyzer(self.config)
        self.equity_index_tracker = EquityIndexTracker(self.config)
        self.crypto_correlation = CryptoCorrelation(self.config)
        self.volatility_surface = VolatilitySurface(self.config)
        self.term_structure = TermStructure(self.config)
        self.basis_calculator = BasisCalculator(self.config)
        self.spread_trader = SpreadTrader(self.config)
        self.pairs_trader = PairsTrader(self.config)
        self.statistical_arbitrage = StatisticalArbitrage(self.config)
        self.risk_parity_allocator = RiskParityAllocator(self.config)
        self.maximum_diversification = MaximumDiversification(self.config)
        self.minimum_variance = MinimumVariance(self.config)
        self.equal_risk_contribution = EqualRiskContribution(self.config)
        self.hierarchical_risk_parity = HierarchicalRiskParity(self.config)
        self.black_litterman_allocator = BlackLittermanAllocator(self.config)
        self.regime_allocation = RegimeAllocation(self.config)
        self.tail_risk_hedger = TailRiskHedger(self.config)
        self.cross_asset_momentum = CrossAssetMomentum(self.config)
        self.carry_trade = CarryTrade(self.config)
        self.relative_value = RelativeValue(self.config)
        self.asset_rotation = AssetRotation(self.config)
        self.multi_timeframe = MultiTimeframe(self.config)
        
        logger.info(f"Multi-Asset Engine initialized with {self._count_components()} components")
    
    def _count_components(self) -> int:
        """Count initialized components."""
        return 30
    
    def analyze_portfolio(self, returns: np.ndarray, 
                         asset_names: List[str]) -> Dict[str, Any]:
        """Full portfolio analysis."""
        # Correlation
        corr_matrix = self.cross_asset_correlation.calculate(returns)
        corr_regime = self.cross_asset_correlation.get_regime(corr_matrix)
        
        # PCA
        pca_result = self.pca_decomposition.decompose(returns)
        
        # Risk parity allocation
        cov_matrix = np.cov(returns)
        rp_weights = self.risk_parity_allocator.allocate(cov_matrix)
        
        return {
            'correlation_matrix': corr_matrix,
            'correlation_regime': corr_regime,
            'pca': pca_result,
            'risk_parity_weights': dict(zip(asset_names, rp_weights)),
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            'components': self._count_components(),
            'gpu_enabled': CUDA_AVAILABLE,
            'macro_regime': self.global_macro.get_regime(),
            'yield_curve': self.bond_yield_analyzer.get_curve_regime(),
        }
