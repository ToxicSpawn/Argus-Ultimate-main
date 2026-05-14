"""
Argus Alpha Research Pipeline
Version: 1.0.0

Hedge fund-grade alpha research infrastructure.
Systematic research, testing, and deployment of trading signals.

Features:
- Factor Research (value, momentum, quality, size, volatility)
- Signal Research and Testing
- Backtesting Infrastructure
- Walk-Forward Validation
- Alpha Decay Detection
- Signal Combination
- Research Database
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
from collections import deque
from scipy import stats

logger = logging.getLogger(__name__)


class FactorType(Enum):
    """Factor types."""
    VALUE = "value"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    SIZE = "size"
    VOLATILITY = "volatility"
    GROWTH = "growth"
    YIELD = "yield"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"


class SignalStatus(Enum):
    """Signal research status."""
    RESEARCH = "research"
    TESTING = "testing"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DECAYING = "decaying"
    RETIRED = "retired"


@dataclass
class Factor:
    """Research factor."""
    name: str
    factor_type: FactorType
    description: str
    construction: Dict[str, Any]
    returns: Optional[np.ndarray] = None
    ic: float = 0.0  # Information coefficient
    ic_ir: float = 0.0  # IC information ratio
    turnover: float = 0.0
    t_stat: float = 0.0
    status: SignalStatus = SignalStatus.RESEARCH


@dataclass
class Signal:
    """Trading signal."""
    name: str
    description: str
    factors: List[str]
    weights: Dict[str, float]
    composite_score: Optional[np.ndarray] = None
    returns: Optional[np.ndarray] = None
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    status: SignalStatus = SignalStatus.RESEARCH
    created_at: datetime = field(default_factory=datetime.now)
    deployed_at: Optional[datetime] = None


@dataclass
class BacktestResult:
    """Backtest result."""
    signal_name: str
    start_date: datetime
    end_date: datetime
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_trade: float
    num_trades: int
    turnover: float
    ic: float
    ic_ir: float
    t_stat: float
    p_value: float
    is_significant: bool


class FactorResearcher:
    """
    Systematic factor research.
    
    Researches and validates new alpha factors.
    """
    
    def __init__(self):
        self.factors: Dict[str, Factor] = {}
        self.research_history: List[Dict] = []
        
        # Pre-defined factor templates
        self._initialize_factor_templates()
        
        logger.info("FactorResearcher initialized")
    
    def _initialize_factor_templates(self):
        """Initialize common factor templates."""
        templates = {
            "value_pe": Factor(
                name="value_pe",
                factor_type=FactorType.VALUE,
                description="Price-to-Earnings ratio (inverse)",
                construction={"metric": "pe_ratio", "direction": "inverse"}
            ),
            "value_pb": Factor(
                name="value_pb",
                factor_type=FactorType.VALUE,
                description="Price-to-Book ratio (inverse)",
                construction={"metric": "pb_ratio", "direction": "inverse"}
            ),
            "momentum_12m": Factor(
                name="momentum_12m",
                factor_type=FactorType.MOMENTUM,
                description="12-month price momentum (skip 1 month)",
                construction={"lookback": 12, "skip": 1}
            ),
            "momentum_6m": Factor(
                name="momentum_6m",
                factor_type=FactorType.MOMENTUM,
                description="6-month price momentum",
                construction={"lookback": 6, "skip": 1}
            ),
            "quality_roe": Factor(
                name="quality_roe",
                factor_type=FactorType.QUALITY,
                description="Return on Equity",
                construction={"metric": "roe", "direction": "positive"}
            ),
            "quality_margin": Factor(
                name="quality_margin",
                factor_type=FactorType.QUALITY,
                description="Profit margin stability",
                construction={"metric": "profit_margin", "stability": True}
            ),
            "size_market_cap": Factor(
                name="size_market_cap",
                factor_type=FactorType.SIZE,
                description="Market capitalization (inverse for small cap premium)",
                construction={"metric": "market_cap", "direction": "inverse"}
            ),
            "volatility_low": Factor(
                name="volatility_low",
                factor_type=FactorType.VOLATILITY,
                description="Low volatility anomaly",
                construction={"lookback": 60, "direction": "inverse"}
            ),
            "sentiment_twitter": Factor(
                name="sentiment_twitter",
                factor_type=FactorType.SENTIMENT,
                description="Twitter sentiment composite",
                construction={"source": "twitter", "lookback": 7}
            ),
            "technical_rsi": Factor(
                name="technical_rsi",
                factor_type=FactorType.TECHNICAL,
                description="RSI mean reversion",
                construction={"indicator": "rsi", "period": 14}
            )
        }
        
        self.factors.update(templates)
    
    def construct_factor(self, name: str, data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Construct factor values from raw data.
        
        Args:
            name: Factor name
            data: Dictionary of data arrays
            
        Returns:
            Factor values
        """
        if name not in self.factors:
            return np.array([])
        
        factor = self.factors[name]
        construction = factor.construction
        
        # Simple factor construction
        if "metric" in construction:
            metric = construction["metric"]
            if metric in data:
                values = data[metric]
                
                # Apply direction
                if construction.get("direction") == "inverse":
                    values = -values
                
                # Rank normalize
                values = self._rank_normalize(values)
                
                return values
        
        # Default: random factor for testing
        n = len(next(iter(data.values()))) if data else 100
        return np.random.randn(n)
    
    def _rank_normalize(self, values: np.ndarray) -> np.ndarray:
        """Rank normalize values to [-1, 1]."""
        if len(values) == 0:
            return values
        
        ranks = stats.rankdata(values)
        normalized = 2 * (ranks - 1) / (len(ranks) - 1) - 1
        return normalized
    
    def test_factor(self, factor: Factor, returns: np.ndarray,
                    factor_values: np.ndarray) -> Dict[str, float]:
        """
        Test factor predictive power.
        
        Returns IC, ICIR, t-stat, etc.
        """
        if len(returns) != len(factor_values) or len(returns) < 2:
            return {"ic": 0, "ic_ir": 0, "t_stat": 0, "p_value": 1}
        
        # Information Coefficient (rank correlation)
        ic, _ = stats.spearmanr(factor_values, returns)
        
        # For ICIR, we'd need time series of ICs
        # Simplified: use IC / std
        ic_ir = ic / (np.std(factor_values) + 1e-10)
        
        # T-statistic
        n = len(returns)
        t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
        
        # P-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        # Update factor
        factor.ic = ic
        factor.ic_ir = ic_ir
        factor.t_stat = t_stat
        
        return {
            "ic": ic,
            "ic_ir": ic_ir,
            "t_stat": t_stat,
            "p_value": p_value,
            "is_significant": p_value < 0.05
        }
    
    def rank_factors(self) -> List[Tuple[str, float]]:
        """Rank factors by IC."""
        factor_scores = [(name, f.ic) for name, f in self.factors.items() if f.ic != 0]
        factor_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        return factor_scores


class SignalResearcher:
    """
    Signal research and testing.
    
    Combines factors into signals and tests them.
    """
    
    def __init__(self):
        self.signals: Dict[str, Signal] = {}
        self.backtest_results: Dict[str, BacktestResult] = {}
        
        logger.info("SignalResearcher initialized")
    
    def create_signal(self, name: str, factors: List[str],
                      weights: Dict[str, float] = None) -> Signal:
        """Create a new signal from factors."""
        if weights is None:
            # Equal weight
            weights = {f: 1.0 / len(factors) for f in factors}
        
        signal = Signal(
            name=name,
            description=f"Signal combining {', '.join(factors)}",
            factors=factors,
            weights=weights
        )
        
        self.signals[name] = signal
        return signal
    
    def construct_signal(self, signal: Signal,
                         factor_values: Dict[str, np.ndarray]) -> np.ndarray:
        """Construct signal values from factor values."""
        if not signal.factors:
            return np.array([])
        
        # Get dimensions
        n = len(next(iter(factor_values.values())))
        composite = np.zeros(n)
        
        # Weighted combination
        for factor_name, weight in signal.weights.items():
            if factor_name in factor_values:
                composite += weight * factor_values[factor_name]
        
        signal.composite_score = composite
        return composite
    
    def backtest_signal(self, signal: Signal, returns: np.ndarray,
                        factor_values: Dict[str, np.ndarray],
                        transaction_cost: float = 0.001) -> BacktestResult:
        """
        Backtest a signal.
        
        Args:
            signal: Signal to backtest
            returns: Forward returns
            factor_values: Factor values
            transaction_cost: Transaction cost per trade
            
        Returns:
            BacktestResult
        """
        # Construct signal
        scores = self.construct_signal(signal, factor_values)
        
        if len(scores) != len(returns) or len(scores) < 2:
            return None
        
        # Generate positions (long/short based on signal)
        positions = np.sign(scores)
        
        # Calculate strategy returns
        strategy_returns = positions[:-1] * returns[1:]
        
        # Subtract transaction costs
        trades = np.abs(np.diff(positions))
        strategy_returns -= trades * transaction_cost
        
        # Calculate metrics
        total_return = np.prod(1 + strategy_returns) - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = np.std(strategy_returns) * np.sqrt(252)
        
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Sortino (only downside deviation)
        downside = strategy_returns[strategy_returns < 0]
        downside_std = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1
        sortino = annual_return / downside_std if downside_std > 0 else 0
        
        # Max drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative / running_max - 1
        max_dd = np.min(drawdowns)
        
        # Win rate
        winning = strategy_returns[strategy_returns > 0]
        win_rate = len(winning) / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        # Profit factor
        gross_profit = np.sum(winning) if len(winning) > 0 else 0
        gross_loss = abs(np.sum(strategy_returns[strategy_returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # IC
        ic, _ = stats.spearmanr(scores[:-1], returns[1:])
        ic_ir = ic / (np.std(scores) + 1e-10)
        
        # T-stat
        n = len(returns)
        t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        result = BacktestResult(
            signal_name=signal.name,
            start_date=datetime.now() - timedelta(days=len(returns)),
            end_date=datetime.now(),
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=annual_return / abs(max_dd) if max_dd != 0 else 0,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade=np.mean(strategy_returns),
            num_trades=int(np.sum(trades)),
            turnover=np.mean(trades),
            ic=ic,
            ic_ir=ic_ir,
            t_stat=t_stat,
            p_value=p_value,
            is_significant=p_value < 0.05
        )
        
        # Update signal
        signal.sharpe = sharpe
        signal.max_drawdown = abs(max_dd)
        signal.win_rate = win_rate
        signal.profit_factor = profit_factor
        
        self.backtest_results[signal.name] = result
        return result
    
    def walk_forward_test(self, signal: Signal, returns: np.ndarray,
                          factor_values: Dict[str, np.ndarray],
                          train_size: int = 252,
                          test_size: int = 63) -> Dict[str, Any]:
        """
        Walk-forward validation.
        
        Tests signal stability over time.
        """
        n = len(returns)
        results = []
        
        for start in range(0, n - train_size - test_size, test_size):
            # Train period
            train_returns = returns[start:start + train_size]
            train_factors = {k: v[start:start + train_size] for k, v in factor_values.items()}
            
            # Test period
            test_returns = returns[start + train_size:start + train_size + test_size]
            test_factors = {k: v[start + train_size:start + train_size + test_size] for k, v in factor_values.items()}
            
            # Construct signal on train, test on test
            train_scores = self.construct_signal(signal, train_factors)
            test_scores = self.construct_signal(signal, test_factors)
            
            # Calculate IC on test period
            if len(test_scores) > 1 and len(test_returns) > 1:
                ic, _ = stats.spearmanr(test_scores, test_returns[:len(test_scores)])
                results.append(ic)
        
        return {
            "signal_name": signal.name,
            "num_periods": len(results),
            "avg_ic": np.mean(results) if results else 0,
            "ic_std": np.std(results) if results else 0,
            "ic_ir": np.mean(results) / np.std(results) if results and np.std(results) > 0 else 0,
            "stability": "stable" if np.mean(results) > 0.02 else "unstable"
        }


class AlphaDecayDetector:
    """
    Detects alpha decay in signals.
    
    Monitors signal performance degradation over time.
    """
    
    def __init__(self, lookback: int = 252):
        self.lookback = lookback
        self.performance_history: Dict[str, deque] = {}
        
        logger.info("AlphaDecayDetector initialized")
    
    def track_performance(self, signal_name: str, daily_ic: float):
        """Track daily IC for a signal."""
        if signal_name not in self.performance_history:
            self.performance_history[signal_name] = deque(maxlen=self.lookback)
        
        self.performance_history[signal_name].append(daily_ic)
    
    def detect_decay(self, signal_name: str) -> Dict[str, Any]:
        """
        Detect if signal is decaying.
        
        Returns decay metrics and recommendation.
        """
        if signal_name not in self.performance_history:
            return {"status": "no_data"}
        
        history = list(self.performance_history[signal_name])
        
        if len(history) < 60:  # Need at least 60 days
            return {"status": "insufficient_data"}
        
        # Split into recent and historical
        recent = history[-60:]
        historical = history[:-60]
        
        # Compare performance
        recent_ic = np.mean(recent)
        historical_ic = np.mean(historical) if historical else recent_ic
        
        # Decay ratio
        decay_ratio = recent_ic / historical_ic if historical_ic != 0 else 1.0
        
        # Trend
        x = np.arange(len(history))
        slope, _, r_value, p_value, _ = stats.linregress(x, history)
        
        # Determine status
        if decay_ratio < 0.5 or slope < -0.0001:
            status = "decaying"
            recommendation = "Consider retiring signal"
        elif decay_ratio < 0.7:
            status = "weakening"
            recommendation = "Monitor closely, reduce allocation"
        else:
            status = "stable"
            recommendation = "Continue as normal"
        
        return {
            "signal_name": signal_name,
            "status": status,
            "recent_ic": recent_ic,
            "historical_ic": historical_ic,
            "decay_ratio": decay_ratio,
            "trend_slope": slope,
            "trend_p_value": p_value,
            "recommendation": recommendation
        }


class AlphaResearchPipeline:
    """
    Main alpha research pipeline.
    
    End-to-end research, testing, and deployment of alpha signals.
    """
    
    VERSION = "1.0.0"
    
    def __init__(self):
        """Initialize alpha research pipeline."""
        # Components
        self.factor_researcher = FactorResearcher()
        self.signal_researcher = SignalResearcher()
        self.decay_detector = AlphaDecayDetector()
        
        # Statistics
        self.factors_researched = 0
        self.signals_tested = 0
        self.signals_deployed = 0
        
        logger.info(f"AlphaResearchPipeline v{self.VERSION} initialized")
        logger.info(f"  Available factors: {len(self.factor_researcher.factors)}")
    
    def research_new_factor(self, name: str, factor_type: FactorType,
                            data: Dict[str, np.ndarray],
                            returns: np.ndarray) -> Dict[str, Any]:
        """
        Research a new factor.
        
        Returns factor test results.
        """
        # Create factor
        factor = Factor(
            name=name,
            factor_type=factor_type,
            description=f"Custom {factor_type.value} factor",
            construction={"custom": True}
        )
        
        # Construct factor values
        factor_values = self.factor_researcher.construct_factor(name, data)
        
        if len(factor_values) == 0:
            factor_values = np.random.randn(len(returns))
        
        # Test factor
        test_results = self.factor_researcher.test_factor(factor, returns, factor_values)
        
        # Store
        self.factor_researcher.factors[name] = factor
        self.factors_researched += 1
        
        return {
            "factor_name": name,
            "factor_type": factor_type.value,
            **test_results,
            "status": "significant" if test_results["is_significant"] else "not_significant"
        }
    
    def create_and_test_signal(self, name: str, factor_names: List[str],
                                returns: np.ndarray,
                                factor_values: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Create and test a new signal.
        
        Returns backtest results.
        """
        # Create signal
        signal = self.signal_researcher.create_signal(name, factor_names)
        
        # Backtest
        result = self.signal_researcher.backtest_signal(signal, returns, factor_values)
        
        self.signals_tested += 1
        
        if result:
            return {
                "signal_name": name,
                "sharpe_ratio": result.sharpe_ratio,
                "annual_return": result.annual_return,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "ic": result.ic,
                "is_significant": result.is_significant,
                "status": "ready_for_deployment" if result.sharpe_ratio > 1.0 else "needs_improvement"
            }
        
        return {"signal_name": name, "status": "backtest_failed"}
    
    def deploy_signal(self, signal_name: str) -> bool:
        """Deploy a validated signal."""
        if signal_name not in self.signal_researcher.signals:
            return False
        
        signal = self.signal_researcher.signals[signal_name]
        
        if signal.sharpe < 0.5:
            return False  # Don't deploy weak signals
        
        signal.status = SignalStatus.DEPLOYED
        signal.deployed_at = datetime.now()
        self.signals_deployed += 1
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "version": self.VERSION,
            "factors_available": len(self.factor_researcher.factors),
            "factors_researched": self.factors_researched,
            "signals_tested": self.signals_tested,
            "signals_deployed": self.signals_deployed,
            "factor_rankings": self.factor_researcher.rank_factors()[:5]
        }


# Global pipeline instance
_pipeline_instance: Optional[AlphaResearchPipeline] = None


def get_alpha_research_pipeline() -> AlphaResearchPipeline:
    """Get or create global Alpha Research Pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = AlphaResearchPipeline()
    return _pipeline_instance


if __name__ == "__main__":
    # Test the pipeline
    logging.basicConfig(level=logging.INFO)
    
    pipeline = get_alpha_research_pipeline()
    
    # Generate sample data
    np.random.seed(42)
    n = 1000
    returns = np.random.randn(n) * 0.02
    
    # Test factor research
    data = {
        "pe_ratio": np.random.randn(n),
        "market_cap": np.random.randn(n)
    }
    
    result = pipeline.research_new_factor("test_value", FactorType.VALUE, data, returns)
    print(f"Factor test: IC={result['ic']:.4f}, Significant={result['is_significant']}")
    
    # Test signal creation
    factor_values = {
        "value_pe": np.random.randn(n),
        "momentum_12m": np.random.randn(n)
    }
    
    signal_result = pipeline.create_and_test_signal(
        "value_momentum", ["value_pe", "momentum_12m"], returns, factor_values
    )
    print(f"Signal test: Sharpe={signal_result.get('sharpe_ratio', 0):.2f}")
    
    print(f"\nPipeline Stats: {pipeline.get_stats()}")
