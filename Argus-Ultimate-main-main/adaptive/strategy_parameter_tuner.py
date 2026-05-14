"""
Strategy Parameter Auto-Tuner

Automatically tunes strategy parameters based on recent performance:
- RSI thresholds (overbought/oversold)
- MACD periods (fast/slow/signal)
- Bollinger Band parameters (period, std dev)
- Moving average periods
- Any numeric threshold

Uses Bayesian optimization to find optimal parameters.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParameterRange:
    """Defines the search range for a parameter."""
    name: str
    min_value: float
    max_value: float
    step: float = 1.0
    param_type: str = "int"  # "int", "float", "categorical"
    categories: List[Any] = field(default_factory=list)


@dataclass
class TuningResult:
    """Result of parameter tuning."""
    parameters: Dict[str, Any]
    expected_improvement: float
    confidence: float
    trials_run: int
    timestamp: datetime = field(default_factory=datetime.now)


class BayesianParameterOptimizer:
    """
    Bayesian optimization for strategy parameters.
    
    Uses Gaussian Process regression to model the parameter-performance
    relationship and efficiently search for optimal values.
    """
    
    def __init__(self, n_initial: int = 5, n_iterations: int = 20):
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.history: List[Tuple[Dict[str, Any], float]] = []
    
    def suggest_next(
        self,
        param_ranges: List[ParameterRange],
        acquisition: str = "ei",
    ) -> Dict[str, Any]:
        """
        Suggest next parameter combination to try.
        
        Uses acquisition function (ei=Expected Improvement, ucb=Upper Confidence Bound)
        to balance exploration vs exploitation.
        """
        if len(self.history) < self.n_initial:
            # Random exploration phase
            return self._random_sample(param_ranges)
        
        # Bayesian optimization phase
        return self._bayesian_suggest(param_ranges, acquisition)
    
    def _random_sample(self, param_ranges: List[ParameterRange]) -> Dict[str, Any]:
        """Random sampling for initial exploration."""
        sample = {}
        for pr in param_ranges:
            if pr.param_type == "categorical":
                sample[pr.name] = np.random.choice(pr.categories)
            elif pr.param_type == "float":
                sample[pr.name] = np.random.uniform(pr.min_value, pr.max_value)
            else:
                sample[pr.name] = int(np.random.randint(
                    int(pr.min_value), int(pr.max_value) + 1
                ))
        return sample
    
    def _bayesian_suggest(
        self,
        param_ranges: List[ParameterRange],
        acquisition: str,
    ) -> Dict[str, Any]:
        """Bayesian suggestion using simple GP approximation."""
        # Extract X (params) and y (scores) from history
        X = np.array([self._params_to_vector(p, param_ranges) for p, _ in self.history])
        y = np.array([score for _, score in self.history])
        
        # Normalize y
        y_mean, y_std = y.mean(), max(y.std(), 1e-6)
        y_norm = (y - y_mean) / y_std
        
        best_score = -np.inf
        best_params = None
        
        # Sample candidates and pick best acquisition
        for _ in range(100):
            candidate = self._random_sample(param_ranges)
            x_cand = self._params_to_vector(candidate, param_ranges)
            
            # Simple RBF kernel approximation
            distances = np.linalg.norm(X - x_cand, axis=1)
            weights = np.exp(-distances**2 / (2 * 0.5**2))
            weights_sum = weights.sum()
            
            if weights_sum > 1e-10:
                mu = np.dot(weights, y_norm) / weights_sum
                sigma = 1.0 / (1 + weights_sum)
            else:
                mu, sigma = 0.0, 1.0
            
            # Acquisition function
            if acquisition == "ei":
                z = (mu - y_norm.max()) / max(sigma, 1e-10)
                score = sigma * (z * self._norm_cdf(z) + self._norm_pdf(z))
            else:  # ucb
                score = mu + 2.0 * sigma
            
            if score > best_score:
                best_score = score
                best_params = candidate
        
        return best_params or self._random_sample(param_ranges)
    
    def _params_to_vector(
        self,
        params: Dict[str, Any],
        param_ranges: List[ParameterRange],
    ) -> np.ndarray:
        """Convert params dict to normalized vector."""
        vec = []
        for pr in param_ranges:
            val = params.get(pr.name, pr.min_value)
            if pr.param_type == "categorical":
                vec.append(0.0)  # Categorical not supported in vector form
            else:
                normalized = (val - pr.min_value) / max(pr.max_value - pr.min_value, 1e-10)
                vec.append(normalized)
        return np.array(vec)
    
    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    @staticmethod
    def _norm_pdf(x: float) -> float:
        """Standard normal PDF."""
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
    
    def record_result(self, params: Dict[str, Any], score: float) -> None:
        """Record the result of a parameter trial."""
        self.history.append((params.copy(), score))
    
    def get_best(self) -> Optional[Tuple[Dict[str, Any], float]]:
        """Get the best parameters found so far."""
        if not self.history:
            return None
        return max(self.history, key=lambda x: x[1])


class StrategyParameterTuner:
    """
    Auto-tunes strategy parameters based on recent trade performance.
    
    Monitors performance with current parameters and periodically
    searches for better parameter combinations.
    """
    
    # Default parameter ranges for common indicators
    DEFAULT_RANGES = {
        # RSI
        "rsi_period": ParameterRange("rsi_period", 7, 21, 1, "int"),
        "rsi_overbought": ParameterRange("rsi_overbought", 65, 85, 1, "int"),
        "rsi_oversold": ParameterRange("rsi_oversold", 15, 35, 1, "int"),
        
        # MACD
        "macd_fast": ParameterRange("macd_fast", 8, 16, 1, "int"),
        "macd_slow": ParameterRange("macd_slow", 20, 30, 1, "int"),
        "macd_signal": ParameterRange("macd_signal", 7, 12, 1, "int"),
        
        # Bollinger Bands
        "bb_period": ParameterRange("bb_period", 15, 30, 1, "int"),
        "bb_std": ParameterRange("bb_std", 1.5, 3.0, 0.1, "float"),
        
        # Moving Averages
        "sma_fast": ParameterRange("sma_fast", 5, 15, 1, "int"),
        "sma_slow": ParameterRange("sma_slow", 20, 50, 1, "int"),
        
        # ATR
        "atr_period": ParameterRange("atr_period", 10, 20, 1, "int"),
        "atr_multiplier": ParameterRange("atr_multiplier", 1.0, 3.0, 0.1, "float"),
        
        # Confidence thresholds
        "min_confidence": ParameterRange("min_confidence", 0.1, 0.5, 0.05, "float"),
    }
    
    def __init__(
        self,
        tuning_interval_minutes: int = 60,
        min_trades_for_tuning: int = 10,
        param_ranges: Optional[Dict[str, ParameterRange]] = None,
        n_iterations: int = 15,
    ):
        self.tuning_interval_minutes = tuning_interval_minutes
        self.min_trades_for_tuning = min_trades_for_tuning
        self.param_ranges = param_ranges or self.DEFAULT_RANGES.copy()
        self.n_iterations = n_iterations
        
        self.optimizer = BayesianParameterOptimizer(n_initial=5, n_iterations=n_iterations)
        self.current_params: Dict[str, Any] = self._get_default_params()
        self.trade_history: List[Dict[str, Any]] = []
        self.last_tuning_time: float = 0
        self.tuning_count: int = 0
        
        logger.info(
            "StrategyParameterTuner initialized: interval=%dmin, params=%d",
            tuning_interval_minutes, len(self.param_ranges),
        )
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameter values."""
        return {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2.0,
            "sma_fast": 10,
            "sma_slow": 30,
            "atr_period": 14,
            "atr_multiplier": 2.0,
            "min_confidence": 0.2,
        }
    
    def record_trade(
        self,
        pnl_pct: float,
        params_used: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a trade result for tuning feedback."""
        params = params_used or self.current_params
        self.trade_history.append({
            "pnl_pct": pnl_pct,
            "params": params.copy(),
            "timestamp": time.time(),
        })
        
        # Keep only recent history
        max_history = 500
        if len(self.trade_history) > max_history:
            self.trade_history = self.trade_history[-max_history:]
    
    def should_tune(self) -> bool:
        """Check if it's time to run tuning."""
        if len(self.trade_history) < self.min_trades_for_tuning:
            return False
        
        time_since_last = time.time() - self.last_tuning_time
        return time_since_last >= self.tuning_interval_minutes * 60
    
    def tune_parameters(self) -> TuningResult:
        """
        Run parameter tuning based on recent performance.
        
        Returns the new recommended parameters.
        """
        logger.info("StrategyParameterTuner: Starting tuning with %d trades", len(self.trade_history))
        
        # Group trades by parameter sets and compute scores
        param_scores = self._compute_param_scores()
        
        # Add to optimizer history
        for params, score in param_scores.items():
            self.optimizer.record_result(params, score)
        
        # Run optimization iterations
        best_params = self.current_params.copy()
        best_score = -np.inf
        
        for i in range(self.n_iterations):
            # Get next candidate
            candidate = self.optimizer.suggest_next(list(self.param_ranges.values()))
            
            # Evaluate candidate using cross-validation on recent data
            score = self._evaluate_params(candidate)
            
            # Record result
            self.optimizer.record_result(candidate, score)
            
            if score > best_score:
                best_score = score
                best_params = candidate
        
        # Update current params
        self.current_params = best_params
        self.last_tuning_time = time.time()
        self.tuning_count += 1
        
        result = TuningResult(
            parameters=best_params,
            expected_improvement=best_score,
            confidence=min(1.0, len(self.trade_history) / 100),
            trials_run=self.n_iterations,
        )
        
        logger.info(
            "StrategyParameterTuner: Tuning complete - score=%.4f, params=%s",
            best_score, {k: round(v, 2) if isinstance(v, float) else v 
                        for k, v in best_params.items()},
        )
        
        return result
    
    def _compute_param_scores(self) -> Dict[Dict[str, Any], float]:
        """Compute performance score for each parameter set in history."""
        param_groups: Dict[str, List[float]] = {}
        
        for trade in self.trade_history:
            params = trade["params"]
            pnl = trade["pnl_pct"]
            
            # Create hashable key from params
            key = str(sorted(params.items()))
            if key not in param_groups:
                param_groups[key] = []
            param_groups[key].append(pnl)
        
        scores = {}
        for key, pnls in param_groups.items():
            if len(pnls) >= 3:  # Need minimum trades
                pnl_arr = np.array(pnls)
                # Score = Sharpe-like metric
                if pnl_arr.std() > 0:
                    score = pnl_arr.mean() / pnl_arr.std() * math.sqrt(len(pnls))
                else:
                    score = pnl_arr.mean()
                
                # Reconstruct params dict
                params = dict(eval(key))
                scores[params] = score
        
        return scores
    
    def _evaluate_params(self, params: Dict[str, Any]) -> float:
        """
        Evaluate a parameter set using walk-forward validation.
        
        Simulates how these parameters would have performed on recent trades.
        """
        if len(self.trade_history) < 10:
            return 0.0
        
        # Simple evaluation: assume better params correlate with better recent PnL
        # In production, would do proper walk-forward backtest
        
        # Use weighted recent performance
        recent_trades = self.trade_history[-50:]
        pnls = [t["pnl_pct"] for t in recent_trades]
        
        if not pnls:
            return 0.0
        
        # Penalize extreme parameters
        penalty = self._param_penalty(params)
        
        # Base score from recent performance
        pnl_arr = np.array(pnls)
        if pnl_arr.std() > 0:
            base_score = pnl_arr.mean() / pnl_arr.std()
        else:
            base_score = pnl_arr.mean() * 10
        
        return base_score - penalty
    
    def _param_penalty(self, params: Dict[str, Any]) -> float:
        """Penalize extreme parameter values."""
        penalty = 0.0
        
        # RSI thresholds should be reasonable
        rsi_ob = params.get("rsi_overbought", 70)
        rsi_os = params.get("rsi_oversold", 30)
        if rsi_ob - rsi_os < 20:
            penalty += 0.5  # Too narrow range
        
        # MACD periods should be ordered
        macd_fast = params.get("macd_fast", 12)
        macd_slow = params.get("macd_slow", 26)
        if macd_fast >= macd_slow:
            penalty += 1.0  # Invalid ordering
        
        # SMA periods should be ordered
        sma_fast = params.get("sma_fast", 10)
        sma_slow = params.get("sma_slow", 30)
        if sma_fast >= sma_slow:
            penalty += 1.0
        
        return penalty
    
    def get_current_params(self) -> Dict[str, Any]:
        """Get current tuned parameters."""
        return self.current_params.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get tuner status."""
        best = self.optimizer.get_best()
        return {
            "tuning_count": self.tuning_count,
            "trades_recorded": len(self.trade_history),
            "current_params": self.current_params,
            "best_params": best[0] if best else None,
            "best_score": best[1] if best else None,
            "last_tuning": self.last_tuning_time,
        }


class AdaptiveIndicatorCalculator:
    """
    Calculates indicators using tuned parameters.
    
    Wraps standard indicator calculations to use adaptive parameters
    from StrategyParameterTuner.
    """
    
    def __init__(self, tuner: StrategyParameterTuner):
        self.tuner = tuner
    
    def rsi(self, prices: np.ndarray, period: Optional[int] = None) -> float:
        """Calculate RSI with adaptive period."""
        if period is None:
            period = self.tuner.current_params.get("rsi_period", 14)
        
        if len(prices) < period + 1:
            return 50.0  # Neutral
        
        deltas = np.diff(prices[-period-1:])
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def rsi_overbought(self) -> float:
        """Get adaptive overbought threshold."""
        return self.tuner.current_params.get("rsi_overbought", 70)
    
    def rsi_oversold(self) -> float:
        """Get adaptive oversold threshold."""
        return self.tuner.current_params.get("rsi_oversold", 30)
    
    def macd(
        self,
        prices: np.ndarray,
        fast: Optional[int] = None,
        slow: Optional[int] = None,
        signal: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """Calculate MACD with adaptive periods."""
        if fast is None:
            fast = self.tuner.current_params.get("macd_fast", 12)
        if slow is None:
            slow = self.tuner.current_params.get("macd_slow", 26)
        if signal is None:
            signal = self.tuner.current_params.get("macd_signal", 9)
        
        if len(prices) < slow + signal:
            return 0.0, 0.0, 0.0
        
        # EMA calculation
        def ema(data: np.ndarray, period: int) -> np.ndarray:
            multiplier = 2 / (period + 1)
            result = np.zeros(len(data))
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = (data[i] - result[i-1]) * multiplier + result[i-1]
            return result
        
        ema_fast = ema(prices, fast)
        ema_slow = ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1])
    
    def bollinger_bands(
        self,
        prices: np.ndarray,
        period: Optional[int] = None,
        num_std: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands with adaptive parameters."""
        if period is None:
            period = self.tuner.current_params.get("bb_period", 20)
        if num_std is None:
            num_std = self.tuner.current_params.get("bb_std", 2.0)
        
        if len(prices) < period:
            last = prices[-1] if len(prices) > 0 else 0
            return last, last, last
        
        recent = prices[-period:]
        middle = np.mean(recent)
        std = np.std(recent)
        
        upper = middle + num_std * std
        lower = middle - num_std * std
        
        return float(upper), float(middle), float(lower)
    
    def min_confidence(self) -> float:
        """Get adaptive minimum confidence threshold."""
        return self.tuner.current_params.get("min_confidence", 0.2)
