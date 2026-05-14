"""
Adaptive Strategy Allocator - Real-Time Learning Component

This component dynamically adjusts strategy weights based on:
- Performance metrics (Sharpe, win rate, drawdown)
- Market regime (volatile, range-bound, trending)
- Correlation between strategies

Key Features:
- Performance-based reallocation
- Regime-aware adjustments
- Bounded parameter tuning
- Full safety validation integration
"""

from __future__ import annotations
import logging
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import spearmanr

from .orchestrator import LearningComponent

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """Tracks performance metrics for a single strategy"""
    
    strategy_name: str
    weight: float = 1.0  # Current allocation weight (0-1)
    sharpe_ratio: float = 0.0
    win_rate: float = 0.5
    max_drawdown: float = 0.0
    recent_trades: List[Dict] = field(default_factory=list)
    regime_performance: Dict[str, Dict] = field(default_factory=dict)  # regime -> metrics
    
    def update_metrics(self, trade_result: Dict) -> None:
        """Update performance metrics from a trade result"""
        self.recent_trades.append(trade_result)
        if len(self.recent_trades) > 100:  # Keep last 100 trades
            self.recent_trades.pop(0)
        
        # Update basic metrics
        self._update_basic_metrics()
    
    def _update_basic_metrics(self) -> None:
        """Recalculate performance metrics from recent trades"""
        if not self.recent_trades:
            return
        
        pnls = [t['pnl'] for t in self.recent_trades if 'pnl' in t]
        returns = [t['return_pct'] for t in self.recent_trades if 'return_pct' in t]
        
        if len(returns) >= 5:
            self.win_rate = len([r for r in returns if r > 0]) / len(returns)
            
            # Simple Sharpe approximation (annualized)
            mean_return = np.mean(returns) * 252 * 24  # Hourly to annual
            std_return = np.std(returns) * np.sqrt(252 * 24)
            self.sharpe_ratio = mean_return / std_return if std_return > 0 else 0
            
            # Simple drawdown calculation
            cumulative = np.cumsum(returns)
            peak = np.maximum.accumulate(cumulative)
            drawdowns = (peak - cumulative) / peak
            self.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

    def get_regime_metrics(self, regime: str) -> Dict:
        """Get performance metrics for a specific regime"""
        if regime not in self.regime_performance:
            self.regime_performance[regime] = {
                'trades': 0,
                'win_rate': 0.5,
                'avg_return': 0.0,
                'sharpe': 0.0
            }
        return self.regime_performance[regime]


class AdaptiveStrategyAllocator(LearningComponent):
    """Dynamically allocates weights to strategies based on performance"""
    
    def __init__(self):
        super().__init__(
            name="strategy_allocator",
            version="1.0",
            enabled=True,
            update_frequency=5  # Update every 5 trade cycles
        )
        
        # Performance tracking
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.current_regime: str = "stable"
        self.min_weight: float = 0.05  # Minimum allocation to any strategy
        self.max_weight: float = 0.50  # Maximum allocation to any strategy
        self.weight_change_limit: float = 0.10  # Max weight change per update
        
        # Regime detection thresholds
        self.regime_thresholds = {
            'volatile': {'volatility': 0.02, 'trend_strength': 0.3},
            'range': {'volatility': 0.005, 'trend_strength': 0.1},
            'trending': {'volatility': 0.01, 'trend_strength': 0.5}
        }
        
        # Correlation tracking
        self.strategy_correlation: Dict[Tuple[str, str], float] = {}
        self.correlation_window = 20  # Number of trades to use for correlation
        
        # State tracking
        self.last_allocation: Dict[str, float] = {}
        self.allocation_history: List[Dict] = []
        
    def initialize_strategies(self, strategy_names: List[str]) -> None:
        """Initialize tracking for all strategies"""
        for name in strategy_names:
            if name not in self.strategy_performance:
                self.strategy_performance[name] = StrategyPerformance(strategy_name=name)
                self.last_allocation[name] = 1.0 / len(strategy_names)  # Equal initial allocation
        
        # Initialize correlation tracking
        for i, strat1 in enumerate(strategy_names):
            for strat2 in strategy_names[i+1:]:
                self.strategy_correlation[(strat1, strat2)] = 0.0
    
    def learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from new market data and update strategy allocations"""
        
        # Update regime detection
        self._update_regime(data)
        
        # Update strategy performance metrics
        self._update_strategy_performance(data)
        
        # Update strategy correlations
        self._update_strategy_correlations()
        
        # Calculate new optimal allocation
        new_allocation = self._calculate_optimal_allocation()
        
        # Apply bounds and limits
        final_allocation = self._apply_constraints(new_allocation)
        
        # Store history
        self.allocation_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'regime': self.current_regime,
            'allocation': final_allocation.copy()
        })
        
        # Keep only last 100 allocations
        if len(self.allocation_history) > 100:
            self.allocation_history.pop(0)
        
        return {'strategy_weights': final_allocation}
    
    def _update_regime(self, data: Dict[str, Any]) -> None:
        """Update current market regime based on volatility and trend"""
        if 'market_data' not in data:
            return
        
        market_data = data['market_data']
        volatility = market_data.get('volatility', 0.01)
        trend_strength = market_data.get('trend_strength', 0.2)
        
        # Determine current regime
        if volatility > self.regime_thresholds['volatile']['volatility']:
            self.current_regime = 'volatile'
        elif trend_strength > self.regime_thresholds['trending']['trend_strength']:
            self.current_regime = 'trending'
        elif volatility < self.regime_thresholds['range']['volatility']:
            self.current_regime = 'range'
        else:
            self.current_regime = 'stable'
    
    def _update_strategy_performance(self, data: Dict[str, Any]) -> None:
        """Update performance metrics for all strategies"""
        if 'trade_results' not in data:
            return
        
        for trade in data['trade_results']:
            strategy_name = trade.get('strategy')
            if strategy_name and strategy_name in self.strategy_performance:
                self.strategy_performance[strategy_name].update_metrics(trade)
                
                # Update regime-specific metrics
                regime_metrics = self.strategy_performance[strategy_name].get_regime_metrics(self.current_regime)
                regime_metrics['trades'] += 1
                regime_metrics['win_rate'] = (
                    (regime_metrics['win_rate'] * (regime_metrics['trades'] - 1) + 
                     (1 if trade.get('pnl', 0) > 0 else 0)) / regime_metrics['trades']
                )
                
                if 'return_pct' in trade:
                    regime_metrics['avg_return'] = (
                        (regime_metrics['avg_return'] * (regime_metrics['trades'] - 1) + 
                         trade['return_pct']) / regime_metrics['trades']
                    )
    
    def _update_strategy_correlations(self) -> None:
        """Update correlation between strategy returns"""
        if len(self.strategy_performance) < 2:
            return
        
        # Collect recent returns for all strategies
        strategy_returns = {}
        for strat_name, perf in self.strategy_performance.items():
            returns = []
            for trade in perf.recent_trades:
                if 'return_pct' in trade:
                    returns.append(trade['return_pct'])
            strategy_returns[strat_name] = returns[-self.correlation_window:]  # Take most recent
        
        # Calculate pairwise correlations
        strat_names = list(strategy_returns.keys())
        for i in range(len(strat_names)):
            for j in range(i+1, len(strat_names)):
                strat1 = strat_names[i]
                strat2 = strat_names[j]
                
                returns1 = strategy_returns[strat1]
                returns2 = strategy_returns[strat2]
                
                if len(returns1) >= 5 and len(returns2) >= 5:
                    # Use minimum length to avoid mismatch
                    min_len = min(len(returns1), len(returns2))
                    corr, _ = spearmanr(returns1[-min_len:], returns2[-min_len:])
                    self.strategy_correlation[(strat1, strat2)] = corr
                    self.strategy_correlation[(strat2, strat1)] = corr
    
    def _calculate_optimal_allocation(self) -> Dict[str, float]:
        """Calculate optimal strategy allocation using performance metrics"""
        if not self.strategy_performance:
            return {}
        
        # Get current performance metrics
        strategies = list(self.strategy_performance.keys())
        
        # Create score for each strategy (composite of metrics)
        scores = {}
        regime_scores = {}
        overall_scores = {}
        
        for strat in strategies:
            perf = self.strategy_performance[strat]
            
            # Get regime-specific metrics
            regime_metrics = perf.get_regime_metrics(self.current_regime)
            
            # Calculate regime-specific score (this is the primary driver)
            regime_score = (
                0.7 * regime_metrics['win_rate'] + 
                0.3 * (regime_metrics['avg_return'] * 100)  # Scale return to percentage
            )
            regime_scores[strat] = regime_score
            
            # Add overall performance as secondary factor
            overall_score = (
                0.5 * perf.sharpe_ratio +
                0.3 * perf.win_rate +
                0.2 * (1 - perf.max_drawdown)
            )
            overall_scores[strat] = overall_score
            
            # Combined score - regime performance is more important
            scores[strat] = 0.8 * regime_score + 0.2 * overall_score
            
            # Debug output
            logger.info(f"Strategy {strat} scores - Regime: {regime_score:.2f}, Overall: {overall_score:.2f}, Combined: {scores[strat]:.2f}")
        
        # Normalize scores to weights
        total_score = sum(scores.values())
        if total_score <= 0:
            # Fallback to equal weights if all scores are zero/negative
            logger.warning("All strategy scores are zero/negative - using equal weights")
            return {strat: 1.0/len(strategies) for strat in strategies}
        
        # Calculate raw weights
        raw_weights = {strat: scores[strat]/total_score for strat in strategies}
        
        # Debug output
        logger.info(f"Raw weights before correlation adjustment: {raw_weights}")
        
        # Apply correlation penalty (reduce weights for highly correlated strategies)
        adjusted_weights = self._apply_correlation_penalty(raw_weights)
        
        # Debug output
        logger.info(f"Final adjusted weights: {adjusted_weights}")
        
        return adjusted_weights
    
    def _apply_correlation_penalty(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Adjust weights based on strategy correlations to improve diversification"""
        if len(self.strategy_performance) < 2:
            return weights
        
        # Calculate average correlation for each strategy
        strat_names = list(weights.keys())
        avg_correlation = {strat: 0.0 for strat in strat_names}
        
        for i in range(len(strat_names)):
            strat1 = strat_names[i]
            count = 0
            
            for j in range(len(strat_names)):
                if i != j:
                    strat2 = strat_names[j]
                    corr = self.strategy_correlation.get((strat1, strat2), 0.0)
                    avg_correlation[strat1] += abs(corr)
                    count += 1
            
            if count > 0:
                avg_correlation[strat1] /= count
        
        # Apply penalty - reduce weight for highly correlated strategies
        adjusted_weights = {}
        total_adjusted = 0.0
        
        for strat in strat_names:
            # Penalty factor - higher correlation = lower weight
            penalty = 1.0 - 0.5 * avg_correlation[strat]  # Reduce up to 50% for perfect correlation
            penalty = max(0.5, min(1.0, penalty))  # Keep between 0.5 and 1.0
            
            adjusted_weights[strat] = weights[strat] * penalty
            total_adjusted += adjusted_weights[strat]
        
        # Renormalize
        if total_adjusted > 0:
            return {strat: weight/total_adjusted for strat, weight in adjusted_weights.items()}
        
        return weights
    
    def _apply_constraints(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Apply bounds and change limits to allocation"""
        constrained = {}
        
        for strat, weight in allocation.items():
            # Apply min/max bounds
            constrained_weight = max(self.min_weight, min(self.max_weight, weight))
            
            # Apply change limit from last allocation
            last_weight = self.last_allocation.get(strat, constrained_weight)
            max_change = self.weight_change_limit * last_weight
            
            # Limit both increases and decreases
            constrained_weight = max(
                last_weight - max_change,
                min(last_weight + max_change, constrained_weight)
            )
            
            constrained[strat] = constrained_weight
        
        # Renormalize to sum to 1.0
        total = sum(constrained.values())
        if total > 0:
            self.last_allocation = {strat: weight/total for strat, weight in constrained.items()}
            return self.last_allocation
        
        # Fallback to equal weights if something went wrong
        strategies = list(allocation.keys())
        equal_weight = 1.0 / len(strategies) if strategies else 0
        self.last_allocation = {strat: equal_weight for strat in strategies}
        return self.last_allocation
    
    def get_params(self) -> Dict[str, Any]:
        """Get current parameters"""
        return {
            'strategy_weights': self.last_allocation.copy(),
            'current_regime': self.current_regime,
            'min_weight': self.min_weight,
            'max_weight': self.max_weight,
            'weight_change_limit': self.weight_change_limit,
            'regime_thresholds': self.regime_thresholds.copy()
        }
    
    def rollback(self) -> None:
        """Revert to last known good state"""
        if len(self.allocation_history) > 1:
            # Revert to previous allocation
            previous = self.allocation_history[-2]
            self.last_allocation = previous['allocation'].copy()
            self.current_regime = previous['regime']
            logger.info(f"Rolled back to previous allocation from {previous['timestamp']}")
        else:
            # Fallback to equal weights
            strategies = list(self.strategy_performance.keys())
            equal_weight = 1.0 / len(strategies) if strategies else 0
            self.last_allocation = {strat: equal_weight for strat in strategies}
            logger.warning("No allocation history - reset to equal weights")
    
    def validate(self, new_params: Dict[str, Any]) -> bool:
        """Validate proposed parameter changes"""
        if 'strategy_weights' not in new_params:
            return False
        
        weights = new_params['strategy_weights']
        
        # Check weights sum to ~1 (allowing for floating point precision)
        total = sum(weights.values())
        if not (0.99 <= total <= 1.01):
            logger.warning(f"Strategy weights don't sum to 1: {total:.3f}")
            return False
        
        # Check individual weight bounds
        for strat, weight in weights.items():
            if weight < self.min_weight or weight > self.max_weight:
                logger.warning(f"Weight for {strat} out of bounds: {weight:.3f}")
                return False
        
        # Check regime is valid
        if 'current_regime' in new_params and new_params['current_regime'] not in self.regime_thresholds:
            logger.warning(f"Invalid regime: {new_params['current_regime']}")
            return False
        
        return True
    
    def learn_from_trade(self, trade: Dict[str, Any]) -> None:
        """Specialized learning from individual trade results"""
        if 'strategy' in trade:
            strategy_name = trade['strategy']
            if strategy_name in self.strategy_performance:
                self.strategy_performance[strategy_name].update_metrics(trade)
                
                # Update regime-specific metrics
                regime_metrics = self.strategy_performance[strategy_name].get_regime_metrics(self.current_regime)
                regime_metrics['trades'] += 1
                regime_metrics['win_rate'] = (
                    (regime_metrics['win_rate'] * (regime_metrics['trades'] - 1) + 
                     (1 if trade.get('pnl', 0) > 0 else 0)) / regime_metrics['trades']
                )
                
                if 'return_pct' in trade:
                    regime_metrics['avg_return'] = (
                        (regime_metrics['avg_return'] * (regime_metrics['trades'] - 1) + 
                         trade['return_pct']) / regime_metrics['trades']
                    )
    
    def _restore_state(self, state: Dict) -> None:
        """Restore state from saved data"""
        if 'params' in state:
            params = state['params']
            if 'strategy_weights' in params:
                self.last_allocation = params['strategy_weights']
            if 'current_regime' in params:
                self.current_regime = params['current_regime']
            if 'allocation_history' in state:
                self.allocation_history = state['allocation_history']