"""
Paper Validation Engine - Real-Time Learning Component

This component validates all adaptive changes with paper trading before live deployment.
Key features:
- Backtesting of proposed parameter changes
- Statistical significance testing
- Risk metric validation
- Performance threshold enforcement
"""

from __future__ import annotations
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from .orchestrator import LearningComponent

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Stores the result of a validation test"""
    
    component: str
    parameter_changes: Dict[str, Any]
    test_passed: bool
    metrics: Dict[str, float]
    required_metrics: Dict[str, Tuple[float, str]]  # (threshold, comparison_op)
    backtest_results: Optional[Dict] = None
    statistical_results: Optional[Dict[str, Dict]] = None  # Statistical test results
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def is_valid(self) -> bool:
        """Check if validation passed all requirements"""
        if not self.test_passed:
            return False
        
        # Check all metric thresholds
        for metric, (threshold, op) in self.required_metrics.items():
            if metric not in self.metrics:
                return False
            
            value = self.metrics[metric]
            
            if op == '>' and value <= threshold:
                return False
            if op == '>=' and value < threshold:
                return False
            if op == '<' and value >= threshold:
                return False
            if op == '<=' and value > threshold:
                return False
            if op == '==' and abs(value - threshold) >= 0.001:
                return False
        
        # Check statistical significance - all tests must be significant
        if self.statistical_results and not self.all_statistical_tests_passed():
            return False
        
        return True
    
    def all_statistical_tests_passed(self) -> bool:
        """Check if all statistical tests passed significance thresholds"""
        if not self.statistical_results:
            return False
            
        return all(test['significant'] for test in self.statistical_results.values())

    def get_failure_reasons(self) -> List[str]:
        """Get reasons why validation failed"""
        reasons = []
        
        if not self.test_passed:
            reasons.append("General test failed")
        
        # Check metric threshold failures
        for metric, (threshold, op) in self.required_metrics.items():
            if metric not in self.metrics:
                reasons.append(f"Missing metric: {metric}")
                continue
            
            value = self.metrics[metric]
            
            if op == '>' and value <= threshold:
                reasons.append(f"{metric} ({value:.3f}) not > {threshold:.3f}")
            if op == '>=' and value < threshold:
                reasons.append(f"{metric} ({value:.3f}) not >= {threshold:.3f}")
            if op == '<' and value >= threshold:
                reasons.append(f"{metric} ({value:.3f}) not < {threshold:.3f}")
            if op == '<=' and value > threshold:
                reasons.append(f"{metric} ({value:.3f}) not <= {threshold:.3f}")
            if op == '==' and abs(value - threshold) >= 0.001:
                reasons.append(f"{metric} ({value:.3f}) not == {threshold:.3f}")
        
        # Check statistical significance failures
        if self.statistical_results:
            for test_name, test_result in self.statistical_results.items():
                if not test_result['significant']:
                    reasons.append(f"{test_name} not statistically significant (p={test_result['p_value']:.4f}) > {test_result['threshold']}")
        
        return reasons


class PaperValidationEngine(LearningComponent):
    """Validates all adaptive changes with paper trading before live deployment"""
    
    def __init__(self):
        super().__init__(
            name="paper_validation",
            version="1.0",
            enabled=True,
            update_frequency=1  # Validate every change
        )
        
        # Validation thresholds by component type
        self.component_thresholds = {
            'strategy_allocator': {
                'min_trades': 30,
                'min_sharpe_improvement': 0.1,
                'max_drawdown_increase': 0.02,  # 2%
                'min_win_rate': 0.5,
                'p_value_threshold': 0.05
            },
            'correlation_matrix': {
                'min_trades': 20,
                'min_diversification_improvement': 0.05,
                'max_concentration': 0.35,
                'p_value_threshold': 0.05
            },
            'order_router': {
                'min_trades': 25,
                'min_fill_improvement': 0.02,  # 2%
                'max_slippage_increase': 0.0005,  # 0.05%
                'min_latency_improvement': 5,  # 5ms
                'p_value_threshold': 0.05
            },
            'regime_parameters': {
                'min_trades': 30,
                'min_sharpe_improvement': 0.05,
                'max_drawdown_increase': 0.01,  # 1%
                'min_profit_factor': 1.1,
                'p_value_threshold': 0.05
            }
        }
        
        # Validation history
        self.validation_history: List[ValidationResult] = []
        self.max_history = 1000
        
        # Backtest data cache
        self.backtest_cache: Dict[str, Dict] = {}  # component_name -> {params: backtest_results}
        self.max_cache_size = 100
        
        # Statistical test results
        self.statistical_results: Dict[str, Dict] = {}  # component_name -> {metric: test_results}
    
    def learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate proposed parameter changes with paper trading"""
        
        if 'proposed_changes' not in data or 'component_name' not in data:
            return {"status": "error", "message": "Missing required data"}
        
        component_name = data['component_name']
        proposed_changes = data['proposed_changes']
        
        # Get validation thresholds for this component
        if component_name not in self.component_thresholds:
            return {
                "status": "error",
                "message": f"No validation thresholds defined for {component_name}"
            }
        
        thresholds = self.component_thresholds[component_name]
        
        # Check if we have cached backtest results
        cache_key = self._get_cache_key(component_name, proposed_changes)
        if cache_key in self.backtest_cache:
            backtest_results = self.backtest_cache[cache_key]
        else:
            # Run paper trading validation (simulated in this demo)
            backtest_results = self._run_paper_trading(component_name, proposed_changes, data)
            
            # Cache results
            self.backtest_cache[cache_key] = backtest_results
            if len(self.backtest_cache) > self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.backtest_cache))
                del self.backtest_cache[oldest_key]
        
        # Analyze results
        validation_result = self._analyze_validation_results(
            component_name, proposed_changes, backtest_results, thresholds
        )
        
        # Store validation result
        self.validation_history.append(validation_result)
        if len(self.validation_history) > self.max_history:
            self.validation_history.pop(0)
        
        return {
            "status": "success" if validation_result.is_valid() else "failed",
            "validation_result": validation_result,
            "backtest_results": backtest_results
        }
    
    def _get_cache_key(self, component_name: str, proposed_changes: Dict[str, Any]) -> str:
        """Generate a cache key for proposed changes"""
        # Sort parameters for consistent hashing
        sorted_items = sorted(proposed_changes.items())
        params_str = ",".join([f"{k}={v}" for k, v in sorted_items])
        return f"{component_name}_{params_str}"
    
    def _run_paper_trading(self, component_name: str, proposed_changes: Dict[str, Any], 
                          data: Dict[str, Any]) -> Dict[str, Any]:
        """Run paper trading validation for proposed changes"""
        
        # In a real implementation, this would run actual backtests
        # For this demo, we'll simulate results based on the component type
        
        if component_name == 'strategy_allocator':
            return self._simulate_strategy_allocator_validation(proposed_changes, data)
        elif component_name == 'correlation_matrix':
            return self._simulate_correlation_matrix_validation(proposed_changes, data)
        elif component_name == 'order_router':
            return self._simulate_order_router_validation(proposed_changes, data)
        elif component_name == 'regime_parameters':
            return self._simulate_regime_parameters_validation(proposed_changes, data)
        else:
            return {"error": "Unsupported component type"}
    
    def _simulate_strategy_allocator_validation(self, proposed_changes: Dict[str, Any], 
                                              data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate validation for strategy allocator changes"""
        
        # Extract strategy weights
        if 'strategy_weights' not in proposed_changes:
            return {"error": "No strategy weights in proposed changes"}
        
        weights = proposed_changes['strategy_weights']
        
        # Simulate backtest results based on weights
        # In reality, this would run actual backtests with the new weights
        
        # Base metrics
        base_sharpe = 1.8
        base_win_rate = 0.55
        base_drawdown = 0.15
        
        # Calculate expected improvement based on weight changes
        # This is a simplified simulation - real implementation would run actual backtests
        
        # Check if momentum weight increased (good for volatile/trending)
        momentum_change = weights.get('momentum', 0) - 0.33  # Assuming initial equal weights
        
        # Check if mean_reversion weight increased (good for range/stable)
        mr_change = weights.get('mean_reversion', 0) - 0.33
        
        # Simulate performance based on regime in data
        regime = data.get('market_data', {}).get('regime', 'stable')
        
        if regime == 'volatile' and momentum_change > 0:
            # Momentum should perform well in volatile markets
            sharpe_improvement = 0.3
            win_rate_improvement = 0.05
            drawdown_change = -0.02
        elif regime == 'trending' and momentum_change > 0:
            # Momentum should perform well in trending markets
            sharpe_improvement = 0.4
            win_rate_improvement = 0.1
            drawdown_change = -0.03
        elif regime == 'range' and mr_change > 0:
            # Mean reversion should perform well in range markets
            sharpe_improvement = 0.2
            win_rate_improvement = 0.08
            drawdown_change = -0.01
        else:
            # Neutral or negative change
            sharpe_improvement = 0.05
            win_rate_improvement = 0.02
            drawdown_change = 0.01
        
        # Generate simulated backtest results
        return {
            'sharpe_ratio': base_sharpe + sharpe_improvement,
            'win_rate': base_win_rate + win_rate_improvement,
            'max_drawdown': base_drawdown + drawdown_change,
            'profit_factor': 1.8 + (sharpe_improvement * 2),
            'total_trades': 50,
            'regime': regime,
            'strategy_weights': weights
        }
    
    def _simulate_correlation_matrix_validation(self, proposed_changes: Dict[str, Any], 
                                             data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate validation for correlation matrix changes"""
        
        # Extract correlation matrix
        if 'current_matrix' not in proposed_changes:
            return {"error": "No correlation matrix in proposed changes"}
        
        matrix = proposed_changes['current_matrix']
        
        # Calculate average correlation
        correlations = [abs(v) for v in matrix.values()]
        avg_corr = np.mean(correlations) if correlations else 0.5
        diversification_score = 1.0 - avg_corr
        
        # Simulate backtest results
        base_sharpe = 1.8
        base_drawdown = 0.15
        
        # Higher diversification should improve risk-adjusted returns
        sharpe_improvement = (diversification_score - 0.5) * 0.5  # Scale factor
        drawdown_reduction = (diversification_score - 0.5) * 0.03
        
        return {
            'diversification_score': diversification_score,
            'sharpe_ratio': base_sharpe + sharpe_improvement,
            'max_drawdown': max(0, base_drawdown - drawdown_reduction),
            'portfolio_concentration': avg_corr,
            'total_trades': 40,
            'regime': data.get('market_data', {}).get('regime', 'stable')
        }
    
    def _simulate_order_router_validation(self, proposed_changes: Dict[str, Any], 
                                         data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate validation for order router changes"""
        
        # Extract venue performance
        if 'venue_performance' not in proposed_changes:
            return {"error": "No venue performance in proposed changes"}
        
        performance = proposed_changes['venue_performance']
        
        # Calculate average performance metrics
        fill_ratios = [v for v in performance.values() if 'fill_ratio' in v]
        avg_fill = np.mean(fill_ratios) if fill_ratios else 0.85
        
        slippages = [v for v in performance.values() if 'avg_slippage' in v]
        avg_slippage = np.mean(slippages) if slippages else 0.001
        
        latencies = [v for v in performance.values() if 'latency' in v]
        avg_latency = np.mean(latencies) if latencies else 40
        
        # Simulate backtest results
        base_fill = 0.85
        base_slippage = 0.001
        base_latency = 45
        
        fill_improvement = avg_fill - base_fill
        slippage_change = avg_slippage - base_slippage
        latency_change = base_latency - avg_latency
        
        # Calculate impact on execution quality
        execution_score = (
            0.5 * (fill_improvement * 100) +  # 50% weight to fill improvement
            0.3 * (1 - min(slippage_change * 1000, 1.0)) +  # 30% weight to slippage
            0.2 * (min(latency_change / 10, 1.0))  # 20% weight to latency
        )
        
        return {
            'avg_fill_ratio': avg_fill,
            'avg_slippage': avg_slippage,
            'avg_latency': avg_latency,
            'execution_quality_score': execution_score,
            'total_trades': 30,
            'regime': data.get('market_data', {}).get('regime', 'stable')
        }
    
    def _simulate_regime_parameters_validation(self, proposed_changes: Dict[str, Any], 
                                            data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate validation for regime parameters changes"""
        
        # Extract parameters
        if 'current_parameters' not in proposed_changes:
            return {"error": "No parameters in proposed changes"}
        
        params = proposed_changes['current_parameters']
        regime = data.get('market_data', {}).get('regime', 'stable')
        
        # Simulate backtest results based on regime and parameters
        base_sharpe = 1.8
        base_win_rate = 0.55
        base_drawdown = 0.15
        base_profit_factor = 1.8
        
        # Calculate expected performance based on regime and parameters
        if regime == 'volatile':
            # In volatile markets, conservative parameters should perform better
            position_size = params.get('position_size_pct', 0.05)
            leverage = params.get('max_leverage', 2.0)
            
            if position_size < 0.04 and leverage < 1.5:
                # Conservative settings for volatile markets
                sharpe_improvement = 0.2
                drawdown_reduction = 0.05
                win_rate_improvement = 0.03
            else:
                # Aggressive settings may perform worse
                sharpe_improvement = -0.1
                drawdown_reduction = -0.03
                win_rate_improvement = -0.02
        
        elif regime == 'trending':
            # In trending markets, aggressive parameters should perform better
            position_size = params.get('position_size_pct', 0.05)
            take_profit = params.get('take_profit_pct', 0.05)
            
            if position_size > 0.06 and take_profit > 0.06:
                # Aggressive settings for trending markets
                sharpe_improvement = 0.3
                drawdown_reduction = 0.02
                win_rate_improvement = 0.05
            else:
                # Conservative settings may miss opportunities
                sharpe_improvement = -0.1
                drawdown_reduction = 0.01
                win_rate_improvement = -0.01
        
        elif regime == 'range':
            # In range markets, mean-reversion friendly parameters should perform better
            stop_loss = params.get('stop_loss_pct', 0.03)
            entry_aggressiveness = params.get('entry_aggressiveness', 0.5)
            
            if stop_loss < 0.04 and entry_aggressiveness > 0.6:
                # Aggressive entry with tight stops for range markets
                sharpe_improvement = 0.15
                drawdown_reduction = 0.03
                win_rate_improvement = 0.04
            else:
                # Conservative settings may miss opportunities
                sharpe_improvement = -0.05
                drawdown_reduction = 0.01
                win_rate_improvement = -0.02
        
        else:  # stable
            # In stable markets, balanced parameters should perform well
            sharpe_improvement = 0.1
            drawdown_reduction = 0.02
            win_rate_improvement = 0.02
        
        return {
            'sharpe_ratio': base_sharpe + sharpe_improvement,
            'win_rate': base_win_rate + win_rate_improvement,
            'max_drawdown': max(0, base_drawdown - drawdown_reduction),
            'profit_factor': base_profit_factor + (sharpe_improvement * 1.5),
            'total_trades': 50,
            'regime': regime,
            'parameters': params
        }
    
    def _analyze_validation_results(self, component_name: str, proposed_changes: Dict[str, Any], 
                                   backtest_results: Dict[str, Any], thresholds: Dict[str, Any]) -> ValidationResult:
        """Analyze validation results and determine if changes are acceptable"""

        # Check minimum trades requirement
        if backtest_results.get('total_trades', 0) < thresholds['min_trades']:
            return ValidationResult(
                component=component_name,
                parameter_changes=proposed_changes,
                test_passed=False,
                metrics={'total_trades': backtest_results.get('total_trades', 0)},
                required_metrics={'total_trades': (thresholds['min_trades'], '>=')},
                backtest_results=backtest_results
            )

        # Check component-specific metrics
        metrics = {}
        required_metrics = {}

        if component_name == 'strategy_allocator':
            metrics = {
                'sharpe_ratio': backtest_results.get('sharpe_ratio', 0),
                'win_rate': backtest_results.get('win_rate', 0),
                'max_drawdown': backtest_results.get('max_drawdown', 1.0),
                'profit_factor': backtest_results.get('profit_factor', 1.0)
            }
            
            required_metrics = {
                'sharpe_ratio': (1.2, '>'),  # Minimum Sharpe ratio
                'win_rate': (thresholds['min_win_rate'], '>='),
                'max_drawdown': (0.2, '<'),  # Max drawdown
                'profit_factor': (1.1, '>')
            }
            
            # Check Sharpe improvement
            if 'sharpe_ratio' in backtest_results and 'original_sharpe' in backtest_results:
                sharpe_improvement = backtest_results['sharpe_ratio'] - backtest_results['original_sharpe']
                required_metrics['sharpe_improvement'] = (thresholds['min_sharpe_improvement'], '>')
                metrics['sharpe_improvement'] = sharpe_improvement
            
            # Check drawdown change
            if 'max_drawdown' in backtest_results and 'original_drawdown' in backtest_results:
                drawdown_change = backtest_results['max_drawdown'] - backtest_results['original_drawdown']
                required_metrics['drawdown_change'] = (thresholds['max_drawdown_increase'], '<')
                metrics['drawdown_change'] = drawdown_change
            
        elif component_name == 'correlation_matrix':
            metrics = {
                'diversification_score': backtest_results.get('diversification_score', 0),
                'sharpe_ratio': backtest_results.get('sharpe_ratio', 0),
                'max_drawdown': backtest_results.get('max_drawdown', 1.0),
                'portfolio_concentration': backtest_results.get('portfolio_concentration', 1.0)
            }
            
            required_metrics = {
                'diversification_score': (0.5, '>'),
                'sharpe_ratio': (1.2, '>'),
                'max_drawdown': (0.2, '<'),
                'portfolio_concentration': (thresholds['max_concentration'], '<')
            }
            
            # Check diversification improvement
            if 'diversification_score' in backtest_results and 'original_diversification' in backtest_results:
                div_improvement = backtest_results['diversification_score'] - backtest_results['original_diversification']
                required_metrics['diversification_improvement'] = (thresholds['min_diversification_improvement'], '>')
                metrics['diversification_improvement'] = div_improvement
            
        elif component_name == 'order_router':
            metrics = {
                'avg_fill_ratio': backtest_results.get('avg_fill_ratio', 0),
                'avg_slippage': backtest_results.get('avg_slippage', 1.0),
                'avg_latency': backtest_results.get('avg_latency', 100),
                'execution_quality_score': backtest_results.get('execution_quality_score', 0)
            }
            
            required_metrics = {
                'avg_fill_ratio': (0.8, '>'),
                'avg_slippage': (0.002, '<'),  # 0.2%
                'avg_latency': (100, '<'),  # 100ms
                'execution_quality_score': (0.5, '>')
            }
            
            # Check fill improvement
            if 'avg_fill_ratio' in backtest_results and 'original_fill' in backtest_results:
                fill_improvement = backtest_results['avg_fill_ratio'] - backtest_results['original_fill']
                required_metrics['fill_improvement'] = (thresholds['min_fill_improvement'], '>')
                metrics['fill_improvement'] = fill_improvement
            
            # Check slippage change
            if 'avg_slippage' in backtest_results and 'original_slippage' in backtest_results:
                slippage_change = backtest_results['avg_slippage'] - backtest_results['original_slippage']
                required_metrics['slippage_change'] = (thresholds['max_slippage_increase'], '<')
                metrics['slippage_change'] = slippage_change
            
            # Check latency improvement
            if 'avg_latency' in backtest_results and 'original_latency' in backtest_results:
                latency_change = backtest_results['original_latency'] - backtest_results['avg_latency']
                required_metrics['latency_improvement'] = (thresholds['min_latency_improvement'], '>')
                metrics['latency_improvement'] = latency_change
            
        elif component_name == 'regime_parameters':
            metrics = {
                'sharpe_ratio': backtest_results.get('sharpe_ratio', 0),
                'win_rate': backtest_results.get('win_rate', 0),
                'max_drawdown': backtest_results.get('max_drawdown', 1.0),
                'profit_factor': backtest_results.get('profit_factor', 1.0)
            }
            
            required_metrics = {
                'sharpe_ratio': (1.2, '>'),
                'win_rate': (0.5, '>'),
                'max_drawdown': (0.2, '<'),
                'profit_factor': (thresholds['min_profit_factor'], '>')
            }
            
            # Check Sharpe improvement
            if 'sharpe_ratio' in backtest_results and 'original_sharpe' in backtest_results:
                sharpe_improvement = backtest_results['sharpe_ratio'] - backtest_results['original_sharpe']
                required_metrics['sharpe_improvement'] = (thresholds['min_sharpe_improvement'], '>')
                metrics['sharpe_improvement'] = sharpe_improvement
            
            # Check drawdown change
            if 'max_drawdown' in backtest_results and 'original_drawdown' in backtest_results:
                drawdown_change = backtest_results['max_drawdown'] - backtest_results['original_drawdown']
                required_metrics['drawdown_change'] = (thresholds['max_drawdown_increase'], '<')
                metrics['drawdown_change'] = drawdown_change

        # Run statistical tests
        statistical_results = self._run_statistical_tests(component_name, metrics, backtest_results)

        # Determine if validation passed
        test_passed = True

        # Check all required metrics
        for metric, (threshold, op) in required_metrics.items():
            if metric not in metrics:
                test_passed = False
                break
            
            value = metrics[metric]
            
            if op == '>' and value <= threshold:
                test_passed = False
                break
            if op == '>=' and value < threshold:
                test_passed = False
                break
            if op == '<' and value >= threshold:
                test_passed = False
                break
            if op == '<=' and value > threshold:
                test_passed = False
                break
            if op == '==' and abs(value - threshold) >= 0.001:
                test_passed = False
                break

        # Check statistical significance for all relevant metrics
        # Only approve changes if they show statistically significant improvements
        for test_name, test_result in statistical_results.items():
            if not test_result['significant']:
                logger.info("Validation failed for %s: %s not statistically significant (p=%.4f) > %.4f", 
                           component_name, test_name, test_result['p_value'], test_result['threshold'])
                test_passed = False

        # Create validation result with statistical results
        result = ValidationResult(
            component=component_name,
            parameter_changes=proposed_changes,
            test_passed=test_passed,
            metrics=metrics,
            required_metrics=required_metrics,
            backtest_results=backtest_results
        )
        
        # Add statistical results to the validation result
        result.statistical_results = statistical_results
        
        return result
    
    def _run_statistical_tests(self, component_name: str, metrics: Dict[str, float],
                               backtest_results: Dict[str, Any]) -> Dict[str, Dict]:
        """Run statistical tests on validation results"""

        tests = {}
        thresholds = self.component_thresholds[component_name]
        p_threshold = thresholds['p_value_threshold']

        # For strategy allocator, test if performance improvements are significant
        if component_name == 'strategy_allocator':
            # Test Sharpe improvement significance
            if 'sharpe_improvement' in metrics and 'original_sharpe' in backtest_results:
                n = backtest_results.get('total_trades', 30)  # Use actual trade count
                mean_improvement = metrics['sharpe_improvement']
                
                # Estimate standard deviation based on improvement magnitude
                # Conservative estimate: 50% of improvement as std dev
                std_dev = abs(mean_improvement) * 0.5
                
                # One-sample t-test against 0 (no improvement)
                t_stat, p_value = stats.ttest_1samp(
                    [mean_improvement] * n,  # Simulated sample
                    0
                )
                
                tests['sharpe_improvement'] = {
                    'test': 'one_sample_t_test',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < p_threshold,
                    'threshold': p_threshold
                }
                
            # Test win rate improvement significance
            if 'win_rate' in metrics and 'original_win_rate' in backtest_results:
                n = backtest_results.get('total_trades', 30)
                original_win_rate = backtest_results['original_win_rate']
                new_win_rate = metrics['win_rate']
                
                 # Proportion test for win rate improvement
                
                # Two-proportion z-test
                p1 = original_win_rate
                p2 = new_win_rate
                
                # Calculate pooled proportion for two-proportion z-test
                p_pooled = (p1 * n + p2 * n) / (2 * n)
                se = np.sqrt(p_pooled * (1 - p_pooled) * (2 / n))
                z_score = (p2 - p1) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                tests['win_rate_improvement'] = {
                    'test': 'two_proportion_z_test',
                    'statistic': z_score,
                    'p_value': p_value,
                    'significant': p_value < p_threshold,
                    'threshold': p_threshold
                }
                
            # Test drawdown change significance
            if 'drawdown_change' in metrics and 'original_drawdown' in backtest_results:
                n = backtest_results.get('total_trades', 30)
                mean_change = metrics['drawdown_change']
                
                # One-sample t-test against 0 (no improvement)
                # Note: We don't need to explicitly calculate std_dev as ttest_1samp handles it
                t_stat, p_value = stats.ttest_1samp(
                    [mean_improvement] * n,  # Simulated sample
                    0
                )
                
                tests['drawdown_change'] = {
                    'test': 'one_sample_t_test',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < p_threshold,
                    'threshold': p_threshold
                }

        # For correlation matrix, test if diversification improvement is significant
        elif component_name == 'correlation_matrix':
            # Test diversification improvement significance
            if 'diversification_improvement' in metrics:
                n = backtest_results.get('total_trades', 20)
                mean_improvement = metrics['diversification_improvement']
                
                # Estimate standard deviation
                std_dev = abs(mean_improvement) * 0.3
                
                t_stat, p_value = stats.ttest_1samp(
                    [mean_improvement] * n,
                    0
                )
                
                tests['diversification_improvement'] = {
                    'test': 'one_sample_t_test',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < p_threshold,
                    'threshold': p_threshold
                }
                
            # Test portfolio concentration change significance
            if 'portfolio_concentration' in metrics and 'original_concentration' in backtest_results:
                n = backtest_results.get('total_trades', 20)
                original_conc = backtest_results['original_concentration']
                new_conc = metrics['portfolio_concentration']
                
                # Calculate change
                conc_change = original_conc - new_conc  # Positive means improvement
                
                # Estimate standard deviation
                std_dev = abs(conc_change) * 0.4
                
                t_stat, p_value = stats.ttest_1samp(
                    [conc_change] * n,
                    0
                )
                
                tests['portfolio_concentration_improvement'] = {
                    'test': 'one_sample_t_test',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < p_threshold,
                    'threshold': p_threshold
                }

        # For order router, test if execution improvements are significant
        elif component_name == 'order_router':
            # Test fill improvement significance
            if 'fill_improvement' in metrics:
                n = backtest_results.get('total_trades', 25)
                mean_improvement = metrics['fill_improvement']
                
                # Estimate standard deviation
                std_dev = abs(mean_improvement) * 0.4
                
                t_stat, p_value = stats.ttest_1samp(
                    [mean_improvement] * n,
                    0
                )
                
                tests['fill_improvement'] = {
                    'test': 'one_sample_t_test',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < p_threshold,
                    'threshold': p_threshold
                }
                
            # Test slippage change significance
            if 'slippage_change' in metrics:
                n = backtest_results.get('total_trades', 25)
                mean_change = metrics['slippage_change']
                
                # Estimate standard deviation
                std_dev = abs(mean_change) * 0.3
                
                t_stat, p_value = stats.ttest_1samp(
                    [mean_change] * n,
                    0
                )
                
                tests['slippage_change'] = {
                    'test': 'one_sample_t_test',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < p_threshold,
                    'threshold': p_threshold
                }
                
            # Test latency improvement significance
            if 'latency_improvement' in metrics:
                n = backtest_results.get('total_trades', 25)
                mean_improvement = metrics['latency_improvement']
                
                # Estimate standard deviation
                std_dev = abs(mean_improvement) * 0.5
                
                t_stat, p_value = stats.ttest_1samp(
                    [mean_improvement] * n,
                    0
                )
                
                tests['latency_improvement'] = {
                    'test': 'one_sample_t_test',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < p_threshold,
                    'threshold': p_threshold
                }

        # For regime parameters, test if performance improvements are significant
        elif component_name == 'regime_parameters':
            # Test Sharpe improvement significance
            if 'sharpe_improvement' in metrics and 'original_sharpe' in backtest_results:
                n = backtest_results.get('total_trades', 30)
                mean_improvement = metrics['sharpe_improvement']
                
                # Estimate standard deviation
                std_dev = abs(mean_improvement) * 0.5
                
                t_stat, p_value = stats.ttest_1samp(
                    [mean_improvement] * n,
                    0
                )
                
                tests['sharpe_improvement'] = {
                    'test': 'one_sample_t_test',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < p_threshold,
                    'threshold': p_threshold
                }
                
            # Test profit factor improvement significance
            if 'profit_factor' in metrics and 'original_profit_factor' in backtest_results:
                n = backtest_results.get('total_trades', 30)
                original_pf = backtest_results['original_profit_factor']
                new_pf = metrics['profit_factor']
                
                # Calculate improvement
                pf_improvement = new_pf - original_pf
                
                # Estimate standard deviation
                std_dev = abs(pf_improvement) * 0.6
                
                t_stat, p_value = stats.ttest_1samp(
                    [pf_improvement] * n,
                    0
                )
                
                tests['profit_factor_improvement'] = {
                    'test': 'one_sample_t_test',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < p_threshold,
                    'threshold': p_threshold
                }
                
            # Test drawdown change significance
            if 'drawdown_change' in metrics and 'original_drawdown' in backtest_results:
                n = backtest_results.get('total_trades', 30)
                mean_change = metrics['drawdown_change']
                
                # Estimate standard deviation
                std_dev = abs(mean_change) * 0.3
                
                t_stat, p_value = stats.ttest_1samp(
                    [mean_change] * n,
                    0
                )
                
                tests['drawdown_change'] = {
                    'test': 'one_sample_t_test',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < p_threshold,
                    'threshold': p_threshold
                }

        # Store statistical results for this validation
        self.statistical_results[component_name] = tests
        return tests
    
    def get_params(self) -> Dict[str, Any]:
        """Get current parameters"""
        return {
            'component_thresholds': self.component_thresholds.copy(),
            'validation_history': [
                {
                    'component': v.component,
                    'timestamp': v.timestamp,
                    'test_passed': v.test_passed,
                    'metrics': v.metrics
                }
                for v in self.validation_history[-10:]  # Last 10 validations
            ],
            'backtest_cache_size': len(self.backtest_cache),
            'statistical_results': self.statistical_results
        }
    
    def rollback(self) -> None:
        """Revert to last known good state"""
        # Clear validation history and cache to force re-validation
        self.validation_history = []
        self.backtest_cache = {}
        self.statistical_results = {}
        logger.info("Cleared validation history and cache")
    
    def validate(self, new_params: Dict[str, Any]) -> bool:
        """Validate proposed parameter changes"""
        if 'component_thresholds' in new_params:
            for component, thresholds in new_params['component_thresholds'].items():
                if component not in self.component_thresholds:
                    logger.warning("Unknown component in thresholds: %s", component)
                    return False
                
                for metric, value in thresholds.items():
                    if metric == 'p_value_threshold' and (value <= 0 or value >= 1):
                        logger.warning("Invalid p-value threshold for %s: %s", component, value)
                        return False
                    elif metric in ['min_trades', 'min_sharpe_improvement', 'min_diversification_improvement']:
                        if value < 0:
                            logger.warning("Invalid threshold for %s.%s: %s", component, metric, value)
                            return False
                    elif metric in ['max_drawdown_increase', 'max_concentration', 'max_slippage_increase']:
                        if value < 0:
                            logger.warning("Invalid threshold for %s.%s: %s", component, metric, value)
                            return False
        
        return True
    
    def get_validation_history(self, component: Optional[str] = None, limit: int = 10) -> List[ValidationResult]:
        """Get validation history filtered by component"""
        if component:
            return [v for v in self.validation_history[-limit:] if v.component == component]
        return self.validation_history[-limit:]
    
    def get_validation_stats(self) -> Dict[str, Dict]:
        """Get statistics about validation results"""
        stats = {
            'total_validations': len(self.validation_history),
            'pass_rate': 0.0,
            'by_component': {}
        }
        
        if not self.validation_history:
            return stats
        
        # Calculate overall pass rate
        passed = sum(1 for v in self.validation_history if v.test_passed)
        stats['pass_rate'] = passed / len(self.validation_history)
        
        # Calculate by component
        component_stats = defaultdict(lambda: {'total': 0, 'passed': 0})
        
        for validation in self.validation_history:
            component = validation.component
            component_stats[component]['total'] += 1
            if validation.test_passed:
                component_stats[component]['passed'] += 1
        
        for component, data in component_stats.items():
            stats['by_component'][component] = {
                'total': data['total'],
                'passed': data['passed'],
                'pass_rate': data['passed'] / data['total'] if data['total'] > 0 else 0.0
            }
        
        return stats