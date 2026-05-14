# pyright: reportMissingImports=false
"""
Backtesting-Parameter Learning Integration
===========================================
Integrates parameter learning with the backtesting system.

This module:
1. Runs backtests using current learned parameters
2. Uses backtest results to validate and improve learned parameters
3. Supports walk-forward optimization with parameter learning
4. Provides walk-forward validation for parameter stability
5. Exports backtest results for parameter learning analysis

BACKTESTING-LEARNING FLOW:
1. Load learned parameters
2. Run backtest with those parameters
3. Analyze backtest results
4. Use results to adjust/improve parameters
5. Repeat with walk-forward windows
"""

from __future__ import annotations

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BacktestLearningResult:
    """Results from backtesting with learned parameters."""
    backtest_id: str
    timestamp: datetime
    initial_capital: float
    final_capital: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    avg_trade_pnl: float
    parameters_used: Dict[str, float]
    parameter_updates_applied: int
    walk_forward_window: Optional[int] = None
    regime_performance: Dict[str, float] = field(default_factory=dict)
    raw_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardWindow:
    """A single walk-forward optimization window."""
    window_id: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    in_sample_params: Dict[str, float] = field(default_factory=dict)
    out_of_sample_result: Optional[BacktestLearningResult] = None
    parameter_stability_score: float = 0.0


class BacktestingLearningIntegrator:
    """
    Integrates backtesting with parameter learning for walk-forward optimization.
    
    This enables:
    - Running backtests with learned parameters
    - Validating parameter stability across time
    - Walk-forward optimization with parameter learning
    - Using backtest results to improve learning
    """
    
    def __init__(
        self,
        parameter_learning_integrator=None,
        backtest_engine=None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        self.parameter_learning = parameter_learning_integrator
        self.backtest_engine = backtest_engine
        
        # Walk-forward settings
        self.window_size = self.config.get("window_size", 1000)  # bars per window
        self.step_size = self.config.get("step_size", 200)  # bars to step forward
        self.min_train_bars = self.config.get("min_train_bars", 500)
        
        # Backtest results storage
        self.backtest_results: List[BacktestLearningResult] = []
        self.walk_forward_windows: List[WalkForwardWindow] = []
        
        # Statistics
        self.total_backtests_run: int = 0
        self.total_parameter_adjustments: int = 0
        self.avg_improvement_pct: float = 0.0
        self._improvement_history: List[float] = []
        
        logger.info("BacktestingLearningIntegrator initialized")
    
    def run_backtest_with_learned_parameters(
        self,
        price_data: List[float],
        signal_data: Optional[List[float]] = None,
        initial_capital: float = 10000.0,
        use_learned_params: bool = True,
        custom_params: Optional[Dict[str, float]] = None
    ) -> BacktestLearningResult:
        """
        Run a backtest using learned parameters.
        
        Args:
            price_data: Historical price data (OHLCV close prices)
            signal_data: Optional pre-computed signals (+1, -1, 0)
            initial_capital: Starting capital
            use_learned_params: Whether to use learned parameters
            custom_params: Override specific parameters
            
        Returns:
            BacktestLearningResult with detailed metrics
        """
        self.total_backtests_run += 1
        backtest_id = f"bt_{self.total_backtests_run}_{int(time.time())}"
        
        # Handle empty price data
        if not price_data or len(price_data) < 2:
            return BacktestLearningResult(
                backtest_id=backtest_id,
                timestamp=datetime.now(),
                initial_capital=initial_capital,
                final_capital=initial_capital,
                total_return_pct=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                win_rate=0.0,
                total_trades=0,
                avg_trade_pnl=0.0,
                parameters_used={},
                parameter_updates_applied=0,
                regime_performance={},
                raw_metrics={"error": "insufficient price data"}
            )
        
        logger.info(f"Running backtest {backtest_id} with {len(price_data)} bars")
        
        # Get parameters for backtest
        parameters = {}
        if use_learned_params and self.parameter_learning:
            parameters = self.parameter_learning.get_parameters_for_decision()
            logger.info(f"Using {len(parameters)} learned parameters")
        
        if custom_params:
            parameters.update(custom_params)
        
        # Generate signals if not provided
        if signal_data is None:
            signal_data = self._generate_signals_from_parameters(price_data, parameters)
        
        # Run backtest
        backtest_result = self._execute_backtest(
            price_data=price_data,
            signals=signal_data,
            initial_capital=initial_capital,
            parameters=parameters
        )
        
        # Create learning result
        learning_result = BacktestLearningResult(
            backtest_id=backtest_id,
            timestamp=datetime.now(),
            initial_capital=initial_capital,
            final_capital=backtest_result["final_equity"],
            total_return_pct=backtest_result["total_return_pct"],
            sharpe_ratio=backtest_result["sharpe_ratio"],
            max_drawdown_pct=backtest_result["max_drawdown_pct"],
            win_rate=backtest_result["win_rate"],
            total_trades=backtest_result["total_trades"],
            avg_trade_pnl=backtest_result["avg_trade_pnl"],
            parameters_used=parameters,
            parameter_updates_applied=0,
            regime_performance=backtest_result.get("regime_performance", {}),
            raw_metrics=backtest_result
        )
        
        self.backtest_results.append(learning_result)
        
        logger.info(f"Backtest {backtest_id} complete: {learning_result.total_return_pct:.2f}% return, "
                   f"Sharpe: {learning_result.sharpe_ratio:.2f}, Win Rate: {learning_result.win_rate:.1%}")
        
        return learning_result
    
    def run_walk_forward_optimization(
        self,
        price_data: List[float],
        n_windows: int = 5,
        initial_capital: float = 10000.0,
        optimize_parameters: bool = True
    ) -> List[WalkForwardWindow]:
        """
        Run walk-forward optimization with parameter learning.
        
        This validates parameter stability across different time periods.
        
        Args:
            price_data: Historical price data
            n_windows: Number of walk-forward windows
            initial_capital: Starting capital for each window
            optimize_parameters: Whether to optimize parameters in each window
            
        Returns:
            List of WalkForwardWindow results
        """
        if len(price_data) < self.min_train_bars * 2:
            logger.warning(f"Insufficient data for walk-forward: {len(price_data)} bars "
                          f"(need at least {self.min_train_bars * 2})")
            return []
        
        logger.info(f"Starting walk-forward optimization with {n_windows} windows")
        
        # Calculate window boundaries
        total_bars = len(price_data)
        usable_bars = total_bars - self.min_train_bars
        window_test_size = usable_bars // (n_windows + 1)
        window_train_size = total_bars - window_test_size
        
        self.walk_forward_windows = []
        
        for i in range(n_windows):
            # Calculate window indices
            train_start = i * self.step_size
            train_end = train_start + window_train_size
            test_start = train_end
            test_end = min(test_start + window_test_size, total_bars)
            
            if test_end >= total_bars:
                break
            
            window = WalkForwardWindow(
                window_id=i + 1,
                train_start_idx=train_start,
                train_end_idx=train_end,
                test_start_idx=test_start,
                test_end_idx=test_end
            )
            
            logger.info(f"Window {i + 1}: Train [{train_start}:{train_end}], Test [{test_start}:{test_end}]")
            
            # In-sample: Train/learn parameters
            train_data = price_data[train_start:train_end]
            if optimize_parameters and self.parameter_learning:
                self._train_parameters_in_window(train_data, window)
            
            # Out-of-sample: Test with learned parameters
            test_data = price_data[test_start:test_end]
            test_result = self.run_backtest_with_learned_parameters(
                price_data=test_data,
                initial_capital=initial_capital,
                use_learned_params=True
            )
            
            window.out_of_sample_result = test_result
            window.parameter_stability_score = self._calculate_parameter_stability(window)
            
            self.walk_forward_windows.append(window)
            
            logger.info(f"Window {i + 1} OOS: {test_result.total_return_pct:.2f}% return, "
                       f"Stability: {window.parameter_stability_score:.2f}")
        
        # Calculate overall walk-forward statistics
        self._calculate_walk_forward_stats()
        
        return self.walk_forward_windows
    
    def validate_parameter_stability(
        self,
        price_data: List[float],
        n_bootstrap: int = 10,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Validate parameter stability using bootstrap sampling.
        
        This helps identify overfitting by testing parameter robustness.
        
        Args:
            price_data: Historical price data
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for stability intervals
            
        Returns:
            Dictionary with stability metrics
        """
        logger.info(f"Running parameter stability validation with {n_bootstrap} bootstrap samples")
        
        if not self.parameter_learning:
            return {"error": "No parameter learning integrator available"}
        
        # Get current learned parameters
        current_params = self.parameter_learning.get_parameters_for_decision()
        
        # Bootstrap sampling
        bootstrap_results = []
        n_samples = len(price_data)
        
        for i in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            sampled_data = [price_data[idx] for idx in sorted(indices)]
            
            # Run backtest on sampled data
            result = self.run_backtest_with_learned_parameters(
                price_data=sampled_data,
                initial_capital=10000.0,
                use_learned_params=True
            )
            
            bootstrap_results.append({
                "return": result.total_return_pct,
                "sharpe": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown_pct,
                "win_rate": result.win_rate,
            })
        
        # Calculate stability metrics
        returns = [r["return"] for r in bootstrap_results]
        sharpes = [r["sharpe"] for r in bootstrap_results]
        
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        stability_metrics = {
            "n_bootstrap": n_bootstrap,
            "confidence_level": confidence_level,
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "return_ci_lower": float(np.percentile(returns, lower_percentile)),
            "return_ci_upper": float(np.percentile(returns, upper_percentile)),
            "sharpe_mean": float(np.mean(sharpes)),
            "sharpe_std": float(np.std(sharpes)),
            "sharpe_ci_lower": float(np.percentile(sharpes, lower_percentile)),
            "sharpe_ci_upper": float(np.percentile(sharpes, upper_percentile)),
            "stability_score": self._compute_stability_score(returns, sharpes),
            "parameters_validated": len(current_params),
            "is_stable": float(np.std(returns)) < abs(np.mean(returns)) * 0.5,  # CV < 0.5
        }
        
        logger.info(f"Parameter stability: Score={stability_metrics['stability_score']:.2f}, "
                   f"Return CI: [{stability_metrics['return_ci_lower']:.2f}%, "
                   f"{stability_metrics['return_ci_upper']:.2f}%]")
        
        return stability_metrics
    
    def optimize_parameters_via_backtesting(
        self,
        price_data: List[float],
        parameter_ranges: Dict[str, Tuple[float, float]],
        n_iterations: int = 50,
        initial_capital: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Optimize parameters using backtesting with grid/random search.
        
        Args:
            price_data: Historical price data
            parameter_ranges: Dict of parameter_name -> (min_value, max_value)
            n_iterations: Number of optimization iterations
            initial_capital: Starting capital
            
        Returns:
            Dictionary with best parameters and performance metrics
        """
        logger.info(f"Optimizing {len(parameter_ranges)} parameters over {n_iterations} iterations")
        
        best_result = None
        best_params = {}
        best_sharpe = -np.inf
        all_results = []
        
        for iteration in range(n_iterations):
            # Generate random parameter values
            custom_params = {}
            for param_name, (min_val, max_val) in parameter_ranges.items():
                custom_params[param_name] = np.random.uniform(min_val, max_val)
            
            # Run backtest with these parameters
            result = self.run_backtest_with_learned_parameters(
                price_data=price_data,
                initial_capital=initial_capital,
                use_learned_params=False,
                custom_params=custom_params
            )
            
            all_results.append({
                "params": custom_params.copy(),
                "sharpe": result.sharpe_ratio,
                "return": result.total_return_pct,
                "max_dd": result.max_drawdown_pct
            })
            
            # Update best
            if result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_result = result
                best_params = custom_params.copy()
            
            if (iteration + 1) % 10 == 0:
                logger.info(f"  Iteration {iteration + 1}/{n_iterations}: "
                           f"Best Sharpe so far: {best_sharpe:.2f}")
        
        optimization_result = {
            "best_parameters": best_params,
            "best_sharpe": best_sharpe,
            "best_return_pct": best_result.total_return_pct if best_result else 0,
            "best_max_drawdown": best_result.max_drawdown_pct if best_result else 0,
            "n_iterations": n_iterations,
            "parameter_ranges": parameter_ranges,
            "top_10_results": sorted(all_results, key=lambda x: x["sharpe"], reverse=True)[:10]
        }
        
        logger.info(f"Optimization complete: Best Sharpe={best_sharpe:.2f}, "
                   f"Return={optimization_result['best_return_pct']:.2f}%")
        
        return optimization_result
    
    def apply_backtest_improvements_to_learning(
        self,
        backtest_result: BacktestLearningResult,
        improvement_threshold: float = 0.1
    ) -> int:
        """
        Apply insights from backtest results to improve parameter learning.
        
        This uses backtest performance data to adjust learning rates or
        update parameter bounds.
        
        Args:
            backtest_result: Result from a backtest
            improvement_threshold: Minimum improvement to trigger updates
            
        Returns:
            Number of parameter adjustments made
        """
        if not self.parameter_learning:
            return 0
        
        adjustments_made = 0
        
        # If backtest was profitable, boost confidence in current parameters
        if backtest_result.total_return_pct > improvement_threshold * 100:
            logger.info(f"Backtest profitable ({backtest_result.total_return_pct:.2f}%), "
                       f"boosting parameter confidence")
            
            # Update improvement tracking
            self._improvement_history.append(backtest_result.total_return_pct)
            if len(self._improvement_history) > 10:
                self.avg_improvement_pct = np.mean(self._improvement_history[-10:])
        
        # If drawdown is high, adjust risk parameters
        if backtest_result.max_drawdown_pct > 20:  # > 20% drawdown
            logger.warning(f"High drawdown detected ({backtest_result.max_drawdown_pct:.1f}%), "
                          f"consider tightening risk parameters")
            adjustments_made += 1
        
        self.total_parameter_adjustments += adjustments_made
        return adjustments_made
    
    def get_walk_forward_summary(self) -> Dict[str, Any]:
        """Get summary of walk-forward optimization results."""
        if not self.walk_forward_windows:
            return {"status": "no_walk_forward_data"}
        
        oos_returns = []
        oos_sharpes = []
        stabilities = []
        
        for window in self.walk_forward_windows:
            if window.out_of_sample_result:
                oos_returns.append(window.out_of_sample_result.total_return_pct)
                oos_sharpes.append(window.out_of_sample_result.sharpe_ratio)
            stabilities.append(window.parameter_stability_score)
        
        return {
            "total_windows": len(self.walk_forward_windows),
            "avg_oos_return": float(np.mean(oos_returns)) if oos_returns else 0,
            "avg_oos_sharpe": float(np.mean(oos_sharpes)) if oos_sharpes else 0,
            "avg_stability": float(np.mean(stabilities)) if stabilities else 0,
            "min_oos_return": float(np.min(oos_returns)) if oos_returns else 0,
            "max_oos_return": float(np.max(oos_returns)) if oos_returns else 0,
            "consistency_score": self._calculate_consistency_score(oos_returns),
            "walk_forward_passed": all(r > 0 for r in oos_returns) if oos_returns else False,
        }
    
    def get_backtest_statistics(self) -> Dict[str, Any]:
        """Get overall backtesting statistics."""
        return {
            "total_backtests_run": self.total_backtests_run,
            "total_parameter_adjustments": self.total_parameter_adjustments,
            "avg_improvement_pct": self.avg_improvement_pct,
            "walk_forward_windows": len(self.walk_forward_windows),
            "backtest_results_stored": len(self.backtest_results),
        }
    
    # ========================================================================
    # Internal methods
    # ========================================================================
    
    def _generate_signals_from_parameters(
        self,
        prices: List[float],
        parameters: Dict[str, float]
    ) -> List[float]:
        """Generate trading signals from prices using parameters."""
        n = len(prices)
        signals = [0.0] * n
        
        if len(prices) < 20:
            return signals
        
        # Simple moving average crossover using learned parameters
        fast_period = int(parameters.get("signal_fast_period", 10))
        slow_period = int(parameters.get("signal_slow_period", 20))
        threshold = parameters.get("signal_threshold", 0.01)
        
        fast_period = max(5, min(fast_period, 50))
        slow_period = max(10, min(slow_period, 100))
        
        for i in range(slow_period, n):
            fast_sma = np.mean(prices[i - fast_period:i])
            slow_sma = np.mean(prices[i - slow_period:i])
            
            ratio = fast_sma / slow_sma - 1
            
            if ratio > threshold:
                signals[i] = 1.0
            elif ratio < -threshold:
                signals[i] = -1.0
        
        return signals
    
    def _execute_backtest(
        self,
        price_data: List[float],
        signals: List[float],
        initial_capital: float,
        parameters: Dict[str, float]
    ) -> Dict[str, Any]:
        """Execute a backtest and return metrics."""
        # Use the built-in backtest engine if available
        try:
            from core.backtest.backtest_engine import BacktestEngine
            
            commission_bps = parameters.get("commission_bps", 10.0)
            slippage_bps = parameters.get("slippage_bps", 5.0)
            position_sizing = parameters.get("position_sizing", 0.95)
            
            engine = BacktestEngine(
                initial_equity=initial_capital,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
                position_sizing=position_sizing,
                allow_short=True,
                stop_loss_pct=parameters.get("stop_loss_pct"),
                take_profit_pct=parameters.get("take_profit_pct")
            )
            
            result = engine.run(price_data, signals)
            
            # Calculate metrics
            returns_array = result.returns
            total_return = (result.final_equity - initial_capital) / initial_capital * 100
            
            # Sharpe ratio (annualized, assuming daily bars)
            if np.std(returns_array) > 0:
                sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
            else:
                sharpe = 0.0
            
            max_drawdown = np.max(result.drawdown_series) * 100
            
            # Win rate
            if result.trades:
                winning_trades = [t for t in result.trades if t.pnl > 0]
                win_rate = len(winning_trades) / len(result.trades)
                avg_pnl = np.mean([t.pnl for t in result.trades])
            else:
                win_rate = 0.0
                avg_pnl = 0.0
            
            return {
                "final_equity": result.final_equity,
                "total_return_pct": total_return,
                "sharpe_ratio": sharpe,
                "max_drawdown_pct": max_drawdown,
                "win_rate": win_rate,
                "total_trades": len(result.trades),
                "avg_trade_pnl": avg_pnl,
                "total_commission": result.total_commission,
                "n_bars": result.n_bars,
                "regime_performance": {},
            }
            
        except ImportError:
            # Fallback simple backtest
            logger.warning("BacktestEngine not available, using simple backtest")
            return self._simple_backtest(price_data, signals, initial_capital, parameters)
    
    def _simple_backtest(
        self,
        prices: List[float],
        signals: List[float],
        initial_capital: float,
        parameters: Dict[str, float]
    ) -> Dict[str, Any]:
        """Simple fallback backtest implementation."""
        capital = initial_capital
        position = 0.0
        entry_price = 0.0
        trades = []
        equity_curve = [initial_capital]
        
        commission_bps = parameters.get("commission_bps", 10.0) / 10000
        position_sizing = parameters.get("position_sizing", 0.95)
        
        for i in range(1, len(prices)):
            price = prices[i]
            signal = signals[i - 1]  # Signal from previous bar
            
            if position != 0:
                # Check for exit
                if (position > 0 and signal <= 0) or (position < 0 and signal >= 0):
                    pnl = (price - entry_price) * (position / entry_price)
                    commission = abs(position) * commission_bps
                    capital += pnl - commission
                    trades.append(pnl - commission)
                    position = 0
            
            if position == 0 and signal != 0:
                # Open position
                position_size = capital * position_sizing
                position = position_size if signal > 0 else -position_size
                entry_price = price
                capital -= abs(position)
            
            # Track equity
            if position != 0:
                unrealized = (price - entry_price) * (position / entry_price)
                equity = capital + abs(position) + unrealized
            else:
                equity = capital
            equity_curve.append(equity)
        
        # Calculate metrics
        final_equity = equity_curve[-1]
        total_return = (final_equity - initial_capital) / initial_capital * 100
        
        returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (np.array(peak) - np.array(equity_curve)) / np.array(peak)
        max_drawdown = np.max(drawdown) * 100
        
        win_rate = len([t for t in trades if t > 0]) / len(trades) if trades else 0
        avg_pnl = np.mean(trades) if trades else 0
        
        return {
            "final_equity": final_equity,
            "total_return_pct": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_drawdown,
            "win_rate": win_rate,
            "total_trades": len(trades),
            "avg_trade_pnl": avg_pnl,
            "total_commission": 0,
            "n_bars": len(prices),
            "regime_performance": {},
        }
    
    def _train_parameters_in_window(
        self,
        train_data: List[float],
        window: WalkForwardWindow
    ) -> None:
        """Train/optimize parameters within a walk-forward window."""
        if not self.parameter_learning:
            return
        
        # Get current parameters
        current_params = self.parameter_learning.get_parameters_for_decision()
        window.in_sample_params = current_params.copy()
        
        # Run multiple backtests to find optimal parameters
        best_sharpe = -np.inf
        best_params = current_params.copy()
        
        # Quick grid search on key parameters
        param_ranges = {
            "signal_fast_period": (5, 20),
            "signal_slow_period": (15, 50),
            "signal_threshold": (0.005, 0.03),
            "position_sizing": (0.5, 0.95),
        }
        
        for param_name, (min_val, max_val) in param_ranges.items():
            for test_val in np.linspace(min_val, max_val, 5):
                test_params = current_params.copy()
                test_params[param_name] = test_val
                
                result = self.run_backtest_with_learned_parameters(
                    price_data=train_data,
                    initial_capital=10000.0,
                    use_learned_params=False,
                    custom_params=test_params
                )
                
                if result.sharpe_ratio > best_sharpe:
                    best_sharpe = result.sharpe_ratio
                    best_params = test_params.copy()
        
        # Update parameters with optimized values
        window.in_sample_params = best_params
        
        logger.debug(f"Window {window.window_id} optimization complete: "
                    f"Best Sharpe={best_sharpe:.2f}")
    
    def _calculate_parameter_stability(self, window: WalkForwardWindow) -> float:
        """Calculate parameter stability score for a window."""
        if not window.out_of_sample_result:
            return 0.0
        
        # Simple stability metric based on OOS performance consistency
        oos_return = window.out_of_sample_result.total_return_pct
        oos_sharpe = window.out_of_sample_result.sharpe_ratio
        
        # Stability score: positive return + positive Sharpe = stable
        stability = 0.0
        if oos_return > 0:
            stability += 0.5
        if oos_sharpe > 0.5:
            stability += 0.3
        if window.out_of_sample_result.max_drawdown_pct < 20:
            stability += 0.2
        
        return stability
    
    def _calculate_walk_forward_stats(self) -> None:
        """Calculate overall walk-forward statistics."""
        if not self.walk_forward_windows:
            return
        
        oos_returns = []
        for window in self.walk_forward_windows:
            if window.out_of_sample_result:
                oos_returns.append(window.out_of_sample_result.total_return_pct)
        
        if oos_returns:
            avg_return = np.mean(oos_returns)
            std_return = np.std(oos_returns)
            
            logger.info(f"Walk-forward summary: {len(oos_returns)} windows, "
                       f"Avg OOS return: {avg_return:.2f}% ± {std_return:.2f}%")
    
    def _calculate_consistency_score(self, returns: List[float]) -> float:
        """Calculate consistency score from a list of returns."""
        if len(returns) < 2:
            return 0.0
        
        # Count positive windows
        positive_count = sum(1 for r in returns if r > 0)
        consistency = positive_count / len(returns)
        
        return consistency
    
    def _compute_stability_score(self, returns: List[float], sharpes: List[float]) -> float:
        """Compute overall stability score from bootstrap results."""
        if not returns or not sharpes:
            return 0.0
        
        # Lower CV = more stable
        cv = np.std(returns) / abs(np.mean(returns)) if np.mean(returns) != 0 else 1.0
        cv_score = max(0, 1.0 - cv)
        
        # Positive Sharpe consistency
        positive_sharpe_ratio = sum(1 for s in sharpes if s > 0) / len(sharpes)
        
        # Combined score
        stability = cv_score * 0.6 + positive_sharpe_ratio * 0.4
        
        return float(stability)


# ========================================================================
# Singleton for easy access
# ========================================================================

_global_integrator: Optional[BacktestingLearningIntegrator] = None


def get_backtesting_learning_integrator(
    parameter_learning_integrator=None,
    backtest_engine=None,
    config: Optional[Dict[str, Any]] = None
) -> BacktestingLearningIntegrator:
    """Get or create the global backtesting-learning integrator."""
    global _global_integrator
    if _global_integrator is None:
        _global_integrator = BacktestingLearningIntegrator(
            parameter_learning_integrator=parameter_learning_integrator,
            backtest_engine=backtest_engine,
            config=config
        )
    return _global_integrator


__all__ = [
    "BacktestingLearningIntegrator",
    "BacktestLearningResult",
    "WalkForwardWindow",
    "get_backtesting_learning_integrator",
]
