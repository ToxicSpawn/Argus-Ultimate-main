"""
Walk-Forward Validation Framework
==================================
Validates trading strategies using walk-forward analysis.

Why this is CRITICAL:
- Prevents overfitting by testing on out-of-sample data
- Proves strategy profitability before risking real money
- Identifies strategy degradation over time
- Provides confidence intervals for expected returns

Method:
1. Split historical data into rolling train/validation windows
2. Train strategy on in-sample data
3. Test on out-of-sample data
4. Roll forward and repeat
5. Aggregate results across all windows

Success criteria:
- Sharpe ratio > 1.0 in validation (not just training)
- >70% of validation windows profitable
- Max drawdown < 20%
- Out-of-sample performance within 20% of in-sample (no overfitting)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for walk-forward validation."""
    
    # Window sizes
    train_days: int = 180      # 6 months training
    validation_days: int = 30  # 1 month validation
    step_days: int = 30        # Step forward by 1 month
    
    # Minimum requirements
    min_train_samples: int = 1000  # Minimum bars in training
    min_validation_samples: int = 200  # Minimum bars in validation
    
    # Success criteria
    min_sharpe: float = 1.0
    min_win_rate: float = 0.45
    max_drawdown: float = 0.20  # 20%
    max_oos_degradation: float = 0.20  # OOS within 20% of IS
    
    # Risk-free rate for Sharpe calculation
    risk_free_rate: float = 0.05  # 5% annual


@dataclass
class TradeResult:
    """Result of a single trade."""
    timestamp: float
    action: str  # "buy" or "sell"
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    duration_hours: float
    regime: str


@dataclass
class WindowResult:
    """Result of a single validation window."""
    window_index: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    
    # Training metrics
    train_trades: int
    train_win_rate: float
    train_sharpe: float
    train_sortino: float
    train_max_drawdown: float
    train_total_return: float
    
    # Validation metrics
    val_trades: int
    val_win_rate: float
    val_sharpe: float
    val_sortino: float
    val_max_drawdown: float
    val_total_return: float
    
    # Comparison
    is_profitable: bool
    oos_degradation: float  # How much worse OOS vs IS


@dataclass
class ValidationSummary:
    """Summary of all validation windows."""
    total_windows: int
    profitable_windows: int
    profitability_rate: float
    
    # Aggregate metrics
    avg_train_sharpe: float
    avg_val_sharpe: float
    avg_train_win_rate: float
    avg_val_win_rate: float
    avg_max_drawdown: float
    
    # Success determination
    passed_min_sharpe: bool
    passed_profitability_rate: bool
    passed_drawdown: bool
    passed_degradation: bool
    overall_passed: bool
    
    # Detailed window results
    window_results: List[WindowResult]


class PerformanceMetrics:
    """Calculate trading performance metrics."""
    
    @staticmethod
    def calculate_sharpe(
        returns: List[float],
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252 * 24,  # Hourly data for crypto
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / periods_per_year)
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
        return float(sharpe)
    
    @staticmethod
    def calculate_sortino(
        returns: List[float],
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252 * 24,
    ) -> float:
        """Calculate Sortino ratio (only penalizes downside volatility)."""
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / periods_per_year)
        
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
        
        sortino = np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)
        return float(sortino)
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: List[float]) -> float:
        """Calculate maximum drawdown."""
        if len(equity_curve) < 2:
            return 0.0
        
        equity = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        
        return float(abs(np.min(drawdowns)))
    
    @staticmethod
    def calculate_win_rate(trades: List[TradeResult]) -> float:
        """Calculate win rate."""
        if not trades:
            return 0.0
        
        wins = sum(1 for t in trades if t.pnl > 0)
        return wins / len(trades)
    
    @staticmethod
    def calculate_total_return(trades: List[TradeResult], initial_capital: float) -> float:
        """Calculate total return."""
        if not trades:
            return 0.0
        
        total_pnl = sum(t.pnl for t in trades)
        return total_pnl / initial_capital


class StrategySimulator:
    """Simulates strategy execution for validation."""
    
    def __init__(self, initial_capital: float = 10000.0, fee_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
    
    def simulate(
        self,
        prices: List[float],
        signals: List[Dict[str, Any]],
        regimes: Optional[List[str]] = None,
    ) -> Tuple[List[TradeResult], List[float]]:
        """
        Simulate trading based on signals.
        
        Returns:
            - List of trade results
            - Equity curve
        """
        trades: List[TradeResult] = []
        equity_curve: List[float] = [self.initial_capital]
        
        capital = self.initial_capital
        position = 0.0
        entry_price = 0.0
        entry_time = 0
        
        for i, signal in enumerate(signals):
            if i >= len(prices):
                break
            
            current_price = prices[i]
            regime = regimes[i] if regimes and i < len(regimes) else "unknown"
            
            action = signal.get("action", "hold")
            confidence = signal.get("confidence", 0.0)
            
            # Scale position by confidence
            position_size = confidence * 0.5  # Max 50% of capital
            
            if action == "buy" and position == 0:
                # Enter long
                entry_price = current_price
                position = (capital * position_size) / current_price
                capital -= position * current_price * (1 + self.fee_rate)
                entry_time = i
            
            elif action == "sell" and position > 0:
                # Exit long
                exit_price = current_price
                pnl = position * (exit_price - entry_price)
                pnl_pct = (exit_price - entry_price) / entry_price
                capital += position * exit_price * (1 - self.fee_rate)
                
                duration_hours = i - entry_time
                
                trades.append(TradeResult(
                    timestamp=float(i),
                    action="sell",
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=position,
                    pnl=pnl - (position * entry_price * self.fee_rate * 2),  # Round-trip fees
                    pnl_pct=pnl_pct,
                    duration_hours=float(duration_hours),
                    regime=regime,
                ))
                
                position = 0
            
            # Update equity
            equity = capital + position * current_price
            equity_curve.append(equity)
        
        # Close any open position at end
        if position > 0 and prices:
            exit_price = prices[-1]
            pnl = position * (exit_price - entry_price)
            capital += position * exit_price * (1 - self.fee_rate)
            equity_curve[-1] = capital
        
        return trades, equity_curve


class WalkForwardValidator:
    """
    Walk-forward validation for trading strategies.
    
    Usage:
        validator = WalkForwardValidator()
        
        results = validator.validate(
            prices=historical_prices,
            signal_generator=my_strategy.generate_signal,
            regime_detector=my_regime_detector,
        )
        
        if results.overall_passed:
            print("Strategy validated!")
        else:
            print("Strategy failed validation")
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.simulator = StrategySimulator()
    
    def validate(
        self,
        prices: List[float],
        signal_generator: Callable[[List[float], List[str]], List[Dict[str, Any]]],
        regimes: Optional[List[str]] = None,
        initial_capital: float = 10000.0,
    ) -> ValidationSummary:
        """
        Run walk-forward validation.
        
        Args:
            prices: Historical price data (hourly bars)
            signal_generator: Function that generates signals from prices
            regimes: Optional regime labels for each bar
            initial_capital: Starting capital
        
        Returns:
            ValidationSummary with results
        """
        self.simulator.initial_capital = initial_capital
        
        # Calculate window sizes in bars (hourly)
        train_bars = self.config.train_days * 24
        val_bars = self.config.validation_days * 24
        step_bars = self.config.step_days * 24
        
        window_results: List[WindowResult] = []
        window_index = 0
        
        # Rolling windows
        start = 0
        while start + train_bars + val_bars <= len(prices):
            # Split data
            train_end = start + train_bars
            val_end = train_end + val_bars
            
            train_prices = prices[start:train_end]
            val_prices = prices[train_end:val_end]
            
            train_regimes = regimes[start:train_end] if regimes else None
            val_regimes = regimes[train_end:val_end] if regimes else None
            
            # Generate signals for training period
            train_signals = signal_generator(train_prices, train_regimes or [])
            
            # Generate signals for validation period
            val_signals = signal_generator(val_prices, val_regimes or [])
            
            # Simulate training period
            train_trades, train_equity = self.simulator.simulate(
                train_prices, train_signals, train_regimes
            )
            
            # Simulate validation period
            val_trades, val_equity = self.simulator.simulate(
                val_prices, val_signals, val_regimes
            )
            
            # Calculate metrics
            train_returns = self._equity_to_returns(train_equity)
            val_returns = self._equity_to_returns(val_equity)
            
            window_result = WindowResult(
                window_index=window_index,
                train_start=start,
                train_end=train_end,
                val_start=train_end,
                val_end=val_end,
                train_trades=len(train_trades),
                train_win_rate=PerformanceMetrics.calculate_win_rate(train_trades),
                train_sharpe=PerformanceMetrics.calculate_sharpe(train_returns, self.config.risk_free_rate),
                train_sortino=PerformanceMetrics.calculate_sortino(train_returns, self.config.risk_free_rate),
                train_max_drawdown=PerformanceMetrics.calculate_max_drawdown(train_equity),
                train_total_return=PerformanceMetrics.calculate_total_return(train_trades, initial_capital),
                val_trades=len(val_trades),
                val_win_rate=PerformanceMetrics.calculate_win_rate(val_trades),
                val_sharpe=PerformanceMetrics.calculate_sharpe(val_returns, self.config.risk_free_rate),
                val_sortino=PerformanceMetrics.calculate_sortino(val_returns, self.config.risk_free_rate),
                val_max_drawdown=PerformanceMetrics.calculate_max_drawdown(val_equity),
                val_total_return=PerformanceMetrics.calculate_total_return(val_trades, initial_capital),
                is_profitable=PerformanceMetrics.calculate_total_return(val_trades, initial_capital) > 0,
                oos_degradation=self._calculate_degradation(
                    PerformanceMetrics.calculate_total_return(train_trades, initial_capital),
                    PerformanceMetrics.calculate_total_return(val_trades, initial_capital),
                ),
            )
            
            window_results.append(window_result)
            window_index += 1
            
            # Move to next window
            start += step_bars
        
        # Aggregate results
        return self._aggregate_results(window_results)
    
    def _equity_to_returns(self, equity_curve: List[float]) -> List[float]:
        """Convert equity curve to returns."""
        if len(equity_curve) < 2:
            return []
        
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i-1] > 0:
                returns.append((equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1])
            else:
                returns.append(0.0)
        
        return returns
    
    def _calculate_degradation(self, is_return: float, oos_return: float) -> float:
        """Calculate out-of-sample degradation vs in-sample."""
        if is_return == 0:
            return 0.0
        
        return abs(oos_return - is_return) / abs(is_return)
    
    def _aggregate_results(self, window_results: List[WindowResult]) -> ValidationSummary:
        """Aggregate results across all windows."""
        if not window_results:
            return ValidationSummary(
                total_windows=0,
                profitable_windows=0,
                profitability_rate=0.0,
                avg_train_sharpe=0.0,
                avg_val_sharpe=0.0,
                avg_train_win_rate=0.0,
                avg_val_win_rate=0.0,
                avg_max_drawdown=0.0,
                passed_min_sharpe=False,
                passed_profitability_rate=False,
                passed_drawdown=False,
                passed_degradation=False,
                overall_passed=False,
                window_results=[],
            )
        
        profitable = sum(1 for w in window_results if w.is_profitable)
        profitability_rate = profitable / len(window_results)
        
        avg_train_sharpe = np.mean([w.train_sharpe for w in window_results])
        avg_val_sharpe = np.mean([w.val_sharpe for w in window_results])
        avg_train_win_rate = np.mean([w.train_win_rate for w in window_results])
        avg_val_win_rate = np.mean([w.val_win_rate for w in window_results])
        avg_max_drawdown = np.mean([w.val_max_drawdown for w in window_results])
        avg_degradation = np.mean([w.oos_degradation for w in window_results])
        
        # Check success criteria
        passed_min_sharpe = avg_val_sharpe >= self.config.min_sharpe
        passed_profitability_rate = profitability_rate >= 0.70
        passed_drawdown = avg_max_drawdown <= self.config.max_drawdown
        passed_degradation = avg_degradation <= self.config.max_oos_degradation
        
        overall_passed = all([
            passed_min_sharpe,
            passed_profitability_rate,
            passed_drawdown,
            passed_degradation,
        ])
        
        return ValidationSummary(
            total_windows=len(window_results),
            profitable_windows=profitable,
            profitability_rate=profitability_rate,
            avg_train_sharpe=float(avg_train_sharpe),
            avg_val_sharpe=float(avg_val_sharpe),
            avg_train_win_rate=float(avg_train_win_rate),
            avg_val_win_rate=float(avg_val_win_rate),
            avg_max_drawdown=float(avg_max_drawdown),
            passed_min_sharpe=passed_min_sharpe,
            passed_profitability_rate=passed_profitability_rate,
            passed_drawdown=passed_drawdown,
            passed_degradation=passed_degradation,
            overall_passed=overall_passed,
            window_results=window_results,
        )
    
    def get_report(self, summary: ValidationSummary) -> str:
        """Generate human-readable validation report."""
        report = []
        report.append("=" * 60)
        report.append("WALK-FORWARD VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        report.append(f"Total Windows: {summary.total_windows}")
        report.append(f"Profitable Windows: {summary.profitable_windows}/{summary.total_windows} ({summary.profitability_rate*100:.1f}%)")
        report.append("")
        
        # Training metrics
        report.append("TRAINING (In-Sample) Metrics:")
        report.append(f"  Sharpe Ratio: {summary.avg_train_sharpe:.2f}")
        report.append(f"  Win Rate: {summary.avg_train_win_rate*100:.1f}%")
        report.append("")
        
        # Validation metrics
        report.append("VALIDATION (Out-of-Sample) Metrics:")
        report.append(f"  Sharpe Ratio: {summary.avg_val_sharpe:.2f} {'✓' if summary.passed_min_sharpe else '✗'}")
        report.append(f"  Win Rate: {summary.avg_val_win_rate*100:.1f}%")
        report.append(f"  Max Drawdown: {summary.avg_max_drawdown*100:.1f}% {'✓' if summary.passed_drawdown else '✗'}")
        report.append("")
        
        # Success criteria
        report.append("SUCCESS CRITERIA:")
        report.append(f"  Sharpe > {self.config.min_sharpe}: {'PASS' if summary.passed_min_sharpe else 'FAIL'}")
        report.append(f"  >70% Profitable Windows: {'PASS' if summary.passed_profitability_rate else 'FAIL'}")
        report.append(f"  Drawdown < {self.config.max_drawdown*100}%: {'PASS' if summary.passed_drawdown else 'FAIL'}")
        report.append(f"  OOS Degradation < {self.config.max_oos_degradation*100}%: {'PASS' if summary.passed_degradation else 'FAIL'}")
        report.append("")
        
        # Overall result
        report.append("=" * 60)
        if summary.overall_passed:
            report.append("✓ STRATEGY VALIDATED - Ready for paper/live trading")
        else:
            report.append("✗ STRATEGY FAILED - Not ready for trading")
        report.append("=" * 60)
        
        return "\n".join(report)


__all__ = [
    "ValidationConfig",
    "TradeResult",
    "WindowResult",
    "ValidationSummary",
    "PerformanceMetrics",
    "StrategySimulator",
    "WalkForwardValidator",
]
