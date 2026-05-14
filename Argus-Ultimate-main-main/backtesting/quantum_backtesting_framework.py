"""
Comprehensive Backtesting Framework for Quantum Strategies
==========================================================
Features:
- Vectorized backtesting for speed
- Walk-forward optimization
- Monte Carlo simulation
- Quantum strategy validation
- Performance metrics (Sharpe, Sortino, Calmar, etc.)
- Drawdown analysis
- Trade-level analytics
"""

import asyncio
import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Backtest execution modes."""
    VECTORIZED = "vectorized"  # Fast, numpy-based
    EVENT_DRIVEN = "event_driven"  # Slower, more accurate
    MONTE_CARLO = "monte_carlo"  # Statistical simulation
    WALK_FORWARD = "walk_forward"  # Out-of-sample validation


@dataclass
class Trade:
    """Individual trade record."""
    entry_time: float
    exit_time: Optional[float]
    entry_price: float
    exit_price: Optional[float]
    size: float
    side: str  # "long" or "short"
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    duration_hours: float = 0.0
    strategy: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 10000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    leverage: float = 1.0
    max_position_size: float = 0.25  # 25% of capital
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.05  # 5%
    mode: BacktestMode = BacktestMode.VECTORIZED
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class BacktestResult:
    """Backtest results."""
    # Performance metrics
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    downside_volatility: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    # Time metrics
    avg_trade_duration_hours: float = 0.0
    avg_win_duration_hours: float = 0.0
    avg_loss_duration_hours: float = 0.0
    
    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    
    # Trades
    trades: List[Trade] = field(default_factory=list)
    
    # Metadata
    config: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0


class QuantumBacktester:
    """
    Quantum-Enhanced Backtesting Engine
    ====================================
    Backtests quantum strategies with full analytics.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.results: Dict[str, BacktestResult] = {}
        
    def generate_synthetic_data(
        self,
        symbol: str,
        days: int = 365,
        start_price: float = 50000.0,
        volatility: float = 0.02,
        trend: float = 0.0001
    ) -> Dict[str, np.ndarray]:
        """Generate synthetic OHLCV data for backtesting."""
        np.random.seed(42)
        
        n_bars = days * 24  # Hourly bars
        
        # Generate returns with trend and volatility
        returns = np.random.normal(trend, volatility, n_bars)
        
        # Add regime changes
        regime_length = 24 * 7  # Weekly regimes
        for i in range(0, n_bars, regime_length):
            regime_vol = volatility * np.random.uniform(0.5, 2.0)
            regime_trend = trend * np.random.uniform(-2, 2)
            end = min(i + regime_length, n_bars)
            returns[i:end] = np.random.normal(regime_trend, regime_vol, end - i)
        
        # Generate prices
        prices = start_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV
        timestamps = np.arange(n_bars)
        open_prices = prices * (1 + np.random.uniform(-0.001, 0.001, n_bars))
        high_prices = prices * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
        low_prices = prices * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))
        close_prices = prices
        volumes = np.random.lognormal(15, 1, n_bars)
        
        return {
            "timestamp": timestamps,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes
        }
    
    def run_backtest(
        self,
        strategy_func: Callable,
        data: Dict[str, np.ndarray],
        strategy_name: str = "strategy"
    ) -> BacktestResult:
        """
        Run backtest for a strategy.
        
        strategy_func: Function(data, config) -> List[Trade]
        """
        start_time = time.time()
        
        # Execute strategy
        trades = strategy_func(data, self.config)
        
        # Calculate metrics
        result = self._calculate_metrics(trades, data, self.config)
        result.config = {
            "initial_capital": self.config.initial_capital,
            "commission": self.config.commission,
            "leverage": self.config.leverage,
            "strategy": strategy_name
        }
        result.duration_seconds = time.time() - start_time
        
        self.results[strategy_name] = result
        return result
    
    def _calculate_metrics(
        self,
        trades: List[Trade],
        data: Dict[str, np.ndarray],
        config: BacktestConfig
    ) -> BacktestResult:
        """Calculate comprehensive backtest metrics."""
        result = BacktestResult()
        result.trades = trades
        result.total_trades = len(trades)
        
        if not trades:
            return result
        
        # Separate wins and losses
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        
        result.winning_trades = len(winning)
        result.losing_trades = len(losing)
        result.win_rate = len(winning) / len(trades) if trades else 0
        
        # PnL metrics
        pnls = [t.pnl for t in trades]
        total_pnl = sum(pnls)
        result.total_return = total_pnl
        result.total_return_pct = total_pnl / config.initial_capital
        
        # Average win/loss
        if winning:
            result.avg_win = np.mean([t.pnl for t in winning])
            result.avg_win_duration_hours = np.mean([t.duration_hours for t in winning])
        if losing:
            result.avg_loss = np.mean([t.pnl for t in losing])
            result.avg_loss_duration_hours = np.mean([t.duration_hours for t in losing])
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning) if winning else 0
        gross_loss = abs(sum(t.pnl for t in losing)) if losing else 1
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        result.expectancy = (result.win_rate * result.avg_win) - ((1 - result.win_rate) * abs(result.avg_loss))
        
        # Average trade duration
        result.avg_trade_duration_hours = np.mean([t.duration_hours for t in trades])
        
        # Build equity curve
        equity = [config.initial_capital]
        for trade in trades:
            equity.append(equity[-1] + trade.pnl)
        result.equity_curve = equity
        
        # Calculate drawdown
        peak = equity[0]
        drawdowns = []
        for e in equity:
            if e > peak:
                peak = e
            drawdown = (peak - e) / peak
            drawdowns.append(drawdown)
        result.drawdown_curve = drawdowns
        result.max_drawdown_pct = max(drawdowns) if drawdowns else 0
        result.max_drawdown = result.max_drawdown_pct * peak
        
        # Sharpe ratio (annualized, assuming hourly data)
        if len(pnls) > 1:
            returns_array = np.array(pnls) / config.initial_capital
            result.volatility = np.std(returns_array) * np.sqrt(8760)  # Annualized
            mean_return = np.mean(returns_array) * 8760
            result.sharpe_ratio = mean_return / result.volatility if result.volatility > 0 else 0
            
            # Sortino ratio
            downside_returns = returns_array[returns_array < 0]
            if len(downside_returns) > 0:
                result.downside_volatility = np.std(downside_returns) * np.sqrt(8760)
                result.sortino_ratio = mean_return / result.downside_volatility if result.downside_volatility > 0 else 0
            
            # Calmar ratio
            result.calmar_ratio = result.annualized_return / result.max_drawdown_pct if result.max_drawdown_pct > 0 else 0
            
            # VaR and CVaR
            result.var_95 = np.percentile(returns_array, 5)
            result.cvar_95 = np.mean(returns_array[returns_array <= result.var_95])
        
        # Annualized return
        if len(trades) > 0:
            total_hours = sum(t.duration_hours for t in trades)
            if total_hours > 0:
                years = total_hours / 8760
                result.annualized_return = (1 + result.total_return_pct) ** (1 / years) - 1 if years > 0 else 0
        
        return result


class QuantumStrategyBacktester:
    """
    Quantum Strategy Backtester
    ===========================
    Specialized backtester for quantum-enhanced strategies.
    """
    
    def __init__(self):
        self.backtester = QuantumBacktester()
        self.strategy_results: Dict[str, BacktestResult] = {}
        
    def backtest_quantum_momentum(
        self,
        data: Dict[str, np.ndarray],
        quantum_factor: float = 0.5
    ) -> BacktestResult:
        """Backtest quantum momentum strategy."""
        
        def strategy(data: Dict[str, np.ndarray], config: BacktestConfig) -> List[Trade]:
            trades = []
            closes = data["close"]
            n = len(closes)
            
            # Quantum-enhanced momentum calculation
            lookback = 20
            position = 0
            
            for i in range(lookback, n):
                # Standard momentum
                momentum = (closes[i] - closes[i-lookback]) / closes[i-lookback]
                
                # Quantum superposition factor (simulated)
                quantum_momentum = momentum * (1 + quantum_factor * np.sin(i * 0.1))
                
                # Generate signals
                if quantum_momentum > 0.02 and position == 0:
                    # Buy signal
                    entry_price = closes[i] * (1 + config.slippage)
                    size = config.initial_capital * config.max_position_size / entry_price
                    position = 1
                    
                    trade = Trade(
                        entry_time=float(i),
                        exit_time=None,
                        entry_price=entry_price,
                        exit_price=None,
                        size=size,
                        side="long",
                        strategy="quantum_momentum"
                    )
                    
                elif quantum_momentum < -0.02 and position == 1:
                    # Sell signal
                    exit_price = closes[i] * (1 - config.slippage)
                    pnl = (exit_price - trade.entry_price) * trade.size
                    pnl -= trade.entry_price * trade.size * config.commission * 2  # Entry + exit fees
                    trade.exit_time = float(i)
                    trade.exit_price = exit_price
                    trade.pnl = pnl
                    trade.pnl_pct = pnl / config.initial_capital
                    trade.duration_hours = float(i - trade.entry_time)
                    trades.append(trade)
                    position = 0
            
            return trades
        
        result = self.backtester.run_backtest(strategy, data, "quantum_momentum")
        self.strategy_results["quantum_momentum"] = result
        return result
    
    def backtest_quantum_mean_reversion(
        self,
        data: Dict[str, np.ndarray],
        quantum_factor: float = 0.3
    ) -> BacktestResult:
        """Backtest quantum mean reversion strategy."""
        
        def strategy(data: Dict[str, np.ndarray], config: BacktestConfig) -> List[Trade]:
            trades = []
            closes = data["close"]
            n = len(closes)
            
            lookback = 50
            position = 0
            entry_trade = None
            
            for i in range(lookback, n):
                # Calculate mean and std
                window = closes[i-lookback:i]
                mean = np.mean(window)
                std = np.std(window)
                
                if std == 0:
                    continue
                
                z_score = (closes[i] - mean) / std
                
                # Quantum tunneling effect (probabilistic entry)
                quantum_prob = np.exp(-quantum_factor * abs(z_score))
                
                # Entry signals
                if position == 0 and z_score < -2 and np.random.random() < quantum_prob:
                    # Oversold - buy
                    entry_price = closes[i] * (1 + config.slippage)
                    size = config.initial_capital * config.max_position_size / entry_price
                    position = 1
                    entry_trade = Trade(
                        entry_time=float(i),
                        exit_time=None,
                        entry_price=entry_price,
                        exit_price=None,
                        size=size,
                        side="long",
                        strategy="quantum_mean_reversion"
                    )
                
                # Exit signals
                elif position == 1 and z_score > 0:
                    exit_price = closes[i] * (1 - config.slippage)
                    pnl = (exit_price - entry_trade.entry_price) * entry_trade.size
                    pnl -= entry_trade.entry_price * entry_trade.size * config.commission * 2
                    entry_trade.exit_time = float(i)
                    entry_trade.exit_price = exit_price
                    entry_trade.pnl = pnl
                    entry_trade.pnl_pct = pnl / config.initial_capital
                    entry_trade.duration_hours = float(i - entry_trade.entry_time)
                    trades.append(entry_trade)
                    position = 0
                    entry_trade = None
            
            return trades
        
        result = self.backtester.run_backtest(strategy, data, "quantum_mean_reversion")
        self.strategy_results["quantum_mean_reversion"] = result
        return result
    
    def backtest_quantum_market_making(
        self,
        data: Dict[str, np.ndarray],
        quantum_factor: float = 0.4
    ) -> BacktestResult:
        """Backtest quantum market making strategy."""
        
        def strategy(data: Dict[str, np.ndarray], config: BacktestConfig) -> List[Trade]:
            trades = []
            closes = data["close"]
            highs = data["high"]
            lows = data["low"]
            n = len(closes)
            
            spread_pct = 0.002  # 0.2% spread
            inventory = 0
            max_inventory = 100
            
            for i in range(1, n - 1):
                mid_price = closes[i]
                
                # Quantum-enhanced spread adjustment
                volatility = (highs[i] - lows[i]) / mid_price
                quantum_spread = spread_pct * (1 + quantum_factor * volatility * 10)
                
                bid_price = mid_price * (1 - quantum_spread)
                ask_price = mid_price * (1 + quantum_spread)
                
                # Simulate fills
                if lows[i+1] <= bid_price and inventory < max_inventory:
                    # Bid filled - buy
                    inventory += 1
                    trade = Trade(
                        entry_time=float(i),
                        exit_time=float(i + 1),
                        entry_price=bid_price,
                        exit_price=ask_price,
                        size=1.0,
                        side="long",
                        strategy="quantum_market_making"
                    )
                    trade.pnl = (ask_price - bid_price) * 1.0 - bid_price * config.commission * 2
                    trade.pnl_pct = trade.pnl / config.initial_capital
                    trade.duration_hours = 1.0
                    trades.append(trade)
                    inventory -= 1
                
                elif highs[i+1] >= ask_price and inventory > -max_inventory:
                    # Ask filled - sell
                    inventory -= 1
                    trade = Trade(
                        entry_time=float(i),
                        exit_time=float(i + 1),
                        entry_price=ask_price,
                        exit_price=bid_price,
                        size=1.0,
                        side="short",
                        strategy="quantum_market_making"
                    )
                    trade.pnl = (bid_price - ask_price) * 1.0 - ask_price * config.commission * 2
                    trade.pnl_pct = trade.pnl / config.initial_capital
                    trade.duration_hours = 1.0
                    trades.append(trade)
                    inventory += 1
            
            return trades
        
        result = self.backtester.run_backtest(strategy, data, "quantum_market_making")
        self.strategy_results["quantum_market_making"] = result
        return result
    
    def backtest_quantum_ensemble(
        self,
        data: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None
    ) -> BacktestResult:
        """Backtest quantum ensemble of multiple strategies."""
        
        # Run individual strategies
        momentum_result = self.backtest_quantum_momentum(data)
        mean_rev_result = self.backtest_quantum_mean_reversion(data)
        mm_result = self.backtest_quantum_market_making(data)
        
        # Default weights
        if weights is None:
            weights = {
                "momentum": 0.4,
                "mean_reversion": 0.3,
                "market_making": 0.3
            }
        
        # Combine trades
        all_trades = []
        for trade in momentum_result.trades:
            trade.size *= weights["momentum"]
            trade.pnl *= weights["momentum"]
            all_trades.append(trade)
        
        for trade in mean_rev_result.trades:
            trade.size *= weights["mean_reversion"]
            trade.pnl *= weights["mean_reversion"]
            all_trades.append(trade)
        
        for trade in mm_result.trades:
            trade.size *= weights["market_making"]
            trade.pnl *= weights["market_making"]
            all_trades.append(trade)
        
        # Sort by time
        all_trades.sort(key=lambda t: t.entry_time)
        
        # Calculate combined metrics
        result = self.backtester._calculate_metrics(
            all_trades,
            data,
            BacktestConfig()
        )
        result.config = {"strategy": "quantum_ensemble", "weights": weights}
        self.strategy_results["quantum_ensemble"] = result
        
        return result


class MonteCarloSimulator:
    """
    Monte Carlo Simulation for Strategy Validation
    ===============================================
    Tests strategy robustness via random sampling.
    """
    
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
        
    def simulate(
        self,
        trades: List[Trade],
        initial_capital: float = 10000.0
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation on trade sequence."""
        if not trades:
            return {"error": "No trades to simulate"}
        
        pnls = [t.pnl for t in trades]
        n_trades = len(pnls)
        
        final_equities = []
        max_drawdowns = []
        sharpe_ratios = []
        
        for _ in range(self.n_simulations):
            # Resample trades with replacement
            sampled_pnls = np.random.choice(pnls, size=n_trades, replace=True)
            
            # Calculate equity curve
            equity = initial_capital
            peak = equity
            max_dd = 0
            
            for pnl in sampled_pnls:
                equity += pnl
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)
            
            final_equities.append(equity)
            max_drawdowns.append(max_dd)
            
            # Sharpe
            if np.std(sampled_pnls) > 0:
                sharpe = np.mean(sampled_pnls) / np.std(sampled_pnls) * np.sqrt(8760)
                sharpe_ratios.append(sharpe)
        
        return {
            "n_simulations": self.n_simulations,
            "final_equity": {
                "mean": np.mean(final_equities),
                "median": np.median(final_equities),
                "std": np.std(final_equities),
                "p5": np.percentile(final_equities, 5),
                "p95": np.percentile(final_equities, 95)
            },
            "max_drawdown": {
                "mean": np.mean(max_drawdowns),
                "median": np.median(max_drawdowns),
                "p95": np.percentile(max_drawdowns, 95)
            },
            "sharpe_ratio": {
                "mean": np.mean(sharpe_ratios) if sharpe_ratios else 0,
                "median": np.median(sharpe_ratios) if sharpe_ratios else 0
            },
            "probability_of_profit": sum(1 for e in final_equities if e > initial_capital) / self.n_simulations
        }


class WalkForwardOptimizer:
    """
    Walk-Forward Optimizer
    ======================
    Validates strategies on out-of-sample data.
    """
    
    def __init__(self, train_pct: float = 0.7, n_windows: int = 5):
        self.train_pct = train_pct
        self.n_windows = n_windows
        
    def optimize(
        self,
        strategy_func: Callable,
        data: Dict[str, np.ndarray],
        param_grid: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Run walk-forward optimization."""
        n_bars = len(data["close"])
        window_size = n_bars // self.n_windows
        
        results = []
        
        for i in range(self.n_windows):
            # Split data
            start = i * window_size
            train_end = start + int(window_size * self.train_pct)
            test_end = min(start + window_size, n_bars)
            
            train_data = {k: v[start:train_end] for k, v in data.items()}
            test_data = {k: v[train_end:test_end] for k, v in data.items()}
            
            # Find best params on train data
            best_params = self._grid_search(strategy_func, train_data, param_grid)
            
            # Test on out-of-sample
            config = BacktestConfig()
            test_result = QuantumBacktester(config).run_backtest(
                lambda d, c: strategy_func(d, c, **best_params),
                test_data,
                f"window_{i}"
            )
            
            results.append({
                "window": i,
                "best_params": best_params,
                "test_return": test_result.total_return_pct,
                "test_sharpe": test_result.sharpe_ratio,
                "test_trades": test_result.total_trades
            })
        
        # Aggregate results
        returns = [r["test_return"] for r in results]
        sharpes = [r["test_sharpe"] for r in results]
        
        return {
            "n_windows": self.n_windows,
            "window_results": results,
            "avg_oos_return": np.mean(returns),
            "avg_oos_sharpe": np.mean(sharpes),
            "consistency": sum(1 for r in returns if r > 0) / len(returns)
        }
    
    def _grid_search(
        self,
        strategy_func: Callable,
        data: Dict[str, np.ndarray],
        param_grid: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Simple grid search for best parameters."""
        best_score = -float('inf')
        best_params = {}
        
        # Simplified: just use first param values
        for key, values in param_grid.items():
            for val in values:
                params = {key: val}
                try:
                    config = BacktestConfig()
                    result = QuantumBacktester(config).run_backtest(
                        lambda d, c: strategy_func(d, c, **params),
                        data,
                        "optimization"
                    )
                    score = result.sharpe_ratio
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                except Exception:
                    continue
        
        return best_params


def format_backtest_report(result: BacktestResult, strategy_name: str) -> str:
    """Format backtest result as readable report."""
    report = f"""
{'='*60}
BACKTEST REPORT: {strategy_name.upper()}
{'='*60}

PERFORMANCE
-----------
Total Return:        {result.total_return_pct*100:.2f}%
Annualized Return:   {result.annualized_return*100:.2f}%
Sharpe Ratio:        {result.sharpe_ratio:.3f}
Sortino Ratio:       {result.sortino_ratio:.3f}
Calmar Ratio:        {result.calmar_ratio:.3f}

TRADE STATISTICS
----------------
Total Trades:        {result.total_trades}
Win Rate:            {result.win_rate*100:.1f}%
Profit Factor:       {result.profit_factor:.2f}
Expectancy:          ${result.expectancy:.2f}
Avg Win:             ${result.avg_win:.2f}
Avg Loss:            ${result.avg_loss:.2f}

RISK METRICS
------------
Max Drawdown:        {result.max_drawdown_pct*100:.2f}%
Volatility:          {result.volatility*100:.2f}%
VaR (95%):           {result.var_95*100:.2f}%
CVaR (95%):          {result.cvar_95*100:.2f}%

TIME METRICS
------------
Avg Trade Duration:  {result.avg_trade_duration_hours:.1f} hours
Avg Win Duration:    {result.avg_win_duration_hours:.1f} hours
Avg Loss Duration:   {result.avg_loss_duration_hours:.1f} hours

Backtest Duration:   {result.duration_seconds:.3f}s
{'='*60}
"""
    return report


# Export
__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Trade",
    "QuantumBacktester",
    "QuantumStrategyBacktester",
    "MonteCarloSimulator",
    "WalkForwardOptimizer",
    "format_backtest_report"
]
