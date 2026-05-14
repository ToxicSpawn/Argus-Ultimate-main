#!/usr/bin/env py
"""
earnings_backtest.py
====================
Comprehensive earnings estimation backtest for Argus Ultimate.

Simulates multiple trading strategies across different market regimes
to estimate annual earnings with confidence intervals.

Usage:
    py scripts/earnings_backtest.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import random
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Market regime definitions
# ---------------------------------------------------------------------------

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_volatility"


@dataclass
class MarketConfig:
    """Configuration for a market regime."""
    name: str
    regime: MarketRegime
    annual_drift: float
    annual_vol: float
    spread_bps: float
    daily_volume_usd: float
    duration_days: int


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    name: str
    position_size_usd: float
    max_position_usd: float
    trade_frequency: float
    maker_ratio: float
    take_profit_bps: float
    stop_loss_bps: float
    capital: float


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    strategy_name: str
    market_regime: str
    total_pnl: float
    annual_return_pct: float
    annual_pnl_usd: float
    sharpe: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    profit_factor: float
    duration_days: int


# ---------------------------------------------------------------------------
# Simple backtest engine (pure Python floats)
# ---------------------------------------------------------------------------

class SimpleBacktestEngine:
    """Simplified backtest engine using pure Python floats."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        maker_fee_bps: float = -2.0,
        taker_fee_bps: float = 7.0,
        max_position_usd: float = 50000.0,
    ):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.maker_fee = float(maker_fee_bps) / 10000.0
        self.taker_fee = float(taker_fee_bps) / 10000.0
        self.max_position_usd = float(max_position_usd)
        
        self.position = 0.0
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        
        self.equity_curve: List[float] = [self.initial_capital]
        self.peak_equity = self.initial_capital
        self.max_drawdown = 0.0

    def get_position_value(self, price: float) -> float:
        return self.position * float(price)

    def can_open_position(self, size_usd: float, price: float) -> bool:
        current_value = abs(self.get_position_value(price))
        return (current_value + size_usd) <= self.max_position_usd

    def execute_trade(
        self,
        side: str,
        price: float,
        size_usd: float,
        is_maker: bool = True,
        market_impact_bps: float = 0.0,
    ) -> Tuple[float, float]:
        price = float(price)
        size_usd = float(size_usd)
        market_impact_bps = float(market_impact_bps)
        
        if side == "buy":
            fill_price = price * (1.0 + market_impact_bps / 10000.0)
        else:
            fill_price = price * (1.0 - market_impact_bps / 10000.0)
        
        size = size_usd / fill_price
        
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        fee = fill_price * size * abs(fee_rate)
        if fee_rate < 0:
            fee = -fee  # rebate
        
        if side == "buy":
            if self.position < 0:
                close_size = min(size, abs(self.position))
                pnl = (self.entry_price - fill_price) * close_size - fee
                self.total_pnl += pnl
                if pnl > 0:
                    self.winning_trades += 1
                    self.gross_profit += pnl
                else:
                    self.gross_loss += abs(pnl)
                
                self.position += close_size
                if self.position >= 0:
                    remaining = size - close_size
                    if remaining > 0:
                        self.position = remaining
                        self.entry_price = fill_price
                    else:
                        self.position = 0.0
                        self.entry_price = 0.0
            else:
                if self.position == 0:
                    self.entry_price = fill_price
                    self.position = size
                else:
                    total_cost = self.entry_price * self.position + fill_price * size
                    self.position += size
                    self.entry_price = total_cost / self.position
                self.cash -= (fill_price * size + fee)
        else:
            if self.position > 0:
                close_size = min(size, self.position)
                pnl = (fill_price - self.entry_price) * close_size - fee
                self.total_pnl += pnl
                if pnl > 0:
                    self.winning_trades += 1
                    self.gross_profit += pnl
                else:
                    self.gross_loss += abs(pnl)
                
                self.position -= close_size
                if self.position <= 0:
                    remaining = size - close_size
                    if remaining > 0:
                        self.position = -remaining
                        self.entry_price = fill_price
                    else:
                        self.position = 0.0
                        self.entry_price = 0.0
            else:
                if self.position == 0:
                    self.entry_price = fill_price
                    self.position = -size
                else:
                    total_cost = self.entry_price * abs(self.position) + fill_price * size
                    self.position -= size
                    self.entry_price = total_cost / abs(self.position) if self.position != 0 else 0.0
                self.cash += (fill_price * size - fee)
        
        self.total_trades += 1
        return fill_price, fee

    def update_equity(self, price: float):
        equity = self.cash + self.position * float(price)
        self.equity_curve.append(equity)
        if equity > self.peak_equity:
            self.peak_equity = equity
        drawdown = (self.peak_equity - equity) / self.peak_equity * 100.0
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    def close_all(self, price: float, market_impact_bps: float = 0.0):
        if abs(self.position) > 1e-9:
            side = "sell" if self.position > 0 else "buy"
            size_usd = abs(self.position * price)
            self.execute_trade(side, price, size_usd, is_maker=False, market_impact_bps=market_impact_bps)
            self.update_equity(price)

    def get_stats(self, duration_days: int) -> BacktestResult:
        final_equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        total_pnl = final_equity - self.initial_capital
        
        annual_factor = 365.0 / duration_days
        annual_pnl = total_pnl * annual_factor
        annual_return = (annual_pnl / self.initial_capital) * 100.0
        
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                if self.equity_curve[i-1] != 0:
                    ret = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
                    returns.append(ret)
            
            if returns:
                mean_return = sum(returns) / len(returns)
                variance = sum((r - mean_return) ** 2 for r in returns) / max(1, len(returns) - 1)
                std_return = math.sqrt(variance) if variance > 0 else 1e-10
                sharpe = (mean_return / std_return) * math.sqrt(252 * 24 * 60) if std_return > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        profit_factor = self.gross_profit / self.gross_loss if self.gross_loss > 0 else float('inf')
        
        return BacktestResult(
            strategy_name="",
            market_regime="",
            total_pnl=total_pnl,
            annual_return_pct=annual_return,
            annual_pnl_usd=annual_pnl,
            sharpe=sharpe,
            max_drawdown_pct=self.max_drawdown,
            win_rate=win_rate,
            total_trades=self.total_trades,
            profit_factor=profit_factor,
            duration_days=duration_days,
        )


# ---------------------------------------------------------------------------
# Price generator (pure Python)
# ---------------------------------------------------------------------------

class PriceGenerator:
    """Generate synthetic price paths using pure Python."""

    def __init__(
        self,
        initial_price: float = 65000.0,
        annual_drift: float = 0.50,
        annual_vol: float = 0.80,
        seed: Optional[int] = None,
    ):
        self.initial_price = initial_price
        self.annual_drift = annual_drift
        self.annual_vol = annual_vol
        if seed is not None:
            random.seed(seed)

    def generate_minute_prices(self, n_minutes: int) -> Tuple[List[float], List[float], List[float]]:
        dt = 1.0 / (365.25 * 24.0 * 60.0)
        
        omega = 0.0000001
        alpha = 0.1
        beta = 0.85
        
        prices = [0.0] * n_minutes
        spreads = [0.0] * n_minutes
        imbalances = [0.0] * n_minutes
        vols = [0.0] * n_minutes
        
        prices[0] = self.initial_price
        vols[0] = self.annual_vol * math.sqrt(dt)
        spreads[0] = 3.0
        imbalances[0] = 0.0
        
        for i in range(1, n_minutes):
            if i > 1:
                prev_ret = math.log(prices[i-1] / prices[i-2]) if prices[i-2] > 0 else 0
                vols[i] = math.sqrt(omega + alpha * prev_ret ** 2 + beta * vols[i-1] ** 2)
            else:
                vols[i] = vols[i-1]
            
            vols[i] = max(vols[i], self.annual_vol * math.sqrt(dt) * 0.3)
            vols[i] = min(vols[i], self.annual_vol * math.sqrt(dt) * 3.0)
            
            drift = self.annual_drift * dt
            shock = random.gauss(0, 1)
            ret = drift + vols[i] * shock
            prices[i] = prices[i-1] * math.exp(ret)
            
            spreads[i] = spreads[i-1] + 0.1 * (3.0 - spreads[i-1]) * dt + random.gauss(0, 0.5 * math.sqrt(dt))
            spreads[i] = max(1.0, min(15.0, spreads[i]))
            
            imbalances[i] = imbalances[i-1] + 0.5 * (-imbalances[i-1]) * dt + random.gauss(0, 0.3 * math.sqrt(dt))
            imbalances[i] = max(-1.0, min(1.0, imbalances[i]))
        
        return prices, spreads, imbalances


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class MarketMakingStrategy:
    def __init__(self, config: StrategyConfig):
        self.config = config

    def on_tick(self, engine: SimpleBacktestEngine, tick: int, price: float, spread_bps: float, imbalance: float):
        trade_interval = max(1, int(60.0 / self.config.trade_frequency))
        if tick % trade_interval != 0:
            return
        
        if not engine.can_open_position(self.config.position_size_usd, price):
            return
        
        half_spread = spread_bps / 2.0
        
        if imbalance > 0.3:
            if engine.position > 0:
                engine.execute_trade("sell", price, self.config.position_size_usd, is_maker=True, market_impact_bps=half_spread)
        elif imbalance < -0.3:
            if engine.position < 0:
                engine.execute_trade("buy", price, self.config.position_size_usd, is_maker=True, market_impact_bps=half_spread)
        else:
            if random.random() < 0.5:
                engine.execute_trade("buy", price, self.config.position_size_usd, is_maker=True, market_impact_bps=half_spread)
            else:
                engine.execute_trade("sell", price, self.config.position_size_usd, is_maker=True, market_impact_bps=half_spread)
        
        engine.update_equity(price)


class MomentumStrategy:
    def __init__(self, config: StrategyConfig, lookback: int = 20):
        self.config = config
        self.lookback = lookback
        self.price_history: List[float] = []

    def on_tick(self, engine: SimpleBacktestEngine, tick: int, price: float, spread_bps: float, imbalance: float):
        self.price_history.append(price)
        if len(self.price_history) > self.lookback:
            self.price_history.pop(0)
        
        if len(self.price_history) < self.lookback:
            return
        
        trade_interval = max(1, int(120.0 / self.config.trade_frequency))
        if tick % trade_interval != 0:
            return
        
        returns = [math.log(self.price_history[i] / self.price_history[i-1]) for i in range(1, len(self.price_history))]
        momentum = (sum(returns) / len(returns)) * 10000.0 if returns else 0
        
        if abs(engine.get_position_value(price)) >= engine.max_position_usd:
            return
        
        if momentum > 3.0:
            if engine.position <= 0:
                engine.execute_trade("buy", price, self.config.position_size_usd, is_maker=False, market_impact_bps=spread_bps / 2.0)
                engine.update_equity(price)
        elif momentum < -3.0:
            if engine.position >= 0:
                engine.execute_trade("sell", price, self.config.position_size_usd, is_maker=False, market_impact_bps=spread_bps / 2.0)
                engine.update_equity(price)


class MeanReversionStrategy:
    """
    Improved Mean Reversion strategy with:
    - Trend detection to avoid trading against strong trends
    - Dynamic position sizing based on signal strength
    - Stop losses to prevent catastrophic losses
    - Regime-aware filtering
    """
    
    def __init__(self, config: StrategyConfig, lookback: int = 50):
        self.config = config
        self.lookback = lookback
        self.price_history: List[float] = []
        self.returns_history: List[float] = []
        self.entry_prices: List[float] = []  # Track entry prices for position management
        self.consecutive_losses = 0
        self.last_trade_was_winner = True
        
    def _calculate_trend_strength(self) -> float:
        """Calculate trend strength using linear regression slope."""
        if len(self.price_history) < 20:
            return 0.0
        
        # Use last 20 prices for trend calculation
        recent = self.price_history[-20:]
        n = len(recent)
        
        # Linear regression slope
        sum_x = sum(range(n))
        sum_y = sum(recent)
        sum_xy = sum(i * y for i, y in enumerate(recent))
        sum_x2 = sum(i * i for i in range(n))
        
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Normalize by price level
        avg_price = sum_y / n
        normalized_slope = (slope / avg_price) * 100.0 if avg_price > 0 else 0.0
        
        return normalized_slope
    
    def _calculate_volatility_regime(self) -> str:
        """Determine if we're in high or low volatility regime."""
        if len(self.returns_history) < 20:
            return "normal"
        
        recent_returns = self.returns_history[-20:]
        mean_ret = sum(recent_returns) / len(recent_returns)
        variance = sum((r - mean_ret) ** 2 for r in recent_returns) / len(recent_returns)
        std = math.sqrt(variance)
        
        if std > 0.003:  # High volatility
            return "high"
        elif std < 0.001:  # Low volatility
            return "low"
        return "normal"
    
    def on_tick(self, engine: SimpleBacktestEngine, tick: int, price: float, spread_bps: float, imbalance: float):
        # Track returns for volatility calculation
        if len(self.price_history) > 0:
            if self.price_history[-1] > 0:
                ret = math.log(price / self.price_history[-1])
                self.returns_history.append(ret)
                if len(self.returns_history) > 100:
                    self.returns_history.pop(0)
        
        self.price_history.append(price)
        if len(self.price_history) > self.lookback:
            self.price_history.pop(0)
        
        if len(self.price_history) < self.lookback:
            return
        
        trade_interval = max(1, int(180.0 / self.config.trade_frequency))
        if tick % trade_interval != 0:
            return
        
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength()
        volatility_regime = self._calculate_volatility_regime()
        
        # Skip trading during strong trends (mean reversion doesn't work in trends)
        if abs(trend_strength) > 0.5:  # Strong trend detected
            return
        
        # Skip trading during high volatility (unreliable signals)
        if volatility_regime == "high":
            return
        
        # Calculate z-score
        mean = sum(self.price_history) / len(self.price_history)
        variance = sum((p - mean) ** 2 for p in self.price_history) / len(self.price_history)
        std = math.sqrt(variance) if variance > 0 else 1e-6
        
        if std < 1e-6:
            return
        
        z_score = (price - mean) / std
        
        # Check position limits
        if abs(engine.get_position_value(price)) >= engine.max_position_usd:
            return
        
        # Dynamic position sizing based on signal strength and recent performance
        base_size = self.config.position_size_usd
        
        # Reduce size after consecutive losses
        if self.consecutive_losses >= 3:
            size_multiplier = 0.5
        elif self.consecutive_losses >= 2:
            size_multiplier = 0.75
        else:
            size_multiplier = 1.0
        
        # Adjust based on signal strength (stronger signals = larger size)
        signal_strength = abs(z_score) - 2.0  # How far beyond threshold
        if signal_strength > 0:
            size_multiplier *= min(1.5, 1.0 + signal_strength * 0.25)
        
        adjusted_size = base_size * size_multiplier
        
        # Entry signals with tighter thresholds for better risk/reward
        if z_score < -2.5 and trend_strength > -0.3:  # Oversold, not in strong downtrend
            engine.execute_trade("buy", price, adjusted_size, is_maker=True, market_impact_bps=spread_bps / 3.0)
            engine.update_equity(price)
            self.entry_prices.append(price)
            self.last_trade_was_winner = False
            
        elif z_score > 2.5 and trend_strength < 0.3:  # Overbought, not in strong uptrend
            engine.execute_trade("sell", price, adjusted_size, is_maker=True, market_impact_bps=spread_bps / 3.0)
            engine.update_equity(price)
            self.entry_prices.append(price)
            self.last_trade_was_winner = False
        
        # Exit logic: close position when mean is reached or small profit
        elif abs(z_score) < 0.5 and len(self.entry_prices) > 0:
            # Close position when price reverts to mean
            if engine.position > 0:
                engine.execute_trade("sell", price, abs(engine.get_position_value(price)), is_maker=True, market_impact_bps=spread_bps / 3.0)
                engine.update_equity(price)
                self._update_trade_result(price, "long")
            elif engine.position < 0:
                engine.execute_trade("buy", price, abs(engine.get_position_value(price)), is_maker=True, market_impact_bps=spread_bps / 3.0)
                engine.update_equity(price)
                self._update_trade_result(price, "short")
        
        # Emergency stop loss: if position is down 8%, close it
        if len(self.entry_prices) > 0:
            avg_entry = sum(self.entry_prices) / len(self.entry_prices)
            if engine.position > 0 and price < avg_entry * 0.92:
                engine.execute_trade("sell", price, abs(engine.get_position_value(price)), is_maker=False, market_impact_bps=spread_bps / 2.0)
                engine.update_equity(price)
                self.consecutive_losses += 1
                self.entry_prices.clear()
            elif engine.position < 0 and price > avg_entry * 1.08:
                engine.execute_trade("buy", price, abs(engine.get_position_value(price)), is_maker=False, market_impact_bps=spread_bps / 2.0)
                engine.update_equity(price)
                self.consecutive_losses += 1
                self.entry_prices.clear()
    
    def _update_trade_result(self, exit_price: float, direction: str):
        """Track trade results for adaptive position sizing."""
        if len(self.entry_prices) > 0:
            avg_entry = self.entry_prices[-1]
            if direction == "long":
                pnl = exit_price - avg_entry
            else:
                pnl = avg_entry - exit_price
            
            if pnl > 0:
                self.consecutive_losses = 0
                self.last_trade_was_winner = True
            else:
                self.consecutive_losses += 1
                self.last_trade_was_winner = False
            
            self.entry_prices.clear()


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    strategy_config: StrategyConfig,
    market_config: MarketConfig,
    seed: Optional[int] = None,
) -> BacktestResult:
    n_minutes = market_config.duration_days * 24 * 60
    
    generator = PriceGenerator(
        initial_price=65000.0,
        annual_drift=market_config.annual_drift,
        annual_vol=market_config.annual_vol,
        seed=seed,
    )
    
    logger.info("Generating %d minutes for %s (%d days)...", n_minutes, market_config.name, market_config.duration_days)
    prices, spreads, imbalances = generator.generate_minute_prices(n_minutes)
    
    engine = SimpleBacktestEngine(
        initial_capital=strategy_config.capital,
        maker_fee_bps=-2.0,
        taker_fee_bps=7.0,
        max_position_usd=strategy_config.max_position_usd,
    )
    
    if "market_making" in strategy_config.name.lower():
        strategy = MarketMakingStrategy(strategy_config)
    elif "momentum" in strategy_config.name.lower():
        strategy = MomentumStrategy(strategy_config, lookback=20)
    elif "mean_reversion" in strategy_config.name.lower():
        strategy = MeanReversionStrategy(strategy_config, lookback=50)
    else:
        # Default to mean reversion for unknown strategies
        strategy = MeanReversionStrategy(strategy_config, lookback=50)
    
    logger.info("Running backtest with %d minute ticks...", n_minutes)
    
    for i in range(n_minutes):
        strategy.on_tick(engine, i, prices[i], spreads[i], imbalances[i])
    
    engine.close_all(prices[-1], market_impact_bps=spreads[-1])
    
    result = engine.get_stats(market_config.duration_days)
    result.strategy_name = strategy_config.name
    result.market_regime = market_config.name
    
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("")
    print("=" * 80)
    print("  ARGUS ULTIMATE - COMPREHENSIVE EARNINGS BACKTEST")
    print("=" * 80)
    print("")
    
    # Extended market regimes with more realistic scenarios
    market_regimes = [
        # Original regimes
        MarketConfig("Bull Market (2024-style)", MarketRegime.BULL, 1.20, 0.65, 3.5, 30_000_000_000, 30),
        MarketConfig("Bear Market (2022-style)", MarketRegime.BEAR, -0.60, 0.85, 5.0, 20_000_000_000, 30),
        MarketConfig("Sideways/Range-bound", MarketRegime.SIDEWAYS, 0.05, 0.45, 3.0, 25_000_000_000, 30),
        MarketConfig("High Volatility (crisis)", MarketRegime.HIGH_VOL, -0.20, 1.50, 8.0, 50_000_000_000, 30),
        # Additional realistic regimes
        MarketConfig("Mild Bull (steady growth)", MarketRegime.BULL, 0.40, 0.35, 2.5, 20_000_000_000, 30),
        MarketConfig("Flash Crash Recovery", MarketRegime.HIGH_VOL, -0.10, 2.00, 15.0, 80_000_000_000, 7),
        MarketConfig("Low Vol Grind", MarketRegime.SIDEWAYS, 0.10, 0.25, 2.0, 15_000_000_000, 30),
        MarketConfig("Volatile Bull", MarketRegime.BULL, 0.80, 1.00, 5.0, 40_000_000_000, 30),
    ]
    
    # Strategies with different capital levels
    strategies = [
        # Original strategies at $100K
        StrategyConfig("Market Making ($100K)", 5000, 50000, 100, 0.9, 3.0, 10.0, 100000),
        StrategyConfig("Momentum ($100K)", 10000, 100000, 20, 0.7, 15.0, 8.0, 100000),
        StrategyConfig("Mean Reversion ($100K)", 8000, 80000, 15, 0.8, 10.0, 12.0, 100000),
        # Larger capital tests
        StrategyConfig("Market Making ($500K)", 25000, 250000, 100, 0.9, 3.0, 10.0, 500000),
        StrategyConfig("Momentum ($500K)", 50000, 500000, 20, 0.7, 15.0, 8.0, 500000),
        StrategyConfig("Mean Reversion ($500K)", 40000, 400000, 15, 0.8, 10.0, 12.0, 500000),
    ]
    
    all_results: List[BacktestResult] = []
    base_seed = 42
    
    for regime_idx, market in enumerate(market_regimes):
        print("")
        print("-" * 60)
        print(f"  Market Regime: {market.name}")
        print(f"  Drift: {market.annual_drift*100:.0f}% | Vol: {market.annual_vol*100:.0f}% | Spread: {market.spread_bps:.1f}bps")
        print("-" * 60)
        print("")
        
        for strat_idx, strategy in enumerate(strategies):
            seed = base_seed + regime_idx * 100 + strat_idx
            logger.info("Running %s in %s regime...", strategy.name, market.name)
            
            try:
                result = run_backtest(strategy, market, seed=seed)
                all_results.append(result)
                
                print(f"  {strategy.name:20s} | "
                      f"Annual PnL: ${result.annual_pnl_usd:>12,.0f} | "
                      f"Return: {result.annual_return_pct:>7.1f}% | "
                      f"Sharpe: {result.sharpe:>5.2f} | "
                      f"MaxDD: {result.max_drawdown_pct:>6.1f}% | "
                      f"Trades: {result.total_trades:>5d} | "
                      f"Win: {result.win_rate:.1%}")
            except Exception as e:
                logger.error("Backtest failed for %s/%s: %s", strategy.name, market.name, e)
                import traceback
                traceback.print_exc()
    
    # Summary
    print("")
    print("=" * 80)
    print("  EARNINGS ESTIMATION SUMMARY")
    print("=" * 80)
    print("")
    
    if not all_results:
        print("  No valid results to analyze.")
        return all_results
    
    strategy_results: Dict[str, List[BacktestResult]] = {}
    for r in all_results:
        strategy_results.setdefault(r.strategy_name, []).append(r)
    
    print(f"  {'Strategy':20s} | {'Avg Annual PnL':>15s} | {'Avg Return':>10s} | {'Avg Sharpe':>10s} | {'Avg MaxDD':>10s} | {'Win Rate':>8s}")
    print(f"  {'-'*20}-+-{'-'*15}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    
    for strat_name, results in strategy_results.items():
        avg_pnl = sum(r.annual_pnl_usd for r in results) / len(results)
        avg_return = sum(r.annual_return_pct for r in results) / len(results)
        avg_sharpe = sum(r.sharpe for r in results) / len(results)
        avg_dd = sum(r.max_drawdown_pct for r in results) / len(results)
        avg_win_rate = sum(r.win_rate for r in results) / len(results)
        
        print(f"  {strat_name:20s} | ${avg_pnl:>14,.0f} | {avg_return:>9.1f}% | {avg_sharpe:>9.2f} | {avg_dd:>9.1f}% | {avg_win_rate:>7.1%}")
    
    print("")
    print("-" * 70)
    
    valid_results = [r for r in all_results if abs(r.annual_pnl_usd) < 1e10]
    if not valid_results:
        valid_results = all_results
    
    worst_case = min(r.annual_pnl_usd for r in valid_results)
    best_case = max(r.annual_pnl_usd for r in valid_results)
    median_case = sorted(r.annual_pnl_usd for r in valid_results)[len(valid_results) // 2]
    mean_case = sum(r.annual_pnl_usd for r in valid_results) / len(valid_results)
    
    weights = [0.25, 0.15, 0.15, 0.10, 0.15, 0.05, 0.10, 0.05]  # 8 regimes
    weighted_pnl = 0.0
    for i, regime in enumerate(market_regimes):
        regime_results = [r for r in valid_results if r.market_regime == regime.name]
        if regime_results:
            regime_avg = sum(r.annual_pnl_usd for r in regime_results) / len(regime_results)
            weighted_pnl += regime_avg * weights[i]
    
    print("")
    print("  EARNINGS ESTIMATES (based on $100,000 starting capital):")
    print("  " + "-" * 50)
    print(f"  Conservative (worst case):     ${worst_case:>12,.0f}")
    print(f"  Median scenario:               ${median_case:>12,.0f}")
    print(f"  Mean across all scenarios:     ${mean_case:>12,.0f}")
    print(f"  Market-weighted estimate:      ${weighted_pnl:>12,.0f}")
    print(f"  Optimistic (best case):        ${best_case:>12,.0f}")
    
    print("")
    print("  SCALED EARNINGS BY CAPITAL LEVEL:")
    print("  " + "-" * 50)
    for cap in [100000, 250000, 500000, 1000000, 5000000, 10000000]:
        scale = cap / 100000.0
        print(f"  ${cap:>12,} | Conservative: ${worst_case * scale:>10,.0f} | Expected: ${weighted_pnl * scale:>10,.0f} | Optimistic: ${best_case * scale:>10,.0f}")
    
    all_sharpes = [r.sharpe for r in valid_results]
    all_drawdowns = [r.max_drawdown_pct for r in valid_results]
    
    print("")
    print("  RISK METRICS:")
    print("  " + "-" * 50)
    print(f"  Average Sharpe Ratio:          {sum(all_sharpes) / len(all_sharpes):>8.2f}")
    print(f"  Best Sharpe Ratio:             {max(all_sharpes):>8.2f}")
    print(f"  Worst Sharpe Ratio:            {min(all_sharpes):>8.2f}")
    print(f"  Average Max Drawdown:          {sum(all_drawdowns) / len(all_drawdowns):>7.1f}%")
    print(f"  Worst Max Drawdown:            {max(all_drawdowns):>7.1f}%")
    
    all_return_pcts = [r.annual_return_pct for r in valid_results]
    print("")
    print("  RETURN METRICS:")
    print("  " + "-" * 50)
    print(f"  Average Annual Return:         {sum(all_return_pcts) / len(all_return_pcts):>7.1f}%")
    print(f"  Median Annual Return:          {sorted(all_return_pcts)[len(all_return_pcts) // 2]:>7.1f}%")
    print(f"  Best Annual Return:            {max(all_return_pcts):>7.1f}%")
    print(f"  Worst Annual Return:           {min(all_return_pcts):>7.1f}%")
    
    print("")
    print("=" * 80)
    print(f"  BACKTEST COMPLETE - {len(all_results)} scenarios evaluated")
    print("=" * 80)
    print("")
    
    return all_results


if __name__ == "__main__":
    main()
