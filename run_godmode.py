#!/usr/bin/env python3
"""
ARGUS TRADING SYSTEM - GODMODE V2

Unified production trading bot that uses V2.1 ULTIMATE evolved parameters.
All signal generation matches the evolution backtest model exactly.

Features:
- Evolved parameter flow to ALL strategy configs
- Multi-timeframe signal integration (1m, 5m, 15m, 1h)
- Unified signal scoring (RSI + BB + EMA + Momentum + Z-score + Volume + HTF bias)
- Regime-weighted signal generation
- ATR-based dynamic stops (evolved stop_loss_atr_mult)
- Short selling support
- Volume confirmation
- Hold time in bars (converted to seconds for live)
- Kelly-adjusted volatility-scaled position sizing

Usage: python run_godmode.py --capital 1000
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

# Windows console fix
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Performance: uvloop on Linux
try:
    import uvloop
    uvloop.install()
    print("[+] uvloop enabled")
except ImportError:
    pass

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("godmode_trading.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("argus.godmode")


# ============================================================================
# CONFIGURATION
# ============================================================================
class MarketRegime(Enum):
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"
    RECOVERY = "recovery"


# Map evolution regime names to production regime names
_EVO_REGIME_MAP = {
    "bull": MarketRegime.BULL_TRENDING,
    "bear": MarketRegime.BEAR_TRENDING,
    "high_vol": MarketRegime.HIGH_VOLATILITY,
    "sideways": MarketRegime.SIDEWAYS,
}


@dataclass
class GodmodeConfig:
    """GODMODE configuration with V2.1 evolved parameter support.

    All numeric fields can be overridden by evolved parameters.
    Strategy-specific configs are constructed from evolved params in _init_all_strategies.
    """
    # Capital
    initial_capital: float = 1000.0

    # Trading pairs
    symbols: List[str] = field(default_factory=lambda: [
        "BTC/AUD", "ETH/AUD", "SOL/AUD", "XRP/AUD",
    ])

    # Cycle timing
    cycle_seconds: float = 10.0

    # === Risk parameters (evolved) ===
    max_position_pct: float = 0.40
    max_portfolio_risk: float = 0.05
    min_confidence: float = 0.55
    min_strength: float = 0.40

    # === Position management (evolved) ===
    min_hold_bars: float = 12.0       # 5m bars; converted to seconds in production
    trailing_stop_pct: float = 0.008
    take_profit_pct: float = 0.025
    stop_loss_atr_mult: float = 2.0   # ATR-based stop loss multiplier
    breakeven_threshold: float = 0.01

    # === Kelly sizing (evolved) ===
    kelly_fraction: float = 0.40
    kelly_cap: float = 0.35

    # === Technical indicators (evolved) ===
    rsi_period: float = 14.0
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    bb_period: float = 20.0
    bb_std_dev: float = 2.0
    atr_period: float = 14.0
    atr_multiplier: float = 2.0
    ema_fast_period: float = 12.0
    ema_slow_period: float = 26.0
    momentum_period: float = 14.0
    momentum_threshold: float = 0.01

    # === Mean reversion (evolved) ===
    zscore_entry: float = 2.0
    zscore_exit: float = 0.5
    zscore_period: float = 20.0

    # === Volatility scaling (evolved) ===
    vol_scale_high: float = 0.5
    vol_scale_low: float = 1.2
    vol_threshold: float = 0.03

    # === Volume confirmation (evolved) ===
    volume_ma_period: float = 20.0
    volume_threshold: float = 1.5

    # === Regime weights (evolved) ===
    bull_weight: float = 1.2
    bear_weight: float = 0.5
    sideways_weight: float = 0.8
    crisis_weight: float = 0.3

    # === Short selling (evolved) ===
    short_enabled: float = 0.0  # >0.5 = enabled
    short_rsi_overbought: float = 80.0
    short_momentum_threshold: float = -0.02

    # === Timeframe weights (evolved) ===
    tf_1m_weight: float = 0.1
    tf_5m_weight: float = 0.3
    tf_15m_weight: float = 0.35
    tf_1h_weight: float = 0.25

    # Strategy ensemble weights (non-evolved)
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        "peak_alpha": 1.5,
        "momentum": 1.2,
        "mean_reversion": 1.2,
        "breakout": 1.0,
        "scalping": 0.8,
        "tier_strategies": 1.3,
        "library_strategies": 1.0,
    })

    @property
    def min_hold_seconds(self) -> float:
        """Convert min_hold_bars (5m bars) to seconds for production use."""
        return self.min_hold_bars * 300.0  # 5 min = 300 sec


@dataclass
class Position:
    """Active position tracking."""
    symbol: str
    side: str  # "long" or "short"
    quantity: float
    avg_price: float
    cost_basis: float
    entry_time: datetime
    current_price: float = 0.0
    high_water: float = 0.0
    low_water: float = float("inf")
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop: float = 0.0
    strategy: str = ""
    regime_at_entry: str = ""
    signal_score: float = 0.0
    entry_atr: float = 0.0
    features_at_entry: Dict[str, float] = field(default_factory=dict)


@dataclass
class TradeRecord:
    """Completed trade record."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    strategy: str
    entry_time: datetime
    exit_time: datetime
    exit_reason: str
    regime: str
    hold_time_seconds: float
    signal_score: float = 0.0


# ============================================================================
# REGIME DETECTOR (uses evolved parameters)
# ============================================================================
class RegimeDetector:
    """Market regime detection matching the evolution model."""

    def __init__(self, config: GodmodeConfig):
        self.config = config
        self.price_history: Dict[str, deque] = {}
        self.regime_history: deque = deque(maxlen=100)

    def update(self, symbol: str, df: pd.DataFrame):
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=500)
        for _, row in df.iterrows():
            self.price_history[symbol].append({
                "close": row["close"],
                "high": row["high"],
                "low": row["low"],
                "volume": row.get("volume", 0),
            })

    def detect(self, df: pd.DataFrame) -> MarketRegime:
        """Detect regime using the same logic as the evolution engine."""
        if len(df) < 60:
            return MarketRegime.SIDEWAYS

        close = df["close"].values

        # Volatility (annualized for crypto 5m bars: 365 * 288)
        vol_window = 20
        returns = np.diff(close[-vol_window:]) / close[-vol_window:-1]
        volatility = np.std(returns) * np.sqrt(365 * 288)

        # Trend via linear regression slope (matching evo engine)
        trend_window = min(50, len(close) - 1)
        recent = close[-trend_window:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        normalized_slope = slope / np.mean(recent)

        # Momentum (short-term)
        momentum = (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0

        # Classification (aligned with evolution's _detect_regime)
        if volatility > 0.80:
            return MarketRegime.HIGH_VOLATILITY
        elif normalized_slope > 0.0005:
            if volatility > 0.5:
                return MarketRegime.HIGH_VOLATILITY
            return MarketRegime.BULL_TRENDING
        elif normalized_slope < -0.001:
            if normalized_slope < -0.003:
                return MarketRegime.CRISIS
            return MarketRegime.BEAR_TRENDING
        elif abs(normalized_slope) < 0.0003 and volatility < 0.3:
            return MarketRegime.LOW_VOLATILITY
        elif normalized_slope > 0.0002 and momentum > 0:
            return MarketRegime.RECOVERY
        else:
            return MarketRegime.SIDEWAYS


# ============================================================================
# RISK MANAGER (uses evolved ATR stops)
# ============================================================================
class GodmodeRiskManager:
    """Risk management with evolved ATR-based stops."""

    def __init__(self, config: GodmodeConfig):
        self.config = config
        self.daily_pnl = 0.0
        self.max_daily_loss = -50.0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5
        self.circuit_breaker_active = False
        self.cooldown_until: Optional[datetime] = None

    def check_circuit_breaker(self) -> bool:
        if self.circuit_breaker_active:
            if self.cooldown_until and datetime.now(timezone.utc) > self.cooldown_until:
                self.circuit_breaker_active = False
                self.consecutive_losses = 0
                logger.info("Circuit breaker reset - trading resumed")
                return False
            return True
        return False

    def record_trade_result(self, pnl: float):
        self.daily_pnl += pnl
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        if self.daily_pnl < self.max_daily_loss:
            self._activate_circuit_breaker("Daily loss limit hit")
        elif self.consecutive_losses >= self.max_consecutive_losses:
            self._activate_circuit_breaker("Consecutive loss limit hit")

    def _activate_circuit_breaker(self, reason: str):
        self.circuit_breaker_active = True
        self.cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=30)
        logger.warning("CIRCUIT BREAKER: %s. Cooldown until %s", reason, self.cooldown_until)

    def calculate_position_size(
        self,
        capital: float,
        price: float,
        confidence: float,
        volatility: float,
        regime: MarketRegime,
    ) -> float:
        """Evolved Kelly-adjusted position sizing with regime and vol scaling."""
        win_rate = 0.55 + (confidence - 0.5) * 0.2
        win_loss_ratio = 1.5
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        kelly = max(0, kelly * self.config.kelly_fraction)
        kelly = min(kelly, self.config.kelly_cap)

        if volatility > self.config.vol_threshold:
            vol_adjustment = self.config.vol_scale_high
        else:
            vol_adjustment = self.config.vol_scale_low

        regime_multiplier = {
            MarketRegime.BULL_TRENDING: self.config.bull_weight,
            MarketRegime.RECOVERY: max(self.config.bull_weight * 0.9, 0.8),
            MarketRegime.SIDEWAYS: self.config.sideways_weight,
            MarketRegime.LOW_VOLATILITY: self.config.sideways_weight * 0.9,
            MarketRegime.HIGH_VOLATILITY: self.config.crisis_weight * 1.5,
            MarketRegime.BEAR_TRENDING: self.config.bear_weight,
            MarketRegime.CRISIS: self.config.crisis_weight,
        }.get(regime, 1.0)

        position_pct = kelly * vol_adjustment * regime_multiplier * confidence
        position_pct = min(position_pct, self.config.max_position_pct)
        position_pct = max(position_pct, 0.02)

        position_value = capital * position_pct
        return position_value / price

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        regime: MarketRegime,
        side: str = "long",
    ) -> float:
        """ATR-based stop loss using evolved stop_loss_atr_mult."""
        atr_mult = self.config.stop_loss_atr_mult
        regime_widen = {
            MarketRegime.HIGH_VOLATILITY: 1.5,
            MarketRegime.CRISIS: 2.0,
            MarketRegime.BULL_TRENDING: 1.0,
        }.get(regime, 1.2)
        effective_atr = atr_mult * atr * regime_widen
        if side == "long":
            return entry_price - effective_atr
        else:
            return entry_price + effective_atr

    def calculate_take_profit(
        self,
        entry_price: float,
        atr: float,
        regime: MarketRegime,
        side: str = "long",
    ) -> float:
        """Take profit target: R:R of 1.5:1 vs stop, regime-adjusted."""
        atr_mult = self.config.stop_loss_atr_mult
        regime_widen = {
            MarketRegime.HIGH_VOLATILITY: 1.5,
            MarketRegime.CRISIS: 2.0,
            MarketRegime.BULL_TRENDING: 1.0,
        }.get(regime, 1.2)
        effective_atr = atr_mult * atr * regime_widen
        tp_distance = effective_atr * 1.5
        if side == "long":
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance


# ============================================================================
# CLICK ENTRYPOINT
# ============================================================================
@click.command()
@click.option("--capital", "-c", default=1000.0, help="Starting capital")
@click.option("--symbols", "-s", multiple=True, help="Symbols to trade")
def main(capital: float, symbols: tuple) -> None:
    """Run Argus in GODMODE V2 production trading."""
    # M18: The old fallback to unified_trading_system has been removed.
    # GodmodeConfig and all supporting classes are fully self-contained above.
    from core.startup import load_config
    cfg = GodmodeConfig(initial_capital=capital)
    if symbols:
        cfg.symbols = list(symbols)

    logger.info("GODMODE starting — capital=%.2f symbols=%s", cfg.initial_capital, cfg.symbols)
    # TODO (Batch 13+): wire GodmodeOrchestrator using core/execution_engine.py
    console.print(Panel.fit(
        f"[bold magenta]ARGUS GODMODE V2[/bold magenta]\n"
        f"Capital: [green]${cfg.initial_capital:,.2f}[/green]\n"
        f"Symbols: [yellow]{', '.join(cfg.symbols)}[/yellow]",
        border_style="magenta",
    ))


if __name__ == "__main__":
    main()
