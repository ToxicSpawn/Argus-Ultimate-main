"""
Advanced Strategy Adapter
=========================

Adapts strategies from the strategy_library_impl to work with the
Argus async framework and Signal dataclass.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from core.types import Signal, SignalAction, MarketRegime
from strategies.base import Strategy, StrategyConfig


class StrategyAdapter(Strategy):
    """
    Adapts a strategy_library_impl strategy to the Argus async framework.

    Wraps strategies that have an analyze(market_data) -> dict method
    and converts their output to Signal objects.
    """

    def __init__(self, wrapped_strategy: Any, strategy_name: str, config: Optional[StrategyConfig] = None):
        self._wrapped = wrapped_strategy
        self._strategy_name = strategy_name
        self._config = config or StrategyConfig(name=strategy_name)
        self._state = None

    @property
    def name(self) -> str:
        return self._strategy_name

    @property
    def config(self) -> StrategyConfig:
        return self._config

    @property
    def required_lookback(self) -> int:
        return 100

    async def generate_signal(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        regime: MarketRegime = MarketRegime.UNKNOWN,
        **kwargs,
    ) -> Optional[Signal]:
        """Generate signal by calling wrapped strategy's analyze method."""

        if ohlcv is None or len(ohlcv) < 50:
            return None

        # Build market_data dict expected by strategy_library_impl strategies
        current_price = float(ohlcv["close"].iloc[-1])
        market_data = {
            "symbol": symbol,
            "price": current_price,
            "ohlcv_df": ohlcv,
            "tickers": {symbol: current_price},
        }

        try:
            result = self._wrapped.analyze(market_data)
        except Exception:
            return None

        if not result or not isinstance(result, dict):
            return None

        action_str = str(result.get("action", "")).upper()
        if action_str not in ("BUY", "SELL"):
            return None

        action = SignalAction.BUY if action_str == "BUY" else SignalAction.SELL
        confidence = float(result.get("confidence", 0.5))

        if confidence < self._config.min_confidence:
            return None

        # Calculate stops based on ATR if available
        atr = self._calculate_atr(ohlcv)
        if action == SignalAction.BUY:
            stop_loss = current_price - (atr * 2.0)
            take_profit = current_price + (atr * 3.0)
        else:
            stop_loss = current_price + (atr * 2.0)
            take_profit = current_price - (atr * 3.0)

        signal_id = f"{self._strategy_name[:4]}_{uuid.uuid4().hex[:8]}"

        return Signal(
            signal_id=signal_id,
            symbol=symbol,
            action=action,
            confidence=confidence,
            strength=0.6,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=self._strategy_name,
            reasoning=f"Source: {result.get('source', self._strategy_name)}",
            regime=regime,
            timestamp=datetime.now(timezone.utc),
            metadata={"wrapped_result": result},
        )

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR for stop/take profit levels."""
        if len(df) < period + 1:
            return float(df["close"].iloc[-1]) * 0.02  # Default 2%

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else float(close.iloc[-1]) * 0.02


def create_adapted_strategies() -> list:
    """Create all adapted strategies from strategy_library_impl."""
    from strategies.strategy_library_impl import (
        MomentumStrategy as LibMomentum,
        MeanReversionStrategy as LibMeanReversion,
        TrendFollowingStrategy as LibTrendFollowing,
        RegimeSwitchingStrategy as LibRegimeSwitching,
        CandlestickPatternStrategy as LibCandlestick,
        HighFreqGridStrategy as LibGrid,
        QuantumBreakoutEliteStrategy as LibBreakout,
        StatArbStrategy as LibStatArb,
    )
    from strategies.tier_strategies_impl import (
        AbsoluteTierStrategy,
        OmegaTierStrategy,
        SingularityTierStrategy,
        ParadoxTierStrategy,
        ChronosTierStrategy,
    )
    from strategies.ultimate_comprehensive_strategy import UltimateComprehensiveStrategy

    strategies = []

    # Core strategies
    strategies.append(StrategyAdapter(
        LibMomentum({"fast": 12, "slow": 26}),
        "lib_momentum",
        StrategyConfig(name="lib_momentum", min_confidence=0.50)
    ))

    strategies.append(StrategyAdapter(
        LibMeanReversion({"window": 40, "entry_z": 1.8}),
        "lib_mean_reversion",
        StrategyConfig(name="lib_mean_reversion", min_confidence=0.50)
    ))

    strategies.append(StrategyAdapter(
        LibTrendFollowing({"fast": 15, "slow": 45}),
        "lib_trend",
        StrategyConfig(name="lib_trend", min_confidence=0.50)
    ))

    strategies.append(StrategyAdapter(
        LibCandlestick({}),
        "candlestick",
        StrategyConfig(name="candlestick", min_confidence=0.55)
    ))

    strategies.append(StrategyAdapter(
        LibGrid({"window": 40, "grid_spacing_pct": 0.004}),
        "hf_grid",
        StrategyConfig(name="hf_grid", min_confidence=0.50)
    ))

    strategies.append(StrategyAdapter(
        LibBreakout({"lookback": 80}),
        "breakout_elite",
        StrategyConfig(name="breakout_elite", min_confidence=0.55)
    ))

    strategies.append(StrategyAdapter(
        LibRegimeSwitching({"vol_threshold": 0.015}),
        "regime_switch",
        StrategyConfig(name="regime_switch", min_confidence=0.50)
    ))

    # Tier ensemble strategies
    strategies.append(StrategyAdapter(
        AbsoluteTierStrategy({"consensus": 1}),
        "absolute_tier",
        StrategyConfig(name="absolute_tier", min_confidence=0.55)
    ))

    strategies.append(StrategyAdapter(
        OmegaTierStrategy({"consensus": 2}),
        "omega_tier",
        StrategyConfig(name="omega_tier", min_confidence=0.60)
    ))

    strategies.append(StrategyAdapter(
        SingularityTierStrategy({"consensus": 2}),
        "singularity_tier",
        StrategyConfig(name="singularity_tier", min_confidence=0.60)
    ))

    strategies.append(StrategyAdapter(
        ParadoxTierStrategy({"consensus": 1}),
        "paradox_tier",
        StrategyConfig(name="paradox_tier", min_confidence=0.50)
    ))

    strategies.append(StrategyAdapter(
        ChronosTierStrategy({"consensus": 1}),
        "chronos_tier",
        StrategyConfig(name="chronos_tier", min_confidence=0.50)
    ))

    # Ultimate strategy
    strategies.append(StrategyAdapter(
        UltimateComprehensiveStrategy({}),
        "ultimate",
        StrategyConfig(name="ultimate", min_confidence=0.50)
    ))

    return strategies
