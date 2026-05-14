"""
Argus Trading System - Peak Alpha Strategy
==========================================

Ultimate multi-factor strategy combining:
- Momentum with volume confirmation
- Mean reversion at extreme levels
- Order flow imbalance detection
- Volatility regime adaptation
- Multi-timeframe confluence

This is the highest-performance strategy designed for maximum alpha generation.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.types import (
    Signal,
    SignalAction,
    MarketRegime,
)
from strategies.base import Strategy, StrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class PeakAlphaConfig(StrategyConfig):
    """Configuration for Peak Alpha strategy."""
    name: str = "peak_alpha"

    # Multi-factor weights
    momentum_weight: float = 0.30
    mean_reversion_weight: float = 0.25
    volume_weight: float = 0.20
    volatility_weight: float = 0.15
    trend_weight: float = 0.10

    # Momentum parameters
    rsi_period: int = 14
    rsi_oversold: float = 25.0
    rsi_overbought: float = 75.0
    macd_fast: int = 8
    macd_slow: int = 21
    macd_signal: int = 5

    # Mean reversion parameters
    bb_period: int = 20
    bb_std: float = 2.0
    zscore_threshold: float = 1.8
    extreme_zscore: float = 2.5

    # Volume parameters
    volume_spike_threshold: float = 2.0
    volume_dry_threshold: float = 0.5

    # Volatility parameters
    atr_period: int = 14
    volatility_expansion_mult: float = 1.5

    # Trend parameters
    ema_fast: int = 9
    ema_medium: int = 21
    ema_slow: int = 55
    adx_period: int = 14
    adx_trend_threshold: float = 25.0

    # Signal thresholds
    min_confidence: float = 0.55
    min_strength: float = 0.40
    min_factor_agreement: int = 3

    # Risk parameters
    stop_loss_atr_mult: float = 1.5
    take_profit_atr_mult: float = 3.0

    # Regime preferences
    preferred_regimes: List[MarketRegime] = field(default_factory=lambda: [
        MarketRegime.TREND_UP,
        MarketRegime.TREND_DOWN,
        MarketRegime.RANGE,
        MarketRegime.LOW_VOL,
        MarketRegime.HIGH_VOL,
    ])


class PeakAlphaStrategy(Strategy):
    """
    Peak Alpha - Multi-factor strategy for maximum returns.

    Combines multiple proven factors:
    1. Momentum: RSI + MACD for direction
    2. Mean Reversion: Bollinger + Z-score for extremes
    3. Volume: Confirms genuine moves — direction from price action, not RSI
    4. Volatility: Adapts position sizing and timing
    5. Trend: EMA alignment + ADX strength

    Entry only when multiple factors align (confluence).
    """

    def __init__(self, config: Optional[PeakAlphaConfig] = None) -> None:
        self._config = config or PeakAlphaConfig()
        self._state = None

    @property
    def name(self) -> str:
        return "peak_alpha"

    @property
    def config(self) -> PeakAlphaConfig:
        return self._config

    @property
    def required_lookback(self) -> int:
        return max(
            self._config.ema_slow + 10,
            self._config.bb_period + 10,
            self._config.atr_period + 10,
            100,
        )

    async def generate_signal(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        regime: MarketRegime = MarketRegime.UNKNOWN,
        **kwargs,
    ) -> Optional[Signal]:
        """Generate peak alpha signal using multi-factor analysis."""

        if len(ohlcv) < self.required_lookback:
            return None

        config = self._config

        close = ohlcv["close"].astype(float)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)
        volume = ohlcv["volume"].astype(float) if "volume" in ohlcv.columns else pd.Series([1.0] * len(close))

        current_price = float(close.iloc[-1])

        indicators = self._calculate_indicators(close, high, low, volume)

        momentum_signal, momentum_score = self._analyze_momentum(indicators)
        reversion_signal, reversion_score = self._analyze_mean_reversion(indicators, current_price)
        volume_signal, volume_score = self._analyze_volume(indicators, close)
        volatility_signal, volatility_score = self._analyze_volatility(indicators, regime)
        trend_signal, trend_score = self._analyze_trend(indicators, current_price)

        signals = [momentum_signal, reversion_signal, volume_signal, trend_signal]
        buy_count = sum(1 for s in signals if s == SignalAction.BUY)
        sell_count = sum(1 for s in signals if s == SignalAction.SELL)

        action = None
        if buy_count >= config.min_factor_agreement:
            action = SignalAction.BUY
        elif sell_count >= config.min_factor_agreement:
            action = SignalAction.SELL

        if action is None:
            return None

        weights = [
            config.momentum_weight,
            config.mean_reversion_weight,
            config.volume_weight,
            config.volatility_weight,
            config.trend_weight,
        ]
        scores = [momentum_score, reversion_score, volume_score, volatility_score, trend_score]

        confidence = sum(w * s for w, s in zip(weights, scores))
        strength = max(buy_count, sell_count) / len(signals)

        agreement_bonus = (max(buy_count, sell_count) - config.min_factor_agreement) * 0.1
        confidence = min(1.0, confidence + agreement_bonus)

        regime_mult = self._get_regime_multiplier(regime, action)
        confidence *= regime_mult

        if confidence < config.min_confidence or strength < config.min_strength:
            return None

        atr = float(indicators["atr"].iloc[-1])
        if action == SignalAction.BUY:
            stop_loss = current_price - (atr * config.stop_loss_atr_mult)
            take_profit = current_price + (atr * config.take_profit_atr_mult)
        else:
            stop_loss = current_price + (atr * config.stop_loss_atr_mult)
            take_profit = current_price - (atr * config.take_profit_atr_mult)

        factors = []
        if momentum_signal == action:
            factors.append(f"Momentum({momentum_score:.0%})")
        if reversion_signal == action:
            factors.append(f"MeanRev({reversion_score:.0%})")
        if volume_signal == action:
            factors.append(f"Volume({volume_score:.0%})")
        if trend_signal == action:
            factors.append(f"Trend({trend_score:.0%})")

        reasoning = f"Peak Alpha: {' + '.join(factors)} | {max(buy_count, sell_count)}/{len(signals)} factors agree"
        signal_id = f"peak_{uuid.uuid4().hex[:8]}"

        return Signal(
            signal_id=signal_id,
            symbol=symbol,
            action=action,
            confidence=confidence,
            strength=strength,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=self.name,
            reasoning=reasoning,
            regime=regime,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "momentum_score": momentum_score,
                "reversion_score": reversion_score,
                "volume_score": volume_score,
                "volatility_score": volatility_score,
                "trend_score": trend_score,
                "buy_factors": buy_count,
                "sell_factors": sell_count,
                "atr": atr,
                "rsi": float(indicators["rsi"].iloc[-1]),
                "zscore": float(indicators["zscore"].iloc[-1]),
            },
        )

    def _calculate_indicators(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
    ) -> Dict[str, pd.Series]:
        config = self._config

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=config.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=config.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))

        ema_fast = close.ewm(span=config.macd_fast, adjust=False).mean()
        ema_slow_macd = close.ewm(span=config.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow_macd
        macd_signal_line = macd_line.ewm(span=config.macd_signal, adjust=False).mean()
        macd_hist = macd_line - macd_signal_line

        bb_middle = close.rolling(config.bb_period).mean()
        bb_std = close.rolling(config.bb_period).std()
        bb_upper = bb_middle + (bb_std * config.bb_std)
        bb_lower = bb_middle - (bb_std * config.bb_std)
        zscore = (close - bb_middle) / bb_std.replace(0, np.inf)

        volume_ma = volume.rolling(20).mean()
        volume_ratio = volume / volume_ma.replace(0, 1)

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=config.atr_period, adjust=False).mean()

        ema_fast_trend = close.ewm(span=config.ema_fast, adjust=False).mean()
        ema_medium = close.ewm(span=config.ema_medium, adjust=False).mean()
        ema_slow_trend = close.ewm(span=config.ema_slow, adjust=False).mean()

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        atr_14 = tr.ewm(span=14, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr_14)
        minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr_14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
        adx = dx.ewm(span=14, adjust=False).mean()

        return {
            "rsi": rsi,
            "macd_line": macd_line,
            "macd_signal": macd_signal_line,
            "macd_hist": macd_hist,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
            "zscore": zscore,
            "volume_ratio": volume_ratio,
            "atr": atr,
            "ema_fast": ema_fast_trend,
            "ema_medium": ema_medium,
            "ema_slow": ema_slow_trend,
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
        }

    def _analyze_momentum(self, indicators: Dict) -> tuple[Optional[SignalAction], float]:
        config = self._config
        rsi = float(indicators["rsi"].iloc[-1])
        macd_hist = float(indicators["macd_hist"].iloc[-1])
        macd_hist_prev = float(indicators["macd_hist"].iloc[-2])

        score = 0.5
        signal = None

        if rsi < config.rsi_oversold:
            score += 0.25
            signal = SignalAction.BUY
        elif rsi > config.rsi_overbought:
            score += 0.25
            signal = SignalAction.SELL
        elif rsi < 40:
            score += 0.1
            signal = SignalAction.BUY
        elif rsi > 60:
            score += 0.1
            signal = SignalAction.SELL

        if macd_hist > 0 and macd_hist_prev <= 0:
            score += 0.2
            signal = SignalAction.BUY
        elif macd_hist < 0 and macd_hist_prev >= 0:
            score += 0.2
            signal = SignalAction.SELL

        score += min(0.1, abs(macd_hist) * 50)

        return signal, min(1.0, score)

    def _analyze_mean_reversion(self, indicators: Dict, price: float) -> tuple[Optional[SignalAction], float]:
        config = self._config
        zscore = float(indicators["zscore"].iloc[-1])
        bb_lower = float(indicators["bb_lower"].iloc[-1])
        bb_upper = float(indicators["bb_upper"].iloc[-1])
        bb_middle = float(indicators["bb_middle"].iloc[-1])

        score = 0.5
        signal = None

        if zscore <= -config.extreme_zscore:
            score += 0.3
            signal = SignalAction.BUY
        elif zscore >= config.extreme_zscore:
            score += 0.3
            signal = SignalAction.SELL
        elif zscore <= -config.zscore_threshold:
            score += 0.2
            signal = SignalAction.BUY
        elif zscore >= config.zscore_threshold:
            score += 0.2
            signal = SignalAction.SELL

        if price <= bb_lower:
            score += 0.15
            signal = SignalAction.BUY
        elif price >= bb_upper:
            score += 0.15
            signal = SignalAction.SELL

        distance_pct = abs(price - bb_middle) / bb_middle
        if distance_pct > 0.02:
            score += min(0.1, distance_pct * 3)

        return signal, min(1.0, score)

    def _analyze_volume(
        self, indicators: Dict, close: pd.Series
    ) -> tuple[Optional[SignalAction], float]:
        """Analyse volume factor.

        Direction is determined by the actual price move on the spike candle
        (close vs prior close), NOT by RSI threshold comparison.
        RSI is still used for score magnitude only.
        """
        config = self._config
        volume_ratio = float(indicators["volume_ratio"].iloc[-1])
        rsi = float(indicators["rsi"].iloc[-1])

        # Price direction from the current candle vs prior close
        price_up = float(close.iloc[-1]) > float(close.iloc[-2])

        score = 0.5
        signal = None

        if volume_ratio >= config.volume_spike_threshold:
            score += 0.3
            # Direction follows actual price movement on the spike
            signal = SignalAction.BUY if price_up else SignalAction.SELL
        elif volume_ratio >= 1.5:
            score += 0.15
            signal = SignalAction.BUY if price_up else SignalAction.SELL
        elif volume_ratio < config.volume_dry_threshold:
            score -= 0.1  # Low volume = weak signal

        # RSI adjusts score magnitude only (not direction)
        if signal == SignalAction.BUY and rsi < 50:
            score += 0.05   # Stronger conviction — oversold + volume spike up
        elif signal == SignalAction.SELL and rsi > 50:
            score += 0.05   # Stronger conviction — overbought + volume spike down

        return signal, max(0.3, min(1.0, score))

    def _analyze_volatility(self, indicators: Dict, regime: MarketRegime) -> tuple[Optional[SignalAction], float]:
        atr = indicators["atr"]
        current_atr = float(atr.iloc[-1])
        avg_atr = float(atr.rolling(50).mean().iloc[-1])

        score = 0.5

        if current_atr > avg_atr * 1.5:
            score += 0.2
        elif current_atr < avg_atr * 0.7:
            score += 0.1

        if regime == MarketRegime.HIGH_VOL:
            score += 0.1
        elif regime == MarketRegime.LOW_VOL:
            score += 0.15

        return None, min(1.0, score)

    def _analyze_trend(self, indicators: Dict, price: float) -> tuple[Optional[SignalAction], float]:
        config = self._config
        ema_fast = float(indicators["ema_fast"].iloc[-1])
        ema_medium = float(indicators["ema_medium"].iloc[-1])
        ema_slow = float(indicators["ema_slow"].iloc[-1])
        adx = float(indicators["adx"].iloc[-1])
        plus_di = float(indicators["plus_di"].iloc[-1])
        minus_di = float(indicators["minus_di"].iloc[-1])

        score = 0.5
        signal = None

        if ema_fast > ema_medium > ema_slow:
            score += 0.2
            signal = SignalAction.BUY
        elif ema_fast < ema_medium < ema_slow:
            score += 0.2
            signal = SignalAction.SELL

        if price > ema_fast > ema_medium:
            score += 0.1
            signal = SignalAction.BUY
        elif price < ema_fast < ema_medium:
            score += 0.1
            signal = SignalAction.SELL

        if adx > config.adx_trend_threshold:
            score += 0.15
            if plus_di > minus_di:
                signal = SignalAction.BUY
            else:
                signal = SignalAction.SELL

        return signal, min(1.0, score)

    def _get_regime_multiplier(self, regime: MarketRegime, action: SignalAction) -> float:
        if regime == MarketRegime.TREND_UP and action == SignalAction.BUY:
            return 1.15
        if regime == MarketRegime.TREND_DOWN and action == SignalAction.SELL:
            return 1.15
        if regime == MarketRegime.RANGE:
            return 1.0
        if regime == MarketRegime.HIGH_VOL:
            return 0.9
        return 1.0


def create_peak_alpha_strategy(**kwargs) -> PeakAlphaStrategy:
    config = PeakAlphaConfig(**kwargs)
    return PeakAlphaStrategy(config)
