"""
Maximum Aggression Scalping Strategy
=====================================
Designed for 70-150% monthly returns.

Features:
- Ultra-fast signal generation (5m candles)
- Tight stops, quick profits
- High win rate targeting
- Multiple entry signals
- Dynamic position sizing
- Session-aware trading
"""

import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4
    EXTREME = 5


@dataclass
class ScalpSignal:
    """Scalping signal."""
    symbol: str
    direction: str  # "long" or "short"
    strength: SignalStrength
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reasons: List[str]
    timestamp: float = field(default_factory=time.time)
    expected_hold_minutes: float = 15.0


class MomentumScalper:
    """
    Momentum Scalper
    ================
    Captures quick momentum moves with tight risk.
    """
    
    def __init__(self):
        self.lookback = 20
        self.momentum_threshold = 0.002  # 0.2%
        self.volume_multiplier = 1.5
        
    def calculate_signals(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        symbol: str
    ) -> List[ScalpSignal]:
        """Calculate momentum scalp signals."""
        if len(closes) < self.lookback + 10:
            return []
        
        signals = []
        
        # Calculate indicators
        ema_fast = self._ema(closes, 9)
        ema_slow = self._ema(closes, 21)
        rsi = self._rsi(closes, 14)
        atr = self._atr(highs, lows, closes, 14)
        volume_sma = self._sma(volumes, 20)
        
        # Current values
        current_idx = -1
        current_price = closes[current_idx]
        current_rsi = rsi[current_idx]
        current_atr = atr[current_idx]
        current_volume = volumes[current_idx]
        avg_volume = volume_sma[current_idx]
        
        # EMA crossover signal
        ema_cross_bullish = (ema_fast[current_idx] > ema_slow[current_idx] and 
                            ema_fast[current_idx-1] <= ema_slow[current_idx-1])
        ema_cross_bearish = (ema_fast[current_idx] < ema_slow[current_idx] and 
                            ema_fast[current_idx-1] >= ema_slow[current_idx-1])
        
        # Volume spike
        volume_spike = current_volume > avg_volume * self.volume_multiplier
        
        # RSI conditions
        rsi_oversold = current_rsi < 35
        rsi_overbought = current_rsi > 65
        
        # Momentum
        momentum = (closes[current_idx] - closes[current_idx - 5]) / closes[current_idx - 5]
        
        # LONG SIGNAL
        if ema_cross_bullish and (rsi_oversold or volume_spike) and momentum > -0.01:
            strength = SignalStrength.STRONG if volume_spike else SignalStrength.MODERATE
            signals.append(ScalpSignal(
                symbol=symbol,
                direction="long",
                strength=strength,
                entry_price=current_price,
                stop_loss=current_price - current_atr * 1.5,
                take_profit=current_price + current_atr * 2.5,
                confidence=0.65 if volume_spike else 0.55,
                reasons=["EMA cross bullish", "RSI oversold" if rsi_oversold else "Volume spike"],
                expected_hold_minutes=15
            ))
        
        # SHORT SIGNAL
        if ema_cross_bearish and (rsi_overbought or volume_spike) and momentum < 0.01:
            strength = SignalStrength.STRONG if volume_spike else SignalStrength.MODERATE
            signals.append(ScalpSignal(
                symbol=symbol,
                direction="short",
                strength=strength,
                entry_price=current_price,
                stop_loss=current_price + current_atr * 1.5,
                take_profit=current_price - current_atr * 2.5,
                confidence=0.65 if volume_spike else 0.55,
                reasons=["EMA cross bearish", "RSI overbought" if rsi_overbought else "Volume spike"],
                expected_hold_minutes=15
            ))
        
        # MEAN REVERSION SIGNAL (counter-trend)
        if rsi_oversold and momentum < -0.02:  # Oversold + big drop = bounce coming
            signals.append(ScalpSignal(
                symbol=symbol,
                direction="long",
                strength=SignalStrength.MODERATE,
                entry_price=current_price,
                stop_loss=current_price - current_atr * 1.0,
                take_profit=current_price + current_atr * 1.5,
                confidence=0.50,
                reasons=["RSI oversold", "Mean reversion"],
                expected_hold_minutes=10
            ))
        
        if rsi_overbought and momentum > 0.02:  # Overbought + big rise = pullback coming
            signals.append(ScalpSignal(
                symbol=symbol,
                direction="short",
                strength=SignalStrength.MODERATE,
                entry_price=current_price,
                stop_loss=current_price + current_atr * 1.0,
                take_profit=current_price - current_atr * 1.5,
                confidence=0.50,
                reasons=["RSI overbought", "Mean reversion"],
                expected_hold_minutes=10
            ))
        
        return signals
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average."""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average."""
        return np.convolve(data, np.ones(period)/period, mode='same')
    
    def _rsi(self, data: np.ndarray, period: int) -> np.ndarray:
        """Relative Strength Index."""
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.zeros(len(data))
        avg_loss = np.zeros(len(data))
        
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        
        for i in range(period + 1, len(data)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
        """Average True Range."""
        tr = np.zeros(len(closes))
        tr[0] = highs[0] - lows[0]
        
        for i in range(1, len(closes)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
        
        atr = np.zeros(len(closes))
        atr[period] = np.mean(tr[:period])
        
        for i in range(period + 1, len(closes)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        return atr


class BreakoutScalper:
    """
    Breakout Scalper
    ================
    Trades breakouts with volume confirmation.
    """
    
    def __init__(self):
        self.lookback = 20
        self.breakout_threshold = 0.001  # 0.1%
        
    def calculate_signals(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        symbol: str
    ) -> List[ScalpSignal]:
        """Calculate breakout signals."""
        if len(closes) < self.lookback + 5:
            return []
        
        signals = []
        current_idx = -1
        current_price = closes[current_idx]
        
        # Calculate levels
        recent_high = np.max(highs[-self.lookback:-1])
        recent_low = np.min(lows[-self.lookback:-1])
        range_size = recent_high - recent_low
        
        # Volume analysis
        volume_sma = np.mean(volumes[-20:-1])
        current_volume = volumes[current_idx]
        volume_ratio = current_volume / volume_sma if volume_sma > 0 else 1
        
        # ATR for stops
        atr = self._atr(highs, lows, closes, 14)
        current_atr = atr[current_idx]
        
        # Breakout detection
        breakout_up = closes[current_idx] > recent_high * (1 + self.breakout_threshold)
        breakout_down = closes[current_idx] < recent_low * (1 - self.breakout_threshold)
        
        # Volume confirmation
        volume_confirm = volume_ratio > 1.5
        
        if breakout_up and volume_confirm:
            signals.append(ScalpSignal(
                symbol=symbol,
                direction="long",
                strength=SignalStrength.STRONG,
                entry_price=current_price,
                stop_loss=recent_high - current_atr * 0.5,
                take_profit=current_price + current_atr * 3,
                confidence=0.70,
                reasons=["Breakout above resistance", f"Volume {volume_ratio:.1f}x average"],
                expected_hold_minutes=20
            ))
        
        if breakout_down and volume_confirm:
            signals.append(ScalpSignal(
                symbol=symbol,
                direction="short",
                strength=SignalStrength.STRONG,
                entry_price=current_price,
                stop_loss=recent_low + current_atr * 0.5,
                take_profit=current_price - current_atr * 3,
                confidence=0.70,
                reasons=["Breakout below support", f"Volume {volume_ratio:.1f}x average"],
                expected_hold_minutes=20
            ))
        
        return signals
    
    def _atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
        """Average True Range."""
        tr = np.zeros(len(closes))
        tr[0] = highs[0] - lows[0]
        
        for i in range(1, len(closes)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
        
        atr = np.zeros(len(closes))
        atr[period] = np.mean(tr[:period])
        
        for i in range(period + 1, len(closes)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        return atr


class FundingRateScalper:
    """
    Funding Rate Scalper
    ====================
    Trades funding rate extremes.
    """
    
    def __init__(self):
        self.high_funding_threshold = 0.001  # 0.1% per 8h
        self.low_funding_threshold = -0.0005
        
    def calculate_signal(
        self,
        symbol: str,
        funding_rate: float,
        predicted_rate: float,
        current_price: float,
        long_oi: float,
        short_oi: float
    ) -> Optional[ScalpSignal]:
        """Calculate funding rate scalp signal."""
        
        # High positive funding = crowded longs = short opportunity
        if funding_rate > self.high_funding_threshold:
            ls_ratio = long_oi / short_oi if short_oi > 0 else 2.0
            
            if ls_ratio > 1.5:  # Crowded longs
                return ScalpSignal(
                    symbol=symbol,
                    direction="short",
                    strength=SignalStrength.STRONG,
                    entry_price=current_price,
                    stop_loss=current_price * 1.015,  # 1.5% stop
                    take_profit=current_price * 0.97,  # 3% target
                    confidence=0.72,
                    reasons=[
                        f"High funding: {funding_rate*100:.3f}%",
                        f"L/S ratio: {ls_ratio:.2f}",
                        "Contrarian short"
                    ],
                    expected_hold_minutes=240  # Hold for funding periods
                )
        
        # High negative funding = crowded shorts = long opportunity
        if funding_rate < self.low_funding_threshold:
            ls_ratio = long_oi / short_oi if short_oi > 0 else 0.5
            
            if ls_ratio < 0.7:  # Crowded shorts
                return ScalpSignal(
                    symbol=symbol,
                    direction="long",
                    strength=SignalStrength.STRONG,
                    entry_price=current_price,
                    stop_loss=current_price * 0.985,  # 1.5% stop
                    take_profit=current_price * 1.03,  # 3% target
                    confidence=0.72,
                    reasons=[
                        f"Low funding: {funding_rate*100:.3f}%",
                        f"L/S ratio: {ls_ratio:.2f}",
                        "Contrarian long"
                    ],
                    expected_hold_minutes=240
                )
        
        return None


class AggressiveSignalAggregator:
    """
    Aggressive Signal Aggregator
    =============================
    Combines multiple scalping strategies for maximum signals.
    """
    
    def __init__(self):
        self.momentum_scalper = MomentumScalper()
        self.breakout_scalper = BreakoutScalper()
        self.funding_scalper = FundingRateScalper()
        
        self.min_confidence = 0.40  # Low threshold for more trades
        self.max_positions = 10
        
    def generate_signals(
        self,
        symbol: str,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        funding_rate: float = 0.0,
        long_oi: float = 0,
        short_oi: float = 0
    ) -> List[ScalpSignal]:
        """Generate all signals for a symbol."""
        all_signals = []
        
        # Momentum signals
        momentum_signals = self.momentum_scalper.calculate_signals(
            closes, highs, lows, volumes, symbol
        )
        signals.extend(momentum_signals)
        
        # Breakout signals
        breakout_signals = self.breakout_scalper.calculate_signals(
            closes, highs, lows, volumes, symbol
        )
        signals.extend(breakout_signals)
        
        # Funding rate signal
        if funding_rate != 0:
            funding_signal = self.funding_scalper.calculate_signal(
                symbol, funding_rate, funding_rate, closes[-1], long_oi, short_oi
            )
            if funding_signal:
                signals.append(funding_signal)
        
        # Filter by confidence
        filtered = [s for s in signals if s.confidence >= self.min_confidence]
        
        # Sort by confidence
        filtered.sort(key=lambda s: s.confidence, reverse=True)
        
        return filtered[:self.max_positions]
    
    def calculate_position_size(
        self,
        signal: ScalpSignal,
        capital: float,
        max_risk_pct: float = 0.02  # 2% risk per trade
    ) -> float:
        """Calculate position size based on risk."""
        risk_amount = capital * max_risk_pct
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        
        if stop_distance == 0:
            return 0
        
        position_size = risk_amount / stop_distance
        
        # Cap at 25% of capital
        max_position = capital * 0.25 / signal.entry_price
        position_size = min(position_size, max_position)
        
        # Adjust for confidence
        position_size *= signal.confidence
        
        return position_size


# Export
__all__ = [
    "MomentumScalper",
    "BreakoutScalper",
    "FundingRateScalper",
    "AggressiveSignalAggregator",
    "ScalpSignal",
    "SignalStrength"
]
