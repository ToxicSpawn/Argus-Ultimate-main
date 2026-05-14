"""
ULTIMATE STRATEGY LIBRARY - Every Edge for Crypto Trading
===========================================================
Combines 50+ proven strategies from research and open source.

Categories:
1. Trend Following (8 strategies)
2. Mean Reversion (6 strategies)
3. Momentum (6 strategies)
4. Breakout (5 strategies)
5. Volume Analysis (5 strategies)
6. On-Chain Analysis (6 strategies)
7. Sentiment Analysis (5 strategies)
8. Arbitrage (5 strategies)
9. Options/Derivatives (4 strategies)
10. ML/AI Enhanced (5 strategies)

Each strategy generates signals with confidence scores.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import math

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal types."""
    STRONG_BUY = 4
    BUY = 3
    NEUTRAL = 2
    SELL = 1
    STRONG_SELL = 0


@dataclass
class StrategySignal:
    """Strategy signal output."""
    strategy_name: str
    symbol: str
    signal: SignalType
    confidence: float  # 0-1
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# TREND FOLLOWING STRATEGIES
# ============================================================================

class TrendFollowing:
    """Trend following strategies - ride the trend."""
    
    @staticmethod
    def sma_crossover(closes: np.ndarray, fast: int = 20, slow: int = 50) -> Tuple[float, str]:
        """SMA Crossover - Classic trend following."""
        if len(closes) < slow + 1:
            return 0, "neutral"
        
        fast_sma = np.mean(closes[-fast:])
        slow_sma = np.mean(closes[-slow:])
        prev_fast = np.mean(closes[-fast-1:-1])
        prev_slow = np.mean(closes[-slow-1:-1])
        
        # Bullish crossover
        if fast_sma > slow_sma and prev_fast <= prev_slow:
            return 0.75, "bullish"
        # Bearish crossover
        elif fast_sma < slow_sma and prev_fast >= prev_slow:
            return 0.75, "bearish"
        # Trend continuation
        elif fast_sma > slow_sma:
            return 0.5, "bullish"
        elif fast_sma < slow_sma:
            return 0.5, "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def ema_crossover(closes: np.ndarray, fast: int = 12, slow: int = 26) -> Tuple[float, str]:
        """EMA Crossover - Faster than SMA."""
        if len(closes) < slow + 1:
            return 0, "neutral"
        
        # Calculate EMAs
        fast_ema = closes[-1]
        slow_ema = closes[-1]
        alpha_fast = 2 / (fast + 1)
        alpha_slow = 2 / (slow + 1)
        
        for i in range(2, min(fast + 2, len(closes))):
            fast_ema = alpha_fast * closes[-i] + (1 - alpha_fast) * fast_ema
        
        for i in range(2, min(slow + 2, len(closes))):
            slow_ema = alpha_slow * closes[-i] + (1 - alpha_slow) * slow_ema
        
        if fast_ema > slow_ema:
            strength = min((fast_ema - slow_ema) / slow_ema * 100, 1)
            return 0.5 + strength * 0.3, "bullish"
        else:
            strength = min((slow_ema - fast_ema) / slow_ema * 100, 1)
            return 0.5 + strength * 0.3, "bearish"
    
    @staticmethod
    def macd(closes: np.ndarray) -> Tuple[float, str]:
        """MACD - Momentum trend indicator."""
        if len(closes) < 35:
            return 0, "neutral"
        
        # MACD calculation
        ema12 = closes[-1]
        ema26 = closes[-1]
        alpha12 = 2 / 13
        alpha26 = 2 / 27
        
        for i in range(1, min(26, len(closes))):
            if i < 13:
                ema12 = alpha12 * closes[-(i+1)] + (1 - alpha12) * ema12
            ema26 = alpha26 * closes[-(i+1)] + (1 - alpha26) * ema26
        
        macd_line = ema12 - ema26
        
        # Signal line (9-period EMA of MACD)
        # Simplified
        if macd_line > 0:
            return 0.6, "bullish"
        else:
            return 0.6, "bearish"
    
    @staticmethod
    def parabolic_sar(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
                      af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2) -> Tuple[float, str]:
        """Parabolic SAR - Trailing stop indicator."""
        if len(closes) < 10:
            return 0, "neutral"
        
        # Simplified SAR
        sar = lows[-1]
        ep = highs[-1]
        af = af_start
        
        trend = 1  # 1 = up, -1 = down
        
        for i in range(-2, -min(20, len(closes)), -1):
            if closes[i] > ep:
                ep = closes[i]
                af = min(af + af_step, af_max)
            elif closes[i] < sar:
                trend = -1
                sar = ep
                af = af_start
        
        if closes[-1] > sar:
            return 0.7, "bullish"
        else:
            return 0.7, "bearish"
    
    @staticmethod
    def adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> Tuple[float, str]:
        """ADX - Trend strength indicator."""
        if len(closes) < period + 1:
            return 0, "neutral"
        
        # Calculate True Range
        tr = []
        for i in range(1, len(closes)):
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            ))
        
        # Simplified ADX
        avg_tr = np.mean(tr[-period:])
        
        # Directional movement
        up_move = highs[-1] - highs[-2]
        down_move = lows[-2] - lows[-1]
        
        if up_move > down_move and up_move > 0:
            plus_di = up_move / avg_tr if avg_tr > 0 else 0
            minus_di = 0
        elif down_move > up_move and down_move > 0:
            minus_di = down_move / avg_tr if avg_tr > 0 else 0
            plus_di = 0
        else:
            plus_di = 0
            minus_di = 0
        
        # ADX value (simplified)
        adx_value = abs(plus_di - minus_di) * 100
        
        if adx_value > 25:
            if plus_di > minus_di:
                return min(adx_value / 50, 1), "bullish"
            else:
                return min(adx_value / 50, 1), "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def ichimoku_cloud(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Tuple[float, str]:
        """Ichimoku Cloud - Complete trend system."""
        if len(closes) < 52:
            return 0, "neutral"
        
        # Tenkan-sen (9-period)
        tenkan = (np.max(highs[-9:]) + np.min(lows[-9:])) / 2
        
        # Kijun-sen (26-period)
        kijun = (np.max(highs[-26:]) + np.min(lows[-26:])) / 2
        
        # Senkou Span A
        senkou_a = (tenkan + kijun) / 2
        
        # Senkou Span B
        senkou_b = (np.max(highs[-52:]) + np.min(lows[-52:])) / 2
        
        # Current price vs cloud
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        current_price = closes[-1]
        
        if current_price > cloud_top:
            strength = (current_price - cloud_top) / cloud_top
            return min(0.6 + strength * 2, 1), "bullish"
        elif current_price < cloud_bottom:
            strength = (cloud_bottom - current_price) / cloud_bottom
            return min(0.6 + strength * 2, 1), "bearish"
        
        return 0.3, "neutral"
    
    @staticmethod
    def supertrend(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
                   period: int = 10, multiplier: float = 3.0) -> Tuple[float, str]:
        """Supertrend - ATR-based trend following."""
        if len(closes) < period + 1:
            return 0, "neutral"
        
        # Calculate ATR
        tr = []
        for i in range(1, len(closes)):
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            ))
        
        atr = np.mean(tr[-period:])
        
        # Basic bands
        hl2 = (highs[-1] + lows[-1]) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr
        
        if closes[-1] > upper_band:
            return 0.7, "bullish"
        elif closes[-1] < lower_band:
            return 0.7, "bearish"
        
        return 0.4, "neutral"
    
    @staticmethod
    def keltner_channel(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                        ema_period: int = 20, atr_period: int = 10) -> Tuple[float, str]:
        """Keltner Channel - Volatility-based trend."""
        if len(closes) < max(ema_period, atr_period) + 1:
            return 0, "neutral"
        
        # EMA
        ema = np.mean(closes[-ema_period:])
        
        # ATR
        tr = []
        for i in range(-atr_period, 0):
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            ))
        atr = np.mean(tr)
        
        upper = ema + 2 * atr
        lower = ema - 2 * atr
        
        current = closes[-1]
        
        if current > upper:
            return 0.65, "bullish"
        elif current < lower:
            return 0.65, "bearish"
        
        return 0, "neutral"


# ============================================================================
# MEAN REVERSION STRATEGIES
# ============================================================================

class MeanReversion:
    """Mean reversion strategies - fade the extremes."""
    
    @staticmethod
    def rsi(closes: np.ndarray, period: int = 14, 
            oversold: int = 30, overbought: int = 70) -> Tuple[float, str]:
        """RSI - Relative Strength Index."""
        if len(closes) < period + 1:
            return 0, "neutral"
        
        deltas = np.diff(closes[-(period+10):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        if rsi < oversold:
            strength = (oversold - rsi) / oversold
            return min(0.5 + strength * 0.5, 1), "bullish"
        elif rsi > overbought:
            strength = (rsi - overbought) / (100 - overbought)
            return min(0.5 + strength * 0.5, 1), "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def bollinger_bands(closes: np.ndarray, period: int = 20, 
                        std_dev: float = 2.0) -> Tuple[float, str]:
        """Bollinger Bands - Volatility bands."""
        if len(closes) < period:
            return 0, "neutral"
        
        sma = np.mean(closes[-period:])
        std = np.std(closes[-period:])
        
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        current = closes[-1]
        
        # %B indicator
        pct_b = (current - lower) / (upper - lower) if upper != lower else 0.5
        
        if pct_b < 0:  # Below lower band
            return min(0.5 + abs(pct_b) * 0.5, 1), "bullish"
        elif pct_b > 1:  # Above upper band
            return min(0.5 + (pct_b - 1) * 0.5, 1), "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def stochastic(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                   k_period: int = 14, d_period: int = 3) -> Tuple[float, str]:
        """Stochastic Oscillator."""
        if len(closes) < k_period:
            return 0, "neutral"
        
        lowest_low = np.min(lows[-k_period:])
        highest_high = np.max(highs[-k_period:])
        
        if highest_high == lowest_low:
            k = 50
        else:
            k = ((closes[-1] - lowest_low) / (highest_high - lowest_low)) * 100
        
        if k < 20:
            return 0.7, "bullish"  # Oversold
        elif k > 80:
            return 0.7, "bearish"  # Overbought
        
        return 0, "neutral"
    
    @staticmethod
    def williams_r(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                   period: int = 14) -> Tuple[float, str]:
        """Williams %R - Momentum oscillator."""
        if len(closes) < period:
            return 0, "neutral"
        
        highest_high = np.max(highs[-period:])
        lowest_low = np.min(lows[-period:])
        
        if highest_high == lowest_low:
            wr = -50
        else:
            wr = ((highest_high - closes[-1]) / (highest_high - lowest_low)) * -100
        
        if wr < -80:  # Oversold
            return 0.7, "bullish"
        elif wr > -20:  # Overbought
            return 0.7, "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def cci(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
            period: int = 20) -> Tuple[float, str]:
        """CCI - Commodity Channel Index."""
        if len(closes) < period:
            return 0, "neutral"
        
        typical_price = (highs[-period:] + lows[-period:] + closes[-period:]) / 3
        sma_tp = np.mean(typical_price)
        mad = np.mean(np.abs(typical_price - sma_tp))
        
        if mad == 0:
            cci = 0
        else:
            cci = (typical_price[-1] - sma_tp) / (0.015 * mad)
        
        if cci < -100:
            return min(0.5 + abs(cci + 100) / 200, 1), "bullish"
        elif cci > 100:
            return min(0.5 + (cci - 100) / 200, 1), "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def mean_reversion_zscore(closes: np.ndarray, period: int = 20) -> Tuple[float, str]:
        """Z-Score Mean Reversion."""
        if len(closes) < period:
            return 0, "neutral"
        
        mean = np.mean(closes[-period:])
        std = np.std(closes[-period:])
        
        if std == 0:
            return 0, "neutral"
        
        z_score = (closes[-1] - mean) / std
        
        if z_score < -2:
            return min(0.5 + abs(z_score + 2) * 0.2, 1), "bullish"
        elif z_score > 2:
            return min(0.5 + (z_score - 2) * 0.2, 1), "bearish"
        
        return 0, "neutral"


# ============================================================================
# MOMENTUM STRATEGIES
# ============================================================================

class Momentum:
    """Momentum strategies - follow the strength."""
    
    @staticmethod
    def roc(closes: np.ndarray, period: int = 12) -> Tuple[float, str]:
        """ROC - Rate of Change."""
        if len(closes) < period + 1:
            return 0, "neutral"
        
        roc = (closes[-1] - closes[-period-1]) / closes[-period-1] * 100
        
        if roc > 5:
            return min(0.5 + roc / 20, 1), "bullish"
        elif roc < -5:
            return min(0.5 + abs(roc) / 20, 1), "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def momentum(closes: np.ndarray, period: int = 10) -> Tuple[float, str]:
        """Simple Momentum."""
        if len(closes) < period + 1:
            return 0, "neutral"
        
        momentum = closes[-1] - closes[-period-1]
        normalized = momentum / closes[-period-1] * 100
        
        if normalized > 2:
            return min(0.5 + normalized / 10, 1), "bullish"
        elif normalized < -2:
            return min(0.5 + abs(normalized) / 10, 1), "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def cmo(closes: np.ndarray, period: int = 14) -> Tuple[float, str]:
        """CMO - Chande Momentum Oscillator."""
        if len(closes) < period + 1:
            return 0, "neutral"
        
        deltas = np.diff(closes[-(period+1):])
        sum_up = np.sum(deltas[deltas > 0])
        sum_down = abs(np.sum(deltas[deltas < 0]))
        
        if sum_up + sum_down == 0:
            cmo = 0
        else:
            cmo = (sum_up - sum_down) / (sum_up + sum_down) * 100
        
        if cmo > 50:
            return min(0.5 + (cmo - 50) / 100, 1), "bullish"
        elif cmo < -50:
            return min(0.5 + abs(cmo + 50) / 100, 1), "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def tsi(closes: np.ndarray, long: int = 25, short: int = 13) -> Tuple[float, str]:
        """TSI - True Strength Index."""
        if len(closes) < long + short:
            return 0, "neutral"
        
        # Double smoothed momentum
        momentum = np.diff(closes)
        
        # First EMA
        ema1 = np.convolve(momentum, np.ones(short)/short, mode='valid')
        # Second EMA
        ema2 = np.convolve(ema1, np.ones(long)/long, mode='valid')
        
        if len(ema2) > 0 and ema2[-1] > 0:
            return 0.6, "bullish"
        elif len(ema2) > 0 and ema2[-1] < 0:
            return 0.6, "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def uo(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
           fast: int = 7, mid: int = 14, slow: int = 28) -> Tuple[float, str]:
        """Ultimate Oscillator."""
        if len(closes) < slow + 1:
            return 0, "neutral"
        
        # Calculate buying pressure
        bp = closes[-1] - lows[-1]
        tr = highs[-1] - lows[-1]
        
        if tr == 0:
            return 0, "neutral"
        
        avg7 = bp / tr if tr > 0 else 0
        avg14 = bp / tr if tr > 0 else 0
        avg28 = bp / tr if tr > 0 else 0
        
        uo = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
        
        if uo > 65:
            return min(0.5 + (uo - 65) / 70, 1), "bullish"
        elif uo < 35:
            return min(0.5 + (35 - uo) / 70, 1), "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def ao(highs: np.ndarray, lows: np.ndarray) -> Tuple[float, str]:
        """Awesome Oscillator."""
        if len(highs) < 35:
            return 0, "neutral"
        
        # SMA of median price
        median = (highs + lows) / 2
        sma5 = np.mean(median[-5:])
        sma34 = np.mean(median[-34:])
        
        ao = sma5 - sma34
        
        if ao > 0 and ao > np.mean(median[-5:-1]) - np.mean(median[-34:-35]):
            return 0.65, "bullish"
        elif ao < 0 and ao < np.mean(median[-5:-1]) - np.mean(median[-34:-35]):
            return 0.65, "bearish"
        
        return 0, "neutral"


# ============================================================================
# BREAKOUT STRATEGIES
# ============================================================================

class Breakout:
    """Breakout strategies - trade the break."""
    
    @staticmethod
    def donchian_breakout(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                          period: int = 20) -> Tuple[float, str]:
        """Donchian Channel Breakout."""
        if len(closes) < period + 1:
            return 0, "neutral"
        
        upper = np.max(highs[-period:])
        lower = np.min(lows[-period:])
        
        if closes[-1] > upper:
            return 0.8, "bullish"
        elif closes[-1] < lower:
            return 0.8, "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def keltner_breakout(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                         period: int = 20, mult: float = 1.5) -> Tuple[float, str]:
        """Keltner Channel Breakout."""
        if len(closes) < period + 1:
            return 0, "neutral"
        
        ema = np.mean(closes[-period:])
        
        # ATR
        tr = []
        for i in range(-period, 0):
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            ))
        atr = np.mean(tr)
        
        upper = ema + mult * atr
        lower = ema - mult * atr
        
        if closes[-1] > upper:
            return 0.75, "bullish"
        elif closes[-1] < lower:
            return 0.75, "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def volatility_breakout(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                            period: int = 20, factor: float = 0.5) -> Tuple[float, str]:
        """Volatility Breakout (Kelterner-style)."""
        if len(closes) < period + 1:
            return 0, "neutral"
        
        # Calculate ATR
        tr = []
        for i in range(-period, 0):
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            ))
        atr = np.mean(tr)
        
        # Breakout level
        close_ma = np.mean(closes[-period:])
        breakout_level = close_ma + factor * atr
        
        if closes[-1] > breakout_level:
            return 0.7, "bullish"
        elif closes[-1] < close_ma - factor * atr:
            return 0.7, "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def opening_range_breakout(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                               range_bars: int = 3) -> Tuple[float, str]:
        """Opening Range Breakout."""
        if len(closes) < range_bars + 1:
            return 0, "neutral"
        
        # First N bars range
        range_high = np.max(highs[-(range_bars+1):-1])
        range_low = np.min(lows[-(range_bars+1):-1])
        
        current = closes[-1]
        
        if current > range_high:
            return 0.75, "bullish"
        elif current < range_low:
            return 0.75, "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def pivot_breakout(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Tuple[float, str]:
        """Pivot Point Breakout."""
        if len(closes) < 2:
            return 0, "neutral"
        
        # Classic pivot points
        high = highs[-2]
        low = lows[-2]
        close = closes[-2]
        
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        
        current = closes[-1]
        
        if current > r2:
            return 0.8, "bullish"
        elif current > r1:
            return 0.6, "bullish"
        elif current < s2:
            return 0.8, "bearish"
        elif current < s1:
            return 0.6, "bearish"
        
        return 0, "neutral"


# ============================================================================
# VOLUME ANALYSIS STRATEGIES
# ============================================================================

class VolumeAnalysis:
    """Volume analysis strategies - follow the money."""
    
    @staticmethod
    def obv(closes: np.ndarray, volumes: np.ndarray) -> Tuple[float, str]:
        """OBV - On Balance Volume."""
        if len(closes) < 20:
            return 0, "neutral"
        
        obv = np.zeros(len(closes))
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif closes[i] < closes[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]
        
        # OBV trend
        obv_sma = np.mean(obv[-20:])
        
        if obv[-1] > obv_sma and closes[-1] > np.mean(closes[-20:]):
            return 0.65, "bullish"
        elif obv[-1] < obv_sma and closes[-1] < np.mean(closes[-20:]):
            return 0.65, "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def vpt(closes: np.ndarray, volumes: np.ndarray) -> Tuple[float, str]:
        """VPT - Volume Price Trend."""
        if len(closes) < 20:
            return 0, "neutral"
        
        vpt = np.zeros(len(closes))
        for i in range(1, len(closes)):
            pct_change = (closes[i] - closes[i-1]) / closes[i-1]
            vpt[i] = vpt[i-1] + volumes[i] * pct_change
        
        vpt_sma = np.mean(vpt[-20:])
        
        if vpt[-1] > vpt_sma:
            return 0.6, "bullish"
        else:
            return 0.6, "bearish"
    
    @staticmethod
    def mfi(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
            volumes: np.ndarray, period: int = 14) -> Tuple[float, str]:
        """MFI - Money Flow Index."""
        if len(closes) < period + 1:
            return 0, "neutral"
        
        # Typical price
        tp = (highs + lows + closes) / 3
        
        # Money flow
        mf = tp * volumes
        
        # Positive and negative flow
        pos_flow = 0
        neg_flow = 0
        
        for i in range(-period, 0):
            if tp[i] > tp[i-1]:
                pos_flow += mf[i]
            elif tp[i] < tp[i-1]:
                neg_flow += mf[i]
        
        if neg_flow == 0:
            mfi = 100
        else:
            mfi = 100 - (100 / (1 + pos_flow / neg_flow))
        
        if mfi < 20:
            return 0.7, "bullish"
        elif mfi > 80:
            return 0.7, "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def volume_weighted_price(closes: np.ndarray, volumes: np.ndarray,
                              period: int = 20) -> Tuple[float, str]:
        """VWAP deviation."""
        if len(closes) < period:
            return 0, "neutral"
        
        # Simplified VWAP
        vwap = np.sum(closes[-period:] * volumes[-period:]) / np.sum(volumes[-period:])
        std = np.std(closes[-period:])
        
        if std == 0:
            return 0, "neutral"
        
        deviation = (closes[-1] - vwap) / std
        
        if deviation < -2:
            return 0.65, "bullish"
        elif deviation > 2:
            return 0.65, "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def volume_profile(closes: np.ndarray, volumes: np.ndarray,
                       period: int = 50) -> Tuple[float, str]:
        """Volume Profile - High Volume Nodes."""
        if len(closes) < period:
            return 0, "neutral"
        
        # Find price levels with highest volume
        price_bins = np.linspace(np.min(closes[-period:]), np.max(closes[-period:]), 20)
        volume_profile = np.zeros(len(price_bins) - 1)
        
        for i in range(len(closes[-period:])):
            price = closes[-period:][i]
            bin_idx = np.digitize(price, price_bins) - 1
            bin_idx = max(0, min(bin_idx, len(volume_profile) - 1))
            volume_profile[bin_idx] += volumes[-period:][i]
        
        # Point of Control (highest volume)
        poc_idx = np.argmax(volume_profile)
        poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
        
        current = closes[-1]
        
        if current > poc_price:
            return 0.55, "bullish"
        else:
            return 0.55, "bearish"


# ============================================================================
# ON-CHAIN ANALYSIS STRATEGIES
# ============================================================================

class OnChainAnalysis:
    """On-chain analysis strategies - blockchain data alpha."""
    
    @staticmethod
    def nvt_ratio(market_cap: float, transaction_volume: float) -> Tuple[float, str]:
        """NVT Ratio - Network Value to Transactions."""
        if transaction_volume == 0:
            return 0, "neutral"
        
        nvt = market_cap / transaction_volume
        
        # NVT > 100 = overvalued, < 50 = undervalued
        if nvt > 100:
            return min(0.5 + (nvt - 100) / 200, 1), "bearish"
        elif nvt < 50:
            return min(0.5 + (50 - nvt) / 50, 1), "bullish"
        
        return 0, "neutral"
    
    @staticmethod
    def mvrv_z_score(market_cap: float, realized_cap: float) -> Tuple[float, str]:
        """MVRV Z-Score - Market Value vs Realized Value."""
        if realized_cap == 0:
            return 0, "neutral"
        
        mvrv = market_cap / realized_cap
        
        if mvrv > 3.5:
            return 0.8, "bearish"  # Overvalued
        elif mvrv < 1:
            return 0.8, "bullish"  # Undervalued
        
        return 0, "neutral"
    
    @staticmethod
    def nupl(supply_in_profit: float, supply_in_loss: float, total_supply: float) -> Tuple[float, str]:
        """NUPL - Net Unrealized Profit/Loss."""
        if total_supply == 0:
            return 0, "neutral"
        
        nupl = (supply_in_profit - supply_in_loss) / total_supply
        
        if nupl > 0.75:
            return 0.75, "bearish"  # Euphoria
        elif nupl < -0.25:
            return 0.75, "bullish"  # Capitulation
        
        return 0, "neutral"
    
    @staticmethod
    def exchange_netflow(exchange_inflow: float, exchange_outflow: float) -> Tuple[float, str]:
        """Exchange Net Flow - Whale movements."""
        net_flow = exchange_inflow - exchange_outflow
        
        if net_flow > 10000000:  # $10M+ inflow
            return 0.7, "bearish"  # Selling pressure
        elif net_flow < -10000000:  # $10M+ outflow
            return 0.7, "bullish"  # Accumulation
        
        return 0, "neutral"
    
    @staticmethod
    def hash_rate_trend(hash_rate_current: float, hash_rate_avg: float) -> Tuple[float, str]:
        """Hash Rate Trend - Miner confidence."""
        if hash_rate_avg == 0:
            return 0, "neutral"
        
        change = (hash_rate_current - hash_rate_avg) / hash_rate_avg
        
        if change > 0.05:
            return 0.6, "bullish"
        elif change < -0.05:
            return 0.6, "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def stablecoin_supply_change(supply_change_pct: float) -> Tuple[float, str]:
        """Stablecoin Supply - Buying power indicator."""
        if supply_change_pct > 5:
            return 0.7, "bullish"  # New money entering
        elif supply_change_pct < -5:
            return 0.7, "bearish"  # Money leaving
        
        return 0, "neutral"


# ============================================================================
# SENTIMENT ANALYSIS STRATEGIES
# ============================================================================

class SentimentAnalysis:
    """Sentiment analysis strategies - crowd psychology."""
    
    @staticmethod
    def fear_greed_index(fgi_value: float) -> Tuple[float, str]:
        """Fear & Greed Index - Contrarian indicator."""
        if fgi_value < 20:
            return 0.8, "bullish"  # Extreme fear = buy
        elif fgi_value > 80:
            return 0.8, "bearish"  # Extreme greed = sell
        elif fgi_value < 40:
            return 0.6, "bullish"
        elif fgi_value > 60:
            return 0.6, "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def funding_rate_sentiment(funding_rate: float, 
                               long_short_ratio: float) -> Tuple[float, str]:
        """Funding Rate Sentiment - Contrarian."""
        # High positive funding = crowded longs
        if funding_rate > 0.001 and long_short_ratio > 2:
            return 0.7, "bearish"  # Contrarian short
        # High negative funding = crowded shorts
        elif funding_rate < -0.0005 and long_short_ratio < 0.6:
            return 0.7, "bullish"  # Contrarian long
        
        return 0, "neutral"
    
    @staticmethod
    def social_volume(social_volume: float, avg_social_volume: float,
                      price_change: float) -> Tuple[float, str]:
        """Social Volume - Sentiment spike detection."""
        if avg_social_volume == 0:
            return 0, "neutral"
        
        volume_ratio = social_volume / avg_social_volume
        
        # High social volume + price up = potential top
        if volume_ratio > 2 and price_change > 0.05:
            return 0.65, "bearish"
        # High social volume + price down = potential bottom
        elif volume_ratio > 2 and price_change < -0.05:
            return 0.65, "bullish"
        
        return 0, "neutral"
    
    @staticmethod
    def long_short_ratio_signal(ls_ratio: float) -> Tuple[float, str]:
        """Long/Short Ratio - Crowd positioning."""
        if ls_ratio > 2.5:
            return 0.7, "bearish"  # Too many longs
        elif ls_ratio < 0.5:
            return 0.7, "bullish"  # Too many shorts
        
        return 0, "neutral"
    
    @staticmethod
    def open_interest_change(oi_change_pct: float, price_change: float) -> Tuple[float, str]:
        """Open Interest Change - Positioning signal."""
        # OI up + Price up = new longs (bullish)
        if oi_change_pct > 10 and price_change > 0:
            return 0.65, "bullish"
        # OI up + Price down = new shorts (bearish)
        elif oi_change_pct > 10 and price_change < 0:
            return 0.65, "bearish"
        # OI down + Price up = short covering (bullish but weak)
        elif oi_change_pct < -10 and price_change > 0:
            return 0.5, "bullish"
        # OI down + Price down = long liquidation (bearish but weak)
        elif oi_change_pct < -10 and price_change < 0:
            return 0.5, "bearish"
        
        return 0, "neutral"


# ============================================================================
# ARBITRAGE STRATEGIES
# ============================================================================

class ArbitrageStrategies:
    """Arbitrage strategies - risk-free profits."""
    
    @staticmethod
    def funding_rate_arb(perp_price: float, spot_price: float,
                         funding_rate: float) -> Tuple[float, str]:
        """Funding Rate Arbitrage."""
        basis = (perp_price - spot_price) / spot_price * 100
        
        # High positive funding = short perp, long spot
        if funding_rate > 0.001:
            return min(funding_rate * 1000, 1), "short_perp"
        # High negative funding = long perp, short spot
        elif funding_rate < -0.0005:
            return min(abs(funding_rate) * 1000, 1), "long_perp"
        
        return 0, "neutral"
    
    @staticmethod
    def cross_exchange_spread(price_exchange_a: float, price_exchange_b: float,
                              fees_a: float = 0.001, fees_b: float = 0.001) -> Tuple[float, str]:
        """Cross-Exchange Arbitrage."""
        spread = (price_exchange_b - price_exchange_a) / price_exchange_a
        total_fees = fees_a + fees_b
        
        if spread > total_fees:
            profit = spread - total_fees
            return min(profit * 100, 1), "buy_a_sell_b"
        elif spread < -total_fees:
            profit = abs(spread) - total_fees
            return min(profit * 100, 1), "buy_b_sell_a"
        
        return 0, "neutral"
    
    @staticmethod
    def triangular_arb(rate_ab: float, rate_bc: float, rate_ca: float) -> Tuple[float, str]:
        """Triangular Arbitrage."""
        # Start with 1 unit of A
        # A -> B -> C -> A
        result = 1.0 * rate_ab * rate_bc * rate_ca
        
        profit = result - 1.0
        
        if profit > 0.001:  # 0.1% minimum
            return min(profit * 100, 1), "profitable"
        
        return 0, "neutral"
    
    @staticmethod
    def basis_arb(futures_price: float, spot_price: float,
                  days_to_expiry: int, risk_free_rate: float = 0.05) -> Tuple[float, str]:
        """Futures-Spot Basis Arbitrage."""
        # Theoretical futures price
        fair_value = spot_price * (1 + risk_free_rate * days_to_expiry / 365)
        
        basis = futures_price - fair_value
        basis_pct = basis / spot_price * 100
        
        if basis > 0.5:  # Futures overpriced
            return min(basis_pct / 5, 1), "short_futures"
        elif basis < -0.5:  # Futures underpriced
            return min(abs(basis_pct) / 5, 1), "long_futures"
        
        return 0, "neutral"
    
    @staticmethod
    def stablecoin_peg(stablecoin_price: float, target: float = 1.0) -> Tuple[float, str]:
        """Stablecoin Depeg Arbitrage."""
        deviation = abs(stablecoin_price - target)
        
        if deviation > 0.005:  # 0.5% depeg
            if stablecoin_price < target:
                return min(deviation * 100, 1), "buy_peg"
            else:
                return min(deviation * 100, 1), "sell_peg"
        
        return 0, "neutral"


# ============================================================================
# OPTIONS/DERIVATIVES STRATEGIES
# ============================================================================

class OptionsStrategies:
    """Options and derivatives strategies."""
    
    @staticmethod
    def gamma_squeeze(call_oi: float, put_oi: float, 
                      gamma_exposure: float) -> Tuple[float, str]:
        """Gamma Squeeze Detection."""
        # High call OI + high gamma = potential gamma squeeze up
        if call_oi > put_oi * 2 and gamma_exposure > 0:
            return 0.75, "bullish"
        # High put OI + negative gamma = potential gamma squeeze down
        elif put_oi > call_oi * 2 and gamma_exposure < 0:
            return 0.75, "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def max_pain(call_oi_by_strike: Dict[float, float], 
                 put_oi_by_strike: Dict[float, float]) -> float:
        """Max Pain - Strike with highest OI."""
        strikes = sorted(set(list(call_oi_by_strike.keys()) + list(put_oi_by_strike.keys())))
        
        min_pain = float('inf')
        max_pain_strike = strikes[0] if strikes else 0
        
        for strike in strikes:
            pain = 0
            for s, oi in call_oi_by_strike.items():
                if s > strike:
                    pain += (s - strike) * oi
            for s, oi in put_oi_by_strike.items():
                if s < strike:
                    pain += (strike - s) * oi
            
            if pain < min_pain:
                min_pain = pain
                max_pain_strike = strike
        
        return max_pain_strike
    
    @staticmethod
    def put_call_ratio(call_volume: float, put_volume: float) -> Tuple[float, str]:
        """Put/Call Ratio - Sentiment indicator."""
        if call_volume == 0:
            return 0, "neutral"
        
        pcr = put_volume / call_volume
        
        if pcr > 1.2:
            return 0.7, "bullish"  # Too many puts = contrarian bullish
        elif pcr < 0.6:
            return 0.7, "bearish"  # Too many calls = contrarian bearish
        
        return 0, "neutral"
    
    @staticmethod
    def iv_rank(current_iv: float, iv_history: List[float]) -> Tuple[float, str]:
        """IV Rank - Volatility timing."""
        if not iv_history:
            return 0, "neutral"
        
        iv_min = min(iv_history)
        iv_max = max(iv_history)
        
        if iv_max == iv_min:
            return 0, "neutral"
        
        iv_rank = (current_iv - iv_min) / (iv_max - iv_min) * 100
        
        if iv_rank > 80:
            return 0.6, "sell_vol"  # High IV = sell options
        elif iv_rank < 20:
            return 0.6, "buy_vol"  # Low IV = buy options
        
        return 0, "neutral"


# ============================================================================
# ML/AI ENHANCED STRATEGIES
# ============================================================================

class MLEnhanced:
    """ML/AI enhanced strategies."""
    
    @staticmethod
    def ensemble_prediction(predictions: List[Tuple[float, str, float]]) -> Tuple[float, str]:
        """Ensemble multiple model predictions."""
        if not predictions:
            return 0, "neutral"
        
        bull_score = 0
        bear_score = 0
        total_weight = 0
        
        for pred, direction, confidence in predictions:
            weight = confidence
            if direction == "bullish":
                bull_score += pred * weight
            else:
                bear_score += pred * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0, "neutral"
        
        bull_score /= total_weight
        bear_score /= total_weight
        
        if bull_score > bear_score * 1.2:
            return bull_score, "bullish"
        elif bear_score > bull_score * 1.2:
            return bear_score, "bearish"
        
        return 0, "neutral"
    
    @staticmethod
    def regime_detection(volatility: float, trend_strength: float,
                         volume_ratio: float) -> str:
        """Market Regime Detection."""
        if volatility > 0.04:
            if abs(trend_strength) > 0.5:
                return "high_vol_trend"
            else:
                return "high_vol_chop"
        else:
            if abs(trend_strength) > 0.3:
                return "low_vol_trend"
            else:
                return "low_vol_chop"
    
    @staticmethod
    def adaptive_threshold(regime: str, base_threshold: float) -> float:
        """Adaptive threshold based on regime."""
        multipliers = {
            "high_vol_trend": 1.5,
            "high_vol_chop": 0.7,
            "low_vol_trend": 1.0,
            "low_vol_chop": 0.8
        }
        return base_threshold * multipliers.get(regime, 1.0)


# ============================================================================
# STRATEGY AGGREGATOR
# ============================================================================

class UltimateStrategyAggregator:
    """
    Ultimate Strategy Aggregator
    =============================
    Combines all 50+ strategies for maximum edge.
    """
    
    def __init__(self):
        self.trend = TrendFollowing()
        self.mean_rev = MeanReversion()
        self.momentum = Momentum()
        self.breakout = Breakout()
        self.volume = VolumeAnalysis()
        self.onchain = OnChainAnalysis()
        self.sentiment = SentimentAnalysis()
        self.arb = ArbitrageStrategies()
        self.options = OptionsStrategies()
        self.ml = MLEnhanced()
        
        # Strategy weights (optimized for crypto)
        self.weights = {
            # Trend (25%)
            "sma_cross": 0.05,
            "ema_cross": 0.05,
            "macd": 0.05,
            "ichimoku": 0.05,
            "supertrend": 0.05,
            
            # Mean Reversion (20%)
            "rsi": 0.06,
            "bollinger": 0.06,
            "stochastic": 0.04,
            "zscore": 0.04,
            
            # Momentum (15%)
            "roc": 0.03,
            "momentum": 0.04,
            "cmo": 0.03,
            "uo": 0.03,
            "ao": 0.02,
            
            # Breakout (10%)
            "donchian": 0.04,
            "volatility_break": 0.03,
            "pivot": 0.03,
            
            # Volume (10%)
            "obv": 0.04,
            "mfi": 0.04,
            "vwap": 0.02,
            
            # On-chain (10%)
            "nvt": 0.03,
            "mvrv": 0.03,
            "exchange_flow": 0.04,
            
            # Sentiment (10%)
            "fear_greed": 0.04,
            "funding_rate": 0.03,
            "ls_ratio": 0.03,
        }
    
    def calculate_all_signals(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate signals from all strategies."""
        closes = data.get("closes", np.array([]))
        highs = data.get("highs", np.array([]))
        lows = data.get("lows", np.array([]))
        volumes = data.get("volumes", np.array([]))
        
        signals = {}
        
        # Trend
        signals["sma_cross"] = self.trend.sma_crossover(closes)
        signals["ema_cross"] = self.trend.ema_crossover(closes)
        signals["macd"] = self.trend.macd(closes)
        signals["ichimoku"] = self.trend.ichimoku_cloud(closes, highs, lows)
        signals["supertrend"] = self.trend.supertrend(highs, lows, closes)
        
        # Mean Reversion
        signals["rsi"] = self.mean_rev.rsi(closes)
        signals["bollinger"] = self.mean_rev.bollinger_bands(closes)
        signals["stochastic"] = self.mean_rev.stochastic(highs, lows, closes)
        signals["zscore"] = self.mean_rev.mean_reversion_zscore(closes)
        
        # Momentum
        signals["roc"] = self.momentum.roc(closes)
        signals["momentum"] = self.momentum.momentum(closes)
        signals["cmo"] = self.momentum.cmo(closes)
        signals["uo"] = self.momentum.uo(highs, lows, closes)
        signals["ao"] = self.momentum.ao(highs, lows)
        
        # Breakout
        signals["donchian"] = self.breakout.donchian_breakout(closes, highs, lows)
        signals["volatility_break"] = self.breakout.volatility_breakout(closes, highs, lows)
        signals["pivot"] = self.breakout.pivot_breakout(highs, lows, closes)
        
        # Volume
        signals["obv"] = self.volume.obv(closes, volumes)
        signals["mfi"] = self.volume.mfi(highs, lows, closes, volumes)
        signals["vwap"] = self.volume.volume_weighted_price(closes, volumes)
        
        # On-chain (if data available)
        if data.get("market_cap") and data.get("transaction_volume"):
            signals["nvt"] = self.onchain.nvt_ratio(
                data["market_cap"], data["transaction_volume"]
            )
        if data.get("market_cap") and data.get("realized_cap"):
            signals["mvrv"] = self.onchain.mvrv_z_score(
                data["market_cap"], data["realized_cap"]
            )
        if data.get("exchange_inflow") and data.get("exchange_outflow"):
            signals["exchange_flow"] = self.onchain.exchange_netflow(
                data["exchange_inflow"], data["exchange_outflow"]
            )
        
        # Sentiment (if data available)
        if data.get("fear_greed_index"):
            signals["fear_greed"] = self.sentiment.fear_greed_index(
                data["fear_greed_index"]
            )
        if data.get("funding_rate"):
            signals["funding_rate"] = self.sentiment.funding_rate_sentiment(
                data["funding_rate"], data.get("long_short_ratio", 1.0)
            )
        if data.get("long_short_ratio"):
            signals["ls_ratio"] = self.sentiment.long_short_ratio_signal(
                data["long_short_ratio"]
            )
        
        return signals
    
    def calculate_weighted_signal(
        self,
        signals: Dict[str, Tuple[float, str]]
    ) -> Tuple[float, str]:
        """Calculate weighted consensus signal."""
        bull_score = 0
        bear_score = 0
        
        for strategy_name, (score, direction) in signals.items():
            weight = self.weights.get(strategy_name, 0.02)
            
            if direction == "bullish":
                bull_score += score * weight
            elif direction == "bearish":
                bear_score += score * weight
        
        total_weight = sum(self.weights.values())
        bull_score /= total_weight
        bear_score /= total_weight
        
        if bull_score > bear_score * 1.3:
            return bull_score, "bullish"
        elif bear_score > bull_score * 1.3:
            return bear_score, "bearish"
        
        return 0, "neutral"
    
    def get_trading_signal(
        self,
        data: Dict[str, Any],
        entry_price: float
    ) -> Dict[str, Any]:
        """Get final trading signal with all analysis."""
        signals = self.calculate_all_signals(data)
        consensus_score, consensus_dir = self.calculate_weighted_signal(signals)
        
        # Count signals
        bullish_count = sum(1 for _, (_, d) in signals.items() if d == "bullish")
        bearish_count = sum(1 for _, (_, d) in signals.items() if d == "bearish")
        total_signals = len([s for s in signals.values() if s[0] > 0])
        
        # Determine action
        if consensus_dir == "bullish" and consensus_score > 0.5:
            action = "BUY"
            stop_loss = entry_price * 0.985  # 1.5% stop
            take_profit = entry_price * 1.03  # 3% target
        elif consensus_dir == "bearish" and consensus_score > 0.5:
            action = "SELL"
            stop_loss = entry_price * 1.015
            take_profit = entry_price * 0.97
        else:
            action = "HOLD"
            stop_loss = 0
            take_profit = 0
        
        return {
            "action": action,
            "consensus_score": consensus_score,
            "consensus_direction": consensus_dir,
            "bullish_signals": bullish_count,
            "bearish_signals": bearish_count,
            "total_signals": total_signals,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "signals": signals
        }


# Export
__all__ = [
    "UltimateStrategyAggregator",
    "TrendFollowing",
    "MeanReversion",
    "Momentum",
    "Breakout",
    "VolumeAnalysis",
    "OnChainAnalysis",
    "SentimentAnalysis",
    "ArbitrageStrategies",
    "OptionsStrategies",
    "MLEnhanced"
]
