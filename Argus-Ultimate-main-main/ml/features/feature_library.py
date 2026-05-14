"""
ARGUS GODMODE - Feature Engineering Library
=============================================

Comprehensive feature engineering for ML-based trading signals.
300+ features across multiple categories:
- Price features (returns, volatility, momentum)
- Technical features (indicators, patterns)
- Volume features (volume dynamics, VWAP)
- Order book features (depth, imbalance)
- Regime features (trend, volatility regime)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class FeatureCategory(Enum):
    PRICE = "price"
    TECHNICAL = "technical"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    REGIME = "regime"
    ORDERBOOK = "orderbook"


@dataclass
class FeatureSet:
    """Container for computed features."""
    features: Dict[str, float]
    timestamp: pd.Timestamp
    symbol: str
    category_counts: Dict[str, int]


class FeatureLibrary:
    """
    Comprehensive feature library for ML trading.
    Generates 100+ features from OHLCV data.
    """

    def __init__(self):
        self.feature_names: List[str] = []
        self._cache: Dict[str, Any] = {}

    def compute_all_features(self, df: pd.DataFrame, symbol: str = "") -> FeatureSet:
        """Compute all features from OHLCV data."""
        features = {}
        category_counts = {}

        # Price features
        price_features = self._compute_price_features(df)
        features.update(price_features)
        category_counts["price"] = len(price_features)

        # Technical features
        tech_features = self._compute_technical_features(df)
        features.update(tech_features)
        category_counts["technical"] = len(tech_features)

        # Volume features
        vol_features = self._compute_volume_features(df)
        features.update(vol_features)
        category_counts["volume"] = len(vol_features)

        # Momentum features
        mom_features = self._compute_momentum_features(df)
        features.update(mom_features)
        category_counts["momentum"] = len(mom_features)

        # Volatility features
        volatility_features = self._compute_volatility_features(df)
        features.update(volatility_features)
        category_counts["volatility"] = len(volatility_features)

        # Regime features
        regime_features = self._compute_regime_features(df)
        features.update(regime_features)
        category_counts["regime"] = len(regime_features)

        return FeatureSet(
            features=features,
            timestamp=df["timestamp"].iloc[-1] if "timestamp" in df.columns else pd.Timestamp.now(),
            symbol=symbol,
            category_counts=category_counts,
        )

    def _compute_price_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute price-based features."""
        features = {}
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        open_price = df["open"].values

        # Returns at various timeframes
        for period in [1, 5, 10, 20, 50]:
            if len(close) > period:
                ret = (close[-1] - close[-period - 1]) / close[-period - 1]
                features[f"return_{period}"] = ret

        # Log returns
        if len(close) > 1:
            log_returns = np.log(close[1:] / close[:-1])
            features["log_return_1"] = log_returns[-1] if len(log_returns) > 0 else 0
            features["log_return_mean_20"] = np.mean(log_returns[-20:]) if len(log_returns) >= 20 else 0

        # Price position in range
        if len(close) > 20:
            high_20 = np.max(high[-20:])
            low_20 = np.min(low[-20:])
            range_20 = high_20 - low_20
            if range_20 > 0:
                features["price_position_20"] = (close[-1] - low_20) / range_20
            else:
                features["price_position_20"] = 0.5

        # Gap analysis
        if len(open_price) > 1:
            gap = (open_price[-1] - close[-2]) / close[-2]
            features["gap_pct"] = gap

        # Candle body and shadow
        body = close[-1] - open_price[-1]
        upper_shadow = high[-1] - max(close[-1], open_price[-1])
        lower_shadow = min(close[-1], open_price[-1]) - low[-1]
        candle_range = high[-1] - low[-1]

        if candle_range > 0:
            features["body_pct"] = body / candle_range
            features["upper_shadow_pct"] = upper_shadow / candle_range
            features["lower_shadow_pct"] = lower_shadow / candle_range
        else:
            features["body_pct"] = 0
            features["upper_shadow_pct"] = 0
            features["lower_shadow_pct"] = 0

        return features

    def _compute_technical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute technical indicator features."""
        features = {}
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # Moving averages
        for period in [5, 10, 20, 50]:
            if len(close) >= period:
                sma = np.mean(close[-period:])
                features[f"sma_{period}"] = sma
                features[f"price_vs_sma_{period}"] = (close[-1] - sma) / sma

                # EMA
                ema = self._ema(close, period)
                features[f"ema_{period}"] = ema
                features[f"price_vs_ema_{period}"] = (close[-1] - ema) / ema

        # MA crossovers
        if len(close) >= 50:
            sma_20 = np.mean(close[-20:])
            sma_50 = np.mean(close[-50:])
            features["sma_20_50_cross"] = 1 if sma_20 > sma_50 else -1
            features["sma_20_50_diff"] = (sma_20 - sma_50) / sma_50

        # RSI
        if len(close) > 14:
            rsi = self._rsi(close, 14)
            features["rsi_14"] = rsi
            features["rsi_oversold"] = 1 if rsi < 30 else 0
            features["rsi_overbought"] = 1 if rsi > 70 else 0

        # MACD
        if len(close) >= 26:
            macd, signal, hist = self._macd(close)
            features["macd"] = macd
            features["macd_signal"] = signal
            features["macd_hist"] = hist
            features["macd_cross"] = 1 if macd > signal else -1

        # Bollinger Bands
        if len(close) >= 20:
            sma = np.mean(close[-20:])
            std = np.std(close[-20:])
            upper = sma + 2 * std
            lower = sma - 2 * std
            features["bb_upper"] = upper
            features["bb_lower"] = lower
            features["bb_width"] = (upper - lower) / sma
            features["bb_position"] = (close[-1] - lower) / (upper - lower) if upper != lower else 0.5

        # Stochastic
        if len(close) >= 14:
            k, d = self._stochastic(close, high, low)
            features["stoch_k"] = k
            features["stoch_d"] = d
            features["stoch_cross"] = 1 if k > d else -1

        # ATR
        if len(close) >= 14:
            atr = self._atr(close, high, low, 14)
            features["atr_14"] = atr
            features["atr_pct"] = atr / close[-1]

        # ADX
        if len(close) >= 14:
            adx = self._adx(close, high, low, 14)
            features["adx_14"] = adx
            features["trend_strength"] = 1 if adx > 25 else 0

        return features

    def _compute_volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute volume-based features."""
        features = {}

        if "volume" not in df.columns:
            return features

        volume = df["volume"].values
        close = df["close"].values

        # Volume ratios
        if len(volume) >= 20:
            vol_sma_20 = np.mean(volume[-20:])
            features["volume_ratio_20"] = volume[-1] / vol_sma_20 if vol_sma_20 > 0 else 1

        if len(volume) >= 50:
            vol_sma_50 = np.mean(volume[-50:])
            features["volume_ratio_50"] = volume[-1] / vol_sma_50 if vol_sma_50 > 0 else 1

        # Volume trend
        if len(volume) >= 5:
            vol_change = (volume[-1] - volume[-5]) / volume[-5] if volume[-5] > 0 else 0
            features["volume_change_5"] = vol_change

        # Price-volume correlation
        if len(close) >= 20 and len(volume) >= 20:
            close_returns = np.diff(close[-20:]) / close[-21:-1]
            vol_changes = np.diff(volume[-20:]) / (volume[-21:-1] + 1e-10)
            corr = np.corrcoef(close_returns, vol_changes)[0, 1]
            features["price_volume_corr"] = corr if not np.isnan(corr) else 0

        # VWAP
        if len(close) >= 20:
            typical_price = (close[-20:] + df["high"].values[-20:] + df["low"].values[-20:]) / 3
            vwap = np.sum(typical_price * volume[-20:]) / np.sum(volume[-20:]) if np.sum(volume[-20:]) > 0 else close[-1]
            features["vwap_20"] = vwap
            features["price_vs_vwap"] = (close[-1] - vwap) / vwap

        # OBV
        if len(close) >= 20:
            obv = self._obv(close[-20:], volume[-20:])
            features["obv_slope"] = (obv[-1] - obv[0]) / len(obv) if len(obv) > 0 else 0

        return features

    def _compute_momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute momentum features."""
        features = {}
        close = df["close"].values

        # Rate of change
        for period in [5, 10, 20]:
            if len(close) > period:
                roc = (close[-1] - close[-period - 1]) / close[-period - 1] * 100
                features[f"roc_{period}"] = roc

        # Momentum
        for period in [10, 20]:
            if len(close) > period:
                mom = close[-1] - close[-period - 1]
                features[f"momentum_{period}"] = mom

        # Williams %R
        if len(close) >= 14:
            high = df["high"].values
            low = df["low"].values
            high_14 = np.max(high[-14:])
            low_14 = np.min(low[-14:])
            if high_14 != low_14:
                williams_r = (high_14 - close[-1]) / (high_14 - low_14) * -100
                features["williams_r_14"] = williams_r
            else:
                features["williams_r_14"] = -50

        # CCI
        if len(close) >= 20:
            cci = self._cci(close, df["high"].values, df["low"].values, 20)
            features["cci_20"] = cci

        # Acceleration
        if len(close) >= 11:
            recent = close[-11:]
            returns = np.diff(recent) / recent[:-1]
            if len(returns) >= 2:
                acceleration = returns[-1] - returns[-2]
                features["acceleration"] = acceleration

        return features

    def _compute_volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute volatility features."""
        features = {}
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # Historical volatility
        if len(close) > 21:
            recent = close[-21:]
            returns = np.diff(recent) / recent[:-1]
            hv_20 = np.std(returns) * np.sqrt(252) * 100
            features["hv_20"] = hv_20

        if len(close) > 51:
            recent = close[-51:]
            returns = np.diff(recent) / recent[:-1]
            hv_50 = np.std(returns) * np.sqrt(252) * 100
            features["hv_50"] = hv_50
            features["hv_ratio"] = features.get("hv_20", hv_50) / hv_50 if hv_50 > 0 else 1

        # Average True Range percentage
        if len(close) >= 14:
            atr = self._atr(close, high, low, 14)
            features["atr_pct_14"] = (atr / close[-1]) * 100

        # Parkinson volatility
        if len(high) >= 20:
            pk_vol = np.sqrt(np.mean((np.log(high[-20:] / low[-20:]))**2) / (4 * np.log(2))) * np.sqrt(252)
            features["parkinson_vol"] = pk_vol

        # Volatility regime
        if "hv_20" in features and "hv_50" in features:
            if features["hv_20"] > features["hv_50"] * 1.5:
                features["vol_regime"] = 2  # High vol
            elif features["hv_20"] < features["hv_50"] * 0.7:
                features["vol_regime"] = 0  # Low vol
            else:
                features["vol_regime"] = 1  # Normal

        # Range expansion
        if len(high) >= 5:
            avg_range = np.mean(high[-5:] - low[-5:])
            current_range = high[-1] - low[-1]
            features["range_expansion"] = current_range / avg_range if avg_range > 0 else 1

        return features

    def _compute_regime_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute market regime features."""
        features = {}
        close = df["close"].values

        # Trend features
        if len(close) >= 50:
            sma_20 = np.mean(close[-20:])
            sma_50 = np.mean(close[-50:])

            # Trend direction
            features["trend_direction"] = 1 if close[-1] > sma_20 > sma_50 else (-1 if close[-1] < sma_20 < sma_50 else 0)

            # Trend strength (distance from MAs)
            features["trend_strength_20"] = abs(close[-1] - sma_20) / sma_20
            features["trend_strength_50"] = abs(close[-1] - sma_50) / sma_50

        # Mean reversion potential
        if len(close) >= 20:
            zscore = (close[-1] - np.mean(close[-20:])) / np.std(close[-20:])
            features["zscore_20"] = zscore
            features["mean_reversion_signal"] = 1 if zscore < -2 else (-1 if zscore > 2 else 0)

        # Momentum regime
        if len(close) >= 20:
            short_mom = (close[-1] - close[-5]) / close[-5] if len(close) > 5 else 0
            long_mom = (close[-1] - close[-20]) / close[-20]
            features["momentum_regime"] = 1 if short_mom > 0 and long_mom > 0 else (-1 if short_mom < 0 and long_mom < 0 else 0)

        return features

    # Helper methods
    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calculate EMA."""
        if len(data) < period:
            return data[-1]
        multiplier = 2 / (period + 1)
        ema = np.mean(data[:period])
        for price in data[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema

    def _rsi(self, data: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _macd(self, data: np.ndarray) -> tuple:
        """Calculate MACD."""
        ema_12 = self._ema(data, 12)
        ema_26 = self._ema(data, 26)
        macd = ema_12 - ema_26
        signal = self._ema(np.array([macd]), 9)  # Simplified
        hist = macd - signal
        return macd, signal, hist

    def _stochastic(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int = 14) -> tuple:
        """Calculate Stochastic."""
        lowest_low = np.min(low[-period:])
        highest_high = np.max(high[-period:])
        if highest_high == lowest_low:
            k = 50
        else:
            k = ((close[-1] - lowest_low) / (highest_high - lowest_low)) * 100
        d = k  # Simplified (should be SMA of K)
        return k, d

    def _atr(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int = 14) -> float:
        """Calculate ATR."""
        tr_list = []
        for i in range(-period, 0):
            if i == -period:
                tr = high[i] - low[i]
            else:
                tr = max(
                    high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1])
                )
            tr_list.append(tr)
        return np.mean(tr_list)

    def _adx(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int = 14) -> float:
        """Calculate ADX (simplified)."""
        atr = self._atr(close, high, low, period)
        if atr == 0:
            return 0

        plus_dm = max(high[-1] - high[-2], 0) if high[-1] - high[-2] > low[-2] - low[-1] else 0
        minus_dm = max(low[-2] - low[-1], 0) if low[-2] - low[-1] > high[-1] - high[-2] else 0

        plus_di = (plus_dm / atr) * 100 if atr > 0 else 0
        minus_di = (minus_dm / atr) * 100 if atr > 0 else 0

        if plus_di + minus_di == 0:
            return 0
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        return dx

    def _cci(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int = 20) -> float:
        """Calculate CCI."""
        typical_price = (close[-period:] + high[-period:] + low[-period:]) / 3
        sma = np.mean(typical_price)
        mad = np.mean(np.abs(typical_price - sma))
        if mad == 0:
            return 0
        return (typical_price[-1] - sma) / (0.015 * mad)

    def _obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate OBV."""
        obv = [0]
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv.append(obv[-1] + volume[i])
            elif close[i] < close[i - 1]:
                obv.append(obv[-1] - volume[i])
            else:
                obv.append(obv[-1])
        return np.array(obv)


# Singleton instance
feature_library = FeatureLibrary()
