"""
Argus Trading System - Feature Engineering
=========================================

Feature engineering pipeline for ML models.

Features organized by category:
- Price features (returns, volatility)
- Technical indicators (RSI, MACD, Bollinger)
- Volume features (volume ratio, OBV)
- Order book features (imbalance, depth)
- Cross-asset features (correlations)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Return windows
    return_windows: List[int] = field(default_factory=lambda: [1, 5, 15, 60, 240])

    # Volatility windows
    volatility_windows: List[int] = field(default_factory=lambda: [5, 15, 30, 60])

    # Technical indicator periods
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 50, 200])
    atr_period: int = 14

    # Volume features
    volume_ma_period: int = 20
    obv_period: int = 20

    # Momentum indicators
    momentum_periods: List[int] = field(default_factory=lambda: [5, 10, 20])

    # Scaling
    normalize_features: bool = True
    clip_outliers: bool = True
    outlier_std: float = 3.0


class FeatureEngineer:
    """
    Feature engineering pipeline.

    Generates features from OHLCV data for ML models.
    Features are designed to be predictive of price movements.
    """

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.config = config or FeatureConfig()
        self._feature_names: List[str] = []

    def generate_features(
        self,
        ohlcv: pd.DataFrame,
        order_book: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Generate all features from OHLCV data.

        Args:
            ohlcv: DataFrame with columns [open, high, low, close, volume]
            order_book: Optional order book data dict

        Returns:
            DataFrame with feature columns
        """
        if ohlcv is None or ohlcv.empty:
            return pd.DataFrame()

        # Ensure required columns
        required = {"open", "high", "low", "close"}
        if not required.issubset(set(ohlcv.columns)):
            logger.warning("Missing required OHLCV columns")
            return pd.DataFrame()

        features = pd.DataFrame(index=ohlcv.index)

        # Price features
        price_features = self._price_features(ohlcv)
        features = pd.concat([features, price_features], axis=1)

        # Volatility features
        vol_features = self._volatility_features(ohlcv)
        features = pd.concat([features, vol_features], axis=1)

        # Technical indicators
        tech_features = self._technical_features(ohlcv)
        features = pd.concat([features, tech_features], axis=1)

        # Volume features (if volume available)
        if "volume" in ohlcv.columns:
            volume_features = self._volume_features(ohlcv)
            features = pd.concat([features, volume_features], axis=1)

        # Momentum features
        momentum_features = self._momentum_features(ohlcv)
        features = pd.concat([features, momentum_features], axis=1)

        # Order book features (if available)
        if order_book is not None:
            ob_features = self._order_book_features(order_book)
            if not ob_features.empty:
                features = pd.concat([features, ob_features], axis=1)

        # Post-processing
        if self.config.normalize_features:
            features = self._normalize(features)

        if self.config.clip_outliers:
            features = self._clip_outliers(features)

        # Store feature names
        self._feature_names = list(features.columns)

        return features

    def get_latest_features(
        self,
        ohlcv: pd.DataFrame,
        order_book: Optional[Dict] = None,
    ) -> Optional[np.ndarray]:
        """
        Get feature vector for the latest bar.

        Args:
            ohlcv: OHLCV DataFrame
            order_book: Optional order book data

        Returns:
            1D numpy array of features, or None if insufficient data
        """
        features = self.generate_features(ohlcv, order_book)
        if features.empty:
            return None

        # Get last row, drop NaN
        latest = features.iloc[-1].dropna()
        if len(latest) == 0:
            return None

        return latest.values

    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self._feature_names

    def _price_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based features."""
        config = self.config
        close = ohlcv["close"].astype(float)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)
        open_ = ohlcv["open"].astype(float)

        features = pd.DataFrame(index=ohlcv.index)

        # Returns at different windows
        for window in config.return_windows:
            features[f"return_{window}"] = close.pct_change(window)

        # Log returns
        features["log_return_1"] = np.log(close / close.shift(1))

        # Price range features
        features["bar_range"] = (high - low) / close
        features["bar_body"] = abs(close - open_) / close
        features["upper_shadow"] = (high - np.maximum(close, open_)) / close
        features["lower_shadow"] = (np.minimum(close, open_) - low) / close

        # Price position within range
        features["price_position"] = (close - low) / (high - low + 1e-10)

        # Gap features
        features["gap"] = (open_ - close.shift(1)) / close.shift(1)

        return features

    def _volatility_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility features."""
        config = self.config
        close = ohlcv["close"].astype(float)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)

        features = pd.DataFrame(index=ohlcv.index)

        # Rolling standard deviation of returns
        returns = close.pct_change()
        for window in config.volatility_windows:
            features[f"volatility_{window}"] = returns.rolling(window).std()

        # ATR (Average True Range)
        tr = self._true_range(high, low, close)
        features["atr"] = tr.rolling(config.atr_period).mean()
        features["atr_pct"] = features["atr"] / close

        # Parkinson volatility (using high-low)
        hl_ratio = np.log(high / low)
        features["parkinson_vol"] = hl_ratio.rolling(20).std() / (2 * np.sqrt(np.log(2)))

        # Volatility regime (current vs historical)
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std()
        features["vol_regime"] = vol_20 / (vol_60 + 1e-10)

        return features

    def _technical_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicator features."""
        config = self.config
        close = ohlcv["close"].astype(float)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)

        features = pd.DataFrame(index=ohlcv.index)

        # RSI
        features["rsi"] = self._calculate_rsi(close, config.rsi_period)
        features["rsi_normalized"] = (features["rsi"] - 50) / 50  # -1 to 1

        # MACD
        macd, signal, hist = self._calculate_macd(
            close, config.macd_fast, config.macd_slow, config.macd_signal
        )
        features["macd"] = macd / close  # Normalize by price
        features["macd_signal"] = signal / close
        features["macd_hist"] = hist / close
        features["macd_hist_change"] = features["macd_hist"].diff()

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger(
            close, config.bb_period, config.bb_std
        )
        bb_width = bb_upper - bb_lower
        features["bb_position"] = (close - bb_lower) / (bb_width + 1e-10)
        features["bb_width"] = bb_width / bb_middle

        # EMAs and distances
        for period in config.ema_periods:
            ema = close.ewm(span=period, adjust=False).mean()
            features[f"ema_{period}_dist"] = (close - ema) / ema

        # EMA crossover signals
        ema_9 = close.ewm(span=9, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()
        features["ema_cross_9_21"] = (ema_9 - ema_21) / ema_21

        # Stochastic
        lowest_low = low.rolling(14).min()
        highest_high = high.rolling(14).max()
        features["stoch_k"] = (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        features["stoch_d"] = features["stoch_k"].rolling(3).mean()

        return features

    def _volume_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features."""
        config = self.config
        close = ohlcv["close"].astype(float)
        volume = ohlcv["volume"].astype(float)

        features = pd.DataFrame(index=ohlcv.index)

        # Volume moving average ratio
        vol_ma = volume.rolling(config.volume_ma_period).mean()
        features["volume_ratio"] = volume / (vol_ma + 1e-10)

        # Volume trend
        features["volume_change"] = volume.pct_change()
        features["volume_trend"] = volume.rolling(5).mean() / volume.rolling(20).mean()

        # On-Balance Volume
        obv = self._calculate_obv(close, volume)
        obv_ma = obv.rolling(config.obv_period).mean()
        features["obv_trend"] = (obv - obv_ma) / (abs(obv_ma) + 1e-10)

        # Price-Volume correlation
        returns = close.pct_change()
        features["pv_corr"] = returns.rolling(20).corr(volume.pct_change())

        # Volume-weighted price
        vwap = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
        features["vwap_dist"] = (close - vwap) / vwap

        return features

    def _momentum_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum features."""
        config = self.config
        close = ohlcv["close"].astype(float)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)

        features = pd.DataFrame(index=ohlcv.index)

        # Rate of Change
        for period in config.momentum_periods:
            features[f"roc_{period}"] = (close - close.shift(period)) / close.shift(period)

        # ADX (simplified)
        tr = self._true_range(high, low, close)
        atr = tr.rolling(14).mean()

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / atr

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        features["adx"] = dx.rolling(14).mean() / 100  # Normalize to 0-1

        # CCI (Commodity Channel Index)
        typical_price = (high + low + close) / 3
        tp_ma = typical_price.rolling(20).mean()
        tp_std = typical_price.rolling(20).std()
        features["cci"] = (typical_price - tp_ma) / (0.015 * tp_std + 1e-10) / 100

        # Williams %R
        highest_high = high.rolling(14).max()
        lowest_low = low.rolling(14).min()
        features["williams_r"] = (highest_high - close) / (highest_high - lowest_low + 1e-10)

        return features

    def _order_book_features(self, order_book: Dict) -> pd.DataFrame:
        """Calculate order book features."""
        features = {}

        try:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])

            if not bids or not asks:
                return pd.DataFrame()

            # Best bid/ask
            best_bid = float(bids[0][0]) if bids else 0
            best_ask = float(asks[0][0]) if asks else 0

            # Spread
            if best_bid > 0 and best_ask > 0:
                mid = (best_bid + best_ask) / 2
                features["spread_bps"] = (best_ask - best_bid) / mid * 10000

            # Depth (top 5 levels)
            bid_depth = sum(float(b[1]) for b in bids[:5]) if bids else 0
            ask_depth = sum(float(a[1]) for a in asks[:5]) if asks else 0

            total_depth = bid_depth + ask_depth
            if total_depth > 0:
                features["book_imbalance"] = (bid_depth - ask_depth) / total_depth

            # Weighted mid price
            if bid_depth + ask_depth > 0:
                weighted_mid = (best_bid * ask_depth + best_ask * bid_depth) / (bid_depth + ask_depth)
                features["weighted_mid_dist"] = (weighted_mid - mid) / mid if mid > 0 else 0

        except Exception as e:
            logger.debug("Error calculating order book features: %s", e)

        return pd.DataFrame([features]) if features else pd.DataFrame()

    def _normalize(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using z-score."""
        for col in features.columns:
            mean = features[col].mean()
            std = features[col].std()
            if std > 0:
                features[col] = (features[col] - mean) / std
        return features

    def _clip_outliers(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers to N standard deviations."""
        std_clip = self.config.outlier_std
        for col in features.columns:
            features[col] = features[col].clip(-std_clip, std_clip)
        return features

    @staticmethod
    def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate True Range."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    @staticmethod
    def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calculate_macd(
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _calculate_bollinger(
        close: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    @staticmethod
    def _calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        direction = np.sign(close.diff())
        return (direction * volume).cumsum()


# Convenience function
def extract_features(
    ohlcv: pd.DataFrame,
    order_book: Optional[Dict] = None,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """
    Extract features from OHLCV data.

    Args:
        ohlcv: OHLCV DataFrame
        order_book: Optional order book data
        config: Feature configuration

    Returns:
        DataFrame with features
    """
    engineer = FeatureEngineer(config)
    return engineer.generate_features(ohlcv, order_book)


__all__ = [
    "FeatureEngineer",
    "FeatureConfig",
    "extract_features",
]
