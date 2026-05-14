"""
Advanced Feature Engineering Module

Generates 100+ features from market data for ML models.
Includes:
- Price-based features (returns, momentum, volatility)
- Volume features (OBV, VWAP, volume profile)
- Technical indicators (RSI, MACD, BB, ATR, Stochastic)
- Cross-timeframe features
- Statistical features (skewness, kurtosis, Hurst)
- Pattern recognition (candlestick patterns)
- Time-based features (hour of day, day of week)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """Generates advanced features for ML models."""
    
    def __init__(self):
        self.feature_names: List[str] = []
    
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features from OHLCV DataFrame."""
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features = pd.concat([features, self._price_features(df)], axis=1)
        
        # Volatility features
        features = pd.concat([features, self._volatility_features(df)], axis=1)
        
        # Volume features
        features = pd.concat([features, self._volume_features(df)], axis=1)
        
        # Technical indicators
        features = pd.concat([features, self._rsi_features(df)], axis=1)
        features = pd.concat([features, self._macd_features(df)], axis=1)
        features = pd.concat([features, self._bollinger_features(df)], axis=1)
        features = pd.concat([features, self._atr_features(df)], axis=1)
        features = pd.concat([features, self._stochastic_features(df)], axis=1)
        
        # Statistical features
        features = pd.concat([features, self._statistical_features(df)], axis=1)
        
        # Pattern features
        features = pd.concat([features, self._pattern_features(df)], axis=1)
        
        # Time features
        features = pd.concat([features, self._time_features(df)], axis=1)
        
        # Cross-timeframe features (if higher TF data available)
        # These are added by the multi-timeframe processor
        
        # Store feature names
        self.feature_names = list(features.columns)
        
        return features
    
    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price-based features."""
        f = pd.DataFrame(index=df.index)
        
        # Returns at different horizons
        for period in [1, 2, 4, 8, 12, 24, 48, 96]:
            f[f'ret_{period}'] = df['close'].pct_change(period)
        
        # Log returns
        f['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price relative to moving averages
        for period in [10, 20, 50, 100]:
            ma = df['close'].rolling(period).mean()
            f[f'price_ma_{period}_ratio'] = df['close'] / ma
        
        # High-Low range
        f['hl_range'] = (df['high'] - df['low']) / df['close']
        f['hl_range_ma'] = f['hl_range'].rolling(20).mean()
        
        # Close position within candle
        f['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).clip(lower=1e-8)
        
        # Gap (open vs previous close)
        f['gap'] = df['open'] / df['close'].shift(1) - 1
        
        # Price acceleration
        f['ret_accel'] = f['ret_1'].diff()
        
        return f
    
    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility features."""
        f = pd.DataFrame(index=df.index)
        ret = df['close'].pct_change()
        
        # Realized volatility at different horizons
        for period in [12, 24, 48, 96]:
            f[f'rvol_{period}'] = ret.rolling(period).std() * np.sqrt(8760)  # Annualized
        
        # Volatility ratio (short/long)
        f['vol_ratio'] = f['rvol_12'] / f['rvol_96'].clip(lower=1e-8)
        
        # Volatility regime
        f['vol_regime'] = pd.qcut(f['rvol_24'].fillna(0), q=5, labels=False, duplicates='drop')
        
        # Parkinson volatility (using high-low)
        f['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(df['high'] / df['low']) ** 2).rolling(24).mean()
        )
        
        # Garman-Klass volatility
        log_hl = np.log(df['high'] / df['low']) ** 2
        log_co = np.log(df['close'] / df['open']) ** 2
        f['gk_vol'] = np.sqrt((0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(24).mean())
        
        return f
    
    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features."""
        f = pd.DataFrame(index=df.index)
        
        # Volume ratios
        f['vol_sma_10'] = df['volume'].rolling(10).mean()
        f['vol_sma_20'] = df['volume'].rolling(20).mean()
        f['vol_ratio_10'] = df['volume'] / f['vol_sma_10'].clip(lower=1e-8)
        f['vol_ratio_20'] = df['volume'] / f['vol_sma_20'].clip(lower=1e-8)
        
        # On-Balance Volume
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        f['obv'] = obv
        f['obv_sma'] = obv.rolling(20).mean()
        f['obv_slope'] = obv.rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0
        )
        
        # Volume-Price Trend
        f['vpt'] = (df['close'].pct_change() * df['volume']).fillna(0).cumsum()
        
        # Accumulation/Distribution
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).clip(lower=1e-8)
        f['ad_line'] = (clv * df['volume']).fillna(0).cumsum()
        
        # Volume rate of change
        f['vol_roc'] = df['volume'].pct_change(10)
        
        return f
    
    def _rsi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI at multiple periods."""
        f = pd.DataFrame(index=df.index)
        delta = df['close'].diff()
        
        for period in [7, 14, 21]:
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss.clip(lower=1e-8)
            rsi = 100 - (100 / (1 + rs))
            f[f'rsi_{period}'] = rsi
        
        # RSI divergence (price vs RSI trend)
        f['rsi_divergence'] = (
            (df['close'] > df['close'].rolling(20).mean()) & 
            (f['rsi_14'] < f['rsi_14'].rolling(20).mean())
        ).astype(float)
        
        return f
    
    def _macd_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD features."""
        f = pd.DataFrame(index=df.index)
        
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        
        f['macd'] = ema12 - ema26
        f['macd_signal'] = f['macd'].ewm(span=9).mean()
        f['macd_histogram'] = f['macd'] - f['macd_signal']
        f['macd_cross'] = np.sign(f['macd'] - f['macd_signal']).diff().fillna(0)
        
        return f
    
    def _bollinger_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Band features."""
        f = pd.DataFrame(index=df.index)
        
        bb_mid = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        
        f['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower).clip(lower=1e-8)
        f['bb_width'] = (bb_upper - bb_lower) / bb_mid
        f['bb_squeeze'] = (f['bb_width'] < f['bb_width'].rolling(100).quantile(0.1)).astype(float)
        
        return f
    
    def _atr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR features."""
        f = pd.DataFrame(index=df.index)
        
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        f['atr_14'] = tr.rolling(14).mean()
        f['atr_ratio'] = f['atr_14'] / df['close']
        f['atr_pct_change'] = f['atr_14'].pct_change(10)
        
        return f
    
    def _stochastic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stochastic oscillator features."""
        f = pd.DataFrame(index=df.index)
        
        for period in [14, 21]:
            low_min = df['low'].rolling(period).min()
            high_max = df['high'].rolling(period).max()
            k = 100 * (df['close'] - low_min) / (high_max - low_min).clip(lower=1e-8)
            d = k.rolling(3).mean()
            f[f'stoch_k_{period}'] = k
            f[f'stoch_d_{period}'] = d
        
        return f
    
    def _statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical features."""
        f = pd.DataFrame(index=df.index)
        ret = df['close'].pct_change()
        
        # Skewness and kurtosis
        f['skew_20'] = ret.rolling(20).skew()
        f['kurt_20'] = ret.rolling(20).kurt()
        
        # Hurst exponent approximation
        for period in [20, 50]:
            f[f'hurst_{period}'] = ret.rolling(period).apply(
                lambda x: self._hurst_exponent(x) if len(x) == period else np.nan
            )
        
        # Autocorrelation
        f['autocorr_10'] = ret.rolling(10).apply(
            lambda x: x.autocorr() if len(x) == 10 else np.nan
        )
        
        return f
    
    def _hurst_exponent(self, series: pd.Series) -> float:
        """Calculate Hurst exponent."""
        try:
            lags = range(2, min(20, len(series) // 2))
            tau = [np.std(np.subtract(series[lag:].values, series[:-lag].values)) for lag in lags]
            tau = [t for t in tau if t > 0]
            if len(tau) < 2:
                return 0.5
            poly = np.polyfit(np.log(range(2, 2 + len(tau))), np.log(tau), 1)
            return poly[0]
        except:
            return 0.5
    
    def _pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Candlestick pattern features."""
        f = pd.DataFrame(index=df.index)
        
        body = df['close'] - df['open']
        body_abs = body.abs()
        range_hl = df['high'] - df['low']
        
        # Doji (small body)
        f['doji'] = (body_abs / range_hl.clip(lower=1e-8) < 0.1).astype(float)
        
        # Hammer (small body, long lower shadow)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        f['hammer'] = (
            (lower_shadow > 2 * body_abs) & 
            (upper_shadow < body_abs * 0.5)
        ).astype(float)
        
        # Engulfing patterns
        f['bullish_engulfing'] = (
            (body.shift(1) < 0) &  # Previous bearish
            (body > 0) &  # Current bullish
            (body.abs() > body.shift(1).abs())  # Larger body
        ).astype(float)
        
        f['bearish_engulfing'] = (
            (body.shift(1) > 0) &  # Previous bullish
            (body < 0) &  # Current bearish
            (body.abs() > body.shift(1).abs())  # Larger body
        ).astype(float)
        
        # Three white soldiers / three black crows
        f['three_white_soldiers'] = (
            (body > 0) & (body.shift(1) > 0) & (body.shift(2) > 0) &
            (df['close'] > df['close'].shift(1)) &
            (df['close'].shift(1) > df['close'].shift(2))
        ).astype(float)
        
        f['three_black_crows'] = (
            (body < 0) & (body.shift(1) < 0) & (body.shift(2) < 0) &
            (df['close'] < df['close'].shift(1)) &
            (df['close'].shift(1) < df['close'].shift(2))
        ).astype(float)
        
        return f
    
    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features."""
        f = pd.DataFrame(index=df.index)
        
        if 'datetime' in df.columns:
            dt = pd.to_datetime(df['datetime'])
        elif df.index.dtype == 'datetime64[ns]':
            dt = df.index
        else:
            return f
        
        f['hour'] = dt.hour
        f['day_of_week'] = dt.dayofweek
        f['day_of_month'] = dt.day
        f['month'] = dt.month
        
        # Cyclical encoding
        f['hour_sin'] = np.sin(2 * np.pi * f['hour'] / 24)
        f['hour_cos'] = np.cos(2 * np.pi * f['hour'] / 24)
        f['dow_sin'] = np.sin(2 * np.pi * f['day_of_week'] / 7)
        f['dow_cos'] = np.cos(2 * np.pi * f['day_of_week'] / 7)
        
        # Session detection (UTC)
        f['asian_session'] = ((f['hour'] >= 0) & (f['hour'] < 8)).astype(float)
        f['european_session'] = ((f['hour'] >= 8) & (f['hour'] < 16)).astype(float)
        f['us_session'] = ((f['hour'] >= 16) & (f['hour'] < 24)).astype(float)
        
        return f
    
    def generate_labels(self, df: pd.DataFrame, 
                        signal_horizon: int = 4,
                        regime_horizon: int = 24) -> pd.DataFrame:
        """Generate labels for training."""
        labels = pd.DataFrame(index=df.index)
        
        fwd_ret_short = df['close'].pct_change(signal_horizon).shift(-signal_horizon)
        fwd_ret_long = df['close'].pct_change(regime_horizon).shift(-regime_horizon)
        
        # Signal (3-class)
        labels['signal'] = pd.cut(
            fwd_ret_short, 
            bins=[-np.inf, -0.01, 0.01, np.inf], 
            labels=[0, 1, 2]
        )
        
        # Regime (3-class)
        labels['regime'] = pd.cut(
            fwd_ret_long,
            bins=[-np.inf, -0.03, 0.03, np.inf],
            labels=[0, 1, 2]
        )
        
        # Position size (continuous 0-1)
        labels['position_size'] = np.clip(np.abs(fwd_ret_short) * 30, 0, 1)
        
        # Volatility (forward)
        labels['volatility'] = fwd_ret_short.rolling(signal_horizon).std().shift(-signal_horizon)
        
        # Trend strength (0-1)
        rolling_up = (fwd_ret_short > 0).rolling(24).mean()
        labels['trend_strength'] = np.abs(rolling_up - 0.5) * 2
        
        return labels
