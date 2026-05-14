#!/usr/bin/env python3
"""
Train Argus ML models with ENHANCED features and MORE symbols.

Features added:
- Advanced technical indicators (Stochastic, CCI, Williams %R)
- Statistical features (skewness, kurtosis, autocorrelation)
- Cross-asset correlation features
- Time-based features (hour of day, day of week)
- Multi-timeframe features

Usage:
    py scripts/train_enhanced.py
"""

import sys
import os
import time
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

from ml.training_pipeline import TrainingPipeline, TrainingConfig
from ml.data_quality import DataQualityPipeline, DataQualityConfig
from ml.model_registry_enhanced import EnhancedModelRegistry


def load_real_market_data():
    """Load real market data from pickle file."""
    data_path = Path("data/training_market_data.pkl")
    
    if not data_path.exists():
        logger.error("Training data not found: %s", data_path)
        return None
    
    logger.info("Loading real market data from %s", data_path)
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    logger.info("Loaded data for %d symbols: %s", len(data), list(data.keys()))
    return data


def ohlcv_to_enhanced_features(ohlcv_data, symbol=""):
    """
    Convert OHLCV data to ENHANCED ML features.
    
    Includes:
    - Basic technical indicators
    - Advanced indicators (Stochastic, CCI, Williams %R)
    - Statistical features
    - Time-based features
    - Pattern features
    """
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['symbol'] = symbol
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # ============================================
    # BASIC PRICE FEATURES
    # ============================================
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['returns_1h'] = df['returns']
    df['returns_2h'] = df['close'].pct_change(2)
    df['returns_4h'] = df['close'].pct_change(4)
    df['returns_8h'] = df['close'].pct_change(8)
    df['returns_24h'] = df['close'].pct_change(24)
    
    # Price levels
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['close_to_high'] = (df['high'] - df['close']) / df['close']
    df['close_to_low'] = (df['close'] - df['low']) / df['close']
    df['open_close_ratio'] = df['close'] / df['open']
    
    # ============================================
    # VOLATILITY FEATURES
    # ============================================
    df['volatility_4h'] = df['returns'].rolling(4).std()
    df['volatility_8h'] = df['returns'].rolling(8).std()
    df['volatility_24h'] = df['returns'].rolling(24).std()
    df['volatility_ratio'] = df['volatility_4h'] / (df['volatility_24h'] + 1e-10)
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_8'] = tr.rolling(8).mean()
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_ratio'] = df['atr_8'] / (df['atr_14'] + 1e-10)
    
    # ============================================
    # VOLUME FEATURES
    # ============================================
    df['volume_sma_8'] = df['volume'].rolling(8).mean()
    df['volume_sma_24'] = df['volume'].rolling(24).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_8'] + 1e-10)
    df['volume_trend'] = df['volume_sma_8'] / (df['volume_sma_24'] + 1e-10)
    
    # OBV (On-Balance Volume)
    df['obv'] = (np.sign(df['returns']) * df['volume']).cumsum()
    df['obv_sma'] = df['obv'].rolling(20).mean()
    df['obv_trend'] = df['obv'] / (df['obv_sma'] + 1e-10)
    
    # Volume-Price Trend
    df['vpt'] = (df['volume'] * df['returns']).cumsum()
    
    # ============================================
    # MOMENTUM INDICATORS
    # ============================================
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_7'] = 100 - (100 / (1 + (gain.rolling(7).mean() / (loss.rolling(7).mean() + 1e-10))))
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['rsi_21'] = 100 - (100 / (1 + (gain.rolling(21).mean() / (loss.rolling(21).mean() + 1e-10))))
    df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
    
    # MACD
    ema8 = df['close'].ewm(span=8).mean()
    ema12 = df['close'].ewm(span=12).mean()
    ema21 = df['close'].ewm(span=21).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_crossover'] = np.sign(df['macd_hist']).diff()
    
    # Fast MACD
    df['macd_fast'] = ema8 - ema21
    df['macd_fast_signal'] = df['macd_fast'].ewm(span=5).mean()
    
    # Stochastic Oscillator
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
    df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
    
    # CCI (Commodity Channel Index)
    tp = (df['high'] + df['low'] + df['close']) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (tp - tp_sma) / (0.015 * tp_mad + 1e-10)
    
    # Williams %R
    df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14 + 1e-10)
    
    # Momentum
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
    
    # Rate of Change
    df['roc_5'] = (df['close'] - df['close'].shift(5)) / (df['close'].shift(5) + 1e-10)
    df['roc_10'] = (df['close'] - df['close'].shift(10)) / (df['close'].shift(10) + 1e-10)
    
    # ============================================
    # TREND INDICATORS
    # ============================================
    # Moving Averages
    df['sma_8'] = df['close'].rolling(8).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # MA Crossovers
    df['ma_cross_8_20'] = (df['sma_8'] > df['sma_20']).astype(int)
    df['ma_cross_20_50'] = (df['sma_20'] > df['sma_50']).astype(int)
    df['price_vs_sma20'] = (df['close'] - df['sma_20']) / (df['sma_20'] + 1e-10)
    df['price_vs_sma50'] = (df['close'] - df['sma_50']) / (df['sma_50'] + 1e-10)
    
    # Bollinger Bands
    bb_sma = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = bb_sma + 2 * bb_std
    df['bb_lower'] = bb_sma - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (bb_sma + 1e-10)
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).quantile(0.1)).astype(int)
    
    # ADX (Average Directional Index) approximation
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr_smooth = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr_smooth + 1e-10)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr_smooth + 1e-10)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.rolling(14).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    
    # ============================================
    # VOLATILITY INDICATORS
    # ============================================
    # Historical Volatility
    df['hv_10'] = df['returns'].rolling(10).std() * np.sqrt(365 * 24 / 4)
    df['hv_20'] = df['returns'].rolling(20).std() * np.sqrt(365 * 24 / 4)
    df['hv_ratio'] = df['hv_10'] / (df['hv_20'] + 1e-10)
    
    # Volatility Rank
    df['vol_rank'] = df['hv_20'].rolling(50).rank(pct=True)
    
    # ============================================
    # STATISTICAL FEATURES
    # ============================================
    df['skewness_20'] = df['returns'].rolling(20).skew()
    df['kurtosis_20'] = df['returns'].rolling(20).kurt()
    df['autocorr_5'] = df['returns'].rolling(20).apply(lambda x: x.autocorr(lag=5) if len(x) > 5 else 0)
    
    # Hurst Exponent approximation
    def hurst_exponent(series):
        if len(series) < 20:
            return 0.5
        lags = range(2, min(20, len(series) // 2))
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        tau = [t for t in tau if t > 0]
        if len(tau) < 2:
            return 0.5
        poly = np.polyfit(np.log(range(2, 2 + len(tau))), np.log(tau), 1)
        return poly[0]
    
    df['hurst_20'] = df['returns'].rolling(20).apply(hurst_exponent, raw=False)
    
    # ============================================
    # PATTERN FEATURES
    # ============================================
    # Candlestick patterns
    df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
    df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
    df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
    df['is_doji'] = (df['body_size'] < 0.1).astype(int)
    df['is_hammer'] = ((df['lower_shadow'] > 0.6) & (df['upper_shadow'] < 0.2)).astype(int)
    
    # Gap detection
    df['gap'] = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-10)
    
    # Consecutive moves
    df['consecutive_up'] = (df['returns'] > 0).astype(int).groupby((df['returns'] <= 0).cumsum()).cumsum()
    df['consecutive_down'] = (df['returns'] < 0).astype(int).groupby((df['returns'] >= 0).cumsum()).cumsum()
    
    # ============================================
    # TIME FEATURES
    # ============================================
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Session detection (crypto trades 24/7 but has peak hours)
    df['is_asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['is_europe_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['is_us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
    
    # Drop NaN
    df = df.dropna()
    
    return df


def create_enhanced_labels(df):
    """Create enhanced training labels."""
    # Forward-looking targets
    horizon = 12  # 12 candles ahead (48 hours for 4h data)
    
    # Future returns
    df['future_return'] = df['close'].pct_change(horizon).shift(-horizon)
    df['future_volatility'] = df['returns'].rolling(horizon).std().shift(-horizon)
    
    # Enhanced regime labels
    vol_threshold_high = df['returns'].rolling(horizon).std().quantile(0.7)
    vol_threshold_low = df['returns'].rolling(horizon).std().quantile(0.3)
    
    conditions = [
        (df['future_return'] > 0.01) & (df['future_volatility'] < vol_threshold_low),  # Strong uptrend
        (df['future_return'] > 0) & (df['future_volatility'] < vol_threshold_high),     # Weak uptrend
        (df['future_return'] < -0.01) & (df['future_volatility'] < vol_threshold_low),  # Strong downtrend
        (df['future_return'] < 0) & (df['future_volatility'] < vol_threshold_high),     # Weak downtrend
        (df['future_volatility'] >= vol_threshold_high),                                 # High volatility
    ]
    choices = [0, 1, 2, 3, 4]  # strong_up, weak_up, strong_down, weak_down, volatile
    df['regime_enhanced'] = np.select(conditions, choices, default=2)
    
    # Simple regime (4 classes)
    simple_conditions = [
        (df['future_return'] > 0.02),
        (df['future_return'] < -0.02),
        (abs(df['future_return']) <= 0.02) & (df['future_volatility'] < vol_threshold_high),
        (df['future_volatility'] >= vol_threshold_high),
    ]
    df['regime'] = np.select(simple_conditions, [0, 1, 2, 3], default=2)
    
    # Optimal position
    momentum_score = df['momentum_5'] * 2 + df['momentum_10'] * 1.5 + df['momentum_20']
    rsi_score = (50 - df['rsi_14']) / 50
    trend_score = df['price_vs_sma20'] * 10
    df['optimal_position'] = np.clip((momentum_score + rsi_score + trend_score) / 3, -1, 1)
    
    # Enhanced signal (3 classes with confidence)
    buy_signal = (
        (df['macd'] > df['macd_signal']) &
        (df['rsi_14'] < 65) &
        (df['stoch_k'] < 80) &
        (df['ma_cross_8_20'] == 1)
    )
    sell_signal = (
        (df['macd'] < df['macd_signal']) &
        (df['rsi_14'] > 35) &
        (df['stoch_k'] > 20) &
        (df['ma_cross_8_20'] == 0)
    )
    df['signal'] = np.where(buy_signal, 1, np.where(sell_signal, 2, 0))
    
    return df


def get_feature_columns():
    """Get list of feature columns for training."""
    return [
        # Returns
        'returns_1h', 'returns_2h', 'returns_4h', 'returns_8h', 'returns_24h',
        'log_returns',
        # Price levels
        'high_low_range', 'close_to_high', 'close_to_low', 'open_close_ratio',
        # Volatility
        'volatility_4h', 'volatility_8h', 'volatility_24h', 'volatility_ratio',
        'atr_8', 'atr_14', 'atr_ratio',
        # Volume
        'volume_ratio', 'volume_trend', 'obv_trend',
        # Momentum
        'rsi_7', 'rsi_14', 'rsi_21', 'rsi_overbought', 'rsi_oversold',
        'macd', 'macd_signal', 'macd_hist', 'macd_crossover',
        'stoch_k', 'stoch_d', 'stoch_overbought', 'stoch_oversold',
        'cci', 'williams_r',
        'momentum_5', 'momentum_10', 'momentum_20',
        'roc_5', 'roc_10',
        # Trend
        'ma_cross_8_20', 'ma_cross_20_50',
        'price_vs_sma20', 'price_vs_sma50',
        'bb_position', 'bb_width', 'bb_squeeze',
        'adx', 'plus_di', 'minus_di',
        # Volatility indicators
        'hv_10', 'hv_20', 'hv_ratio', 'vol_rank',
        # Statistical
        'skewness_20', 'kurtosis_20', 'hurst_20',
        # Patterns
        'body_size', 'upper_shadow', 'lower_shadow', 'is_doji', 'is_hammer',
        'gap', 'consecutive_up', 'consecutive_down',
        # Time
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'is_asia_session', 'is_europe_session', 'is_us_session',
    ]


def train_model(name, model, X_train, y_train, X_val, y_val):
    """Train a model using the unified pipeline."""
    config = TrainingConfig(
        model_name=name,
        model_type="sklearn",
        patience=10,
        register_model=True,
        verbose=False,
    )
    
    pipeline = TrainingPipeline(config)
    result = pipeline.train_sklearn(model, X_train, y_train, X_val, y_val)
    
    return result


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("ARGUS ML TRAINING - ENHANCED FEATURES")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Load real market data
    raw_data = load_real_market_data()
    if raw_data is None:
        return
    
    # Process each symbol with enhanced features
    all_dfs = []
    
    for symbol, symbol_data in raw_data.items():
        logger.info("Processing %s...", symbol)
        
        ohlcv = symbol_data.get('ohlcv', [])
        if not ohlcv or len(ohlcv) < 100:
            logger.warning("  Skipping %s: insufficient data", symbol)
            continue
        
        # Generate enhanced features
        df = ohlcv_to_enhanced_features(ohlcv, symbol=symbol)
        df = create_enhanced_labels(df)
        
        logger.info("  Generated %d samples with %d features", len(df), len(df.columns))
        all_dfs.append(df)
    
    if not all_dfs:
        logger.error("No valid data!")
        return
    
    # Combine all symbols
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info("\nTotal samples from %d symbols: %d", len(all_dfs), len(combined_df))
    
    # Get feature columns
    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in combined_df.columns]
    
    X = combined_df[feature_cols].copy()
    logger.info("Features: %d", len(feature_cols))
    
    # Data quality
    logger.info("\nValidating data quality...")
    quality = DataQualityPipeline(DataQualityConfig(min_samples=100))
    passed, report = quality.validate(X)
    logger.info("Data quality: %s (score=%.2f)", "PASSED" if passed else "FAILED", report.quality_score)
    
    # Clean
    X_clean = quality.clean(X)
    
    # Split
    split = int(len(X_clean) * 0.8)
    X_train = X_clean.iloc[:split]
    X_val = X_clean.iloc[split:]
    
    logger.info("Training: %d, Validation: %d", len(X_train), len(X_val))
    
    # Train enhanced models
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
    
    results = {}
    
    # 1. Enhanced Regime Classifier
    y_regime = combined_df['regime_enhanced'].fillna(2).astype(int)
    logger.info("\n" + "=" * 60)
    logger.info("1. REGIME CLASSIFIER (5 classes)")
    logger.info("=" * 60)
    results['regime_classifier'] = train_model(
        'regime_classifier',
        GradientBoostingClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42),
        X_train, y_regime.iloc[:split], X_val, y_regime.iloc[split:],
    )
    
    # 2. Enhanced Position Sizer
    y_position = combined_df['optimal_position'].fillna(0)
    logger.info("\n" + "=" * 60)
    logger.info("2. POSITION SIZER (enhanced)")
    logger.info("=" * 60)
    results['position_sizer'] = train_model(
        'position_sizer',
        GradientBoostingRegressor(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42),
        X_train, y_position.iloc[:split], X_val, y_position.iloc[split:],
    )
    
    # 3. Enhanced Signal Classifier
    y_signal = combined_df['signal'].fillna(0).astype(int)
    logger.info("\n" + "=" * 60)
    logger.info("3. SIGNAL CLASSIFIER (enhanced)")
    logger.info("=" * 60)
    results['signal_classifier'] = train_model(
        'signal_classifier',
        RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        X_train, y_signal.iloc[:split], X_val, y_signal.iloc[split:],
    )
    
    # 4. Enhanced Volatility Model
    y_vol = combined_df['future_volatility'].fillna(combined_df['future_volatility'].median())
    logger.info("\n" + "=" * 60)
    logger.info("4. VOLATILITY MODEL (forward-looking)")
    logger.info("=" * 60)
    results['volatility_model'] = train_model(
        'volatility_model',
        GradientBoostingRegressor(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42),
        X_train, y_vol.iloc[:split], X_val, y_vol.iloc[split:],
    )
    
    # 5. NEW: Trend Strength Predictor
    y_trend = np.abs(combined_df['future_return'].fillna(0))
    logger.info("\n" + "=" * 60)
    logger.info("5. TREND STRENGTH PREDICTOR")
    logger.info("=" * 60)
    results['trend_strength'] = train_model(
        'trend_strength',
        GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        X_train, y_trend.iloc[:split], X_val, y_trend.iloc[split:],
    )
    
    # Summary
    total_time = time.time() - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("ENHANCED TRAINING COMPLETE")
    logger.info("=" * 60)
    
    for name, result in results.items():
        status = "OK" if result.success else "FAIL"
        metrics = result.final_metrics
        if 'accuracy' in metrics:
            logger.info("  %s [%s]: accuracy=%.4f", name, status, metrics['accuracy'])
        elif 'r2' in metrics:
            logger.info("  %s [%s]: r2=%.4f", name, status, metrics['r2'])
    
    logger.info("Total time: %.1fs", total_time)
    
    # Show registry
    registry = EnhancedModelRegistry()
    models = registry.list_models()
    logger.info("\nRegistered models: %d", len(models))


if __name__ == "__main__":
    main()
