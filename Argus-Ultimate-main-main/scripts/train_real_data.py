#!/usr/bin/env python3
"""
Train Argus ML models using REAL market data.

Loads BTC/USD OHLCV data and trains:
1. Regime Classifier
2. Position Sizer
3. Signal Classifier
4. Volatility Model

Usage:
    py scripts/train_real_data.py
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


def ohlcv_to_features(ohlcv_data):
    """
    Convert OHLCV data to ML features.
    
    Input: List of [timestamp, open, high, low, close, volume]
    Output: DataFrame with engineered features
    """
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Price features
    df['returns'] = df['close'].pct_change()
    df['returns_1h'] = df['returns']
    df['returns_4h'] = df['close'].pct_change(4)
    df['returns_24h'] = df['close'].pct_change(24)
    
    # Volatility
    df['volatility_1h'] = df['returns'].rolling(6).std()
    df['volatility_4h'] = df['returns'].rolling(24).std()
    
    # Volume features
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)
    
    # Technical indicators
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    
    # Price ratios
    df['high_low_ratio'] = df['high'] / (df['low'] + 1e-10)
    df['close_open_ratio'] = df['close'] / (df['open'] + 1e-10)
    
    # Momentum
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
    
    # Mean reversion
    sma20 = df['close'].rolling(20).mean()
    df['mean_reversion_z'] = (df['close'] - sma20) / (std20 + 1e-10)
    
    # Drop NaN
    df = df.dropna()
    
    return df


def create_labels(df):
    """Create training labels from price data."""
    # Regime labels (0=trending_up, 1=trending_down, 2=ranging, 3=volatile)
    future_returns = df['close'].pct_change(12).shift(-12)  # 12 candles ahead
    volatility = df['returns'].rolling(12).std()
    vol_threshold = volatility.median()
    
    conditions = [
        (future_returns > 0.02) & (volatility < vol_threshold),  # Trending up
        (future_returns < -0.02) & (volatility < vol_threshold),  # Trending down
        (abs(future_returns) <= 0.02) & (volatility < vol_threshold),  # Ranging
        (volatility >= vol_threshold),  # Volatile
    ]
    df['regime'] = np.select(conditions, [0, 1, 2, 3], default=2)
    
    # Position labels (optimal position based on momentum and RSI)
    momentum = df['momentum_5']
    rsi = df['rsi_14']
    position = np.clip(momentum * 10 + (50 - rsi) / 50, -1, 1)
    df['optimal_position'] = position
    
    # Signal labels (0=no trade, 1=buy, 2=sell)
    signal_conditions = [
        (df['macd'] > df['macd_signal']) & (df['rsi_14'] < 60) & (df['returns_4h'] > 0),  # Buy
        (df['macd'] < df['macd_signal']) & (df['rsi_14'] > 40) & (df['returns_4h'] < 0),  # Sell
    ]
    df['signal'] = np.select(signal_conditions, [1, 2], default=0)
    
    # Volatility labels (forward volatility)
    df['future_volatility'] = df['returns'].rolling(12).std().shift(-12)
    
    return df


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
    logger.info("ARGUS ML TRAINING - REAL MARKET DATA")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Load real market data
    raw_data = load_real_market_data()
    if raw_data is None:
        return
    
    # Process each symbol
    all_features = []
    all_labels = []
    
    for symbol, symbol_data in raw_data.items():
        logger.info("\nProcessing %s...", symbol)
        
        ohlcv = symbol_data.get('ohlcv', [])
        if not ohlcv or len(ohlcv) < 100:
            logger.warning("  Skipping %s: insufficient data (%d candles)", symbol, len(ohlcv))
            continue
        
        # Convert to features
        df = ohlcv_to_features(ohlcv)
        df = create_labels(df)
        
        logger.info("  Generated %d samples with %d features", len(df), len(df.columns))
        
        all_features.append(df)
        all_labels.append(symbol)
    
    if not all_features:
        logger.error("No valid data found!")
        return
    
    # Combine all symbols
    logger.info("\nCombining data from %d symbols...", len(all_features))
    combined_df = pd.concat(all_features, ignore_index=True)
    logger.info("Total samples: %d", len(combined_df))
    
    # Select feature columns
    feature_cols = [
        'returns_1h', 'returns_4h', 'returns_24h',
        'volatility_1h', 'volatility_4h',
        'volume_ratio',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bb_position', 'atr_14',
        'high_low_ratio', 'close_open_ratio',
        'momentum_5', 'momentum_20',
        'mean_reversion_z',
    ]
    
    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in combined_df.columns]
    X = combined_df[feature_cols].copy()
    
    # Data quality check
    logger.info("\nValidating data quality...")
    quality = DataQualityPipeline(DataQualityConfig())
    passed, report = quality.validate(X)
    logger.info("Data quality: %s (score=%.2f)", "PASSED" if passed else "FAILED", report.quality_score)
    
    # Clean data
    X_clean = quality.clean(X)
    
    # Split 80/20
    split = int(len(X_clean) * 0.8)
    X_train = X_clean.iloc[:split]
    X_val = X_clean.iloc[split:]
    
    logger.info("\nTraining samples: %d", len(X_train))
    logger.info("Validation samples: %d", len(X_val))
    logger.info("Features: %s", feature_cols)
    
    # Train models
    results = {}
    
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
    
    # 1. Regime Classifier
    y_regime = combined_df['regime'].fillna(2).astype(int)
    logger.info("\n" + "=" * 60)
    logger.info("1. REGIME CLASSIFIER")
    logger.info("=" * 60)
    results['regime_classifier'] = train_model(
        'regime_classifier',
        GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        X_train, y_regime.iloc[:split], X_val, y_regime.iloc[split:],
    )
    if results['regime_classifier'].success:
        acc = results['regime_classifier'].final_metrics.get('accuracy', 0)
        logger.info("  Accuracy: %.4f", acc)
    
    # 2. Position Sizer
    y_position = combined_df['optimal_position'].fillna(0)
    logger.info("\n" + "=" * 60)
    logger.info("2. POSITION SIZER")
    logger.info("=" * 60)
    results['position_sizer'] = train_model(
        'position_sizer',
        GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        X_train, y_position.iloc[:split], X_val, y_position.iloc[split:],
    )
    if results['position_sizer'].success:
        r2 = results['position_sizer'].final_metrics.get('r2', 0)
        logger.info("  R²: %.4f", r2)
    
    # 3. Signal Classifier
    y_signal = combined_df['signal'].fillna(0).astype(int)
    logger.info("\n" + "=" * 60)
    logger.info("3. SIGNAL CLASSIFIER")
    logger.info("=" * 60)
    results['signal_classifier'] = train_model(
        'signal_classifier',
        RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        X_train, y_signal.iloc[:split], X_val, y_signal.iloc[split:],
    )
    if results['signal_classifier'].success:
        acc = results['signal_classifier'].final_metrics.get('accuracy', 0)
        logger.info("  Accuracy: %.4f", acc)
    
    # 4. Volatility Model
    y_vol = combined_df['future_volatility'].fillna(combined_df['future_volatility'].median())
    logger.info("\n" + "=" * 60)
    logger.info("4. VOLATILITY MODEL")
    logger.info("=" * 60)
    results['volatility_model'] = train_model(
        'volatility_model',
        GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        X_train, y_vol.iloc[:split], X_val, y_vol.iloc[split:],
    )
    if results['volatility_model'].success:
        r2 = results['volatility_model'].final_metrics.get('r2', 0)
        logger.info("  R²: %.4f", r2)
    
    # Summary
    total_time = time.time() - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE - REAL MARKET DATA")
    logger.info("=" * 60)
    
    success_count = sum(1 for r in results.values() if r.success)
    logger.info("Models trained: %d/%d", success_count, len(results))
    logger.info("Total time: %.1fs", total_time)
    
    # Show registered models
    registry = EnhancedModelRegistry()
    models = registry.list_models()
    logger.info("\nAll registered models: %d", len(models))
    for m in models:
        metrics = m.get('metrics', {})
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)])
        logger.info("  - %s v%d (%s) %s", m['name'], m['version'], m['status'], metrics_str)


if __name__ == "__main__":
    main()
