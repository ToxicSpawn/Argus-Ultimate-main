#!/usr/bin/env python3
"""
Retrain all ML models with consistent 17 features.

This ensures all models can work together in production.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def load_market_data():
    """Load market data from pickle."""
    import pickle
    
    data_path = Path("data/training_market_data.pkl")
    if not data_path.exists():
        raise FileNotFoundError(f"Market data not found: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Loaded data for {len(data)} symbols")
    return data


def generate_features(df):
    """Generate the 17 base features."""
    features = pd.DataFrame(index=df.index)
    
    # Returns
    features['returns_1'] = df['close'].pct_change(1)
    features['returns_4'] = df['close'].pct_change(4)
    features['returns_12'] = df['close'].pct_change(12)
    features['returns_24'] = df['close'].pct_change(24)
    
    # Volatility
    features['volatility_12'] = features['returns_1'].rolling(12).std()
    features['volatility_24'] = features['returns_1'].rolling(24).std()
    
    # Volume
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Price position
    features['price_position'] = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min())
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    features['macd'] = ema12 - ema26
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features['atr_ratio'] = true_range.rolling(14).mean() / df['close']
    
    # OBV
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    features['obv_change'] = obv.pct_change(10)
    
    # BB position
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
    
    return features


def generate_labels(df):
    """Generate labels for each model."""
    labels = pd.DataFrame(index=df.index)
    
    # Returns for forward periods
    fwd_returns_4 = df['close'].pct_change(4).shift(-4)
    fwd_returns_12 = df['close'].pct_change(12).shift(-12)
    
    # Regime (5 classes based on 12-period return)
    fwd_return_12 = df['close'].pct_change(12).shift(-12)
    labels['regime'] = pd.cut(fwd_return_12, bins=[-np.inf, -0.05, -0.01, 0.01, 0.05, np.inf], labels=[0, 1, 2, 3, 4])
    
    # Signal (3 classes: sell, hold, buy)
    labels['signal'] = pd.cut(fwd_returns_4, bins=[-np.inf, -0.01, 0.01, np.inf], labels=[0, 1, 2])
    
    # Position size (0-1 based on absolute return)
    labels['position_size'] = np.clip(np.abs(fwd_returns_4) * 20, 0, 1)
    
    # Volatility (forward realized vol)
    labels['volatility'] = fwd_returns_4.rolling(4).std().shift(-4)
    
    # Trend strength (0-1 based on consistency of direction)
    rolling_up = (fwd_returns_4 > 0).rolling(12).mean()
    labels['trend_strength'] = np.abs(rolling_up - 0.5) * 2  # 0 to 1
    
    return labels


def train_all_models():
    """Train all models with consistent 17 features."""
    logger.info("="*60)
    logger.info("RETRAINING ALL MODELS WITH 17 FEATURES")
    logger.info("="*60)
    
    # Load data
    data = load_market_data()
    
    # Process all symbols
    all_features = []
    all_labels_list = []
    
    for symbol, symbol_data in data.items():
        logger.info(f"Processing {symbol}...")
        
        # Extract OHLCV data and convert to DataFrame
        ohlcv_raw = symbol_data.get('ohlcv') if isinstance(symbol_data, dict) else None
        if ohlcv_raw is None or len(ohlcv_raw) < 100:
            logger.warning(f"  Skipped {symbol}: no valid data")
            continue
        
        # Convert list to DataFrame
        df = pd.DataFrame(ohlcv_raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df = df.sort_index()
        
        features = generate_features(df)
        labels = generate_labels(df)
        
        # Combine and drop NaN
        combined = pd.concat([features, labels], axis=1).dropna()
        
        if len(combined) < 100:
            logger.warning(f"  Skipped {symbol}: only {len(combined)} samples")
            continue
        
        all_features.append(combined[features.columns])
        all_labels_list.append(combined[labels.columns])
        
        logger.info(f"  {len(combined)} samples")
    
    # Combine all data
    X = pd.concat(all_features, ignore_index=True)
    labels_df = pd.concat(all_labels_list, ignore_index=True)
    y = {col: labels_df[col] for col in labels_df.columns}
    
    logger.info(f"\nTotal samples: {len(X)}")
    logger.info(f"Features: {list(X.columns)}")
    
    # Train/test split
    X_train, X_test, idx_train, idx_test = train_test_split(X, X.index, test_size=0.2, random_state=42)
    
    # Train models
    models = {}
    metrics = {}
    
    # 1. Regime Classifier
    logger.info("\n" + "="*60)
    logger.info("1. REGIME CLASSIFIER")
    logger.info("="*60)
    
    y_train_regime = y['regime'].iloc[idx_train]
    y_test_regime = y['regime'].iloc[idx_test]
    
    regime_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    regime_model.fit(X_train, y_train_regime)
    
    train_acc = regime_model.score(X_train, y_train_regime)
    test_acc = regime_model.score(X_test, y_test_regime)
    
    models['regime_classifier'] = regime_model
    metrics['regime_classifier'] = {'train_accuracy': train_acc, 'test_accuracy': test_acc}
    logger.info(f"Train: {train_acc:.4f}, Test: {test_acc:.4f}")
    
    # 2. Signal Classifier
    logger.info("\n" + "="*60)
    logger.info("2. SIGNAL CLASSIFIER")
    logger.info("="*60)
    
    y_train_signal = y['signal'].iloc[idx_train]
    y_test_signal = y['signal'].iloc[idx_test]
    
    signal_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    signal_model.fit(X_train, y_train_signal)
    
    train_acc = signal_model.score(X_train, y_train_signal)
    test_acc = signal_model.score(X_test, y_test_signal)
    
    models['signal_classifier'] = signal_model
    metrics['signal_classifier'] = {'train_accuracy': train_acc, 'test_accuracy': test_acc}
    logger.info(f"Train: {train_acc:.4f}, Test: {test_acc:.4f}")
    
    # 3. Position Sizer
    logger.info("\n" + "="*60)
    logger.info("3. POSITION SIZER")
    logger.info("="*60)
    
    y_train_pos = y['position_size'].iloc[idx_train]
    y_test_pos = y['position_size'].iloc[idx_test]
    
    pos_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    pos_model.fit(X_train, y_train_pos)
    
    train_r2 = pos_model.score(X_train, y_train_pos)
    test_r2 = pos_model.score(X_test, y_test_pos)
    
    models['position_sizer'] = pos_model
    metrics['position_sizer'] = {'train_r2': train_r2, 'test_r2': test_r2}
    logger.info(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
    # 4. Volatility Model
    logger.info("\n" + "="*60)
    logger.info("4. VOLATILITY MODEL")
    logger.info("="*60)
    
    y_train_vol = y['volatility'].iloc[idx_train]
    y_test_vol = y['volatility'].iloc[idx_test]
    
    vol_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    vol_model.fit(X_train, y_train_vol)
    
    train_r2 = vol_model.score(X_train, y_train_vol)
    test_r2 = vol_model.score(X_test, y_test_vol)
    
    models['volatility_model'] = vol_model
    metrics['volatility_model'] = {'train_r2': train_r2, 'test_r2': test_r2}
    logger.info(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
    # 5. Trend Strength
    logger.info("\n" + "="*60)
    logger.info("5. TREND STRENGTH")
    logger.info("="*60)
    
    y_train_trend = y['trend_strength'].iloc[idx_train]
    y_test_trend = y['trend_strength'].iloc[idx_test]
    
    trend_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    trend_model.fit(X_train, y_train_trend)
    
    train_r2 = trend_model.score(X_train, y_train_trend)
    test_r2 = trend_model.score(X_test, y_test_trend)
    
    models['trend_strength'] = trend_model
    metrics['trend_strength'] = {'train_r2': train_r2, 'test_r2': test_r2}
    logger.info(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
    # Save models
    logger.info("\n" + "="*60)
    logger.info("SAVING MODELS")
    logger.info("="*60)
    
    output_dir = Path("data/models_unified")
    output_dir.mkdir(exist_ok=True)
    
    for name, model in models.items():
        output_path = output_dir / f"{name}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"  Saved {name} to {output_path}")
    
    # Save feature names
    feature_path = output_dir / "feature_names.pkl"
    with open(feature_path, 'wb') as f:
        pickle.dump(list(X.columns), f)
    logger.info(f"  Saved feature names to {feature_path}")
    
    # Save metrics
    import json
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Saved metrics to {metrics_path}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    
    for name, m in metrics.items():
        if 'test_accuracy' in m:
            logger.info(f"  {name}: accuracy={m['test_accuracy']:.4f}")
        else:
            logger.info(f"  {name}: R²={m['test_r2']:.4f}")
    
    return models, metrics


if __name__ == "__main__":
    train_all_models()
