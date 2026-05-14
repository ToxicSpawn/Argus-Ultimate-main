#!/usr/bin/env python3
"""
Multi-Timeframe ML Training with 3 Years of Historical Data.

Features:
- Cross-timeframe features (daily trend + hourly entry)
- Multi-timeframe regime detection
- Ensemble models with walk-forward validation
- Regime-specific models (bull/bear/sideways)
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def load_historical_data() -> Dict:
    """Load historical data from pickle."""
    data_path = Path("data/historical/historical_data.pkl")
    
    if not data_path.exists():
        logger.error(f"Historical data not found: {data_path}")
        return {}
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Loaded data for {len(data)} symbols")
    return data


def ohlcv_to_df(ohlcv_list: List[Dict]) -> pd.DataFrame:
    """Convert OHLCV list to DataFrame."""
    df = pd.DataFrame(ohlcv_list)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    df = df.sort_index()
    return df


def generate_features_single_tf(df: pd.DataFrame, tf_name: str) -> pd.DataFrame:
    """Generate features for a single timeframe."""
    features = pd.DataFrame(index=df.index)
    prefix = tf_name + "_"
    
    # Returns
    for period in [1, 4, 12, 24, 48]:
        features[f'{prefix}returns_{period}'] = df['close'].pct_change(period)
    
    # Volatility
    returns = df['close'].pct_change()
    for period in [12, 24, 48]:
        features[f'{prefix}volatility_{period}'] = returns.rolling(period).std()
    
    # Volume
    vol_sma = df['volume'].rolling(24).mean()
    features[f'{prefix}volume_ratio'] = df['volume'] / vol_sma.clip(lower=1e-8)
    
    # Price position
    high_24 = df['high'].rolling(24).max()
    low_24 = df['low'].rolling(24).min()
    features[f'{prefix}price_position'] = (df['close'] - low_24) / (high_24 - low_24).clip(lower=1e-8)
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.clip(lower=1e-8)
    features[f'{prefix}rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    features[f'{prefix}macd'] = ema12 - ema26
    features[f'{prefix}macd_signal'] = features[f'{prefix}macd'].ewm(span=9).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features[f'{prefix}atr'] = true_range.rolling(14).mean() / df['close']
    
    # Bollinger position
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    features[f'{prefix}bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower).clip(lower=1e-8)
    
    # Trend direction
    features[f'{prefix}trend_up'] = (df['close'] > df['close'].rolling(20).mean()).astype(float)
    features[f'{prefix}trend_strength'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std().clip(lower=1e-8)
    
    return features


def generate_multi_tf_features(symbol_data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate features from multiple timeframes aligned to 1h base."""
    # Target timeframe for alignment
    target_tf = "1h"
    
    if target_tf not in symbol_data:
        logger.warning(f"Missing {target_tf} data")
        return None, None
    
    # Get base DataFrame (1h)
    base_df = ohlcv_to_df(symbol_data[target_tf])
    
    if len(base_df) < 1000:
        logger.warning(f"Insufficient data: {len(base_df)} candles")
        return None, None
    
    # Generate base features
    all_features = generate_features_single_tf(base_df, "1h")
    
    # Add higher timeframe features by resampling
    for tf_name, tf_data in [("4h", "4h"), ("1d", "1d")]:
        if tf_name in symbol_data:
            tf_df = ohlcv_to_df(symbol_data[tf_name])
            
            # Resample to 1h and forward-fill
            tf_features = generate_features_single_tf(tf_df, tf_name)
            
            # Align to 1h index (forward fill higher TF values)
            tf_features_resampled = tf_features.reindex(base_df.index, method='ffill')
            all_features = pd.concat([all_features, tf_features_resampled], axis=1)
    
    # Add cross-timeframe features
    if "1d_returns_1" in all_features.columns and "1h_returns_4" in all_features.columns:
        # Daily trend vs hourly momentum
        all_features['daily_hourly_alignment'] = (
            np.sign(all_features["1d_returns_1"]) == np.sign(all_features["1h_returns_4"])
        ).astype(float)
        
        # Multi-timeframe RSI divergence
        if "1d_rsi" in all_features.columns and "1h_rsi" in all_features.columns:
            all_features['rsi_divergence'] = all_features["1h_rsi"] - all_features["1d_rsi"]
    
    # Generate labels (based on 4h forward returns)
    labels = pd.DataFrame(index=base_df.index)
    
    fwd_4h = base_df['close'].pct_change(4).shift(-4)
    fwd_24h = base_df['close'].pct_change(24).shift(-24)
    
    # Signal (3-class: sell, hold, buy)
    labels['signal'] = pd.cut(fwd_4h, bins=[-np.inf, -0.01, 0.01, np.inf], labels=[0, 1, 2])
    
    # Regime (3-class: bear, sideways, bull)
    labels['regime'] = pd.cut(fwd_24h, bins=[-np.inf, -0.03, 0.03, np.inf], labels=[0, 1, 2])
    
    # Position size (0-1)
    labels['position_size'] = np.clip(np.abs(fwd_4h) * 30, 0, 1)
    
    # Volatility (forward 24h)
    labels['volatility'] = fwd_24h.rolling(24).std().shift(-24)
    
    # Trend strength
    rolling_up = (fwd_4h > 0).rolling(24).mean()
    labels['trend_strength'] = np.abs(rolling_up - 0.5) * 2
    
    return all_features, labels


def create_ensemble_classifier(n_estimators: int = 150) -> VotingClassifier:
    """Create ensemble classifier."""
    estimators = [
        ('gb', GradientBoostingClassifier(
            n_estimators=n_estimators, 
            max_depth=5, 
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )),
        ('rf', RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )),
    ]
    return VotingClassifier(estimators=estimators, voting='soft')


def create_ensemble_regressor(n_estimators: int = 150) -> VotingRegressor:
    """Create ensemble regressor."""
    estimators = [
        ('gb', GradientBoostingRegressor(
            n_estimators=n_estimators, 
            max_depth=5, 
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )),
        ('rf', RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )),
    ]
    return VotingRegressor(estimators=estimators)


def walk_forward_validation(X: pd.DataFrame, y: pd.Series, model, n_splits: int = 5) -> Dict:
    """Perform walk-forward validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    scores = []
    train_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        train_scores.append(train_score)
        scores.append(test_score)
        
        logger.info(f"    Fold {fold+1}: train={train_score:.4f}, test={test_score:.4f}")
    
    return {
        'mean_test': np.mean(scores),
        'std_test': np.std(scores),
        'mean_train': np.mean(train_scores),
        'scores': scores,
    }


def main():
    """Main training pipeline."""
    logger.info("="*70)
    logger.info("MULTI-TIMEFRAME ML TRAINING (3 YEARS DATA)")
    logger.info("="*70)
    
    # Load historical data
    data = load_historical_data()
    if not data:
        return
    
    # Process all symbols
    all_features = []
    all_labels = []
    
    for i, (symbol, symbol_data) in enumerate(data.items()):
        logger.info(f"\n[{i+1}/{len(data)}] Processing {symbol}...")
        
        features, labels = generate_multi_tf_features(symbol_data)
        if features is None:
            continue
        
        combined = pd.concat([features, labels], axis=1).dropna()
        
        if len(combined) < 2000:
            logger.warning(f"  Skipped: only {len(combined)} samples")
            continue
        
        all_features.append(combined[features.columns])
        all_labels.append(combined[labels.columns])
        
        logger.info(f"  {len(combined)} samples, {len(features.columns)} features")
    
    # Combine all data
    X = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_labels, ignore_index=True)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"TOTAL DATASET")
    logger.info(f"{'='*70}")
    logger.info(f"Samples: {len(X):,}")
    logger.info(f"Features: {len(X.columns)}")
    
    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # Time-based split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
    
    logger.info(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    # Output directory
    output_dir = Path("data/models_mtf")
    output_dir.mkdir(exist_ok=True)
    
    all_models = {}
    all_metrics = {}
    
    # ========================================================================
    # 1. SIGNAL CLASSIFIER
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("1. SIGNAL CLASSIFIER")
    logger.info(f"{'='*70}")
    
    y_signal_train = y['signal'].iloc[:split_idx]
    y_signal_test = y['signal'].iloc[split_idx:]
    
    # Walk-forward validation
    logger.info("\n  Walk-forward validation:")
    wf_scores = walk_forward_validation(
        X_train, y_signal_train, 
        GradientBoostingClassifier(n_estimators=100, max_depth=5)
    )
    logger.info(f"  WF Mean: {wf_scores['mean_test']:.4f} (+/- {wf_scores['std_test']:.4f})")
    
    # Train ensemble
    logger.info("\n  Training ensemble...")
    signal_model = create_ensemble_classifier()
    signal_model.fit(X_train, y_signal_train)
    
    train_acc = signal_model.score(X_train, y_signal_train)
    test_acc = signal_model.score(X_test, y_signal_test)
    logger.info(f"  Train: {train_acc:.4f}, Test: {test_acc:.4f}")
    
    all_models['signal_classifier'] = signal_model
    all_metrics['signal_classifier'] = {
        'train_accuracy': train_acc, 
        'test_accuracy': test_acc,
        'wf_mean': wf_scores['mean_test'],
        'wf_std': wf_scores['std_test'],
    }
    
    # ========================================================================
    # 2. REGIME CLASSIFIER
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("2. REGIME CLASSIFIER")
    logger.info(f"{'='*70}")
    
    y_regime_train = y['regime'].iloc[:split_idx]
    y_regime_test = y['regime'].iloc[split_idx:]
    
    regime_model = create_ensemble_classifier()
    regime_model.fit(X_train, y_regime_train)
    
    train_acc = regime_model.score(X_train, y_regime_train)
    test_acc = regime_model.score(X_test, y_regime_test)
    logger.info(f"  Train: {train_acc:.4f}, Test: {test_acc:.4f}")
    
    all_models['regime_classifier'] = regime_model
    all_metrics['regime_classifier'] = {'train_accuracy': train_acc, 'test_accuracy': test_acc}
    
    # ========================================================================
    # 3. REGIME-SPECIFIC SIGNAL MODELS
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("3. REGIME-SPECIFIC SIGNAL MODELS")
    logger.info(f"{'='*70}")
    
    regime_names = {0: 'bear', 1: 'sideways', 2: 'bull'}
    
    for regime_id, regime_name in regime_names.items():
        # Filter training data for this regime
        regime_mask = y_regime_train == regime_id
        X_regime = X_train[regime_mask]
        y_regime_signal = y_signal_train[regime_mask]
        
        if len(X_regime) < 500:
            logger.warning(f"  Skipping {regime_name}: only {len(X_regime)} samples")
            continue
        
        logger.info(f"\n  {regime_name.upper()} regime ({len(X_regime):,} samples):")
        
        model = create_ensemble_classifier(n_estimators=100)
        model.fit(X_regime, y_regime_signal)
        
        train_acc = model.score(X_regime, y_regime_signal)
        logger.info(f"    Train accuracy: {train_acc:.4f}")
        
        all_models[f'signal_{regime_name}'] = model
        all_metrics[f'signal_{regime_name}'] = {
            'train_accuracy': train_acc,
            'n_samples': len(X_regime),
        }
    
    # ========================================================================
    # 4. POSITION SIZER
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("4. POSITION SIZER")
    logger.info(f"{'='*70}")
    
    y_pos_train = y['position_size'].iloc[:split_idx]
    y_pos_test = y['position_size'].iloc[split_idx:]
    
    pos_model = create_ensemble_regressor()
    pos_model.fit(X_train, y_pos_train)
    
    train_r2 = pos_model.score(X_train, y_pos_train)
    test_r2 = pos_model.score(X_test, y_pos_test)
    logger.info(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
    all_models['position_sizer'] = pos_model
    all_metrics['position_sizer'] = {'train_r2': train_r2, 'test_r2': test_r2}
    
    # ========================================================================
    # 5. VOLATILITY MODEL
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("5. VOLATILITY MODEL")
    logger.info(f"{'='*70}")
    
    y_vol_train = y['volatility'].iloc[:split_idx]
    y_vol_test = y['volatility'].iloc[split_idx:]
    
    vol_model = create_ensemble_regressor()
    vol_model.fit(X_train, y_vol_train)
    
    train_r2 = vol_model.score(X_train, y_vol_train)
    test_r2 = vol_model.score(X_test, y_vol_test)
    logger.info(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
    all_models['volatility_model'] = vol_model
    all_metrics['volatility_model'] = {'train_r2': train_r2, 'test_r2': test_r2}
    
    # ========================================================================
    # 6. TREND STRENGTH
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("6. TREND STRENGTH")
    logger.info(f"{'='*70}")
    
    y_trend_train = y['trend_strength'].iloc[:split_idx]
    y_trend_test = y['trend_strength'].iloc[split_idx:]
    
    trend_model = create_ensemble_regressor()
    trend_model.fit(X_train, y_trend_train)
    
    train_r2 = trend_model.score(X_train, y_trend_train)
    test_r2 = trend_model.score(X_test, y_trend_test)
    logger.info(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
    all_models['trend_strength'] = trend_model
    all_metrics['trend_strength'] = {'train_r2': train_r2, 'test_r2': test_r2}
    
    # ========================================================================
    # SAVE MODELS
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("SAVING MODELS")
    logger.info(f"{'='*70}")
    
    # Save models
    for name, model in all_models.items():
        with open(output_dir / f"{name}.pkl", 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"  Saved {name}")
    
    # Save scaler
    with open(output_dir / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open(output_dir / "feature_names.pkl", 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    # Save metrics
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Data: {len(X):,} samples from {len(data)} symbols")
    logger.info(f"Features: {len(X.columns)} (multi-timeframe)")
    logger.info(f"\nModel Performance:")
    
    for name, metrics in all_metrics.items():
        if 'test_accuracy' in metrics:
            logger.info(f"  {name}: accuracy={metrics['test_accuracy']:.4f}")
        elif 'test_r2' in metrics:
            logger.info(f"  {name}: R²={metrics['test_r2']:.4f}")
    
    logger.info(f"\nModels saved to: {output_dir}")


if __name__ == "__main__":
    main()
