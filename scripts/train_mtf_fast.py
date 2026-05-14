#!/usr/bin/env python3
"""
Fast Multi-Timeframe Training (skip slow walk-forward).
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

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
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def load_historical_data() -> Dict:
    with open("data/historical/historical_data.pkl", 'rb') as f:
        return pickle.load(f)


def ohlcv_to_df(ohlcv_list: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv_list)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df.sort_index()


def generate_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    
    for period in [1, 4, 12, 24, 48]:
        features[f'{prefix}returns_{period}'] = df['close'].pct_change(period)
    
    returns = df['close'].pct_change()
    for period in [12, 24, 48]:
        features[f'{prefix}volatility_{period}'] = returns.rolling(period).std()
    
    vol_sma = df['volume'].rolling(24).mean()
    features[f'{prefix}volume_ratio'] = df['volume'] / vol_sma.clip(lower=1e-8)
    
    high_24 = df['high'].rolling(24).max()
    low_24 = df['low'].rolling(24).min()
    features[f'{prefix}price_position'] = (df['close'] - low_24) / (high_24 - low_24).clip(lower=1e-8)
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.clip(lower=1e-8)
    features[f'{prefix}rsi'] = 100 - (100 / (1 + rs))
    
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    features[f'{prefix}macd'] = ema12 - ema26
    
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    features[f'{prefix}bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower).clip(lower=1e-8)
    
    return features


def process_symbol(symbol_data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_df = ohlcv_to_df(symbol_data['1h'])
    
    all_features = generate_features(base_df, "1h")
    
    for tf in ['4h', '1d']:
        if tf in symbol_data:
            tf_df = ohlcv_to_df(symbol_data[tf])
            tf_features = generate_features(tf_df, tf)
            tf_features_aligned = tf_features.reindex(base_df.index, method='ffill')
            all_features = pd.concat([all_features, tf_features_aligned], axis=1)
    
    fwd_4h = base_df['close'].pct_change(4).shift(-4)
    fwd_24h = base_df['close'].pct_change(24).shift(-24)
    
    labels = pd.DataFrame(index=base_df.index)
    labels['signal'] = pd.cut(fwd_4h, bins=[-np.inf, -0.01, 0.01, np.inf], labels=[0, 1, 2])
    labels['regime'] = pd.cut(fwd_24h, bins=[-np.inf, -0.03, 0.03, np.inf], labels=[0, 1, 2])
    labels['position_size'] = np.clip(np.abs(fwd_4h) * 30, 0, 1)
    labels['volatility'] = fwd_24h.rolling(24).std().shift(-24)
    labels['trend_strength'] = np.abs((fwd_4h > 0).rolling(24).mean() - 0.5) * 2
    
    return all_features, labels


def main():
    logger.info("="*70)
    logger.info("FAST MULTI-TIMEFRAME TRAINING (3 YEARS)")
    logger.info("="*70)
    
    data = load_historical_data()
    logger.info(f"Loaded {len(data)} symbols")
    
    all_X, all_y = [], []
    
    for symbol, symbol_data in data.items():
        logger.info(f"Processing {symbol}...")
        features, labels = process_symbol(symbol_data)
        combined = pd.concat([features, labels], axis=1).dropna()
        if len(combined) > 2000:
            all_X.append(combined[features.columns])
            all_y.append(combined[labels.columns])
            logger.info(f"  {len(combined)} samples, {len(features.columns)} features")
    
    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X_scaled.iloc[:split], X_scaled.iloc[split:]
    
    logger.info(f"\nTotal: {len(X):,} samples, {len(X.columns)} features")
    logger.info(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    output_dir = Path("data/models_mtf")
    output_dir.mkdir(exist_ok=True)
    
    metrics = {}
    
    # Signal Classifier
    logger.info("\n" + "="*70)
    logger.info("SIGNAL CLASSIFIER")
    y_sig_train, y_sig_test = y['signal'].iloc[:split], y['signal'].iloc[split:]
    
    sig_model = VotingClassifier([
        ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10)),
    ], voting='soft')
    sig_model.fit(X_train, y_sig_train)
    
    train_acc = sig_model.score(X_train, y_sig_train)
    test_acc = sig_model.score(X_test, y_sig_test)
    logger.info(f"Train: {train_acc:.4f}, Test: {test_acc:.4f}")
    metrics['signal_classifier'] = {'train': train_acc, 'test': test_acc}
    pickle.dump(sig_model, open(output_dir / "signal_classifier.pkl", 'wb'))
    
    # Regime Classifier
    logger.info("\nREGIME CLASSIFIER")
    y_reg_train, y_reg_test = y['regime'].iloc[:split], y['regime'].iloc[split:]
    
    reg_model = VotingClassifier([
        ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10)),
    ], voting='soft')
    reg_model.fit(X_train, y_reg_train)
    
    train_acc = reg_model.score(X_train, y_reg_train)
    test_acc = reg_model.score(X_test, y_reg_test)
    logger.info(f"Train: {train_acc:.4f}, Test: {test_acc:.4f}")
    metrics['regime_classifier'] = {'train': train_acc, 'test': test_acc}
    pickle.dump(reg_model, open(output_dir / "regime_classifier.pkl", 'wb'))
    
    # Regime-specific models
    logger.info("\nREGIME-SPECIFIC MODELS")
    for regime_id, regime_name in [(0, 'bear'), (1, 'sideways'), (2, 'bull')]:
        mask = y_reg_train == regime_id
        X_r = X_train[mask]
        y_r = y_sig_train[mask]
        
        if len(X_r) < 500:
            logger.info(f"  {regime_name}: skipped ({len(X_r)} samples)")
            continue
        
        m = VotingClassifier([
            ('gb', GradientBoostingClassifier(n_estimators=80, max_depth=5)),
            ('rf', RandomForestClassifier(n_estimators=80, max_depth=8)),
        ], voting='soft')
        m.fit(X_r, y_r)
        acc = m.score(X_r, y_r)
        logger.info(f"  {regime_name}: train={acc:.4f} ({len(X_r):,} samples)")
        metrics[f'signal_{regime_name}'] = {'train': acc, 'samples': len(X_r)}
        pickle.dump(m, open(output_dir / f"signal_{regime_name}.pkl", 'wb'))
    
    # Position Sizer
    logger.info("\nPOSITION SIZER")
    y_pos_train, y_pos_test = y['position_size'].iloc[:split], y['position_size'].iloc[split:]
    
    pos_model = VotingRegressor([
        ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=5)),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10)),
    ])
    pos_model.fit(X_train, y_pos_train)
    
    train_r2 = pos_model.score(X_train, y_pos_train)
    test_r2 = pos_model.score(X_test, y_pos_test)
    logger.info(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    metrics['position_sizer'] = {'train_r2': train_r2, 'test_r2': test_r2}
    pickle.dump(pos_model, open(output_dir / "position_sizer.pkl", 'wb'))
    
    # Volatility Model
    logger.info("\nVOLATILITY MODEL")
    y_vol_train, y_vol_test = y['volatility'].iloc[:split], y['volatility'].iloc[split:]
    
    vol_model = VotingRegressor([
        ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=5)),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10)),
    ])
    vol_model.fit(X_train, y_vol_train)
    
    train_r2 = vol_model.score(X_train, y_vol_train)
    test_r2 = vol_model.score(X_test, y_vol_test)
    logger.info(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    metrics['volatility_model'] = {'train_r2': train_r2, 'test_r2': test_r2}
    pickle.dump(vol_model, open(output_dir / "volatility_model.pkl", 'wb'))
    
    # Trend Strength
    logger.info("\nTREND STRENGTH")
    y_trend_train, y_trend_test = y['trend_strength'].iloc[:split], y['trend_strength'].iloc[split:]
    
    trend_model = VotingRegressor([
        ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=5)),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10)),
    ])
    trend_model.fit(X_train, y_trend_train)
    
    train_r2 = trend_model.score(X_train, y_trend_train)
    test_r2 = trend_model.score(X_test, y_trend_test)
    logger.info(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    metrics['trend_strength'] = {'train_r2': train_r2, 'test_r2': test_r2}
    pickle.dump(trend_model, open(output_dir / "trend_strength.pkl", 'wb'))
    
    # Save scaler and features
    pickle.dump(scaler, open(output_dir / "scaler.pkl", 'wb'))
    pickle.dump(list(X.columns), open(output_dir / "feature_names.pkl", 'wb'))
    json.dump(metrics, open(output_dir / "metrics.json", 'w'), indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    for name, m in metrics.items():
        if 'test' in m:
            logger.info(f"  {name}: test accuracy={m['test']:.4f}")
        elif 'test_r2' in m:
            logger.info(f"  {name}: test R²={m['test_r2']:.4f}")
        else:
            logger.info(f"  {name}: {m}")


if __name__ == "__main__":
    main()
