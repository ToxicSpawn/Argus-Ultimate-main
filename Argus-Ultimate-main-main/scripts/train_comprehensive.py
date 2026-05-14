#!/usr/bin/env python3
"""
Comprehensive ML training with all improvements:
1. Extended data (6+ months 15-min candles)
2. Advanced features (order book, funding rates, open interest)
3. Walk-forward validation
4. Ensemble methods (multiple model types)
5. Regime-specific models (bull/bear/sideways)
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
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def generate_advanced_features(ohlcv: pd.DataFrame, order_book: Optional[Dict] = None,
                                funding_rates: Optional[List] = None,
                                open_interest: Optional[List] = None) -> pd.DataFrame:
    """Generate advanced features including order book, funding, OI."""
    df = ohlcv.copy()
    features = pd.DataFrame(index=df.index)
    
    # === Basic Price Features ===
    features['returns_1'] = df['close'].pct_change(1)
    features['returns_4'] = df['close'].pct_change(4)
    features['returns_12'] = df['close'].pct_change(12)
    features['returns_24'] = df['close'].pct_change(24)
    features['returns_48'] = df['close'].pct_change(48)
    features['returns_96'] = df['close'].pct_change(96)  # 24 hours at 15m
    
    # === Volatility Features ===
    features['volatility_8'] = features['returns_1'].rolling(8).std()
    features['volatility_24'] = features['returns_1'].rolling(24).std()
    features['volatility_48'] = features['returns_1'].rolling(48).std()
    features['volatility_96'] = features['returns_1'].rolling(96).std()
    features['volatility_ratio'] = features['volatility_8'] / features['volatility_96'].clip(lower=1e-8)
    
    # === Volume Features ===
    features['volume_sma_24'] = df['volume'].rolling(24).mean()
    features['volume_sma_96'] = df['volume'].rolling(96).mean()
    features['volume_ratio'] = df['volume'] / features['volume_sma_24'].clip(lower=1e-8)
    features['volume_trend'] = features['volume_sma_24'] / features['volume_sma_96'].clip(lower=1e-8)
    
    # === Price Position Features ===
    high_24 = df['high'].rolling(24).max()
    low_24 = df['low'].rolling(24).min()
    features['price_position_24'] = (df['close'] - low_24) / (high_24 - low_24).clip(lower=1e-8)
    
    high_96 = df['high'].rolling(96).max()
    low_96 = df['low'].rolling(96).min()
    features['price_position_96'] = (df['close'] - low_96) / (high_96 - low_96).clip(lower=1e-8)
    
    # === RSI ===
    for period in [7, 14, 28]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.clip(lower=1e-8)
        features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # === MACD ===
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    features['macd'] = ema12 - ema26
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    features['macd_histogram'] = features['macd'] - features['macd_signal']
    
    # === Bollinger Bands ===
    bb_period = 20
    bb_mid = df['close'].rolling(bb_period).mean()
    bb_std = df['close'].rolling(bb_period).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower).clip(lower=1e-8)
    features['bb_width'] = (bb_upper - bb_lower) / bb_mid
    
    # === ATR ===
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features['atr_14'] = true_range.rolling(14).mean()
    features['atr_ratio'] = features['atr_14'] / df['close']
    
    # === OBV ===
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    features['obv_sma_24'] = obv.rolling(24).mean()
    features['obv_slope'] = obv.rolling(24).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 24 else 0)
    
    # === Momentum ===
    features['momentum_12'] = df['close'] / df['close'].shift(12) - 1
    features['momentum_24'] = df['close'] / df['close'].shift(24) - 1
    features['momentum_48'] = df['close'] / df['close'].shift(48) - 1
    
    # === Stochastic ===
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    features['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14).clip(lower=1e-8)
    features['stoch_d'] = features['stoch_k'].rolling(3).mean()
    
    # === Time Features ===
    if 'timestamp' in df.columns:
        dt = pd.to_datetime(df['timestamp'], unit='ms')
        features['hour'] = dt.dt.hour
        features['day_of_week'] = dt.dt.dayofweek
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    
    # === Order Book Features (if available) ===
    if order_book and 'b' in order_book and 'a' in order_book:
        bids = order_book['b'][:20]  # Top 20 bids
        asks = order_book['a'][:20]  # Top 20 asks
        
        if bids and asks:
            # Bid-ask spread
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            features['spread'] = (best_ask - best_bid) / best_bid
            
            # Order book imbalance
            bid_volume = sum(float(b[1]) for b in bids)
            ask_volume = sum(float(a[1]) for a in asks)
            total_volume = bid_volume + ask_volume
            features['ob_imbalance'] = (bid_volume - ask_volume) / max(total_volume, 1e-8)
            
            # Depth imbalance at different levels
            for depth in [5, 10]:
                bids_depth = sum(float(b[1]) for b in bids[:depth])
                asks_depth = sum(float(a[1]) for a in asks[:depth])
                total_depth = bids_depth + asks_depth
                features[f'ob_imbalance_{depth}'] = (bids_depth - asks_depth) / max(total_depth, 1e-8)
    
    # === Funding Rate Features (if available) ===
    if funding_rates and len(funding_rates) > 0:
        fr_df = pd.DataFrame(funding_rates)
        fr_df['fundingRate'] = fr_df['fundingRate'].astype(float)
        fr_df['timestamp'] = pd.to_numeric(fr_df['fundingRateTimestamp'])
        fr_df = fr_df.sort_values('timestamp')
        
        features['funding_rate'] = fr_df['fundingRate'].iloc[0] if len(fr_df) > 0 else 0
        features['funding_rate_sma'] = fr_df['fundingRate'].rolling(8).mean().iloc[-1] if len(fr_df) >= 8 else 0
        features['funding_rate_trend'] = fr_df['fundingRate'].diff().iloc[-1] if len(fr_df) > 1 else 0
    
    # === Open Interest Features (if available) ===
    if open_interest and len(open_interest) > 0:
        oi_df = pd.DataFrame(open_interest)
        oi_df['openInterest'] = oi_df['openInterest'].astype(float)
        oi_df['timestamp'] = pd.to_numeric(oi_df['timestamp'])
        oi_df = oi_df.sort_values('timestamp')
        
        features['open_interest'] = oi_df['openInterest'].iloc[-1] if len(oi_df) > 0 else 0
        features['oi_change_24h'] = oi_df['openInterest'].pct_change(24).iloc[-1] if len(oi_df) >= 24 else 0
    
    return features


def generate_labels(df: pd.DataFrame, fwd_periods: Dict[str, int] = None) -> pd.DataFrame:
    """Generate labels for training."""
    if fwd_periods is None:
        fwd_periods = {
            'signal': 4,      # 1 hour ahead for signal
            'regime': 24,     # 6 hours ahead for regime
            'volatility': 12, # 3 hours ahead for volatility
            'trend': 48,      # 12 hours ahead for trend
        }
    
    labels = pd.DataFrame(index=df.index)
    
    # Forward returns
    for name, periods in fwd_periods.items():
        fwd_return = df['close'].pct_change(periods).shift(-periods)
        
        if name == 'signal':
            # 3-class: sell, hold, buy
            labels['signal'] = pd.cut(fwd_return, 
                                       bins=[-np.inf, -0.005, 0.005, np.inf], 
                                       labels=[0, 1, 2])
        
        elif name == 'regime':
            # 3-class regime for regime-specific models
            fwd_return_48 = df['close'].pct_change(48).shift(-48)
            labels['regime'] = pd.cut(fwd_return_48,
                                       bins=[-np.inf, -0.02, 0.02, np.inf],
                                       labels=[0, 1, 2])  # bear, sideways, bull
        
        elif name == 'volatility':
            labels['volatility'] = fwd_return.rolling(periods).std().shift(-periods)
        
        elif name == 'trend':
            # Trend strength 0-1
            rolling_up = (fwd_return > 0).rolling(periods).mean()
            labels['trend_strength'] = np.abs(rolling_up - 0.5) * 2
    
    # Position size (based on Kelly-like criterion)
    fwd_ret_12 = df['close'].pct_change(12).shift(-12)
    win_rate = (fwd_ret_12 > 0).rolling(48).mean()
    avg_win = fwd_ret_12.where(fwd_ret_12 > 0, 0).rolling(48).mean()
    avg_loss = fwd_ret_12.where(fwd_ret_12 < 0, 0).abs().rolling(48).mean()
    kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss.clip(lower=1e-8)).clip(lower=1e-8)
    labels['position_size'] = np.clip(kelly * 2, 0, 1)  # Half-Kelly
    
    return labels


# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

def walk_forward_validation(X: pd.DataFrame, y: pd.Series, model, 
                            n_splits: int = 5, min_train_size: int = 1000) -> Dict:
    """Perform walk-forward validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    scores = []
    train_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(train_idx) < min_train_size:
            continue
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Score
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        train_scores.append(train_score)
        scores.append(test_score)
        
        logger.info(f"  Fold {fold+1}: train={train_score:.4f}, test={test_score:.4f}")
    
    return {
        'mean_test_score': np.mean(scores),
        'std_test_score': np.std(scores),
        'mean_train_score': np.mean(train_scores),
        'test_scores': scores,
        'train_scores': train_scores,
    }


# ============================================================================
# ENSEMBLE MODELS
# ============================================================================

def create_ensemble_classifier(X: pd.DataFrame, y: pd.Series, name: str) -> VotingClassifier:
    """Create ensemble classifier with multiple model types."""
    estimators = [
        ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ]
    
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    ensemble.fit(X, y)
    
    return ensemble


def create_ensemble_regressor(X: pd.DataFrame, y: pd.Series, name: str) -> VotingRegressor:
    """Create ensemble regressor with multiple model types."""
    estimators = [
        ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ('ridge', Ridge(alpha=1.0)),
    ]
    
    ensemble = VotingRegressor(estimators=estimators)
    ensemble.fit(X, y)
    
    return ensemble


def create_stacking_classifier(X: pd.DataFrame, y: pd.Series) -> StackingClassifier:
    """Create stacking classifier."""
    base_estimators = [
        ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
    ]
    
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3
    )
    stacking.fit(X, y)
    
    return stacking


# ============================================================================
# REGIME-SPECIFIC MODELS
# ============================================================================

def train_regime_specific_models(X: pd.DataFrame, y_signal: pd.Series, 
                                  y_regime: pd.Series) -> Dict:
    """Train separate signal models for each regime."""
    regime_models = {}
    
    regime_names = {0: 'bear', 1: 'sideways', 2: 'bull'}
    
    for regime_id, regime_name in regime_names.items():
        # Filter data for this regime
        regime_mask = y_regime == regime_id
        X_regime = X[regime_mask]
        y_regime_signal = y_signal[regime_mask]
        
        if len(X_regime) < 100:
            logger.warning(f"  Skipping {regime_name} regime: only {len(X_regime)} samples")
            continue
        
        logger.info(f"\n  Training {regime_name} regime model ({len(X_regime)} samples)...")
        
        # Train ensemble for this regime
        model = create_ensemble_classifier(X_regime, y_regime_signal, f"signal_{regime_name}")
        
        # Evaluate
        train_score = model.score(X_regime, y_regime_signal)
        logger.info(f"    Train accuracy: {train_score:.4f}")
        
        regime_models[regime_name] = {
            'model': model,
            'train_accuracy': train_score,
            'n_samples': len(X_regime),
        }
    
    return regime_models


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def load_extended_data() -> Dict:
    """Load extended market data."""
    data_path = Path("data/extended_market_data.pkl")
    
    if not data_path.exists():
        logger.error(f"Extended data not found: {data_path}")
        logger.info("Run fetch_extended_data.py first!")
        return {}
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Loaded data for {len(data)} symbols")
    return data


def process_symbol(symbol: str, symbol_data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process a single symbol's data."""
    # Extract OHLCV
    ohlcv_list = symbol_data.get('ohlcv_15m', [])
    if len(ohlcv_list) < 1000:
        return None, None
    
    ohlcv_df = pd.DataFrame(ohlcv_list)
    ohlcv_df['datetime'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ms')
    ohlcv_df.set_index('datetime', inplace=True)
    ohlcv_df = ohlcv_df.sort_index()
    
    # Generate features
    order_book = symbol_data.get('order_book')
    funding_rates = symbol_data.get('funding_rates')
    open_interest = symbol_data.get('open_interest')
    
    features = generate_advanced_features(ohlcv_df, order_book, funding_rates, open_interest)
    
    # Generate labels
    labels = generate_labels(ohlcv_df)
    
    return features, labels


def main():
    """Main training pipeline."""
    logger.info("="*70)
    logger.info("COMPREHENSIVE ML TRAINING - ALL IMPROVEMENTS")
    logger.info("="*70)
    
    # Load data
    data = load_extended_data()
    if not data:
        return
    
    # Process all symbols
    all_features = []
    all_labels = []
    
    for i, (symbol, symbol_data) in enumerate(data.items()):
        logger.info(f"\n[{i+1}/{len(data)}] Processing {symbol}...")
        
        features, labels = process_symbol(symbol, symbol_data)
        if features is None:
            logger.warning(f"  Skipped {symbol}")
            continue
        
        # Combine and drop NaN
        combined = pd.concat([features, labels], axis=1).dropna()
        
        if len(combined) < 500:
            logger.warning(f"  Skipped {symbol}: only {len(combined)} samples")
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
    logger.info(f"Samples: {len(X)}")
    logger.info(f"Features: {len(X.columns)}")
    logger.info(f"Feature names: {list(X.columns[:20])}...")
    
    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # Split data (80% train, 20% test - time-based)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Output directory
    output_dir = Path("data/models_enhanced")
    output_dir.mkdir(exist_ok=True)
    
    all_models = {}
    all_metrics = {}
    
    # ========================================================================
    # 1. SIGNAL CLASSIFIER (with ensemble + stacking)
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("1. SIGNAL CLASSIFIER (ENSEMBLE)")
    logger.info(f"{'='*70}")
    
    y_signal_train = y['signal'].iloc[:split_idx]
    y_signal_test = y['signal'].iloc[split_idx:]
    
    # Walk-forward validation
    logger.info("\nWalk-forward validation:")
    wf_scores = walk_forward_validation(X_train, y_signal_train, 
                                         GradientBoostingClassifier(n_estimators=100, max_depth=5))
    logger.info(f"WF Mean: {wf_scores['mean_test_score']:.4f} (+/- {wf_scores['std_test_score']:.4f})")
    
    # Train ensemble
    logger.info("\nTraining ensemble classifier...")
    signal_ensemble = create_ensemble_classifier(X_train, y_signal_train, "signal_ensemble")
    train_acc = signal_ensemble.score(X_train, y_signal_train)
    test_acc = signal_ensemble.score(X_test, y_signal_test)
    logger.info(f"Train: {train_acc:.4f}, Test: {test_acc:.4f}")
    
    # Train stacking
    logger.info("\nTraining stacking classifier...")
    signal_stacking = create_stacking_classifier(X_train, y_signal_train)
    train_acc_stack = signal_stacking.score(X_train, y_signal_train)
    test_acc_stack = signal_stacking.score(X_test, y_signal_test)
    logger.info(f"Train: {train_acc_stack:.4f}, Test: {test_acc_stack:.4f}")
    
    all_models['signal_ensemble'] = signal_ensemble
    all_models['signal_stacking'] = signal_stacking
    all_metrics['signal_ensemble'] = {'train_accuracy': train_acc, 'test_accuracy': test_acc}
    all_metrics['signal_stacking'] = {'train_accuracy': train_acc_stack, 'test_accuracy': test_acc_stack}
    
    # ========================================================================
    # 2. REGIME CLASSIFIER
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("2. REGIME CLASSIFIER")
    logger.info(f"{'='*70}")
    
    y_regime_train = y['regime'].iloc[:split_idx]
    y_regime_test = y['regime'].iloc[split_idx:]
    
    regime_model = create_ensemble_classifier(X_train, y_regime_train, "regime")
    train_acc = regime_model.score(X_train, y_regime_train)
    test_acc = regime_model.score(X_test, y_regime_test)
    logger.info(f"Train: {train_acc:.4f}, Test: {test_acc:.4f}")
    
    all_models['regime_classifier'] = regime_model
    all_metrics['regime_classifier'] = {'train_accuracy': train_acc, 'test_accuracy': test_acc}
    
    # ========================================================================
    # 3. REGIME-SPECIFIC SIGNAL MODELS
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("3. REGIME-SPECIFIC SIGNAL MODELS")
    logger.info(f"{'='*70}")
    
    regime_models = train_regime_specific_models(X_train, y_signal_train, y_regime_train)
    
    for regime_name, regime_data in regime_models.items():
        all_models[f'signal_{regime_name}'] = regime_data['model']
        all_metrics[f'signal_{regime_name}'] = {
            'train_accuracy': regime_data['train_accuracy'],
            'n_samples': regime_data['n_samples']
        }
    
    # ========================================================================
    # 4. POSITION SIZER
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("4. POSITION SIZER (ENSEMBLE)")
    logger.info(f"{'='*70}")
    
    y_pos_train = y['position_size'].iloc[:split_idx]
    y_pos_test = y['position_size'].iloc[split_idx:]
    
    pos_ensemble = create_ensemble_regressor(X_train, y_pos_train, "position")
    train_r2 = pos_ensemble.score(X_train, y_pos_train)
    test_r2 = pos_ensemble.score(X_test, y_pos_test)
    logger.info(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
    all_models['position_sizer'] = pos_ensemble
    all_metrics['position_sizer'] = {'train_r2': train_r2, 'test_r2': test_r2}
    
    # ========================================================================
    # 5. VOLATILITY MODEL
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("5. VOLATILITY MODEL")
    logger.info(f"{'='*70}")
    
    y_vol_train = y['volatility'].iloc[:split_idx]
    y_vol_test = y['volatility'].iloc[split_idx:]
    
    vol_model = create_ensemble_regressor(X_train, y_vol_train, "volatility")
    train_r2 = vol_model.score(X_train, y_vol_train)
    test_r2 = vol_model.score(X_test, y_vol_test)
    logger.info(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
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
    
    trend_model = create_ensemble_regressor(X_train, y_trend_train, "trend")
    train_r2 = trend_model.score(X_train, y_trend_train)
    test_r2 = trend_model.score(X_test, y_trend_test)
    logger.info(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
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
        model_path = output_dir / f"{name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"  Saved {name}")
    
    # Save scaler
    scaler_path = output_dir / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"  Saved scaler")
    
    # Save feature names
    features_path = output_dir / "feature_names.pkl"
    with open(features_path, 'wb') as f:
        pickle.dump(list(X.columns), f)
    logger.info(f"  Saved feature names ({len(X.columns)} features)")
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"  Saved metrics")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info(f"{'='*70}")
    
    for name, metrics in all_metrics.items():
        if 'test_accuracy' in metrics:
            logger.info(f"  {name}: accuracy={metrics['test_accuracy']:.4f}")
        elif 'test_r2' in metrics:
            logger.info(f"  {name}: R²={metrics['test_r2']:.4f}")
        else:
            logger.info(f"  {name}: {metrics}")
    
    logger.info(f"\nModels saved to: {output_dir}")


if __name__ == "__main__":
    main()
