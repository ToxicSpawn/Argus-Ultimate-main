#!/usr/bin/env python3
"""
Train Argus ML models using the NEW unified training pipeline.

Uses:
- Data quality validation
- Early stopping
- LR scheduling
- Model registry with metadata

Usage:
    py scripts/train_with_pipeline.py
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


def generate_training_data(n_samples=5000):
    """Generate synthetic market data for training."""
    logger.info("Generating synthetic training data...")
    np.random.seed(42)
    
    X = pd.DataFrame({
        'returns_1h': np.random.randn(n_samples),
        'returns_4h': np.random.randn(n_samples),
        'returns_1d': np.random.randn(n_samples),
        'volume_ratio': np.random.uniform(0.5, 2.0, n_samples),
        'volatility_1h': np.random.uniform(0.001, 0.05, n_samples),
        'rsi_14': np.random.uniform(20, 80, n_samples),
        'macd_signal': np.random.randn(n_samples),
        'bb_position': np.random.uniform(0, 1, n_samples),
        'atr_14': np.random.uniform(0.001, 0.03, n_samples),
        'order_imbalance': np.random.uniform(-1, 1, n_samples),
        'spread_bps': np.random.uniform(1, 50, n_samples),
        'sentiment_score': np.random.uniform(-1, 1, n_samples),
        'hurst_exponent': np.random.uniform(0.3, 0.7, n_samples),
        'momentum_5': np.random.randn(n_samples),
        'momentum_20': np.random.randn(n_samples),
        'mean_reversion_z': np.random.uniform(-3, 3, n_samples),
    })
    
    y_regime = np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.2, 0.35, 0.15])
    y_position = np.clip(np.random.randn(n_samples) * 0.5, -1, 1)
    y_signal = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.2, 0.2])
    y_volatility = np.random.uniform(0.001, 0.05, n_samples)
    
    return X, y_regime, y_position, y_signal, y_volatility


def train_model(name, model, X_train, y_train, X_val, y_val, task_type="classification"):
    """Train a single model using the pipeline."""
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING: {name}")
    logger.info(f"{'='*60}")
    
    config = TrainingConfig(
        model_name=name,
        model_type="sklearn",
        patience=10,
        register_model=True,
        verbose=False,
    )
    
    pipeline = TrainingPipeline(config)
    result = pipeline.train_sklearn(model, X_train, y_train, X_val, y_val)
    
    if result.success:
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in result.final_metrics.items()])
        logger.info(f"[OK] {name}: {metrics_str}")
    else:
        logger.error(f"[FAIL] {name}: training failed")
    
    return result


def main():
    """Main training function."""
    logger.info("="*60)
    logger.info("ARGUS ML TRAINING - UNIFIED PIPELINE")
    logger.info("="*60)
    
    start_time = time.time()
    
    # Generate data
    X, y_regime, y_position, y_signal, y_volatility = generate_training_data()
    
    # Split 80/20
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    
    # Data quality check
    logger.info("\nValidating data quality...")
    quality = DataQualityPipeline(DataQualityConfig())
    passed, report = quality.validate(X_train)
    logger.info(f"Data quality: {'PASSED' if passed else 'FAILED'} (score={report.quality_score:.2f})")
    
    # Train models
    results = {}
    
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
    
    # 1. Regime Classifier
    results['regime_classifier'] = train_model(
        'regime_classifier',
        GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        X_train, y_regime[:split], X_val, y_regime[split:],
    )
    
    # 2. Position Sizer
    results['position_sizer'] = train_model(
        'position_sizer',
        GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        X_train, y_position[:split], X_val, y_position[split:],
    )
    
    # 3. Signal Classifier
    results['signal_classifier'] = train_model(
        'signal_classifier',
        RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        X_train, y_signal[:split], X_val, y_signal[split:],
    )
    
    # 4. Volatility Model
    results['volatility_model'] = train_model(
        'volatility_model',
        GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        X_train, y_volatility[:split], X_val, y_volatility[split:],
    )
    
    # Summary
    total_time = time.time() - start_time
    
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    
    success_count = sum(1 for r in results.values() if r.success)
    logger.info(f"Models trained: {success_count}/{len(results)}")
    logger.info(f"Total time: {total_time:.1f}s")
    
    # Show registered models
    registry = EnhancedModelRegistry()
    models = registry.list_models()
    logger.info(f"\nRegistered models in registry: {len(models)}")
    for m in models:
        logger.info(f"  - {m['name']} v{m['version']} ({m['status']})")


if __name__ == "__main__":
    main()
