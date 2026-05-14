#!/usr/bin/env python3
"""
Integrate trained ML models into Argus live trading.

This script:
1. Loads production models from the enhanced registry
2. Creates model wrappers for live prediction
3. Integrates with Argus trading system
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class MLModelIntegrator:
    """Integrates trained ML models into Argus trading."""
    
    def __init__(self, models_dir: str = "data/models_unified"):
        self.models_dir = Path(models_dir)
        self.models: Dict[str, Any] = {}
        self.feature_names: list = []
        self.model_info: Dict[str, Dict] = {}
        
    def load_production_models(self) -> Dict[str, Any]:
        """Load unified models from models directory."""
        import pickle
        import json
        
        model_names = ['regime_classifier', 'signal_classifier', 'position_sizer', 'volatility_model', 'trend_strength']
        
        logger.info(f"Loading unified models from {self.models_dir}...")
        
        # Load feature names
        feature_path = self.models_dir / "feature_names.pkl"
        if feature_path.exists():
            with open(feature_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            logger.info(f"  Features: {len(self.feature_names)}")
        
        # Load metrics
        metrics_path = self.models_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {}
        
        # Load each model
        for name in model_names:
            model_path = self.models_dir / f"{name}.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                self.models[name] = model
                self.model_info[name] = metrics.get(name, {})
                logger.info(f"  Loaded {name}: {self.model_info[name]}")
            else:
                logger.warning(f"  Missing {name}")
        
        return self.models
    
    def predict_regime(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict market regime."""
        if 'regime_classifier' not in self.models:
            return {'error': 'Model not loaded'}
        
        model = self.models['regime_classifier']
        
        # Create DataFrame with feature names
        import pandas as pd
        X = pd.DataFrame([features], columns=self.feature_names)
        
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        regimes = ['strong_bear', 'bear', 'sideways', 'bull', 'strong_bull']
        
        return {
            'regime': regimes[int(prediction)],
            'regime_id': int(prediction),
            'confidence': float(np.max(probability)),
            'probabilities': {
                regimes[i]: float(prob) 
                for i, prob in enumerate(probability)
            }
        }
    
    def predict_signal_strength(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict signal strength (buy/sell/hold)."""
        if 'signal_classifier' not in self.models:
            return {'error': 'Model not loaded'}
        
        model = self.models['signal_classifier']
        
        import pandas as pd
        X = pd.DataFrame([features], columns=self.feature_names)
        
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        signals = ['sell', 'hold', 'buy']
        
        return {
            'signal': signals[int(prediction)],
            'signal_id': int(prediction),
            'confidence': float(np.max(probability)),
            'strength': float(probability[2] - probability[0])  # buy - sell
        }
    
    def predict_position_size(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict optimal position size (0-1 scale)."""
        if 'position_sizer' not in self.models:
            return {'error': 'Model not loaded'}
        
        model = self.models['position_sizer']
        
        import pandas as pd
        X = pd.DataFrame([features], columns=self.feature_names)
        
        size = float(model.predict(X)[0])
        
        # Clamp to [0, 1]
        size = max(0.0, min(1.0, size))
        
        return {
            'position_size': size,
            'size_pct': size * 100,
            'recommendation': 'full' if size > 0.8 else 'reduced' if size > 0.3 else 'minimal'
        }
    
    def predict_volatility(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict forward volatility."""
        if 'volatility_model' not in self.models:
            return {'error': 'Model not loaded'}
        
        model = self.models['volatility_model']
        
        import pandas as pd
        X = pd.DataFrame([features], columns=self.feature_names)
        
        vol = float(model.predict(X)[0])
        
        return {
            'volatility': vol,
            'vol_pct': vol * 100,
            'regime': 'high' if vol > 0.03 else 'medium' if vol > 0.015 else 'low'
        }
    
    def predict_trend_strength(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict trend strength."""
        if 'trend_strength' not in self.models:
            return {'error': 'Model not loaded'}
        
        model = self.models['trend_strength']
        
        import pandas as pd
        X = pd.DataFrame([features], columns=self.feature_names)
        
        strength = float(model.predict(X)[0])
        
        # Normalize to [0, 1]
        strength = max(0.0, min(1.0, strength))
        
        return {
            'trend_strength': strength,
            'strength_pct': strength * 100,
            'direction': 'strong_up' if strength > 0.7 else 'up' if strength > 0.5 else 'neutral' if strength > 0.3 else 'down' if strength > 0.1 else 'strong_down'
        }
    
    def get_trading_advisory(self, features: np.ndarray) -> Dict[str, Any]:
        """Get comprehensive trading advisory from all models."""
        advisory = {
            'regime': self.predict_regime(features),
            'signal': self.predict_signal_strength(features),
            'position_size': self.predict_position_size(features),
            'volatility': self.predict_volatility(features),
            'trend_strength': self.predict_trend_strength(features),
        }
        
        # Calculate composite score
        signal_strength = advisory['signal'].get('strength', 0)
        position_size = advisory['position_size'].get('position_size', 0.5)
        trend_strength = advisory['trend_strength'].get('trend_strength', 0.5)
        volatility = advisory['volatility'].get('volatility', 0.02)
        
        # Composite score: higher = more confident to trade
        composite = (signal_strength + position_size + trend_strength) / 3
        composite *= (1 - min(volatility * 10, 0.5))  # Penalize high volatility
        
        advisory['composite'] = {
            'score': max(0.0, min(1.0, composite)),
            'action': 'trade' if composite > 0.6 else 'wait' if composite > 0.4 else 'avoid',
            'confidence': composite
        }
        
        return advisory


def main():
    """Test the ML model integration."""
    logger.info("="*60)
    logger.info("ML MODEL INTEGRATION TEST")
    logger.info("="*60)
    
    integrator = MLModelIntegrator()
    models = integrator.load_production_models()
    
    logger.info(f"\nLoaded {len(models)} models")
    
    # Test with random features (14 features - model input size)
    test_features = np.random.randn(14)
    
    logger.info("\n" + "="*60)
    logger.info("TEST PREDICTIONS (random features)")
    logger.info("="*60)
    
    advisory = integrator.get_trading_advisory(test_features)
    
    logger.info(f"\nRegime: {advisory['regime']['regime']} (confidence: {advisory['regime']['confidence']:.2%})")
    logger.info(f"Signal: {advisory['signal']['signal']} (strength: {advisory['signal']['strength']:.2f})")
    logger.info(f"Position Size: {advisory['position_size']['position_size']:.2%}")
    logger.info(f"Volatility: {advisory['volatility']['vol_pct']:.2f}%")
    logger.info(f"Trend Strength: {advisory['trend_strength']['strength_pct']:.1f}%")
    logger.info(f"\nComposite Score: {advisory['composite']['score']:.2f}")
    logger.info(f"Action: {advisory['composite']['action'].upper()}")
    
    logger.info("\n" + "="*60)
    logger.info("INTEGRATION COMPLETE")
    logger.info("="*60)
    
    return integrator


if __name__ == "__main__":
    main()
