"""
ML Trading Integration for Argus

Integrates all ML improvements:
- Advanced feature engineering (100+ features)
- Online learning (real-time adaptation)
- Advanced risk management
- Ensemble voting (multiple models)
- Deep learning models (LSTM, Bayesian NN, Autoencoder, RL)

This is the main entry point for ML-powered trading decisions.
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.winrate_enhancement import (
    CalibratedConfidenceAdapter,
    ConflictAbstentionGate,
    DynamicModelWeightManager,
    ModelVote,
    compact_deep_features,
    compact_gb_features,
)

logger = logging.getLogger(__name__)


class DeepLearningModels:
    """Wrapper for deep learning models."""
    
    def __init__(self, models_dir: str = "data/models_deep"):
        import torch
        import torch.nn as nn
        
        # Define model classes inline (matching training scripts)
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size=64):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden_size, 3)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])
        
        class BayesianNN(nn.Module):
            def __init__(self, input_size, hidden_size=64):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size, 32),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(32, 3)
                )
            def forward(self, x):
                return self.net(x)
        
        class AutoencoderModel(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_size, 32), nn.ReLU(),
                    nn.Linear(32, 16), nn.ReLU(),
                    nn.Linear(16, 8)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(8, 16), nn.ReLU(),
                    nn.Linear(16, 32), nn.ReLU(),
                    nn.Linear(32, input_size)
                )
            def forward(self, x):
                return self.decoder(self.encoder(x))
        
        class DQNAgent(nn.Module):
            def __init__(self, input_size, n_actions=3):
                super().__init__()
                self.q = nn.Sequential(
                    nn.Linear(input_size, 64), nn.ReLU(),
                    nn.Linear(64, 32), nn.ReLU(),
                    nn.Linear(32, n_actions)
                )
            def forward(self, x):
                return self.q(x)
        
        self.LSTMModel = LSTMModel
        self.BayesianNN = BayesianNN
        self.AutoencoderModel = AutoencoderModel
        self.DQNAgent = DQNAgent
        self.torch = torch
        
        self.models_dir = Path(models_dir)
        self.lstm = None
        self.bayesian_nn = None
        self.autoencoder = None
        self.rl_agent = None
        self.scaler = None
        self.ae_threshold = 0.5
        self.is_loaded = False
        
    def load(self):
        """Load all deep learning models."""
        torch = self.torch
        
        try:
            # Load scaler
            scaler_path = self.models_dir / 'scaler.pkl'
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Loaded deep learning scaler")
            
            # Load LSTM
            lstm_path = self.models_dir / 'lstm_predictor.pth'
            if lstm_path.exists():
                self.lstm = self.LSTMModel(input_size=7, hidden_size=64)
                self.lstm.load_state_dict(torch.load(lstm_path, map_location='cpu'))
                self.lstm.eval()
                logger.info("Loaded LSTM model")
            
            # Load Bayesian NN
            bn_path = self.models_dir / 'bayesian_nn.pth'
            if bn_path.exists():
                self.bayesian_nn = self.BayesianNN(input_size=7, hidden_size=64)
                self.bayesian_nn.load_state_dict(torch.load(bn_path, map_location='cpu'))
                self.bayesian_nn.eval()
                logger.info("Loaded Bayesian NN model")
            
            # Load Autoencoder
            ae_path = self.models_dir / 'autoencoder.pth'
            if ae_path.exists():
                self.autoencoder = self.AutoencoderModel(input_size=7)
                self.autoencoder.load_state_dict(torch.load(ae_path, map_location='cpu'))
                self.autoencoder.eval()
                
                # Load threshold
                threshold_path = self.models_dir / 'autoencoder_threshold.pkl'
                if threshold_path.exists():
                    with open(threshold_path, 'rb') as f:
                        data = pickle.load(f)
                        self.ae_threshold = data.get('threshold', 0.5)
                logger.info("Loaded Autoencoder model")
            
            # Load RL Agent
            rl_path = self.models_dir / 'rl_dqn.pth'
            if rl_path.exists():
                self.rl_agent = self.DQNAgent(input_size=7)
                self.rl_agent.load_state_dict(torch.load(rl_path, map_location='cpu'))
                self.rl_agent.eval()
                logger.info("Loaded RL DQN agent")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load deep learning models: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> Dict:
        """Get predictions from all deep learning models."""
        import torch
        
        if not self.is_loaded:
            return {}
        
        results = {}
        raw_features = np.asarray(features, dtype=np.float32)
        if raw_features.ndim == 1:
            raw_features = raw_features.reshape(1, -1)
        if self.scaler is not None:
            raw_features = self.scaler.transform(raw_features).astype(np.float32)

        dense_tensor = torch.FloatTensor(raw_features[-1:])
        sequence_tensor = torch.FloatTensor(raw_features.reshape(1, raw_features.shape[0], raw_features.shape[1]))
        
        # LSTM prediction
        if self.lstm is not None:
            try:
                with torch.no_grad():
                    lstm_out = self.lstm(sequence_tensor)
                    lstm_probs = torch.softmax(lstm_out, dim=-1).numpy()[0]
                    results['lstm'] = {
                        'signal': int(np.argmax(lstm_probs)),
                        'confidence': float(np.max(lstm_probs)),
                        'probs': lstm_probs.tolist()
                    }
            except Exception as e:
                logger.debug(f"LSTM prediction failed: {e}")
        
        # Bayesian NN prediction
        if self.bayesian_nn is not None:
            try:
                with torch.no_grad():
                    bn_out = self.bayesian_nn(dense_tensor)
                    bn_probs = torch.softmax(bn_out, dim=-1).numpy()[0]
                    results['bayesian'] = {
                        'signal': int(np.argmax(bn_probs)),
                        'confidence': float(np.max(bn_probs)),
                        'probs': bn_probs.tolist()
                    }
            except Exception as e:
                logger.debug(f"Bayesian NN prediction failed: {e}")
        
        # Autoencoder anomaly detection
        if self.autoencoder is not None:
            try:
                with torch.no_grad():
                    recon = self.autoencoder(dense_tensor)
                    error = ((recon - dense_tensor) ** 2).mean().item()
                    results['autoencoder'] = {
                        'anomaly_score': error,
                        'is_anomaly': error > self.ae_threshold,
                        'threshold': self.ae_threshold
                    }
            except Exception as e:
                logger.debug(f"Autoencoder prediction failed: {e}")
        
        # RL Agent prediction
        if self.rl_agent is not None:
            try:
                with torch.no_grad():
                    rl_q = self.rl_agent(dense_tensor)
                    rl_action = int(torch.argmax(rl_q, dim=-1).item())
                    results['rl'] = {
                        'signal': rl_action,  # 0=sell, 1=hold, 2=buy
                        'q_values': rl_q.numpy()[0].tolist()
                    }
            except Exception as e:
                logger.debug(f"RL agent prediction failed: {e}")
        
        return results


class MLTradingEngine:
    """
    Main ML trading engine that integrates all components.
    
    Components:
    1. FeatureEngineer: Generates 100+ features from market data
    2. EnsembleVoter: Combines predictions from multiple models
    3. OnlineLearner: Adapts models in real-time
    4. RiskManager: Manages position sizing and risk
    """
    
    def __init__(self, models_dir: str = "data/models_mtf"):
        self.models_dir = Path(models_dir)
        self.ensemble_db_path = "data/ensemble_voter.db"
        
        # Components (lazy loaded)
        self.feature_engineer = None
        self.ensemble_voter = None
        self.online_learner = None
        self.risk_manager = None
        self.deep_models = None  # Deep learning models
        self.gb_models = {}
        self.gb_scaler = None
        self.gb_feature_names = []
        self.calibrator = CalibratedConfidenceAdapter()
        self.weight_manager = DynamicModelWeightManager()
        self.abstention_gate = ConflictAbstentionGate()
        
        # State
        self.is_initialized = False
        self.prediction_count = 0
        self.last_prediction_time = None
        
    def initialize(self):
        """Initialize all components."""
        logger.info("="*70)
        logger.info("INITIALIZING ML TRADING ENGINE")
        logger.info("="*70)
        
        # Import components
        from ml.advanced_features import AdvancedFeatureEngineer
        from ml.ensemble_voter import EnsembleVoter
        from ml.online_learner import OnlineLearner
        from risk.advanced_risk_manager import RiskManager
        
        # Ensure data directory exists
        Path("data").mkdir(exist_ok=True)
        
        # Initialize components
        self.feature_engineer = AdvancedFeatureEngineer()
        
        self.ensemble_voter = EnsembleVoter(self.ensemble_db_path)
        # Note: EnsembleVoter doesn't have load_models - it accumulates votes
        
        self.online_learner = OnlineLearner(str(self.models_dir))
        
        self.risk_manager = RiskManager()
        
        # Load deep learning models
        self.deep_models = DeepLearningModels("data/models_deep")
        self.deep_models.load()
        
        # Load gradient boosting models
        self._load_gb_models()
        
        self.is_initialized = True
        
        logger.info("ML Trading Engine initialized successfully")
        logger.info(f"Gradient boosting models: {list(self.gb_models.keys())}")
        logger.info(f"Deep learning models: {'loaded' if self.deep_models.is_loaded else 'not loaded'}")
    
    def _load_gb_models(self):
        """Load gradient boosting models."""
        import joblib
        
        models_dir = self.models_dir
        try:
            # Load scaler
            scaler_path = models_dir / 'scaler.pkl'
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.gb_scaler = pickle.load(f)
                logger.info("Loaded GB scaler")

            feature_names_path = models_dir / 'feature_names.pkl'
            if feature_names_path.exists():
                with open(feature_names_path, 'rb') as f:
                    self.gb_feature_names = list(pickle.load(f))
            
            # Load models
            model_names = ['signal_classifier', 'regime_classifier', 'position_sizer', 
                          'volatility_model', 'trend_strength']
            for name in model_names:
                model_path = models_dir / f'{name}.pkl'
                if model_path.exists():
                    self.gb_models[name] = joblib.load(model_path)
                    logger.info(f"Loaded GB model: {name}")
        except Exception as e:
            logger.error(f"Failed to load GB models: {e}")
    
    def generate_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate the compact 9-feature vector expected by GB artifacts."""
        try:
            return compact_gb_features(df)
        except Exception as e:
            logger.error(f"Feature generation failed: {e}")
            return None

    def generate_deep_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate the compact 7-feature vector expected by deep artifacts."""
        try:
            return compact_deep_features(df)
        except Exception as e:
            logger.error(f"Deep feature generation failed: {e}")
            return None
    
    def predict(
        self,
        df: pd.DataFrame,
        current_price: float,
        symbol: str = "BTC/USD"
    ) -> Dict:
        """
        Make a trading prediction using gradient boosting + deep learning ensemble.
        """
        if not self.is_initialized:
            self.initialize()
        
        self.prediction_count += 1
        self.last_prediction_time = datetime.now()
        
        # Generate features
        features = self.generate_features(df)
        if features is None:
            return self._default_prediction("feature_generation_failed")
        
        # Get gradient boosting predictions
        gb_prediction = self._predict_gb(features)
        
        # Get deep learning predictions
        dl_predictions = {}
        if self.deep_models and self.deep_models.is_loaded:
            deep_features = self.generate_deep_features(df)
            if deep_features is not None:
                dl_predictions = self.deep_models.predict(deep_features)
        
        # Combine predictions
        final_prediction = self._combine_predictions(gb_prediction, dl_predictions)
        
        # Map signal to action
        signal_map = {0: 'sell', 1: 'hold', 2: 'buy'}
        action = signal_map.get(final_prediction['signal'], 'wait')
        
        anomaly_detected = bool(dl_predictions.get('autoencoder', {}).get('is_anomaly', False))
        should_abstain, abstain_reason, adjusted_confidence = self.abstention_gate.evaluate(
            final_prediction['combined_vote'],
            anomaly_detected=anomaly_detected,
            model_agreement=final_prediction.get('model_agreement', True),
        )
        final_prediction['confidence'] = adjusted_confidence

        if should_abstain:
            action = 'wait'
        
        # Calculate stops
        atr = self._calculate_atr(df)
        stop_loss, take_profit = self._calculate_stops(
            current_price, action, atr, final_prediction['regime']
        )
        
        # Position size
        position_size = 0.0
        if action in ['buy', 'sell'] and self.risk_manager:
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                entry_price=current_price,
                stop_loss=stop_loss,
                confidence=final_prediction['confidence'],
                win_rate=0.65,
                avg_win=0.02,
                avg_loss=0.01
            )
            position_size *= final_prediction.get('position_size_multiplier', 1.0)
        
        result = {
            'action': action,
            'signal': final_prediction['signal'],
            'confidence': final_prediction['confidence'],
            'regime': ['bear', 'sideways', 'bull'][final_prediction['regime']],
            'regime_confidence': final_prediction['regime_confidence'],
            'position_size_multiplier': final_prediction.get('position_size_multiplier', 1.0),
            'suggested_position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'disagreement': final_prediction.get('disagreement', 0.0),
            'prediction_count': self.prediction_count,
            'gradient_boosting': gb_prediction,
            'deep_learning': dl_predictions,
            'model_agreement': final_prediction.get('model_agreement', True),
            'abstention_reason': abstain_reason,
            'dynamic_weights': final_prediction.get('weights_used', {}),
        }
        
        logger.info(f"Prediction: {action} {symbol} @ {current_price:.2f} "
                   f"(confidence={final_prediction['confidence']:.2%}, "
                   f"regime={result['regime']})")
        
        return result
    
    def _predict_gb(self, features: np.ndarray) -> Dict:
        """Get gradient boosting predictions."""
        if not self.gb_models or self.gb_scaler is None:
            return {'signal': 1, 'confidence': 0.5, 'regime': 1, 'regime_confidence': 0.5}
        
        # Scale features
        features_for_model = features
        if self.gb_feature_names:
            features_for_model = pd.DataFrame(features, columns=self.gb_feature_names)
        features_scaled = self.gb_scaler.transform(features_for_model)
        if self.gb_feature_names:
            features_scaled = pd.DataFrame(features_scaled, columns=self.gb_feature_names)
        
        # Get signal prediction
        signal_model = self.gb_models.get('signal_classifier')
        if signal_model:
            signal_pred = signal_model.predict(features_scaled)[0]
            signal_proba = signal_model.predict_proba(features_scaled)[0]
            confidence = self.calibrator.calibrate('signal_classifier', float(np.max(signal_proba)))
        else:
            signal_pred = 1
            confidence = 0.5
        
        # Get regime prediction
        regime_model = self.gb_models.get('regime_classifier')
        if regime_model:
            regime_pred = regime_model.predict(features_scaled)[0]
            regime_proba = regime_model.predict_proba(features_scaled)[0]
            regime_confidence = self.calibrator.calibrate('regime_classifier', float(np.max(regime_proba)))
        else:
            regime_pred = 1
            regime_confidence = 0.5
        
        return {
            'signal': int(signal_pred),
            'confidence': confidence,
            'regime': int(regime_pred),
            'regime_confidence': regime_confidence,
        }
    
    def _combine_predictions(
        self,
        gb_prediction: Dict,
        dl_predictions: Dict
    ) -> Dict:
        """Combine gradient boosting and deep learning predictions."""
        votes = [ModelVote('gradient_boosting', gb_prediction['signal'], gb_prediction['confidence'])]

        for model_name in ('lstm', 'bayesian'):
            pred = dl_predictions.get(model_name)
            if pred:
                confidence = self.calibrator.calibrate(model_name, pred['confidence'])
                votes.append(ModelVote(model_name, pred['signal'], confidence, pred.get('probs')))

        rl_pred = dl_predictions.get('rl')
        if rl_pred:
            q_values = np.asarray(rl_pred.get('q_values', [0.0, 0.0, 0.0]), dtype=float)
            q_conf = float(np.max(q_values) - np.median(q_values)) if q_values.size else 0.2
            votes.append(ModelVote('rl', rl_pred['signal'], min(0.8, max(0.2, q_conf))))

        combined = self.weight_manager.combine(votes, regime=gb_prediction.get('regime'))
        result = dict(gb_prediction)
        result['signal'] = combined.signal
        result['confidence'] = combined.confidence
        result['disagreement'] = combined.disagreement
        result['model_agreement'] = combined.model_agreement
        result['weights_used'] = combined.weights_used
        result['combined_vote'] = combined
        return result
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR from DataFrame."""
        if len(df) < period + 1:
            return df['close'].iloc[-1] * 0.02  # Default 2% if not enough data
        
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return float(atr)
    
    def _calculate_stops(
        self,
        entry_price: float,
        action: str,
        atr: float,
        regime: int
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit."""
        if action == 'wait' or action == 'hold':
            return 0.0, 0.0
        
        # Adjust based on regime
        regime_multiplier = {0: 1.5, 1: 1.0, 2: 0.8}  # bear, sideways, bull
        multiplier = regime_multiplier.get(regime, 1.0)
        
        stop_distance = atr * 2.0 * multiplier
        tp_distance = atr * 3.0 * multiplier
        
        if action == 'buy':
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + tp_distance
        else:  # sell
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - tp_distance
        
        return stop_loss, take_profit
    
    def _default_prediction(self, reason: str) -> Dict:
        """Return default prediction on error."""
        return {
            'action': 'wait',
            'confidence': 0.0,
            'regime': 'unknown',
            'regime_confidence': 0.0,
            'position_size_multiplier': 0.0,
            'suggested_position_size': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'disagreement': 1.0,
            'prediction_count': self.prediction_count,
            'error': reason,
        }
    
    def record_outcome(self, symbol: str, prediction: Dict, actual_result: float):
        """
        Record actual outcome for online learning.
        
        Args:
            symbol: Trading pair
            prediction: The prediction dict from predict()
            actual_result: Actual return (>0 for profit, <0 for loss)
        """
        if self.online_learner is None:
            return
        
        # Determine actual signal
        if actual_result > 0.01:
            actual_signal = 2  # buy was correct
        elif actual_result < -0.01:
            actual_signal = 0  # sell was correct
        else:
            actual_signal = 1  # hold was correct
        
        # Record for each model
        for model_name in ['signal_classifier', 'signal_bear', 'signal_sideways', 'signal_bull']:
            self.online_learner.record_prediction(model_name, prediction.get('signal', 1), actual_signal)

        predicted_signal = prediction.get('signal')
        was_correct = predicted_signal == actual_signal if predicted_signal is not None else actual_result > 0
        self.calibrator.record('signal_classifier', prediction.get('confidence', 0.0), bool(was_correct))
        for model_name in prediction.get('dynamic_weights', {}).keys():
            self.weight_manager.update_outcome(model_name, bool(was_correct))
        
        logger.debug(f"Recorded outcome: {symbol} actual={actual_signal}, predicted={prediction.get('signal')}")
    
    def get_status(self) -> Dict:
        """Get engine status."""
        dl_models = []
        if self.deep_models and self.deep_models.is_loaded:
            if self.deep_models.lstm is not None:
                dl_models.append('lstm')
            if self.deep_models.bayesian_nn is not None:
                dl_models.append('bayesian_nn')
            if self.deep_models.autoencoder is not None:
                dl_models.append('autoencoder')
            if self.deep_models.rl_agent is not None:
                dl_models.append('rl_dqn')
        
        return {
            'initialized': self.is_initialized,
            'prediction_count': self.prediction_count,
            'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            'gradient_boosting_models': list(self.gb_models.keys()) if self.gb_models else [],
            'deep_learning_models': dl_models,
            'total_models': len(self.gb_models) + len(dl_models),
        }


# Singleton instance
_engine: Optional[MLTradingEngine] = None


def get_ml_engine() -> MLTradingEngine:
    """Get or create singleton ML trading engine."""
    global _engine
    
    if _engine is None:
        _engine = MLTradingEngine()
        _engine.initialize()
    
    return _engine


def predict_trading_signal(
    df: pd.DataFrame,
    current_price: float,
    symbol: str = "BTC/USD"
) -> Dict:
    """
    Convenience function to get trading prediction.
    
    Usage:
        from ml.ml_trading_integration import predict_trading_signal
        
        prediction = predict_trading_signal(ohlcv_df, 50000.0, "BTC/USD")
        print(prediction['action'])  # 'buy', 'sell', 'hold', 'wait'
        print(prediction['confidence'])  # 0.0 - 1.0
    """
    engine = get_ml_engine()
    return engine.predict(df, current_price, symbol)
