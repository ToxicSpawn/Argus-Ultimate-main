"""
Quantum-Enhanced Regime Detection

This module implements quantum-enhanced market regime detection using:
- Quantum clustering for regime identification
- Quantum feature extraction from market data
- Hybrid quantum-classical classification
- Real-time regime adaptation
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime types"""
    STABLE = auto()
    TRENDING = auto()
    VOLATILE = auto()
    RANGE = auto()
    UNCERTAIN = auto()

@dataclass
class MarketDataFeatures:
    """Market data features for regime detection"""
    returns: np.ndarray  # Price returns
    volatility: np.ndarray  # Volatility measures
    volume: np.ndarray     # Trading volume
    momentum: np.ndarray   # Momentum indicators
    correlation: np.ndarray  # Asset correlations
    sentiment: np.ndarray   # Market sentiment
    
    def to_array(self) -> np.ndarray:
        """Convert to feature array"""
        # Stack all features horizontally
        features = []
        if hasattr(self, 'returns') and self.returns is not None:
            features.append(self.returns)
        if hasattr(self, 'volatility') and self.volatility is not None:
            features.append(self.volatility)
        if hasattr(self, 'volume') and self.volume is not None:
            features.append(self.volume)
        if hasattr(self, 'momentum') and self.momentum is not None:
            features.append(self.momentum)
        if hasattr(self, 'correlation') and self.correlation is not None:
            features.append(self.correlation)
        if hasattr(self, 'sentiment') and self.sentiment is not None:
            features.append(self.sentiment)
            
        return np.column_stack(features) if features else np.array([])

@dataclass
class RegimeDetectionResult:
    """Result of regime detection"""
    regime: MarketRegime
    confidence: float
    features: MarketDataFeatures
    quantum_contribution: float
    detection_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'regime': self.regime.name,
            'confidence': self.confidence,
            'quantum_contribution': self.quantum_contribution,
            'detection_time': self.detection_time.isoformat(),
            'feature_dimensions': {
                'returns': self.features.returns.shape[0] if hasattr(self.features, 'returns') else 0,
                'volatility': self.features.volatility.shape[0] if hasattr(self.features, 'volatility') else 0,
                'volume': self.features.volume.shape[0] if hasattr(self.features, 'volume') else 0,
                'momentum': self.features.momentum.shape[0] if hasattr(self.features, 'momentum') else 0,
                'correlation': self.features.correlation.shape[0] if hasattr(self.features, 'correlation') else 0,
                'sentiment': self.features.sentiment.shape[0] if hasattr(self.features, 'sentiment') else 0
            }
        }

class QuantumFeatureExtractor:
    """Quantum feature extraction for regime detection"""
    
    def __init__(self, num_qubits: int = 4, hardware_backend: str = "simulator"):
        self.num_qubits = num_qubits
        self.hardware_backend = hardware_backend
        self.quantum_circuit_cache = {}
        
    def extract_quantum_features(self, features: np.ndarray) -> np.ndarray:
        """
        Extract quantum features from classical market data
        
        Args:
            features: Classical market features (n_samples, n_features)
            
        Returns:
            Quantum-enhanced features (n_samples, num_qubits)
        """
        if features.ndim != 2:
            raise ValueError("Features must be 2D array (n_samples, n_features)")
        
        # Create circuit hash for caching
        circuit_hash = self._get_circuit_hash(features.shape[1])
        
        # Check cache
        if circuit_hash in self.quantum_circuit_cache:
            return self.quantum_circuit_cache[circuit_hash](features)
        
        # Simulate quantum feature extraction
        def quantum_feature_function(X: np.ndarray) -> np.ndarray:
            """Simulated quantum feature extraction"""
            n_samples = X.shape[0]
            quantum_features = np.zeros((n_samples, self.num_qubits))
            
            # For each sample, create quantum-enhanced features
            for i in range(n_samples):
                # Simple simulation: use trigonometric functions with different frequencies
                for q in range(self.num_qubits):
                    # Combine multiple features with different weights
                    feature_combination = 0
                    for f in range(X.shape[1]):
                        # Use qubit index as frequency multiplier
                        feature_combination += X[i, f] * (q + 1)
                    
                    # Apply quantum-inspired transformation
                    quantum_features[i, q] = np.sin(feature_combination) * 0.5 + np.cos(feature_combination * 0.5)
            
            return quantum_features
        
        # Cache the function
        self.quantum_circuit_cache[circuit_hash] = quantum_feature_function
        return quantum_feature_function(features)
    
    def _get_circuit_hash(self, input_features: int) -> str:
        """Get a hash for the quantum circuit configuration"""
        return f"qfe_{self.num_qubits}_{input_features}_{self.hardware_backend}"

class QuantumRegimeClassifier:
    """Quantum-enhanced regime classifier"""
    
    def __init__(self, feature_extractor: QuantumFeatureExtractor):
        self.feature_extractor = feature_extractor
        self.classical_weights = {}
        self.quantum_weights = {}
        self.training_history = []
        
    def _initialize_weights(self, num_features: int):
        """Initialize classifier weights"""
        num_regimes = len(MarketRegime)
        
        # Classical weights (feature space -> regimes)
        self.classical_weights = {
            'weights': np.random.normal(0, 0.01, (num_features, num_regimes)),
            'bias': np.zeros(num_regimes)
        }
        
        # Quantum weights (quantum features -> regimes)
        self.quantum_weights = {
            'weights': np.random.normal(0, 0.01, (self.feature_extractor.num_qubits, num_regimes)),
            'bias': np.zeros(num_regimes)
        }
    
    def train(self, features_list: List[MarketDataFeatures], regimes: List[MarketRegime],
              epochs: int = 100, learning_rate: float = 0.01) -> List[Dict[str, Any]]:
        """
        Train the quantum-enhanced regime classifier
        
        Args:
            features_list: List of market data features
            regimes: Corresponding market regimes
            epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        if len(features_list) != len(regimes):
            raise ValueError("Features and regimes lists must have the same length")
        
        if len(features_list) == 0:
            return []
        
        # Convert features to arrays
        classical_features = np.array([f.to_array() for f in features_list])
        quantum_features = self.feature_extractor.extract_quantum_features(classical_features)
        
        # One-hot encode regimes
        regime_targets = np.zeros((len(regimes), len(MarketRegime)))
        for i, regime in enumerate(regimes):
            regime_targets[i, regime.value - 1] = 1  # MarketRegime starts at 1
        
        # Initialize weights if not already done
        if not self.classical_weights:
            num_features = classical_features.shape[1]
            self._initialize_weights(num_features)
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            classical_logits = self._classical_forward(classical_features)
            quantum_logits = self._quantum_forward(quantum_features)
            combined_logits = classical_logits + quantum_logits
            
            # Softmax for probabilities
            probs = self._softmax(combined_logits)
            
            # Compute loss
            loss = self._cross_entropy_loss(probs, regime_targets)
            
            # Compute accuracy
            accuracy = self._compute_accuracy(probs, regime_targets)
            
            # Quantum contribution metric
            quantum_contribution = self._compute_quantum_contribution(
                classical_logits, quantum_logits, regime_targets
            )
            
            # Backward pass (simplified)
            self._backward_pass(classical_features, quantum_features, regime_targets, probs, learning_rate)
            
            # Record training metrics
            self.training_history.append({
                'epoch': epoch + 1,
                'loss': loss,
                'accuracy': accuracy,
                'quantum_contribution': quantum_contribution,
                'timestamp': datetime.now().isoformat()
            })
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch + 1}/{epochs}: loss={loss:.4f}, "
                           f"accuracy={accuracy:.4f}, quantum_contribution={quantum_contribution:.4f}")
        
        return self.training_history
    
    def _classical_forward(self, features: np.ndarray) -> np.ndarray:
        """Classical forward pass"""
        return np.dot(features, self.classical_weights['weights']) + self.classical_weights['bias']
    
    def _quantum_forward(self, features: np.ndarray) -> np.ndarray:
        """Quantum forward pass"""
        return np.dot(features, self.quantum_weights['weights']) + self.quantum_weights['bias']
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Softmax function"""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def _cross_entropy_loss(self, probs: np.ndarray, targets: np.ndarray) -> float:
        """Cross entropy loss"""
        return -np.mean(np.sum(targets * np.log(probs + 1e-10), axis=1))
    
    def _compute_accuracy(self, probs: np.ndarray, targets: np.ndarray) -> float:
        """Compute classification accuracy"""
        predictions = np.argmax(probs, axis=1)
        true_labels = np.argmax(targets, axis=1)
        return np.mean(predictions == true_labels)
    
    def _compute_quantum_contribution(self, classical_logits: np.ndarray,
                                    quantum_logits: np.ndarray,
                                    targets: np.ndarray) -> float:
        """Compute quantum contribution to classification"""
        # Classical-only probabilities
        classical_probs = self._softmax(classical_logits)
        classical_acc = self._compute_accuracy(classical_probs, targets)
        
        # Combined probabilities
        combined_probs = self._softmax(classical_logits + quantum_logits)
        combined_acc = self._compute_accuracy(combined_probs, targets)
        
        # Quantum contribution is the improvement from adding quantum features
        return max(0, combined_acc - classical_acc)
    
    def _backward_pass(self, classical_features: np.ndarray, quantum_features: np.ndarray,
                       targets: np.ndarray, probs: np.ndarray, learning_rate: float):
        """Backward pass for training"""
        # Compute gradient (simplified)
        error = probs - targets
        
        # Update classical weights
        classical_grad = np.dot(classical_features.T, error) / len(classical_features)
        self.classical_weights['weights'] -= learning_rate * classical_grad
        self.classical_weights['bias'] -= learning_rate * np.mean(error, axis=0)
        
        # Update quantum weights
        quantum_grad = np.dot(quantum_features.T, error) / len(quantum_features)
        self.quantum_weights['weights'] -= learning_rate * quantum_grad
        self.quantum_weights['bias'] -= learning_rate * np.mean(error, axis=0)
    
    def predict(self, features: MarketDataFeatures) -> RegimeDetectionResult:
        """
        Predict market regime from features
        
        Args:
            features: Market data features
            
        Returns:
            Regime detection result
        """
        # Convert to arrays
        classical_features = features.to_array()
        if classical_features.ndim == 1:
            classical_features = classical_features[np.newaxis, :]
        
        quantum_features = self.feature_extractor.extract_quantum_features(classical_features)
        
        # Forward pass
        classical_logits = self._classical_forward(classical_features)
        quantum_logits = self._quantum_forward(quantum_features)
        combined_logits = classical_logits + quantum_logits
        
        # Get probabilities and predicted regime
        probs = self._softmax(combined_logits)
        predicted_regime = MarketRegime(np.argmax(probs) + 1)  # +1 because MarketRegime starts at 1
        confidence = np.max(probs)
        
        # Compute quantum contribution
        classical_probs = self._softmax(classical_logits)
        classical_acc = self._compute_accuracy(classical_probs, np.eye(len(MarketRegime))[np.argmax(probs)])
        combined_acc = self._compute_accuracy(probs, np.eye(len(MarketRegime))[np.argmax(probs)])
        quantum_contribution = max(0, combined_acc - classical_acc)
        
        return RegimeDetectionResult(
            regime=predicted_regime,
            confidence=confidence,
            features=features,
            quantum_contribution=quantum_contribution
        )
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        if not self.training_history:
            return {"status": "not_trained"}
            
        best_epoch = max(self.training_history, key=lambda x: x['accuracy'])
        final_epoch = self.training_history[-1]
        
        return {
            "status": "trained",
            "epochs": len(self.training_history),
            "best_accuracy": best_epoch['accuracy'],
            "final_accuracy": final_epoch['accuracy'],
            "avg_quantum_contribution": np.mean([e['quantum_contribution'] for e in self.training_history]),
            "training_history": self.training_history
        }

class QuantumRegimeDetector:
    """Complete quantum-enhanced regime detection system"""
    
    def __init__(self, num_qubits: int = 4, hardware_backend: str = "simulator"):
        self.feature_extractor = QuantumFeatureExtractor(num_qubits, hardware_backend)
        self.classifier = QuantumRegimeClassifier(self.feature_extractor)
        self.detection_history = []
        
    def train_detector(self, training_data: List[Tuple[MarketDataFeatures, MarketRegime]],
                       epochs: int = 100, learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        Train the quantum regime detector
        
        Args:
            training_data: List of (features, regime) tuples
            epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Training summary
        """
        features = [data[0] for data in training_data]
        regimes = [data[1] for data in training_data]
        
        training_history = self.classifier.train(features, regimes, epochs, learning_rate)
        
        return {
            "status": "training_completed",
            "epochs": epochs,
            "final_accuracy": training_history[-1]['accuracy'] if training_history else 0,
            "avg_quantum_contribution": np.mean([e['quantum_contribution'] for e in training_history]) if training_history else 0
        }
    
    def detect_regime(self, features: MarketDataFeatures) -> RegimeDetectionResult:
        """
        Detect market regime from features
        
        Args:
            features: Market data features
            
        Returns:
            Regime detection result
        """
        result = self.classifier.predict(features)
        self.detection_history.append(result)
        return result
    
    def get_detection_history(self, limit: int = 100) -> List[RegimeDetectionResult]:
        """Get recent detection history"""
        return self.detection_history[-limit:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.detection_history:
            return {"status": "no_detections"}
            
        # Calculate regime distribution
        regime_counts = {regime: 0 for regime in MarketRegime}
        for detection in self.detection_history:
            regime_counts[detection.regime] += 1
        
        # Calculate quantum contribution stats
        quantum_contributions = [d.quantum_contribution for d in self.detection_history]
        avg_quantum_contribution = np.mean(quantum_contributions)
        
        return {
            "status": "active",
            "total_detections": len(self.detection_history),
            "regime_distribution": {r.name: count for r, count in regime_counts.items()},
            "avg_quantum_contribution": avg_quantum_contribution,
            "recent_regime": self.detection_history[-1].regime.name if self.detection_history else None,
            "recent_confidence": self.detection_history[-1].confidence if self.detection_history else 0
        }
    
    def adapt_to_new_data(self, new_data: List[Tuple[MarketDataFeatures, MarketRegime]],
                         epochs: int = 50, learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Adapt the detector to new market data
        
        Args:
            new_data: New training data
            epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Training summary
        """
        features = [data[0] for data in new_data]
        regimes = [data[1] for data in new_data]
        
        training_history = self.classifier.train(features, regimes, epochs, learning_rate)
        
        return {
            "status": "adaptation_completed",
            "epochs": epochs,
            "final_accuracy": training_history[-1]['accuracy'] if training_history else 0,
            "avg_quantum_contribution": np.mean([e['quantum_contribution'] for e in training_history]) if training_history else 0
        }

class QuantumRegimeAdaptationSystem:
    """Complete quantum-enhanced regime adaptation system"""
    
    def __init__(self, num_qubits: int = 4, hardware_backend: str = "simulator"):
        self.detector = QuantumRegimeDetector(num_qubits, hardware_backend)
        self.adaptation_strategies = {
            MarketRegime.STABLE: self._adapt_to_stable,
            MarketRegime.TRENDING: self._adapt_to_trending,
            MarketRegime.VOLATILE: self._adapt_to_volatile,
            MarketRegime.RANGE: self._adapt_to_range,
            MarketRegime.UNCERTAIN: self._adapt_to_uncertain
        }
        self.current_regime = None
        self.adaptation_history = []
    
    def train_system(self, training_data: List[Tuple[MarketDataFeatures, MarketRegime]],
                     epochs: int = 100, learning_rate: float = 0.01) -> Dict[str, Any]:
        """Train the complete regime adaptation system"""
        return self.detector.train_detector(training_data, epochs, learning_rate)
    
    def detect_and_adapt(self, features: MarketDataFeatures) -> Dict[str, Any]:
        """
        Detect market regime and adapt trading strategy
        
        Args:
            features: Market data features
            
        Returns:
            Adaptation result
        """
        # Detect regime
        detection = self.detector.detect_regime(features)
        self.current_regime = detection.regime
        
        # Apply regime-specific adaptation
        adaptation_strategy = self.adaptation_strategies.get(detection.regime, self._adapt_to_uncertain)
        adaptation_result = adaptation_strategy(features)
        
        # Record adaptation
        self.adaptation_history.append({
            'timestamp': datetime.now().isoformat(),
            'regime': detection.regime.name,
            'confidence': detection.confidence,
            'quantum_contribution': detection.quantum_contribution,
            'adaptation': adaptation_result
        })
        
        return {
            'detection': detection.to_dict(),
            'adaptation': adaptation_result,
            'system_status': self.get_system_status()
        }
    
    def _adapt_to_stable(self, features: MarketDataFeatures) -> Dict[str, Any]:
        """Adaptation strategy for stable market regime"""
        return {
            'strategy_weights': {
                'momentum': 0.2,
                'mean_reversion': 0.3,
                'breakout': 0.1,
                'arbitrage': 0.4
            },
            'risk_parameters': {
                'max_leverage': 3.0,
                'position_size': 0.05,
                'stop_loss': 0.02,
                'take_profit': 0.06
            },
            'execution_parameters': {
                'aggressiveness': 0.3,
                'venue_preference': ['binance', 'okx', 'bybit'],
                'order_type': 'limit'
            },
            'description': "Stable market adaptation: balanced strategy with moderate risk"
        }
    
    def _adapt_to_trending(self, features: MarketDataFeatures) -> Dict[str, Any]:
        """Adaptation strategy for trending market regime"""
        # Analyze trend strength from features
        trend_strength = np.mean(features.momentum) if hasattr(features, 'momentum') else 0.5
        
        return {
            'strategy_weights': {
                'momentum': 0.5 + min(0.3, trend_strength * 0.5),
                'mean_reversion': 0.1,
                'breakout': 0.3 - min(0.2, trend_strength * 0.4),
                'arbitrage': 0.1
            },
            'risk_parameters': {
                'max_leverage': 2.0 + min(1.0, trend_strength),
                'position_size': 0.07 - min(0.03, trend_strength * 0.1),
                'stop_loss': 0.03 + min(0.02, trend_strength * 0.05),
                'take_profit': 0.08 + min(0.04, trend_strength * 0.1)
            },
            'execution_parameters': {
                'aggressiveness': 0.5 + min(0.3, trend_strength * 0.6),
                'venue_preference': ['binance', 'bybit', 'okx'],
                'order_type': 'market' if trend_strength > 0.7 else 'limit'
            },
            'description': f"Trending market adaptation (strength: {trend_strength:.2f}): "
                           f"momentum-focused with dynamic risk based on trend strength"
        }
    
    def _adapt_to_volatile(self, features: MarketDataFeatures) -> Dict[str, Any]:
        """Adaptation strategy for volatile market regime"""
        # Analyze volatility from features
        volatility = np.mean(features.volatility) if hasattr(features, 'volatility') else 0.5
        
        return {
            'strategy_weights': {
                'momentum': 0.2,
                'mean_reversion': 0.4 + min(0.3, volatility * 0.5),
                'breakout': 0.3,
                'arbitrage': 0.1
            },
            'risk_parameters': {
                'max_leverage': max(1.0, 2.0 - volatility),
                'position_size': max(0.02, 0.05 - volatility * 0.03),
                'stop_loss': min(0.08, 0.04 + volatility * 0.06),
                'take_profit': max(0.04, 0.06 - volatility * 0.02)
            },
            'execution_parameters': {
                'aggressiveness': max(0.2, 0.4 - volatility * 0.3),
                'venue_preference': ['okx', 'binance', 'bybit'],
                'order_type': 'limit'
            },
            'description': f"Volatile market adaptation (volatility: {volatility:.2f}): "
                           f"conservative with mean-reversion focus"
        }
    
    def _adapt_to_range(self, features: MarketDataFeatures) -> Dict[str, Any]:
        """Adaptation strategy for range-bound market regime"""
        # Analyze range characteristics from features
        range_width = np.std(features.returns) if hasattr(features, 'returns') else 0.3
        
        return {
            'strategy_weights': {
                'momentum': 0.1,
                'mean_reversion': 0.6 + min(0.2, range_width * 0.5),
                'breakout': 0.2 + min(0.1, (0.5 - range_width) * 0.4),
                'arbitrage': 0.1
            },
            'risk_parameters': {
                'max_leverage': 2.5 - min(1.0, range_width),
                'position_size': 0.06 - min(0.02, range_width * 0.05),
                'stop_loss': 0.03 + min(0.03, (0.5 - range_width) * 0.08),
                'take_profit': 0.05 + min(0.03, range_width * 0.06)
            },
            'execution_parameters': {
                'aggressiveness': 0.4 - min(0.2, range_width * 0.4),
                'venue_preference': ['bybit', 'okx', 'binance'],
                'order_type': 'limit'
            },
            'description': f"Range-bound market adaptation (range width: {range_width:.2f}): "
                           f"mean-reversion dominant with breakout opportunities"
        }
    
    def _adapt_to_uncertain(self, features: MarketDataFeatures) -> Dict[str, Any]:
        """Adaptation strategy for uncertain market regime"""
        return {
            'strategy_weights': {
                'momentum': 0.25,
                'mean_reversion': 0.25,
                'breakout': 0.25,
                'arbitrage': 0.25
            },
            'risk_parameters': {
                'max_leverage': 1.5,
                'position_size': 0.03,
                'stop_loss': 0.05,
                'take_profit': 0.05
            },
            'execution_parameters': {
                'aggressiveness': 0.3,
                'venue_preference': ['binance', 'okx', 'bybit'],
                'order_type': 'limit'
            },
            'description': "Uncertain market adaptation: balanced conservative approach"
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        detector_status = self.detector.get_performance_metrics()
        
        return {
            'current_regime': self.current_regime.name if self.current_regime else None,
            'detector_status': detector_status,
            'total_adaptations': len(self.adaptation_history),
            'recent_adaptations': [a['adaptation']['description'] for a in self.adaptation_history[-5:]]
        }
    
    def adapt_to_new_market_conditions(self, new_data: List[Tuple[MarketDataFeatures, MarketRegime]],
                                      epochs: int = 30, learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Adapt the system to new market conditions
        
        Args:
            new_data: New training data (features, regime)
            epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Adaptation summary
        """
        adaptation_result = self.detector.adapt_to_new_data(new_data, epochs, learning_rate)
        
        return {
            'status': 'adaptation_completed',
            'new_data_samples': len(new_data),
            'adaptation_metrics': adaptation_result,
            'system_status': self.get_system_status()
        }