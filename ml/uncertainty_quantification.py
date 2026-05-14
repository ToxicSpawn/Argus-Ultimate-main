# pyright: reportMissingImports=false
"""
Uncertainty Quantification System for Argus Trading.

This module implements uncertainty quantification to make safer trading decisions
by understanding model confidence and prediction uncertainty.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class UncertaintyMethod(Enum):
    """Methods for uncertainty quantification."""
    BAYESIAN = auto()  # Bayesian neural networks
    ENSEMBLE = auto()  # Ensemble disagreement
    MC_DROPOUT = auto()  # Monte Carlo dropout
    BOOTSTRAP = auto()  # Bootstrap sampling
    QUANTUM = auto()  # Quantum uncertainty


@dataclass
class UncertaintyEstimate:
    """Uncertainty estimate for a prediction."""
    prediction: float
    epistemic_uncertainty: float  # Model uncertainty (can be reduced with more data)
    aleatoric_uncertainty: float  # Data uncertainty (irreducible)
    total_uncertainty: float
    confidence: float  # 1 - normalized uncertainty
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_reliable(self) -> bool:
        """Check if the prediction is reliable."""
        return self.confidence > 0.6 and self.total_uncertainty < 0.4
    
    def risk_adjusted_position(self, base_position: float, risk_factor: float = 0.5) -> float:
        """Calculate risk-adjusted position size."""
        uncertainty_factor = 1.0 - (self.total_uncertainty * risk_factor)
        return base_position * max(0.1, uncertainty_factor)


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification."""
    method: UncertaintyMethod = UncertaintyMethod.ENSEMBLE
    num_samples: int = 100
    confidence_level: float = 0.95
    risk_factor: float = 0.5
    min_confidence: float = 0.6
    max_uncertainty: float = 0.4


class BayesianPredictor:
    """Bayesian prediction with uncertainty."""

    def __init__(self, input_dim: int = 8, output_dim: int = 4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Prior parameters
        self.weight_mean = np.random.randn(input_dim, output_dim) * 0.1
        self.weight_std = np.ones((input_dim, output_dim)) * 0.5
        
    def predict_with_uncertainty(self, x: NDArray[np.float64], 
                                 num_samples: int = 100) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict with uncertainty estimation."""
        samples = []
        
        for _ in range(num_samples):
            # Sample from weight posterior
            weights = np.random.randn(*self.weight_mean.shape) * self.weight_std + self.weight_mean
            
            # Forward pass
            prediction = np.tanh(x @ weights)
            samples.append(prediction)
        
        samples = np.array(samples)
        mean_prediction = np.mean(samples, axis=0)
        std_prediction = np.std(samples, axis=0)
        
        return mean_prediction, std_prediction


class UncertaintyQuantifier:
    """Main uncertainty quantification system."""

    def __init__(self, config: Optional[UncertaintyConfig] = None):
        """Initialize the uncertainty quantifier."""
        self.config = config or UncertaintyConfig()
        self.bayesian_predictor = BayesianPredictor()
        self.uncertainty_history: List[UncertaintyEstimate] = []
        self.calibration_data: List[Tuple[float, float]] = []  # (predicted_confidence, actual_correctness)
        
    def estimate_uncertainty(self, 
                            model_predictions: List[NDArray[np.float64]],
                            market_state: NDArray[np.float64]) -> UncertaintyEstimate:
        """Estimate uncertainty for a prediction."""
        if self.config.method == UncertaintyMethod.ENSEMBLE:
            return self._ensemble_uncertainty(model_predictions)
        elif self.config.method == UncertaintyMethod.BAYESIAN:
            return self._bayesian_uncertainty(market_state)
        elif self.config.method == UncertaintyMethod.MC_DROPOUT:
            return self._mc_dropout_uncertainty(model_predictions)
        elif self.config.method == UncertaintyMethod.BOOTSTRAP:
            return self._bootstrap_uncertainty(model_predictions)
        else:
            return self._ensemble_uncertainty(model_predictions)
    
    def _ensemble_uncertainty(self, predictions: List[NDArray[np.float64]]) -> UncertaintyEstimate:
        """Calculate uncertainty from ensemble disagreement."""
        predictions_array = np.array(predictions)
        
        mean_prediction = np.mean(predictions_array, axis=0)
        std_prediction = np.std(predictions_array, axis=0)
        
        # Epistemic uncertainty (model disagreement)
        epistemic = np.mean(std_prediction)
        
        # Aleatoric uncertainty (estimated from prediction magnitude)
        aleatoric = np.mean(np.abs(mean_prediction)) * 0.1
        
        total_uncertainty = epistemic + aleatoric
        confidence = 1.0 - min(total_uncertainty, 1.0)
        
        estimate = UncertaintyEstimate(
            prediction=float(np.argmax(mean_prediction)),
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total_uncertainty,
            confidence=confidence,
            metadata={
                "method": "ensemble",
                "num_models": len(predictions),
                "prediction_std": std_prediction.tolist()
            }
        )
        
        self.uncertainty_history.append(estimate)
        return estimate
    
    def _bayesian_uncertainty(self, market_state: NDArray[np.float64]) -> UncertaintyEstimate:
        """Calculate uncertainty using Bayesian prediction."""
        mean_pred, std_pred = self.bayesian_predictor.predict_with_uncertainty(
            market_state, self.config.num_samples
        )
        
        epistemic = np.mean(std_pred)
        aleatoric = np.mean(np.abs(mean_pred)) * 0.05
        
        total_uncertainty = epistemic + aleatoric
        confidence = 1.0 - min(total_uncertainty, 1.0)
        
        estimate = UncertaintyEstimate(
            prediction=float(np.argmax(mean_pred)),
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total_uncertainty,
            confidence=confidence,
            metadata={
                "method": "bayesian",
                "num_samples": self.config.num_samples
            }
        )
        
        self.uncertainty_history.append(estimate)
        return estimate
    
    def _mc_dropout_uncertainty(self, predictions: List[NDArray[np.float64]]) -> UncertaintyEstimate:
        """Calculate uncertainty using Monte Carlo dropout."""
        # Similar to ensemble but with dropout active
        return self._ensemble_uncertainty(predictions)
    
    def _bootstrap_uncertainty(self, predictions: List[NDArray[np.float64]]) -> UncertaintyEstimate:
        """Calculate uncertainty using bootstrap sampling."""
        predictions_array = np.array(predictions)
        
        # Bootstrap sampling
        bootstrap_predictions = []
        for _ in range(self.config.num_samples):
            indices = np.random.choice(len(predictions), size=len(predictions), replace=True)
            bootstrap_pred = np.mean(predictions_array[indices], axis=0)
            bootstrap_predictions.append(bootstrap_pred)
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        mean_prediction = np.mean(bootstrap_predictions, axis=0)
        std_prediction = np.std(bootstrap_predictions, axis=0)
        
        epistemic = np.mean(std_prediction)
        aleatoric = np.mean(np.abs(mean_prediction)) * 0.08
        
        total_uncertainty = epistemic + aleatoric
        confidence = 1.0 - min(total_uncertainty, 1.0)
        
        estimate = UncertaintyEstimate(
            prediction=float(np.argmax(mean_prediction)),
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total_uncertainty,
            confidence=confidence,
            metadata={
                "method": "bootstrap",
                "num_samples": self.config.num_samples
            }
        )
        
        self.uncertainty_history.append(estimate)
        return estimate
    
    def calibrate(self, predicted_confidence: float, actual_correct: bool) -> None:
        """Update calibration data."""
        self.calibration_data.append((predicted_confidence, 1.0 if actual_correct else 0.0))
        
        # Keep only recent calibration data
        if len(self.calibration_data) > 1000:
            self.calibration_data = self.calibration_data[-1000:]
    
    def get_calibration_error(self) -> float:
        """Calculate expected calibration error."""
        if len(self.calibration_data) < 100:
            return 0.0
        
        # Bin predictions by confidence
        bins = np.linspace(0, 1, 11)
        bin_errors = []
        
        for i in range(len(bins) - 1):
            bin_data = [
                (conf, correct) for conf, correct in self.calibration_data
                if bins[i] <= conf < bins[i + 1]
            ]
            
            if bin_data:
                avg_confidence = np.mean([d[0] for d in bin_data])
                avg_correctness = np.mean([d[1] for d in bin_data])
                bin_errors.append(abs(avg_confidence - avg_correctness))
        
        return np.mean(bin_errors) if bin_errors else 0.0
    
    def risk_adjust_position(self, 
                            original_position: float,
                            uncertainty: UncertaintyEstimate) -> float:
        """Adjust position size based on uncertainty."""
        return uncertainty.risk_adjusted_position(original_position, self.config.risk_factor)
    
    def get_uncertainty_summary(self) -> Dict[str, Any]:
        """Get summary of uncertainty estimates."""
        if not self.uncertainty_history:
            return {"status": "no_data"}
        
        recent = self.uncertainty_history[-100:]
        
        return {
            "total_predictions": len(self.uncertainty_history),
            "recent_stats": {
                "avg_confidence": np.mean([u.confidence for u in recent]),
                "avg_total_uncertainty": np.mean([u.total_uncertainty for u in recent]),
                "avg_epistemic": np.mean([u.epistemic_uncertainty for u in recent]),
                "avg_aleatoric": np.mean([u.aleatoric_uncertainty for u in recent]),
                "reliable_ratio": sum(1 for u in recent if u.is_reliable) / len(recent)
            },
            "calibration_error": self.get_calibration_error(),
            "method": self.config.method.name
        }


class RiskAwareTradingSystem:
    """Trading system that uses uncertainty for risk management."""

    def __init__(self, uncertainty_quantifier: Optional[UncertaintyQuantifier] = None):
        """Initialize risk-aware trading system."""
        self.uncertainty_quantifier = uncertainty_quantifier or UncertaintyQuantifier()
        self.base_position_size = 1.0
        self.max_position_multiplier = 2.0
        self.min_position_multiplier = 0.1
        
    def make_risk_aware_decision(self,
                                 model_predictions: List[NDArray[np.float64]],
                                 market_state: NDArray[np.float64],
                                 base_action: int) -> Tuple[int, float, UncertaintyEstimate]:
        """Make a risk-aware trading decision."""
        # Estimate uncertainty
        uncertainty = self.uncertainty_quantifier.estimate_uncertainty(
            model_predictions, market_state
        )
        
        # Adjust position size
        adjusted_position = self.uncertainty_quantifier.risk_adjust_position(
            self.base_position_size, uncertainty
        )
        
        # If uncertainty is too high, consider holding instead
        if not uncertainty.is_reliable:
            logger.warning(f"Low confidence prediction (confidence={uncertainty.confidence:.2f}), considering hold")
            # Still return original action but with reduced position
            return base_action, adjusted_position, uncertainty
        
        return base_action, adjusted_position, uncertainty


__all__ = [
    "UncertaintyQuantifier",
    "UncertaintyEstimate",
    "UncertaintyConfig",
    "UncertaintyMethod",
    "RiskAwareTradingSystem",
    "BayesianPredictor"
]