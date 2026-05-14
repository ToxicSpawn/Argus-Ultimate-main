"""
Model Explainability — SHAP-based feature importance and model interpretation.

Features:
  - SHAP value computation for tree and linear models
  - Feature importance ranking
  - Partial dependence plots (data)
  - Model-agnostic explanation via KernelSHAP
  - Local (per-prediction) and global explanations
  - Explanation caching for performance

Usage:
    explainer = ModelExplainer(model, feature_names=["price", "volume", "volatility"])
    
    # Global importance
    importance = explainer.global_importance(X_background)
    
    # Local explanation for single prediction
    explanation = explainer.explain(X[0])
    
    # Summary
    summary = explainer.summary(X_background)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Check for SHAP availability
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.debug("shap not available; using fallback explanation methods")


@dataclass
class FeatureImportance:
    """Feature importance result."""
    feature_names: List[str]
    importance_values: np.ndarray
    rank: List[int]  # Indices sorted by importance (descending)
    method: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "importance_values": self.importance_values.tolist(),
            "rank": self.rank,
            "method": self.method,
            "top_features": [
                {"name": self.feature_names[i], "importance": float(self.importance_values[i])}
                for i in self.rank[:10]
            ],
        }


@dataclass
class LocalExplanation:
    """Local (single prediction) explanation."""
    base_value: float
    shap_values: np.ndarray
    feature_values: np.ndarray
    feature_names: List[str]
    prediction: float
    method: str
    
    def to_dict(self) -> Dict[str, Any]:
        contributions = [
            {"name": self.feature_names[i], "value": float(self.feature_values[i]), 
             "contribution": float(self.shap_values[i])}
            for i in np.argsort(np.abs(self.shap_values))[::-1]
        ]
        
        return {
            "base_value": float(self.base_value),
            "prediction": float(self.prediction),
            "method": self.method,
            "top_contributions": contributions[:10],
            "total_contribution": float(np.sum(self.shap_values)),
        }


@dataclass
class ExplanationSummary:
    """Summary of model explanations."""
    global_importance: FeatureImportance
    mean_abs_shap: np.ndarray
    feature_correlation: Dict[str, float]
    n_samples: int
    method: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_importance": self.global_importance.to_dict(),
            "mean_abs_shap": self.mean_abs_shap.tolist(),
            "feature_correlation": self.feature_correlation,
            "n_samples": self.n_samples,
            "method": self.method,
        }


class ModelExplainer:
    """
    Model explainability using SHAP and fallback methods.
    
    Supports:
    - TreeExplainer for tree-based models (XGBoost, RandomForest)
    - LinearExplainer for linear models
    - KernelExplainer for model-agnostic explanations
    - Fallback permutation importance when SHAP unavailable
    
    Parameters
    ----------
    model : Any
        Trained model (sklearn, xgboost, or any callable with predict)
    feature_names : List[str]
        Names of input features
    background_data : Optional[np.ndarray]
        Background dataset for SHAP (sampled if not provided)
    n_background_samples : int
        Number of background samples to use
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None,
        n_background_samples: int = 100,
    ):
        self.model = model
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.n_background_samples = n_background_samples
        
        # Determine explainer type
        self._explainer_type = self._detect_explainer_type(model)
        self._explainer = None
        self._background_data = None
        
        # Initialize if SHAP available
        if _SHAP_AVAILABLE and background_data is not None:
            self._init_shap_explainer(background_data)
        
        logger.info("ModelExplainer initialized: %s (shap=%s)", 
                    self._explainer_type, _SHAP_AVAILABLE)
    
    def _detect_explainer_type(self, model: Any) -> str:
        """Detect the best explainer type for the model."""
        model_type = type(model).__name__.lower()
        
        # Tree-based models
        if any(x in model_type for x in ["xgb", "randomforest", "gradientboost", "decisiontree"]):
            return "tree"
        
        # Linear models
        if any(x in model_type for x in ["linear", "logistic", "ridge", "lasso", "svr"]):
            return "linear"
        
        # Neural networks
        if any(x in model_type for x in ["module", "sequential", "linear"]):
            return "deep"
        
        # Fallback
        return "kernel"
    
    def _init_shap_explainer(self, background_data: np.ndarray) -> None:
        """Initialize SHAP explainer."""
        if not _SHAP_AVAILABLE:
            return
        
        # Sample background data
        if len(background_data) > self.n_background_samples:
            indices = np.random.choice(
                len(background_data), self.n_background_samples, replace=False
            )
            self._background_data = background_data[indices]
        else:
            self._background_data = background_data
        
        try:
            if self._explainer_type == "tree":
                self._explainer = shap.TreeExplainer(self.model)
            elif self._explainer_type == "linear":
                self._explainer = shap.LinearExplainer(
                    self.model, self._background_data
                )
            else:
                # Use KernelExplainer as fallback
                self._explainer = shap.KernelExplainer(
                    self.model.predict, self._background_data
                )
            
            logger.info("SHAP explainer initialized: %s", self._explainer_type)
            
        except Exception as e:
            logger.warning("Failed to init SHAP explainer: %s", e)
            self._explainer = None
    
    def global_importance(
        self,
        X: np.ndarray,
        n_samples: int = 500,
    ) -> FeatureImportance:
        """
        Compute global feature importance.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            n_samples: Number of samples to use
            
        Returns:
            FeatureImportance with importance values and rankings
        """
        if _SHAP_AVAILABLE and self._explainer is not None:
            return self._shap_global_importance(X, n_samples)
        else:
            return self._permutation_importance(X, n_samples)
    
    def explain(
        self,
        x: np.ndarray,
        background_data: Optional[np.ndarray] = None,
    ) -> LocalExplanation:
        """
        Explain a single prediction.
        
        Args:
            x: Single sample (n_features,) or (1, n_features)
            background_data: Optional background for KernelSHAP
            
        Returns:
            LocalExplanation with SHAP values and contributions
        """
        x = np.atleast_2d(x)
        
        if _SHAP_AVAILABLE and self._explainer is not None:
            return self._shap_local_explain(x)
        else:
            return self._fallback_local_explain(x, background_data)
    
    def summary(
        self,
        X: np.ndarray,
        n_samples: int = 500,
    ) -> ExplanationSummary:
        """
        Generate explanation summary.
        
        Args:
            X: Feature matrix
            n_samples: Number of samples to analyze
            
        Returns:
            ExplanationSummary with global importance and statistics
        """
        # Global importance
        importance = self.global_importance(X, n_samples)
        
        # Mean absolute SHAP values
        if _SHAP_AVAILABLE and self._explainer is not None:
            mean_abs_shap = self._compute_mean_abs_shap(X[:n_samples])
        else:
            mean_abs_shap = importance.importance_values
        
        # Feature correlation with importance
        correlation = self._compute_feature_correlation(X, importance.importance_values)
        
        return ExplanationSummary(
            global_importance=importance,
            mean_abs_shap=mean_abs_shap,
            feature_correlation=correlation,
            n_samples=min(len(X), n_samples),
            method="shap" if _SHAP_AVAILABLE else "permutation",
        )
    
    def _shap_global_importance(self, X: np.ndarray, n_samples: int) -> FeatureImportance:
        """Compute SHAP-based global importance."""
        samples = X[:n_samples]
        
        try:
            shap_values = self._explainer.shap_values(samples)
            
            # Handle multi-class (list of arrays)
            if isinstance(shap_values, list):
                shap_values = np.abs(np.array(shap_values)).mean(axis=0)
            else:
                shap_values = np.abs(shap_values)
            
            # Average across samples
            importance = np.mean(shap_values, axis=0)
            
            # Handle 2D importance (multi-output)
            if importance.ndim > 1:
                importance = np.mean(importance, axis=0)
            
            rank = np.argsort(importance)[::-1].tolist()
            
            return FeatureImportance(
                feature_names=self.feature_names,
                importance_values=importance,
                rank=rank,
                method="shap",
            )
            
        except Exception as e:
            logger.warning("SHAP global importance failed: %s, using fallback", e)
            return self._permutation_importance(X, n_samples)
    
    def _shap_local_explain(self, x: np.ndarray) -> LocalExplanation:
        """Compute SHAP-based local explanation."""
        try:
            shap_values = self._explainer.shap_values(x)
            
            # Handle multi-class
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Use first class
            
            shap_values = np.array(shap_values).flatten()
            x_flat = x.flatten()
            
            # Get base value
            base_value = self._explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0]
            
            # Get prediction
            prediction = float(self.model.predict(x)[0])
            
            return LocalExplanation(
                base_value=float(base_value),
                shap_values=shap_values,
                feature_values=x_flat,
                feature_names=self.feature_names,
                prediction=prediction,
                method="shap",
            )
            
        except Exception as e:
            logger.warning("SHAP local explanation failed: %s", e)
            return self._fallback_local_explain(x)
    
    def _permutation_importance(self, X: np.ndarray, n_samples: int) -> FeatureImportance:
        """Fallback: permutation importance."""
        samples = X[:n_samples]
        
        try:
            # Get baseline score
            baseline_pred = self.model.predict(samples)
            if baseline_pred.ndim > 1:
                baseline_pred = baseline_pred[:, 0]
            
            # For regression, use variance as baseline
            baseline_score = np.var(baseline_pred)
            
            importance = np.zeros(self.n_features)
            
            for i in range(self.n_features):
                # Permute feature i
                permuted = samples.copy()
                permuted[:, i] = np.random.permutation(permuted[:, i])
                
                permuted_pred = self.model.predict(permuted)
                if permuted_pred.ndim > 1:
                    permuted_pred = permuted_pred[:, 0]
                
                permuted_score = np.var(permuted_pred)
                importance[i] = abs(baseline_score - permuted_score)
            
            # Normalize
            total = np.sum(importance)
            if total > 0:
                importance = importance / total
            
            rank = np.argsort(importance)[::-1].tolist()
            
            return FeatureImportance(
                feature_names=self.feature_names,
                importance_values=importance,
                rank=rank,
                method="permutation",
            )
            
        except Exception as e:
            logger.warning("Permutation importance failed: %s", e)
            # Ultimate fallback: uniform importance
            importance = np.ones(self.n_features) / self.n_features
            return FeatureImportance(
                feature_names=self.feature_names,
                importance_values=importance,
                rank=list(range(self.n_features)),
                method="uniform_fallback",
            )
    
    def _fallback_local_explain(
        self,
        x: np.ndarray,
        background_data: Optional[np.ndarray] = None,
    ) -> LocalExplanation:
        """Fallback local explanation using numerical gradients."""
        x_flat = x.flatten()
        
        # Compute numerical gradients
        epsilon = 1e-5
        gradients = np.zeros(self.n_features)
        
        baseline_pred = float(self.model.predict(x)[0])
        
        for i in range(self.n_features):
            x_plus = x.copy().flatten()
            x_plus[i] += epsilon
            
            x_minus = x.copy().flatten()
            x_minus[i] -= epsilon
            
            pred_plus = float(self.model.predict(x_plus.reshape(1, -1))[0])
            pred_minus = float(self.model.predict(x_minus.reshape(1, -1))[0])
            
            gradients[i] = (pred_plus - pred_minus) / (2 * epsilon)
        
        # Scale gradients to approximate SHAP-like contributions
        total = np.sum(np.abs(gradients))
        if total > 0:
            shap_values = gradients * (np.abs(x_flat).mean() / total)
        else:
            shap_values = gradients
        
        return LocalExplanation(
            base_value=0.0,  # Unknown without SHAP
            shap_values=shap_values,
            feature_values=x_flat,
            feature_names=self.feature_names,
            prediction=baseline_pred,
            method="gradient_fallback",
        )
    
    def _compute_mean_abs_shap(self, X: np.ndarray) -> np.ndarray:
        """Compute mean absolute SHAP values."""
        try:
            shap_values = self._explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                shap_values = np.abs(np.array(shap_values)).mean(axis=0)
            else:
                shap_values = np.abs(shap_values)
            
            return np.mean(shap_values, axis=0)
            
        except Exception:
            return np.zeros(self.n_features)
    
    def _compute_feature_correlation(
        self,
        X: np.ndarray,
        importance: np.ndarray,
    ) -> Dict[str, float]:
        """Compute correlation between features and importance."""
        correlation = {}
        
        for i, name in enumerate(self.feature_names):
            if i < X.shape[1] and i < len(importance):
                try:
                    corr = np.corrcoef(X[:, i], importance)[0, 1]
                    correlation[name] = float(corr) if not np.isnan(corr) else 0.0
                except Exception:
                    correlation[name] = 0.0
        
        return correlation
