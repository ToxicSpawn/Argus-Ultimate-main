"""
Causal Inference Learning
=========================
Goes beyond correlation to understand CAUSAL relationships between
market features, parameters, and outcomes.

Key Features:
1. Causal Graph Learning - Learns causal structure from data
2. Counterfactual Analysis - "What would have happened if X was different?"
3. Feature Attribution - Understand which features drive outcomes
4. Intervention Planning - Plan parameter changes with known effects

NEW: Integrated with DoWhy-based CausalEngine for advanced inference
"""

from __future__ import annotations

import logging
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Try to import the new CausalEngine
try:
    from learning.causal_engine import CausalEngine
    CAUSAL_ENGINE_AVAILABLE = True
except ImportError:
    CAUSAL_ENGINE_AVAILABLE = False
    logger.warning("CausalEngine not available (DoWhy not installed)")


@dataclass
class CausalEdge:
    """Represents a causal relationship."""
    cause: str
    effect: str
    strength: float  # -1 to 1
    confidence: float  # 0 to 1
    observations: int
    
    def __repr__(self):
        return f"{self.cause} → {self.effect} (strength={self.strength:.2f}, conf={self.confidence:.2f})"


class CausalGraph:
    """Learns and maintains causal relationships."""
    
    def __init__(self):
        self.edges: Dict[Tuple[str, str], CausalEdge] = {}
        self.node_effects: Dict[str, List[str]] = defaultdict(list)
        self.node_causes: Dict[str, List[str]] = defaultdict(list)
    
    def add_observation(self, cause: str, effect: str, 
                        cause_value: float, effect_value: float) -> None:
        """Add an observation of a potential causal relationship."""
        key = (cause, effect)
        
        if key not in self.edges:
            self.edges[key] = CausalEdge(
                cause=cause,
                effect=effect,
                strength=0.0,
                confidence=0.1,
                observations=0
            )
            self.node_effects[cause].append(effect)
            self.node_causes[effect].append(cause)
        
        edge = self.edges[key]
        edge.observations += 1
        
        # Update strength using incremental correlation
        # Simplified: use sign and magnitude of relationship
        if edge.observations == 1:
            edge.strength = 1.0 if cause_value * effect_value > 0 else -1.0
        else:
            # Exponential moving average
            alpha = 1.0 / (edge.observations + 1)
            new_strength = 1.0 if cause_value * effect_value > 0 else -1.0
            edge.strength = edge.strength * (1 - alpha) + new_strength * alpha
        
        # Confidence increases with observations
        edge.confidence = min(0.95, 0.1 + np.log1p(edge.observations) * 0.2)
    
    def get_effects(self, node: str, min_confidence: float = 0.3) -> List[CausalEdge]:
        """Get all effects of a node."""
        return [
            edge for key, edge in self.edges.items()
            if key[0] == node and edge.confidence >= min_confidence
        ]
    
    def get_causes(self, node: str, min_confidence: float = 0.3) -> List[CausalEdge]:
        """Get all causes of a node."""
        return [
            edge for key, edge in self.edges.items()
            if key[1] == node and edge.confidence >= min_confidence
        ]
    
    def get_direct_causes(self, effect: str, min_confidence: float = 0.5) -> List[str]:
        """Get direct causes of an effect."""
        causes = self.get_causes(effect, min_confidence)
        return [c.cause for c in sorted(causes, key=lambda x: x.confidence, reverse=True)]
    
    def get_direct_effects(self, cause: str, min_confidence: float = 0.5) -> List[str]:
        """Get direct effects of a cause."""
        effects = self.get_effects(cause, min_confidence)
        return [e.effect for e in sorted(effects, key=lambda x: x.confidence, reverse=True)]
    
    def get_causal_path(self, start: str, end: str, 
                         max_depth: int = 5) -> Optional[List[str]]:
        """Find a causal path from start to end."""
        visited = set()
        
        def dfs(current: str, path: List[str]) -> Optional[List[str]]:
            if current == end:
                return path
            if current in visited or len(path) > max_depth:
                return None
            
            visited.add(current)
            
            for effect in self.get_direct_effects(current):
                result = dfs(effect, path + [effect])
                if result:
                    return result
            
            visited.remove(current)
            return None
        
        return dfs(start, [start])
    
    def get_intervention_effect(self, intervention: str, 
                                 target: str) -> Optional[float]:
        """
        Estimate the effect of intervening on 'intervention' on 'target'.
        Uses backdoor adjustment when possible.
        """
        path = self.get_causal_path(intervention, target)
        if not path:
            return None
        
        # Sum up edge strengths along the path
        total_effect = 1.0
        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            if key in self.edges:
                total_effect *= self.edges[key].strength
        
        return total_effect


class CounterfactualAnalyzer:
    """Analyzes 'what if' scenarios."""
    
    def __init__(self, causal_graph: CausalGraph):
        self.graph = causal_graph
        self.scenarios: Deque[Dict] = deque(maxlen=100)
    
    def analyze_scenario(self,
                         actual_params: Dict[str, float],
                         actual_outcome: float,
                         hypothetical_params: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze what would have happened with different parameters.
        """
        # Calculate parameter differences
        param_changes = {}
        for key in set(actual_params.keys()) | set(hypothetical_params.keys()):
            actual = actual_params.get(key, 0)
            hypothetical = hypothetical_params.get(key, 0)
            param_changes[key] = hypothetical - actual
        
        # Estimate outcome change using causal graph
        estimated_outcome_change = 0.0
        contributing_factors = []
        
        for param, change in param_changes.items():
            if abs(change) < 0.001:
                continue
            
            # Get causal effect on outcome
            effect = self.graph.get_intervention_effect(param, "trade_pnl")
            if effect is not None:
                contribution = change * effect
                estimated_outcome_change += contribution
                contributing_factors.append({
                    "parameter": param,
                    "change": change,
                    "effect_strength": effect,
                    "contribution": contribution,
                })
        
        # Hypothetical outcome
        hypothetical_outcome = actual_outcome + estimated_outcome_change
        
        scenario = {
            "actual_params": dict(actual_params),
            "actual_outcome": actual_outcome,
            "hypothetical_params": dict(hypothetical_params),
            "hypothetical_outcome": hypothetical_outcome,
            "outcome_difference": estimated_outcome_change,
            "contributing_factors": contributing_factors,
            "would_have_improved": estimated_outcome_change > 0,
            "timestamp": time.time(),
        }
        
        self.scenarios.append(scenario)
        
        return scenario
    
    def get_best_known_parameters(self, 
                                   target_metric: str = "trade_pnl") -> Dict[str, float]:
        """Get best parameters based on counterfactual analysis."""
        if not self.scenarios:
            return {}
        
        # Find scenarios that improved outcomes
        improvements = [
            s for s in self.scenarios 
            if s.get("would_have_improved", False)
        ]
        
        if not improvements:
            return {}
        
        # Aggregate successful parameter changes
        param_improvements: Dict[str, List[float]] = defaultdict(list)
        
        for scenario in improvements:
            for factor in scenario.get("contributing_factors", []):
                if factor["contribution"] > 0:
                    param = factor["parameter"]
                    change = factor["change"]
                    param_improvements[param].append(change)
        
        # Take median of successful changes
        best_params = {}
        for param, changes in param_improvements.items():
            if len(changes) >= 3:
                best_params[param] = np.median(changes)
        
        return best_params


class FeatureAttribution:
    """Determines which market features drive trading outcomes."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Feature-outcome pairs
        self.observations: Deque[Dict[str, float]] = deque(maxlen=window_size)
        self.outcomes: Deque[float] = deque(maxlen=window_size)
        
        # Attribution scores
        self.attribution_scores: Dict[str, float] = {}
    
    def record(self, features: Dict[str, float], outcome: float) -> None:
        """Record a feature-outcome observation."""
        self.observations.append(features)
        self.outcomes.append(outcome)
    
    def calculate_attributions(self) -> Dict[str, float]:
        """Calculate feature attributions using linear regression."""
        if len(self.observations) < 20:
            return {}
        
        # Get all feature names
        feature_names = set()
        for obs in self.observations:
            feature_names.update(obs.keys())
        
        # Build feature matrix and outcome vector
        n = len(self.observations)
        X = np.zeros((n, len(feature_names)))
        y = np.array(list(self.outcomes))
        
        feature_list = sorted(feature_names)
        for i, obs in enumerate(self.observations):
            for j, fname in enumerate(feature_list):
                X[i, j] = obs.get(fname, 0.0)
        
        # Normalize features
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1.0  # Avoid division by zero
        X_normalized = (X - X_mean) / X_std
        
        # Simple linear regression for each feature
        attributions = {}
        for j, fname in enumerate(feature_list):
            # Correlation-based attribution
            if np.std(X_normalized[:, j]) > 0:
                correlation = np.corrcoef(X_normalized[:, j], y)[0, 1]
                if not np.isnan(correlation):
                    attributions[fname] = correlation
        
        self.attribution_scores = attributions
        return attributions
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most impactful features."""
        if not self.attribution_scores:
            self.calculate_attributions()
        
        sorted_features = sorted(
            self.attribution_scores.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return sorted_features[:n]
    
    def get_recommendation(self) -> Dict[str, Any]:
        """Get recommendation based on feature attributions."""
        top_features = self.get_top_features(5)
        
        if not top_features:
            return {"recommendation": "insufficient_data"}
        
        # Analyze recent feature values
        if self.observations:
            recent = self.observations[-1]
            
            recommendations = []
            for feature, attribution in top_features:
                value = recent.get(feature, 0)
                
                if attribution > 0.3:
                    recommendations.append({
                        "feature": feature,
                        "attribution": attribution,
                        "current_value": value,
                        "action": "increase" if value < 0.5 else "maintain",
                    })
                elif attribution < -0.3:
                    recommendations.append({
                        "feature": feature,
                        "attribution": attribution,
                        "current_value": value,
                        "action": "decrease" if value > 0.5 else "maintain",
                    })
            
            return {
                "top_features": [(f, a) for f, a in top_features],
                "recommendations": recommendations,
            }
        
        return {"recommendation": "analyzing"}


class CausalLearningSystem:
    """
    Unified causal learning system that combines:
    1. Causal graph learning
    2. Counterfactual analysis
    3. Feature attribution
    4. DoWhy-based causal inference (NEW)
    """
    
    def __init__(self):
        self.causal_graph = CausalGraph()
        self.counterfactual = CounterfactualAnalyzer(self.causal_graph)
        self.feature_attribution = FeatureAttribution()
        
        # DoWhy-based causal engine (NEW)
        if CAUSAL_ENGINE_AVAILABLE:
            self.causal_engine = CausalEngine()
        else:
            self.causal_engine = None
        
        # Learning state
        self.observations_made = 0
        self.causal_insights: List[Dict] = []
    
    def record_trade(self,
                     parameters: Dict[str, float],
                     market_features: Dict[str, float],
                     outcome: float) -> None:
        """Record a trade for causal learning."""
        self.observations_made += 1
        
        # Update causal graph
        for param_name, param_value in parameters.items():
            self.causal_graph.add_observation(
                cause=f"param_{param_name}",
                effect="trade_pnl",
                cause_value=param_value,
                effect_value=outcome
            )
        
        for feature_name, feature_value in market_features.items():
            self.causal_graph.add_observation(
                cause=f"feature_{feature_name}",
                effect="trade_pnl",
                cause_value=feature_value,
                effect_value=outcome
            )
        
        # Update feature attribution
        all_features = {**parameters, **market_features}
        self.feature_attribution.record(all_features, outcome)
        
        # Store scenario for counterfactual
        self.counterfactual.analyze_scenario(
            actual_params=parameters,
            actual_outcome=outcome,
            hypothetical_params=parameters  # Store baseline
        )
        
        # Update DoWhy causal engine if available
        if self.causal_engine and len(market_features) > 0:
            try:
                # Convert to DataFrame
                data = pd.DataFrame([{**parameters, **market_features, "trade_pnl": outcome}])
                for param in parameters:
                    self.causal_engine.add_data(
                        data=data,
                        treatment=param,
                        outcome="trade_pnl",
                    )
            except Exception as e:
                logger.debug(f"DoWhy causal inference failed: {e}")
    
    def analyze_causality_with_dowhy(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
    ) -> Dict[str, Any]:
        """
        Analyze causality using DoWhy (NEW).
        Args:
            data: DataFrame with market data
            treatment: Variable to treat as cause
            outcome: Variable to treat as effect
        Returns:
            Dict with causal effect and model info
        """
        if not CAUSAL_ENGINE_AVAILABLE or self.causal_engine is None:
            return {"error": "DoWhy not available"}
        
        return self.causal_engine.add_data(data, treatment, outcome)
    
    def predict_intervention_with_dowhy(
        self,
        treatment: str,
        outcome: str,
        intervention_value: float,
    ) -> Dict[str, Any]:
        """
        Predict the effect of an intervention using DoWhy (NEW).
        Args:
            treatment: Variable to intervene on
            outcome: Outcome variable
            intervention_value: Value to set for treatment
        Returns:
            Dict with predicted outcome and effect
        """
        if not CAUSAL_ENGINE_AVAILABLE or self.causal_engine is None:
            return {"error": "DoWhy not available"}
        
        return self.causal_engine.predict_intervention(treatment, outcome, intervention_value)
    
    def get_causal_insights(self) -> Dict[str, Any]:
        """Get causal insights from both engines."""
        # Recalculate feature attributions
        self.feature_attribution.calculate_attributions()
        
        # Get top drivers of PnL
        top_drivers = self.feature_attribution.get_top_features(10)
        
        # Get causal graph stats
        total_edges = len(self.causal_graph.edges)
        high_confidence_edges = sum(
            1 for e in self.causal_graph.edges.values() 
            if e.confidence > 0.5
        )
        
        insights = {
            "observations_made": self.observations_made,
            "causal_graph": {
                "total_edges": total_edges,
                "high_confidence_edges": high_confidence_edges,
            },
            "top_pnl_drivers": [
                {"feature": f, "attribution": a}
                for f, a in top_drivers
            ],
            "recommendations": self.feature_attribution.get_recommendation(),
        }
        
        # Add DoWhy insights if available
        if CAUSAL_ENGINE_AVAILABLE and self.causal_engine:
            insights["dowhy_insights"] = {
                "graph": self.causal_engine.get_graph().to_dict(),
                "history": self.causal_engine.get_history(),
            }
        
        return insights
    
    def suggest_parameter_change(self,
                                  current_params: Dict[str, float],
                                  target_outcome: float = 0.0) -> Dict[str, float]:
        """Suggest parameter changes to achieve target outcome."""
        suggestions = {}
        
        # Get top drivers
        top_drivers = self.feature_attribution.get_top_features(5)
        
        for feature, attribution in top_drivers:
            if feature in current_params and abs(attribution) > 0.2:
                # Suggest direction based on attribution sign
                current = current_params[feature]
                if attribution > 0:
                    # Positive correlation - increase to improve
                    suggestions[feature] = current * 0.05  # 5% increase
                else:
                    # Negative correlation - decrease to improve
                    suggestions[feature] = current * -0.05  # 5% decrease
        
        return suggestions


# Singleton
_causal_system: Optional[CausalLearningSystem] = None


def get_causal_system() -> CausalLearningSystem:
    """Get or create singleton causal learning system."""
    global _causal_system
    if _causal_system is None:
        _causal_system = CausalLearningSystem()
    return _causal_system


def reset_causal_system() -> None:
    """Reset singleton (for testing)."""
    global _causal_system
    _causal_system = None
