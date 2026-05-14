"""
Causal Inference Engine for Argus Ultimate
==========================================
Uses DoWhy to infer causal relationships between market variables.
Example: "BTC price drop → ETH price drop → SOL volume spike"

Dependencies:
- dowhy
- pandas
- numpy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dowhy import CausalModel
from dowhy.causal_identifier import Identify
from dowhy.causal_estimator import Estimate
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class CausalGraph:
    """
    Represents a causal graph of market relationships.
    """
    def __init__(self):
        self.nodes: List[str] = []
        self.edges: List[Tuple[str, str]] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        self.causal_effects: Dict[Tuple[str, str], float] = {}

    def add_node(self, node: str):
        if node not in self.nodes:
            self.nodes.append(node)

    def add_edge(self, source: str, target: str, effect: float = 0.0):
        """Add a directed edge from source to target with a causal effect."""
        self.add_node(source)
        self.add_node(target)
        if (source, target) not in self.edges:
            self.edges.append((source, target))
        self.adjacency[source].append(target)
        self.causal_effects[(source, target)] = effect

    def get_children(self, node: str) -> List[str]:
        """Get all nodes that are directly influenced by the given node."""
        return self.adjacency.get(node, [])

    def get_parents(self, node: str) -> List[str]:
        """Get all nodes that directly influence the given node."""
        return [src for src, tgt in self.edges if tgt == node]

    def get_effect(self, source: str, target: str) -> Optional[float]:
        """Get the causal effect from source to target."""
        return self.causal_effects.get((source, target))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the graph to a dictionary."""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "causal_effects": self.causal_effects,
        }


class CausalEngine:
    """
    Causal Inference Engine for market analysis.
    Uses DoWhy to:
    1. Identify causal relationships
    2. Estimate causal effects
    3. Predict interventions (e.g., "What if BTC drops 5%?")
    """
    def __init__(self):
        self.graph = CausalGraph()
        self.models: Dict[str, CausalModel] = {}
        self.history: deque = deque(maxlen=1000)

    def add_data(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        graph: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add data and infer causal relationships.
        Args:
            data: DataFrame with market data
            treatment: Variable to treat as cause (e.g., "BTC_price")
            outcome: Variable to treat as effect (e.g., "ETH_price")
            graph: Optional causal graph string (e.g., "digraph {BTC_price -> ETH_price}")
        Returns:
            Dict with causal effect and model info
        """
        try:
            # Create causal model
            model = CausalModel(
                data=data,
                treatment=treatment,
                outcome=outcome,
                graph=graph,
            )

            # Identify causal effect
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            if identified_estimand is None:
                logger.warning(f"Could not identify causal effect for {treatment} -> {outcome}")
                return {"error": "Unidentifiable", "treatment": treatment, "outcome": outcome}

            # Estimate causal effect
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.propensity_score_matching",
                target_units="ate",
            )

            # Add to graph
            effect = estimate.value if estimate is not None else 0.0
            self.graph.add_edge(treatment, outcome, effect)

            # Store model
            model_key = f"{treatment}_{outcome}"
            self.models[model_key] = model

            # Log history
            self.history.append({
                "treatment": treatment,
                "outcome": outcome,
                "effect": effect,
                "timestamp": pd.Timestamp.now().isoformat(),
            })

            return {
                "treatment": treatment,
                "outcome": outcome,
                "causal_effect": float(effect) if effect is not None else 0.0,
                "confidence_interval": estimate.get_confidence_intervals() if estimate else None,
                "model_key": model_key,
            }
        except Exception as e:
            logger.error(f"Error in add_data: {e}")
            return {"error": str(e), "treatment": treatment, "outcome": outcome}

    def predict_intervention(
        self,
        treatment: str,
        outcome: str,
        intervention_value: float,
        current_value: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Predict the effect of an intervention (e.g., "What if BTC drops to 45000?").
        Args:
            treatment: Variable to intervene on (e.g., "BTC_price")
            outcome: Outcome variable (e.g., "ETH_price")
            intervention_value: Value to set for treatment
            current_value: Current value of treatment (if None, uses mean)
        Returns:
            Dict with predicted outcome and effect
        """
        model_key = f"{treatment}_{outcome}"
        if model_key not in self.models:
            return {"error": f"No model for {treatment} -> {outcome}", "model_key": model_key}

        model = self.models[model_key]

        try:
            # Get current value if not provided
            if current_value is None:
                current_value = float(model.data[treatment].mean())

            # Predict effect of intervention
            result = model.predict_effect(
                intervention_value,
                target_units="ate",
            )
            predicted_outcome = float(result.value) if result is not None else None

            return {
                "treatment": treatment,
                "outcome": outcome,
                "intervention_value": intervention_value,
                "current_value": current_value,
                "predicted_outcome": predicted_outcome,
                "effect": predicted_outcome - current_value if predicted_outcome is not None else None,
            }
        except Exception as e:
            logger.error(f"Error in predict_intervention: {e}")
            return {"error": str(e), "treatment": treatment, "outcome": outcome}

    def get_causal_chain(
        self,
        start_node: str,
        end_node: str,
        max_depth: int = 3,
    ) -> List[List[str]]:
        """
        Find all causal chains from start_node to end_node.
        Example: [["BTC_price", "ETH_price", "SOL_volume"]]
        Args:
            start_node: Starting node for the chain
            end_node: Target node for the chain
            max_depth: Maximum length of the chain
        Returns:
            List of causal chains (each chain is a list of nodes)
        """
        chains = []

        def dfs(node: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            if node == end_node:
                chains.append(path.copy())
                return
            for child in self.graph.get_children(node):
                if child not in path:
                    path.append(child)
                    dfs(child, path, depth + 1)
                    path.pop()

        dfs(start_node, [start_node], 0)
        return chains

    def get_graph(self) -> CausalGraph:
        """Get the current causal graph."""
        return self.graph

    def get_history(self) -> List[Dict]:
        """Get history of causal inferences."""
        return list(self.history)

    def clear(self):
        """Clear all models and history."""
        self.models.clear()
        self.history.clear()
        self.graph = CausalGraph()
        logger.info("CausalEngine cleared")
