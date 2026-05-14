"""
Example implementation of a LearningComponent - AdaptiveVolatilityCluster

This component demonstrates how to implement a real adaptive component
that learns from market data and adjusts its parameters.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List
import numpy as np
from .orchestrator import LearningComponent


class AdaptiveVolatilityCluster(LearningComponent):
    """
    Adaptively clusters volatility regimes and adjusts thresholds.
    
    Learns from:
    - Realized volatility
    - Price jumps
    - Volume spikes
    
    Adjusts:
    - Volatility regime thresholds
    - Cluster centers
    - Transition probabilities
    """

    def __init__(self):
        super().__init__(
            name="adaptive_volatility",
            version="1.0",
            update_frequency=5  # Update every 5 cycles
        )
        
        # Current parameters
        self.params = {
            "low_vol_threshold": 0.01,    # 1% daily move = low vol
            "high_vol_threshold": 0.03,   # 3% daily move = high vol
            "cluster_centers": [0.01, 0.02, 0.04],  # Low, medium, high vol centers
            "transition_matrix": {        # Regime transition probabilities
                "low": {"low": 0.7, "medium": 0.25, "high": 0.05},
                "medium": {"low": 0.15, "medium": 0.7, "high": 0.15},
                "high": {"low": 0.05, "medium": 0.25, "high": 0.7}
            }
        }
        
        # Learning state
        self.volatility_history = []
        self.max_history = 1000
        self.current_regime = "medium"

    def learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Learn from new market data and update volatility parameters.
        
        Args:
            data: Market data dictionary containing:
                - realized_volatility: float (daily vol)
                - price_jumps: List[float] (intraday jumps)
                - volume: float (relative volume)
                - timestamp: datetime
        
        Returns:
            Updated parameters dictionary
        """
        # Extract relevant data
        vol = data.get("realized_volatility", 0.01)
        jumps = data.get("price_jumps", [])
        volume = data.get("volume", 1.0)
        timestamp = data.get("timestamp", datetime.now(timezone.utc))
        
        # Store volatility history
        self.volatility_history.append({
            "volatility": vol,
            "jumps": jumps,
            "volume": volume,
            "timestamp": timestamp
        })
        
        # Keep history bounded
        if len(self.volatility_history) > self.max_history:
            self.volatility_history.pop(0)
            
        # Only learn if we have enough data
        if len(self.volatility_history) < 100:
            return self.params
            
        # Update cluster centers using simple online k-means
        volatilities = [x["volatility"] for x in self.volatility_history]
        new_centers = self._online_kmeans(volatilities, self.params["cluster_centers"])
        self.params["cluster_centers"] = new_centers
        
        # Sort centers and update thresholds
        sorted_centers = sorted(new_centers)
        self.params["low_vol_threshold"] = (sorted_centers[0] + sorted_centers[1]) / 2
        self.params["high_vol_threshold"] = (sorted_centers[1] + sorted_centers[2]) / 2
        
        # Update transition matrix based on recent regime transitions
        self._update_transition_matrix()
        
        # Determine current regime
        self.current_regime = self._determine_regime(vol)
        
        self.last_updated = datetime.now(timezone.utc)
        return self.params

    def _online_kmeans(self, data: List[float], centers: List[float], learning_rate: float = 0.01) -> List[float]:
        """
        Simple online k-means update for volatility clusters.
        """
        if not data:
            return centers
            
        # Assign each point to nearest cluster
        assignments = []
        for point in data:
            distances = [abs(point - c) for c in centers]
            assignments.append(distances.index(min(distances)))
        
        # Update cluster centers
        counts = [0] * len(centers)
        new_centers = [0.0] * len(centers)
        
        for point, cluster in zip(data, assignments):
            counts[cluster] += 1
            new_centers[cluster] += point
        
        # Calculate new centers (with learning rate for stability)
        updated_centers = []
        for i in range(len(centers)):
            if counts[i] > 0:
                updated_centers.append(
                    centers[i] * (1 - learning_rate) +
                    (new_centers[i] / counts[i]) * learning_rate
                )
            else:
                updated_centers.append(centers[i])
                
        return updated_centers

    def _update_transition_matrix(self) -> None:
        """
        Update regime transition probabilities based on recent history.
        """
        if len(self.volatility_history) < 2:
            return
            
        # Get recent regimes
        regimes = []
        for data in self.volatility_history[-100:]:
            vol = data["volatility"]
            regimes.append(self._determine_regime(vol))
        
        # Count transitions
        transitions = {
            "low": {"low": 0, "medium": 0, "high": 0},
            "medium": {"low": 0, "medium": 0, "high": 0},
            "high": {"low": 0, "medium": 0, "high": 0}
        }
        
        for i in range(1, len(regimes)):
            from_regime = regimes[i-1]
            to_regime = regimes[i]
            transitions[from_regime][to_regime] += 1
        
        # Normalize to probabilities
        for from_regime in transitions:
            total = sum(transitions[from_regime].values())
            if total > 0:
                for to_regime in transitions[from_regime]:
                    transitions[from_regime][to_regime] /= total
        
        # Blend with existing probabilities for stability
        blend_factor = 0.1  # 10% new data, 90% existing
        for from_regime in self.params["transition_matrix"]:
            for to_regime in self.params["transition_matrix"][from_regime]:
                self.params["transition_matrix"][from_regime][to_regime] = (
                    self.params["transition_matrix"][from_regime][to_regime] * (1 - blend_factor) +
                    transitions[from_regime][to_regime] * blend_factor
                )

    def _determine_regime(self, volatility: float) -> str:
        """Determine volatility regime based on current thresholds."""
        if volatility < self.params["low_vol_threshold"]:
            return "low"
        elif volatility > self.params["high_vol_threshold"]:
            return "high"
        else:
            return "medium"

    def get_params(self) -> Dict[str, Any]:
        """Get current parameters."""
        return {
            **self.params,
            "current_regime": self.current_regime,
            "last_updated": self.last_updated
        }

    def rollback(self) -> None:
        """Revert to last known good state."""
        # In a real implementation, we would restore from a backup
        # For this example, we'll just reset to initial values
        self.params = {
            "low_vol_threshold": 0.01,
            "high_vol_threshold": 0.03,
            "cluster_centers": [0.01, 0.02, 0.04],
            "transition_matrix": {
                "low": {"low": 0.7, "medium": 0.25, "high": 0.05},
                "medium": {"low": 0.15, "medium": 0.7, "high": 0.15},
                "high": {"low": 0.05, "medium": 0.25, "high": 0.7}
            }
        }

    def validate(self, new_params: Dict[str, Any]) -> bool:
        """Validate proposed parameter changes."""
        # Check threshold separation
        if (new_params["high_vol_threshold"] - new_params["low_vol_threshold"]) < 0.005:
            return False  # Thresholds too close
            
        # Check cluster center separation
        centers = sorted(new_params["cluster_centers"])
        if centers[1] - centers[0] < 0.005 or centers[2] - centers[1] < 0.005:
            return False  # Centers too close
            
        # Check transition matrix probabilities sum to ~1
        for regime in new_params["transition_matrix"].values():
            if not (0.99 <= sum(regime.values()) <= 1.01):
                return False
                
        return True

    def learn_from_trade(self, trade: Dict[str, Any]) -> None:
        """
        Optional: Learn from trade execution data.
        This component doesn't need trade data, but the method is required by the interface.
        """
        pass

    def _restore_state(self, state: Dict[str, Any]) -> None:
        """Restore component state from saved data."""
        self.params = state["params"]
        self.current_regime = state.get("current_regime", "medium")
        self.last_updated = datetime.fromisoformat(state["last_updated"])
        if isinstance(self.last_updated, str):
            self.last_updated = datetime.fromisoformat(self.last_updated)