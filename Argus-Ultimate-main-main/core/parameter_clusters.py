"""
Parameter Clusters — group related parameters for joint adaptation.

With 500+ parameters in ARGUS, you can't adapt each one individually
(not enough samples). Instead, parameters with correlated outcomes are
grouped into CLUSTERS that adapt together as a unit.

Each cluster:
  - Has a single multiplier (e.g. 0.95 = "tighten everything by 5%")
  - The multiplier scales all member parameters proportionally
  - All members revert together if the cluster hurts performance
  - Health is tracked per-cluster, not per-parameter

This reduces 500 individual params to ~30 cluster-tunables that CAN be
adapted with normal trade volume.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ClusterDefinition:
    """A logical group of parameters that adapt together."""
    name: str
    description: str
    multiplier: float = 1.0          # current multiplier (1.0 = baseline)
    min_multiplier: float = 0.5      # how much to tighten/loosen at most
    max_multiplier: float = 1.5
    learning_rate: float = 0.005     # max change per cycle
    member_params: List[str] = field(default_factory=list)
    base_values: Dict[str, float] = field(default_factory=dict)
    drift_history: deque = field(default_factory=lambda: deque(maxlen=100))
    sample_count: int = 0
    last_pnl_contribution: float = 0.0
    helped_count: int = 0
    hurt_count: int = 0


class ClusterManager:
    """
    Manages parameter clusters and their joint adaptation.

    Usage::

        mgr = ClusterManager(registry=universal_registry)

        # Define a cluster
        mgr.define_cluster(
            "position_sizing",
            description="All sizing-related parameters",
            members=["max_position_pct", "kelly_fraction", "min_size_aud"],
        )

        # Apply a multiplier to the entire cluster
        mgr.apply_multiplier("position_sizing", 0.95)  # tighten 5%

        # Revert
        mgr.revert_cluster("position_sizing")
    """

    DEFAULT_CLUSTERS = {
        "position_sizing": {
            "description": "All position sizing parameters",
            "min_mult": 0.5,
            "max_mult": 1.5,
            "lr": 0.005,
        },
        "stops": {
            "description": "Stop loss parameters",
            "min_mult": 0.5,
            "max_mult": 2.0,
            "lr": 0.005,
        },
        "take_profits": {
            "description": "Take profit parameters",
            "min_mult": 0.5,
            "max_mult": 2.0,
            "lr": 0.008,
        },
        "risk_limits": {
            "description": "VaR/CVaR/loss limit parameters",
            "min_mult": 0.7,         # never loosen risk limits much
            "max_mult": 1.2,
            "lr": 0.003,             # very slow on risk
        },
        "confidence_thresholds": {
            "description": "Signal confidence gate thresholds",
            "min_mult": 0.8,
            "max_mult": 1.3,
            "lr": 0.008,
        },
        "ml_hyperparams": {
            "description": "ML model hyperparameters",
            "min_mult": 0.5,
            "max_mult": 2.0,
            "lr": 0.015,
        },
        "execution_timing": {
            "description": "Execution timing/latency parameters",
            "min_mult": 0.5,
            "max_mult": 2.0,
            "lr": 0.020,
        },
        "drawdown_thresholds": {
            "description": "Drawdown gate thresholds",
            "min_mult": 0.7,
            "max_mult": 1.3,
            "lr": 0.005,
        },
        "regime_params": {
            "description": "Regime detection parameters",
            "min_mult": 0.7,
            "max_mult": 1.5,
            "lr": 0.005,
        },
        "feature_thresholds": {
            "description": "Feature significance thresholds",
            "min_mult": 0.5,
            "max_mult": 2.0,
            "lr": 0.010,
        },
    }

    def __init__(self, registry: Any = None) -> None:
        self._registry = registry
        self._clusters: Dict[str, ClusterDefinition] = {}
        self._cycle_count = 0
        self._adaptation_count = 0
        self._revert_count = 0
        self._initialize_default_clusters()
        logger.info("ClusterManager: initialized with %d default clusters", len(self._clusters))

    def _initialize_default_clusters(self) -> None:
        """Set up the canonical ARGUS clusters."""
        for name, cfg in self.DEFAULT_CLUSTERS.items():
            self._clusters[name] = ClusterDefinition(
                name=name,
                description=str(cfg.get("description", "")),
                min_multiplier=float(cfg.get("min_mult", 0.5)),
                max_multiplier=float(cfg.get("max_mult", 1.5)),
                learning_rate=float(cfg.get("lr", 0.005)),
            )

    def define_cluster(
        self,
        name: str,
        description: str = "",
        members: Optional[List[str]] = None,
        min_multiplier: float = 0.5,
        max_multiplier: float = 1.5,
        learning_rate: float = 0.005,
    ) -> bool:
        """Define a new cluster (or replace an existing one)."""
        if name in self._clusters and members is None:
            return False
        cluster = ClusterDefinition(
            name=name,
            description=description,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
            learning_rate=learning_rate,
            member_params=list(members or []),
        )
        # Snapshot base values
        if self._registry is not None and members:
            for param_name in members:
                value = self._registry.get_value(param_name)
                if value is not None:
                    cluster.base_values[param_name] = value
        self._clusters[name] = cluster
        return True

    def add_member(self, cluster_name: str, param_name: str) -> bool:
        """Add a parameter to an existing cluster."""
        cluster = self._clusters.get(cluster_name)
        if cluster is None:
            return False
        if param_name in cluster.member_params:
            return False
        cluster.member_params.append(param_name)
        if self._registry is not None:
            value = self._registry.get_value(param_name)
            if value is not None:
                cluster.base_values[param_name] = value
        return True

    def auto_populate_from_registry(self) -> int:
        """
        Auto-populate clusters by reading registry's cluster assignments.
        Returns total members added.
        """
        if self._registry is None:
            return 0

        added = 0
        for cluster_name in self._clusters.keys():
            param_names = self._registry.list_parameters(cluster=cluster_name)
            for param_name in param_names:
                if self.add_member(cluster_name, param_name):
                    added += 1
        return added

    def apply_multiplier(self, cluster_name: str, multiplier: float) -> int:
        """
        Apply a multiplier to all parameters in a cluster.
        Returns count of parameters updated.
        """
        cluster = self._clusters.get(cluster_name)
        if cluster is None or self._registry is None:
            return 0

        # Clamp multiplier to bounds
        multiplier = max(cluster.min_multiplier, min(multiplier, cluster.max_multiplier))

        if abs(multiplier - cluster.multiplier) < 1e-9:
            return 0

        old_mult = cluster.multiplier
        cluster.multiplier = multiplier
        cluster.drift_history.append({
            "timestamp": time.time(),
            "old_mult": old_mult,
            "new_mult": multiplier,
        })

        count = 0
        for param_name in cluster.member_params:
            base = cluster.base_values.get(param_name)
            if base is None:
                base = self._registry.get_value(param_name)
                if base is None:
                    continue
                cluster.base_values[param_name] = base
            new_value = base * multiplier
            if self._registry.set_value(param_name, new_value):
                count += 1

        self._adaptation_count += 1
        logger.debug(
            "ClusterManager: applied multiplier %.4f to %s (%d params updated)",
            multiplier, cluster_name, count,
        )
        return count

    def drift_cluster(
        self,
        cluster_name: str,
        gradient: float,
    ) -> bool:
        """
        Drift a cluster's multiplier in the gradient direction.
        gradient > 0 → increase multiplier (loosen)
        gradient < 0 → decrease multiplier (tighten)
        """
        cluster = self._clusters.get(cluster_name)
        if cluster is None:
            return False
        if abs(gradient) < 1e-9:
            return False

        # Compute drift step
        direction = 1.0 if gradient > 0 else -1.0
        max_step = cluster.learning_rate
        # Confidence weighting based on sample count
        confidence = min(1.0, cluster.sample_count / 100.0)
        drift = direction * max_step * confidence

        new_mult = cluster.multiplier + drift
        return self.apply_multiplier(cluster_name, new_mult) > 0

    def revert_cluster(self, cluster_name: str) -> int:
        """Revert a cluster to its baseline (multiplier=1.0)."""
        return self.apply_multiplier(cluster_name, 1.0)

    def revert_all(self) -> int:
        """Emergency: revert all clusters."""
        total = 0
        for cluster_name in self._clusters.keys():
            total += self.revert_cluster(cluster_name)
        self._revert_count += 1
        logger.warning("ClusterManager: REVERT ALL — %d params reset", total)
        return total

    def record_outcome(
        self,
        cluster_name: str,
        pnl_contribution: float,
        helped: bool,
    ) -> None:
        """Record the outcome of trades using this cluster's current settings."""
        cluster = self._clusters.get(cluster_name)
        if cluster is None:
            return
        cluster.sample_count += 1
        cluster.last_pnl_contribution = pnl_contribution
        if helped:
            cluster.helped_count += 1
        else:
            cluster.hurt_count += 1

    def get_cluster(self, name: str) -> Optional[ClusterDefinition]:
        return self._clusters.get(name)

    def list_clusters(self) -> List[str]:
        return sorted(self._clusters.keys())

    def get_cluster_state(self, name: str) -> Optional[Dict[str, Any]]:
        cluster = self._clusters.get(name)
        if cluster is None:
            return None
        total_outcomes = cluster.helped_count + cluster.hurt_count
        return {
            "name": cluster.name,
            "description": cluster.description,
            "multiplier": cluster.multiplier,
            "min_multiplier": cluster.min_multiplier,
            "max_multiplier": cluster.max_multiplier,
            "learning_rate": cluster.learning_rate,
            "member_count": len(cluster.member_params),
            "members": list(cluster.member_params),
            "sample_count": cluster.sample_count,
            "helped_count": cluster.helped_count,
            "hurt_count": cluster.hurt_count,
            "effectiveness": (
                cluster.helped_count / total_outcomes if total_outcomes else 0.5
            ),
            "drift_count": len(cluster.drift_history),
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_clusters": len(self._clusters),
            "total_members": sum(len(c.member_params) for c in self._clusters.values()),
            "adaptation_count": self._adaptation_count,
            "revert_count": self._revert_count,
            "clusters": {
                name: {
                    "multiplier": c.multiplier,
                    "members": len(c.member_params),
                    "samples": c.sample_count,
                }
                for name, c in self._clusters.items()
            },
        }
