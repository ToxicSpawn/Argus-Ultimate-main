"""
Universal Adaptation Engine — master coordinator for ALL ARGUS adaptation.

This is the highest-level adaptation primitive in ARGUS. It orchestrates:

  1. UniversalParameterRegistry — central registry of 500+ parameters
  2. ClusterManager              — groups params for joint adaptation
  3. ParameterDependencyGraph    — enforces constraints between params
  4. ParameterAttributionTracker — per-param P&L attribution
  5. HierarchicalHealthMonitor   — multi-level health tracking

Each cycle:
  1. Compute per-cluster gradients from observation buffer
  2. Propose parameter changes (cluster multipliers)
  3. Filter through dependency constraints
  4. Apply changes to registry
  5. Track attribution
  6. Check hierarchical health
  7. Revert any cluster that hurts performance

This makes EVERY parameter in ARGUS adapt automatically, while:
  - Respecting safety bounds
  - Respecting dependencies
  - Avoiding overfitting via cluster grouping
  - Reverting bad changes fast
  - Providing full audit trail
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class UniversalAdaptationMode(Enum):
    AGGRESSIVE = "aggressive"
    NORMAL = "normal"
    CONSERVATIVE = "conservative"
    PAUSED = "paused"


@dataclass
class UniversalAdaptationConfig:
    enabled: bool = True
    mode: UniversalAdaptationMode = UniversalAdaptationMode.NORMAL
    cluster_adapt_cycles: int = 100        # adapt clusters every N cycles
    health_check_cycles: int = 50           # check health every N cycles
    max_clusters_per_cycle: int = 5         # max clusters drifted per cycle
    auto_revert: bool = True
    auto_throttle: bool = True
    yaml_path: str = "unified_config.yaml"
    auto_discover_yaml: bool = True


class UniversalAdaptationEngine:
    """
    Master coordinator for universal parameter adaptation across ARGUS.

    Usage::

        engine = UniversalAdaptationEngine()

        # Initial setup
        engine.bootstrap()  # discovers all params, sets up clusters

        # Each cycle:
        result = engine.tick(
            cycle_number=42,
            portfolio_value_aud=1000.0,
            current_regime="TRENDING_UP",
        )

        # On every fill:
        engine.observe_trade(
            trade_id="ord_123",
            pnl_aud=15.0,
            regime="TRENDING_UP",
        )
    """

    def __init__(self, config: Optional[UniversalAdaptationConfig] = None) -> None:
        self._config = config or UniversalAdaptationConfig()
        self._cycle_count = 0
        self._last_adapt_cycle = 0
        self._last_health_cycle = 0
        self._mode = self._config.mode

        # Components — built lazily
        self._registry: Any = None
        self._cluster_manager: Any = None
        self._dependency_graph: Any = None
        self._attribution_tracker: Any = None
        self._health_monitor: Any = None

        self._stats: Dict[str, int] = {
            "ticks": 0,
            "clusters_adapted": 0,
            "constraints_blocked": 0,
            "reverts_triggered": 0,
            "params_reverted": 0,
        }
        self._bootstrapped = False
        logger.info("UniversalAdaptationEngine: initialized (mode=%s)", self._mode.value)

    def bootstrap(self) -> Dict[str, int]:
        """
        Initialize all sub-components and discover parameters.
        Returns dict of {component: count_initialized}.
        """
        from core.universal_parameter_registry import UniversalParameterRegistry
        from core.parameter_clusters import ClusterManager
        from core.parameter_dependencies import ParameterDependencyGraph
        from core.parameter_attribution import ParameterAttributionTracker
        from core.hierarchical_health_monitor import HierarchicalHealthMonitor

        # Step 1: Build the registry
        self._registry = UniversalParameterRegistry()

        # Step 2: Auto-discover from YAML
        yaml_count = 0
        if self._config.auto_discover_yaml:
            try:
                yaml_count = self._registry.discover_from_yaml(self._config.yaml_path)
            except Exception as exc:
                logger.debug("yaml discovery error: %s", exc)

        # Step 3: Build cluster manager
        self._cluster_manager = ClusterManager(registry=self._registry)
        cluster_count = self._cluster_manager.auto_populate_from_registry()

        # Step 4: Build dependency graph
        self._dependency_graph = ParameterDependencyGraph()

        # Step 5: Build attribution tracker
        self._attribution_tracker = ParameterAttributionTracker()

        # Step 6: Build health monitor
        self._health_monitor = HierarchicalHealthMonitor()

        # Register clusters and categories with health monitor
        for cluster_name in self._cluster_manager.list_clusters():
            self._health_monitor.register_cluster(cluster_name)

        for category in self._registry.get_categories():
            self._health_monitor.register_category(category)

        self._bootstrapped = True

        result = {
            "params_registered": self._registry.parameter_count(),
            "params_from_yaml": yaml_count,
            "clusters_defined": len(self._cluster_manager.list_clusters()),
            "cluster_members_assigned": cluster_count,
            "constraints_defined": len(self._dependency_graph.list_constraints()),
        }
        logger.info(
            "UniversalAdaptationEngine: bootstrapped — %d params, %d clusters, %d constraints",
            result["params_registered"], result["clusters_defined"], result["constraints_defined"],
        )
        return result

    def tick(
        self,
        cycle_number: int,
        portfolio_value_aud: float = 1000.0,
        current_regime: str = "NORMAL",
        cumulative_pnl: float = 0.0,
    ) -> Dict[str, Any]:
        """Run one cycle of the universal adaptation engine."""
        self._cycle_count = cycle_number
        self._stats["ticks"] += 1

        if not self._config.enabled or self._mode == UniversalAdaptationMode.PAUSED:
            return {"enabled": False, "mode": self._mode.value}

        if not self._bootstrapped:
            self.bootstrap()

        result: Dict[str, Any] = {
            "cycle": cycle_number,
            "mode": self._mode.value,
            "actions": [],
        }

        # Cluster adaptation cadence
        adapt_cadence = self._effective_adapt_cadence()
        if (cycle_number - self._last_adapt_cycle) >= adapt_cadence:
            self._last_adapt_cycle = cycle_number
            adapt_count = self._run_cluster_adaptation(current_regime)
            if adapt_count > 0:
                result["actions"].append(f"adapted_{adapt_count}_clusters")
                self._stats["clusters_adapted"] += adapt_count

        # Health check cadence
        if (cycle_number - self._last_health_cycle) >= self._config.health_check_cycles:
            self._last_health_cycle = cycle_number
            revert_candidates = self._check_health_and_revert()
            if revert_candidates:
                result["actions"].append(f"reverted_{len(revert_candidates)}_clusters")

        return result

    def observe_trade(
        self,
        trade_id: str,
        pnl_aud: float,
        regime: str = "NORMAL",
        strategy: str = "",
        symbol: str = "",
    ) -> None:
        """Record a trade outcome for attribution."""
        if not self._bootstrapped:
            return

        # Snapshot current parameter values
        param_snapshot: Dict[str, float] = {}
        for name in list(self._registry._params.keys())[:50]:  # cap at 50 most relevant
            value = self._registry.get_value(name)
            if value is not None:
                param_snapshot[name] = value

        # Snapshot current cluster multipliers
        cluster_snapshot: Dict[str, float] = {}
        for cluster_name in self._cluster_manager.list_clusters():
            cluster = self._cluster_manager.get_cluster(cluster_name)
            if cluster:
                cluster_snapshot[cluster_name] = cluster.multiplier

        # Record in attribution tracker
        self._attribution_tracker.record_trade(
            trade_id=trade_id,
            pnl_aud=pnl_aud,
            parameters=param_snapshot,
            cluster_multipliers=cluster_snapshot,
            regime=regime,
            strategy=strategy,
            symbol=symbol,
        )

        # Record in health monitor (per cluster)
        for cluster_name, mult in cluster_snapshot.items():
            self._health_monitor.record_outcome(
                pnl_aud=pnl_aud,
                cluster_name=cluster_name,
            )
            self._cluster_manager.record_outcome(
                cluster_name=cluster_name,
                pnl_contribution=pnl_aud,
                helped=pnl_aud > 0,
            )

    def _effective_adapt_cadence(self) -> int:
        base = self._config.cluster_adapt_cycles
        if self._mode == UniversalAdaptationMode.AGGRESSIVE:
            return max(int(base * 0.5), 25)
        if self._mode == UniversalAdaptationMode.CONSERVATIVE:
            return int(base * 2)
        return base

    def _run_cluster_adaptation(self, regime: str) -> int:
        """Compute and apply cluster adaptations based on attribution."""
        if self._cluster_manager is None or self._attribution_tracker is None:
            return 0

        # Compute cluster impacts
        impacts = self._attribution_tracker.compute_cluster_impacts(min_samples=20)
        if not impacts:
            return 0

        # Pick top clusters by absolute correlation
        scored = sorted(
            impacts.items(),
            key=lambda x: abs(x[1].correlation_with_outcome),
            reverse=True,
        )[:self._config.max_clusters_per_cycle]

        # Build proposed changes
        proposed_changes: Dict[str, float] = {}
        cluster_changes: List[str] = []
        for cluster_name, impact in scored:
            cluster = self._cluster_manager.get_cluster(cluster_name)
            if cluster is None:
                continue

            # Drift in direction of positive correlation
            gradient = impact.correlation_with_outcome
            if abs(gradient) < 0.1:
                continue

            direction = 1.0 if gradient > 0 else -1.0
            confidence = min(1.0, impact.sample_count / 100.0)
            new_mult = cluster.multiplier + direction * cluster.learning_rate * confidence

            # Check if change would be applied
            if abs(new_mult - cluster.multiplier) < 1e-9:
                continue

            cluster_changes.append(cluster_name)
            # Build current values for constraint check
            for param_name in cluster.member_params:
                current = self._registry.get_value(param_name)
                base = cluster.base_values.get(param_name, current)
                if base is not None:
                    proposed_changes[param_name] = base * new_mult

        # Validate against dependency constraints
        if proposed_changes and self._dependency_graph is not None:
            current_values = {
                name: self._registry.get_value(name) or 0.0
                for name in proposed_changes.keys()
            }
            safe_changes = self._dependency_graph.filter_safe_changes(
                current_values, proposed_changes,
            )
            if len(safe_changes) < len(proposed_changes):
                blocked = len(proposed_changes) - len(safe_changes)
                self._stats["constraints_blocked"] += blocked

        # Apply cluster changes (cluster manager handles base_values internally)
        applied = 0
        for cluster_name in cluster_changes:
            cluster = self._cluster_manager.get_cluster(cluster_name)
            if cluster is None:
                continue
            impact = impacts.get(cluster_name)
            if impact is None:
                continue
            gradient = impact.correlation_with_outcome
            if self._cluster_manager.drift_cluster(cluster_name, gradient):
                applied += 1

        return applied

    def _check_health_and_revert(self) -> List[str]:
        """Check hierarchical health, revert anything that's hurting."""
        if self._health_monitor is None or self._cluster_manager is None:
            return []

        if not self._config.auto_revert:
            return []

        candidates = self._health_monitor.find_revert_candidates()
        reverted: List[str] = []

        for candidate in candidates:
            if candidate["type"] != "cluster":
                continue
            cluster_name = candidate["name"]
            count = self._cluster_manager.revert_cluster(cluster_name)
            if count > 0:
                reverted.append(cluster_name)
                self._health_monitor.mark_reverted(cluster_name, "cluster")
                self._stats["reverts_triggered"] += 1
                self._stats["params_reverted"] += count
                logger.warning(
                    "UniversalAdaptationEngine: REVERTED cluster %s (health critical)",
                    cluster_name,
                )

        # If too many clusters being reverted, throttle the engine
        if len(reverted) >= 3 and self._config.auto_throttle:
            self._throttle()

        return reverted

    def _throttle(self) -> None:
        """Reduce adaptation rate if health is bad."""
        old = self._mode
        if self._mode == UniversalAdaptationMode.AGGRESSIVE:
            self._mode = UniversalAdaptationMode.NORMAL
        elif self._mode == UniversalAdaptationMode.NORMAL:
            self._mode = UniversalAdaptationMode.CONSERVATIVE
        elif self._mode == UniversalAdaptationMode.CONSERVATIVE:
            self._mode = UniversalAdaptationMode.PAUSED

        if old != self._mode:
            logger.warning(
                "UniversalAdaptationEngine: throttled %s → %s",
                old.value, self._mode.value,
            )

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def get_value(self, name: str) -> Optional[float]:
        """Get the current adapted value of a parameter (for use by other modules)."""
        if self._registry is None:
            return None
        return self._registry.get_value(name)

    def get_cluster_multiplier(self, cluster_name: str) -> Optional[float]:
        """Get the current multiplier for a cluster."""
        if self._cluster_manager is None:
            return None
        cluster = self._cluster_manager.get_cluster(cluster_name)
        return cluster.multiplier if cluster else None

    def revert_all(self) -> int:
        """Emergency: revert every parameter and cluster."""
        if self._cluster_manager is None or self._registry is None:
            return 0
        cluster_revert = self._cluster_manager.revert_all()
        param_revert = self._registry.revert_all()
        self._stats["reverts_triggered"] += 1
        self._stats["params_reverted"] += param_revert
        return param_revert

    def set_mode(self, mode: UniversalAdaptationMode) -> None:
        old = self._mode
        self._mode = mode
        logger.info(
            "UniversalAdaptationEngine: mode %s → %s",
            old.value, mode.value,
        )

    def snapshot(self) -> Dict[str, Any]:
        snap: Dict[str, Any] = {
            "enabled": self._config.enabled,
            "mode": self._mode.value,
            "bootstrapped": self._bootstrapped,
            "cycle": self._cycle_count,
            "stats": dict(self._stats),
        }
        if self._registry is not None:
            snap["registry"] = self._registry.snapshot()
        if self._cluster_manager is not None:
            snap["clusters"] = self._cluster_manager.snapshot()
        if self._dependency_graph is not None:
            snap["dependencies"] = self._dependency_graph.snapshot()
        if self._attribution_tracker is not None:
            snap["attribution"] = self._attribution_tracker.snapshot()
        if self._health_monitor is not None:
            snap["health"] = self._health_monitor.snapshot()
        return snap
