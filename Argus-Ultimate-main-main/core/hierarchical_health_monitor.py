"""
Hierarchical Health Monitor — track health at multiple levels.

When you have 500+ adaptive parameters, "is the system healthy?" becomes
a multi-layered question:

  - Per-parameter health    — is THIS specific param helping or hurting?
  - Per-cluster health      — is THIS group of params working as a unit?
  - Per-module health       — is THIS module's parameter set viable?
  - Per-category health     — are all sizing params OK collectively?
  - Overall system health   — is the ensemble functioning?

This monitor tracks all 5 levels simultaneously and triggers reverts at
the appropriate level when something goes wrong.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class HealthLevel(Enum):
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"            # 70-89
    OK = "ok"                # 50-69
    POOR = "poor"            # 30-49
    CRITICAL = "critical"    # 0-29


@dataclass
class HealthSnapshot:
    """Health state at one point in time."""
    timestamp: float
    score: float                    # 0.0 - 100.0
    level: HealthLevel
    sample_count: int
    pnl_window_aud: float
    win_rate_window: float
    drawdown_pct: float
    notes: str = ""


@dataclass
class HealthEntity:
    """Health tracking for one entity (parameter / cluster / module)."""
    name: str
    entity_type: str               # "parameter" / "cluster" / "module" / "category"
    samples: deque = field(default_factory=lambda: deque(maxlen=200))
    snapshots: deque = field(default_factory=lambda: deque(maxlen=100))
    consecutive_bad_cycles: int = 0
    consecutive_good_cycles: int = 0
    total_helped: int = 0
    total_hurt: int = 0
    last_revert: float = 0.0
    revert_count: int = 0


class HierarchicalHealthMonitor:
    """
    Multi-level health tracking for the universal adaptation system.

    Usage::

        monitor = HierarchicalHealthMonitor()

        # Register entities (auto-discovered from registry)
        monitor.register_parameter("max_position_pct")
        monitor.register_cluster("position_sizing")
        monitor.register_module("strategies.momentum")

        # On every fill:
        monitor.record_outcome(
            parameter_name="max_position_pct",
            cluster_name="position_sizing",
            module_path="strategies.momentum",
            pnl_aud=15.0,
            is_win=True,
        )

        # Periodic check:
        unhealthy = monitor.find_unhealthy()
        for entity in unhealthy:
            if entity.level == HealthLevel.CRITICAL:
                # Trigger revert
                ...
    """

    HURT_STREAK_REVERT = 5  # revert after this many consecutive bad cycles
    SAMPLE_WINDOW_HOURS = 24

    def __init__(self) -> None:
        self._parameters: Dict[str, HealthEntity] = {}
        self._clusters: Dict[str, HealthEntity] = {}
        self._modules: Dict[str, HealthEntity] = {}
        self._categories: Dict[str, HealthEntity] = {}
        self._system_score: float = 100.0
        self._cycle_count = 0
        logger.info("HierarchicalHealthMonitor: initialized")

    # ─────────────────────────────────────────────────────────────────
    # Registration
    # ─────────────────────────────────────────────────────────────────

    def register_parameter(self, name: str) -> None:
        if name not in self._parameters:
            self._parameters[name] = HealthEntity(name=name, entity_type="parameter")

    def register_cluster(self, name: str) -> None:
        if name not in self._clusters:
            self._clusters[name] = HealthEntity(name=name, entity_type="cluster")

    def register_module(self, name: str) -> None:
        if name not in self._modules:
            self._modules[name] = HealthEntity(name=name, entity_type="module")

    def register_category(self, name: str) -> None:
        if name not in self._categories:
            self._categories[name] = HealthEntity(name=name, entity_type="category")

    # ─────────────────────────────────────────────────────────────────
    # Outcome recording
    # ─────────────────────────────────────────────────────────────────

    def record_outcome(
        self,
        pnl_aud: float,
        parameter_name: Optional[str] = None,
        cluster_name: Optional[str] = None,
        module_path: Optional[str] = None,
        category: Optional[str] = None,
    ) -> None:
        """Record an outcome attributable to one or more entities."""
        is_win = pnl_aud > 0

        for entity, lookup in [
            (parameter_name, self._parameters),
            (cluster_name, self._clusters),
            (module_path, self._modules),
            (category, self._categories),
        ]:
            if entity is None:
                continue
            if entity not in lookup:
                continue
            ent = lookup[entity]
            ent.samples.append({
                "timestamp": time.time(),
                "pnl": pnl_aud,
                "is_win": is_win,
            })
            if is_win:
                ent.total_helped += 1
            else:
                ent.total_hurt += 1

        # Update system-level
        self._update_system_score()

    def _update_system_score(self) -> None:
        """Compute overall system health score."""
        # Aggregate over all clusters
        total_helped = sum(c.total_helped for c in self._clusters.values())
        total_hurt = sum(c.total_hurt for c in self._clusters.values())
        total = total_helped + total_hurt
        if total == 0:
            self._system_score = 100.0
        else:
            self._system_score = (total_helped / total) * 100.0

    # ─────────────────────────────────────────────────────────────────
    # Health computation
    # ─────────────────────────────────────────────────────────────────

    def compute_health(self, entity_name: str, entity_type: str) -> Optional[HealthSnapshot]:
        """Compute health for one entity. Returns None if not enough samples."""
        lookup_map = {
            "parameter": self._parameters,
            "cluster": self._clusters,
            "module": self._modules,
            "category": self._categories,
        }
        lookup = lookup_map.get(entity_type)
        if lookup is None:
            return None

        ent = lookup.get(entity_name)
        if ent is None:
            return None

        samples = list(ent.samples)
        if len(samples) < 5:
            return None

        # Filter to last N hours
        cutoff = time.time() - (self.SAMPLE_WINDOW_HOURS * 3600)
        recent = [s for s in samples if s["timestamp"] >= cutoff]
        if len(recent) < 5:
            recent = samples[-20:]  # fall back to last 20 samples

        n = len(recent)
        wins = sum(1 for s in recent if s["is_win"])
        win_rate = wins / n
        cumulative_pnl = sum(s["pnl"] for s in recent)

        # Compute drawdown from running max
        running_max = 0.0
        running_pnl = 0.0
        max_dd = 0.0
        for s in recent:
            running_pnl += s["pnl"]
            if running_pnl > running_max:
                running_max = running_pnl
            dd = running_max - running_pnl
            if dd > max_dd:
                max_dd = dd

        drawdown_pct = (max_dd / max(abs(running_max), 1.0)) * 100

        # Health score: weighted combo
        # 50% win rate, 30% PnL sign, 20% drawdown
        win_score = win_rate * 100
        pnl_score = 100 if cumulative_pnl > 0 else (50 if cumulative_pnl == 0 else 0)
        dd_score = max(0, 100 - drawdown_pct * 2)
        score = 0.5 * win_score + 0.3 * pnl_score + 0.2 * dd_score

        if score >= 90:
            level = HealthLevel.EXCELLENT
        elif score >= 70:
            level = HealthLevel.GOOD
        elif score >= 50:
            level = HealthLevel.OK
        elif score >= 30:
            level = HealthLevel.POOR
        else:
            level = HealthLevel.CRITICAL

        snap = HealthSnapshot(
            timestamp=time.time(),
            score=score,
            level=level,
            sample_count=n,
            pnl_window_aud=cumulative_pnl,
            win_rate_window=win_rate,
            drawdown_pct=drawdown_pct,
        )
        ent.snapshots.append(snap)

        # Update streak counters
        if level in (HealthLevel.POOR, HealthLevel.CRITICAL):
            ent.consecutive_bad_cycles += 1
            ent.consecutive_good_cycles = 0
        else:
            ent.consecutive_good_cycles += 1
            ent.consecutive_bad_cycles = 0

        return snap

    def find_unhealthy(
        self,
        threshold: HealthLevel = HealthLevel.POOR,
    ) -> List[Dict[str, Any]]:
        """Find all entities currently at or below the threshold."""
        threshold_score = {
            HealthLevel.EXCELLENT: 90,
            HealthLevel.GOOD: 70,
            HealthLevel.OK: 50,
            HealthLevel.POOR: 30,
            HealthLevel.CRITICAL: 0,
        }[threshold]

        unhealthy: List[Dict[str, Any]] = []
        for lookup, ent_type in [
            (self._parameters, "parameter"),
            (self._clusters, "cluster"),
            (self._modules, "module"),
            (self._categories, "category"),
        ]:
            for name in lookup.keys():
                snap = self.compute_health(name, ent_type)
                if snap is None:
                    continue
                if snap.score < threshold_score:
                    unhealthy.append({
                        "name": name,
                        "type": ent_type,
                        "score": snap.score,
                        "level": snap.level.value,
                        "consecutive_bad": lookup[name].consecutive_bad_cycles,
                        "should_revert": (
                            lookup[name].consecutive_bad_cycles >= self.HURT_STREAK_REVERT
                        ),
                    })

        return unhealthy

    def find_revert_candidates(self) -> List[Dict[str, Any]]:
        """Find entities that should be reverted (5+ consecutive bad cycles)."""
        unhealthy = self.find_unhealthy(threshold=HealthLevel.POOR)
        return [u for u in unhealthy if u["should_revert"]]

    def mark_reverted(self, entity_name: str, entity_type: str) -> bool:
        """Reset hurt streak after a revert."""
        lookup_map = {
            "parameter": self._parameters,
            "cluster": self._clusters,
            "module": self._modules,
            "category": self._categories,
        }
        lookup = lookup_map.get(entity_type)
        if lookup is None:
            return False
        ent = lookup.get(entity_name)
        if ent is None:
            return False
        ent.last_revert = time.time()
        ent.revert_count += 1
        ent.consecutive_bad_cycles = 0
        return True

    def get_system_score(self) -> float:
        return self._system_score

    def get_system_level(self) -> HealthLevel:
        s = self._system_score
        if s >= 90:
            return HealthLevel.EXCELLENT
        if s >= 70:
            return HealthLevel.GOOD
        if s >= 50:
            return HealthLevel.OK
        if s >= 30:
            return HealthLevel.POOR
        return HealthLevel.CRITICAL

    def snapshot(self) -> Dict[str, Any]:
        return {
            "system_score": round(self._system_score, 1),
            "system_level": self.get_system_level().value,
            "parameters_tracked": len(self._parameters),
            "clusters_tracked": len(self._clusters),
            "modules_tracked": len(self._modules),
            "categories_tracked": len(self._categories),
            "unhealthy_count": len(self.find_unhealthy()),
            "revert_candidates": len(self.find_revert_candidates()),
        }
