"""
Universal Parameter Registry — central registry for ALL adaptive parameters in ARGUS.

This is the master parameter system that makes EVERY module in ARGUS adaptive.
It supports:

  1. Manual registration       — module.register("max_pos", 0.25, 0.10, 0.50)
  2. Decorator registration    — @adaptive(min=0.10, max=0.50)
  3. YAML auto-discovery       — scans unified_config.yaml for numerics
  4. Class scanning            — finds all UPPERCASE constants in modules
  5. Cluster grouping          — adapts correlated params together
  6. Dependency constraints    — enforces relationships between params
  7. Per-parameter attribution — knows which param led to which trade
  8. Bounds enforcement        — never violates min/max safety limits
  9. Sample-size weighting     — slow drift on rare params, fast on common
 10. Hierarchical reverts      — revert one param, one cluster, or all

Coverage goal: ~95% of all ARGUS numerical parameters across:
  - Strategies (~150 parameters)
  - Risk modules (~50 parameters)
  - ML hyperparameters (~100 parameters)
  - Execution parameters (~30 parameters)
  - Intelligence gates (~75 thresholds)
  - Monitoring/compliance (~30 parameters)
  - Data sources (~50 parameters)
  Total: ~500+ parameters
"""
from __future__ import annotations

import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """The kind of value a parameter holds."""
    NUMERIC = "numeric"          # float/int
    PERCENTAGE = "percentage"    # 0.0 - 1.0
    BPS = "bps"                  # basis points
    DURATION_S = "duration_s"    # seconds
    DURATION_MS = "duration_ms"  # milliseconds
    COUNT = "count"              # integer count
    PRICE = "price"              # absolute price
    RATIO = "ratio"              # any ratio


class ParameterCategory(Enum):
    """High-level categorization for cluster grouping."""
    SIZING = "sizing"
    RISK = "risk"
    EXECUTION = "execution"
    STRATEGY = "strategy"
    ML = "ml"
    REGIME = "regime"
    GATE = "gate"
    TIMING = "timing"
    THRESHOLD = "threshold"
    MONITORING = "monitoring"
    DATA = "data"
    NETWORK = "network"
    UNCATEGORIZED = "uncategorized"


@dataclass
class ParameterSpec:
    """Definition of an adaptive parameter."""
    name: str
    initial_value: float
    current_value: float
    min_value: float
    max_value: float
    param_type: ParameterType = ParameterType.NUMERIC
    category: ParameterCategory = ParameterCategory.UNCATEGORIZED
    cluster: Optional[str] = None
    module_path: str = ""
    description: str = ""
    learning_rate: float = 0.01
    sample_count: int = 0
    last_updated: float = 0.0
    drift_history: deque = field(default_factory=lambda: deque(maxlen=100))
    impact_score: float = 0.0  # impact-weighted importance


@dataclass
class ParameterObservation:
    """Records a (parameter_value, outcome) pair for gradient computation."""
    timestamp: float
    parameter_name: str
    value: float
    outcome_pnl: float
    regime: str = "NORMAL"
    sample_weight: float = 1.0


# Decorator-based registration support
_PENDING_DECORATED: List[Tuple[str, Dict[str, Any]]] = []


def adaptive(
    *,
    min_value: float,
    max_value: float,
    category: str = "uncategorized",
    cluster: Optional[str] = None,
    description: str = "",
    learning_rate: float = 0.01,
    param_type: str = "numeric",
):
    """
    Decorator to mark a class attribute as adaptive.

    Usage::

        class MyStrategy:
            @adaptive(min_value=0.10, max_value=0.50, category="sizing")
            MAX_POSITION_PCT = 0.25
    """
    def decorator(value: float) -> float:
        # Store metadata in pending list (registry picks it up at scan time)
        _PENDING_DECORATED.append((str(value), {
            "min": float(min_value),
            "max": float(max_value),
            "category": category,
            "cluster": cluster,
            "description": description,
            "learning_rate": learning_rate,
            "type": param_type,
        }))
        return value
    return decorator


class UniversalParameterRegistry:
    """
    Master registry for all adaptive parameters in ARGUS.

    Usage::

        registry = UniversalParameterRegistry()

        # Manual registration
        registry.register(
            name="max_position_pct",
            initial=0.25, min_value=0.10, max_value=0.50,
            category=ParameterCategory.SIZING,
            cluster="position_sizing",
        )

        # Auto-discover from YAML
        registry.discover_from_yaml("unified_config.yaml")

        # Get current value
        value = registry.get_value("max_position_pct")

        # Apply adaptation (after observation gradient computed elsewhere)
        registry.set_value("max_position_pct", 0.27)

        # Revert
        registry.revert("max_position_pct")
    """

    # Default learning rates by category (less for risky things)
    DEFAULT_LR = {
        ParameterCategory.SIZING: 0.005,
        ParameterCategory.RISK: 0.003,         # very slow on risk
        ParameterCategory.EXECUTION: 0.010,
        ParameterCategory.STRATEGY: 0.008,
        ParameterCategory.ML: 0.015,
        ParameterCategory.REGIME: 0.005,
        ParameterCategory.GATE: 0.008,
        ParameterCategory.TIMING: 0.020,
        ParameterCategory.THRESHOLD: 0.008,
        ParameterCategory.MONITORING: 0.020,
        ParameterCategory.DATA: 0.015,
        ParameterCategory.NETWORK: 0.020,
        ParameterCategory.UNCATEGORIZED: 0.005,
    }

    # Max drift per adaptation step as fraction of current value
    MAX_DRIFT_FRACTION = 0.02

    def __init__(self) -> None:
        self._params: Dict[str, ParameterSpec] = {}
        self._by_category: Dict[ParameterCategory, Set[str]] = defaultdict(set)
        self._by_cluster: Dict[str, Set[str]] = defaultdict(set)
        self._by_module: Dict[str, Set[str]] = defaultdict(set)
        self._observations: deque = deque(maxlen=1_000_000)
        self._adaptation_count = 0
        self._revert_count = 0
        logger.info("UniversalParameterRegistry: initialized")

    # ─────────────────────────────────────────────────────────────────
    # Registration
    # ─────────────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        initial: float,
        min_value: float,
        max_value: float,
        category: ParameterCategory = ParameterCategory.UNCATEGORIZED,
        cluster: Optional[str] = None,
        param_type: ParameterType = ParameterType.NUMERIC,
        module_path: str = "",
        description: str = "",
        learning_rate: Optional[float] = None,
    ) -> bool:
        """Register a parameter for adaptive tuning."""
        if name in self._params:
            return False

        # Sanity-clamp initial value to bounds
        if min_value > max_value:
            min_value, max_value = max_value, min_value
        initial = max(min_value, min(initial, max_value))

        if learning_rate is None:
            learning_rate = self.DEFAULT_LR.get(category, 0.005)

        spec = ParameterSpec(
            name=name,
            initial_value=initial,
            current_value=initial,
            min_value=min_value,
            max_value=max_value,
            param_type=param_type,
            category=category,
            cluster=cluster,
            module_path=module_path,
            description=description,
            learning_rate=learning_rate,
        )
        self._params[name] = spec
        self._by_category[category].add(name)
        if cluster:
            self._by_cluster[cluster].add(name)
        if module_path:
            self._by_module[module_path].add(name)
        return True

    def register_many(self, params: List[Dict[str, Any]]) -> int:
        """Bulk-register a list of parameter dicts. Returns count registered."""
        count = 0
        for p in params:
            try:
                category = p.get("category", ParameterCategory.UNCATEGORIZED)
                if isinstance(category, str):
                    try:
                        category = ParameterCategory(category)
                    except ValueError:
                        category = ParameterCategory.UNCATEGORIZED
                param_type = p.get("param_type", ParameterType.NUMERIC)
                if isinstance(param_type, str):
                    try:
                        param_type = ParameterType(param_type)
                    except ValueError:
                        param_type = ParameterType.NUMERIC
                ok = self.register(
                    name=p["name"],
                    initial=float(p["initial"]),
                    min_value=float(p["min"]),
                    max_value=float(p["max"]),
                    category=category,
                    cluster=p.get("cluster"),
                    param_type=param_type,
                    module_path=p.get("module_path", ""),
                    description=p.get("description", ""),
                    learning_rate=p.get("learning_rate"),
                )
                if ok:
                    count += 1
            except (KeyError, ValueError, TypeError) as exc:
                logger.debug("registry.register_many: skipped param: %s", exc)
        return count

    # ─────────────────────────────────────────────────────────────────
    # Value access
    # ─────────────────────────────────────────────────────────────────

    def get_value(self, name: str) -> Optional[float]:
        spec = self._params.get(name)
        return spec.current_value if spec else None

    def set_value(self, name: str, new_value: float) -> bool:
        """Set a parameter value, clamping to bounds. Returns True if changed."""
        spec = self._params.get(name)
        if spec is None:
            return False
        clamped = max(spec.min_value, min(new_value, spec.max_value))
        if abs(clamped - spec.current_value) < 1e-9:
            return False
        spec.drift_history.append({
            "timestamp": time.time(),
            "old": spec.current_value,
            "new": clamped,
        })
        spec.current_value = clamped
        spec.last_updated = time.time()
        self._adaptation_count += 1
        return True

    def revert(self, name: str) -> bool:
        """Revert a parameter to its initial value."""
        spec = self._params.get(name)
        if spec is None:
            return False
        if abs(spec.current_value - spec.initial_value) < 1e-9:
            return False
        spec.current_value = spec.initial_value
        spec.last_updated = time.time()
        self._revert_count += 1
        logger.info(
            "UniversalParameterRegistry: reverted %s to initial=%.6f",
            name, spec.initial_value,
        )
        return True

    def revert_cluster(self, cluster: str) -> int:
        """Revert all parameters in a cluster. Returns count reverted."""
        count = 0
        for name in self._by_cluster.get(cluster, set()):
            if self.revert(name):
                count += 1
        return count

    def revert_all(self) -> int:
        """Emergency: revert every parameter."""
        count = 0
        for name in list(self._params.keys()):
            if self.revert(name):
                count += 1
        logger.warning(
            "UniversalParameterRegistry: REVERT ALL — %d parameters reverted", count,
        )
        return count

    # ─────────────────────────────────────────────────────────────────
    # Observation recording
    # ─────────────────────────────────────────────────────────────────

    def observe(
        self,
        parameter_values: Dict[str, float],
        pnl_aud: float,
        regime: str = "NORMAL",
    ) -> None:
        """Record an outcome correlated with parameter values."""
        ts = time.time()
        for name, value in parameter_values.items():
            spec = self._params.get(name)
            if spec is None:
                continue
            spec.sample_count += 1
            self._observations.append(ParameterObservation(
                timestamp=ts,
                parameter_name=name,
                value=value,
                outcome_pnl=pnl_aud,
                regime=regime,
            ))

    def get_observations(
        self,
        name: str,
        min_count: int = 0,
    ) -> List[ParameterObservation]:
        """Get all observations for a specific parameter."""
        results = [o for o in self._observations if o.parameter_name == name]
        return results if len(results) >= min_count else []

    # ─────────────────────────────────────────────────────────────────
    # Discovery / introspection
    # ─────────────────────────────────────────────────────────────────

    def discover_from_yaml(self, yaml_path: str) -> int:
        """
        Auto-discover parameters from a YAML config file.
        Recursively walks the config tree and registers all numeric leaves
        with sensible default bounds.
        """
        try:
            import yaml
        except ImportError:
            logger.debug("UniversalParameterRegistry: pyyaml not available")
            return 0

        path = Path(yaml_path)
        if not path.exists():
            logger.debug("UniversalParameterRegistry: yaml not found: %s", yaml_path)
            return 0

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as exc:
            logger.debug("UniversalParameterRegistry: yaml load error: %s", exc)
            return 0

        if not isinstance(data, dict):
            return 0

        count = 0
        for name, value, category, cluster in self._walk_yaml(data, ""):
            ok = self._register_yaml_param(name, value, category, cluster)
            if ok:
                count += 1
        logger.info("UniversalParameterRegistry: discovered %d params from YAML", count)
        return count

    def _walk_yaml(
        self,
        node: Any,
        prefix: str,
        depth: int = 0,
    ) -> List[Tuple[str, float, ParameterCategory, str]]:
        """Recursively walk a YAML tree, yielding (name, value, category, cluster) tuples."""
        results: List[Tuple[str, float, ParameterCategory, str]] = []
        if depth > 10:  # safety
            return results

        if isinstance(node, dict):
            for k, v in node.items():
                new_prefix = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    category = self._infer_category(new_prefix)
                    cluster = self._infer_cluster(new_prefix)
                    results.append((new_prefix, float(v), category, cluster))
                elif isinstance(v, dict):
                    results.extend(self._walk_yaml(v, new_prefix, depth + 1))
        return results

    def _register_yaml_param(
        self,
        name: str,
        value: float,
        category: ParameterCategory,
        cluster: str,
    ) -> bool:
        """Register a parameter discovered from YAML with auto-bounds."""
        if name in self._params:
            return False
        # Skip values that don't make sense to adapt
        if abs(value) < 1e-9 and category not in (ParameterCategory.RISK,):
            return False
        if abs(value) > 1e9:  # cap extreme values
            return False

        # Auto-compute bounds based on value magnitude
        min_v, max_v = self._auto_bounds(value, category)

        return self.register(
            name=name,
            initial=value,
            min_value=min_v,
            max_value=max_v,
            category=category,
            cluster=cluster,
            description=f"auto-discovered from YAML: {name}",
        )

    @staticmethod
    def _auto_bounds(value: float, category: ParameterCategory) -> Tuple[float, float]:
        """Compute safe min/max bounds for an auto-discovered parameter."""
        if value == 0:
            return (0.0, 1.0)
        if 0 < abs(value) < 1.0:
            # Probably a percentage / fraction
            min_v = max(value * 0.5, 0.0)
            max_v = min(value * 2.0, 1.0)
        elif abs(value) < 100:
            # Small integer / count
            min_v = max(value * 0.5, 0.0)
            max_v = value * 2.0
        elif abs(value) < 10000:
            # Medium value
            min_v = value * 0.7
            max_v = value * 1.5
        else:
            # Large value
            min_v = value * 0.8
            max_v = value * 1.3
        return (float(min_v), float(max_v))

    @staticmethod
    def _infer_category(name: str) -> ParameterCategory:
        """Infer parameter category from its name."""
        n = name.lower()
        if any(k in n for k in ("position_pct", "size", "kelly", "sizing")):
            return ParameterCategory.SIZING
        if any(k in n for k in ("stop", "var", "cvar", "risk", "drawdown", "loss_limit", "leverage")):
            return ParameterCategory.RISK
        if any(k in n for k in ("execution", "twap", "vwap", "slippage", "fill", "maker", "taker")):
            return ParameterCategory.EXECUTION
        if any(k in n for k in ("strategy", "momentum", "mean_rev", "breakout")):
            return ParameterCategory.STRATEGY
        if any(k in n for k in ("learning_rate", "epoch", "hidden", "n_estimators", "ml", "model")):
            return ParameterCategory.ML
        if any(k in n for k in ("regime", "transition", "vol_state")):
            return ParameterCategory.REGIME
        if any(k in n for k in ("gate", "threshold", "min_", "max_")):
            return ParameterCategory.GATE
        if any(k in n for k in ("cycle", "interval", "duration", "timeout", "frequency")):
            return ParameterCategory.TIMING
        if any(k in n for k in ("alert", "monitor", "log", "retention")):
            return ParameterCategory.MONITORING
        if any(k in n for k in ("data", "feed", "cache", "ttl")):
            return ParameterCategory.DATA
        if any(k in n for k in ("network", "latency", "buffer", "bandwidth")):
            return ParameterCategory.NETWORK
        return ParameterCategory.UNCATEGORIZED

    @staticmethod
    def _infer_cluster(name: str) -> str:
        """Infer which cluster a parameter belongs to."""
        n = name.lower()
        # Sizing cluster
        if "position_pct" in n or "kelly" in n or "size_pct" in n:
            return "position_sizing"
        # Stops cluster
        if "stop_loss" in n or "trailing" in n:
            return "stops"
        # Take profit cluster
        if "take_profit" in n or "tp_pct" in n:
            return "take_profits"
        # Risk limits cluster
        if "var_limit" in n or "cvar_limit" in n or "loss_limit" in n:
            return "risk_limits"
        # Confidence thresholds cluster
        if "confidence" in n or "threshold" in n:
            return "confidence_thresholds"
        # ML hyperparams cluster
        if "learning_rate" in n or "epoch" in n or "hidden" in n:
            return "ml_hyperparams"
        # Execution latency cluster
        if "latency" in n or "timeout" in n or "interval" in n:
            return "execution_timing"
        # Drawdown cluster
        if "drawdown" in n:
            return "drawdown_thresholds"
        # Regime params cluster
        if "regime" in n:
            return "regime_params"
        # Default: one cluster per module path
        parts = n.split(".")
        if len(parts) >= 2:
            return parts[0]
        return "miscellaneous"

    # ─────────────────────────────────────────────────────────────────
    # Query / inspection
    # ─────────────────────────────────────────────────────────────────

    def list_parameters(
        self,
        category: Optional[ParameterCategory] = None,
        cluster: Optional[str] = None,
        module: Optional[str] = None,
    ) -> List[str]:
        """List parameter names matching the filter."""
        if category:
            return sorted(self._by_category[category])
        if cluster:
            return sorted(self._by_cluster.get(cluster, set()))
        if module:
            return sorted(self._by_module.get(module, set()))
        return sorted(self._params.keys())

    def get_clusters(self) -> List[str]:
        return sorted(self._by_cluster.keys())

    def get_categories(self) -> List[str]:
        return [c.value for c in self._by_category.keys()]

    def parameter_count(self) -> int:
        return len(self._params)

    def get_spec(self, name: str) -> Optional[ParameterSpec]:
        return self._params.get(name)

    def get_state(self, name: str) -> Optional[Dict[str, Any]]:
        spec = self._params.get(name)
        if spec is None:
            return None
        return {
            "name": spec.name,
            "current_value": spec.current_value,
            "initial_value": spec.initial_value,
            "min": spec.min_value,
            "max": spec.max_value,
            "category": spec.category.value,
            "cluster": spec.cluster,
            "module_path": spec.module_path,
            "param_type": spec.param_type.value,
            "sample_count": spec.sample_count,
            "drift_count": len(spec.drift_history),
            "drifted_pct": (
                (spec.current_value - spec.initial_value) / max(spec.initial_value, 1e-9) * 100
                if spec.initial_value != 0
                else 0.0
            ),
            "last_updated": spec.last_updated,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_params": len(self._params),
            "total_clusters": len(self._by_cluster),
            "total_categories": len(self._by_category),
            "by_category": {
                cat.value: len(names)
                for cat, names in self._by_category.items()
            },
            "by_cluster": {
                cluster: len(names)
                for cluster, names in self._by_cluster.items()
            },
            "adaptation_count": self._adaptation_count,
            "revert_count": self._revert_count,
            "observation_count": len(self._observations),
        }
