"""
core/component_registry_base.py
================================
ComponentRegistryBase — the thread-safe base extracted from the
6 591-line component_registry.py god-object (H02, Phase 1).

This module provides:
- A generic ``register`` / ``get`` / ``unregister`` lifecycle.
- Dependency-aware startup ordering via a simple topological sort.
- Component health reporting.
- Hook points (``on_register``, ``on_unregister``) for subclasses.

The full ComponentRegistry in component_registry.py should subclass this
base in a subsequent batch once the remaining concerns have been split out.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger("argus.core.component_registry")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass
class ComponentMeta:
    """Runtime metadata for a registered component."""
    name: str
    instance: Any
    depends_on: List[str] = field(default_factory=list)
    healthy: bool = True
    start_order: int = 0
    tags: Dict[str, str] = field(default_factory=dict)

    def health_check(self) -> bool:
        """Call component's ``is_healthy()`` if available, else return stored flag."""
        fn: Optional[Callable[[], bool]] = getattr(self.instance, "is_healthy", None)
        if callable(fn):
            try:
                self.healthy = bool(fn())
            except Exception:
                logger.exception("Health check raised for component '%s'", self.name)
                self.healthy = False
        return self.healthy


# ---------------------------------------------------------------------------
# Base registry
# ---------------------------------------------------------------------------

class ComponentRegistryBase:
    """
    Thread-safe component lifecycle manager.

    Usage
    -----
    registry = ComponentRegistryBase()
    registry.register("risk", risk_manager, depends_on=[])
    registry.register("engine", execution_engine, depends_on=["risk"])

    for name in registry.startup_order():
        component = registry.get(name)
        ...
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._components: Dict[str, ComponentMeta] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        instance: Any,
        *,
        depends_on: Optional[List[str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Register a component.  Raises ``ValueError`` on duplicate names."""
        with self._lock:
            if name in self._components:
                raise ValueError(f"Component '{name}' is already registered.")
            meta = ComponentMeta(
                name=name,
                instance=instance,
                depends_on=list(depends_on or []),
                tags=dict(tags or {}),
            )
            self._components[name] = meta
            self.on_register(meta)
            logger.debug("Registered component '%s'", name)

    def unregister(self, name: str) -> None:
        """Remove a component.  Silent no-op if not found."""
        with self._lock:
            meta = self._components.pop(name, None)
            if meta is not None:
                self.on_unregister(meta)
                logger.debug("Unregistered component '%s'", name)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, name: str) -> Any:
        """Return the component instance.  Raises ``KeyError`` if absent."""
        with self._lock:
            return self._components[name].instance

    def get_optional(self, name: str) -> Optional[Any]:
        """Return the component instance or *None* if absent."""
        with self._lock:
            meta = self._components.get(name)
            return meta.instance if meta is not None else None

    def names(self) -> List[str]:
        """Snapshot of registered component names."""
        with self._lock:
            return list(self._components.keys())

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._components

    def __len__(self) -> int:
        with self._lock:
            return len(self._components)

    # ------------------------------------------------------------------
    # Startup ordering
    # ------------------------------------------------------------------

    def startup_order(self) -> List[str]:
        """
        Return component names in dependency-safe startup order.

        Uses Kahn's algorithm (BFS topological sort).
        Raises ``RuntimeError`` if a dependency cycle is detected.
        """
        with self._lock:
            names = list(self._components.keys())
            deps: Dict[str, List[str]] = {
                n: list(self._components[n].depends_on) for n in names
            }

        # Validate all deps exist
        all_names = set(names)
        for name, d_list in deps.items():
            for dep in d_list:
                if dep not in all_names:
                    raise RuntimeError(
                        f"Component '{name}' depends on unknown component '{dep}'"
                    )

        # Kahn's algorithm
        in_degree: Dict[str, int] = {n: 0 for n in names}
        dependants: Dict[str, List[str]] = {n: [] for n in names}
        for name, d_list in deps.items():
            for dep in d_list:
                in_degree[name] += 1
                dependants[dep].append(name)

        queue: List[str] = [n for n in names if in_degree[n] == 0]
        order: List[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for dependent in dependants[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(names):
            cycle_nodes = [n for n in names if n not in order]
            raise RuntimeError(
                f"Dependency cycle detected among components: {cycle_nodes}"
            )
        return order

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_report(self) -> Dict[str, bool]:
        """Run health checks for all components; return name → healthy mapping."""
        report: Dict[str, bool] = {}
        with self._lock:
            snapshot = list(self._components.values())
        for meta in snapshot:
            report[meta.name] = meta.health_check()
        return report

    @property
    def all_healthy(self) -> bool:
        """True only if every component reports healthy."""
        return all(self.health_report().values())

    # ------------------------------------------------------------------
    # Hook points for subclasses
    # ------------------------------------------------------------------

    def on_register(self, meta: ComponentMeta) -> None:  # noqa: B027
        """Called after a component is registered.  Override in subclasses."""

    def on_unregister(self, meta: ComponentMeta) -> None:  # noqa: B027
        """Called after a component is unregistered.  Override in subclasses."""

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        with self._lock:
            return f"{self.__class__.__name__}(components={list(self._components.keys())})"
