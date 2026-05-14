"""
Tool registry — a catalog of named callable tools with JSON schemas.

Each tool is a ``ToolSpec`` with:
  - name
  - description (human-readable)
  - parameters schema (JSON schema-ish dict)
  - handler (Callable)
  - tags (for filtering by category)

Usage
-----
>>> registry = ToolRegistry()
>>> registry.register(
...     name="rag.retrieve_similar",
...     description="Retrieve top-k historical trades similar to context",
...     parameters={"context": "dict", "k": "int (default 5)"},
...     handler=lambda context, k=5: trade_rag.retrieve_similar(context, k),
...     tags=["rag", "memory"],
... )
>>> result = registry.call("rag.retrieve_similar", context={...}, k=3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# ToolSpec
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class ToolSpec:
    """A single registered tool."""

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]
    tags: List[str] = field(default_factory=list)
    returns: str = "Any"

    def to_json_schema(self) -> Dict[str, Any]:
        """Return an OpenAI/Anthropic-compatible JSON schema for this tool."""
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for param_name, param_type in self.parameters.items():
            prop_schema = {"type": _map_type(param_type), "description": str(param_type)}
            properties[param_name] = prop_schema
            if "default" not in str(param_type).lower() and "optional" not in str(param_type).lower():
                required.append(param_name)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


def _map_type(type_hint: Any) -> str:
    """Map a Python type description to a JSON schema type string."""
    s = str(type_hint).lower()
    if "int" in s:
        return "integer"
    if "float" in s or "number" in s:
        return "number"
    if "bool" in s:
        return "boolean"
    if "list" in s or "array" in s:
        return "array"
    if "dict" in s or "object" in s:
        return "object"
    return "string"


# ═════════════════════════════════════════════════════════════════════════════
# ToolRegistry
# ═════════════════════════════════════════════════════════════════════════════


class ToolRegistry:
    """
    Catalog of registered tools with JSON schemas.

    Thread safety: not guaranteed. For a singleton use
    ``get_default_registry()``.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Any],
        *,
        tags: Optional[List[str]] = None,
        returns: str = "Any",
    ) -> None:
        """Register a tool under ``name``."""
        if name in self._tools:
            logger.debug("Tool %s already registered — overwriting", name)
        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            parameters=dict(parameters),
            handler=handler,
            tags=list(tags or []),
            returns=returns,
        )

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry."""
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def list_tools(self, *, tags: Optional[List[str]] = None) -> List[ToolSpec]:
        """Return all registered tools, optionally filtered by tags."""
        if tags is None:
            return list(self._tools.values())
        tag_set = set(tags)
        return [t for t in self._tools.values() if tag_set.intersection(t.tags)]

    def call(self, name: str, **kwargs: Any) -> Any:
        """Invoke a registered tool by name with keyword arguments."""
        spec = self._tools.get(name)
        if spec is None:
            raise KeyError(f"Tool not registered: {name}")
        try:
            return spec.handler(**kwargs)
        except Exception as exc:
            logger.warning("Tool %s failed: %s", name, exc)
            raise

    def publish_schemas(self) -> List[Dict[str, Any]]:
        """Return all tools as a list of JSON schemas for LLM consumption."""
        return [t.to_json_schema() for t in self._tools.values()]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_tools": len(self._tools),
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "tags": t.tags,
                    "parameters": t.parameters,
                }
                for t in self._tools.values()
            ],
        }


# ═════════════════════════════════════════════════════════════════════════════
# Default singleton
# ═════════════════════════════════════════════════════════════════════════════


_DEFAULT_REGISTRY: Optional[ToolRegistry] = None


def get_default_registry() -> ToolRegistry:
    """Return the global default tool registry."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = ToolRegistry()
    return _DEFAULT_REGISTRY
