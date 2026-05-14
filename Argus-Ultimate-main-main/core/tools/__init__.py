"""
Tool registry for ARGUS — catalogs callable tools with JSON schemas.

Lets the reasoning layer dispatch named tools (retrieve similar setups,
run what-if, evaluate MCTS plan, query decision journal, score competence)
with a single uniform interface.

v1 is rule-based dispatch; v2 will be LLM-driven via
``core/reasoning/reflection_loop.py``.
"""

from .tool_registry import ToolRegistry, ToolSpec, get_default_registry
from .tool_dispatcher import ToolDispatcher

__all__ = [
    "ToolRegistry",
    "ToolSpec",
    "ToolDispatcher",
    "get_default_registry",
]
