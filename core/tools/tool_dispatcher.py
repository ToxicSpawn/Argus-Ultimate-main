"""
Tool dispatcher — context-sensitive tool selection.

v1: simple rule-based dispatch. Given a "query intent" (e.g. "similar past
setups", "what-if counterfactual", "score competence"), resolve to the
appropriate tool and invoke it.

v2 (future): LLM-driven dispatch via function calling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Intent → tool mappings
# ═════════════════════════════════════════════════════════════════════════════


_INTENT_RULES: List[Tuple[List[str], str]] = [
    (["similar", "historical", "past", "like this"], "rag.retrieve_similar"),
    (["what if", "counterfactual", "alternative"], "counterfactual.what_if"),
    (["plan", "mcts", "search", "lookahead"], "mcts.plan"),
    (["competence", "capability", "how good", "am i good"], "metacognition.score"),
    (["event", "news", "macro"], "rag.retrieve_events"),
    (["decision", "journal"], "decision_journal.query"),
    (["regret", "missed"], "counterfactual.regret"),
]


# ═════════════════════════════════════════════════════════════════════════════
# ToolDispatcher
# ═════════════════════════════════════════════════════════════════════════════


class ToolDispatcher:
    """
    Route query intents to registered tools.

    Parameters
    ----------
    registry : ToolRegistry
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def resolve(self, query: str) -> Optional[str]:
        """Map a natural-language query to a tool name."""
        q = query.lower()
        for keywords, tool_name in _INTENT_RULES:
            if any(kw in q for kw in keywords):
                if self.registry.get(tool_name) is not None:
                    return tool_name
        return None

    def dispatch(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Dispatch a query to the best-matching tool.

        Returns
        -------
        Dict with keys:
            - ``tool``: tool name (or None)
            - ``result``: tool return value (or None)
            - ``error``: str (or None)
        """
        tool_name = self.resolve(query)
        if tool_name is None:
            return {
                "tool": None,
                "result": None,
                "error": f"No tool matched query: {query!r}",
            }

        try:
            result = self.registry.call(tool_name, **kwargs)
            return {"tool": tool_name, "result": result, "error": None}
        except Exception as exc:
            logger.warning("Dispatch %s failed: %s", tool_name, exc)
            return {"tool": tool_name, "result": None, "error": str(exc)}

    def dispatch_named(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        """Dispatch directly by tool name (bypasses intent matching)."""
        if self.registry.get(tool_name) is None:
            return {
                "tool": tool_name,
                "result": None,
                "error": f"Tool not registered: {tool_name}",
            }
        try:
            result = self.registry.call(tool_name, **kwargs)
            return {"tool": tool_name, "result": result, "error": None}
        except Exception as exc:
            logger.warning("Direct dispatch %s failed: %s", tool_name, exc)
            return {"tool": tool_name, "result": None, "error": str(exc)}

    def available_tool_names(self) -> List[str]:
        return [t.name for t in self.registry.list_tools()]
