"""
Stable LLM client adapter for ARGUS.

Wraps the existing ``ml/llm_signal.py`` Ollama/OpenAI client in a uniform
interface with timeout, retry, and JSON-mode toggle. Used by:
- ``core/reasoning/reflection_loop.py`` (Commit 3)
- ``core/memory/consolidation.py`` (Commit 5)
- ``core/tools/llm_tool_calling.py`` (Commit 3)

DO NOT fork the existing LLM client — inherit its bugs and fix them in
one place.
"""

from .client import LLMClient, get_llm_client, LLMResponse

__all__ = ["LLMClient", "get_llm_client", "LLMResponse"]
