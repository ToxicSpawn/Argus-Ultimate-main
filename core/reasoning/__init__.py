"""
Reasoning layer for ARGUS — reflection and self-critique loops.

Wraps the LLM signal path with a 1-step reflection cycle that asks the
model to critique its own signal against decision-journal evidence and
RAG-retrieved similar historical setups.

Per Plan-agent review: Tree-of-Thought with 5×3×5 branching would cost
75 LLM calls per decision → minutes per cycle on local Ollama. We ship
reflection-only (1 critique) as the realistic v1.
"""

from .reflection_loop import ReflectionLoop, ReflectionResult, reflect_signal

__all__ = ["ReflectionLoop", "ReflectionResult", "reflect_signal"]
