"""
Memory consolidation v2 — episodic → semantic compression.

Design principle (per Plan-agent risk R1)
-----------------------------------------
``core/cross_session_memory.py`` is read at startup by
``get_startup_briefing()`` with a threading lock. A "sleep cycle" that
rewrites content in-place can corrupt that startup path. So v2 runs
**alongside** the existing system with a new ``semantic_memory.db`` and
the existing ``cross_session_memory.sqlite`` stays untouched.

Components
----------
- ``episodic_memory``: append-only event log with vector embeddings from
  Commit 1's RAG.
- ``semantic_memory``: compressed knowledge graph extracted from episodes.
- ``consolidation``: scheduled worker (every N cycles) that compresses
  recent episodic entries into semantic summaries.
- ``memory_replay``: offline replay buffer for continual learning
  (feeds ``core/ewc_continual_learner.py``).
"""

from .episodic_memory import EpisodicMemory, EpisodicEntry
from .semantic_memory import SemanticMemory, SemanticFact
from .consolidation import MemoryConsolidator
from .memory_replay import MemoryReplayBuffer

__all__ = [
    "EpisodicMemory",
    "EpisodicEntry",
    "SemanticMemory",
    "SemanticFact",
    "MemoryConsolidator",
    "MemoryReplayBuffer",
]
