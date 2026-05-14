# pyright: reportMissingImports=false
"""
Memory-Augmented Networks for Argus Trading.

This module implements memory-augmented neural networks for
long-term memory and experience replay in trading.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory."""
    EPISODIC = auto()  # Specific experiences
    SEMANTIC = auto()  # General knowledge
    WORKING = auto()  # Short-term active memory
    PROCEDURAL = auto()  # How-to knowledge


@dataclass
class MemoryEntry:
    """A single memory entry."""
    memory_id: str
    memory_type: MemoryType
    content: NDArray[np.float64]
    metadata: Dict[str, Any]
    timestamp: float
    access_count: int = 0
    importance: float = 1.0


@dataclass
class MemoryConfig:
    """Configuration for memory systems."""
    episodic_capacity: int = 10000
    semantic_capacity: int = 5000
    working_capacity: int = 100
    retrieval_k: int = 5  # Number of memories to retrieve
    decay_rate: float = 0.99  # Memory decay per access
    importance_boost: float = 1.1  # Boost for important memories


class EpisodicMemory:
    """Stores and retrieves specific experiences."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memories: deque = deque(maxlen=capacity)
        self.memory_index: Dict[str, int] = {}

    def store(self,
             content: NDArray[np.float64],
             metadata: Dict[str, Any],
             importance: float = 1.0) -> str:
        """Store an episodic memory."""
        memory_id = f"episodic_{len(self.memories)}_{id(content) % 10000}"

        memory = MemoryEntry(
            memory_id=memory_id,
            memory_type=MemoryType.EPISODIC,
            content=content.copy(),
            metadata=metadata,
            timestamp=np.random.uniform(0, 10000),  # Simulated timestamp
            importance=importance
        )

        self.memories.append(memory)
        self.memory_index[memory_id] = len(self.memories) - 1

        return memory_id

    def retrieve(self, 
                query: NDArray[np.float64],
                k: int = 5) -> List[MemoryEntry]:
        """Retrieve similar memories."""
        if not self.memories:
            return []

        # Compute similarities
        similarities = []
        for memory in self.memories:
            sim = self._compute_similarity(query, memory.content)
            # Weight by importance and recency
            weighted_sim = sim * memory.importance
            similarities.append((memory, weighted_sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        retrieved = [mem for mem, _ in similarities[:k]]
        
        # Update access counts
        for memory in retrieved:
            memory.access_count += 1

        return retrieved

    def _compute_similarity(self, 
                           a: NDArray[np.float64],
                           b: NDArray[np.float64]) -> float:
        """Compute similarity between vectors."""
        if len(a) != len(b):
            min_len = min(len(a), len(b))
            a = a[:min_len]
            b = b[:min_len]

        norm_a = np.linalg.norm(a) + 1e-8
        norm_b = np.linalg.norm(b) + 1e-8
        
        return float(np.dot(a, b) / (norm_a * norm_b))

    def consolidate(self) -> None:
        """Consolidate memories (merge similar ones)."""
        if len(self.memories) < 100:
            return

        # Find and merge similar memories
        merged_count = 0
        memories_list = list(self.memories)

        for i in range(len(memories_list)):
            for j in range(i + 1, len(memories_list)):
                sim = self._compute_similarity(
                    memories_list[i].content,
                    memories_list[j].content
                )

                if sim > 0.9:  # Very similar
                    # Merge into the more accessed one
                    if memories_list[i].access_count >= memories_list[j].access_count:
                        memories_list[i].importance *= 1.1
                        memories_list[i].access_count += memories_list[j].access_count
                        memories_list[j].importance = 0  # Mark for deletion
                    else:
                        memories_list[j].importance *= 1.1
                        memories_list[j].access_count += memories_list[i].access_count
                        memories_list[i].importance = 0
                    merged_count += 1

        # Remove marked memories
        self.memories = deque(
            [m for m in memories_list if m.importance > 0],
            maxlen=self.capacity
        )

        if merged_count > 0:
            logger.debug(f"Consolidated {merged_count} memories")


class SemanticMemory:
    """Stores general knowledge and patterns."""

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.concepts: Dict[str, MemoryEntry] = {}
        self.concept_graph: Dict[str, List[str]] = {}

    def store_concept(self,
                     concept_name: str,
                     content: NDArray[np.float64],
                     metadata: Dict[str, Any]) -> None:
        """Store a semantic concept."""
        memory = MemoryEntry(
            memory_id=f"semantic_{concept_name}",
            memory_type=MemoryType.SEMANTIC,
            content=content.copy(),
            metadata=metadata,
            timestamp=np.random.uniform(0, 10000),
            importance=1.0
        )

        self.concepts[concept_name] = memory
        self.concept_graph.setdefault(concept_name, [])

    def retrieve_concept(self, concept_name: str) -> Optional[MemoryEntry]:
        """Retrieve a specific concept."""
        return self.concepts.get(concept_name)

    def find_related(self, 
                    concept_name: str,
                    max_related: int = 5) -> List[str]:
        """Find related concepts."""
        if concept_name not in self.concept_graph:
            return []

        related = self.concept_graph[concept_name]
        return related[:max_related]

    def update_concept(self,
                      concept_name: str,
                      new_content: NDArray[np.float64],
                      learning_rate: float = 0.1) -> None:
        """Update a concept with new information."""
        if concept_name not in self.concepts:
            self.store_concept(concept_name, new_content, {"source": "update"})
            return

        memory = self.concepts[concept_name]
        # Exponential moving average update
        memory.content = (
            memory.content * (1 - learning_rate) +
            new_content * learning_rate
        )
        memory.access_count += 1


class WorkingMemory:
    """Short-term active memory for current task."""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.attention_weights: Dict[str, float] = {}

    def add(self, content: NDArray[np.float64], label: str = "") -> None:
        """Add to working memory."""
        self.buffer.append({
            "content": content,
            "label": label,
            "timestamp": len(self.buffer)
        })
        
        # Initialize attention
        if label:
            self.attention_weights[label] = 1.0

    def get_recent(self, n: int = 10) -> List[NDArray[np.float64]]:
        """Get recent items from working memory."""
        items = list(self.buffer)[-n:]
        return [item["content"] for item in items]

    def attend(self, label: str) -> Optional[NDArray[np.float64]]:
        """Focus attention on a specific item."""
        if label in self.attention_weights:
            self.attention_weights[label] *= 1.5  # Boost attention

        for item in reversed(self.buffer):
            if item["label"] == label:
                return item["content"]
        return None

    def clear(self) -> None:
        """Clear working memory."""
        self.buffer.clear()
        self.attention_weights.clear()


class MemoryAugmentedNetwork:
    """Memory-augmented neural network for trading."""

    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize the memory-augmented network."""
        self.config = config or MemoryConfig()

        # Initialize memory systems
        self.episodic = EpisodicMemory(self.config.episodic_capacity)
        self.semantic = SemanticMemory(self.config.semantic_capacity)
        self.working = WorkingMemory(self.config.working_capacity)

        self.query_count = 0
        self.retrieval_history: List[Dict[str, Any]] = []

    def store_experience(self,
                        state: NDArray[np.float64],
                        action: int,
                        reward: float,
                        next_state: NDArray[np.float64],
                        done: bool) -> str:
        """Store an experience in episodic memory."""
        content = np.concatenate([state, np.array([action, reward]), next_state])
        
        metadata = {
            "action": action,
            "reward": reward,
            "done": done,
            "state_shape": state.shape
        }

        importance = 1.0 + abs(reward)  # More important for larger rewards

        memory_id = self.episodic.store(content, metadata, importance)

        # Also update semantic memory with patterns
        if abs(reward) > 0.5:
            # Learn from significant experiences
            pattern_name = f"reward_{reward:.1f}_action_{action}"
            self.semantic.update_concept(pattern_name, state)

        return memory_id

    def retrieve_relevant(self, 
                         current_state: NDArray[np.float64],
                         k: int = 5) -> List[MemoryEntry]:
        """Retrieve relevant memories for current state."""
        self.query_count += 1

        # Retrieve from episodic memory
        episodic_memories = self.episodic.retrieve(current_state, k)

        # Record retrieval
        self.retrieval_history.append({
            "query": self.query_count,
            "memories_retrieved": len(episodic_memories),
            "avg_importance": np.mean([m.importance for m in episodic_memories]) if episodic_memories else 0.0
        })

        return episodic_memories

    def get_retrieved_context(self,
                             current_state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get context from retrieved memories."""
        memories = self.retrieve_relevant(current_state)

        if not memories:
            return np.zeros(8)

        # Aggregate retrieved memories
        contents = [m.content[:8] for m in memories]  # Take first 8 elements
        context = np.mean(contents, axis=0)

        # Normalize
        norm = np.linalg.norm(context) + 1e-8
        return context / norm

    def consolidate_memories(self) -> Dict[str, Any]:
        """Consolidate all memories."""
        initial_count = len(self.episodic.memories)
        self.episodic.consolidate()
        final_count = len(self.episodic.memories)

        return {
            "initial_memories": initial_count,
            "final_memories": final_count,
            "consolidated": initial_count - final_count
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            "episodic": {
                "count": len(self.episodic.memories),
                "capacity": self.config.episodic_capacity
            },
            "semantic": {
                "concepts": len(self.semantic.concepts),
                "capacity": self.config.semantic_capacity
            },
            "working": {
                "items": len(self.working.buffer),
                "capacity": self.config.working_capacity
            },
            "queries": self.query_count,
            "avg_retrieval": np.mean([r["memories_retrieved"] for r in self.retrieval_history[-100:]])
            if self.retrieval_history else 0.0
        }


__all__ = [
    "MemoryAugmentedNetwork",
    "MemoryConfig",
    "EpisodicMemory",
    "SemanticMemory",
    "WorkingMemory",
    "MemoryEntry",
    "MemoryType"
]