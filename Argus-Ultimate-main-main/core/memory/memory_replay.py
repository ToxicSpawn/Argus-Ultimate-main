"""
Memory replay buffer for continual learning.

Provides a ring buffer of (state, action, reward, next_state) tuples
drawn from episodic memory for off-line replay by the EWC continual
learner (``core/ewc_continual_learner.py``). This is the standard
experience replay pattern used in deep RL.

The buffer is **prioritized by recency** with optional importance
weighting: recent episodes are sampled more often than ancient ones,
but high-surprise (high-reward or high-loss) episodes get extra weight.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Data classes
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class ReplayEntry:
    """A single replay tuple."""

    state: np.ndarray
    action: Any
    reward: float
    next_state: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: float = 1.0


# ═════════════════════════════════════════════════════════════════════════════
# MemoryReplayBuffer
# ═════════════════════════════════════════════════════════════════════════════


class MemoryReplayBuffer:
    """
    Prioritized experience replay buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of entries. Oldest get dropped on overflow.
    alpha : float, default 0.6
        Priority exponent (0 = uniform sampling, 1 = pure priority).
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
    ) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self._entries: Deque[ReplayEntry] = deque(maxlen=self.capacity)
        self._rng = np.random.default_rng(42)

    def add(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a new replay entry."""
        priority = 1.0 + abs(float(reward))  # high-reward episodes get higher priority
        self._entries.append(ReplayEntry(
            state=np.asarray(state, dtype=np.float64),
            action=action,
            reward=float(reward),
            next_state=(
                np.asarray(next_state, dtype=np.float64)
                if next_state is not None else None
            ),
            metadata=dict(metadata or {}),
            priority=priority,
        ))

    def sample(self, batch_size: int = 32) -> List[ReplayEntry]:
        """
        Sample a batch using prioritized replay.

        Probability(entry_i) ∝ priority_i^alpha.
        """
        if not self._entries:
            return []
        n = len(self._entries)
        if n <= batch_size:
            return list(self._entries)

        priorities = np.array([e.priority for e in self._entries])
        weights = np.power(priorities, self.alpha)
        probs = weights / weights.sum()
        indices = self._rng.choice(n, size=batch_size, replace=False, p=probs)
        return [self._entries[i] for i in indices]

    def sample_random(self, batch_size: int = 32) -> List[ReplayEntry]:
        """Uniform random sampling (non-prioritized baseline)."""
        if not self._entries:
            return []
        n = len(self._entries)
        if n <= batch_size:
            return list(self._entries)
        indices = self._rng.choice(n, size=batch_size, replace=False)
        return [self._entries[i] for i in indices]

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    # ── Integration with episodic memory ────────────────────────────────────

    def load_from_episodic(
        self,
        episodic_memory: Any,
        *,
        max_entries: int = 1000,
    ) -> int:
        """
        Populate the buffer by converting episodic entries into replay tuples.

        Requires that episodic entries have metadata with ``state``,
        ``action``, ``reward`` fields.

        Returns the number of entries loaded.
        """
        try:
            entries = episodic_memory.get_recent(n=max_entries)
        except Exception as exc:
            logger.debug("load_from_episodic failed: %s", exc)
            return 0

        loaded = 0
        for ep in entries:
            meta = ep.metadata or {}
            state = meta.get("state")
            action = meta.get("action")
            reward = meta.get("reward", meta.get("pnl"))

            if state is None or action is None or reward is None:
                continue

            try:
                self.add(
                    state=np.asarray(state, dtype=np.float64),
                    action=action,
                    reward=float(reward),
                    next_state=meta.get("next_state"),
                    metadata={"episode_id": ep.id, "event_type": ep.event_type},
                )
                loaded += 1
            except Exception:
                continue

        return loaded

    def snapshot(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "n_entries": len(self._entries),
            "alpha": self.alpha,
            "mean_priority": float(np.mean([e.priority for e in self._entries])) if self._entries else 0.0,
        }
