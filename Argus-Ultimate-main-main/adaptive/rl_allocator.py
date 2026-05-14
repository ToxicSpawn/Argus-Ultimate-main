"""
RL for allocation: simple value-based / Q-learning for strategy weights.

Learns weights or signal priority from (state, action) -> reward (e.g. realized PnL).
Integrates with strategy_allocator as optional multiplier source.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class RLAllocatorStub:
    """
    Stub: Q-table or value function over (strategy, regime, sentiment_bucket).
    update(state, action, reward) and get_weight(state, action) for integration.
    """

    def __init__(self, persist_path: str = "data/rl_allocator.json", learning_rate: float = 0.1) -> None:
        self.persist_path = str(persist_path)
        self.lr = float(learning_rate)
        self._q: Dict[str, Dict[str, float]] = {}  # state -> action -> value
        self._load()

    def _state_key(self, strategy: str, regime: str, sentiment_bucket: str = "mid") -> str:
        return f"{strategy}|{regime}|{sentiment_bucket}"

    def _load(self) -> None:
        try:
            p = Path(self.persist_path)
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "q" in data:
                    self._q = {str(k): dict(v) for k, v in (data.get("q") or {}).items()}
        except Exception as e:
            logger.debug("RLAllocator failed to load state from %s: %s", self.persist_path, e)

    def save(self) -> None:
        try:
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.persist_path).write_text(
                json.dumps({"q": self._q}, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.debug("RLAllocator failed to save state to %s: %s", self.persist_path, e)

    def update(self, state: str, action: str, reward: float) -> None:
        """Q-learning update: Q(s,a) += lr * (reward - Q(s,a))."""
        key = state
        if key not in self._q:
            self._q[key] = {}
        old = self._q[key].get(action, 0.0)
        self._q[key][action] = old + self.lr * (float(reward) - old)

    def get_weight(self, strategy: str, regime: str, sentiment_bucket: str = "mid") -> float:
        """Return weight in [0.5, 1.5] from Q-values for state (strategy, regime, sentiment)."""
        state = self._state_key(strategy, regime, sentiment_bucket)
        if state not in self._q or not self._q[state]:
            return 1.0
        vals = list(self._q[state].values())
        if not vals:
            return 1.0
        # Normalize to weight: max is 1.5, min 0.5
        mean_q = sum(vals) / len(vals)
        w = 1.0 + (mean_q * 0.2)
        return max(0.5, min(1.5, w))
