"""
Monte Carlo Tree Search (MCTS) with UCB1.

Standard algorithm: selection (UCB1) → expansion → simulation (random
rollout) → backpropagation. Uses ``core/world_model.py`` as the generative
model for rollouts.

Parameters
----------
- ``n_simulations``: MCTS iterations per planning call
- ``max_depth``: depth of each rollout
- ``c_ucb``: UCB1 exploration constant (default √2)
- ``discount``: reward discount factor

Trading use
-----------
Wired into ``adaptive/predictive_planner.py`` as an optional planner for
multi-step trade sequences with world-model rollouts.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# MCTS node + config
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class MCTSConfig:
    n_simulations: int = 200
    max_depth: int = 10
    c_ucb: float = math.sqrt(2.0)
    discount: float = 0.99
    rollout_policy: str = "random"  # "random" or "greedy"
    time_budget_s: float = 0.5


class MCTSNode:
    """A single node in the MCTS tree."""

    __slots__ = (
        "state",
        "parent",
        "action",
        "children",
        "visit_count",
        "total_reward",
        "untried_actions",
        "depth",
    )

    def __init__(
        self,
        state: np.ndarray,
        parent: Optional["MCTSNode"] = None,
        action: Any = None,
        untried_actions: Optional[List[Any]] = None,
        depth: int = 0,
    ) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[MCTSNode] = []
        self.visit_count = 0
        self.total_reward = 0.0
        self.untried_actions = list(untried_actions) if untried_actions else []
        self.depth = int(depth)

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(self.visit_count, 1)

    def best_child(self, c_ucb: float) -> "MCTSNode":
        """Select the child with the highest UCB1 score."""
        if not self.children:
            raise ValueError("best_child called on leaf")
        log_N = math.log(max(self.visit_count, 1))
        scores = []
        for child in self.children:
            exploit = child.mean_reward
            explore = c_ucb * math.sqrt(log_N / max(child.visit_count, 1))
            scores.append(exploit + explore)
        return self.children[int(np.argmax(scores))]


# ═════════════════════════════════════════════════════════════════════════════
# MCTS
# ═════════════════════════════════════════════════════════════════════════════


class MCTS:
    """
    Monte Carlo Tree Search planner.

    Parameters
    ----------
    world_model : Any
        Must implement ``predict_next(state, action) → (next_state, reward,
        uncertainty)``.
    action_space : Sequence[Any]
        The discrete set of actions available at each state.
    config : MCTSConfig
    """

    def __init__(
        self,
        world_model: Any,
        action_space: Sequence[Any],
        config: Optional[MCTSConfig] = None,
    ) -> None:
        if not hasattr(world_model, "predict_next"):
            raise ValueError("world_model must implement predict_next()")
        self.world_model = world_model
        self.action_space = list(action_space)
        self.config = config or MCTSConfig()
        self._rng = random.Random(42)

    def plan(
        self,
        initial_state: np.ndarray,
        *,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run MCTS from ``initial_state`` and return the best action.

        Returns
        -------
        Dict[str, Any]
            ``{"best_action", "action_values", "n_simulations_run",
              "tree_size", "elapsed_ms"}``
        """
        t0 = time.perf_counter()
        if seed is not None:
            self._rng = random.Random(seed)

        root = MCTSNode(
            state=initial_state.copy(),
            untried_actions=list(self.action_space),
        )

        n_sims = 0
        time_budget = float(self.config.time_budget_s)
        for _ in range(self.config.n_simulations):
            if time.perf_counter() - t0 > time_budget:
                break
            # 1. Selection
            node = self._select(root)
            # 2. Expansion
            if not node.is_fully_expanded and node.depth < self.config.max_depth:
                node = self._expand(node)
            # 3. Simulation
            reward = self._simulate(node)
            # 4. Backpropagation
            self._backpropagate(node, reward)
            n_sims += 1

        # Best action: most-visited child
        if not root.children:
            best_action = self.action_space[0] if self.action_space else None
            action_values: Dict[Any, float] = {}
        else:
            best_child = max(root.children, key=lambda c: c.visit_count)
            best_action = best_child.action
            action_values = {
                str(c.action): c.mean_reward for c in root.children
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000
        tree_size = self._count_nodes(root)

        return {
            "best_action": best_action,
            "action_values": action_values,
            "n_simulations_run": n_sims,
            "tree_size": tree_size,
            "root_visits": root.visit_count,
            "elapsed_ms": elapsed_ms,
            "method": "mcts_ucb1",
        }

    # ── MCTS phases ──────────────────────────────────────────────────────────

    def _select(self, root: MCTSNode) -> MCTSNode:
        """Descend tree using UCB1 until reaching an unexpanded or leaf node."""
        node = root
        while node.is_fully_expanded and node.children:
            node = node.best_child(self.config.c_ucb)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Create a new child node by trying an untried action."""
        if not node.untried_actions:
            return node
        action = node.untried_actions.pop()
        try:
            next_state, _, _ = self.world_model.predict_next(node.state, action)
        except Exception:
            return node

        child = MCTSNode(
            state=np.asarray(next_state, dtype=float),
            parent=node,
            action=action,
            untried_actions=list(self.action_space),
            depth=node.depth + 1,
        )
        node.children.append(child)
        return child

    def _simulate(self, node: MCTSNode) -> float:
        """Random rollout from ``node`` until ``max_depth``."""
        state = node.state
        total_reward = 0.0
        discount = 1.0
        remaining_depth = max(0, self.config.max_depth - node.depth)

        for _ in range(remaining_depth):
            if not self.action_space:
                break
            if self.config.rollout_policy == "greedy":
                # Greedy: evaluate all actions and pick the best 1-step reward
                best_r = -float("inf")
                best_next = state
                for action in self.action_space:
                    try:
                        next_state, r, _ = self.world_model.predict_next(state, action)
                        if r > best_r:
                            best_r = r
                            best_next = next_state
                    except Exception:
                        continue
                total_reward += discount * max(best_r, 0.0)
                state = np.asarray(best_next, dtype=float)
            else:
                # Random rollout
                action = self._rng.choice(self.action_space)
                try:
                    next_state, r, _ = self.world_model.predict_next(state, action)
                except Exception:
                    break
                total_reward += discount * float(r)
                state = np.asarray(next_state, dtype=float)
            discount *= self.config.discount
        return float(total_reward)

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Update visit counts and total rewards up the tree."""
        cur: Optional[MCTSNode] = node
        while cur is not None:
            cur.visit_count += 1
            cur.total_reward += reward
            cur = cur.parent

    def _count_nodes(self, root: MCTSNode) -> int:
        """Count nodes in the tree."""
        count = 1
        for child in root.children:
            count += self._count_nodes(child)
        return count
