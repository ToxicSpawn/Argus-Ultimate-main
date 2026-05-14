"""Hierarchical reinforcement learning for multi-timescale trading decisions.

ARGUS operates on several timescales simultaneously:

* Sub-second micro decisions — order placement, cancel / replace, spread
  crossing. These must react in milliseconds.
* Minute-scale tactical decisions — open / close / rebalance intraday
  positions in response to signals.
* Hour / day scale strategic decisions — regime allocation, risk budgeting,
  which strategy family to favour.

Upgrade (2026-04 peak-potential):
- UCB1 agent selection: agents track uncertainty and get exploration bonuses,
  preventing permanent stagnation in suboptimal routing.
- Entropy bonus: policy updates include an entropy regularisation term to
  maintain exploration even after many updates.
- Cross-agent credit: when a slow agent governs a fast one, the fast agent
  also receives a partial credit signal from the slow agent's outcomes.
- Weight persistence: snapshot/restore serialises full weight matrices so
  the hierarchy survives hot-reloads and process restarts.
- Adaptive exploration decay: exploration sigma decays as sqrt(1/n_updates)
  so agents converge without collapsing prematurely.
- Per-agent gradient clipping to prevent runaway weight updates on extreme
  reward spikes.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent data classes
# ---------------------------------------------------------------------------


@dataclass
class ActionTrace:
    """One (state, action) record awaiting a reward signal."""

    state: np.ndarray
    action: np.ndarray
    timestamp: float
    logprob: float = 0.0


@dataclass
class AgentStats:
    """Lightweight running stats per agent for monitoring."""

    decisions: int = 0
    updates: int = 0
    total_reward: float = 0.0
    last_reward: float = 0.0
    reward_ema: float = 0.0
    ucb_bonus: float = 0.0


# ---------------------------------------------------------------------------
# RL agent
# ---------------------------------------------------------------------------


class RLAgent:
    """Linear Gaussian policy + REINFORCE updates with UCB and entropy bonus.

    Parameters
    ----------
    name
        Display name.
    state_dim
        Dimensionality of the state vector.
    action_dim
        Number of scalar action components.
    timescale_seconds
        Minimum time between decisions this agent handles.
    learning_rate
        Step size for REINFORCE policy update.
    exploration
        Initial std-dev of Gaussian action noise; decays adaptively.
    baseline_ema
        EMA coefficient for the reward baseline.
    entropy_beta
        Weight of entropy regularisation bonus (higher = more exploration).
    grad_clip
        Maximum L2 norm of gradient before clipping.
    exploration_decay
        If True, exploration decays as init_exploration / sqrt(1 + n_updates).
    """

    def __init__(
        self,
        name: str,
        state_dim: int = 8,
        action_dim: int = 1,
        timescale_seconds: float = 1.0,
        learning_rate: float = 1e-2,
        exploration: float = 0.1,
        baseline_ema: float = 0.1,
        entropy_beta: float = 0.01,
        grad_clip: float = 1.0,
        exploration_decay: bool = True,
        seed: int = 0,
    ) -> None:
        self.name = str(name)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.timescale_seconds = float(timescale_seconds)
        self.learning_rate = float(learning_rate)
        self._init_exploration = max(1e-6, float(exploration))
        self.exploration = self._init_exploration
        self.baseline_ema = float(baseline_ema)
        self.entropy_beta = float(entropy_beta)
        self.grad_clip = float(grad_clip)
        self.exploration_decay = bool(exploration_decay)
        self._rng = np.random.default_rng(seed)
        self.weights = self._rng.normal(scale=0.05, size=(self.state_dim, self.action_dim))
        self.bias = np.zeros(self.action_dim, dtype=np.float64)
        self.baseline = 0.0
        self.last_decision_time: float = -float("inf")
        self.pending: Deque[ActionTrace] = deque(maxlen=256)
        self.stats = AgentStats()
        # UCB tracking: total selections and total reward for upper confidence bound
        self._ucb_pulls: int = 0
        self._ucb_reward_sum: float = 0.0

    # -- policy ----------------------------------------------------------

    def policy_mean(self, state: np.ndarray) -> np.ndarray:
        return state @ self.weights + self.bias

    def _log_prob(self, action: np.ndarray, mean: np.ndarray) -> float:
        """Log probability of action under current Gaussian policy."""
        diff = action - mean
        lp = -0.5 * float(np.sum(diff ** 2)) / (self.exploration ** 2)
        lp -= self.action_dim * math.log(self.exploration * math.sqrt(2 * math.pi))
        return float(lp)

    def _entropy(self) -> float:
        """Differential entropy of the Gaussian policy."""
        return float(0.5 * self.action_dim * (1.0 + math.log(2 * math.pi * self.exploration ** 2)))

    def ucb_score(self, total_pulls: int) -> float:
        """UCB1 score for agent selection by the manager."""
        if self._ucb_pulls == 0:
            return float('inf')
        mean_r = self._ucb_reward_sum / self._ucb_pulls
        bonus = math.sqrt(2.0 * math.log(max(1, total_pulls)) / self._ucb_pulls)
        self.stats.ucb_bonus = bonus
        return mean_r + bonus

    def decide(self, state: np.ndarray, now: float) -> np.ndarray:
        """Sample an action for ``state`` and record the trace."""
        mean = self.policy_mean(state)
        noise = self._rng.normal(scale=self.exploration, size=self.action_dim)
        action = mean + noise
        lp = self._log_prob(action, mean)
        self.pending.append(ActionTrace(
            state=state.copy(), action=action.copy(), timestamp=now, logprob=lp
        ))
        self.last_decision_time = now
        self.stats.decisions += 1
        self._ucb_pulls += 1
        return action

    def record_outcome(self, reward: float) -> float:
        """REINFORCE update with entropy bonus and adaptive exploration.

        Returns the effective advantage used for diagnostics.
        """
        if not self.pending:
            return 0.0
        trace = self.pending.pop()
        advantage = float(reward) - self.baseline

        # Entropy bonus: encourages the policy to maintain spread.
        entropy_reward = advantage + self.entropy_beta * self._entropy()

        mean = self.policy_mean(trace.state)
        grad_log = (trace.action - mean) / (self.exploration ** 2)

        w_update = self.learning_rate * entropy_reward * np.outer(trace.state, grad_log)
        b_update = self.learning_rate * entropy_reward * grad_log

        # Gradient clipping.
        w_norm = float(np.linalg.norm(w_update))
        if w_norm > self.grad_clip:
            w_update = w_update * (self.grad_clip / w_norm)

        self.weights += w_update
        self.bias += b_update

        # Adaptive exploration decay.
        if self.exploration_decay and self.stats.updates > 0:
            self.exploration = self._init_exploration / math.sqrt(1.0 + self.stats.updates)
            self.exploration = max(self._init_exploration * 0.05, self.exploration)

        self.baseline = (1 - self.baseline_ema) * self.baseline + self.baseline_ema * float(reward)
        self._ucb_reward_sum += float(reward)
        self.stats.updates += 1
        self.stats.total_reward += float(reward)
        self.stats.last_reward = float(reward)
        self.stats.reward_ema = (
            (1 - self.baseline_ema) * self.stats.reward_ema + self.baseline_ema * float(reward)
        )
        return advantage

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timescale_seconds": self.timescale_seconds,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "decisions": self.stats.decisions,
            "updates": self.stats.updates,
            "total_reward": float(self.stats.total_reward),
            "last_reward": float(self.stats.last_reward),
            "reward_ema": float(self.stats.reward_ema),
            "baseline": float(self.baseline),
            "pending_traces": len(self.pending),
            "weight_norm": float(np.linalg.norm(self.weights)),
            "exploration": float(self.exploration),
            "ucb_pulls": int(self._ucb_pulls),
            "ucb_bonus": float(self.stats.ucb_bonus),
        }

    def get_weights(self) -> Dict[str, Any]:
        """Serialise weights for persistence / hot-reload."""
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "baseline": float(self.baseline),
            "exploration": float(self.exploration),
            "ucb_pulls": int(self._ucb_pulls),
            "ucb_reward_sum": float(self._ucb_reward_sum),
        }

    def set_weights(self, data: Dict[str, Any]) -> None:
        """Restore weights from a previous :meth:`get_weights` call."""
        w = np.asarray(data["weights"], dtype=np.float64)
        b = np.asarray(data["bias"], dtype=np.float64)
        if w.shape == self.weights.shape:
            self.weights = w
        if b.shape == self.bias.shape:
            self.bias = b
        self.baseline = float(data.get("baseline", self.baseline))
        self.exploration = float(data.get("exploration", self.exploration))
        self._ucb_pulls = int(data.get("ucb_pulls", self._ucb_pulls))
        self._ucb_reward_sum = float(data.get("ucb_reward_sum", self._ucb_reward_sum))


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class HierarchicalRLManager:
    """Route decisions to the agent responsible for the current timescale.

    Upgrade notes (2026-04):
    - UCB1-based routing: among agents that are "due", we pick the one with
      the highest UCB score rather than always the fastest. This prevents
      fast agents from completely dominating slow but high-value ones.
    - Cross-agent credit: record_outcome propagates partial reward both
      upward (to slower/strategic agents) and downward (to faster subordinates
      that contributed to the outcome).
    - Full weight serialisation via get_all_weights / set_all_weights.
    """

    def __init__(self, default_state_dim: int = 8) -> None:
        self.default_state_dim = int(default_state_dim)
        self.agents: Dict[str, RLAgent] = {}
        self._order: List[str] = []
        self.last_router_name: Optional[str] = None
        self.step_count = 0
        self._total_pulls = 0

    # -- registration --------------------------------------------------------

    def add_agent(
        self,
        name: str,
        timescale_seconds: float,
        state_dim: Optional[int] = None,
        action_dim: int = 1,
        **agent_kwargs: Any,
    ) -> RLAgent:
        if name in self.agents:
            raise ValueError(f"agent {name!r} already registered")
        agent = RLAgent(
            name=name,
            state_dim=int(state_dim if state_dim is not None else self.default_state_dim),
            action_dim=int(action_dim),
            timescale_seconds=float(timescale_seconds),
            **agent_kwargs,
        )
        self.agents[name] = agent
        self._order.append(name)
        self._order.sort(key=lambda n: self.agents[n].timescale_seconds)
        return agent

    # -- decision making -----------------------------------------------------

    def step(
        self,
        state: np.ndarray,
        current_time: Optional[float] = None,
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Route ``state`` to the best-fit agent via UCB selection.

        Among agents that are "due" (elapsed >= timescale), we pick the one
        with the highest UCB score. If no agent is due, fall back to UCB
        across all agents regardless of cooldown.
        """
        if not self.agents:
            return None, None
        now = float(current_time if current_time is not None else time.time())
        self.step_count += 1
        self._total_pulls += 1

        due = [
            name for name in self._order
            if (now - self.agents[name].last_decision_time) >= self.agents[name].timescale_seconds
        ]
        candidates = due if due else self._order

        # UCB1 selection among candidates.
        best_name = max(candidates, key=lambda n: self.agents[n].ucb_score(self._total_pulls))
        chosen = self.agents[best_name]

        # Pad/clip state to the agent's state_dim.
        st = np.zeros(chosen.state_dim, dtype=np.float64)
        src = np.asarray(state, dtype=np.float64).flatten()
        n = min(chosen.state_dim, src.size)
        st[:n] = src[:n]
        action = chosen.decide(st, now)
        self.last_router_name = chosen.name
        return chosen.name, action

    def record_outcome(self, agent_name: str, reward: float) -> float:
        """Update the named agent and propagate credit bi-directionally.

        Upward (to slower strategic agents): discounted by timescale ratio.
        Downward (to faster sub-agents): small cross-credit to reinforce
        that the fast agent contributed to the strategic outcome.
        """
        agent = self.agents.get(agent_name)
        if agent is None:
            logger.debug("hierarchical_rl: unknown agent %s", agent_name)
            return 0.0
        advantage = agent.record_outcome(reward)
        own_scale = agent.timescale_seconds

        for name, other in self.agents.items():
            if name == agent_name:
                continue
            if other.timescale_seconds > own_scale:
                # Upward propagation — slower strategic agents.
                discount = own_scale / max(own_scale, other.timescale_seconds)
                if other.pending:
                    other.record_outcome(reward * discount)
            else:
                # Downward cross-credit — faster sub-agents get a small boost.
                cross = 0.1 * reward * (other.timescale_seconds / max(own_scale, 1e-6))
                if other.pending:
                    other.record_outcome(cross)
        return advantage

    # -- weight persistence --------------------------------------------------

    def get_all_weights(self) -> Dict[str, Any]:
        """Serialise all agent weights for checkpointing."""
        return {name: agent.get_weights() for name, agent in self.agents.items()}

    def set_all_weights(self, data: Dict[str, Any]) -> None:
        """Restore agent weights from a previous :meth:`get_all_weights` call."""
        for name, w_data in data.items():
            if name in self.agents:
                self.agents[name].set_weights(w_data)

    # -- introspection -------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        return {
            "step_count": self.step_count,
            "total_pulls": self._total_pulls,
            "last_router_name": self.last_router_name,
            "agents": {name: self.agents[name].snapshot() for name in self._order},
        }
