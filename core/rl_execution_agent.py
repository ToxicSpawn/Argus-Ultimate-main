"""Reinforcement-learning execution agent (pure-numpy PPO-lite).

This module provides a lightweight reinforcement learning agent that decides
*how* to execute an order once the strategy engine has decided *what* to trade.
Because the broader ARGUS runtime cannot always import PyTorch, we implement a
simplified policy-gradient agent using only numpy: a linear policy head per
action dimension, a linear value baseline, and advantage-weighted SGD updates.

State features (fixed order expected by callers)::

    state = {
        "spread_bps":     spread of the top of book in basis points,
        "volatility":     recent realised volatility (e.g. 1-min std),
        "volume_ratio":   traded volume / median volume,
        "urgency_needed": 0..1 how quickly the order must go out,
        "inventory_pct":  current position / target position in [-1, 1],
    }

Actions (discrete buckets along each axis)::

    order_type   in {"market", "limit", "post_only"}
    urgency      in {0.0, 0.25, 0.5, 0.75, 1.0}
    slice_count  in {1, 3, 5}

The agent is intentionally tiny — we rely on the strategy layer for alpha and
only use RL to learn micro-structure nuances (when to pay the spread, when to
post, how finely to slice).  Training happens online via :meth:`record_outcome`
after each executed order.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


STATE_KEYS: Tuple[str, ...] = (
    "spread_bps",
    "volatility",
    "volume_ratio",
    "urgency_needed",
    "inventory_pct",
)
STATE_DIM: int = len(STATE_KEYS) + 1  # +1 bias term

ORDER_TYPES: Tuple[str, ...] = ("market", "limit", "post_only")
URGENCIES: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
SLICE_COUNTS: Tuple[int, ...] = (1, 3, 5)


# ---------------------------------------------------------------------------
# Policy + value function
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    s = float(np.sum(e))
    if s <= 0:
        return np.ones_like(x) / len(x)
    return e / s


@dataclass
class LinearPolicy:
    """Linear softmax policy with one weight matrix per action dimension."""

    n_order_types: int = len(ORDER_TYPES)
    n_urgencies: int = len(URGENCIES)
    n_slices: int = len(SLICE_COUNTS)
    state_dim: int = STATE_DIM
    scale: float = 0.01

    W_order: np.ndarray = field(init=False)
    W_urgency: np.ndarray = field(init=False)
    W_slice: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(0)
        self.W_order = rng.normal(0.0, self.scale, size=(self.state_dim, self.n_order_types))
        self.W_urgency = rng.normal(0.0, self.scale, size=(self.state_dim, self.n_urgencies))
        self.W_slice = rng.normal(0.0, self.scale, size=(self.state_dim, self.n_slices))

    def logits(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return state @ self.W_order, state @ self.W_urgency, state @ self.W_slice

    def action_probs(
        self, state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lo, lu, ls = self.logits(state)
        return _softmax(lo), _softmax(lu), _softmax(ls)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "W_order_norm": float(np.linalg.norm(self.W_order)),
            "W_urgency_norm": float(np.linalg.norm(self.W_urgency)),
            "W_slice_norm": float(np.linalg.norm(self.W_slice)),
            "state_dim": int(self.state_dim),
        }


@dataclass
class LinearValueFunction:
    """Simple linear V(s) baseline used to compute advantage."""

    state_dim: int = STATE_DIM
    lr: float = 0.01
    w: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.w = np.zeros(self.state_dim, dtype=np.float64)

    def predict(self, state: np.ndarray) -> float:
        return float(state @ self.w)

    def update(self, state: np.ndarray, target: float) -> None:
        pred = self.predict(state)
        error = target - pred
        self.w += self.lr * error * state

    def snapshot(self) -> Dict[str, Any]:
        return {
            "w_norm": float(np.linalg.norm(self.w)),
            "lr": float(self.lr),
        }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


def _state_to_vector(state: Dict[str, Any]) -> np.ndarray:
    vec = np.zeros(STATE_DIM, dtype=np.float64)
    for i, key in enumerate(STATE_KEYS):
        try:
            vec[i] = float(state.get(key, 0.0))
        except (TypeError, ValueError):
            vec[i] = 0.0
    # Light normalisation so feature magnitudes don't blow up the softmax.
    vec[0] = float(np.clip(vec[0] / 100.0, -5.0, 5.0))  # spread_bps
    vec[1] = float(np.clip(vec[1], -5.0, 5.0))  # volatility
    vec[2] = float(np.clip(vec[2] - 1.0, -5.0, 5.0))  # volume_ratio around 1
    vec[3] = float(np.clip(vec[3], 0.0, 1.0))
    vec[4] = float(np.clip(vec[4], -1.0, 1.0))
    vec[-1] = 1.0  # bias
    return vec


class RLExecutionAgent:
    """Pure-numpy PPO-lite execution agent.

    Usage::

        agent = RLExecutionAgent()
        action = agent.decide(state)
        # ... execute order ...
        reward = compute_reward(fill)
        agent.record_outcome(state, action, reward)
    """

    def __init__(
        self,
        policy_lr: float = 0.05,
        value_lr: float = 0.02,
        entropy_coef: float = 0.01,
        seed: Optional[int] = None,
    ) -> None:
        self.policy = LinearPolicy()
        self.value_fn = LinearValueFunction(lr=value_lr)
        self.policy_lr = float(policy_lr)
        self.entropy_coef = float(entropy_coef)
        self._rng = np.random.default_rng(seed)
        self._n_updates = 0
        self._last_entropy = math.log(len(ORDER_TYPES))  # uniform entropy

    # -- decision ------------------------------------------------------------

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        vec = _state_to_vector(state)
        p_order, p_urg, p_slice = self.policy.action_probs(vec)

        order_idx = int(self._rng.choice(len(ORDER_TYPES), p=p_order))
        urg_idx = int(self._rng.choice(len(URGENCIES), p=p_urg))
        slice_idx = int(self._rng.choice(len(SLICE_COUNTS), p=p_slice))

        return {
            "order_type": ORDER_TYPES[order_idx],
            "urgency": float(URGENCIES[urg_idx]),
            "slice_count": int(SLICE_COUNTS[slice_idx]),
            "_order_idx": order_idx,
            "_urg_idx": urg_idx,
            "_slice_idx": slice_idx,
            "_probs": {
                "order": p_order.tolist(),
                "urgency": p_urg.tolist(),
                "slice": p_slice.tolist(),
            },
        }

    # -- update --------------------------------------------------------------

    @staticmethod
    def _gradient_step(
        W: np.ndarray,
        state: np.ndarray,
        chosen_idx: int,
        probs: np.ndarray,
        advantage: float,
        lr: float,
        entropy_coef: float,
    ) -> np.ndarray:
        """SGD step for a softmax head with entropy bonus."""
        # ∂log π / ∂ logits = one_hot - probs
        one_hot = np.zeros_like(probs)
        one_hot[chosen_idx] = 1.0
        grad_log = one_hot - probs  # shape (A,)

        # Outer product of state with grad_log gives the weight gradient.
        policy_grad = np.outer(state, grad_log) * advantage

        # Entropy bonus gradient: ∂H/∂logits = -(log π + 1).  Use -(log π + 1) then
        # take outer with state; the minus is flipped for ascent.
        with np.errstate(divide="ignore"):
            log_probs = np.where(probs > 0, np.log(probs), -30.0)
        entropy_grad = -(log_probs + 1.0) - np.sum(probs * -(log_probs + 1.0))
        entropy_weight_grad = np.outer(state, entropy_grad) * entropy_coef

        return W + lr * (policy_grad + entropy_weight_grad)

    def record_outcome(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
    ) -> None:
        vec = _state_to_vector(state)

        # Baseline update first — advantage uses OLD prediction.
        v_old = self.value_fn.predict(vec)
        advantage = float(reward) - v_old
        self.value_fn.update(vec, float(reward))

        probs = action.get("_probs")
        if not probs:
            # Recompute if not cached from decide().
            p_order, p_urg, p_slice = self.policy.action_probs(vec)
        else:
            p_order = np.asarray(probs["order"], dtype=np.float64)
            p_urg = np.asarray(probs["urgency"], dtype=np.float64)
            p_slice = np.asarray(probs["slice"], dtype=np.float64)

        order_idx = int(action.get("_order_idx", 0))
        urg_idx = int(action.get("_urg_idx", 0))
        slice_idx = int(action.get("_slice_idx", 0))

        self.policy.W_order = self._gradient_step(
            self.policy.W_order, vec, order_idx, p_order, advantage, self.policy_lr, self.entropy_coef
        )
        self.policy.W_urgency = self._gradient_step(
            self.policy.W_urgency, vec, urg_idx, p_urg, advantage, self.policy_lr, self.entropy_coef
        )
        self.policy.W_slice = self._gradient_step(
            self.policy.W_slice, vec, slice_idx, p_slice, advantage, self.policy_lr, self.entropy_coef
        )

        # Light weight clamp to keep softmax well-conditioned.
        self.policy.W_order = np.clip(self.policy.W_order, -5.0, 5.0)
        self.policy.W_urgency = np.clip(self.policy.W_urgency, -5.0, 5.0)
        self.policy.W_slice = np.clip(self.policy.W_slice, -5.0, 5.0)

        self._n_updates += 1
        self._last_entropy = self._entropy(p_order, p_urg, p_slice)

    # -- utilities -----------------------------------------------------------

    @staticmethod
    def _entropy(*probs_tuple: Sequence[float]) -> float:
        total = 0.0
        for p in probs_tuple:
            arr = np.asarray(p, dtype=np.float64)
            safe = np.where(arr > 0, arr, 1e-12)
            total += float(-np.sum(arr * np.log(safe)))
        return total / max(len(probs_tuple), 1)

    def get_policy_entropy(self) -> float:
        return float(self._last_entropy)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.snapshot(),
            "value_fn": self.value_fn.snapshot(),
            "n_updates": int(self._n_updates),
            "policy_lr": float(self.policy_lr),
            "entropy_coef": float(self.entropy_coef),
            "last_entropy": float(self._last_entropy),
            "state_keys": list(STATE_KEYS),
            "order_types": list(ORDER_TYPES),
            "urgencies": list(URGENCIES),
            "slice_counts": list(SLICE_COUNTS),
        }


__all__ = [
    "STATE_KEYS",
    "ORDER_TYPES",
    "URGENCIES",
    "SLICE_COUNTS",
    "LinearPolicy",
    "LinearValueFunction",
    "RLExecutionAgent",
]
