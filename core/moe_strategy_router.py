"""Mixture-of-Experts strategy router for ARGUS.

Upgrade (2026-04 peak-potential):
- Temperature annealing on softmax gating: starts exploratory (high temp)
  and decays toward exploitation over routing steps, controlled by
  temp_init, temp_min, and temp_decay_rate.
- Expert load-balancing loss: tracks expert utilisation and applies a
  penalty to over-used experts so the router doesn't collapse onto one
  expert in non-volatile markets.
- Per-expert EMA momentum on gating weights: smooths out noisy one-off
  outcomes before they shift routing behaviour.
- Dynamic expert addition at runtime: add_expert() now re-initialises
  the gating column for the new expert without resetting existing weights.
- Regime-lock: when regime_confidence > lock_threshold, the top expert
  for that regime is hard-pinned with weight 1.0 (bypass gating).
- All previous behaviour (delta-rule updates, strategy score EMA,
  top-k routing, snapshot) is preserved.
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
    "volatility",
    "trend_strength",
    "volume_ratio",
    "regime_confidence",
    "funding_rate",
    "spread_bps",
)
STATE_DIM: int = len(STATE_KEYS) + 1  # +1 bias


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = (x - np.max(x)) / max(temperature, 1e-6)
    e = np.exp(x)
    s = float(np.sum(e))
    if s <= 0:
        return np.ones_like(x) / max(len(x), 1)
    return e / s


def _state_to_vector(state: Dict[str, Any]) -> np.ndarray:
    vec = np.zeros(STATE_DIM, dtype=np.float64)
    for i, key in enumerate(STATE_KEYS):
        try:
            vec[i] = float(np.clip(float(state.get(key, 0.0)), -5.0, 5.0))
        except (TypeError, ValueError):
            vec[i] = 0.0
    vec[-1] = 1.0  # bias
    return vec


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------


@dataclass
class ExpertRouter:
    """A single expert specialised to a regime."""

    name: str
    regime: str
    strategies: List[str] = field(default_factory=list)
    _strategy_scores: Dict[str, float] = field(default_factory=dict)
    n_used: int = 0
    cum_pnl: float = 0.0
    ema_pnl: float = 0.0
    ema_alpha: float = 0.1
    # Load-balancing: fraction of routes this expert was in top-k.
    _route_count: int = 0
    _total_routes_ref: List[int] = field(default_factory=lambda: [0])  # shared ref

    def add_strategy(self, strategy_name: str) -> None:
        if strategy_name not in self.strategies:
            self.strategies.append(strategy_name)
            self._strategy_scores.setdefault(strategy_name, 0.0)

    def record(self, pnl: float) -> None:
        self.n_used += 1
        self.cum_pnl += float(pnl)
        self.ema_pnl = (1 - self.ema_alpha) * self.ema_pnl + self.ema_alpha * float(pnl)

    def mark_used(self) -> None:
        self._route_count += 1

    @property
    def utilisation(self) -> float:
        total = self._total_routes_ref[0]
        return self._route_count / max(total, 1)

    def pick(self, weight: float) -> Dict[str, float]:
        if not self.strategies or weight <= 0:
            return {}
        scores = np.array(
            [self._strategy_scores.get(s, 0.0) for s in self.strategies],
            dtype=np.float64,
        )
        if float(np.max(scores) - np.min(scores)) < 1e-9:
            per = np.ones(len(self.strategies)) / len(self.strategies)
        else:
            per = _softmax(scores)
        return {name: float(weight * per[i]) for i, name in enumerate(self.strategies)}

    def update_strategy_score(self, strategy_name: str, pnl: float) -> None:
        self._strategy_scores.setdefault(strategy_name, 0.0)
        prev = self._strategy_scores[strategy_name]
        self._strategy_scores[strategy_name] = (
            (1 - self.ema_alpha) * prev + self.ema_alpha * float(pnl)
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "regime": self.regime,
            "strategies": list(self.strategies),
            "strategy_scores": dict(self._strategy_scores),
            "n_used": int(self.n_used),
            "cum_pnl": float(self.cum_pnl),
            "ema_pnl": float(self.ema_pnl),
            "utilisation": float(self.utilisation),
        }


# ---------------------------------------------------------------------------
# Gating network
# ---------------------------------------------------------------------------


@dataclass
class GatingNetwork:
    """Linear softmax gating with temperature annealing and EMA momentum."""

    n_experts: int
    state_dim: int = STATE_DIM
    lr: float = 0.05
    momentum: float = 0.9          # EMA momentum on weight updates
    load_balance_coeff: float = 0.01
    W: np.ndarray = field(init=False)
    _velocity: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(0)
        self.W = rng.normal(0.0, 0.01, size=(self.state_dim, max(self.n_experts, 1)))
        self._velocity = np.zeros_like(self.W)

    def resize(self, n_experts: int) -> None:
        """Add a new column for a new expert without resetting existing weights."""
        if n_experts <= self.n_experts:
            return
        rng = np.random.default_rng(n_experts)  # reproducible per count
        extra = n_experts - self.n_experts
        new_cols = rng.normal(0.0, 0.01, size=(self.state_dim, extra))
        self.W = np.concatenate([self.W, new_cols], axis=1)
        self._velocity = np.concatenate(
            [self._velocity, np.zeros((self.state_dim, extra))], axis=1
        )
        self.n_experts = n_experts

    def weights(
        self,
        state_vec: np.ndarray,
        temperature: float = 1.0,
        utilisation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        logits = state_vec @ self.W
        # Load-balancing penalty: push down over-used experts.
        if utilisation is not None and len(utilisation) == self.n_experts:
            mean_util = float(np.mean(utilisation))
            penalty = self.load_balance_coeff * (utilisation - mean_util)
            logits = logits - penalty
        return _softmax(logits, temperature=temperature)

    def update(
        self,
        state_vec: np.ndarray,
        expert_indices: Sequence[int],
        reward: float,
        temperature: float = 1.0,
        utilisation: Optional[np.ndarray] = None,
    ) -> None:
        if not expert_indices:
            return
        probs = self.weights(state_vec, temperature=temperature, utilisation=utilisation)
        target = probs.copy()
        share = float(reward) / len(expert_indices)
        for idx in expert_indices:
            if 0 <= idx < self.n_experts:
                target[idx] += share
        target = target - float(np.mean(target))
        grad = np.outer(state_vec, (target - probs))
        # EMA momentum update.
        self._velocity = self.momentum * self._velocity + (1 - self.momentum) * grad
        self.W += self.lr * self._velocity
        self.W = np.clip(self.W, -5.0, 5.0)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_experts": int(self.n_experts),
            "state_dim": int(self.state_dim),
            "W_norm": float(np.linalg.norm(self.W)),
            "lr": float(self.lr),
            "momentum": float(self.momentum),
        }


# ---------------------------------------------------------------------------
# MoE router
# ---------------------------------------------------------------------------


class MoEStrategyRouter:
    """Mixture-of-Experts router with temperature annealing and regime-lock."""

    def __init__(
        self,
        top_k: int = 2,
        seed: Optional[int] = None,
        temp_init: float = 2.0,
        temp_min: float = 0.5,
        temp_decay_rate: float = 0.9995,
        regime_lock_threshold: float = 0.90,
    ) -> None:
        self.top_k = int(max(1, top_k))
        self._experts: List[ExpertRouter] = []
        self._expert_by_name: Dict[str, int] = {}
        self._regime_to_experts: Dict[str, List[int]] = {}
        self._strategy_to_experts: Dict[str, List[int]] = {}
        self._gate = GatingNetwork(n_experts=0)
        self._rng = np.random.default_rng(seed)
        self._n_routes = 0
        self._total_routes = [0]  # shared reference for utilisation
        # Temperature annealing.
        self._temperature = float(temp_init)
        self._temp_min = float(temp_min)
        self._temp_decay = float(temp_decay_rate)
        # Regime-lock.
        self._lock_threshold = float(regime_lock_threshold)

    # -- registration --------------------------------------------------------

    def add_expert(self, name: str, regime: str) -> ExpertRouter:
        if name in self._expert_by_name:
            return self._experts[self._expert_by_name[name]]
        expert = ExpertRouter(name=name, regime=regime)
        expert._total_routes_ref = self._total_routes
        idx = len(self._experts)
        self._expert_by_name[name] = idx
        self._experts.append(expert)
        self._regime_to_experts.setdefault(regime, []).append(idx)
        self._gate.resize(len(self._experts))
        logger.debug("MoE: added expert %s regime=%s idx=%d", name, regime, idx)
        return expert

    def register_strategy(self, name: str, regime: str) -> None:
        assigned = False
        for idx, expert in enumerate(self._experts):
            if expert.regime == regime:
                expert.add_strategy(name)
                self._strategy_to_experts.setdefault(name, []).append(idx)
                assigned = True
        if not assigned:
            expert = self.add_expert(f"expert_{regime}", regime)
            expert.add_strategy(name)
            self._strategy_to_experts.setdefault(name, []).append(
                self._expert_by_name[expert.name]
            )

    # -- routing -------------------------------------------------------------

    def route(
        self,
        state: Dict[str, Any],
        regime: Optional[str] = None,
        regime_confidence: float = 0.0,
    ) -> Dict[str, float]:
        if not self._experts:
            return {}

        vec = _state_to_vector(state)
        self._n_routes += 1
        self._total_routes[0] += 1

        # Regime-lock: bypass gating when confidence is very high.
        locked_indices: Optional[List[int]] = None
        if (
            regime is not None
            and regime_confidence >= self._lock_threshold
            and regime in self._regime_to_experts
            and self._regime_to_experts[regime]
        ):
            locked_indices = self._regime_to_experts[regime]
            logger.debug(
                "MoE: regime-lock active regime=%s conf=%.2f experts=%s",
                regime, regime_confidence, locked_indices,
            )

        utilisation = np.array(
            [e.utilisation for e in self._experts], dtype=np.float64
        )

        if locked_indices:
            # Hard-pin locked experts with equal weight.
            weights = np.zeros(len(self._experts))
            for idx in locked_indices:
                weights[idx] = 1.0 / len(locked_indices)
            top_idx = locked_indices[:self.top_k]
        else:
            weights = self._gate.weights(
                vec,
                temperature=self._temperature,
                utilisation=utilisation,
            )
            top_idx = list(np.argsort(weights)[::-1][: self.top_k])

        # Anneal temperature.
        self._temperature = max(
            self._temp_min, self._temperature * self._temp_decay
        )

        active: Dict[str, float] = {}
        for idx in top_idx:
            expert = self._experts[int(idx)]
            expert.mark_used()
            picks = expert.pick(float(weights[idx]))
            for s, w in picks.items():
                active[s] = active.get(s, 0.0) + w

        total = sum(active.values())
        if total > 0:
            active = {k: v / total for k, v in active.items()}

        return active

    # -- outcome updates -----------------------------------------------------

    def record_outcome(
        self,
        experts_used: Sequence[str],
        pnl: float,
        state: Optional[Dict[str, Any]] = None,
        strategy_name: Optional[str] = None,
    ) -> None:
        expert_indices: List[int] = []
        for name in experts_used:
            idx = self._expert_by_name.get(name)
            if idx is None:
                continue
            expert_indices.append(idx)
            self._experts[idx].record(float(pnl))
            if strategy_name is not None:
                self._experts[idx].update_strategy_score(strategy_name, float(pnl))

        if state is not None and expert_indices:
            vec = _state_to_vector(state)
            utilisation = np.array(
                [e.utilisation for e in self._experts], dtype=np.float64
            )
            self._gate.update(
                vec, expert_indices, float(pnl),
                temperature=self._temperature,
                utilisation=utilisation,
            )

    # -- inspection ----------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_experts": len(self._experts),
            "top_k": int(self.top_k),
            "n_routes": int(self._n_routes),
            "temperature": float(self._temperature),
            "experts": [e.snapshot() for e in self._experts],
            "gate": self._gate.snapshot(),
            "strategy_to_experts": {
                k: list(v) for k, v in self._strategy_to_experts.items()
            },
        }


__all__ = ["ExpertRouter", "GatingNetwork", "MoEStrategyRouter"]
