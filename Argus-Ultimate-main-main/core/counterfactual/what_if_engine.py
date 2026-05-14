"""
What-If engine: counterfactual replay of past decisions.

Given a decision journal entry (pre-trade state + actual action + outcome),
replay the decision with alternative actions via ``core/world_model.py``
and compare the counterfactual outcome to the actual outcome.

This gives ARGUS "I could have made more money by doing X" signals that
feed the Thompson bandit router for much faster arm updates than waiting
for real fills.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# WhatIfEngine
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class CounterfactualResult:
    actual_action: Any
    alternative_action: Any
    actual_reward: float
    alternative_reward: float
    improvement: float  # alternative - actual (positive = alternative was better)
    uncertainty: float  # world model uncertainty
    horizon: int


class WhatIfEngine:
    """
    Counterfactual simulator wired to ``core/world_model.py``.

    Parameters
    ----------
    world_model : Any
        Instance of ``core/world_model.WorldModel`` (or any object with
        ``predict_next(state, action) → (next_state, reward, uncertainty)``).
    """

    def __init__(self, world_model: Any) -> None:
        if not hasattr(world_model, "predict_next"):
            raise ValueError("world_model must implement predict_next()")
        self.world_model = world_model

    def simulate(
        self,
        state: np.ndarray,
        actual_action: Any,
        alternative_action: Any,
        actual_reward: Optional[float] = None,
        horizon: int = 1,
    ) -> CounterfactualResult:
        """
        Simulate both actual and alternative action trajectories.

        Parameters
        ----------
        state : np.ndarray
            Pre-decision state vector.
        actual_action : Any
            The action that was actually taken (can be anything the world
            model understands).
        alternative_action : Any
            The counterfactual action.
        actual_reward : float, optional
            If known from decision journal, use this instead of re-simulating.
        horizon : int
            Number of steps to simulate forward.

        Returns
        -------
        CounterfactualResult
        """
        # Simulate the actual action if not provided
        if actual_reward is None:
            actual_traj = self._rollout_single(state, actual_action, horizon)
            actual_r = float(actual_traj["total_reward"])
            actual_u = float(actual_traj["mean_uncertainty"])
        else:
            actual_r = float(actual_reward)
            actual_u = 0.0

        # Simulate the alternative
        alt_traj = self._rollout_single(state, alternative_action, horizon)
        alt_r = float(alt_traj["total_reward"])
        alt_u = float(alt_traj["mean_uncertainty"])

        return CounterfactualResult(
            actual_action=actual_action,
            alternative_action=alternative_action,
            actual_reward=actual_r,
            alternative_reward=alt_r,
            improvement=alt_r - actual_r,
            uncertainty=0.5 * (actual_u + alt_u),
            horizon=horizon,
        )

    def _rollout_single(
        self, state: np.ndarray, action: Any, horizon: int
    ) -> Dict[str, Any]:
        """Single-action rollout for ``horizon`` steps."""
        try:
            # Use the world model's rollout method if available
            if hasattr(self.world_model, "rollout"):
                result = self.world_model.rollout(state, [action] * horizon)
                return {
                    "total_reward": float(getattr(result, "total_reward", 0.0)),
                    "mean_uncertainty": float(getattr(result, "mean_uncertainty", 0.0)),
                    "states": getattr(result, "states", None),
                }
        except Exception as exc:
            logger.debug("world_model.rollout failed: %s", exc)

        # Manual rollout via predict_next
        current = state.copy()
        total_reward = 0.0
        uncertainties: List[float] = []
        for _ in range(horizon):
            try:
                next_state, r, u = self.world_model.predict_next(current, action)
                total_reward += float(r)
                uncertainties.append(float(u))
                current = next_state
            except Exception as exc:
                logger.debug("world_model.predict_next failed: %s", exc)
                break
        return {
            "total_reward": total_reward,
            "mean_uncertainty": float(np.mean(uncertainties)) if uncertainties else 0.0,
        }

    def evaluate_action_space(
        self,
        state: np.ndarray,
        candidate_actions: Sequence[Any],
        horizon: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Score every candidate action in the given state.

        Returns a list sorted by expected reward (descending).
        """
        results = []
        for action in candidate_actions:
            traj = self._rollout_single(state, action, horizon)
            results.append({
                "action": action,
                "expected_reward": traj["total_reward"],
                "uncertainty": traj["mean_uncertainty"],
            })
        return sorted(results, key=lambda r: -r["expected_reward"])


# ═════════════════════════════════════════════════════════════════════════════
# Stateless convenience
# ═════════════════════════════════════════════════════════════════════════════


def simulate_counterfactual(
    world_model: Any,
    state: np.ndarray,
    actual_action: Any,
    alternative_action: Any,
    actual_reward: Optional[float] = None,
    horizon: int = 1,
) -> CounterfactualResult:
    """Convenience function: one-shot counterfactual simulation."""
    engine = WhatIfEngine(world_model)
    return engine.simulate(
        state=state,
        actual_action=actual_action,
        alternative_action=alternative_action,
        actual_reward=actual_reward,
        horizon=horizon,
    )
