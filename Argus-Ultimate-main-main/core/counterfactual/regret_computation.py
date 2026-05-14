"""
Regret computation for counterfactual analysis.

Given a set of past decisions and their counterfactual alternatives, compute
per-decision and aggregate regret.

Regret_i = max_a reward(s_i, a) - reward(s_i, actual_action_i)

High aggregate regret means "we're leaving money on the table" and suggests
policy improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# RegretComputation
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class DecisionRegret:
    decision_id: str
    actual_reward: float
    best_alternative_reward: float
    regret: float
    best_alternative_action: Any
    uncertainty: float


class RegretComputation:
    """
    Compute counterfactual regret for a batch of decisions.

    Parameters
    ----------
    what_if_engine : WhatIfEngine
        Source of counterfactual rollouts.
    candidate_actions : List[Any]
        The set of alternative actions to consider for each decision.
    """

    def __init__(self, what_if_engine: Any, candidate_actions: Sequence[Any]) -> None:
        self.engine = what_if_engine
        self.candidate_actions = list(candidate_actions)

    def compute_batch(
        self,
        decisions: List[Dict[str, Any]],
    ) -> List[DecisionRegret]:
        """
        Compute per-decision regret.

        Parameters
        ----------
        decisions : List[Dict]
            Each decision has ``id``, ``state``, ``action``, ``reward``.
        """
        out: List[DecisionRegret] = []
        for d in decisions:
            state = np.asarray(d["state"], dtype=float)
            actual_action = d["action"]
            actual_reward = float(d.get("reward", 0.0))
            decision_id = str(d.get("id", ""))

            # Score every candidate action
            action_rewards = []
            for alt in self.candidate_actions:
                cf = self.engine.simulate(
                    state=state,
                    actual_action=actual_action,
                    alternative_action=alt,
                    actual_reward=actual_reward,
                    horizon=1,
                )
                action_rewards.append((alt, cf.alternative_reward, cf.uncertainty))

            # Best alternative
            best = max(action_rewards, key=lambda t: t[1])
            best_alt_action = best[0]
            best_alt_reward = float(best[1])
            best_alt_uncertainty = float(best[2])

            regret = max(0.0, best_alt_reward - actual_reward)
            out.append(DecisionRegret(
                decision_id=decision_id,
                actual_reward=actual_reward,
                best_alternative_reward=best_alt_reward,
                regret=regret,
                best_alternative_action=best_alt_action,
                uncertainty=best_alt_uncertainty,
            ))
        return out

    def aggregate(self, regrets: List[DecisionRegret]) -> Dict[str, Any]:
        """Aggregate regret statistics."""
        if not regrets:
            return {
                "n_decisions": 0,
                "total_regret": 0.0,
                "mean_regret": 0.0,
                "max_regret": 0.0,
                "regret_ratio": 0.0,
            }
        regrets_arr = np.array([r.regret for r in regrets])
        actual_arr = np.array([r.actual_reward for r in regrets])
        total_actual = float(np.sum(actual_arr))
        total_regret = float(np.sum(regrets_arr))
        return {
            "n_decisions": len(regrets),
            "total_regret": total_regret,
            "mean_regret": float(np.mean(regrets_arr)),
            "max_regret": float(np.max(regrets_arr)),
            "total_actual_reward": total_actual,
            "regret_ratio": float(total_regret / max(abs(total_actual), 1e-6)),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Stateless convenience
# ═════════════════════════════════════════════════════════════════════════════


def compute_regret(
    world_model: Any,
    decisions: List[Dict[str, Any]],
    candidate_actions: Sequence[Any],
) -> Dict[str, Any]:
    """One-shot regret computation for a batch of decisions."""
    from .what_if_engine import WhatIfEngine

    engine = WhatIfEngine(world_model)
    rc = RegretComputation(engine, candidate_actions)
    regrets = rc.compute_batch(decisions)
    return {
        "per_decision": regrets,
        "aggregate": rc.aggregate(regrets),
    }
