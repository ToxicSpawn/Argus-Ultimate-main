"""
Off-policy policy evaluation via importance sampling.

Given a historical dataset of (state, action, reward, action_probability),
estimate the expected value of a NEW policy without actually running it
in live trading.

Used to safely compare candidate strategies before deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# PolicyEvaluator
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class PolicyValue:
    estimated_value: float
    n_samples: int
    ess: float  # effective sample size
    variance: float


class PolicyEvaluator:
    """
    Off-policy policy evaluator using importance sampling.

    Parameters
    ----------
    historical_data : List[Dict]
        Each entry has: ``state``, ``action``, ``reward``, ``action_prob``.
        ``action_prob`` is the probability under the behavior policy
        (whatever took the action).
    """

    def __init__(self, historical_data: List[Dict[str, Any]]) -> None:
        self.data = list(historical_data)

    def evaluate(
        self,
        new_policy: Callable[[Any], Dict[Any, float]],
        *,
        clip_ratio: float = 10.0,
    ) -> PolicyValue:
        """
        Estimate the value of ``new_policy`` on the historical data.

        Parameters
        ----------
        new_policy : Callable[[state], Dict[action, probability]]
            Function mapping state to a distribution over actions.
        clip_ratio : float
            Maximum importance weight to clip (reduces variance at cost
            of bias).

        Returns
        -------
        PolicyValue
        """
        if not self.data:
            return PolicyValue(0.0, 0, 0.0, 0.0)

        weighted_returns: List[float] = []
        weights: List[float] = []

        for sample in self.data:
            state = sample["state"]
            action = sample["action"]
            reward = float(sample["reward"])
            behavior_prob = float(sample.get("action_prob", 1.0))
            if behavior_prob <= 1e-9:
                continue

            new_probs = new_policy(state)
            new_prob = float(new_probs.get(action, 0.0))
            ratio = min(new_prob / behavior_prob, clip_ratio)
            weighted_returns.append(ratio * reward)
            weights.append(ratio)

        if not weighted_returns:
            return PolicyValue(0.0, 0, 0.0, 0.0)

        mean = float(np.mean(weighted_returns))
        var = float(np.var(weighted_returns))
        weights_arr = np.array(weights)
        w_sum = float(weights_arr.sum())
        w_sq_sum = float(np.sum(weights_arr ** 2))
        ess = (w_sum ** 2) / max(w_sq_sum, 1e-9)

        return PolicyValue(
            estimated_value=mean,
            n_samples=len(weighted_returns),
            ess=ess,
            variance=var,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Stateless convenience
# ═════════════════════════════════════════════════════════════════════════════


def importance_sampled_value(
    historical_data: List[Dict[str, Any]],
    new_policy: Callable[[Any], Dict[Any, float]],
    *,
    clip_ratio: float = 10.0,
) -> PolicyValue:
    """One-shot off-policy evaluation."""
    return PolicyEvaluator(historical_data).evaluate(new_policy, clip_ratio=clip_ratio)
