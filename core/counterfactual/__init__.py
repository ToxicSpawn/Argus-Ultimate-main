"""
Counterfactual simulation engine.

"What if I had done X instead?" — uses the learned world model
(``core/world_model.py``) to replay past decisions with alternative actions
and compute expected regret.

- ``what_if_engine``: replay a decision with an alternative action
- ``regret_computation``: per-decision and aggregate regret
- ``policy_evaluator``: off-policy policy evaluation via importance sampling
"""

from .what_if_engine import WhatIfEngine, simulate_counterfactual
from .regret_computation import RegretComputation, compute_regret
from .policy_evaluator import PolicyEvaluator, importance_sampled_value

__all__ = [
    "WhatIfEngine",
    "simulate_counterfactual",
    "RegretComputation",
    "compute_regret",
    "PolicyEvaluator",
    "importance_sampled_value",
]
