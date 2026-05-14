"""
Metacognitive self-monitoring.

"Am I good at this right now?" — ARGUS aggregates its recent performance
per regime, per symbol, and per strategy into a single 0-1 competence score
that gates trade sizing via `pre_order_check`.

- ``competence_estimator``: combines confidence calibration, recent P&L,
  decision-journal accuracy → competence_score(symbol, regime, strategy)
- ``capability_bounds``: rolling Sharpe per regime → "which regimes am I
  strong/weak in?"
- ``metacognitive_monitor``: singleton collector, snapshot API
"""

from .competence_estimator import CompetenceEstimator, compute_competence_score
from .capability_bounds import CapabilityBounds
from .metacognitive_monitor import (
    MetacognitiveMonitor,
    get_metacognitive_monitor,
)

__all__ = [
    "CompetenceEstimator",
    "compute_competence_score",
    "CapabilityBounds",
    "MetacognitiveMonitor",
    "get_metacognitive_monitor",
]
