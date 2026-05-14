"""
A/B & shadow trading: track baseline vs candidate config for promotion.

Live A/B: run candidate in shadow mode (no real orders), compare metrics to baseline.
Promotion gates already decide when to apply; this module records candidate id and metrics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ShadowCandidate:
    candidate_id: str
    config_snapshot: Dict[str, Any]
    started_utc: float = field(default_factory=time.time)
    metrics: Dict[str, float] = field(default_factory=dict)


class ShadowTracker:
    """
    Track shadow candidate (A/B) for promotion.
    get_current_candidate() / record_candidate_metrics() for live A/B before promotion.
    """

    def __init__(self) -> None:
        self._baseline_metrics: Dict[str, float] = {}
        self._candidate: Optional[ShadowCandidate] = None

    def set_baseline_metrics(self, metrics: Dict[str, float]) -> None:
        """Set baseline (current live) metrics for comparison."""
        self._baseline_metrics = dict(metrics)

    def start_candidate(self, candidate_id: str, config_snapshot: Dict[str, Any]) -> None:
        """Start tracking a shadow candidate (A/B)."""
        self._candidate = ShadowCandidate(
            candidate_id=str(candidate_id),
            config_snapshot=dict(config_snapshot),
        )

    def record_candidate_metrics(self, metrics: Dict[str, float]) -> None:
        """Update candidate metrics (e.g. from shadow paper run)."""
        if self._candidate is not None:
            self._candidate.metrics.update(metrics)

    def get_current_candidate(self) -> Optional[ShadowCandidate]:
        """Return current shadow candidate or None."""
        return self._candidate

    def clear_candidate(self) -> None:
        """Clear candidate after promotion or discard."""
        self._candidate = None

    def shadow_mode(self) -> bool:
        """True if a candidate is being tracked (shadow mode)."""
        return self._candidate is not None
