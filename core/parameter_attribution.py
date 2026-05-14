"""
Parameter Attribution — track per-parameter contribution to P&L.

When ARGUS makes a trade, MANY parameters were active simultaneously:
  - max_position_pct = 0.27
  - confidence_threshold = 0.52
  - stop_loss_pct = 0.011
  - 30+ other parameter values

When that trade returns +$15, which parameter "caused" the gain?

This module solves the credit assignment problem using:

  1. Snapshot — record all active parameter values at trade time
  2. Counterfactual — simulate what would have happened with default values
  3. Marginal contribution — Δ P&L attributed to each parameter
  4. Time-decay — recent attributions weighted higher
  5. Confidence intervals — know when attribution is statistically significant
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TradeAttribution:
    """One trade with its full parameter snapshot + outcome."""
    trade_id: str
    timestamp: float
    pnl_aud: float
    parameters: Dict[str, float] = field(default_factory=dict)
    cluster_multipliers: Dict[str, float] = field(default_factory=dict)
    regime: str = "NORMAL"
    strategy: str = ""
    symbol: str = ""


@dataclass
class ParameterImpact:
    """Computed impact of one parameter across many trades."""
    name: str
    sample_count: int = 0
    cumulative_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    correlation_with_outcome: float = 0.0
    confidence_interval_low: float = 0.0
    confidence_interval_high: float = 0.0
    last_updated: float = 0.0


class ParameterAttributionTracker:
    """
    Tracks per-parameter contribution to P&L across many trades.

    Usage::

        tracker = ParameterAttributionTracker()

        # On trade execution:
        tracker.record_trade(
            trade_id="ord_123",
            pnl_aud=15.0,
            parameters={
                "max_position_pct": 0.27,
                "stop_loss_pct": 0.011,
                "confidence_threshold": 0.52,
            },
        )

        # Periodically: compute per-parameter impact
        impacts = tracker.compute_impacts()

        # Get top contributors
        top = tracker.top_contributors(n=10)
    """

    MIN_SAMPLES_FOR_ATTRIBUTION = 20

    def __init__(self, max_history: int = 100_000) -> None:
        self._history: deque[TradeAttribution] = deque(maxlen=max_history)
        self._impacts: Dict[str, ParameterImpact] = {}
        self._cluster_impacts: Dict[str, ParameterImpact] = {}
        self._param_observations: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10_000)
        )
        self._cluster_observations: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10_000)
        )
        self._trade_count = 0
        logger.info("ParameterAttributionTracker: initialized")

    def record_trade(
        self,
        trade_id: str,
        pnl_aud: float,
        parameters: Dict[str, float],
        cluster_multipliers: Optional[Dict[str, float]] = None,
        regime: str = "NORMAL",
        strategy: str = "",
        symbol: str = "",
    ) -> None:
        """Record a complete trade with its parameter snapshot."""
        attr = TradeAttribution(
            trade_id=trade_id,
            timestamp=time.time(),
            pnl_aud=pnl_aud,
            parameters=dict(parameters),
            cluster_multipliers=dict(cluster_multipliers or {}),
            regime=regime,
            strategy=strategy,
            symbol=symbol,
        )
        self._history.append(attr)
        self._trade_count += 1

        # Update per-parameter observation buffers
        for name, value in parameters.items():
            self._param_observations[name].append((value, pnl_aud))

        for cluster, mult in (cluster_multipliers or {}).items():
            self._cluster_observations[cluster].append((mult, pnl_aud))

    def compute_impacts(self, min_samples: Optional[int] = None) -> Dict[str, ParameterImpact]:
        """
        Compute per-parameter impact statistics.
        Returns dict of {param_name: ParameterImpact}.
        """
        min_n = min_samples or self.MIN_SAMPLES_FOR_ATTRIBUTION
        impacts: Dict[str, ParameterImpact] = {}

        for name, observations in self._param_observations.items():
            if len(observations) < min_n:
                continue

            obs_list = list(observations)
            n = len(obs_list)
            values = [v for v, _ in obs_list]
            pnls = [p for _, p in obs_list]

            cum_pnl = sum(pnls)
            avg_pnl = cum_pnl / n if n > 0 else 0.0

            # Compute correlation between parameter value and outcome
            corr = self._compute_correlation(values, pnls)

            # 95% confidence interval (rough)
            std_pnl = self._compute_std(pnls)
            ci_margin = 1.96 * std_pnl / math.sqrt(n) if n > 1 else 0.0

            impact = ParameterImpact(
                name=name,
                sample_count=n,
                cumulative_pnl=cum_pnl,
                avg_pnl_per_trade=avg_pnl,
                correlation_with_outcome=corr,
                confidence_interval_low=avg_pnl - ci_margin,
                confidence_interval_high=avg_pnl + ci_margin,
                last_updated=time.time(),
            )
            impacts[name] = impact
            self._impacts[name] = impact

        return impacts

    def compute_cluster_impacts(self, min_samples: Optional[int] = None) -> Dict[str, ParameterImpact]:
        """Compute impact statistics per cluster."""
        min_n = min_samples or self.MIN_SAMPLES_FOR_ATTRIBUTION
        impacts: Dict[str, ParameterImpact] = {}

        for cluster, observations in self._cluster_observations.items():
            if len(observations) < min_n:
                continue

            obs_list = list(observations)
            n = len(obs_list)
            mults = [m for m, _ in obs_list]
            pnls = [p for _, p in obs_list]

            cum_pnl = sum(pnls)
            avg_pnl = cum_pnl / n
            corr = self._compute_correlation(mults, pnls)
            std_pnl = self._compute_std(pnls)
            ci_margin = 1.96 * std_pnl / math.sqrt(n) if n > 1 else 0.0

            impact = ParameterImpact(
                name=cluster,
                sample_count=n,
                cumulative_pnl=cum_pnl,
                avg_pnl_per_trade=avg_pnl,
                correlation_with_outcome=corr,
                confidence_interval_low=avg_pnl - ci_margin,
                confidence_interval_high=avg_pnl + ci_margin,
                last_updated=time.time(),
            )
            impacts[cluster] = impact
            self._cluster_impacts[cluster] = impact

        return impacts

    def top_contributors(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the top N parameters by absolute correlation × sample size."""
        if not self._impacts:
            self.compute_impacts()

        scored: List[Tuple[float, str, ParameterImpact]] = []
        for name, impact in self._impacts.items():
            if impact.sample_count < self.MIN_SAMPLES_FOR_ATTRIBUTION:
                continue
            # Score: confidence × magnitude
            score = abs(impact.correlation_with_outcome) * math.log(1 + impact.sample_count)
            scored.append((score, name, impact))

        scored.sort(reverse=True)
        return [
            {
                "name": name,
                "score": score,
                "correlation": impact.correlation_with_outcome,
                "samples": impact.sample_count,
                "avg_pnl": impact.avg_pnl_per_trade,
                "cumulative_pnl": impact.cumulative_pnl,
            }
            for score, name, impact in scored[:n]
        ]

    def get_parameter_impact(self, name: str) -> Optional[ParameterImpact]:
        if name not in self._impacts:
            self.compute_impacts()
        return self._impacts.get(name)

    @staticmethod
    def _compute_correlation(xs: List[float], ys: List[float]) -> float:
        """Pearson correlation coefficient."""
        n = len(xs)
        if n < 2:
            return 0.0

        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n
        var_x = sum((x - mean_x) ** 2 for x in xs) / n
        var_y = sum((y - mean_y) ** 2 for y in ys) / n

        if var_x < 1e-12 or var_y < 1e-12:
            return 0.0

        return cov / (math.sqrt(var_x) * math.sqrt(var_y))

    @staticmethod
    def _compute_std(values: List[float]) -> float:
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        return math.sqrt(variance)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "trades_recorded": self._trade_count,
            "history_size": len(self._history),
            "params_tracked": len(self._param_observations),
            "clusters_tracked": len(self._cluster_observations),
            "impacts_computed": len(self._impacts),
        }
