"""
Meta-Learner — ARGUS optimizes HOW it optimizes.

This is the layer above the self-optimizer. While the self-optimizer
tunes strategies, the meta-learner tunes the optimization PROCESS itself:

- Evolver hyperparameters (mutation rate, crossover rate, population size)
- Generator tree depth and operator mix
- Promotion pipeline thresholds
- Conviction sizer weights
- Self-optimizer sensitivity

It learns which optimization settings produce the best strategies
over time, creating a feedback loop that accelerates improvement.

Architecture:
  Meta-Learner
    ├── tracks: evolver generation → best fitness trajectory
    ├── tracks: generator generation → discovery rate
    ├── tracks: promotion pipeline → success/failure rate
    ├── tracks: live strategy → realized Sharpe per optimization config
    └── adjusts: hyperparameters every N cycles based on what worked

This is "learning to learn" — the system gets better at getting better.
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HyperparamConfig:
    """A snapshot of optimization hyperparameters."""
    evolver_mutation_rate: float = 0.3
    evolver_crossover_rate: float = 0.5
    evolver_population_size: int = 200
    generator_tree_depth: int = 3
    generator_population_size: int = 30
    promotion_min_sharpe: float = 0.3
    conviction_max_multiplier: float = 3.0
    optimizer_interval: int = 100
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evolver_mutation_rate": self.evolver_mutation_rate,
            "evolver_crossover_rate": self.evolver_crossover_rate,
            "evolver_population_size": self.evolver_population_size,
            "generator_tree_depth": self.generator_tree_depth,
            "generator_population_size": self.generator_population_size,
            "promotion_min_sharpe": self.promotion_min_sharpe,
            "conviction_max_multiplier": self.conviction_max_multiplier,
            "optimizer_interval": self.optimizer_interval,
        }


@dataclass
class MetaObservation:
    """One observation: config → outcome."""
    config: HyperparamConfig
    evolver_best_fitness: float = 0.0
    generator_best_fitness: float = 0.0
    promotion_success_rate: float = 0.0
    live_avg_sharpe: float = 0.0
    discovery_rate: float = 0.0     # new strategies per 100 cycles
    cycles_observed: int = 0

    @property
    def score(self) -> float:
        """Composite score of this configuration."""
        return (
            self.evolver_best_fitness * 0.25
            + self.generator_best_fitness * 0.20
            + self.live_avg_sharpe * 0.30
            + self.discovery_rate * 0.15
            + self.promotion_success_rate * 0.10
        )


class MetaLearner:
    """
    Optimizes the optimization process itself.

    Tracks which hyperparameter configurations produce the best outcomes,
    then adjusts parameters toward better configurations.

    Uses a simple bandit-style approach: explore new configs occasionally,
    exploit the best-known config most of the time.
    """

    def __init__(
        self,
        adjustment_interval: int = 1000,    # cycles between meta-adjustments
        explore_rate: float = 0.2,          # 20% explore, 80% exploit
        history_capacity: int = 50,
    ):
        self._interval = adjustment_interval
        self._explore_rate = explore_rate
        self._history: List[MetaObservation] = []
        self._capacity = history_capacity
        self._current_config = HyperparamConfig()
        self._cycle_count = 0
        self._last_adjustment = 0

        # Running metrics for current config period
        self._period_evolver_fitness: List[float] = []
        self._period_generator_fitness: List[float] = []
        self._period_promotions = 0
        self._period_submissions = 0
        self._period_live_pnls: List[float] = []
        self._period_discoveries = 0

    def record_evolver_result(self, best_fitness: float) -> None:
        self._period_evolver_fitness.append(best_fitness)

    def record_generator_result(self, best_fitness: float) -> None:
        self._period_generator_fitness.append(best_fitness)

    def record_promotion(self, success: bool) -> None:
        self._period_submissions += 1
        if success:
            self._period_promotions += 1

    def record_live_trade(self, pnl: float) -> None:
        self._period_live_pnls.append(pnl)

    def record_discovery(self) -> None:
        self._period_discoveries += 1

    def step(self, cycle: int, advisory: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Called every cycle. Returns adjustment directives when it's time.
        """
        self._cycle_count = cycle
        if cycle - self._last_adjustment < self._interval:
            return None

        self._last_adjustment = cycle
        return self._meta_optimize()

    def _meta_optimize(self) -> Dict[str, Any]:
        """Run one meta-optimization step."""
        # Record observation for current config
        obs = MetaObservation(
            config=HyperparamConfig(**self._current_config.to_dict()),
            evolver_best_fitness=max(self._period_evolver_fitness) if self._period_evolver_fitness else 0,
            generator_best_fitness=max(self._period_generator_fitness) if self._period_generator_fitness else 0,
            promotion_success_rate=self._period_promotions / max(self._period_submissions, 1),
            live_avg_sharpe=(sum(self._period_live_pnls) / len(self._period_live_pnls)) if self._period_live_pnls else 0,
            discovery_rate=self._period_discoveries / max(self._interval / 100, 1),
            cycles_observed=self._interval,
        )
        self._history.append(obs)
        if len(self._history) > self._capacity:
            self._history = self._history[-self._capacity:]

        # Reset period counters
        self._period_evolver_fitness.clear()
        self._period_generator_fitness.clear()
        self._period_promotions = 0
        self._period_submissions = 0
        self._period_live_pnls.clear()
        self._period_discoveries = 0

        # Find best historical config
        if len(self._history) < 3:
            return {"action": "explore", "reason": "insufficient_history"}

        best_obs = max(self._history, key=lambda o: o.score)
        current_score = obs.score

        import random
        if random.random() < self._explore_rate:
            # Explore: perturb best config randomly
            new_config = self._perturb_config(best_obs.config)
            action = "explore"
        else:
            # Exploit: move toward best config
            new_config = self._interpolate_configs(self._current_config, best_obs.config, alpha=0.3)
            action = "exploit"

        self._current_config = new_config

        adjustments = {
            "action": action,
            "current_score": current_score,
            "best_score": best_obs.score,
            "new_config": new_config.to_dict(),
            "history_size": len(self._history),
        }

        logger.info(
            "MetaLearner: %s (score=%.3f → best=%.3f) — adjusting %d hyperparams",
            action, current_score, best_obs.score, len(new_config.to_dict()),
        )
        return adjustments

    def _perturb_config(self, base: HyperparamConfig) -> HyperparamConfig:
        """Randomly perturb a config for exploration."""
        import random
        rng = random.Random()
        return HyperparamConfig(
            evolver_mutation_rate=max(0.05, min(0.8, base.evolver_mutation_rate * rng.uniform(0.7, 1.3))),
            evolver_crossover_rate=max(0.1, min(0.9, base.evolver_crossover_rate * rng.uniform(0.8, 1.2))),
            evolver_population_size=max(20, min(500, int(base.evolver_population_size * rng.uniform(0.7, 1.3)))),
            generator_tree_depth=max(2, min(6, base.generator_tree_depth + rng.choice([-1, 0, 1]))),
            generator_population_size=max(10, min(100, int(base.generator_population_size * rng.uniform(0.7, 1.3)))),
            promotion_min_sharpe=max(0.0, min(1.0, base.promotion_min_sharpe * rng.uniform(0.7, 1.3))),
            conviction_max_multiplier=max(1.5, min(5.0, base.conviction_max_multiplier * rng.uniform(0.8, 1.2))),
            optimizer_interval=max(50, min(500, int(base.optimizer_interval * rng.uniform(0.8, 1.2)))),
        )

    def _interpolate_configs(self, current: HyperparamConfig, target: HyperparamConfig,
                             alpha: float = 0.3) -> HyperparamConfig:
        """Move current config toward target by alpha fraction."""
        def lerp(a, b, t):
            return a + (b - a) * t
        return HyperparamConfig(
            evolver_mutation_rate=lerp(current.evolver_mutation_rate, target.evolver_mutation_rate, alpha),
            evolver_crossover_rate=lerp(current.evolver_crossover_rate, target.evolver_crossover_rate, alpha),
            evolver_population_size=int(lerp(current.evolver_population_size, target.evolver_population_size, alpha)),
            generator_tree_depth=round(lerp(current.generator_tree_depth, target.generator_tree_depth, alpha)),
            generator_population_size=int(lerp(current.generator_population_size, target.generator_population_size, alpha)),
            promotion_min_sharpe=lerp(current.promotion_min_sharpe, target.promotion_min_sharpe, alpha),
            conviction_max_multiplier=lerp(current.conviction_max_multiplier, target.conviction_max_multiplier, alpha),
            optimizer_interval=int(lerp(current.optimizer_interval, target.optimizer_interval, alpha)),
        )

    def get_current_config(self) -> HyperparamConfig:
        return self._current_config

    def get_stats(self) -> Dict[str, Any]:
        return {
            "history_size": len(self._history),
            "current_config": self._current_config.to_dict(),
            "best_score": max((o.score for o in self._history), default=0),
            "avg_score": sum(o.score for o in self._history) / max(len(self._history), 1),
            "last_adjustment_cycle": self._last_adjustment,
        }


# ════════════════════════════════════════════════════════════════════════════
# Market Memory — recall similar historical conditions
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class MarketState:
    """A snapshot of market conditions at a point in time."""
    timestamp: float
    regime: str
    volatility: float
    trend_strength: float       # -1 (strong down) to +1 (strong up)
    volume_ratio: float         # current vol / avg vol
    rsi: float
    spread_bps: float
    correlation_btc_eth: float
    best_strategy: str
    outcome_pnl: float = 0.0   # what happened next (filled after N cycles)

    def feature_vector(self) -> Tuple[float, ...]:
        regime_map = {"trending": 1.0, "ranging": 0.0, "volatile": -1.0}
        return (
            regime_map.get(self.regime, 0.0),
            self.volatility,
            self.trend_strength,
            self.volume_ratio,
            self.rsi / 100.0,
            min(self.spread_bps / 10.0, 1.0),
            self.correlation_btc_eth,
        )


class MarketMemory:
    """
    Episodic memory of market conditions and what worked.

    When current conditions match a historical episode, recall which
    strategies performed best and adjust accordingly.

    Uses k-NN in feature space (no ML dependency).
    """

    def __init__(self, capacity: int = 2000, k_nearest: int = 10):
        self._memories: List[MarketState] = []
        self._capacity = capacity
        self._k = k_nearest

    def remember(self, state: MarketState) -> None:
        self._memories.append(state)
        if len(self._memories) > self._capacity:
            self._memories.pop(0)

    def recall(self, current: MarketState) -> Dict[str, Any]:
        """Find similar past conditions and what worked."""
        if len(self._memories) < self._k + 5:
            return {"similar_count": 0, "recommended_strategy": "", "expected_pnl": 0.0}

        current_vec = current.feature_vector()
        distances = []
        for mem in self._memories:
            if mem.outcome_pnl == 0.0:
                continue  # skip memories without outcome
            vec = mem.feature_vector()
            d = sum((a - b) ** 2 for a, b in zip(current_vec, vec)) ** 0.5
            distances.append((d, mem))

        if not distances:
            return {"similar_count": 0, "recommended_strategy": "", "expected_pnl": 0.0}

        distances.sort(key=lambda x: x[0])
        nearest = distances[:self._k]

        # What strategy worked best in similar conditions?
        strategy_pnl: Dict[str, List[float]] = defaultdict(list)
        for _, mem in nearest:
            strategy_pnl[mem.best_strategy].append(mem.outcome_pnl)

        # Find strategy with best avg PnL in similar conditions
        best_strat = ""
        best_avg = -999.0
        for strat, pnls in strategy_pnl.items():
            avg = sum(pnls) / len(pnls)
            if avg > best_avg:
                best_avg = avg
                best_strat = strat

        # Expected PnL from distance-weighted average
        total_w = 0.0
        weighted_pnl = 0.0
        for d, mem in nearest:
            w = 1.0 / max(d, 1e-9)
            weighted_pnl += w * mem.outcome_pnl
            total_w += w
        expected_pnl = weighted_pnl / total_w if total_w > 0 else 0.0

        # Similarity score (inverse of average distance)
        avg_dist = sum(d for d, _ in nearest) / len(nearest)
        similarity = 1.0 / (1.0 + avg_dist)

        return {
            "similar_count": len(nearest),
            "recommended_strategy": best_strat,
            "expected_pnl": expected_pnl,
            "similarity": similarity,
            "avg_distance": avg_dist,
            "strategies_seen": dict({s: len(p) for s, p in strategy_pnl.items()}),
        }

    def update_outcome(self, timestamp: float, pnl: float) -> None:
        """Update the most recent memory with its outcome."""
        for mem in reversed(self._memories):
            if abs(mem.timestamp - timestamp) < 60 and mem.outcome_pnl == 0.0:
                mem.outcome_pnl = pnl
                break

    def size(self) -> int:
        return len(self._memories)

    def get_stats(self) -> Dict[str, Any]:
        with_outcome = sum(1 for m in self._memories if m.outcome_pnl != 0.0)
        return {
            "total_memories": len(self._memories),
            "with_outcomes": with_outcome,
            "regimes_seen": list(set(m.regime for m in self._memories)),
        }


# ════════════════════════════════════════════════════════════════════════════
# Information-Theoretic Signal Filter
# ════════════════════════════════════════════════════════════════════════════

class EntropyFilter:
    """
    Filters signals using information theory.

    High entropy in recent returns → market is random → don't trade.
    Low entropy → market has structure → trade with confidence.

    Also computes mutual information between signal sources to identify
    redundant signals (two signals that always agree add no information).
    """

    def __init__(self, window: int = 50, entropy_threshold: float = 0.85, n_bins: int = 10):
        self._window = window
        self._threshold = entropy_threshold
        self._n_bins = n_bins
        self._returns_history: List[float] = []
        self._signal_history: Dict[str, List[float]] = defaultdict(list)

    def record_return(self, ret: float) -> None:
        self._returns_history.append(ret)
        if len(self._returns_history) > self._window * 3:
            self._returns_history = self._returns_history[-self._window * 2:]

    def record_signal(self, source: str, value: float) -> None:
        self._signal_history[source].append(value)
        if len(self._signal_history[source]) > self._window * 3:
            self._signal_history[source] = self._signal_history[source][-self._window * 2:]

    def should_trade(self) -> Tuple[bool, float]:
        """Returns (should_trade, entropy_score). Trade when entropy < threshold."""
        if len(self._returns_history) < self._window:
            return True, 0.5  # insufficient data, allow trading

        recent = self._returns_history[-self._window:]
        entropy = self._compute_entropy(recent)
        normalised = entropy / math.log(self._n_bins) if self._n_bins > 1 else 0.5

        return normalised < self._threshold, normalised

    def signal_redundancy(self, source_a: str, source_b: str) -> float:
        """Compute mutual information between two signal sources.
        Returns 0 (independent) to 1 (fully redundant)."""
        hist_a = self._signal_history.get(source_a, [])
        hist_b = self._signal_history.get(source_b, [])
        if len(hist_a) < 20 or len(hist_b) < 20:
            return 0.0

        n = min(len(hist_a), len(hist_b))
        a = hist_a[-n:]
        b = hist_b[-n:]

        # Discretise into bins
        a_bins = self._discretise(a)
        b_bins = self._discretise(b)

        # Joint and marginal distributions
        joint: Dict[Tuple[int, int], int] = defaultdict(int)
        marg_a: Dict[int, int] = defaultdict(int)
        marg_b: Dict[int, int] = defaultdict(int)

        for ai, bi in zip(a_bins, b_bins):
            joint[(ai, bi)] += 1
            marg_a[ai] += 1
            marg_b[bi] += 1

        # Mutual information
        mi = 0.0
        for (ai, bi), count in joint.items():
            p_ab = count / n
            p_a = marg_a[ai] / n
            p_b = marg_b[bi] / n
            if p_ab > 0 and p_a > 0 and p_b > 0:
                mi += p_ab * math.log(p_ab / (p_a * p_b))

        # Normalise by min entropy
        h_a = self._compute_entropy(a)
        h_b = self._compute_entropy(b)
        max_mi = min(h_a, h_b)
        return mi / max_mi if max_mi > 0 else 0.0

    def _compute_entropy(self, values: List[float]) -> float:
        if not values:
            return 0.0
        counts = [0] * self._n_bins
        lo, hi = min(values), max(values)
        rng = hi - lo if hi != lo else 1.0
        for v in values:
            idx = min(self._n_bins - 1, int((v - lo) / rng * self._n_bins))
            counts[idx] += 1
        n = len(values)
        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / n
                entropy -= p * math.log(p)
        return entropy

    def _discretise(self, values: List[float]) -> List[int]:
        if not values:
            return []
        lo, hi = min(values), max(values)
        rng = hi - lo if hi != lo else 1.0
        return [min(self._n_bins - 1, int((v - lo) / rng * self._n_bins)) for v in values]

    def get_stats(self) -> Dict[str, Any]:
        should, entropy = self.should_trade()
        return {
            "should_trade": should,
            "entropy": entropy,
            "returns_recorded": len(self._returns_history),
            "signals_tracked": len(self._signal_history),
        }
