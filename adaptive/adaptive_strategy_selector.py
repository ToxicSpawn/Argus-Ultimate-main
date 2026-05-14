"""
Adaptive Strategy Selector - Dynamic strategy selection based on market conditions.

Implements intelligent strategy evaluation, capital allocation, rotation,
conflict resolution, and performance tracking for multi-strategy portfolios.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    CRISIS = "crisis"


@dataclass
class StrategyPerformance:
    """Performance metrics for a single strategy."""
    strategy_name: str
    returns: np.ndarray
    sharpe: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    recent_pnl: float
    regime_performance: Dict[str, float]
    last_updated: datetime

    def __post_init__(self):
        if not isinstance(self.returns, np.ndarray):
            self.returns = np.array(self.returns)


@dataclass
class StrategyScore:
    """Score and recommendation for a strategy."""
    strategy_name: str
    score: float
    components: Dict[str, float]
    recommendation: str
    confidence: float

    def __post_init__(self):
        self.score = np.clip(self.score, 0.0, 100.0)
        self.confidence = np.clip(self.confidence, 0.0, 1.0)


@dataclass
class StrategyChange:
    """Proposed change to strategy allocation."""
    strategy_name: str
    action: str
    current_weight: float
    target_weight: float
    reason: str
    confidence: float


@dataclass
class RotationResult:
    """Result of a strategy rotation."""
    timestamp: datetime
    changes: List[StrategyChange]
    expected_improvement: float
    rotation_cost: float
    new_allocation: Dict[str, float]


@dataclass
class Conflict:
    """Conflict between strategies."""
    strategy_a: str
    strategy_b: str
    conflict_type: str
    correlation: float
    severity: float


class RegimeStrategyMapping:
    """Maps market regimes to recommended strategies with weights."""

    def __init__(self):
        self.regime_to_strategies: Dict[MarketRegime, List[str]] = {
            MarketRegime.BULLISH: ["momentum", "trend_following", "breakout"],
            MarketRegime.BEARISH: ["mean_reversion", "short_selling", "defensive"],
            MarketRegime.RANGING: ["mean_reversion", "grid_trading", "pairs_trading"],
            MarketRegime.HIGH_VOLATILITY: ["volatility_arb", "breakout", "momentum"],
            MarketRegime.LOW_VOLATILITY: ["carry_trade", "mean_reversion", "grid_trading"],
            MarketRegime.TRENDING_UP: ["trend_following", "momentum", "breakout"],
            MarketRegime.TRENDING_DOWN: ["short_selling", "inverse_etf", "defensive"],
            MarketRegime.CRISIS: ["tail_hedge", "cash", "defensive"],
        }
        self.strategy_weights: Dict[MarketRegime, Dict[str, float]] = {}
        self._initialize_default_weights()

    def _initialize_default_weights(self):
        for regime, strategies in self.regime_to_strategies.items():
            n = len(strategies)
            self.strategy_weights[regime] = {s: 1.0 / n for s in strategies}

    def get_recommended_strategies(self, regime: MarketRegime) -> List[str]:
        return self.regime_to_strategies.get(regime, [])

    def get_strategy_weight(self, regime: MarketRegime, strategy: str) -> float:
        return self.strategy_weights.get(regime, {}).get(strategy, 0.0)

    def update_mapping(
        self,
        regime: MarketRegime,
        strategy: str,
        performance: StrategyPerformance,
    ) -> None:
        if regime not in self.regime_to_strategies:
            self.regime_to_strategies[regime] = []
        if strategy not in self.regime_to_strategies[regime]:
            self.regime_to_strategies[regime].append(strategy)

        if regime not in self.strategy_weights:
            self.strategy_weights[regime] = {}

        current_weights = self.strategy_weights[regime]
        perf_score = (
            0.4 * performance.sharpe
            + 0.3 * performance.win_rate
            + 0.2 * performance.profit_factor
            - 0.1 * abs(performance.max_drawdown)
        )
        perf_score = max(perf_score, 0.0)

        current_weights[strategy] = perf_score

        total = sum(current_weights.values())
        if total > 0:
            self.strategy_weights[regime] = {
                s: w / total for s, w in current_weights.items()
            }

        logger.info(
            "Updated regime mapping for %s: strategy=%s, perf_score=%.2f",
            regime.value,
            strategy,
            perf_score,
        )


class StrategyEvaluator:
    """Evaluates and ranks strategies based on performance metrics."""

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self._performance_cache: Dict[str, StrategyPerformance] = {}

    def evaluate_strategy(
        self,
        strategy: Any,
        lookback_days: int = 90,
    ) -> StrategyPerformance:
        returns = self._get_strategy_returns(strategy, lookback_days)
        sharpe = self._compute_sharpe(returns)
        max_dd = self._compute_max_drawdown(returns)
        win_rate = self._compute_win_rate(returns)
        profit_factor = self._compute_profit_factor(returns)
        recent_pnl = float(np.sum(returns[-20:])) if len(returns) >= 20 else float(np.sum(returns))
        regime_perf = self._compute_regime_performance(strategy, returns)
        last_updated = datetime.now()

        perf = StrategyPerformance(
            strategy_name=strategy if isinstance(strategy, str) else getattr(strategy, "name", "unknown"),
            returns=returns,
            sharpe=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            recent_pnl=recent_pnl,
            regime_performance=regime_perf,
            last_updated=last_updated,
        )
        self._performance_cache[perf.strategy_name] = perf
        return perf

    def compute_regime_fitness(self, strategy: str, regime: MarketRegime) -> float:
        perf = self._performance_cache.get(strategy)
        if perf is None:
            return 0.0
        regime_key = regime.value
        return perf.regime_performance.get(regime_key, 0.0)

    def compute_consistency_score(self, strategy: str) -> float:
        perf = self._performance_cache.get(strategy)
        if perf is None or len(perf.returns) < 10:
            return 0.0
        returns = perf.returns
        rolling_sharpe = self._compute_rolling_sharpe(returns, window=20)
        if len(rolling_sharpe) < 2:
            return 0.0
        consistency = 1.0 - min(np.std(rolling_sharpe) / max(np.mean(np.abs(rolling_sharpe)), 1e-6), 1.0)
        return float(np.clip(consistency, 0.0, 1.0))

    def compute_recovery_score(self, strategy: str) -> float:
        perf = self._performance_cache.get(strategy)
        if perf is None or len(perf.returns) < 10:
            return 0.0
        returns = perf.returns
        drawdowns = self._compute_drawdown_series(returns)
        recovery_times = []
        in_drawdown = False
        drawdown_start = 0
        for i, dd in enumerate(drawdowns):
            if dd < -0.05 and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
            elif dd >= -0.01 and in_drawdown:
                recovery_times.append(i - drawdown_start)
                in_drawdown = False
        if not recovery_times:
            return 0.5
        avg_recovery = np.mean(recovery_times)
        score = max(0.0, 1.0 - (avg_recovery / len(returns)))
        return float(np.clip(score, 0.0, 1.0))

    def rank_strategies(
        self,
        strategies: List[str],
        regime: MarketRegime,
    ) -> List[StrategyScore]:
        scores = []
        for strategy in strategies:
            perf = self._performance_cache.get(strategy)
            if perf is None:
                continue

            regime_fitness = self.compute_regime_fitness(strategy, regime)
            consistency = self.compute_consistency_score(strategy)
            recovery = self.compute_recovery_score(strategy)

            sharpe_norm = np.clip((perf.sharpe + 1) / 4, 0, 1)
            win_norm = np.clip(perf.win_rate, 0, 1)
            pf_norm = np.clip(perf.profit_factor / 3, 0, 1)
            dd_norm = np.clip(1 - abs(perf.max_drawdown) / 0.5, 0, 1)

            score = (
                0.25 * regime_fitness
                + 0.20 * sharpe_norm
                + 0.15 * win_norm
                + 0.15 * pf_norm
                + 0.10 * dd_norm
                + 0.10 * consistency
                + 0.05 * recovery
            ) * 100

            if score >= 75:
                recommendation = "use"
            elif score >= 50:
                recommendation = "reduce"
            elif score >= 25:
                recommendation = "monitor"
            else:
                recommendation = "disable"

            confidence = min(
                0.9,
                0.5 + 0.1 * min(len(perf.returns) / 100, 1.0) + 0.2 * consistency,
            )

            scores.append(
                StrategyScore(
                    strategy_name=strategy,
                    score=float(score),
                    components={
                        "regime_fitness": regime_fitness,
                        "sharpe": sharpe_norm,
                        "win_rate": win_norm,
                        "profit_factor": pf_norm,
                        "drawdown": dd_norm,
                        "consistency": consistency,
                        "recovery": recovery,
                    },
                    recommendation=recommendation,
                    confidence=confidence,
                )
            )

        scores.sort(key=lambda s: s.score, reverse=True)
        return scores

    def _get_strategy_returns(self, strategy: Any, lookback_days: int) -> np.ndarray:
        if hasattr(strategy, "get_returns"):
            returns = strategy.get_returns(lookback_days)
            return np.array(returns) if returns is not None else np.array([])
        return np.random.randn(lookback_days) * 0.01

    def _compute_sharpe(self, returns: np.ndarray) -> float:
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        excess = np.mean(returns) - self.risk_free_rate / 252
        return float(excess / np.std(returns) * np.sqrt(252))

    def _compute_max_drawdown(self, returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))

    def _compute_win_rate(self, returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.5
        return float(np.mean(returns > 0))

    def _compute_profit_factor(self, returns: np.ndarray) -> float:
        gains = returns[returns > 0]
        losses = np.abs(returns[returns < 0])
        if np.sum(losses) == 0:
            return float(np.sum(gains)) if len(gains) > 0 else 1.0
        return float(np.sum(gains) / np.sum(losses))

    def _compute_regime_performance(
        self, strategy: Any, returns: np.ndarray
    ) -> Dict[str, float]:
        if hasattr(strategy, "get_regime_performance"):
            return strategy.get_regime_performance()
        return {r.value: 0.5 for r in MarketRegime}

    def _compute_rolling_sharpe(self, returns: np.ndarray, window: int) -> np.ndarray:
        if len(returns) < window:
            return np.array([])
        rolling_mean = np.convolve(returns, np.ones(window) / window, mode="valid")
        rolling_std = np.array(
            [np.std(returns[i : i + window]) for i in range(len(returns) - window + 1)]
        )
        rolling_std = np.where(rolling_std == 0, 1e-6, rolling_std)
        return rolling_mean / rolling_std * np.sqrt(252)

    def _compute_drawdown_series(self, returns: np.ndarray) -> np.ndarray:
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        return (cumulative - running_max) / running_max


class AdaptiveAllocator:
    """Allocates capital across strategies using Kelly criterion and risk parity."""

    def __init__(
        self,
        max_single_allocation: float = 0.40,
        min_allocation: float = 0.05,
        kelly_fraction: float = 0.25,
    ):
        self.max_single_allocation = max_single_allocation
        self.min_allocation = min_allocation
        self.kelly_fraction = kelly_fraction

    def allocate_capital(
        self,
        strategies: List[StrategyScore],
        total_capital: float,
    ) -> Dict[str, float]:
        if not strategies:
            return {}

        kelly_allocations = {}
        for s in strategies:
            if s.recommendation == "disable":
                kelly_allocations[s.strategy_name] = 0.0
                continue
            win_rate = s.components.get("win_rate", 0.5)
            avg_win = s.components.get("profit_factor", 1.0)
            avg_loss = 1.0
            kelly = self.kelly_criterion(win_rate, avg_win, avg_loss)
            fk = self.fractional_kelly(kelly, self.kelly_fraction)
            kelly_allocations[s.strategy_name] = max(fk, self.min_allocation)

        total_kelly = sum(kelly_allocations.values())
        if total_kelly == 0:
            n = len(strategies)
            return {s.strategy_name: total_capital / n for s in strategies}

        allocations = {}
        for name, alloc in kelly_allocations.items():
            normalized = alloc / total_kelly
            allocations[name] = min(normalized, self.max_single_allocation)

        total_alloc = sum(allocations.values())
        if total_alloc > 0:
            allocations = {k: v / total_alloc for k, v in allocations.items()}

        return {k: v * total_capital for k, v in allocations.items()}

    def kelly_criterion(
        self, win_rate: float, avg_win: float, avg_loss: float
    ) -> float:
        if avg_loss == 0:
            return 0.0
        b = avg_win / avg_loss
        q = 1.0 - win_rate
        kelly = (b * win_rate - q) / b
        return float(np.clip(kelly, 0.0, 1.0))

    def fractional_kelly(self, kelly_pct: float, fraction: float = 0.25) -> float:
        return kelly_pct * fraction

    def risk_parity_allocation(
        self,
        strategies: List[str],
        volatilities: Dict[str, float],
    ) -> Dict[str, float]:
        if not strategies:
            return {}

        inv_vol = {s: 1.0 / max(volatilities.get(s, 1.0), 1e-6) for s in strategies}
        total_inv_vol = sum(inv_vol.values())
        return {s: w / total_inv_vol for s, w in inv_vol.items()}


class StrategyRotator:
    """Manages strategy rotation with cost-aware gradual transitions."""

    def __init__(
        self,
        min_rotation_threshold: float = 10.0,
        max_rotation_per_step: float = 0.10,
        rotation_cost_factor: float = 0.001,
    ):
        self.min_rotation_threshold = min_rotation_threshold
        self.max_rotation_per_step = max_rotation_per_step
        self.rotation_cost_factor = rotation_cost_factor

    def should_rotate(
        self,
        current_strategies: List[str],
        new_ranking: List[StrategyScore],
    ) -> bool:
        if not current_strategies or not new_ranking:
            return False

        current_scores = {s.strategy_name: s.score for s in new_ranking}
        avg_current = np.mean(
            [current_scores.get(s, 50.0) for s in current_strategies]
        )
        top_new = [s.score for s in new_ranking[: len(current_strategies)]]
        avg_new = np.mean(top_new) if top_new else 0.0

        return (avg_new - avg_current) > self.min_rotation_threshold

    def compute_rotation_cost(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
    ) -> float:
        total_change = sum(abs(target.get(k, 0) - current.get(k, 0)) for k in set(current) | set(target))
        return total_change * self.rotation_cost_factor

    def gradual_rotation(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
        steps: int = 5,
    ) -> List[Dict[str, float]]:
        all_strategies = set(current) | set(target)
        path = []
        for step in range(1, steps + 1):
            alpha = step / steps
            intermediate = {}
            for s in all_strategies:
                curr = current.get(s, 0.0)
                tgt = target.get(s, 0.0)
                intermediate[s] = curr + alpha * (tgt - curr)
            total = sum(intermediate.values())
            if total > 0:
                intermediate = {k: v / total for k, v in intermediate.items()}
            path.append(intermediate)
        return path


class PerformanceTracker:
    """Tracks strategy performance over time with rolling metrics."""

    def __init__(self, default_window: int = 20):
        self.default_window = default_window
        self._trade_history: Dict[str, List[Dict[str, Any]]] = {}
        self._returns_history: Dict[str, List[float]] = {}

    def track_trade(self, strategy: str, trade: Dict[str, Any]) -> None:
        if strategy not in self._trade_history:
            self._trade_history[strategy] = []
            self._returns_history[strategy] = []

        self._trade_history[strategy].append(trade)
        pnl = trade.get("pnl", 0.0)
        self._returns_history[strategy].append(pnl)

    def get_strategy_stats(self, strategy: str) -> Optional[StrategyPerformance]:
        if strategy not in self._returns_history:
            return None

        returns = np.array(self._returns_history[strategy])
        if len(returns) == 0:
            return None

        sharpe = self._compute_sharpe(returns)
        max_dd = self._compute_max_drawdown(returns)
        win_rate = float(np.mean(returns > 0)) if len(returns) > 0 else 0.5
        profit_factor = self._compute_profit_factor(returns)
        recent_pnl = float(np.sum(returns[-self.default_window:]))
        last_updated = datetime.now()

        return StrategyPerformance(
            strategy_name=strategy,
            returns=returns,
            sharpe=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            recent_pnl=recent_pnl,
            regime_performance={},
            last_updated=last_updated,
        )

    def compute_rolling_metrics(
        self, strategy: str, window: Optional[int] = None
    ) -> Dict[str, float]:
        w = window or self.default_window
        returns = self._returns_history.get(strategy, [])
        if len(returns) < w:
            return {"sharpe": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "volatility": 0.0}

        recent = np.array(returns[-w:])
        return {
            "sharpe": self._compute_sharpe(recent),
            "win_rate": float(np.mean(recent > 0)),
            "profit_factor": self._compute_profit_factor(recent),
            "volatility": float(np.std(recent)),
            "mean_return": float(np.mean(recent)),
            "skewness": float(self._compute_skewness(recent)),
            "kurtosis": float(self._compute_kurtosis(recent)),
        }

    def detect_strategy_decay(
        self, strategy: str, threshold: float = 0.15
    ) -> bool:
        returns = self._returns_history.get(strategy, [])
        if len(returns) < 40:
            return False

        recent = np.array(returns[-20:])
        older = np.array(returns[-40:-20])

        recent_perf = np.mean(recent)
        older_perf = np.mean(older)

        if older_perf == 0:
            return False

        decay = (older_perf - recent_perf) / abs(older_perf)
        return decay > threshold

    def _compute_sharpe(self, returns: np.ndarray) -> float:
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(252))

    def _compute_max_drawdown(self, returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))

    def _compute_profit_factor(self, returns: np.ndarray) -> float:
        gains = returns[returns > 0]
        losses = np.abs(returns[returns < 0])
        if np.sum(losses) == 0:
            return float(np.sum(gains)) if len(gains) > 0 else 1.0
        return float(np.sum(gains) / np.sum(losses))

    def _compute_skewness(self, returns: np.ndarray) -> float:
        if len(returns) < 3:
            return 0.0
        std = np.std(returns)
        if std == 0:
            return 0.0
        return float(np.mean(((returns - np.mean(returns)) / std) ** 3))

    def _compute_kurtosis(self, returns: np.ndarray) -> float:
        if len(returns) < 4:
            return 0.0
        std = np.std(returns)
        if std == 0:
            return 0.0
        return float(np.mean(((returns - np.mean(returns)) / std) ** 4) - 3)


class StrategyConflictResolver:
    """Detects and resolves conflicts between strategies."""

    def __init__(self, correlation_threshold: float = 0.7):
        self.correlation_threshold = correlation_threshold

    def detect_conflicts(
        self, strategies: List[str], returns_data: Optional[Dict[str, np.ndarray]] = None
    ) -> List[Conflict]:
        if len(strategies) < 2:
            return []

        corr_matrix = self.check_correlation(strategies, returns_data)
        conflicts = []

        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                corr = corr_matrix[i, j]
                if abs(corr) > self.correlation_threshold:
                    severity = abs(corr) - self.correlation_threshold
                    conflict_type = "high_positive_correlation" if corr > 0 else "hedging_conflict"
                    conflicts.append(
                        Conflict(
                            strategy_a=strategies[i],
                            strategy_b=strategies[j],
                            conflict_type=conflict_type,
                            correlation=float(corr),
                            severity=float(severity),
                        )
                    )

        return conflicts

    def resolve_conflicts(
        self,
        strategies: List[str],
        market_state: Optional[Dict[str, Any]] = None,
        returns_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[str]:
        conflicts = self.detect_conflicts(strategies, returns_data)
        if not conflicts:
            return strategies

        conflict_pairs = set()
        for c in conflicts:
            conflict_pairs.add((c.strategy_a, c.strategy_b))

        strategy_scores = {s: 0.0 for s in strategies}
        for c in conflicts:
            if c.correlation > 0:
                strategy_scores[c.strategy_a] -= c.severity
                strategy_scores[c.strategy_b] -= c.severity
            else:
                strategy_scores[c.strategy_a] += c.severity * 0.5
                strategy_scores[c.strategy_b] += c.severity * 0.5

        resolved = sorted(strategies, key=lambda s: strategy_scores[s], reverse=True)

        for c in conflicts:
            if c.correlation > 0.85:
                if strategy_scores[c.strategy_a] > strategy_scores[c.strategy_b]:
                    resolved = [s for s in resolved if s != c.strategy_b]
                else:
                    resolved = [s for s in resolved if s != c.strategy_a]

        logger.info("Resolved conflicts: %d strategies removed", len(strategies) - len(resolved))
        return resolved

    def check_correlation(
        self,
        strategies: List[str],
        returns_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        n = len(strategies)
        corr_matrix = np.eye(n)

        if returns_data is None:
            returns_data = {s: np.random.randn(100) * 0.01 for s in strategies}

        for i in range(n):
            for j in range(i + 1, n):
                ret_i = returns_data.get(strategies[i], np.array([]))
                ret_j = returns_data.get(strategies[j], np.array([]))
                if len(ret_i) > 1 and len(ret_j) > 1:
                    min_len = min(len(ret_i), len(ret_j))
                    corr = np.corrcoef(ret_i[:min_len], ret_j[:min_len])[0, 1]
                    if not np.isnan(corr):
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr

        return corr_matrix

    def diversification_score(self, strategies: List[str], returns_data: Optional[Dict[str, np.ndarray]] = None) -> float:
        if len(strategies) < 2:
            return 1.0

        corr_matrix = self.check_correlation(strategies, returns_data)
        n = len(strategies)
        off_diag = []
        for i in range(n):
            for j in range(i + 1, n):
                off_diag.append(corr_matrix[i, j])

        if not off_diag:
            return 1.0

        avg_corr = np.mean(np.abs(off_diag))
        return float(np.clip(1.0 - avg_corr, 0.0, 1.0))


class AdaptiveStrategySelector:
    """Main class for adaptive strategy selection and rotation."""

    def __init__(
        self,
        total_capital: float = 1_000_000.0,
        max_active_strategies: int = 5,
        min_rotation_threshold: float = 10.0,
        rotation_steps: int = 5,
    ):
        self.total_capital = total_capital
        self.max_active_strategies = max_active_strategies
        self.rotation_steps = rotation_steps

        self._strategies: Dict[str, Any] = {}
        self._strategy_configs: Dict[str, Dict[str, Any]] = {}
        self._active_strategies: List[str] = []
        self._allocation: Dict[str, float] = {}
        self._performance_history: List[Dict[str, Any]] = []

        self.regime_mapping = RegimeStrategyMapping()
        self.evaluator = StrategyEvaluator()
        self.allocator = AdaptiveAllocator()
        self.rotator = StrategyRotator(min_rotation_threshold=min_rotation_threshold)
        self.tracker = PerformanceTracker()
        self.conflict_resolver = StrategyConflictResolver()

        self._current_regime: Optional[MarketRegime] = None
        self._last_rotation: Optional[datetime] = None
        self._rotation_history: List[RotationResult] = []

        logger.info(
            "AdaptiveStrategySelector initialized: capital=%.0f, max_strategies=%d",
            total_capital,
            max_active_strategies,
        )

    def register_strategy(
        self, name: str, strategy: Any, config: Optional[Dict[str, Any]] = None
    ) -> None:
        self._strategies[name] = strategy
        self._strategy_configs[name] = config or {}
        logger.info("Registered strategy: %s", name)

    def update(
        self, market_data: Optional[Dict[str, Any]] = None, regime: Optional[MarketRegime] = None
    ) -> List[StrategyScore]:
        if regime is not None:
            self._current_regime = regime

        if self._current_regime is None:
            logger.warning("No regime set, using default ranking")
            return self._rank_all_strategies(MarketRegime.RANGING)

        for name in self._strategies:
            try:
                perf = self.evaluator.evaluate_strategy(name)
                self.regime_mapping.update_mapping(self._current_regime, name, perf)
            except Exception as e:
                logger.error("Error evaluating strategy %s: %s", name, e)

        scores = self._rank_all_strategies(self._current_regime)
        self._active_strategies = [
            s.strategy_name
            for s in scores
            if s.recommendation in ("use", "reduce")
        ][: self.max_active_strategies]

        self._allocation = self.allocator.allocate_capital(scores, self.total_capital)
        self._performance_history.append(
            {
                "timestamp": datetime.now(),
                "regime": self._current_regime.value,
                "scores": {s.strategy_name: s.score for s in scores},
            }
        )

        return scores

    def get_active_strategies(self) -> List[str]:
        return list(self._active_strategies)

    def get_allocation(self) -> Dict[str, float]:
        return dict(self._allocation)

    def get_recommended_changes(self) -> List[StrategyChange]:
        changes = []
        current_set = set(self._active_strategies)

        scores = self._rank_all_strategies(
            self._current_regime or MarketRegime.RANGING
        )
        recommended = {s.strategy_name: s for s in scores[: self.max_active_strategies]}
        recommended_set = set(recommended.keys())

        for name in current_set - recommended_set:
            changes.append(
                StrategyChange(
                    strategy_name=name,
                    action="remove",
                    current_weight=self._allocation.get(name, 0.0),
                    target_weight=0.0,
                    reason=f"Strategy {name} no longer in top {self.max_active_strategies}",
                    confidence=recommended.get(name, StrategyScore(name, 0, {}, "disable", 0.5)).confidence,
                )
            )

        for name in recommended_set - current_set:
            s = recommended[name]
            changes.append(
                StrategyChange(
                    strategy_name=name,
                    action="add",
                    current_weight=0.0,
                    target_weight=self._allocation.get(name, 0.0),
                    reason=f"Strategy {name} entered top {self.max_active_strategies} (score={s.score:.1f})",
                    confidence=s.confidence,
                )
            )

        for name in current_set & recommended_set:
            current_w = self._allocation.get(name, 0.0)
            target_w = self._allocation.get(name, 0.0)
            s = recommended.get(name)
            if s and s.recommendation == "reduce":
                changes.append(
                    StrategyChange(
                        strategy_name=name,
                        action="decrease",
                        current_weight=current_w,
                        target_weight=target_w * 0.7,
                        reason=f"Strategy {name} score declining (score={s.score:.1f})",
                        confidence=s.confidence,
                    )
                )
            elif s and s.recommendation == "use" and current_w < target_w * 1.1:
                changes.append(
                    StrategyChange(
                        strategy_name=name,
                        action="increase",
                        current_weight=current_w,
                        target_weight=target_w * 1.1,
                        reason=f"Strategy {name} performing well (score={s.score:.1f})",
                        confidence=s.confidence,
                    )
                )

        return changes

    def execute_rotation(self) -> RotationResult:
        changes = self.get_recommended_changes()
        if not changes:
            return RotationResult(
                timestamp=datetime.now(),
                changes=[],
                expected_improvement=0.0,
                rotation_cost=0.0,
                new_allocation=dict(self._allocation),
            )

        current_alloc = {
            name: self._allocation.get(name, 0.0)
            for name in self._active_strategies
        }
        target_alloc = {
            c.strategy_name: c.target_weight for c in changes
        }

        rotation_cost = self.rotator.compute_rotation_cost(current_alloc, target_alloc)

        rotation_path = self.rotator.gradual_rotation(
            current_alloc, target_alloc, self.rotation_steps
        )
        final_alloc = rotation_path[-1] if rotation_path else target_alloc

        self._active_strategies = [
            name for name, alloc in final_alloc.items() if alloc > 0.01
        ]
        self._allocation = final_alloc
        self._last_rotation = datetime.now()

        expected_improvement = self._estimate_improvement(changes)

        result = RotationResult(
            timestamp=datetime.now(),
            changes=changes,
            expected_improvement=expected_improvement,
            rotation_cost=rotation_cost,
            new_allocation=dict(self._allocation),
        )

        self._rotation_history.append(result)
        logger.info(
            "Rotation executed: %d changes, cost=%.4f, expected_improvement=%.2f%%",
            len(changes),
            rotation_cost,
            expected_improvement * 100,
        )

        return result

    def _rank_all_strategies(self, regime: MarketRegime) -> List[StrategyScore]:
        return self.evaluator.rank_strategies(list(self._strategies.keys()), regime)

    def _estimate_improvement(self, changes: List[StrategyChange]) -> float:
        if not changes:
            return 0.0
        total_score_change = sum(
            abs(c.target_weight - c.current_weight) for c in changes
        )
        return float(np.clip(total_score_change * 0.1, 0.0, 0.5))
