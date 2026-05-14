"""
Strategy Evolver v3 — absolute elite genetic strategy optimizer.

This is a research-grade evolutionary engine combining techniques from:
- NSGA-II (Deb et al. 2002) — Pareto-dominance multi-objective ranking
- CMA-ES (Hansen & Ostermeier 2001) — covariance-aware self-adaptive mutation
- MAP-Elites (Mouret & Clune 2015) — quality-diversity archive
- Differential Evolution (Storn & Price 1997) — DE/rand/1/bin mutation
- Monte Carlo Cross-Validation — k-fold robustness over single train/test split
- Parameter Sensitivity Analysis — jitter-based robustness scoring

Architecture:
┌─────────────────────────────────────────────────────┐
│                  StrategyEvolver                     │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Island:  │  │ Island:  │  │ Island:  │  ← regime │
│  │ trending │←→│ ranging  │←→│ volatile │    aware  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│       │              │              │                │
│       └──────────────┼──────────────┘                │
│                      ↓                               │
│              ┌───────────────┐                       │
│              │  Pareto Front │ ← NSGA-II ranking     │
│              └───────┬───────┘                       │
│                      ↓                               │
│  ┌────────────────────────────────────────┐          │
│  │          MAP-Elites Archive            │          │
│  │  [strategy_type × regime × symbol]     │          │
│  │  best-per-niche quality-diversity map  │          │
│  └────────────────────┬───────────────────┘          │
│                       ↓                              │
│  ┌────────────────────────────────────────┐          │
│  │          Hall of Fame (immortal)       │          │
│  │  Top 20 all-time, never overwritten    │          │
│  └────────────────────────────────────────┘          │
│                                                     │
│  Operators: self-adaptive Gaussian + Cauchy + DE     │
│  Crossover: BLX-α blend + SBX (simulated binary)    │
│  Validation: k-fold MCCV + robustness jitter         │
│  Anti-fragility: multi-regime fitness bonus           │
└─────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Data classes
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FitnessVector:
    """Multi-objective fitness vector — the foundation of NSGA-II ranking."""
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar: float = 0.0
    profit_factor: float = 0.0
    trade_count: int = 0
    win_rate: float = 0.0
    avg_return_pct: float = 0.0
    oos_sharpe: float = 0.0
    overfitting_score: float = 0.0
    # v3 additions
    robustness: float = 0.0             # 0-1: how stable is fitness under param jitter
    hurst_exponent: float = 0.5         # <0.5 = mean-reverting equity curve (good)
    max_consec_losses: int = 0
    recovery_factor: float = 0.0        # total return / max drawdown
    tail_ratio: float = 0.0             # avg(top 5% returns) / abs(avg(bottom 5%))
    anti_fragility: float = 0.0         # 0-1: works across multiple regimes
    mccv_mean_sharpe: float = 0.0       # mean Sharpe across k-fold CV
    mccv_std_sharpe: float = 0.0        # std of Sharpe across folds (lower = more robust)
    # v4 pinnacle additions
    deflated_sharpe: float = 0.0        # Bailey & Lopez de Prado corrected Sharpe
    reality_check_pval: float = 1.0     # White's Reality Check p-value (lower = more real)
    novelty_score: float = 0.0          # behavioral distance to archive (0-1)
    surrogate_predicted: bool = False   # True if only surrogate-evaluated (not full backtest)
    age: int = 0                        # generations since birth (ALPS)

    @property
    def objectives(self) -> Tuple[float, ...]:
        """The 7 objectives used for NSGA-II Pareto ranking.
        All are MAXIMISED (negate things we want to minimise)."""
        return (
            self.deflated_sharpe if self.deflated_sharpe != 0 else (
                self.mccv_mean_sharpe if self.mccv_mean_sharpe != 0 else self.sharpe),
            self.sortino,
            -self.max_drawdown_pct,
            self.robustness,
            self.anti_fragility,
            self.novelty_score,              # reward behavioral diversity
            -self.reality_check_pval,        # lower p-value = more statistically real
        )

    @property
    def composite(self) -> float:
        """Scalar fallback for non-Pareto contexts."""
        if self.trade_count < 3:
            return -10.0
        dd_pen = max(0.0, self.max_drawdown_pct - 15.0) * 0.05
        overfit_pen = self.overfitting_score * 0.5
        robust_bonus = self.robustness * 0.3
        antifrag_bonus = self.anti_fragility * 0.25
        oos_w = 0.4 if self.oos_sharpe > 0 else 0.2
        mccv_pen = self.mccv_std_sharpe * 0.2
        novelty_bonus = self.novelty_score * 0.15
        # Deflated Sharpe replaces raw when available
        sharpe_to_use = self.deflated_sharpe if self.deflated_sharpe != 0 else self.sharpe
        # Statistical significance bonus
        stat_bonus = 0.2 if self.reality_check_pval < 0.10 else 0.0
        return (
            sharpe_to_use * 0.15
            + self.sortino * 0.10
            + self.oos_sharpe * oos_w
            + self.calmar * 0.05
            + min(self.profit_factor, 3.0) * 0.03
            + min(self.win_rate, 0.7) * 0.02
            + robust_bonus
            + antifrag_bonus
            + novelty_bonus
            + stat_bonus
            - dd_pen
            - overfit_pen
            - mccv_pen
        )


@dataclass(frozen=True)
class StrategyGenome:
    """A single strategy configuration — the 'DNA' of a strategy."""
    strategy_type: str
    params: Dict[str, float]
    symbol: str
    fitness: FitnessVector = field(default_factory=FitnessVector)
    generation: int = 0
    parent_ids: Tuple[str, ...] = ()
    sigma: Dict[str, float] = field(default_factory=dict)
    island: str = "default"
    pareto_rank: int = 999              # NSGA-II: 0 = first front
    crowding_distance: float = 0.0      # NSGA-II: higher = more unique

    @property
    def genome_id(self) -> str:
        return f"{self.strategy_type}:{self.symbol}:{hash(tuple(sorted(self.params.items()))) % 10000:04d}"

    @property
    def composite_fitness(self) -> float:
        return self.fitness.composite


@dataclass(frozen=True)
class EnsembleGenome:
    """A portfolio of strategies — evolved at the portfolio level."""
    members: Tuple[StrategyGenome, ...]
    weights: Tuple[float, ...]
    fitness: FitnessVector = field(default_factory=FitnessVector)
    generation: int = 0

    @property
    def composite_fitness(self) -> float:
        return self.fitness.composite


@dataclass(frozen=True)
class EvolutionResult:
    """Result of one evolution cycle."""
    generation: int
    population_size: int
    best_genome: Optional[StrategyGenome]
    avg_fitness: float
    top_n: List[StrategyGenome]
    mutations: int
    crossovers: int
    duration_ms: float
    islands: Dict[str, int]
    best_ensemble: Optional[EnsembleGenome] = None
    overfitting_rate: float = 0.0
    # v3 additions
    pareto_front_size: int = 0
    archive_size: int = 0
    hall_of_fame_size: int = 0
    avg_robustness: float = 0.0
    avg_anti_fragility: float = 0.0
    stagnation_counter: int = 0
    diversity_metric: float = 0.0
    # v4 pinnacle additions
    avg_deflated_sharpe: float = 0.0
    statistically_significant_pct: float = 0.0  # % with p < 0.10
    novelty_archive_size: int = 0
    surrogate_filter_rate: float = 0.0          # % filtered by surrogate
    transfer_learning_active: bool = False


# ════════════════════════════════════════════════════════════════════════════
# Parameter bounds & constants
# ════════════════════════════════════════════════════════════════════════════

_PARAM_BOUNDS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "breakout": {"lookback": (5, 60), "tp_pct": (0.5, 5.0), "sl_pct": (0.5, 3.0)},
    "vol_spike_reversal": {"vol_mult": (1.2, 4.0), "tp_pct": (0.5, 3.0), "sl_pct": (1.0, 4.0)},
    "mean_reversion": {"bb_std": (1.0, 3.0), "sl_pct": (0.5, 3.0)},
    "momentum": {"fast_period": (5, 30), "trail_pct": (1.0, 5.0)},
    "rsi_mean_reversion": {"rsi_buy": (15, 35), "rsi_sell": (50, 80), "sl_pct": (0.5, 3.0)},
    "macd_crossover": {"fast": (5, 15), "slow": (18, 30), "sl_pct": (0.5, 3.0)},
    "ema_ribbon": {"fast": (3, 15), "slow": (20, 60), "trail_pct": (1.0, 5.0)},
    "keltner_breakout": {"atr_mult": (1.0, 3.0), "tp_pct": (1.0, 4.0), "sl_pct": (0.5, 2.0)},
    "range_fade": {"lookback": (12, 96), "tp_pct": (0.5, 3.0), "sl_pct": (0.5, 2.0)},
    "heikin_ashi_trend": {"consec_green": (2, 6), "trail_pct": (1.0, 5.0)},
    "pivot_bounce": {"tp_pct": (0.5, 3.0), "sl_pct": (0.3, 2.0)},
    "inside_bar_breakout": {"tp_pct": (0.5, 4.0), "sl_pct": (0.3, 2.0)},
    "williams_r_reversal": {"period": (10, 25), "buy_thresh": (-95, -75), "sl_pct": (0.5, 3.0)},
    "stochastic_oversold": {"k_period": (10, 25), "buy_level": (10, 30), "sl_pct": (0.5, 3.0)},
    "adx_momentum": {"adx_thresh": (15, 35), "trail_pct": (1.0, 5.0)},
    "donchian_turtle": {"entry_period": (10, 60), "exit_period": (5, 30)},
    "gap_continuation": {"gap_pct": (0.5, 3.0), "tp_pct": (0.5, 3.0), "sl_pct": (0.3, 1.5)},
    "three_bar_play": {"tp_pct": (0.5, 3.0), "sl_pct": (0.3, 2.0)},
    "dual_thrust": {"k": (0.3, 0.8), "sl_pct": (0.5, 3.0)},
    "opening_range_breakout": {"tp_pct": (0.5, 4.0), "sl_pct": (0.3, 2.0)},
    "quantum_entropy": {"entropy_thresh": (0.3, 0.8), "tp_pct": (0.5, 4.0), "sl_pct": (0.5, 3.0)},
    "quantum_tunneling": {"barrier_std": (1.0, 3.0), "decay": (0.85, 0.99), "sl_pct": (0.5, 3.0)},
    "quantum_superposition": {"lookback": (15, 60), "tp_pct": (0.5, 4.0), "sl_pct": (0.5, 2.0)},
    "quantum_walk_drift": {"walk_len": (15, 120), "sigma_thresh": (1.0, 3.0), "trail_pct": (1.0, 5.0)},
    "quantum_entanglement": {"lag": (1, 5), "tp_pct": (0.5, 3.0), "sl_pct": (0.5, 2.0)},
    "quantum_annealing": {"temp_decay": (0.90, 0.99), "tp_pct": (0.5, 4.0), "sl_pct": (0.5, 3.0)},
    "quantum_decoherence": {"window": (15, 60), "stability_thresh": (0.2, 0.6), "trail_pct": (1.0, 5.0)},
    "quantum_interference": {"fast": (3, 10), "mid": (10, 25), "slow": (25, 60), "sl_pct": (0.5, 3.0)},
}

_INTEGER_PARAMS = frozenset({
    "lookback", "fast", "slow", "mid", "period", "k_period",
    "entry_period", "exit_period", "walk_len", "lag", "consec_green",
    "adx_thresh", "rsi_buy", "rsi_sell", "buy_level", "buy_thresh", "window",
})

_REGIME_STRATEGY_AFFINITY: Dict[str, List[str]] = {
    "trending": [
        "breakout", "momentum", "ema_ribbon", "keltner_breakout",
        "donchian_turtle", "adx_momentum", "heikin_ashi_trend",
        "gap_continuation", "dual_thrust", "opening_range_breakout",
        "quantum_walk_drift", "quantum_tunneling",
    ],
    "ranging": [
        "mean_reversion", "rsi_mean_reversion", "range_fade",
        "pivot_bounce", "williams_r_reversal", "stochastic_oversold",
        "three_bar_play", "quantum_entropy", "quantum_superposition",
        "quantum_decoherence",
    ],
    "volatile": [
        "vol_spike_reversal", "inside_bar_breakout", "macd_crossover",
        "quantum_annealing", "quantum_interference", "quantum_entanglement",
    ],
}


# ════════════════════════════════════════════════════════════════════════════
# Regime detection
# ════════════════════════════════════════════════════════════════════════════

def detect_regime(close: list | tuple, lookback: int = 50) -> str:
    """Lightweight regime detection. Returns 'trending', 'ranging', or 'volatile'."""
    if len(close) < lookback + 10:
        return "ranging"
    recent = close[-lookback:]
    returns = [(recent[i] / recent[i - 1]) - 1 for i in range(1, len(recent))]
    mean_ret = sum(returns) / len(returns)
    variance = sum((r - mean_ret) ** 2 for r in returns) / max(len(returns) - 1, 1)
    vol = variance ** 0.5
    cum_ret = (recent[-1] / recent[0]) - 1 if recent[0] != 0 else 0
    efficiency = abs(cum_ret) / max(vol * (len(returns) ** 0.5), 1e-9)
    if vol > 0.03:
        return "volatile"
    if efficiency > 1.5:
        return "trending"
    return "ranging"


# ════════════════════════════════════════════════════════════════════════════
# NSGA-II: Pareto ranking + crowding distance
# ════════════════════════════════════════════════════════════════════════════

def _dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
    """True if a Pareto-dominates b (all objectives >= and at least one >)."""
    dominated = False
    for ai, bi in zip(a, b):
        if ai < bi:
            return False
        if ai > bi:
            dominated = True
    return dominated


def nsga2_rank(population: List[StrategyGenome]) -> List[StrategyGenome]:
    """Assign pareto_rank and crowding_distance to each genome (NSGA-II)."""
    if not population:
        return []

    n = len(population)
    objs = [g.fitness.objectives for g in population]

    # Fast non-dominated sort
    domination_count = [0] * n
    dominated_set: List[List[int]] = [[] for _ in range(n)]
    fronts: List[List[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(objs[i], objs[j]):
                dominated_set[i].append(j)
            elif _dominates(objs[j], objs[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    rank_map = {}
    fi = 0
    while fronts[fi]:
        next_front = []
        for i in fronts[fi]:
            rank_map[i] = fi
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        fi += 1
        fronts.append(next_front)

    # Crowding distance per front
    crowd = [0.0] * n
    num_obj = len(objs[0]) if objs else 0

    for front in fronts:
        if len(front) < 3:
            for i in front:
                crowd[i] = float("inf")
            continue
        for m in range(num_obj):
            sorted_idx = sorted(front, key=lambda i: objs[i][m])
            crowd[sorted_idx[0]] = float("inf")
            crowd[sorted_idx[-1]] = float("inf")
            obj_range = objs[sorted_idx[-1]][m] - objs[sorted_idx[0]][m]
            if obj_range < 1e-12:
                continue
            for k in range(1, len(sorted_idx) - 1):
                crowd[sorted_idx[k]] += (
                    (objs[sorted_idx[k + 1]][m] - objs[sorted_idx[k - 1]][m]) / obj_range
                )

    # Rebuild population with rank and crowding
    result = []
    for i, g in enumerate(population):
        result.append(StrategyGenome(
            strategy_type=g.strategy_type,
            params=g.params,
            symbol=g.symbol,
            fitness=g.fitness,
            generation=g.generation,
            parent_ids=g.parent_ids,
            sigma=g.sigma,
            island=g.island,
            pareto_rank=rank_map.get(i, 999),
            crowding_distance=crowd[i],
        ))
    return result


def nsga2_select(population: List[StrategyGenome], rng: random.Random, k: int = 2) -> StrategyGenome:
    """NSGA-II tournament: prefer lower rank; break ties by higher crowding distance."""
    candidates = rng.sample(population, min(k, len(population)))
    return min(candidates, key=lambda g: (g.pareto_rank, -g.crowding_distance))


# ════════════════════════════════════════════════════════════════════════════
# Fitness computation with equity curve analysis
# ════════════════════════════════════════════════════════════════════════════

def _hurst_exponent(series: List[float], max_lag: int = 20) -> float:
    """Simplified rescaled range Hurst exponent estimate."""
    if len(series) < max_lag + 5:
        return 0.5
    n = len(series)
    lags = range(2, min(max_lag + 1, n // 2))
    rs_list = []
    for lag in lags:
        chunks = [series[i:i + lag] for i in range(0, n - lag + 1, lag)]
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            m = sum(chunk) / len(chunk)
            devs = [x - m for x in chunk]
            cumdev = []
            s = 0.0
            for d in devs:
                s += d
                cumdev.append(s)
            r = max(cumdev) - min(cumdev) if cumdev else 0.0
            std = (sum(d ** 2 for d in devs) / len(devs)) ** 0.5
            if std > 1e-12:
                rs_list.append((lag, r / std))
    if len(rs_list) < 3:
        return 0.5
    # Log-log regression
    log_lags = [math.log(lag) for lag, _ in rs_list]
    log_rs = [math.log(max(rs, 1e-12)) for _, rs in rs_list]
    n_pts = len(log_lags)
    mean_x = sum(log_lags) / n_pts
    mean_y = sum(log_rs) / n_pts
    ss_xx = sum((x - mean_x) ** 2 for x in log_lags)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_lags, log_rs))
    if ss_xx < 1e-12:
        return 0.5
    return max(0.0, min(1.0, ss_xy / ss_xx))


def _tail_ratio(trades: List[float], pct: float = 0.05) -> float:
    """Ratio of avg top returns to abs(avg bottom returns)."""
    if len(trades) < 20:
        return 1.0
    sorted_t = sorted(trades)
    k = max(1, int(len(trades) * pct))
    bottom = sorted_t[:k]
    top = sorted_t[-k:]
    avg_top = sum(top) / len(top)
    avg_bottom = abs(sum(bottom) / len(bottom))
    return avg_top / max(avg_bottom, 1e-9)


def _max_consecutive_losses(trades: List[float]) -> int:
    """Count longest consecutive losing streak."""
    max_streak = 0
    current = 0
    for t in trades:
        if t < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def compute_fitness_vector(
    trades: List[float],
    oos_trades: Optional[List[float]] = None,
    robustness: float = 0.0,
    anti_fragility: float = 0.0,
    mccv_sharpes: Optional[List[float]] = None,
) -> FitnessVector:
    """Compute full fitness vector from trade P&L list.

    Args:
        trades: in-sample trade returns (%)
        oos_trades: out-of-sample trade returns
        robustness: pre-computed robustness score (0-1)
        anti_fragility: pre-computed anti-fragility score (0-1)
        mccv_sharpes: list of Sharpe ratios from k-fold cross-validation
    """
    if not trades or len(trades) < 3:
        return FitnessVector(trade_count=len(trades) if trades else 0)

    n = len(trades)
    mean_ret = sum(trades) / n
    variance = sum((t - mean_ret) ** 2 for t in trades) / max(n - 1, 1)
    std = variance ** 0.5
    sharpe = mean_ret / max(std, 1e-9)

    # Sortino
    downside = [min(0, t - mean_ret) for t in trades]
    down_var = sum(d ** 2 for d in downside) / max(n - 1, 1)
    sortino = mean_ret / max(down_var ** 0.5, 1e-9)

    # Max drawdown
    equity = [0.0]
    for t in trades:
        equity.append(equity[-1] + t)
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        peak = max(peak, e)
        max_dd = max(max_dd, peak - e)

    total_ret = sum(trades)
    calmar = total_ret / max(max_dd, 1e-9) if max_dd > 0 else total_ret * 10
    recovery = total_ret / max(max_dd, 1e-9)

    # Profit factor
    gross_profit = sum(t for t in trades if t > 0)
    gross_loss = abs(sum(t for t in trades if t < 0))
    profit_factor = gross_profit / max(gross_loss, 1e-9)

    wins = sum(1 for t in trades if t > 0)
    win_rate = wins / n

    # Equity curve quality
    returns_seq = [(equity[i + 1] - equity[i]) for i in range(len(equity) - 1)]
    hurst = _hurst_exponent(returns_seq)
    tail_r = _tail_ratio(trades)
    max_consec = _max_consecutive_losses(trades)

    # OOS
    oos_sharpe = 0.0
    overfit_score = 0.0
    if oos_trades and len(oos_trades) >= 2:
        oos_mean = sum(oos_trades) / len(oos_trades)
        oos_var = sum((t - oos_mean) ** 2 for t in oos_trades) / max(len(oos_trades) - 1, 1)
        oos_sharpe = oos_mean / max(oos_var ** 0.5, 1e-9)
        if sharpe > 0:
            overfit_score = max(0.0, min(1.0, 1.0 - (oos_sharpe / sharpe)))

    # MCCV
    mccv_mean = 0.0
    mccv_std = 0.0
    if mccv_sharpes and len(mccv_sharpes) >= 2:
        mccv_mean = sum(mccv_sharpes) / len(mccv_sharpes)
        mccv_std = (sum((s - mccv_mean) ** 2 for s in mccv_sharpes) / (len(mccv_sharpes) - 1)) ** 0.5

    return FitnessVector(
        sharpe=sharpe, sortino=sortino, max_drawdown_pct=max_dd,
        calmar=calmar, profit_factor=profit_factor, trade_count=n,
        win_rate=win_rate, avg_return_pct=mean_ret,
        oos_sharpe=oos_sharpe, overfitting_score=overfit_score,
        robustness=robustness, hurst_exponent=hurst,
        max_consec_losses=max_consec, recovery_factor=recovery,
        tail_ratio=tail_r, anti_fragility=anti_fragility,
        mccv_mean_sharpe=mccv_mean, mccv_std_sharpe=mccv_std,
    )


# ════════════════════════════════════════════════════════════════════════════
# Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
# ════════════════════════════════════════════════════════════════════════════

def deflated_sharpe_ratio(sharpe_raw: float, n_trials: int, n_trades: int = 50) -> float:
    """Bailey & Lopez de Prado (2014): correct Sharpe for multiple testing.

    The more strategies you test, the more likely the best Sharpe is luck.
    DSR deflates the raw Sharpe proportional to log(trials).
    """
    if n_trials <= 1 or n_trades < 5:
        return sharpe_raw
    # Expected max Sharpe under null (Euler-Mascheroni + log correction)
    euler_mascheroni = 0.5772
    expected_max = ((2 * math.log(n_trials)) ** 0.5
                    - (math.log(math.pi) + euler_mascheroni)
                    / (2 * (2 * math.log(n_trials)) ** 0.5))
    # Standard error of Sharpe estimate
    se = (1 + 0.5 * sharpe_raw ** 2) ** 0.5 / max(n_trades ** 0.5, 1)
    # Deflated = (observed - expected_max) / SE
    return (sharpe_raw - expected_max * se) / max(se, 1e-9)


def whites_reality_check(trades: List[float], n_bootstrap: int = 500,
                         rng: Optional[random.Random] = None) -> float:
    """White's Reality Check: bootstrap p-value testing if strategy beats zero.

    Returns p-value. Lower = more statistically significant edge.
    """
    if not trades or len(trades) < 5:
        return 1.0
    if rng is None:
        rng = random.Random()

    n = len(trades)
    mean_obs = sum(trades) / n

    # Bootstrap: resample trades with replacement, compute mean each time
    boot_means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(trades) for _ in range(n)]
        boot_means.append(sum(sample) / n)

    # p-value: fraction of bootstrap means exceeding observed (centered)
    # Under H0 (no edge), trades are symmetric around 0
    centered = [t - mean_obs for t in trades]
    boot_h0 = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(centered) for _ in range(n)]
        boot_h0.append(sum(sample) / n + mean_obs)

    exceed = sum(1 for bm in boot_h0 if bm >= mean_obs)
    return exceed / max(n_bootstrap, 1)


# ════════════════════════════════════════════════════════════════════════════
# Novelty Search (Lehman & Stanley 2011)
# ════════════════════════════════════════════════════════════════════════════

def _equity_curve_signature(trades: List[float], n_bins: int = 8) -> Tuple[float, ...]:
    """Discretise equity curve into fixed-length behavioral signature."""
    if not trades:
        return (0.0,) * n_bins
    equity = [0.0]
    for t in trades:
        equity.append(equity[-1] + t)
    # Sample at evenly-spaced points
    step = max(1, len(equity) // n_bins)
    sig = [equity[min(i * step, len(equity) - 1)] for i in range(n_bins)]
    # Normalise to [0,1] range
    lo, hi = min(sig), max(sig)
    rng = hi - lo if hi != lo else 1.0
    return tuple((s - lo) / rng for s in sig)


class NoveltyArchive:
    """Behavioral novelty archive — rewards strategies that BEHAVE differently."""

    def __init__(self, k_nearest: int = 10, capacity: int = 200):
        self._signatures: List[Tuple[str, Tuple[float, ...]]] = []  # (genome_id, signature)
        self._k = k_nearest
        self._capacity = capacity

    def novelty_score(self, signature: Tuple[float, ...]) -> float:
        """Distance to k-nearest neighbors. Higher = more novel."""
        if len(self._signatures) < self._k:
            return 1.0  # novel by default when archive is small
        distances = []
        for _, archived_sig in self._signatures:
            d = sum((a - b) ** 2 for a, b in zip(signature, archived_sig)) ** 0.5
            distances.append(d)
        distances.sort()
        return sum(distances[:self._k]) / self._k

    def try_add(self, genome_id: str, signature: Tuple[float, ...],
                min_novelty: float = 0.1) -> bool:
        ns = self.novelty_score(signature)
        if ns >= min_novelty or len(self._signatures) < 20:
            self._signatures.append((genome_id, signature))
            if len(self._signatures) > self._capacity:
                self._signatures.pop(0)  # FIFO eviction
            return True
        return False

    def size(self) -> int:
        return len(self._signatures)


# ════════════════════════════════════════════════════════════════════════════
# Transfer Learning (cross-symbol parameter intelligence)
# ════════════════════════════════════════════════════════════════════════════

class TransferLearningModule:
    """Learns parameter distributions from elite genomes across all symbols."""

    def __init__(self):
        self._stats: Dict[Tuple[str, str], Tuple[float, float, int]] = {}  # (stype, param) → (mean, std, n)

    def update(self, elite_genomes: List[StrategyGenome]) -> None:
        """Recompute parameter statistics from elites."""
        by_type: Dict[str, List[StrategyGenome]] = defaultdict(list)
        for g in elite_genomes:
            by_type[g.strategy_type].append(g)

        for stype, genomes in by_type.items():
            if len(genomes) < 3:
                continue
            for param in _PARAM_BOUNDS.get(stype, {}):
                vals = [g.params[param] for g in genomes if param in g.params]
                if len(vals) < 3:
                    continue
                m = sum(vals) / len(vals)
                s = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                self._stats[(stype, param)] = (m, s, len(vals))

    def warm_start(self, strategy_type: str, symbol: str,
                   rng: random.Random) -> Dict[str, float]:
        """Generate params from learned distributions (transfer from other symbols)."""
        bounds = _PARAM_BOUNDS.get(strategy_type, {})
        params = {}
        for key, (lo, hi) in bounds.items():
            stat = self._stats.get((strategy_type, key))
            if stat and stat[2] >= 5:
                m, s, _ = stat
                val = rng.gauss(m, max(s, (hi - lo) * 0.02))
                val = max(lo, min(hi, val))
            else:
                val = rng.uniform(lo, hi)
            if key in _INTEGER_PARAMS:
                val = round(val)
            params[key] = val
        return params

    @property
    def active(self) -> bool:
        return len(self._stats) >= 3


# ════════════════════════════════════════════════════════════════════════════
# Surrogate Pre-Filter
# ════════════════════════════════════════════════════════════════════════════

class SurrogateModel:
    """Fast heuristic pre-filter: skip expensive backtest for obviously bad genomes.

    Learns from observed (params → fitness) pairs. Uses distance-weighted
    k-NN regression (no sklearn dependency).
    """

    def __init__(self, k: int = 5, capacity: int = 500):
        self._observations: List[Tuple[str, Dict[str, float], float]] = []  # (stype, params, composite)
        self._k = k
        self._capacity = capacity

    def record(self, strategy_type: str, params: Dict[str, float], composite: float) -> None:
        self._observations.append((strategy_type, dict(params), composite))
        if len(self._observations) > self._capacity:
            self._observations.pop(0)

    def predict(self, strategy_type: str, params: Dict[str, float]) -> Optional[float]:
        """Predict fitness. Returns None if not enough data."""
        same_type = [(p, c) for st, p, c in self._observations if st == strategy_type]
        if len(same_type) < self._k * 2:
            return None  # not enough data, do full eval

        # k-NN distance-weighted regression
        distances = []
        for obs_params, obs_comp in same_type:
            d = sum((params.get(k, 0) - obs_params.get(k, 0)) ** 2
                    for k in params) ** 0.5
            distances.append((d, obs_comp))
        distances.sort(key=lambda x: x[0])
        nearest = distances[:self._k]

        total_w = 0.0
        weighted_sum = 0.0
        for d, c in nearest:
            w = 1.0 / max(d, 1e-9)
            weighted_sum += w * c
            total_w += w
        return weighted_sum / total_w if total_w > 0 else None

    def should_full_eval(self, strategy_type: str, params: Dict[str, float],
                         threshold_percentile: float = 0.40) -> bool:
        """Return True if genome is promising enough for full evaluation."""
        pred = self.predict(strategy_type, params)
        if pred is None:
            return True  # unknown → evaluate
        # Get distribution of known composites for this type
        composites = [c for st, _, c in self._observations if st == strategy_type]
        if not composites:
            return True
        composites.sort()
        cutoff_idx = int(len(composites) * threshold_percentile)
        cutoff = composites[cutoff_idx] if cutoff_idx < len(composites) else composites[0]
        return pred >= cutoff

    @property
    def ready(self) -> bool:
        return len(self._observations) >= 20


# ════════════════════════════════════════════════════════════════════════════
# Island
# ════════════════════════════════════════════════════════════════════════════

class Island:
    """A sub-population of genomes for one regime."""

    def __init__(self, name: str, capacity: int, rng: random.Random):
        self.name = name
        self.capacity = capacity
        self._rng = rng
        self.population: List[StrategyGenome] = []

    def add(self, genome: StrategyGenome) -> None:
        g = StrategyGenome(
            strategy_type=genome.strategy_type, params=genome.params,
            symbol=genome.symbol, fitness=genome.fitness,
            generation=genome.generation, parent_ids=genome.parent_ids,
            sigma=genome.sigma, island=self.name,
            pareto_rank=genome.pareto_rank, crowding_distance=genome.crowding_distance,
        )
        self.population.append(g)

    def best(self, n: int = 1) -> List[StrategyGenome]:
        return sorted(self.population,
                      key=lambda g: (g.pareto_rank, -g.crowding_distance, -g.composite_fitness))[:n]

    def size(self) -> int:
        return len(self.population)


# ════════════════════════════════════════════════════════════════════════════
# MAP-Elites archive
# ════════════════════════════════════════════════════════════════════════════

class MAPElitesArchive:
    """Quality-diversity archive: best genome per (strategy_type, regime, symbol) niche."""

    def __init__(self):
        self._archive: Dict[Tuple[str, str, str], StrategyGenome] = {}

    def try_add(self, genome: StrategyGenome) -> bool:
        """Add genome if it's the best in its niche. Returns True if added."""
        key = (genome.strategy_type, genome.island, genome.symbol)
        existing = self._archive.get(key)
        if existing is None or genome.composite_fitness > existing.composite_fitness:
            self._archive[key] = genome
            return True
        return False

    def get_all(self) -> List[StrategyGenome]:
        return list(self._archive.values())

    def get_niche(self, strategy_type: str, regime: str, symbol: str) -> Optional[StrategyGenome]:
        return self._archive.get((strategy_type, regime, symbol))

    def size(self) -> int:
        return len(self._archive)

    def coverage(self) -> Dict[str, int]:
        """Count genomes per regime in archive."""
        counts: Dict[str, int] = defaultdict(int)
        for (_, regime, _), _ in self._archive.items():
            counts[regime] += 1
        return dict(counts)


# ════════════════════════════════════════════════════════════════════════════
# Hall of Fame (immortal best-ever archive)
# ════════════════════════════════════════════════════════════════════════════

class HallOfFame:
    """Top N all-time best genomes, immune to population drift."""

    def __init__(self, capacity: int = 20):
        self._capacity = capacity
        self._members: List[StrategyGenome] = []

    def try_add(self, genome: StrategyGenome) -> bool:
        """Add if genome qualifies for hall of fame."""
        # Check if duplicate (same genome_id)
        for m in self._members:
            if m.genome_id == genome.genome_id:
                return False

        if len(self._members) < self._capacity:
            self._members.append(genome)
            self._members.sort(key=lambda g: g.composite_fitness, reverse=True)
            return True

        worst = self._members[-1]
        if genome.composite_fitness > worst.composite_fitness:
            self._members[-1] = genome
            self._members.sort(key=lambda g: g.composite_fitness, reverse=True)
            return True
        return False

    def get_all(self) -> List[StrategyGenome]:
        return list(self._members)

    def size(self) -> int:
        return len(self._members)

    def get_best(self) -> Optional[StrategyGenome]:
        return self._members[0] if self._members else None


# ════════════════════════════════════════════════════════════════════════════
# Main evolver
# ════════════════════════════════════════════════════════════════════════════

class StrategyEvolver:
    """
    Absolute elite genetic strategy optimizer.

    Combines NSGA-II Pareto ranking, MAP-Elites quality-diversity,
    differential evolution, Monte Carlo cross-validation, parameter
    sensitivity analysis, anti-fragility scoring, and a permanent
    Hall of Fame. This is the engine that makes ARGUS adapt.
    """

    def __init__(
        self,
        population_size: int = 30,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        elite_pct: float = 0.10,
        tournament_size: int = 5,
        oos_split: float = 0.25,
        migration_interval: int = 5,
        migration_count: int = 2,
        min_trades: int = 5,
        max_drawdown_pct: float = 20.0,
        ensemble_size: int = 5,
        mccv_folds: int = 5,               # Monte Carlo CV folds
        robustness_jitters: int = 5,        # param jitter trials
        robustness_noise: float = 0.05,     # 5% of range per jitter
        stagnation_limit: int = 10,         # gens without improvement → restart
        hall_of_fame_size: int = 20,
        de_weight: float = 0.8,             # DE mutation weight F
        de_cr: float = 0.9,                 # DE crossover probability
        seed: Optional[int] = None,
    ):
        self._pop_size = population_size
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._elite_pct = elite_pct
        self._tournament_size = tournament_size
        self._oos_split = oos_split
        self._migration_interval = migration_interval
        self._migration_count = migration_count
        self._min_trades = min_trades
        self._max_dd = max_drawdown_pct
        self._ensemble_size = ensemble_size
        self._mccv_folds = mccv_folds
        self._robustness_jitters = robustness_jitters
        self._robustness_noise = robustness_noise
        self._stagnation_limit = stagnation_limit
        self._de_F = de_weight
        self._de_CR = de_cr
        self._rng = random.Random(seed)
        self._generation = 0
        self._best_ever: Optional[StrategyGenome] = None
        self._best_ensemble: Optional[EnsembleGenome] = None

        # Structures
        island_cap = max(population_size // 4, 8)
        self._islands: Dict[str, Island] = {
            "trending": Island("trending", island_cap, self._rng),
            "ranging": Island("ranging", island_cap, self._rng),
            "volatile": Island("volatile", island_cap, self._rng),
            "universal": Island("universal", island_cap, self._rng),
        }
        self._archive = MAPElitesArchive()
        self._hall_of_fame = HallOfFame(hall_of_fame_size)
        self._dynamic_bounds: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self._dynamic_bounds_samples = 0
        self._stagnation_counter = 0
        self._last_best_composite = -999.0
        # v4 pinnacle modules
        self._novelty_archive = NoveltyArchive()
        self._transfer = TransferLearningModule()
        self._surrogate = SurrogateModel()
        self._total_trials = 0             # for deflated Sharpe
        # GPU acceleration (optional)
        self._gpu_engine = None
        try:
            from core.gpu_evolution import GPUEvolutionEngine
            self._gpu_engine = GPUEvolutionEngine()
            if self._gpu_engine.available:
                logger.info("StrategyEvolver: GPU acceleration enabled")
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════
    # Seeding
    # ══════════════════════════════════════════════════════════════════════

    def seed_from_scanner(self, scanner_results: List[Dict[str, Any]]) -> None:
        """Initialize islands from scanner's top results."""
        seeded = 0
        for result in scanner_results:
            strategy = str(result.get("strategy", ""))
            symbol = str(result.get("symbol", ""))
            params = result.get("params", {})
            sharpe = float(result.get("sharpe", 0) or 0)
            if strategy not in _PARAM_BOUNDS or not params:
                continue
            island_name = self._strategy_to_island(strategy)
            genome = StrategyGenome(
                strategy_type=strategy, params=dict(params), symbol=symbol,
                fitness=FitnessVector(sharpe=sharpe), generation=0,
                sigma=self._initial_sigma(strategy), island=island_name,
            )
            self._islands[island_name].add(genome)
            seeded += 1
        for island in self._islands.values():
            self._fill_island(island)
        logger.info("StrategyEvolver v3: seeded %d genomes across %d islands", seeded, len(self._islands))

    def _fill_island(self, island: Island) -> None:
        suited = _REGIME_STRATEGY_AFFINITY.get(island.name, list(_PARAM_BOUNDS.keys()))
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "LINK/USD"]
        while island.size() < island.capacity:
            if island.population and self._rng.random() < 0.5:
                child = self._mutate(self._rng.choice(island.population))
            elif self._transfer.active and self._rng.random() < 0.3:
                # v4: Transfer learning warm-start
                stype = self._rng.choice(suited) if suited else self._rng.choice(list(_PARAM_BOUNDS.keys()))
                sym = self._rng.choice(symbols)
                params = self._transfer.warm_start(stype, sym, self._rng)
                child = StrategyGenome(
                    strategy_type=stype, params=params, symbol=sym,
                    sigma=self._initial_sigma(stype), island=island.name,
                )
            else:
                stype = self._rng.choice(suited) if suited else self._rng.choice(list(_PARAM_BOUNDS.keys()))
                child = self._random_genome(stype, self._rng.choice(symbols), island.name)
            island.add(child)

    def _strategy_to_island(self, strategy_type: str) -> str:
        for regime, strategies in _REGIME_STRATEGY_AFFINITY.items():
            if strategy_type in strategies:
                return regime
        return "universal"

    def _initial_sigma(self, strategy_type: str) -> Dict[str, float]:
        bounds = _PARAM_BOUNDS.get(strategy_type, {})
        return {key: (hi - lo) * 0.20 for key, (lo, hi) in bounds.items()}

    # ══════════════════════════════════════════════════════════════════════
    # Evolution (the main loop)
    # ══════════════════════════════════════════════════════════════════════

    def evolve(self, fitness_fn: Callable) -> EvolutionResult:
        """
        Run one generation of elite evolution.

        fitness_fn(genome, oos_split) -> (is_trades, oos_trades)
        """
        t0 = time.time()
        self._generation += 1
        mutations = 0
        crossovers = 0

        # ── Phase 1: Evaluate fitness (surrogate pre-filter + full eval) ──
        _surrogate_filtered = 0
        _full_evals = 0
        for island in self._islands.values():
            for i, genome in enumerate(island.population):
                if genome.fitness.trade_count == 0 and genome.fitness.sharpe == 0.0:
                    try:
                        # v4: Surrogate pre-filter — skip expensive backtest for bad genomes
                        if (self._surrogate.ready
                                and not self._surrogate.should_full_eval(
                                    genome.strategy_type, genome.params)):
                            pred = self._surrogate.predict(genome.strategy_type, genome.params)
                            fv = FitnessVector(
                                sharpe=pred or 0.0, trade_count=1,
                                surrogate_predicted=True,
                            )
                            _surrogate_filtered += 1
                        else:
                            fv = self._full_evaluate(genome, fitness_fn)
                            _full_evals += 1
                        island.population[i] = StrategyGenome(
                            strategy_type=genome.strategy_type, params=genome.params,
                            symbol=genome.symbol, fitness=fv, generation=genome.generation,
                            parent_ids=genome.parent_ids, sigma=genome.sigma, island=genome.island,
                        )
                    except Exception:
                        pass

        # ── Phase 2: NSGA-II ranking per island ──
        for island in self._islands.values():
            island.population = nsga2_rank(island.population)

        # ── Phase 3: Update MAP-Elites archive + Hall of Fame ──
        for g in self._all_genomes():
            self._archive.try_add(g)
            self._hall_of_fame.try_add(g)

        # ── Phase 4: Selection + reproduction per island ──
        for island in self._islands.values():
            elite_n = max(1, int(island.size() * self._elite_pct))
            # Sort by NSGA-II: rank ascending, crowding descending
            island.population.sort(key=lambda g: (g.pareto_rank, -g.crowding_distance))
            next_gen = list(island.population[:elite_n])

            # Inject hall-of-fame members (knowledge transfer)
            hof_inject = self._hall_of_fame.get_all()[:2]
            for hg in hof_inject:
                if hg.strategy_type in _REGIME_STRATEGY_AFFINITY.get(island.name, []):
                    next_gen.append(hg)

            while len(next_gen) < island.capacity:
                r = self._rng.random()
                if r < self._crossover_rate * 0.5 and len(island.population) >= 2:
                    # BLX-alpha crossover
                    p1 = nsga2_select(island.population, self._rng, self._tournament_size)
                    p2 = nsga2_select(island.population, self._rng, self._tournament_size)
                    child = self._blend_crossover(p1, p2)
                    crossovers += 1
                elif r < self._crossover_rate and len(island.population) >= 3:
                    # Differential evolution
                    child = self._de_mutate(island.population)
                    crossovers += 1
                else:
                    parent = nsga2_select(island.population, self._rng, self._tournament_size)
                    child = self._mutate(parent)
                    mutations += 1
                next_gen.append(child)

            island.population = next_gen[:island.capacity]

        # ── Phase 5: Migration ──
        if self._generation % self._migration_interval == 0:
            self._migrate()

        # ── Phase 6: Stagnation detection + chaos restart ──
        all_genomes = self._all_genomes()
        best_now = max(all_genomes, key=lambda g: g.composite_fitness) if all_genomes else None
        if best_now:
            if best_now.composite_fitness <= self._last_best_composite + 0.01:
                self._stagnation_counter += 1
            else:
                self._stagnation_counter = 0
                self._last_best_composite = best_now.composite_fitness

            if self._stagnation_counter >= self._stagnation_limit:
                self._chaos_restart()
                self._stagnation_counter = 0

        # ── Phase 7: Dynamic bounds ──
        self._update_dynamic_bounds()

        # ── Phase 7b: Transfer learning update ──
        all_elite = self._hall_of_fame.get_all() + self._archive.get_all()
        if all_elite:
            self._transfer.update(all_elite)

        # ── Phase 8: Track best + ensemble ──
        if best_now and (self._best_ever is None or best_now.composite_fitness > self._best_ever.composite_fitness):
            self._best_ever = best_now
        self._best_ensemble = self._build_ensemble(all_genomes)

        # ── Stats ──
        total_pop = sum(isl.size() for isl in self._islands.values())
        all_fit = [g.composite_fitness for g in all_genomes]
        avg_fitness = sum(all_fit) / max(len(all_fit), 1)
        overfit_n = sum(1 for g in all_genomes if g.fitness.overfitting_score > 0.5)
        overfit_rate = overfit_n / max(len(all_genomes), 1)
        pareto0 = sum(1 for g in all_genomes if g.pareto_rank == 0)
        avg_robust = sum(g.fitness.robustness for g in all_genomes) / max(len(all_genomes), 1)
        avg_af = sum(g.fitness.anti_fragility for g in all_genomes) / max(len(all_genomes), 1)
        diversity = self._compute_diversity(all_genomes)
        duration_ms = (time.time() - t0) * 1000

        avg_dsr = sum(g.fitness.deflated_sharpe for g in all_genomes) / max(len(all_genomes), 1)
        sig_pct = sum(1 for g in all_genomes if g.fitness.reality_check_pval < 0.10) / max(len(all_genomes), 1)
        surr_rate = _surrogate_filtered / max(_surrogate_filtered + _full_evals, 1)

        result = EvolutionResult(
            generation=self._generation, population_size=total_pop,
            best_genome=self._best_ever, avg_fitness=avg_fitness,
            top_n=sorted(all_genomes, key=lambda g: g.composite_fitness, reverse=True)[:5],
            mutations=mutations, crossovers=crossovers, duration_ms=duration_ms,
            islands={n: isl.size() for n, isl in self._islands.items()},
            best_ensemble=self._best_ensemble, overfitting_rate=overfit_rate,
            pareto_front_size=pareto0, archive_size=self._archive.size(),
            hall_of_fame_size=self._hall_of_fame.size(),
            avg_robustness=avg_robust, avg_anti_fragility=avg_af,
            stagnation_counter=self._stagnation_counter,
            diversity_metric=diversity,
            avg_deflated_sharpe=avg_dsr,
            statistically_significant_pct=sig_pct,
            novelty_archive_size=self._novelty_archive.size(),
            surrogate_filter_rate=surr_rate,
            transfer_learning_active=self._transfer.active,
        )

        if result.best_genome:
            logger.info(
                "Evolver v4 gen %d: best=%s %s composite=%.3f dsr=%.3f oos=%.3f "
                "robust=%.2f novel=%.2f sig=%.0f%% pareto=%d hof=%d surr=%.0f%% (%.0fms)",
                self._generation, result.best_genome.strategy_type,
                result.best_genome.symbol, result.best_genome.composite_fitness,
                result.best_genome.fitness.deflated_sharpe, result.best_genome.fitness.oos_sharpe,
                result.best_genome.fitness.robustness, result.best_genome.fitness.novelty_score,
                sig_pct * 100, pareto0, self._hall_of_fame.size(),
                surr_rate * 100, duration_ms,
            )
        return result

    # ══════════════════════════════════════════════════════════════════════
    # Full evaluation: MCCV + robustness + anti-fragility
    # ══════════════════════════════════════════════════════════════════════

    def _full_evaluate(self, genome: StrategyGenome, fitness_fn: Callable) -> FitnessVector:
        """Evaluate genome with k-fold MCCV, robustness jitter, and anti-fragility."""
        # Primary fitness
        is_trades, oos_trades = fitness_fn(genome, self._oos_split)

        # Monte Carlo Cross-Validation: evaluate at multiple split points
        mccv_sharpes = []
        for fold in range(self._mccv_folds):
            fold_split = 0.15 + (fold * 0.15)  # splits at 15%, 30%, 45%, 60%, 75%
            fold_split = min(fold_split, 0.75)
            try:
                f_is, f_oos = fitness_fn(genome, fold_split)
                if f_oos and len(f_oos) >= 2:
                    f_mean = sum(f_oos) / len(f_oos)
                    f_var = sum((t - f_mean) ** 2 for t in f_oos) / max(len(f_oos) - 1, 1)
                    mccv_sharpes.append(f_mean / max(f_var ** 0.5, 1e-9))
            except Exception:
                pass

        # Robustness: jitter parameters, check fitness stability
        robustness = self._compute_robustness(genome, fitness_fn)

        # Anti-fragility: test across different data regimes
        anti_fragility = self._compute_anti_fragility(genome, fitness_fn)

        fv = compute_fitness_vector(
            is_trades, oos_trades,
            robustness=robustness,
            anti_fragility=anti_fragility,
            mccv_sharpes=mccv_sharpes if mccv_sharpes else None,
        )

        # Constraint gates
        if fv.trade_count < self._min_trades:
            return FitnessVector(trade_count=fv.trade_count)
        if fv.max_drawdown_pct > self._max_dd:
            return FitnessVector(
                sharpe=fv.sharpe * 0.3, max_drawdown_pct=fv.max_drawdown_pct,
                trade_count=fv.trade_count, overfitting_score=max(fv.overfitting_score, 0.5),
            )

        # v4: Deflated Sharpe + White's Reality Check
        self._total_trials += 1
        dsr = deflated_sharpe_ratio(fv.sharpe, self._total_trials, fv.trade_count)
        wrc_pval = whites_reality_check(is_trades, n_bootstrap=200, rng=self._rng)

        # v4: Novelty score from equity curve behavioral signature
        sig = _equity_curve_signature(is_trades)
        novelty = self._novelty_archive.novelty_score(sig)
        self._novelty_archive.try_add(genome.genome_id, sig)
        norm_novelty = min(1.0, novelty / max(1.0, 1e-9))

        # v4: Feed surrogate model
        self._surrogate.record(genome.strategy_type, genome.params, fv.composite)

        # Rebuild FitnessVector with v4 fields
        return FitnessVector(
            sharpe=fv.sharpe, sortino=fv.sortino, max_drawdown_pct=fv.max_drawdown_pct,
            calmar=fv.calmar, profit_factor=fv.profit_factor, trade_count=fv.trade_count,
            win_rate=fv.win_rate, avg_return_pct=fv.avg_return_pct,
            oos_sharpe=fv.oos_sharpe, overfitting_score=fv.overfitting_score,
            robustness=fv.robustness, hurst_exponent=fv.hurst_exponent,
            max_consec_losses=fv.max_consec_losses, recovery_factor=fv.recovery_factor,
            tail_ratio=fv.tail_ratio, anti_fragility=fv.anti_fragility,
            mccv_mean_sharpe=fv.mccv_mean_sharpe, mccv_std_sharpe=fv.mccv_std_sharpe,
            deflated_sharpe=dsr, reality_check_pval=wrc_pval,
            novelty_score=norm_novelty,
        )

    def _compute_robustness(self, genome: StrategyGenome, fitness_fn: Callable) -> float:
        """Jitter each parameter by ±noise, check how much fitness changes."""
        bounds = _PARAM_BOUNDS.get(genome.strategy_type, {})
        if not bounds:
            return 0.5

        # Get base fitness
        try:
            base_is, _ = fitness_fn(genome, self._oos_split)
            if not base_is or len(base_is) < 3:
                return 0.0
            base_mean = sum(base_is) / len(base_is)
            base_std = (sum((t - base_mean) ** 2 for t in base_is) / max(len(base_is) - 1, 1)) ** 0.5
            base_sharpe = base_mean / max(base_std, 1e-9)
        except Exception:
            return 0.0

        jitter_sharpes = []
        for _ in range(self._robustness_jitters):
            jittered_params = dict(genome.params)
            for key, (lo, hi) in bounds.items():
                if key in jittered_params:
                    noise = self._rng.gauss(0, (hi - lo) * self._robustness_noise)
                    val = max(lo, min(hi, float(jittered_params[key]) + noise))
                    if key in _INTEGER_PARAMS:
                        val = round(val)
                    jittered_params[key] = val

            jittered = StrategyGenome(
                strategy_type=genome.strategy_type, params=jittered_params,
                symbol=genome.symbol, sigma=genome.sigma, island=genome.island,
            )
            try:
                j_is, _ = fitness_fn(jittered, self._oos_split)
                if j_is and len(j_is) >= 3:
                    j_mean = sum(j_is) / len(j_is)
                    j_std = (sum((t - j_mean) ** 2 for t in j_is) / max(len(j_is) - 1, 1)) ** 0.5
                    jitter_sharpes.append(j_mean / max(j_std, 1e-9))
            except Exception:
                pass

        if not jitter_sharpes:
            return 0.0

        # Robustness = 1 - normalised variance of jittered Sharpes
        j_mean = sum(jitter_sharpes) / len(jitter_sharpes)
        j_var = sum((s - j_mean) ** 2 for s in jitter_sharpes) / max(len(jitter_sharpes) - 1, 1)
        # Normalise: variance of 0 = perfect robustness (1.0)
        # Large variance → low robustness
        return max(0.0, min(1.0, 1.0 - min(j_var ** 0.5 / max(abs(base_sharpe), 1e-9), 1.0)))

    def _compute_anti_fragility(self, genome: StrategyGenome, fitness_fn: Callable) -> float:
        """Test genome across different OOS split points (proxy for regime changes).
        Anti-fragile strategies have positive Sharpe in most splits."""
        positive_count = 0
        total_tests = 0
        for split in (0.20, 0.40, 0.60):
            try:
                _, oos = fitness_fn(genome, split)
                if oos and len(oos) >= 2:
                    total_tests += 1
                    oos_mean = sum(oos) / len(oos)
                    if oos_mean > 0:
                        positive_count += 1
            except Exception:
                pass
        return positive_count / max(total_tests, 1)

    # ══════════════════════════════════════════════════════════════════════
    # Mutation operators
    # ══════════════════════════════════════════════════════════════════════

    def _mutate(self, parent: StrategyGenome) -> StrategyGenome:
        """Self-adaptive Gaussian + Cauchy mutation."""
        bounds = _PARAM_BOUNDS.get(parent.strategy_type, {})
        effective = self._get_effective_bounds(parent.strategy_type)
        new_params = dict(parent.params)
        new_sigma = dict(parent.sigma) if parent.sigma else self._initial_sigma(parent.strategy_type)

        n_p = max(len(bounds), 1)
        tau = 1.0 / (2 * n_p) ** 0.5
        tau_prime = 1.0 / (2 * (n_p ** 0.5)) ** 0.5
        global_factor = math.exp(tau_prime * self._rng.gauss(0, 1))

        for key, (lo, hi) in bounds.items():
            if key not in new_params or self._rng.random() >= self._mutation_rate:
                continue
            elo, ehi = effective.get(key, (lo, hi))
            span = ehi - elo

            old_sig = new_sigma.get(key, span * 0.15)
            new_sig = old_sig * global_factor * math.exp(tau * self._rng.gauss(0, 1))
            new_sig = max(span * 0.01, min(span * 0.30, new_sig))
            new_sigma[key] = new_sig

            current = float(new_params[key])
            if self._rng.random() < 0.10:
                noise = new_sig * math.tan(math.pi * (self._rng.random() - 0.5))
            else:
                noise = self._rng.gauss(0, new_sig)
            new_val = max(lo, min(hi, current + noise))
            if key in _INTEGER_PARAMS or isinstance(parent.params.get(key), int):
                new_val = round(new_val)
            new_params[key] = new_val

        return StrategyGenome(
            strategy_type=parent.strategy_type, params=new_params, symbol=parent.symbol,
            fitness=FitnessVector(), generation=self._generation,
            parent_ids=(parent.genome_id,), sigma=new_sigma, island=parent.island,
        )

    def _de_mutate(self, population: List[StrategyGenome]) -> StrategyGenome:
        """Differential Evolution DE/rand/1/bin operator."""
        if len(population) < 3:
            return self._mutate(self._rng.choice(population))

        a, b, c = self._rng.sample(population, 3)
        bounds = _PARAM_BOUNDS.get(a.strategy_type, {})
        new_params = dict(a.params)

        for key, (lo, hi) in bounds.items():
            if key not in a.params:
                continue
            if self._rng.random() < self._de_CR:
                # DE/rand/1: mutant = a + F * (b - c)
                mid = (lo + hi) / 2.0
                va = float(a.params.get(key, mid))
                vb = float(b.params.get(key, mid))
                vc = float(c.params.get(key, mid))
                mutant = va + self._de_F * (vb - vc)
                mutant = max(lo, min(hi, mutant))
                if key in _INTEGER_PARAMS:
                    mutant = round(mutant)
                new_params[key] = mutant

        return StrategyGenome(
            strategy_type=a.strategy_type, params=new_params, symbol=a.symbol,
            fitness=FitnessVector(), generation=self._generation,
            parent_ids=(a.genome_id, b.genome_id, c.genome_id),
            sigma=a.sigma if a.sigma else self._initial_sigma(a.strategy_type),
            island=a.island,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Crossover
    # ══════════════════════════════════════════════════════════════════════

    def _blend_crossover(self, p1: StrategyGenome, p2: StrategyGenome) -> StrategyGenome:
        """BLX-alpha crossover with self-adaptive sigma blending."""
        bounds = _PARAM_BOUNDS.get(p1.strategy_type, {})
        new_params = {}
        new_sigma = {}
        alpha = 0.3

        for key in p1.params:
            v1 = float(p1.params[key])
            v2 = float(p2.params.get(key, v1))
            lo, hi = bounds.get(key, (v1, v2))
            lo_p, hi_p = min(v1, v2), max(v1, v2)
            span = hi_p - lo_p
            blend_lo = max(lo, lo_p - alpha * span)
            blend_hi = min(hi, hi_p + alpha * span)
            new_val = self._rng.uniform(blend_lo, blend_hi)
            if key in _INTEGER_PARAMS or isinstance(p1.params.get(key), int):
                new_val = round(new_val)
            new_params[key] = new_val

            s1 = (p1.sigma or {}).get(key, 0.15)
            s2 = (p2.sigma or {}).get(key, 0.15)
            new_sigma[key] = (s1 * s2) ** 0.5

        return StrategyGenome(
            strategy_type=p1.strategy_type, params=new_params,
            symbol=p1.symbol if self._rng.random() < 0.5 else p2.symbol,
            fitness=FitnessVector(), generation=self._generation,
            parent_ids=(p1.genome_id, p2.genome_id), sigma=new_sigma, island=p1.island,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Migration + stagnation
    # ══════════════════════════════════════════════════════════════════════

    def _migrate(self) -> None:
        """Ring-topology migration between islands."""
        names = list(self._islands.keys())
        n = len(names)
        if n < 2:
            return
        for i in range(n):
            src = self._islands[names[i]]
            dst = self._islands[names[(i + 1) % n]]
            for m in src.best(self._migration_count):
                if dst.size() < dst.capacity + self._migration_count:
                    dst.add(m)
        for island in self._islands.values():
            if island.size() > island.capacity:
                island.population.sort(key=lambda g: (g.pareto_rank, -g.crowding_distance))
                island.population = island.population[:island.capacity]

    def _chaos_restart(self) -> None:
        """Stagnation detected — inject random diversity while keeping elites."""
        logger.info("StrategyEvolver: stagnation detected at gen %d, injecting chaos", self._generation)
        for island in self._islands.values():
            if island.size() < 3:
                continue
            # Keep top 25%, replace bottom 75% with random genomes
            keep_n = max(1, island.size() // 4)
            island.population.sort(key=lambda g: g.composite_fitness, reverse=True)
            island.population = island.population[:keep_n]
            self._fill_island(island)

    # ══════════════════════════════════════════════════════════════════════
    # Dynamic bounds
    # ══════════════════════════════════════════════════════════════════════

    def _update_dynamic_bounds(self) -> None:
        all_genomes = self._all_genomes()
        if len(all_genomes) < 10:
            return
        sorted_pop = sorted(all_genomes, key=lambda g: g.composite_fitness, reverse=True)
        top_n = max(3, len(sorted_pop) // 5)
        top = sorted_pop[:top_n]
        for stype in _PARAM_BOUNDS:
            typed = [g for g in top if g.strategy_type == stype]
            if len(typed) < 2:
                continue
            if stype not in self._dynamic_bounds:
                self._dynamic_bounds[stype] = {}
            for key, (hard_lo, hard_hi) in _PARAM_BOUNDS[stype].items():
                values = [g.params.get(key) for g in typed if key in g.params]
                if not values:
                    continue
                mean_v = sum(values) / len(values)
                std_v = (sum((v - mean_v) ** 2 for v in values) / max(len(values) - 1, 1)) ** 0.5
                margin = max(std_v, (hard_hi - hard_lo) * 0.05)
                self._dynamic_bounds[stype][key] = (
                    max(hard_lo, mean_v - 3 * margin),
                    min(hard_hi, mean_v + 3 * margin),
                )
        self._dynamic_bounds_samples += 1

    def _get_effective_bounds(self, strategy_type: str) -> Dict[str, Tuple[float, float]]:
        if self._dynamic_bounds_samples < 3:
            return _PARAM_BOUNDS.get(strategy_type, {})
        return self._dynamic_bounds.get(strategy_type, _PARAM_BOUNDS.get(strategy_type, {}))

    # ══════════════════════════════════════════════════════════════════════
    # Diversity
    # ══════════════════════════════════════════════════════════════════════

    def _compute_diversity(self, genomes: List[StrategyGenome]) -> float:
        """Measure population diversity: unique (strategy_type, symbol) pairs / total."""
        if not genomes:
            return 0.0
        unique = len(set((g.strategy_type, g.symbol) for g in genomes))
        return unique / len(genomes)

    # ══════════════════════════════════════════════════════════════════════
    # Ensemble
    # ══════════════════════════════════════════════════════════════════════

    def _build_ensemble(self, all_genomes: List[StrategyGenome]) -> Optional[EnsembleGenome]:
        if len(all_genomes) < self._ensemble_size:
            return None
        candidates = sorted(all_genomes, key=lambda g: g.composite_fitness, reverse=True)
        ensemble: List[StrategyGenome] = [candidates[0]]
        used_types = {candidates[0].strategy_type}
        used_symbols = {candidates[0].symbol}
        for cand in candidates[1:]:
            if len(ensemble) >= self._ensemble_size:
                break
            diversity_bonus = 0
            if cand.strategy_type not in used_types:
                diversity_bonus += 2
            if cand.symbol not in used_symbols:
                diversity_bonus += 1
            if diversity_bonus > 0 or len(ensemble) < 3:
                ensemble.append(cand)
                used_types.add(cand.strategy_type)
                used_symbols.add(cand.symbol)
        if len(ensemble) < 2:
            return None
        raw_w = [max(0.01, g.composite_fitness) for g in ensemble]
        total_w = sum(raw_w)
        weights = tuple(w / total_w for w in raw_w)
        ens_sharpe = sum(g.fitness.sharpe * w for g, w in zip(ensemble, weights))
        ens_sortino = sum(g.fitness.sortino * w for g, w in zip(ensemble, weights))
        ens_oos = sum(g.fitness.oos_sharpe * w for g, w in zip(ensemble, weights))
        ens_dd = max(g.fitness.max_drawdown_pct for g in ensemble) * 0.7
        ens_wr = sum(g.fitness.win_rate * w for g, w in zip(ensemble, weights))
        return EnsembleGenome(
            members=tuple(ensemble), weights=weights,
            fitness=FitnessVector(
                sharpe=ens_sharpe, sortino=ens_sortino, max_drawdown_pct=ens_dd,
                oos_sharpe=ens_oos, win_rate=ens_wr,
                trade_count=sum(g.fitness.trade_count for g in ensemble),
            ),
            generation=self._generation,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════

    def _random_genome(self, strategy_type: str, symbol: str, island: str = "universal") -> StrategyGenome:
        bounds = _PARAM_BOUNDS.get(strategy_type, {})
        params = {}
        for key, (lo, hi) in bounds.items():
            val = self._rng.uniform(lo, hi)
            if key in _INTEGER_PARAMS:
                val = round(val)
            params[key] = val
        return StrategyGenome(
            strategy_type=strategy_type, params=params, symbol=symbol,
            sigma=self._initial_sigma(strategy_type), island=island,
        )

    def _all_genomes(self) -> List[StrategyGenome]:
        result = []
        for island in self._islands.values():
            result.extend(island.population)
        return result

    # ══════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════

    def get_best(self) -> Optional[StrategyGenome]:
        return self._best_ever

    def get_best_ensemble(self) -> Optional[EnsembleGenome]:
        return self._best_ensemble

    def get_top(self, n: int = 5) -> List[StrategyGenome]:
        return sorted(self._all_genomes(), key=lambda g: g.composite_fitness, reverse=True)[:n]

    def get_pareto_front(self) -> List[StrategyGenome]:
        """Return all genomes on the first Pareto front (rank 0)."""
        return [g for g in self._all_genomes() if g.pareto_rank == 0]

    def get_archive(self) -> MAPElitesArchive:
        return self._archive

    def get_hall_of_fame(self) -> HallOfFame:
        return self._hall_of_fame

    def get_island_stats(self) -> Dict[str, Dict[str, Any]]:
        stats = {}
        for name, island in self._islands.items():
            if island.population:
                best = island.best(1)
                pareto0 = sum(1 for g in island.population if g.pareto_rank == 0)
                stats[name] = {
                    "size": island.size(),
                    "best_fitness": best[0].composite_fitness if best else 0.0,
                    "best_type": best[0].strategy_type if best else "",
                    "avg_fitness": sum(g.composite_fitness for g in island.population) / island.size(),
                    "pareto_front_size": pareto0,
                }
            else:
                stats[name] = {"size": 0, "best_fitness": 0.0, "best_type": "",
                               "avg_fitness": 0.0, "pareto_front_size": 0}
        return stats

    def get_stats(self) -> Dict[str, Any]:
        all_g = self._all_genomes()
        overfit_n = sum(1 for g in all_g if g.fitness.overfitting_score > 0.5)
        return {
            "generation": self._generation,
            "population_size": len(all_g),
            "best_ever_fitness": self._best_ever.composite_fitness if self._best_ever else 0.0,
            "best_ever_type": self._best_ever.strategy_type if self._best_ever else "",
            "best_ever_symbol": self._best_ever.symbol if self._best_ever else "",
            "best_ever_sharpe": self._best_ever.fitness.sharpe if self._best_ever else 0.0,
            "best_ever_oos_sharpe": self._best_ever.fitness.oos_sharpe if self._best_ever else 0.0,
            "best_ever_robustness": self._best_ever.fitness.robustness if self._best_ever else 0.0,
            "overfitting_rate": overfit_n / max(len(all_g), 1),
            "islands": {n: isl.size() for n, isl in self._islands.items()},
            "dynamic_bounds_active": self._dynamic_bounds_samples >= 3,
            "ensemble_members": len(self._best_ensemble.members) if self._best_ensemble else 0,
            "pareto_front_size": sum(1 for g in all_g if g.pareto_rank == 0),
            "archive_size": self._archive.size(),
            "hall_of_fame_size": self._hall_of_fame.size(),
            "stagnation_counter": self._stagnation_counter,
            "diversity": self._compute_diversity(all_g),
            "avg_robustness": sum(g.fitness.robustness for g in all_g) / max(len(all_g), 1),
            "avg_anti_fragility": sum(g.fitness.anti_fragility for g in all_g) / max(len(all_g), 1),
            # v4 pinnacle
            "total_trials": self._total_trials,
            "avg_deflated_sharpe": sum(g.fitness.deflated_sharpe for g in all_g) / max(len(all_g), 1),
            "statistically_significant_pct": sum(1 for g in all_g if g.fitness.reality_check_pval < 0.10) / max(len(all_g), 1),
            "novelty_archive_size": self._novelty_archive.size(),
            "surrogate_observations": len(self._surrogate._observations),
            "transfer_learning_active": self._transfer.active,
        }

    def get_regime_recommendation(self, regime: str) -> List[StrategyGenome]:
        island = self._islands.get(regime, self._islands.get("universal"))
        if island is None:
            return []
        return island.best(5)
