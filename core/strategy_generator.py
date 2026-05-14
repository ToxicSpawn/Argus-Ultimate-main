"""
Strategy Generator — ARGUS invents its own trading strategies.

This goes beyond the evolver (which mutates parameters of known strategies).
The generator uses Genetic Programming to compose ENTIRELY NEW strategies
from atomic building blocks — indicators, comparisons, and logical operators.

Instead of "breakout with lookback=20", the generator can discover:
"BUY when RSI(14) < 30 AND volume > 2*SMA(volume,20) AND close > EMA(50)"

Architecture:
┌─────────────────────────────────────────────────────┐
│              Strategy = Expression Tree              │
│                                                     │
│  Entry Rule:   AND                                  │
│               /    |                                │
│           LT        GT                              │
│          /  |     /  |                              │
│      RSI(14) 30  VOL  MUL                           │
│                      /  |                            │
│                  SMA(V,20) 2.0                      │
│                                                     │
│  Exit Rule:   OR                                    │
│              /    |                                  │
│          GT        LT                               │
│         /  |     /  |                               │
│    gain  2.0  gain  -1.5                            │
│                                                     │
│  Genetic operators: subtree crossover, point        │
│  mutation, grow, shrink, hoist                      │
└─────────────────────────────────────────────────────┘

Building blocks:
- Indicators: RSI, SMA, EMA, MACD, BB_upper, BB_lower, ATR, volume_ratio, high_of(N), low_of(N)
- Comparisons: GT, LT, CROSS_ABOVE, CROSS_BELOW
- Logic: AND, OR, NOT
- Constants: evolved floating-point values
- Price refs: close, open, high, low, volume
"""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Expression tree nodes
# ════════════════════════════════════════════════════════════════════════════

class NodeType(Enum):
    # Indicators (take price array + period → array)
    RSI = "RSI"
    SMA = "SMA"
    EMA = "EMA"
    MACD_HIST = "MACD_HIST"
    BB_UPPER = "BB_UPPER"
    BB_LOWER = "BB_LOWER"
    ATR = "ATR"
    VOL_RATIO = "VOL_RATIO"       # volume / SMA(volume, N)
    HIGH_OF = "HIGH_OF"           # rolling max of high over N bars
    LOW_OF = "LOW_OF"             # rolling min of low over N bars
    ROC = "ROC"                   # rate of change over N bars
    STDEV = "STDEV"               # rolling standard deviation

    # Price references (leaf nodes → array)
    CLOSE = "CLOSE"
    OPEN = "OPEN"
    HIGH = "HIGH"
    LOW = "LOW"
    VOLUME = "VOLUME"

    # Constants (leaf nodes → scalar broadcast to array)
    CONST = "CONST"

    # Comparisons (take two arrays → bool array)
    GT = "GT"
    LT = "LT"
    CROSS_ABOVE = "CROSS_ABOVE"
    CROSS_BELOW = "CROSS_BELOW"

    # Logic (take bool arrays → bool array)
    AND = "AND"
    OR = "OR"
    NOT = "NOT"

    # Arithmetic (take arrays → array)
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"


# Classification of nodes
_INDICATORS = {NodeType.RSI, NodeType.SMA, NodeType.EMA, NodeType.MACD_HIST,
               NodeType.BB_UPPER, NodeType.BB_LOWER, NodeType.ATR, NodeType.VOL_RATIO,
               NodeType.HIGH_OF, NodeType.LOW_OF, NodeType.ROC, NodeType.STDEV}
_PRICE_REFS = {NodeType.CLOSE, NodeType.OPEN, NodeType.HIGH, NodeType.LOW, NodeType.VOLUME}
_COMPARISONS = {NodeType.GT, NodeType.LT, NodeType.CROSS_ABOVE, NodeType.CROSS_BELOW}
_LOGIC = {NodeType.AND, NodeType.OR, NodeType.NOT}
_ARITHMETIC = {NodeType.ADD, NodeType.SUB, NodeType.MUL, NodeType.DIV}
_LEAVES = _PRICE_REFS | {NodeType.CONST}


@dataclass
class TreeNode:
    """A node in the expression tree."""
    node_type: NodeType
    children: List['TreeNode'] = field(default_factory=list)
    value: float = 0.0          # for CONST nodes
    period: int = 14            # for indicator nodes (lookback period)

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def copy(self) -> 'TreeNode':
        return TreeNode(
            node_type=self.node_type,
            children=[c.copy() for c in self.children],
            value=self.value,
            period=self.period,
        )

    def to_string(self) -> str:
        if self.node_type == NodeType.CONST:
            return f"{self.value:.2f}"
        if self.node_type in _PRICE_REFS:
            return self.node_type.value.lower()
        if self.node_type in _INDICATORS:
            child_str = self.children[0].to_string() if self.children else "close"
            return f"{self.node_type.value}({child_str},{self.period})"
        if self.node_type == NodeType.NOT:
            return f"NOT({self.children[0].to_string()})" if self.children else "NOT(?)"
        if len(self.children) == 2:
            return f"({self.children[0].to_string()} {self.node_type.value} {self.children[1].to_string()})"
        return self.node_type.value


@dataclass
class GeneratedStrategy:
    """A complete generated strategy with entry and exit rules."""
    entry_rule: TreeNode            # expression tree → bool array
    exit_rule: TreeNode             # expression tree → bool array
    name: str = ""                  # auto-generated name
    fitness: float = 0.0
    sharpe: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    max_drawdown_pct: float = 0.0
    generation: int = 0

    def to_string(self) -> str:
        return f"ENTRY: {self.entry_rule.to_string()}\nEXIT: {self.exit_rule.to_string()}"


# ════════════════════════════════════════════════════════════════════════════
# Indicator computation (vectorised numpy)
# ════════════════════════════════════════════════════════════════════════════

def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    cumsum = np.cumsum(arr)
    cumsum[period:] = cumsum[period:] - cumsum[:-period]
    result = np.full_like(arr, np.nan)
    result[period - 1:] = cumsum[period - 1:] / period
    return result


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    result = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period:
        return result
    alpha = 2.0 / (period + 1)
    result[period - 1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def _rsi(arr: np.ndarray, period: int) -> np.ndarray:
    if len(arr) < period + 1:
        return np.full_like(arr, 50.0)
    delta = np.diff(arr)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _sma(gain, period)
    avg_loss = _sma(loss, period)
    rs = avg_gain / np.maximum(avg_loss, 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    result = np.full(len(arr), 50.0)
    result[1:] = rsi
    return np.nan_to_num(result, nan=50.0)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    if len(close) < 2:
        return np.zeros_like(close)
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    atr = np.zeros(len(close))
    atr[1:] = tr
    return _sma(atr, period)


# ════════════════════════════════════════════════════════════════════════════
# Tree evaluation
# ════════════════════════════════════════════════════════════════════════════

def evaluate_tree(
    node: TreeNode,
    data: Dict[str, np.ndarray],
) -> np.ndarray:
    """Evaluate an expression tree on OHLCV data. Returns numpy array."""
    T = len(data.get("close", []))
    if T == 0:
        return np.array([])

    close = data["close"]
    high = data.get("high", close)
    low = data.get("low", close)
    volume = data.get("volume", np.ones(T))

    try:
        # Leaf nodes
        if node.node_type == NodeType.CLOSE:
            return close.copy()
        if node.node_type == NodeType.OPEN:
            return data.get("open", close).copy()
        if node.node_type == NodeType.HIGH:
            return high.copy()
        if node.node_type == NodeType.LOW:
            return low.copy()
        if node.node_type == NodeType.VOLUME:
            return volume.copy()
        if node.node_type == NodeType.CONST:
            return np.full(T, node.value)

        # Indicators
        if node.node_type in _INDICATORS:
            child_arr = evaluate_tree(node.children[0], data) if node.children else close
            p = max(2, node.period)
            if node.node_type == NodeType.RSI:
                return _rsi(child_arr, p)
            if node.node_type == NodeType.SMA:
                return np.nan_to_num(_sma(child_arr, p), nan=child_arr[0] if len(child_arr) > 0 else 0)
            if node.node_type == NodeType.EMA:
                return np.nan_to_num(_ema(child_arr, p), nan=child_arr[0] if len(child_arr) > 0 else 0)
            if node.node_type == NodeType.ATR:
                return np.nan_to_num(_atr(high, low, close, p), nan=0)
            if node.node_type == NodeType.BB_UPPER:
                sma = _sma(close, p)
                std = np.full(T, 0.0)
                for i in range(p, T):
                    std[i] = np.std(close[i - p:i])
                return np.nan_to_num(sma + 2 * std, nan=close[0] if T > 0 else 0)
            if node.node_type == NodeType.BB_LOWER:
                sma = _sma(close, p)
                std = np.full(T, 0.0)
                for i in range(p, T):
                    std[i] = np.std(close[i - p:i])
                return np.nan_to_num(sma - 2 * std, nan=close[0] if T > 0 else 0)
            if node.node_type == NodeType.VOL_RATIO:
                vol_sma = _sma(volume, p)
                return np.nan_to_num(volume / np.maximum(vol_sma, 1e-9), nan=1.0)
            if node.node_type == NodeType.HIGH_OF:
                result = np.zeros(T)
                for i in range(p, T):
                    result[i] = np.max(high[i - p:i])
                return result
            if node.node_type == NodeType.LOW_OF:
                result = np.full(T, 1e18)
                for i in range(p, T):
                    result[i] = np.min(low[i - p:i])
                return result
            if node.node_type == NodeType.ROC:
                result = np.zeros(T)
                for i in range(p, T):
                    if child_arr[i - p] != 0:
                        result[i] = (child_arr[i] / child_arr[i - p] - 1) * 100
                return result
            if node.node_type == NodeType.STDEV:
                result = np.zeros(T)
                for i in range(p, T):
                    result[i] = np.std(child_arr[i - p:i])
                return result
            if node.node_type == NodeType.MACD_HIST:
                fast = _ema(child_arr, 12)
                slow = _ema(child_arr, 26)
                macd = fast - slow
                signal = _ema(np.nan_to_num(macd, nan=0), 9)
                return np.nan_to_num(macd - signal, nan=0)

        # Comparisons
        if node.node_type in _COMPARISONS and len(node.children) == 2:
            left = evaluate_tree(node.children[0], data)
            right = evaluate_tree(node.children[1], data)
            if node.node_type == NodeType.GT:
                return (left > right).astype(float)
            if node.node_type == NodeType.LT:
                return (left < right).astype(float)
            if node.node_type == NodeType.CROSS_ABOVE:
                prev = np.roll(left - right, 1)
                curr = left - right
                return ((curr > 0) & (prev <= 0)).astype(float)
            if node.node_type == NodeType.CROSS_BELOW:
                prev = np.roll(left - right, 1)
                curr = left - right
                return ((curr < 0) & (prev >= 0)).astype(float)

        # Logic
        if node.node_type == NodeType.AND and len(node.children) == 2:
            left = evaluate_tree(node.children[0], data)
            right = evaluate_tree(node.children[1], data)
            return ((left > 0.5) & (right > 0.5)).astype(float)
        if node.node_type == NodeType.OR and len(node.children) == 2:
            left = evaluate_tree(node.children[0], data)
            right = evaluate_tree(node.children[1], data)
            return ((left > 0.5) | (right > 0.5)).astype(float)
        if node.node_type == NodeType.NOT and len(node.children) >= 1:
            child = evaluate_tree(node.children[0], data)
            return (child <= 0.5).astype(float)

        # Arithmetic
        if node.node_type in _ARITHMETIC and len(node.children) == 2:
            left = evaluate_tree(node.children[0], data)
            right = evaluate_tree(node.children[1], data)
            if node.node_type == NodeType.ADD:
                return left + right
            if node.node_type == NodeType.SUB:
                return left - right
            if node.node_type == NodeType.MUL:
                return left * right
            if node.node_type == NodeType.DIV:
                return left / np.maximum(np.abs(right), 1e-9) * np.sign(right + 1e-12)

    except Exception:
        pass

    return np.zeros(T)


# ════════════════════════════════════════════════════════════════════════════
# Strategy backtesting
# ════════════════════════════════════════════════════════════════════════════

def backtest_generated_strategy(
    entry_signals: np.ndarray,
    exit_signals: np.ndarray,
    close: np.ndarray,
    fee_pct: float = 0.26,
) -> Dict[str, float]:
    """Backtest a generated strategy. Returns metrics dict."""
    T = len(close)
    trades = []
    in_position = False
    entry_price = 0.0

    for i in range(1, T):
        if not in_position and entry_signals[i] > 0.5:
            in_position = True
            entry_price = close[i]
        elif in_position and (exit_signals[i] > 0.5 or i == T - 1):
            gain = (close[i] / entry_price - 1) * 100 - 2 * fee_pct
            trades.append(gain)
            in_position = False

    if not trades:
        return {"sharpe": 0.0, "win_rate": 0.0, "trade_count": 0,
                "total_return": 0.0, "max_drawdown": 0.0}

    n = len(trades)
    mean_ret = sum(trades) / n
    std_ret = (sum((t - mean_ret) ** 2 for t in trades) / max(n - 1, 1)) ** 0.5
    sharpe = mean_ret / max(std_ret, 1e-9)
    wins = sum(1 for t in trades if t > 0)

    equity = [0.0]
    for t in trades:
        equity.append(equity[-1] + t)
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        peak = max(peak, e)
        max_dd = max(max_dd, peak - e)

    return {
        "sharpe": sharpe,
        "win_rate": wins / n,
        "trade_count": n,
        "total_return": sum(trades),
        "max_drawdown": max_dd,
    }


# ════════════════════════════════════════════════════════════════════════════
# Tree generation (random)
# ════════════════════════════════════════════════════════════════════════════

def _random_condition(rng: random.Random, max_depth: int = 3) -> TreeNode:
    """Generate a random boolean condition tree."""
    if max_depth <= 1:
        # Leaf comparison: indicator vs constant
        indicator = rng.choice([NodeType.RSI, NodeType.SMA, NodeType.EMA,
                                NodeType.VOL_RATIO, NodeType.ROC, NodeType.MACD_HIST])
        period = rng.randint(5, 50)
        price_ref = TreeNode(node_type=rng.choice([NodeType.CLOSE, NodeType.HIGH, NodeType.LOW]))
        ind_node = TreeNode(node_type=indicator, children=[price_ref], period=period)

        # Sensible constant ranges per indicator
        if indicator == NodeType.RSI:
            const_val = rng.uniform(15, 85)
        elif indicator == NodeType.VOL_RATIO:
            const_val = rng.uniform(0.5, 3.0)
        elif indicator == NodeType.ROC:
            const_val = rng.uniform(-5, 5)
        elif indicator == NodeType.MACD_HIST:
            const_val = rng.uniform(-2, 2)
        else:
            const_val = rng.uniform(0.5, 1.5)  # relative to price

        cmp = rng.choice([NodeType.GT, NodeType.LT, NodeType.CROSS_ABOVE, NodeType.CROSS_BELOW])
        const_node = TreeNode(node_type=NodeType.CONST, value=const_val)

        return TreeNode(node_type=cmp, children=[ind_node, const_node])

    # Internal node: AND or OR of two sub-conditions
    logic = rng.choice([NodeType.AND, NodeType.OR])
    left = _random_condition(rng, max_depth - 1)
    right = _random_condition(rng, max_depth - 1)
    return TreeNode(node_type=logic, children=[left, right])


def _random_exit_rule(rng: random.Random) -> TreeNode:
    """Generate a random exit rule (gain/loss based + optional indicator)."""
    # Gain-based exit: gain > tp_pct
    gain_node = TreeNode(node_type=NodeType.ROC, children=[TreeNode(node_type=NodeType.CLOSE)],
                         period=1)
    tp = rng.uniform(1.0, 5.0)
    sl = rng.uniform(0.5, 3.0)

    tp_exit = TreeNode(node_type=NodeType.GT,
                       children=[gain_node, TreeNode(node_type=NodeType.CONST, value=tp)])
    sl_exit = TreeNode(node_type=NodeType.LT,
                       children=[gain_node.copy(), TreeNode(node_type=NodeType.CONST, value=-sl)])

    # OR: take profit OR stop loss
    return TreeNode(node_type=NodeType.OR, children=[tp_exit, sl_exit])


# ════════════════════════════════════════════════════════════════════════════
# Genetic Programming operators
# ════════════════════════════════════════════════════════════════════════════

def _get_all_nodes(tree: TreeNode) -> List[TreeNode]:
    """Flatten tree to list of all nodes."""
    nodes = [tree]
    for c in tree.children:
        nodes.extend(_get_all_nodes(c))
    return nodes


def _subtree_crossover(p1: TreeNode, p2: TreeNode, rng: random.Random) -> TreeNode:
    """Replace a random subtree in p1 with a random subtree from p2."""
    child = p1.copy()
    nodes1 = _get_all_nodes(child)
    nodes2 = _get_all_nodes(p2)
    if len(nodes1) < 2 or not nodes2:
        return child

    # Pick a non-root node in child to replace
    parent_candidates = [n for n in nodes1 if n.children]
    if not parent_candidates:
        return child
    parent = rng.choice(parent_candidates)
    idx = rng.randrange(len(parent.children))

    # Pick a subtree from p2
    donor = rng.choice(nodes2).copy()
    parent.children[idx] = donor
    return child


def _point_mutation(tree: TreeNode, rng: random.Random) -> TreeNode:
    """Mutate a single node: change constant, period, or operator."""
    child = tree.copy()
    nodes = _get_all_nodes(child)
    if not nodes:
        return child

    target = rng.choice(nodes)

    if target.node_type == NodeType.CONST:
        target.value *= rng.uniform(0.7, 1.3)
    elif target.node_type in _INDICATORS:
        target.period = max(2, target.period + rng.randint(-5, 5))
    elif target.node_type in _COMPARISONS:
        target.node_type = rng.choice(list(_COMPARISONS))
    elif target.node_type in _LOGIC - {NodeType.NOT}:
        target.node_type = rng.choice([NodeType.AND, NodeType.OR])

    return child


def _hoist_mutation(tree: TreeNode, rng: random.Random) -> TreeNode:
    """Replace tree with a random subtree (simplification)."""
    nodes = _get_all_nodes(tree)
    subtrees = [n for n in nodes if n.children and n.depth() < tree.depth()]
    if not subtrees:
        return tree.copy()
    return rng.choice(subtrees).copy()


# ════════════════════════════════════════════════════════════════════════════
# Strategy Generator (Genetic Programming engine)
# ════════════════════════════════════════════════════════════════════════════

class StrategyGenerator:
    """
    Genetic Programming engine that invents new trading strategies.

    Evolves expression trees that define entry/exit rules from atomic
    building blocks (indicators, comparisons, logic operators).

    Usage:
        gen = StrategyGenerator(population_size=50)
        gen.initialize()
        for _ in range(20):
            result = gen.evolve(market_data)
        best = gen.get_best()
        logger.info(best.to_string())
    """

    def __init__(
        self,
        population_size: int = 50,
        max_tree_depth: int = 4,
        tournament_size: int = 5,
        crossover_rate: float = 0.5,
        mutation_rate: float = 0.3,
        hoist_rate: float = 0.1,
        min_trades: int = 5,
        max_drawdown_pct: float = 20.0,
        fee_pct: float = 0.26,
        seed: Optional[int] = None,
    ):
        self._pop_size = population_size
        self._max_depth = max_tree_depth
        self._tournament_size = tournament_size
        self._crossover_rate = crossover_rate
        self._mutation_rate = mutation_rate
        self._hoist_rate = hoist_rate
        self._min_trades = min_trades
        self._max_dd = max_drawdown_pct
        self._fee = fee_pct
        self._rng = random.Random(seed)
        self._population: List[GeneratedStrategy] = []
        self._generation = 0
        self._best_ever: Optional[GeneratedStrategy] = None
        self._hall_of_fame: List[GeneratedStrategy] = []

    def initialize(self) -> None:
        """Create random initial population."""
        self._population = []
        for i in range(self._pop_size):
            entry = _random_condition(self._rng, self._max_depth)
            exit_rule = _random_exit_rule(self._rng)
            strat = GeneratedStrategy(
                entry_rule=entry, exit_rule=exit_rule,
                name=f"gen0_strat{i}",
            )
            self._population.append(strat)
        logger.info("StrategyGenerator: initialized %d random strategies", len(self._population))

    def evolve(self, market_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Run one generation of GP evolution.

        Args:
            market_data: {close, high, low, volume} numpy arrays

        Returns:
            Dict with generation stats.
        """
        if not self._population:
            self.initialize()

        t0 = time.time()
        self._generation += 1

        # Evaluate fitness with walk-forward validation + parsimony pressure
        T = len(market_data.get("close", []))
        _split = int(T * 0.75)  # 75% train, 25% test
        _is_data = {k: v[:_split] for k, v in market_data.items()} if _split > 50 else market_data
        _oos_data = {k: v[_split:] for k, v in market_data.items()} if T - _split > 20 else None

        for i, strat in enumerate(self._population):
            if strat.trade_count == 0 and strat.sharpe == 0.0:
                try:
                    # In-sample evaluation
                    entry_signals = evaluate_tree(strat.entry_rule, _is_data)
                    exit_signals = evaluate_tree(strat.exit_rule, _is_data)
                    is_metrics = backtest_generated_strategy(
                        entry_signals, exit_signals, _is_data["close"], self._fee,
                    )

                    # Out-of-sample evaluation
                    oos_sharpe = 0.0
                    if _oos_data and is_metrics["trade_count"] >= 3:
                        oos_entry = evaluate_tree(strat.entry_rule, _oos_data)
                        oos_exit = evaluate_tree(strat.exit_rule, _oos_data)
                        oos_metrics = backtest_generated_strategy(
                            oos_entry, oos_exit, _oos_data["close"], self._fee,
                        )
                        oos_sharpe = oos_metrics["sharpe"]

                    # Parsimony pressure: penalise complex trees (Occam's razor)
                    tree_complexity = strat.entry_rule.size() + strat.exit_rule.size()
                    parsimony_penalty = max(0, tree_complexity - 10) * 0.02  # -0.02 per node over 10

                    # Overfitting penalty: IS >> OOS = overfit
                    overfit_penalty = 0.0
                    if is_metrics["sharpe"] > 0 and oos_sharpe > 0:
                        degradation = 1.0 - (oos_sharpe / is_metrics["sharpe"])
                        overfit_penalty = max(0, degradation) * 0.3

                    # Combined fitness
                    raw_fitness = is_metrics["sharpe"] * (1 - min(1, is_metrics["max_drawdown"] / max(self._max_dd, 1)))
                    fitness = raw_fitness + oos_sharpe * 0.4 - parsimony_penalty - overfit_penalty

                    self._population[i] = GeneratedStrategy(
                        entry_rule=strat.entry_rule,
                        exit_rule=strat.exit_rule,
                        name=strat.name,
                        fitness=fitness,
                        sharpe=is_metrics["sharpe"],
                        win_rate=is_metrics["win_rate"],
                        trade_count=is_metrics["trade_count"],
                        max_drawdown_pct=is_metrics["max_drawdown"],
                        generation=self._generation,
                    )
                except Exception:
                    pass

        # Sort by fitness
        self._population.sort(key=lambda s: s.fitness, reverse=True)

        # Track best
        if self._population and (self._best_ever is None or self._population[0].fitness > self._best_ever.fitness):
            self._best_ever = self._population[0]
            # Add to hall of fame
            self._hall_of_fame.append(self._best_ever)
            if len(self._hall_of_fame) > 20:
                self._hall_of_fame = sorted(self._hall_of_fame, key=lambda s: s.fitness, reverse=True)[:20]

        # Selection + reproduction
        elite_n = max(2, self._pop_size // 10)
        next_gen = list(self._population[:elite_n])

        while len(next_gen) < self._pop_size:
            r = self._rng.random()
            if r < self._crossover_rate and len(self._population) >= 2:
                p1 = self._tournament_select()
                p2 = self._tournament_select()
                child_entry = _subtree_crossover(p1.entry_rule, p2.entry_rule, self._rng)
                child_exit = _subtree_crossover(p1.exit_rule, p2.exit_rule, self._rng)
                # Limit depth
                if child_entry.depth() > self._max_depth + 2:
                    child_entry = _hoist_mutation(child_entry, self._rng)
                child = GeneratedStrategy(entry_rule=child_entry, exit_rule=child_exit,
                                          name=f"gen{self._generation}_cross")
            elif r < self._crossover_rate + self._mutation_rate:
                parent = self._tournament_select()
                child_entry = _point_mutation(parent.entry_rule, self._rng)
                child_exit = _point_mutation(parent.exit_rule, self._rng)
                child = GeneratedStrategy(entry_rule=child_entry, exit_rule=child_exit,
                                          name=f"gen{self._generation}_mut")
            elif r < self._crossover_rate + self._mutation_rate + self._hoist_rate:
                parent = self._tournament_select()
                child = GeneratedStrategy(
                    entry_rule=_hoist_mutation(parent.entry_rule, self._rng),
                    exit_rule=parent.exit_rule.copy(),
                    name=f"gen{self._generation}_hoist",
                )
            else:
                # Random new strategy (exploration)
                entry = _random_condition(self._rng, self._max_depth)
                exit_rule = _random_exit_rule(self._rng)
                child = GeneratedStrategy(entry_rule=entry, exit_rule=exit_rule,
                                          name=f"gen{self._generation}_new")
            next_gen.append(child)

        self._population = next_gen[:self._pop_size]
        duration_ms = (time.time() - t0) * 1000

        best = self._population[0] if self._population else None
        if best and best.fitness > 0.3:
            logger.info(
                "StrategyGenerator gen %d: best fitness=%.3f sharpe=%.3f wr=%.0f%% trades=%d\n  %s",
                self._generation, best.fitness, best.sharpe,
                best.win_rate * 100, best.trade_count,
                best.entry_rule.to_string(),
            )

        return {
            "generation": self._generation,
            "population_size": len(self._population),
            "best_fitness": best.fitness if best else 0.0,
            "best_sharpe": best.sharpe if best else 0.0,
            "best_win_rate": best.win_rate if best else 0.0,
            "best_trades": best.trade_count if best else 0,
            "best_rule": best.entry_rule.to_string() if best else "",
            "hall_of_fame_size": len(self._hall_of_fame),
            "duration_ms": duration_ms,
        }

    def _tournament_select(self) -> GeneratedStrategy:
        k = min(self._tournament_size, len(self._population))
        candidates = self._rng.sample(self._population, k)
        return max(candidates, key=lambda s: s.fitness)

    def get_best(self) -> Optional[GeneratedStrategy]:
        return self._best_ever

    def get_hall_of_fame(self) -> List[GeneratedStrategy]:
        return list(self._hall_of_fame)

    def get_top(self, n: int = 5) -> List[GeneratedStrategy]:
        return sorted(self._population, key=lambda s: s.fitness, reverse=True)[:n]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "generation": self._generation,
            "population_size": len(self._population),
            "best_ever_fitness": self._best_ever.fitness if self._best_ever else 0.0,
            "best_ever_rule": self._best_ever.entry_rule.to_string() if self._best_ever else "",
            "hall_of_fame_size": len(self._hall_of_fame),
        }
