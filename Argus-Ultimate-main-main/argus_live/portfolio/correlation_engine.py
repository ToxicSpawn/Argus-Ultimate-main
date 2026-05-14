from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class CorrelationPair:
    left: str
    right: str
    correlation: float


@dataclass(frozen=True)
class CorrelationReport:
    strategy_pairs: List[CorrelationPair]
    symbol_pairs: List[CorrelationPair]
    average_strategy_correlation: float
    average_symbol_correlation: float
    reason: str


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation coefficient. Returns 0.0 on degenerate input."""
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    mx = _mean(xs[:n])
    my = _mean(ys[:n])
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


def _pairwise(
    series_map: Dict[str, Sequence[float]],
) -> List[CorrelationPair]:
    keys = sorted(series_map.keys())
    pairs: List[CorrelationPair] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            c = _corr(series_map[keys[i]], series_map[keys[j]])
            pairs.append(CorrelationPair(left=keys[i], right=keys[j], correlation=c))
    return pairs


def build_correlation_report(
    strategy_return_series: Dict[str, Sequence[float]],
    symbol_return_series: Dict[str, Sequence[float]],
) -> CorrelationReport:
    strat_pairs = _pairwise(strategy_return_series)
    sym_pairs = _pairwise(symbol_return_series)
    avg_strat = _mean([p.correlation for p in strat_pairs])
    avg_sym = _mean([p.correlation for p in sym_pairs])
    return CorrelationReport(
        strategy_pairs=strat_pairs,
        symbol_pairs=sym_pairs,
        average_strategy_correlation=avg_strat,
        average_symbol_correlation=avg_sym,
        reason=(
            f"correlation report: {len(strat_pairs)} strategy pairs "
            f"(avg {avg_strat:.4f}), {len(sym_pairs)} symbol pairs "
            f"(avg {avg_sym:.4f})"
        ),
    )
