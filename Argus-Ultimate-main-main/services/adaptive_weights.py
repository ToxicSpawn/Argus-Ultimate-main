"""
Adaptive strategy weights: update weights from recent Sharpe or win rate per strategy.
Used by strategy synergy / router to tune allocation.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# strategy_id -> (timestamp, list of recent returns or pnls)
_strategy_returns: Dict[str, List[float]] = {}
_strategy_weights: Dict[str, float] = {}
UPDATE_INTERVAL_S = 300.0
_last_update_ts = 0.0


def record_strategy_result(strategy_id: str, pnl: float) -> None:
    """Record a strategy's trade result for adaptive weight tuning."""
    if strategy_id not in _strategy_returns:
        _strategy_returns[strategy_id] = []
    _strategy_returns[strategy_id].append(pnl)
    while len(_strategy_returns[strategy_id]) > 200:
        _strategy_returns[strategy_id].pop(0)


def _sharpe_like(returns: List[float]) -> float:
    if not returns:
        return 0.0
    import numpy as np
    r = np.array(returns)
    mean_r = np.mean(r)
    std_r = np.std(r)
    if std_r <= 0:
        return 0.0
    return float(mean_r / std_r)


def get_adaptive_weights(
    strategy_ids: List[str],
    base_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Update weights from recent performance (Sharpe-like ratio per strategy).
    Returns normalized weights; if no history, returns base_weights or equal.
    """
    global _last_update_ts
    now = time.time()
    if base_weights is None:
        base_weights = {s: 1.0 / max(len(strategy_ids), 1) for s in strategy_ids}
    out: Dict[str, float] = {}
    for sid in strategy_ids:
        returns = _strategy_returns.get(sid) or []
        sharpe = _sharpe_like(returns)
        base = base_weights.get(sid, 1.0 / max(len(strategy_ids), 1))
        # Boost weight for positive Sharpe, reduce for negative
        adj = 1.0 + sharpe * 0.5 if sharpe > 0 else 1.0 / (1.0 - sharpe * 0.5) if sharpe < 0 else 1.0
        out[sid] = max(0.01, base * adj)
    total = sum(out.values())
    if total <= 0:
        return base_weights
    return {s: out[s] / total for s in out}
