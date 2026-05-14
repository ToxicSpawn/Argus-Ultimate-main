"""Opportunity scoring and ranking engine."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Opportunity:
    strategy_id: str
    symbol: str
    venue: str
    score: float
    expected_edge_bps: float
    confidence: float
    capital_priority: float
    reason: str


def score_opportunity(
    strategy_id: str,
    symbol: str,
    venue: str,
    expected_edge_bps: float,
    confidence: float,
    regime_score: float,
    alpha_score: float,
    efficiency_score: float,
    capital_priority: float = 1.0,
) -> Opportunity:
    """Score a single trading opportunity using a weighted formula.

    score = edge*0.35 + conf*10*0.20 + regime*10*0.15 + alpha*0.20 + efficiency*10*0.10
    """
    score = (
        expected_edge_bps * 0.35
        + confidence * 10.0 * 0.20
        + regime_score * 10.0 * 0.15
        + alpha_score * 0.20
        + efficiency_score * 10.0 * 0.10
    )

    reason = (
        f"edge={expected_edge_bps:.2f}*0.35 + conf={confidence:.2f}*10*0.20 + "
        f"regime={regime_score:.2f}*10*0.15 + alpha={alpha_score:.2f}*0.20 + "
        f"eff={efficiency_score:.2f}*10*0.10 => {score:.2f}"
    )

    logger.debug("opportunity scored: %s/%s score=%.2f", strategy_id, symbol, score)

    return Opportunity(
        strategy_id=strategy_id,
        symbol=symbol,
        venue=venue,
        score=score,
        expected_edge_bps=expected_edge_bps,
        confidence=confidence,
        capital_priority=capital_priority,
        reason=reason,
    )


def rank_opportunities(opportunities: List[Opportunity]) -> List[Opportunity]:
    """Rank opportunities by (-capital_priority, -score, -confidence)."""
    return sorted(
        opportunities,
        key=lambda o: (-o.capital_priority, -o.score, -o.confidence),
    )
