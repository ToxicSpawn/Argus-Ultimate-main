"""
Microstructure signals from order book: spread, flow imbalance.

Used to boost/penalize signal confidence (e.g. tight spread + positive flow -> boost).
Leverage 10GbE for low-latency order book feeds when available.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def spread_bps(bids: list, asks: list) -> float:
    """
    Mid-to-spread in bps. bids/asks are [(price, size), ...] or dict with 'bids'/'asks'.
    Returns 0 if insufficient data.
    """
    try:
        if isinstance(bids, dict):
            bids = (bids.get("bids") or bids.get("bids_list") or [])[:20]
            asks = (asks.get("asks") or asks.get("asks_list") or [])[:20]
        if not bids or not asks:
            return 0.0
        def top(l):
            if not l:
                return None
            x = l[0]
            return float(x[0]) if isinstance(x, (list, tuple)) else float(x.get("price", x.get("p", 0)))
        b = top(bids)
        a = top(asks)
        if b is None or a is None or b <= 0:
            return 0.0
        mid = (b + a) / 2.0
        return (a - b) / mid * 10000.0 if mid else 0.0
    except Exception as e:
        logger.debug("spread_bps: %s", e)
        return 0.0


def flow_imbalance(bids: list, asks: list, depth: int = 10) -> float:
    """
    Order book flow imbalance in [-1, 1]. Positive = more bid pressure.
    Entries are (price, size) or dict with price/size.
    """
    try:
        if isinstance(bids, dict):
            bids = (bids.get("bids") or bids.get("bids_list") or [])[:depth]
            asks = (asks.get("asks") or asks.get("asks_list") or [])[:depth]
        def vol(l):
            return sum(
                float(x[1]) if isinstance(x, (list, tuple)) else float(x.get("size", x.get("s", 0)) or 0)
                for x in (l or [])[:depth]
            )
        bv = vol(bids)
        av = vol(asks)
        total = bv + av
        if total <= 0:
            return 0.0
        return (bv - av) / total
    except Exception as e:
        logger.debug("flow_imbalance: %s", e)
        return 0.0


def microstructure_boost(
    order_book: Optional[Dict[str, Any]] = None,
    *,
    spread_bps_max: float = 20.0,
    flow_boost: float = 0.1,
) -> float:
    """
    Returns a confidence multiplier in [0.9, 1.1] from spread and flow.
    Tight spread + positive flow -> boost; wide spread or negative flow -> slight penalty.
    """
    if not order_book or not isinstance(order_book, dict):
        return 1.0
    bids = order_book.get("bids") or order_book.get("bids_list") or []
    asks = order_book.get("asks") or order_book.get("asks_list") or []
    sp = spread_bps(bids, asks)
    flow = flow_imbalance(bids, asks)
    # Lower spread = better (multiply up to 1.05); flow in [-1,1] adds ±flow_boost
    spread_term = 1.0 - (min(sp, spread_bps_max) / spread_bps_max) * 0.05
    flow_term = flow * flow_boost
    return max(0.9, min(1.1, spread_term + flow_term))
