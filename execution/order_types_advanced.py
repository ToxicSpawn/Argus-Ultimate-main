"""
Advanced order types: IOC, FOK, GTD, hidden, pegged, post-only, reduce-only.

Use where venue supports them; fallback to limit/market when not available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TimeInForce(Enum):
    GTC = "GTC"
    IOC = "IOC"   # Immediate-or-Cancel
    FOK = "FOK"   # Fill-or-Kill
    GTD = "GTD"   # Good-Till-Date


@dataclass
class AdvancedOrderSpec:
    symbol: str
    side: str
    size: float
    order_type: str          # limit, market, ioc, fok, stop
    price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    hidden: bool = False
    pegged: bool = False
    peg_offset_bps: Optional[float] = None  # e.g. -5 = 5 bps below mid
    post_only: bool = False                  # maker-only; reject if would take
    reduce_only: bool = False                # futures: only reduce open position
    client_order_id: Optional[str] = None
    expire_time: Optional[str] = None        # ISO-8601 string; used with GTD


def normalize_to_venue(
    spec: AdvancedOrderSpec,
    venue: str,
    mid_price: float,
) -> Dict[str, Any]:
    """
    Map advanced order spec to venue-specific API payload.
    Venues that don't support hidden/pegged get standard limit/market.
    """
    payload: Dict[str, Any] = {
        "symbol": spec.symbol,
        "side":   spec.side,
        "size":   spec.size,
        "client_order_id": spec.client_order_id,
    }
    if spec.order_type in ("market", "market_order"):
        payload["type"] = "market"
        return payload

    # Time-in-force
    if spec.time_in_force == TimeInForce.IOC:
        payload["time_in_force"] = "IOC"
    elif spec.time_in_force == TimeInForce.FOK:
        payload["time_in_force"] = "FOK"
    elif spec.time_in_force == TimeInForce.GTD:
        payload["time_in_force"] = "GTD"
        if spec.expire_time:
            payload["expire_time"] = spec.expire_time

    # Price
    if spec.price is not None:
        payload["price"] = spec.price
        payload["type"]  = "limit"
    elif spec.pegged and spec.peg_offset_bps is not None:
        payload["type"] = "limit"
        if spec.side == "buy":
            payload["price"] = mid_price * (1 + spec.peg_offset_bps / 10_000.0)
        else:
            payload["price"] = mid_price * (1 - spec.peg_offset_bps / 10_000.0)

    # Execution flags
    if spec.post_only:
        if venue in ("bybit", "kraken", "binance", "okx"):
            payload["postOnly"] = True
        else:
            payload["post_only"] = True

    if spec.reduce_only:
        if venue in ("bybit", "binance", "okx"):
            payload["reduceOnly"] = True
        else:
            payload["reduce_only"] = True

    if spec.hidden:
        payload["hidden"] = True

    return payload
