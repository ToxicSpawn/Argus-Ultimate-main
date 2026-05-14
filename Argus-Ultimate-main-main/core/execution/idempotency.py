#!/usr/bin/env python3
"""
Argus Trading Bot - Order Idempotency Core
Prevents duplicate order submissions using deterministic client order IDs
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderIntent:
    """Immutable order intent for idempotency"""

    strategy: str
    symbol: str
    side: str
    qty: float
    price: float | None
    time_bucket_s: int  # Time bucket in seconds (e.g., 60 for 1-minute buckets)


def client_order_id(intent: OrderIntent) -> str:
    """
    Generate deterministic client order ID from order intent

    Uses SHA256 hash of order parameters to ensure idempotency.
    Same parameters in same time bucket = same order ID.

    Args:
        intent: OrderIntent with all order parameters

    Returns:
        Deterministic client order ID (e.g., "argus_a1b2c3d4e5f6...")

    Example:
        >>> intent = OrderIntent(
        ...     strategy="core_portfolio",
        ...     symbol="BTC/USDT",
        ...     side="buy",
        ...     qty=0.1,
        ...     price=50000.0,
        ...     time_bucket_s=60
        ... )
        >>> order_id = client_order_id(intent)
        >>> logger.debug(order_id)
        argus_a1b2c3d4e5f6...
    """
    # Create deterministic string from intent
    raw = (
        f"{intent.strategy}|{intent.symbol}|{intent.side}|"
        f"{intent.qty:.8f}|{intent.price or 'market'}|{intent.time_bucket_s}"
    )

    # Generate hash
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    return f"argus_{h}"


def get_time_bucket(timestamp: datetime | None = None, bucket_seconds: int = 60) -> int:
    """
    Get time bucket for order idempotency

    Args:
        timestamp: Timestamp (defaults to now)
        bucket_seconds: Bucket size in seconds (default: 60)

    Returns:
        Time bucket integer
    """
    if timestamp is None:
        timestamp = datetime.now()

    return int(timestamp.timestamp() // bucket_seconds)
