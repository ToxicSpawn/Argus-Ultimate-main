"""
RAM (64GB): in-memory cache config for OHLCV, order book, strategy state.

MarketDataService already has TTL caches; this adds max_size and preload options.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def cache_config(
    *,
    ohlcv_max_entries: int = 500,
    ticker_max_entries: int = 100,
    order_book_max_entries: int = 200,
    preload_symbols: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Return config dict for in-memory caches (OHLCV, order book, strategy state).
    Pass to MarketDataService or cache layer when supported.
    """
    return {
        "ohlcv_max_entries": int(ohlcv_max_entries),
        "ticker_max_entries": int(ticker_max_entries),
        "order_book_max_entries": int(order_book_max_entries),
        "preload_symbols": list(preload_symbols or []),
    }
