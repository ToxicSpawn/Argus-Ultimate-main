"""Batch 2 — Level 2 order-book depth analyser.

Consumes L2 snapshots and produces micro-structure signals:
  * Order book imbalance (OBI)
  * Weighted mid-price
  * Cumulative depth at N levels
  * Large-order detection (iceberg proxy)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Book = List[Tuple[float, float]]  # [(price, size), ...]


@dataclass
class DepthSignals:
    symbol: str
    mid_price: float
    weighted_mid: float
    obi: float              # [-1, 1] bid-heavy → +1, ask-heavy → -1
    bid_depth_n: float      # cumulative bid size at N levels
    ask_depth_n: float
    depth_ratio: float      # bid_depth / ask_depth
    large_bid: bool         # unusually large single bid level
    large_ask: bool
    spread_bps: float


class L2DepthAnalyser:
    """Compute micro-structure signals from L2 order book snapshots."""

    def __init__(
        self,
        n_levels: int = 10,
        large_order_z: float = 3.0,
        history_len: int = 200,
    ) -> None:
        self._n = n_levels
        self._large_z = large_order_z
        self._history_len = history_len
        # Running size history per side
        self._bid_sizes: List[float] = []
        self._ask_sizes: List[float] = []

    def analyse(self, symbol: str, bids: Book, asks: Book) -> DepthSignals:
        """Process one L2 snapshot and return depth signals."""
        if not bids or not asks:
            return self._empty(symbol)

        bids_n = bids[: self._n]
        asks_n = asks[: self._n]

        best_bid = bids_n[0][0]
        best_ask = asks_n[0][0]
        mid = (best_bid + best_ask) / 2.0
        spread_bps = (best_ask - best_bid) / mid * 10_000

        bid_sizes = np.array([s for _, s in bids_n])
        ask_sizes = np.array([s for _, s in asks_n])
        bid_prices = np.array([p for p, _ in bids_n])
        ask_prices = np.array([p for p, _ in asks_n])

        bid_depth = float(bid_sizes.sum())
        ask_depth = float(ask_sizes.sum())
        total_depth = bid_depth + ask_depth
        obi = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0

        # Weighted mid-price
        weighted_mid = float(
            (bid_prices * bid_sizes).sum() + (ask_prices * ask_sizes).sum()
        ) / (bid_depth + ask_depth)

        # Large order detection
        large_bid, large_ask = self._detect_large(
            float(bid_sizes.max()), float(ask_sizes.max())
        )

        self._update_history(bid_depth, ask_depth)

        return DepthSignals(
            symbol=symbol,
            mid_price=mid,
            weighted_mid=weighted_mid,
            obi=obi,
            bid_depth_n=bid_depth,
            ask_depth_n=ask_depth,
            depth_ratio=bid_depth / ask_depth if ask_depth > 0 else float("inf"),
            large_bid=large_bid,
            large_ask=large_ask,
            spread_bps=spread_bps,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_large(self, max_bid: float, max_ask: float) -> Tuple[bool, bool]:
        if len(self._bid_sizes) < 10:
            return False, False
        b_arr = np.array(self._bid_sizes)
        a_arr = np.array(self._ask_sizes)
        large_bid = max_bid > b_arr.mean() + self._large_z * b_arr.std()
        large_ask = max_ask > a_arr.mean() + self._large_z * a_arr.std()
        return bool(large_bid), bool(large_ask)

    def _update_history(self, bid: float, ask: float) -> None:
        self._bid_sizes.append(bid)
        self._ask_sizes.append(ask)
        if len(self._bid_sizes) > self._history_len:
            self._bid_sizes.pop(0)
            self._ask_sizes.pop(0)

    @staticmethod
    def _empty(symbol: str) -> DepthSignals:
        return DepthSignals(
            symbol=symbol,
            mid_price=0.0,
            weighted_mid=0.0,
            obi=0.0,
            bid_depth_n=0.0,
            ask_depth_n=0.0,
            depth_ratio=1.0,
            large_bid=False,
            large_ask=False,
            spread_bps=0.0,
        )
