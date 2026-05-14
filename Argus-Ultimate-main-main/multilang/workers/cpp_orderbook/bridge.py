"""
C++ Order Book engine bridge for ARGUS.

Tries the compiled C++ binary first (JSON stdin/stdout protocol).
Falls back to a pure-Python dict-based order book when the binary is not available.

Build:
    cd multilang/workers/cpp_orderbook
    mkdir build && cd build && cmake .. && make
    # or: g++ -O3 -std=c++17 -o cpp_orderbook orderbook.cpp

Usage:
    book = CppOrderBook()
    book.update("bid", 65000.0, 1.5)
    book.update("ask", 65010.0, 0.8)
    state = book.get_state()
    vwap = book.get_vwap(10000.0)
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_BINARY_NAME = "cpp_orderbook.exe" if platform.system() == "Windows" else "cpp_orderbook"
_BINARY_PATH = _THIS_DIR / _BINARY_NAME


class CppOrderBook:
    """High-performance L2 order book via C++ subprocess, with Python fallback."""

    def __init__(self, binary_path: Optional[str] = None) -> None:
        self._binary = Path(binary_path) if binary_path else _BINARY_PATH
        self._native_available = self._binary.is_file()
        self._backend = "native" if self._native_available else "fallback"
        self._call_count = 0
        self._total_latency = 0.0

        # Fallback state: Python dict-based order book
        # bids: {price: quantity} — we want descending order for best bid
        # asks: {price: quantity} — we want ascending order for best ask
        self._bids: Dict[float, float] = {}
        self._asks: Dict[float, float] = {}

        if self._native_available:
            logger.info("CppOrderBook: native binary found at %s", self._binary)
        else:
            logger.info("CppOrderBook: binary not found, using Python fallback")

    @property
    def available(self) -> bool:
        return self._native_available

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def avg_latency_ms(self) -> float:
        if self._call_count == 0:
            return 0.0
        return (self._total_latency / self._call_count) * 1000.0

    # ── Public API ────────────────────────────────────────────────────

    def update(self, side: str, price: float, quantity: float) -> None:
        """
        Apply an L2 order book update.

        Args:
            side: "bid" or "ask"
            price: Price level.
            quantity: New quantity at this level (0 to remove).
        """
        t0 = time.monotonic()
        try:
            self._fb_update(side, price, quantity)
        finally:
            self._call_count += 1
            self._total_latency += time.monotonic() - t0

    def get_state(self) -> Dict[str, Any]:
        """
        Get full order book state: mid, spread, imbalance, walls.

        Returns:
            {"mid_price": float, "spread_bps": float, "imbalance": float,
             "walls": [{"side": str, "price": float, "quantity": float, "multiple": float}],
             "best_bid": float, "best_ask": float, "bid_levels": int, "ask_levels": int}
        """
        t0 = time.monotonic()
        try:
            return self._fb_get_state()
        finally:
            self._call_count += 1
            self._total_latency += time.monotonic() - t0

    def get_vwap(self, depth_usd: float) -> float:
        """
        Compute volume-weighted average price for a given USD depth on the ask side.

        Args:
            depth_usd: Total USD to fill.

        Returns:
            VWAP price.
        """
        t0 = time.monotonic()
        try:
            return self._fb_get_vwap(depth_usd)
        finally:
            self._call_count += 1
            self._total_latency += time.monotonic() - t0

    def clear(self) -> None:
        """Reset the order book."""
        self._bids.clear()
        self._asks.clear()

    # ── Fallback implementations (Python) ─────────────────────────────

    def _fb_update(self, side: str, price: float, quantity: float) -> None:
        book = self._bids if side in ("bid", "buy") else self._asks
        if quantity <= 1e-15:
            book.pop(price, None)
        else:
            book[price] = quantity

    def _fb_get_state(self) -> Dict[str, Any]:
        mid = self._fb_mid_price()
        spread = self._fb_spread_bps()
        imbalance = self._fb_imbalance()
        walls = self._fb_detect_walls()
        best_bid = max(self._bids.keys()) if self._bids else 0.0
        best_ask = min(self._asks.keys()) if self._asks else 0.0

        return {
            "mid_price": mid,
            "spread_bps": spread,
            "imbalance": imbalance,
            "walls": walls,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_levels": len(self._bids),
            "ask_levels": len(self._asks),
        }

    def _fb_mid_price(self) -> float:
        if not self._bids or not self._asks:
            return 0.0
        best_bid = max(self._bids.keys())
        best_ask = min(self._asks.keys())
        return (best_bid + best_ask) / 2.0

    def _fb_spread_bps(self) -> float:
        if not self._bids or not self._asks:
            return 0.0
        best_bid = max(self._bids.keys())
        best_ask = min(self._asks.keys())
        mid = (best_bid + best_ask) / 2.0
        if mid < 1e-15:
            return 0.0
        return ((best_ask - best_bid) / mid) * 10000.0

    def _fb_imbalance(self, levels: int = 5) -> float:
        sorted_bids = sorted(self._bids.keys(), reverse=True)[:levels]
        sorted_asks = sorted(self._asks.keys())[:levels]

        bid_vol = sum(self._bids[p] for p in sorted_bids)
        ask_vol = sum(self._asks[p] for p in sorted_asks)
        total = bid_vol + ask_vol
        if total < 1e-15:
            return 0.0
        return (bid_vol - ask_vol) / total

    def _fb_detect_walls(self, min_size_multiple: float = 5.0) -> List[Dict[str, Any]]:
        walls = []

        # Average bid size
        if self._bids:
            avg_bid = sum(self._bids.values()) / len(self._bids)
            if avg_bid > 1e-15:
                for price, qty in self._bids.items():
                    mult = qty / avg_bid
                    if mult >= min_size_multiple:
                        walls.append({
                            "side": "bid",
                            "price": price,
                            "quantity": qty,
                            "multiple": round(mult, 2),
                        })

        # Average ask size
        if self._asks:
            avg_ask = sum(self._asks.values()) / len(self._asks)
            if avg_ask > 1e-15:
                for price, qty in self._asks.items():
                    mult = qty / avg_ask
                    if mult >= min_size_multiple:
                        walls.append({
                            "side": "ask",
                            "price": price,
                            "quantity": qty,
                            "multiple": round(mult, 2),
                        })

        return walls

    def _fb_get_vwap(self, depth_usd: float) -> float:
        if not self._asks:
            return 0.0
        sorted_asks = sorted(self._asks.items())  # ascending price
        total_cost = 0.0
        total_qty = 0.0

        for price, qty in sorted_asks:
            level_cost = price * qty
            if total_cost + level_cost >= depth_usd:
                remaining = depth_usd - total_cost
                partial_qty = remaining / price
                total_qty += partial_qty
                total_cost += remaining
                break
            total_cost += level_cost
            total_qty += qty

        if total_qty < 1e-15:
            return 0.0
        return total_cost / total_qty
