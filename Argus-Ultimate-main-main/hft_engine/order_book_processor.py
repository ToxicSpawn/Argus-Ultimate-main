"""
Production-grade L2/L3 Order Book Processor.

Components
----------
- PriceLevel           — immutable dataclass for a single price level
- L2OrderBook          — price-level aggregated book with alpha signals
- L3OrderBook          — individual-order book with queue-position tracking
- OrderBookSignals     — rolling-window alpha engine (OBI, VPIN, CVD, Kyle-λ, …)

All hot paths are O(log n) via sortedcontainers.SortedDict (falls back to plain
dict + sort if the package is absent).
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SortedDict import with graceful fallback
# ---------------------------------------------------------------------------
try:
    from sortedcontainers import SortedDict as _SortedDict  # type: ignore

    _HAS_SORTED = True
except ImportError:  # pragma: no cover
    _HAS_SORTED = False
    _SortedDict = None  # type: ignore


def _make_bid_book() -> Any:
    """Return a SortedDict keyed by *negative* price so highest bid is first."""
    if _HAS_SORTED:
        return _SortedDict()
    return {}


def _make_ask_book() -> Any:
    if _HAS_SORTED:
        return _SortedDict()
    return {}


# ---------------------------------------------------------------------------
# PriceLevel
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PriceLevel:
    price: float
    size: float
    order_count: int = 1

    def __repr__(self) -> str:  # noqa: D105
        return f"PriceLevel(price={self.price:.6g}, size={self.size:.6g}, n={self.order_count})"


# ---------------------------------------------------------------------------
# L2OrderBook
# ---------------------------------------------------------------------------


class L2OrderBook:
    """
    Aggregated limit order book (Level-2).

    Bids are stored with negated keys so index-0 is always best bid.
    Asks are stored with positive keys so index-0 is always best ask.
    """

    __slots__ = ("symbol", "_bids", "_asks", "_prev_depth_vol")

    def __init__(self, symbol: str = "") -> None:
        self.symbol: str = symbol
        # bids: key = -price  → first key = cheapest neg = highest price
        self._bids: Any = _make_bid_book()
        # asks: key = +price  → first key = lowest ask
        self._asks: Any = _make_ask_book()
        self._prev_depth_vol: Optional[float] = None  # for sweep detection

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update(self, side: str, price: float, size: float, order_count: int = 1) -> None:
        """
        Add, update, or remove a price level.

        size == 0  → remove the level.
        side       → 'bid' / 'b' or 'ask' / 'a' (case-insensitive).
        O(log n) with SortedDict, O(n log n) with plain dict fallback.
        """
        side = side.lower()
        is_bid = side.startswith("b")
        book = self._bids if is_bid else self._asks
        key = -price if is_bid else price

        if size <= 0.0:
            book.pop(key, None)
        else:
            existing = book.get(key)
            if existing is not None:
                existing.size = size
                existing.order_count = order_count
            else:
                book[key] = PriceLevel(price=price, size=size, order_count=order_count)

    def clear(self) -> None:
        self._bids.clear()
        self._asks.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_bid(self) -> Optional[PriceLevel]:
        if not self._bids:
            return None
        key = next(iter(self._bids)) if _HAS_SORTED else min(self._bids)
        return self._bids[key]

    @property
    def best_ask(self) -> Optional[PriceLevel]:
        if not self._asks:
            return None
        key = next(iter(self._asks)) if _HAS_SORTED else min(self._asks)
        return self._asks[key]

    @property
    def mid_price(self) -> Optional[float]:
        bb = self.best_bid
        ba = self.best_ask
        if bb is None or ba is None:
            return None
        return (bb.price + ba.price) / 2.0

    @property
    def spread_bps(self) -> float:
        bb = self.best_bid
        ba = self.best_ask
        if bb is None or ba is None:
            return 0.0
        mid = (bb.price + ba.price) / 2.0
        if mid <= 0.0:
            return 0.0
        return (ba.price - bb.price) / mid * 10_000.0

    # ------------------------------------------------------------------
    # Depth / aggregation helpers
    # ------------------------------------------------------------------

    def _sorted_bid_levels(self) -> List[PriceLevel]:
        if _HAS_SORTED:
            return list(self._bids.values())
        return [self._bids[k] for k in sorted(self._bids)]  # keys are -price

    def _sorted_ask_levels(self) -> List[PriceLevel]:
        if _HAS_SORTED:
            return list(self._asks.values())
        return [self._asks[k] for k in sorted(self._asks)]

    def depth(self, n: int = 10) -> Dict[str, List[Tuple[float, float]]]:
        """Return top-n bid/ask levels as {bids: [(price, size)], asks: [(price, size)]}."""
        bids = [(lvl.price, lvl.size) for lvl in self._sorted_bid_levels()[:n]]
        asks = [(lvl.price, lvl.size) for lvl in self._sorted_ask_levels()[:n]]
        return {"bids": bids, "asks": asks}

    def imbalance(self, levels: int = 5) -> float:
        """
        Order Book Imbalance (OBI) over top-N levels.

        Returns (bid_vol - ask_vol) / (bid_vol + ask_vol) ∈ [-1, 1].
        """
        bid_vol = sum(lvl.size for lvl in self._sorted_bid_levels()[:levels])
        ask_vol = sum(lvl.size for lvl in self._sorted_ask_levels()[:levels])
        total = bid_vol + ask_vol
        if total <= 0.0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def weighted_mid(self, levels: int = 5) -> Optional[float]:
        """
        Volume-weighted mid price.

        Better than arithmetic mid because it weights the mid towards the
        side with more volume.  Equivalent to the microprice formula at depth-1.
        """
        bid_levels = self._sorted_bid_levels()[:levels]
        ask_levels = self._sorted_ask_levels()[:levels]
        bid_vol = sum(lvl.size for lvl in bid_levels)
        ask_vol = sum(lvl.size for lvl in ask_levels)
        total = bid_vol + ask_vol
        if total <= 0.0:
            return self.mid_price
        bid_vwap = (
            sum(lvl.price * lvl.size for lvl in bid_levels) / bid_vol if bid_vol > 0 else 0.0
        )
        ask_vwap = (
            sum(lvl.price * lvl.size for lvl in ask_levels) / ask_vol if ask_vol > 0 else 0.0
        )
        return (bid_vwap * bid_vol + ask_vwap * ask_vol) / total

    def microprice(self, levels: int = 5) -> Optional[float]:
        """
        Glosten–Milgrom microprice.

        microprice = mid + imbalance * half_spread

        Encodes the direction in which the fair value is likely to move given
        current order book pressure.
        """
        mid = self.mid_price
        if mid is None:
            return None
        bb = self.best_bid
        ba = self.best_ask
        if bb is None or ba is None:
            return mid
        half_spread = (ba.price - bb.price) / 2.0
        obi = self.imbalance(levels=levels)
        return mid + obi * half_spread

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of the current book state."""
        d = self.depth(10)
        mid = self.mid_price
        return {
            "symbol": self.symbol,
            "ts": time.time(),
            "best_bid": self.best_bid.price if self.best_bid else None,
            "best_ask": self.best_ask.price if self.best_ask else None,
            "mid_price": mid,
            "spread_bps": self.spread_bps,
            "imbalance": self.imbalance(),
            "microprice": self.microprice(),
            "weighted_mid": self.weighted_mid(),
            "bids": d["bids"],
            "asks": d["asks"],
        }

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"L2OrderBook(symbol={self.symbol!r}, "
            f"bid={self.best_bid}, ask={self.best_ask})"
        )


# ---------------------------------------------------------------------------
# L3OrderBook
# ---------------------------------------------------------------------------


@dataclass
class _Order:
    order_id: str
    side: str  # 'bid' | 'ask'
    price: float
    size: float
    ts: float = field(default_factory=time.time)


class L3OrderBook:
    """
    Individual-order book (Level-3).

    Maintains a per-price queue of order IDs in arrival order, enabling
    queue-position queries.
    """

    def __init__(self, symbol: str = "") -> None:
        self.symbol = symbol
        self._orders: Dict[str, _Order] = {}
        # side → price → list[order_id] (arrival order)
        self._queues: Dict[str, Dict[float, List[str]]] = {"bid": {}, "ask": {}}

    def add_order(self, order_id: str, side: str, price: float, size: float) -> None:
        side = "bid" if side.lower().startswith("b") else "ask"
        if order_id in self._orders:
            # Re-add = cancel + re-add (worst case queue position)
            self.cancel_order(order_id)
        order = _Order(order_id=order_id, side=side, price=price, size=size)
        self._orders[order_id] = order
        q = self._queues[side].setdefault(price, [])
        q.append(order_id)

    def cancel_order(self, order_id: str) -> Optional[_Order]:
        order = self._orders.pop(order_id, None)
        if order is None:
            return None
        q = self._queues[order.side].get(order.price, [])
        try:
            q.remove(order_id)
        except ValueError:
            pass
        if not q:
            self._queues[order.side].pop(order.price, None)
        return order

    def fill_order(self, order_id: str, fill_size: float) -> Optional[_Order]:
        """Reduce order size by fill_size; cancel entirely if fully filled."""
        order = self._orders.get(order_id)
        if order is None:
            return None
        order.size = max(0.0, order.size - fill_size)
        if order.size <= 0.0:
            self.cancel_order(order_id)
        return order

    def queue_position(self, order_id: str) -> int:
        """
        Number of orders *ahead* in the queue at the same price level.

        Returns -1 if the order is not found.
        """
        order = self._orders.get(order_id)
        if order is None:
            return -1
        q = self._queues[order.side].get(order.price, [])
        try:
            pos = q.index(order_id)
        except ValueError:
            return -1
        return pos  # 0 = first in queue (no orders ahead)

    def orders_ahead_size(self, order_id: str) -> float:
        """Total size of all orders ahead in the queue."""
        order = self._orders.get(order_id)
        if order is None:
            return 0.0
        q = self._queues[order.side].get(order.price, [])
        try:
            pos = q.index(order_id)
        except ValueError:
            return 0.0
        ahead = q[:pos]
        return sum(self._orders[oid].size for oid in ahead if oid in self._orders)

    def __len__(self) -> int:
        return len(self._orders)


# ---------------------------------------------------------------------------
# OrderBookSignals
# ---------------------------------------------------------------------------

_EPS = 1e-9


class OrderBookSignals:
    """
    Rolling-window alpha signal engine for a single symbol.

    Call ``update(book)`` on every book update tick, then query individual
    signals or ``signal_summary()`` for the full picture.
    """

    def __init__(self, symbol: str, window: int = 100) -> None:
        self.symbol = symbol
        self.window = max(10, int(window))

        # Rolling snapshot history
        self._snapshots: Deque[Dict[str, Any]] = deque(maxlen=self.window)

        # OBI history for z-score
        self._obi_hist: Deque[float] = deque(maxlen=self.window)

        # Spread history (bps) for percentile
        self._spread_hist: Deque[float] = deque(maxlen=self.window)

        # Trade stream (each entry: {size, side, price, ts})
        self._trades: Deque[Dict[str, Any]] = deque(maxlen=5000)

        # VPIN bucket accumulation
        self._vpin_bucket_buy: float = 0.0
        self._vpin_bucket_sell: float = 0.0
        self._vpin_bucket_size: float = 50.0
        self._vpin_buckets: Deque[float] = deque(maxlen=50)  # |buy_pct - 0.5| * 2

        # Last book reference for sweep detection
        self._prev_depth_vol: Optional[float] = None
        self._last_book: Optional[L2OrderBook] = None

    # ------------------------------------------------------------------
    # Feed
    # ------------------------------------------------------------------

    def update(self, book: L2OrderBook) -> None:
        """Ingest a new L2 book state."""
        snap = book.snapshot()
        self._snapshots.append(snap)
        self._obi_hist.append(snap["imbalance"])
        sp = snap["spread_bps"]
        if sp > 0:
            self._spread_hist.append(sp)
        self._last_book = book

    def add_trade(
        self, price: float, size: float, side: str, ts: Optional[float] = None
    ) -> None:
        """
        Register an aggressor trade.

        side: 'buy' if buyer is aggressor, 'sell' otherwise.
        """
        trade = {
            "price": price,
            "size": size,
            "side": side.lower(),
            "ts": ts if ts is not None else time.time(),
        }
        self._trades.append(trade)
        self._accumulate_vpin_bucket(trade)

    def _accumulate_vpin_bucket(self, trade: Dict[str, Any]) -> None:
        size = trade["size"]
        if trade["side"] == "buy":
            self._vpin_bucket_buy += size
        else:
            self._vpin_bucket_sell += size

        total = self._vpin_bucket_buy + self._vpin_bucket_sell
        if total >= self._vpin_bucket_size:
            buy_pct = self._vpin_bucket_buy / total
            self._vpin_buckets.append(abs(buy_pct - 0.5) * 2.0)
            # reset
            self._vpin_bucket_buy = 0.0
            self._vpin_bucket_sell = 0.0

    # ------------------------------------------------------------------
    # OBI signal (z-score normalised)
    # ------------------------------------------------------------------

    def obi_signal(self) -> float:
        """
        Current OBI z-score over the rolling window.

        Positive = bid-heavy; negative = ask-heavy.
        Returns 0.0 if insufficient history.
        """
        if len(self._obi_hist) < 2:
            return 0.0
        vals = list(self._obi_hist)
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        sigma = math.sqrt(var) if var > 0 else _EPS
        current = vals[-1]
        return (current - mu) / sigma

    # ------------------------------------------------------------------
    # VPIN
    # ------------------------------------------------------------------

    def vpin(self, bucket_size: float = 50.0, n_buckets: int = 10) -> float:
        """
        Volume-synchronised Probability of Informed Trading.

        Accumulates trades into volume buckets of `bucket_size`.
        VPIN = rolling mean of |buy_pct − 0.5| × 2 over last `n_buckets`
        complete buckets.

        Returns float ∈ [0, 1].  0.5 baseline ≈ uninformed; >0.65 ≈ informed.
        """
        if self._vpin_bucket_size != bucket_size:
            self._vpin_bucket_size = bucket_size
        buckets = list(self._vpin_buckets)
        if not buckets:
            return 0.0
        tail = buckets[-n_buckets:]
        return sum(tail) / len(tail)

    # ------------------------------------------------------------------
    # Trade arrival rate
    # ------------------------------------------------------------------

    def trade_arrival_rate(self, window_seconds: float = 60.0) -> float:
        """Trades per second over the last `window_seconds`."""
        if not self._trades:
            return 0.0
        now = time.time()
        cutoff = now - window_seconds
        count = sum(1 for t in self._trades if t["ts"] >= cutoff)
        return count / window_seconds

    # ------------------------------------------------------------------
    # Spread percentile
    # ------------------------------------------------------------------

    def spread_percentile(self) -> float:
        """
        Where the current spread sits vs the last 100 spread observations.

        0.0 = tightest ever seen; 1.0 = widest ever seen.

        Uses a midpoint rank formula so that a perfectly stable spread
        (all equal) returns 0.5 rather than 1.0, matching intuition that
        a constant spread is neither tight nor wide relative to itself.
        """
        if len(self._spread_hist) < 2:
            return 0.5
        hist = list(self._spread_hist)
        current = hist[-1]
        n = len(hist)
        # Count strictly-below and equal observations (excluding current tick)
        below = sum(1 for v in hist[:-1] if v < current)
        equal = sum(1 for v in hist[:-1] if v == current)
        # Midpoint rank: (below + 0.5 * equal) / (n - 1), clamped to [0, 1]
        if n <= 1:
            return 0.5
        rank = (below + 0.5 * equal) / (n - 1)
        return max(0.0, min(1.0, rank))

    # ------------------------------------------------------------------
    # Liquidity sweep detection
    # ------------------------------------------------------------------

    def liquidity_sweep_detected(self, threshold_pct: float = 0.05) -> bool:
        """
        True if top-of-book liquidity dropped by more than `threshold_pct`
        of total depth in one tick.

        Uses consecutive snapshot pairs.
        """
        snaps = list(self._snapshots)
        if len(snaps) < 2:
            return False
        prev = snaps[-2]
        curr = snaps[-1]

        def _top_vol(snap: Dict) -> float:
            bids = snap.get("bids") or []
            asks = snap.get("asks") or []
            vol = 0.0
            if bids:
                vol += bids[0][1]
            if asks:
                vol += asks[0][1]
            return vol

        prev_vol = _top_vol(prev)
        curr_vol = _top_vol(curr)
        if prev_vol <= 0:
            return False
        drop = (prev_vol - curr_vol) / prev_vol
        return drop > threshold_pct

    # ------------------------------------------------------------------
    # CVD
    # ------------------------------------------------------------------

    def cvd(self, lookback: int = 50) -> float:
        """
        Cumulative Volume Delta over the last `lookback` trades.

        CVD = Σ (buy_vol − sell_vol).
        Positive → net buying pressure; negative → net selling.
        """
        tail = list(self._trades)[-lookback:]
        delta = 0.0
        for t in tail:
            if t["side"] == "buy":
                delta += t["size"]
            else:
                delta -= t["size"]
        return delta

    # ------------------------------------------------------------------
    # Book pressure (proximity-weighted OBI)
    # ------------------------------------------------------------------

    def book_pressure(self, levels: int = 5) -> float:
        """
        Weighted OBI giving exponentially more weight to levels closer to mid.

        Returns float ∈ [-1, 1].
        """
        if self._last_book is None:
            return 0.0
        book = self._last_book
        bid_levels = book._sorted_bid_levels()[:levels]
        ask_levels = book._sorted_ask_levels()[:levels]

        mid = book.mid_price
        if mid is None or mid <= 0:
            return book.imbalance(levels)

        def weighted_vol(lvls: List[PriceLevel]) -> float:
            total = 0.0
            for i, lvl in enumerate(lvls):
                dist = abs(mid - lvl.price) / mid + _EPS
                weight = math.exp(-i)  # exponential decay by rank
                total += lvl.size * weight / dist
            return total

        bw = weighted_vol(bid_levels)
        aw = weighted_vol(ask_levels)
        denom = bw + aw
        if denom <= 0:
            return 0.0
        return (bw - aw) / denom

    # ------------------------------------------------------------------
    # Kyle's Lambda (simplified)
    # ------------------------------------------------------------------

    def kyle_lambda(self, lookback: int = 50) -> float:
        """
        Simplified Kyle's λ — price impact per unit of signed volume.

        λ = Cov(ΔP, ΔQ) / Var(ΔQ)

        where ΔP = mid-price change, ΔQ = signed order flow (buy − sell).
        Returns 0.0 if insufficient data.
        """
        snaps = list(self._snapshots)
        trades = list(self._trades)
        n = min(lookback, len(snaps) - 1, len(trades))
        if n < 5:
            return 0.0

        # Build paired (Δmid, signed_vol) per snapshot interval
        delta_p: List[float] = []
        delta_q: List[float] = []

        for i in range(1, n + 1):
            prev_mid = snaps[-i - 1].get("mid_price") or 0.0
            curr_mid = snaps[-i].get("mid_price") or 0.0
            if prev_mid <= 0 or curr_mid <= 0:
                continue
            dp = curr_mid - prev_mid
            # Approximate ΔQ from CVD over the last few trades — use trade size
            t = trades[-i]
            dq = t["size"] if t["side"] == "buy" else -t["size"]
            delta_p.append(dp)
            delta_q.append(dq)

        if len(delta_p) < 5:
            return 0.0

        n2 = len(delta_p)
        mean_p = sum(delta_p) / n2
        mean_q = sum(delta_q) / n2
        cov = sum((delta_p[i] - mean_p) * (delta_q[i] - mean_q) for i in range(n2)) / n2
        var_q = sum((v - mean_q) ** 2 for v in delta_q) / n2
        if var_q <= 0:
            return 0.0
        return cov / var_q

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def signal_summary(self) -> Dict[str, Any]:
        """Return all signals as a single dict with timestamp and symbol."""
        return {
            "symbol": self.symbol,
            "ts": time.time(),
            "obi_z": self.obi_signal(),
            "obi_raw": self._obi_hist[-1] if self._obi_hist else 0.0,
            "vpin": self.vpin(),
            "trade_arrival_rate": self.trade_arrival_rate(),
            "spread_percentile": self.spread_percentile(),
            "liquidity_sweep": self.liquidity_sweep_detected(),
            "cvd": self.cvd(),
            "book_pressure": self.book_pressure(),
            "kyle_lambda": self.kyle_lambda(),
            "spread_bps": self._snapshots[-1]["spread_bps"] if self._snapshots else None,
            "microprice": self._snapshots[-1]["microprice"] if self._snapshots else None,
            "mid_price": self._snapshots[-1]["mid_price"] if self._snapshots else None,
        }


# ---------------------------------------------------------------------------
# Helper: build L2OrderBook from a dict payload
# ---------------------------------------------------------------------------


def _book_from_dict(symbol: str, ob_dict: Dict[str, Any]) -> L2OrderBook:
    """
    Build (or rebuild) an L2OrderBook from a snapshot dict.

    Accepted formats:
    - {"bids": [[price, size], ...], "asks": [[price, size], ...]}
    - {"bids": [{"price": p, "size": s}, ...], ...}
    """
    book = L2OrderBook(symbol=symbol)
    for side, key in [("bid", "bids"), ("ask", "asks")]:
        for entry in ob_dict.get(key) or []:
            if isinstance(entry, (list, tuple)):
                price, size = float(entry[0]), float(entry[1])
                oc = int(entry[2]) if len(entry) > 2 else 1
            elif isinstance(entry, dict):
                price = float(entry.get("price", entry.get("p", 0)))
                size = float(entry.get("size", entry.get("s", 0)))
                oc = int(entry.get("order_count", 1))
            else:
                continue
            if price > 0 and size > 0:
                book.update(side, price, size, oc)
    return book
