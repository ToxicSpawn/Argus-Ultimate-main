"""
core/feeds/lob_feed.py
======================
Limit Order Book (LOB) feed — market-by-order (MBO) depth data.

Extends ws_feed_base.py to stream full order book snapshots and deltas
from Kraken (default/AU), Bybit, and OKX. Produces LOBSnapshot objects
consumed by:
  - JaxRLEnvironment (training)
  - feature_store.py  (LOB-aware features)
  - causal_gnn.py     (graph edges from order flow)

Default exchange is Kraken — the only major exchange with full
Australian retail API access following Binance AU's exit in 2023.

Schema
------
LOBSnapshot.bids / .asks : list of (price, qty) tuples, depth-sorted.
LOBDelta               : incremental update (price, qty, side, type).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("argus.core.feeds.lob_feed")

LOB_DEPTH = 20          # levels to maintain per side
SNAPSHOT_BUFFER = 2000  # rolling history for JAX array export


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class LOBSnapshot:
    """Full LOB state at a single point in time."""
    symbol: str
    exchange: str
    ts_ns: int                              # nanosecond timestamp
    bids: List[Tuple[float, float]]         # [(price, qty), ...] best-first
    asks: List[Tuple[float, float]]
    sequence: int = 0

    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.bids[0][0] + self.asks[0][0]) / 2
        return 0.0

    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return 0.0

    @property
    def bid_depth(self) -> float:
        return sum(q for _, q in self.bids)

    @property
    def ask_depth(self) -> float:
        return sum(q for _, q in self.asks)

    @property
    def order_imbalance(self) -> float:
        """(bid_depth - ask_depth) / (bid_depth + ask_depth)"""
        b, a = self.bid_depth, self.ask_depth
        total = b + a
        return (b - a) / total if total > 0 else 0.0

    def to_feature_vector(self) -> List[float]:
        """Flatten to fixed-length feature vector for ML models."""
        features = []
        for i in range(LOB_DEPTH):
            bp = self.bids[i][0] if i < len(self.bids) else 0.0
            bq = self.bids[i][1] if i < len(self.bids) else 0.0
            ap = self.asks[i][0] if i < len(self.asks) else 0.0
            aq = self.asks[i][1] if i < len(self.asks) else 0.0
            features.extend([bp, bq, ap, aq])
        features.extend([self.mid_price, self.spread,
                         self.order_imbalance, float(self.ts_ns)])
        return features


@dataclass
class LOBDelta:
    """Incremental LOB update (market-by-order event)."""
    symbol: str
    exchange: str
    ts_ns: int
    side: str           # "bid" | "ask"
    price: float
    qty: float          # 0.0 means level removed
    delta_type: str     # "insert" | "update" | "delete"
    sequence: int = 0


# ---------------------------------------------------------------------------
# LOB state manager (maintains sorted book)
# ---------------------------------------------------------------------------

class LOBBook:
    """Maintains a live sorted order book for one symbol."""

    def __init__(self, symbol: str, exchange: str, depth: int = LOB_DEPTH) -> None:
        self.symbol = symbol
        self.exchange = exchange
        self.depth = depth
        self._bids: Dict[float, float] = {}   # price -> qty
        self._asks: Dict[float, float] = {}
        self._seq: int = 0
        self._history: Deque[LOBSnapshot] = deque(maxlen=SNAPSHOT_BUFFER)

    def apply_snapshot(self, bids: List[Tuple[float, float]],
                       asks: List[Tuple[float, float]], seq: int = 0) -> LOBSnapshot:
        """Replace full book from a REST snapshot."""
        self._bids = {p: q for p, q in bids}
        self._asks = {p: q for p, q in asks}
        self._seq = seq
        return self._emit()

    def apply_delta(self, delta: LOBDelta) -> LOBSnapshot:
        """Apply a single incremental update and return new snapshot."""
        book = self._bids if delta.side == "bid" else self._asks
        if delta.qty == 0.0 or delta.delta_type == "delete":
            book.pop(delta.price, None)
        else:
            book[delta.price] = delta.qty
        self._seq = delta.sequence
        return self._emit()

    def snapshot(self) -> LOBSnapshot:
        return self._emit()

    @property
    def history(self) -> Deque[LOBSnapshot]:
        return self._history

    def to_jax_array(self) -> Any:
        """
        Export rolling history as a numpy array of shape
        (T, depth, 4) — [bid_px, bid_qty, ask_px, ask_qty].
        Used directly by JaxRLEnvironment.
        """
        import numpy as np
        out = []
        for snap in self._history:
            row = []
            for i in range(self.depth):
                bp = snap.bids[i][0] if i < len(snap.bids) else 0.0
                bq = snap.bids[i][1] if i < len(snap.bids) else 0.0
                ap = snap.asks[i][0] if i < len(snap.asks) else 0.0
                aq = snap.asks[i][1] if i < len(snap.asks) else 0.0
                row.append([bp, bq, ap, aq])
            out.append(row)
        return np.array(out, dtype=np.float32)

    def _emit(self) -> LOBSnapshot:
        bids = sorted(self._bids.items(), reverse=True)[:self.depth]
        asks = sorted(self._asks.items())[:self.depth]
        snap = LOBSnapshot(
            symbol=self.symbol,
            exchange=self.exchange,
            ts_ns=time.time_ns(),
            bids=bids,
            asks=asks,
            sequence=self._seq,
        )
        self._history.append(snap)
        return snap


# ---------------------------------------------------------------------------
# WebSocket LOB feed (exchange-agnostic base)
# ---------------------------------------------------------------------------

class LOBFeed:
    """
    Async WebSocket LOB feed.

    Subclass and implement _parse_message() for each exchange.
    KrakenLOBFeed (AU default), BybitLOBFeed, OKXLOBFeed follow below.
    """

    WS_URL: str = ""

    def __init__(self, symbol: str, depth: int = LOB_DEPTH,
                 on_snapshot: Optional[Any] = None) -> None:
        self.symbol = symbol.upper()
        self._book = LOBBook(self.symbol, self.__class__.__name__, depth)
        self._on_snapshot = on_snapshot   # callback(LOBSnapshot)
        self._running = False
        self._ws = None

    async def start(self) -> None:
        self._running = True
        logger.info("%s LOB feed starting for %s", self.__class__.__name__, self.symbol)
        await self._connect()

    async def stop(self) -> None:
        self._running = False
        if self._ws is not None:
            await self._ws.close()
        logger.info("%s LOB feed stopped for %s", self.__class__.__name__, self.symbol)

    @property
    def book(self) -> LOBBook:
        return self._book

    async def _connect(self) -> None:
        try:
            import websockets  # type: ignore
        except ImportError:
            logger.error("websockets not installed — pip install websockets")
            return

        url = self._ws_url()
        logger.info("Connecting LOB WS: %s", url)
        try:
            async with websockets.connect(url, ping_interval=20) as ws:
                self._ws = ws
                await self._on_open(ws)
                async for raw in ws:
                    if not self._running:
                        break
                    await self._handle(raw)
        except Exception as exc:
            logger.error("%s WS error: %s", self.__class__.__name__, exc)

    async def _on_open(self, ws: Any) -> None:
        pass

    async def _handle(self, raw: str) -> None:
        import json
        try:
            msg = json.loads(raw)
            snap = self._parse_message(msg)
            if snap and self._on_snapshot:
                await self._fire(snap)
        except Exception as exc:
            logger.debug("LOB parse error: %s", exc)

    async def _fire(self, snap: LOBSnapshot) -> None:
        try:
            if asyncio.iscoroutinefunction(self._on_snapshot):
                await self._on_snapshot(snap)
            else:
                self._on_snapshot(snap)
        except Exception:
            logger.exception("LOB snapshot callback raised")

    def _ws_url(self) -> str:
        return self.WS_URL

    def _parse_message(self, msg: dict) -> Optional[LOBSnapshot]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# KrakenLOBFeed  — AU default
# ---------------------------------------------------------------------------

class KrakenLOBFeed(LOBFeed):
    """
    Kraken WS v2 order book — ``book`` channel, depth 10.
    Primary LOB feed for Australian deployments.
    """

    def _ws_url(self) -> str:
        return "wss://ws.kraken.com/v2"

    async def _on_open(self, ws: Any) -> None:
        import json as _json
        from core.feeds.kraken_feed import _kraken_sym
        sym = _kraken_sym(self.symbol)
        sub = {
            "method": "subscribe",
            "params": {"channel": "book", "symbol": [sym], "depth": 10},
        }
        await ws.send(_json.dumps(sub))
        logger.info("KrakenLOBFeed: subscribed book for %s (%s)", self.symbol, sym)

    def _parse_message(self, msg: dict) -> Optional[LOBSnapshot]:
        if msg.get("channel") != "book":
            return None
        data_list = msg.get("data", [])
        if not data_list:
            return None
        data = data_list[0]
        msg_type = msg.get("type", "update")

        bids: List[Tuple[float, float]] = [
            (float(b["price"]), float(b["qty"])) for b in data.get("bids", [])
        ]
        asks: List[Tuple[float, float]] = [
            (float(a["price"]), float(a["qty"])) for a in data.get("asks", [])
        ]

        if msg_type == "snapshot":
            return self._book.apply_snapshot(bids, asks)

        ts = time.time_ns()
        for p, q in bids:
            self._book.apply_delta(
                LOBDelta(self.symbol, "Kraken", ts, "bid", p, q,
                         "delete" if q == 0.0 else "update")
            )
        for p, q in asks:
            self._book.apply_delta(
                LOBDelta(self.symbol, "Kraken", ts, "ask", p, q,
                         "delete" if q == 0.0 else "update")
            )
        return self._book.snapshot()


# ---------------------------------------------------------------------------
# BinanceLOBFeed  — kept for non-AU deployments / VPN use only
# ---------------------------------------------------------------------------

class BinanceLOBFeed(LOBFeed):
    """Binance diff depth stream.  NOT available to Australian retail."""

    def _ws_url(self) -> str:
        sym = self.symbol.lower().replace("/", "")
        return f"wss://stream.binance.com:9443/ws/{sym}@depth20@100ms"

    def _parse_message(self, msg: dict) -> Optional[LOBSnapshot]:
        if "bids" not in msg or "asks" not in msg:
            return None
        bids = [(float(p), float(q)) for p, q in msg["bids"]]
        asks = [(float(p), float(q)) for p, q in msg["asks"]]
        return self._book.apply_snapshot(bids, asks, seq=msg.get("lastUpdateId", 0))


class BybitLOBFeed(LOBFeed):
    """Bybit orderbook.20 stream."""

    def _ws_url(self) -> str:
        return "wss://stream.bybit.com/v5/public/linear"

    async def _on_open(self, ws: Any) -> None:
        import json as _json
        sym = self.symbol.replace("/", "")
        await ws.send(_json.dumps({
            "op": "subscribe",
            "args": [f"orderbook.20.{sym}"]
        }))

    def _parse_message(self, msg: dict) -> Optional[LOBSnapshot]:
        data = msg.get("data", {})
        if not data or "b" not in data:
            return None
        bids = [(float(p), float(q)) for p, q in data.get("b", [])]
        asks = [(float(p), float(q)) for p, q in data.get("a", [])]
        op = msg.get("type", "delta")
        if op == "snapshot":
            return self._book.apply_snapshot(bids, asks)
        for p, q in bids:
            self._book.apply_delta(LOBDelta(self.symbol, "Bybit",
                time.time_ns(), "bid", p, q, "update"))
        for p, q in asks:
            self._book.apply_delta(LOBDelta(self.symbol, "Bybit",
                time.time_ns(), "ask", p, q, "update"))
        return self._book.snapshot()


class OKXLOBFeed(LOBFeed):
    """OKX books channel."""

    def _ws_url(self) -> str:
        return "wss://ws.okx.com:8443/ws/v5/public"

    async def _on_open(self, ws: Any) -> None:
        import json as _json
        inst = self.symbol.replace("/", "-")
        await ws.send(_json.dumps({
            "op": "subscribe",
            "args": [{"channel": "books", "instId": inst}]
        }))

    def _parse_message(self, msg: dict) -> Optional[LOBSnapshot]:
        data = msg.get("data", [{}])
        if not data:
            return None
        d = data[0]
        bids = [(float(p), float(q)) for p, q, *_ in d.get("bids", [])]
        asks = [(float(p), float(q)) for p, q, *_ in d.get("asks", [])]
        action = msg.get("action", "update")
        if action == "snapshot":
            return self._book.apply_snapshot(bids, asks)
        for p, q in bids:
            self._book.apply_delta(LOBDelta(self.symbol, "OKX",
                time.time_ns(), "bid", p, q, "update"))
        for p, q in asks:
            self._book.apply_delta(LOBDelta(self.symbol, "OKX",
                time.time_ns(), "ask", p, q, "update"))
        return self._book.snapshot()
