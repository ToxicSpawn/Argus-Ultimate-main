"""
HFT Scalping Engine — order book and trade flow signals.

Replaces the original stub.  Uses L2OrderBook + OrderBookSignals from
order_book_processor for production-grade OBI, VPIN, CVD, and microprice
signals.

Signal thresholds (tunable via config):
    OBI z-score  > 1.5 AND spread_percentile < 0.3  →  order book signal
    VPIN         > 0.65                              →  informed-flow signal
    |CVD trend|  > cvd_threshold                    →  momentum signal
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from hft_engine.order_book_processor import (
    L2OrderBook,
    OrderBookSignals,
    _book_from_dict,
)

logger = logging.getLogger(__name__)

_DEFAULT_OBI_Z_THRESHOLD: float = 1.5
_DEFAULT_SPREAD_PCT_THRESHOLD: float = 0.3
_DEFAULT_VPIN_THRESHOLD: float = 0.65
_DEFAULT_CVD_THRESHOLD: float = 1000.0  # notional units


class HFTScalpingEngine:
    """
    HFT signals from order book (OBI) and trade flow.

    Parameters
    ----------
    config : any
        Configuration object or dict.  Recognises keys:
        ``obi_z_threshold``, ``spread_pct_threshold``,
        ``vpin_threshold``, ``cvd_threshold``.
    exchanges : dict, optional
        Exchange connectors (unused internally, forwarded to callers).
    """

    def __init__(self, config: Any, *, exchanges: Any = None) -> None:
        self.config = config or {}
        self.exchanges = exchanges or {}

        cfg = config if isinstance(config, dict) else {}
        self._obi_z_thresh: float = float(
            cfg.get("obi_z_threshold", _DEFAULT_OBI_Z_THRESHOLD)
        )
        self._spread_pct_thresh: float = float(
            cfg.get("spread_pct_threshold", _DEFAULT_SPREAD_PCT_THRESHOLD)
        )
        self._vpin_thresh: float = float(
            cfg.get("vpin_threshold", _DEFAULT_VPIN_THRESHOLD)
        )
        self._cvd_thresh: float = float(
            cfg.get("cvd_threshold", _DEFAULT_CVD_THRESHOLD)
        )

        # Per-symbol state
        self._books: Dict[str, L2OrderBook] = {}
        self._signals_engines: Dict[str, OrderBookSignals] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_engine(self, symbol: str) -> OrderBookSignals:
        if symbol not in self._signals_engines:
            self._signals_engines[symbol] = OrderBookSignals(symbol)
        return self._signals_engines[symbol]

    def _get_book(self, symbol: str) -> L2OrderBook:
        if symbol not in self._books:
            self._books[symbol] = L2OrderBook(symbol=symbol)
        return self._books[symbol]

    def _ingest_book_dict(self, symbol: str, order_book_dict: Dict[str, Any]) -> L2OrderBook:
        """
        Rebuild the book from a dict snapshot (stateless feed) OR apply
        incremental updates if the dict contains an 'updates' key.
        """
        book = _book_from_dict(symbol, order_book_dict)
        self._books[symbol] = book
        return book

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze_order_book(
        self, symbol: str, order_book: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Produce a signal from the order book snapshot.

        Returns a signal dict when:
            |OBI z-score| > obi_z_threshold  AND
            spread_percentile < spread_pct_threshold

        Signal keys:
            symbol, direction, obi, microprice, spread_bps,
            confidence, timestamp
        """
        if not order_book:
            return None

        try:
            book = self._ingest_book_dict(symbol, order_book)
            engine = self._get_engine(symbol)
            engine.update(book)

            obi_z = engine.obi_signal()
            spread_pct = engine.spread_percentile()
            snap = engine._snapshots[-1] if engine._snapshots else {}

            # Gate: strong imbalance in a tight book
            if abs(obi_z) <= self._obi_z_thresh or spread_pct >= self._spread_pct_thresh:
                return None

            direction = "long" if obi_z > 0 else "short"
            confidence = min(1.0, abs(obi_z) / (self._obi_z_thresh * 2.0))
            # Tighter spread → higher confidence
            confidence *= 1.0 - spread_pct

            return {
                "symbol": symbol,
                "direction": direction,
                "obi": snap.get("imbalance", 0.0),
                "obi_z": obi_z,
                "microprice": snap.get("microprice"),
                "mid_price": snap.get("mid_price"),
                "spread_bps": snap.get("spread_bps", 0.0),
                "spread_percentile": spread_pct,
                "confidence": round(confidence, 6),
                "source": "order_book",
                "timestamp": time.time(),
            }
        except Exception as exc:
            logger.warning("analyze_order_book(%s): %s", symbol, exc, exc_info=True)
            return None

    async def analyze_trade_flow(
        self, symbol: str, recent_trades: List[Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Produce a signal from recent trade flow.

        Returns a signal dict when:
            VPIN > vpin_threshold  (informed flow detected), OR
            |CVD| > cvd_threshold  (strong directional momentum)

        Each trade entry may be:
            [price, size, side]  or  {"price": p, "size": s, "side": s}
            side: 'buy' | 'sell'
        """
        if not recent_trades:
            return None

        try:
            engine = self._get_engine(symbol)

            # Ingest trades
            for entry in recent_trades:
                if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                    price, size, side = float(entry[0]), float(entry[1]), str(entry[2])
                    ts = float(entry[3]) if len(entry) > 3 else None
                elif isinstance(entry, dict):
                    price = float(entry.get("price", entry.get("p", 0)))
                    size = float(entry.get("size", entry.get("s", entry.get("qty", 0))))
                    side = str(entry.get("side", entry.get("direction", "buy")))
                    ts = entry.get("ts") or entry.get("timestamp")
                    ts = float(ts) if ts is not None else None
                else:
                    continue
                engine.add_trade(price=price, size=size, side=side, ts=ts)

            vpin_val = engine.vpin()
            cvd_val = engine.cvd()
            summary = engine.signal_summary()

            vpin_triggered = vpin_val > self._vpin_thresh
            cvd_triggered = abs(cvd_val) > self._cvd_thresh

            if not (vpin_triggered or cvd_triggered):
                return None

            direction: str
            if cvd_triggered:
                direction = "long" if cvd_val > 0 else "short"
            else:
                # VPIN alone → use book pressure for direction if available
                bp = summary.get("book_pressure", 0.0)
                direction = "long" if bp >= 0 else "short"

            confidence = 0.0
            if vpin_triggered:
                confidence = (vpin_val - self._vpin_thresh) / (1.0 - self._vpin_thresh)
            if cvd_triggered:
                confidence = max(confidence, min(1.0, abs(cvd_val) / (self._cvd_thresh * 2.0)))

            return {
                "symbol": symbol,
                "direction": direction,
                "vpin": vpin_val,
                "cvd": cvd_val,
                "trade_arrival_rate": summary.get("trade_arrival_rate", 0.0),
                "book_pressure": summary.get("book_pressure", 0.0),
                "confidence": round(min(1.0, confidence), 6),
                "source": "trade_flow",
                "timestamp": time.time(),
            }
        except Exception as exc:
            logger.warning("analyze_trade_flow(%s): %s", symbol, exc, exc_info=True)
            return None

    async def scan_for_opportunities(
        self, request: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Scan multiple symbols for HFT/scalping opportunities.

        Expected request format::

            {
                "symbols": ["BTC-USD", "ETH-USD"],
                "order_books": {
                    "BTC-USD": {"bids": [...], "asks": [...]},
                    ...
                },
                "recent_trades": {
                    "BTC-USD": [{"price": ..., "size": ..., "side": ...}, ...],
                    ...
                }
            }

        Returns a list of opportunity dicts sorted by confidence (descending).
        Each opportunity has an ``expected_profit_pct`` field derived from
        spread, confidence, and microprice displacement.
        """
        opportunities: List[Dict[str, Any]] = []

        symbols = request.get("symbols") or list(
            set(list(request.get("order_books", {}).keys()))
            | set(list(request.get("recent_trades", {}).keys()))
        )

        for symbol in symbols:
            ob = (request.get("order_books") or {}).get(symbol)
            trades = (request.get("recent_trades") or {}).get(symbol) or []

            ob_signal: Optional[Dict[str, Any]] = None
            tf_signal: Optional[Dict[str, Any]] = None

            if ob:
                ob_signal = await self.analyze_order_book(symbol, ob)
            if trades:
                tf_signal = await self.analyze_trade_flow(symbol, trades)

            for sig in [ob_signal, tf_signal]:
                if sig is None:
                    continue
                spread_bps = sig.get("spread_bps") or 0.0
                confidence = sig.get("confidence", 0.0)
                # Rough expected profit: half the microprice displacement
                # minus a round-trip cost of ~0.5 * spread
                mp = sig.get("microprice") or 0.0
                mid = sig.get("mid_price") or 0.0
                displacement_bps = abs(mp - mid) / max(mid, 1e-9) * 10_000.0 if mid > 0 else 0.0
                expected_profit_bps = max(0.0, displacement_bps - 0.5 * spread_bps)
                expected_profit_pct = expected_profit_bps / 10_000.0

                opportunities.append(
                    {
                        **sig,
                        "expected_profit_pct": round(expected_profit_pct, 8),
                    }
                )

        # Sort by confidence descending
        opportunities.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        return opportunities
