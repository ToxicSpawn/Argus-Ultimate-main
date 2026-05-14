"""
feed_normaliser.py
------------------
Per-venue schema translation → canonical dataclasses.

All timestamps are normalised to UTC epoch seconds (float).
All prices and quantities are coerced to Python Decimal for precision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Canonical dataclasses
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CanonicalTick:
    venue: str
    symbol: str                      # normalised e.g. "BTC/USDT"
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume_24h: Decimal
    ts: float                        # UTC epoch seconds

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid


@dataclass(slots=True)
class CanonicalBook:
    venue: str
    symbol: str
    bids: List[Tuple[Decimal, Decimal]]   # [(price, qty), ...] best-first
    asks: List[Tuple[Decimal, Decimal]]
    ts: float

    @property
    def best_bid(self) -> Optional[Decimal]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        return self.asks[0][0] if self.asks else None


@dataclass(slots=True)
class CanonicalTrade:
    venue: str
    symbol: str
    price: Decimal
    qty: Decimal
    side: str                        # "buy" | "sell"
    trade_id: str
    ts: float


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------

def _d(v: Any) -> Decimal:
    """Safe Decimal coercion."""
    try:
        return Decimal(str(v))
    except (InvalidOperation, TypeError):
        return Decimal("0")


def _ts_ms(v: Any) -> float:
    """Convert millisecond epoch int/str to float seconds."""
    try:
        return float(v) / 1000.0
    except (TypeError, ValueError):
        return time.time()


def _ts_s(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return time.time()


class FeedNormaliser:
    """
    Translate venue-specific raw dicts to canonical dataclasses.

    Supported venues: bybit, binance, okx
    """

    # -----------------------------------------------------------------------
    # Bybit v5
    # -----------------------------------------------------------------------

    @staticmethod
    def bybit_ticker(raw: Dict[str, Any]) -> Optional[CanonicalTick]:
        """
        raw = data field from Bybit tickers.linear/spot topic message.
        Expected keys: symbol, bid1Price, ask1Price, lastPrice, volume24h, ts (ms).
        """
        try:
            d = raw.get("data", raw)
            sym = d.get("symbol", "")
            return CanonicalTick(
                venue="bybit",
                symbol=FeedNormaliser._bybit_symbol(sym),
                bid=_d(d.get("bid1Price", 0)),
                ask=_d(d.get("ask1Price", 0)),
                last=_d(d.get("lastPrice", 0)),
                volume_24h=_d(d.get("volume24h", 0)),
                ts=_ts_ms(raw.get("ts", d.get("ts", time.time() * 1000))),
            )
        except Exception:
            return None

    @staticmethod
    def bybit_book(raw: Dict[str, Any], symbol: str) -> Optional[CanonicalBook]:
        """
        raw = full orderbook.1 data dict from Bybit.
        Keys: b (bids list [[price,qty],...]), a (asks list), ts.
        """
        try:
            d = raw.get("data", raw)
            bids = [(Decimal(p), Decimal(q)) for p, q in (d.get("b") or [])]
            asks = [(Decimal(p), Decimal(q)) for p, q in (d.get("a") or [])]
            return CanonicalBook(
                venue="bybit",
                symbol=FeedNormaliser._bybit_symbol(symbol),
                bids=sorted(bids, key=lambda x: x[0], reverse=True),
                asks=sorted(asks, key=lambda x: x[0]),
                ts=_ts_ms(raw.get("ts", d.get("ts", time.time() * 1000))),
            )
        except Exception:
            return None

    @staticmethod
    def bybit_trade(raw: Dict[str, Any]) -> List[CanonicalTrade]:
        try:
            trades = []
            d_list = raw.get("data", [])
            for d in d_list:
                trades.append(CanonicalTrade(
                    venue="bybit",
                    symbol=FeedNormaliser._bybit_symbol(d.get("s", "")),
                    price=_d(d.get("p", 0)),
                    qty=_d(d.get("v", 0)),
                    side="buy" if d.get("S") == "Buy" else "sell",
                    trade_id=str(d.get("i", "")),
                    ts=_ts_ms(d.get("T", time.time() * 1000)),
                ))
            return trades
        except Exception:
            return []

    @staticmethod
    def _bybit_symbol(s: str) -> str:
        """BTCUSDT → BTC/USDT (best-effort for common suffixes)."""
        for quote in ("USDT", "USDC", "BTC", "ETH", "BNB"):
            if s.endswith(quote) and len(s) > len(quote):
                return f"{s[:-len(quote)]}/{quote}"
        return s

    # -----------------------------------------------------------------------
    # Binance
    # -----------------------------------------------------------------------

    @staticmethod
    def binance_ticker(raw: Dict[str, Any]) -> Optional[CanonicalTick]:
        """
        raw = individual stream event from bookTicker or 24h miniTicker.
        bookTicker keys: s, b (bestBidPrice), B (bestBidQty), a, A, T.
        """
        try:
            sym = raw.get("s", "")
            return CanonicalTick(
                venue="binance",
                symbol=FeedNormaliser._binance_symbol(sym),
                bid=_d(raw.get("b", raw.get("b", 0))),
                ask=_d(raw.get("a", 0)),
                last=_d(raw.get("c", raw.get("b", 0))),   # miniTicker uses c
                volume_24h=_d(raw.get("v", 0)),
                ts=_ts_ms(raw.get("T", raw.get("E", time.time() * 1000))),
            )
        except Exception:
            return None

    @staticmethod
    def binance_book(raw: Dict[str, Any], symbol: str) -> Optional[CanonicalBook]:
        """
        raw = depthUpdate event (full or incremental; caller manages book state).
        Keys: b (bids), a (asks), E (event time ms).
        """
        try:
            bids = [(Decimal(p), Decimal(q)) for p, q in (raw.get("b") or [])]
            asks = [(Decimal(p), Decimal(q)) for p, q in (raw.get("a") or [])]
            return CanonicalBook(
                venue="binance",
                symbol=FeedNormaliser._binance_symbol(symbol),
                bids=sorted(bids, key=lambda x: x[0], reverse=True),
                asks=sorted(asks, key=lambda x: x[0]),
                ts=_ts_ms(raw.get("E", time.time() * 1000)),
            )
        except Exception:
            return None

    @staticmethod
    def binance_trade(raw: Dict[str, Any]) -> Optional[CanonicalTrade]:
        """
        raw = aggTrade event. Keys: s, p, q, m (isBuyerMaker), T, a (aggId).
        """
        try:
            return CanonicalTrade(
                venue="binance",
                symbol=FeedNormaliser._binance_symbol(raw.get("s", "")),
                price=_d(raw.get("p", 0)),
                qty=_d(raw.get("q", 0)),
                side="sell" if raw.get("m") else "buy",
                trade_id=str(raw.get("a", "")),
                ts=_ts_ms(raw.get("T", time.time() * 1000)),
            )
        except Exception:
            return None

    @staticmethod
    def _binance_symbol(s: str) -> str:
        for quote in ("USDT", "USDC", "BUSD", "BTC", "ETH", "BNB"):
            if s.endswith(quote) and len(s) > len(quote):
                return f"{s[:-len(quote)]}/{quote}"
        return s

    # -----------------------------------------------------------------------
    # OKX v5
    # -----------------------------------------------------------------------

    @staticmethod
    def okx_ticker(raw: Dict[str, Any]) -> Optional[CanonicalTick]:
        """
        raw = single item from OKX tickers data array.
        Keys: instId, bidPx, askPx, last, vol24h, ts.
        """
        try:
            return CanonicalTick(
                venue="okx",
                symbol=FeedNormaliser._okx_symbol(raw.get("instId", "")),
                bid=_d(raw.get("bidPx", 0)),
                ask=_d(raw.get("askPx", 0)),
                last=_d(raw.get("last", 0)),
                volume_24h=_d(raw.get("vol24h", 0)),
                ts=_ts_ms(raw.get("ts", time.time() * 1000)),
            )
        except Exception:
            return None

    @staticmethod
    def okx_book(raw: Dict[str, Any]) -> Optional[CanonicalBook]:
        """
        raw = single item from OKX books5 data array.
        Keys: instId, bids [[price,qty,liquidOrd,numOrds],...], asks, ts.
        """
        try:
            bids = [(Decimal(b[0]), Decimal(b[1])) for b in (raw.get("bids") or [])]
            asks = [(Decimal(a[0]), Decimal(a[1])) for a in (raw.get("asks") or [])]
            return CanonicalBook(
                venue="okx",
                symbol=FeedNormaliser._okx_symbol(raw.get("instId", "")),
                bids=sorted(bids, key=lambda x: x[0], reverse=True),
                asks=sorted(asks, key=lambda x: x[0]),
                ts=_ts_ms(raw.get("ts", time.time() * 1000)),
            )
        except Exception:
            return None

    @staticmethod
    def okx_trade(raw: Dict[str, Any]) -> Optional[CanonicalTrade]:
        """
        raw = single item from OKX trades data array.
        Keys: instId, px, sz, side, tradeId, ts.
        """
        try:
            return CanonicalTrade(
                venue="okx",
                symbol=FeedNormaliser._okx_symbol(raw.get("instId", "")),
                price=_d(raw.get("px", 0)),
                qty=_d(raw.get("sz", 0)),
                side=raw.get("side", "buy").lower(),
                trade_id=str(raw.get("tradeId", "")),
                ts=_ts_ms(raw.get("ts", time.time() * 1000)),
            )
        except Exception:
            return None

    @staticmethod
    def _okx_symbol(s: str) -> str:
        """BTC-USDT-SWAP or BTC-USDT → BTC/USDT."""
        parts = s.split("-")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return s
