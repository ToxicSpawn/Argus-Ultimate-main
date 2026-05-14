"""
alpha/microstructure/ws_feed_adapter.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
WS feed adapter — bridges raw exchange WebSocket messages to LiveSignalBus.

This is the single place that translates exchange-specific WS payloads
into the canonical LiveSignalBus.on_trade() / on_book_update() calls,
and simultaneously records LatencyTelemetry MARKET_DATA_RX timestamps.

Supported feeds
---------------
  - Hyperliquid WebSocket (fills + L2Book)
  - Kraken WebSocket v2 (trade + book)
  - Coinbase Advanced Trade WebSocket (market_trades + level2)

Architecture
------------
  WSFeedAdapter
    ├─ .attach_hyperliquid(ws_client)
    ├─ .attach_kraken(ws_client)
    └─ .attach_coinbase(ws_client)

  Each attach_*() method monkey-patches / registers callbacks on the
  exchange client so that incoming messages flow through this adapter.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from alpha.microstructure.live_signal_bus import LiveSignalBus
from hft_engine.latency_telemetry import JitterMonitor, LatencyStage, LatencyTelemetry

logger = logging.getLogger(__name__)


class WSFeedAdapter:
    """
    Bridges raw WS payloads to LiveSignalBus + LatencyTelemetry.

    Parameters
    ----------
    bus : LiveSignalBus
        The singleton signal aggregator.
    telemetry : LatencyTelemetry | None
        Latency telemetry singleton (optional; creates new if None).
    jitter : JitterMonitor | None
        Jitter monitor (optional; creates new if None).
    """

    def __init__(
        self,
        bus: LiveSignalBus,
        telemetry: Optional[LatencyTelemetry] = None,
        jitter: Optional[JitterMonitor] = None,
    ) -> None:
        self._bus = bus
        self._tel = telemetry or LatencyTelemetry.get_instance()
        self._jitter = jitter or JitterMonitor()
        self._journey_map: Dict[str, str] = {}  # symbol -> current journey_id

    # ── Hyperliquid ───────────────────────────────────────────────────────────

    def on_hyperliquid_trade(self, msg: Dict[str, Any]) -> None:
        """Handle a Hyperliquid trade WebSocket message.

        Expected shape (channel=trades)::

            {"data": [{"coin": "BTC", "side": "B", "sz": "0.01",
                       "px": "30000.0", "time": 1234567890123}]}
        """
        try:
            ts_ns = time.time_ns()
            trades: List[Dict] = msg.get("data", [])
            if not isinstance(trades, list):
                trades = [trades]

            for t in trades:
                sym = str(t.get("coin", "")).upper()
                if not sym:
                    continue

                # Stamp MARKET_DATA_RX
                jid = self._tel.start_journey(sym)
                self._journey_map[sym] = jid
                self._jitter.record_tick(sym)

                side = "buy" if str(t.get("side", "")).upper() in ("B", "BUY") else "sell"
                size = float(t.get("sz", 0.0))
                price = float(t.get("px", 0.0))

                self._bus.on_trade(sym, side, size, price, ts_ns)

                self._tel.mark(jid, LatencyStage.SIGNAL_COMPUTE)
                logger.debug("WSFeedAdapter[HL] trade: %s %s %.6f @ %.4f", sym, side, size, price)

        except Exception as exc:  # noqa: BLE001
            logger.warning("WSFeedAdapter: HL trade parse error: %s", exc)

    def on_hyperliquid_book(self, msg: Dict[str, Any]) -> None:
        """Handle a Hyperliquid L2Book WebSocket message.

        Expected shape (channel=l2Book)::

            {"data": {"coin": "BTC", "levels": [[bids], [asks]],
                      "time": 1234567890123}}
        """
        try:
            ts_ns = time.time_ns()
            data = msg.get("data", {})
            sym = str(data.get("coin", "")).upper()
            if not sym:
                return

            levels = data.get("levels", [[], []])
            raw_bids: List = levels[0] if len(levels) > 0 else []
            raw_asks: List = levels[1] if len(levels) > 1 else []

            bids: List[Tuple[float, float]] = [
                (float(b.get("px", 0)), float(b.get("sz", 0))) for b in raw_bids
            ]
            asks: List[Tuple[float, float]] = [
                (float(a.get("px", 0)), float(a.get("sz", 0))) for a in raw_asks
            ]

            self._bus.on_book_update(sym, bids, asks, ts_ns)
            logger.debug("WSFeedAdapter[HL] book: %s bids=%d asks=%d", sym, len(bids), len(asks))

        except Exception as exc:  # noqa: BLE001
            logger.warning("WSFeedAdapter: HL book parse error: %s", exc)

    # ── Kraken ────────────────────────────────────────────────────────────────

    def on_kraken_trade(self, msg: Any) -> None:
        """Handle a Kraken WebSocket v2 trade message.

        Expected shape::

            {"channel": "trade", "data": [{"symbol": "BTC/USD",
             "side": "buy", "qty": 0.1, "price": 30000.0,
             "timestamp": "2024-01-01T00:00:00.000000Z"}]}
        """
        try:
            ts_ns = time.time_ns()
            trades = msg.get("data", []) if isinstance(msg, dict) else []

            for t in trades:
                raw_sym = str(t.get("symbol", ""))
                sym = raw_sym.replace("/", "").upper()
                if not sym:
                    continue

                jid = self._tel.start_journey(sym)
                self._journey_map[sym] = jid
                self._jitter.record_tick(sym)

                side = str(t.get("side", "buy")).lower()
                size = float(t.get("qty", 0.0))
                price = float(t.get("price", 0.0))

                self._bus.on_trade(sym, side, size, price, ts_ns)
                self._tel.mark(jid, LatencyStage.SIGNAL_COMPUTE)

        except Exception as exc:  # noqa: BLE001
            logger.warning("WSFeedAdapter: Kraken trade parse error: %s", exc)

    def on_kraken_book(self, msg: Any) -> None:
        """Handle a Kraken WebSocket v2 book snapshot/update.

        Expected shape::

            {"channel": "book", "data": [{"symbol": "BTC/USD",
              "bids": [{"price": 30000.0, "qty": 0.5}],
              "asks": [{"price": 30001.0, "qty": 0.3}]}]}
        """
        try:
            ts_ns = time.time_ns()
            entries = msg.get("data", []) if isinstance(msg, dict) else []

            for entry in entries:
                raw_sym = str(entry.get("symbol", ""))
                sym = raw_sym.replace("/", "").upper()
                if not sym:
                    continue

                bids: List[Tuple[float, float]] = [
                    (float(b["price"]), float(b["qty"])) for b in entry.get("bids", [])
                ]
                asks: List[Tuple[float, float]] = [
                    (float(a["price"]), float(a["qty"])) for a in entry.get("asks", [])
                ]

                self._bus.on_book_update(sym, bids, asks, ts_ns)

        except Exception as exc:  # noqa: BLE001
            logger.warning("WSFeedAdapter: Kraken book parse error: %s", exc)

    # ── Coinbase ──────────────────────────────────────────────────────────────

    def on_coinbase_trade(self, msg: Dict[str, Any]) -> None:
        """Handle a Coinbase Advanced Trade WebSocket market_trades message.

        Expected shape::

            {"type": "market_trades", "events": [{"trades": [
              {"product_id": "BTC-USD", "side": "BUY",
               "size": "0.01", "price": "30000.00"}]}]}
        """
        try:
            ts_ns = time.time_ns()
            for event in msg.get("events", []):
                for t in event.get("trades", []):
                    raw_sym = str(t.get("product_id", ""))
                    sym = raw_sym.replace("-", "").upper()
                    if not sym:
                        continue

                    jid = self._tel.start_journey(sym)
                    self._journey_map[sym] = jid
                    self._jitter.record_tick(sym)

                    side = str(t.get("side", "BUY")).lower()
                    size = float(t.get("size", 0.0))
                    price = float(t.get("price", 0.0))

                    self._bus.on_trade(sym, side, size, price, ts_ns)
                    self._tel.mark(jid, LatencyStage.SIGNAL_COMPUTE)

        except Exception as exc:  # noqa: BLE001
            logger.warning("WSFeedAdapter: Coinbase trade parse error: %s", exc)

    def on_coinbase_book(self, msg: Dict[str, Any]) -> None:
        """Handle a Coinbase Advanced Trade WebSocket level2 message.

        Expected shape::

            {"type": "l2_data", "events": [{"product_id": "BTC-USD",
              "updates": [{"side": "bid", "price_level": "30000", "new_quantity": "0.5"}]}]}
        """
        try:
            ts_ns = time.time_ns()
            for event in msg.get("events", []):
                raw_sym = str(event.get("product_id", ""))
                sym = raw_sym.replace("-", "").upper()
                if not sym:
                    continue

                bids: List[Tuple[float, float]] = []
                asks: List[Tuple[float, float]] = []

                for upd in event.get("updates", []):
                    side_str = str(upd.get("side", "")).lower()
                    px = float(upd.get("price_level", 0.0))
                    qty = float(upd.get("new_quantity", 0.0))
                    if side_str == "bid":
                        bids.append((px, qty))
                    else:
                        asks.append((px, qty))

                if bids or asks:
                    bids.sort(key=lambda x: x[0], reverse=True)
                    asks.sort(key=lambda x: x[0])
                    self._bus.on_book_update(sym, bids, asks, ts_ns)

        except Exception as exc:  # noqa: BLE001
            logger.warning("WSFeedAdapter: Coinbase book parse error: %s", exc)

    # ── Journey completion helper ─────────────────────────────────────────────

    def complete_journey(self, symbol: str, stage: LatencyStage = LatencyStage.ORDER_SUBMIT) -> None:
        """Mark final stage and complete the journey for *symbol*.

        Call this from executor after place_order() returns.
        """
        sym = symbol.upper()
        jid = self._journey_map.pop(sym, None)
        if jid is None:
            return
        self._tel.mark(jid, stage)
        self._tel.complete_journey(jid)

    def complete_fill(self, symbol: str) -> None:
        """Mark FILL_RX and complete journey — call from fill handler."""
        self.complete_journey(symbol, LatencyStage.FILL_RX)

    # ── Jitter / health ───────────────────────────────────────────────────────

    def jitter_report(self) -> Dict[str, float]:
        """Return per-symbol jitter in microseconds."""
        return {sym: self._jitter.jitter_us(sym) for sym in self._bus._symbols}

    def latency_stats(self) -> dict:
        """Return full latency stats from telemetry."""
        return self._tel.get_stats()
