from __future__ import annotations

import time
from typing import Any, Optional

from argus_live.execution.venue_adapter import VenueAdapter, VenueOrderResult


class CcxtKrakenAdapter(VenueAdapter):
    """
    Thin CCXT wrapper for Kraken with optional WSFeedAdapter injection.

    When ws_adapter is supplied the on_trade() and on_book() methods should
    be registered as callbacks on the live Kraken WebSocket client so that
    OFI / VPIN update on every tick rather than once per OHLCV poll cycle.

    Kraken WS trade payload shape (v2 public/trades):
        {
            "type": "trade",
            "symbol": "BTC/USD",
            "trades": [
                {"price": 65000.0, "qty": 0.012, "side": "buy", "timestamp": 1713200000.123}
            ]
        }

    Kraken WS book payload shape (v2 public/book snapshot or update):
        {
            "type": "snapshot" | "update",
            "symbol": "BTC/USD",
            "bids": [[price, qty], ...],
            "asks": [[price, qty], ...],
            "timestamp": 1713200000.456
        }
    """

    def __init__(
        self,
        exchange_client: Any,
        dry_run: bool = True,
        ws_adapter: Optional[Any] = None,
    ) -> None:
        self.exchange_client = exchange_client
        self.dry_run = dry_run
        self._ws_adapter = ws_adapter

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def submit_limit_order(
        self, *, symbol: str, side: str, quantity: float, price: float
    ) -> VenueOrderResult:
        if self.dry_run:
            return VenueOrderResult(
                True,
                "dryrun_kraken_order",
                "dry-run accepted",
                {"symbol": symbol, "side": side, "quantity": quantity, "price": price},
            )
        try:
            result = self.exchange_client.create_order(
                symbol=symbol, type="limit", side=side, amount=quantity, price=price
            )
            return VenueOrderResult(True, str(result.get("id")), "order submitted", result)
        except Exception as exc:
            return VenueOrderResult(False, None, f"submit failed: {exc}", None)

    def fetch_order(self, *, venue_order_id: str, symbol: str) -> dict[str, Any]:
        if self.dry_run:
            return {
                "id": venue_order_id,
                "symbol": symbol,
                "status": "closed",
                "filled": 1.0,
                "average": 0.0,
            }
        return self.exchange_client.fetch_order(venue_order_id, symbol)

    # ------------------------------------------------------------------
    # WebSocket callbacks — register these with your Kraken WS client
    # ------------------------------------------------------------------

    def on_trade(self, payload: dict[str, Any]) -> None:
        """
        Called for every Kraken WS trade message.

        Expected payload matches Kraken v2 public/trades shape.  Each
        individual trade in payload["trades"] is forwarded to WSFeedAdapter
        so OFI / VPIN accumulate on every fill, not every 5-second poll.

        Usage (inside your Kraken WS client):
            ws_client.on_trade_callback = kraken_adapter.on_trade
        """
        if self._ws_adapter is None:
            return
        symbol_raw: str = payload.get("symbol", "")
        # Normalise "BTC/USD" -> "BTC", "BTC/AUD" -> "BTC"
        base = symbol_raw.split("/")[0] if "/" in symbol_raw else symbol_raw
        trades: list[dict[str, Any]] = payload.get("trades", [])
        for trade in trades:
            try:
                normalised = {
                    "symbol": base,
                    "price": float(trade["price"]),
                    "qty": float(trade.get("qty", trade.get("size", 0.0))),
                    "side": trade.get("side", "buy"),
                    "timestamp": float(
                        trade.get("timestamp", payload.get("timestamp", time.time()))
                    ),
                }
                self._ws_adapter.on_trade(normalised)
            except Exception:
                # Never crash the WS loop on a single malformed tick
                pass

    def on_book(self, payload: dict[str, Any]) -> None:
        """
        Called for every Kraken WS order-book snapshot or update message.

        Expected payload matches Kraken v2 public/book shape.  Best bid/ask
        are extracted and forwarded to WSFeedAdapter so the book-imbalance
        dimension of the LiveSignalBus updates on every quote change.

        Usage (inside your Kraken WS client):
            ws_client.on_book_callback = kraken_adapter.on_book
        """
        if self._ws_adapter is None:
            return
        symbol_raw: str = payload.get("symbol", "")
        base = symbol_raw.split("/")[0] if "/" in symbol_raw else symbol_raw
        bids: list[list[float]] = payload.get("bids", [])
        asks: list[list[float]] = payload.get("asks", [])
        if not bids or not asks:
            return
        try:
            best_bid_price = float(bids[0][0])
            best_bid_qty = float(bids[0][1])
            best_ask_price = float(asks[0][0])
            best_ask_qty = float(asks[0][1])
            normalised = {
                "symbol": base,
                "bid": best_bid_price,
                "bid_qty": best_bid_qty,
                "ask": best_ask_price,
                "ask_qty": best_ask_qty,
                "timestamp": float(payload.get("timestamp", time.time())),
            }
            self._ws_adapter.on_book_update(normalised)
        except Exception:
            pass
