from __future__ import annotations

from typing import Any

from argus_live.execution.venue_adapter import VenueAdapter, VenueOrderResult


class CcxtCoinbaseAdapter(VenueAdapter):
    def __init__(self, exchange_client: Any, dry_run: bool = True) -> None:
        self.exchange_client = exchange_client
        self.dry_run = dry_run

    def submit_limit_order(self, *, symbol: str, side: str, quantity: float, price: float) -> VenueOrderResult:
        if self.dry_run:
            return VenueOrderResult(True, "dryrun_coinbase_order", "dry-run accepted", {"symbol": symbol, "side": side, "quantity": quantity, "price": price})
        try:
            result = self.exchange_client.create_order(symbol=symbol, type="limit", side=side, amount=quantity, price=price)
            return VenueOrderResult(True, str(result.get("id")), "order submitted", result)
        except Exception as exc:
            return VenueOrderResult(False, None, f"submit failed: {exc}", None)

    def fetch_order(self, *, venue_order_id: str, symbol: str) -> dict[str, Any]:
        if self.dry_run:
            return {"id": venue_order_id, "symbol": symbol, "status": "closed", "filled": 1.0, "average": 0.0}
        return self.exchange_client.fetch_order(venue_order_id, symbol)
