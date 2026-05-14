import tempfile

from argus_live.execution.adapter_registry import AdapterRegistry
from argus_live.execution.entrypoint import ExecutionEntrypoint
from argus_live.execution.intent_runtime import IntentRuntime
from argus_live.execution.venue_adapter import VenueAdapter, VenueOrderResult
from argus_live.ledger.event_journal import EventJournal
from argus_live.ledger.trade_ledger import TradeLedger


class DummyAdapter(VenueAdapter):
    def submit_limit_order(self, *, symbol, side, quantity, price):
        return VenueOrderResult(True, "dummy_order", "ok", {})

    def fetch_order(self, *, venue_order_id, symbol):
        return {"id": venue_order_id, "symbol": symbol, "status": "closed", "filled": 1.0, "average": 50000.0}


def test_phase_d_flow_reaches_attributed() -> None:
    with tempfile.NamedTemporaryFile() as journal_file, tempfile.NamedTemporaryFile() as ledger_file:
        journal = EventJournal(journal_file.name)
        ledger = TradeLedger(ledger_file.name)
        runtime = IntentRuntime(journal, AdapterRegistry({"kraken": DummyAdapter(), "coinbase_advanced": DummyAdapter()}), ledger)
        entry = ExecutionEntrypoint(runtime, manifest_hash="sha256:test")
        intent_id = entry.submit_order(
            symbol="BTC/AUD",
            side="buy",
            quantity=1.0,
            strategy_id="test_strategy",
            price=50000.0,
            equity=100000.0,
            symbol_notional_after=5000.0,
            cluster_notional_after=5000.0,
            gross_notional_after=5000.0,
            top_of_book_notional=100000.0,
            spread_bps=3.0,
            volatility_bps=10.0,
            fee_rate=0.002,
            allow_market_orders=False,
        )
        assert runtime.state[intent_id] == "ATTRIBUTED"
