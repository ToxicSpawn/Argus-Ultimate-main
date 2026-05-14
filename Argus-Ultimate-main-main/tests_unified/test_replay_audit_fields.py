import tempfile
from pathlib import Path

from argus_live.execution.adapter_registry import AdapterRegistry
from argus_live.execution.entrypoint import ExecutionEntrypoint
from argus_live.execution.intent_runtime import IntentRuntime
from argus_live.execution.venue_adapter import VenueAdapter, VenueOrderResult
from argus_live.governance.coordinator import ArgusGovernanceCoordinator
from argus_live.ledger.event_journal import EventJournal
from argus_live.ledger.trade_ledger import TradeLedger
from argus_live.replay.replay_audit import ReplayAuditStore


class DummyAdapter(VenueAdapter):
    def submit_limit_order(self, *, symbol, side, quantity, price):
        return VenueOrderResult(True, "dummy_order", "ok", {})

    def fetch_order(self, *, venue_order_id, symbol):
        return {"id": venue_order_id, "symbol": symbol, "status": "closed", "filled": 1.0, "average": 50000.0}


def test_replay_audit_has_checksums_and_hashes() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        journal = EventJournal(td_path / "journal.jsonl")
        ledger = TradeLedger(td_path / "ledger.jsonl")
        replay_store = ReplayAuditStore(td_path / "replay.jsonl")
        runtime = IntentRuntime(
            journal,
            AdapterRegistry({"kraken": DummyAdapter(), "coinbase_advanced": DummyAdapter()}),
            ledger,
            governance_coordinator=ArgusGovernanceCoordinator(db_path=str(td_path / "incidents.db")),
            replay_audit_store=replay_store,
        )
        entry = ExecutionEntrypoint(runtime, manifest_hash="sha256:test")
        entry.submit_order(
            symbol="BTC/AUD",
            side="buy",
            quantity=0.1,
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
        latest = replay_store.latest_for_run("sha256:test")
        assert latest is not None
        assert latest.journal_checksum.startswith("sha256:")
        assert latest.terminal_state_hash.startswith("sha256:")
