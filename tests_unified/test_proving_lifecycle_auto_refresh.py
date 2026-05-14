import json
import tempfile
from pathlib import Path

from argus_live.execution.adapter_registry import AdapterRegistry
from argus_live.execution.entrypoint import ExecutionEntrypoint
from argus_live.execution.intent_runtime import IntentRuntime
from argus_live.execution.venue_adapter import VenueAdapter, VenueOrderResult
from argus_live.governance.coordinator import ArgusGovernanceCoordinator
from argus_live.ledger.event_journal import EventJournal
from argus_live.ledger.trade_ledger import TradeLedger
from argus_live.proving.session_lifecycle import AutoProvingLifecycle
from argus_live.replay.replay_audit import ReplayAuditStore


class DummyAdapter(VenueAdapter):
    def submit_limit_order(self, *, symbol, side, quantity, price):
        return VenueOrderResult(True, "dummy_order", "ok", {})

    def fetch_order(self, *, venue_order_id, symbol):
        return {"id": venue_order_id, "symbol": symbol, "status": "closed", "filled": 1.0, "average": 50000.0}


def test_proving_lifecycle_auto_refresh_writes_review_artifact() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        journal_path = td_path / "event_journal.jsonl"
        ledger_path = td_path / "trade_ledger.jsonl"
        governance_db = td_path / "governance_incidents.db"
        replay_path = td_path / "replay_audit.jsonl"
        report_path = td_path / "proving_review.json"

        runtime = IntentRuntime(
            EventJournal(journal_path),
            AdapterRegistry({"kraken": DummyAdapter(), "coinbase_advanced": DummyAdapter()}),
            TradeLedger(ledger_path),
            governance_coordinator=ArgusGovernanceCoordinator(db_path=str(governance_db)),
            replay_audit_store=ReplayAuditStore(replay_path),
        )
        lifecycle = AutoProvingLifecycle(
            incidents_path=governance_db,
            replay_audit_path=replay_path,
            report_path=report_path,
        )
        entry = ExecutionEntrypoint(runtime, manifest_hash="sha256:test", proving_lifecycle=lifecycle)

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
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["run_id"] == "sha256:test"
        assert payload["trade_count"] >= 1
        journal_text = journal_path.read_text(encoding="utf-8")
        assert "PROVING_REVIEW_REFRESHED" in journal_text
