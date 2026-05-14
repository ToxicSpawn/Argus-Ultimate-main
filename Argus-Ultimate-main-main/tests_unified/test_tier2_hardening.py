from pathlib import Path

from argus_live.execution.adapter_registry import AdapterRegistry
from argus_live.execution.entrypoint import ExecutionEntrypoint
from argus_live.execution.intent_runtime import IntentRuntime
from argus_live.execution.router_policy import select_route
from argus_live.execution.venue_adapter import VenueAdapter, VenueOrderResult
from argus_live.execution.venue_health import VenueHealthModel
from argus_live.governance.coordinator import ArgusGovernanceCoordinator
from argus_live.ledger.event_journal import EventJournal
from argus_live.ledger.trade_ledger import LedgerFillRecord, TradeLedger
from argus_live.replay.replay_audit import ReplayAudit, ReplayAuditStore


class Adapter(VenueAdapter):
    def submit_limit_order(self, *, symbol: str, side: str, quantity: float, price: float) -> VenueOrderResult:
        return VenueOrderResult(True, 'oid-1', 'accepted')

    def fetch_order(self, *, venue_order_id: str, symbol: str) -> dict:
        return {'id': venue_order_id, 'symbol': symbol, 'status': 'filled', 'filled': 1.0, 'average': 100.0}


def _runtime(tmp_path: Path):
    journal = EventJournal(tmp_path / 'journal.jsonl')
    ledger = TradeLedger(tmp_path / 'fills.jsonl')
    registry = AdapterRegistry({'kraken': Adapter(), 'coinbase_advanced': Adapter()})
    replay_store = ReplayAuditStore(tmp_path / 'replay.jsonl')
    gov = ArgusGovernanceCoordinator(db_path=str(tmp_path / 'incidents.db'))
    return IntentRuntime(journal=journal, adapter_registry=registry, trade_ledger=ledger, governance_coordinator=gov, replay_audit_store=replay_store, strict_governance=True), ledger, replay_store


def test_venue_health_prefers_better_venue(tmp_path: Path):
    model = VenueHealthModel()
    good = [LedgerFillRecord(intent_id='1', symbol='BTC/AUD', side='buy', quantity=1, price=100, venue='kraken', fill_qty=1, latency_ms=10, execution_alpha_bps=1.5)]
    bad = [LedgerFillRecord(intent_id='2', symbol='BTC/AUD', side='buy', quantity=1, price=100, venue='coinbase_advanced', fill_qty=1, latency_ms=900, execution_alpha_bps=-5.0, reject_flag=1, adverse_price_move_bps=8.0)]
    snaps = [model.snapshot(venue='kraken', fills=good), model.snapshot(venue='coinbase_advanced', fills=bad)]
    route = select_route(symbol='BTC/AUD', spread_bps=3.0, volatility_bps=10.0, allow_market_orders=False, venue_health=snaps)
    assert route.venue == 'kraken'


def test_entrypoint_blocks_on_existing_replay_mismatch(tmp_path: Path):
    runtime, _ledger, replay_store = _runtime(tmp_path)
    replay_store.append(ReplayAudit(ts='2026-04-02T00:00:00+00:00', run_id='run-1', status='FAIL', mismatch_count=2, notes='bad', journal_checksum='x', terminal_state_hash='y'))
    ep = ExecutionEntrypoint(runtime=runtime, manifest_hash='run-1', strict_governance=True)
    try:
        ep.submit_order(symbol='BTC/AUD', side='buy', quantity=1.0, strategy_id='s1', price=100.0, equity=10000.0, symbol_notional_after=100.0, cluster_notional_after=100.0, gross_notional_after=100.0, top_of_book_notional=10000.0, spread_bps=2.0, volatility_bps=5.0, fee_rate=0.001)
    except RuntimeError as e:
        assert 'Replay mismatch present' in str(e)
    else:
        raise AssertionError('expected replay blocker')


def test_execution_context_uses_market_state_and_adverse_selection(tmp_path: Path):
    runtime, ledger, _replay_store = _runtime(tmp_path)
    ledger.append_fill(LedgerFillRecord(intent_id='1', symbol='BTC/AUD', side='buy', quantity=1, price=100, venue='kraken', strategy_id='s1', fill_qty=1, expected_price=100, fill_price=100.2, slippage_bps=20.0, execution_alpha_bps=-20.0, adverse_price_move_bps=9.0))

    class Intent:
        symbol = 'BTC/AUD'
        strategy_id = 's1'
        limit_price = 100.0

    ctx = runtime._build_execution_context(Intent(), spread_bps=4.0, volatility_bps=12.0, top_of_book_notional=1000.0, approved_quantity=1.0, market_state={'order_book_imbalance': 0.7, 'microprice_drift_bps': 1.8, 'queue_position_score': 0.9, 'fill_probability': 0.88, 'urgency_score': 1.2, 'venue': 'kraken'})
    assert ctx.imbalance_score > 0.5
    assert ctx.fill_probability >= 0.88 - 1e-9
    assert ctx.adverse_selection_score > 0.0
