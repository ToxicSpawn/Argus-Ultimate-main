from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, List

from argus_live.execution.adapter_registry import AdapterRegistry
from argus_live.execution.entrypoint import ExecutionEntrypoint
from argus_live.execution.intent_runtime import IntentRuntime
from argus_live.execution.venue_adapter import VenueAdapter, VenueOrderResult
from argus_live.execution.fill_tracker import set_fill_simulator, clear_fill_simulator
from argus_live.governance.coordinator import ArgusGovernanceCoordinator
from argus_live.ledger.event_journal import EventJournal
from argus_live.ledger.trade_ledger import TradeLedger
from argus_live.proving.session_lifecycle import AutoProvingLifecycle
from argus_live.replay.replay_audit import ReplayAuditStore

from .fill_realism import FillRealismEngine
from .hostile_scenarios import HostileScenarioInjector, MarketState, ScenarioPlan


@dataclass(frozen=True)
class ScenarioOrder:
    symbol: str
    side: str
    quantity: float
    strategy_id: str
    price: float
    equity: float = 10000.0
    symbol_notional_after: float = 500.0
    cluster_notional_after: float = 1000.0
    gross_notional_after: float = 1500.0
    fee_rate: float = 0.0026
    allow_market_orders: bool = False


@dataclass(frozen=True)
class HarnessResult:
    run_id: str
    report_path: str
    operator_summary_path: str
    journal_path: str
    ledger_path: str
    replay_audit_path: str
    incidents_db_path: str
    scenario_path: str
    trade_count: int


class SimulatedVenueAdapter(VenueAdapter):
    def __init__(self, venue: str, state_ref: dict[str, MarketState]):
        self.venue = venue
        self.state_ref = state_ref
        self._last_order: dict[str, Any] = {}

    def submit_limit_order(self, *, symbol: str, side: str, quantity: float, price: float) -> VenueOrderResult:
        self._last_order = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
        }
        return VenueOrderResult(success=True, venue_order_id=f"{self.venue}-{symbol}-1", reason="simulated")

    def fetch_order(self, *, venue_order_id: str, symbol: str) -> dict[str, Any]:
        state = self.state_ref[symbol]
        return {
            "id": venue_order_id,
            "status": "closed",
            "filled": self._last_order.get("quantity", 0.0),
            "remaining": 0.0,
            "average": self._last_order.get("price", state.mid_price),
            "symbol": symbol,
            "side": self._last_order.get("side", "buy"),
        }


class HostileReplayHarness:
    def __init__(self, artifacts_dir: str | Path, *, run_id: str = "hostile-run", strict_governance: bool = True) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.strict_governance = strict_governance

    def run(self, *, scenario: ScenarioPlan, orders: Iterable[ScenarioOrder]) -> HarnessResult:
        state_ref: dict[str, MarketState] = {scenario.base_state.symbol: scenario.base_state}
        journal_path = self.artifacts_dir / "journal.jsonl"
        ledger_path = self.artifacts_dir / "fills.jsonl"
        incidents_db = self.artifacts_dir / "governance_incidents.db"
        replay_audit_path = self.artifacts_dir / "replay_audit.jsonl"
        report_path = self.artifacts_dir / "proving_review.json"
        operator_summary = self.artifacts_dir / "operator_summary.json"
        scenario_path = self.artifacts_dir / "scenario_trace.jsonl"

        journal = EventJournal(journal_path)
        ledger = TradeLedger(ledger_path)
        governance = ArgusGovernanceCoordinator(db_path=str(incidents_db))
        replay_store = ReplayAuditStore(replay_audit_path)
        proving = AutoProvingLifecycle(
            incidents_path=incidents_db,
            replay_audit_path=replay_audit_path,
            report_path=report_path,
            operator_summary_path=operator_summary,
        )
        registry = AdapterRegistry({
            "kraken": SimulatedVenueAdapter("kraken", state_ref),
            "coinbase_advanced": SimulatedVenueAdapter("coinbase_advanced", state_ref),
        })
        runtime = IntentRuntime(
            journal=journal,
            adapter_registry=registry,
            trade_ledger=ledger,
            governance_coordinator=governance,
            replay_audit_store=replay_store,
            strict_governance=self.strict_governance,
        )
        entry = ExecutionEntrypoint(runtime=runtime, manifest_hash=self.run_id, proving_lifecycle=proving, profile="paper")
        injector = HostileScenarioInjector(scenario)
        fill_engine = FillRealismEngine()
        scenario_rows: list[dict[str, Any]] = []

        def simulator(intent_id: str, quantity: float, price: float, metadata: dict[str, Any] | None = None):
            md = dict(metadata or {})
            symbol = str(md.get("symbol") or scenario.base_state.symbol)
            md["market_state"] = state_ref[symbol]
            return fill_engine.simulate(intent_id, quantity, price, md)

        set_fill_simulator(simulator)
        try:
            for step, order in enumerate(list(orders)):
                market_state = injector.state_at(step)
                state_ref[order.symbol] = market_state
                scenario_rows.append({"step": step, **asdict(market_state)})
                entry.submit_order(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    strategy_id=order.strategy_id,
                    price=order.price,
                    equity=order.equity,
                    symbol_notional_after=order.symbol_notional_after,
                    cluster_notional_after=order.cluster_notional_after,
                    gross_notional_after=order.gross_notional_after,
                    top_of_book_notional=market_state.top_of_book_notional,
                    spread_bps=market_state.spread_bps,
                    volatility_bps=market_state.volatility_bps,
                    fee_rate=order.fee_rate,
                    allow_market_orders=order.allow_market_orders,
                )
        finally:
            clear_fill_simulator()

        scenario_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in scenario_rows) + ("\n" if scenario_rows else ""), encoding="utf-8")
        trade_count = len(ledger.load_recent_fills())
        return HarnessResult(
            run_id=self.run_id,
            report_path=str(report_path),
            operator_summary_path=str(operator_summary),
            journal_path=str(journal_path),
            ledger_path=str(ledger_path),
            replay_audit_path=str(replay_audit_path),
            incidents_db_path=str(incidents_db),
            scenario_path=str(scenario_path),
            trade_count=trade_count,
        )
