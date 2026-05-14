import tempfile

from argus_live.execution.venue_adapter import VenueAdapter, VenueOrderResult
from unified_execution_engine import UnifiedExecutionEngine, UnifiedExecutionRequest
from unified_trading_system import UnifiedTradingInput, UnifiedTradingSystem


class DummyAdapter(VenueAdapter):
    def submit_limit_order(self, *, symbol, side, quantity, price):
        return VenueOrderResult(True, "dummy_order", "ok", {})

    def fetch_order(self, *, venue_order_id, symbol):
        return {
            "id": venue_order_id,
            "symbol": symbol,
            "status": "closed",
            "filled": 1.0,
            "average": 50000.0,
        }


def test_unified_execution_engine_forwards_into_argus_live() -> None:
    with tempfile.NamedTemporaryFile() as journal_file, tempfile.NamedTemporaryFile() as ledger_file:
        engine = UnifiedExecutionEngine(
            manifest_hash="sha256:test",
            journal_path=journal_file.name,
            ledger_path=ledger_file.name,
            adapters={"kraken": DummyAdapter(), "coinbase_advanced": DummyAdapter()},
        )
        intent_id = engine.execute(
            UnifiedExecutionRequest(
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
            )
        )
        assert isinstance(intent_id, str)


def test_unified_trading_system_uses_target_driven_flow() -> None:
    with tempfile.NamedTemporaryFile() as journal_file, tempfile.NamedTemporaryFile() as ledger_file:
        engine = UnifiedExecutionEngine(
            manifest_hash="sha256:test",
            journal_path=journal_file.name,
            ledger_path=ledger_file.name,
            adapters={"kraken": DummyAdapter(), "coinbase_advanced": DummyAdapter()},
        )
        system = UnifiedTradingSystem(engine)
        result = system.process(
            UnifiedTradingInput(
                strategy_id="target_test",
                symbol="BTC/AUD",
                target_weight=0.10,
                current_weight=0.02,
                reference_price=50000.0,
                manifest_hash="sha256:test",
                portfolio_equity=100000.0,
                symbol_notional_after=5000.0,
                cluster_notional_after=5000.0,
                gross_notional_after=5000.0,
                top_of_book_notional=100000.0,
                spread_bps=3.0,
                volatility_bps=10.0,
                fee_rate=0.002,
            )
        )
        assert result is not None
