from exchanges.centralized.coinbase_advanced import build_coinbase_advanced_adapter
from exchanges.centralized.kraken import build_kraken_adapter
from strategies import StrategyTargetProposal, TargetOnlyStrategyMixin


class DummyClient:
    def create_order(self, **kwargs):
        return {"id": "order_1", **kwargs}

    def fetch_order(self, order_id, symbol):
        return {"id": order_id, "symbol": symbol, "status": "closed", "filled": 1.0, "average": 100.0}


class DemoStrategy(TargetOnlyStrategyMixin):
    def produce_target_proposal(self):
        return StrategyTargetProposal(
            strategy_id="demo",
            symbol="BTC/AUD",
            target_weight=0.10,
            current_weight=0.02,
            reference_price=50_000.0,
            reason="demo target output",
        )


def test_exchange_builders_return_adapters() -> None:
    kraken = build_kraken_adapter(DummyClient(), dry_run=True)
    coinbase = build_coinbase_advanced_adapter(DummyClient(), dry_run=True)
    assert kraken.submit_limit_order(symbol="BTC/AUD", side="buy", quantity=1.0, price=1.0).success is True
    assert coinbase.submit_limit_order(symbol="ETH/AUD", side="sell", quantity=1.0, price=1.0).success is True


def test_strategy_contract_is_target_only() -> None:
    proposal = DemoStrategy().produce_target_proposal()
    assert proposal.symbol == "BTC/AUD"
    assert proposal.target_weight > proposal.current_weight
