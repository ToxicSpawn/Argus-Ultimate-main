from argus_live.optimization.live_learning import LiveLearningIntegrator
from argus_live.state.learning_state import LearningStateStore

def test_live_learning_updates_strategy_and_venue_state(tmp_path) -> None:
    store = LearningStateStore(tmp_path / "learning_state.json")
    integrator = LiveLearningIntegrator(
        state_store=store,
        recommendation_log_path=tmp_path / "learning_recommendations.jsonl",
    )
    result = integrator.process_trade_outcome(
        strategy_id="s1",
        symbol="BTC/AUD",
        venue="kraken",
        expected_edge_bps=10.0,
        realized_pnl_bps=8.0,
        slippage_bps=1.0,
        fee_bps=1.0,
        drawdown_pct=0.01,
        stability_score=1.2,
    )
    state = store.load()
    assert "s1" in state.strategy_edge_bps
    assert "kraken" in state.venue_slippage_bps
    assert result.lifecycle.strategy_id == "s1"
