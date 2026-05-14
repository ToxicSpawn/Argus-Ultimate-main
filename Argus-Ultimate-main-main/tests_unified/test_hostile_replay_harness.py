import json
from pathlib import Path

from argus_live.simulation.hostile_replay import HostileReplayHarness, ScenarioOrder
from argus_live.simulation.hostile_scenarios import MarketState, ScenarioPlan, ShockWindow


def test_hostile_replay_harness_emits_artifacts(tmp_path: Path):
    harness = HostileReplayHarness(tmp_path, run_id="hostile-test")
    scenario = ScenarioPlan(
        name="hostile",
        base_state=MarketState(symbol="BTC/AUD", mid_price=100000.0, spread_bps=6.0, volatility_bps=10.0, top_of_book_notional=5000.0),
        shocks=[
            ShockWindow(name="latency", start_step=1, end_step=1, latency_ms_add=80.0, spread_multiplier=1.8),
            ShockWindow(name="thin_book", start_step=2, end_step=2, liquidity_multiplier=0.2, stale_quote=True),
        ],
    )
    orders = [
        ScenarioOrder(symbol="BTC/AUD", side="buy", quantity=0.01, strategy_id="s1", price=100000.0),
        ScenarioOrder(symbol="BTC/AUD", side="buy", quantity=0.02, strategy_id="s1", price=100100.0),
        ScenarioOrder(symbol="BTC/AUD", side="sell", quantity=0.015, strategy_id="s2", price=99950.0),
    ]
    result = harness.run(scenario=scenario, orders=orders)
    assert Path(result.report_path).exists()
    assert Path(result.operator_summary_path).exists()
    assert Path(result.scenario_path).exists()
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert report["run_id"] == "hostile-test"
    assert Path(result.ledger_path).read_text(encoding="utf-8").strip()
