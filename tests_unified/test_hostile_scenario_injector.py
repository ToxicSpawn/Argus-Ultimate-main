from argus_live.simulation.hostile_scenarios import MarketState, ScenarioPlan, ShockWindow, HostileScenarioInjector


def test_hostile_injector_applies_shocks():
    plan = ScenarioPlan(
        name="shock",
        base_state=MarketState(symbol="BTC/AUD", mid_price=100000.0, spread_bps=5.0, volatility_bps=8.0, top_of_book_notional=5000.0),
        shocks=[ShockWindow(name="news", start_step=1, end_step=2, spread_multiplier=2.0, liquidity_multiplier=0.4, reject_probability_add=0.2, latency_ms_add=50.0)],
    )
    injector = HostileScenarioInjector(plan)
    base = injector.state_at(0)
    shocked = injector.state_at(1)
    assert shocked.spread_bps > base.spread_bps
    assert shocked.top_of_book_notional < base.top_of_book_notional
    assert shocked.reject_probability > base.reject_probability
