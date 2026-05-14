from argus_live.execution.routing_mesh import VenueScore, rank_venues

def test_routing_mesh_picks_best_score() -> None:
    decision = rank_venues(
        symbol="BTC/AUD",
        venue_inputs=[
            VenueScore("kraken", 9.0, 2.0, 30.0, 10.0, 100000.0, "good"),
            VenueScore("coinbase_advanced", 8.0, 3.0, 40.0, 12.0, 90000.0, "ok"),
        ],
    )
    assert decision.primary_venue == "kraken"
    assert decision.backup_venue == "coinbase_advanced"
