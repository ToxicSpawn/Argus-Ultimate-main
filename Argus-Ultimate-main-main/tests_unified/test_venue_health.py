from argus_live.execution.venue_health import assess_venue_health

def test_venue_health_detects_latency_breach() -> None:
    health = assess_venue_health(
        venue="kraken",
        median_latency_ms=500.0,
        reject_rate=0.01,
        stale_data=False,
    )
    assert health.healthy is False
    assert "latency breach" in health.reason
