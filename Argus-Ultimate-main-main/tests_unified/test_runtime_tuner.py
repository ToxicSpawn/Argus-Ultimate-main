from argus_live.optimization.runtime_tuner import suggest_tuning

def test_runtime_tuner_tightens_threshold_when_slippage_high() -> None:
    suggestion = suggest_tuning(
        current_spread_threshold_bps=5.0,
        observed_slippage_bps=8.0,
        target_slippage_bps=4.0,
    )
    assert suggestion.proposed_value < suggestion.current_value
