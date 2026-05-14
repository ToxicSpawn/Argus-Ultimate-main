from argus_live.risk.constitution_gate import evaluate_constitution


def test_constitution_rejects_excess_symbol_exposure() -> None:
    result = evaluate_constitution(
        order_notional=5000,
        symbol_notional_after=20000,
        cluster_notional_after=20000,
        gross_notional_after=20000,
        equity=100000,
        max_gross_exposure_pct=0.25,
        max_single_symbol_exposure_pct=0.08,
        max_cluster_exposure_pct=0.15,
    )
    assert result.allowed is False
