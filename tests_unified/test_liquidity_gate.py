from argus_live.risk.liquidity_gate import apply_liquidity_haircut


def test_liquidity_gate_haircuts_requested_size() -> None:
    result = apply_liquidity_haircut(
        requested_quantity=10.0,
        reference_price=100.0,
        top_of_book_notional=200.0,
        max_book_take_ratio=0.5,
    )
    assert result.approved_notional == 100.0
    assert result.approved_quantity == 1.0
