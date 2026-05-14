from argus_live.portfolio.strategy_allocator import allocate_capital

def test_strategy_allocator_weights_capital() -> None:
    allocations = allocate_capital(total_capital=1000.0, strategy_scores={"a": 2.0, "b": 1.0})
    by_id = {a.strategy_id: a for a in allocations}
    assert by_id["a"].target_capital > by_id["b"].target_capital

def test_strategy_allocator_equal_fallback() -> None:
    allocations = allocate_capital(total_capital=1000.0, strategy_scores={"a": 0.0, "b": 0.0})
    assert len(allocations) == 2
    assert allocations[0].target_capital == allocations[1].target_capital
