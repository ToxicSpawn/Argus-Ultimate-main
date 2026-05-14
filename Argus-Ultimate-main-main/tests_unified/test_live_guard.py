from argus_live.control_plane.live_guard import check_live_allowed


def test_strategy_node_cannot_run_live() -> None:
    result = check_live_allowed(
        requested_mode="live",
        node_role="strategy-node",
        operator_ack_present=True,
        soak_ok=True,
        reconciliation_ok=True,
        operator_halted=False,
        operator_frozen=False,
    )
    assert result.allowed is False
