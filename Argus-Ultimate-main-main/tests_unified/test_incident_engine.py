"""Tests for the incident engine and execution alpha tuning pack."""
from argus_live.governance.incident_and_execution_alpha_pack import (
    ArgusGovernanceCoordinator,
    ExecutionAlphaConfig,
    ExecutionAlphaTuningPack,
    ExecutionContext,
    IncidentEngine,
    PositionRecord,
    ReplayAuditRecord,
    RuntimeSnapshot,
    Thresholds,
    TradeRecord,
    utc_now_iso,
)


def _make_trade(**overrides) -> TradeRecord:
    defaults = dict(
        ts=utc_now_iso(), strategy_id="s1", symbol="BTC/AUD", venue="kraken",
        side="buy", qty=0.01, expected_price=50000.0, fill_price=50010.0,
        fees=1.0, gross_pnl=-0.5, net_pnl=-1.5, slippage_bps=2.0,
        execution_alpha_bps=1.0, maker_flag=1, partial_fill_flag=0,
        reject_flag=0, latency_ms=100.0, ladder_stage="PAPER",
    )
    defaults.update(overrides)
    return TradeRecord(**defaults)


def _make_snapshot(trades=None, positions=None, drawdown=None, replay_mismatch=0, metrics_lag=5.0):
    return RuntimeSnapshot(
        run_id="test-run",
        ladder_stage="PAPER",
        recent_trades=trades or [],
        recent_positions=positions or [],
        latest_replay_audit=ReplayAuditRecord(
            ts=utc_now_iso(), run_id="test-run", status="OK",
            mismatch_count=replay_mismatch,
        ),
        metrics_lag_seconds=metrics_lag,
        strategy_drawdown_pct=drawdown or {},
    )


def test_replay_mismatch_stops_trading():
    engine = IncidentEngine(Thresholds())
    snapshot = _make_snapshot(replay_mismatch=2)
    incidents, actions = engine.evaluate(snapshot)
    assert any(i.severity == "CRITICAL" for i in incidents)
    assert actions.stop_trading is True
    assert actions.block_promotions is True


def test_slippage_spike_detected():
    trades = [_make_trade(slippage_bps=50.0) for _ in range(15)]
    engine = IncidentEngine(Thresholds())
    snapshot = _make_snapshot(trades=trades)
    incidents, actions = engine.evaluate(snapshot)
    assert any("slippage" in i.title.lower() for i in incidents)


def test_strategy_drawdown_reduces_weight():
    engine = IncidentEngine(Thresholds())
    snapshot = _make_snapshot(drawdown={"s1": 3.5})
    incidents, actions = engine.evaluate(snapshot)
    assert "s1" in actions.reduce_strategy_weight
    assert actions.reduce_strategy_weight["s1"] < 1.0


def test_concentration_pauses_symbol():
    positions = [PositionRecord(
        ts=utc_now_iso(), symbol="BTC/AUD", strategy_id="s1",
        notional=500.0, exposure_pct=40.0,
    )]
    engine = IncidentEngine(Thresholds())
    snapshot = _make_snapshot(positions=positions)
    incidents, actions = engine.evaluate(snapshot)
    assert "BTC/AUD" in actions.pause_symbols


def test_execution_alpha_tuning_cancel():
    pack = ExecutionAlphaTuningPack(ExecutionAlphaConfig())
    ctx = ExecutionContext(
        strategy_id="s1", symbol="BTC/AUD", venue="kraken",
        expected_edge_bps=5.0, spread_bps=2.0, short_horizon_drift_bps=1.0,
        volatility_score=2.0, imbalance_score=0.5, adverse_selection_score=0.3,
        fill_probability=0.20, queue_position_score=0.3, urgency_score=0.5,
        top_book_notional=1000.0, remaining_notional=200.0,
        maker_retries_used=0, elapsed_wait_ms=100,
        current_edge_decay_bps=0.5, spread_widen_bps=0.3,
    )
    decision = pack.decide(ctx)
    assert decision.cancel is True


def test_governance_coordinator_roundtrip(tmp_path):
    db = str(tmp_path / "test_incidents.db")
    coordinator = ArgusGovernanceCoordinator(db_path=db)
    trades = [_make_trade(slippage_bps=25.0, execution_alpha_bps=-3.0) for _ in range(15)]
    snapshot = _make_snapshot(trades=trades)
    outcome = coordinator.evaluate_snapshot(snapshot)
    assert len(outcome.incidents) > 0
    assert isinstance(outcome.execution_tuning_pack, ExecutionAlphaTuningPack)


def test_no_incidents_on_healthy_snapshot():
    engine = IncidentEngine(Thresholds())
    trades = [_make_trade() for _ in range(15)]
    snapshot = _make_snapshot(trades=trades)
    incidents, actions = engine.evaluate(snapshot)
    assert len(incidents) == 0
    assert actions.stop_trading is False
