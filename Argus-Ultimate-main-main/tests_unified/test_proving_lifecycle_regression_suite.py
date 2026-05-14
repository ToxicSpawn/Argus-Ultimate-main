from pathlib import Path

from argus_live.ledger.trade_ledger import LedgerFillRecord, TradeLedger
from argus_live.proving.session_lifecycle import AutoProvingLifecycle
from argus_live.replay.replay_audit import ReplayAudit, ReplayAuditStore


def test_proving_lifecycle_runs_regression_suite(tmp_path: Path):
    ledger = TradeLedger(tmp_path / "fills.jsonl")
    ledger.append_fill(LedgerFillRecord(
        intent_id="i1", symbol="BTC/AUD", side="buy", quantity=0.01, price=100000.0,
        run_id="run-1", manifest_hash="run-1", fees=-5.0, execution_alpha_bps=1.0, slippage_bps=2.0
    ))
    ReplayAuditStore(tmp_path / "replay.jsonl").append(ReplayAudit(
        ts="2026-04-02T00:00:00+00:00", run_id="run-1", status="OK", mismatch_count=0, notes=""
    ))
    lifecycle = AutoProvingLifecycle(
        incidents_path=tmp_path / "incidents.db",
        replay_audit_path=tmp_path / "replay.jsonl",
        report_path=tmp_path / "review.json",
        operator_summary_path=tmp_path / "operator.json",
        regression_artifacts_root=tmp_path / "regression",
        enable_regression_suite=True,
    )
    result = lifecycle.refresh(run_id="run-1", trade_ledger=ledger)
    assert result.regression_summary_path
    assert Path(result.regression_summary_path).exists()
    assert result.review.regression_summary_path == result.regression_summary_path
