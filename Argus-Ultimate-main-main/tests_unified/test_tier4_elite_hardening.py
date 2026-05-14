from __future__ import annotations

import json
from pathlib import Path

from argus_live.config.quarantine import ConfigQuarantineStore
from argus_live.ledger.trade_ledger import LedgerFillRecord, TradeLedger
from argus_live.proving.comparative import compare_reviews
from argus_live.proving.runner import run_proving_review
from argus_live.replay.replay_audit import ReplayAudit, ReplayAuditStore
from argus_live.simulation.batch_runner import ScenarioBatchRunner
from argus_live.simulation.regression_library import build_regression_library


def _append_fill(ledger: TradeLedger, run_id: str, config_hash: str) -> None:
    ledger.append_fill(
        LedgerFillRecord(
            intent_id='i1',
            symbol='BTC/AUD',
            side='buy',
            quantity=1.0,
            price=100.1,
            ts='2026-04-02T00:00:00+00:00',
            strategy_id='s1',
            venue='kraken',
            requested_qty=1.0,
            approved_qty=1.0,
            fill_qty=1.0,
            expected_price=100.0,
            fill_price=100.1,
            fees=0.1,
            latency_ms=10.0,
            reject_reason='',
            slippage_bps=10.0,
            execution_alpha_bps=-10.0,
            adverse_price_move_bps=1.5,
            venue_order_id='v1',
            parent_intent_id='p1',
            child_order_id='c1',
            status='ATTRIBUTED',
            gross_pnl=1.0,
            realized_pnl=0.8,
            unrealized_pnl=0.0,
            net_pnl_value=0.7,
            manifest_hash=run_id,
            run_id=run_id,
            config_hash=config_hash,
            ladder_stage='PAPER',
        )
    )


def test_runner_quarantines_bad_config(tmp_path: Path) -> None:
    ledger = TradeLedger(tmp_path / 'fills.jsonl')
    _append_fill(ledger, 'run-a', 'cfg-a')
    (tmp_path / 'incidents.jsonl').write_text(json.dumps({'run_id': 'run-a', 'severity': 'CRITICAL'}) + '\n', encoding='utf-8')
    replay = ReplayAuditStore(tmp_path / 'replay.jsonl')
    replay.append(ReplayAudit(ts='2026-04-02T00:00:00+00:00', run_id='run-a', status='FAIL', mismatch_count=1, notes='bad'))
    review = run_proving_review(
        run_id='run-a',
        trade_ledger=ledger,
        incidents_path=tmp_path / 'incidents.jsonl',
        replay_audit_path=tmp_path / 'replay.jsonl',
        report_path=tmp_path / 'review.json',
        config_quarantine_path=tmp_path / 'quarantine.jsonl',
    )
    assert review.config_quarantined is True
    assert review.rollback_tag == 'AUTO_ROLLBACK_RECOMMENDED'
    assert ConfigQuarantineStore(tmp_path / 'quarantine.jsonl').is_quarantined('cfg-a') is True


def test_comparative_review_summary(tmp_path: Path) -> None:
    base = tmp_path / 'base.json'
    cand = tmp_path / 'cand.json'
    base.write_text(json.dumps({'config_hash': 'a', 'promotion_decision': 'HOLD', 'net_pnl': 1, 'execution_alpha_bps': 1, 'max_drawdown_pct': 2, 'regression_overall_score': 80, 'replay_mismatch_count': 0}), encoding='utf-8')
    cand.write_text(json.dumps({'config_hash': 'b', 'promotion_decision': 'GO', 'net_pnl': 3, 'execution_alpha_bps': 4, 'max_drawdown_pct': 1, 'regression_overall_score': 92, 'replay_mismatch_count': 0}), encoding='utf-8')
    base_cfg = tmp_path / 'base_cfg.json'
    cand_cfg = tmp_path / 'cand_cfg.json'
    base_cfg.write_text(json.dumps({'config_hash': 'a', 'config': {'x': 1, 'y': 2}}), encoding='utf-8')
    cand_cfg.write_text(json.dumps({'config_hash': 'b', 'config': {'x': 1, 'y': 3}}), encoding='utf-8')
    summary = compare_reviews(
        baseline_report_path=base,
        candidate_report_path=cand,
        baseline_config_path=base_cfg,
        candidate_config_path=cand_cfg,
    )
    assert summary.decision_changed is True
    assert summary.config_change_count == 1
    assert summary.net_pnl_delta == 2.0


def test_regression_library_and_batch_scores(tmp_path: Path) -> None:
    scenarios = build_regression_library()
    names = {s.name for s in scenarios}
    assert 'venue_desync' in names
    assert 'adverse_selection_spiral' in names
    batch = ScenarioBatchRunner(tmp_path).run(batch_name='elite-batch', scenarios=scenarios[:2])
    assert len(batch.scenario_results) == 2
    first = batch.scenario_results[0]
    assert hasattr(first, 'governance_response_score')
    assert hasattr(first, 'adverse_selection_damage_bps')
