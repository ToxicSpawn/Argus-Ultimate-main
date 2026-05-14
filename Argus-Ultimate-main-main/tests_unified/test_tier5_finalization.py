from __future__ import annotations

import json
from pathlib import Path

from argus_live.proving.control_surface import build_control_surface
from argus_live.proving.day_review import build_day_review
from argus_live.proving.session_lifecycle import AutoProvingLifecycle
from argus_live.risk.capital_scaling import derive_capital_scaling_policy


class _DummyLedger:
    def load_recent_fills(self, limit: int = 100):
        return []


def test_capital_scaling_blocks_live_safe_bootstrap_when_scale_not_ready():
    policy = derive_capital_scaling_policy(
        equity=60000.0,
        ladder_stage='LIVE_SAFE',
        replay_ok=True,
        execution_alpha_bps=-1.0,
        critical_incident_count=0,
        regression_overall_score=90.0,
    )
    assert policy.scale_blocked is True
    assert policy.scale_readiness_score < 85.0


def test_day_review_and_control_surface_include_scaling_fields():
    review = build_day_review(
        run_id='r1',
        config_hash='c1',
        trade_count=10,
        net_pnl=100.0,
        execution_alpha_bps=2.0,
        max_drawdown_pct=1.0,
        reject_rate_pct=0.0,
        slippage_tail_bps=5.0,
        replay_ok=True,
        replay_mismatch_count=0,
        critical_incident_count=0,
        regression_overall_score=95.0,
        regression_pass_rate=100.0,
        ladder_stage='LIVE_SAFE',
        scale_readiness_score=96.0,
        scale_blocked=False,
        capital_throttled=False,
        throttle_reason='',
        max_safe_aum=50000.0,
    )
    surface = build_control_surface(review)
    assert surface['capital_scaling_ready'] is True
    assert surface['ladder']['current'] == 'LIVE_SAFE'
    assert surface['scaling']['readiness_score'] == 96.0


def test_lifecycle_writes_control_surface(tmp_path: Path):
    report = tmp_path / 'review.json'
    lifecycle = AutoProvingLifecycle(
        incidents_path=tmp_path / 'incidents.db',
        replay_audit_path=tmp_path / 'replay.jsonl',
        report_path=report,
        control_surface_path=tmp_path / 'control_surface.json',
    )
    result = lifecycle.refresh(run_id='run-1', trade_ledger=_DummyLedger())
    payload = json.loads(Path(result.control_surface_path).read_text(encoding='utf-8'))
    assert 'status' in payload
    assert 'scaling' in payload
