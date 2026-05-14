from __future__ import annotations

import json
from pathlib import Path

from argus_live.config.config_lock import build_config_diff, lock_config, write_config_artifacts
from argus_live.execution.entrypoint import ExecutionEntrypoint
from argus_live.ledger.trade_ledger import LedgerFillRecord


class _RuntimeStub:
    def __init__(self):
        class _Journal:
            def __init__(self):
                self.events = []
            def append(self, event):
                self.events.append(event)
        self.journal = _Journal()
        self.governance = object()
        self.strict_governance = False
        self.trade_ledger = object()
    def process_intent(self, intent, **kwargs):
        self.last_kwargs = kwargs


def test_config_lock_artifacts(tmp_path: Path):
    cfg = {'b': 2, 'a': 1}
    prev = {'a': 1}
    locked = lock_config(cfg)
    config_path, diff_path = write_config_artifacts(artifact_dir=tmp_path, run_id='run-1', config_lock=locked, previous_config=prev)
    assert config_path.exists()
    assert diff_path.exists()
    payload = json.loads(config_path.read_text())
    diff = json.loads(diff_path.read_text())
    assert payload['config_hash'].startswith('sha256:')
    assert diff['diff']['changed']['b']['new'] == 2


def test_execution_entrypoint_requires_governance_when_strict():
    runtime = _RuntimeStub()
    runtime.governance = None
    try:
        ExecutionEntrypoint(runtime=runtime, manifest_hash='run-1', strict_governance=True)
    except RuntimeError as exc:
        assert 'Governance coordinator required' in str(exc)
    else:
        raise AssertionError('expected RuntimeError')


def test_ledger_net_pnl_prefers_attribution_truth():
    record = LedgerFillRecord(
        intent_id='i1', symbol='BTC/AUD', side='buy', quantity=1.0, price=100.0,
        realized_pnl=5.0, unrealized_pnl=1.5, fees=0.5, net_pnl_value=0.0,
    ).normalized()
    assert abs(record.net_pnl - 6.0) < 1e-9
