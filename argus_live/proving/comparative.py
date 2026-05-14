from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


@dataclass(frozen=True)
class ComparativeReviewSummary:
    baseline_report_path: str
    candidate_report_path: str
    baseline_config_hash: str
    candidate_config_hash: str
    config_change_count: int
    baseline_decision: str
    candidate_decision: str
    decision_changed: bool
    net_pnl_delta: float
    execution_alpha_delta_bps: float
    drawdown_delta_pct: float
    regression_score_delta: float
    replay_delta: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_reviews(
    *,
    baseline_report_path: str | Path,
    candidate_report_path: str | Path,
    baseline_config_path: str | Path | None = None,
    candidate_config_path: str | Path | None = None,
) -> ComparativeReviewSummary:
    base = _load_json(baseline_report_path)
    cand = _load_json(candidate_report_path)
    base_cfg = _load_json(baseline_config_path) if baseline_config_path is not None else {}
    cand_cfg = _load_json(candidate_config_path) if candidate_config_path is not None else {}
    changed = 0
    if isinstance(base_cfg.get('config'), dict) and isinstance(cand_cfg.get('config'), dict):
        keys = set(base_cfg['config']) | set(cand_cfg['config'])
        changed = sum(1 for k in keys if base_cfg['config'].get(k) != cand_cfg['config'].get(k))
    return ComparativeReviewSummary(
        baseline_report_path=str(baseline_report_path),
        candidate_report_path=str(candidate_report_path),
        baseline_config_hash=str(base.get('config_hash', base_cfg.get('config_hash', ''))),
        candidate_config_hash=str(cand.get('config_hash', cand_cfg.get('config_hash', ''))),
        config_change_count=changed,
        baseline_decision=str(base.get('promotion_decision', 'UNKNOWN')),
        candidate_decision=str(cand.get('promotion_decision', 'UNKNOWN')),
        decision_changed=str(base.get('promotion_decision', 'UNKNOWN')) != str(cand.get('promotion_decision', 'UNKNOWN')),
        net_pnl_delta=float(cand.get('net_pnl', 0.0) or 0.0) - float(base.get('net_pnl', 0.0) or 0.0),
        execution_alpha_delta_bps=float(cand.get('execution_alpha_bps', 0.0) or 0.0) - float(base.get('execution_alpha_bps', 0.0) or 0.0),
        drawdown_delta_pct=float(cand.get('max_drawdown_pct', 0.0) or 0.0) - float(base.get('max_drawdown_pct', 0.0) or 0.0),
        regression_score_delta=float(cand.get('regression_overall_score', 0.0) or 0.0) - float(base.get('regression_overall_score', 0.0) or 0.0),
        replay_delta=int(cand.get('replay_mismatch_count', 0) or 0) - int(base.get('replay_mismatch_count', 0) or 0),
    )


def write_comparative_summary(summary: ComparativeReviewSummary, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding='utf-8')
    return str(p)
