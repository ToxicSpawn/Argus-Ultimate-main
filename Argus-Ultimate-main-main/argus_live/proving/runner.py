from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from statistics import mean

from argus_live.config.quarantine import ConfigQuarantineStore
from argus_live.proving.comparative import compare_reviews, write_comparative_summary
from argus_live.proving.day_review import DayReview, build_day_review
from argus_live.proving.report_builder import write_review_report
from argus_live.replay.replay_audit import ReplayAuditStore
from argus_live.risk.capital_scaling import derive_capital_scaling_policy


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _load_incidents(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    if p.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        try:
            with sqlite3.connect(p) as conn:
                cursor = conn.execute(
                    "SELECT incident_id, severity, class, subsystem, title, summary, run_id FROM incidents ORDER BY ts_open ASC"
                )
                cols = [d[0] for d in cursor.description]
                return [dict(zip(cols, row)) for row in cursor.fetchall()]
        except sqlite3.Error:
            return []
    rows: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_regression_summary(path: str | Path, run_id: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if isinstance(data, dict) and str(data.get("batch_name", "")) == f"{run_id}-regression":
        return data
    return {}


def run_proving_review(*, run_id: str, trade_ledger, incidents_path: str | Path, replay_audit_path: str | Path, report_path: str | Path, regression_summary_path: str | Path | None = None, baseline_report_path: str | Path | None = None, baseline_config_path: str | Path | None = None, candidate_config_path: str | Path | None = None, config_quarantine_path: str | Path | None = None) -> DayReview:
    fills = [f for f in trade_ledger.load_recent_fills(limit=500) if str(getattr(f, 'run_id', '')) in ('', str(run_id))]
    incidents = [i for i in _load_incidents(incidents_path) if str(i.get('run_id', '')) in ('', str(run_id))]
    latest_audit = ReplayAuditStore(replay_audit_path).latest_for_run(run_id) or ReplayAuditStore(replay_audit_path).latest()

    net_pnl = sum(float(getattr(f, 'net_pnl', 0.0)) for f in fills)
    execution_alpha = _safe_mean([float(getattr(f, 'execution_alpha_bps', 0.0)) for f in fills])
    slippage_tail = max([abs(float(getattr(f, 'slippage_bps', 0.0))) for f in fills], default=0.0)
    rejects = sum(1 for f in fills if int(getattr(f, 'reject_flag', 0)) == 1)
    reject_rate = 0.0 if not fills else 100.0 * rejects / len(fills)
    config_hash = str(getattr(fills[-1], 'config_hash', '') or getattr(fills[-1], 'manifest_hash', run_id)) if fills else str(run_id)
    ladder_stage = str(getattr(fills[-1], 'ladder_stage', 'PAPER')) if fills else 'PAPER'
    equity_proxy = max(1.0, sum(float(getattr(f, 'approved_qty', 0.0) or getattr(f, 'quantity', 0.0)) * float(getattr(f, 'limit_price', 0.0) or getattr(f, 'price', 0.0)) for f in fills))

    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for f in fills:
        cumulative += float(getattr(f, 'net_pnl', 0.0))
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_drawdown = max(max_drawdown, dd)
    base = max(abs(peak), 1.0)
    max_drawdown_pct = 100.0 * max_drawdown / base

    critical_incidents = sum(1 for i in incidents if i.get('severity') == 'CRITICAL')
    replay_ok = bool(latest_audit and latest_audit.status == 'OK')
    replay_mismatch_count = int(getattr(latest_audit, 'mismatch_count', 0) or 0)

    regression = _load_regression_summary(regression_summary_path, run_id) if regression_summary_path is not None else {}
    capital_policy = derive_capital_scaling_policy(
        equity=equity_proxy,
        ladder_stage=ladder_stage,
        replay_ok=replay_ok,
        execution_alpha_bps=execution_alpha,
        critical_incident_count=critical_incidents,
        regression_overall_score=float(regression.get('overall_score', 0.0) or 0.0),
    )

    comparative_summary_path = ""
    comparative_score_delta = 0.0
    config_change_count = 0

    review = build_day_review(
        run_id=run_id,
        config_hash=config_hash,
        trade_count=len(fills),
        net_pnl=net_pnl,
        execution_alpha_bps=execution_alpha,
        max_drawdown_pct=max_drawdown_pct,
        reject_rate_pct=reject_rate,
        slippage_tail_bps=slippage_tail,
        replay_ok=replay_ok,
        replay_mismatch_count=replay_mismatch_count,
        critical_incident_count=critical_incidents,
        regression_overall_score=float(regression.get("overall_score", 0.0) or 0.0),
        regression_pass_rate=float(regression.get("pass_rate", 0.0) or 0.0),
        regression_summary_path=str(regression_summary_path or ""),
        ladder_stage=ladder_stage,
        scale_readiness_score=capital_policy.scale_readiness_score,
        scale_blocked=capital_policy.scale_blocked,
        capital_throttled=capital_policy.throttled,
        throttle_reason=capital_policy.throttle_reason,
        max_safe_aum=capital_policy.max_safe_aum,
    )

    write_review_report(review, report_path)

    if baseline_report_path is not None and Path(baseline_report_path).exists():
        comparative_summary = compare_reviews(
            baseline_report_path=baseline_report_path,
            candidate_report_path=report_path,
            baseline_config_path=baseline_config_path,
            candidate_config_path=candidate_config_path,
        )
        comparative_summary_path = write_comparative_summary(
            comparative_summary,
            Path(report_path).with_name(Path(report_path).stem + '_comparative.json'),
        )
        comparative_score_delta = comparative_summary.regression_score_delta
        config_change_count = comparative_summary.config_change_count
        review = replace(
            review,
            comparative_summary_path=comparative_summary_path,
            comparative_score_delta=comparative_score_delta,
            config_change_count=config_change_count,
        )
        write_review_report(review, report_path)

    if config_quarantine_path is not None:
        quarantine = ConfigQuarantineStore(config_quarantine_path)
        should_quarantine = review.promotion_decision == 'NO_GO' or not review.replay_ok or review.regression_overall_score < 70.0
        if should_quarantine:
            record = quarantine.quarantine(
                run_id=run_id,
                config_hash=review.config_hash,
                decision=review.promotion_decision,
                reasons=review.active_blockers or review.reasons,
            )
            review = replace(review, config_quarantined=True, rollback_tag=record.rollback_tag)
            write_review_report(review, report_path)
        elif quarantine.is_quarantined(review.config_hash):
            review = replace(review, config_quarantined=True, rollback_tag='CONFIG_ALREADY_QUARANTINED')
            write_review_report(review, report_path)

    return review
