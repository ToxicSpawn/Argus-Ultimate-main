from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from argus_live.promotion.ladder import LadderTransitionDecision, evaluate_ladder_transition
from argus_live.promotion.promotion_bundle import PromotionBundle


@dataclass(frozen=True)
class PromotionGateDecision:
    decision: str
    reasons: list[str]
    blockers: list[str]
    ladder_transition: LadderTransitionDecision | None = None
    scale_readiness_score: float = 0.0


def promotion_allowed(bundle: PromotionBundle) -> bool:
    return bool(
        bundle.replay_passed
        and bundle.signature
        and bundle.approved_by
        and bundle.walk_forward_score > 0
        and bundle.stress_score > 0
    )


def evaluate_promotion_gate(
    *,
    net_pnl: float,
    execution_alpha_bps: float,
    max_drawdown_pct: float,
    replay_ok: bool,
    critical_incident_count: int,
    reject_rate_pct: float,
    slippage_tail_bps: float,
    regression_overall_score: float | None = None,
    regression_pass_rate: float | None = None,
    current_stage: str = "PAPER",
    scale_readiness_score: float | None = None,
) -> PromotionGateDecision:
    reasons: list[str] = []
    stage = str(current_stage or 'PAPER').upper()
    readiness = float(scale_readiness_score if scale_readiness_score is not None else 100.0)
    if not replay_ok:
        reasons.append("replay mismatch present")
    if critical_incident_count > 0:
        reasons.append("critical incidents present")
    if execution_alpha_bps < 0:
        reasons.append("execution alpha negative")
    if max_drawdown_pct > 3.0:
        reasons.append("drawdown exceeds 3%")
    if reject_rate_pct > 8.0:
        reasons.append("reject rate exceeds 8%")
    if slippage_tail_bps > 45.0:
        reasons.append("slippage tail exceeds 45bps")
    if regression_overall_score is not None and regression_overall_score < 70.0:
        reasons.append("regression overall score below 70")
    if regression_pass_rate is not None and regression_pass_rate < 100.0:
        reasons.append("regression suite did not fully pass")
    if stage in {"LIVE_SAFE", "SCALE"} and readiness < 85.0:
        reasons.append("scale readiness below 85")
    if reasons:
        ladder = evaluate_ladder_transition(current_stage=stage, promotion_decision="NO_GO", replay_ok=replay_ok, critical_incident_count=critical_incident_count, regression_pass_rate=float(regression_pass_rate or 0.0), scale_readiness_score=readiness, active_blockers=reasons)
        return PromotionGateDecision("NO_GO", reasons, reasons, ladder, readiness)

    hold_reasons: list[str] = []
    if net_pnl <= 0:
        hold_reasons.append("net pnl non-positive")
    if max_drawdown_pct > 2.0:
        hold_reasons.append("drawdown above proving comfort band")
    if reject_rate_pct > 4.0:
        hold_reasons.append("reject rate elevated")
    if slippage_tail_bps > 25.0:
        hold_reasons.append("slippage tail elevated")
    if regression_overall_score is not None and regression_overall_score < 85.0:
        hold_reasons.append("regression score below elite band")
    if stage in {"LIVE_SAFE", "SCALE"} and readiness < 92.0:
        hold_reasons.append("scale readiness below elite band")
    if hold_reasons:
        ladder = evaluate_ladder_transition(current_stage=stage, promotion_decision="HOLD", replay_ok=replay_ok, critical_incident_count=critical_incident_count, regression_pass_rate=float(regression_pass_rate or 0.0), scale_readiness_score=readiness, active_blockers=hold_reasons)
        return PromotionGateDecision("HOLD", hold_reasons, hold_reasons, ladder, readiness)

    reasons = ["all proving criteria passed"]
    ladder = evaluate_ladder_transition(current_stage=stage, promotion_decision="GO", replay_ok=replay_ok, critical_incident_count=critical_incident_count, regression_pass_rate=float(regression_pass_rate or 100.0), scale_readiness_score=readiness, active_blockers=[])
    return PromotionGateDecision("GO", reasons, [], ladder, readiness)


def load_approved_strategies(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def assert_strategy_allowed(
    strategy_id: str,
    manifest_hash: str,
    approved: dict[str, str] | None = None,
    manifest_path: Path | None = None,
) -> None:
    if approved is None:
        if manifest_path is None:
            raise RuntimeError("No approved-strategy manifest provided")
        approved = load_approved_strategies(manifest_path)

    if strategy_id not in approved:
        raise RuntimeError(f"Strategy '{strategy_id}' is not in the approved manifest")
    if approved[strategy_id] != manifest_hash:
        raise RuntimeError(
            f"Strategy '{strategy_id}' manifest hash mismatch: expected {approved[strategy_id]}, got {manifest_hash}"
        )
