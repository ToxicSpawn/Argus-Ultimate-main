from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapitalScalingPolicy:
    ladder_stage: str
    max_safe_aum: float
    max_book_take_ratio: float
    max_gross_exposure_pct: float
    max_single_symbol_exposure_pct: float
    max_cluster_exposure_pct: float
    throttled: bool
    blocked: bool
    blockers: list[str]
    scale_readiness_score: float
    scale_blocked: bool
    throttle_reason: str = ""


_STAGE_CAPS = {
    "PAPER": (0.20, 0.25, 0.08, 0.15, 25_000.0),
    "SHADOW": (0.18, 0.22, 0.07, 0.14, 25_000.0),
    "LIVE_SAFE_MICRO": (0.12, 0.18, 0.05, 0.10, 10_000.0),
    "LIVE_SAFE": (0.08, 0.15, 0.04, 0.08, 50_000.0),
    "SCALE": (0.05, 0.12, 0.03, 0.06, 100_000.0),
}


def derive_capital_scaling_policy(
    *,
    equity: float,
    ladder_stage: str = "PAPER",
    replay_ok: bool = True,
    execution_alpha_bps: float = 0.0,
    critical_incident_count: int = 0,
    regression_overall_score: float | None = None,
    max_safe_aum: float | None = None,
) -> CapitalScalingPolicy:
    stage = str(ladder_stage or "PAPER").upper()
    max_book_take_ratio, max_gross, max_symbol, max_cluster, default_max_aum = _STAGE_CAPS.get(stage, _STAGE_CAPS["PAPER"])
    stage_max_safe_aum = float(max_safe_aum if max_safe_aum is not None else default_max_aum)
    blockers: list[str] = []
    throttled = False
    throttle_reason = ""

    if equity > stage_max_safe_aum:
        blockers.append("equity exceeds max_safe_aum")
    if not replay_ok:
        blockers.append("replay mismatch present")
    if critical_incident_count > 0:
        blockers.append("critical incidents present")
    if regression_overall_score is not None and regression_overall_score < 70.0:
        blockers.append("regression overall score below 70")

    if execution_alpha_bps < 0:
        throttled = True
        throttle_reason = "execution alpha negative"
        max_book_take_ratio *= 0.75
        max_gross *= 0.85
        max_symbol *= 0.85
        max_cluster *= 0.85
    elif regression_overall_score is not None and regression_overall_score < 85.0:
        throttled = True
        throttle_reason = "regression score below elite band"
        max_book_take_ratio *= 0.85
        max_gross *= 0.90
        max_symbol *= 0.90
        max_cluster *= 0.90

    scale_blocked = stage in {"LIVE_SAFE", "SCALE"} and (bool(blockers) or throttled)

    readiness = 100.0
    if blockers:
        readiness -= min(60.0, 20.0 * len(blockers))
    if throttled:
        readiness -= 20.0
    if execution_alpha_bps < 0:
        readiness -= min(20.0, abs(execution_alpha_bps) * 2.0)
    if regression_overall_score is not None:
        readiness -= max(0.0, 90.0 - float(regression_overall_score)) * 0.5
    scale_readiness_score = max(0.0, min(100.0, readiness))

    blocked = bool(blockers)
    return CapitalScalingPolicy(
        ladder_stage=stage,
        max_safe_aum=stage_max_safe_aum,
        max_book_take_ratio=max_book_take_ratio,
        max_gross_exposure_pct=max_gross,
        max_single_symbol_exposure_pct=max_symbol,
        max_cluster_exposure_pct=max_cluster,
        throttled=throttled,
        blocked=blocked,
        blockers=blockers,
        scale_readiness_score=scale_readiness_score,
        scale_blocked=scale_blocked,
        throttle_reason=throttle_reason,
    )
