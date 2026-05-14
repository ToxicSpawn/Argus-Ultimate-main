from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstitutionDecision:
    allowed: bool
    reason: str
    gross_exposure_after: float
    symbol_exposure_after: float
    cluster_exposure_after: float


def evaluate_constitution(*, order_notional: float, symbol_notional_after: float, cluster_notional_after: float, gross_notional_after: float, equity: float, max_gross_exposure_pct: float, max_single_symbol_exposure_pct: float, max_cluster_exposure_pct: float) -> ConstitutionDecision:
    if equity <= 0:
        return ConstitutionDecision(False, "non-positive equity", gross_notional_after, symbol_notional_after, cluster_notional_after)
    gross_pct = gross_notional_after / equity
    symbol_pct = symbol_notional_after / equity
    cluster_pct = cluster_notional_after / equity
    if gross_pct > max_gross_exposure_pct:
        return ConstitutionDecision(False, "gross exposure exceeds constitution", gross_notional_after, symbol_notional_after, cluster_notional_after)
    if symbol_pct > max_single_symbol_exposure_pct:
        return ConstitutionDecision(False, "single-symbol exposure exceeds constitution", gross_notional_after, symbol_notional_after, cluster_notional_after)
    if cluster_pct > max_cluster_exposure_pct:
        return ConstitutionDecision(False, "cluster exposure exceeds constitution", gross_notional_after, symbol_notional_after, cluster_notional_after)
    return ConstitutionDecision(True, "constitution ok", gross_notional_after, symbol_notional_after, cluster_notional_after)
