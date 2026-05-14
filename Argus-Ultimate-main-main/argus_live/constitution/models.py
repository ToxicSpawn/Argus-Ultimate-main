from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    max_gross_exposure_pct: float
    max_single_symbol_exposure_pct: float
    max_cluster_exposure_pct: float
    max_daily_loss_pct: float
    max_drawdown_pct: float
    max_open_intents: int
    max_consecutive_losses: int


@dataclass(frozen=True)
class SafetyRules:
    block_new_risk_on_data_degradation: bool
    block_new_risk_on_reconciliation_staleness: bool
    require_signed_promotion_bundle: bool
    block_research_modules_from_live: bool
    require_soak_evidence: bool
    require_operator_ack_after_reconciliation_freeze: bool


@dataclass(frozen=True)
class VenueRules:
    allow_cross_venue_live: bool
    allow_market_orders: bool
    allow_only_approved_symbols: bool


@dataclass(frozen=True)
class Constitution:
    version: int
    profile: str
    limits: Limits
    safety: SafetyRules
    venues: VenueRules
