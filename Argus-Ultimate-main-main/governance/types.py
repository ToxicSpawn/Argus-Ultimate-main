"""Governance types — all dataclasses, enums, configs, and utility functions."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence


# -----------------------------
# Utilities
# -----------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _recent_trade_dicts(trades: Sequence["TradeRecord"], limit: int = 20) -> List[Dict[str, Any]]:
    return [asdict(t) for t in list(trades)[-limit:]]


def _rolling_avg(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return mean(values)


def _pct(part: int, total: int) -> float:
    return 0.0 if total <= 0 else 100.0 * part / total


# -----------------------------
# Enums
# -----------------------------


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    MAJOR = "MAJOR"
    CRITICAL = "CRITICAL"


class IncidentStatus(str, Enum):
    OPEN = "OPEN"
    ACKED = "ACKED"
    MITIGATED = "MITIGATED"
    RESOLVED = "RESOLVED"
    POSTMORTEM_REQUIRED = "POSTMORTEM_REQUIRED"


class IncidentClass(str, Enum):
    EXECUTION = "EXECUTION"
    STRATEGY = "STRATEGY"
    PORTFOLIO = "PORTFOLIO"
    LEARNING = "LEARNING"
    RISK = "RISK"
    RUNTIME = "RUNTIME"


# -----------------------------
# Config dataclasses
# -----------------------------


@dataclass(frozen=True)
class Thresholds:
    rolling_slippage_warn_bps: float = 12.0
    rolling_slippage_major_bps: float = 20.0
    worst_trade_slippage_critical_bps: float = 45.0
    rolling_exec_alpha_major_bps: float = -2.0
    rolling_exec_alpha_critical_bps: float = -6.0
    reject_rate_warn_pct: float = 4.0
    reject_rate_major_pct: float = 8.0
    venue_exec_alpha_major_bps: float = -3.0
    concentration_major_pct: float = 35.0
    concentration_critical_pct: float = 50.0
    replay_mismatch_critical_count: int = 1
    metrics_lag_major_seconds: float = 30.0
    drawdown_major_pct: float = 2.0
    drawdown_critical_pct: float = 3.0


@dataclass(frozen=True)
class AggressionConfig:
    min_edge_to_cross_bps: float = 6.0
    max_cross_spread_bps: float = 4.0
    urgency_multiplier: float = 1.15
    volatility_penalty: float = 0.80
    imbalance_boost: float = 0.50
    adverse_selection_penalty: float = 1.25


@dataclass(frozen=True)
class SlicingConfig:
    base_slice_notional: float = 250.0
    min_slice_notional: float = 50.0
    max_slice_pct_top_book: float = 0.12
    slice_growth_factor: float = 1.10
    slice_decay_on_reject: float = 0.80
    slice_decay_on_slippage: float = 0.70
    max_child_orders: int = 8


@dataclass(frozen=True)
class RoutingConfig:
    maker_min_fill_prob: float = 0.62
    maker_max_wait_ms: int = 1200
    taker_min_edge_bps: float = 8.0
    queue_position_threshold: float = 0.55
    drift_escape_threshold: float = 3.0
    maker_retry_limit: int = 2


@dataclass(frozen=True)
class AbandonConfig:
    abandon_wait_ms: int = 1800
    abandon_on_edge_decay_bps: float = 2.5
    abandon_on_spread_widen_bps: float = 1.5
    abandon_on_vol_spike: float = 1.8
    cancel_if_fill_prob_below: float = 0.35


@dataclass(frozen=True)
class ExecutionAlphaConfig:
    aggression: AggressionConfig = AggressionConfig()
    slicing: SlicingConfig = SlicingConfig()
    routing: RoutingConfig = RoutingConfig()
    abandon: AbandonConfig = AbandonConfig()


# -----------------------------
# Record dataclasses
# -----------------------------


@dataclass
class TradeRecord:
    ts: str
    strategy_id: str
    symbol: str
    venue: str
    side: str
    qty: float
    expected_price: float
    fill_price: float
    fees: float
    gross_pnl: float
    net_pnl: float
    slippage_bps: float
    execution_alpha_bps: float
    maker_flag: int
    partial_fill_flag: int
    reject_flag: int
    latency_ms: float
    ladder_stage: str
    regime: Optional[str] = None


@dataclass
class PositionRecord:
    ts: str
    symbol: str
    strategy_id: str
    notional: float
    exposure_pct: float


@dataclass
class ReplayAuditRecord:
    ts: str
    run_id: str
    status: str
    mismatch_count: int
    notes: str = ""


@dataclass
class RuntimeSnapshot:
    run_id: str
    ladder_stage: str
    recent_trades: List[TradeRecord]
    recent_positions: List[PositionRecord]
    latest_replay_audit: Optional[ReplayAuditRecord]
    metrics_lag_seconds: float
    strategy_drawdown_pct: Dict[str, float]


@dataclass
class Incident:
    incident_id: str
    ts_open: str
    ts_last_update: str
    status: str
    severity: str
    incident_class: str
    subsystem: str
    entity_type: Optional[str]
    entity_id: Optional[str]
    title: str
    summary: str
    trigger_metric: Optional[str]
    trigger_value: Optional[float]
    threshold_value: Optional[float]
    ladder_stage: str
    recommended_action: str
    auto_action_taken: str
    operator_action_required: int
    evidence_blob: str
    run_id: str


@dataclass
class ActionSet:
    reduce_strategy_weight: Dict[str, float]
    degrade_venue_score: Dict[str, float]
    pause_symbols: List[str]
    pause_venues: List[str]
    execution_overrides: Dict[str, Any]
    block_promotions: bool
    stop_trading: bool

    @staticmethod
    def empty() -> "ActionSet":
        return ActionSet(
            reduce_strategy_weight={},
            degrade_venue_score={},
            pause_symbols=[],
            pause_venues=[],
            execution_overrides={},
            block_promotions=False,
            stop_trading=False,
        )

    def merge(self, other: "ActionSet") -> "ActionSet":
        merged = ActionSet.empty()
        merged.reduce_strategy_weight = {**self.reduce_strategy_weight, **other.reduce_strategy_weight}
        merged.degrade_venue_score = {**self.degrade_venue_score, **other.degrade_venue_score}
        merged.pause_symbols = sorted(set(self.pause_symbols + other.pause_symbols))
        merged.pause_venues = sorted(set(self.pause_venues + other.pause_venues))
        merged.execution_overrides = {**self.execution_overrides, **other.execution_overrides}
        merged.block_promotions = self.block_promotions or other.block_promotions
        merged.stop_trading = self.stop_trading or other.stop_trading
        return merged


# -----------------------------
# Execution context / decision
# -----------------------------


@dataclass
class ExecutionContext:
    strategy_id: str
    symbol: str
    venue: str
    expected_edge_bps: float
    spread_bps: float
    short_horizon_drift_bps: float
    volatility_score: float
    imbalance_score: float
    adverse_selection_score: float
    fill_probability: float
    queue_position_score: float
    urgency_score: float
    top_book_notional: float
    remaining_notional: float
    maker_retries_used: int
    elapsed_wait_ms: int
    current_edge_decay_bps: float
    spread_widen_bps: float


@dataclass
class ExecutionDecision:
    mode: str
    aggression_score: float
    slice_notional: float
    cancel: bool
    reason: str


@dataclass
class GovernanceOutcome:
    incidents: List[Incident]
    actions: ActionSet
    execution_tuning_pack: Any  # ExecutionAlphaTuningPack (avoid circular import)
