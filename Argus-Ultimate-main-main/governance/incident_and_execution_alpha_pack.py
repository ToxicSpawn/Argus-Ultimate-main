from __future__ import annotations

import json
import math
import sqlite3
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ============================================================
# ARGUS Incident Engine + Execution Alpha Tuning Pack
# ------------------------------------------------------------
# Drop-in proving-phase skeleton designed to plug into an
# existing journal/metrics/SQLite pipeline.
#
# Goals:
# - Detect operational incidents deterministically
# - Take bounded, reversible auto-actions only
# - Improve execution quality without changing strategy logic
# - Persist evidence for postmortem and promotion decisions
# ============================================================


# -----------------------------
# Utilities
# -----------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# -----------------------------
# Enums / Dataclasses
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
# SQLite persistence
# -----------------------------


INCIDENTS_DDL = """
CREATE TABLE IF NOT EXISTS incidents (
  incident_id TEXT PRIMARY KEY,
  ts_open TEXT NOT NULL,
  ts_last_update TEXT NOT NULL,
  status TEXT NOT NULL,
  severity TEXT NOT NULL,
  class TEXT NOT NULL,
  subsystem TEXT NOT NULL,
  entity_type TEXT,
  entity_id TEXT,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  trigger_metric TEXT,
  trigger_value REAL,
  threshold_value REAL,
  ladder_stage TEXT,
  recommended_action TEXT,
  auto_action_taken TEXT,
  operator_action_required INTEGER DEFAULT 0,
  evidence_blob TEXT,
  run_id TEXT
);
"""


class IncidentRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def init_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(INCIDENTS_DDL)
            conn.commit()

    def insert_many(self, incidents: Sequence[Incident]) -> None:
        if not incidents:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO incidents (
                    incident_id, ts_open, ts_last_update, status, severity, class,
                    subsystem, entity_type, entity_id, title, summary, trigger_metric,
                    trigger_value, threshold_value, ladder_stage, recommended_action,
                    auto_action_taken, operator_action_required, evidence_blob, run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        i.incident_id,
                        i.ts_open,
                        i.ts_last_update,
                        i.status,
                        i.severity,
                        i.incident_class,
                        i.subsystem,
                        i.entity_type,
                        i.entity_id,
                        i.title,
                        i.summary,
                        i.trigger_metric,
                        i.trigger_value,
                        i.threshold_value,
                        i.ladder_stage,
                        i.recommended_action,
                        i.auto_action_taken,
                        i.operator_action_required,
                        i.evidence_blob,
                        i.run_id,
                    )
                    for i in incidents
                ],
            )
            conn.commit()


# -----------------------------
# Incident rule helpers
# -----------------------------


class IncidentFactory:
    @staticmethod
    def build(
        *,
        severity: Severity,
        incident_class: IncidentClass,
        subsystem: str,
        ladder_stage: str,
        run_id: str,
        title: str,
        summary: str,
        trigger_metric: Optional[str],
        trigger_value: Optional[float],
        threshold_value: Optional[float],
        recommended_action: str,
        auto_action_taken: str,
        operator_action_required: bool,
        evidence: Dict[str, Any],
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        status: IncidentStatus = IncidentStatus.OPEN,
    ) -> Incident:
        now = utc_now_iso()
        return Incident(
            incident_id=str(uuid.uuid4()),
            ts_open=now,
            ts_last_update=now,
            status=status.value,
            severity=severity.value,
            incident_class=incident_class.value,
            subsystem=subsystem,
            entity_type=entity_type,
            entity_id=entity_id,
            title=title,
            summary=summary,
            trigger_metric=trigger_metric,
            trigger_value=trigger_value,
            threshold_value=threshold_value,
            ladder_stage=ladder_stage,
            recommended_action=recommended_action,
            auto_action_taken=auto_action_taken,
            operator_action_required=1 if operator_action_required else 0,
            evidence_blob=json.dumps(evidence, default=str, sort_keys=True),
            run_id=run_id,
        )


def _recent_trade_dicts(trades: Sequence[TradeRecord], limit: int = 20) -> List[Dict[str, Any]]:
    return [asdict(t) for t in list(trades)[-limit:]]


def _rolling_avg(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return mean(values)


def _pct(part: int, total: int) -> float:
    return 0.0 if total <= 0 else 100.0 * part / total


# -----------------------------
# Incident Engine
# -----------------------------


class IncidentEngine:
    def __init__(self, thresholds: Thresholds) -> None:
        self.thresholds = thresholds

    def evaluate(self, snapshot: RuntimeSnapshot) -> Tuple[List[Incident], ActionSet]:
        incidents: List[Incident] = []
        actions = ActionSet.empty()

        for rule in (
            self._rule_replay_mismatch,
            self._rule_metrics_lag,
            self._rule_negative_execution_alpha,
            self._rule_slippage_spike,
            self._rule_reject_burst,
            self._rule_strategy_drawdown,
            self._rule_concentration,
            self._rule_venue_quality,
        ):
            new_incidents, new_actions = rule(snapshot)
            incidents.extend(new_incidents)
            actions = actions.merge(new_actions)

        return incidents, actions

    def _rule_replay_mismatch(self, snapshot: RuntimeSnapshot) -> Tuple[List[Incident], ActionSet]:
        audit = snapshot.latest_replay_audit
        if not audit or audit.mismatch_count < self.thresholds.replay_mismatch_critical_count:
            return [], ActionSet.empty()

        incident = IncidentFactory.build(
            severity=Severity.CRITICAL,
            incident_class=IncidentClass.RUNTIME,
            subsystem="replay_audit",
            ladder_stage=snapshot.ladder_stage,
            run_id=snapshot.run_id,
            title="Replay mismatch detected",
            summary="Replay mismatch is non-zero. Proving must stop until determinism is restored.",
            trigger_metric="replay.mismatch_count",
            trigger_value=float(audit.mismatch_count),
            threshold_value=float(self.thresholds.replay_mismatch_critical_count),
            recommended_action="Stop trading, block promotions, reconcile journal/ordering, open postmortem.",
            auto_action_taken="stop_trading=true, block_promotions=true",
            operator_action_required=True,
            evidence={"replay_audit": asdict(audit)},
            entity_type="runtime",
            entity_id="replay",
        )
        actions = ActionSet.empty()
        actions.stop_trading = True
        actions.block_promotions = True
        return [incident], actions

    def _rule_metrics_lag(self, snapshot: RuntimeSnapshot) -> Tuple[List[Incident], ActionSet]:
        if snapshot.metrics_lag_seconds <= self.thresholds.metrics_lag_major_seconds:
            return [], ActionSet.empty()

        incident = IncidentFactory.build(
            severity=Severity.MAJOR,
            incident_class=IncidentClass.RUNTIME,
            subsystem="telemetry",
            ladder_stage=snapshot.ladder_stage,
            run_id=snapshot.run_id,
            title="Metrics lag exceeded threshold",
            summary="Telemetry freshness degraded beyond proving threshold. Promotion decisions must be blocked.",
            trigger_metric="telemetry.metrics_lag_seconds",
            trigger_value=snapshot.metrics_lag_seconds,
            threshold_value=self.thresholds.metrics_lag_major_seconds,
            recommended_action="Block promotions and investigate DB lag / journal backlog.",
            auto_action_taken="block_promotions=true",
            operator_action_required=True,
            evidence={"metrics_lag_seconds": snapshot.metrics_lag_seconds},
            entity_type="runtime",
            entity_id="telemetry",
        )
        actions = ActionSet.empty()
        actions.block_promotions = True
        return [incident], actions

    def _rule_negative_execution_alpha(self, snapshot: RuntimeSnapshot) -> Tuple[List[Incident], ActionSet]:
        trades = snapshot.recent_trades[-15:]
        if not trades:
            return [], ActionSet.empty()

        avg_alpha = _rolling_avg([t.execution_alpha_bps for t in trades])
        if avg_alpha is None or avg_alpha >= self.thresholds.rolling_exec_alpha_major_bps:
            return [], ActionSet.empty()

        severity = Severity.CRITICAL if avg_alpha <= self.thresholds.rolling_exec_alpha_critical_bps else Severity.MAJOR
        auto_action = "raise taker threshold, reduce urgency, shrink slices"
        incident = IncidentFactory.build(
            severity=severity,
            incident_class=IncidentClass.EXECUTION,
            subsystem="execution_alpha",
            ladder_stage=snapshot.ladder_stage,
            run_id=snapshot.run_id,
            title="Execution alpha turned negative",
            summary="Rolling execution alpha is below threshold. The system is leaking edge through execution.",
            trigger_metric="execution_alpha.rolling_avg_bps_15",
            trigger_value=avg_alpha,
            threshold_value=self.thresholds.rolling_exec_alpha_major_bps,
            recommended_action="Clamp aggression, prefer cleaner fills, inspect venue/symbol cohorts.",
            auto_action_taken=auto_action,
            operator_action_required=True,
            evidence={"recent_trades": _recent_trade_dicts(trades)},
            entity_type="portfolio",
            entity_id="global_execution",
        )
        actions = ActionSet.empty()
        actions.execution_overrides = {
            "taker_min_edge_bps_multiplier": 1.15,
            "urgency_multiplier": 0.85,
            "max_slice_pct_top_book_multiplier": 0.8,
        }
        if severity == Severity.CRITICAL:
            actions.block_promotions = True
        return [incident], actions

    def _rule_slippage_spike(self, snapshot: RuntimeSnapshot) -> Tuple[List[Incident], ActionSet]:
        trades = snapshot.recent_trades[-15:]
        if not trades:
            return [], ActionSet.empty()

        rolling = _rolling_avg([t.slippage_bps for t in trades])
        worst = max((t.slippage_bps for t in trades), default=0.0)
        if rolling is None:
            return [], ActionSet.empty()
        if rolling <= self.thresholds.rolling_slippage_warn_bps and worst <= self.thresholds.worst_trade_slippage_critical_bps:
            return [], ActionSet.empty()

        if worst > self.thresholds.worst_trade_slippage_critical_bps:
            severity = Severity.CRITICAL
            threshold = self.thresholds.worst_trade_slippage_critical_bps
            trigger = worst
            metric = "slippage.worst_trade_bps_15"
        elif rolling > self.thresholds.rolling_slippage_major_bps:
            severity = Severity.MAJOR
            threshold = self.thresholds.rolling_slippage_major_bps
            trigger = rolling
            metric = "slippage.rolling_avg_bps_15"
        else:
            severity = Severity.WARNING
            threshold = self.thresholds.rolling_slippage_warn_bps
            trigger = rolling
            metric = "slippage.rolling_avg_bps_15"

        incident = IncidentFactory.build(
            severity=severity,
            incident_class=IncidentClass.EXECUTION,
            subsystem="slippage_monitor",
            ladder_stage=snapshot.ladder_stage,
            run_id=snapshot.run_id,
            title="Slippage spike detected",
            summary="Recent slippage breached the configured band. Execution quality is deteriorating.",
            trigger_metric=metric,
            trigger_value=trigger,
            threshold_value=threshold,
            recommended_action="Reduce slice size, lower aggression, inspect liquidity/venue concentration.",
            auto_action_taken="max_slice_pct_top_book reduced, slice_decay_on_slippage increased",
            operator_action_required=severity != Severity.WARNING,
            evidence={"recent_trades": _recent_trade_dicts(trades)},
            entity_type="portfolio",
            entity_id="global_execution",
        )
        actions = ActionSet.empty()
        actions.execution_overrides = {
            "max_slice_pct_top_book_multiplier": 0.7,
            "slice_decay_on_slippage_override": 0.55,
        }
        if severity == Severity.CRITICAL:
            actions.block_promotions = True
        return [incident], actions

    def _rule_reject_burst(self, snapshot: RuntimeSnapshot) -> Tuple[List[Incident], ActionSet]:
        trades = snapshot.recent_trades[-25:]
        if not trades:
            return [], ActionSet.empty()

        reject_rate = _pct(sum(t.reject_flag for t in trades), len(trades))
        if reject_rate < self.thresholds.reject_rate_warn_pct:
            return [], ActionSet.empty()

        severity = Severity.MAJOR if reject_rate >= self.thresholds.reject_rate_major_pct else Severity.WARNING
        incident = IncidentFactory.build(
            severity=severity,
            incident_class=IncidentClass.EXECUTION,
            subsystem="order_rejects",
            ladder_stage=snapshot.ladder_stage,
            run_id=snapshot.run_id,
            title="Reject burst detected",
            summary="Order reject rate exceeded the configured threshold over the recent execution window.",
            trigger_metric="reject_rate.recent_pct_25",
            trigger_value=reject_rate,
            threshold_value=self.thresholds.reject_rate_warn_pct,
            recommended_action="Reduce order cadence, inspect venue health, downgrade weak route if localized.",
            auto_action_taken="maker_retry_limit lowered, weak venue may be paused",
            operator_action_required=severity == Severity.MAJOR,
            evidence={"recent_trades": _recent_trade_dicts(trades)},
            entity_type="portfolio",
            entity_id="global_execution",
        )
        actions = ActionSet.empty()
        actions.execution_overrides = {"maker_retry_limit_override": 1}
        return [incident], actions

    def _rule_strategy_drawdown(self, snapshot: RuntimeSnapshot) -> Tuple[List[Incident], ActionSet]:
        incidents: List[Incident] = []
        actions = ActionSet.empty()

        for strategy_id, dd in snapshot.strategy_drawdown_pct.items():
            if dd < self.thresholds.drawdown_major_pct:
                continue
            severity = Severity.CRITICAL if dd >= self.thresholds.drawdown_critical_pct else Severity.MAJOR
            action_weight = 0.25 if severity == Severity.CRITICAL else 0.60
            incident = IncidentFactory.build(
                severity=severity,
                incident_class=IncidentClass.STRATEGY,
                subsystem="strategy_drawdown",
                ladder_stage=snapshot.ladder_stage,
                run_id=snapshot.run_id,
                title=f"Strategy drawdown breach: {strategy_id}",
                summary="Strategy drawdown exceeded its proving threshold and should be degraded.",
                trigger_metric="strategy.drawdown_pct",
                trigger_value=dd,
                threshold_value=self.thresholds.drawdown_major_pct,
                recommended_action="Reduce strategy allocation and review regime fit before re-enabling.",
                auto_action_taken=f"strategy_weight[{strategy_id}]={action_weight}",
                operator_action_required=True,
                evidence={"strategy_id": strategy_id, "drawdown_pct": dd},
                entity_type="strategy",
                entity_id=strategy_id,
            )
            incidents.append(incident)
            actions.reduce_strategy_weight[strategy_id] = action_weight
            if severity == Severity.CRITICAL:
                actions.block_promotions = True

        return incidents, actions

    def _rule_concentration(self, snapshot: RuntimeSnapshot) -> Tuple[List[Incident], ActionSet]:
        if not snapshot.recent_positions:
            return [], ActionSet.empty()

        worst = max(snapshot.recent_positions, key=lambda p: p.exposure_pct)
        if worst.exposure_pct < self.thresholds.concentration_major_pct:
            return [], ActionSet.empty()

        severity = Severity.CRITICAL if worst.exposure_pct >= self.thresholds.concentration_critical_pct else Severity.MAJOR
        incident = IncidentFactory.build(
            severity=severity,
            incident_class=IncidentClass.PORTFOLIO,
            subsystem="allocator",
            ladder_stage=snapshot.ladder_stage,
            run_id=snapshot.run_id,
            title=f"Concentration breach: {worst.symbol}",
            summary="Position concentration exceeded the allowed proving band.",
            trigger_metric="portfolio.max_exposure_pct",
            trigger_value=worst.exposure_pct,
            threshold_value=self.thresholds.concentration_major_pct,
            recommended_action="Clamp allocator and freeze further adds to concentrated symbol until normalized.",
            auto_action_taken=f"pause_symbol={worst.symbol}",
            operator_action_required=True,
            evidence={"position": asdict(worst)},
            entity_type="symbol",
            entity_id=worst.symbol,
        )
        actions = ActionSet.empty()
        actions.pause_symbols.append(worst.symbol)
        if severity == Severity.CRITICAL:
            actions.block_promotions = True
        return [incident], actions

    def _rule_venue_quality(self, snapshot: RuntimeSnapshot) -> Tuple[List[Incident], ActionSet]:
        incidents: List[Incident] = []
        actions = ActionSet.empty()

        grouped: Dict[str, List[TradeRecord]] = {}
        for t in snapshot.recent_trades[-30:]:
            grouped.setdefault(t.venue, []).append(t)

        for venue, trades in grouped.items():
            if len(trades) < 5:
                continue
            avg_alpha = mean(t.execution_alpha_bps for t in trades)
            if avg_alpha >= self.thresholds.venue_exec_alpha_major_bps:
                continue
            incident = IncidentFactory.build(
                severity=Severity.MAJOR,
                incident_class=IncidentClass.EXECUTION,
                subsystem="venue_router",
                ladder_stage=snapshot.ladder_stage,
                run_id=snapshot.run_id,
                title=f"Venue quality degraded: {venue}",
                summary="Venue-specific execution alpha fell below the acceptable band.",
                trigger_metric="venue.execution_alpha_bps",
                trigger_value=avg_alpha,
                threshold_value=self.thresholds.venue_exec_alpha_major_bps,
                recommended_action="Downweight route and inspect venue-specific slippage/reject patterns.",
                auto_action_taken=f"degrade_venue_score[{venue}]=0.5",
                operator_action_required=True,
                evidence={"venue": venue, "recent_trades": _recent_trade_dicts(trades, limit=10)},
                entity_type="venue",
                entity_id=venue,
            )
            incidents.append(incident)
            actions.degrade_venue_score[venue] = 0.5

        return incidents, actions


# -----------------------------
# Execution Alpha Tuning Pack
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


class ExecutionAlphaTuningPack:
    def __init__(self, config: ExecutionAlphaConfig) -> None:
        self.config = config

    def apply_overrides(self, overrides: Dict[str, Any]) -> "ExecutionAlphaTuningPack":
        cfg = self.config
        a = cfg.aggression
        s = cfg.slicing
        r = cfg.routing
        ab = cfg.abandon

        new_cfg = ExecutionAlphaConfig(
            aggression=AggressionConfig(
                min_edge_to_cross_bps=a.min_edge_to_cross_bps * overrides.get("taker_min_edge_bps_multiplier", 1.0),
                max_cross_spread_bps=a.max_cross_spread_bps,
                urgency_multiplier=overrides.get("urgency_multiplier", a.urgency_multiplier),
                volatility_penalty=a.volatility_penalty,
                imbalance_boost=a.imbalance_boost,
                adverse_selection_penalty=a.adverse_selection_penalty,
            ),
            slicing=SlicingConfig(
                base_slice_notional=s.base_slice_notional,
                min_slice_notional=s.min_slice_notional,
                max_slice_pct_top_book=s.max_slice_pct_top_book * overrides.get("max_slice_pct_top_book_multiplier", 1.0),
                slice_growth_factor=s.slice_growth_factor,
                slice_decay_on_reject=s.slice_decay_on_reject,
                slice_decay_on_slippage=overrides.get("slice_decay_on_slippage_override", s.slice_decay_on_slippage),
                max_child_orders=s.max_child_orders,
            ),
            routing=RoutingConfig(
                maker_min_fill_prob=r.maker_min_fill_prob,
                maker_max_wait_ms=r.maker_max_wait_ms,
                taker_min_edge_bps=r.taker_min_edge_bps * overrides.get("taker_min_edge_bps_multiplier", 1.0),
                queue_position_threshold=r.queue_position_threshold,
                drift_escape_threshold=r.drift_escape_threshold,
                maker_retry_limit=overrides.get("maker_retry_limit_override", r.maker_retry_limit),
            ),
            abandon=ab,
        )
        return ExecutionAlphaTuningPack(new_cfg)

    def decide(self, ctx: ExecutionContext) -> ExecutionDecision:
        if self._should_cancel(ctx):
            return ExecutionDecision(
                mode="CANCEL",
                aggression_score=0.0,
                slice_notional=0.0,
                cancel=True,
                reason="edge/volatility/spread/fill-probability abandon condition met",
            )

        aggression = self._aggression_score(ctx)
        slice_notional = self._slice_size(ctx, aggression)

        use_taker = self._should_cross(ctx, aggression)
        mode = "TAKER" if use_taker else "MAKER"
        return ExecutionDecision(
            mode=mode,
            aggression_score=aggression,
            slice_notional=slice_notional,
            cancel=False,
            reason=self._decision_reason(ctx, aggression, mode),
        )

    def _aggression_score(self, ctx: ExecutionContext) -> float:
        c = self.config.aggression
        score = 0.0
        score += ctx.expected_edge_bps
        score += c.imbalance_boost * ctx.imbalance_score
        score += c.urgency_multiplier * ctx.urgency_score
        score += 0.50 * ctx.short_horizon_drift_bps
        score -= c.volatility_penalty * ctx.volatility_score
        score -= c.adverse_selection_penalty * ctx.adverse_selection_score
        score -= 0.50 * max(0.0, ctx.spread_bps - c.max_cross_spread_bps)
        return score

    def _slice_size(self, ctx: ExecutionContext, aggression_score: float) -> float:
        s = self.config.slicing
        base = s.base_slice_notional
        top_book_cap = max(s.min_slice_notional, ctx.top_book_notional * s.max_slice_pct_top_book)
        slice_size = min(base, top_book_cap, ctx.remaining_notional)

        if aggression_score > self.config.routing.taker_min_edge_bps:
            slice_size *= s.slice_growth_factor

        if ctx.fill_probability < self.config.routing.maker_min_fill_prob:
            slice_size *= s.slice_decay_on_reject

        if ctx.spread_widen_bps > self.config.abandon.abandon_on_spread_widen_bps:
            slice_size *= s.slice_decay_on_slippage

        return round(clamp(slice_size, s.min_slice_notional, max(s.min_slice_notional, ctx.remaining_notional)), 4)

    def _should_cross(self, ctx: ExecutionContext, aggression_score: float) -> bool:
        r = self.config.routing
        a = self.config.aggression
        edge_ok = ctx.expected_edge_bps >= max(a.min_edge_to_cross_bps, r.taker_min_edge_bps)
        spread_ok = ctx.spread_bps <= a.max_cross_spread_bps
        drift_escape = ctx.short_horizon_drift_bps >= r.drift_escape_threshold
        maker_unfavorable = (
            ctx.fill_probability < r.maker_min_fill_prob
            or ctx.queue_position_score < r.queue_position_threshold
            or ctx.maker_retries_used >= r.maker_retry_limit
            or ctx.elapsed_wait_ms > r.maker_max_wait_ms
        )
        return (edge_ok and spread_ok and aggression_score > 0.0 and (drift_escape or maker_unfavorable))

    def _should_cancel(self, ctx: ExecutionContext) -> bool:
        ab = self.config.abandon
        if ctx.fill_probability < ab.cancel_if_fill_prob_below:
            return True
        if ctx.elapsed_wait_ms > ab.abandon_wait_ms:
            return True
        if ctx.current_edge_decay_bps > ab.abandon_on_edge_decay_bps:
            return True
        if ctx.spread_widen_bps > ab.abandon_on_spread_widen_bps:
            return True
        if ctx.volatility_score > ab.abandon_on_vol_spike:
            return True
        return False

    @staticmethod
    def _decision_reason(ctx: ExecutionContext, aggression_score: float, mode: str) -> str:
        return (
            f"mode={mode}; edge={ctx.expected_edge_bps:.2f}bps; spread={ctx.spread_bps:.2f}bps; "
            f"drift={ctx.short_horizon_drift_bps:.2f}bps; fill_prob={ctx.fill_probability:.2f}; "
            f"aggression={aggression_score:.2f}"
        )


# -----------------------------
# Orchestration glue
# -----------------------------


@dataclass
class GovernanceOutcome:
    incidents: List[Incident]
    actions: ActionSet
    execution_tuning_pack: ExecutionAlphaTuningPack


class ArgusGovernanceCoordinator:
    def __init__(
        self,
        db_path: str,
        thresholds: Optional[Thresholds] = None,
        exec_cfg: Optional[ExecutionAlphaConfig] = None,
    ) -> None:
        self.repo = IncidentRepository(db_path)
        self.repo.init_schema()
        self.incident_engine = IncidentEngine(thresholds or Thresholds())
        self.base_tuning_pack = ExecutionAlphaTuningPack(exec_cfg or ExecutionAlphaConfig())

    def evaluate_snapshot(self, snapshot: RuntimeSnapshot) -> GovernanceOutcome:
        incidents, actions = self.incident_engine.evaluate(snapshot)
        self.repo.insert_many(incidents)
        tuned_pack = self.base_tuning_pack.apply_overrides(actions.execution_overrides)
        return GovernanceOutcome(
            incidents=incidents,
            actions=actions,
            execution_tuning_pack=tuned_pack,
        )


# -----------------------------
# Example integration points
# -----------------------------


def decide_order(
    coordinator: ArgusGovernanceCoordinator,
    snapshot: RuntimeSnapshot,
    exec_ctx: ExecutionContext,
) -> Dict[str, Any]:
    """
    Example proving-phase flow:
    1. Evaluate incidents and bounded auto-actions.
    2. If critical stop is active, refuse order.
    3. Run execution tuning pack to decide maker/taker/cancel/slice.
    4. Return decision payload for the execution engine.
    """
    outcome = coordinator.evaluate_snapshot(snapshot)

    if outcome.actions.stop_trading:
        return {
            "allowed": False,
            "reason": "critical incident stop_trading asserted",
            "incidents": [asdict(i) for i in outcome.incidents],
        }

    decision = outcome.execution_tuning_pack.decide(exec_ctx)

    return {
        "allowed": not decision.cancel,
        "decision": asdict(decision),
        "actions": asdict(outcome.actions),
        "incidents": [asdict(i) for i in outcome.incidents],
    }


# -----------------------------
# Minimal smoke-test example
# -----------------------------


if __name__ == "__main__":
    db_path = "argus_runtime.db"
    coordinator = ArgusGovernanceCoordinator(db_path=db_path)

    trades = [
        TradeRecord(
            ts=utc_now_iso(),
            strategy_id="mean_rev_1",
            symbol="BTC/AUD",
            venue="kraken",
            side="buy",
            qty=0.01,
            expected_price=100000.0,
            fill_price=100080.0,
            fees=1.2,
            gross_pnl=-2.5,
            net_pnl=-3.7,
            slippage_bps=8.0 + i,
            execution_alpha_bps=-1.5 - (i * 0.2),
            maker_flag=0,
            partial_fill_flag=0,
            reject_flag=1 if i in (3, 7) else 0,
            latency_ms=120 + (i * 4),
            ladder_stage="PAPER",
            regime="volatile",
        )
        for i in range(15)
    ]

    positions = [
        PositionRecord(
            ts=utc_now_iso(),
            symbol="BTC/AUD",
            strategy_id="mean_rev_1",
            notional=620.0,
            exposure_pct=41.0,
        )
    ]

    snapshot = RuntimeSnapshot(
        run_id="paper-run-001",
        ladder_stage="PAPER",
        recent_trades=trades,
        recent_positions=positions,
        latest_replay_audit=ReplayAuditRecord(
            ts=utc_now_iso(),
            run_id="paper-run-001",
            status="OK",
            mismatch_count=0,
        ),
        metrics_lag_seconds=5.0,
        strategy_drawdown_pct={"mean_rev_1": 2.4},
    )

    exec_ctx = ExecutionContext(
        strategy_id="mean_rev_1",
        symbol="BTC/AUD",
        venue="kraken",
        expected_edge_bps=9.5,
        spread_bps=2.4,
        short_horizon_drift_bps=3.6,
        volatility_score=1.2,
        imbalance_score=0.8,
        adverse_selection_score=0.6,
        fill_probability=0.48,
        queue_position_score=0.42,
        urgency_score=0.5,
        top_book_notional=900.0,
        remaining_notional=300.0,
        maker_retries_used=1,
        elapsed_wait_ms=850,
        current_edge_decay_bps=1.0,
        spread_widen_bps=0.7,
    )

    payload = decide_order(coordinator, snapshot, exec_ctx)
    print(json.dumps(payload, indent=2, default=str))
