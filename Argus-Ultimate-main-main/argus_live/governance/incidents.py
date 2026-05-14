"""Governance incidents — repository, factory, and the 8-rule incident engine."""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .types import (
    ActionSet,
    Incident,
    IncidentClass,
    IncidentStatus,
    PositionRecord,
    ReplayAuditRecord,
    RuntimeSnapshot,
    Severity,
    Thresholds,
    TradeRecord,
    _pct,
    _recent_trade_dicts,
    _rolling_avg,
    utc_now_iso,
)


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
