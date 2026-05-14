from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HardeningBlock:
    module: str
    reason: str
    details: Dict[str, Any]
    severity: str = "warning"


class HardeningModulePack:
    """Unified deterministic hardening pack implementing 15 runtime safety modules."""

    def __init__(self, config: Any, db_path: str):
        self.enabled = bool(getattr(config, "hardening_pack_enabled", True))
        self.fail_closed = bool(getattr(config, "hardening_pack_fail_closed", True))
        self.db_path = str(db_path or "data/unified_trades.db")

        self.venue_health_max_stale_seconds = float(
            getattr(config, "hardening_venue_health_max_stale_seconds", 30.0) or 30.0
        )
        self.venue_health_max_heartbeat_age_seconds = float(
            getattr(config, "hardening_venue_health_max_heartbeat_age_seconds", 15.0) or 15.0
        )
        self.latency_spike_threshold_ms = float(
            getattr(config, "hardening_latency_spike_threshold_ms", 8000.0) or 8000.0
        )
        _latency_grace_raw = getattr(config, "hardening_latency_spike_grace_cycles", None)
        if _latency_grace_raw is None:
            _latency_grace_raw = 2
        self.latency_spike_grace_cycles = int(_latency_grace_raw)
        self.latency_spike_consecutive_limit = int(
            getattr(config, "hardening_latency_spike_consecutive_limit", 3) or 3
        )
        self.fill_anomaly_max_slippage_bps = float(
            getattr(config, "hardening_fill_anomaly_max_slippage_bps", 40.0) or 40.0
        )
        self.fill_anomaly_max_abs_pnl_aud = float(
            getattr(config, "hardening_fill_anomaly_max_abs_pnl_aud", 500.0) or 500.0
        )
        self.ledger_invariant_tolerance_aud = float(
            getattr(config, "hardening_ledger_invariant_tolerance_aud", 5.0) or 5.0
        )
        # In ARGUS runtime accounting, realized/unrealized PnL are fee-net by default.
        self.ledger_pnl_is_fee_net = bool(
            getattr(config, "hardening_ledger_pnl_is_fee_net", True)
        )
        self.journal_gap_max_idle_cycles = int(
            getattr(config, "hardening_journal_gap_max_idle_cycles", 3) or 3
        )
        self.position_drift_max_exposure_multiplier = float(
            getattr(config, "hardening_position_drift_max_exposure_multiplier", 1.25) or 1.25
        )
        self.execution_timeout_halt_after = int(
            getattr(config, "hardening_execution_timeout_halt_after", 5) or 5
        )
        self.cancel_storm_max_cancels_per_cycle = int(
            getattr(config, "hardening_cancel_storm_max_cancels_per_cycle", 5) or 5
        )
        self.market_impact_max_ratio = float(
            getattr(config, "hardening_market_impact_max_ratio", 5.0) or 5.0
        )
        self.slippage_regime_threshold_bps = float(
            getattr(config, "hardening_slippage_regime_threshold_bps", 25.0) or 25.0
        )
        self.slippage_regime_edge_buffer_mult = float(
            getattr(config, "hardening_slippage_regime_edge_buffer_mult", 1.2) or 1.2
        )
        self.replay_coverage_min_ratio = float(
            getattr(config, "hardening_replay_coverage_min_ratio", 0.80) or 0.80
        )
        self.promotion_rollback_drawdown_pct = float(
            getattr(config, "hardening_promotion_rollback_drawdown_pct", 0.12) or 0.12
        )
        self.alert_escalation_repeat_threshold = int(
            getattr(config, "hardening_alert_escalation_repeat_threshold", 3) or 3
        )

        self._latency_spike_streak = 0
        self._journal_last_snapshot_id = 0
        self._journal_idle_cycles = 0
        self._timeout_count = 0
        self._cancel_count_by_cycle: Dict[int, int] = defaultdict(int)
        self._slippage_by_regime: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self._alert_counts: Dict[str, int] = defaultdict(int)
        self._force_halt = False
        self._halt_reason = ""
        self._last_blocks: List[HardeningBlock] = []
        self._last_notes: List[Dict[str, Any]] = []
        self._promotion_rollback_recommended = False
        self._ledger_invariant_baseline_aud: Optional[float] = None

    @staticmethod
    def _signal_get(signal: Any, field: str, default: Any = None) -> Any:
        if isinstance(signal, dict):
            return signal.get(field, default)
        return getattr(signal, field, default)

    @staticmethod
    def _signal_set(signal: Any, field: str, value: Any) -> None:
        if isinstance(signal, dict):
            signal[field] = value
        else:
            setattr(signal, field, value)

    @staticmethod
    def _side(signal: Any) -> str:
        raw = str(
            HardeningModulePack._signal_get(signal, "side", None)
            or HardeningModulePack._signal_get(signal, "action", None)
            or ""
        ).upper()
        if raw in {"BUY", "LONG"}:
            return "BUY"
        if raw in {"SELL", "SHORT"}:
            return "SELL"
        return raw

    def _escalate(self, key: str) -> str:
        self._alert_counts[key] += 1
        if self._alert_counts[key] >= self.alert_escalation_repeat_threshold:
            return "critical"
        return "warning"

    def _maybe_halt(self, reason: str) -> None:
        if not self.fail_closed:
            return
        self._force_halt = True
        self._halt_reason = str(reason or "hardening_halt")

    def force_halt_state(self) -> Tuple[bool, str]:
        return bool(self._force_halt), str(self._halt_reason or "")

    def clear_cycle_state(self, cycle_id: int) -> None:
        self._last_blocks = []
        self._last_notes = []
        # Keep only recent cancel counters.
        for old in list(self._cancel_count_by_cycle.keys()):
            if old < int(cycle_id) - 5:
                self._cancel_count_by_cycle.pop(old, None)

    def _block(self, module: str, reason: str, details: Dict[str, Any]) -> HardeningBlock:
        severity = self._escalate(f"{module}:{reason}")
        blk = HardeningBlock(module=module, reason=reason, details=dict(details or {}), severity=severity)
        self._last_blocks.append(blk)
        return blk

    def _note(self, module: str, details: Dict[str, Any]) -> None:
        self._last_notes.append({"module": module, **dict(details or {})})

    def _safe_json_ratio(self, details_json: Any) -> bool:
        if details_json is None:
            return False
        if isinstance(details_json, dict):
            return bool(details_json)
        txt = str(details_json or "").strip()
        if not txt:
            return False
        try:
            parsed = json.loads(txt)
            return bool(parsed)
        except Exception:
            return False

    def _db_max_snapshot_id(self) -> int:
        try:
            con = sqlite3.connect(self.db_path, timeout=10.0)
            try:
                row = con.execute("SELECT COALESCE(MAX(id), 0) FROM decision_snapshots").fetchone()
                return int((row or [0])[0] or 0)
            finally:
                con.close()
        except Exception:
            return 0

    def _db_replay_coverage_ratio(self, lookback: int = 100) -> float:
        try:
            con = sqlite3.connect(self.db_path, timeout=10.0)
            try:
                rows = con.execute(
                    "SELECT details_json FROM decision_snapshots ORDER BY id DESC LIMIT ?",
                    (int(max(1, lookback)),),
                ).fetchall()
            finally:
                con.close()
        except Exception:
            return 0.0
        if not rows:
            return 1.0
        covered = sum(1 for r in rows if self._safe_json_ratio((r or [None])[0]))
        return float(covered / max(1, len(rows)))

    def pre_signal_filter(
        self,
        signals: Iterable[Any],
        *,
        cycle_id: int,
        regime_label: str,
    ) -> Tuple[List[Any], List[HardeningBlock]]:
        rows = list(signals or [])
        if not self.enabled:
            return rows, []
        self.clear_cycle_state(int(cycle_id))

        kept: List[Any] = []
        blocks: List[HardeningBlock] = []
        for sig in rows:
            symbol = str(self._signal_get(sig, "symbol", "") or "")
            stale_age = float(self._signal_get(sig, "market_data_age_seconds", 0.0) or 0.0)
            heartbeat_age = float(
                self._signal_get(sig, "exchange_heartbeat_age_seconds", 0.0) or 0.0
            )
            spread_bps = float(self._signal_get(sig, "spread_bps", 0.0) or 0.0)
            bid = float(
                self._signal_get(sig, "top_of_book_bid_size", None)
                or self._signal_get(sig, "bid_size_1", 0.0)
                or 0.0
            )
            ask = float(
                self._signal_get(sig, "top_of_book_ask_size", None)
                or self._signal_get(sig, "ask_size_1", 0.0)
                or 0.0
            )
            depth = float(
                self._signal_get(sig, "orderbook_depth_estimate", None) or max(0.0, bid + ask)
            )
            price = float(
                self._signal_get(sig, "entry_price", None)
                or self._signal_get(sig, "price", None)
                or 0.0
            )
            qty = float(
                self._signal_get(sig, "quantity", None)
                or self._signal_get(sig, "planned_order_size", None)
                or 0.0
            )
            notional = max(0.0, float(price * qty)) if price > 0.0 and qty > 0.0 else 0.0
            impact_ratio: Optional[float] = None
            if depth > 0.0 and notional > 0.0:
                impact_ratio = float(notional / max(depth, 1e-9))

            self._signal_set(sig, "hardening_regime_label", str(regime_label or ""))
            self._signal_set(sig, "venue_health_stale_age_seconds", float(stale_age))
            self._signal_set(sig, "venue_health_heartbeat_age_seconds", float(heartbeat_age))
            self._signal_set(sig, "market_impact_ratio", float(impact_ratio or 0.0))

            if stale_age > self.venue_health_max_stale_seconds:
                blk = self._block(
                    "VenueHealthGate",
                    "stale_market_data",
                    {"symbol": symbol, "stale_age_seconds": stale_age},
                )
                self._signal_set(sig, "hardening_module", blk.module)
                self._signal_set(sig, "hardening_reason", blk.reason)
                blocks.append(blk)
                continue
            if heartbeat_age > self.venue_health_max_heartbeat_age_seconds:
                blk = self._block(
                    "VenueHealthGate",
                    "stale_exchange_heartbeat",
                    {"symbol": symbol, "heartbeat_age_seconds": heartbeat_age},
                )
                self._signal_set(sig, "hardening_module", blk.module)
                self._signal_set(sig, "hardening_reason", blk.reason)
                blocks.append(blk)
                continue

            # MarketImpactLimiter
            if impact_ratio is not None and impact_ratio > self.market_impact_max_ratio:
                blk = self._block(
                    "MarketImpactLimiter",
                    "impact_ratio_exceeded",
                    {
                        "symbol": symbol,
                        "impact_ratio": float(impact_ratio),
                        "max_ratio": float(self.market_impact_max_ratio),
                        "notional": float(notional),
                        "depth": float(depth),
                    },
                )
                self._signal_set(sig, "hardening_module", blk.module)
                self._signal_set(sig, "hardening_reason", blk.reason)
                blocks.append(blk)
                continue

            # SlippageRegimeGate
            regime_key = str(regime_label or "unknown")
            slippage_series = self._slippage_by_regime.get(regime_key)
            p90 = 0.0
            if slippage_series:
                ordered = sorted(float(x) for x in slippage_series)
                idx = max(0, min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1)))))
                p90 = float(ordered[idx])
            edge_bps = float(self._signal_get(sig, "expected_net_edge_bps", 0.0) or 0.0)
            self._signal_set(sig, "slippage_regime_p90_bps", float(p90))
            if p90 > self.slippage_regime_threshold_bps and edge_bps < (p90 * self.slippage_regime_edge_buffer_mult):
                blk = self._block(
                    "SlippageRegimeGate",
                    "edge_below_regime_slippage_budget",
                    {"symbol": symbol, "regime_p90_slippage_bps": p90, "expected_net_edge_bps": edge_bps},
                )
                self._signal_set(sig, "hardening_module", blk.module)
                self._signal_set(sig, "hardening_reason", blk.reason)
                blocks.append(blk)
                continue

            kept.append(sig)
        return kept, blocks

    def pre_execution_filter(
        self,
        signals: Iterable[Any],
        *,
        cycle_id: int,
    ) -> Tuple[List[Any], List[HardeningBlock]]:
        rows = list(signals or [])
        if not self.enabled:
            return rows, []
        cancel_count = int(self._cancel_count_by_cycle.get(int(cycle_id), 0) or 0)
        if cancel_count <= self.cancel_storm_max_cancels_per_cycle:
            return rows, []
        blk = self._block(
            "CancelStormProtector",
            "cancel_storm_active",
            {
                "cycle_id": int(cycle_id),
                "cancel_count": cancel_count,
                "limit": int(self.cancel_storm_max_cancels_per_cycle),
            },
        )
        self._maybe_halt("cancel_storm_active")
        for sig in rows:
            self._signal_set(sig, "hardening_module", blk.module)
            self._signal_set(sig, "hardening_reason", blk.reason)
        return [], [blk]

    def on_trade_result(self, result: Dict[str, Any], *, cycle_id: int, regime_label: str) -> None:
        if not self.enabled:
            return
        status = str(result.get("status", "") or "").lower()
        slippage_bps = float(
            result.get("slippage_bps")
            or (result.get("raw") or {}).get("slippage_bps")
            or 0.0
        )
        pnl = float(
            result.get("pnl")
            or result.get("net_pnl_aud")
            or (result.get("raw") or {}).get("pnl")
            or 0.0
        )

        # FillAnomalyDetector
        if abs(slippage_bps) > self.fill_anomaly_max_slippage_bps:
            self._block(
                "FillAnomalyDetector",
                "extreme_slippage",
                {"slippage_bps": float(slippage_bps)},
            )
        if abs(pnl) > self.fill_anomaly_max_abs_pnl_aud:
            self._block(
                "FillAnomalyDetector",
                "extreme_trade_pnl",
                {"pnl_aud": float(pnl)},
            )

        if status in {"canceled", "cancelled"}:
            self._cancel_count_by_cycle[int(cycle_id)] += 1

        # ExecutionTimeoutClassifier
        if "timeout" in status or "timeout" in str(result.get("error", "")).lower():
            self._timeout_count += 1
            self._block(
                "ExecutionTimeoutClassifier",
                "execution_timeout",
                {"timeout_count": int(self._timeout_count)},
            )
            if self._timeout_count >= self.execution_timeout_halt_after:
                self._maybe_halt("execution_timeout_threshold")
        elif status in {"filled", "success", "ok"}:
            self._timeout_count = max(0, self._timeout_count - 1)

        # Slippage series for SlippageRegimeGate.
        regime_key = str(regime_label or "unknown")
        if slippage_bps != 0.0:
            self._slippage_by_regime[regime_key].append(float(abs(slippage_bps)))

    def post_cycle(
        self,
        *,
        cycle_id: int,
        cycle_duration_ms: float,
        had_signals: bool,
        portfolio_value_aud: float,
        cash_balance_aud: float,
        realized_pnl_aud: float,
        unrealized_pnl_aud: float,
        total_fees_aud: float,
        starting_capital_aud: float,
        positions: Dict[str, Any],
        max_total_exposure_pct: float,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {"halt": False, "halt_reason": "", "blocks": [], "notes": []}

        # LatencySpikeGuard
        if int(cycle_id) <= max(0, int(self.latency_spike_grace_cycles)):
            self._latency_spike_streak = 0
            self._note(
                "LatencySpikeGuard",
                {
                    "latency_spike_grace_active": True,
                    "cycle_id": int(cycle_id),
                    "grace_cycles": int(self.latency_spike_grace_cycles),
                    "cycle_ms": float(cycle_duration_ms or 0.0),
                },
            )
        else:
            if float(cycle_duration_ms or 0.0) > self.latency_spike_threshold_ms:
                self._latency_spike_streak += 1
                self._block(
                    "LatencySpikeGuard",
                    "latency_spike",
                    {
                        "cycle_ms": float(cycle_duration_ms),
                        "threshold_ms": float(self.latency_spike_threshold_ms),
                        "streak": int(self._latency_spike_streak),
                    },
                )
            else:
                self._latency_spike_streak = 0
            if self._latency_spike_streak >= self.latency_spike_consecutive_limit:
                self._maybe_halt("latency_spike_streak_exhausted")

        # LedgerInvariantGuard
        pnl_component = float(realized_pnl_aud) + float(unrealized_pnl_aud)
        if not self.ledger_pnl_is_fee_net:
            pnl_component -= float(total_fees_aud)
        startup_invariant = float(starting_capital_aud) + float(pnl_component)
        startup_diff = abs(float(portfolio_value_aud) - float(startup_invariant))
        if self._ledger_invariant_baseline_aud is None:
            if startup_diff <= self.ledger_invariant_tolerance_aud:
                self._ledger_invariant_baseline_aud = float(starting_capital_aud)
            else:
                # Rebase once if runtime restores persisted state that doesn't equal configured start capital.
                self._ledger_invariant_baseline_aud = float(portfolio_value_aud) - float(pnl_component)
                self._note(
                    "LedgerInvariantGuard",
                    {
                        "baseline_rebased": True,
                        "baseline_aud": float(self._ledger_invariant_baseline_aud),
                        "starting_capital_aud": float(starting_capital_aud),
                        "startup_diff_aud": float(startup_diff),
                    },
                )
        invariant_equity = float(self._ledger_invariant_baseline_aud or 0.0) + float(pnl_component)
        invariant_diff = abs(float(portfolio_value_aud) - float(invariant_equity))
        if invariant_diff > self.ledger_invariant_tolerance_aud:
            self._block(
                "LedgerInvariantGuard",
                "equity_invariant_drift",
                {
                    "portfolio_value_aud": float(portfolio_value_aud),
                    "invariant_equity_aud": float(invariant_equity),
                    "diff_aud": float(invariant_diff),
                    "ledger_pnl_is_fee_net": bool(self.ledger_pnl_is_fee_net),
                },
            )

        # PositionDriftSentinel
        pos_notional = 0.0
        try:
            for p in list((positions or {}).values()):
                if not isinstance(p, dict):
                    continue
                qty = abs(float(p.get("quantity", 0.0) or 0.0))
                px = float(p.get("current_price", p.get("entry_price", 0.0)) or 0.0)
                pos_notional += max(0.0, qty * px)
        except Exception:
            pos_notional = 0.0
        denom = max(float(portfolio_value_aud or 0.0), 1e-9)
        exposure_ratio = float(pos_notional / denom)
        max_allowed = float(max_total_exposure_pct or 0.0) * self.position_drift_max_exposure_multiplier
        if max_allowed > 0.0 and exposure_ratio > max_allowed:
            self._block(
                "PositionDriftSentinel",
                "position_exposure_drift",
                {
                    "exposure_ratio": float(exposure_ratio),
                    "allowed_ratio": float(max_allowed),
                    "position_notional": float(pos_notional),
                },
            )

        # JournalGapDetector
        snap_id = self._db_max_snapshot_id()
        if had_signals and snap_id <= self._journal_last_snapshot_id:
            self._journal_idle_cycles += 1
        else:
            self._journal_idle_cycles = 0
        self._journal_last_snapshot_id = max(self._journal_last_snapshot_id, snap_id)
        if self._journal_idle_cycles >= self.journal_gap_max_idle_cycles:
            self._block(
                "JournalGapDetector",
                "snapshot_id_not_growing",
                {
                    "journal_idle_cycles": int(self._journal_idle_cycles),
                    "snapshot_id": int(self._journal_last_snapshot_id),
                },
            )

        # ReplayCoverageAuditor
        coverage_ratio = self._db_replay_coverage_ratio(lookback=100)
        self._note("ReplayCoverageAuditor", {"replay_coverage_ratio": float(coverage_ratio)})
        if coverage_ratio < self.replay_coverage_min_ratio:
            self._block(
                "ReplayCoverageAuditor",
                "replay_coverage_below_threshold",
                {
                    "coverage_ratio": float(coverage_ratio),
                    "min_ratio": float(self.replay_coverage_min_ratio),
                },
            )

        # PromotionRollbackEngine recommendation (advisory only)
        drawdown_pct = 0.0
        if float(starting_capital_aud or 0.0) > 0:
            drawdown_pct = max(
                0.0,
                (float(starting_capital_aud) - float(portfolio_value_aud)) / float(starting_capital_aud),
            )
        self._promotion_rollback_recommended = bool(
            drawdown_pct >= self.promotion_rollback_drawdown_pct
        )
        self._note(
            "PromotionRollbackEngine",
            {
                "promotion_rollback_recommended": bool(self._promotion_rollback_recommended),
                "drawdown_pct": float(drawdown_pct),
            },
        )

        # CircuitBreakerOrchestrator + RecoveryPlaybookEngine
        if self._force_halt:
            self._note(
                "RecoveryPlaybookEngine",
                {
                    "actions": [
                        "stop_new_intents",
                        "cancel_open_orders",
                        "run_reconciliation",
                        "review_last_decision_snapshots",
                    ],
                    "halt_reason": str(self._halt_reason),
                },
            )

        return {
            "halt": bool(self._force_halt),
            "halt_reason": str(self._halt_reason or ""),
            "blocks": [asdict(b) for b in self._last_blocks],
            "notes": list(self._last_notes),
            "promotion_rollback_recommended": bool(self._promotion_rollback_recommended),
        }
