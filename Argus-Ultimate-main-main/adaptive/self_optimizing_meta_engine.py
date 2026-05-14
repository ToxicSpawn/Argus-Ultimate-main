from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StrategyWeightState:
    strategy_name: str
    current_weight: float
    proposed_weight: float
    smoothed_weight: float
    last_updated_ts: float
    trades_count: int
    expectancy: float
    profit_factor: float
    sharpe_like_score: float
    # v1 spec aliases (kept alongside legacy names for compatibility)
    drawdown_pct: float
    slippage_bps: float
    max_drawdown_pct: float
    avg_realized_slippage_bps: float
    fee_ratio: float
    regime_label: Optional[str] = None
    enabled_for_weighting: bool = False
    reasons: List[str] = field(default_factory=list)


@dataclass(slots=True)
class MetaWeightSnapshot:
    run_id: str
    trace_id: str
    ts: float
    regime_label: str
    weights_json: str
    reasons_json: str
    source_metrics_json: str


class SelfOptimizingMetaEngine:
    """Deterministic strategy re-weighting engine (advisory for candidate ranking only)."""

    def __init__(
        self,
        *,
        db_path: str = "data/meta_weights.db",
        enabled: bool = True,
        advisory_only: bool = False,
        update_interval_cycles: int = 10,
        min_trades_for_reweighting: int = 5,
        meta_alpha: float = 0.2,
        max_weight_change_per_update: float = 0.10,
        min_weight_per_strategy: float = 0.05,
        max_weight_per_strategy: float = 0.45,
        baseline_weight_mode: str = "equal",
        score_weights: Optional[Dict[str, float]] = None,
        regime_multipliers: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.db_path = str(db_path or "data/meta_weights.db")
        self.enabled = bool(enabled)
        self.advisory_only = bool(advisory_only)
        self.update_interval_cycles = max(1, int(update_interval_cycles or 10))
        self.min_trades_for_reweighting = max(1, int(min_trades_for_reweighting or 5))
        self.meta_alpha = max(0.0, min(1.0, float(meta_alpha or 0.2)))
        self.max_weight_change_per_update = max(0.0, min(1.0, float(max_weight_change_per_update or 0.10)))
        self.min_weight_per_strategy = max(0.0, min(1.0, float(min_weight_per_strategy or 0.05)))
        self.max_weight_per_strategy = max(0.0, min(1.0, float(max_weight_per_strategy or 0.45)))
        self.baseline_weight_mode = str(baseline_weight_mode or "equal")
        raw_score_weights = dict(score_weights or {})
        self.score_weights: Dict[str, float] = {
            "expectancy": float(raw_score_weights.get("expectancy", 1.0) or 1.0),
            "sharpe_like": float(raw_score_weights.get("sharpe_like", 1.0) or 1.0),
            "profit_factor": float(raw_score_weights.get("profit_factor", 0.75) or 0.75),
            "drawdown_penalty": float(raw_score_weights.get("drawdown_penalty", 1.0) or 1.0),
            "fee_penalty": float(raw_score_weights.get("fee_penalty", 0.5) or 0.5),
            "slippage_penalty": float(raw_score_weights.get("slippage_penalty", 0.5) or 0.5),
        }
        self.regime_multipliers: Dict[str, Dict[str, float]] = {
            str(k): {str(sk): float(sv) for sk, sv in dict(v or {}).items()}
            for k, v in dict(regime_multipliers or {}).items()
        }

        self._weights: Dict[str, float] = {}
        self._states: Dict[str, StrategyWeightState] = {}
        self._weight_change_cache: Dict[str, float] = {}
        self._last_update_ts: float = 0.0
        self._last_update_cycle: int = 0
        self._init_schema()
        self.load_from_db()

    def _connect(self) -> sqlite3.Connection:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.db_path, timeout=15.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        return con

    def _init_schema(self) -> None:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_weights (
                    strategy_name TEXT PRIMARY KEY,
                    current_weight REAL NOT NULL,
                    proposed_weight REAL NOT NULL,
                    smoothed_weight REAL NOT NULL,
                    state_json TEXT NOT NULL,
                    last_updated_ts REAL NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS meta_weight_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    ts REAL NOT NULL,
                    regime_label TEXT,
                    weights_json TEXT NOT NULL,
                    reasons_json TEXT NOT NULL,
                    source_metrics_json TEXT NOT NULL
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_meta_weight_snapshots_ts ON meta_weight_snapshots(ts)")
            con.commit()

    @staticmethod
    def _strategy_name(signal: Any) -> str:
        if isinstance(signal, dict):
            out = signal.get("source_strategy") or signal.get("strategy") or ""
        else:
            out = getattr(signal, "source_strategy", None) or getattr(signal, "strategy", None) or ""
        s = str(out or "").strip()
        return s or "unknown"

    @staticmethod
    def _signal_get(signal: Any, name: str, default: Any = None) -> Any:
        if isinstance(signal, dict):
            return signal.get(name, default)
        return getattr(signal, name, default)

    def _baseline_weights(self, strategies: Iterable[str]) -> Dict[str, float]:
        uniq = sorted({str(s or "").strip() for s in strategies if str(s or "").strip()})
        if not uniq:
            return {}
        if str(self.baseline_weight_mode).lower() != "equal":
            logger.debug("Unsupported baseline_weight_mode=%s, using equal", self.baseline_weight_mode)
        w = 1.0 / float(len(uniq))
        return {s: w for s in uniq}

    @staticmethod
    def _norm_0_1(values: Dict[str, float], *, invert: bool = False) -> Dict[str, float]:
        if not values:
            return {}
        vals = [float(v) for v in values.values()]
        lo = min(vals)
        hi = max(vals)
        if hi - lo <= 1e-12:
            out = {k: 0.5 for k in values}
        else:
            out = {k: (float(v) - lo) / (hi - lo) for k, v in values.items()}
        if invert:
            return {k: 1.0 - v for k, v in out.items()}
        return out

    def _bounded_normalized(self, raw: Dict[str, float]) -> Dict[str, float]:
        if not raw:
            return {}
        keys = sorted(raw)
        n = len(keys)
        if n == 1:
            return {keys[0]: 1.0}

        min_w = max(0.0, min(1.0, float(self.min_weight_per_strategy or 0.0)))
        max_w = max(0.0, min(1.0, float(self.max_weight_per_strategy or 1.0)))
        if min_w * n > 1.0:
            min_w = 1.0 / float(n)
        # If configured max is infeasible with min floor, relax max to keep normalization possible.
        feasible_max = 1.0 - (min_w * float(max(0, n - 1)))
        max_w = max(max_w, feasible_max)
        max_w = max(min_w, min(1.0, max_w))

        floor = {k: min_w for k in keys}
        cap = {k: max(0.0, max_w - min_w) for k in keys}
        leftover = max(0.0, 1.0 - sum(floor.values()))
        score = {k: max(0.0, float(raw.get(k, 0.0) or 0.0)) for k in keys}
        alloc = {k: 0.0 for k in keys}
        active = set(keys)

        while leftover > 1e-12 and active:
            denom = sum(score[k] for k in active)
            if denom <= 1e-12:
                share = leftover / float(len(active))
                for k in list(active):
                    add = min(share, cap[k] - alloc[k])
                    alloc[k] += max(0.0, add)
                leftover = max(0.0, 1.0 - sum(floor[k] + alloc[k] for k in keys))
            else:
                spent = 0.0
                for k in list(active):
                    desired = leftover * (score[k] / denom)
                    add = min(desired, cap[k] - alloc[k])
                    add = max(0.0, add)
                    alloc[k] += add
                    spent += add
                if spent <= 1e-12:
                    break
                leftover = max(0.0, leftover - spent)
            active = {k for k in active if alloc[k] < (cap[k] - 1e-12)}

        out = {k: floor[k] + alloc[k] for k in keys}
        total = sum(out.values())
        if total <= 1e-12:
            eq = 1.0 / float(n)
            return {k: eq for k in keys}
        return {k: max(0.0, float(v) / total) for k, v in out.items()}

    def _regime_key(self, regime_label: str) -> str:
        txt = str(regime_label or "").strip().lower()
        if not txt:
            return ""
        if ":" in txt:
            return txt.split(":", 1)[0]
        return txt

    def _load_metric_map(
        self,
        *,
        strategy_names: List[str],
        strategy_evaluation_engine: Any,
        regime_label: str,
    ) -> Dict[str, Any]:
        metric_map: Dict[str, Any] = {}
        for s in strategy_names:
            m = None
            try:
                m = strategy_evaluation_engine.get_metrics(
                    strategy_name=str(s),
                    symbol=None,
                    regime_label=(regime_label or None),
                )
            except Exception:
                m = None
            metric_map[s] = m
        return metric_map

    def compute_weights(
        self,
        *,
        strategy_names: Iterable[str],
        strategy_evaluation_engine: Any,
        regime_label: str = "",
        execution_telemetry: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
        names = sorted({str(s or "").strip() for s in strategy_names if str(s or "").strip()})
        if not names:
            return {}, {}, {}

        baseline = self._baseline_weights(names)
        metric_map = self._load_metric_map(
            strategy_names=names,
            strategy_evaluation_engine=strategy_evaluation_engine,
            regime_label=str(regime_label or ""),
        )

        eligible: List[str] = []
        reasons: Dict[str, List[str]] = {s: [] for s in names}
        source_metrics: Dict[str, Dict[str, Any]] = {}

        expectancy_vals: Dict[str, float] = {}
        sharpe_vals: Dict[str, float] = {}
        profit_vals: Dict[str, float] = {}
        drawdown_vals: Dict[str, float] = {}
        fee_vals: Dict[str, float] = {}
        slippage_vals: Dict[str, float] = {}

        for s in names:
            m = metric_map.get(s)
            trades = int(getattr(m, "trades_count", 0) or 0) if m is not None else 0
            expectancy = float(getattr(m, "expectancy", 0.0) or 0.0) if m is not None else 0.0
            sharpe = float(getattr(m, "sharpe_like_score", 0.0) or 0.0) if m is not None else 0.0
            profit_factor = float(getattr(m, "profit_factor", 0.0) or 0.0) if m is not None else 0.0
            drawdown = float(getattr(m, "max_drawdown_pct", 0.0) or 0.0) if m is not None else 0.0
            gross = float(getattr(m, "gross_pnl_aud", 0.0) or 0.0) if m is not None else 0.0
            fees = float(getattr(m, "total_fees_aud", 0.0) or 0.0) if m is not None else 0.0
            slippage = float(getattr(m, "avg_realized_slippage_bps", 0.0) or 0.0) if m is not None else 0.0
            if execution_telemetry:
                t = execution_telemetry.get(s) or {}
                slippage = float(t.get("slippage_p90", slippage) or slippage)

            fee_ratio = fees / max(abs(gross), 1e-9)
            source_metrics[s] = {
                "trades_count": trades,
                "expectancy": expectancy,
                "sharpe_like_score": sharpe,
                "profit_factor": profit_factor,
                "max_drawdown_pct": drawdown,
                "fee_ratio": fee_ratio,
                "avg_realized_slippage_bps": slippage,
            }
            if trades < self.min_trades_for_reweighting:
                reasons[s].append("insufficient_trades_baseline")
                continue
            eligible.append(s)
            expectancy_vals[s] = expectancy
            sharpe_vals[s] = sharpe
            profit_vals[s] = profit_factor
            drawdown_vals[s] = drawdown
            fee_vals[s] = fee_ratio
            slippage_vals[s] = slippage

        norm_expectancy = self._norm_0_1(expectancy_vals)
        norm_sharpe = self._norm_0_1(sharpe_vals)
        norm_profit = self._norm_0_1(profit_vals)
        norm_drawdown = self._norm_0_1(drawdown_vals)
        norm_fee = self._norm_0_1(fee_vals)
        norm_slippage = self._norm_0_1(slippage_vals)

        raw_scores: Dict[str, float] = {}
        w = self.score_weights
        regime_key = self._regime_key(regime_label)
        regime_mult_map = dict(self.regime_multipliers.get(regime_key, {}) or {})

        for s in names:
            if s not in eligible:
                raw_scores[s] = 0.0
                continue
            score = (
                float(w["expectancy"]) * float(norm_expectancy.get(s, 0.5))
                + float(w["sharpe_like"]) * float(norm_sharpe.get(s, 0.5))
                + float(w["profit_factor"]) * float(norm_profit.get(s, 0.5))
                - float(w["drawdown_penalty"]) * float(norm_drawdown.get(s, 0.5))
                - float(w["fee_penalty"]) * float(norm_fee.get(s, 0.5))
                - float(w["slippage_penalty"]) * float(norm_slippage.get(s, 0.5))
            )
            mult = float(regime_mult_map.get(s, 1.0) or 1.0)
            if mult != 1.0:
                reasons[s].append(f"regime_multiplier:{regime_key}:{mult:.4f}")
            score *= mult
            raw_scores[s] = max(0.0, float(score))

        proposed = dict(baseline)
        pos_sum = sum(raw_scores.get(s, 0.0) for s in eligible)
        if eligible and pos_sum > 1e-12:
            for s in eligible:
                proposed[s] = float(raw_scores.get(s, 0.0) / pos_sum)
        proposed = self._bounded_normalized(proposed)

        bounded_proposed: Dict[str, float] = {}
        alpha = float(self.meta_alpha)
        max_delta = float(self.max_weight_change_per_update)
        prev_weights = dict(self._weights or {})

        for s in names:
            prev = float(prev_weights.get(s, baseline.get(s, 0.0)) or 0.0)
            target = float(proposed.get(s, baseline.get(s, 0.0)) or 0.0)
            delta = max(-max_delta, min(max_delta, target - prev))
            limited = prev + delta
            smoothed = ((1.0 - alpha) * prev) + (alpha * limited)
            bounded_proposed[s] = max(0.0, float(smoothed))

        final_weights = self._bounded_normalized(bounded_proposed)
        self._weight_change_cache = {
            s: float(final_weights.get(s, 0.0) - float(prev_weights.get(s, baseline.get(s, 0.0)) or 0.0))
            for s in names
        }

        now = float(time.time())
        states: Dict[str, StrategyWeightState] = {}
        for s in names:
            src = source_metrics.get(s, {})
            state = StrategyWeightState(
                strategy_name=s,
                current_weight=float(prev_weights.get(s, baseline.get(s, 0.0)) or 0.0),
                proposed_weight=float(proposed.get(s, baseline.get(s, 0.0)) or 0.0),
                smoothed_weight=float(final_weights.get(s, baseline.get(s, 0.0)) or 0.0),
                last_updated_ts=now,
                trades_count=int(src.get("trades_count", 0) or 0),
                expectancy=float(src.get("expectancy", 0.0) or 0.0),
                profit_factor=float(src.get("profit_factor", 0.0) or 0.0),
                sharpe_like_score=float(src.get("sharpe_like_score", 0.0) or 0.0),
                drawdown_pct=float(src.get("max_drawdown_pct", 0.0) or 0.0),
                slippage_bps=float(src.get("avg_realized_slippage_bps", 0.0) or 0.0),
                max_drawdown_pct=float(src.get("max_drawdown_pct", 0.0) or 0.0),
                avg_realized_slippage_bps=float(src.get("avg_realized_slippage_bps", 0.0) or 0.0),
                fee_ratio=float(src.get("fee_ratio", 0.0) or 0.0),
                regime_label=(str(regime_label) if regime_label else None),
                enabled_for_weighting=(s in eligible),
                reasons=list(reasons.get(s, [])),
            )
            states[s] = state

        self._weights = dict(final_weights)
        self._states = dict(states)
        self._last_update_ts = now
        return final_weights, reasons, source_metrics

    def persist_current(self) -> None:
        with self._connect() as con:
            cur = con.cursor()
            for strategy, st in self._states.items():
                cur.execute(
                    """
                    INSERT INTO strategy_weights
                    (strategy_name, current_weight, proposed_weight, smoothed_weight, state_json, last_updated_ts)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(strategy_name) DO UPDATE SET
                        current_weight=excluded.current_weight,
                        proposed_weight=excluded.proposed_weight,
                        smoothed_weight=excluded.smoothed_weight,
                        state_json=excluded.state_json,
                        last_updated_ts=excluded.last_updated_ts
                    """,
                    (
                        str(strategy),
                        float(st.current_weight),
                        float(st.proposed_weight),
                        float(st.smoothed_weight),
                        json.dumps(asdict(st), ensure_ascii=True, default=str),
                        float(st.last_updated_ts),
                    ),
                )
            con.commit()

    def persist_snapshot(
        self,
        *,
        run_id: str,
        trace_id: str,
        regime_label: str,
        reasons: Dict[str, List[str]],
        source_metrics: Dict[str, Dict[str, Any]],
    ) -> MetaWeightSnapshot:
        snap = MetaWeightSnapshot(
            run_id=str(run_id or ""),
            trace_id=str(trace_id or ""),
            ts=float(time.time()),
            regime_label=str(regime_label or ""),
            weights_json=json.dumps(self._weights, ensure_ascii=True, sort_keys=True, default=str),
            reasons_json=json.dumps(reasons or {}, ensure_ascii=True, sort_keys=True, default=str),
            source_metrics_json=json.dumps(source_metrics or {}, ensure_ascii=True, sort_keys=True, default=str),
        )
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO meta_weight_snapshots
                (run_id, trace_id, ts, regime_label, weights_json, reasons_json, source_metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(snap.run_id),
                    str(snap.trace_id),
                    float(snap.ts),
                    str(snap.regime_label),
                    str(snap.weights_json),
                    str(snap.reasons_json),
                    str(snap.source_metrics_json),
                ),
            )
            con.commit()
        return snap

    def load_from_db(self) -> None:
        self._weights.clear()
        self._states.clear()
        with self._connect() as con:
            rows = list(
                con.execute(
                    "SELECT strategy_name, smoothed_weight, state_json, last_updated_ts FROM strategy_weights"
                )
            )
        for row in rows:
            strategy = str(row["strategy_name"] or "")
            state_json = row["state_json"]
            try:
                payload = json.loads(state_json) if state_json else {}
            except Exception:
                payload = {}
            try:
                st = StrategyWeightState(
                    strategy_name=str(strategy),
                    current_weight=float(payload.get("current_weight", row["smoothed_weight"] or 0.0) or 0.0),
                    proposed_weight=float(payload.get("proposed_weight", row["smoothed_weight"] or 0.0) or 0.0),
                    smoothed_weight=float(payload.get("smoothed_weight", row["smoothed_weight"] or 0.0) or 0.0),
                    last_updated_ts=float(payload.get("last_updated_ts", row["last_updated_ts"] or 0.0) or 0.0),
                    trades_count=int(payload.get("trades_count", 0) or 0),
                    expectancy=float(payload.get("expectancy", 0.0) or 0.0),
                    profit_factor=float(payload.get("profit_factor", 0.0) or 0.0),
                    sharpe_like_score=float(payload.get("sharpe_like_score", 0.0) or 0.0),
                    drawdown_pct=float(payload.get("drawdown_pct", payload.get("max_drawdown_pct", 0.0)) or 0.0),
                    slippage_bps=float(payload.get("slippage_bps", payload.get("avg_realized_slippage_bps", 0.0)) or 0.0),
                    max_drawdown_pct=float(payload.get("max_drawdown_pct", 0.0) or 0.0),
                    avg_realized_slippage_bps=float(payload.get("avg_realized_slippage_bps", 0.0) or 0.0),
                    fee_ratio=float(payload.get("fee_ratio", 0.0) or 0.0),
                    regime_label=(str(payload.get("regime_label")) if payload.get("regime_label") else None),
                    enabled_for_weighting=bool(payload.get("enabled_for_weighting", False)),
                    reasons=list(payload.get("reasons", []) or []),
                )
            except Exception:
                st = StrategyWeightState(
                    strategy_name=str(strategy),
                    current_weight=float(row["smoothed_weight"] or 0.0),
                    proposed_weight=float(row["smoothed_weight"] or 0.0),
                    smoothed_weight=float(row["smoothed_weight"] or 0.0),
                    last_updated_ts=float(row["last_updated_ts"] or 0.0),
                    trades_count=0,
                    expectancy=0.0,
                    profit_factor=0.0,
                    sharpe_like_score=0.0,
                    drawdown_pct=0.0,
                    slippage_bps=0.0,
                    max_drawdown_pct=0.0,
                    avg_realized_slippage_bps=0.0,
                    fee_ratio=0.0,
                )
            self._states[strategy] = st
            self._weights[strategy] = float(st.smoothed_weight)
            self._last_update_ts = max(self._last_update_ts, float(st.last_updated_ts))

    def maybe_update(
        self,
        *,
        cycle_id: int,
        strategy_names: Iterable[str],
        strategy_evaluation_engine: Any,
        regime_label: str,
        execution_telemetry: Optional[Dict[str, Dict[str, Any]]] = None,
        run_id: str = "",
        trace_id: str = "",
    ) -> Dict[str, float]:
        if not self.enabled:
            return dict(self._weights)
        if int(cycle_id) <= 0:
            return dict(self._weights)
        if int(self._last_update_cycle) > 0 and int(cycle_id) - int(self._last_update_cycle) < int(self.update_interval_cycles):
            return dict(self._weights)
        try:
            weights, reasons, source_metrics = self.compute_weights(
                strategy_names=strategy_names,
                strategy_evaluation_engine=strategy_evaluation_engine,
                regime_label=str(regime_label or ""),
                execution_telemetry=execution_telemetry,
            )
            self.persist_current()
            self.persist_snapshot(
                run_id=str(run_id or ""),
                trace_id=str(trace_id or ""),
                regime_label=str(regime_label or ""),
                reasons=reasons,
                source_metrics=source_metrics,
            )
            self._last_update_cycle = int(cycle_id)
            if weights:
                logger.info("meta engine updated %s strategy weights", len(weights))
                for s, st in sorted(self._states.items()):
                    if st.reasons and any("insufficient_trades" in r for r in st.reasons):
                        logger.info("%s held at baseline due to insufficient trades", s)
                    else:
                        logger.info("%s weight %.4f -> %.4f", s, float(st.current_weight), float(st.smoothed_weight))
            return dict(self._weights)
        except Exception as e:
            logger.warning("Self-optimizing meta engine failed, using baseline/no-change weights: %s", e)
            return dict(self._weights)

    def apply_to_candidates(self, signals: Iterable[Any]) -> List[Any]:
        out: List[Any] = []
        rows = list(signals or [])
        if not rows:
            return out
        strategies = [self._strategy_name(s) for s in rows]
        baseline = self._baseline_weights(strategies)
        for sig in rows:
            strategy = self._strategy_name(sig)
            strategy_weight = float(self._weights.get(strategy, baseline.get(strategy, 0.0)) or 0.0)
            baseline_w = float(baseline.get(strategy, 1.0 / max(1, len(baseline))) or (1.0 / max(1, len(baseline))))
            ratio = strategy_weight / max(baseline_w, 1e-9)
            meta_adjust = max(0.5, min(1.5, ratio))
            base_priority = float(self._signal_get(sig, "priority_score", self._signal_get(sig, "confidence", 0.0)) or 0.0)
            weighted_priority = base_priority * max(0.0, strategy_weight)
            confidence = float(self._signal_get(sig, "confidence", 0.0) or 0.0)
            weighted_conf = max(0.0, min(1.0, confidence * meta_adjust))
            reason = "meta_weight_applied"
            if strategy not in self._weights:
                reason = "meta_baseline_fallback"

            if isinstance(sig, dict):
                sig["strategy_weight"] = float(strategy_weight)
                sig["meta_priority_adjustment"] = float(meta_adjust)
                sig["weighting_reason"] = str(reason)
                sig["priority_score"] = float(weighted_priority)
                sig["confidence"] = float(weighted_conf)
            else:
                setattr(sig, "strategy_weight", float(strategy_weight))
                setattr(sig, "meta_priority_adjustment", float(meta_adjust))
                setattr(sig, "weighting_reason", str(reason))
                setattr(sig, "priority_score", float(weighted_priority))
                setattr(sig, "confidence", float(weighted_conf))
            out.append(sig)
        return out

    def current_strategy_weights(self) -> Dict[str, float]:
        return dict(self._weights)

    def last_update_time(self) -> float:
        return float(self._last_update_ts)

    def top_weighted_strategies(self, *, limit: int = 5) -> List[Tuple[str, float]]:
        lim = max(1, int(limit or 5))
        rows = sorted(self._weights.items(), key=lambda kv: float(kv[1]), reverse=True)
        return [(str(k), float(v)) for k, v in rows[:lim]]

    def biggest_recent_weight_changes(self, *, limit: int = 5) -> List[Tuple[str, float]]:
        lim = max(1, int(limit or 5))
        rows = sorted(self._weight_change_cache.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
        return [(str(k), float(v)) for k, v in rows[:lim]]

    def baseline_pinned_strategies(self) -> List[str]:
        out: List[str] = []
        for s, st in self._states.items():
            if any("insufficient_trades" in str(r) for r in list(st.reasons or [])):
                out.append(str(s))
        return sorted(out)
