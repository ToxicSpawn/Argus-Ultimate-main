from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StrategyMetrics:
    strategy_name: str
    symbol: str
    trades_count: int
    wins_count: int
    losses_count: int
    win_rate: float
    gross_pnl_aud: float
    net_pnl_aud: float
    total_fees_aud: float
    avg_net_pnl_per_trade: float
    avg_expected_net_edge_bps: float
    avg_realized_slippage_bps: float
    avg_hold_time_seconds: float
    max_drawdown_pct: float
    profit_factor: float
    expectancy: float
    sharpe_like_score: float
    last_updated_ts: float
    regime_label: Optional[str] = None
    enabled_for_ranking: bool = False


@dataclass(slots=True)
class _MetricState:
    trades_count: int = 0
    wins_count: int = 0
    losses_count: int = 0
    gross_pnl_aud: float = 0.0
    net_pnl_aud: float = 0.0
    total_fees_aud: float = 0.0
    sum_expected_net_edge_bps: float = 0.0
    sum_realized_slippage_bps: float = 0.0
    sum_hold_time_seconds: float = 0.0
    wins_pnl_sum: float = 0.0
    losses_pnl_sum: float = 0.0
    gross_profit_sum: float = 0.0
    gross_loss_sum: float = 0.0
    cumulative_net_pnl: float = 0.0
    peak_cumulative_net_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    pnl_series: List[float] = None  # type: ignore[assignment]
    last_updated_ts: float = 0.0

    def __post_init__(self) -> None:
        if self.pnl_series is None:
            self.pnl_series = []


class StrategyEvaluationEngine:
    """
    Deterministic strategy evaluation spine.

    Tracks per-strategy and per-strategy-per-symbol metrics in memory and persists
    snapshots to SQLite for reload and reporting.
    """

    GLOBAL_SYMBOL = "__ALL__"

    def __init__(
        self,
        *,
        db_path: str = "data/strategy_metrics.db",
        enabled: bool = True,
        persist_interval_cycles: int = 10,
        min_trades_for_ranking: int = 0,
        use_regime_scoped_metrics: bool = True,
        sharpe_like_min_trades: int = 5,
        max_metrics_history_points: int = 500,
    ) -> None:
        self.db_path = str(db_path or "data/strategy_metrics.db")
        self.enabled = bool(enabled)
        self.persist_interval_cycles = max(1, int(persist_interval_cycles or 10))
        self.min_trades_for_ranking = max(0, int(min_trades_for_ranking or 0))
        self.use_regime_scoped_metrics = bool(use_regime_scoped_metrics)
        self.sharpe_like_min_trades = max(1, int(sharpe_like_min_trades or 5))
        self.max_metrics_history_points = max(20, int(max_metrics_history_points or 500))
        self._states: Dict[Tuple[str, str, str], _MetricState] = {}
        self._metrics: Dict[Tuple[str, str, str], StrategyMetrics] = {}
        self._open_entries: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        self._last_persist_cycle: int = 0
        self._init_schema()
        self.load_from_db()

    def _connect(self) -> sqlite3.Connection:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_metrics (
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    regime_label TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    last_updated_ts REAL NOT NULL,
                    PRIMARY KEY (strategy_name, symbol, regime_label)
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_strategy_metrics_updated ON strategy_metrics(last_updated_ts)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_strategy_metrics_strategy ON strategy_metrics(strategy_name)")
            conn.commit()

    @staticmethod
    def _norm_strategy(name: str) -> str:
        s = str(name or "").strip()
        return s or "unknown"

    @staticmethod
    def _norm_symbol(symbol: Optional[str]) -> str:
        s = str(symbol or "").strip()
        return s or StrategyEvaluationEngine.GLOBAL_SYMBOL

    @staticmethod
    def _norm_regime(regime_label: Optional[str]) -> str:
        s = str(regime_label or "").strip()
        return s

    def _key(self, strategy_name: str, symbol: Optional[str], regime_label: Optional[str]) -> Tuple[str, str, str]:
        return (
            self._norm_strategy(strategy_name),
            self._norm_symbol(symbol),
            self._norm_regime(regime_label),
        )

    def _state_for(self, key: Tuple[str, str, str]) -> _MetricState:
        st = self._states.get(key)
        if st is None:
            st = _MetricState()
            self._states[key] = st
        return st

    def _build_metrics(self, key: Tuple[str, str, str], st: _MetricState) -> StrategyMetrics:
        strategy_name, symbol, regime_label = key
        trades = max(0, int(st.trades_count))
        wins = max(0, int(st.wins_count))
        losses = max(0, int(st.losses_count))
        win_rate = float(wins / trades) if trades > 0 else 0.0
        avg_net = float(st.net_pnl_aud / trades) if trades > 0 else 0.0
        avg_edge = float(st.sum_expected_net_edge_bps / trades) if trades > 0 else 0.0
        avg_slip = float(st.sum_realized_slippage_bps / trades) if trades > 0 else 0.0
        avg_hold = float(st.sum_hold_time_seconds / trades) if trades > 0 else 0.0
        if losses > 0:
            profit_factor = float(st.gross_profit_sum / max(st.gross_loss_sum, 1e-9))
        else:
            profit_factor = float(st.gross_profit_sum) if st.gross_profit_sum > 0 else 0.0
        avg_win = float(st.wins_pnl_sum / wins) if wins > 0 else 0.0
        avg_loss = float(st.losses_pnl_sum / losses) if losses > 0 else 0.0
        expectancy = float((win_rate * avg_win) + ((1.0 - win_rate) * avg_loss))

        sharpe_like = 0.0
        if trades >= self.sharpe_like_min_trades and len(st.pnl_series) >= self.sharpe_like_min_trades:
            arr = np.asarray(st.pnl_series, dtype=float)
            std = float(np.std(arr))
            if std > 1e-12:
                sharpe_like = float((np.mean(arr) / std) * math.sqrt(len(arr)))

        enabled_for_ranking = bool(trades >= self.min_trades_for_ranking)
        return StrategyMetrics(
            strategy_name=strategy_name,
            symbol=symbol,
            trades_count=trades,
            wins_count=wins,
            losses_count=losses,
            win_rate=win_rate,
            gross_pnl_aud=float(st.gross_pnl_aud),
            net_pnl_aud=float(st.net_pnl_aud),
            total_fees_aud=float(st.total_fees_aud),
            avg_net_pnl_per_trade=avg_net,
            avg_expected_net_edge_bps=avg_edge,
            avg_realized_slippage_bps=avg_slip,
            avg_hold_time_seconds=avg_hold,
            max_drawdown_pct=float(st.max_drawdown_pct),
            profit_factor=float(profit_factor),
            expectancy=float(expectancy),
            sharpe_like_score=float(sharpe_like),
            last_updated_ts=float(st.last_updated_ts),
            regime_label=(regime_label if regime_label else None),
            enabled_for_ranking=enabled_for_ranking,
        )

    def _apply_trade_update(
        self,
        *,
        strategy_name: str,
        symbol: str,
        regime_label: Optional[str],
        gross_pnl_aud: float,
        net_pnl_aud: float,
        fees_aud: float,
        expected_net_edge_bps: float,
        realized_slippage_bps: float,
        hold_time_seconds: float,
        ts: float,
    ) -> None:
        keys: List[Tuple[str, str, str]] = [
            self._key(strategy_name, self.GLOBAL_SYMBOL, None),
            self._key(strategy_name, symbol, None),
        ]
        if self.use_regime_scoped_metrics and regime_label:
            keys.extend(
                [
                    self._key(strategy_name, self.GLOBAL_SYMBOL, regime_label),
                    self._key(strategy_name, symbol, regime_label),
                ]
            )

        for key in keys:
            st = self._state_for(key)
            st.trades_count += 1
            st.gross_pnl_aud += float(gross_pnl_aud)
            st.net_pnl_aud += float(net_pnl_aud)
            st.total_fees_aud += max(0.0, float(fees_aud))
            st.sum_expected_net_edge_bps += float(expected_net_edge_bps)
            st.sum_realized_slippage_bps += float(realized_slippage_bps)
            st.sum_hold_time_seconds += max(0.0, float(hold_time_seconds))
            st.cumulative_net_pnl += float(net_pnl_aud)
            st.peak_cumulative_net_pnl = max(st.peak_cumulative_net_pnl, st.cumulative_net_pnl)
            drawdown_den = max(1.0, abs(st.peak_cumulative_net_pnl))
            drawdown_pct = max(0.0, (st.peak_cumulative_net_pnl - st.cumulative_net_pnl) / drawdown_den * 100.0)
            st.max_drawdown_pct = max(st.max_drawdown_pct, drawdown_pct)
            st.pnl_series.append(float(net_pnl_aud))
            if len(st.pnl_series) > self.max_metrics_history_points:
                st.pnl_series = st.pnl_series[-self.max_metrics_history_points :]
            if float(net_pnl_aud) > 0.0:
                st.wins_count += 1
                st.wins_pnl_sum += float(net_pnl_aud)
                st.gross_profit_sum += float(max(gross_pnl_aud, 0.0))
            elif float(net_pnl_aud) < 0.0:
                st.losses_count += 1
                st.losses_pnl_sum += float(net_pnl_aud)
                st.gross_loss_sum += float(abs(min(gross_pnl_aud, 0.0)))
            st.last_updated_ts = float(ts)
            metrics = self._build_metrics(key, st)
            self._metrics[key] = metrics

    def record_open(
        self,
        *,
        strategy_name: str,
        symbol: str,
        quantity: float,
        ts: Optional[float] = None,
        regime_label: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        qty = float(quantity or 0.0)
        if qty <= 0.0:
            return
        ts_now = float(ts if ts is not None else time.time())
        strategy = self._norm_strategy(strategy_name)
        symbol_n = self._norm_symbol(symbol)
        regime_n = self._norm_regime(regime_label)
        keys: List[Tuple[str, str, str]] = [
            self._key(strategy, self.GLOBAL_SYMBOL, None),
            self._key(strategy, symbol_n, None),
        ]
        if self.use_regime_scoped_metrics and regime_n:
            keys.extend(
                [
                    self._key(strategy, self.GLOBAL_SYMBOL, regime_n),
                    self._key(strategy, symbol_n, regime_n),
                ]
            )
        for metric_key in keys:
            st = self._state_for(metric_key)
            if st.last_updated_ts <= 0:
                st.last_updated_ts = ts_now
            self._metrics.setdefault(metric_key, self._build_metrics(metric_key, st))
        key = (self._norm_strategy(strategy_name), self._norm_symbol(symbol))
        row = {
            "qty": qty,
            "ts": ts_now,
            "regime": self._norm_regime(regime_label),
        }
        self._open_entries.setdefault(key, []).append(row)

    def consume_hold_time_seconds(
        self,
        *,
        strategy_name: str,
        symbol: str,
        quantity: float,
        ts: Optional[float] = None,
    ) -> float:
        if not self.enabled:
            return 0.0
        qty = max(0.0, float(quantity or 0.0))
        if qty <= 0.0:
            return 0.0
        now = float(ts if ts is not None else time.time())
        key = (self._norm_strategy(strategy_name), self._norm_symbol(symbol))
        entries = self._open_entries.get(key) or []
        if not entries:
            return 0.0
        remaining = qty
        weighted_sum = 0.0
        consumed = 0.0
        while remaining > 1e-12 and entries:
            top = entries[0]
            top_qty = max(0.0, float(top.get("qty", 0.0) or 0.0))
            top_ts = float(top.get("ts", now) or now)
            if top_qty <= 1e-12:
                entries.pop(0)
                continue
            take = min(remaining, top_qty)
            hold = max(0.0, now - top_ts)
            weighted_sum += hold * take
            consumed += take
            top["qty"] = top_qty - take
            remaining -= take
            if float(top.get("qty", 0.0) or 0.0) <= 1e-12:
                entries.pop(0)
        if not entries:
            self._open_entries.pop(key, None)
        if consumed <= 1e-12:
            return 0.0
        return float(weighted_sum / consumed)

    def record_trade_close(
        self,
        *,
        strategy_name: str,
        symbol: str,
        gross_pnl_aud: float,
        net_pnl_aud: float,
        fees_aud: float,
        expected_net_edge_bps: float,
        realized_slippage_bps: float,
        hold_time_seconds: float,
        regime_label: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return
        strategy = self._norm_strategy(strategy_name)
        symbol_n = self._norm_symbol(symbol)
        now = float(ts if ts is not None else time.time())
        self._apply_trade_update(
            strategy_name=strategy,
            symbol=symbol_n,
            regime_label=regime_label,
            gross_pnl_aud=float(gross_pnl_aud),
            net_pnl_aud=float(net_pnl_aud),
            fees_aud=float(fees_aud),
            expected_net_edge_bps=float(expected_net_edge_bps),
            realized_slippage_bps=float(realized_slippage_bps),
            hold_time_seconds=float(hold_time_seconds),
            ts=now,
        )
        snap = self.get_metrics(strategy_name=strategy, symbol=symbol_n, regime_label=regime_label)
        if snap is not None:
            logger.info(
                "strategy metrics updated for %s on %s (expectancy=%.4f, profit_factor=%.4f)",
                strategy,
                symbol_n,
                float(snap.expectancy),
                float(snap.profit_factor),
            )

    def get_metrics(
        self,
        *,
        strategy_name: str,
        symbol: Optional[str] = None,
        regime_label: Optional[str] = None,
    ) -> Optional[StrategyMetrics]:
        candidates = [
            self._key(strategy_name, symbol, regime_label if (self.use_regime_scoped_metrics and regime_label) else None),
            self._key(strategy_name, symbol, None),
            self._key(strategy_name, self.GLOBAL_SYMBOL, regime_label if (self.use_regime_scoped_metrics and regime_label) else None),
            self._key(strategy_name, self.GLOBAL_SYMBOL, None),
        ]
        for key in candidates:
            metrics = self._metrics.get(key)
            if metrics is not None:
                return metrics
        return None

    def get_decision_context(
        self,
        *,
        strategy_name: str,
        symbol: Optional[str] = None,
        regime_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        metrics = self.get_metrics(strategy_name=strategy_name, symbol=symbol, regime_label=regime_label)
        if metrics is None:
            return {}
        return {
            "strategy_trades_count": int(metrics.trades_count),
            "strategy_win_rate": float(metrics.win_rate),
            "strategy_expectancy": float(metrics.expectancy),
            "strategy_profit_factor": float(metrics.profit_factor),
        }

    def _iter_global_metrics(self, *, regime_label: Optional[str] = None) -> Iterable[StrategyMetrics]:
        reg = self._norm_regime(regime_label)
        for key, metrics in self._metrics.items():
            _strategy, symbol, regime = key
            if symbol != self.GLOBAL_SYMBOL:
                continue
            if reg and regime != reg:
                continue
            if (not reg) and regime:
                continue
            yield metrics

    def _top_by(self, metric_name: str, *, limit: int, reverse: bool = True, regime_label: Optional[str] = None) -> List[StrategyMetrics]:
        lim = max(1, int(limit or 5))
        rows = list(self._iter_global_metrics(regime_label=regime_label))
        rows = [m for m in rows if bool(m.enabled_for_ranking)]
        rows.sort(key=lambda m: float(getattr(m, metric_name, 0.0) or 0.0), reverse=reverse)
        return rows[:lim]

    def top_by_net_pnl(self, *, limit: int = 5, regime_label: Optional[str] = None) -> List[StrategyMetrics]:
        return self._top_by("net_pnl_aud", limit=limit, reverse=True, regime_label=regime_label)

    def top_by_expectancy(self, *, limit: int = 5, regime_label: Optional[str] = None) -> List[StrategyMetrics]:
        return self._top_by("expectancy", limit=limit, reverse=True, regime_label=regime_label)

    def top_by_sharpe_like(self, *, limit: int = 5, regime_label: Optional[str] = None) -> List[StrategyMetrics]:
        return self._top_by("sharpe_like_score", limit=limit, reverse=True, regime_label=regime_label)

    def worst_by_drawdown(self, *, limit: int = 5, regime_label: Optional[str] = None) -> List[StrategyMetrics]:
        return self._top_by("max_drawdown_pct", limit=limit, reverse=True, regime_label=regime_label)

    def worst_by_expectancy(self, *, limit: int = 5, regime_label: Optional[str] = None) -> List[StrategyMetrics]:
        return self._top_by("expectancy", limit=limit, reverse=False, regime_label=regime_label)

    def rankable_strategy_count(self) -> int:
        return len([m for m in self._iter_global_metrics() if bool(m.enabled_for_ranking)])

    def maybe_persist(self, *, cycle_id: int) -> None:
        if not self.enabled:
            return
        if int(cycle_id) <= 0:
            return
        if (int(cycle_id) - int(self._last_persist_cycle)) < int(self.persist_interval_cycles):
            return
        self.persist_to_db()
        self._last_persist_cycle = int(cycle_id)

    def persist_to_db(self) -> None:
        if not self.enabled:
            return
        with self._connect() as conn:
            cur = conn.cursor()
            for key, metrics in self._metrics.items():
                st = self._states.get(key) or _MetricState()
                strategy_name, symbol, regime_label = key
                cur.execute(
                    """
                    INSERT INTO strategy_metrics (strategy_name, symbol, regime_label, metrics_json, state_json, last_updated_ts)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(strategy_name, symbol, regime_label) DO UPDATE SET
                        metrics_json=excluded.metrics_json,
                        state_json=excluded.state_json,
                        last_updated_ts=excluded.last_updated_ts
                    """,
                    (
                        str(strategy_name),
                        str(symbol),
                        str(regime_label),
                        json.dumps(asdict(metrics), ensure_ascii=True, default=str),
                        json.dumps(asdict(st), ensure_ascii=True, default=str),
                        float(metrics.last_updated_ts or time.time()),
                    ),
                )
            conn.commit()

    def load_from_db(self) -> None:
        self._states.clear()
        self._metrics.clear()
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT strategy_name, symbol, regime_label, metrics_json, state_json FROM strategy_metrics")
            rows = cur.fetchall()
        for row in rows:
            strategy_name = str(row["strategy_name"] or "")
            symbol = str(row["symbol"] or "")
            regime_label = str(row["regime_label"] or "")
            key = (strategy_name, symbol, regime_label)
            state_raw = row["state_json"] or "{}"
            metrics_raw = row["metrics_json"] or "{}"
            try:
                st_dict = json.loads(state_raw)
            except Exception:
                st_dict = {}
            try:
                m_dict = json.loads(metrics_raw)
            except Exception:
                m_dict = {}
            try:
                st = _MetricState(**{k: v for k, v in dict(st_dict or {}).items() if k in _MetricState.__dataclass_fields__})
            except Exception:
                st = _MetricState()
            if len(st.pnl_series) > self.max_metrics_history_points:
                st.pnl_series = list(st.pnl_series)[-self.max_metrics_history_points :]
            self._states[key] = st
            try:
                metrics = StrategyMetrics(**{k: v for k, v in dict(m_dict or {}).items() if k in StrategyMetrics.__dataclass_fields__})
            except Exception:
                metrics = self._build_metrics(key, st)
            self._metrics[key] = metrics
