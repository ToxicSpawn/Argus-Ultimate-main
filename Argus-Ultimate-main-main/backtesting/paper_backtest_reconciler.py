"""
Paper-trading vs backtest reconciliation tool.

Compares paper-trading results (from TradeLedger SQLite) against backtest
results (from WalkForwardBacktester) to quantify how realistic the backtest
is and identify sources of divergence (slippage, timing, missed fills, etc.).

Usage:
    from backtesting.paper_backtest_reconciler import PaperBacktestReconciler

    reconciler = PaperBacktestReconciler(
        paper_ledger_path="data/unified_trades.db",
        backtest_results=backtest_result_dict,
    )
    report = reconciler.reconcile()
    print(report.summary)
    sources = reconciler.identify_divergence_sources()
    ok = reconciler.is_backtest_realistic(max_divergence_pct=15.0)
"""
from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReconciliationReport:
    """Structured comparison between paper-trading and backtest results."""

    trade_count_paper: int = 0
    trade_count_backtest: int = 0

    total_pnl_paper: float = 0.0
    total_pnl_backtest: float = 0.0
    pnl_divergence_pct: float = 0.0

    win_rate_paper: float = 0.0
    win_rate_backtest: float = 0.0

    avg_holding_period_paper: float = 0.0  # seconds
    avg_holding_period_backtest: float = 0.0  # seconds

    sharpe_paper: float = 0.0
    sharpe_backtest: float = 0.0

    max_drawdown_paper: float = 0.0
    max_drawdown_backtest: float = 0.0

    signal_agreement_pct: float = 0.0
    slippage_impact_bps: float = 0.0

    divergent_trades: List[dict] = field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------

class PaperBacktestReconciler:
    """Compare paper-trading ledger against walk-forward backtest results.

    Parameters
    ----------
    paper_ledger_path : str
        Path to the TradeLedger SQLite database (``data/unified_trades.db``).
    backtest_results : dict
        A dictionary (or ``BacktestResult.__dict__``) from
        ``WalkForwardBacktester.run()``.  Expected keys include
        ``all_trades``, ``combined_sharpe``, ``combined_max_drawdown_pct``,
        ``combined_win_rate``, ``total_trades``, and optionally
        ``combined_equity_curve``.
    """

    def __init__(
        self,
        paper_ledger_path: str,
        backtest_results: dict,
    ) -> None:
        self._ledger_path = paper_ledger_path
        self._bt = backtest_results
        self._paper_trades: List[Dict[str, Any]] = []
        self._bt_trades: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_paper_trades(self) -> List[Dict[str, Any]]:
        """Read trades from the TradeLedger SQLite file."""
        try:
            conn = sqlite3.connect(self._ledger_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM trades ORDER BY timestamp ASC"
            )
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            return rows
        except Exception as exc:
            logger.warning("Could not read paper ledger: %s", exc)
            return []

    def _normalise_bt_trades(self) -> List[Dict[str, Any]]:
        """Normalise backtest trades into a flat dict list.

        BacktestResult stores trades as ``Trade`` dataclass instances with
        fields ``entry_time``, ``exit_time``, ``side``, ``entry_price``,
        ``exit_price``, ``quantity``, ``pnl_usd``, ``impact_cost_usd``, etc.
        They may arrive as dicts if already serialised.
        """
        raw = self._bt.get("all_trades", [])
        out: List[Dict[str, Any]] = []
        for t in raw:
            if isinstance(t, dict):
                d = t
            elif hasattr(t, "__dict__"):
                d = t.__dict__
            else:
                continue
            out.append({
                "timestamp": d.get("entry_time", 0.0),
                "exit_time": d.get("exit_time"),
                "symbol": d.get("symbol", ""),
                "side": str(d.get("side", "")).upper(),
                "price": float(d.get("entry_price", 0.0)),
                "exit_price": d.get("exit_price"),
                "size": float(d.get("quantity", 0.0)),
                "pnl": float(d.get("pnl_usd", 0.0)),
                "impact_cost": float(d.get("impact_cost_usd", 0.0)),
                "exit_reason": d.get("exit_reason", ""),
            })
        return out

    @staticmethod
    def _compute_win_rate(trades: List[Dict[str, Any]], pnl_key: str = "pnl") -> float:
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if (t.get(pnl_key) or 0.0) > 0)
        return wins / len(trades)

    @staticmethod
    def _compute_avg_holding(trades: List[Dict[str, Any]]) -> float:
        """Average holding period in seconds.  Handles both paper and BT formats."""
        durations: List[float] = []
        for t in trades:
            entry = t.get("timestamp") or t.get("entry_time") or 0.0
            exit_ = t.get("exit_time")
            if exit_ is not None and entry:
                dur = float(exit_) - float(entry)
                if dur > 0:
                    durations.append(dur)
        return sum(durations) / len(durations) if durations else 0.0

    @staticmethod
    def _compute_sharpe(pnl_series: List[float], periods_per_year: int = 8760) -> float:
        """Annualised Sharpe from a series of per-trade PnL values."""
        if len(pnl_series) < 2:
            return 0.0
        import numpy as np  # deferred import to keep module lightweight
        arr = np.array(pnl_series, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        if std < 1e-12:
            return 0.0
        return (mean / std) * math.sqrt(periods_per_year)

    @staticmethod
    def _compute_max_drawdown(pnl_series: List[float]) -> float:
        """Max drawdown as a fraction (0.0 .. 1.0) from cumulative PnL."""
        if not pnl_series:
            return 0.0
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnl_series:
            cum += pnl
            if cum > peak:
                peak = cum
            if peak > 0:
                dd = (peak - cum) / peak
                if dd > max_dd:
                    max_dd = dd
        return max_dd

    @staticmethod
    def _pct_divergence(a: float, b: float) -> float:
        """Percentage divergence between two values.  Uses the larger absolute
        value as the denominator to avoid division-by-zero."""
        denom = max(abs(a), abs(b))
        if denom < 1e-12:
            return 0.0
        return abs(a - b) / denom * 100.0

    # ------------------------------------------------------------------
    # Pair matching for signal agreement
    # ------------------------------------------------------------------

    def _pair_trades(
        self,
        paper: List[Dict[str, Any]],
        bt: List[Dict[str, Any]],
        time_tolerance: float = 3600.0,
    ) -> List[Dict[str, Any]]:
        """Pair paper trades with backtest trades by timestamp proximity.

        Returns a list of dicts with keys ``paper``, ``backtest``, and
        ``matched`` (bool).
        """
        used_bt: set = set()
        pairs: List[Dict[str, Any]] = []

        for pt in paper:
            pt_ts = float(pt.get("timestamp", 0))
            pt_sym = str(pt.get("symbol", ""))
            best_idx: Optional[int] = None
            best_dt: float = float("inf")

            for i, bt_t in enumerate(bt):
                if i in used_bt:
                    continue
                bt_ts = float(bt_t.get("timestamp", 0))
                bt_sym = str(bt_t.get("symbol", ""))
                if pt_sym and bt_sym and pt_sym != bt_sym:
                    continue
                dt = abs(pt_ts - bt_ts)
                if dt < best_dt and dt <= time_tolerance:
                    best_dt = dt
                    best_idx = i

            if best_idx is not None:
                used_bt.add(best_idx)
                pairs.append({
                    "paper": pt,
                    "backtest": bt[best_idx],
                    "matched": True,
                })
            else:
                pairs.append({
                    "paper": pt,
                    "backtest": None,
                    "matched": False,
                })

        # Unmatched backtest trades
        for i, bt_t in enumerate(bt):
            if i not in used_bt:
                pairs.append({
                    "paper": None,
                    "backtest": bt_t,
                    "matched": False,
                })

        return pairs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconcile(self) -> ReconciliationReport:
        """Run the full reconciliation and return a ``ReconciliationReport``."""
        self._paper_trades = self._load_paper_trades()
        self._bt_trades = self._normalise_bt_trades()

        paper = self._paper_trades
        bt = self._bt_trades

        report = ReconciliationReport()

        # --- trade counts ---
        report.trade_count_paper = len(paper)
        report.trade_count_backtest = len(bt)

        # --- PnL ---
        paper_pnls = [float(t.get("pnl") or 0.0) for t in paper]
        bt_pnls = [float(t.get("pnl") or 0.0) for t in bt]

        report.total_pnl_paper = sum(paper_pnls)
        report.total_pnl_backtest = sum(bt_pnls)
        report.pnl_divergence_pct = self._pct_divergence(
            report.total_pnl_paper, report.total_pnl_backtest
        )

        # --- win rates ---
        report.win_rate_paper = self._compute_win_rate(paper)
        report.win_rate_backtest = self._compute_win_rate(bt)

        # --- holding periods ---
        report.avg_holding_period_paper = self._compute_avg_holding(paper)
        report.avg_holding_period_backtest = self._compute_avg_holding(bt)

        # --- Sharpe ---
        report.sharpe_paper = self._compute_sharpe(paper_pnls)
        # Prefer the pre-computed backtest Sharpe if available.
        bt_sharpe = self._bt.get("combined_sharpe")
        if bt_sharpe is not None:
            report.sharpe_backtest = float(bt_sharpe)
        else:
            report.sharpe_backtest = self._compute_sharpe(bt_pnls)

        # --- max drawdown ---
        report.max_drawdown_paper = self._compute_max_drawdown(paper_pnls)
        bt_mdd = self._bt.get("combined_max_drawdown_pct")
        if bt_mdd is not None:
            report.max_drawdown_backtest = float(bt_mdd)
        else:
            report.max_drawdown_backtest = self._compute_max_drawdown(bt_pnls)

        # --- signal agreement & slippage ---
        pairs = self._pair_trades(paper, bt)
        matched_pairs = [p for p in pairs if p["matched"]]
        if matched_pairs:
            agreements = 0
            total_slippage_bps = 0.0
            for pair in matched_pairs:
                p_side = str(pair["paper"].get("side", "")).upper()
                b_side = str(pair["backtest"].get("side", "")).upper()
                if p_side == b_side:
                    agreements += 1

                p_price = float(pair["paper"].get("price") or 0.0)
                b_price = float(pair["backtest"].get("price") or 0.0)
                if b_price > 0:
                    slip = abs(p_price - b_price) / b_price * 10_000
                    total_slippage_bps += slip

            report.signal_agreement_pct = (agreements / len(matched_pairs)) * 100.0
            report.slippage_impact_bps = total_slippage_bps / len(matched_pairs)
        else:
            report.signal_agreement_pct = 0.0
            report.slippage_impact_bps = 0.0

        # --- divergent trades ---
        divergent: List[dict] = []
        for pair in pairs:
            if not pair["matched"]:
                divergent.append({
                    "type": "unmatched",
                    "paper": pair["paper"],
                    "backtest": pair["backtest"],
                    "reason": "missed_fill" if pair["paper"] else "extra_backtest_trade",
                })
            elif pair["paper"] and pair["backtest"]:
                p_side = str(pair["paper"].get("side", "")).upper()
                b_side = str(pair["backtest"].get("side", "")).upper()
                p_pnl = float(pair["paper"].get("pnl") or 0.0)
                b_pnl = float(pair["backtest"].get("pnl") or 0.0)
                pnl_div = self._pct_divergence(p_pnl, b_pnl)

                if p_side != b_side or pnl_div > 25.0:
                    divergent.append({
                        "type": "direction_mismatch" if p_side != b_side else "pnl_mismatch",
                        "paper": pair["paper"],
                        "backtest": pair["backtest"],
                        "pnl_divergence_pct": pnl_div,
                        "reason": (
                            f"direction: paper={p_side} vs bt={b_side}"
                            if p_side != b_side
                            else f"pnl divergence {pnl_div:.1f}%"
                        ),
                    })
        report.divergent_trades = divergent

        # --- summary ---
        lines = [
            "=== Paper vs Backtest Reconciliation ===",
            f"Trades: paper={report.trade_count_paper}, backtest={report.trade_count_backtest}",
            f"Total PnL: paper=${report.total_pnl_paper:,.2f}, backtest=${report.total_pnl_backtest:,.2f} "
            f"(divergence={report.pnl_divergence_pct:.1f}%)",
            f"Win rate: paper={report.win_rate_paper:.1%}, backtest={report.win_rate_backtest:.1%}",
            f"Sharpe: paper={report.sharpe_paper:.3f}, backtest={report.sharpe_backtest:.3f}",
            f"Max DD: paper={report.max_drawdown_paper:.2%}, backtest={report.max_drawdown_backtest:.2%}",
            f"Signal agreement: {report.signal_agreement_pct:.1f}%",
            f"Avg slippage impact: {report.slippage_impact_bps:.1f} bps",
            f"Divergent trades: {len(report.divergent_trades)}",
        ]
        report.summary = "\n".join(lines)

        return report

    def identify_divergence_sources(self) -> List[str]:
        """Categorise the main reasons paper and backtest results differ.

        Must call ``reconcile()`` first (otherwise returns generic list).
        """
        if not self._paper_trades and not self._bt_trades:
            # reconcile() not called yet — load data
            self._paper_trades = self._load_paper_trades()
            self._bt_trades = self._normalise_bt_trades()

        sources: List[str] = []
        paper = self._paper_trades
        bt = self._bt_trades

        # 1. Trade count mismatch → missed fills or extra signals
        count_diff = abs(len(paper) - len(bt))
        if count_diff > 0:
            pct = count_diff / max(len(paper), len(bt), 1) * 100
            if len(paper) < len(bt):
                sources.append(
                    f"missed_fills: paper has {count_diff} fewer trades "
                    f"({pct:.0f}% of backtest count) — likely unfilled limit orders "
                    f"or exchange rejections"
                )
            else:
                sources.append(
                    f"extra_paper_trades: paper has {count_diff} more trades "
                    f"({pct:.0f}%) — backtest may not model all signal triggers"
                )

        # 2. Slippage
        pairs = self._pair_trades(paper, bt)
        matched = [p for p in pairs if p["matched"]]
        slip_bps_list: List[float] = []
        for pair in matched:
            p_price = float(pair["paper"].get("price") or 0.0)
            b_price = float(pair["backtest"].get("price") or 0.0)
            if b_price > 0:
                slip_bps_list.append(abs(p_price - b_price) / b_price * 10_000)

        if slip_bps_list:
            avg_slip = sum(slip_bps_list) / len(slip_bps_list)
            max_slip = max(slip_bps_list)
            if avg_slip > 1.0:
                sources.append(
                    f"slippage: avg={avg_slip:.1f} bps, max={max_slip:.1f} bps — "
                    f"live execution received worse prices than backtest assumed"
                )

        # 3. Timing differences
        timing_deltas: List[float] = []
        for pair in matched:
            pt_ts = float(pair["paper"].get("timestamp") or 0)
            bt_ts = float(pair["backtest"].get("timestamp") or 0)
            if pt_ts > 0 and bt_ts > 0:
                timing_deltas.append(abs(pt_ts - bt_ts))
        if timing_deltas:
            avg_delay = sum(timing_deltas) / len(timing_deltas)
            if avg_delay > 60:
                sources.append(
                    f"timing_delay: avg={avg_delay:.0f}s between backtest signal "
                    f"and paper fill — latency or order queue delays"
                )

        # 4. Direction disagreements
        dir_mismatches = 0
        for pair in matched:
            p_side = str(pair["paper"].get("side", "")).upper()
            b_side = str(pair["backtest"].get("side", "")).upper()
            if p_side != b_side:
                dir_mismatches += 1
        if dir_mismatches > 0:
            sources.append(
                f"direction_mismatch: {dir_mismatches} trades where paper and "
                f"backtest took opposite sides — signal interpretation differs"
            )

        # 5. Commission / fee differences
        paper_commissions = sum(float(t.get("commission") or 0.0) for t in paper)
        bt_impact = sum(float(t.get("impact_cost") or 0.0) for t in self._bt_trades)
        fee_diff = abs(paper_commissions - bt_impact)
        if fee_diff > 0.01:
            sources.append(
                f"fee_model: paper commissions=${paper_commissions:.2f} vs "
                f"backtest impact costs=${bt_impact:.2f} — "
                f"cost model calibration may differ"
            )

        if not sources:
            sources.append("no_significant_divergence: results are closely aligned")

        return sources

    def is_backtest_realistic(self, max_divergence_pct: float = 15.0) -> bool:
        """Return True if paper and backtest PnL are within tolerance.

        This is a convenience check: if ``pnl_divergence_pct`` exceeds
        *max_divergence_pct* the backtest is considered unrealistic.

        If ``reconcile()`` has not been called yet, it is called automatically.
        """
        report = self.reconcile()
        return report.pnl_divergence_pct <= max_divergence_pct
