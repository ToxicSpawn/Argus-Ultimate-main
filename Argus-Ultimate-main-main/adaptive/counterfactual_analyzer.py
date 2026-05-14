#!/usr/bin/env python3
"""
Counterfactual Analyzer — "What If" analysis for trades and skipped signals.

For every trade, answers: what if we held longer, used a different size,
routed to a different venue, or used a limit order?  For every skipped signal,
answers: what would have happened if we took it?

Identifies biggest missed opportunities and biggest mistakes.
SQLite persistence at data/counterfactuals.db.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualReport:
    """What-if analysis of a single trade or skipped signal."""
    trade_id: str
    actual_pnl: float = 0.0
    what_if_held_longer: Dict[str, float] = field(default_factory=dict)  # {"1h": pnl, "4h": pnl, "24h": pnl}
    what_if_larger_size: float = 0.0
    what_if_smaller_size: float = 0.0
    what_if_different_venue: float = 0.0
    what_if_limit_order: float = 0.0
    optimal_action: str = ""
    lesson: str = ""
    opportunity_cost: float = 0.0
    analyzed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TradeRecord:
    """Minimal trade record for counterfactual analysis."""
    trade_id: str
    symbol: str
    side: str  # "buy" / "sell"
    entry_price: float
    exit_price: float
    size: float
    entry_time: str
    exit_time: str
    venue: str = ""
    slippage_bps: float = 0.0
    was_limit: bool = False


@dataclass
class SkippedSignal:
    """Record of a signal that was not acted upon."""
    signal_id: str
    symbol: str
    direction: str  # "long" / "short"
    signal_score: float
    price_at_signal: float
    timestamp: str
    reason_skipped: str = ""


@dataclass
class PriceSnapshot:
    """Price data at specific horizons after a trade or signal."""
    symbol: str
    base_price: float
    prices_after: Dict[str, float] = field(default_factory=dict)  # {"1h": price, "4h": price, "24h": price}


# ---------------------------------------------------------------------------
# CounterfactualAnalyzer
# ---------------------------------------------------------------------------

class CounterfactualAnalyzer:
    """
    What-if analysis engine for ARGUS trades and skipped signals.

    Parameters
    ----------
    config : dict, optional
        ``counterfactual`` section from unified_config.yaml.
    db_path : str, optional
        Override SQLite path.
    """

    _DEFAULTS: Dict[str, Any] = {
        "enabled": True,
        "db_path": "data/counterfactuals.db",
        "size_multiplier_large": 1.5,
        "size_multiplier_small": 0.5,
        "venue_fee_diff_bps": 5.0,
        "limit_order_improvement_bps": 3.0,
        "horizons": ["1h", "4h", "24h"],
        "lookback_days": 30,
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        db_path: Optional[str] = None,
    ) -> None:
        cfg = dict(self._DEFAULTS)
        if config:
            cfg.update(config)
        self._cfg = cfg
        self._enabled = bool(cfg.get("enabled", True))
        self._size_large = float(cfg.get("size_multiplier_large", 1.5))
        self._size_small = float(cfg.get("size_multiplier_small", 0.5))
        self._venue_fee_diff = float(cfg.get("venue_fee_diff_bps", 5.0))
        self._limit_improve = float(cfg.get("limit_order_improvement_bps", 3.0))
        self._horizons = cfg.get("horizons", ["1h", "4h", "24h"])
        self._lookback_days = int(cfg.get("lookback_days", 30))

        # In-memory caches
        self._trades: Dict[str, TradeRecord] = {}
        self._skipped: Dict[str, SkippedSignal] = {}
        self._price_snapshots: Dict[str, PriceSnapshot] = {}
        self._reports: Dict[str, CounterfactualReport] = {}
        self._lock = threading.Lock()

        db = db_path or str(cfg.get("db_path", "data/counterfactuals.db"))
        self._db_path = db
        self._init_db()

        logger.info("CounterfactualAnalyzer initialised (enabled=%s)", self._enabled)

    # ------------------------------------------------------------------
    # SQLite
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    size REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    venue TEXT,
                    slippage_bps REAL DEFAULT 0.0,
                    was_limit INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skipped_signals (
                    signal_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    signal_score REAL NOT NULL,
                    price_at_signal REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    reason_skipped TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_snapshots (
                    symbol TEXT NOT NULL,
                    base_price REAL NOT NULL,
                    prices_after TEXT NOT NULL DEFAULT '{}',
                    reference_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    PRIMARY KEY (reference_id, symbol)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS counterfactual_reports (
                    trade_id TEXT PRIMARY KEY,
                    report TEXT NOT NULL,
                    analyzed_at TEXT NOT NULL
                )
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_trade(self, trade: TradeRecord) -> None:
        """Record a completed trade for later counterfactual analysis."""
        with self._lock:
            self._trades[trade.trade_id] = trade
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO trades (trade_id, symbol, side, entry_price, exit_price, "
                    "size, entry_time, exit_time, venue, slippage_bps, was_limit) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (trade.trade_id, trade.symbol, trade.side, trade.entry_price,
                     trade.exit_price, trade.size, trade.entry_time, trade.exit_time,
                     trade.venue, trade.slippage_bps, int(trade.was_limit)),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("CounterfactualAnalyzer: trade persist failed — %s", exc)

    def record_skipped_signal(self, signal: SkippedSignal) -> None:
        """Record a signal that was skipped."""
        with self._lock:
            self._skipped[signal.signal_id] = signal
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO skipped_signals (signal_id, symbol, direction, "
                    "signal_score, price_at_signal, timestamp, reason_skipped) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (signal.signal_id, signal.symbol, signal.direction,
                     signal.signal_score, signal.price_at_signal, signal.timestamp,
                     signal.reason_skipped),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("CounterfactualAnalyzer: signal persist failed — %s", exc)

    def record_price_snapshot(self, reference_id: str, snapshot: PriceSnapshot) -> None:
        """Record prices at various horizons after a trade or signal."""
        with self._lock:
            self._price_snapshots[reference_id] = snapshot
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO price_snapshots (symbol, base_price, prices_after, reference_id, timestamp) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (snapshot.symbol, snapshot.base_price, json.dumps(snapshot.prices_after),
                     reference_id, datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("CounterfactualAnalyzer: snapshot persist failed — %s", exc)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze_trade(self, trade_id: str) -> Optional[CounterfactualReport]:
        """
        Perform counterfactual analysis on a completed trade.

        Requires a price snapshot to have been recorded for the trade.
        """
        trade = self._trades.get(trade_id)
        if not trade:
            # Try loading from DB
            trade = self._load_trade(trade_id)
        if not trade:
            logger.warning("CounterfactualAnalyzer: trade %s not found", trade_id)
            return None

        snapshot = self._price_snapshots.get(trade_id)
        if not snapshot:
            snapshot = self._load_snapshot(trade_id)

        now = datetime.now(timezone.utc).isoformat()
        actual_pnl = self._compute_pnl(trade.side, trade.entry_price, trade.exit_price, trade.size)

        # What if held longer?
        held_longer: Dict[str, float] = {}
        if snapshot and snapshot.prices_after:
            for horizon, price in snapshot.prices_after.items():
                held_longer[horizon] = self._compute_pnl(trade.side, trade.entry_price, price, trade.size)
        else:
            # Estimate: extrapolate from actual return
            ret_pct = (trade.exit_price - trade.entry_price) / trade.entry_price if trade.entry_price else 0
            for h in self._horizons:
                # Decay/extend heuristic
                mult = {"1h": 0.5, "4h": 1.2, "24h": 0.8}.get(h, 1.0)
                est_price = trade.entry_price * (1 + ret_pct * mult)
                held_longer[h] = self._compute_pnl(trade.side, trade.entry_price, est_price, trade.size)

        # What if larger size?
        larger_pnl = self._compute_pnl(
            trade.side, trade.entry_price, trade.exit_price, trade.size * self._size_large
        )

        # What if smaller size?
        smaller_pnl = self._compute_pnl(
            trade.side, trade.entry_price, trade.exit_price, trade.size * self._size_small
        )

        # What if different venue? (better/worse fees)
        fee_improvement = trade.size * trade.entry_price * (self._venue_fee_diff / 10000.0)
        venue_pnl = actual_pnl + fee_improvement

        # What if limit order? (better fill price)
        improvement = trade.entry_price * (self._limit_improve / 10000.0)
        if trade.was_limit:
            limit_pnl = actual_pnl  # already was limit
        else:
            if trade.side == "buy":
                limit_entry = trade.entry_price - improvement
            else:
                limit_entry = trade.entry_price + improvement
            limit_pnl = self._compute_pnl(trade.side, limit_entry, trade.exit_price, trade.size)

        # Determine optimal action
        all_options = {
            "actual": actual_pnl,
            "held_longer_best": max(held_longer.values()) if held_longer else actual_pnl,
            "larger_size": larger_pnl,
            "smaller_size": smaller_pnl,
            "different_venue": venue_pnl,
            "limit_order": limit_pnl,
        }
        best_option = max(all_options, key=all_options.get)  # type: ignore
        best_pnl = all_options[best_option]

        # Lesson
        if best_option == "actual":
            lesson = "Trade execution was optimal"
            optimal = "Current approach was best"
        elif best_option == "held_longer_best":
            best_h = max(held_longer, key=held_longer.get)  # type: ignore
            lesson = f"Holding {best_h} longer would have been better (PnL: {best_pnl:.2f} vs {actual_pnl:.2f})"
            optimal = f"Hold for {best_h} instead of exiting early"
        elif best_option == "larger_size":
            lesson = f"Larger position ({self._size_large:.1f}x) would have yielded {best_pnl:.2f} vs {actual_pnl:.2f}"
            optimal = "Increase position size for high-conviction trades"
        elif best_option == "smaller_size":
            lesson = f"Smaller position ({self._size_small:.1f}x) would have limited loss to {best_pnl:.2f}"
            optimal = "Reduce position size when conviction is lower"
        elif best_option == "different_venue":
            lesson = f"Better venue would have saved {fee_improvement:.2f} in fees"
            optimal = "Route to lower-fee venue"
        else:
            lesson = f"Limit order would have improved fill by {self._limit_improve:.1f}bps"
            optimal = "Use limit orders instead of market orders"

        opportunity_cost = best_pnl - actual_pnl

        report = CounterfactualReport(
            trade_id=trade_id,
            actual_pnl=round(actual_pnl, 4),
            what_if_held_longer=held_longer,
            what_if_larger_size=round(larger_pnl, 4),
            what_if_smaller_size=round(smaller_pnl, 4),
            what_if_different_venue=round(venue_pnl, 4),
            what_if_limit_order=round(limit_pnl, 4),
            optimal_action=optimal,
            lesson=lesson,
            opportunity_cost=round(opportunity_cost, 4),
            analyzed_at=now,
        )

        with self._lock:
            self._reports[trade_id] = report
        self._persist_report(report)
        return report

    def analyze_skipped_signal(self, signal_id: str) -> Optional[CounterfactualReport]:
        """
        Analyze what would have happened if we acted on a skipped signal.
        """
        signal = self._skipped.get(signal_id)
        if not signal:
            signal = self._load_skipped(signal_id)
        if not signal:
            logger.warning("CounterfactualAnalyzer: signal %s not found", signal_id)
            return None

        snapshot = self._price_snapshots.get(signal_id)
        if not snapshot:
            snapshot = self._load_snapshot(signal_id)

        now = datetime.now(timezone.utc).isoformat()
        hypothetical_size = 1.0  # normalised unit
        entry = signal.price_at_signal

        held_longer: Dict[str, float] = {}
        if snapshot and snapshot.prices_after:
            for horizon, price in snapshot.prices_after.items():
                side = "buy" if signal.direction == "long" else "sell"
                held_longer[horizon] = self._compute_pnl(side, entry, price, hypothetical_size)
        else:
            # Can't compute without price data — return zero estimates
            for h in self._horizons:
                held_longer[h] = 0.0

        best_pnl = max(held_longer.values()) if held_longer else 0.0
        best_horizon = max(held_longer, key=held_longer.get) if held_longer and any(v != 0 for v in held_longer.values()) else "unknown"

        if best_pnl > 0:
            lesson = f"Skipped signal would have yielded {best_pnl:.2f} at {best_horizon}"
            optimal = f"Should have taken the {signal.direction} signal"
        elif best_pnl < 0:
            lesson = f"Correctly skipped — would have lost {abs(best_pnl):.2f}"
            optimal = "Skip decision was correct"
        else:
            lesson = "Insufficient data to determine outcome"
            optimal = "Monitor similar signals more closely"

        report = CounterfactualReport(
            trade_id=signal_id,
            actual_pnl=0.0,
            what_if_held_longer=held_longer,
            optimal_action=optimal,
            lesson=lesson,
            opportunity_cost=round(max(best_pnl, 0), 4),
            analyzed_at=now,
        )

        with self._lock:
            self._reports[signal_id] = report
        self._persist_report(report)
        return report

    # ------------------------------------------------------------------
    # Aggregation queries
    # ------------------------------------------------------------------

    def get_biggest_missed_opportunities(self, lookback_days: int = 7) -> List[Dict[str, Any]]:
        """Return skipped signals that would have been profitable, sorted by opportunity cost."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
        results: List[Dict[str, Any]] = []

        for sid, report in self._reports.items():
            if sid not in self._skipped:
                continue
            if report.analyzed_at < cutoff:
                continue
            if report.opportunity_cost > 0:
                results.append({
                    "signal_id": sid,
                    "symbol": self._skipped[sid].symbol,
                    "direction": self._skipped[sid].direction,
                    "opportunity_cost": report.opportunity_cost,
                    "lesson": report.lesson,
                })

        results.sort(key=lambda x: x["opportunity_cost"], reverse=True)
        return results[:20]

    def get_biggest_mistakes(self, lookback_days: int = 7) -> List[Dict[str, Any]]:
        """Return trades where a significantly better outcome was possible."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
        results: List[Dict[str, Any]] = []

        for tid, report in self._reports.items():
            if tid in self._skipped:
                continue  # skip signal analyses
            if report.analyzed_at < cutoff:
                continue
            if report.opportunity_cost > 0 and report.actual_pnl < 0:
                results.append({
                    "trade_id": tid,
                    "actual_pnl": report.actual_pnl,
                    "opportunity_cost": report.opportunity_cost,
                    "optimal_action": report.optimal_action,
                    "lesson": report.lesson,
                })

        results.sort(key=lambda x: x["opportunity_cost"], reverse=True)
        return results[:20]

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Aggregate counterfactual statistics."""
        trade_reports = [r for tid, r in self._reports.items() if tid not in self._skipped]
        signal_reports = [r for sid, r in self._reports.items() if sid in self._skipped]

        total_opp_cost = sum(r.opportunity_cost for r in trade_reports if r.opportunity_cost > 0)
        total_missed = sum(r.opportunity_cost for r in signal_reports if r.opportunity_cost > 0)

        return {
            "trades_analyzed": len(trade_reports),
            "signals_analyzed": len(signal_reports),
            "total_trade_opportunity_cost": round(total_opp_cost, 2),
            "total_missed_opportunity_cost": round(total_missed, 2),
            "avg_trade_opportunity_cost": round(total_opp_cost / max(len(trade_reports), 1), 4),
            "avg_missed_opportunity_cost": round(total_missed / max(len(signal_reports), 1), 4),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_pnl(side: str, entry: float, exit: float, size: float) -> float:
        if side in ("buy", "long"):
            return (exit - entry) * size
        else:
            return (entry - exit) * size

    def _load_trade(self, trade_id: str) -> Optional[TradeRecord]:
        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT trade_id, symbol, side, entry_price, exit_price, size, "
                    "entry_time, exit_time, venue, slippage_bps, was_limit FROM trades WHERE trade_id=?",
                    (trade_id,),
                ).fetchone()
            if row:
                t = TradeRecord(
                    trade_id=row[0], symbol=row[1], side=row[2],
                    entry_price=row[3], exit_price=row[4], size=row[5],
                    entry_time=row[6], exit_time=row[7], venue=row[8] or "",
                    slippage_bps=row[9] or 0.0, was_limit=bool(row[10]),
                )
                self._trades[trade_id] = t
                return t
        except Exception:
            pass
        return None

    def _load_skipped(self, signal_id: str) -> Optional[SkippedSignal]:
        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT signal_id, symbol, direction, signal_score, price_at_signal, timestamp, reason_skipped "
                    "FROM skipped_signals WHERE signal_id=?",
                    (signal_id,),
                ).fetchone()
            if row:
                s = SkippedSignal(
                    signal_id=row[0], symbol=row[1], direction=row[2],
                    signal_score=row[3], price_at_signal=row[4],
                    timestamp=row[5], reason_skipped=row[6] or "",
                )
                self._skipped[signal_id] = s
                return s
        except Exception:
            pass
        return None

    def _load_snapshot(self, ref_id: str) -> Optional[PriceSnapshot]:
        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT symbol, base_price, prices_after FROM price_snapshots WHERE reference_id=?",
                    (ref_id,),
                ).fetchone()
            if row:
                ps = PriceSnapshot(
                    symbol=row[0], base_price=row[1],
                    prices_after=json.loads(row[2]) if row[2] else {},
                )
                self._price_snapshots[ref_id] = ps
                return ps
        except Exception:
            pass
        return None

    def _persist_report(self, report: CounterfactualReport) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO counterfactual_reports (trade_id, report, analyzed_at) VALUES (?, ?, ?)",
                    (report.trade_id, json.dumps(report.to_dict()), report.analyzed_at),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("CounterfactualAnalyzer: report persist failed — %s", exc)
