"""
Trade Journal — structured logging of every trade with rich context.

Records for each trade:
  - Entry/exit prices, quantities, P&L
  - Signal source and confidence
  - Market regime at entry
  - Reason for exit (TP/SL/timeout/signal_reversal)
  - Post-trade notes (optional)
  - Performance vs benchmark (BTC buy-and-hold)

Stores in SQLite. Generates markdown reports.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS journal (
    trade_id          TEXT PRIMARY KEY,
    symbol            TEXT NOT NULL,
    strategy          TEXT,
    entry_ts          REAL,
    exit_ts           REAL,
    entry_price       REAL,
    exit_price        REAL,
    qty_usd           REAL,
    pnl_usd           REAL,
    pnl_pct           REAL,
    regime_at_entry   TEXT,
    signal_confidence REAL,
    exit_reason       TEXT,
    notes             TEXT,
    tags              TEXT
);
CREATE INDEX IF NOT EXISTS idx_journal_ts ON journal(entry_ts);
CREATE INDEX IF NOT EXISTS idx_journal_strategy ON journal(strategy);
CREATE TABLE IF NOT EXISTS journal_notes (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id  TEXT NOT NULL,
    note      TEXT,
    tag       TEXT,
    ts        REAL
);
"""


@dataclass
class JournalEntry:
    trade_id: str
    symbol: str
    strategy: str = ""
    entry_ts: float = 0.0
    exit_ts: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    qty_usd: float = 0.0
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    regime_at_entry: str = ""
    signal_confidence: float = 0.0
    exit_reason: str = ""
    notes: str = ""
    tags: List[str] = field(default_factory=list)


class TradeJournal:
    """SQLite-backed trade journal with markdown reporting."""

    def __init__(self, db_path: str = "data/trade_journal.db") -> None:
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(self, entry: JournalEntry) -> None:
        """Persist a JournalEntry to SQLite."""
        tags_json = json.dumps(entry.tags) if entry.tags else "[]"
        try:
            conn = self._connect()
            conn.execute(
                """INSERT OR REPLACE INTO journal VALUES
                   (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    entry.trade_id, entry.symbol, entry.strategy,
                    entry.entry_ts, entry.exit_ts,
                    entry.entry_price, entry.exit_price, entry.qty_usd,
                    entry.pnl_usd, entry.pnl_pct,
                    entry.regime_at_entry, entry.signal_confidence,
                    entry.exit_reason, entry.notes, tags_json,
                ),
            )
            conn.commit()
            conn.close()
        except Exception:
            logger.exception("TradeJournal: failed to record trade %s", entry.trade_id)

    def add_note(self, trade_id: str, note: str, tag: str = "") -> None:
        """Append a note/tag to an existing trade."""
        try:
            conn = self._connect()
            conn.execute(
                "INSERT INTO journal_notes (trade_id, note, tag, ts) VALUES (?,?,?,?)",
                (trade_id, note, tag, time.time()),
            )
            conn.commit()
            conn.close()
        except Exception:
            logger.exception("TradeJournal: add_note failed for %s", trade_id)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        strategy: Optional[str] = None,
        symbol: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[JournalEntry]:
        """Query trades with optional filters."""
        try:
            conn = self._connect()
            where, params = [], []
            if strategy:
                where.append("strategy = ?")
                params.append(strategy)
            if symbol:
                where.append("symbol = ?")
                params.append(symbol)
            if since is not None:
                where.append("entry_ts >= ?")
                params.append(since)
            sql = "SELECT * FROM journal"
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += f" ORDER BY entry_ts DESC LIMIT {int(limit)}"
            cur = conn.execute(sql, params)
            rows = cur.fetchall()
            conn.close()
            return [self._row_to_entry(r) for r in rows]
        except Exception:
            logger.exception("TradeJournal: query failed")
            return []

    def get_stats(self, since_days: int = 30) -> Dict:
        """Compute aggregate stats for the last N days."""
        since = time.time() - since_days * 86400
        trades = self.query(since=since, limit=10000)
        if not trades:
            return {"trade_count": 0, "win_rate": 0.0, "avg_pnl": 0.0}

        wins = [t for t in trades if t.pnl_usd > 0]
        by_reason: Dict[str, int] = {}
        for t in trades:
            r = t.exit_reason or "unknown"
            by_reason[r] = by_reason.get(r, 0) + 1

        pnls = [t.pnl_usd for t in trades]
        return {
            "trade_count": len(trades),
            "win_rate": len(wins) / len(trades),
            "avg_pnl": sum(pnls) / len(pnls),
            "total_pnl": sum(pnls),
            "best_trade": max(pnls),
            "worst_trade": min(pnls),
            "by_exit_reason": by_reason,
        }

    def generate_markdown_report(self, since_days: int = 30) -> str:
        """Generate a markdown table + stats summary."""
        stats = self.get_stats(since_days)
        since = time.time() - since_days * 86400
        trades = self.query(since=since, limit=50)

        lines = [
            f"# Trade Journal — Last {since_days} Days",
            "",
            "## Summary",
            f"- Trades: {stats.get('trade_count', 0)}",
            f"- Win rate: {stats.get('win_rate', 0)*100:.1f}%",
            f"- Avg P&L: ${stats.get('avg_pnl', 0):.2f}",
            f"- Total P&L: ${stats.get('total_pnl', 0):.2f}",
            f"- Best trade: ${stats.get('best_trade', 0):.2f}",
            f"- Worst trade: ${stats.get('worst_trade', 0):.2f}",
            "",
            "## Recent Trades",
            "| ID | Symbol | Strategy | P&L | Exit Reason | Confidence |",
            "|----|--------|----------|-----|-------------|------------|",
        ]
        for t in trades[:20]:
            pnl_str = f"${t.pnl_usd:+.2f}"
            lines.append(
                f"| {t.trade_id[:8]} | {t.symbol} | {t.strategy} | "
                f"{pnl_str} | {t.exit_reason} | {t.signal_confidence:.2f} |"
            )

        return "\n".join(lines)

    def benchmark_comparison(
        self, since_days: int = 30, btc_start_price: Optional[float] = None
    ) -> Dict:
        """Compare our returns vs BTC buy-and-hold."""
        since = time.time() - since_days * 86400
        trades = self.query(since=since, limit=10000)
        our_total = sum(t.pnl_usd for t in trades)
        # BTC comparison is approximate without live price feed
        btc_return_pct = None
        if btc_start_price is not None and btc_start_price > 0:
            # Placeholder: requires current price to complete comparison
            btc_return_pct = 0.0  # caller should fill in
        return {
            "our_total_pnl_usd": our_total,
            "our_return_pct": our_total / 1000.0 * 100,  # assume $1000 capital
            "btc_hold_return_pct": btc_return_pct,
            "trade_count": len(trades),
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        import os
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        try:
            conn = self._connect()
            conn.executescript(_SCHEMA)
            conn.commit()
            conn.close()
        except Exception:
            logger.exception("TradeJournal: failed to initialise DB at %s", self.db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def _row_to_entry(row) -> JournalEntry:
        d = dict(row)
        tags = json.loads(d.get("tags") or "[]")
        return JournalEntry(
            trade_id=d["trade_id"],
            symbol=d.get("symbol", ""),
            strategy=d.get("strategy", ""),
            entry_ts=d.get("entry_ts", 0.0),
            exit_ts=d.get("exit_ts", 0.0),
            entry_price=d.get("entry_price", 0.0),
            exit_price=d.get("exit_price", 0.0),
            qty_usd=d.get("qty_usd", 0.0),
            pnl_usd=d.get("pnl_usd", 0.0),
            pnl_pct=d.get("pnl_pct", 0.0),
            regime_at_entry=d.get("regime_at_entry", ""),
            signal_confidence=d.get("signal_confidence", 0.0),
            exit_reason=d.get("exit_reason", ""),
            notes=d.get("notes", ""),
            tags=tags,
        )
