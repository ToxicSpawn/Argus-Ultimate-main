"""
Performance Attribution — decomposes P&L by strategy, regime, time-of-day, asset.

Answers questions like:
  - Which strategy contributes most to returns?
  - Does the system perform better in trending vs ranging markets?
  - Are there intraday patterns in P&L?
  - Which assets are most/least profitable?

Reads from trade ledger SQLite. Updates incrementally.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AttributionBucket:
    key: str
    total_pnl_usd: float = 0.0
    trade_count: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    sharpe_like: float = 0.0

    def __repr__(self) -> str:
        return (
            f"AttributionBucket(key={self.key!r}, total_pnl={self.total_pnl_usd:.2f}, "
            f"trades={self.trade_count}, win_rate={self.win_rate:.1%}, "
            f"sharpe={self.sharpe_like:.3f})"
        )


class PerformanceAttribution:
    """Decomposes P&L into attribution buckets across multiple dimensions."""

    def __init__(
        self,
        trade_db: str = "data/paper_trades.db",
        lookback_days: int = 30,
    ) -> None:
        self.trade_db = trade_db
        self.lookback_days = lookback_days

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_by_strategy(self) -> Dict[str, AttributionBucket]:
        """Attribution broken down by strategy name."""
        trades = self._load_trades(self._since_ts())
        return self._bucket(trades, lambda t: t.get("strategy") or "unknown")

    def compute_by_regime(self) -> Dict[str, AttributionBucket]:
        """Attribution broken down by market regime recorded at trade time."""
        trades = self._load_trades(self._since_ts())
        return self._bucket(trades, lambda t: t.get("regime") or "unknown")

    def compute_by_hour(self) -> Dict[int, AttributionBucket]:
        """Attribution broken down by hour-of-day (0-23, UTC)."""
        trades = self._load_trades(self._since_ts())
        raw = self._bucket(
            trades,
            lambda t: str(self._hour_from_ts(t.get("exit_ts") or t.get("entry_ts", 0))),
        )
        # Return as Dict[int, AttributionBucket] with integer keys
        result: Dict[int, AttributionBucket] = {}
        for hour_str, bucket in raw.items():
            try:
                hour_int = int(hour_str)
            except ValueError:
                hour_int = -1
            bucket.key = str(hour_int)
            result[hour_int] = bucket
        return result

    def compute_by_asset(self) -> Dict[str, AttributionBucket]:
        """Attribution broken down by trading pair / asset symbol."""
        trades = self._load_trades(self._since_ts())
        return self._bucket(trades, lambda t: t.get("symbol") or t.get("pair") or "unknown")

    def full_report(self) -> dict:
        """Return all four attribution breakdowns plus summary stats."""
        since_ts = self._since_ts()
        trades = self._load_trades(since_ts)

        by_strategy = self._bucket(trades, lambda t: t.get("strategy") or "unknown")
        by_regime = self._bucket(trades, lambda t: t.get("regime") or "unknown")
        by_hour_raw = self._bucket(
            trades,
            lambda t: str(self._hour_from_ts(t.get("exit_ts") or t.get("entry_ts", 0))),
        )
        by_hour: Dict[int, AttributionBucket] = {}
        for k, v in by_hour_raw.items():
            try:
                by_hour[int(k)] = v
            except ValueError:
                by_hour[-1] = v

        by_asset = self._bucket(trades, lambda t: t.get("symbol") or t.get("pair") or "unknown")

        total_pnl = sum(t.get("pnl_usd", 0.0) for t in trades)
        wins = [t for t in trades if (t.get("pnl_usd") or 0.0) > 0]
        overall_win_rate = len(wins) / len(trades) if trades else 0.0

        return {
            "by_strategy": {k: self._bucket_to_dict(v) for k, v in by_strategy.items()},
            "by_regime": {k: self._bucket_to_dict(v) for k, v in by_regime.items()},
            "by_hour": {k: self._bucket_to_dict(v) for k, v in by_hour.items()},
            "by_asset": {k: self._bucket_to_dict(v) for k, v in by_asset.items()},
            "summary": {
                "total_pnl_usd": round(total_pnl, 4),
                "trade_count": len(trades),
                "win_rate": round(overall_win_rate, 4),
                "lookback_days": self.lookback_days,
                "since_ts": since_ts,
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_trades(self, since_ts: float) -> List[dict]:
        """Load trades from the SQLite trade ledger since a given timestamp."""
        import os

        if not os.path.exists(self.trade_db):
            logger.warning("Trade DB not found at %s — returning empty list", self.trade_db)
            return []

        try:
            conn = sqlite3.connect(self.trade_db)
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.execute(
                    "SELECT * FROM trades WHERE exit_ts >= ? ORDER BY exit_ts ASC",
                    (since_ts,),
                )
                rows = [dict(row) for row in cursor.fetchall()]
            except sqlite3.OperationalError:
                # Table may not exist yet or column names may differ; try alternative
                try:
                    cursor = conn.execute(
                        "SELECT * FROM trades WHERE timestamp >= ? ORDER BY timestamp ASC",
                        (since_ts,),
                    )
                    rows = [dict(row) for row in cursor.fetchall()]
                except sqlite3.OperationalError as exc:
                    logger.warning("Could not query trades table: %s", exc)
                    rows = []
            finally:
                conn.close()
            return rows
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error loading trades from %s: %s", self.trade_db, exc)
            return []

    def _bucket(
        self,
        trades: List[dict],
        key_fn: Callable[[dict], str],
    ) -> Dict[str, AttributionBucket]:
        """Group trades by the result of key_fn and compute attribution metrics."""
        groups: Dict[str, List[float]] = {}
        for trade in trades:
            key = key_fn(trade)
            pnl = float(trade.get("pnl_usd", 0.0) or 0.0)
            groups.setdefault(key, []).append(pnl)

        result: Dict[str, AttributionBucket] = {}
        for key, pnls in groups.items():
            total = sum(pnls)
            count = len(pnls)
            wins = sum(1 for p in pnls if p > 0)
            win_rate = wins / count if count else 0.0
            avg = total / count if count else 0.0
            sharpe = self._sharpe_like(pnls)
            result[key] = AttributionBucket(
                key=key,
                total_pnl_usd=round(total, 4),
                trade_count=count,
                win_rate=round(win_rate, 4),
                avg_pnl=round(avg, 4),
                sharpe_like=round(sharpe, 4),
            )
        return result

    @staticmethod
    def _sharpe_like(pnls: List[float]) -> float:
        """Return a Sharpe-like ratio: mean / std of P&L series."""
        if len(pnls) < 2:
            return 0.0
        n = len(pnls)
        mean = sum(pnls) / n
        variance = sum((p - mean) ** 2 for p in pnls) / (n - 1)
        std = math.sqrt(variance)
        if std == 0.0:
            return 0.0
        return mean / std

    @staticmethod
    def _hour_from_ts(ts: object) -> int:
        """Extract UTC hour from a Unix timestamp."""
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc).hour  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            return -1

    def _since_ts(self) -> float:
        """Return Unix timestamp for lookback_days ago."""
        import time

        return time.time() - self.lookback_days * 86400

    @staticmethod
    def _bucket_to_dict(bucket: AttributionBucket) -> dict:
        return {
            "key": bucket.key,
            "total_pnl_usd": bucket.total_pnl_usd,
            "trade_count": bucket.trade_count,
            "win_rate": bucket.win_rate,
            "avg_pnl": bucket.avg_pnl,
            "sharpe_like": bucket.sharpe_like,
        }
