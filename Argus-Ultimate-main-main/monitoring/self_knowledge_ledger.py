"""
SelfKnowledgeLedger — SQLite-backed persistent self-memory.

ARGUS writes one observation row every record_interval_cycles and reads
back the last N days on startup — so the system remembers its own
history across restarts.

Schema (table: self_observations):
  ts              REAL     Unix timestamp
  cycle           INTEGER  Cycle number
  health_score    INTEGER  0–100
  health_label    TEXT     PEAK/GOOD/MARGINAL/POOR/CRITICAL
  regime          TEXT     Current regime label
  edge_score      REAL     0–1
  has_pre_hedge   INTEGER  0 or 1
  critical_count  INTEGER
  warning_count   INTEGER
  position_scale  REAL     Current position scale (from intelligence_directive)

Summary metrics:
  health_trend    : "improving" | "stable" | "deteriorating"
  best/worst_performing_regime: by avg health score
  edge_score_trend: "improving" | "stable" | "deteriorating"

Output: advisory["self_knowledge"]
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS self_observations (
    ts             REAL    NOT NULL,
    cycle          INTEGER NOT NULL,
    health_score   INTEGER NOT NULL DEFAULT 70,
    health_label   TEXT    NOT NULL DEFAULT 'GOOD',
    regime         TEXT    NOT NULL DEFAULT 'UNKNOWN',
    edge_score     REAL    NOT NULL DEFAULT 1.0,
    has_pre_hedge  INTEGER NOT NULL DEFAULT 0,
    critical_count INTEGER NOT NULL DEFAULT 0,
    warning_count  INTEGER NOT NULL DEFAULT 0,
    position_scale REAL    NOT NULL DEFAULT 1.0
);
"""
_CREATE_IDX = "CREATE INDEX IF NOT EXISTS idx_ts ON self_observations (ts);"


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class SelfKnowledgeSummary:
    health_trend: str               # "improving" | "stable" | "deteriorating"
    avg_health_score_7d: float
    best_performing_regime: str
    worst_performing_regime: str
    edge_score_trend: str
    total_records: int
    insights: List[str]
    ts: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# SelfKnowledgeLedger
# ---------------------------------------------------------------------------

class SelfKnowledgeLedger:
    """
    Persistent self-memory backed by SQLite.

    Parameters
    ----------
    db_path                : path to SQLite file (created if absent)
    record_interval_cycles : how often to write a row (default 100)
    config                 : optional config object
    """

    def __init__(
        self,
        db_path: str = "data/self_knowledge.db",
        record_interval_cycles: int = 100,
        config: Optional[Any] = None,
    ) -> None:
        self.db_path                 = str(db_path)
        self.record_interval_cycles  = max(1, int(record_interval_cycles))
        self.config                  = config

        self._last_record_cycle: int          = -999999
        self._last_summary: Optional[SelfKnowledgeSummary] = None
        self._conn: Optional[sqlite3.Connection] = None

        # Initialise DB
        self._init_db()

        # Bootstrap summary from existing data
        try:
            self._last_summary = self.summarize(lookback_days=7)
        except Exception as exc:
            logger.debug("SelfKnowledgeLedger: bootstrap summarize failed: %s", exc)

    # ── Public API ─────────────────────────────────────────────────────────

    def record(self, advisory: Dict[str, Any], cycle: int) -> None:
        """
        Write one observation row (rate-limited to record_interval_cycles).
        """
        if cycle - self._last_record_cycle < self.record_interval_cycles:
            return

        try:
            _hs  = advisory.get("health_score") or {}
            _em  = advisory.get("edge_monitor")  or {}
            _rt  = advisory.get("regime_transition") or {}
            _re  = advisory.get("regime_ensemble") or {}
            _sd  = advisory.get("self_diagnosis")  or {}
            _id  = advisory.get("intelligence_directive") or {}

            row = (
                time.time(),
                int(cycle),
                int(_hs.get("score", 70) or 70),
                str(_hs.get("label", "GOOD") or "GOOD"),
                str(_re.get("regime", "UNKNOWN") or "UNKNOWN"),
                float(_em.get("edge_score", 1.0) or 1.0),
                1 if bool(_rt.get("pre_hedge_signal", False)) else 0,
                len(_sd.get("critical") or []),
                len(_sd.get("warnings") or []),
                float(_id.get("position_scale", 1.0) or 1.0),
            )
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO self_observations
                   (ts, cycle, health_score, health_label, regime,
                    edge_score, has_pre_hedge, critical_count,
                    warning_count, position_scale)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                row,
            )
            conn.commit()
            self._last_record_cycle = cycle

        except Exception as exc:
            logger.debug("SelfKnowledgeLedger.record failed: %s", exc)

    def summarize(self, lookback_days: int = 7) -> SelfKnowledgeSummary:
        """Return summary of recent self-observations."""
        cutoff = time.time() - lookback_days * 86400.0
        insights: List[str] = []

        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT ts, health_score, regime, edge_score FROM self_observations "
                "WHERE ts >= ? ORDER BY ts ASC",
                (cutoff,),
            ).fetchall()
        except Exception as exc:
            logger.debug("SelfKnowledgeLedger.summarize query failed: %s", exc)
            rows = []

        total_records = self._count_total()

        if not rows:
            summary = SelfKnowledgeSummary(
                health_trend            = "stable",
                avg_health_score_7d     = 70.0,
                best_performing_regime  = "UNKNOWN",
                worst_performing_regime = "UNKNOWN",
                edge_score_trend        = "stable",
                total_records           = total_records,
                insights                = ["Insufficient history for summary"],
            )
            self._last_summary = summary
            return summary

        health_vals = [float(r[1]) for r in rows]
        regime_vals = [str(r[2]) for r in rows]
        edge_vals   = [float(r[3]) for r in rows]

        avg_health = sum(health_vals) / len(health_vals)

        # Trend: compare first half vs second half
        health_trend  = self._half_trend(health_vals,  higher_is_better=True)
        edge_trend    = self._half_trend(edge_vals,    higher_is_better=True)

        # Regime performance
        regime_health: Dict[str, List[float]] = {}
        for h, reg in zip(health_vals, regime_vals):
            regime_health.setdefault(reg, []).append(h)
        regime_avgs = {
            reg: sum(vals) / len(vals)
            for reg, vals in regime_health.items()
            if vals
        }
        if regime_avgs:
            best_regime  = max(regime_avgs, key=lambda r: regime_avgs[r])
            worst_regime = min(regime_avgs, key=lambda r: regime_avgs[r])
        else:
            best_regime  = "UNKNOWN"
            worst_regime = "UNKNOWN"

        # Insights
        insights.append(
            f"Health over {lookback_days}d: avg={avg_health:.0f}, "
            f"trend={health_trend}"
        )
        if best_regime != "UNKNOWN":
            insights.append(
                f"Best regime: {best_regime} "
                f"(avg health {regime_avgs.get(best_regime, 0):.0f})"
            )
        if worst_regime != best_regime and worst_regime != "UNKNOWN":
            insights.append(
                f"Worst regime: {worst_regime} "
                f"(avg health {regime_avgs.get(worst_regime, 0):.0f})"
            )
        insights.append(f"Edge trend: {edge_trend}")

        summary = SelfKnowledgeSummary(
            health_trend            = health_trend,
            avg_health_score_7d     = round(avg_health, 2),
            best_performing_regime  = best_regime,
            worst_performing_regime = worst_regime,
            edge_score_trend        = edge_trend,
            total_records           = total_records,
            insights                = insights[:5],
        )
        self._last_summary = summary
        return summary

    def snapshot(self) -> Dict[str, Any]:
        s = self._last_summary
        if s is None:
            return {
                "health_trend": "stable",
                "avg_health_score_7d": 70.0,
                "best_performing_regime": "UNKNOWN",
                "worst_performing_regime": "UNKNOWN",
                "edge_score_trend": "stable",
                "total_records": 0,
                "insights": [],
            }
        return {
            "health_trend":             s.health_trend,
            "avg_health_score_7d":      s.avg_health_score_7d,
            "best_performing_regime":   s.best_performing_regime,
            "worst_performing_regime":  s.worst_performing_regime,
            "edge_score_trend":         s.edge_score_trend,
            "total_records":            s.total_records,
            "insights":                 s.insights,
            "ts":                       s.ts,
        }

    # ── Internal ───────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = self._get_conn()
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_IDX)
            conn.commit()
        except Exception as exc:
            logger.warning("SelfKnowledgeLedger: DB init failed: %s", exc)

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            try:
                self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            except Exception as exc:
                # Fallback: in-memory DB so the class still works
                logger.warning(
                    "SelfKnowledgeLedger: cannot open %s (%s) — using :memory:",
                    self.db_path, exc,
                )
                self._conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._conn.execute(_CREATE_TABLE)
                self._conn.execute(_CREATE_IDX)
                self._conn.commit()
        return self._conn

    def _count_total(self) -> int:
        try:
            row = self._get_conn().execute(
                "SELECT COUNT(*) FROM self_observations"
            ).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    @staticmethod
    def _half_trend(vals: List[float], higher_is_better: bool = True) -> str:
        """Compare first half vs second half to determine trend."""
        n = len(vals)
        if n < 4:
            return "stable"
        mid   = n // 2
        first = sum(vals[:mid])  / mid
        second= sum(vals[mid:]) / (n - mid)
        delta = second - first
        threshold = 2.0
        if abs(delta) < threshold:
            return "stable"
        improving = delta > 0 if higher_is_better else delta < 0
        return "improving" if improving else "deteriorating"
