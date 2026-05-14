"""
Decision Journal — Append-only JSONL logging of every trade decision chain.

Records the full decision pipeline for every signal:
  signal → gates applied → sizing → execution outcome

Enables:
  - Deterministic replay: reproduce any past decision from the log
  - Post-mortem debugging: see exactly why a trade was sized/blocked
  - Edge analysis: compare decisions across regimes, strategies, time-of-day
  - Regulatory audit: tamper-evident record of all trading decisions

File format: one JSON object per line (JSONL), UTC timestamps, sorted keys.
Files rotate daily: data/decision_journal/decisions_YYYYMMDD.jsonl
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    """One gate's contribution to the decision."""
    gate_name: str
    action: str            # "pass", "block", "reduce", "skip"
    detail: str = ""       # human-readable reason
    size_multiplier: float = 1.0  # 1.0 = no change, 0.5 = halved, 0.0 = blocked


@dataclass
class DecisionRecord:
    """Full decision chain for one signal in one cycle."""
    # ── Identity ─────────────────────────────────────────────────────────
    decision_id: str              # unique ID for this decision
    cycle_number: int             # which trading cycle
    timestamp_utc: str            # ISO-8601 UTC timestamp
    timestamp_ms: int             # epoch milliseconds

    # ── Signal ───────────────────────────────────────────────────────────
    symbol: str
    side: str                     # BUY / SELL
    strategy: str                 # source strategy name
    confidence: float             # signal confidence [0, 1]
    signal_price: float           # price at signal time

    # ── Context ──────────────────────────────────────────────────────────
    regime: str                   # market regime label
    portfolio_value_aud: float
    position_count: int
    session_mult: float = 1.0     # session-based sizing multiplier
    regime_pos_mult: float = 1.0  # regime position multiplier

    # ── Gate chain ───────────────────────────────────────────────────────
    gates: List[GateResult] = field(default_factory=list)

    # ── Sizing ───────────────────────────────────────────────────────────
    raw_size_pct: float = 0.0     # before any gate adjustments
    final_size_pct: float = 0.0   # after all gates
    final_size_aud: float = 0.0   # actual AUD amount
    conviction_score: float = 0.0  # conviction sizer output

    # ── Outcome ──────────────────────────────────────────────────────────
    outcome: str = "pending"      # "executed", "blocked", "reduced", "error"
    block_reason: str = ""        # why it was blocked (if blocked)
    fill_price: float = 0.0       # actual fill price (0 if not filled)
    slippage_bps: float = 0.0     # fill slippage in basis points
    order_id: str = ""            # exchange order ID

    # ── Metadata ─────────────────────────────────────────────────────────
    metadata: Dict[str, Any] = field(default_factory=dict)

    def total_gate_multiplier(self) -> float:
        """Product of all gate size_multipliers."""
        result = 1.0
        for g in self.gates:
            result *= g.size_multiplier
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Journal writer
# ─────────────────────────────────────────────────────────────────────────────

class DecisionJournal:
    """
    Append-only JSONL writer for trade decisions.

    Usage::

        journal = DecisionJournal("data/decision_journal")
        record = DecisionRecord(...)
        record.gates.append(GateResult("circuit_breaker", "pass"))
        record.gates.append(GateResult("daily_loss", "pass"))
        record.gates.append(GateResult("meta_gate", "reduce", size_multiplier=0.5))
        record.outcome = "executed"
        journal.write(record)

    Files are daily-rotated: decisions_20260409.jsonl
    """

    def __init__(self, base_dir: str = "data/decision_journal") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._current_date: str = ""
        self._handle: Optional[Any] = None
        self._write_count: int = 0
        self._error_count: int = 0
        logger.info("DecisionJournal: initialized at %s", self.base_dir)

    def _get_handle(self) -> Any:
        """Get file handle, rotating daily."""
        today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        if today != self._current_date:
            if self._handle is not None:
                try:
                    self._handle.close()
                except Exception:
                    pass
            self._current_date = today
            path = self.base_dir / f"decisions_{today}.jsonl"
            self._handle = open(path, "a", encoding="utf-8")
            logger.debug("DecisionJournal: opened %s", path)
        return self._handle

    def write(self, record: DecisionRecord) -> None:
        """Append one decision record as a JSON line."""
        try:
            handle = self._get_handle()
            # Convert dataclass to dict, handling nested GateResults
            data = {
                "decision_id": record.decision_id,
                "cycle_number": record.cycle_number,
                "timestamp_utc": record.timestamp_utc,
                "timestamp_ms": record.timestamp_ms,
                "symbol": record.symbol,
                "side": record.side,
                "strategy": record.strategy,
                "confidence": round(record.confidence, 4),
                "signal_price": record.signal_price,
                "regime": record.regime,
                "portfolio_value_aud": round(record.portfolio_value_aud, 2),
                "position_count": record.position_count,
                "session_mult": round(record.session_mult, 3),
                "regime_pos_mult": round(record.regime_pos_mult, 3),
                "gates": [asdict(g) for g in record.gates],
                "total_gate_multiplier": round(record.total_gate_multiplier(), 4),
                "raw_size_pct": round(record.raw_size_pct, 6),
                "final_size_pct": round(record.final_size_pct, 6),
                "final_size_aud": round(record.final_size_aud, 2),
                "conviction_score": round(record.conviction_score, 4),
                "outcome": record.outcome,
                "block_reason": record.block_reason,
                "fill_price": record.fill_price,
                "slippage_bps": round(record.slippage_bps, 2),
                "order_id": record.order_id,
                "metadata": record.metadata,
            }
            line = json.dumps(data, sort_keys=True, default=str)
            handle.write(line + "\n")
            handle.flush()
            self._write_count += 1
        except Exception as exc:
            self._error_count += 1
            logger.warning("DecisionJournal.write error: %s", exc)

    def write_many(self, records: List[DecisionRecord]) -> None:
        """Batch write multiple records."""
        for rec in records:
            self.write(rec)

    # ─────────────────────────────────────────────────────────────────────
    # Replay / query helpers
    # ─────────────────────────────────────────────────────────────────────

    def read_day(self, date_str: str) -> List[Dict[str, Any]]:
        """
        Read all decisions for a given date (YYYYMMDD format).
        Returns list of dicts (raw JSON objects).
        """
        path = self.base_dir / f"decisions_{date_str}.jsonl"
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def read_all(self) -> List[Dict[str, Any]]:
        """Read all decision files in chronological order."""
        all_records: List[Dict[str, Any]] = []
        for path in sorted(self.base_dir.glob("decisions_*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return all_records

    def query(
        self,
        *,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        outcome: Optional[str] = None,
        regime: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Filter decisions by criteria. Reads all files.
        For large datasets, use read_day() instead.
        """
        results = []
        for rec in self.read_all():
            if symbol and rec.get("symbol") != symbol:
                continue
            if strategy and rec.get("strategy") != strategy:
                continue
            if outcome and rec.get("outcome") != outcome:
                continue
            if regime and rec.get("regime") != regime:
                continue
            if rec.get("confidence", 0) < min_confidence:
                continue
            results.append(rec)
        return results

    def summary(self) -> Dict[str, Any]:
        """Quick stats about the journal."""
        all_recs = self.read_all()
        if not all_recs:
            return {"total": 0}

        outcomes = {}
        strategies = {}
        symbols = {}
        total_gate_mult = []

        for rec in all_recs:
            o = rec.get("outcome", "unknown")
            outcomes[o] = outcomes.get(o, 0) + 1
            s = rec.get("strategy", "unknown")
            strategies[s] = strategies.get(s, 0) + 1
            sym = rec.get("symbol", "unknown")
            symbols[sym] = symbols.get(sym, 0) + 1
            total_gate_mult.append(rec.get("total_gate_multiplier", 1.0))

        avg_gate = sum(total_gate_mult) / len(total_gate_mult) if total_gate_mult else 1.0

        return {
            "total": len(all_recs),
            "outcomes": outcomes,
            "strategies": strategies,
            "symbols": symbols,
            "avg_gate_multiplier": round(avg_gate, 4),
            "write_count": self._write_count,
            "error_count": self._error_count,
        }

    def close(self) -> None:
        """Close file handle."""
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass
            self._handle = None


# ─────────────────────────────────────────────────────────────────────────────
# Helper: create a decision ID
# ─────────────────────────────────────────────────────────────────────────────

_decision_counter = 0


def make_decision_id(cycle: int, symbol: str) -> str:
    """Generate a unique decision ID."""
    global _decision_counter
    _decision_counter += 1
    ts = int(time.time() * 1000)
    return f"dec_{cycle}_{symbol.replace('/', '_')}_{ts}_{_decision_counter}"
