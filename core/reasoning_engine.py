"""
core/reasoning_engine.py --- Explainable Decision Reasoning.

Builds structured, human-readable reasoning chains for every autonomous
decision.  Each chain contains premises, supporting evidence, a conclusion,
confidence score, and alternative actions that were considered.

Usage::

    engine = ReasoningEngine()
    chain = engine.build_reasoning_chain(inputs)
    logger.info(engine.explain_last_decision())
    for d in engine.get_decision_log(lookback_hours=24):
        logger.info(d)
Standalone --- no hard imports on the rest of the ARGUS tree at module load.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReasoningChain:
    """A structured reasoning chain for one decision."""

    premises: List[str]                 # factual statements feeding the decision
    evidence: Dict[str, Any]            # supporting numeric / categorical evidence
    conclusion: str                     # the final decision statement
    confidence: float                   # 0.0 -- 1.0
    alternatives: List[str]             # other options that were considered
    decision_type: str = ""             # e.g. "deactivate_strategy"
    target: str = ""                    # what the decision targets
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_explanation(self) -> str:
        """Format as a human-readable paragraph."""
        lines: List[str] = []
        lines.append(f"Decision: {self.conclusion}")
        lines.append(f"Confidence: {self.confidence:.0%}")
        if self.premises:
            lines.append("Reasoning:")
            for i, p in enumerate(self.premises, 1):
                lines.append(f"  ({i}) {p}")
        if self.evidence:
            lines.append(f"Evidence: {json.dumps(self.evidence, default=str)}")
        if self.alternatives:
            lines.append(f"Alternatives considered: {'; '.join(self.alternatives)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_DB_PATH = os.path.join(_DB_DIR, "reasoning_log.db")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS reasoning_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    decision_type TEXT  NOT NULL,
    target      TEXT    NOT NULL,
    premises    TEXT    NOT NULL,
    evidence    TEXT    NOT NULL,
    conclusion  TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    alternatives TEXT   NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reasoning_ts ON reasoning_log(ts);
"""


# ---------------------------------------------------------------------------
# ReasoningEngine
# ---------------------------------------------------------------------------

class ReasoningEngine:
    """Produces and stores explainable reasoning chains.

    Parameters
    ----------
    db_path : str, optional
        Override the default SQLite log path.
    max_log_entries : int
        Maximum entries retained in the in-memory log before pruning.
    """

    def __init__(
        self,
        *,
        db_path: Optional[str] = None,
        max_log_entries: int = 500,
    ) -> None:
        self._db_path = db_path or _DB_PATH
        self._max_log_entries = int(max_log_entries)
        self._log: List[ReasoningChain] = []
        self._lock = threading.Lock()
        self._init_db()
        logger.info("ReasoningEngine initialised (db=%s)", self._db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_reasoning_chain(self, inputs: Dict[str, Any]) -> ReasoningChain:
        """Build a reasoning chain from structured *inputs*.

        Parameters
        ----------
        inputs : dict
            Expected keys (all optional):
            - ``decision_type`` (str): kind of decision
            - ``target`` (str): what the decision targets
            - ``metrics`` (dict): numeric evidence (Sharpe, drawdown, etc.)
            - ``regime`` (str): current market regime
            - ``thresholds`` (dict): thresholds that were checked
            - ``additional_context`` (dict): any extra context
        """
        decision_type = str(inputs.get("decision_type", "unknown"))
        target = str(inputs.get("target", ""))
        metrics = inputs.get("metrics") or {}
        regime = str(inputs.get("regime", "unknown"))
        thresholds = inputs.get("thresholds") or {}
        context = inputs.get("additional_context") or {}

        # Build premises based on decision type
        premises = self._build_premises(decision_type, target, metrics, regime, thresholds, context)

        # Gather evidence
        evidence: Dict[str, Any] = {}
        evidence.update(metrics)
        if regime != "unknown":
            evidence["regime"] = regime
        evidence.update({f"threshold_{k}": v for k, v in thresholds.items()})

        # Build conclusion
        conclusion = self._build_conclusion(decision_type, target, premises)

        # Compute confidence
        confidence = self._compute_confidence(decision_type, metrics, premises)

        # Generate alternatives
        alternatives = self._generate_alternatives(decision_type, target, metrics)

        chain = ReasoningChain(
            premises=premises,
            evidence=evidence,
            conclusion=conclusion,
            confidence=confidence,
            alternatives=alternatives,
            decision_type=decision_type,
            target=target,
        )

        # Persist
        self._append_to_log(chain)
        self._persist_chain(chain)

        logger.debug(
            "ReasoningChain: %s(%s) conf=%.2f premises=%d",
            decision_type, target, confidence, len(premises),
        )
        return chain

    def explain_last_decision(self) -> str:
        """Return a human-readable explanation of the most recent decision."""
        with self._lock:
            if not self._log:
                return "No decisions recorded yet."
            return self._log[-1].to_explanation()

    def get_decision_log(self, lookback_hours: float = 24) -> List[Dict[str, Any]]:
        """Return recent decisions within *lookback_hours* from the database."""
        cutoff = datetime.now(timezone.utc).timestamp() - lookback_hours * 3600
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()

        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM reasoning_log WHERE ts >= ? ORDER BY id DESC",
                (cutoff_iso,),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("Failed to query reasoning log: %s", exc)
            return []

    @property
    def log_size(self) -> int:
        """Number of entries in the in-memory log."""
        return len(self._log)

    # ------------------------------------------------------------------
    # Premise builders
    # ------------------------------------------------------------------

    def _build_premises(
        self,
        decision_type: str,
        target: str,
        metrics: Dict[str, Any],
        regime: str,
        thresholds: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[str]:
        """Generate a list of factual premise strings."""
        premises: List[str] = []

        # Strategy-related premises
        if decision_type in ("deactivate_strategy", "activate_strategy"):
            sharpe = metrics.get("sharpe")
            if sharpe is not None:
                sharpe_thresh = thresholds.get("sharpe", 0.0)
                if float(sharpe) < float(sharpe_thresh):
                    premises.append(
                        f"Sharpe ratio for '{target}' is {float(sharpe):.2f}, below threshold {float(sharpe_thresh):.2f}."
                    )
                elif float(sharpe) > float(sharpe_thresh):
                    premises.append(
                        f"Sharpe ratio for '{target}' is {float(sharpe):.2f}, above threshold {float(sharpe_thresh):.2f}."
                    )

            decay_mult = metrics.get("decay_mult")
            if decay_mult is not None and float(decay_mult) < 0.5:
                premises.append(
                    f"Decay multiplier for '{target}' is {float(decay_mult):.2f}, indicating alpha erosion."
                )

            if not metrics.get("regime_match", True):
                premises.append(
                    f"Current regime '{regime}' is mismatched with strategy '{target}' preferred regime."
                )

            consecutive_losses = metrics.get("consecutive_losses", 0)
            if int(consecutive_losses) >= 3:
                premises.append(
                    f"Strategy '{target}' has {int(consecutive_losses)} consecutive losses."
                )

            trades_14d = metrics.get("trades_14d", -1)
            if int(trades_14d) == 0:
                premises.append(
                    f"Strategy '{target}' has generated zero trades in the last 14 days."
                )

        # Risk-related premises
        if decision_type == "adjust_risk":
            dd = metrics.get("drawdown_pct", 0.0)
            if float(dd) > 5:
                premises.append(f"Current drawdown is {float(dd):.1f}%, requiring risk reduction.")

            vol = metrics.get("volatility", 0.0)
            if float(vol) > 0.8:
                premises.append(f"Realised volatility is {float(vol):.2f}, classified as high.")
            elif float(vol) < 0.2:
                premises.append(f"Realised volatility is {float(vol):.2f}, classified as low.")

            loss_streak = metrics.get("loss_streak", 0)
            if int(loss_streak) >= 3:
                premises.append(f"Current loss streak is {int(loss_streak)} trades.")

        # Model-related premises
        if decision_type == "retrain_model":
            age = metrics.get("age_days", 0)
            if float(age) > 7:
                premises.append(f"Model '{target}' was last trained {float(age):.0f} days ago (stale).")

            accuracy = metrics.get("accuracy", 0.5)
            peak = metrics.get("peak_accuracy", accuracy)
            if float(peak) > 0 and (float(peak) - float(accuracy)) / max(float(peak), 0.01) > 0.1:
                premises.append(
                    f"Model '{target}' accuracy dropped from {float(peak):.2f} to {float(accuracy):.2f}."
                )

            drift = metrics.get("drift_score", 0.0)
            if float(drift) > 0.3:
                premises.append(f"Feature drift score is {float(drift):.2f}, above threshold 0.3.")

        # Pause/resume premises
        if decision_type in ("pause_trading", "resume_trading"):
            dd = metrics.get("drawdown_pct", 0.0)
            if float(dd) > 10:
                premises.append(f"Drawdown is {float(dd):.1f}%, indicating capital preservation needed.")

            event = context.get("event")
            if event:
                premises.append(f"Upcoming macro event: {event}.")

        # Generic context
        if regime != "unknown":
            premises.append(f"Current market regime is '{regime}'.")

        # Fallback if no premises
        if not premises:
            premises.append(f"Decision '{decision_type}' for target '{target}' based on available metrics.")

        return premises

    def _build_conclusion(
        self,
        decision_type: str,
        target: str,
        premises: List[str],
    ) -> str:
        """Build a conclusion sentence."""
        n = len(premises)
        conclusions = {
            "deactivate_strategy": f"Strategy '{target}' should be deactivated based on {n} signal(s).",
            "activate_strategy": f"Strategy '{target}' should be activated or promoted based on {n} signal(s).",
            "adjust_position_size": f"Position size for '{target}' should be adjusted based on {n} factor(s).",
            "switch_venue": f"Execution should be routed to venue '{target}' based on {n} criterion(ia).",
            "retrain_model": f"Model '{target}' should be retrained based on {n} indicator(s).",
            "adjust_risk": f"Global risk posture should be adjusted based on {n} factor(s).",
            "pause_trading": f"Trading should be paused based on {n} warning(s).",
            "resume_trading": f"Trading can be resumed as {n} condition(s) have normalised.",
        }
        return conclusions.get(decision_type, f"Action '{decision_type}' on '{target}' recommended ({n} premises).")

    def _compute_confidence(
        self,
        decision_type: str,
        metrics: Dict[str, Any],
        premises: List[str],
    ) -> float:
        """Heuristic confidence based on evidence strength."""
        base = 0.3
        # More premises = more confidence
        base += min(0.3, len(premises) * 0.08)

        # Decision-specific boosts
        if decision_type == "pause_trading":
            dd = float(metrics.get("drawdown_pct", 0.0))
            if dd > 15:
                base += 0.3
            elif dd > 10:
                base += 0.2

        if decision_type == "retrain_model":
            drift = float(metrics.get("drift_score", 0.0))
            base += min(0.2, drift * 0.5)

        if decision_type in ("deactivate_strategy", "activate_strategy"):
            sharpe = abs(float(metrics.get("sharpe", 0.0)))
            base += min(0.2, sharpe * 0.1)

        return min(1.0, max(0.0, base))

    def _generate_alternatives(
        self,
        decision_type: str,
        target: str,
        metrics: Dict[str, Any],
    ) -> List[str]:
        """Generate alternative actions that were considered."""
        alts: List[str] = []

        if decision_type == "deactivate_strategy":
            alts.append(f"Reduce '{target}' weight by 50% instead of full deactivation.")
            alts.append(f"Keep '{target}' active but lower position sizing to minimum.")
            alts.append("Wait one more review cycle to confirm trend.")

        elif decision_type == "activate_strategy":
            alts.append(f"Keep '{target}' at current weight (do nothing).")
            alts.append(f"Increase '{target}' weight gradually over 3 cycles.")

        elif decision_type == "pause_trading":
            alts.append("Reduce position sizes by 75% instead of full pause.")
            alts.append("Close only the losing positions and keep winners running.")

        elif decision_type == "retrain_model":
            alts.append(f"Disable model '{target}' and fall back to rule-based signals.")
            alts.append(f"Schedule retrain for off-hours to avoid latency impact.")

        elif decision_type == "adjust_risk":
            alts.append("Maintain current risk level and reassess next cycle.")

        elif decision_type == "switch_venue":
            alts.append(f"Split execution 50/50 between venues instead of full switch.")

        elif decision_type == "resume_trading":
            alts.append("Resume at 50% position sizes for one cycle as a test.")

        return alts

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        try:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._db_path)
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("Failed to init ReasoningEngine DB: %s", exc)

    def _persist_chain(self, chain: ReasoningChain) -> None:
        with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                conn.execute(
                    "INSERT INTO reasoning_log (ts, decision_type, target, premises, evidence, conclusion, confidence, alternatives) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        chain.timestamp, chain.decision_type, chain.target,
                        json.dumps(chain.premises), json.dumps(chain.evidence, default=str),
                        chain.conclusion, chain.confidence, json.dumps(chain.alternatives),
                    ),
                )
                conn.commit()
                conn.close()
            except Exception as exc:
                logger.warning("Failed to persist reasoning chain: %s", exc)

    def _append_to_log(self, chain: ReasoningChain) -> None:
        with self._lock:
            self._log.append(chain)
            if len(self._log) > self._max_log_entries:
                self._log = self._log[-self._max_log_entries:]
