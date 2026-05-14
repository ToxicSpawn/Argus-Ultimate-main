"""
Multi-Strategy Ensemble Voting — consensus-based signal aggregation.

When multiple strategies independently agree on a direction for the same
symbol, the probability of a correct signal is significantly higher than
any single strategy alone.  This module collects votes from all active
strategies, computes accuracy-weighted consensus, and only emits a
ConsensusResult when agreement exceeds configurable thresholds.

Accuracy weighting
------------------
Each voter's weight is proportional to their historical accuracy.  New
voters start with a neutral weight of 0.5.  After 20+ outcomes, their
weight tracks their actual win rate (exponentially smoothed).

Conflict resolution
-------------------
If buy and sell votes are close to balanced, no consensus is emitted
(ambiguity → stay flat).  Only when agreement_pct >= min_agreement
does the module emit a direction.

Example:
  3 strategies vote BUY at confidence [0.7, 0.8, 0.6], accuracies [0.65, 0.72, 0.55]
  1 strategy votes SELL at confidence [0.5], accuracy [0.48]
  Weighted BUY = 0.7×0.65 + 0.8×0.72 + 0.6×0.55 = 1.36
  Weighted SELL = 0.5×0.48 = 0.24
  Agreement = 1.36 / (1.36 + 0.24) = 0.85 → PASS (>0.60)
  → ConsensusResult(direction="buy", avg_confidence=0.70, agreement_pct=0.85)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = "data/ensemble_voter.db"
_DEFAULT_MIN_VOTES = 2
_DEFAULT_MIN_AGREEMENT = 0.60
_DEFAULT_VOTE_TTL_SECONDS = 300  # votes older than 5 min are discarded
_EMA_ALPHA = 0.1  # exponential smoothing for accuracy updates
_MIN_OUTCOMES_FOR_WEIGHT = 20


@dataclass
class ConsensusResult:
    """Result of an ensemble vote aggregation."""

    symbol: str
    direction: str  # "buy" or "sell"
    avg_confidence: float
    num_votes: int
    agreement_pct: float
    voters: List[str]
    weighted_score: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return (
            f"ConsensusResult({self.symbol} {self.direction} "
            f"conf={self.avg_confidence:.3f} agree={self.agreement_pct:.1%} "
            f"votes={self.num_votes} voters={self.voters})"
        )


@dataclass
class _Vote:
    """Internal representation of a single strategy vote."""

    strategy_name: str
    symbol: str
    direction: str  # "buy" or "sell"
    confidence: float
    timestamp: float  # time.time()


class EnsembleVoter:
    """
    Multi-strategy ensemble voting system.

    Collects votes from multiple strategies, weights them by historical
    accuracy, and emits a consensus direction when agreement is sufficient.

    Parameters
    ----------
    db_path : str
        SQLite database for persisting voter accuracy history.
    min_votes : int
        Minimum number of votes required to form consensus.
    min_agreement : float
        Minimum agreement percentage (0–1) to emit consensus.
    vote_ttl_seconds : float
        Time-to-live for votes; stale votes are ignored.
    """

    def __init__(
        self,
        db_path: str = _DEFAULT_DB_PATH,
        min_votes: int = _DEFAULT_MIN_VOTES,
        min_agreement: float = _DEFAULT_MIN_AGREEMENT,
        vote_ttl_seconds: float = _DEFAULT_VOTE_TTL_SECONDS,
    ) -> None:
        self.db_path = db_path
        self.min_votes = max(1, min_votes)
        self.min_agreement = max(0.1, min(1.0, min_agreement))
        self.vote_ttl_seconds = max(10.0, vote_ttl_seconds)

        self._lock = threading.Lock()

        # Active votes: {symbol: [_Vote, ...]}
        self._votes: Dict[str, List[_Vote]] = defaultdict(list)

        # Voter accuracy: {strategy_name: {"wins": int, "losses": int, "accuracy": float}}
        self._voter_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"wins": 0, "losses": 0, "accuracy": 0.5, "total_pnl": 0.0}
        )

        # Outcome tracking: {(symbol, direction, timestamp_bucket): [voter_names]}
        self._pending_outcomes: Dict[str, Dict] = {}

        self._init_db()
        self._load_voter_stats()

        logger.info(
            "EnsembleVoter initialised (min_votes=%d, min_agreement=%.2f, "
            "vote_ttl=%.0fs, db=%s)",
            self.min_votes,
            self.min_agreement,
            self.vote_ttl_seconds,
            self.db_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_vote(
        self,
        strategy_name: str,
        symbol: str,
        direction: str,
        confidence: float,
    ) -> None:
        """
        Register a vote from a strategy.

        Parameters
        ----------
        strategy_name : str
            Unique name of the voting strategy.
        symbol : str
            Trading pair (e.g. ``"BTC/USD"``).
        direction : str
            ``"buy"`` or ``"sell"``.
        confidence : float
            Strategy's confidence in this direction (0.0–1.0).
        """
        direction = direction.lower().strip()
        if direction not in ("buy", "sell"):
            logger.warning("EnsembleVoter: invalid direction '%s' from %s", direction, strategy_name)
            return

        confidence = max(0.0, min(1.0, float(confidence)))

        vote = _Vote(
            strategy_name=strategy_name,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timestamp=time.time(),
        )

        with self._lock:
            # Remove any existing vote from this strategy for this symbol
            self._votes[symbol] = [
                v for v in self._votes[symbol]
                if v.strategy_name != strategy_name
            ]
            self._votes[symbol].append(vote)

        logger.debug(
            "Vote received: %s → %s %s conf=%.3f",
            strategy_name, symbol, direction, confidence,
        )

    def get_consensus(
        self,
        symbol: str,
        min_votes: Optional[int] = None,
        min_agreement: Optional[float] = None,
    ) -> Optional[ConsensusResult]:
        """
        Compute consensus for *symbol* from active votes.

        Parameters
        ----------
        symbol : str
            Trading pair.
        min_votes : int or None
            Override instance ``min_votes`` for this call.
        min_agreement : float or None
            Override instance ``min_agreement`` for this call.

        Returns
        -------
        ConsensusResult or None
            Consensus if thresholds are met, else None.
        """
        if min_votes is None:
            min_votes = self.min_votes
        if min_agreement is None:
            min_agreement = self.min_agreement

        with self._lock:
            # Prune stale votes
            now = time.time()
            self._votes[symbol] = [
                v for v in self._votes[symbol]
                if (now - v.timestamp) <= self.vote_ttl_seconds
            ]

            active_votes = self._votes.get(symbol, [])

        if len(active_votes) < min_votes:
            return None

        # Compute accuracy-weighted scores per direction
        buy_score = 0.0
        sell_score = 0.0
        buy_confidences: List[float] = []
        sell_confidences: List[float] = []
        buy_voters: List[str] = []
        sell_voters: List[str] = []

        for vote in active_votes:
            weight = self._get_voter_weight(vote.strategy_name)
            weighted = vote.confidence * weight

            if vote.direction == "buy":
                buy_score += weighted
                buy_confidences.append(vote.confidence)
                buy_voters.append(vote.strategy_name)
            else:
                sell_score += weighted
                sell_confidences.append(vote.confidence)
                sell_voters.append(vote.strategy_name)

        total_score = buy_score + sell_score
        if total_score <= 0:
            return None

        # Determine winning direction
        if buy_score >= sell_score:
            direction = "buy"
            agreement = buy_score / total_score
            confidences = buy_confidences
            voters = buy_voters
            score = buy_score
        else:
            direction = "sell"
            agreement = sell_score / total_score
            confidences = sell_confidences
            voters = sell_voters
            score = sell_score

        if agreement < min_agreement:
            logger.debug(
                "Consensus for %s: %s agreement=%.3f < %.3f (rejected)",
                symbol, direction, agreement, min_agreement,
            )
            return None

        if len(voters) < min_votes:
            return None

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        result = ConsensusResult(
            symbol=symbol,
            direction=direction,
            avg_confidence=round(avg_conf, 4),
            num_votes=len(active_votes),
            agreement_pct=round(agreement, 4),
            voters=voters,
            weighted_score=round(score, 4),
        )

        # Record pending outcome for later tracking
        outcome_key = f"{symbol}_{direction}_{int(time.time())}"
        self._pending_outcomes[outcome_key] = {
            "symbol": symbol,
            "direction": direction,
            "voters": list(voters),
            "timestamp": time.time(),
        }

        logger.info(
            "Consensus: %s %s (agreement=%.1f%%, votes=%d, conf=%.3f, voters=%s)",
            symbol, direction, agreement * 100, len(active_votes), avg_conf, voters,
        )
        return result

    def record_outcome(
        self,
        symbol: str,
        direction: str,
        pnl: float,
    ) -> None:
        """
        Record the outcome of a consensus trade to update voter accuracy.

        Parameters
        ----------
        symbol : str
            Trading pair.
        direction : str
            Direction that was traded (``"buy"`` or ``"sell"``).
        pnl : float
            Realised P&L from the trade.
        """
        direction = direction.lower().strip()
        won = pnl > 0

        # Find matching pending outcome
        matched_voters: List[str] = []
        to_remove: List[str] = []

        for key, outcome in self._pending_outcomes.items():
            if outcome["symbol"] == symbol and outcome["direction"] == direction:
                matched_voters.extend(outcome["voters"])
                to_remove.append(key)

        for key in to_remove:
            del self._pending_outcomes[key]

        # If no pending outcome found, update all voters who voted this direction
        if not matched_voters:
            with self._lock:
                for vote in self._votes.get(symbol, []):
                    if vote.direction == direction:
                        matched_voters.append(vote.strategy_name)

        # Update voter stats
        with self._lock:
            for voter in set(matched_voters):
                stats = self._voter_stats[voter]
                if won:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1
                stats["total_pnl"] += pnl

                total = stats["wins"] + stats["losses"]
                if total >= _MIN_OUTCOMES_FOR_WEIGHT:
                    new_accuracy = stats["wins"] / total
                    # Exponential smoothing
                    stats["accuracy"] = (
                        _EMA_ALPHA * new_accuracy
                        + (1 - _EMA_ALPHA) * stats["accuracy"]
                    )

                self._persist_voter_stats(voter, stats)

        logger.info(
            "Outcome recorded: %s %s pnl=%.4f (%s) → updated %d voters",
            symbol, direction, pnl, "WIN" if won else "LOSS", len(set(matched_voters)),
        )

    def get_voter_accuracy(self, strategy_name: str) -> float:
        """
        Return the historical accuracy for a specific voter.

        Returns 0.5 (neutral) if the voter has no history.
        """
        with self._lock:
            stats = self._voter_stats.get(strategy_name)
            if stats is None:
                return 0.5
            return round(stats["accuracy"], 4)

    def get_all_voter_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return accuracy stats for all known voters."""
        with self._lock:
            return {
                name: dict(stats)
                for name, stats in self._voter_stats.items()
            }

    def get_active_votes(self, symbol: str) -> List[Dict[str, Any]]:
        """Return all active (non-stale) votes for *symbol*."""
        now = time.time()
        with self._lock:
            return [
                {
                    "strategy": v.strategy_name,
                    "direction": v.direction,
                    "confidence": v.confidence,
                    "age_seconds": round(now - v.timestamp, 1),
                }
                for v in self._votes.get(symbol, [])
                if (now - v.timestamp) <= self.vote_ttl_seconds
            ]

    def clear_votes(self, symbol: Optional[str] = None) -> None:
        """Clear votes for *symbol*, or all votes if None."""
        with self._lock:
            if symbol is not None:
                self._votes[symbol] = []
            else:
                self._votes.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_voter_weight(self, strategy_name: str) -> float:
        """Return the accuracy-based weight for a voter."""
        stats = self._voter_stats.get(strategy_name)
        if stats is None:
            return 0.5
        return stats["accuracy"]

    def _init_db(self) -> None:
        """Create SQLite tables."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS voter_stats (
                    strategy_name TEXT PRIMARY KEY,
                    wins INTEGER NOT NULL DEFAULT 0,
                    losses INTEGER NOT NULL DEFAULT 0,
                    accuracy REAL NOT NULL DEFAULT 0.5,
                    total_pnl REAL NOT NULL DEFAULT 0.0,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _persist_voter_stats(self, voter: str, stats: Dict[str, Any]) -> None:
        """Persist voter stats to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO voter_stats
                        (strategy_name, wins, losses, accuracy, total_pnl, updated_at)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
                    """,
                    (voter, stats["wins"], stats["losses"], stats["accuracy"], stats["total_pnl"]),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            logger.exception("Failed to persist voter stats for %s", voter)

    def _load_voter_stats(self) -> None:
        """Load voter accuracy history from the database."""
        if not os.path.exists(self.db_path):
            return
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    "SELECT strategy_name, wins, losses, accuracy, total_pnl FROM voter_stats"
                )
                for name, wins, losses, accuracy, total_pnl in cursor:
                    self._voter_stats[name] = {
                        "wins": wins,
                        "losses": losses,
                        "accuracy": accuracy,
                        "total_pnl": total_pnl,
                    }
                if self._voter_stats:
                    logger.info(
                        "Loaded accuracy data for %d voters from database",
                        len(self._voter_stats),
                    )
            finally:
                conn.close()
        except Exception:
            logger.exception("Failed to load voter stats from %s", self.db_path)
