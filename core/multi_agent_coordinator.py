"""
Multi-Agent Trade Voting Coordinator — consensus-based trade decisions.

Multiple agents (alpha generators, risk models, execution analyzers, sentiment
engines) independently vote on trade direction for each symbol.  The coordinator
aggregates votes using weighted majority and enforces a minimum agreement
threshold before signalling consensus.

Agent types: ``alpha``, ``risk``, ``execution``, ``sentiment``.

Voting rounds are held in memory; completed rounds are persisted to SQLite
for accuracy tracking and historical analysis.

Usage:
    from core.multi_agent_coordinator import MultiAgentCoordinator

    coord = MultiAgentCoordinator()
    coord.register_agent("momentum_alpha", "alpha", weight=1.5)
    coord.register_agent("risk_overlay", "risk", weight=1.0)
    coord.submit_vote("momentum_alpha", "BTC/USD", "buy", 0.85, "Strong uptrend")
    coord.submit_vote("risk_overlay", "BTC/USD", "hold", 0.6, "VaR limit close")
    result = coord.get_consensus("BTC/USD")
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

_VALID_AGENT_TYPES = frozenset({"alpha", "risk", "execution", "sentiment"})
_VALID_DIRECTIONS = frozenset({"buy", "sell", "hold"})
_DEFAULT_AGREEMENT_THRESHOLD = 0.60  # 60% weighted agreement


@dataclass
class AgentInfo:
    """Registered agent metadata."""

    name: str
    agent_type: str
    weight: float = 1.0


@dataclass
class Vote:
    """A single agent vote for a symbol."""

    agent_name: str
    symbol: str
    direction: str
    confidence: float
    reason: str
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class ConsensusResult:
    """Aggregated consensus for a symbol."""

    symbol: str
    direction: str  # "buy" / "sell" / "hold"
    weighted_confidence: float
    votes_for: int
    votes_against: int
    agents_voted: List[str] = field(default_factory=list)
    unanimous: bool = False
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# MultiAgentCoordinator
# ---------------------------------------------------------------------------

class MultiAgentCoordinator:
    """Coordinate multi-agent trade voting with weighted consensus.

    Parameters
    ----------
    db_path : str or Path
        SQLite database for vote history and accuracy tracking.
    agreement_threshold : float
        Minimum fraction of weighted votes required for directional consensus
        (default 0.60 = 60%).
    """

    def __init__(
        self,
        db_path: str = "data/agent_votes.db",
        *,
        agreement_threshold: float = _DEFAULT_AGREEMENT_THRESHOLD,
    ) -> None:
        self.db_path = Path(db_path)
        self.agreement_threshold = agreement_threshold

        self._agents: Dict[str, AgentInfo] = {}
        # In-memory voting rounds: symbol -> list of Vote
        self._current_votes: Dict[str, List[Vote]] = defaultdict(list)
        self._lock = threading.Lock()
        self._ensure_db()

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _ensure_db(self) -> None:
        """Create SQLite tables if missing."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vote_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name  TEXT    NOT NULL,
                    symbol      TEXT    NOT NULL,
                    direction   TEXT    NOT NULL,
                    confidence  REAL    NOT NULL,
                    reason      TEXT    DEFAULT '',
                    ts          TEXT    NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consensus_history (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol          TEXT    NOT NULL,
                    direction       TEXT    NOT NULL,
                    weighted_conf   REAL    NOT NULL,
                    votes_for       INTEGER NOT NULL,
                    votes_against   INTEGER NOT NULL,
                    agents_voted    TEXT    DEFAULT '',
                    unanimous       INTEGER NOT NULL,
                    ts              TEXT    NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS outcome_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name      TEXT    NOT NULL,
                    symbol          TEXT    NOT NULL,
                    direction       TEXT    NOT NULL,
                    was_correct     INTEGER NOT NULL,
                    ts              TEXT    NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vote_agent_ts ON vote_history(agent_name, ts)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_outcome_agent ON outcome_log(agent_name, ts)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def register_agent(self, name: str, agent_type: str, weight: float = 1.0) -> None:
        """Register a voting agent.

        Parameters
        ----------
        name : str
            Unique agent identifier.
        agent_type : str
            One of ``alpha``, ``risk``, ``execution``, ``sentiment``.
        weight : float
            Voting weight (default 1.0).  Higher = more influence on consensus.

        Raises
        ------
        ValueError
            If agent_type is not recognised.
        """
        if agent_type not in _VALID_AGENT_TYPES:
            raise ValueError(
                f"Invalid agent_type '{agent_type}'; must be one of {sorted(_VALID_AGENT_TYPES)}"
            )
        with self._lock:
            self._agents[name] = AgentInfo(name=name, agent_type=agent_type, weight=max(0.01, weight))
        logger.info("MultiAgentCoordinator: registered agent '%s' type=%s weight=%.2f", name, agent_type, weight)

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    def submit_vote(
        self,
        agent_name: str,
        symbol: str,
        direction: str,
        confidence: float,
        reason: str = "",
    ) -> None:
        """Submit a directional vote from an agent.

        Parameters
        ----------
        agent_name : str
            Must be a previously registered agent.
        symbol : str
            Trading pair (e.g. "BTC/USD").
        direction : str
            ``buy``, ``sell``, or ``hold``.
        confidence : float
            Agent confidence in [0.0, 1.0].
        reason : str
            Human-readable justification.

        Raises
        ------
        ValueError
            If agent not registered or direction invalid.
        """
        if agent_name not in self._agents:
            raise ValueError(f"Agent '{agent_name}' is not registered")
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(f"Invalid direction '{direction}'; must be one of {sorted(_VALID_DIRECTIONS)}")

        confidence = max(0.0, min(1.0, confidence))
        vote = Vote(
            agent_name=agent_name,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            reason=reason,
        )

        with self._lock:
            self._current_votes[symbol].append(vote)

        # Persist to history
        self._persist_vote(vote)
        logger.debug(
            "MultiAgentCoordinator: vote from '%s' on %s — %s (conf=%.2f)",
            agent_name, symbol, direction, confidence,
        )

    def get_consensus(self, symbol: str) -> ConsensusResult:
        """Compute weighted consensus for a symbol from current voting round.

        Aggregation logic:
        1. Sum weighted confidence per direction (buy/sell/hold).
        2. Direction with highest weighted score wins if it exceeds the
           agreement threshold.
        3. Otherwise direction is ``hold``.

        After computing consensus, the current round for this symbol is
        cleared and the result is persisted to SQLite.

        Parameters
        ----------
        symbol : str
            Trading pair.

        Returns
        -------
        ConsensusResult
        """
        with self._lock:
            votes = list(self._current_votes.get(symbol, []))
            # Clear current round
            self._current_votes[symbol] = []

        if not votes:
            return ConsensusResult(
                symbol=symbol,
                direction="hold",
                weighted_confidence=0.0,
                votes_for=0,
                votes_against=0,
                agents_voted=[],
                unanimous=True,
            )

        # Weighted tally per direction
        direction_scores: Dict[str, float] = defaultdict(float)
        direction_counts: Dict[str, int] = defaultdict(int)
        agents_voted: List[str] = []
        total_weight = 0.0

        for v in votes:
            agent_info = self._agents.get(v.agent_name)
            weight = agent_info.weight if agent_info else 1.0
            weighted_score = weight * v.confidence
            direction_scores[v.direction] += weighted_score
            direction_counts[v.direction] += 1
            total_weight += weighted_score
            if v.agent_name not in agents_voted:
                agents_voted.append(v.agent_name)

        # Find winner
        if total_weight < 1e-12:
            winner = "hold"
            winner_score = 0.0
        else:
            winner = max(direction_scores, key=direction_scores.get)  # type: ignore[arg-type]
            winner_score = direction_scores[winner]

        # Check agreement threshold
        agreement_pct = winner_score / total_weight if total_weight > 1e-12 else 0.0
        if agreement_pct < self.agreement_threshold:
            winner = "hold"

        # Votes for/against
        votes_for = direction_counts.get(winner, 0)
        votes_against = len(votes) - votes_for

        # Unanimous check
        directions_seen = set(v.direction for v in votes)
        unanimous = len(directions_seen) == 1

        weighted_confidence = agreement_pct

        result = ConsensusResult(
            symbol=symbol,
            direction=winner,
            weighted_confidence=round(weighted_confidence, 4),
            votes_for=votes_for,
            votes_against=votes_against,
            agents_voted=agents_voted,
            unanimous=unanimous,
        )

        self._persist_consensus(result)
        logger.info(
            "MultiAgentCoordinator: consensus for %s — %s (conf=%.2f, for=%d, against=%d, unanimous=%s)",
            symbol, result.direction, result.weighted_confidence,
            result.votes_for, result.votes_against, result.unanimous,
        )
        return result

    # ------------------------------------------------------------------
    # Accuracy tracking
    # ------------------------------------------------------------------

    def record_outcome(self, agent_name: str, symbol: str, direction: str, was_correct: bool) -> None:
        """Record whether an agent's vote was correct (for accuracy tracking).

        Parameters
        ----------
        agent_name : str
            Agent that voted.
        symbol : str
            Trading pair.
        direction : str
            The direction the agent voted.
        was_correct : bool
            Whether the vote aligned with the actual market outcome.
        """
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO outcome_log (agent_name, symbol, direction, was_correct, ts) VALUES (?, ?, ?, ?, ?)",
                    (agent_name, symbol, direction, int(was_correct), ts),
                )
        logger.debug("MultiAgentCoordinator: recorded outcome for '%s' — correct=%s", agent_name, was_correct)

    def get_agent_accuracy(self, agent_name: str, lookback_days: int = 30) -> Dict[str, Any]:
        """Compute accuracy statistics for an agent.

        Parameters
        ----------
        agent_name : str
            Agent identifier.
        lookback_days : int
            How many days of history to consider.

        Returns
        -------
        dict
            Keys: accuracy (float), avg_confidence (float), vote_count (int).
        """
        with self._lock:
            with self._connect() as conn:
                # Accuracy from outcome_log
                cursor = conn.execute(
                    """
                    SELECT was_correct FROM outcome_log
                    WHERE agent_name = ?
                    ORDER BY ts DESC LIMIT ?
                    """,
                    (agent_name, lookback_days * 10),  # Generous limit
                )
                outcomes = [row[0] for row in cursor.fetchall()]

                # Average confidence from vote_history
                cursor2 = conn.execute(
                    """
                    SELECT confidence FROM vote_history
                    WHERE agent_name = ?
                    ORDER BY ts DESC LIMIT ?
                    """,
                    (agent_name, lookback_days * 10),
                )
                confidences = [row[0] for row in cursor2.fetchall()]

        if not outcomes:
            return {"accuracy": 0.0, "avg_confidence": 0.0, "vote_count": 0}

        accuracy = sum(outcomes) / len(outcomes)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "avg_confidence": round(avg_conf, 4),
            "vote_count": len(outcomes),
        }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist_vote(self, v: Vote) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO vote_history (agent_name, symbol, direction, confidence, reason, ts) VALUES (?, ?, ?, ?, ?, ?)",
                    (v.agent_name, v.symbol, v.direction, v.confidence, v.reason, v.timestamp),
                )

    def _persist_consensus(self, r: ConsensusResult) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO consensus_history
                        (symbol, direction, weighted_conf, votes_for, votes_against, agents_voted, unanimous, ts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (r.symbol, r.direction, r.weighted_confidence, r.votes_for,
                     r.votes_against, json.dumps(r.agents_voted), int(r.unanimous), r.timestamp),
                )

    def clear_votes(self, symbol: Optional[str] = None) -> None:
        """Clear in-memory votes for a symbol (or all symbols if None).

        Parameters
        ----------
        symbol : str or None
            If None, clears all current voting rounds.
        """
        with self._lock:
            if symbol is None:
                self._current_votes.clear()
            else:
                self._current_votes.pop(symbol, None)
        logger.debug("MultiAgentCoordinator: cleared votes for %s", symbol or "all symbols")
