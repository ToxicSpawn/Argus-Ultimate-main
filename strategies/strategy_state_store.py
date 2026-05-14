"""
Strategy State Persistence + Cooldown Enforcement.

Provides SQLite-backed persistence for per-strategy trading state (win/loss
streaks, PnL, trade counts) and enforces cooldowns after consecutive-loss
streaks.  Designed to survive process restarts.

Usage:
    store = StrategyStateStore("data/strategy_states.db")
    store.load_all()                       # on startup
    store.check_cooldown("unified_engine")  # before accepting a signal
    store.update_after_trade("unified_engine", pnl=-12.5, timestamp=time.time())
    store.save_all()                       # on shutdown (also auto-saved)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DB_PATH = "data/strategy_states.db"
DEFAULT_MAX_CONSECUTIVE_LOSSES = 5
DEFAULT_COOLDOWN_MINUTES = 60


class StrategyState:
    """In-memory representation of a single strategy's persistent state."""

    __slots__ = (
        "strategy_name",
        "trade_count",
        "win_count",
        "loss_count",
        "total_pnl",
        "consecutive_losses",
        "consecutive_wins",
        "last_trade_time",
        "cooldown_until",
        "parameters",
    )

    def __init__(
        self,
        strategy_name: str,
        trade_count: int = 0,
        win_count: int = 0,
        loss_count: int = 0,
        total_pnl: float = 0.0,
        consecutive_losses: int = 0,
        consecutive_wins: int = 0,
        last_trade_time: Optional[float] = None,
        cooldown_until: Optional[float] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.strategy_name = strategy_name
        self.trade_count = trade_count
        self.win_count = win_count
        self.loss_count = loss_count
        self.total_pnl = total_pnl
        self.consecutive_losses = consecutive_losses
        self.consecutive_wins = consecutive_wins
        self.last_trade_time = last_trade_time
        self.cooldown_until = cooldown_until
        self.parameters = parameters or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "total_pnl": self.total_pnl,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "last_trade_time": self.last_trade_time,
            "cooldown_until": self.cooldown_until,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyState":
        return cls(
            strategy_name=str(d.get("strategy_name", "unknown")),
            trade_count=int(d.get("trade_count", 0)),
            win_count=int(d.get("win_count", 0)),
            loss_count=int(d.get("loss_count", 0)),
            total_pnl=float(d.get("total_pnl", 0.0)),
            consecutive_losses=int(d.get("consecutive_losses", 0)),
            consecutive_wins=int(d.get("consecutive_wins", 0)),
            last_trade_time=d.get("last_trade_time"),
            cooldown_until=d.get("cooldown_until"),
            parameters=d.get("parameters") or {},
        )

    @property
    def win_rate(self) -> float:
        if self.trade_count == 0:
            return 0.0
        return self.win_count / self.trade_count


class StrategyStateStore:
    """
    SQLite-backed persistence for per-strategy trading state with cooldown
    enforcement.

    Thread-safe: all DB access is serialised via a threading.Lock.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        max_consecutive_losses: int = DEFAULT_MAX_CONSECUTIVE_LOSSES,
        cooldown_minutes: int = DEFAULT_COOLDOWN_MINUTES,
    ) -> None:
        self.db_path = str(db_path)
        self.max_consecutive_losses = int(max_consecutive_losses)
        self.cooldown_minutes = int(cooldown_minutes)
        self._lock = threading.Lock()
        self._states: Dict[str, StrategyState] = {}

        # Ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # DB setup
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS strategy_states (
                        strategy_name   TEXT PRIMARY KEY,
                        trade_count     INTEGER NOT NULL DEFAULT 0,
                        win_count       INTEGER NOT NULL DEFAULT 0,
                        loss_count      INTEGER NOT NULL DEFAULT 0,
                        total_pnl       REAL    NOT NULL DEFAULT 0.0,
                        consecutive_losses INTEGER NOT NULL DEFAULT 0,
                        consecutive_wins   INTEGER NOT NULL DEFAULT 0,
                        last_trade_time REAL,
                        cooldown_until  REAL,
                        parameters      TEXT,
                        updated_at      REAL NOT NULL
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Core persistence API
    # ------------------------------------------------------------------
    def save_state(self, strategy_name: str, state: dict) -> None:
        """Upsert a single strategy state into SQLite."""
        s = StrategyState.from_dict({**state, "strategy_name": strategy_name})
        self._states[strategy_name] = s
        self._write_one(s)

    def load_state(self, strategy_name: str) -> Optional[dict]:
        """Load a single strategy state.  Returns None if not found."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                row = conn.execute(
                    "SELECT * FROM strategy_states WHERE strategy_name = ?",
                    (strategy_name,),
                ).fetchone()
                if row is None:
                    return None
                state = self._row_to_state(row)
                self._states[strategy_name] = state
                return state.to_dict()
            finally:
                conn.close()

    def load_all(self) -> Dict[str, dict]:
        """Load all persisted strategy states.  Returns {name: dict}."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute("SELECT * FROM strategy_states").fetchall()
                result: Dict[str, dict] = {}
                for row in rows:
                    state = self._row_to_state(row)
                    self._states[state.strategy_name] = state
                    result[state.strategy_name] = state.to_dict()
                return result
            finally:
                conn.close()

    def save_all(self) -> None:
        """Persist every in-memory state to SQLite (for shutdown)."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                for s in self._states.values():
                    self._upsert(conn, s)
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------
    def update_after_trade(
        self,
        strategy_name: str,
        pnl: float,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Atomic update after a trade completes.

        Increments counts, updates PnL, tracks consecutive streaks, and
        activates cooldown when the consecutive-loss threshold is breached.

        Returns the updated state dict.
        """
        ts = float(timestamp or time.time())
        s = self._states.get(strategy_name) or StrategyState(strategy_name=strategy_name)
        s.trade_count += 1
        s.total_pnl += float(pnl)
        s.last_trade_time = ts

        if pnl >= 0:
            s.win_count += 1
            s.consecutive_wins += 1
            s.consecutive_losses = 0
            # A win clears any active cooldown
            if s.cooldown_until is not None and s.cooldown_until > 0:
                logger.info(
                    "Strategy '%s' cooldown cleared by winning trade (PnL=%.4f)",
                    strategy_name, pnl,
                )
                s.cooldown_until = None
        else:
            s.loss_count += 1
            s.consecutive_losses += 1
            s.consecutive_wins = 0
            # Activate cooldown if threshold breached
            if s.consecutive_losses >= self.max_consecutive_losses:
                cooldown_end = ts + (self.cooldown_minutes * 60)
                s.cooldown_until = cooldown_end
                logger.warning(
                    "Strategy '%s' entering cooldown until %.0f "
                    "(%d consecutive losses >= threshold %d, cooldown=%d min)",
                    strategy_name,
                    cooldown_end,
                    s.consecutive_losses,
                    self.max_consecutive_losses,
                    self.cooldown_minutes,
                )

        self._states[strategy_name] = s
        self._write_one(s)
        return s.to_dict()

    # ------------------------------------------------------------------
    # Cooldown enforcement
    # ------------------------------------------------------------------
    def check_cooldown(self, strategy_name: str, now: Optional[float] = None) -> bool:
        """
        Return True if the strategy is currently in cooldown and signals
        should be dropped.
        """
        s = self._states.get(strategy_name)
        if s is None:
            return False
        cd = s.cooldown_until
        if cd is None or cd <= 0:
            return False
        current = float(now or time.time())
        if current < cd:
            return True
        # Cooldown has expired — clear it
        s.cooldown_until = None
        self._states[strategy_name] = s
        return False

    def cooldown_remaining_seconds(self, strategy_name: str, now: Optional[float] = None) -> float:
        """Seconds remaining in cooldown (0.0 if not in cooldown)."""
        s = self._states.get(strategy_name)
        if s is None:
            return 0.0
        cd = s.cooldown_until
        if cd is None or cd <= 0:
            return 0.0
        current = float(now or time.time())
        return max(0.0, cd - current)

    def clear_cooldown(self, strategy_name: str) -> None:
        """Manually clear a strategy's cooldown."""
        s = self._states.get(strategy_name)
        if s is not None:
            s.cooldown_until = None
            self._write_one(s)

    # ------------------------------------------------------------------
    # FIX 16: Aggressive strategy dampening
    # ------------------------------------------------------------------

    def get_strategy_multiplier(self, strategy_name: str, portfolio_value: float = 1000.0) -> float:
        """
        Return a sizing multiplier for the given strategy based on recent
        performance.  Losing strategies are dampened aggressively; winners
        get a modest boost.

        Rules:
          - consecutive_losses >= 5: return 0.10 (near-disabled)
          - PnL_last < -3%: return 0.25
          - PnL_last < -1%: return 0.50
          - PnL_last > +3%: return 1.30
          - Otherwise: 1.0
        """
        s = self._states.get(strategy_name)
        if s is None:
            return 1.0

        # Consecutive losses is the most aggressive dampener
        if s.consecutive_losses >= 5:
            return 0.10

        if s.trade_count < 5:
            return 1.0  # not enough history

        # Approximate PnL percentage
        pnl_pct = s.total_pnl / max(portfolio_value, 1.0) * 100.0

        if pnl_pct < -3.0:
            return 0.25
        if pnl_pct < -1.0:
            return 0.50
        if pnl_pct > 3.0:
            return 1.30
        return 1.0

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_state(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        s = self._states.get(strategy_name)
        return s.to_dict() if s else None

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        return {k: v.to_dict() for k, v in self._states.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _write_one(self, s: StrategyState) -> None:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                self._upsert(conn, s)
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _upsert(conn: sqlite3.Connection, s: StrategyState) -> None:
        params_json = json.dumps(s.parameters) if s.parameters else "{}"
        conn.execute(
            """
            INSERT INTO strategy_states
                (strategy_name, trade_count, win_count, loss_count,
                 total_pnl, consecutive_losses, consecutive_wins,
                 last_trade_time, cooldown_until, parameters, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(strategy_name) DO UPDATE SET
                trade_count        = excluded.trade_count,
                win_count          = excluded.win_count,
                loss_count         = excluded.loss_count,
                total_pnl          = excluded.total_pnl,
                consecutive_losses = excluded.consecutive_losses,
                consecutive_wins   = excluded.consecutive_wins,
                last_trade_time    = excluded.last_trade_time,
                cooldown_until     = excluded.cooldown_until,
                parameters         = excluded.parameters,
                updated_at         = excluded.updated_at
            """,
            (
                s.strategy_name,
                s.trade_count,
                s.win_count,
                s.loss_count,
                s.total_pnl,
                s.consecutive_losses,
                s.consecutive_wins,
                s.last_trade_time,
                s.cooldown_until,
                params_json,
                time.time(),
            ),
        )

    @staticmethod
    def _row_to_state(row: sqlite3.Row) -> StrategyState:
        params_raw = row["parameters"]
        try:
            params = json.loads(params_raw) if params_raw else {}
        except (json.JSONDecodeError, TypeError):
            params = {}
        return StrategyState(
            strategy_name=str(row["strategy_name"]),
            trade_count=int(row["trade_count"]),
            win_count=int(row["win_count"]),
            loss_count=int(row["loss_count"]),
            total_pnl=float(row["total_pnl"]),
            consecutive_losses=int(row["consecutive_losses"]),
            consecutive_wins=int(row["consecutive_wins"]),
            last_trade_time=row["last_trade_time"],
            cooldown_until=row["cooldown_until"],
            parameters=params,
        )


# ---------------------------------------------------------------------------
# Parameter Validation
# ---------------------------------------------------------------------------
def validate_strategy_parameters(config: Any) -> List[str]:
    """
    Validate strategy-engine parameters from a config object.

    Returns a list of error strings.  Empty list means all valid.
    """
    errors: List[str] = []

    def _get(attr: str, default: Any = None) -> Any:
        return getattr(config, attr, default)

    # RSI periods
    rsi_period = _get("se_rsi_period", 14)
    if rsi_period is not None:
        try:
            if int(rsi_period) < 2:
                errors.append(f"se_rsi_period must be >= 2, got {rsi_period}")
        except (TypeError, ValueError):
            errors.append(f"se_rsi_period must be an integer, got {rsi_period!r}")

    # Bollinger period
    bb_period = _get("se_bb_period", 20)
    if bb_period is not None:
        try:
            if int(bb_period) < 2:
                errors.append(f"se_bb_period must be >= 2, got {bb_period}")
        except (TypeError, ValueError):
            errors.append(f"se_bb_period must be an integer, got {bb_period!r}")

    # Confidence thresholds in [0, 1]
    for attr in ("min_signal_confidence", "live_min_signal_confidence"):
        val = _get(attr)
        if val is not None:
            try:
                fval = float(val)
                if not (0.0 <= fval <= 1.0):
                    errors.append(f"{attr} must be in [0, 1], got {fval}")
            except (TypeError, ValueError):
                errors.append(f"{attr} must be a float, got {val!r}")

    # RSI thresholds
    buy_rsi = _get("se_buy_rsi", 35.0)
    sell_rsi = _get("se_sell_rsi", 65.0)
    if buy_rsi is not None and sell_rsi is not None:
        try:
            b, s = float(buy_rsi), float(sell_rsi)
            if not (0 < b < 100):
                errors.append(f"se_buy_rsi must be in (0, 100), got {b}")
            if not (0 < s < 100):
                errors.append(f"se_sell_rsi must be in (0, 100), got {s}")
            if b >= s:
                errors.append(f"se_buy_rsi ({b}) must be < se_sell_rsi ({s})")
        except (TypeError, ValueError):
            pass

    # BB position thresholds in [0, 1]
    for attr in ("se_buy_bb", "se_sell_bb"):
        val = _get(attr)
        if val is not None:
            try:
                fval = float(val)
                if not (0.0 <= fval <= 1.0):
                    errors.append(f"{attr} must be in [0, 1], got {fval}")
            except (TypeError, ValueError):
                errors.append(f"{attr} must be a float, got {val!r}")

    # Stop-loss / take-profit percentages > 0
    for attr in ("stop_loss_pct", "take_profit_pct"):
        val = _get(attr)
        if val is not None:
            try:
                fval = float(val)
                if fval <= 0:
                    errors.append(f"{attr} must be > 0, got {fval}")
                if fval > 1.0:
                    errors.append(f"{attr} seems too large (> 100%), got {fval}")
            except (TypeError, ValueError):
                errors.append(f"{attr} must be a float, got {val!r}")

    # Max concurrent signals
    mcs = _get("max_concurrent_signals", 2)
    if mcs is not None:
        try:
            if int(mcs) < 1:
                errors.append(f"max_concurrent_signals must be >= 1, got {mcs}")
        except (TypeError, ValueError):
            errors.append(f"max_concurrent_signals must be an integer, got {mcs!r}")

    # Lookback / bar count
    lookback = _get("ohlcv_lookback", None)
    if lookback is not None:
        try:
            if int(lookback) < 1:
                errors.append(f"ohlcv_lookback must be >= 1, got {lookback}")
        except (TypeError, ValueError):
            errors.append(f"ohlcv_lookback must be an integer, got {lookback!r}")

    # Max consecutive losses (for cooldown)
    mcl = _get("strategy_max_consecutive_losses", DEFAULT_MAX_CONSECUTIVE_LOSSES)
    if mcl is not None:
        try:
            if int(mcl) < 1:
                errors.append(f"strategy_max_consecutive_losses must be >= 1, got {mcl}")
        except (TypeError, ValueError):
            errors.append(f"strategy_max_consecutive_losses must be an integer, got {mcl!r}")

    # Cooldown minutes
    cdm = _get("strategy_cooldown_minutes", DEFAULT_COOLDOWN_MINUTES)
    if cdm is not None:
        try:
            if int(cdm) < 0:
                errors.append(f"strategy_cooldown_minutes must be >= 0, got {cdm}")
        except (TypeError, ValueError):
            errors.append(f"strategy_cooldown_minutes must be an integer, got {cdm!r}")

    return errors
