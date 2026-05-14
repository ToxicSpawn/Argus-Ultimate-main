#!/usr/bin/env python3
"""
Goal-Oriented Trading — explicit goal pursuit instead of blind PnL maximisation.

GoalManager lets ARGUS set, track, and adapt to concrete trading goals:
monthly return, max drawdown, Sharpe ratio, win rate, total trades, reduce exposure.

Strategy adjustments are generated automatically based on goal progress.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_GOAL_TYPES = frozenset({
    "monthly_return",
    "max_drawdown",
    "sharpe_ratio",
    "win_rate",
    "total_trades",
    "reduce_exposure",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Goal:
    """A single trading goal."""
    goal_id: str
    goal_type: str
    target: float
    deadline: Optional[str] = None  # ISO-8601 or None
    created_at: str = ""
    current_value: float = 0.0
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GoalProgress:
    """Progress report for a single goal."""
    goal: Goal
    current_value: float
    target: float
    pct_complete: float
    on_track: bool
    estimated_completion: Optional[str]
    status: str  # "ahead" | "on_track" | "behind" | "completed" | "failed"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["goal"] = self.goal.to_dict()
        return d


@dataclass
class StrategyAdjustment:
    """A recommended strategy adjustment to meet goals."""
    goal_id: str
    adjustment_type: str  # "increase_risk", "decrease_risk", "switch_strategy", etc.
    description: str
    magnitude: float  # 0..1 scale
    priority: int  # 1=highest

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# GoalManager
# ---------------------------------------------------------------------------

class GoalManager:
    """
    Goal-oriented trading controller.

    Parameters
    ----------
    config : dict, optional
        ``goal_manager`` section from unified_config.yaml.
    db_path : str, optional
        Override SQLite path.
    """

    _DEFAULTS: Dict[str, Any] = {
        "enabled": True,
        "db_path": "data/goals.db",
        "max_active_goals": 10,
        "risk_bump_pct": 10.0,
        "risk_cut_pct": 15.0,
        "exposure_cut_pct": 25.0,
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
        self._max_goals = int(cfg.get("max_active_goals", 10))
        self._risk_bump = float(cfg.get("risk_bump_pct", 10.0))
        self._risk_cut = float(cfg.get("risk_cut_pct", 15.0))
        self._exposure_cut = float(cfg.get("exposure_cut_pct", 25.0))

        self._goals: Dict[str, Goal] = {}
        self._lock = threading.Lock()

        db = db_path or str(cfg.get("db_path", "data/goals.db"))
        self._db_path = db
        self._init_db()
        self._load_goals()

        # Performance metrics — injected externally
        self._metrics: Dict[str, float] = {
            "monthly_return": 0.0,
            "current_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "total_trades": 0.0,
            "exposure_pct": 0.0,
        }

        logger.info("GoalManager initialised (enabled=%s, goals=%d)", self._enabled, len(self._goals))

    # ------------------------------------------------------------------
    # SQLite
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    goal_id TEXT PRIMARY KEY,
                    goal_type TEXT NOT NULL,
                    target REAL NOT NULL,
                    deadline TEXT,
                    created_at TEXT NOT NULL,
                    current_value REAL NOT NULL DEFAULT 0.0,
                    active INTEGER NOT NULL DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS goal_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_id TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    pct_complete REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()

    def _load_goals(self) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT goal_id, goal_type, target, deadline, created_at, current_value, active FROM goals"
                ).fetchall()
            for gid, gtype, target, deadline, created, current, active in rows:
                self._goals[gid] = Goal(
                    goal_id=gid, goal_type=gtype, target=target,
                    deadline=deadline, created_at=created,
                    current_value=current, active=bool(active),
                )
        except Exception as exc:
            logger.warning("GoalManager: load failed — %s", exc)

    def _persist_goal(self, goal: Goal) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO goals (goal_id, goal_type, target, deadline, created_at, current_value, active) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (goal.goal_id, goal.goal_type, goal.target,
                     goal.deadline, goal.created_at, goal.current_value, int(goal.active)),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("GoalManager: persist failed — %s", exc)

    # ------------------------------------------------------------------
    # Goal management
    # ------------------------------------------------------------------

    def set_goal(
        self,
        goal_type: str,
        target: float,
        deadline: Optional[str] = None,
        goal_id: Optional[str] = None,
    ) -> Goal:
        """
        Create or update a trading goal.

        Parameters
        ----------
        goal_type : str
            One of VALID_GOAL_TYPES.
        target : float
            Target value (e.g. 0.05 for 5% monthly return).
        deadline : str, optional
            ISO-8601 deadline (e.g. "2026-04-30").
        goal_id : str, optional
            Custom ID; defaults to goal_type.
        """
        if goal_type not in VALID_GOAL_TYPES:
            raise ValueError(f"Invalid goal_type '{goal_type}'. Must be one of {sorted(VALID_GOAL_TYPES)}")

        gid = goal_id or goal_type
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            if len(self._goals) >= self._max_goals and gid not in self._goals:
                raise RuntimeError(f"Maximum active goals ({self._max_goals}) reached")

            goal = Goal(
                goal_id=gid, goal_type=goal_type, target=target,
                deadline=deadline, created_at=now,
            )
            self._goals[gid] = goal
            self._persist_goal(goal)
            logger.info("Goal set: %s → target=%s deadline=%s", gid, target, deadline)
            return goal

    def remove_goal(self, goal_id: str) -> bool:
        """Remove a goal."""
        with self._lock:
            if goal_id in self._goals:
                self._goals[goal_id].active = False
                self._persist_goal(self._goals[goal_id])
                del self._goals[goal_id]
                return True
            return False

    def get_goals(self) -> List[Goal]:
        """Return all active goals."""
        with self._lock:
            return [g for g in self._goals.values() if g.active]

    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Inject current performance metrics from the trading system."""
        self._metrics.update(metrics)

    # ------------------------------------------------------------------
    # Progress evaluation
    # ------------------------------------------------------------------

    def evaluate_progress(self, goal_id: Optional[str] = None) -> List[GoalProgress]:
        """
        Evaluate progress toward one or all goals.

        Returns
        -------
        list of GoalProgress
        """
        goals = []
        with self._lock:
            if goal_id:
                if goal_id in self._goals:
                    goals = [self._goals[goal_id]]
            else:
                goals = [g for g in self._goals.values() if g.active]

        results: List[GoalProgress] = []
        for goal in goals:
            current = self._get_current_value(goal)
            goal.current_value = current
            self._persist_goal(goal)

            pct = self._compute_pct_complete(goal, current)
            on_track = self._is_on_track(goal, pct)
            estimated = self._estimate_completion(goal, current, pct)
            status = self._determine_status(goal, pct, current)

            progress = GoalProgress(
                goal=goal, current_value=current, target=goal.target,
                pct_complete=round(pct, 2), on_track=on_track,
                estimated_completion=estimated, status=status,
            )
            results.append(progress)

            # Persist history
            try:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute(
                        "INSERT INTO goal_history (goal_id, current_value, pct_complete, timestamp) VALUES (?, ?, ?, ?)",
                        (goal.goal_id, current, pct, datetime.now(timezone.utc).isoformat()),
                    )
                    conn.commit()
            except Exception:
                pass

        return results

    def _get_current_value(self, goal: Goal) -> float:
        """Map goal type to current metric value."""
        mapping = {
            "monthly_return": "monthly_return",
            "max_drawdown": "current_drawdown_pct",
            "sharpe_ratio": "sharpe_ratio",
            "win_rate": "win_rate",
            "total_trades": "total_trades",
            "reduce_exposure": "exposure_pct",
        }
        key = mapping.get(goal.goal_type, goal.goal_type)
        return self._metrics.get(key, 0.0)

    def _compute_pct_complete(self, goal: Goal, current: float) -> float:
        """Compute percentage completion toward goal."""
        if goal.target == 0:
            return 100.0 if current == 0 else 0.0

        if goal.goal_type in ("max_drawdown", "reduce_exposure"):
            # These are "keep below target" goals
            if current <= goal.target:
                return 100.0
            # How close are we (lower is better)?
            return max(0.0, min(100.0, (1.0 - (current - goal.target) / max(goal.target, 1e-9)) * 100.0))
        else:
            # Standard "reach target" goals
            return max(0.0, min(100.0, (current / goal.target) * 100.0))

    def _is_on_track(self, goal: Goal, pct: float) -> bool:
        """Determine if we're on track to meet the deadline."""
        if not goal.deadline:
            return pct >= 50.0

        try:
            deadline = datetime.fromisoformat(goal.deadline.replace("Z", "+00:00"))
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            return pct >= 50.0

        created = datetime.fromisoformat(goal.created_at.replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)

        total_duration = (deadline - created).total_seconds()
        elapsed = (now - created).total_seconds()
        if total_duration <= 0:
            return pct >= 100.0

        time_pct = (elapsed / total_duration) * 100.0
        # On track if progress pct >= time pct (with 10% grace)
        return pct >= (time_pct - 10.0)

    def _estimate_completion(self, goal: Goal, current: float, pct: float) -> Optional[str]:
        """Estimate when the goal will be met at current pace."""
        if pct >= 100.0:
            return "completed"
        if pct <= 0.0 or not goal.created_at:
            return None

        try:
            created = datetime.fromisoformat(goal.created_at.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            return None

        now = datetime.now(timezone.utc)
        elapsed = (now - created).total_seconds()
        if elapsed <= 0 or pct <= 0:
            return None

        # Linear extrapolation
        total_needed = elapsed * (100.0 / pct)
        remaining = total_needed - elapsed
        est = now + timedelta(seconds=remaining)
        return est.isoformat()

    def _determine_status(self, goal: Goal, pct: float, current: float) -> str:
        """Determine goal status."""
        if pct >= 100.0:
            return "completed"
        if goal.deadline:
            try:
                deadline = datetime.fromisoformat(goal.deadline.replace("Z", "+00:00"))
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) > deadline and pct < 100.0:
                    return "failed"
            except (ValueError, AttributeError):
                pass
        if self._is_on_track(goal, pct):
            if pct > 80.0:
                return "ahead"
            return "on_track"
        return "behind"

    # ------------------------------------------------------------------
    # Strategy adjustments
    # ------------------------------------------------------------------

    def adjust_strategy_for_goals(self) -> List[StrategyAdjustment]:
        """
        Generate strategy adjustments based on goal progress.

        Returns
        -------
        list of StrategyAdjustment
        """
        adjustments: List[StrategyAdjustment] = []
        progress_list = self.evaluate_progress()

        for prog in progress_list:
            goal = prog.goal

            if goal.goal_type == "monthly_return":
                if prog.status == "behind":
                    adjustments.append(StrategyAdjustment(
                        goal_id=goal.goal_id,
                        adjustment_type="increase_risk",
                        description=f"Behind on return target ({prog.current_value:.2%} vs {goal.target:.2%}) — increase position sizing by {self._risk_bump:.0f}%",
                        magnitude=self._risk_bump / 100.0,
                        priority=2,
                    ))
                elif prog.status == "ahead":
                    adjustments.append(StrategyAdjustment(
                        goal_id=goal.goal_id,
                        adjustment_type="lock_profits",
                        description=f"Ahead of return target ({prog.current_value:.2%} vs {goal.target:.2%}) — tighten stops, reduce risk",
                        magnitude=self._risk_cut / 100.0,
                        priority=3,
                    ))

            elif goal.goal_type == "max_drawdown":
                if prog.current_value > goal.target * 0.8:
                    adjustments.append(StrategyAdjustment(
                        goal_id=goal.goal_id,
                        adjustment_type="reduce_exposure",
                        description=f"Drawdown {prog.current_value:.1f}% approaching limit {goal.target:.1f}% — aggressively reduce exposure",
                        magnitude=self._exposure_cut / 100.0,
                        priority=1,
                    ))

            elif goal.goal_type == "sharpe_ratio":
                if prog.status == "behind":
                    adjustments.append(StrategyAdjustment(
                        goal_id=goal.goal_id,
                        adjustment_type="reduce_volatility",
                        description=f"Sharpe {prog.current_value:.2f} below target {goal.target:.2f} — reduce volatile positions",
                        magnitude=0.15,
                        priority=2,
                    ))

            elif goal.goal_type == "win_rate":
                if prog.status == "behind":
                    adjustments.append(StrategyAdjustment(
                        goal_id=goal.goal_id,
                        adjustment_type="switch_strategy",
                        description=f"Win rate {prog.current_value:.1%} below target {goal.target:.1%} — switch to higher-probability strategies",
                        magnitude=0.3,
                        priority=2,
                    ))

            elif goal.goal_type == "reduce_exposure":
                if prog.current_value > goal.target:
                    adjustments.append(StrategyAdjustment(
                        goal_id=goal.goal_id,
                        adjustment_type="reduce_exposure",
                        description=f"Exposure {prog.current_value:.1f}% above target {goal.target:.1f}% — close smallest positions first",
                        magnitude=min((prog.current_value - goal.target) / 100.0, 0.5),
                        priority=1,
                    ))

        # Sort by priority (1=highest)
        adjustments.sort(key=lambda a: a.priority)
        return adjustments

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def get_goal_dashboard(self) -> Dict[str, Any]:
        """Return a full goal dashboard."""
        progress_list = self.evaluate_progress()
        adjustments = self.adjust_strategy_for_goals()
        return {
            "goals": [p.to_dict() for p in progress_list],
            "adjustments": [a.to_dict() for a in adjustments],
            "total_goals": len(progress_list),
            "completed": sum(1 for p in progress_list if p.status == "completed"),
            "on_track": sum(1 for p in progress_list if p.on_track),
            "behind": sum(1 for p in progress_list if p.status == "behind"),
            "metrics": dict(self._metrics),
        }
