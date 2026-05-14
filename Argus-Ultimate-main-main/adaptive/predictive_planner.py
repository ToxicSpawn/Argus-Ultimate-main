#!/usr/bin/env python3
"""
Predictive Planner — forward-looking trade planning.

Instead of reacting to current signals, the planner PRE-COMPUTES trade plans
with explicit entry conditions, exit conditions, expected returns, risk/reward
ratios, and contingency plans.  Plans are monitored each cycle and triggered
when conditions align.

SQLite persistence at data/trade_plans.db.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    """A single trigger condition."""
    indicator: str  # e.g. "price", "rsi", "volume_ratio", "regime"
    operator: str  # "lt", "gt", "eq", "between", "in"
    value: Any  # threshold or [low, high] for "between"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def evaluate(self, current_value: Any) -> bool:
        """Check if this condition is satisfied."""
        try:
            if self.operator == "lt":
                return float(current_value) < float(self.value)
            elif self.operator == "gt":
                return float(current_value) > float(self.value)
            elif self.operator == "eq":
                return str(current_value).lower() == str(self.value).lower()
            elif self.operator == "between":
                if isinstance(self.value, (list, tuple)) and len(self.value) == 2:
                    return float(self.value[0]) <= float(current_value) <= float(self.value[1])
            elif self.operator == "in":
                if isinstance(self.value, (list, tuple)):
                    return current_value in self.value
        except (TypeError, ValueError):
            pass
        return False


@dataclass
class TradePlan:
    """A complete forward-looking trade plan."""
    plan_id: str
    symbol: str
    direction: str  # "long" / "short"
    entry_conditions: List[Condition] = field(default_factory=list)
    exit_conditions: List[Condition] = field(default_factory=list)
    expected_return_pct: float = 0.0
    risk_reward_ratio: float = 0.0
    confidence: float = 0.5
    contingency_plan: str = ""
    horizon_hours: int = 24
    created_at: str = ""
    expires_at: str = ""
    status: str = "active"  # active, triggered, expired, cancelled
    entry_price_target: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    size_pct: float = 1.0  # position size as fraction of default

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["entry_conditions"] = [c.to_dict() for c in self.entry_conditions]
        d["exit_conditions"] = [c.to_dict() for c in self.exit_conditions]
        return d


# ---------------------------------------------------------------------------
# Plan generators (rule-based templates)
# ---------------------------------------------------------------------------

def _generate_support_bounce_plan(
    symbol: str, support: float, resistance: float, current_price: float,
    rsi: float = 50.0, avg_volume_ratio: float = 1.0,
) -> Optional[TradePlan]:
    """Generate a plan to buy at support with exit at resistance."""
    if current_price <= 0 or support <= 0 or resistance <= support:
        return None

    entry_target = support * 1.005  # just above support
    stop = support * 0.98  # 2% below support
    take_profit = resistance * 0.995  # just below resistance

    risk = abs(entry_target - stop)
    reward = abs(take_profit - entry_target)
    rr = reward / risk if risk > 0 else 0

    if rr < 1.5:
        return None  # not worth it

    now = datetime.now(timezone.utc)
    plan = TradePlan(
        plan_id=str(uuid.uuid4())[:12],
        symbol=symbol,
        direction="long",
        entry_conditions=[
            Condition("price", "lt", round(entry_target, 2),
                      f"Price drops to support zone ({entry_target:.2f})"),
            Condition("rsi", "lt", 40.0, "RSI below 40 (oversold)"),
            Condition("volume_ratio", "gt", 1.5, "Volume spike >1.5x avg"),
        ],
        exit_conditions=[
            Condition("price", "gt", round(take_profit, 2),
                      f"Price reaches resistance ({take_profit:.2f})"),
            Condition("price", "lt", round(stop, 2),
                      f"Stop loss at {stop:.2f}"),
        ],
        expected_return_pct=round((take_profit - entry_target) / entry_target * 100, 2),
        risk_reward_ratio=round(rr, 2),
        confidence=round(min(0.4 + rr * 0.1, 0.8), 2),
        contingency_plan=f"If BTC drops >3%, cancel plan. If support breaks, do not chase.",
        horizon_hours=24,
        created_at=now.isoformat(),
        expires_at=(now + timedelta(hours=24)).isoformat(),
        entry_price_target=round(entry_target, 2),
        stop_loss=round(stop, 2),
        take_profit=round(take_profit, 2),
    )
    return plan


def _generate_breakout_plan(
    symbol: str, resistance: float, current_price: float,
    volatility_pct: float = 2.0,
) -> Optional[TradePlan]:
    """Generate a plan to buy a breakout above resistance."""
    if current_price <= 0 or resistance <= 0:
        return None
    if current_price > resistance:
        return None  # already broken out

    entry_target = resistance * 1.005
    stop = resistance * 0.98
    take_profit = entry_target * (1 + volatility_pct / 100 * 2)

    risk = abs(entry_target - stop)
    reward = abs(take_profit - entry_target)
    rr = reward / risk if risk > 0 else 0

    now = datetime.now(timezone.utc)
    plan = TradePlan(
        plan_id=str(uuid.uuid4())[:12],
        symbol=symbol,
        direction="long",
        entry_conditions=[
            Condition("price", "gt", round(resistance, 2),
                      f"Price breaks above resistance ({resistance:.2f})"),
            Condition("volume_ratio", "gt", 2.0, "Volume spike >2x avg on breakout"),
        ],
        exit_conditions=[
            Condition("price", "gt", round(take_profit, 2),
                      f"Take profit at {take_profit:.2f}"),
            Condition("price", "lt", round(stop, 2),
                      f"Stop loss if breakout fails at {stop:.2f}"),
        ],
        expected_return_pct=round((take_profit - entry_target) / entry_target * 100, 2),
        risk_reward_ratio=round(rr, 2),
        confidence=0.55,
        contingency_plan=f"If volume dries up post-breakout, exit immediately — likely false breakout.",
        horizon_hours=12,
        created_at=now.isoformat(),
        expires_at=(now + timedelta(hours=12)).isoformat(),
        entry_price_target=round(entry_target, 2),
        stop_loss=round(stop, 2),
        take_profit=round(take_profit, 2),
    )
    return plan


def _generate_mean_revert_plan(
    symbol: str, mean_price: float, current_price: float,
    std_dev: float = 0.0,
) -> Optional[TradePlan]:
    """Generate a plan to fade an extended move back to the mean."""
    if current_price <= 0 or mean_price <= 0 or std_dev <= 0:
        return None

    deviation = (current_price - mean_price) / std_dev
    if abs(deviation) < 1.5:
        return None  # not extended enough

    if deviation > 0:
        # Price above mean — short
        direction = "short"
        entry_target = mean_price + std_dev * 2.0
        stop = mean_price + std_dev * 3.0
        take_profit = mean_price + std_dev * 0.5
    else:
        # Price below mean — long
        direction = "long"
        entry_target = mean_price - std_dev * 2.0
        stop = mean_price - std_dev * 3.0
        take_profit = mean_price - std_dev * 0.5

    risk = abs(entry_target - stop)
    reward = abs(take_profit - entry_target)
    rr = reward / risk if risk > 0 else 0

    now = datetime.now(timezone.utc)
    plan = TradePlan(
        plan_id=str(uuid.uuid4())[:12],
        symbol=symbol,
        direction=direction,
        entry_conditions=[
            Condition("price", "gt" if direction == "short" else "lt",
                      round(entry_target, 2),
                      f"Price reaches 2-sigma zone ({entry_target:.2f})"),
            Condition("regime", "eq", "mean_revert",
                      "Regime supports mean reversion"),
        ],
        exit_conditions=[
            Condition("price", "lt" if direction == "short" else "gt",
                      round(take_profit, 2),
                      f"Take profit near mean ({take_profit:.2f})"),
            Condition("price", "gt" if direction == "short" else "lt",
                      round(stop, 2),
                      f"Stop at 3-sigma ({stop:.2f})"),
        ],
        expected_return_pct=round(reward / entry_target * 100, 2),
        risk_reward_ratio=round(rr, 2),
        confidence=round(min(0.4 + abs(deviation) * 0.1, 0.75), 2),
        contingency_plan="Cancel if regime shifts to trending. Mean reversion fails in trends.",
        horizon_hours=48,
        created_at=now.isoformat(),
        expires_at=(now + timedelta(hours=48)).isoformat(),
        entry_price_target=round(entry_target, 2),
        stop_loss=round(stop, 2),
        take_profit=round(take_profit, 2),
    )
    return plan


# ---------------------------------------------------------------------------
# PredictivePlanner
# ---------------------------------------------------------------------------

class PredictivePlanner:
    """
    Forward-looking trade planning engine.

    Parameters
    ----------
    config : dict, optional
        ``predictive_planner`` section from unified_config.yaml.
    db_path : str, optional
        Override SQLite path.
    """

    _DEFAULTS: Dict[str, Any] = {
        "enabled": True,
        "db_path": "data/trade_plans.db",
        "max_active_plans": 20,
        "min_risk_reward": 1.5,
        "default_horizon_hours": 24,
        "min_confidence": 0.4,
        # Phase W2 — MCTS re-scoring
        "use_mcts_planning": False,
        "mcts_n_simulations": 100,
        "mcts_max_depth": 5,
        "mcts_time_budget_s": 0.25,
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        db_path: Optional[str] = None,
        world_model: Optional[Any] = None,
    ) -> None:
        cfg = dict(self._DEFAULTS)
        if config:
            cfg.update(config)
        self._cfg = cfg
        self._enabled = bool(cfg.get("enabled", True))
        self._max_plans = int(cfg.get("max_active_plans", 20))
        self._min_rr = float(cfg.get("min_risk_reward", 1.5))
        self._default_horizon = int(cfg.get("default_horizon_hours", 24))
        self._min_conf = float(cfg.get("min_confidence", 0.4))
        self._use_mcts = bool(cfg.get("use_mcts_planning", False))
        self._world_model = world_model

        self._plans: Dict[str, TradePlan] = {}
        self._lock = threading.Lock()

        db = db_path or str(cfg.get("db_path", "data/trade_plans.db"))
        self._db_path = db
        self._init_db()
        self._load_plans()

        logger.info(
            "PredictivePlanner initialised (enabled=%s, plans=%d, mcts=%s)",
            self._enabled, len(self._plans), self._use_mcts,
        )

    # ------------------------------------------------------------------
    # SQLite
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_plans (
                    plan_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    plan_data TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plan_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_id TEXT NOT NULL,
                    triggered INTEGER NOT NULL DEFAULT 0,
                    actual_pnl REAL,
                    notes TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()

    def _load_plans(self) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT plan_id, plan_data, status FROM trade_plans WHERE status='active'"
                ).fetchall()
            for pid, data_json, status in rows:
                try:
                    d = json.loads(data_json)
                    plan = self._dict_to_plan(d)
                    plan.status = status
                    self._plans[pid] = plan
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("PredictivePlanner: load failed — %s", exc)

    def _persist_plan(self, plan: TradePlan) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO trade_plans (plan_id, symbol, direction, plan_data, status, created_at, expires_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (plan.plan_id, plan.symbol, plan.direction,
                     json.dumps(plan.to_dict()), plan.status,
                     plan.created_at, plan.expires_at),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("PredictivePlanner: persist failed — %s", exc)

    @staticmethod
    def _dict_to_plan(d: Dict[str, Any]) -> TradePlan:
        """Reconstruct a TradePlan from its dict representation."""
        entry_conds = [
            Condition(**c) for c in d.get("entry_conditions", [])
        ]
        exit_conds = [
            Condition(**c) for c in d.get("exit_conditions", [])
        ]
        return TradePlan(
            plan_id=d["plan_id"],
            symbol=d["symbol"],
            direction=d["direction"],
            entry_conditions=entry_conds,
            exit_conditions=exit_conds,
            expected_return_pct=d.get("expected_return_pct", 0.0),
            risk_reward_ratio=d.get("risk_reward_ratio", 0.0),
            confidence=d.get("confidence", 0.5),
            contingency_plan=d.get("contingency_plan", ""),
            horizon_hours=d.get("horizon_hours", 24),
            created_at=d.get("created_at", ""),
            expires_at=d.get("expires_at", ""),
            status=d.get("status", "active"),
            entry_price_target=d.get("entry_price_target", 0.0),
            stop_loss=d.get("stop_loss", 0.0),
            take_profit=d.get("take_profit", 0.0),
            size_pct=d.get("size_pct", 1.0),
        )

    # ------------------------------------------------------------------
    # Plan generation
    # ------------------------------------------------------------------

    def generate_plan(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        horizon_hours: int = 24,
    ) -> List[TradePlan]:
        """
        Generate trade plans for a symbol based on current market data.

        Parameters
        ----------
        symbol : str
        market_data : dict
            Keys: price, support, resistance, rsi, volume_ratio, volatility_pct,
            mean_price, std_dev, regime.
        horizon_hours : int

        Returns
        -------
        list of TradePlan
        """
        generated: List[TradePlan] = []
        price = market_data.get("price", 0.0)
        support = market_data.get("support", 0.0)
        resistance = market_data.get("resistance", 0.0)
        rsi = market_data.get("rsi", 50.0)
        vol_ratio = market_data.get("volume_ratio", 1.0)
        vol_pct = market_data.get("volatility_pct", 2.0)
        mean_price = market_data.get("mean_price", 0.0)
        std_dev = market_data.get("std_dev", 0.0)

        # Support bounce
        if support > 0 and resistance > 0:
            plan = _generate_support_bounce_plan(
                symbol, support, resistance, price, rsi, vol_ratio,
            )
            if plan and plan.risk_reward_ratio >= self._min_rr and plan.confidence >= self._min_conf:
                plan.horizon_hours = horizon_hours
                plan.expires_at = (
                    datetime.now(timezone.utc) + timedelta(hours=horizon_hours)
                ).isoformat()
                generated.append(plan)

        # Breakout
        if resistance > 0:
            plan = _generate_breakout_plan(symbol, resistance, price, vol_pct)
            if plan and plan.risk_reward_ratio >= self._min_rr:
                generated.append(plan)

        # Mean reversion
        if mean_price > 0 and std_dev > 0:
            plan = _generate_mean_revert_plan(symbol, mean_price, price, std_dev)
            if plan and plan.risk_reward_ratio >= self._min_rr:
                generated.append(plan)

        # Phase W2: optionally re-score plan confidence via MCTS rollouts.
        if self._use_mcts and self._world_model is not None and generated:
            for plan in generated:
                try:
                    mcts_score = self._mcts_rescore_plan(plan, market_data)
                    if mcts_score is not None:
                        # Blend: 60% original rule-based confidence, 40% MCTS
                        blended = 0.6 * plan.confidence + 0.4 * mcts_score
                        plan.confidence = round(max(0.0, min(1.0, blended)), 2)
                except Exception as exc:
                    logger.debug("MCTS re-score failed for %s: %s", plan.plan_id, exc)

        # Store generated plans
        with self._lock:
            for plan in generated:
                if len(self._plans) < self._max_plans:
                    self._plans[plan.plan_id] = plan
                    self._persist_plan(plan)

        return generated

    # ------------------------------------------------------------------
    # Phase W2 — MCTS plan re-scoring
    # ------------------------------------------------------------------

    def _mcts_rescore_plan(
        self,
        plan: TradePlan,
        market_data: Dict[str, Any],
    ) -> Optional[float]:
        """
        Score a plan by running MCTS rollouts from its entry state.

        Uses ``core/world_model.py`` as the generative model and treats the
        plan's entry as a ``long``/``short``/``flat`` decision in action
        space ``[-1, 0, 1]``. Returns a normalized confidence in ``[0, 1]``
        based on best-action expected value, or ``None`` if MCTS cannot run.
        """
        try:
            from core.planning.mcts import MCTS, MCTSConfig
        except ImportError:
            return None

        wm = self._world_model
        if wm is None or not hasattr(wm, "encode_state") or not hasattr(wm, "predict_next"):
            return None

        try:
            import numpy as _np
            obs = {
                "price": float(market_data.get("price", 0.0)),
                "volume": float(market_data.get("volume", 0.0)),
                "regime": str(market_data.get("regime", "ranging")).lower(),
                "regime_confidence": float(market_data.get("regime_confidence", 0.5)),
                "volatility": float(market_data.get("volatility_pct", 0.0)) / 100.0,
                "spread": float(market_data.get("spread", 0.0)),
                "position": 0.0,
                "unrealized_pnl": 0.0,
            }
            state = wm.encode_state(obs)

            mcts_cfg = MCTSConfig(
                n_simulations=int(self._cfg.get("mcts_n_simulations", 100)),
                max_depth=int(self._cfg.get("mcts_max_depth", 5)),
                time_budget_s=float(self._cfg.get("mcts_time_budget_s", 0.25)),
            )
            mcts = MCTS(
                world_model=wm,
                action_space=[-1.0, 0.0, 1.0],
                config=mcts_cfg,
            )
            result = mcts.plan(state)

            action_values = result.get("action_values", {})
            if not action_values:
                return None

            # Map plan direction to the MCTS action we care about
            target_action = "1.0" if plan.direction == "long" else "-1.0"
            target_value = float(action_values.get(target_action, 0.0))
            max_value = max(float(v) for v in action_values.values())

            if max_value <= 0.0:
                return 0.5  # neutral

            # Normalize: target reward / max reward across actions.
            # If target is the max, score approaches 1.0; else proportionally lower.
            norm = max(0.0, min(1.0, target_value / max_value))
            return float(norm)
        except Exception as exc:
            logger.debug("MCTS rescore inner failure: %s", exc)
            return None

    def add_plan(self, plan: TradePlan) -> None:
        """Add a manually created plan."""
        with self._lock:
            if len(self._plans) >= self._max_plans:
                self.expire_stale_plans()
            self._plans[plan.plan_id] = plan
        self._persist_plan(plan)

    # ------------------------------------------------------------------
    # Plan checking
    # ------------------------------------------------------------------

    def check_plan_triggers(self, current_data: Dict[str, Dict[str, Any]]) -> List[TradePlan]:
        """
        Check which plans have all entry conditions met.

        Parameters
        ----------
        current_data : dict
            Keyed by symbol: {"BTC/USD": {"price": 65000, "rsi": 35, ...}}

        Returns
        -------
        list of TradePlan that were triggered
        """
        self.expire_stale_plans()
        triggered: List[TradePlan] = []

        with self._lock:
            for plan in list(self._plans.values()):
                if plan.status != "active":
                    continue

                sym_data = current_data.get(plan.symbol, {})
                if not sym_data:
                    continue

                # Check if ALL entry conditions are met
                all_met = True
                for cond in plan.entry_conditions:
                    val = sym_data.get(cond.indicator)
                    if val is None or not cond.evaluate(val):
                        all_met = False
                        break

                if all_met and plan.entry_conditions:
                    plan.status = "triggered"
                    self._persist_plan(plan)
                    triggered.append(plan)
                    logger.info("Plan triggered: %s %s %s", plan.plan_id, plan.direction, plan.symbol)

        return triggered

    def check_exit_conditions(
        self, plan_id: str, current_data: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Check if exit conditions are met for a triggered plan.

        Returns
        -------
        (should_exit, reason)
        """
        plan = self._plans.get(plan_id)
        if not plan:
            return False, "plan not found"

        for cond in plan.exit_conditions:
            val = current_data.get(cond.indicator)
            if val is not None and cond.evaluate(val):
                return True, cond.description

        return False, ""

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def get_active_plans(self) -> List[TradePlan]:
        """Return all active (non-expired, non-triggered) plans."""
        with self._lock:
            return [p for p in self._plans.values() if p.status == "active"]

    def get_triggered_plans(self) -> List[TradePlan]:
        """Return plans that have been triggered."""
        with self._lock:
            return [p for p in self._plans.values() if p.status == "triggered"]

    def cancel_plan(self, plan_id: str) -> bool:
        """Cancel a plan."""
        with self._lock:
            plan = self._plans.get(plan_id)
            if plan and plan.status == "active":
                plan.status = "cancelled"
                self._persist_plan(plan)
                return True
        return False

    def expire_stale_plans(self) -> int:
        """Remove plans past their horizon. Returns count expired."""
        now = datetime.now(timezone.utc)
        expired = 0
        with self._lock:
            for plan in list(self._plans.values()):
                if plan.status != "active":
                    continue
                try:
                    exp = datetime.fromisoformat(plan.expires_at.replace("Z", "+00:00"))
                    if exp.tzinfo is None:
                        exp = exp.replace(tzinfo=timezone.utc)
                    if now > exp:
                        plan.status = "expired"
                        self._persist_plan(plan)
                        expired += 1
                except (ValueError, AttributeError):
                    continue
        return expired

    def record_outcome(self, plan_id: str, triggered: bool, actual_pnl: float = 0.0, notes: str = "") -> None:
        """Record the outcome of a plan."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT INTO plan_outcomes (plan_id, triggered, actual_pnl, notes, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (plan_id, int(triggered), actual_pnl, notes, datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("PredictivePlanner: outcome persist failed — %s", exc)

    def get_plan_success_rate(self) -> Dict[str, Any]:
        """Return success rate statistics for completed plans."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT triggered, actual_pnl FROM plan_outcomes"
                ).fetchall()
        except Exception:
            rows = []

        if not rows:
            return {"total": 0, "triggered": 0, "profitable": 0, "success_rate": 0.0}

        triggered = [r for r in rows if r[0]]
        profitable = [r for r in triggered if r[1] and r[1] > 0]
        return {
            "total": len(rows),
            "triggered": len(triggered),
            "profitable": len(profitable),
            "success_rate": round(len(profitable) / max(len(triggered), 1), 4),
            "avg_pnl": round(sum(r[1] or 0 for r in triggered) / max(len(triggered), 1), 4),
        }
