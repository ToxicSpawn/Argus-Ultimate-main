"""
core/autonomous_brain.py --- Central Decision Engine (the "CEO" of Argus).

Makes high-level trading decisions every cycle by synthesising market state,
strategy performance, model health, execution quality, and risk posture into
a ranked list of AutonomousAction items. Decisions are fully logged with
reasoning chains and persisted to SQLite for post-hoc analysis.

Peak-potential upgrade (#1)
---------------------------
* Hierarchical routing via a MetaAgent that fans out decision-making to
  specialised sub-agents instead of relying purely on a flat evaluator stack.
* Sub-agents included here: RiskSubAgent, StrategySubAgent,
  ExecutionSubAgent, and ModelSubAgent.
* Conflict arbitration: higher-priority / higher-confidence actions win when
  sub-agents disagree on the same target; conflicts are stored in memory and
  persisted in the decision snapshot.
* AutonomousBrain remains standalone and backwards-compatible: if meta-agent
  routing is disabled, the original flat evaluator pipeline still runs.
* market_state["mm_spread"] support: the execution sub-agent can react to the
  standalone market_maker_spread.py module by reducing risk or pausing MM.

Usage::

    brain = AutonomousBrain(config=cfg)
    actions = brain.decide(market_state)
    for a in actions:
        logger.info(a.action_type, a.target, a.reason)

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AutonomousAction:
    """A single decision produced by the AutonomousBrain."""

    action_type: str
    target: str
    params: Dict[str, Any]
    reason: str
    confidence: float
    priority: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentConflict:
    """Conflict record produced during meta-agent arbitration."""

    winner_agent: str
    loser_agent: str
    winner_action: str
    loser_action: str
    target: str
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_ACTION_TYPES = frozenset({
    "activate_strategy",
    "deactivate_strategy",
    "adjust_position_size",
    "switch_venue",
    "retrain_model",
    "adjust_risk",
    "pause_trading",
    "resume_trading",
})

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_DB_PATH = os.path.join(_DB_DIR, "autonomous_decisions.db")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    action_type TEXT    NOT NULL,
    target      TEXT    NOT NULL,
    params      TEXT    NOT NULL,
    reason      TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    priority    INTEGER NOT NULL,
    cycle_id    TEXT,
    market_snapshot TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts);
CREATE INDEX IF NOT EXISTS idx_decisions_action ON decisions(action_type);

CREATE TABLE IF NOT EXISTS cross_session_memory (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    ts    TEXT NOT NULL
);
"""

_AGENT_PRIORITY = {
    "RiskSubAgent": 1,
    "ExecutionSubAgent": 2,
    "StrategySubAgent": 3,
    "ModelSubAgent": 4,
}

_TERMINAL_ACTIONS = frozenset({"pause_trading", "resume_trading"})


# ---------------------------------------------------------------------------
# Sub-agent framework
# ---------------------------------------------------------------------------

class DecisionSubAgent:
    """Base class for hierarchical decision sub-agents."""

    name: str = "DecisionSubAgent"

    def __init__(self, brain: "AutonomousBrain") -> None:
        self.brain = brain

    def decide(self, market_state: Dict[str, Any]) -> List[AutonomousAction]:
        raise NotImplementedError


class StrategySubAgent(DecisionSubAgent):
    name = "StrategySubAgent"

    def decide(self, market_state: Dict[str, Any]) -> List[AutonomousAction]:
        return self.brain._evaluate_strategies(market_state)


class ModelSubAgent(DecisionSubAgent):
    name = "ModelSubAgent"

    def decide(self, market_state: Dict[str, Any]) -> List[AutonomousAction]:
        return self.brain._evaluate_models(market_state)


class RiskSubAgent(DecisionSubAgent):
    name = "RiskSubAgent"

    def decide(self, market_state: Dict[str, Any]) -> List[AutonomousAction]:
        actions: List[AutonomousAction] = []
        actions.extend(self.brain._evaluate_risk(market_state))
        actions.extend(self.brain._evaluate_pause_resume(market_state))

        mm = market_state.get("mm_spread") or {}
        inv_util = float(mm.get("inventory_utilisation", 0.0))
        last_spread_bps = mm.get("last_spread_bps")
        fill_win_rate = float(mm.get("fill_win_rate_50", 1.0))
        if inv_util >= 0.90:
            actions.append(AutonomousAction(
                action_type="adjust_risk",
                target="market_maker",
                params={
                    "inventory_utilisation": round(inv_util, 4),
                    "position_multiplier": 0.5,
                },
                reason=(
                    f"MM inventory utilisation {inv_util:.2f} near cap; "
                    f"halving MM risk until inventory mean-reverts."
                ),
                confidence=min(1.0, 0.70 + 0.2 * inv_util),
                priority=1,
            ))
        if last_spread_bps is not None and float(last_spread_bps) >= 120.0 and fill_win_rate < 0.45:
            actions.append(AutonomousAction(
                action_type="pause_trading",
                target="market_maker",
                params={
                    "spread_bps": round(float(last_spread_bps), 3),
                    "fill_win_rate_50": round(fill_win_rate, 4),
                },
                reason=(
                    f"MM spread widened to {float(last_spread_bps):.1f}bps with weak fill quality "
                    f"(win_rate={fill_win_rate:.2f}); pausing MM flow."
                ),
                confidence=0.82,
                priority=1,
            ))
        return actions


class ExecutionSubAgent(DecisionSubAgent):
    name = "ExecutionSubAgent"

    def decide(self, market_state: Dict[str, Any]) -> List[AutonomousAction]:
        actions: List[AutonomousAction] = []
        actions.extend(self.brain._evaluate_venues(market_state))
        actions.extend(self.brain._evaluate_position_sizing(market_state))

        mm = market_state.get("mm_spread") or {}
        if mm:
            spread_bps = mm.get("last_spread_bps")
            active = bool(mm.get("last_active", False))
            regime = str(mm.get("last_regime", "UNKNOWN"))
            if active and spread_bps is not None and float(spread_bps) > 80.0:
                actions.append(AutonomousAction(
                    action_type="adjust_position_size",
                    target="market_maker",
                    params={
                        "spread_bps": round(float(spread_bps), 3),
                        "new_size_pct": 0.5,
                        "regime": regime,
                    },
                    reason=(
                        f"Execution sub-agent detected wide MM spread ({float(spread_bps):.1f}bps) "
                        f"in regime '{regime}'; reducing quote size."
                    ),
                    confidence=0.68,
                    priority=2,
                ))
        return actions


class MetaAgent:
    """Hierarchical router that merges specialised sub-agent outputs."""

    def __init__(self, brain: "AutonomousBrain", sub_agents: Optional[Sequence[DecisionSubAgent]] = None) -> None:
        self.brain = brain
        self.sub_agents: List[DecisionSubAgent] = list(sub_agents) if sub_agents is not None else [
            RiskSubAgent(brain),
            ExecutionSubAgent(brain),
            StrategySubAgent(brain),
            ModelSubAgent(brain),
        ]

    def decide(self, market_state: Dict[str, Any]) -> Tuple[List[AutonomousAction], List[AgentConflict], Dict[str, List[AutonomousAction]]]:
        all_tagged: List[Tuple[str, AutonomousAction]] = []
        by_agent: Dict[str, List[AutonomousAction]] = {}
        for agent in self.sub_agents:
            try:
                actions = list(agent.decide(market_state))
            except Exception:
                logger.exception("MetaAgent: sub-agent %s failed", agent.name)
                actions = []
            by_agent[agent.name] = actions
            all_tagged.extend((agent.name, a) for a in actions)
        merged, conflicts = self._arbitrate(all_tagged)
        return merged, conflicts, by_agent

    def _arbitrate(self, tagged_actions: Sequence[Tuple[str, AutonomousAction]]) -> Tuple[List[AutonomousAction], List[AgentConflict]]:
        winners: Dict[Tuple[str, str], Tuple[str, AutonomousAction]] = {}
        conflicts: List[AgentConflict] = []
        terminal_winner: Optional[Tuple[str, AutonomousAction]] = None

        for agent_name, action in tagged_actions:
            if action.action_type in _TERMINAL_ACTIONS:
                terminal_winner = self._choose_better(terminal_winner, (agent_name, action), conflicts, target_key="system-terminal")
                continue

            key = (action.target, action.action_type)
            current = winners.get(key)
            winners[key] = self._choose_better(current, (agent_name, action), conflicts, target_key=action.target)

        merged = [winner for _, winner in winners.values()]
        if terminal_winner is not None:
            terminal_action = terminal_winner[1]
            if terminal_action.action_type == "pause_trading":
                merged = [a for a in merged if a.action_type != "resume_trading"]
            elif terminal_action.action_type == "resume_trading":
                merged = [a for a in merged if a.action_type != "pause_trading"]
            merged.append(terminal_action)

        merged.sort(key=lambda a: (a.priority, -a.confidence))
        return merged, conflicts

    def _choose_better(
        self,
        current: Optional[Tuple[str, AutonomousAction]],
        challenger: Tuple[str, AutonomousAction],
        conflicts: List[AgentConflict],
        target_key: str,
    ) -> Tuple[str, AutonomousAction]:
        if current is None:
            return challenger
        winner = current
        loser = challenger
        cur_agent, cur_action = current
        ch_agent, ch_action = challenger
        cur_rank = (_AGENT_PRIORITY.get(cur_agent, 99), cur_action.priority, -cur_action.confidence)
        ch_rank = (_AGENT_PRIORITY.get(ch_agent, 99), ch_action.priority, -ch_action.confidence)
        if ch_rank < cur_rank:
            winner, loser = challenger, current
        elif ch_rank == cur_rank and ch_action.confidence > cur_action.confidence:
            winner, loser = challenger, current
        if winner != current or loser != challenger:
            conflicts.append(AgentConflict(
                winner_agent=winner[0],
                loser_agent=loser[0],
                winner_action=winner[1].action_type,
                loser_action=loser[1].action_type,
                target=target_key,
                reason=(
                    f"{winner[0]} outranked {loser[0]} on target '{target_key}' "
                    f"(winner priority={winner[1].priority}, conf={winner[1].confidence:.2f})."
                ),
            ))
        return winner


# ---------------------------------------------------------------------------
# AutonomousBrain
# ---------------------------------------------------------------------------

class AutonomousBrain:
    """Central decision engine that produces ranked AutonomousAction lists."""

    def __init__(
        self,
        *,
        config: Any = None,
        db_path: Optional[str] = None,
    ) -> None:
        self._config = config
        self._cfg = self._extract_section(config)

        self._enabled: bool = bool(self._cfg.get("enabled", True))
        self._max_actions_per_cycle: int = int(self._cfg.get("max_actions_per_cycle", 10))
        self._min_confidence: float = float(self._cfg.get("min_confidence", 0.3))
        self._pause_drawdown_pct: float = float(self._cfg.get("pause_drawdown_pct", 15.0))
        self._resume_recovery_pct: float = float(self._cfg.get("resume_recovery_pct", 5.0))
        self._model_stale_days: int = int(self._cfg.get("model_stale_days", 7))
        self._strategy_disable_sharpe: float = float(self._cfg.get("strategy_disable_sharpe", 0.0))
        self._strategy_promote_sharpe: float = float(self._cfg.get("strategy_promote_sharpe", 1.5))

        self._kelly_max_fraction: float = float(self._cfg.get("kelly_max_fraction", 0.25))
        self._kelly_half_kelly: bool = bool(self._cfg.get("kelly_half_kelly", True))

        self._moe_low_weight_threshold: float = float(self._cfg.get("moe_low_weight_threshold", 0.05))
        self._moe_high_weight_threshold: float = float(self._cfg.get("moe_high_weight_threshold", 0.25))

        self._use_meta_agent: bool = bool(self._cfg.get("use_meta_agent", True))
        self._memory: Dict[str, Any] = {}
        self._db_path = db_path or str(self._cfg.get("db_path", _DB_PATH))
        self._lock = threading.Lock()
        self._cycle_counter = 0
        self._meta_agent: Optional[MetaAgent] = None
        self._init_db()
        self._load_memory()
        if self._use_meta_agent:
            self._meta_agent = MetaAgent(self)

        logger.info(
            "AutonomousBrain initialised (enabled=%s, min_conf=%.2f, pause_dd=%.1f%%, "
            "kelly_max=%.2f, meta_agent=%s, moe_thresholds=[%.2f, %.2f])",
            self._enabled, self._min_confidence, self._pause_drawdown_pct,
            self._kelly_max_fraction, self._use_meta_agent,
            self._moe_low_weight_threshold, self._moe_high_weight_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_meta_agent(self, meta_agent: Optional[MetaAgent]) -> None:
        """Inject a custom MetaAgent implementation."""
        self._meta_agent = meta_agent
        self._use_meta_agent = meta_agent is not None

    def decide(self, market_state: Dict[str, Any]) -> List[AutonomousAction]:
        if not self._enabled:
            return []

        self._cycle_counter += 1
        cycle_id = f"cycle-{self._cycle_counter}-{int(time.time())}"
        actions: List[AutonomousAction] = []
        conflicts: List[AgentConflict] = []
        by_agent: Dict[str, List[AutonomousAction]] = {}

        try:
            if self._use_meta_agent and self._meta_agent is not None:
                actions, conflicts, by_agent = self._meta_agent.decide(market_state)
            else:
                actions.extend(self._evaluate_strategies(market_state))
                actions.extend(self._evaluate_position_sizing(market_state))
                actions.extend(self._evaluate_venues(market_state))
                actions.extend(self._evaluate_models(market_state))
                actions.extend(self._evaluate_risk(market_state))
                actions.extend(self._evaluate_pause_resume(market_state))
        except Exception:
            logger.exception("AutonomousBrain.decide() error in cycle %s", cycle_id)

        actions = [a for a in actions if a.confidence >= self._min_confidence]
        actions.sort(key=lambda a: (a.priority, -a.confidence))
        actions = actions[: self._max_actions_per_cycle]

        self._persist_actions(actions, cycle_id, market_state, conflicts=conflicts, agent_actions=by_agent)
        self._update_memory(market_state, actions, conflicts)

        if actions:
            logger.info(
                "AutonomousBrain produced %d action(s) in %s: %s",
                len(actions), cycle_id,
                ", ".join(f"{a.action_type}({a.target})" for a in actions),
            )
        else:
            logger.debug("AutonomousBrain: no actions in %s", cycle_id)

        if conflicts:
            logger.info("AutonomousBrain resolved %d sub-agent conflict(s) in %s", len(conflicts), cycle_id)

        return actions

    def get_decision_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
                conn.close()
                return [dict(r) for r in rows]
            except Exception as exc:
                logger.warning("Failed to read decision history: %s", exc)
                return []

    def recall(self, key: str, default: Any = None) -> Any:
        return self._memory.get(key, default)

    def remember(self, key: str, value: Any) -> None:
        self._memory[key] = value
        self._save_memory_key(key, value)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def cycle_counter(self) -> int:
        return self._cycle_counter

    # ------------------------------------------------------------------
    # Flat evaluator modules (retained for compatibility)
    # ------------------------------------------------------------------

    def _evaluate_strategies(self, state: Dict[str, Any]) -> List[AutonomousAction]:
        actions: List[AutonomousAction] = []
        strategies = state.get("strategies") or {}
        regime = state.get("regime", "unknown")
        regime_confidence = float(state.get("regime_confidence", 1.0))
        moe_weights: Dict[str, float] = state.get("moe_weights") or {}

        for name, metrics in strategies.items():
            sharpe = float(metrics.get("sharpe", 0.0))
            decay_mult = float(metrics.get("decay_mult", 1.0))
            trades_14d = int(metrics.get("trades_14d", 0))
            regime_match = bool(metrics.get("regime_match", True))
            is_active = bool(metrics.get("is_active", True))
            n_strategies = max(len(strategies), 1)
            moe_w = float(moe_weights.get(name, 1.0 / n_strategies))

            if is_active and sharpe < self._strategy_disable_sharpe and decay_mult < 0.5:
                moe_boost = 0.15 if moe_w < self._moe_low_weight_threshold else 0.0
                actions.append(AutonomousAction(
                    action_type="deactivate_strategy",
                    target=name,
                    params={
                        "sharpe": sharpe,
                        "decay_mult": decay_mult,
                        "moe_weight": round(moe_w, 4),
                        "regime": regime,
                    },
                    reason=(
                        f"Strategy '{name}' Sharpe={sharpe:.2f} (below {self._strategy_disable_sharpe}), "
                        f"decay_mult={decay_mult:.2f}, MoE_weight={moe_w:.3f}. Deactivating to preserve capital."
                    ),
                    confidence=min(1.0, 0.6 + abs(sharpe) * 0.2 + moe_boost),
                    priority=2,
                ))
            elif is_active and sharpe > self._strategy_promote_sharpe and decay_mult > 0.7:
                moe_boost = 0.15 if moe_w > self._moe_high_weight_threshold else 0.0
                actions.append(AutonomousAction(
                    action_type="activate_strategy",
                    target=name,
                    params={
                        "sharpe": sharpe,
                        "increase_weight": True,
                        "moe_weight": round(moe_w, 4),
                        "regime": regime,
                        "regime_confidence": round(regime_confidence, 3),
                    },
                    reason=(
                        f"Strategy '{name}' Sharpe={sharpe:.2f} > {self._strategy_promote_sharpe}, "
                        f"decay_mult={decay_mult:.2f}, MoE_weight={moe_w:.3f} "
                        f"(regime='{regime}' conf={regime_confidence:.2f}). Increasing allocation."
                    ),
                    confidence=min(1.0, 0.5 + sharpe * 0.1 + moe_boost),
                    priority=3,
                ))

            if is_active and not regime_match and regime_confidence > 0.6:
                actions.append(AutonomousAction(
                    action_type="deactivate_strategy",
                    target=name,
                    params={
                        "regime": regime,
                        "regime_match": False,
                        "regime_confidence": round(regime_confidence, 3),
                        "moe_weight": round(moe_w, 4),
                    },
                    reason=(
                        f"Strategy '{name}' mismatched with regime '{regime}' (conf={regime_confidence:.2f}). "
                        f"MoE weight={moe_w:.3f}. Reducing exposure."
                    ),
                    confidence=min(1.0, 0.40 + regime_confidence * 0.25),
                    priority=4,
                ))

            if is_active and trades_14d == 0:
                actions.append(AutonomousAction(
                    action_type="deactivate_strategy",
                    target=name,
                    params={"trades_14d": 0, "moe_weight": round(moe_w, 4)},
                    reason=(
                        f"Strategy '{name}' generated 0 trades in 14 days (MoE weight={moe_w:.3f}). "
                        f"Flagging for review."
                    ),
                    confidence=0.35,
                    priority=5,
                ))
        return actions

    def _evaluate_position_sizing(self, state: Dict[str, Any]) -> List[AutonomousAction]:
        actions: List[AutonomousAction] = []
        drawdown_pct = float(state.get("drawdown_pct", 0.0))
        volatility = float(state.get("volatility", 0.0))
        regime_confidence = float(state.get("regime_confidence", 1.0))
        kelly_estimates: Dict[str, Any] = state.get("kelly_estimates") or {}
        strategies = state.get("strategies") or {}

        for name, metrics in strategies.items():
            if name in kelly_estimates:
                ke = kelly_estimates[name]
                kelly = float(ke.get("fraction", ke.get("kelly_fraction", 0.0)))
                ci_lo = float(ke.get("ci_low", kelly))
                ci_hi = float(ke.get("ci_high", kelly))
                ci_width = max(0.0, ci_hi - ci_lo)
                shrinkage = max(0.0, 1.0 - ci_width)
                kelly = ci_lo + (kelly - ci_lo) * shrinkage
                sizing_source = "KellySizer"
            else:
                win_rate = float(metrics.get("win_rate", 0.5))
                avg_win = float(metrics.get("avg_win_pct", 1.0))
                avg_loss = float(metrics.get("avg_loss_pct", 1.0))
                ci_lo, ci_hi = 0.0, 0.0
                b = avg_win / max(avg_loss, 0.01)
                kelly = (win_rate * b - (1 - win_rate)) / max(b, 0.01)
                sizing_source = "derived"

            if self._kelly_half_kelly:
                kelly *= 0.5
            kelly = max(0.0, min(kelly, self._kelly_max_fraction))
            regime_mult = 0.5 + 0.5 * regime_confidence
            kelly *= regime_mult

            dd_factor = max(0.3, 1.0 - (drawdown_pct / 100.0) ** 1.5 * 3.0)
            if volatility > 0.8:
                vol_factor = 0.6
            elif volatility > 0.5:
                vol_factor = 0.8
            elif volatility < 0.2:
                vol_factor = 1.2
            else:
                vol_factor = 1.0

            adjusted_size = kelly * dd_factor * vol_factor
            current_size = float(metrics.get("current_size_pct", kelly))
            if abs(adjusted_size - current_size) > 0.02:
                actions.append(AutonomousAction(
                    action_type="adjust_position_size",
                    target=name,
                    params={
                        "kelly": round(kelly, 4),
                        "kelly_source": sizing_source,
                        "ci_low": round(ci_lo, 4) if sizing_source == "KellySizer" else None,
                        "ci_high": round(ci_hi, 4) if sizing_source == "KellySizer" else None,
                        "regime_mult": round(regime_mult, 3),
                        "dd_factor": round(dd_factor, 3),
                        "vol_factor": round(vol_factor, 3),
                        "new_size_pct": round(adjusted_size, 4),
                    },
                    reason=(
                        f"Position size '{name}': Kelly={kelly:.3f} ({sizing_source}), regime_mult={regime_mult:.2f}, "
                        f"DD={dd_factor:.2f}, vol={vol_factor:.2f} -> new size {adjusted_size:.3f} (was {current_size:.3f})"
                    ),
                    confidence=0.6,
                    priority=3,
                ))
        return actions

    def _evaluate_venues(self, state: Dict[str, Any]) -> List[AutonomousAction]:
        actions: List[AutonomousAction] = []
        venues = state.get("venues") or {}
        if len(venues) < 2:
            return actions

        scored: List[Tuple[str, float, Dict[str, Any]]] = []
        for vname, vm in venues.items():
            fill_rate = float(vm.get("fill_rate", 0.95))
            slippage = float(vm.get("avg_slippage_bps", 5.0))
            fees = float(vm.get("fee_bps", 10.0))
            latency = float(vm.get("latency_ms", 100.0))
            score = fill_rate * 100 - slippage * 2 - fees - latency * 0.01
            scored.append((vname, score, vm))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_name, best_score, _ = scored[0]
        current_primary = state.get("current_venue", scored[-1][0])
        if best_name != current_primary and len(scored) >= 2:
            gap = best_score - scored[1][1]
            if gap > 5.0:
                actions.append(AutonomousAction(
                    action_type="switch_venue",
                    target=best_name,
                    params={"from_venue": current_primary, "score_gap": round(gap, 2)},
                    reason=f"Venue '{best_name}' scores {gap:.1f} pts above current primary '{current_primary}'.",
                    confidence=min(1.0, 0.5 + gap * 0.02),
                    priority=3,
                ))
        return actions

    def _evaluate_models(self, state: Dict[str, Any]) -> List[AutonomousAction]:
        actions: List[AutonomousAction] = []
        models = state.get("models") or {}
        now_ts = time.time()
        for mname, mm in models.items():
            last_train_ts = float(mm.get("last_train_ts", 0))
            accuracy = float(mm.get("accuracy", 0.5))
            peak_accuracy = float(mm.get("peak_accuracy", accuracy))
            drift_score = float(mm.get("drift_score", 0.0))
            samples_since_train = int(mm.get("samples_since_train", 0))
            age_days = (now_ts - last_train_ts) / 86400.0 if last_train_ts > 0 else 999

            reasons: List[str] = []
            urgency = 5
            if age_days > self._model_stale_days:
                reasons.append(f"stale ({age_days:.0f}d)")
                urgency = min(urgency, 3)
            if drift_score > 0.3:
                reasons.append(f"feature drift ({drift_score:.2f})")
                urgency = min(urgency, 2)
            if peak_accuracy > 0 and (peak_accuracy - accuracy) / max(peak_accuracy, 0.01) > 0.10:
                reasons.append(f"accuracy -{((peak_accuracy - accuracy) / peak_accuracy) * 100:.0f}% from peak")
                urgency = min(urgency, 3)
            if accuracy < 0.45 and samples_since_train > 100:
                reasons.append(f"acc={accuracy:.2f} over {samples_since_train} samples")
                urgency = min(urgency, 2)
            if samples_since_train > 1000:
                reasons.append(f"{samples_since_train} new samples")
                urgency = min(urgency, 4)
            if reasons:
                actions.append(AutonomousAction(
                    action_type="retrain_model",
                    target=mname,
                    params={
                        "age_days": round(age_days, 1),
                        "accuracy": round(accuracy, 3),
                        "drift_score": round(drift_score, 3),
                        "samples_since_train": samples_since_train,
                    },
                    reason=f"Model '{mname}' needs retraining: {'; '.join(reasons)}.",
                    confidence=min(1.0, 0.4 + len(reasons) * 0.15),
                    priority=urgency,
                ))
        return actions

    def _evaluate_risk(self, state: Dict[str, Any]) -> List[AutonomousAction]:
        actions: List[AutonomousAction] = []
        drawdown_pct = float(state.get("drawdown_pct", 0.0))
        volatility = float(state.get("volatility", 0.0))
        regime = state.get("regime", "unknown")
        regime_confidence = float(state.get("regime_confidence", 1.0))
        risk = state.get("risk") or {}
        loss_streak = int(risk.get("loss_streak", 0))
        win_streak = int(risk.get("win_streak", 0))

        mult = 1.0
        reasons: List[str] = []
        if drawdown_pct > 5:
            dd_mult = max(0.3, 1.0 - (drawdown_pct / 30.0) ** 1.5)
            mult *= dd_mult
            reasons.append(f"DD {drawdown_pct:.1f}% -> {dd_mult:.2f}x")
        if volatility > 0.8:
            mult *= 0.7
            reasons.append(f"high vol ({volatility:.2f}) -> 0.7x")
        elif volatility < 0.2:
            mult *= 1.15
            reasons.append(f"low vol ({volatility:.2f}) -> 1.15x")
        if loss_streak >= 5:
            streak_mult = max(0.4, 1.0 - loss_streak * 0.08)
            mult *= streak_mult
            reasons.append(f"loss streak {loss_streak} -> {streak_mult:.2f}x")
        elif win_streak >= 5:
            streak_mult = min(1.15, 1.0 + win_streak * 0.02)
            mult *= streak_mult
            reasons.append(f"win streak {win_streak} -> {streak_mult:.2f}x")
        if regime_confidence < 0.5:
            rc_mult = 0.8
            mult *= rc_mult
            reasons.append(f"low regime conf ({regime_confidence:.2f}) -> {rc_mult}x")
        mult = max(0.2, min(1.5, mult))
        if abs(mult - 1.0) > 0.05:
            actions.append(AutonomousAction(
                action_type="adjust_risk",
                target="global",
                params={
                    "position_multiplier": round(mult, 3),
                    "regime": regime,
                    "regime_confidence": round(regime_confidence, 3),
                },
                reason=f"Risk multiplier {mult:.2f}x: {'; '.join(reasons)}.",
                confidence=min(1.0, 0.5 + abs(1.0 - mult) * 0.5),
                priority=2,
            ))
        return actions

    def _evaluate_pause_resume(self, state: Dict[str, Any]) -> List[AutonomousAction]:
        actions: List[AutonomousAction] = []
        drawdown_pct = float(state.get("drawdown_pct", 0.0))
        is_paused = bool(state.get("is_paused", False))
        upcoming = state.get("upcoming_events") or []
        volatility = float(state.get("volatility", 0.0))

        if not is_paused and drawdown_pct >= self._pause_drawdown_pct:
            actions.append(AutonomousAction(
                action_type="pause_trading",
                target="system",
                params={"drawdown_pct": drawdown_pct, "threshold": self._pause_drawdown_pct},
                reason=f"Drawdown {drawdown_pct:.1f}% >= pause threshold {self._pause_drawdown_pct:.1f}%.",
                confidence=0.95,
                priority=1,
            ))
        if not is_paused:
            for evt in upcoming:
                hours_until = float(evt.get("hours_until", 999))
                event_name = str(evt.get("name", "macro event"))
                if hours_until <= 2.0:
                    actions.append(AutonomousAction(
                        action_type="pause_trading",
                        target="system",
                        params={"event": event_name, "hours_until": hours_until},
                        reason=f"Event '{event_name}' in {hours_until:.1f}h.",
                        confidence=0.7,
                        priority=2,
                    ))
                    break
        if not is_paused and volatility > 1.5:
            actions.append(AutonomousAction(
                action_type="pause_trading",
                target="system",
                params={"volatility": volatility},
                reason=f"Vol spike ({volatility:.2f}) — potential black swan.",
                confidence=0.8,
                priority=1,
            ))
        if is_paused:
            recovered = drawdown_pct < (self._pause_drawdown_pct - self._resume_recovery_pct)
            vol_ok = volatility < 1.0
            no_imminent = all(float(e.get("hours_until", 999)) > 4.0 for e in upcoming)
            if recovered and vol_ok and no_imminent:
                actions.append(AutonomousAction(
                    action_type="resume_trading",
                    target="system",
                    params={"drawdown_pct": drawdown_pct, "volatility": volatility},
                    reason=f"Conditions normalised: DD={drawdown_pct:.1f}%, vol={volatility:.2f}.",
                    confidence=0.75,
                    priority=2,
                ))
        return actions

    # ------------------------------------------------------------------
    # Cross-session memory
    # ------------------------------------------------------------------

    def _load_memory(self) -> None:
        try:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute("SELECT key, value FROM cross_session_memory").fetchall()
            conn.close()
            for k, v in rows:
                try:
                    self._memory[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    self._memory[k] = v
            logger.debug("Loaded %d cross-session memory entries", len(self._memory))
        except Exception as exc:
            logger.debug("Could not load cross-session memory: %s", exc)

    def _save_memory_key(self, key: str, value: Any) -> None:
        with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                conn.execute(
                    "INSERT OR REPLACE INTO cross_session_memory (key, value, ts) VALUES (?, ?, ?)",
                    (key, json.dumps(value, default=str), datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
                conn.close()
            except Exception as exc:
                logger.warning("Failed to persist memory key '%s': %s", key, exc)

    def _update_memory(self, state: Dict[str, Any], actions: List[AutonomousAction], conflicts: Optional[List[AgentConflict]] = None) -> None:
        self.remember("last_cycle_ts", time.time())
        self.remember("last_regime", state.get("regime", "unknown"))
        self.remember("last_regime_confidence", state.get("regime_confidence", 1.0))
        self.remember("last_drawdown_pct", state.get("drawdown_pct", 0.0))
        self.remember("total_decisions", self.recall("total_decisions", 0) + len(actions))
        freq = self.recall("action_type_freq", {})
        if not isinstance(freq, dict):
            freq = {}
        for a in actions:
            freq[a.action_type] = freq.get(a.action_type, 0) + 1
        self.remember("action_type_freq", freq)
        if conflicts is not None:
            self.remember("last_conflicts", [c.to_dict() for c in conflicts])
            self.remember("total_conflicts", self.recall("total_conflicts", 0) + len(conflicts))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_actions(
        self,
        actions: List[AutonomousAction],
        cycle_id: str,
        market_state: Dict[str, Any],
        conflicts: Optional[List[AgentConflict]] = None,
        agent_actions: Optional[Dict[str, List[AutonomousAction]]] = None,
    ) -> None:
        if not actions:
            return
        snapshot = json.dumps(
            {
                "core": {
                    k: v for k, v in market_state.items()
                    if k in ("regime", "regime_confidence", "drawdown_pct", "volatility", "mm_spread")
                },
                "conflicts": [c.to_dict() for c in (conflicts or [])],
                "agent_actions": {
                    name: [a.to_dict() for a in acts]
                    for name, acts in (agent_actions or {}).items()
                },
            },
            default=str,
        )
        with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                conn.execute("BEGIN IMMEDIATE")
                for a in actions:
                    conn.execute(
                        "INSERT INTO decisions (ts, action_type, target, params, reason, confidence, priority, cycle_id, market_snapshot) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            a.timestamp, a.action_type, a.target,
                            json.dumps(a.params, default=str), a.reason,
                            a.confidence, a.priority, cycle_id, snapshot,
                        ),
                    )
                conn.commit()
                conn.close()
            except Exception as exc:
                logger.warning("Failed to persist %d actions: %s", len(actions), exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_section(config: Any) -> Dict[str, Any]:
        if config is None:
            return {}
        if isinstance(config, dict):
            return config.get("autonomous_brain", {}) or {}
        return getattr(config, "autonomous_brain", None) or {}


__all__ = [
    "AutonomousAction",
    "AgentConflict",
    "DecisionSubAgent",
    "StrategySubAgent",
    "ModelSubAgent",
    "RiskSubAgent",
    "ExecutionSubAgent",
    "MetaAgent",
    "AutonomousBrain",
]
