from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from argus_live.evidence.edge_attribution import EdgeAttribution, attribute_trade
from argus_live.optimization.learning_loop import LearningUpdate, compute_learning_update
from argus_live.promotion.strategy_lifecycle import (
    LifecycleState,
    StrategyLifecycleDecision,
    evaluate_lifecycle,
)
from argus_live.state.learning_state import LearningState, LearningStateStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveLearningResult:
    """Immutable result of a single trade learning pass."""

    edge: EdgeAttribution
    lifecycle: StrategyLifecycleDecision
    learning_update: LearningUpdate


class LiveLearningIntegrator:
    """Orchestrates per-trade learning: attribution, lifecycle eval, and update suggestion."""

    def __init__(
        self,
        *,
        state_store: LearningStateStore | None = None,
        recommendation_log_path: str | Path | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        self._state_store = state_store or LearningStateStore()
        self._state: LearningState = self._state_store.load()
        _lp = recommendation_log_path or log_path
        self._log_path = Path(_lp) if _lp else None

    @property
    def state(self) -> LearningState:
        return self._state

    def process_trade_outcome(
        self,
        strategy_id: str,
        symbol: str,
        venue: str,
        expected_edge_bps: float,
        realized_pnl_bps: float,
        slippage_bps: float,
        fee_bps: float,
        drawdown_pct: float = 0.0,
        stability_score: float = 0.0,
    ) -> LiveLearningResult:
        """Full learning pass for one trade outcome.

        1. Attribute trade edge
        2. Update running-average strategy edge
        3. Update running-average venue slippage
        4. Evaluate lifecycle
        5. Compute learning update suggestions
        6. Persist state and append JSONL log
        """
        # 1. Edge attribution
        edge = attribute_trade(
            strategy_id=strategy_id,
            symbol=symbol,
            expected_edge_bps=expected_edge_bps,
            realized_pnl_bps=realized_pnl_bps,
            slippage_bps=slippage_bps,
            fee_bps=fee_bps,
        )

        # 2. Running average for strategy edge
        prev_edge = self._state.strategy_edge_bps.get(strategy_id, 0.0)
        prev_count = self._state.strategy_trade_count.get(strategy_id, 0)
        new_count = prev_count + 1
        new_edge = prev_edge + (edge.net_edge_bps - prev_edge) / new_count
        self._state.strategy_edge_bps[strategy_id] = new_edge
        self._state.strategy_trade_count[strategy_id] = new_count

        # 3. Running average for venue slippage
        prev_slip = self._state.venue_slippage_bps.get(venue, 0.0)
        prev_vcount = self._state.venue_trade_count.get(venue, 0)
        new_vcount = prev_vcount + 1
        new_slip = prev_slip + (slippage_bps - prev_slip) / new_vcount
        self._state.venue_slippage_bps[venue] = new_slip
        self._state.venue_trade_count[venue] = new_vcount

        # 4. Lifecycle evaluation
        current_lc = LifecycleState(
            self._state.lifecycle_state.get(strategy_id, LifecycleState.SHADOW.value)
        )
        lifecycle = evaluate_lifecycle(
            strategy_id=strategy_id,
            net_edge_bps=new_edge,
            drawdown_pct=drawdown_pct,
            stability_score=stability_score,
            current_state=current_lc,
        )
        self._state.lifecycle_state[strategy_id] = lifecycle.new_state.value

        # 5. Learning update
        learning_update = compute_learning_update(
            strategy_edges=self._state.strategy_edge_bps,
            venue_slippage=self._state.venue_slippage_bps,
        )

        # 6. Persist
        self._state_store.save(self._state)
        self._append_log(edge, lifecycle, learning_update)

        result = LiveLearningResult(
            edge=edge,
            lifecycle=lifecycle,
            learning_update=learning_update,
        )
        logger.info(
            "Learning pass: strategy=%s net_edge=%.1fbps lifecycle=%s->%s",
            strategy_id,
            edge.net_edge_bps,
            lifecycle.current_state.value,
            lifecycle.new_state.value,
        )
        return result

    # ------------------------------------------------------------------
    def _append_log(
        self,
        edge: EdgeAttribution,
        lifecycle: StrategyLifecycleDecision,
        update: LearningUpdate,
    ) -> None:
        if self._log_path is None:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "edge": {
                "strategy_id": edge.strategy_id,
                "symbol": edge.symbol,
                "net_edge_bps": edge.net_edge_bps,
            },
            "lifecycle": {
                "strategy_id": lifecycle.strategy_id,
                "current": lifecycle.current_state.value,
                "new": lifecycle.new_state.value,
                "reason": lifecycle.reason,
            },
            "update": {
                "weights": update.strategy_weight_suggestions,
                "penalties": update.venue_penalty_suggestions,
                "reason": update.reason,
            },
        }
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception:
            logger.exception("Failed to append learning log to %s", self._log_path)
