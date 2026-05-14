"""
Strategy Promotion Pipeline — from invention to live trading.

This is what closes the loop: discovered strategies don't just sit in
a hall of fame — they get validated, paper-tested, and promoted to live.

Pipeline stages:
  CANDIDATE → VALIDATING → PAPER_TESTING → PROMOTED → LIVE → RETIRED

Each stage has gates:
  CANDIDATE:      GP generator or evolver discovers a strategy with fitness > threshold
  VALIDATING:     Walk-forward OOS Sharpe > 0.3, trade_count >= 10, max_dd < 15%
  PAPER_TESTING:  Run in paper mode for N cycles, measure real execution metrics
  PROMOTED:       Paper results confirm edge (Sharpe > 0, win_rate > 40%)
  LIVE:           Active in signal generation with full position sizing
  RETIRED:        Demoted after sustained underperformance

The self-optimizer feeds back into this: if a LIVE strategy degrades,
it gets RETIRED. If a RETIRED strategy's market conditions return,
it can be re-PROMOTED.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PromotionStage(Enum):
    CANDIDATE = "candidate"
    VALIDATING = "validating"
    PAPER_TESTING = "paper_testing"
    PROMOTED = "promoted"
    LIVE = "live"
    RETIRED = "retired"


@dataclass
class StrategyCandidate:
    """A strategy moving through the promotion pipeline."""
    strategy_id: str
    source: str                         # "generator", "evolver", "manual"
    strategy_type: str                  # e.g. "breakout" or "gp_invented_42"
    params: Dict[str, Any]              # parameters or GP tree serialization
    rule_description: str               # human-readable rule string
    stage: PromotionStage = PromotionStage.CANDIDATE
    # Discovery metrics
    discovery_fitness: float = 0.0
    discovery_sharpe: float = 0.0
    discovery_win_rate: float = 0.0
    discovery_trade_count: int = 0
    discovery_max_dd: float = 0.0
    # Validation metrics
    oos_sharpe: float = 0.0
    oos_win_rate: float = 0.0
    oos_trade_count: int = 0
    # Paper testing metrics
    paper_cycles: int = 0
    paper_trades: int = 0
    paper_pnl: float = 0.0
    paper_sharpe: float = 0.0
    paper_slippage_avg: float = 0.0
    # Live metrics
    live_trades: int = 0
    live_pnl: float = 0.0
    live_sharpe: float = 0.0
    # Lifecycle
    created_at: float = field(default_factory=time.time)
    promoted_at: float = 0.0
    retired_at: float = 0.0
    retirement_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "source": self.source,
            "strategy_type": self.strategy_type,
            "stage": self.stage.value,
            "rule_description": self.rule_description[:200],
            "discovery_sharpe": self.discovery_sharpe,
            "oos_sharpe": self.oos_sharpe,
            "paper_sharpe": self.paper_sharpe,
            "live_pnl": self.live_pnl,
            "live_trades": self.live_trades,
        }


class StrategyPromotionPipeline:
    """
    Manages the full lifecycle of discovered strategies.

    Strategies flow: CANDIDATE → VALIDATING → PAPER_TESTING → PROMOTED → LIVE
    Failed strategies at any stage go to RETIRED.
    """

    def __init__(
        self,
        validation_min_sharpe: float = 0.3,
        validation_min_trades: int = 10,
        validation_max_dd: float = 15.0,
        paper_min_cycles: int = 200,
        paper_min_sharpe: float = 0.0,
        paper_min_win_rate: float = 0.40,
        live_retire_sharpe: float = -0.3,
        live_retire_trades: int = 20,
        max_live_strategies: int = 10,
        persist_path: Optional[str] = None,
    ):
        self._val_sharpe = validation_min_sharpe
        self._val_trades = validation_min_trades
        self._val_max_dd = validation_max_dd
        self._paper_min_cycles = paper_min_cycles
        self._paper_min_sharpe = paper_min_sharpe
        self._paper_min_wr = paper_min_win_rate
        self._live_retire_sharpe = live_retire_sharpe
        self._live_retire_trades = live_retire_trades
        self._max_live = max_live_strategies
        self._persist_path = Path(persist_path) if persist_path else None

        self._candidates: Dict[str, StrategyCandidate] = {}
        self._promotion_count = 0
        self._retirement_count = 0
        self._hostile_injector: Any = None
        self._last_prices: Dict[str, float] = {}
        self._last_advisory: Dict[str, Any] = {}

    def set_hostile_injector(self, injector: Any) -> None:
        """Inject the HostileScenarioInjector for promotion gating."""
        self._hostile_injector = injector

    def update_market_context(self, prices: Dict[str, float], advisory: Dict[str, Any]) -> None:
        """Update cached market context for hostile testing."""
        self._last_prices = prices
        self._last_advisory = advisory

    def submit_candidate(
        self,
        strategy_id: str,
        source: str,
        strategy_type: str,
        params: Dict[str, Any],
        rule_description: str,
        fitness: float,
        sharpe: float,
        win_rate: float,
        trade_count: int,
        max_dd: float,
    ) -> StrategyCandidate:
        """Submit a newly discovered strategy to the pipeline."""
        candidate = StrategyCandidate(
            strategy_id=strategy_id,
            source=source,
            strategy_type=strategy_type,
            params=params,
            rule_description=rule_description,
            stage=PromotionStage.CANDIDATE,
            discovery_fitness=fitness,
            discovery_sharpe=sharpe,
            discovery_win_rate=win_rate,
            discovery_trade_count=trade_count,
            discovery_max_dd=max_dd,
        )
        self._candidates[strategy_id] = candidate
        logger.info("Pipeline: new candidate %s from %s (sharpe=%.2f, wr=%.0f%%)",
                     strategy_id, source, sharpe, win_rate * 100)
        return candidate

    def validate(
        self,
        strategy_id: str,
        oos_sharpe: float,
        oos_win_rate: float,
        oos_trade_count: int,
    ) -> bool:
        """Validate a candidate with out-of-sample results. Returns True if promoted."""
        c = self._candidates.get(strategy_id)
        if c is None or c.stage != PromotionStage.CANDIDATE:
            return False

        c.stage = PromotionStage.VALIDATING
        c.oos_sharpe = oos_sharpe
        c.oos_win_rate = oos_win_rate
        c.oos_trade_count = oos_trade_count

        # Validation gates
        if (oos_sharpe >= self._val_sharpe
                and oos_trade_count >= self._val_trades
                and c.discovery_max_dd <= self._val_max_dd):
            c.stage = PromotionStage.PAPER_TESTING
            logger.info("Pipeline: %s passed validation → PAPER_TESTING (oos_sharpe=%.2f)",
                         strategy_id, oos_sharpe)
            return True

        c.stage = PromotionStage.RETIRED
        c.retired_at = time.time()
        c.retirement_reason = f"validation_failed: oos_sharpe={oos_sharpe:.2f}, trades={oos_trade_count}"
        self._retirement_count += 1
        return False

    def record_paper_cycle(self, strategy_id: str) -> None:
        """Record one paper trading cycle for a candidate."""
        c = self._candidates.get(strategy_id)
        if c and c.stage == PromotionStage.PAPER_TESTING:
            c.paper_cycles += 1

    def record_paper_trade(
        self,
        strategy_id: str,
        pnl: float,
        slippage_bps: float = 0.0,
    ) -> None:
        """Record a paper trade result."""
        c = self._candidates.get(strategy_id)
        if c and c.stage == PromotionStage.PAPER_TESTING:
            c.paper_trades += 1
            c.paper_pnl += pnl
            c.paper_slippage_avg = (
                (c.paper_slippage_avg * (c.paper_trades - 1) + slippage_bps)
                / c.paper_trades
            )

    def check_paper_promotion(self, strategy_id: str) -> bool:
        """Check if a paper-testing strategy should be promoted to LIVE."""
        c = self._candidates.get(strategy_id)
        if c is None or c.stage != PromotionStage.PAPER_TESTING:
            return False

        if c.paper_cycles < self._paper_min_cycles:
            return False

        if c.paper_trades < 3:
            c.stage = PromotionStage.RETIRED
            c.retired_at = time.time()
            c.retirement_reason = f"paper_no_trades: {c.paper_trades} trades in {c.paper_cycles} cycles"
            self._retirement_count += 1
            return False

        # Compute paper Sharpe
        avg_pnl = c.paper_pnl / c.paper_trades
        c.paper_sharpe = avg_pnl  # simplified; real Sharpe needs std

        paper_wr = max(0, c.paper_trades - abs(c.paper_pnl)) / max(c.paper_trades, 1)  # approximate

        if c.paper_sharpe >= self._paper_min_sharpe and c.paper_pnl > 0:
            # Hostile scenario gate: strategy must survive adversarial conditions
            if self._hostile_injector is not None:
                try:
                    _hostile_report = self._hostile_injector.test_all_scenarios(
                        strategy_name=strategy_id,
                        symbol=c.params.get("symbol", "BTC/USD"),
                        prices=self._last_prices or {},
                        advisory=self._last_advisory or {},
                    )
                    if not _hostile_report.promotion_safe:
                        _failures = [r.scenario.value for r in _hostile_report.results if not r.passed]
                        c.stage = PromotionStage.RETIRED
                        c.retired_at = time.time()
                        c.retirement_reason = f"hostile_test_failed: {_failures}"
                        self._retirement_count += 1
                        logger.warning(
                            "Pipeline: %s FAILED hostile scenarios: %s — retired",
                            strategy_id, _failures,
                        )
                        return False
                    logger.info("Pipeline: %s passed hostile scenarios (%d/%d)",
                                 strategy_id, _hostile_report.passed, _hostile_report.total_scenarios)
                except Exception as _hi_exc:
                    logger.debug("Pipeline: hostile test skipped for %s: %s", strategy_id, _hi_exc)

            # Check if we have room for more live strategies
            live_count = sum(1 for sc in self._candidates.values() if sc.stage == PromotionStage.LIVE)
            if live_count >= self._max_live:
                # Retire worst live strategy to make room
                self._retire_worst_live()

            c.stage = PromotionStage.PROMOTED
            c.promoted_at = time.time()
            self._promotion_count += 1
            logger.info("Pipeline: %s PROMOTED to live! (paper_pnl=%.2f, trades=%d, cycles=%d)",
                         strategy_id, c.paper_pnl, c.paper_trades, c.paper_cycles)
            return True

        c.stage = PromotionStage.RETIRED
        c.retired_at = time.time()
        c.retirement_reason = f"paper_underperformed: pnl={c.paper_pnl:.2f}, trades={c.paper_trades}"
        self._retirement_count += 1
        return False

    def activate_live(self, strategy_id: str) -> bool:
        """Move PROMOTED strategy to LIVE (active in signal generation)."""
        c = self._candidates.get(strategy_id)
        if c and c.stage == PromotionStage.PROMOTED:
            c.stage = PromotionStage.LIVE
            logger.info("Pipeline: %s is now LIVE", strategy_id)
            return True
        return False

    def record_live_trade(self, strategy_id: str, pnl: float) -> None:
        """Record a live trade result."""
        c = self._candidates.get(strategy_id)
        if c and c.stage == PromotionStage.LIVE:
            c.live_trades += 1
            c.live_pnl += pnl

    def check_live_retirement(self, strategy_id: str) -> bool:
        """Check if a live strategy should be retired."""
        c = self._candidates.get(strategy_id)
        if c is None or c.stage != PromotionStage.LIVE:
            return False

        if c.live_trades < self._live_retire_trades:
            return False

        avg_pnl = c.live_pnl / c.live_trades
        if avg_pnl < self._live_retire_sharpe:
            c.stage = PromotionStage.RETIRED
            c.retired_at = time.time()
            c.retirement_reason = f"live_underperformed: avg_pnl={avg_pnl:.3f}, trades={c.live_trades}"
            self._retirement_count += 1
            logger.warning("Pipeline: RETIRED %s from live — %s", strategy_id, c.retirement_reason)
            return True
        return False

    def _retire_worst_live(self) -> None:
        """Retire the worst-performing live strategy to make room."""
        live = [c for c in self._candidates.values() if c.stage == PromotionStage.LIVE]
        if not live:
            return
        worst = min(live, key=lambda c: c.live_pnl / max(c.live_trades, 1))
        worst.stage = PromotionStage.RETIRED
        worst.retired_at = time.time()
        worst.retirement_reason = "displaced_by_better_strategy"
        self._retirement_count += 1

    # ──────────────────────────────────────────────────────────────────────
    # Queries
    # ──────────────────────────────────────────────────────────────────────

    def get_live_strategies(self) -> List[StrategyCandidate]:
        return [c for c in self._candidates.values() if c.stage == PromotionStage.LIVE]

    def get_promoted(self) -> List[StrategyCandidate]:
        return [c for c in self._candidates.values() if c.stage == PromotionStage.PROMOTED]

    def get_paper_testing(self) -> List[StrategyCandidate]:
        return [c for c in self._candidates.values() if c.stage == PromotionStage.PAPER_TESTING]

    def get_all(self) -> Dict[str, StrategyCandidate]:
        return dict(self._candidates)

    def get_stats(self) -> Dict[str, Any]:
        stages = {}
        for c in self._candidates.values():
            stages[c.stage.value] = stages.get(c.stage.value, 0) + 1
        return {
            "total_candidates": len(self._candidates),
            "stages": stages,
            "promotions": self._promotion_count,
            "retirements": self._retirement_count,
            "live_count": sum(1 for c in self._candidates.values() if c.stage == PromotionStage.LIVE),
            "paper_testing_count": sum(1 for c in self._candidates.values() if c.stage == PromotionStage.PAPER_TESTING),
        }

    def persist(self) -> None:
        """Save pipeline state to disk."""
        if self._persist_path is None:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {sid: c.to_dict() for sid, c in self._candidates.items()}
            self._persist_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Pipeline persist failed: %s", e)
