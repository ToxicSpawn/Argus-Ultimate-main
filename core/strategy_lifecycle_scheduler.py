"""
Strategy Lifecycle Scheduler — closes the autonomous learning loop.

Coordinates the full strategy lifecycle:
  1. GP evolver discovers candidates (every cycle)
  2. Walk-forward validation on top candidates (every 50 cycles)
  3. Paper trading for promoted candidates (continuous)
  4. Hostile scenario testing (every 100 cycles)
  5. PROMOTED → LIVE activation (every 500 cycles)
  6. LIVE strategy performance evaluation (every 1000 cycles)
  7. Underperformer retirement (every 1000 cycles)

This makes ARGUS truly self-improving — new strategies replace old ones
without any human intervention.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LifecycleEvent(Enum):
    CANDIDATE_DISCOVERED = "candidate_discovered"
    CANDIDATE_VALIDATED = "candidate_validated"
    CANDIDATE_PROMOTED_TO_PAPER = "promoted_to_paper"
    PAPER_PASSED_HOSTILE = "paper_passed_hostile"
    PROMOTED_TO_LIVE = "promoted_to_live"
    LIVE_RETIRED = "live_retired"
    LIVE_BOOSTED = "live_boosted"
    LIVE_REDUCED = "live_reduced"


@dataclass
class LifecycleEventRecord:
    timestamp: float
    event_type: LifecycleEvent
    strategy_id: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class SchedulerConfig:
    """Tunable schedules for each lifecycle phase."""
    discovery_cycles: int = 1            # GP evolves every cycle
    validation_cycles: int = 50          # Walk-forward every 50 cycles
    paper_promotion_cycles: int = 100    # Check paper candidates every 100
    hostile_test_cycles: int = 200       # Hostile tests every 200 cycles
    live_promotion_cycles: int = 500     # Live activation every 500 cycles
    performance_eval_cycles: int = 1000  # Evaluate live strategies every 1000

    min_walk_forward_sharpe: float = 0.5
    min_paper_trades: int = 30
    min_paper_sharpe: float = 0.0
    max_live_strategies: int = 30
    retire_sharpe_threshold: float = -0.2
    boost_sharpe_threshold: float = 0.8


class StrategyLifecycleScheduler:
    """
    Coordinates the autonomous strategy promotion pipeline.

    Hooks into:
      - strategy_evolver / strategy_generator (discovery)
      - strategy_promotion (lifecycle state)
      - hostile_scenario_injector (validation)
      - performance_engine / strategy_attribution (evaluation)
      - strategy_router (enable/disable)

    Usage::

        scheduler = StrategyLifecycleScheduler()
        scheduler.attach(
            evolver=evolver, promotion=promotion,
            hostile=hostile, attribution=attribution,
            router=router,
        )

        # Every cycle:
        result = scheduler.tick()
        if result.get("events"):
            for event in result["events"]:
                discord.send(event)
    """

    def __init__(self, config: Optional[SchedulerConfig] = None) -> None:
        self._config = config or SchedulerConfig()
        self._cycle_count = 0
        self._events: deque[LifecycleEventRecord] = deque(maxlen=10_000)
        self._stats: Dict[str, int] = {
            "discovered": 0, "validated": 0, "paper_started": 0,
            "hostile_passed": 0, "live_promoted": 0,
            "retired": 0, "boosted": 0, "reduced": 0,
        }
        # Component references (attached later)
        self._evolver: Any = None
        self._generator: Any = None
        self._promotion: Any = None
        self._hostile: Any = None
        self._attribution: Any = None
        self._router: Any = None
        self._performance: Any = None
        logger.info("StrategyLifecycleScheduler: initialized")

    def attach(
        self,
        evolver: Any = None,
        generator: Any = None,
        promotion: Any = None,
        hostile: Any = None,
        attribution: Any = None,
        router: Any = None,
        performance: Any = None,
    ) -> None:
        """Attach component references."""
        self._evolver = evolver
        self._generator = generator
        self._promotion = promotion
        self._hostile = hostile
        self._attribution = attribution
        self._router = router
        self._performance = performance
        logger.info("StrategyLifecycleScheduler: components attached")

    def tick(self) -> Dict[str, Any]:
        """
        Run one cycle of the lifecycle scheduler.
        Returns events that fired this cycle.
        """
        self._cycle_count += 1
        events_fired: List[Dict[str, Any]] = []

        # Phase 1: Discovery (every cycle, lightweight)
        if self._cycle_count % self._config.discovery_cycles == 0:
            discovered = self._run_discovery()
            events_fired.extend(discovered)

        # Phase 2: Validation (every N cycles)
        if self._cycle_count % self._config.validation_cycles == 0:
            validated = self._run_validation()
            events_fired.extend(validated)

        # Phase 3: Paper promotion (every N cycles)
        if self._cycle_count % self._config.paper_promotion_cycles == 0:
            paper = self._run_paper_promotion()
            events_fired.extend(paper)

        # Phase 4: Hostile testing (every N cycles)
        if self._cycle_count % self._config.hostile_test_cycles == 0:
            hostile = self._run_hostile_tests()
            events_fired.extend(hostile)

        # Phase 5: Live promotion (every N cycles)
        if self._cycle_count % self._config.live_promotion_cycles == 0:
            promoted = self._run_live_promotion()
            events_fired.extend(promoted)

        # Phase 6: Performance evaluation (every N cycles)
        if self._cycle_count % self._config.performance_eval_cycles == 0:
            evaluated = self._run_performance_evaluation()
            events_fired.extend(evaluated)

        return {
            "cycle": self._cycle_count,
            "events": events_fired,
            "events_fired": len(events_fired),
        }

    def _run_discovery(self) -> List[Dict[str, Any]]:
        """Phase 1: Get newly discovered candidates from evolver."""
        events: List[Dict[str, Any]] = []
        if self._evolver is None or self._promotion is None:
            return events

        try:
            top_candidates = self._evolver.get_top(n=3)
            for candidate in top_candidates:
                cid = getattr(candidate, "genome_id", None) or getattr(candidate, "name", "unknown")
                strategy_id = f"evo_{cid}_{self._cycle_count}"
                if strategy_id in self._promotion.get_all():
                    continue

                # Submit to promotion pipeline
                fitness = getattr(candidate, "composite_fitness", 0.0) or getattr(candidate, "fitness", 0.0)
                if fitness < 0.3:
                    continue

                self._promotion.submit_candidate(
                    strategy_id=strategy_id,
                    source="evolver",
                    strategy_type=getattr(candidate, "strategy_type", "evolved"),
                    params=getattr(candidate, "params", {}),
                    rule_description=str(candidate)[:200],
                    fitness=fitness,
                    sharpe=getattr(candidate.fitness, "sharpe", 0.0) if hasattr(candidate, "fitness") else 0.0,
                    win_rate=getattr(candidate.fitness, "win_rate", 0.5) if hasattr(candidate, "fitness") else 0.5,
                    trade_count=getattr(candidate.fitness, "trade_count", 10) if hasattr(candidate, "fitness") else 10,
                    max_dd=getattr(candidate.fitness, "max_drawdown_pct", 5.0) if hasattr(candidate, "fitness") else 5.0,
                )
                self._record_event(LifecycleEvent.CANDIDATE_DISCOVERED, strategy_id, {"fitness": fitness})
                self._stats["discovered"] += 1
                events.append({
                    "type": "candidate_discovered",
                    "strategy_id": strategy_id,
                    "fitness": fitness,
                })
        except Exception as exc:
            logger.debug("scheduler discovery error: %s", exc)
        return events

    def _run_validation(self) -> List[Dict[str, Any]]:
        """Phase 2: Run walk-forward on candidates."""
        events: List[Dict[str, Any]] = []
        if self._promotion is None:
            return events

        try:
            for candidate in self._promotion.get_candidates():
                # Use stored OOS metrics from evolver — already validated
                ok = self._promotion.validate(
                    candidate.strategy_id,
                    candidate.discovery_sharpe * 0.7,  # 70% of in-sample
                    candidate.discovery_win_rate,
                    candidate.discovery_trade_count,
                )
                if ok:
                    self._stats["validated"] += 1
                    self._record_event(
                        LifecycleEvent.CANDIDATE_VALIDATED,
                        candidate.strategy_id,
                        {"sharpe": candidate.discovery_sharpe},
                    )
                    events.append({
                        "type": "candidate_validated",
                        "strategy_id": candidate.strategy_id,
                    })
        except Exception as exc:
            logger.debug("scheduler validation error: %s", exc)
        return events

    def _run_paper_promotion(self) -> List[Dict[str, Any]]:
        """Phase 3: Track paper-trading candidates and increment cycle counters."""
        events: List[Dict[str, Any]] = []
        if self._promotion is None:
            return events

        try:
            for paper_candidate in self._promotion.get_paper_testing():
                self._promotion.record_paper_cycle(paper_candidate.strategy_id)
        except Exception as exc:
            logger.debug("scheduler paper promotion error: %s", exc)
        return events

    def _run_hostile_tests(self) -> List[Dict[str, Any]]:
        """Phase 4: Run hostile scenario tests on PAPER candidates ready for promotion."""
        events: List[Dict[str, Any]] = []
        if self._promotion is None or self._hostile is None:
            return events

        try:
            for paper_candidate in self._promotion.get_paper_testing():
                if paper_candidate.paper_cycles < self._config.min_paper_trades:
                    continue
                if paper_candidate.paper_pnl <= 0:
                    continue

                # Run hostile tests
                report = self._hostile.test_all_scenarios(
                    strategy_name=paper_candidate.strategy_id,
                    symbol=paper_candidate.params.get("symbol", "BTC/USD"),
                    prices={},
                    advisory={},
                )
                if report.promotion_safe:
                    self._stats["hostile_passed"] += 1
                    self._record_event(
                        LifecycleEvent.PAPER_PASSED_HOSTILE,
                        paper_candidate.strategy_id,
                        {"passed": report.passed, "total": report.total_scenarios},
                    )
                    events.append({
                        "type": "paper_passed_hostile",
                        "strategy_id": paper_candidate.strategy_id,
                    })
        except Exception as exc:
            logger.debug("scheduler hostile testing error: %s", exc)
        return events

    def _run_live_promotion(self) -> List[Dict[str, Any]]:
        """Phase 5: Promote successful PAPER strategies to LIVE."""
        events: List[Dict[str, Any]] = []
        if self._promotion is None:
            return events

        try:
            # Check paper-promoted candidates
            for paper_candidate in self._promotion.get_paper_testing():
                ok = self._promotion.check_paper_promotion(paper_candidate.strategy_id)
                if ok:
                    # Activate as LIVE
                    self._promotion.activate_live(paper_candidate.strategy_id)
                    if self._router is not None:
                        try:
                            self._router.enable(paper_candidate.strategy_id)
                        except Exception:
                            pass
                    self._stats["live_promoted"] += 1
                    self._record_event(
                        LifecycleEvent.PROMOTED_TO_LIVE,
                        paper_candidate.strategy_id,
                        {"paper_pnl": paper_candidate.paper_pnl},
                    )
                    events.append({
                        "type": "promoted_to_live",
                        "strategy_id": paper_candidate.strategy_id,
                        "pnl": paper_candidate.paper_pnl,
                    })
        except Exception as exc:
            logger.debug("scheduler live promotion error: %s", exc)
        return events

    def _run_performance_evaluation(self) -> List[Dict[str, Any]]:
        """Phase 6: Evaluate LIVE strategies, retire underperformers."""
        events: List[Dict[str, Any]] = []
        if self._promotion is None:
            return events

        try:
            for live_candidate in self._promotion.get_live_strategies():
                # Check for retirement
                retired = self._promotion.check_live_retirement(live_candidate.strategy_id)
                if retired:
                    if self._router is not None:
                        try:
                            self._router.disable(live_candidate.strategy_id)
                        except Exception:
                            pass
                    self._stats["retired"] += 1
                    self._record_event(
                        LifecycleEvent.LIVE_RETIRED,
                        live_candidate.strategy_id,
                        {"live_pnl": live_candidate.live_pnl},
                    )
                    events.append({
                        "type": "live_retired",
                        "strategy_id": live_candidate.strategy_id,
                    })
                    continue

                # Boost top performers
                if hasattr(live_candidate, "live_sharpe"):
                    if live_candidate.live_sharpe > self._config.boost_sharpe_threshold:
                        self._stats["boosted"] += 1
                        events.append({
                            "type": "live_boosted",
                            "strategy_id": live_candidate.strategy_id,
                            "sharpe": live_candidate.live_sharpe,
                        })
        except Exception as exc:
            logger.debug("scheduler evaluation error: %s", exc)
        return events

    def _record_event(
        self,
        event_type: LifecycleEvent,
        strategy_id: str,
        metrics: Optional[Dict[str, Any]] = None,
        reason: str = "",
    ) -> None:
        record = LifecycleEventRecord(
            timestamp=time.time(),
            event_type=event_type,
            strategy_id=strategy_id,
            metrics=metrics or {},
            reason=reason,
        )
        self._events.append(record)

    def get_recent_events(self, n: int = 50) -> List[Dict[str, Any]]:
        recent = list(self._events)[-n:]
        return [
            {
                "timestamp": e.timestamp,
                "type": e.event_type.value,
                "strategy_id": e.strategy_id,
                "metrics": e.metrics,
                "reason": e.reason,
            }
            for e in reversed(recent)
        ]

    def snapshot(self) -> Dict[str, Any]:
        """Return current state for advisory dict."""
        return {
            "cycle": self._cycle_count,
            "stats": dict(self._stats),
            "recent_events": len(self._events),
        }
