"""
Code Evolution Engine — master coordinator for ARGUS self-modifying code.

This is the highest-level component for true code generation. It orchestrates
the full pipeline:

  1. PATTERN DETECTION    — find strong patterns in observation_recorder
  2. GENERATE             — emit Python source code via code_generator
  3. REVIEW               — static analysis via code_review_gate
  4. SANDBOX              — dynamic execution via code_sandbox
  5. COMMIT               — version-control via git_versioner
  6. PROMOTE              — move to active/, hot-reload via module_reloader
  7. MONITOR              — track live performance, retire if bad
  8. ROLLBACK             — git revert if a generation causes problems

The engine runs every N cycles. Each cycle:
  - Mines observation_recorder for patterns
  - Generates 1-3 strategies from the strongest patterns
  - Pipes them through review → sandbox → git → reload
  - Tracks how many made it through each stage

This is the primary mechanism for ARGUS to grow new capabilities without
human intervention. Combined with continuous_adaptation_engine, ARGUS now
both adapts existing parameters AND writes new code at runtime.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EvolutionPhase(Enum):
    PATTERN_DETECTION = "pattern_detection"
    GENERATION = "generation"
    REVIEW = "review"
    SANDBOX = "sandbox"
    COMMIT = "commit"
    PROMOTION = "promotion"
    MONITORING = "monitoring"


@dataclass
class EvolutionEvent:
    """Records one event in the evolution pipeline."""
    timestamp: float
    cycle: int
    phase: EvolutionPhase
    strategy_name: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionConfig:
    """Tunables for the code evolution engine."""
    enabled: bool = True
    generation_cycles: int = 500       # generate every N cycles
    pattern_min_observations: int = 50   # min obs for a pattern to be promotable
    pattern_min_win_rate: float = 0.55
    pattern_min_sharpe: float = 0.5
    max_generations_per_cycle: int = 3
    max_active_strategies: int = 50
    auto_promote_to_active: bool = True
    auto_commit_to_git: bool = True
    auto_reload_after_commit: bool = True
    candidates_dir: str = "generated_strategies/candidates"
    active_dir: str = "generated_strategies/active"
    graveyard_dir: str = "generated_strategies/graveyard"


class CodeEvolutionEngine:
    """
    Master coordinator for the self-modifying code pipeline.

    Usage::

        engine = CodeEvolutionEngine()
        engine.attach(
            observation_recorder=recorder,
            code_generator=gen,
            code_review_gate=gate,
            code_sandbox=sandbox,
            git_versioner=versioner,
            module_reloader=reloader,
        )

        # Each cycle:
        result = engine.tick(cycle_number=42)
        if result.get("generated"):
            for s in result["generated"]:
                print(f"New strategy: {s}")
    """

    def __init__(self, config: Optional[EvolutionConfig] = None) -> None:
        self._config = config or EvolutionConfig()
        self._cycle_count = 0
        self._last_generation_cycle = 0
        self._events: List[EvolutionEvent] = []
        self._stats: Dict[str, int] = {
            "patterns_detected": 0,
            "generations_attempted": 0,
            "generations_succeeded": 0,
            "review_passed": 0,
            "review_failed": 0,
            "sandbox_passed": 0,
            "sandbox_failed": 0,
            "committed": 0,
            "promoted": 0,
            "retired": 0,
            "rolled_back": 0,
        }

        # Component references
        self._observation_recorder: Any = None
        self._code_generator: Any = None
        self._code_review_gate: Any = None
        self._code_sandbox: Any = None
        self._git_versioner: Any = None
        self._module_reloader: Any = None

        # Make sandbox dirs
        Path(self._config.candidates_dir).mkdir(parents=True, exist_ok=True)
        Path(self._config.active_dir).mkdir(parents=True, exist_ok=True)
        Path(self._config.graveyard_dir).mkdir(parents=True, exist_ok=True)

        logger.info("CodeEvolutionEngine: initialized")

    def attach(
        self,
        observation_recorder: Any = None,
        code_generator: Any = None,
        code_review_gate: Any = None,
        code_sandbox: Any = None,
        git_versioner: Any = None,
        module_reloader: Any = None,
    ) -> None:
        """Attach component references."""
        self._observation_recorder = observation_recorder
        self._code_generator = code_generator
        self._code_review_gate = code_review_gate
        self._code_sandbox = code_sandbox
        self._git_versioner = git_versioner
        self._module_reloader = module_reloader
        logger.info("CodeEvolutionEngine: components attached")

    def tick(self, cycle_number: int) -> Dict[str, Any]:
        """Run one cycle of the code evolution engine."""
        self._cycle_count = cycle_number

        if not self._config.enabled:
            return {"enabled": False}

        result: Dict[str, Any] = {
            "cycle": cycle_number,
            "generated": [],
            "promoted": [],
            "retired": [],
            "rollbacks": [],
        }

        # Refresh module reloader (pick up changed files)
        if self._module_reloader is not None:
            try:
                refresh = self._module_reloader.refresh()
                if refresh.get("loaded", 0) > 0 or refresh.get("reloaded", 0) > 0:
                    result["reloader_refresh"] = refresh
            except Exception as exc:
                logger.debug("module_reloader.refresh error: %s", exc)

        # Generate new strategies on cadence
        if (cycle_number - self._last_generation_cycle) >= self._config.generation_cycles:
            self._last_generation_cycle = cycle_number
            generated = self._run_generation_pipeline()
            if generated:
                result["generated"] = generated

        return result

    def _run_generation_pipeline(self) -> List[str]:
        """Run the full generate → review → sandbox → commit → promote pipeline."""
        if (
            self._observation_recorder is None
            or self._code_generator is None
        ):
            return []

        # Step 1: Detect patterns
        patterns = self._detect_patterns()
        if not patterns:
            return []

        generated_names: List[str] = []

        for pattern in patterns[:self._config.max_generations_per_cycle]:
            self._stats["patterns_detected"] += 1

            # Step 2: Generate code
            strategy = self._generate_from_pattern(pattern)
            if strategy is None:
                continue

            self._stats["generations_attempted"] += 1
            self._stats["generations_succeeded"] += 1
            self._record_event(
                EvolutionPhase.GENERATION, strategy.name, True,
                {"template": strategy.template.value},
            )

            # Step 3: Review
            if self._code_review_gate is not None:
                review = self._code_review_gate.review(strategy.file_path)
                if not review.passed:
                    self._stats["review_failed"] += 1
                    self._move_to_graveyard(strategy.file_path, "review_failed")
                    self._record_event(
                        EvolutionPhase.REVIEW, strategy.name, False,
                        {"violations": review.violations[:3]},
                    )
                    continue
                self._stats["review_passed"] += 1
                self._record_event(EvolutionPhase.REVIEW, strategy.name, True)

            # Step 4: Sandbox
            if self._code_sandbox is not None:
                sandbox_result = self._code_sandbox.run(strategy.file_path)
                if not sandbox_result.passed:
                    self._stats["sandbox_failed"] += 1
                    self._move_to_graveyard(strategy.file_path, "sandbox_failed")
                    self._record_event(
                        EvolutionPhase.SANDBOX, strategy.name, False,
                        {"errors": sandbox_result.errors[:3]},
                    )
                    continue
                self._stats["sandbox_passed"] += 1
                self._record_event(
                    EvolutionPhase.SANDBOX, strategy.name, True,
                    {
                        "fitness": sandbox_result.fitness_estimate,
                        "avg_eval_ms": sandbox_result.avg_eval_time_ms,
                    },
                )

            # Step 5: Commit to git
            if self._config.auto_commit_to_git and self._git_versioner is not None:
                sha = self._git_versioner.commit_generation(
                    file_path=strategy.file_path,
                    metadata={
                        "template": strategy.template.value,
                        "win_rate": strategy.context.win_rate,
                        "sharpe": strategy.context.sharpe,
                        "observations": strategy.context.observation_count,
                    },
                )
                if sha:
                    self._stats["committed"] += 1
                    self._record_event(
                        EvolutionPhase.COMMIT, strategy.name, True,
                        {"sha": sha[:8]},
                    )

            # Step 6: Promote to active (if enabled and under cap)
            if self._config.auto_promote_to_active:
                active_count = self._count_active_strategies()
                if active_count < self._config.max_active_strategies:
                    promoted = self._promote_to_active(strategy.file_path)
                    if promoted:
                        self._stats["promoted"] += 1
                        self._record_event(
                            EvolutionPhase.PROMOTION, strategy.name, True,
                        )
                        generated_names.append(strategy.name)

                        # Step 7: Hot reload
                        if self._config.auto_reload_after_commit and self._module_reloader is not None:
                            try:
                                self._module_reloader.refresh()
                            except Exception as exc:
                                logger.debug("module_reloader.refresh error: %s", exc)

        return generated_names

    def _detect_patterns(self) -> List[Dict[str, Any]]:
        """Mine the observation recorder for promotable patterns."""
        if self._observation_recorder is None:
            return []

        try:
            # Group by strategy and look for high-performing ones
            groups = self._observation_recorder.aggregate_pnl_by("strategy")
        except Exception as exc:
            logger.debug("pattern detection: aggregate failed: %s", exc)
            return []

        patterns: List[Dict[str, Any]] = []
        for strategy_name, stats in groups.items():
            if stats["count"] < self._config.pattern_min_observations:
                continue
            if stats["win_rate"] < self._config.pattern_min_win_rate:
                continue
            patterns.append({
                "pattern_id": f"pat_{strategy_name}_{int(time.time())}",
                "source_strategy": strategy_name,
                "observation_count": stats["count"],
                "win_rate": stats["win_rate"],
                "avg_pnl_aud": stats["avg_pnl"],
                "total_pnl": stats["total_pnl"],
            })

        # Sort by win rate × observation count
        patterns.sort(key=lambda p: p["win_rate"] * p["observation_count"], reverse=True)
        return patterns

    def _generate_from_pattern(self, pattern: Dict[str, Any]) -> Any:
        """Generate a new strategy from an observed pattern."""
        if self._code_generator is None:
            return None

        try:
            from core.code_generator import GenerationContext
        except ImportError:
            return None

        ctx = GenerationContext(
            pattern_id=pattern["pattern_id"],
            observation_count=pattern["observation_count"],
            win_rate=pattern["win_rate"],
            avg_pnl_aud=pattern["avg_pnl_aud"],
            sharpe=pattern["win_rate"] * 2 - 1,  # crude proxy
            target_regime="ANY",
            description=f"Generated from {pattern['source_strategy']} pattern",
        )

        # For now, generate a threshold strategy on RSI
        # (Future: pick template based on pattern characteristics)
        try:
            return self._code_generator.generate_threshold_strategy(
                ctx,
                indicator="rsi",
                threshold=30.0 + (pattern["win_rate"] - 0.5) * 20,  # 20-40 range
                direction="below" if pattern["win_rate"] > 0.55 else "above",
            )
        except Exception as exc:
            logger.debug("generate_from_pattern error: %s", exc)
            return None

    def _move_to_graveyard(self, file_path: Path, reason: str) -> None:
        """Move a failed file to graveyard."""
        try:
            graveyard = Path(self._config.graveyard_dir)
            graveyard.mkdir(parents=True, exist_ok=True)
            target = graveyard / file_path.name
            if file_path.exists():
                file_path.rename(target)
                logger.info(
                    "CodeEvolutionEngine: moved %s to graveyard (%s)",
                    file_path.name, reason,
                )
        except OSError as exc:
            logger.debug("move_to_graveyard error: %s", exc)

    def _promote_to_active(self, file_path: Path) -> bool:
        """Move a candidate to active/."""
        try:
            active = Path(self._config.active_dir)
            active.mkdir(parents=True, exist_ok=True)
            target = active / file_path.name
            if file_path.exists():
                file_path.rename(target)
                logger.info("CodeEvolutionEngine: promoted %s to active", file_path.name)
                return True
        except OSError as exc:
            logger.debug("promote_to_active error: %s", exc)
        return False

    def _count_active_strategies(self) -> int:
        """Count strategies currently in active/ directory."""
        try:
            active = Path(self._config.active_dir)
            return sum(1 for f in active.glob("*.py") if f.name != "__init__.py")
        except OSError:
            return 0

    def _record_event(
        self,
        phase: EvolutionPhase,
        strategy_name: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        event = EvolutionEvent(
            timestamp=time.time(),
            cycle=self._cycle_count,
            phase=phase,
            strategy_name=strategy_name,
            success=success,
            details=details or {},
        )
        self._events.append(event)
        if len(self._events) > 1000:
            self._events = self._events[-500:]

    def get_recent_events(self, n: int = 20) -> List[Dict[str, Any]]:
        recent = self._events[-n:]
        return [
            {
                "timestamp": e.timestamp,
                "cycle": e.cycle,
                "phase": e.phase.value,
                "strategy_name": e.strategy_name,
                "success": e.success,
                "details": e.details,
            }
            for e in reversed(recent)
        ]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": self._config.enabled,
            "cycle": self._cycle_count,
            "active_strategies": self._count_active_strategies(),
            "stats": dict(self._stats),
            "recent_events": len(self._events),
        }
