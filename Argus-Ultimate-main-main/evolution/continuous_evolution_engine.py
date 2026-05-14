"""
Continuous evolution engine: run evolution on a schedule (e.g. every N hours).

Optional background loop; can be driven by self_improvement tick or external cron.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from evolution.evolution_unified import evolve_once

logger = logging.getLogger(__name__)


class ContinuousEvolutionEngine:
    """
    Run evolution periodically. schedule_interval_hours: run every N hours;
    run_once() runs a single evolution and returns result.
    """

    def __init__(
        self,
        schedule_interval_hours: float = 24.0,
        generations: int = 8,
        population_size: int = 16,
        fitness_days: int = 7,
        config_path: str = "unified_config.yaml",
        on_complete: Optional[Callable[[dict], None]] = None,
    ):
        self.schedule_interval_hours = schedule_interval_hours
        self.generations = generations
        self.population_size = population_size
        self.fitness_days = fitness_days
        self.config_path = config_path
        self.on_complete = on_complete
        self._last_run_time: float = 0.0

    def run_once(self, **kwargs: Any) -> dict:
        result = evolve_once(
            generations=kwargs.get("generations", self.generations),
            population_size=kwargs.get("population_size", self.population_size),
            fitness_days=kwargs.get("fitness_days", self.fitness_days),
            config_path=kwargs.get("config_path", self.config_path),
        )
        self._last_run_time = time.time()
        if self.on_complete:
            try:
                self.on_complete(result)
            except Exception as e:
                logger.warning("on_complete callback failed: %s", e)
        return result

    def tick(self) -> bool:
        """
        Call periodically (e.g. from self_improvement loop).
        Returns True if an evolution run was started this tick.
        """
        now = time.time()
        interval_sec = self.schedule_interval_hours * 3600.0
        if self._last_run_time <= 0 or (now - self._last_run_time) >= interval_sec:
            self.run_once()
            return True
        return False
