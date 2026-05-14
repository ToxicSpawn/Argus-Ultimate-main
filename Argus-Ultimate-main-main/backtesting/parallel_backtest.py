"""Parallel backtest runner — merged from backtest/."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ParallelBacktest:
    """Run backtests in parallel across parameter combinations."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def run(self, param_grid: list[dict]) -> list[dict]:
        logger.info('Running parallel backtest across %d configs', len(param_grid))
        return []
