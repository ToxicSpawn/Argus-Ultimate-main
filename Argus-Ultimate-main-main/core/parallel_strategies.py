"""
Parallel strategy evaluation — runs multiple strategies concurrently.

Uses ThreadPoolExecutor to evaluate 18+ strategies simultaneously instead
of sequentially, reducing per-cycle latency by 3-5x.
"""

import asyncio
import concurrent.futures
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    """Result from a single strategy evaluation."""
    strategy_name: str
    signals: list = field(default_factory=list)
    eval_time_ms: float = 0.0
    error: Optional[str] = None
    timed_out: bool = False


class ParallelStrategyEvaluator:
    """Run multiple strategies concurrently via thread pool."""

    def __init__(self, max_workers: Optional[int] = None, timeout_seconds: float = 5.0):
        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, 16)
        self._max_workers = max_workers
        self._timeout = timeout_seconds
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        # Stats
        self._total_evals = 0
        self._total_time_ms = 0.0
        self._timeouts = 0
        self._errors = 0

        logger.info("ParallelStrategyEvaluator: workers=%d, timeout=%.1fs", max_workers, timeout_seconds)

    def _eval_one(self, strategy: Any, market_data: dict, regime: str) -> StrategyResult:
        """Evaluate a single strategy (runs in thread)."""
        name = getattr(strategy, "name", getattr(strategy, "__class__", type(strategy)).__name__)
        t0 = time.monotonic()
        try:
            # Try different method signatures
            if hasattr(strategy, "generate_signal"):
                result = strategy.generate_signal(market_data)
            elif hasattr(strategy, "analyze"):
                result = strategy.analyze(market_data)
            elif hasattr(strategy, "on_cycle"):
                result = strategy.on_cycle(market_data)
            elif callable(strategy):
                result = strategy(market_data)
            else:
                return StrategyResult(strategy_name=name, error="no_callable_method")

            elapsed_ms = (time.monotonic() - t0) * 1000

            # Normalize result to list of signals
            signals = []
            if result is None:
                signals = []
            elif isinstance(result, list):
                signals = [s for s in result if s is not None]
            else:
                signals = [result]

            return StrategyResult(
                strategy_name=name,
                signals=signals,
                eval_time_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.debug("ParallelStrategyEvaluator: '%s' failed: %s", name, exc)
            return StrategyResult(
                strategy_name=name,
                eval_time_ms=elapsed_ms,
                error=str(exc),
            )

    async def evaluate_all(self, strategies: list, market_data: dict,
                           regime: str = "NORMAL") -> List[StrategyResult]:
        """Run all strategies concurrently, return combined results."""
        if not strategies:
            return []

        loop = asyncio.get_running_loop()
        t0 = time.monotonic()

        # Submit all to thread pool
        futures = []
        for strategy in strategies:
            fut = loop.run_in_executor(
                self._executor,
                self._eval_one,
                strategy, market_data, regime,
            )
            futures.append(fut)

        # Gather with timeout
        results = []
        done, pending = await asyncio.wait(futures, timeout=self._timeout)

        for task in done:
            try:
                result = task.result()
                results.append(result)
                self._total_evals += 1
                self._total_time_ms += result.eval_time_ms
                if result.error:
                    self._errors += 1
            except Exception as exc:
                logger.debug("ParallelStrategyEvaluator: task exception: %s", exc)
                self._errors += 1

        # Handle timeouts
        for task in pending:
            task.cancel()
            self._timeouts += 1
            results.append(StrategyResult(
                strategy_name="unknown",
                timed_out=True,
                error="timeout",
            ))

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug(
            "ParallelStrategyEvaluator: %d strategies in %.1fms (%d signals, %d errors, %d timeouts)",
            len(strategies), elapsed_ms,
            sum(len(r.signals) for r in results),
            sum(1 for r in results if r.error),
            sum(1 for r in results if r.timed_out),
        )

        return results

    def evaluate_sync(self, strategies: list, market_data: dict,
                      regime: str = "NORMAL") -> List[StrategyResult]:
        """Synchronous version using ThreadPoolExecutor.map()."""
        if not strategies:
            return []

        t0 = time.monotonic()
        results = []

        futures_map = {}
        for strategy in strategies:
            fut = self._executor.submit(self._eval_one, strategy, market_data, regime)
            futures_map[fut] = strategy

        done, _ = concurrent.futures.wait(
            futures_map.keys(),
            timeout=self._timeout,
        )

        for fut in done:
            try:
                result = fut.result(timeout=0.1)
                results.append(result)
                self._total_evals += 1
                self._total_time_ms += result.eval_time_ms
            except Exception as exc:
                logger.debug("ParallelStrategyEvaluator: sync exception: %s", exc)
                self._errors += 1

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug("ParallelStrategyEvaluator: sync eval %d strategies in %.1fms", len(strategies), elapsed_ms)
        return results

    def get_all_signals(self, results: List[StrategyResult]) -> list:
        """Extract all signals from evaluation results."""
        signals = []
        for r in results:
            signals.extend(r.signals)
        return signals

    def get_stats(self) -> dict:
        """Return evaluator statistics."""
        return {
            "max_workers": self._max_workers,
            "timeout_seconds": self._timeout,
            "total_evaluations": self._total_evals,
            "avg_eval_time_ms": round(self._total_time_ms / max(self._total_evals, 1), 3),
            "timeouts": self._timeouts,
            "errors": self._errors,
        }

    def shutdown(self):
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=False)
