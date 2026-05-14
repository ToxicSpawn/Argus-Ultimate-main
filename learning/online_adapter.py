"""
OnlineAdapter — online learning loop.

Updates strategy weights after each closed trade based on a rolling
50-trade win-rate window. Strategies with improving win rates receive
higher weights; underperforming strategies are down-weighted.

Design
------
- Thread-safe via threading.Lock (compatible with asyncio via run_in_executor)
- Weights are bounded to [MIN_WEIGHT, MAX_WEIGHT] to prevent collapse
- Emits weight-change events via optional callback for downstream consumers
- Persists weight state to JSON for restart continuity
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Optional

logger = logging.getLogger(__name__)

ROLLING_WINDOW: int = 50
MIN_WEIGHT: float = 0.05
MAX_WEIGHT: float = 5.0
LEARNING_RATE: float = 0.1          # step size for weight update
DEFAULT_WEIGHT: float = 1.0


@dataclass
class TradeResult:
    strategy_name: str
    is_win: bool                     # True = profitable closed trade
    pnl: float = 0.0
    timestamp: float = field(default_factory=lambda: __import__('time').time())


class StrategyStats:
    """Tracks rolling win-rate for a single strategy."""

    def __init__(self, window: int = ROLLING_WINDOW) -> None:
        self._results: Deque[bool] = deque(maxlen=window)
        self.window = window

    def record(self, is_win: bool) -> None:
        self._results.append(is_win)

    @property
    def win_rate(self) -> float:
        if not self._results:
            return 0.5          # neutral prior
        return sum(self._results) / len(self._results)

    @property
    def sample_count(self) -> int:
        return len(self._results)


class OnlineAdapter:
    """
    Maintains and updates per-strategy weights based on rolling win-rate.

    Parameters
    ----------
    strategy_names   : initial list of strategy identifiers
    rolling_window   : number of recent trades used for win-rate calculation
    learning_rate    : magnitude of weight adjustment per update
    on_weight_change : optional callback(strategy_name, old_w, new_w)
    state_path       : file path for JSON weight persistence (None = disabled)
    """

    def __init__(
        self,
        strategy_names: Optional[list] = None,
        rolling_window: int = ROLLING_WINDOW,
        learning_rate: float = LEARNING_RATE,
        on_weight_change: Optional[Callable[[str, float, float], None]] = None,
        state_path: Optional[str] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._rolling_window = rolling_window
        self._lr = learning_rate
        self._on_weight_change = on_weight_change
        self._state_path = state_path

        self._stats: Dict[str, StrategyStats] = {}
        self._weights: Dict[str, float] = {}

        for name in (strategy_names or []):
            self._register(name)

        if state_path and os.path.exists(state_path):
            self._load_state(state_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_trade(self, result: TradeResult) -> None:
        """
        Record a closed trade outcome and update the strategy's weight.
        Auto-registers unknown strategy names.
        """
        with self._lock:
            name = result.strategy_name
            if name not in self._stats:
                self._register(name)

            self._stats[name].record(result.is_win)
            self._update_weight(name)

            if self._state_path:
                self._save_state(self._state_path)

    def get_weight(self, strategy_name: str) -> float:
        with self._lock:
            return self._weights.get(strategy_name, DEFAULT_WEIGHT)

    def get_all_weights(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._weights)

    def get_win_rate(self, strategy_name: str) -> float:
        with self._lock:
            stats = self._stats.get(strategy_name)
            return stats.win_rate if stats else 0.5

    def summary(self) -> list:
        with self._lock:
            return [
                {
                    "strategy": name,
                    "weight": self._weights[name],
                    "win_rate": self._stats[name].win_rate,
                    "samples": self._stats[name].sample_count,
                }
                for name in sorted(self._weights)
            ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _register(self, name: str) -> None:
        self._stats[name] = StrategyStats(self._rolling_window)
        self._weights[name] = DEFAULT_WEIGHT

    def _update_weight(self, name: str) -> None:
        """
        Gradient-style update:
            delta = learning_rate * (win_rate - 0.5) * 2
        A win_rate of 1.0 → delta = +lr
        A win_rate of 0.0 → delta = -lr
        A win_rate of 0.5 → delta =  0  (neutral)
        """
        stats = self._stats[name]
        if stats.sample_count < 5:
            return          # wait for minimum sample before adjusting

        win_rate = stats.win_rate
        delta = self._lr * (win_rate - 0.5) * 2.0
        old_w = self._weights[name]
        new_w = max(MIN_WEIGHT, min(MAX_WEIGHT, old_w + delta))
        self._weights[name] = new_w

        if new_w != old_w:
            logger.debug(
                "[OnlineAdapter] %s weight %.4f -> %.4f (win_rate=%.3f)",
                name, old_w, new_w, win_rate,
            )
            if self._on_weight_change:
                self._on_weight_change(name, old_w, new_w)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self, path: str) -> None:
        try:
            state = {
                "weights": self._weights,
                "stats": {
                    name: list(s._results)
                    for name, s in self._stats.items()
                },
            }
            with open(path, "w") as fh:
                json.dump(state, fh, indent=2)
        except OSError as exc:
            logger.error("OnlineAdapter: failed to save state: %s", exc)

    def _load_state(self, path: str) -> None:
        try:
            with open(path) as fh:
                state = json.load(fh)
            for name, w in state.get("weights", {}).items():
                if name not in self._weights:
                    self._register(name)
                self._weights[name] = float(w)
            for name, results in state.get("stats", {}).items():
                if name not in self._stats:
                    self._register(name)
                for r in results:
                    self._stats[name].record(bool(r))
            logger.info("OnlineAdapter: loaded state from %s", path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("OnlineAdapter: could not load state from %s: %s", path, exc)
