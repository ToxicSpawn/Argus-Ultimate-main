#!/usr/bin/env python3
"""
Self-Improvement Orchestrator — drives the shadow-tune + quantum on/off
trial lifecycle for continuous strategy and parameter improvement.

This module coordinates:
1. Shadow backtest trials on recent data (train/test split)
2. Quantum feature on/off A/B trials
3. Promotion decision: apply improved params to live config
4. Debounce / safety gate: never apply if drawdown exceeds threshold
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    trial_id: str
    timeframe: str
    quantum_on: bool
    total_return: float
    sharpe_like: float
    max_drawdown_pct: float
    n_trades: int
    params: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class SelfImprovementOrchestrator:
    """
    Async orchestrator for the shadow-tuning lifecycle.

    Intended usage (non-blocking):
        orch = SelfImprovementOrchestrator(config, trading_system)
        asyncio.create_task(orch.run_forever())
    """

    def __init__(self, config: Any, trading_system: Any = None):
        self.config = config
        self.trading_system = trading_system
        self._last_run_ts: float = 0.0
        self._trial_history: List[TrialResult] = []
        self._running = False
        self._state_path = str(getattr(config, "self_improvement_state_path", "data/self_improvement_state.json") or "data/self_improvement_state.json")
        self._load_state()

    # ---------------------------------------------------------------- lifecycle

    async def run_forever(self) -> None:
        """Background loop — wakes at self_improvement_tick_seconds."""
        self._running = True
        tick = max(1, int(getattr(self.config, "self_improvement_tick_seconds", 60) or 60))
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("SelfImprovementOrchestrator tick: %s", e)
            await asyncio.sleep(tick)

    async def _tick(self) -> None:
        enabled = bool(getattr(self.config, "self_improvement_enabled", True))
        if not enabled:
            return
        mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
        modes = list(getattr(self.config, "self_improvement_modes", ["paper", "backtest"]) or ["paper", "backtest"])
        if mode not in modes:
            return
        shadow_interval_min = max(30, int(getattr(self.config, "self_improvement_shadow_interval_minutes", 240) or 240))
        debounce_min = max(5, float(getattr(self.config, "evolution_debounce_minutes", 15) or 15))
        now = time.time()
        if (now - self._last_run_ts) < debounce_min * 60:
            return
        if (now - self._last_run_ts) < shadow_interval_min * 60:
            return
        shadow_enabled = bool(getattr(self.config, "self_improvement_shadow_tune_enabled", True))
        if not shadow_enabled:
            return
        await self._run_shadow_trial()

    async def _run_shadow_trial(self) -> None:
        """Run one shadow backtest trial (best-effort, all errors swallowed)."""
        try:
            from self_improvement import SelfImprovingBot
            ts = self.trading_system
            market_data = getattr(ts, "market_data_service", None) if ts else None
            bot = SelfImprovingBot(self.config, market_data_service=market_data)
            result = await asyncio.wait_for(
                bot.run_shadow_tune_trial(),
                timeout=300.0,
            )
            if result and isinstance(result, dict):
                trial = TrialResult(
                    trial_id=result.get("trial_id", f"t_{int(time.time())}"),
                    timeframe=str(result.get("timeframe", "1h")),
                    quantum_on=bool(result.get("quantum_on", False)),
                    total_return=float(result.get("total_return", 0.0)),
                    sharpe_like=float(result.get("sharpe_like", 0.0)),
                    max_drawdown_pct=float(result.get("max_drawdown_pct", 0.0)),
                    n_trades=int(result.get("n_trades", 0)),
                    params=dict(result.get("params", {})),
                )
                self._trial_history.append(trial)
                self._last_run_ts = time.time()
                self._maybe_promote(trial)
                self._save_state()
                logger.info(
                    "SelfImprovementOrchestrator: trial complete return=%.2f%% sharpe=%.2f drawdown=%.2f%% trades=%d",
                    trial.total_return * 100, trial.sharpe_like, trial.max_drawdown_pct * 100, trial.n_trades,
                )
        except asyncio.TimeoutError:
            logger.warning("SelfImprovementOrchestrator: shadow trial timed out")
        except Exception as e:
            logger.debug("SelfImprovementOrchestrator shadow trial: %s", e)
            self._last_run_ts = time.time()  # prevent spin on repeated failures

    def _maybe_promote(self, trial: TrialResult) -> None:
        """Apply trial params to live config if improvement thresholds are met."""
        try:
            min_delta = float(getattr(self.config, "self_improvement_min_delta_return_pct", 0.10) or 0.10) / 100.0
            max_dd = float(getattr(self.config, "self_improvement_max_drawdown_pct", 2.0) or 2.0) / 100.0
            min_trades = int(getattr(self.config, "self_improvement_min_trades", 3) or 3)
            apply_only_on_improvement = bool(getattr(self.config, "self_improvement_apply_on_improvement_only", True))
            dry_run = bool(getattr(self.config, "evolution_dry_run", False))

            if trial.n_trades < min_trades:
                return
            if trial.max_drawdown_pct > max_dd:
                return
            if apply_only_on_improvement and trial.total_return < min_delta:
                return
            if dry_run:
                logger.info("SelfImprovementOrchestrator: dry_run — skipping param apply")
                return
            if not trial.params:
                return
            # Apply quantum choice
            apply_q = bool(getattr(self.config, "self_improvement_apply_quantum_choice", True))
            if apply_q:
                setattr(self.config, "quantum_features_enabled", bool(trial.quantum_on))
            # Apply evolved params
            from evolution.apply_evolved_strategies import apply_params_dict
            n = apply_params_dict(self.config, trial.params)
            logger.info(
                "SelfImprovementOrchestrator: promoted trial %s → applied %d params (quantum=%s)",
                trial.trial_id, n, trial.quantum_on,
            )
        except Exception as e:
            logger.debug("SelfImprovementOrchestrator promote: %s", e)

    # ---------------------------------------------------------------- state persistence

    def _save_state(self) -> None:
        try:
            Path(Path(self._state_path).parent).mkdir(parents=True, exist_ok=True)
            state = {
                "last_run_ts": self._last_run_ts,
                "trial_history": [
                    {
                        "trial_id": t.trial_id,
                        "timeframe": t.timeframe,
                        "quantum_on": t.quantum_on,
                        "total_return": t.total_return,
                        "sharpe_like": t.sharpe_like,
                        "max_drawdown_pct": t.max_drawdown_pct,
                        "n_trades": t.n_trades,
                        "ts": t.ts,
                    }
                    for t in self._trial_history[-50:]
                ],
            }
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, default=str)
        except Exception as e:
            logger.debug("SelfImprovementOrchestrator save_state: %s", e)

    def _load_state(self) -> None:
        try:
            p = Path(self._state_path)
            if not p.exists():
                return
            with open(p, encoding="utf-8") as f:
                state = json.load(f)
            self._last_run_ts = float(state.get("last_run_ts", 0.0) or 0.0)
        except Exception as e:
            logger.debug("SelfImprovementOrchestrator load_state: %s", e)

    def stop(self) -> None:
        self._running = False
