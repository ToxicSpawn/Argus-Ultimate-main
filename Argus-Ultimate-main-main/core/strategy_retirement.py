"""Automated strategy retirement — Push 100 StatePersist.

Tracks per-strategy performance metrics over a rolling evaluation
window and automatically retires (disables) strategies that fail
pre-configured gates.  Retired strategies are:
  • Flagged in Redis (survives restarts)
  • Published to the ``argus:strategy:retired`` Redis channel
  • Recorded in Prometheus counter  ``argus_strategy_retirements_total``

Retirement gates (all configurable)::

    sharpe_ratio   < min_sharpe         (default 0.3  over eval_window)
    drawdown       > max_drawdown       (default 0.25  absolute)
    win_rate       < min_win_rate       (default 0.35)
    trade_count    < min_trade_count    (within eval window — skip if sparse)
    consecutive_losses > max_cons_loss  (default 10)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Optional Prometheus
try:
    from prometheus_client import Counter as _PCounter
    _retirement_counter = _PCounter(
        "argus_strategy_retirements_total",
        "Total number of automated strategy retirements",
        ["strategy", "reason"],
    )
except Exception:  # noqa: BLE001
    _retirement_counter = None  # type: ignore[assignment]


@dataclass
class RetirementConfig:
    min_sharpe:        float = 0.30
    max_drawdown:      float = 0.25
    min_win_rate:      float = 0.35
    min_trade_count:   int   = 20      # skip evaluation if fewer trades
    max_cons_loss:     int   = 10
    eval_window_secs:  float = 86_400.0  # 24 h rolling window
    cooldown_secs:     float = 3_600.0   # don't re-evaluate within 1 h of retirement


@dataclass
class StrategyStats:
    strategy:          str
    sharpe:            float = 0.0
    drawdown:          float = 0.0
    win_rate:          float = 0.5
    trade_count:       int   = 0
    consecutive_losses: int  = 0
    last_updated:      float = field(default_factory=time.time)


@dataclass
class RetirementRecord:
    strategy:    str
    reason:      str
    retired_at:  float = field(default_factory=time.time)
    stats_snapshot: Dict[str, Any] = field(default_factory=dict)


class StrategyRetirementManager:
    """Evaluates strategy health and retires underperformers."""

    def __init__(
        self,
        config: Optional[RetirementConfig] = None,
        state_persist: Optional[Any]       = None,   # core.state_persist.StatePersist
        redis_url: Optional[str]           = None,
    ) -> None:
        self._cfg  = config or RetirementConfig()
        self._sp   = state_persist
        self._redis_url  = redis_url
        self._retired:   Dict[str, RetirementRecord] = {}
        self._last_eval: Dict[str, float]            = {}
        self._restore()

    # ─── Public API ──────────────────────────────────────────────────────────

    def evaluate(self, stats: StrategyStats) -> Optional[str]:
        """Evaluate a strategy; return retirement reason string or None.

        Call this after each evaluation period or on significant
        performance degradation.
        """
        name = stats.strategy

        # Skip if already retired
        if name in self._retired:
            return f"already_retired:{self._retired[name].reason}"

        # Cooldown guard
        last = self._last_eval.get(name, 0.0)
        if time.time() - last < self._cfg.cooldown_secs and last > 0:
            return None
        self._last_eval[name] = time.time()

        # Skip sparse strategies
        if stats.trade_count < self._cfg.min_trade_count:
            log.debug("%s: too few trades (%d) — skipping evaluation", name, stats.trade_count)
            return None

        reason = self._check_gates(stats)
        if reason:
            self._retire(stats, reason)
        return reason

    def is_retired(self, strategy: str) -> bool:
        return strategy in self._retired

    def active_strategies(self, all_strategies: List[str]) -> List[str]:
        """Filter out retired strategies from a list."""
        return [s for s in all_strategies if s not in self._retired]

    def reinstate(self, strategy: str) -> None:
        """Manually un-retire a strategy (e.g. after patching)."""
        if strategy in self._retired:
            del self._retired[strategy]
            self._save()
            log.info("Strategy %s reinstated", strategy)

    def list_retired(self) -> List[RetirementRecord]:
        return list(self._retired.values())

    # ─── Internal ────────────────────────────────────────────────────────────

    def _check_gates(self, s: StrategyStats) -> Optional[str]:
        if s.sharpe < self._cfg.min_sharpe:
            return f"sharpe={s.sharpe:.3f}<{self._cfg.min_sharpe}"
        if s.drawdown > self._cfg.max_drawdown:
            return f"drawdown={s.drawdown:.3f}>{self._cfg.max_drawdown}"
        if s.win_rate < self._cfg.min_win_rate:
            return f"win_rate={s.win_rate:.3f}<{self._cfg.min_win_rate}"
        if s.consecutive_losses > self._cfg.max_cons_loss:
            return f"consecutive_losses={s.consecutive_losses}>{self._cfg.max_cons_loss}"
        return None

    def _retire(self, stats: StrategyStats, reason: str) -> None:
        rec = RetirementRecord(
            strategy=stats.strategy,
            reason=reason,
            stats_snapshot={
                "sharpe":             stats.sharpe,
                "drawdown":           stats.drawdown,
                "win_rate":           stats.win_rate,
                "trade_count":        stats.trade_count,
                "consecutive_losses": stats.consecutive_losses,
            },
        )
        self._retired[stats.strategy] = rec
        self._save()
        self._publish(rec)

        if _retirement_counter:
            try:
                _retirement_counter.labels(
                    strategy=stats.strategy, reason=reason
                ).inc()
            except Exception:  # noqa: BLE001
                pass

        log.warning(
            "Strategy RETIRED: %s | reason=%s | sharpe=%.3f dd=%.3f wr=%.3f",
            stats.strategy, reason, stats.sharpe, stats.drawdown, stats.win_rate,
        )

    def _publish(self, rec: RetirementRecord) -> None:
        """Publish retirement event to Redis pub/sub channel."""
        if not self._redis_url:
            return
        try:
            import redis  # type: ignore
            import json
            r = redis.from_url(self._redis_url)
            payload = json.dumps({
                "strategy":   rec.strategy,
                "reason":     rec.reason,
                "retired_at": rec.retired_at,
            })
            r.publish("argus:strategy:retired", payload)
        except Exception as exc:  # noqa: BLE001
            log.error("_publish retirement event failed: %s", exc)

    def _save(self) -> None:
        if self._sp is None:
            return
        try:
            import json
            data = {
                name: {
                    "strategy":       r.strategy,
                    "reason":         r.reason,
                    "retired_at":     r.retired_at,
                    "stats_snapshot": r.stats_snapshot,
                }
                for name, r in self._retired.items()
            }
            self._sp._set(  # type: ignore[attr-defined]
                self._sp._key("retired_strategies"),  # type: ignore[attr-defined]
                data,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("StrategyRetirementManager._save failed: %s", exc)

    def _restore(self) -> None:
        if self._sp is None:
            return
        try:
            data = self._sp._get(  # type: ignore[attr-defined]
                self._sp._key("retired_strategies")  # type: ignore[attr-defined]
            )
            if not data:
                return
            for name, r in data.items():
                self._retired[name] = RetirementRecord(
                    strategy=r["strategy"],
                    reason=r["reason"],
                    retired_at=r["retired_at"],
                    stats_snapshot=r.get("stats_snapshot", {}),
                )
            log.info("StrategyRetirementManager: restored %d retired strategies", len(self._retired))
        except Exception as exc:  # noqa: BLE001
            log.warning("StrategyRetirementManager._restore failed: %s", exc)
