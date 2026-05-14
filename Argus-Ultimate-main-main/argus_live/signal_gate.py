"""
Signal Gate — MTF confluence pre-execution filter.

Wraps MTFConfluenceFilter.check() as a synchronous gate that the
main trading loop calls on each candidate signal list BEFORE passing
them to KrakenDCAExecutionEngine.execute_signals().

Features
--------
* Per-symbol override: force-allow or force-block specific symbols.
* Dry-run mode: logs the gate decision but always allows through
  (useful for shadow-monitoring without blocking live orders).
* Gate stats: tracks allow/block counts per symbol for dashboards.
* Decision audit: writes each gate decision to jsonl_logger when provided.

Usage::

    from argus_live.signal_gate import SignalGate
    from strategies.mtf_confluence import MTFConfluenceFilter

    gate = SignalGate(
        mtf_filter=mtf_filter,
        dry_run=False,       # True = log only, never block
        min_score=0.6,       # override min confluence score (optional)
    )

    # In the main trading loop, before execute_signals():
    approved_signals = gate.filter_signals(signals, run_id=run_id, cycle_id=cycle_id)
    results = await execution_engine.execute_signals(approved_signals)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SignalGate:
    """
    MTF confluence gate for live signal filtering.

    Attributes:
        mtf_filter:     MTFConfluenceFilter instance (push-based cache must be warm).
        dry_run:        If True, never block — only log decisions.
        min_score:      Minimum confluence score to pass (overrides filter default).
        force_allow:    Set of symbols that always pass regardless of MTF score.
        force_block:    Set of symbols that always block regardless of MTF score.
        jsonl_logger:   Optional JSONLLogger instance for audit trail.
    """

    def __init__(
        self,
        mtf_filter: Any,
        dry_run: bool = False,
        min_score: Optional[float] = None,
        force_allow: Optional[Set[str]] = None,
        force_block: Optional[Set[str]] = None,
        jsonl_logger: Optional[Any] = None,
    ):
        self.mtf_filter = mtf_filter
        self.dry_run = dry_run
        self.min_score = min_score
        self.force_allow: Set[str] = set(force_allow or [])
        self.force_block: Set[str] = set(force_block or [])
        self.jsonl_logger = jsonl_logger

        # Stats: per-symbol {"allowed": int, "blocked": int, "no_data": int}
        self._stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"allowed": 0, "blocked": 0, "no_data": 0})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_signals(
        self,
        signals: List[Any],
        *,
        run_id: str = "",
        cycle_id: int = 0,
    ) -> List[Any]:
        """
        Filter a list of signals through the MTF confluence gate.

        Signals that do not pass are dropped (or passed through in dry_run mode).

        Args:
            signals:   List of signal objects with ``.symbol`` and ``.action`` attributes.
            run_id:    Run ID for audit logging.
            cycle_id:  Cycle ID for audit logging.

        Returns:
            Filtered list of approved signals.
        """
        if not signals:
            return []

        approved = []
        for signal in signals:
            result = self._gate_one(signal, run_id=run_id, cycle_id=cycle_id)
            if result:
                approved.append(signal)

        logger.debug(
            "SignalGate cycle %d: %d/%d signals approved",
            cycle_id, len(approved), len(signals),
        )
        return approved

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Return per-symbol gate statistics."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset all gate statistics counters."""
        self._stats.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _gate_one(
        self,
        signal: Any,
        *,
        run_id: str,
        cycle_id: int,
    ) -> bool:
        """
        Gate a single signal through MTF confluence check.

        Returns True if the signal should proceed to execution.
        """
        symbol = str(getattr(signal, "symbol", "") or "")
        action = str(getattr(signal, "action", "") or getattr(signal, "side", "") or "").lower()
        direction = "buy" if action in ("buy", "long") else "sell"

        # Force-allow / force-block overrides
        if symbol in self.force_block:
            self._stats[symbol]["blocked"] += 1
            self._audit(run_id, cycle_id, symbol, direction,
                        approved=False, score=0.0, reason="force_block_override")
            if self.dry_run:
                logger.info("SignalGate [DRY-RUN] BLOCK %s %s: force_block_override", direction, symbol)
                return True
            logger.info("SignalGate BLOCK %s %s: force_block_override", direction, symbol)
            return False

        if symbol in self.force_allow:
            self._stats[symbol]["allowed"] += 1
            self._audit(run_id, cycle_id, symbol, direction,
                        approved=True, score=1.0, reason="force_allow_override")
            logger.debug("SignalGate ALLOW %s %s: force_allow_override", direction, symbol)
            return True

        # MTF confluence check
        try:
            approved, score, reason = self.mtf_filter.check(
                symbol=symbol,
                signal_direction=direction,
            )
        except Exception as exc:
            logger.warning(
                "SignalGate MTF check failed for %s: %s — allowing through", symbol, exc
            )
            self._stats[symbol]["no_data"] += 1
            return True  # fail-open: don't block on filter errors

        # Override min_score if configured
        if self.min_score is not None and score < self.min_score and approved:
            approved = False
            reason = f"score={score:.2f}_below_gate_min={self.min_score:.2f}"

        # Track "no_data_available" reason as no_data stat
        if reason == "no_mtf_data_available":
            self._stats[symbol]["no_data"] += 1
        elif approved:
            self._stats[symbol]["allowed"] += 1
        else:
            self._stats[symbol]["blocked"] += 1

        self._audit(run_id, cycle_id, symbol, direction,
                    approved=approved, score=score, reason=reason)

        if self.dry_run and not approved:
            logger.info(
                "SignalGate [DRY-RUN] would BLOCK %s %s: %s (score=%.2f)",
                direction, symbol, reason, score,
            )
            return True  # dry_run: let through anyway

        if not approved:
            logger.info(
                "SignalGate BLOCK %s %s: %s (score=%.2f)",
                direction, symbol, reason, score,
            )
        else:
            logger.debug(
                "SignalGate ALLOW %s %s: %s (score=%.2f)",
                direction, symbol, reason, score,
            )

        return approved

    def _audit(
        self,
        run_id: str,
        cycle_id: int,
        symbol: str,
        direction: str,
        *,
        approved: bool,
        score: float,
        reason: str,
    ) -> None:
        """Write gate decision to jsonl_logger if configured."""
        if self.jsonl_logger is None:
            return
        try:
            self.jsonl_logger.write({
                "kind": "mtf_gate_decision",
                "run_id": str(run_id or ""),
                "cycle_id": int(cycle_id),
                "symbol": symbol,
                "direction": direction,
                "approved": bool(approved),
                "score": float(score),
                "reason": str(reason),
                "dry_run": bool(self.dry_run),
                "ts": float(time.time()),
            })
        except Exception:
            pass
