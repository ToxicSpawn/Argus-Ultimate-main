"""
EscalatingGate — Temporal MetaGate memory with escalation.

Wraps the MetaGate *output* (not the MetaGate itself) with a state machine
that remembers how long the system has been at each gate level.

Escalation rules:
  REDUCE  for >= reduce_escalation_cycles  → escalated decision = PAUSE
  PAUSE   for >= pause_escalation_cycles   → escalated decision = HALT

De-escalation rules:
  ALLOW   for >= de_escalation_cycles      → de-escalate from PAUSE back to REDUCE
                                             (de-escalation only fires if there was a
                                              previous escalation, not from natural ALLOW)

The raw MetaGate decision is always preserved in the result so callers can
distinguish original signal from temporal override.

Does NOT instantiate or call MetaGate internally — that runs separately in
on_cycle() and its decision is passed in via evaluate(raw_decision).
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Deque, Optional, Tuple

logger = logging.getLogger(__name__)

# Decision rank for comparisons
_RANK = {"allow": 0, "reduce": 1, "pause": 2, "halt": 3}
_DE_ESCALATION_MAP = {"halt": "pause", "pause": "reduce", "reduce": "allow"}


@dataclass
class EscalatingGateDecision:
    current: str              # raw MetaGate decision (allow/reduce/pause/halt)
    escalated: str            # final decision after temporal rules
    cycles_at_level: int      # consecutive cycles at the CURRENT raw level
    was_escalated: bool       # True if escalated != current (temporal rule fired)
    de_escalated: bool        # True if gate improved since last evaluation
    ts: float = field(default_factory=time.time)


class EscalatingGate:
    """
    Temporal escalation layer on top of MetaGate.

    Parameters
    ----------
    reduce_escalation_cycles  : consecutive REDUCE cycles → escalate to PAUSE
    pause_escalation_cycles   : consecutive PAUSE cycles → escalate to HALT
    de_escalation_cycles      : consecutive ALLOW cycles → de-escalate by one step
    config                    : optional config for threshold overrides
    """

    def __init__(
        self,
        reduce_escalation_cycles: int = 100,
        pause_escalation_cycles: int = 200,
        de_escalation_cycles: int = 50,
        config: Optional[Any] = None,
    ) -> None:
        self.reduce_escalation_cycles = max(1, int(reduce_escalation_cycles))
        self.pause_escalation_cycles  = max(1, int(pause_escalation_cycles))
        self.de_escalation_cycles     = max(1, int(de_escalation_cycles))

        # State tracking
        self._consecutive: Dict[str, int] = {
            "allow": 0, "reduce": 0, "pause": 0, "halt": 0
        }
        self._current_level: str = "allow"    # most recent raw decision
        self._escalated_level: Optional[str] = None  # None = no temporal escalation active
        self._allow_streak: int = 0            # consecutive allow cycles (for de-escalation)
        self._prev_escalated: str = "allow"
        self._last_decision: Optional[EscalatingGateDecision] = None

        # History for snapshot
        self._history: Deque[Tuple[str, str, float]] = deque(maxlen=50)  # (raw, final, ts)

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        raw_decision: str,
        meta_gate_inputs: Optional[Dict[str, float]] = None,
    ) -> EscalatingGateDecision:
        """
        Apply temporal escalation rules to the raw MetaGate decision.

        Parameters
        ----------
        raw_decision      : string from advisory["trade_gate"]["decision"]
        meta_gate_inputs  : optional dict for logging context (not used in logic)
        """
        raw = str(raw_decision or "allow").lower().strip()
        if raw not in _RANK:
            raw = "allow"

        de_escalated = False

        # ── Update consecutive counters ───────────────────────────────────────
        if raw != self._current_level:
            # Level changed — reset streak for previous level
            self._consecutive[self._current_level] = 0
        self._current_level = raw
        self._consecutive[raw] = self._consecutive.get(raw, 0) + 1
        cycles_at_level = self._consecutive[raw]

        # ── De-escalation logic ───────────────────────────────────────────────
        # If raw_decision is ALLOW and there was a prior escalation, track it
        if raw == "allow":
            if self._escalated_level is not None:
                self._allow_streak += 1
                if self._allow_streak >= self.de_escalation_cycles:
                    # De-escalate one step
                    self._escalated_level = _DE_ESCALATION_MAP.get(self._escalated_level)
                    self._allow_streak = 0
                    if self._escalated_level == "allow":
                        self._escalated_level = None  # fully recovered
                    de_escalated = True
                    logger.info(
                        "EscalatingGate: DE-ESCALATED after %d ALLOW cycles → %s",
                        self.de_escalation_cycles,
                        self._escalated_level or "allow",
                    )
        else:
            self._allow_streak = 0  # reset allow streak on any non-ALLOW

        # ── Escalation logic ──────────────────────────────────────────────────
        if raw == "reduce" and cycles_at_level >= self.reduce_escalation_cycles:
            if (self._escalated_level is None or
                    _RANK.get(self._escalated_level, 0) < _RANK["pause"]):
                self._escalated_level = "pause"
                logger.warning(
                    "EscalatingGate: ESCALATED reduce→pause after %d cycles",
                    cycles_at_level,
                )
                self._send_log("REDUCE→PAUSE", cycles_at_level)

        elif raw == "pause" and cycles_at_level >= self.pause_escalation_cycles:
            if (self._escalated_level is None or
                    _RANK.get(self._escalated_level, 0) < _RANK["halt"]):
                self._escalated_level = "halt"
                logger.warning(
                    "EscalatingGate: ESCALATED pause→halt after %d cycles",
                    cycles_at_level,
                )
                self._send_log("PAUSE→HALT", cycles_at_level)

        # ── Build final decision ──────────────────────────────────────────────
        # Final = max severity of (raw, escalated)
        escalated_str = self._escalated_level or raw
        final_rank = max(_RANK.get(raw, 0), _RANK.get(escalated_str, 0))
        # Map rank back to string
        final = next(k for k, v in _RANK.items() if v == final_rank)
        was_escalated = (final != raw)

        decision = EscalatingGateDecision(
            current=raw,
            escalated=final,
            cycles_at_level=cycles_at_level,
            was_escalated=was_escalated,
            de_escalated=de_escalated,
        )
        self._prev_escalated = final
        self._last_decision = decision
        self._history.append((raw, final, decision.ts))
        return decision

    def snapshot(self) -> Dict[str, Any]:
        d = self._last_decision
        return {
            "current":       d.current if d else "allow",
            "escalated":     d.escalated if d else "allow",
            "was_escalated": d.was_escalated if d else False,
            "de_escalated":  d.de_escalated if d else False,
            "cycles_at_level": d.cycles_at_level if d else 0,
            "consecutive":   dict(self._consecutive),
            "escalated_level_active": self._escalated_level,
            "thresholds": {
                "reduce_escalation_cycles": self.reduce_escalation_cycles,
                "pause_escalation_cycles":  self.pause_escalation_cycles,
                "de_escalation_cycles":     self.de_escalation_cycles,
            },
            "ts": d.ts if d else None,
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _send_log(self, escalation: str, cycles: int) -> None:
        logger.warning(
            "EscalatingGate: temporal escalation %s (cycles=%d)", escalation, cycles
        )
