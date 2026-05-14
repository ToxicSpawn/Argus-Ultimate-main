"""Push 78 — RiskEvent: typed risk alert dataclass + pub/sub bus.

RiskEventType covers all breach categories:
  DRAWDOWN_BREACH   — symbol drawdown exceeded limit
  PORTFOLIO_HEAT    — portfolio heat exceeded threshold
  VAR_BREACH        — VaR limit breached
  KILL_SWITCH       — global trading halt activated
  KILL_SWITCH_RESET — kill switch cleared
  MARGIN_SOFT       — margin ratio above soft threshold
  MARGIN_HARD       — margin ratio above hard threshold (force reduce)
  POSITION_REDUCED  — position auto-reduced by margin watcher
  DAILY_LOSS_LIMIT  — daily loss limit hit
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class RiskEventType(str, Enum):
    DRAWDOWN_BREACH  = "DRAWDOWN_BREACH"
    PORTFOLIO_HEAT   = "PORTFOLIO_HEAT"
    VAR_BREACH       = "VAR_BREACH"
    KILL_SWITCH      = "KILL_SWITCH"
    KILL_SWITCH_RESET = "KILL_SWITCH_RESET"
    MARGIN_SOFT      = "MARGIN_SOFT"
    MARGIN_HARD      = "MARGIN_HARD"
    POSITION_REDUCED = "POSITION_REDUCED"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"


@dataclass
class RiskEvent:
    event_type:  RiskEventType
    message:     str
    symbol:      Optional[str]  = None
    value:       Optional[float] = None   # e.g. current drawdown %
    threshold:   Optional[float] = None   # e.g. drawdown limit
    timestamp:   float = field(default_factory=time.time)
    metadata:    Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"RiskEvent({self.event_type.value} "
            f"sym={self.symbol} val={self.value:.4f if self.value else 'N/A'})"
        )


class RiskEventBus:
    """Lightweight synchronous pub/sub for risk alerts.

    Usage:
        bus = RiskEventBus()
        bus.subscribe(my_handler)
        bus.emit(RiskEvent(...))
    """

    def __init__(self):
        self._handlers: List[Callable[[RiskEvent], None]] = []
        self._history:  List[RiskEvent] = []

    def subscribe(self, handler: Callable[[RiskEvent], None]) -> None:
        self._handlers.append(handler)

    def emit(self, event: RiskEvent) -> None:
        self._history.append(event)
        for h in self._handlers:
            try:
                h(event)
            except Exception:
                pass

    @property
    def history(self) -> List[RiskEvent]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    def events_of_type(self, etype: RiskEventType) -> List[RiskEvent]:
        return [e for e in self._history if e.event_type == etype]
