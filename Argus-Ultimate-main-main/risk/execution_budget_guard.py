from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(slots=True)
class ExecutionBudgetDecision:
    symbol: str
    notional_aud: float
    blocked: bool
    reason: str
    cycle_notional_after_aud: float
    symbol_notional_after_aud: float


class ExecutionBudgetGuard:
    """Conservative per-cycle notional and intent cap before order intent creation."""

    def __init__(self, config: Any):
        self.enabled = bool(getattr(config, "execution_budget_guard_enabled", True))
        self.max_cycle_notional_aud = float(
            getattr(config, "execution_budget_guard_max_cycle_notional_aud", 250.0) or 250.0
        )
        self.max_cycle_intents = int(
            getattr(config, "execution_budget_guard_max_cycle_intents", 3) or 3
        )
        self.max_symbol_notional_aud = float(
            getattr(config, "execution_budget_guard_max_symbol_notional_aud", 150.0) or 150.0
        )
        self.fail_closed = bool(getattr(config, "execution_budget_guard_fail_closed", True))

    @staticmethod
    def _signal_get(signal: Any, field: str, default: Any = None) -> Any:
        if isinstance(signal, dict):
            return signal.get(field, default)
        return getattr(signal, field, default)

    @staticmethod
    def _signal_set(signal: Any, field: str, value: Any) -> None:
        if isinstance(signal, dict):
            signal[field] = value
        else:
            setattr(signal, field, value)

    def _estimate_notional_aud(self, signal: Any, *, portfolio_value_aud: float) -> float:
        price = float(
            self._signal_get(signal, "entry_price", None)
            or self._signal_get(signal, "price", None)
            or self._signal_get(signal, "current_price", None)
            or 0.0
        )
        qty = float(
            self._signal_get(signal, "quantity", None)
            or self._signal_get(signal, "planned_order_size", None)
            or 0.0
        )
        if price > 0.0 and qty > 0.0:
            return max(0.0, float(price * qty))
        delta = float(self._signal_get(signal, "delta_exposure_pct", 0.0) or 0.0)
        if abs(delta) > 0.0 and portfolio_value_aud > 0.0:
            return max(0.0, float(abs(delta) * portfolio_value_aud))
        return 0.0

    def filter_execution_signals(
        self,
        signals: Iterable[Any],
        *,
        portfolio_value_aud: float,
    ) -> Tuple[List[Any], List[Tuple[Any, ExecutionBudgetDecision]]]:
        rows = list(signals or [])
        if not self.enabled:
            return rows, []

        kept: List[Any] = []
        blocked: List[Tuple[Any, ExecutionBudgetDecision]] = []
        cycle_notional = 0.0
        symbol_notionals: Dict[str, float] = {}

        for sig in rows:
            symbol = str(self._signal_get(sig, "symbol", "") or "")
            notional = self._estimate_notional_aud(sig, portfolio_value_aud=float(portfolio_value_aud or 0.0))
            symbol_after = float(symbol_notionals.get(symbol, 0.0) + notional)
            cycle_after = float(cycle_notional + notional)
            reason = ""
            blocked_flag = False

            if len(kept) >= max(1, self.max_cycle_intents):
                reason = "max_cycle_intents"
                blocked_flag = bool(self.fail_closed)
            elif notional <= 0.0 and self.fail_closed:
                reason = "unknown_notional"
                blocked_flag = True
            elif cycle_after > self.max_cycle_notional_aud:
                reason = "max_cycle_notional"
                blocked_flag = bool(self.fail_closed)
            elif symbol_after > self.max_symbol_notional_aud:
                reason = "max_symbol_notional"
                blocked_flag = bool(self.fail_closed)

            decision = ExecutionBudgetDecision(
                symbol=symbol,
                notional_aud=float(notional),
                blocked=bool(blocked_flag),
                reason=str(reason),
                cycle_notional_after_aud=float(cycle_after),
                symbol_notional_after_aud=float(symbol_after),
            )

            self._signal_set(sig, "execution_budget_notional_aud", float(notional))
            self._signal_set(sig, "execution_budget_reason", str(reason))
            self._signal_set(sig, "execution_budget_blocked", bool(blocked_flag))
            self._signal_set(sig, "execution_budget_cycle_notional_aud", float(cycle_after))
            self._signal_set(sig, "execution_budget_symbol_notional_aud", float(symbol_after))

            if blocked_flag:
                blocked.append((sig, decision))
                continue

            cycle_notional = cycle_after
            symbol_notionals[symbol] = symbol_after
            kept.append(sig)

        return kept, blocked
