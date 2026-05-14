"""
Typed Protocol interfaces for ARGUS components.

These protocols define the contracts that components must satisfy,
enabling type checking and documentation of expected behavior.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class FillTracker(Protocol):
    """Records fill execution data for slippage analysis."""
    def record_fill(self, record: Any) -> None: ...


@runtime_checkable
class RiskGate(Protocol):
    """Pre-trade risk check that can block or modify orders."""
    def check(self, symbol: str, side: str, size_usd: float,
              portfolio_value: float) -> Dict[str, Any]: ...


@runtime_checkable
class SignalSource(Protocol):
    """Produces trading signals from market data."""
    def update(self, prices: Dict[str, float], regime: str) -> Optional[Dict[str, Any]]: ...


@runtime_checkable
class PositionTracker(Protocol):
    """Tracks position state."""
    def update(self, exchange: str, symbol: str, side: str,
               quantity: float, price: float) -> None: ...


@runtime_checkable
class ComplianceRecorder(Protocol):
    """Records trades for regulatory compliance."""
    def record_transaction(self, tx: Any) -> None: ...


@runtime_checkable
class AlertChannel(Protocol):
    """Sends alerts/notifications."""
    @property
    def is_configured(self) -> bool: ...
    def send_alert(self, message: str, level: str = "info") -> None: ...


@runtime_checkable
class ModelManager(Protocol):
    """Manages ML model lifecycle."""
    @property
    def registry(self) -> Dict[str, Any]: ...
    def performance_check(self) -> Dict[str, Any]: ...


@runtime_checkable
class EnsembleHub(Protocol):
    """Aggregates signals from multiple sources."""
    def compute(self) -> Dict[str, Any]: ...
    def update_source_weights(self, weights: Dict[str, float]) -> None: ...


@runtime_checkable
class StrategyRouter(Protocol):
    """Routes signals to appropriate strategies."""
    def get_active_strategies(self) -> List[str]: ...
    def get_strategy_stats(self) -> Dict[str, Any]: ...
    def disable(self, strategy: str) -> None: ...
    def enable(self, strategy: str) -> None: ...


@runtime_checkable
class ExchangeConnector(Protocol):
    """Exchange API abstraction."""
    async def execute_order(self, order: Dict[str, Any]) -> Optional[Dict[str, Any]]: ...
    async def get_balances(self) -> Optional[Dict[str, float]]: ...
    async def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]: ...


@runtime_checkable
class StateStore(Protocol):
    """Persists state across restarts."""
    def get_account_value(self, key: str) -> Optional[Any]: ...
    def set_account_value(self, key: str, value: Any) -> None: ...
    def get_positions(self) -> Dict[str, Dict]: ...


@runtime_checkable
class WriteQueue(Protocol):
    """Async write queue for batching database writes."""
    def enqueue(self, table: str, data: Dict[str, Any]) -> None: ...
    async def flush(self) -> int: ...
