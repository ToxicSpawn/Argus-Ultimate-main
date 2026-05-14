from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_cache_value(value: Any) -> dict:
    return {
        "result": deepcopy(value),
        "cached_at": _utc_now().isoformat(),
    }


@dataclass(slots=True)
class Command:
    command_id: str = field(default_factory=lambda: str(uuid4()))
    command_type: str = ""
    timestamp: datetime = field(default_factory=_utc_now)
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None


@dataclass(slots=True)
class Query:
    query_id: str = field(default_factory=lambda: str(uuid4()))
    query_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


class ReadModel:
    """Read-optimized projection store for portfolio and trading queries."""

    def __init__(self) -> None:
        self._positions: Dict[str, dict] = {}
        self._portfolio_snapshot: dict = {}
        self._pnl_calculations: dict = {}
        self._risk_metrics: dict = {}
        self._trade_history: List[dict] = []
        self._query_cache: Dict[str, dict] = {}
        self._event_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._lock = asyncio.Lock()

    async def publish_event(self, event_type: str, payload: dict) -> None:
        """Queue an event for asynchronous projection updates."""
        await self._event_queue.put({"event_type": event_type, "payload": deepcopy(payload)})

    async def process_pending_events(self) -> None:
        """Drain queued events and update read-side projections."""
        while not self._event_queue.empty():
            event = await self._event_queue.get()
            try:
                await self._apply_event(event["event_type"], event["payload"])
            finally:
                self._event_queue.task_done()

    async def _apply_event(self, event_type: str, payload: dict) -> None:
        async with self._lock:
            if event_type in {"order_placed", "position_modified"}:
                symbol = str(payload.get("symbol", "")).upper()
                if symbol:
                    self.update_position(symbol, payload)
            elif event_type == "order_cancelled":
                order_id = payload.get("order_id")
                for symbol, position in list(self._positions.items()):
                    if position.get("order_id") == order_id:
                        updated = deepcopy(position)
                        updated["status"] = "cancelled"
                        self.update_position(symbol, updated)
                        break
            elif event_type == "risk_limit_updated":
                self._risk_metrics.update(deepcopy(payload))

            portfolio_snapshot = payload.get("portfolio_snapshot")
            if isinstance(portfolio_snapshot, dict):
                self.update_portfolio_snapshot(portfolio_snapshot)

            pnl_data = payload.get("pnl_data")
            if isinstance(pnl_data, dict):
                self.update_pnl_calculations(pnl_data)

            trade_record = payload.get("trade_record")
            if isinstance(trade_record, dict):
                self._trade_history.append(deepcopy(trade_record))

            self._invalidate_cache()

    def update_position(self, symbol: str, data: dict) -> None:
        self._positions[symbol.upper()] = deepcopy(data)

    def update_portfolio_snapshot(self, snapshot: dict) -> None:
        self._portfolio_snapshot = deepcopy(snapshot)

    def update_pnl_calculations(self, pnl_data: dict) -> None:
        self._pnl_calculations = deepcopy(pnl_data)

    def get_cached_query(self, query_type: str, params: dict) -> Optional[dict]:
        cache_key = self._cache_key(query_type, params)
        cached = self._query_cache.get(cache_key)
        return deepcopy(cached) if cached is not None else None

    def cache_query_result(self, query_type: str, params: dict, result: Any) -> None:
        cache_key = self._cache_key(query_type, params)
        self._query_cache[cache_key] = _normalize_cache_value(result)

    def portfolio_snapshot(self) -> dict:
        return deepcopy(self._portfolio_snapshot)

    def positions(self) -> List[dict]:
        return [deepcopy(position) for position in self._positions.values()]

    def pnl_calculations(self) -> dict:
        return deepcopy(self._pnl_calculations)

    def risk_metrics(self) -> dict:
        return deepcopy(self._risk_metrics)

    def trade_history(self) -> List[dict]:
        return [deepcopy(trade) for trade in self._trade_history]

    def _invalidate_cache(self) -> None:
        self._query_cache.clear()

    @staticmethod
    def _cache_key(query_type: str, params: dict) -> str:
        normalized_items = tuple(sorted((str(key), repr(value)) for key, value in params.items()))
        return f"{query_type}:{normalized_items!r}"


class CommandHandler:
    """Write-side handler for trading commands with validation and audit logs."""

    _SUPPORTED_COMMANDS = {
        "place_order",
        "cancel_order",
        "modify_position",
        "update_risk_limit",
    }

    def __init__(self, read_model: Optional[ReadModel] = None) -> None:
        self.read_model = read_model or ReadModel()
        self.audit_log: List[dict] = []

    def handle_place_order(self, command: Command) -> dict:
        self._ensure_valid(command)
        payload = deepcopy(command.payload)
        payload.setdefault("status", "accepted")
        payload.setdefault("placed_at", command.timestamp.isoformat())
        payload.setdefault("command_id", command.command_id)

        symbol = str(payload.get("symbol", "")).upper()
        event_payload = {
            **payload,
            "trade_record": {
                "command_id": command.command_id,
                "correlation_id": command.correlation_id,
                "event": "order_placed",
                "symbol": symbol,
                "side": payload.get("side"),
                "quantity": payload.get("quantity"),
                "price": payload.get("price"),
                "timestamp": command.timestamp.isoformat(),
            },
        }
        self._publish_event("order_placed", event_payload)

        result = {
            "status": "accepted",
            "command_id": command.command_id,
            "command_type": command.command_type,
            "symbol": symbol,
            "correlation_id": command.correlation_id,
        }
        self._audit(command, result)
        return result

    def handle_cancel_order(self, command: Command) -> dict:
        self._ensure_valid(command)
        payload = deepcopy(command.payload)
        payload["status"] = "cancelled"
        payload["cancelled_at"] = command.timestamp.isoformat()
        self._publish_event("order_cancelled", payload)

        result = {
            "status": "cancelled",
            "command_id": command.command_id,
            "order_id": payload.get("order_id"),
            "correlation_id": command.correlation_id,
        }
        self._audit(command, result)
        return result

    def handle_modify_position(self, command: Command) -> dict:
        self._ensure_valid(command)
        payload = deepcopy(command.payload)
        symbol = str(payload.get("symbol", "")).upper()
        payload.setdefault("modified_at", command.timestamp.isoformat())

        self._publish_event("position_modified", payload)
        result = {
            "status": "updated",
            "command_id": command.command_id,
            "symbol": symbol,
            "changes": deepcopy(payload),
            "correlation_id": command.correlation_id,
        }
        self._audit(command, result)
        return result

    def handle_update_risk_limit(self, command: Command) -> dict:
        self._ensure_valid(command)
        payload = deepcopy(command.payload)
        payload["updated_at"] = command.timestamp.isoformat()
        self._publish_event("risk_limit_updated", payload)

        result = {
            "status": "updated",
            "command_id": command.command_id,
            "limits": deepcopy(payload),
            "correlation_id": command.correlation_id,
        }
        self._audit(command, result)
        return result

    def validate_command(self, command: Command) -> bool:
        try:
            UUID(str(command.command_id))
        except (TypeError, ValueError, AttributeError):
            logger.warning("Command validation failed: invalid command_id=%r", command.command_id)
            return False

        if command.command_type not in self._SUPPORTED_COMMANDS:
            logger.warning("Command validation failed: unsupported command_type=%s", command.command_type)
            return False

        if not isinstance(command.timestamp, datetime):
            logger.warning("Command validation failed: timestamp must be datetime")
            return False

        if not isinstance(command.payload, dict):
            logger.warning("Command validation failed: payload must be dict")
            return False

        if command.command_type == "place_order":
            return self._validate_required_fields(command, {"symbol", "side", "quantity"})
        if command.command_type == "cancel_order":
            return self._validate_required_fields(command, {"order_id"})
        if command.command_type == "modify_position":
            return self._validate_required_fields(command, {"symbol"})
        if command.command_type == "update_risk_limit":
            return bool(command.payload)

        return True

    def _validate_required_fields(self, command: Command, required_fields: set[str]) -> bool:
        missing = sorted(field for field in required_fields if field not in command.payload)
        if missing:
            logger.warning(
                "Command validation failed: missing payload fields for %s: %s",
                command.command_type,
                ", ".join(missing),
            )
            return False
        return True

    def _ensure_valid(self, command: Command) -> None:
        if not self.validate_command(command):
            raise ValueError(f"Invalid command for type {command.command_type!r}")

    def _publish_event(self, event_type: str, payload: dict) -> None:
        async def _publish_and_process() -> None:
            await self.read_model.publish_event(event_type, payload)
            await self.read_model.process_pending_events()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_publish_and_process())
            return

        loop.create_task(_publish_and_process())

    def _audit(self, command: Command, result: dict) -> None:
        entry = {
            "command_id": command.command_id,
            "command_type": command.command_type,
            "timestamp": command.timestamp.isoformat(),
            "correlation_id": command.correlation_id,
            "payload": deepcopy(command.payload),
            "result": deepcopy(result),
        }
        self.audit_log.append(entry)
        logger.info(
            "CQRS command processed: %s",
            command.command_type,
            extra={
                "extra_data": {
                    "command_id": command.command_id,
                    "correlation_id": command.correlation_id,
                    "status": result.get("status"),
                }
            },
        )


class QueryHandler:
    """Read-side handler backed by a cache-aware read model."""

    def __init__(self, read_model: ReadModel) -> None:
        self.read_model = read_model

    def handle_get_portfolio(self, query: Query) -> dict:
        cached = self.read_model.get_cached_query(query.query_type, query.parameters)
        if cached is not None:
            return cached["result"]

        result = self.read_model.portfolio_snapshot()
        if query.parameters:
            result = {**result, "filters": deepcopy(query.parameters)}
        self.read_model.cache_query_result(query.query_type, query.parameters, result)
        return result

    def handle_get_positions(self, query: Query) -> List[dict]:
        cached = self.read_model.get_cached_query(query.query_type, query.parameters)
        if cached is not None:
            return cached["result"]

        positions = self.read_model.positions()
        symbol_filter = query.parameters.get("symbol")
        if symbol_filter:
            positions = [
                position for position in positions if position.get("symbol") == str(symbol_filter).upper()
            ]
        status_filter = query.parameters.get("status")
        if status_filter:
            positions = [position for position in positions if position.get("status") == status_filter]

        self.read_model.cache_query_result(query.query_type, query.parameters, positions)
        return positions

    def handle_get_pnl(self, query: Query) -> dict:
        cached = self.read_model.get_cached_query(query.query_type, query.parameters)
        if cached is not None:
            return cached["result"]

        result = self.read_model.pnl_calculations()
        self.read_model.cache_query_result(query.query_type, query.parameters, result)
        return result

    def handle_get_risk_metrics(self, query: Query) -> dict:
        cached = self.read_model.get_cached_query(query.query_type, query.parameters)
        if cached is not None:
            return cached["result"]

        result = self.read_model.risk_metrics()
        self.read_model.cache_query_result(query.query_type, query.parameters, result)
        return result

    def handle_get_trade_history(self, query: Query) -> List[dict]:
        cached = self.read_model.get_cached_query(query.query_type, query.parameters)
        if cached is not None:
            return cached["result"]

        trades = self.read_model.trade_history()
        symbol_filter = query.parameters.get("symbol")
        if symbol_filter:
            trades = [trade for trade in trades if trade.get("symbol") == str(symbol_filter).upper()]

        limit = query.parameters.get("limit")
        if isinstance(limit, int) and limit > 0:
            trades = trades[-limit:]

        self.read_model.cache_query_result(query.query_type, query.parameters, trades)
        return trades
