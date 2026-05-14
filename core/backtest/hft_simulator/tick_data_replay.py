"""Tick-level replay and L3 reconstruction utilities."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Iterator, Mapping
from typing import cast

from core.backtest.hft_simulator.order_book_l3 import L3Order, L3OrderBook, Side

logger = logging.getLogger(__name__)


@dataclass
class TickMessage:
    timestamp_ns: int
    message_type: str
    order_id: str
    side: Side
    price: float
    quantity: float
    symbol: str
    sequence: int = 0


class TickDataReplay:
    """Replays normalised ITCH/OUCH style event streams."""

    def __init__(self, symbol: str, timestamp_offset_ns: int = 0) -> None:
        self.symbol = symbol
        self.timestamp_offset_ns = timestamp_offset_ns

    def load_messages(self, path: str | Path) -> list[TickMessage]:
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix == ".jsonl":
            messages = list(self._load_jsonl(file_path))
        else:
            messages = list(self._load_csv(file_path))
        return self.align_timestamps(self.validate_messages(messages))

    def parse_message(self, raw: Mapping[str, object]) -> TickMessage:
        message_type = str(raw.get("message_type") or raw.get("type") or "").lower()
        if message_type not in {"add", "cancel", "execute", "trade", "replace"}:
            raise ValueError(f"unsupported message_type={message_type}")
        side = cast(Side, str(raw.get("side") or "buy").lower())
        return TickMessage(
            timestamp_ns=self._as_int(raw.get("timestamp_ns") or raw.get("ts") or 0),
            message_type=message_type,
            order_id=str(raw.get("order_id") or raw.get("id") or ""),
            side=side,
            price=self._as_float(raw.get("price") or 0.0),
            quantity=self._as_float(raw.get("quantity") or raw.get("size") or 0.0),
            symbol=str(raw.get("symbol") or self.symbol),
            sequence=self._as_int(raw.get("sequence") or 0),
        )

    def reconstruct_book(self, messages: Iterable[TickMessage]) -> L3OrderBook:
        book = L3OrderBook(symbol=self.symbol)
        for message in sorted(messages, key=lambda item: (item.timestamp_ns, item.sequence)):
            self.apply_message(book, message)
        return book

    def apply_message(self, book: L3OrderBook, message: TickMessage) -> None:
        if message.message_type == "add":
            book.add_order(
                L3Order(
                    order_id=message.order_id,
                    side=message.side,
                    price=message.price,
                    quantity=message.quantity,
                    timestamp_ns=message.timestamp_ns,
                )
            )
        elif message.message_type == "cancel":
            book.cancel_order(message.order_id, message.quantity or None)
        elif message.message_type in {"execute", "trade"}:
            book.execute_order(message.order_id, message.quantity)
        elif message.message_type == "replace":
            order = book.order_index.get(message.order_id)
            if order is None:
                logger.debug("Replace ignored for missing order_id=%s", message.order_id)
                return
            remaining = order.remaining_quantity
            book.cancel_order(message.order_id)
            book.add_order(
                L3Order(
                    order_id=message.order_id,
                    side=cast(Side, message.side or order.side),
                    price=message.price or order.price,
                    quantity=message.quantity or remaining,
                    timestamp_ns=message.timestamp_ns,
                    owner=order.owner,
                )
            )

    def align_timestamps(self, messages: list[TickMessage]) -> list[TickMessage]:
        if not messages:
            return messages
        floor = min(message.timestamp_ns for message in messages)
        aligned = []
        for message in messages:
            aligned.append(
                TickMessage(
                    timestamp_ns=(message.timestamp_ns - floor + self.timestamp_offset_ns),
                    message_type=message.message_type,
                    order_id=message.order_id,
                    side=message.side,
                    price=message.price,
                    quantity=message.quantity,
                    symbol=message.symbol,
                    sequence=message.sequence,
                )
            )
        return aligned

    def validate_messages(self, messages: list[TickMessage]) -> list[TickMessage]:
        validated: list[TickMessage] = []
        for message in messages:
            if not message.order_id:
                raise ValueError("tick message missing order_id")
            if message.side not in {"buy", "sell"}:
                raise ValueError(f"invalid side={message.side}")
            if message.quantity < 0 or message.price < 0:
                raise ValueError("message price/quantity must be non-negative")
            validated.append(message)
        return sorted(validated, key=lambda item: (item.timestamp_ns, item.sequence))

    def _load_csv(self, path: Path) -> Iterator[TickMessage]:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield self.parse_message(row)

    def _load_jsonl(self, path: Path) -> Iterator[TickMessage]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield self.parse_message(json.loads(line))

    @staticmethod
    def _as_int(value: object) -> int:
        return int(cast(int | float | str, value))

    @staticmethod
    def _as_float(value: object) -> float:
        return float(cast(int | float | str, value))
