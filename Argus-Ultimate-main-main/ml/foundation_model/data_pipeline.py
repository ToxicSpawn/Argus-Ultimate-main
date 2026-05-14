"""Training-data preparation for TradeFM-style microstructure modelling."""

# pyright: reportMissingImports=false

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .tokenizer import UniversalMicrostructureTokenizer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TickEvent:
    """Tick-level trade or order-book event."""

    symbol: str
    timestamp: float
    price: float
    size: float
    bid: float
    ask: float
    order_type: str
    side: str = "buy"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "price": self.price,
            "size": self.size,
            "bid": self.bid,
            "ask": self.ask,
            "mid_price": (self.bid + self.ask) / 2.0 if self.ask >= self.bid else self.price,
            "spread": max(0.0, self.ask - self.bid),
            "order_type": self.order_type,
            "side": self.side,
        }
        payload.update(self.metadata)
        return payload


class MarketMicrostructureDataset(Dataset[Dict[str, torch.Tensor]]):
    """Torch dataset backed by tokenised event sequences."""

    def __init__(self, sequences: Sequence[Sequence[int]], asset_ids: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        self.sequences = [torch.tensor(sequence, dtype=torch.long) for sequence in sequences if len(sequence) > 1]
        self.asset_ids = list(asset_ids) if asset_ids is not None else [0] * len(self.sequences)
        if len(self.asset_ids) != len(self.sequences):
            raise ValueError("asset_ids length must match sequences length")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[index]
        return {
            "input_ids": sequence,
            "attention_mask": torch.ones_like(sequence),
            "asset_ids": torch.tensor(self.asset_ids[index], dtype=torch.long),
        }


class TradeFMDataPipeline:
    """Loads tick data, constructs event sequences, and prepares train splits."""

    def __init__(self, tokenizer: Optional[UniversalMicrostructureTokenizer] = None) -> None:
        self.tokenizer = tokenizer or UniversalMicrostructureTokenizer()

    def load_tick_data(self, path: str | Path) -> List[TickEvent]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Tick data not found: {file_path}")
        events: List[TickEvent] = []
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    events.append(
                        TickEvent(
                            symbol=str(row.get("symbol", "UNKNOWN")),
                            timestamp=float(row.get("timestamp", 0.0)),
                            price=float(row.get("price", 0.0)),
                            size=float(row.get("size", 0.0)),
                            bid=float(row.get("bid", row.get("best_bid", row.get("price", 0.0)))),
                            ask=float(row.get("ask", row.get("best_ask", row.get("price", 0.0)))),
                            order_type=str(row.get("order_type", "market")),
                            side=str(row.get("side", "buy")),
                            metadata={key: value for key, value in row.items() if key not in {"symbol", "timestamp", "price", "size", "bid", "ask", "best_bid", "best_ask", "order_type", "side"}},
                        )
                    )
                except Exception:
                    logger.exception("Failed to parse tick row: %s", row)
        return events

    def create_event_sequences(self, events: Sequence[TickEvent], *, seq_len: int = 128, stride: int = 32) -> List[List[Dict[str, Any]]]:
        serialized = [event.as_dict() for event in events]
        sequences: List[List[Dict[str, Any]]] = []
        for start in range(0, max(0, len(serialized) - seq_len + 1), max(1, stride)):
            sequences.append(serialized[start : start + seq_len])
        return sequences

    def augment_sequences(self, sequences: Sequence[Sequence[Mapping[str, Any]]], *, size_noise_std: float = 0.05, time_jitter: float = 0.05) -> List[List[Dict[str, Any]]]:
        augmented: List[List[Dict[str, Any]]] = []
        for sequence in sequences:
            transformed: List[Dict[str, Any]] = []
            for event in sequence:
                cloned = dict(event)
                cloned["size"] = max(0.0, float(cloned.get("size", 0.0)) * (1.0 + torch.randn(1).item() * size_noise_std))
                cloned["timestamp"] = float(cloned.get("timestamp", 0.0)) + torch.randn(1).item() * time_jitter
                transformed.append(cloned)
            augmented.append(transformed)
        return augmented

    def train_val_test_split(
        self,
        sequences: Sequence[Any],
        *,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Dict[str, List[Any]]:
        total = len(sequences)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        return {
            "train": list(sequences[:train_end]),
            "val": list(sequences[train_end:val_end]),
            "test": list(sequences[val_end:]),
        }

    def tokenise_sequences(self, sequences: Sequence[Sequence[Mapping[str, Any]]]) -> List[List[int]]:
        return [self.tokenizer.to_token_ids(sequence) for sequence in sequences]

    def prepare_dataset(self, sequences: Sequence[Sequence[Mapping[str, Any]]], *, asset_ids: Optional[Sequence[int]] = None) -> MarketMicrostructureDataset:
        tokenised = self.tokenise_sequences(sequences)
        return MarketMicrostructureDataset(tokenised, asset_ids=asset_ids)
