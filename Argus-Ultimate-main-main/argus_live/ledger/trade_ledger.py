from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Literal

LedgerRecordStatus = Literal["submitted", "filled", "rejected", "cancelled", "reconciled", "attributed"]


@dataclass(frozen=True)
class LedgerFillRecord:
    intent_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    ts: str = ""
    strategy_id: str = "unknown"
    venue: str = "unknown"
    requested_qty: float = 0.0
    approved_qty: float = 0.0
    fill_qty: float = 0.0
    limit_price: float = 0.0
    expected_price: float = 0.0
    fill_price: float = 0.0
    fees: float = 0.0
    maker_taker: str = "maker"
    latency_ms: float = 0.0
    slippage_bps: float = 0.0
    execution_alpha_bps: float = 0.0
    adverse_price_move_bps: float = 0.0
    reject_flag: int = 0
    reject_reason: str = ""
    status: LedgerRecordStatus = "filled"
    venue_order_id: str = ""
    parent_intent_id: str = ""
    child_order_id: str = ""
    manifest_hash: str = ""
    config_hash: str = ""
    run_id: str = ""
    ladder_stage: str = "PAPER"
    gross_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    net_pnl_value: float = 0.0

    def normalized(self) -> "LedgerFillRecord":
        ts = self.ts or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        fill_qty = self.fill_qty if self.fill_qty > 0 else self.quantity
        fill_price = self.fill_price or self.price
        approved_qty = self.approved_qty if self.approved_qty > 0 else max(self.quantity, fill_qty)
        requested_qty = self.requested_qty if self.requested_qty > 0 else approved_qty
        limit_price = self.limit_price or self.price
        expected_price = self.expected_price or limit_price
        config_hash = self.config_hash or self.manifest_hash
        run_id = self.run_id or self.manifest_hash or ""
        net_pnl_value = self.net_pnl_value
        if net_pnl_value == 0.0 and (self.realized_pnl != 0.0 or self.unrealized_pnl != 0.0 or self.fees != 0.0):
            net_pnl_value = float(self.realized_pnl) + float(self.unrealized_pnl) - float(self.fees)
        return LedgerFillRecord(
            intent_id=self.intent_id,
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            price=self.price,
            ts=ts,
            strategy_id=self.strategy_id,
            venue=self.venue,
            requested_qty=requested_qty,
            approved_qty=approved_qty,
            fill_qty=fill_qty,
            limit_price=limit_price,
            expected_price=expected_price,
            fill_price=fill_price,
            fees=self.fees,
            maker_taker=self.maker_taker,
            latency_ms=self.latency_ms,
            slippage_bps=self.slippage_bps,
            execution_alpha_bps=self.execution_alpha_bps,
            adverse_price_move_bps=self.adverse_price_move_bps,
            reject_flag=self.reject_flag,
            reject_reason=self.reject_reason,
            status=self.status,
            venue_order_id=self.venue_order_id,
            parent_intent_id=self.parent_intent_id,
            child_order_id=self.child_order_id,
            manifest_hash=self.manifest_hash,
            config_hash=config_hash,
            run_id=run_id,
            ladder_stage=self.ladder_stage,
            gross_pnl=self.gross_pnl,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            net_pnl_value=net_pnl_value,
        )

    @property
    def net_pnl(self) -> float:
        return float(self.net_pnl_value)


class TradeLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_fill(self, record: LedgerFillRecord) -> None:
        normalized = record.normalized()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(normalized), sort_keys=True) + "\n")

    def append_event(self, record: LedgerFillRecord) -> None:
        self.append_fill(record)

    def load_recent_fills(self, limit: int = 100) -> list[LedgerFillRecord]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()][-limit:]
        rows: list[LedgerFillRecord] = []
        for line in lines:
            raw = json.loads(line)
            rows.append(LedgerFillRecord(**raw).normalized())
        return rows
