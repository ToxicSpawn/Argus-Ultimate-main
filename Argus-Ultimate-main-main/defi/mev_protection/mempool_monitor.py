"""Mempool monitoring for MEV detection across Ethereum and L2s."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

if importlib.util.find_spec("websockets") is not None:  # pragma: no branch
    websockets = importlib.import_module("websockets")
    _websockets_available = True
else:  # pragma: no cover - optional dependency
    websockets = None
    _websockets_available = False
    logger.warning("websockets not installed. Live mempool monitoring disabled.")


SWAP_METHOD_SELECTORS = {
    "0x38ed1739": "swapExactTokensForTokens",
    "0x18cbafe5": "swapExactETHForTokens",
    "0x7ff36ab5": "swapExactETHForTokensSupportingFeeOnTransferTokens",
    "0x4a25d94a": "swapTokensForExactETH",
    "0x5ae401dc": "multicall",
    "0x414bf389": "exactInputSingle",
    "0xb858183f": "exactInput",
    "0x04e45aaf": "exactOutputSingle",
}


@dataclass
class MempoolTransaction:
    tx_hash: str
    chain: str = "ethereum"
    from_address: str = ""
    to_address: str = ""
    nonce: int = 0
    value: float = 0.0
    gas_limit: int = 0
    gas_price_gwei: float = 0.0
    max_fee_per_gas_gwei: float = 0.0
    max_priority_fee_gwei: float = 0.0
    amount_in_usd: float = 0.0
    amount_out_min_usd: float = 0.0
    method_id: str = ""
    method_name: str = "unknown"
    token_in: str | None = None
    token_out: str | None = None
    dex: str | None = None
    timestamp: float = field(default_factory=time.time)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_gas_gwei(self) -> float:
        if self.max_fee_per_gas_gwei > 0:
            if self.max_priority_fee_gwei > 0:
                return min(self.max_fee_per_gas_gwei, self.gas_price_gwei + self.max_priority_fee_gwei)
            return self.max_fee_per_gas_gwei
        return self.gas_price_gwei

    @property
    def is_swap(self) -> bool:
        return self.method_id in SWAP_METHOD_SELECTORS or "swap" in self.method_name.lower()

    @property
    def is_large_swap(self) -> bool:
        return self.amount_in_usd >= 100_000


class MempoolMonitor:
    """Monitor pending transactions, gas conditions, and large swap flow."""

    def __init__(
        self,
        websocket_url: str = "",
        chain: str = "ethereum",
        large_swap_threshold_usd: float = 100_000.0,
        max_transactions: int = 5000,
    ) -> None:
        self.websocket_url: str = websocket_url
        self.chain: str = chain.lower()
        self.large_swap_threshold_usd: float = large_swap_threshold_usd
        self._transactions: deque[MempoolTransaction] = deque(maxlen=max_transactions)
        self._callbacks: list[Callable[[MempoolTransaction], Awaitable[None]]] = []
        self._ws: Any | None = None
        self._subscription_id: str | None = None
        self._connected: bool = False
        self._last_heartbeat: float = 0.0

    def add_callback(self, callback: Callable[[MempoolTransaction], Awaitable[None]]) -> None:
        self._callbacks.append(callback)

    @property
    def transactions(self) -> list[MempoolTransaction]:
        return list(self._transactions)

    async def connect(self) -> None:
        if not self.websocket_url:
            logger.warning("No websocket URL configured for mempool monitor")
            return
        if not _websockets_available or websockets is None:
            logger.warning("websockets dependency unavailable; cannot connect to mempool")
            return
        try:
            self._ws = await websockets.connect(self.websocket_url, ping_interval=20, ping_timeout=20)
            self._connected = True
            logger.info("Connected to %s mempool websocket", self.chain)
            await self._subscribe_pending_transactions()
        except Exception as exc:
            self._connected = False
            logger.error("Failed to connect to mempool websocket: %s", exc)

    async def disconnect(self) -> None:
        if self._ws is not None:
            await self._ws.close()
        self._ws = None
        self._connected = False

    async def _subscribe_pending_transactions(self) -> None:
        if self._ws is None:
            return
        payload = {
            "id": 1,
            "method": "eth_subscribe",
            "params": ["newPendingTransactions", True],
            "jsonrpc": "2.0",
        }
        await self._ws.send(json.dumps(payload))

    async def run_forever(self) -> None:
        while True:
            if not self._connected:
                await self.connect()
                if not self._connected:
                    await asyncio.sleep(5)
                    continue
            try:
                assert self._ws is not None
                message = await self._ws.recv()
                self._last_heartbeat = time.time()
                await self._handle_message(message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Mempool stream error: %s", exc)
                await self.disconnect()
                await asyncio.sleep(2)

    async def _handle_message(self, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            logger.debug("Ignoring non-JSON websocket message")
            return
        if payload.get("id") == 1 and payload.get("result"):
            self._subscription_id = payload["result"]
            logger.info("Subscribed to pending transactions: %s", self._subscription_id)
            return

        tx_payload = payload.get("params", {}).get("result")
        if isinstance(tx_payload, dict):
            tx = self.parse_transaction(tx_payload)
            if tx is not None:
                await self.ingest_transaction(tx)

    def parse_transaction(self, tx_payload: dict[str, Any]) -> MempoolTransaction | None:
        try:
            input_data = tx_payload.get("input", "") or ""
            method_id = input_data[:10] if len(input_data) >= 10 else ""
            gas_price_wei = self._hex_to_int(tx_payload.get("gasPrice"))
            max_fee_wei = self._hex_to_int(tx_payload.get("maxFeePerGas"))
            max_priority_wei = self._hex_to_int(tx_payload.get("maxPriorityFeePerGas"))
            gas_limit = self._hex_to_int(tx_payload.get("gas"))
            value_wei = self._hex_to_int(tx_payload.get("value"))
            nonce = self._hex_to_int(tx_payload.get("nonce"))
            amount_in_usd = self._estimate_usd_value(value_wei / 1e18, tx_payload)

            return MempoolTransaction(
                tx_hash=tx_payload.get("hash", ""),
                chain=self.chain,
                from_address=tx_payload.get("from", ""),
                to_address=tx_payload.get("to", ""),
                nonce=nonce,
                value=value_wei / 1e18,
                gas_limit=gas_limit,
                gas_price_gwei=gas_price_wei / 1e9,
                max_fee_per_gas_gwei=max_fee_wei / 1e9,
                max_priority_fee_gwei=max_priority_wei / 1e9,
                amount_in_usd=amount_in_usd,
                amount_out_min_usd=max(amount_in_usd * 0.97, 0.0),
                method_id=method_id,
                method_name=SWAP_METHOD_SELECTORS.get(method_id, "unknown"),
                dex=self._infer_dex(tx_payload.get("to", "")),
                raw=tx_payload,
            )
        except Exception as exc:
            logger.debug("Failed to parse mempool transaction: %s", exc)
            return None

    async def ingest_transaction(self, transaction: MempoolTransaction) -> None:
        if transaction.is_swap and transaction.amount_in_usd <= 0:
            transaction.amount_in_usd = self.large_swap_threshold_usd * 0.4
        self._transactions.append(transaction)
        for callback in self._callbacks:
            try:
                await callback(transaction)
            except Exception as exc:
                logger.warning("Mempool callback error: %s", exc)

    def detect_large_swaps(self) -> list[MempoolTransaction]:
        return [
            tx for tx in self._transactions
            if tx.is_swap and tx.amount_in_usd >= self.large_swap_threshold_usd
        ]

    def get_gas_statistics(self) -> dict[str, float]:
        if not self._transactions:
            return {"avg_gas_gwei": 0.0, "avg_priority_fee_gwei": 0.0, "p95_gas_gwei": 0.0}

        effective_gas = sorted(tx.effective_gas_gwei for tx in self._transactions if tx.effective_gas_gwei > 0)
        priority_fees = [tx.max_priority_fee_gwei for tx in self._transactions if tx.max_priority_fee_gwei > 0]
        if not effective_gas:
            effective_gas = [0.0]
        p95_index = min(len(effective_gas) - 1, int(len(effective_gas) * 0.95))
        return {
            "avg_gas_gwei": mean(effective_gas),
            "avg_priority_fee_gwei": mean(priority_fees) if priority_fees else 0.0,
            "p95_gas_gwei": effective_gas[p95_index],
        }

    def opportunities_seed(self) -> list[MempoolTransaction]:
        return sorted(self.detect_large_swaps(), key=lambda tx: tx.amount_in_usd, reverse=True)

    def _estimate_usd_value(self, native_amount: float, tx_payload: dict[str, Any]) -> float:
        native_price = {"ethereum": 3200.0, "arbitrum": 3200.0, "optimism": 3200.0}.get(self.chain, 3000.0)
        if native_amount > 0:
            return native_amount * native_price
        if (tx_payload.get("input") or "")[:10] in SWAP_METHOD_SELECTORS:
            return self.large_swap_threshold_usd * 0.4
        return 0.0

    def _infer_dex(self, to_address: str) -> str | None:
        address = (to_address or "").lower()
        known = {
            "0x7a250d56": "uniswap_v2",
            "0xe592427a": "uniswap_v3",
            "0x11111112": "1inch",
            "0xdef1c0de": "cow_swap",
        }
        for prefix, dex in known.items():
            if address.startswith(prefix):
                return dex
        return None

    @staticmethod
    def _hex_to_int(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            if value.startswith("0x"):
                return int(value, 16)
            if value.isdigit():
                return int(value)
        return 0
