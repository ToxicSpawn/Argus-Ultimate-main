"""Flashbots-style bundle submission with retry and inclusion tracking."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

if importlib.util.find_spec("aiohttp") is not None:  # pragma: no branch
    aiohttp = importlib.import_module("aiohttp")
    _aiohttp_available = True
else:  # pragma: no cover - optional dependency
    aiohttp = None
    _aiohttp_available = False
    logger.warning("aiohttp not installed. Flashbots relay submission disabled.")


@dataclass
class BundleTransaction:
    signed_transaction: str
    can_revert: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BundleSubmissionResult:
    success: bool
    bundle_hash: str
    relay_url: str
    target_block: int
    included: bool = False
    attempts: int = 0
    error: str = ""
    response: dict[str, Any] = field(default_factory=dict)


class FlashbotsIntegration:
    """Submit MEV bundles to Flashbots or compatible private relays."""

    def __init__(
        self,
        relay_url: str = "https://relay.flashbots.net",
        chain: str = "ethereum",
        max_retries: int = 3,
        retry_delay: float = 1.5,
    ) -> None:
        self.relay_url: str = relay_url
        self.chain: str = chain.lower()
        self.max_retries: int = max_retries
        self.retry_delay: float = retry_delay
        self._last_inclusion_events: list[dict[str, Any]] = []

    def create_bundle(self, signed_transactions: list[str], allow_reverts: bool = False) -> list[BundleTransaction]:
        return [
            BundleTransaction(signed_transaction=tx, can_revert=allow_reverts)
            for tx in signed_transactions
        ]

    async def submit_bundle(
        self,
        bundle: list[BundleTransaction],
        target_block: int,
        replacement_uuid: str | None = None,
    ) -> BundleSubmissionResult:
        if not bundle:
            return BundleSubmissionResult(False, "", self.relay_url, target_block, error="empty bundle")
        if not _aiohttp_available or aiohttp is None:
            return BundleSubmissionResult(False, "", self.relay_url, target_block, error="aiohttp unavailable")

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_sendBundle",
            "params": [{
                "txs": [tx.signed_transaction for tx in bundle],
                "blockNumber": hex(target_block),
                **({"replacementUuid": replacement_uuid} if replacement_uuid else {}),
            }],
        }

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.relay_url, json=payload, timeout=10) as response:
                        data = await response.json(content_type=None)
                        bundle_hash = data.get("result", {}).get("bundleHash", "") if isinstance(data.get("result"), dict) else ""
                        success = response.status < 400 and "error" not in data
                        result = BundleSubmissionResult(
                            success=success,
                            bundle_hash=bundle_hash,
                            relay_url=self.relay_url,
                            target_block=target_block,
                            attempts=attempt,
                            error=str(data.get("error", "")) if not success else "",
                            response=data,
                        )
                        if success:
                            logger.info("Bundle submitted to %s for block %s", self.relay_url, target_block)
                            return result
                        last_error = result.error or f"HTTP {response.status}"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = str(exc)
                logger.warning("Bundle submission attempt %d failed: %s", attempt, exc)

            await asyncio.sleep(self.retry_delay * attempt)

        return BundleSubmissionResult(
            success=False,
            bundle_hash="",
            relay_url=self.relay_url,
            target_block=target_block,
            attempts=self.max_retries,
            error=last_error or "submission failed",
        )

    async def wait_for_inclusion(
        self,
        bundle_hash: str,
        target_block: int,
        max_wait_seconds: float = 20.0,
    ) -> bool:
        deadline = time.time() + max_wait_seconds
        while time.time() < deadline:
            for event in self._last_inclusion_events:
                if event.get("bundle_hash") == bundle_hash and event.get("block") == target_block:
                    return bool(event.get("included"))
            await asyncio.sleep(1.0)
        return False

    def handle_bundle_inclusion_event(self, event: dict[str, Any]) -> None:
        self._last_inclusion_events.append(dict(event))
        self._last_inclusion_events = self._last_inclusion_events[-100:]
