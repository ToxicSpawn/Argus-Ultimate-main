from __future__ import annotations

from typing import Any

from argus_live.execution.ccxt_coinbase_adapter import CcxtCoinbaseAdapter


def build_coinbase_advanced_adapter(exchange_client: Any, dry_run: bool = True) -> CcxtCoinbaseAdapter:
    return CcxtCoinbaseAdapter(exchange_client=exchange_client, dry_run=dry_run)
