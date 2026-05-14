"""Etherscan whale monitor for Ethereum whale tracking.

Monitors large Ethereum transactions using Etherscan API
to detect whale movements.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from data.onchain.whale_tracker import WhaleTransaction

logger = logging.getLogger(__name__)

ETHERSCAN_API_URL = "https://api.etherscan.io/api"

# Known exchange addresses (simplified list)
EXCHANGE_ADDRESSES = {
    "0x28c6c06298d514db089934071355e5743bf21d60": "binance",
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549": "binance",
    "0x56eeddb30e2ec7457a1e54a5b4e1d8b6b5b0f0f0": "kraken",
    "0x40b38765696e3d5d8d9d834d8aad4bb6e418e489": "kraken",
}


class EtherscanWhaleMonitor:
    """Monitor for Ethereum whale transactions via Etherscan.
    
    Parameters
    ----------
    api_key : str
        Etherscan API key
    min_value_usd : float
        Minimum transaction value to track (default 100000)
    poll_interval_s : float
        Polling interval in seconds (default 300)
    """
    
    def __init__(
        self,
        api_key: str = "",
        min_value_usd: float = 100_000.0,
        poll_interval_s: float = 300.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("ETHERSCAN_API_KEY", "")
        self._min_value_usd = min_value_usd
        self._poll_interval_s = poll_interval_s
        self._last_poll_time: float = 0.0
        self._whales: List[WhaleTransaction] = []
    
    async def poll_recent_whales(
        self,
        max_results: int = 100,
    ) -> List[WhaleTransaction]:
        """Poll for recent whale transactions.
        
        Returns empty list if no API key is set.
        Respects polling interval to avoid rate limits.
        """
        if not self._api_key:
            return []
        
        now = time.time()
        if (now - self._last_poll_time) < self._poll_interval_s:
            return self._whales
        
        try:
            import aiohttp
            
            params = {
                "module": "account",
                "action": "txlist",
                "address": "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance hot wallet
                "startblock": 0,
                "endblock": 99999999,
                "page": 1,
                "offset": max_results,
                "sort": "desc",
                "apikey": self._api_key,
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(ETHERSCAN_API_URL, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("result", [])
                        
                        self._whales = []
                        for tx in results:
                            value_wei = int(tx.get("value", 0))
                            value_eth = value_wei / 1e18
                            # Rough USD value (would need price feed in production)
                            value_usd = value_eth * 3000  # Placeholder
                            
                            if value_usd >= self._min_value_usd:
                                from_addr = tx.get("from", "").lower()
                                to_addr = tx.get("to", "").lower()
                                
                                direction = "transfer"
                                if from_addr in EXCHANGE_ADDRESSES:
                                    direction = "exchange_out"
                                elif to_addr in EXCHANGE_ADDRESSES:
                                    direction = "exchange_in"
                                
                                whale = WhaleTransaction(
                                    tx_hash=tx.get("hash", ""),
                                    symbol="ETH",
                                    amount=value_eth,
                                    amount_usd=value_usd,
                                    from_address=from_addr,
                                    to_address=to_addr,
                                    timestamp=float(tx.get("timeStamp", 0)),
                                    direction=direction,
                                )
                                self._whales.append(whale)
                        
                        self._last_poll_time = now
                        return self._whales
                    else:
                        logger.warning("Etherscan API returned status %d", resp.status)
                        return self._whales
                        
        except Exception as e:
            logger.warning("Failed to poll Etherscan whales: %s", e)
            return self._whales
    
    def get_whale_summary(self) -> Dict[str, Any]:
        """Get summary of recent whale activity."""
        total_in = sum(w.amount_usd for w in self._whales if w.direction == "exchange_in")
        total_out = sum(w.amount_usd for w in self._whales if w.direction == "exchange_out")
        
        return {
            "total_whales": len(self._whales),
            "total_inflow_usd": total_in,
            "total_outflow_usd": total_out,
            "net_flow_usd": total_in - total_out,
            "last_poll_time": self._last_poll_time,
        }
