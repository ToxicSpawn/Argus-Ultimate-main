"""Whale transaction tracker for on-chain analysis.

Tracks large cryptocurrency transactions to detect whale activity.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WhaleTransaction:
    """Large cryptocurrency transaction.
    
    Attributes
    ----------
    tx_hash : str
        Transaction hash
    symbol : str
        Cryptocurrency symbol (e.g., "ETH", "BTC")
    amount : float
        Transaction amount
    amount_usd : float
        Transaction value in USD
    from_address : str
        Sender address
    to_address : str
        Recipient address
    timestamp : float
        Unix timestamp
    direction : str
        "exchange_in", "exchange_out", or "transfer"
    """
    tx_hash: str = ""
    symbol: str = ""
    amount: float = 0.0
    amount_usd: float = 0.0
    from_address: str = ""
    to_address: str = ""
    timestamp: float = field(default_factory=time.time)
    direction: str = "transfer"


class WhaleTracker:
    """Generic whale transaction tracker.
    
    Parameters
    ----------
    min_value_usd : float
        Minimum transaction value to track (default 100000)
    """
    
    def __init__(self, min_value_usd: float = 100_000.0) -> None:
        self._min_value_usd = min_value_usd
        self._transactions: List[WhaleTransaction] = []
    
    def add_transaction(self, tx: WhaleTransaction) -> bool:
        """Add a transaction if it meets the minimum value threshold.
        
        Returns True if added, False if below threshold.
        """
        if tx.amount_usd >= self._min_value_usd:
            self._transactions.append(tx)
            # Keep only last 10000 transactions
            if len(self._transactions) > 10000:
                self._transactions = self._transactions[-10000:]
            return True
        return False
    
    def get_recent_transactions(
        self,
        symbol: Optional[str] = None,
        hours: float = 24.0,
    ) -> List[WhaleTransaction]:
        """Get recent whale transactions."""
        now = time.time()
        cutoff = now - (hours * 3600)
        txs = [t for t in self._transactions if t.timestamp >= cutoff]
        if symbol:
            txs = [t for t in txs if t.symbol == symbol]
        return txs
    
    def get_total_volume(
        self,
        symbol: Optional[str] = None,
        hours: float = 24.0,
    ) -> float:
        """Get total transaction volume in USD."""
        txs = self.get_recent_transactions(symbol, hours)
        return sum(t.amount_usd for t in txs)
