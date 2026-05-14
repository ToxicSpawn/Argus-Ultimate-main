"""
Whale Tracker — on-chain whale movement detection and analysis.

Monitors large wallet movements to detect:
- Whale accumulation/distribution patterns
- Exchange inflows/outflows
- Smart money movements
- Wallet clustering and labeling
- Transfer patterns and timing

Example::

    tracker = WhaleTracker()
    tracker.update_transfer(
        token="ETH",
        from_address="0x123...",
        to_address="0x456...",
        amount=10000,
        value_usd=20000000,
        tx_type="transfer",
    )
    whale_signal = tracker.get_signal("ETH")
    print(whale_signal.direction, whale_signal.strength)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WhaleWallet:
    """Known whale wallet information."""
    address: str
    label: str  # "exchange", "fund", "whale", "unknown"
    total_holdings_usd: float
    first_seen: float
    last_activity: float
    transfer_count: int
    total_volume_usd: float
    tags: List[str] = field(default_factory=list)


@dataclass
class Transfer:
    """Token transfer record."""
    tx_hash: str
    token: str
    from_address: str
    to_address: str
    amount: float
    value_usd: float
    timestamp: float
    tx_type: str  # "transfer", "exchange_deposit", "exchange_withdrawal", "defi"
    gas_price_gwei: float = 0.0


@dataclass
class WhaleSignal:
    """Whale activity signal."""
    token: str
    timestamp: float
    direction: str  # "accumulation", "distribution", "neutral"
    strength: float  # 0-1
    net_flow_usd: float  # Positive = inflow to exchanges (sell), Negative = outflow (hold)
    large_transfers_count: int
    exchange_inflow_usd: float
    exchange_outflow_usd: float
    whale_count: int
    confidence: float
    reasoning: List[str] = field(default_factory=list)


@dataclass
class _TokenState:
    transfers: Deque[Transfer] = field(
        default_factory=lambda: deque(maxlen=10000)
    )
    whale_wallets: Dict[str, WhaleWallet] = field(default_factory=dict)
    exchange_addresses: Set[str] = field(default_factory=set)
    last_signal: Optional[WhaleSignal] = None
    cumulative_inflow: float = 0.0
    cumulative_outflow: float = 0.0


class WhaleTracker:
    """
    On-chain whale movement tracker.

    Parameters
    ----------
    large_transfer_threshold_usd : float
        Minimum transfer value to consider "large" (default $1M).
    whale_threshold_usd : float
        Minimum holdings to be considered a whale (default $10M).
    signal_window_hours : int
        Hours of data to analyze for signals (default 24).
    exchange_labels : list[str]
        Known exchange labels to track.
    """

    def __init__(
        self,
        large_transfer_threshold_usd: float = 1_000_000,
        whale_threshold_usd: float = 10_000_000,
        signal_window_hours: int = 24,
        exchange_labels: Optional[List[str]] = None,
    ) -> None:
        self._large_threshold = large_transfer_threshold_usd
        self._whale_threshold = whale_threshold_usd
        self._signal_window = signal_window_hours * 3600
        self._exchange_labels = exchange_labels or [
            "binance", "coinbase", "kraken", "huobi", "okx", "bybit",
            "ftx", "gemini", "bitfinex", "kucoin",
        ]
        self._token_states: Dict[str, _TokenState] = {}
        self._known_wallets: Dict[str, WhaleWallet] = {}

        logger.info(
            "WhaleTracker initialized: large_transfer=$%.1fM whale=$%.1fM window=%dh",
            large_transfer_threshold_usd / 1_000_000,
            whale_threshold_usd / 1_000_000,
            signal_window_hours,
        )

    def register_exchange_address(self, address: str, exchange: str) -> None:
        """Register an address as belonging to an exchange."""
        for token_state in self._token_states.values():
            token_state.exchange_addresses.add(address.lower())

    def update_transfer(
        self,
        token: str,
        from_address: str,
        to_address: str,
        amount: float,
        value_usd: float,
        tx_type: str = "transfer",
        tx_hash: str = "",
        gas_price_gwei: float = 0.0,
    ) -> None:
        """Record a token transfer."""
        if token not in self._token_states:
            self._token_states[token] = _TokenState()

        state = self._token_states[token]
        
        transfer = Transfer(
            tx_hash=tx_hash or f"tx_{int(time.time())}_{len(state.transfers)}",
            token=token,
            from_address=from_address.lower(),
            to_address=to_address.lower(),
            amount=amount,
            value_usd=value_usd,
            timestamp=time.time(),
            tx_type=tx_type,
            gas_price_gwei=gas_price_gwei,
        )
        
        state.transfers.append(transfer)

        # Update whale tracking
        self._update_wallet_activity(token, from_address, -value_usd)
        self._update_wallet_activity(token, to_address, value_usd)

        # Track exchange flows
        if from_address.lower() in state.exchange_addresses:
            state.cumulative_outflow += value_usd  # Withdrawal from exchange
        if to_address.lower() in state.exchange_addresses:
            state.cumulative_inflow += value_usd  # Deposit to exchange

        # Generate signal if large transfer
        if value_usd >= self._large_threshold:
            self._analyze_whale_signal(token)

    def _update_wallet_activity(self, token: str, address: str, value_change: float) -> None:
        """Update wallet activity tracking."""
        state = self._token_states[token]
        
        if address not in state.whale_wallets:
            # Check if this is a known wallet
            if address in self._known_wallets:
                wallet = self._known_wallets[address]
            else:
                wallet = WhaleWallet(
                    address=address,
                    label="unknown",
                    total_holdings_usd=0.0,
                    first_seen=time.time(),
                    last_activity=time.time(),
                    transfer_count=0,
                    total_volume_usd=0.0,
                )
                state.whale_wallets[address] = wallet
        
        wallet = state.whale_wallets[address]
        wallet.total_holdings_usd += value_change
        wallet.last_activity = time.time()
        wallet.transfer_count += 1
        wallet.total_volume_usd += abs(value_change)

        # Auto-label if holdings exceed whale threshold
        if wallet.total_holdings_usd >= self._whale_threshold:
            wallet.label = "whale"
            if wallet.address not in self._known_wallets:
                logger.info("New whale detected: %s (%.1fM holdings)", 
                           address[:10], wallet.total_holdings_usd / 1_000_000)

    def _analyze_whale_signal(self, token: str) -> None:
        """Analyze recent whale activity and generate signal."""
        state = self._token_states[token]
        
        # Get recent transfers
        cutoff = time.time() - self._signal_window
        recent_transfers = [t for t in state.transfers if t.timestamp >= cutoff]
        
        if not recent_transfers:
            return

        # Analyze flows
        exchange_inflow = sum(
            t.value_usd for t in recent_transfers
            if t.to_address in state.exchange_addresses
        )
        exchange_outflow = sum(
            t.value_usd for t in recent_transfers
            if t.from_address in state.exchange_addresses
        )

        # Count large transfers
        large_transfers = [t for t in recent_transfers if t.value_usd >= self._large_threshold]
        
        # Count active whales
        active_whales = set()
        for t in recent_transfers:
            if t.value_usd >= self._large_threshold:
                if t.from_address in state.whale_wallets:
                    active_whales.add(t.from_address)
                if t.to_address in state.whale_wallets:
                    active_whales.add(t.to_address)

        # Net flow (positive = selling pressure, negative = accumulation)
        net_flow = exchange_inflow - exchange_outflow

        # Determine direction
        if net_flow > self._large_threshold * 2:
            direction = "distribution"  # Whales depositing to exchanges = selling
        elif net_flow < -self._large_threshold * 2:
            direction = "accumulation"  # Whales withdrawing from exchanges = holding
        else:
            direction = "neutral"

        # Calculate strength
        flow_strength = min(1.0, abs(net_flow) / (self._large_threshold * 10))
        transfer_count_factor = min(1.0, len(large_transfers) / 10)
        whale_factor = min(1.0, len(active_whales) / 5)
        
        strength = (flow_strength * 0.5 + transfer_count_factor * 0.3 + whale_factor * 0.2)

        # Build reasoning
        reasoning = []
        if exchange_inflow > exchange_outflow * 1.5:
            reasoning.append(f"Exchange inflow ${exchange_inflow/1e6:.1f}M > outflow ${exchange_outflow/1e6:.1f}M")
        elif exchange_outflow > exchange_inflow * 1.5:
            reasoning.append(f"Exchange outflow ${exchange_outflow/1e6:.1f}M > inflow ${exchange_inflow/1e6:.1f}M")
        reasoning.append(f"{len(large_transfers)} large transfers (>${self._large_threshold/1e6:.0f}M)")
        reasoning.append(f"{len(active_whales)} active whale wallets")

        signal = WhaleSignal(
            token=token,
            timestamp=time.time(),
            direction=direction,
            strength=strength,
            net_flow_usd=net_flow,
            large_transfers_count=len(large_transfers),
            exchange_inflow_usd=exchange_inflow,
            exchange_outflow_usd=exchange_outflow,
            whale_count=len(active_whales),
            confidence=min(1.0, strength * 1.2),
            reasoning=reasoning,
        )

        state.last_signal = signal

    def get_signal(self, token: str) -> Optional[WhaleSignal]:
        """Get current whale signal for token."""
        if token in self._token_states:
            return self._token_states[token].last_signal
        return None

    def get_whale_wallets(self, token: str) -> List[WhaleWallet]:
        """Get all whale wallets for token."""
        if token in self._token_states:
            return [
                w for w in self._token_states[token].whale_wallets.values()
                if w.label == "whale" or w.total_holdings_usd >= self._whale_threshold
            ]
        return []

    def get_recent_large_transfers(
        self,
        token: str,
        hours: int = 24,
    ) -> List[Transfer]:
        """Get recent large transfers."""
        if token not in self._token_states:
            return []
        
        cutoff = time.time() - (hours * 3600)
        return [
            t for t in self._token_states[token].transfers
            if t.timestamp >= cutoff and t.value_usd >= self._large_threshold
        ]

    def get_exchange_flow(self, token: str, hours: int = 24) -> Dict[str, float]:
        """Get exchange flow summary."""
        if token not in self._token_states:
            return {"inflow": 0.0, "outflow": 0.0, "net": 0.0}
        
        state = self._token_states[token]
        cutoff = time.time() - (hours * 3600)
        recent = [t for t in state.transfers if t.timestamp >= cutoff]
        
        inflow = sum(t.value_usd for t in recent if t.to_address in state.exchange_addresses)
        outflow = sum(t.value_usd for t in recent if t.from_address in state.exchange_addresses)
        
        return {
            "inflow": inflow,
            "outflow": outflow,
            "net": inflow - outflow,
        }

    def get_all_tokens(self) -> List[str]:
        """Get all tracked tokens."""
        return sorted(self._token_states.keys())


__all__ = ["WhaleTracker", "WhaleSignal", "WhaleWallet", "Transfer"]
