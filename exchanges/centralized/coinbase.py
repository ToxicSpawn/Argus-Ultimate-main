"""
Coinbase Exchange Connector - Re-export from coinbase_advanced.

For backward compatibility, this module re-exports the
CoinbaseAdvancedClient from coinbase_advanced.py.
"""

from __future__ import annotations

from exchanges.centralized.coinbase_advanced import CoinbaseAdvancedClient

# Re-export for compatibility
CoinbaseClient = CoinbaseAdvancedClient
CoinbaseExchange = CoinbaseAdvancedClient

__all__ = ["CoinbaseClient", "CoinbaseExchange", "CoinbaseAdvancedClient"]
