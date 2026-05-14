from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from argus_live.execution.adapter_registry import AdapterRegistry
from argus_live.execution.venue_adapter import VenueAdapter


@dataclass
class ExchangeConfig:
    """Configuration for an exchange connection.
    
    Attributes
    ----------
    name : str
        Exchange name (e.g., "binance", "kraken")
    api_key : str
        API key for authentication
    api_secret : str
        API secret for authentication
    sandbox : bool
        Whether to use sandbox/testnet mode
    enabled : bool
        Whether this exchange is enabled
    symbols : list[str]
        List of trading symbols to subscribe to
    rate_limit : float
        Rate limit in requests per second
    """
    name: str = ""
    api_key: str = ""
    api_secret: str = ""
    sandbox: bool = True
    enabled: bool = True
    symbols: List[str] = field(default_factory=list)
    rate_limit: float = 10.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExchangeConfig":
        """Create ExchangeConfig from a dictionary."""
        return cls(
            name=data.get("name", ""),
            api_key=data.get("api_key", ""),
            api_secret=data.get("api_secret", ""),
            sandbox=data.get("sandbox", True),
            enabled=data.get("enabled", True),
            symbols=data.get("symbols", []),
            rate_limit=data.get("rate_limit", 10.0),
        )


@dataclass
class ExchangeManager:
    """Legacy compatibility wrapper around adapter registry."""

    registry: AdapterRegistry

    @classmethod
    def from_adapters(cls, adapters: dict[str, VenueAdapter]) -> "ExchangeManager":
        return cls(AdapterRegistry(adapters))

    def get_adapter(self, venue: str) -> VenueAdapter:
        return self.registry.get(venue)
