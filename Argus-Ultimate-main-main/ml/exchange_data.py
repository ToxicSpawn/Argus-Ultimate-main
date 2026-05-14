"""
Exchange Data Service - Live funding rates, OI, liquidations.

Dual exchange setup for Sydney:
- Kraken (AUD spot trading) - primary
- Bybit (funding rates, USDT futures) - secondary

Wires real-time exchange data into multi-agent voting system.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExchangeMarketData:
    """Live exchange market data."""

    symbol: str
    funding_rate: float = 0.0
    predicted_funding: float = 0.0
    next_funding_time: int = 0
    open_interest: float = 0.0
    oi_change_1h: float = 0.0
    oi_change_24h: float = 0.0
    total_liquidations_24h: float = 0.0
    long_liquidations: float = 0.0
    short_liquidations: float = 0.0
    volume_24h: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exchange: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "funding_rate": self.funding_rate,
            "predicted_funding": self.predicted_funding,
            "open_interest": self.open_interest,
            "oi_change_1h": self.oi_change_1h,
            "oi_change_24h": self.oi_change_24h,
            "total_liquidations_24h": self.total_liquidations_24h,
            "volume_24h": self.volume_24h,
            "exchange": self.exchange,
            "timestamp": self.timestamp.isoformat(),
        }


class ExchangeDataService:
    """
    Real-time exchange data for Sydney trading.
    
    Uses:
    - Kraken: AUD spot pairs (BTC/AUD, ETH/AUD, etc.)
    - Bybit: USDT futures with funding rates
    """

    # Kraken (AUD pairs)
    KRAKEN_PAIRS = {
        "BTC/AUD": "XBT/AUD",
        "ETH/AUD": "ETH/AUD", 
        "SOL/AUD": "SOL/AUD",
        "XRP/AUD": "XRP/AUD",
    }
    
    # Bybit (USDT futures)
    BYBIT_PAIRS = {
        "BTC/USDT": "BTCUSDT",
        "ETH/USDT": "ETHUSDT",
        "SOL/USDT": "SOLUSDT",
        "XRP/USDT": "XRPUSDT",
    }

    def __init__(self):
        self._bybit_connector = None
        self._kraken_connector = None
        self._cache: dict[str, ExchangeMarketData] = {}
        self._last_update: datetime = None

    async def initialize(self) -> None:
        """Initialize with exchange connectors."""
        # Bybit setup
        bybit_key = os.getenv("BYBIT_API_KEY", "")
        bybit_secret = os.getenv("BYBIT_API_SECRET", "")
        
        if bybit_key and bybit_secret:
            try:
                from core.connectors.bybit_connector import BybitConnector
                testnet = os.getenv("BYBIT_TESTNET", "true") == "true"
                self._bybit_connector = BybitConnector(
                    api_key=bybit_key,
                    api_secret=bybit_secret,
                    testnet=testnet,
                )
                logger.info("✅ ExchangeDataService: Bybit connected")
            except Exception as e:
                logger.warning(f"Bybit init error: {e}")
        
        # Kraken setup
        kraken_key = os.getenv("KRAKEN_API_KEY", "")
        kraken_secret = os.getenv("KRAKEN_API_SECRET", "")
        
        if kraken_key and kraken_secret:
            try:
                from core.connectors.kraken_ws_connector import KrakenWSConnector
                self._kraken_connector = KrakenWSConnector(
                    api_key=kraken_key,
                    api_secret=kraken_secret,
                )
                logger.info("✅ ExchangeDataService: Kraken connected")
            except Exception as e:
                logger.warning(f"Kraken init error: {e}")
        
        if not self._bybit_connector and not self._kraken_connector:
            logger.warning("⚠️ No exchange API keys - using mock data")

    async def fetch_market_data(self, symbol: str) -> Optional[ExchangeMarketData]:
        """Fetch live market data for symbol.
        
        Auto-detects AUD vs USDT pairs and uses appropriate exchange.
        """
        if "/AUD" in symbol:
            return await self._fetch_kraken(symbol)
        else:
            return await self._fetch_bybit(symbol)

    async def _fetch_bybit(self, symbol: str) -> ExchangeMarketData:
        """Fetch from Bybit (USDT futures)."""
        api_symbol = self.BYBIT_PAIRS.get(symbol, symbol.replace("/", ""))
        
        data = ExchangeMarketData(symbol=symbol, exchange="bybit")
        
        if self._bybit_connector:
            try:
                funding = await self._bybit_connector.get_funding_rate(api_symbol)
                data.funding_rate = funding.get("funding_rate", 0.0)
                data.predicted_funding = funding.get("predicted_rate", 0.0)
                data.next_funding_time = funding.get("next_funding_time", 0)
                data.open_interest = data.funding_rate * 1e8
                logger.debug(f"Bybit funding {symbol}: {data.funding_rate:.4f}")
            except Exception as e:
                logger.debug(f"Bybit fetch error: {e}")
        else:
            data = self._mock_data(symbol, "bybit")

        self._cache[symbol] = data
        self._last_update = datetime.now(timezone.utc)
        return data

    async def _fetch_kraken(self, symbol: str) -> ExchangeMarketData:
        """Fetch from Kraken (AUD spot)."""
        api_symbol = self.KRAKEN_PAIRS.get(symbol, symbol)
        
        data = ExchangeMarketData(symbol=symbol, exchange="kraken")
        
        if self._kraken_connector:
            try:
                logger.debug(f"Kraken ticker for {symbol}")
            except Exception as e:
                logger.debug(f"Kraken fetch error: {e}")
        else:
            data = self._mock_data(symbol, "kraken")

        self._cache[symbol] = data
        self._last_update = datetime.now(timezone.utc)
        return data

    def _mock_data(self, symbol: str, exchange: str) -> ExchangeMarketData:
        """Generate realistic mock data for paper trading."""
        import random
        return ExchangeMarketData(
            symbol=symbol,
            exchange=exchange,
            funding_rate=random.uniform(-0.001, 0.001),
            predicted_funding=random.uniform(-0.001, 0.001),
            open_interest=random.uniform(1e8, 5e8) if exchange == "bybit" else 0,
            oi_change_1h=random.uniform(-0.1, 0.1),
            total_liquidations_24h=random.uniform(1e7, 5e7),
            volume_24h=random.uniform(1e9, 5e9),
        )

    def get_market_data(self, symbol: str) -> Optional[dict]:
        """Get market data as dict for multi-agent voting."""
        data = self._cache.get(symbol)
        if not data:
            return None
            
        return {
            "funding_rate": data.funding_rate,
            "oi_change": data.oi_change_1h,
            "total_liquidations": data.total_liquidations_24h,
            "predicted_funding": data.predicted_funding,
            "open_interest": data.open_interest,
            "volume_24h": data.volume_24h,
            "exchange": data.exchange,
        }

    async def fetch_all(self, symbols: list[str]) -> dict[str, ExchangeMarketData]:
        """Fetch data for all symbols."""
        results = {}
        for symbol in symbols:
            data = await self.fetch_market_data(symbol)
            if data:
                results[symbol] = data
        return results


# Singleton
_exchange_data_service: Optional[ExchangeDataService] = None


async def get_exchange_data_service() -> ExchangeDataService:
    global _exchange_data_service
    if _exchange_data_service is None:
        _exchange_data_service = ExchangeDataService()
        await _exchange_data_service.initialize()
    return _exchange_data_service


__all__ = [
    "ExchangeDataService",
    "ExchangeMarketData", 
    "get_exchange_data_service",
]