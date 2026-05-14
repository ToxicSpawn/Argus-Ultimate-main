"""Kraken exchange connector.

Provides:
  KrakenClient   — async CCXT wrapper (REST: connect / fetch_ticker /
                   fetch_ohlcv / close).  Exposes .client for the raw
                   ccxt.async_support.kraken instance so CcxtKrakenAdapter
                   and KrakenWSClient can share one connection.
  build_kraken_adapter — factory that wires WSFeedAdapter into
                         CcxtKrakenAdapter for tick-level OFI/VPIN.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class KrakenClient:
    """Thin async CCXT wrapper around kraken.

    Parameters
    ----------
    api_key, secret : str | None
        Kraken API credentials.  Pass None for public-only / dry-run.
    dry_run : bool
        When True, place_order() logs only and never hits the REST API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        dry_run: bool = True,
    ) -> None:
        self.api_key = api_key
        self.secret = secret
        self.dry_run = dry_run
        self._client: Any = None  # ccxt async exchange object

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def client(self) -> Any:
        """Raw ccxt.async_support.kraken instance (None before connect())."""
        return self._client

    async def connect(self) -> bool:
        """Initialise the underlying CCXT exchange.  Returns True on success."""
        try:
            import ccxt.async_support as ccxt  # type: ignore

            self._client = ccxt.kraken(
                {
                    "apiKey": self.api_key or "",
                    "secret": self.secret or "",
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                }
            )
            # Load markets so symbol lookups work immediately.
            await self._client.load_markets()
            logger.info("KrakenClient connected (dry_run=%s)", self.dry_run)
            return True
        except Exception as exc:
            logger.error("KrakenClient.connect failed: %s", exc)
            return False

    async def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return CCXT ticker dict or None on error."""
        try:
            return await self._client.fetch_ticker(symbol)
        except Exception as exc:
            logger.debug("fetch_ticker(%s) error: %s", symbol, exc)
            return None

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 150,
    ) -> Optional[List[List[Any]]]:
        """Return list of [ts, o, h, l, c, v] candles or None on error."""
        try:
            return await self._client.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as exc:
            logger.debug("fetch_ohlcv(%s) error: %s", symbol, exc)
            return None

    async def close(self) -> None:
        """Close the underlying CCXT session."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as exc:
                logger.debug("KrakenClient.close error: %s", exc)


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------

def build_kraken_adapter(
    exchange_client: Any,
    dry_run: bool = True,
    ws_adapter: Optional[Any] = None,
) -> "CcxtKrakenAdapter":  # noqa: F821
    """
    Factory for CcxtKrakenAdapter.

    Pass ws_adapter to enable tick-level OFI/VPIN updates via
    CcxtKrakenAdapter.on_trade() and CcxtKrakenAdapter.on_book().
    Without ws_adapter the adapter is fully functional but microstructure
    signals only update on the OHLCV poll cadence (~5 s).
    """
    from argus_live.execution.ccxt_kraken_adapter import CcxtKrakenAdapter  # noqa: PLC0415

    return CcxtKrakenAdapter(
        exchange_client=exchange_client,
        dry_run=dry_run,
        ws_adapter=ws_adapter,
    )
