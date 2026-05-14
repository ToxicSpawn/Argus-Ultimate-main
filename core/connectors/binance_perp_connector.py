"""Binance perpetual futures connector via CCXT."""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import ccxt

logger = logging.getLogger(__name__)


class BinancePerpConnector:
    """Connector for Binance USDT-margined perpetual futures.

    Uses CCXT with ``defaultType="future"`` so all symbol lookups resolve
    against the ``/fapi`` endpoint.  The connector can operate in read-only
    mode (no API keys) for public endpoints such as tickers and funding
    rates.  Authenticated endpoints (create_order, get_position, …) require
    valid API credentials.
    """

    health_check_symbol: str = "BTC/USDT"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> None:
        """Initialise the connector.

        Args:
            api_key: Binance API key.  Falls back to the ``BINANCE_API_KEY``
                environment variable when *None*.
            api_secret: Binance API secret.  Falls back to the
                ``BINANCE_API_SECRET`` environment variable when *None*.
        """
        self._api_key: str = api_key or os.environ.get("BINANCE_API_KEY", "")
        self._api_secret: str = api_secret or os.environ.get("BINANCE_API_SECRET", "")
        self._connected: bool = False

        self._exchange: ccxt.binanceusdm = ccxt.binanceusdm(
            {
                "apiKey": self._api_key or None,
                "secret": self._api_secret or None,
                "options": {"defaultType": "future"},
                "enableRateLimit": True,
            }
        )

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Establish connection and load markets.

        In read-only mode (no API keys) the connection is still considered
        successful; authenticated calls will simply fail at call time.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        try:
            self._exchange.load_markets()
            self._connected = True
            auth = bool(self._api_key and self._api_secret)
            logger.info(
                "BinancePerpConnector connected (authenticated=%s)", auth
            )
            return True
        except Exception as exc:
            logger.warning("BinancePerpConnector.connect failed: %s", exc)
            self._connected = False
            return False

    def disconnect(self) -> bool:
        """Tear down the connection.

        Returns:
            Always ``True`` (no persistent socket to close with CCXT REST).
        """
        self._connected = False
        logger.info("BinancePerpConnector disconnected")
        return True

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Fetch the current best bid/ask and last price for *symbol*.

        Args:
            symbol: CCXT unified symbol, e.g. ``"BTC/USDT"``.

        Returns:
            CCXT ticker dict on success, ``None`` on failure.
        """
        try:
            ticker = self._exchange.fetch_ticker(symbol)
            return ticker
        except Exception as exc:
            logger.warning("BinancePerpConnector.get_ticker(%s) failed: %s", symbol, exc)
            return None

    def fetch_funding_rate(self, symbol: str) -> Dict:
        """Fetch the current funding rate for a single perpetual contract.

        Args:
            symbol: CCXT unified symbol, e.g. ``"BTC/USDT"``.

        Returns:
            Dict with keys: ``symbol``, ``funding_rate`` (float),
            ``next_funding_time`` (int ms or None), ``exchange`` (str).
            Returns zeroed-out dict on failure so callers can always
            access the keys safely.
        """
        default: Dict = {
            "symbol": symbol,
            "funding_rate": 0.0,
            "next_funding_time": None,
            "exchange": "binance",
        }
        try:
            raw = self._exchange.fetch_funding_rate(symbol)
            return {
                "symbol": symbol,
                "funding_rate": float(raw.get("fundingRate", 0.0)),
                "next_funding_time": raw.get("fundingDatetime") or raw.get("nextFundingDatetime"),
                "exchange": "binance",
            }
        except Exception as exc:
            logger.warning(
                "BinancePerpConnector.fetch_funding_rate(%s) failed: %s", symbol, exc
            )
            return default

    def fetch_funding_rates(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch funding rates for multiple symbols in one call when possible.

        Args:
            symbols: List of CCXT unified symbols.

        Returns:
            Mapping of symbol -> funding rate (float).  Missing symbols map
            to ``0.0``.
        """
        result: Dict[str, float] = {s: 0.0 for s in symbols}
        try:
            raw_map = self._exchange.fetch_funding_rates(symbols)
            for sym, data in raw_map.items():
                if sym in result:
                    result[sym] = float(data.get("fundingRate", 0.0))
        except Exception as exc:
            logger.warning(
                "BinancePerpConnector.fetch_funding_rates bulk call failed (%s); "
                "falling back to individual calls",
                exc,
            )
            for sym in symbols:
                result[sym] = self.fetch_funding_rate(sym).get("funding_rate", 0.0)
        return result

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def create_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "market",
        price: Optional[float] = None,
    ) -> Optional[Dict]:
        """Place an order on the exchange.

        Args:
            symbol: CCXT unified symbol, e.g. ``"BTC/USDT"``.
            side: ``"buy"`` or ``"sell"``.
            amount: Contract/coin quantity.
            order_type: ``"market"`` (default) or ``"limit"``.
            price: Required for limit orders; ignored for market orders.

        Returns:
            CCXT order dict on success, ``None`` on failure.
        """
        if not (self._api_key and self._api_secret):
            logger.warning(
                "BinancePerpConnector.create_order: no API keys configured"
            )
            return None
        try:
            params: Dict = {}
            order = self._exchange.create_order(
                symbol, order_type, side, amount, price, params
            )
            logger.info(
                "BinancePerpConnector order placed: %s %s %s qty=%.6f",
                order_type,
                side,
                symbol,
                amount,
            )
            return order
        except Exception as exc:
            logger.warning(
                "BinancePerpConnector.create_order(%s %s %s) failed: %s",
                side,
                order_type,
                symbol,
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Fetch the open position for *symbol* (if any).

        Args:
            symbol: CCXT unified symbol.

        Returns:
            CCXT position dict on success, ``None`` when no position exists
            or on failure.
        """
        if not (self._api_key and self._api_secret):
            logger.warning(
                "BinancePerpConnector.get_position: no API keys configured"
            )
            return None
        try:
            positions = self._exchange.fetch_positions([symbol])
            for pos in positions:
                if float(pos.get("contracts", 0) or 0) != 0:
                    return pos
            return None
        except Exception as exc:
            logger.warning(
                "BinancePerpConnector.get_position(%s) failed: %s", symbol, exc
            )
            return None

    def close_position(self, symbol: str) -> Optional[Dict]:
        """Close the entire open position for *symbol* with a market order.

        Determines the current position side/size and places the opposite
        market order.

        Args:
            symbol: CCXT unified symbol.

        Returns:
            CCXT order dict for the closing order, or ``None`` on failure.
        """
        position = self.get_position(symbol)
        if position is None:
            logger.warning(
                "BinancePerpConnector.close_position(%s): no open position found",
                symbol,
            )
            return None
        try:
            contracts = float(position.get("contracts", 0) or 0)
            side_raw = (position.get("side") or "").lower()
            # Determine closing side
            close_side = "sell" if side_raw == "long" else "buy"
            amount = abs(contracts)
            params = {"reduceOnly": True}
            order = self._exchange.create_order(
                symbol, "market", close_side, amount, None, params
            )
            logger.info(
                "BinancePerpConnector closed position %s qty=%.6f", symbol, amount
            )
            return order
        except Exception as exc:
            logger.warning(
                "BinancePerpConnector.close_position(%s) failed: %s", symbol, exc
            )
            return None

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_balance(self) -> Dict:
        """Fetch account balances.

        Returns:
            CCXT balance dict on success; empty dict on failure.
        """
        if not (self._api_key and self._api_secret):
            logger.warning(
                "BinancePerpConnector.get_balance: no API keys configured"
            )
            return {}
        try:
            balance = self._exchange.fetch_balance()
            return balance
        except Exception as exc:
            logger.warning("BinancePerpConnector.get_balance failed: %s", exc)
            return {}
