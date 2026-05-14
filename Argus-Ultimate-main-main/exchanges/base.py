"""
Argus Trading System - Exchange Base Interface
==============================================

Abstract base class defining the exchange interface.
All exchange implementations must inherit from this class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.types import (
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    Side,
    Ticker,
    OrderBook,
    OrderBookLevel,
    OHLCV,
)
from core.exceptions import (
    ExchangeError,
    ExchangeConnectionError,
    ExchangeAuthenticationError,
)


@dataclass
class ExchangeConfig:
    """Base exchange configuration."""
    name: str = "unknown"
    dry_run: bool = True
    sandbox: bool = False
    enable_rate_limit: bool = True
    timeout_ms: int = 30000


@dataclass
class ExchangeInfo:
    """Exchange metadata and capabilities."""
    name: str
    display_name: str
    supported_symbols: List[str]
    maker_fee: float  # As decimal (e.g., 0.0016 for 0.16%)
    taker_fee: float
    min_order_sizes: Dict[str, float]  # symbol -> min size
    price_precisions: Dict[str, int]   # symbol -> decimal places
    qty_precisions: Dict[str, int]     # symbol -> decimal places
    supports_market_orders: bool = True
    supports_limit_orders: bool = True
    supports_stop_orders: bool = True
    supports_websocket: bool = False


class Exchange(ABC):
    """
    Abstract base class for exchange implementations.

    All exchange connectors must implement this interface to ensure
    consistent behavior across different exchanges.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        *,
        dry_run: bool = True,
        sandbox: bool = False,
    ) -> None:
        """
        Initialize exchange connection.

        Args:
            api_key: API key for authentication
            api_secret: API secret for authentication
            dry_run: If True, simulate orders without executing
            sandbox: If True, use exchange sandbox/testnet
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.dry_run = dry_run
        self.sandbox = sandbox
        self._connected = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Exchange identifier (e.g., 'kraken', 'coinbase')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable exchange name."""
        pass

    @property
    def is_connected(self) -> bool:
        """Check if connected to exchange."""
        return self._connected

    # =========================================================================
    # Connection Management
    # =========================================================================

    @abstractmethod
    async def connect(self) -> bool:
        """
        Initialize connection to exchange.

        Returns:
            True if connection successful

        Raises:
            ExchangeConnectionError: If connection fails
            ExchangeAuthenticationError: If authentication fails
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connection and cleanup resources."""
        pass

    async def __aenter__(self) -> "Exchange":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    # =========================================================================
    # Market Data
    # =========================================================================

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Ticker:
        """
        Fetch current ticker data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/AUD')

        Returns:
            Ticker object with current prices

        Raises:
            ExchangeError: If fetch fails
        """
        pass

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 200,
        since: Optional[datetime] = None,
    ) -> List[OHLCV]:
        """
        Fetch OHLCV candle data.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles to fetch
            since: Start time for candles

        Returns:
            List of OHLCV candles, oldest first

        Raises:
            ExchangeError: If fetch fails
        """
        pass

    @abstractmethod
    async def fetch_order_book(
        self,
        symbol: str,
        limit: int = 20,
    ) -> OrderBook:
        """
        Fetch order book depth.

        Args:
            symbol: Trading pair
            limit: Number of levels per side

        Returns:
            OrderBook with bids and asks

        Raises:
            ExchangeError: If fetch fails
        """
        pass

    async def fetch_tickers(self, symbols: List[str]) -> Dict[str, Ticker]:
        """
        Fetch tickers for multiple symbols.

        Default implementation calls fetch_ticker for each symbol.
        Override for exchanges with batch ticker endpoints.

        Args:
            symbols: List of trading pairs

        Returns:
            Dict mapping symbol to Ticker
        """
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = await self.fetch_ticker(symbol)
            except ExchangeError:
                continue
        return result

    # =========================================================================
    # Account Information
    # =========================================================================

    @abstractmethod
    async def fetch_balance(self) -> Dict[str, float]:
        """
        Fetch account balances.

        Returns:
            Dict mapping currency to available balance
            (e.g., {'AUD': 1000.0, 'BTC': 0.5})

        Raises:
            ExchangeError: If fetch fails
        """
        pass

    async def fetch_trading_balance(self, currency: str = "AUD") -> float:
        """
        Fetch available trading balance for a specific currency.

        Args:
            currency: Currency code

        Returns:
            Available balance

        Raises:
            ExchangeError: If fetch fails
        """
        balances = await self.fetch_balance()
        return balances.get(currency, 0.0)

    # =========================================================================
    # Order Management
    # =========================================================================

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: Side,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        *,
        client_order_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> OrderResult:
        """
        Place a new order.

        Args:
            symbol: Trading pair
            side: Buy or sell
            order_type: Market, limit, etc.
            quantity: Order size in base currency
            price: Limit price (required for limit orders)
            client_order_id: Client-assigned order ID
            params: Additional exchange-specific parameters

        Returns:
            OrderResult with order details

        Raises:
            ExchangeOrderError: If order fails
            InsufficientFundsError: If balance too low
        """
        pass

    @abstractmethod
    async def cancel_order(
        self,
        order_id: str,
        symbol: Optional[str] = None,
    ) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair (required by some exchanges)

        Returns:
            True if cancellation successful

        Raises:
            ExchangeOrderError: If cancellation fails
        """
        pass

    @abstractmethod
    async def fetch_order(
        self,
        order_id: str,
        symbol: Optional[str] = None,
    ) -> OrderResult:
        """
        Fetch order details.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair

        Returns:
            OrderResult with current order state

        Raises:
            ExchangeOrderError: If fetch fails
        """
        pass

    @abstractmethod
    async def fetch_open_orders(
        self,
        symbol: Optional[str] = None,
    ) -> List[OrderResult]:
        """
        Fetch all open orders.

        Args:
            symbol: Filter by trading pair (optional)

        Returns:
            List of open orders

        Raises:
            ExchangeError: If fetch fails
        """
        pass

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def market_buy(
        self,
        symbol: str,
        quantity: float,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        """Place a market buy order."""
        return await self.create_order(
            symbol=symbol,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            client_order_id=client_order_id,
        )

    async def market_sell(
        self,
        symbol: str,
        quantity: float,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        """Place a market sell order."""
        return await self.create_order(
            symbol=symbol,
            side=Side.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            client_order_id=client_order_id,
        )

    async def limit_buy(
        self,
        symbol: str,
        quantity: float,
        price: float,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        """Place a limit buy order."""
        return await self.create_order(
            symbol=symbol,
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            client_order_id=client_order_id,
        )

    async def limit_sell(
        self,
        symbol: str,
        quantity: float,
        price: float,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        """Place a limit sell order."""
        return await self.create_order(
            symbol=symbol,
            side=Side.SELL,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            client_order_id=client_order_id,
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_min_order_size(self, symbol: str) -> float:
        """Get minimum order size for a symbol."""
        return 0.0001  # Default, override in implementations

    def get_price_precision(self, symbol: str) -> int:
        """Get price precision (decimal places) for a symbol."""
        return 2  # Default, override in implementations

    def get_quantity_precision(self, symbol: str) -> int:
        """Get quantity precision (decimal places) for a symbol."""
        return 8  # Default, override in implementations

    def round_price(self, price: float, symbol: str) -> float:
        """Round price to valid precision for symbol."""
        precision = self.get_price_precision(symbol)
        return round(price, precision)

    def round_quantity(self, quantity: float, symbol: str) -> float:
        """Round quantity to valid precision for symbol."""
        precision = self.get_quantity_precision(symbol)
        return round(quantity, precision)
