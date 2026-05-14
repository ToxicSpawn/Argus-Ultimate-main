"""
Argus Trading System - Custom Exceptions
========================================

Centralized exception definitions for the trading system.
All exceptions inherit from ArgusError for easy catching.
"""

from __future__ import annotations

from typing import Any, Optional


class ArgusError(Exception):
    """Base exception for all Argus trading system errors."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# =============================================================================
# Exchange Errors
# =============================================================================

class ExchangeError(ArgusError):
    """Base exception for exchange-related errors."""
    pass


class ExchangeConnectionError(ExchangeError):
    """Failed to connect to exchange."""
    pass


class ExchangeAuthenticationError(ExchangeError):
    """Authentication failed with exchange."""
    pass


class ExchangeRateLimitError(ExchangeError):
    """Rate limit exceeded on exchange."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        details: Optional[dict] = None
    ) -> None:
        super().__init__(message, details)
        self.retry_after = retry_after


class ExchangeOrderError(ExchangeError):
    """Error placing or managing an order."""

    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        details: Optional[dict] = None
    ) -> None:
        super().__init__(message, details)
        self.order_id = order_id
        self.symbol = symbol


class InsufficientFundsError(ExchangeError):
    """Insufficient balance for the requested operation."""

    def __init__(
        self,
        message: str,
        required: Optional[float] = None,
        available: Optional[float] = None,
        currency: Optional[str] = None,
        details: Optional[dict] = None
    ) -> None:
        super().__init__(message, details)
        self.required = required
        self.available = available
        self.currency = currency


class InvalidSymbolError(ExchangeError):
    """Invalid or unsupported trading symbol."""

    def __init__(self, symbol: str, exchange: Optional[str] = None) -> None:
        message = f"Invalid symbol: {symbol}"
        if exchange:
            message += f" on {exchange}"
        super().__init__(message, {"symbol": symbol, "exchange": exchange})
        self.symbol = symbol
        self.exchange = exchange


# =============================================================================
# Risk Errors
# =============================================================================

class RiskError(ArgusError):
    """Base exception for risk-related errors."""
    pass


class RiskLimitExceededError(RiskError):
    """A risk limit has been exceeded."""

    def __init__(
        self,
        message: str,
        limit_type: str,
        limit_value: float,
        current_value: float,
        details: Optional[dict] = None
    ) -> None:
        super().__init__(message, details)
        self.limit_type = limit_type
        self.limit_value = limit_value
        self.current_value = current_value


class CircuitBreakerError(RiskError):
    """Circuit breaker has been triggered."""

    def __init__(
        self,
        message: str,
        reason: str,
        cooldown_remaining: Optional[float] = None,
        details: Optional[dict] = None
    ) -> None:
        super().__init__(message, details)
        self.reason = reason
        self.cooldown_remaining = cooldown_remaining


class PositionSizingError(RiskError):
    """Error calculating position size."""
    pass


class MaxDrawdownError(RiskError):
    """Maximum drawdown limit exceeded."""

    def __init__(
        self,
        message: str,
        current_drawdown: float,
        max_drawdown: float,
        details: Optional[dict] = None
    ) -> None:
        super().__init__(message, details)
        self.current_drawdown = current_drawdown
        self.max_drawdown = max_drawdown


# =============================================================================
# Strategy Errors
# =============================================================================

class StrategyError(ArgusError):
    """Base exception for strategy-related errors."""
    pass


class StrategyInitializationError(StrategyError):
    """Failed to initialize strategy."""

    def __init__(
        self,
        strategy_name: str,
        message: str,
        details: Optional[dict] = None
    ) -> None:
        full_message = f"Failed to initialize strategy '{strategy_name}': {message}"
        super().__init__(full_message, details)
        self.strategy_name = strategy_name


class InsufficientDataError(StrategyError):
    """Not enough data for strategy calculation."""

    def __init__(
        self,
        message: str,
        required_bars: int,
        available_bars: int,
        symbol: Optional[str] = None,
        details: Optional[dict] = None
    ) -> None:
        super().__init__(message, details)
        self.required_bars = required_bars
        self.available_bars = available_bars
        self.symbol = symbol


class SignalValidationError(StrategyError):
    """Signal failed validation."""

    def __init__(
        self,
        message: str,
        signal: Any,
        validation_errors: list,
        details: Optional[dict] = None
    ) -> None:
        super().__init__(message, details)
        self.signal = signal
        self.validation_errors = validation_errors


# =============================================================================
# Execution Errors
# =============================================================================

class ExecutionError(ArgusError):
    """Base exception for execution-related errors."""
    pass


class OrderRejectedError(ExecutionError):
    """Order was rejected by exchange or risk system."""

    def __init__(
        self,
        message: str,
        reason: str,
        order_request: Any = None,
        details: Optional[dict] = None
    ) -> None:
        super().__init__(message, details)
        self.reason = reason
        self.order_request = order_request


class OrderTimeoutError(ExecutionError):
    """Order did not fill within timeout period."""

    def __init__(
        self,
        message: str,
        order_id: str,
        timeout_seconds: float,
        details: Optional[dict] = None
    ) -> None:
        super().__init__(message, details)
        self.order_id = order_id
        self.timeout_seconds = timeout_seconds


class SlippageExceededError(ExecutionError):
    """Actual slippage exceeded maximum allowed."""

    def __init__(
        self,
        message: str,
        expected_price: float,
        actual_price: float,
        max_slippage_bps: float,
        details: Optional[dict] = None
    ) -> None:
        super().__init__(message, details)
        self.expected_price = expected_price
        self.actual_price = actual_price
        self.max_slippage_bps = max_slippage_bps
        self.actual_slippage_bps = abs(actual_price - expected_price) / expected_price * 10000


# =============================================================================
# Data Errors
# =============================================================================

class DataError(ArgusError):
    """Base exception for data-related errors."""
    pass


class DataFetchError(DataError):
    """Failed to fetch market data."""

    def __init__(
        self,
        message: str,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        details: Optional[dict] = None
    ) -> None:
        super().__init__(message, details)
        self.symbol = symbol
        self.exchange = exchange


class DataValidationError(DataError):
    """Market data failed validation."""
    pass


class StaleDataError(DataError):
    """Data is too old to be reliable."""

    def __init__(
        self,
        message: str,
        data_age_seconds: float,
        max_age_seconds: float,
        details: Optional[dict] = None
    ) -> None:
        super().__init__(message, details)
        self.data_age_seconds = data_age_seconds
        self.max_age_seconds = max_age_seconds


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(ArgusError):
    """Configuration error."""
    pass


class InvalidConfigError(ConfigurationError):
    """Invalid configuration value."""

    def __init__(
        self,
        message: str,
        config_key: str,
        config_value: Any,
        details: Optional[dict] = None
    ) -> None:
        super().__init__(message, details)
        self.config_key = config_key
        self.config_value = config_value


class MissingConfigError(ConfigurationError):
    """Required configuration is missing."""

    def __init__(self, config_key: str) -> None:
        message = f"Missing required configuration: {config_key}"
        super().__init__(message, {"config_key": config_key})
        self.config_key = config_key


# =============================================================================
# Backtest Errors
# =============================================================================

class BacktestError(ArgusError):
    """Base exception for backtest-related errors."""
    pass


class BacktestDataError(BacktestError):
    """Error with backtest data."""
    pass


class BacktestConfigError(BacktestError):
    """Invalid backtest configuration."""
    pass
