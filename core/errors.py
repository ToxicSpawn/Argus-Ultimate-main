#!/usr/bin/env python3
"""
Argus Error Classes - S+ Tier
Comprehensive error handling for the Argus trading system.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ArgusError(Exception):
    """Base exception for Argus system"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}"


class ConfigurationError(ArgusError):
    """Configuration-related errors"""

    pass


class ValidationError(ArgusError):
    """Data validation errors"""

    pass


class ExchangeError(ArgusError):
    """Exchange API errors"""
    pass


class RateLimitExceeded(ExchangeError):
    """Rate limit exceeded"""

    pass


class NetworkError(ExchangeError):
    """Network connectivity errors"""

    pass


class AuthenticationError(ExchangeError):
    """API authentication errors"""

    pass


class InsufficientFunds(ExchangeError):
    """Insufficient account funds"""

    pass


class InvalidOrder(ExchangeError):
    """Invalid order parameters"""

    pass


class DatabaseError(ArgusError):
    """Database operation errors"""
    pass


class DatabaseConnectionError(DatabaseError):
    """Database connection errors"""

    pass


class DatabaseQueryError(DatabaseError):
    """Database query errors"""

    pass


class DatabaseTransactionError(DatabaseError):
    """Database transaction errors"""

    pass


class StrategyError(ArgusError):
    """Strategy execution errors"""
    pass


class StrategyInitializationError(StrategyError):
    """Strategy initialization errors"""

    pass


class StrategyExecutionError(StrategyError):
    """Strategy execution errors"""

    pass


class StrategyParameterError(StrategyError):
    """Strategy parameter errors"""

    pass


class RiskError(ArgusError):
    """Risk management errors"""
    pass


class RiskLimitExceeded(RiskError):
    """Risk management violations"""

    pass


class PositionSizeError(RiskError):
    """Position sizing errors"""

    pass


class ExposureError(RiskError):
    """Portfolio exposure errors"""

    pass


class BacktestError(ArgusError):
    """Backtesting errors"""
    pass


class BacktestDataError(BacktestError):
    """Backtest data errors"""

    pass


class BacktestSimulationError(BacktestError):
    """Backtest simulation errors"""

    pass


class OptimizationError(ArgusError):
    """Optimization errors"""
    pass


class OptimizationConvergenceError(OptimizationError):
    """Optimization convergence errors"""

    pass


class OptimizationParameterError(OptimizationError):
    """Optimization parameter errors"""

    pass


class MachineLearningError(ArgusError):
    """Machine learning errors"""
    pass


class MLTrainingError(MachineLearningError):
    """Model training errors"""

    pass


class MLPredictionError(MachineLearningError):
    """Model prediction errors"""

    pass


class MLDataError(MachineLearningError):
    """ML data processing errors"""

    pass


def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Handle and format errors for logging and reporting.

    Args:
        error: The exception that occurred
        context: Additional context information

    Returns:
        Formatted error information
    """
    error_info = {
        "error_type": error.__class__.__name__,
        "message": str(error),
        "module": getattr(error, "__module__", "unknown"),
    }

    if context:
        error_info["context"] = context

    if isinstance(error, ArgusError) and hasattr(error, "details"):
        error_info["details"] = error.details

    return error_info


def log_error(error: Exception, logger, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an error with appropriate level and formatting.

    Args:
        error: The exception to log
        logger: Logger instance
        context: Additional context
    """
    error_info = handle_error(error, context)

    if isinstance(error, (ExchangeError, DatabaseError)):
        logger.error(f"Critical system error: {error_info}")
    elif isinstance(error, (StrategyError, RiskError)):
        logger.warning(f"Operational error: {error_info}")
    else:
        logger.info(f"Error occurred: {error_info}")


def create_error_response(error: Exception, request_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a standardized error response for API endpoints.

    Args:
        error: The exception that occurred
        request_id: Optional request identifier

    Returns:
        Standardized error response
    """
    error_info = handle_error(error)

    response = {
        "success": False,
        "error": error_info,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
    }

    if request_id:
        response["request_id"] = request_id

    return response
