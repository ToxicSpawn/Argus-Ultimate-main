"""
Argus Ultimate - Exception Management System
============================================

Comprehensive exception handling to replace silent failures.
Provides custom exception hierarchy and proper error logging.
"""

import logging
import traceback
from typing import Optional, Dict, Any, Type
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
import sys

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exception Hierarchy
# ============================================================================

class ArgusException(Exception):
    """Base exception for all Argus errors."""
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = False
    ):
        super().__init__(message)
        self.message = message
        self.code = code or "ARGUS_ERROR"
        self.details = details or {}
        self.recoverable = recoverable
        self.timestamp = datetime.utcnow().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/monitoring."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "details": self.details,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp,
            "traceback": traceback.format_exc()
        }


# Trading Exceptions
class TradingException(ArgusException):
    """Base exception for trading errors."""
    pass


class OrderProcessingError(TradingException):
    """Order processing failed."""
    
    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            code="ORDER_ERROR",
            details={"order_id": order_id, "symbol": symbol, **kwargs},
            recoverable=False
        )


class RiskViolationError(TradingException):
    """Order violates risk limits."""
    
    def __init__(
        self,
        message: str,
        violation_type: str,
        current_value: float,
        limit: float,
        **kwargs
    ):
        super().__init__(
            message,
            code="RISK_VIOLATION",
            details={
                "violation_type": violation_type,
                "current_value": current_value,
                "limit": limit,
                **kwargs
            },
            recoverable=False
        )


class ExecutionError(TradingException):
    """Order execution failed."""
    
    def __init__(
        self,
        message: str,
        venue: Optional[str] = None,
        retry_possible: bool = True,
        **kwargs
    ):
        super().__init__(
            message,
            code="EXECUTION_ERROR",
            details={"venue": venue, "retry_possible": retry_possible, **kwargs},
            recoverable=retry_possible
        )


class VenueUnavailableError(TradingException):
    """Trading venue is unavailable."""
    
    def __init__(self, message: str, venue: str, **kwargs):
        super().__init__(
            message,
            code="VENUE_UNAVAILABLE",
            details={"venue": venue, **kwargs},
            recoverable=True
        )


class InsufficientFundsError(TradingException):
    """Insufficient funds for operation."""
    
    def __init__(
        self,
        message: str,
        required: float,
        available: float,
        currency: str,
        **kwargs
    ):
        super().__init__(
            message,
            code="INSUFFICIENT_FUNDS",
            details={
                "required": required,
                "available": available,
                "currency": currency,
                **kwargs
            },
            recoverable=False
        )


# Data Exceptions
class DataException(ArgusException):
    """Base exception for data errors."""
    pass


class DataFeedError(DataException):
    """Data feed error."""
    
    def __init__(self, message: str, feed: str, symbol: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            code="DATA_FEED_ERROR",
            details={"feed": feed, "symbol": symbol, **kwargs},
            recoverable=True
        )


class DataValidationError(DataException):
    """Data validation failed."""
    
    def __init__(
        self,
        message: str,
        field: str,
        expected_type: str,
        actual_value: Any,
        **kwargs
    ):
        super().__init__(
            message,
            code="DATA_VALIDATION_ERROR",
            details={
                "field": field,
                "expected_type": expected_type,
                "actual_value": str(actual_value),
                **kwargs
            },
            recoverable=True
        )


class DataMissingError(DataException):
    """Required data is missing."""
    
    def __init__(self, message: str, data_key: str, source: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            code="DATA_MISSING",
            details={"data_key": data_key, "source": source, **kwargs},
            recoverable=True
        )


# ML/Strategy Exceptions
class MLException(ArgusException):
    """Base exception for ML errors."""
    pass


class ModelPredictionError(MLException):
    """Model prediction failed."""
    
    def __init__(
        self,
        message: str,
        model_name: str,
        symbol: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            code="MODEL_PREDICTION_ERROR",
            details={"model_name": model_name, "symbol": symbol, **kwargs},
            recoverable=True
        )


class ModelTrainingError(MLException):
    """Model training failed."""
    
    def __init__(self, message: str, model_name: str, **kwargs):
        super().__init__(
            message,
            code="MODEL_TRAINING_ERROR",
            details={"model_name": model_name, **kwargs},
            recoverable=True
        )


class StrategyError(MLException):
    """Strategy execution error."""
    
    def __init__(
        self,
        message: str,
        strategy_name: str,
        symbol: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            code="STRATEGY_ERROR",
            details={"strategy_name": strategy_name, "symbol": symbol, **kwargs},
            recoverable=True
        )


# Configuration Exceptions
class ConfigException(ArgusException):
    """Base exception for configuration errors."""
    pass


class ConfigLoadError(ConfigException):
    """Configuration loading failed."""
    
    def __init__(self, message: str, config_path: str, **kwargs):
        super().__init__(
            message,
            code="CONFIG_LOAD_ERROR",
            details={"config_path": config_path, **kwargs},
            recoverable=False
        )


class ConfigValidationError(ConfigException):
    """Configuration validation failed."""
    
    def __init__(
        self,
        message: str,
        field: str,
        value: Any,
        constraint: str,
        **kwargs
    ):
        super().__init__(
            message,
            code="CONFIG_VALIDATION_ERROR",
            details={
                "field": field,
                "value": str(value),
                "constraint": constraint,
                **kwargs
            },
            recoverable=False
        )


class ConfigMissingError(ConfigException):
    """Required configuration is missing."""
    
    def __init__(self, message: str, config_key: str, **kwargs):
        super().__init__(
            message,
            code="CONFIG_MISSING",
            details={"config_key": config_key, **kwargs},
            recoverable=False
        )


# Network/External Service Exceptions
class NetworkException(ArgusException):
    """Base exception for network errors."""
    pass


class ExchangeAPIError(NetworkException):
    """Exchange API error."""
    
    def __init__(
        self,
        message: str,
        exchange: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message,
            code="EXCHANGE_API_ERROR",
            details={
                "exchange": exchange,
                "endpoint": endpoint,
                "status_code": status_code,
                **kwargs
            },
            recoverable=True
        )


class RateLimitError(NetworkException):
    """API rate limit exceeded."""
    
    def __init__(
        self,
        message: str,
        service: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message,
            code="RATE_LIMIT_ERROR",
            details={
                "service": service,
                "retry_after": retry_after,
                **kwargs
            },
            recoverable=True
        )


# Database Exceptions
class DatabaseException(ArgusException):
    """Base exception for database errors."""
    pass


class DatabaseConnectionError(DatabaseException):
    """Database connection failed."""
    
    def __init__(self, message: str, db_name: str, **kwargs):
        super().__init__(
            message,
            code="DB_CONNECTION_ERROR",
            details={"db_name": db_name, **kwargs},
            recoverable=True
        )


class DatabaseQueryError(DatabaseException):
    """Database query failed."""
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        table: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            code="DB_QUERY_ERROR",
            details={"query": query, "table": table, **kwargs},
            recoverable=True
        )


# ============================================================================
# Exception Manager
# ============================================================================

@dataclass
class ExceptionRecord:
    """Record of an exception for monitoring."""
    exception_type: str
    message: str
    code: str
    recoverable: bool
    timestamp: datetime
    file: str
    line: int
    function: str
    trace: str
    count: int = 1


class ExceptionManager:
    """Centralized exception management and monitoring."""
    
    def __init__(self):
        self._records: Dict[str, ExceptionRecord] = {}
        self._handlers: Dict[Type[Exception], callable] = {}
        
    def register_handler(self, exc_type: Type[Exception], handler: callable):
        """Register custom handler for exception type."""
        self._handlers[exc_type] = handler
        
    def handle_exception(
        self,
        exc: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle exception with proper logging and monitoring."""
        
        # Get exception details
        exc_type = type(exc).__name__
        exc_message = str(exc)
        exc_code = getattr(exc, 'code', 'UNKNOWN_ERROR')
        recoverable = getattr(exc, 'recoverable', False)
        
        # Get stack trace
        tb = traceback.extract_tb(sys.exc_info()[2])
        if tb:
            last_frame = tb[-1]
            file = last_frame.filename
            line = last_frame.lineno
            function = last_frame.name
        else:
            file = "unknown"
            line = 0
            function = "unknown"
        
        # Create record key
        record_key = f"{exc_type}:{file}:{line}:{exc_message}"
        
        # Update or create record
        if record_key in self._records:
            self._records[record_key].count += 1
        else:
            self._records[record_key] = ExceptionRecord(
                exception_type=exc_type,
                message=exc_message,
                code=exc_code,
                recoverable=recoverable,
                timestamp=datetime.utcnow(),
                file=file,
                line=line,
                function=function,
                trace=traceback.format_exc()
            )
        
        # Log appropriately
        if recoverable:
            logger.warning(
                f"Recoverable error [{exc_code}]: {exc_message}",
                extra={
                    "context": context,
                    "exception": exc.to_dict() if hasattr(exc, 'to_dict') else str(exc)
                }
            )
        else:
            logger.error(
                f"Critical error [{exc_code}]: {exc_message}",
                exc_info=True,
                extra={
                    "context": context,
                    "exception": exc.to_dict() if hasattr(exc, 'to_dict') else str(exc)
                }
            )
        
        # Call custom handler if registered
        for exc_class, handler in self._handlers.items():
            if isinstance(exc, exc_class):
                try:
                    handler(exc, context)
                except Exception as handler_exc:
                    logger.error(f"Exception handler failed: {handler_exc}")
        
        # Return error details for API responses
        return {
            "success": False,
            "error": {
                "type": exc_type,
                "code": exc_code,
                "message": exc_message,
                "recoverable": recoverable,
                "context": context
            }
        }
    
    def get_exception_stats(self) -> Dict[str, Any]:
        """Get exception statistics for monitoring."""
        critical = sum(1 for r in self._records.values() if not r.recoverable)
        recoverable = sum(1 for r in self._records.values() if r.recoverable)
        total_count = sum(r.count for r in self._records.values())
        
        return {
            "total_unique_exceptions": len(self._records),
            "critical_exceptions": critical,
            "recoverable_exceptions": recoverable,
            "total_occurrences": total_count,
            "top_exceptions": sorted(
                self._records.values(),
                key=lambda x: x.count,
                reverse=True
            )[:10]
        }
    
    def reset_stats(self):
        """Reset exception statistics."""
        self._records.clear()


# Global exception manager instance
exception_manager = ExceptionManager()


def safe_execute(
    func,
    *args,
    default=None,
    on_error=None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    Safely execute a function with proper exception handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        default: Default value to return on error
        on_error: Callback to call on error
        context: Additional context for error logging
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default value on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        exception_manager.handle_exception(e, context or {})
        if on_error:
            try:
                on_error(e)
            except Exception as callback_error:
                logger.error(f"Error callback failed: {callback_error}")
        return default


def retry_on_error(
    max_retries: int = 3,
    exceptions: tuple = (Exception,),
    delay: float = 1.0,
    backoff: float = 2.0,
    on_retry: Optional[callable] = None
):
    """
    Decorator for retrying function on specified exceptions.
    
    Args:
        max_retries: Maximum number of retry attempts
        exceptions: Tuple of exception types to retry on
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        on_retry: Callback function called on each retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}"
                        )
                        if on_retry:
                            try:
                                on_retry(e, attempt + 1)
                            except Exception as callback_error:
                                logger.error(f"Retry callback failed: {callback_error}")
                        
                        import time
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}"
                        )
            
            # All retries exhausted
            raise last_exception
        
        return wrapper
    return decorator


def validate_required(
    value: Any,
    name: str,
    expected_type: Optional[type] = None,
    allow_empty: bool = False
):
    """
    Validate that a required value is present and valid.
    
    Args:
        value: Value to validate
        name: Name of the field (for error messages)
        expected_type: Expected type of the value
        allow_empty: Whether empty values are allowed
        
    Raises:
        DataValidationError: If validation fails
    """
    if value is None:
        raise DataValidationError(
            f"Required field '{name}' is None",
            field=name,
            expected_type=expected_type.__name__ if expected_type else "any",
            actual_value=None
        )
    
    if not allow_empty and value in ["", [], {}, set()]:
        raise DataValidationError(
            f"Required field '{name}' is empty",
            field=name,
            expected_type=expected_type.__name__ if expected_type else "any",
            actual_value=value
        )
    
    if expected_type and not isinstance(value, expected_type):
        raise DataValidationError(
            f"Field '{name}' must be {expected_type.__name__}, got {type(value).__name__}",
            field=name,
            expected_type=expected_type.__name__,
            actual_value=value
        )


# Convenience function for common error handling pattern
def handle_errors(
    logger_name: Optional[str] = None,
    reraise: bool = True,
    default=None
):
    """
    Context manager/decorator for consistent error handling.
    
    Usage:
        @handle_errors(logger_name="trading", reraise=False, default=None)
        def process_order(order):
            # Process order
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = logging.getLogger(logger_name) if logger_name else logger
            
            try:
                return func(*args, **kwargs)
            except ArgusException as e:
                # Already properly formatted
                if e.recoverable:
                    log.warning(f"[{e.code}] {e.message}")
                else:
                    log.error(f"[{e.code}] {e.message}", exc_info=True)
                
                if reraise:
                    raise
                return default
            except Exception as e:
                # Wrap in generic Argus exception
                log.error(f"Unexpected error: {e}", exc_info=True)
                exception_manager.handle_exception(e)
                
                if reraise:
                    raise ArgusException(f"Unexpected error: {e}") from e
                return default
        
        return wrapper
    return decorator
