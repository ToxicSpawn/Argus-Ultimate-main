#!/usr/bin/env python3
"""
Argus Logger - S+ Tier
Advanced logging system with structured logging, performance monitoring, and alerting.
"""

import logging
import logging.handlers
import json
import time
import queue
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import threading
import os
import atexit


@dataclass
class LogEntry:
    """Structured log entry"""

    timestamp: float
    level: str
    module: str
    message: str
    extra_data: Optional[Dict[str, Any]] = None


class ArgusFormatter(logging.Formatter):
    """Custom formatter for Argus logs"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured data"""
        # Add timestamp if not present
        if not hasattr(record, "timestamp"):
            record.timestamp = time.time()

        # Correlation ID prefix (if set)
        cid = getattr(record, "correlation_id", "") or ""
        cid_str = f" [{cid}]" if cid else ""

        # Base format
        base_format = f"[{record.levelname}]{cid_str} {record.name}: {record.getMessage()}"

        # Add extra fields if present
        if hasattr(record, "extra_data") and record.extra_data:
            extra_str = " | ".join(f"{k}={v}" for k, v in record.extra_data.items())
            base_format += f" | {extra_str}"

        return base_format


class CorrelationFilter(logging.Filter):
    """Injects the current cycle correlation ID into every log record."""

    _correlation_id: str = ""

    @classmethod
    def set_correlation_id(cls, cid: str) -> None:
        cls._correlation_id = str(cid)

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = self._correlation_id  # type: ignore[attr-defined]
        return True


class AsyncQueueHandler(logging.Handler):
    """Non-blocking handler that queues records and writes from a background thread.

    Prevents file I/O from stalling the asyncio event loop.
    """

    def __init__(self, target_handler: logging.Handler, max_queue: int = 10_000):
        super().__init__()
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._target = target_handler
        self._thread = threading.Thread(target=self._drain, daemon=True, name="async-logger")
        self._shutdown = threading.Event()
        self._thread.start()
        atexit.register(self.close)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            pass  # drop on overflow rather than block

    def _drain(self) -> None:
        while not self._shutdown.is_set():
            try:
                record = self._queue.get(timeout=0.5)
                self._target.emit(record)
            except queue.Empty:
                continue
            except Exception as _e:
                logger.debug("logger error: %s", _e)
        # Flush remaining
        while not self._queue.empty():
            try:
                self._target.emit(self._queue.get_nowait())
            except Exception:
                break

    def close(self) -> None:
        self._shutdown.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)
        super().close()


class ArgusLogger:
    """
    Argus Logger - S+ Tier
    Advanced logging with structured output, performance monitoring, and alerting.
    """

    def __init__(self, name: str = "argus", log_level: str = "INFO"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Prevent duplicate handlers
        if self.logger.handlers:
            return

        # Correlation ID filter (injects cycle correlation ID into records)
        self._corr_filter = CorrelationFilter()
        self.logger.addFilter(self._corr_filter)

        # Console handler (direct — console writes are fast)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ArgusFormatter())
        self.logger.addHandler(console_handler)

        # File handler (rotating, wrapped in async queue to avoid blocking event loop)
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        _raw_file_handler = logging.handlers.RotatingFileHandler(
            f"{log_dir}/{name}.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        _raw_file_handler.setFormatter(ArgusFormatter())
        async_handler = AsyncQueueHandler(_raw_file_handler)
        async_handler.setFormatter(ArgusFormatter())
        self.logger.addHandler(async_handler)

        # Performance tracking
        self.log_count = 0
        self.error_count = 0
        self.warning_count = 0

        self.logger.info(f"Argus Logger initialized: {name}")

    def log(self, level: str, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a message with optional extra data.

        Args:
            level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            message: Log message
            extra_data: Additional structured data
        """
        log_method = getattr(self.logger, level.lower(), self.logger.info)

        self.log_count += 1

        if level.upper() == "ERROR":
            self.error_count += 1
        elif level.upper() == "WARNING":
            self.warning_count += 1

        if extra_data:
            log_method(message, extra={"extra_data": extra_data})
        else:
            log_method(message)

    def debug(self, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message"""
        self.log("DEBUG", message, extra_data)

    def info(self, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log info message"""
        self.log("INFO", message, extra_data)

    def warning(self, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message"""
        self.log("WARNING", message, extra_data)

    def error(self, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log error message"""
        self.log("ERROR", message, extra_data)

    def critical(self, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log critical message"""
        self.log("CRITICAL", message, extra_data)

    def trade_log(self, symbol: str, side: str, quantity: float, price: float, pnl: Optional[float] = None) -> None:
        """
        Log trade execution.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Trade quantity
            price: Trade price
            pnl: Profit/loss (optional)
        """
        extra_data = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
        }

        if pnl is not None:
            extra_data["pnl"] = pnl

        self.info(f"Trade executed: {symbol} {side} {quantity} @ ${price:.2f}", extra_data)

    def performance_log(self, component: str, operation: str, duration_ms: float, success: bool = True) -> None:
        """
        Log performance metrics.

        Args:
            component: Component name
            operation: Operation name
            duration_ms: Duration in milliseconds
            success: Whether operation was successful
        """
        extra_data = {
            "component": component,
            "operation": operation,
            "duration_ms": duration_ms,
            "success": success,
        }

        if success:
            self.debug(f"Performance: {component}.{operation} took {duration_ms:.2f}ms", extra_data)
        else:
            self.warning(f"Performance: {component}.{operation} failed after {duration_ms:.2f}ms", extra_data)

    def alert(self, alert_type: str, message: str, severity: str = "medium") -> None:
        """
        Log system alert.

        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity ('low', 'medium', 'high', 'critical')
        """
        extra_data = {
            "alert_type": alert_type,
            "severity": severity,
            "timestamp": time.time(),
        }

        if severity == "critical":
            self.critical(f"🚨 ALERT: {message}", extra_data)
        elif severity == "high":
            self.error(f"⚠️ ALERT: {message}", extra_data)
        elif severity == "medium":
            self.warning(f"⚠️ ALERT: {message}", extra_data)
        else:
            self.info(f"ℹ️ ALERT: {message}", extra_data)

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            "total_logs": self.log_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "error_rate": self.error_count / max(1, self.log_count),
            "warning_rate": self.warning_count / max(1, self.log_count),
        }

    def set_level(self, level: str) -> None:
        """Set logging level"""
        self.logger.setLevel(getattr(logging, level.upper()))

    def flush(self) -> None:
        """Flush all handlers"""
        for handler in self.logger.handlers:
            handler.flush()


# Global logger instance
_logger_instance: Optional[ArgusLogger] = None
_logger_lock = threading.Lock()


def get_logger(name: str = "argus") -> ArgusLogger:
    """
    Get or create logger instance.

    Args:
        name: Logger name

    Returns:
        ArgusLogger instance
    """
    global _logger_instance

    if _logger_instance is None:
        with _logger_lock:
            if _logger_instance is None:
                _logger_instance = ArgusLogger(name)

    return _logger_instance


def log_trade(symbol: str, side: str, quantity: float, price: float, pnl: Optional[float] = None) -> None:
    """Convenience function for trade logging"""
    logger = get_logger()
    logger.trade_log(symbol, side, quantity, price, pnl)


def log_performance(component: str, operation: str, duration_ms: float, success: bool = True) -> None:
    """Convenience function for performance logging"""
    logger = get_logger()
    logger.performance_log(component, operation, duration_ms, success)


def log_alert(alert_type: str, message: str, severity: str = "medium") -> None:
    """Convenience function for alert logging"""
    logger = get_logger()
    logger.alert(alert_type, message, severity)
