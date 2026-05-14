"""M07 — Structured JSON logging via structlog.

Call ``configure_logging()`` once at process startup (before any log output).
Falls back to standard ``logging`` if structlog is not installed.
"""
from __future__ import annotations

import logging
import logging.config
import sys
from typing import Any

_STRUCTLOG_AVAILABLE: bool
try:
    import structlog  # type: ignore[import-untyped]
    _STRUCTLOG_AVAILABLE = True
except ImportError:
    _STRUCTLOG_AVAILABLE = False


def configure_logging(
    level: str = "INFO",
    json_output: bool = True,
    service_name: str = "argus",
) -> None:
    """Configure root logger and (optionally) structlog JSON renderer.

    Args:
        level: Logging level string (DEBUG/INFO/WARNING/ERROR/CRITICAL).
        json_output: Emit JSON lines when True; human-readable when False.
        service_name: Value of the ``service`` key in every JSON log line.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # ── stdlib baseline ──────────────────────────────────────────────────────
    logging.basicConfig(
        stream=sys.stdout,
        level=numeric_level,
        format="%(message)s",
    )
    logging.getLogger().setLevel(numeric_level)

    if not _STRUCTLOG_AVAILABLE or not json_output:
        # Plain human-readable fallback
        fmt = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
        logging.basicConfig(stream=sys.stdout, level=numeric_level, format=fmt, force=True)
        return

    # ── structlog JSON pipeline ───────────────────────────────────────────────
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also bind service name to every log record
    structlog.contextvars.bind_contextvars(service=service_name)


def get_logger(name: str) -> Any:
    """Return a structlog (or stdlib) logger for *name*."""
    if _STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    return logging.getLogger(name)
