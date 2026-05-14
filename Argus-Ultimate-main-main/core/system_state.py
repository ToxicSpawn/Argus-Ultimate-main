#!/usr/bin/env python3
"""
core/system_state.py
====================
SystemState enum and _ArgusJsonFormatter — extracted from unified_trading_system.py.

These two objects have no runtime dependencies beyond stdlib and are safe
to import anywhere without triggering the full unified_trading_system module
load (which pulls in ccxt, numpy, pandas, etc.).

Backward-compat re-export: core/__init__.py exposes both.
"""
from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Dict


class SystemState(Enum):
    """System operational states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    SHUTDOWN = "shutdown"


class _ArgusJsonFormatter(logging.Formatter):
    """Minimal JSON formatter for Loki-friendly structured logging.

    Each emitted line is a JSON object with at minimum:
        timestamp, level, logger, message
    plus any extra fields passed via the ``extra=`` kwarg on the logger call.

    Activated by setting the env var ARGUS_JSON_LOGS=1 before startup.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        entry: Dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%f"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Stdlib LogRecord attributes that must not be forwarded as extras.
        _STDLIB_ATTRS = frozenset(
            logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
        ) | {"message", "asctime"}
        for k, v in record.__dict__.items():
            if k not in _STDLIB_ATTRS and not k.startswith("_"):
                try:
                    json.dumps(v)  # only include JSON-serialisable extras
                    entry[k] = v
                except (TypeError, ValueError):
                    entry[k] = str(v)
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)
