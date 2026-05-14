"""
Logging utilities (import-safe).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# Backwards-compat: some legacy modules do `from utils.logger import logger`.
logger = logging.getLogger("argus")


def _try_color_formatter() -> Optional[logging.Formatter]:
    try:
        import colorlog  # type: ignore

        return colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    except Exception:
        return None


class ArgusLogger:
    _loggers: Dict[str, logging.Logger] = {}

    @classmethod
    def get_logger(cls, name: str, log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))
        logger.handlers.clear()

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(_try_color_formatter() or logging.Formatter("%(asctime)s %(name)-20s %(levelname)-8s %(message)s"))
        logger.addHandler(ch)

        # Optional file handler
        if log_file:
            Path("logs").mkdir(exist_ok=True)
            fh = logging.FileHandler(str(Path("logs") / str(log_file)))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter("%(asctime)s %(name)-20s %(levelname)-8s %(message)s"))
            logger.addHandler(fh)

        cls._loggers[name] = logger
        return logger


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    log_file = f"{name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.log"
    return ArgusLogger.get_logger(name, log_level, log_file=log_file)


def log_trade(
    logger: logging.Logger,
    action: str,
    symbol: str,
    side: str,
    price: float,
    quantity: float,
    **kwargs: Any,
) -> None:
    msg = f"TRADE {action} | {symbol} | {side} Price=${price:,.2f} Qty={quantity:.8f}"
    if kwargs:
        msg += " | " + " | ".join(f"{k}: {v}" for k, v in kwargs.items())
    logger.info(msg)


def log_order(
    logger: logging.Logger,
    action: str,
    order_id: str,
    symbol: str,
    side: str,
    order_type: str,
    **kwargs: Any,
) -> None:
    msg = f"ORDER {action} ID={order_id} | {symbol} | {side} {order_type}"
    if kwargs:
        msg += " | " + " | ".join(f"{k}: {v}" for k, v in kwargs.items())
    logger.info(msg)


def log_pnl(logger: logging.Logger, symbol: str, realized_pnl: float, unrealized_pnl: float, total_equity: float) -> None:
    msg = f"PNL {symbol} Realized=${realized_pnl:,.2f} Unrealized=${unrealized_pnl:,.2f} Equity=${total_equity:,.2f}"
    logger.info(msg)
