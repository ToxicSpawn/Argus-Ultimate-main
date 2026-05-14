#!/usr/bin/env python3
"""
Shadow executor — records what trades WOULD have happened without placing real orders.

Runs the same signal-processing and risk-check pipeline as the live executor,
persists hypothetical fills to SQLite, and exposes shadow P&L analytics.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ShadowTrade:
    """A hypothetical trade that would have been placed."""

    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    hypothetical_price: float
    timestamp: float
    would_have_filled: bool
    reason: str


class ShadowExecutor:
    """Process signals in shadow mode — no real orders, full analytics."""

    def __init__(
        self,
        config: Any = None,
        risk_manager: Any = None,
    ) -> None:
        self._config = config
        self._risk_manager = risk_manager
        self._enabled: bool = bool(
            _cfg(config, "shadow_execution.enabled", False)
        )

        db_path_str = str(
            _cfg(config, "shadow_execution.db_path", "data/shadow_trades.db")
        )
        self._db_path = Path(db_path_str)
        self._lock = threading.Lock()
        self._trades: List[ShadowTrade] = []

        self._init_db()
        logger.info(
            "ShadowExecutor initialised  enabled=%s  db=%s",
            self._enabled,
            self._db_path,
        )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the shadow_trades table if it doesn't exist."""
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS shadow_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        hypothetical_price REAL NOT NULL,
                        timestamp REAL NOT NULL,
                        would_have_filled INTEGER NOT NULL,
                        reason TEXT
                    )
                    """
                )
                conn.commit()
        except Exception:
            logger.exception("Failed to initialise shadow trades DB")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_signals(self, signals: list) -> List[ShadowTrade]:
        """Process a list of signal dicts and return shadow trades."""
        if not self._enabled:
            return []

        results: List[ShadowTrade] = []

        for sig in signals:
            try:
                shadow = self._evaluate_signal(sig)
                if shadow is not None:
                    results.append(shadow)
                    self._persist(shadow)
            except Exception:
                logger.exception("Shadow signal evaluation failed: %s", sig)

        with self._lock:
            self._trades.extend(results)

        if results:
            logger.info("Shadow executor processed %d signals -> %d trades", len(signals), len(results))

        return results

    def get_shadow_pnl(self, days: int = 7) -> float:
        """Calculate hypothetical P&L over the last N days from persisted trades."""
        cutoff = time.time() - (days * 86400)
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                rows = conn.execute(
                    """
                    SELECT side, quantity, hypothetical_price
                    FROM shadow_trades
                    WHERE timestamp >= ? AND would_have_filled = 1
                    ORDER BY timestamp ASC
                    """,
                    (cutoff,),
                ).fetchall()
        except Exception:
            logger.exception("Failed to read shadow trades for P&L")
            return 0.0

        # Simplified P&L: sum signed notional (buy positive exposure, sell closes)
        pnl = 0.0
        for side, qty, price in rows:
            notional = qty * price
            if side == "sell":
                pnl += notional
            else:
                pnl -= notional

        return round(pnl, 2)

    @property
    def trade_count(self) -> int:
        with self._lock:
            return len(self._trades)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evaluate_signal(self, sig: dict) -> Optional[ShadowTrade]:
        """Evaluate a single signal dict and return a ShadowTrade or None."""
        symbol = sig.get("symbol", "")
        side = sig.get("side", sig.get("direction", "buy"))
        quantity = float(sig.get("quantity", sig.get("size", 0.0)))
        price = float(sig.get("price", sig.get("current_price", 0.0)))
        confidence = float(sig.get("confidence", sig.get("strength", 0.5)))

        if not symbol or quantity <= 0:
            return None

        # Risk check (optional)
        would_fill = True
        reason = "shadow_fill"
        if self._risk_manager is not None:
            try:
                allowed = getattr(self._risk_manager, "check_order", lambda *a, **k: True)(
                    symbol=symbol, side=side, quantity=quantity
                )
                if not allowed:
                    would_fill = False
                    reason = "risk_rejected"
            except Exception:
                logger.debug("Risk check unavailable for shadow trade")

        # Apply hypothetical slippage (0.1%)
        slippage = 0.001
        if side == "buy":
            hypo_price = price * (1.0 + slippage)
        else:
            hypo_price = price * (1.0 - slippage)

        return ShadowTrade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            hypothetical_price=round(hypo_price, 6),
            timestamp=time.time(),
            would_have_filled=would_fill,
            reason=reason,
        )

    def _persist(self, trade: ShadowTrade) -> None:
        """Write a shadow trade to SQLite."""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    """
                    INSERT INTO shadow_trades
                        (symbol, side, quantity, hypothetical_price, timestamp, would_have_filled, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade.symbol,
                        trade.side,
                        trade.quantity,
                        trade.hypothetical_price,
                        trade.timestamp,
                        int(trade.would_have_filled),
                        trade.reason,
                    ),
                )
                conn.commit()
        except Exception:
            logger.exception("Failed to persist shadow trade")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(config: Any, dotted_key: str, default: Any) -> Any:
    if config is None:
        return default
    parts = dotted_key.split(".")
    obj = config
    for part in parts:
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None)
        if obj is None:
            return default
    return obj
