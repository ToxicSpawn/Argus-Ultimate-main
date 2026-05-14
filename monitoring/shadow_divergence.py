#!/usr/bin/env python3
"""
Shadow divergence tracker — compares real trades against shadow trades
to detect systematic differences in fill rate, P&L, and timing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DivergenceReport:
    """Summary of divergences between real and shadow execution."""

    fill_rate_divergence_pct: float
    pnl_divergence_usd: float
    timing_divergence_ms: float
    alert_triggered: bool


@dataclass
class _TradeRecord:
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: float
    filled: bool


class ShadowDivergenceTracker:
    """Track divergence between real and shadow execution paths."""

    def __init__(self, config: Any = None) -> None:
        self._config = config
        self._alert_threshold_pct: float = float(
            _cfg(config, "shadow_execution.divergence_alert_threshold_pct", 10.0)
        )

        self._real_trades: List[_TradeRecord] = []
        self._shadow_trades: List[_TradeRecord] = []

        logger.info(
            "ShadowDivergenceTracker initialised  alert_threshold=%.1f%%",
            self._alert_threshold_pct,
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_real_trade(self, trade_data: dict) -> None:
        """Record a real trade for divergence analysis."""
        try:
            rec = _TradeRecord(
                symbol=trade_data.get("symbol", ""),
                side=trade_data.get("side", "buy"),
                quantity=float(trade_data.get("quantity", 0.0)),
                price=float(trade_data.get("price", 0.0)),
                timestamp=float(trade_data.get("timestamp", time.time())),
                filled=bool(trade_data.get("filled", True)),
            )
            self._real_trades.append(rec)
        except Exception:
            logger.exception("Failed to record real trade")

    def record_shadow_trade(self, shadow: Any) -> None:
        """Record a shadow trade (ShadowTrade dataclass or dict)."""
        try:
            if isinstance(shadow, dict):
                data = shadow
            else:
                # Assume ShadowTrade dataclass
                data = {
                    "symbol": getattr(shadow, "symbol", ""),
                    "side": getattr(shadow, "side", "buy"),
                    "quantity": float(getattr(shadow, "quantity", 0.0)),
                    "price": float(getattr(shadow, "hypothetical_price", 0.0)),
                    "timestamp": float(getattr(shadow, "timestamp", time.time())),
                    "filled": bool(getattr(shadow, "would_have_filled", True)),
                }

            rec = _TradeRecord(
                symbol=data.get("symbol", ""),
                side=data.get("side", "buy"),
                quantity=float(data.get("quantity", 0.0)),
                price=float(data.get("price", 0.0)),
                timestamp=float(data.get("timestamp", time.time())),
                filled=bool(data.get("filled", True)),
            )
            self._shadow_trades.append(rec)
        except Exception:
            logger.exception("Failed to record shadow trade")

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def calculate_divergence(self) -> DivergenceReport:
        """Compare real vs shadow trades and return a divergence report."""

        # Fill rate divergence
        real_fill_rate = self._fill_rate(self._real_trades)
        shadow_fill_rate = self._fill_rate(self._shadow_trades)
        fill_div = abs(real_fill_rate - shadow_fill_rate) * 100.0

        # P&L divergence (signed notional sum)
        real_pnl = self._signed_notional(self._real_trades)
        shadow_pnl = self._signed_notional(self._shadow_trades)
        pnl_div = abs(real_pnl - shadow_pnl)

        # Timing divergence (average timestamp gap for matched trades)
        timing_div_ms = self._timing_divergence()

        # Alert check
        alert = fill_div > self._alert_threshold_pct or (
            pnl_div > 0 and real_pnl != 0 and abs(pnl_div / abs(real_pnl)) * 100 > self._alert_threshold_pct
        )

        report = DivergenceReport(
            fill_rate_divergence_pct=round(fill_div, 2),
            pnl_divergence_usd=round(pnl_div, 2),
            timing_divergence_ms=round(timing_div_ms, 2),
            alert_triggered=alert,
        )

        if alert:
            logger.warning("Shadow divergence ALERT: %s", report)
        else:
            logger.info("Shadow divergence: %s", report)

        return report

    def clear(self) -> None:
        """Reset all recorded trades."""
        self._real_trades.clear()
        self._shadow_trades.clear()

    @property
    def real_count(self) -> int:
        return len(self._real_trades)

    @property
    def shadow_count(self) -> int:
        return len(self._shadow_trades)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_rate(trades: List[_TradeRecord]) -> float:
        if not trades:
            return 0.0
        filled = sum(1 for t in trades if t.filled)
        return filled / len(trades)

    @staticmethod
    def _signed_notional(trades: List[_TradeRecord]) -> float:
        total = 0.0
        for t in trades:
            if not t.filled:
                continue
            notional = t.quantity * t.price
            if t.side == "sell":
                total += notional
            else:
                total -= notional
        return total

    def _timing_divergence(self) -> float:
        """Match trades by index and compute avg timestamp delta in ms."""
        if not self._real_trades or not self._shadow_trades:
            return 0.0
        pairs = min(len(self._real_trades), len(self._shadow_trades))
        total_ms = 0.0
        for i in range(pairs):
            delta = abs(self._real_trades[i].timestamp - self._shadow_trades[i].timestamp)
            total_ms += delta * 1000.0
        return total_ms / pairs


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
