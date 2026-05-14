#!/usr/bin/env python3
"""
Enhanced Trade Attribution V2 — full P&L decomposition into alpha, beta,
timing, execution, slippage, fees, funding, and FX impact.

Builds on the original AttributionEngine with richer decomposition, per-strategy
alpha tracking, cost breakdowns, and actionable improvement suggestions.

Persistence: SQLite at ``data/attribution_v2.db``.

Usage::

    attr = TradeAttributionV2()
    attr.record_trade({
        "symbol": "BTC/USD", "strategy": "momentum",
        "side": "buy", "quantity": 0.01,
        "entry_price": 60000, "exit_price": 61000,
        "slippage_bps": 3.0, "fees_usd": 1.5,
        "funding_usd": -0.25, "fx_impact_usd": 0.0,
        "market_return_pct": 1.2,
    })
    decomp = attr.decompose_pnl(lookback_days=30)
    logger.info(decomp.alpha, decomp.net_pnl)
"""
from __future__ import annotations

import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = Path("data")
_DEFAULT_DB_NAME = "attribution_v2.db"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PnLDecomposition:
    """Full P&L decomposition for a lookback period."""

    gross_pnl: float = 0.0          # raw price change * quantity (USD)
    market_beta: float = 0.0        # P&L attributable to market movement
    alpha: float = 0.0              # skill-based P&L (gross - beta)
    timing: float = 0.0             # entry/exit timing contribution
    execution_cost: float = 0.0     # total execution drag (slippage + fees)
    slippage_cost: float = 0.0      # slippage component
    fee_cost: float = 0.0           # exchange fee component
    funding_cost: float = 0.0       # funding/swap rate cost
    fx_impact: float = 0.0          # currency conversion impact
    net_pnl: float = 0.0            # gross - all costs
    trade_count: int = 0
    lookback_days: int = 30


@dataclass
class TradeRecord:
    """Internal representation of a recorded trade."""

    ts: float
    symbol: str
    strategy: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    market_return_pct: float
    slippage_bps: float
    fees_usd: float
    funding_usd: float
    fx_impact_usd: float


# ---------------------------------------------------------------------------
# Trade Attribution V2
# ---------------------------------------------------------------------------


class TradeAttributionV2:
    """Enhanced trade attribution engine with full P&L decomposition.

    Parameters
    ----------
    db_path : str or Path, optional
        SQLite database path.  Defaults to ``data/attribution_v2.db``.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            self._db_path = _DEFAULT_DB_DIR / _DEFAULT_DB_NAME
        else:
            self._db_path = Path(db_path)

        self._lock = threading.Lock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("TradeAttributionV2 initialised — db=%s", self._db_path)

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trades_v2 (
                        id                INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts                REAL    NOT NULL,
                        symbol            TEXT    NOT NULL,
                        strategy          TEXT    NOT NULL DEFAULT 'unknown',
                        side              TEXT    NOT NULL,
                        quantity          REAL    NOT NULL,
                        entry_price       REAL    NOT NULL,
                        exit_price        REAL    NOT NULL,
                        gross_pnl         REAL    NOT NULL,
                        market_return_pct REAL    NOT NULL DEFAULT 0,
                        slippage_bps      REAL    NOT NULL DEFAULT 0,
                        fees_usd          REAL    NOT NULL DEFAULT 0,
                        funding_usd       REAL    NOT NULL DEFAULT 0,
                        fx_impact_usd     REAL    NOT NULL DEFAULT 0
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trades_v2_ts
                    ON trades_v2 (ts)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trades_v2_strategy
                    ON trades_v2 (strategy)
                """)
                conn.commit()
            finally:
                conn.close()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_trade(self, trade: Dict[str, Any]) -> None:
        """Record a completed trade with full context.

        Parameters
        ----------
        trade : dict
            Required keys: ``symbol``, ``side``, ``quantity``, ``entry_price``,
            ``exit_price``.
            Optional keys: ``strategy``, ``market_return_pct``, ``slippage_bps``,
            ``fees_usd``, ``funding_usd``, ``fx_impact_usd``, ``timestamp``.
        """
        symbol = str(trade.get("symbol", ""))
        side = str(trade.get("side", "buy")).lower()
        quantity = float(trade.get("quantity", 0))
        entry = float(trade.get("entry_price", 0))
        exit_ = float(trade.get("exit_price", 0))
        strategy = str(trade.get("strategy", "unknown"))

        # Gross P&L
        if side == "buy":
            gross_pnl = (exit_ - entry) * quantity
        else:
            gross_pnl = (entry - exit_) * quantity

        market_ret = float(trade.get("market_return_pct", 0))
        slippage = float(trade.get("slippage_bps", 0))
        fees = float(trade.get("fees_usd", 0))
        funding = float(trade.get("funding_usd", 0))
        fx = float(trade.get("fx_impact_usd", 0))
        ts = float(trade.get("timestamp", time.time()))

        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    """
                    INSERT INTO trades_v2
                        (ts, symbol, strategy, side, quantity, entry_price,
                         exit_price, gross_pnl, market_return_pct,
                         slippage_bps, fees_usd, funding_usd, fx_impact_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (ts, symbol, strategy, side, quantity, entry, exit_,
                     gross_pnl, market_ret, slippage, fees, funding, fx),
                )
                conn.commit()
            finally:
                conn.close()

        logger.debug(
            "TradeAttributionV2: recorded %s %s %s — gross=%.2f, strat=%s",
            symbol, side, quantity, gross_pnl, strategy,
        )

    # ------------------------------------------------------------------
    # Decomposition
    # ------------------------------------------------------------------

    def _load_trades(self, lookback_days: int) -> List[TradeRecord]:
        cutoff = time.time() - lookback_days * 86400
        conn = self._conn()
        try:
            rows = conn.execute(
                """
                SELECT ts, symbol, strategy, side, quantity, entry_price,
                       exit_price, gross_pnl, market_return_pct,
                       slippage_bps, fees_usd, funding_usd, fx_impact_usd
                FROM trades_v2
                WHERE ts >= ?
                ORDER BY ts ASC
                """,
                (cutoff,),
            ).fetchall()
        finally:
            conn.close()

        return [
            TradeRecord(
                ts=r[0], symbol=r[1], strategy=r[2], side=r[3],
                quantity=r[4], entry_price=r[5], exit_price=r[6],
                gross_pnl=r[7], market_return_pct=r[8],
                slippage_bps=r[9], fees_usd=r[10],
                funding_usd=r[11], fx_impact_usd=r[12],
            )
            for r in rows
        ]

    def decompose_pnl(self, lookback_days: int = 30) -> PnLDecomposition:
        """Decompose P&L into components over a lookback period.

        Parameters
        ----------
        lookback_days : int
            Number of days to look back.

        Returns
        -------
        PnLDecomposition
            Full decomposition with gross, beta, alpha, costs, net.
        """
        trades = self._load_trades(lookback_days)
        if not trades:
            return PnLDecomposition(lookback_days=lookback_days)

        gross_pnl = sum(t.gross_pnl for t in trades)
        slippage_cost = 0.0
        fee_cost = sum(t.fees_usd for t in trades)
        funding_cost = sum(t.funding_usd for t in trades)
        fx_impact = sum(t.fx_impact_usd for t in trades)

        # Slippage cost: convert bps to USD using trade notional
        for t in trades:
            notional = abs(t.quantity * t.entry_price)
            slippage_cost += notional * t.slippage_bps / 10_000.0

        execution_cost = slippage_cost + fee_cost

        # Beta: what you'd have earned from market exposure alone
        market_beta = 0.0
        for t in trades:
            notional = abs(t.quantity * t.entry_price)
            sign = 1.0 if t.side == "buy" else -1.0
            market_beta += sign * notional * (t.market_return_pct / 100.0)

        # Alpha = gross P&L minus market beta
        alpha = gross_pnl - market_beta

        # Timing: difference between ideal entry (open) and actual entry
        # Approximated as residual after alpha and beta
        timing = 0.0  # TODO: requires OHLCV data for proper calculation

        net_pnl = gross_pnl - slippage_cost - fee_cost - funding_cost - fx_impact

        decomp = PnLDecomposition(
            gross_pnl=round(gross_pnl, 2),
            market_beta=round(market_beta, 2),
            alpha=round(alpha, 2),
            timing=round(timing, 2),
            execution_cost=round(execution_cost, 2),
            slippage_cost=round(slippage_cost, 2),
            fee_cost=round(fee_cost, 2),
            funding_cost=round(funding_cost, 2),
            fx_impact=round(fx_impact, 2),
            net_pnl=round(net_pnl, 2),
            trade_count=len(trades),
            lookback_days=lookback_days,
        )

        logger.info(
            "TradeAttributionV2: decomposition — gross=$%.2f, alpha=$%.2f, "
            "beta=$%.2f, exec_cost=$%.2f, net=$%.2f (%d trades, %dd)",
            decomp.gross_pnl, decomp.alpha, decomp.market_beta,
            decomp.execution_cost, decomp.net_pnl,
            decomp.trade_count, lookback_days,
        )
        return decomp

    # ------------------------------------------------------------------
    # Per-strategy alpha
    # ------------------------------------------------------------------

    def get_alpha_by_strategy(self, lookback_days: int = 30) -> Dict[str, float]:
        """Return alpha contribution per strategy.

        Returns
        -------
        dict
            strategy_name → alpha in USD.
        """
        trades = self._load_trades(lookback_days)
        if not trades:
            return {}

        by_strategy: Dict[str, List[TradeRecord]] = {}
        for t in trades:
            by_strategy.setdefault(t.strategy, []).append(t)

        result: Dict[str, float] = {}
        for strategy, strades in by_strategy.items():
            gross = sum(t.gross_pnl for t in strades)
            beta = 0.0
            for t in strades:
                notional = abs(t.quantity * t.entry_price)
                sign = 1.0 if t.side == "buy" else -1.0
                beta += sign * notional * (t.market_return_pct / 100.0)
            result[strategy] = round(gross - beta, 2)

        logger.info("TradeAttributionV2: alpha by strategy → %s", result)
        return result

    # ------------------------------------------------------------------
    # Cost breakdown
    # ------------------------------------------------------------------

    def get_cost_breakdown(self, lookback_days: int = 30) -> Dict[str, float]:
        """Return total costs by category.

        Returns
        -------
        dict
            cost_type → total_usd (positive = cost).
        """
        trades = self._load_trades(lookback_days)
        if not trades:
            return {}

        slippage = 0.0
        for t in trades:
            notional = abs(t.quantity * t.entry_price)
            slippage += notional * t.slippage_bps / 10_000.0

        return {
            "slippage": round(slippage, 2),
            "fees": round(sum(t.fees_usd for t in trades), 2),
            "funding": round(sum(t.funding_usd for t in trades), 2),
            "fx_impact": round(sum(t.fx_impact_usd for t in trades), 2),
            "total": round(
                slippage
                + sum(t.fees_usd for t in trades)
                + sum(t.funding_usd for t in trades)
                + sum(t.fx_impact_usd for t in trades),
                2,
            ),
        }

    # ------------------------------------------------------------------
    # Improvement suggestions
    # ------------------------------------------------------------------

    def get_improvement_suggestions(
        self, lookback_days: int = 30
    ) -> List[str]:
        """Generate actionable suggestions based on attribution data.

        Returns
        -------
        list of str
            Human-readable improvement suggestions.
        """
        decomp = self.decompose_pnl(lookback_days)
        costs = self.get_cost_breakdown(lookback_days)
        alpha_by_strat = self.get_alpha_by_strategy(lookback_days)
        suggestions: List[str] = []

        if decomp.trade_count == 0:
            return ["No trades in lookback period — cannot generate suggestions."]

        # High slippage
        if costs.get("slippage", 0) > 0:
            avg_slip_per_trade = costs["slippage"] / decomp.trade_count
            if avg_slip_per_trade > 2.0:
                suggestions.append(
                    f"High average slippage (${avg_slip_per_trade:.2f}/trade). "
                    f"Consider using limit orders or TWAP execution for larger orders."
                )

        # High fees
        if costs.get("fees", 0) > 0:
            fee_pct = (costs["fees"] / max(abs(decomp.gross_pnl), 1.0)) * 100
            if fee_pct > 20:
                suggestions.append(
                    f"Fees consume {fee_pct:.0f}% of gross P&L. "
                    f"Upgrade exchange tier or use maker orders to reduce fees."
                )

        # Negative alpha strategies
        for strat, alpha_val in alpha_by_strat.items():
            if alpha_val < -10.0:
                suggestions.append(
                    f"Strategy '{strat}' has negative alpha (${alpha_val:.2f}). "
                    f"Review signal quality or reduce allocation."
                )

        # Execution cost vs alpha
        if decomp.execution_cost > 0 and decomp.alpha > 0:
            exec_alpha_ratio = decomp.execution_cost / decomp.alpha
            if exec_alpha_ratio > 0.5:
                suggestions.append(
                    f"Execution costs consume {exec_alpha_ratio * 100:.0f}% of alpha. "
                    f"Focus on reducing slippage and fees to capture more edge."
                )

        # Funding costs
        if abs(costs.get("funding", 0)) > 5.0:
            suggestions.append(
                f"Funding costs total ${abs(costs['funding']):.2f}. "
                f"Monitor funding rates and consider closing perp positions "
                f"before high-funding periods."
            )

        # Beta dominance
        if decomp.market_beta != 0 and decomp.alpha != 0:
            if abs(decomp.market_beta) > abs(decomp.alpha) * 3:
                suggestions.append(
                    "P&L is dominated by market beta rather than alpha. "
                    "Consider delta-hedging or focusing on market-neutral strategies."
                )

        if not suggestions:
            suggestions.append(
                "Attribution looks healthy. Continue monitoring for degradation."
            )

        logger.info(
            "TradeAttributionV2: %d improvement suggestions generated",
            len(suggestions),
        )
        return suggestions
