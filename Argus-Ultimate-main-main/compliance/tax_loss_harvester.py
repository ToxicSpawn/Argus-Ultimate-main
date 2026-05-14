#!/usr/bin/env python3
"""
Tax-Loss Harvesting Automation — Australian tax-optimised loss realisation.

Scans open positions for unrealised losses, estimates the AUD tax saving at
the configured marginal rate, detects wash-sale risk (30-day ATO "bed and
breakfast" window), and persists harvest history for FY reporting.

Key assumptions:
- Marginal tax rate defaults to **32.5%** (AU $45k–$120k bracket).
- Wash-sale detection uses a **30-day lookback** (conservative; AU law is
  less strict than US IRC §1091 but the ATO may flag aggressive patterns).
- All monetary values are in AUD.

Persistence: SQLite at ``data/tax_harvest.db`` (configurable).

Usage::

    harvester = TaxLossHarvester()
    opportunities = harvester.scan_positions(positions, current_prices)
    for opp in opportunities:
        if not opp.wash_sale_risk:
            harvester.execute_harvest(opp, replacement_symbol="ETH/AUD")
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HarvestOpportunity:
    """A position eligible for tax-loss harvesting."""

    symbol: str
    quantity: float
    entry_price_aud: float
    current_price_aud: float
    unrealized_loss_aud: float
    tax_saving_aud: float
    wash_sale_risk: bool
    days_held: int


@dataclass
class HarvestAction:
    """Record of an executed harvest."""

    harvest_id: int
    symbol: str
    quantity: float
    sale_price_aud: float
    loss_realised_aud: float
    tax_saving_aud: float
    replacement_symbol: Optional[str]
    timestamp: float


# ---------------------------------------------------------------------------
# Tax-Loss Harvester
# ---------------------------------------------------------------------------


class TaxLossHarvester:
    """Automated tax-loss harvesting for Australian crypto positions.

    Parameters
    ----------
    db_path:
        SQLite database for harvest history.  Defaults to
        ``data/tax_harvest.db`` relative to the repo root.
    tax_rate:
        Marginal income tax rate.  Default 0.325 (32.5%).
    wash_sale_lookback_days:
        Number of days to look back for wash-sale detection.
    min_loss_aud:
        Minimum unrealised loss (AUD) to consider harvesting.
    financial_year_start_month:
        Month number when the Australian financial year starts (default 7 = July).
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        tax_rate: float = 0.325,
        wash_sale_lookback_days: int = 30,
        min_loss_aud: float = 10.0,
        financial_year_start_month: int = 7,
    ) -> None:
        if db_path is None:
            repo_root = Path(__file__).resolve().parent.parent
            db_path = str(repo_root / "data" / "tax_harvest.db")

        self._db_path = db_path
        self.tax_rate = tax_rate
        self.wash_sale_lookback_days = wash_sale_lookback_days
        self.min_loss_aud = min_loss_aud
        self.fy_start_month = financial_year_start_month

        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._conn = self._connect()
        self._create_tables()
        logger.info(
            "TaxLossHarvester initialised — rate=%.1f%% wash_sale=%dd min_loss=%.0f AUD db=%s",
            tax_rate * 100,
            wash_sale_lookback_days,
            min_loss_aud,
            self._db_path,
        )

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _create_tables(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS harvests (
                    harvest_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol              TEXT    NOT NULL,
                    quantity            REAL    NOT NULL,
                    entry_price_aud     REAL    NOT NULL,
                    sale_price_aud      REAL    NOT NULL,
                    loss_realised_aud   REAL    NOT NULL,
                    tax_saving_aud      REAL    NOT NULL,
                    replacement_symbol  TEXT,
                    fiscal_year         TEXT    NOT NULL,
                    timestamp           REAL    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_harvests_symbol_ts
                    ON harvests (symbol, timestamp);
                CREATE INDEX IF NOT EXISTS idx_harvests_fy
                    ON harvests (fiscal_year);
                """
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Fiscal year helper
    # ------------------------------------------------------------------

    def _get_fiscal_year(self, ts: Optional[float] = None) -> str:
        """Return fiscal year label like ``"2025-26"`` for a Unix timestamp."""
        import datetime

        dt = datetime.datetime.fromtimestamp(ts or time.time())
        if dt.month >= self.fy_start_month:
            start_year = dt.year
        else:
            start_year = dt.year - 1
        end_year = start_year + 1
        return f"{start_year}-{str(end_year)[-2:]}"

    # ------------------------------------------------------------------
    # Wash-sale detection
    # ------------------------------------------------------------------

    def _has_wash_sale_risk(self, symbol: str) -> bool:
        """Check if *symbol* was harvested within the wash-sale window."""
        cutoff = time.time() - self.wash_sale_lookback_days * 86400.0

        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM harvests WHERE symbol = ? AND timestamp > ?",
                (symbol, cutoff),
            ).fetchone()

        count: int = row[0] if row else 0
        if count > 0:
            logger.debug(
                "TaxLossHarvester wash-sale risk for %s — %d harvest(s) in last %d days",
                symbol,
                count,
                self.wash_sale_lookback_days,
            )
        return count > 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_positions(
        self,
        positions: List[Dict[str, Any]],
        current_prices: Dict[str, float],
    ) -> List[HarvestOpportunity]:
        """Scan positions for tax-loss harvesting opportunities.

        Parameters
        ----------
        positions:
            List of position dicts, each with at least:
            ``symbol``, ``quantity``, ``entry_price``, ``days_held``.
        current_prices:
            Mapping of ``symbol → current_price_aud``.

        Returns
        -------
        list[HarvestOpportunity]
            Opportunities sorted by ``tax_saving_aud`` descending.
        """
        opportunities: List[HarvestOpportunity] = []

        for pos in positions:
            symbol = pos.get("symbol", "")
            quantity = float(pos.get("quantity", 0))
            entry_price = float(pos.get("entry_price", 0))
            days_held = int(pos.get("days_held", 0))

            current_price = current_prices.get(symbol)
            if current_price is None:
                logger.debug(
                    "TaxLossHarvester: no price for %s — skipping", symbol
                )
                continue

            # Unrealised P&L
            unrealized_pnl = (current_price - entry_price) * quantity

            # Only losses
            if unrealized_pnl >= 0:
                continue

            unrealized_loss = abs(unrealized_pnl)
            if unrealized_loss < self.min_loss_aud:
                continue

            tax_saving = unrealized_loss * self.tax_rate
            wash_risk = self._has_wash_sale_risk(symbol)

            opportunities.append(
                HarvestOpportunity(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price_aud=entry_price,
                    current_price_aud=current_price,
                    unrealized_loss_aud=round(unrealized_loss, 2),
                    tax_saving_aud=round(tax_saving, 2),
                    wash_sale_risk=wash_risk,
                    days_held=days_held,
                )
            )

        # Sort by tax saving descending
        opportunities.sort(key=lambda o: o.tax_saving_aud, reverse=True)

        logger.info(
            "TaxLossHarvester scanned %d positions — %d harvest opportunities",
            len(positions),
            len(opportunities),
        )
        return opportunities

    def execute_harvest(
        self,
        opportunity: HarvestOpportunity,
        replacement_symbol: Optional[str] = None,
    ) -> HarvestAction:
        """Record a harvest execution.

        In production this would also place the sell order via the execution
        engine.  For now it records the decision for audit/reporting.

        Parameters
        ----------
        opportunity:
            The harvest opportunity to execute.
        replacement_symbol:
            Optional correlated asset to buy as a replacement (e.g. swap
            BTC/AUD for ETH/AUD to maintain market exposure).

        Returns
        -------
        HarvestAction
        """
        now = time.time()
        fy = self._get_fiscal_year(now)

        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO harvests "
                "(symbol, quantity, entry_price_aud, sale_price_aud, "
                " loss_realised_aud, tax_saving_aud, replacement_symbol, "
                " fiscal_year, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    opportunity.symbol,
                    opportunity.quantity,
                    opportunity.entry_price_aud,
                    opportunity.current_price_aud,
                    opportunity.unrealized_loss_aud,
                    opportunity.tax_saving_aud,
                    replacement_symbol,
                    fy,
                    now,
                ),
            )
            self._conn.commit()
            harvest_id: int = cur.lastrowid  # type: ignore[assignment]

        logger.info(
            "TaxLossHarvester executed harvest #%d — %s loss=%.2f AUD saving=%.2f AUD fy=%s",
            harvest_id,
            opportunity.symbol,
            opportunity.unrealized_loss_aud,
            opportunity.tax_saving_aud,
            fy,
        )

        return HarvestAction(
            harvest_id=harvest_id,
            symbol=opportunity.symbol,
            quantity=opportunity.quantity,
            sale_price_aud=opportunity.current_price_aud,
            loss_realised_aud=opportunity.unrealized_loss_aud,
            tax_saving_aud=opportunity.tax_saving_aud,
            replacement_symbol=replacement_symbol,
            timestamp=now,
        )

    def get_fy_summary(
        self, fiscal_year: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return harvest summary for a fiscal year.

        Parameters
        ----------
        fiscal_year:
            E.g. ``"2025-26"``.  Defaults to the current fiscal year.

        Returns
        -------
        dict
            Keys: ``fiscal_year``, ``total_harvested_aud``, ``tax_saved_aud``,
            ``positions_affected``, ``harvests``.
        """
        if fiscal_year is None:
            fiscal_year = self._get_fiscal_year()

        with self._lock:
            rows = self._conn.execute(
                "SELECT symbol, quantity, loss_realised_aud, tax_saving_aud, "
                "replacement_symbol, timestamp "
                "FROM harvests WHERE fiscal_year = ? ORDER BY timestamp",
                (fiscal_year,),
            ).fetchall()

        total_loss = sum(r[2] for r in rows)
        total_saving = sum(r[3] for r in rows)
        symbols_affected = set(r[0] for r in rows)

        summary = {
            "fiscal_year": fiscal_year,
            "total_harvested_aud": round(total_loss, 2),
            "tax_saved_aud": round(total_saving, 2),
            "positions_affected": len(symbols_affected),
            "harvest_count": len(rows),
            "harvests": [
                {
                    "symbol": r[0],
                    "quantity": r[1],
                    "loss_aud": r[2],
                    "saving_aud": r[3],
                    "replacement": r[4],
                    "timestamp": r[5],
                }
                for r in rows
            ],
        }

        logger.debug(
            "TaxLossHarvester FY %s summary — %d harvests, %.2f AUD saved",
            fiscal_year,
            len(rows),
            total_saving,
        )
        return summary

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()
        logger.info("TaxLossHarvester closed — db=%s", self._db_path)
