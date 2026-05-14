"""
AUSTRAC DCE Compliance — Digital Currency Exchange reporting module.

Covers:
  - Threshold Transaction Reports (TTR): AUD 10,000+ cash / digital currency
  - Suspicious Matter Reports (SMR): anomalous patterns
  - International Funds Transfer Instructions (IFTI): cross-border transfers
  - Customer Due Diligence (CDD) record keeping

Reference: AUSTRAC AML/CTF Act 2006, Part 7 reporting obligations.

IMPORTANT: This module tracks obligations and generates report templates.
Actual AUSTRAC submission is done via the AUSTRAC Online portal by a
designated AUSTRAC Reporting Entity (RE) — not automated here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# AUD thresholds
TTR_THRESHOLD_AUD = 10_000.0
RECORD_RETENTION_YEARS = 7  # AUSTRAC requires 7 years

# SMR trigger patterns
SMR_ROUND_AMOUNT_THRESHOLD = 0.01  # within 1% of round number
SMR_RAPID_TXNS_COUNT = 5          # 5+ transactions in 10 minutes = structuring risk
SMR_RAPID_TXNS_WINDOW_S = 600


@dataclass
class AUSTRACTransaction:
    tx_id: str
    timestamp: datetime
    asset: str                  # BTC, ETH, AUD etc.
    amount_asset: float
    amount_aud: float           # AUD equivalent at time of tx
    direction: str              # BUY / SELL / DEPOSIT / WITHDRAWAL
    counterparty_exchange: str  # Kraken, Coinbase, etc.
    customer_id: str            # Internal customer/account ID
    tx_hash: Optional[str] = None  # On-chain hash if applicable
    notes: str = ""


@dataclass
class TTRRecord:
    """Threshold Transaction Report."""
    report_id: str
    generated_at: datetime
    transaction: AUSTRACTransaction
    requires_filing: bool
    filed: bool = False
    filed_at: Optional[datetime] = None


@dataclass
class SMRRecord:
    """Suspicious Matter Report."""
    report_id: str
    generated_at: datetime
    transactions: List[AUSTRACTransaction]
    reason: str                 # Why this is suspicious
    requires_filing: bool = True
    filed: bool = False


@dataclass
class CDDRecord:
    """Customer Due Diligence record."""
    customer_id: str
    verified_at: datetime
    id_type: str               # passport / drivers_licence / company_acn
    id_reference: str          # Document reference (DO NOT store raw ID numbers)
    risk_rating: str           # LOW / MEDIUM / HIGH
    pep: bool = False          # Politically Exposed Person
    enhanced_dd_required: bool = False


class AUSTRACReporter:
    """
    AUSTRAC compliance tracking and report generation for ARGUS.

    Usage::

        reporter = AUSTRACReporter(output_dir=Path("compliance/reports"))
        tx = AUSTRACTransaction(...)
        reporter.record_transaction(tx)
        # At end of day:
        pending = reporter.get_pending_ttrs()
        reporter.export_ttr_report(pending[0])
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        entity_name: str = "ARGUS Trading",
        reporting_entity_id: str = "UNREG-001",
    ) -> None:
        self._output_dir = output_dir or Path("compliance/reports")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._entity_name = entity_name
        self._re_id = reporting_entity_id
        self._transactions: List[AUSTRACTransaction] = []
        self._ttrs: List[TTRRecord] = []
        self._smrs: List[SMRRecord] = []
        self._cdds: Dict[str, CDDRecord] = {}

    # ------------------------------------------------------------------
    def record_transaction(self, tx: AUSTRACTransaction) -> None:
        """Record a transaction and automatically assess TTR/SMR obligations."""
        self._transactions.append(tx)
        self._assess_ttr(tx)
        self._assess_smr()
        logger.debug("AUSTRAC: recorded tx %s (AUD %.2f)", tx.tx_id, tx.amount_aud)

    def register_customer(self, cdd: CDDRecord) -> None:
        self._cdds[cdd.customer_id] = cdd

    # ------------------------------------------------------------------
    def get_pending_ttrs(self) -> List[TTRRecord]:
        return [t for t in self._ttrs if t.requires_filing and not t.filed]

    def get_pending_smrs(self) -> List[SMRRecord]:
        return [s for s in self._smrs if s.requires_filing and not s.filed]

    def mark_ttr_filed(self, report_id: str) -> None:
        for t in self._ttrs:
            if t.report_id == report_id:
                t.filed = True
                t.filed_at = datetime.now(tz=timezone.utc)
                logger.info("AUSTRAC: TTR %s marked as filed", report_id)

    def mark_smr_filed(self, report_id: str) -> None:
        for s in self._smrs:
            if s.report_id == report_id:
                s.filed = True
                logger.info("AUSTRAC: SMR %s marked as filed", report_id)

    # ------------------------------------------------------------------
    def export_ttr_report(self, ttr: TTRRecord) -> Path:
        """Generate TTR JSON file for manual submission via AUSTRAC Online."""
        tx = ttr.transaction
        report = {
            "report_type": "TTR",
            "report_id": ttr.report_id,
            "generated_at": ttr.generated_at.isoformat(),
            "reporting_entity": {
                "name": self._entity_name,
                "id": self._re_id,
            },
            "transaction": {
                "id": tx.tx_id,
                "timestamp": tx.timestamp.isoformat(),
                "asset": tx.asset,
                "amount_aud": tx.amount_aud,
                "direction": tx.direction,
                "counterparty": tx.counterparty_exchange,
                "customer_id": tx.customer_id,
                "tx_hash": tx.tx_hash,
            },
            "threshold_aud": TTR_THRESHOLD_AUD,
            "notes": tx.notes,
        }
        path = self._output_dir / f"TTR_{ttr.report_id}.json"
        path.write_text(json.dumps(report, indent=2))
        logger.info("AUSTRAC: TTR report exported to %s", path)
        return path

    def export_smr_report(self, smr: SMRRecord) -> Path:
        """Generate SMR JSON file for manual submission."""
        report = {
            "report_type": "SMR",
            "report_id": smr.report_id,
            "generated_at": smr.generated_at.isoformat(),
            "reporting_entity": {
                "name": self._entity_name,
                "id": self._re_id,
            },
            "reason": smr.reason,
            "transactions": [
                {
                    "id": tx.tx_id,
                    "timestamp": tx.timestamp.isoformat(),
                    "amount_aud": tx.amount_aud,
                    "direction": tx.direction,
                }
                for tx in smr.transactions
            ],
        }
        path = self._output_dir / f"SMR_{smr.report_id}.json"
        path.write_text(json.dumps(report, indent=2))
        logger.info("AUSTRAC: SMR report exported to %s", path)
        return path

    def compliance_summary(self) -> Dict:
        """Return current compliance status summary."""
        return {
            "total_transactions": len(self._transactions),
            "pending_ttrs": len(self.get_pending_ttrs()),
            "pending_smrs": len(self.get_pending_smrs()),
            "registered_customers": len(self._cdds),
            "total_ttrs": len(self._ttrs),
            "total_smrs": len(self._smrs),
            "ttr_threshold_aud": TTR_THRESHOLD_AUD,
            "record_retention_years": RECORD_RETENTION_YEARS,
        }

    # ------------------------------------------------------------------
    def _assess_ttr(self, tx: AUSTRACTransaction) -> None:
        """Auto-generate TTR if transaction meets threshold."""
        if tx.amount_aud < TTR_THRESHOLD_AUD:
            return
        report_id = self._make_id("TTR", tx.tx_id)
        ttr = TTRRecord(
            report_id=report_id,
            generated_at=datetime.now(tz=timezone.utc),
            transaction=tx,
            requires_filing=True,
        )
        self._ttrs.append(ttr)
        logger.warning(
            "AUSTRAC TTR REQUIRED: tx %s AUD %.2f exceeds threshold %.2f",
            tx.tx_id, tx.amount_aud, TTR_THRESHOLD_AUD,
        )

    def _assess_smr(self) -> None:
        """Check recent transactions for suspicious patterns."""
        now = time.time()
        window_start = now - SMR_RAPID_TXNS_WINDOW_S
        recent = [
            tx for tx in self._transactions
            if tx.timestamp.timestamp() >= window_start
        ]
        if len(recent) < SMR_RAPID_TXNS_COUNT:
            return

        # Potential structuring: many small transactions near threshold
        near_threshold = [
            tx for tx in recent
            if tx.amount_aud > TTR_THRESHOLD_AUD * 0.8
            and tx.amount_aud < TTR_THRESHOLD_AUD
        ]
        if len(near_threshold) >= 3:
            report_id = self._make_id("SMR", str(now))
            # Check not already reported
            existing_ids = {s.report_id for s in self._smrs}
            if report_id not in existing_ids:
                smr = SMRRecord(
                    report_id=report_id,
                    generated_at=datetime.now(tz=timezone.utc),
                    transactions=near_threshold,
                    reason=(
                        f"Potential structuring: {len(near_threshold)} transactions "
                        f"just below AUD {TTR_THRESHOLD_AUD:.0f} threshold "
                        f"within {SMR_RAPID_TXNS_WINDOW_S}s"
                    ),
                )
                self._smrs.append(smr)
                logger.warning("AUSTRAC SMR generated: %s", smr.reason)

    @staticmethod
    def _make_id(prefix: str, seed: str) -> str:
        h = hashlib.sha256(seed.encode()).hexdigest()[:8].upper()
        return f"{prefix}-{h}"
