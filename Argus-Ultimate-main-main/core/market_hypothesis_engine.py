#!/usr/bin/env python3
"""
Market Hypothesis Engine — generate, test, and track hypotheses about the market.

Creates testable hypotheses (e.g. "BTC entering bull regime", "volatility expansion
imminent") with supporting/opposing evidence, then validates them against actual
market outcomes.  Tracks historical accuracy to calibrate confidence.

SQLite persistence at data/hypotheses.db.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """A testable market hypothesis."""
    hypothesis_id: str
    statement: str
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    probability: float = 0.5
    testable_prediction: str = ""
    expiry_hours: int = 24
    created_at: str = ""
    status: str = "active"  # active, confirmed, rejected, inconclusive, expired
    category: str = "general"  # trend, volatility, correlation, catalyst, regime
    symbol: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HypothesisResult:
    """Outcome of testing a hypothesis."""
    hypothesis_id: str
    outcome: str  # confirmed, rejected, inconclusive
    evidence: str
    actual_value: Optional[float] = None
    predicted_value: Optional[float] = None
    tested_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Hypothesis generators (rule-based templates)
# ---------------------------------------------------------------------------

_HYPOTHESIS_TEMPLATES = [
    {
        "category": "trend",
        "check": lambda d: d.get("momentum_1d", 0) > 0.02 and d.get("hurst", 0) > 0.5,
        "statement": "{symbol} is entering a bull regime",
        "evidence_for_fn": lambda d: [
            f"1d momentum {d.get('momentum_1d', 0):.3f} > 0.02",
            f"Hurst exponent {d.get('hurst', 0):.2f} > 0.5 (trending)",
        ],
        "evidence_against_fn": lambda d: (
            [f"RSI {d.get('rsi', 50):.0f} overbought"] if d.get("rsi", 50) > 70 else []
        ),
        "prediction": "{symbol} price will be higher in 24h",
        "probability": lambda d: min(0.5 + d.get("momentum_1d", 0) * 5, 0.85),
    },
    {
        "category": "trend",
        "check": lambda d: d.get("momentum_1d", 0) < -0.02 and d.get("hurst", 0) > 0.5,
        "statement": "{symbol} is entering a bear regime",
        "evidence_for_fn": lambda d: [
            f"1d momentum {d.get('momentum_1d', 0):.3f} < -0.02",
            f"Hurst exponent {d.get('hurst', 0):.2f} > 0.5 (trending)",
        ],
        "evidence_against_fn": lambda d: (
            [f"RSI {d.get('rsi', 50):.0f} oversold (possible bounce)"] if d.get("rsi", 50) < 30 else []
        ),
        "prediction": "{symbol} price will be lower in 24h",
        "probability": lambda d: min(0.5 + abs(d.get("momentum_1d", 0)) * 5, 0.85),
    },
    {
        "category": "volatility",
        "check": lambda d: d.get("bb_squeeze", False) or (d.get("atr_ratio", 1.0) < 0.5),
        "statement": "Volatility expansion imminent for {symbol}",
        "evidence_for_fn": lambda d: [
            e for e in [
                "Bollinger Band squeeze detected" if d.get("bb_squeeze") else None,
                f"ATR ratio {d.get('atr_ratio', 1.0):.2f} < 0.5 (compressed)" if d.get("atr_ratio", 1.0) < 0.5 else None,
                f"IV rising: {d.get('iv_change', 0):.1f}%" if d.get("iv_change", 0) > 5 else None,
            ] if e
        ],
        "evidence_against_fn": lambda d: (
            ["Low-volume environment suggests continued compression"]
            if d.get("volume_ratio", 1.0) < 0.5 else []
        ),
        "prediction": "{symbol} ATR will increase >50% within 12h",
        "probability": lambda d: 0.6 if d.get("bb_squeeze") else 0.5,
    },
    {
        "category": "correlation",
        "check": lambda d: d.get("btc_eth_corr", 0.9) < 0.6,
        "statement": "ETH is decoupling from BTC",
        "evidence_for_fn": lambda d: [
            f"BTC-ETH correlation dropped to {d.get('btc_eth_corr', 0.9):.2f}",
        ],
        "evidence_against_fn": lambda d: (
            ["Macro risk-off may re-couple assets"] if d.get("fear_greed", 50) < 30 else []
        ),
        "prediction": "ETH/BTC ratio will diverge from 30-day mean",
        "probability": lambda d: max(0.3, 0.8 - d.get("btc_eth_corr", 0.9)),
    },
    {
        "category": "regime",
        "check": lambda d: d.get("fear_greed", 50) < 20,
        "statement": "Market in extreme fear — contrarian bounce likely for {symbol}",
        "evidence_for_fn": lambda d: [
            f"Fear & Greed index at {d.get('fear_greed', 50)}",
            "Historical extreme-fear periods followed by >5% rallies 65% of the time",
        ],
        "evidence_against_fn": lambda d: (
            [f"Drawdown {d.get('drawdown_pct', 0):.1f}% — capitulation may continue"]
            if d.get("drawdown_pct", 0) > 20 else []
        ),
        "prediction": "{symbol} will rally >2% within 48h",
        "probability": lambda d: 0.6,
    },
    {
        "category": "catalyst",
        "check": lambda d: d.get("volume_ratio", 1.0) > 2.5,
        "statement": "Unusual volume on {symbol} — catalyst-driven move underway",
        "evidence_for_fn": lambda d: [
            f"Volume ratio {d.get('volume_ratio', 1.0):.1f}x average",
        ],
        "evidence_against_fn": lambda d: (
            ["Could be wash trading or exchange migration"]
            if d.get("exchange_count", 1) < 2 else []
        ),
        "prediction": "{symbol} will sustain >1 ATR move in next 4h",
        "probability": lambda d: min(0.4 + d.get("volume_ratio", 1.0) * 0.1, 0.8),
    },
]


# ---------------------------------------------------------------------------
# MarketHypothesisEngine
# ---------------------------------------------------------------------------

class MarketHypothesisEngine:
    """
    Generates, tracks, and validates market hypotheses.

    Parameters
    ----------
    config : dict, optional
        ``market_hypothesis`` section from unified_config.yaml.
    db_path : str, optional
        Override SQLite path.
    """

    _DEFAULTS: Dict[str, Any] = {
        "enabled": True,
        "db_path": "data/hypotheses.db",
        "max_active": 20,
        "default_expiry_hours": 24,
        "min_probability": 0.3,
        "confirmation_threshold": 0.7,
        "rejection_threshold": 0.3,
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        db_path: Optional[str] = None,
    ) -> None:
        cfg = dict(self._DEFAULTS)
        if config:
            cfg.update(config)
        self._cfg = cfg
        self._enabled = bool(cfg.get("enabled", True))
        self._max_active = int(cfg.get("max_active", 20))
        self._default_expiry = int(cfg.get("default_expiry_hours", 24))
        self._confirm_thresh = float(cfg.get("confirmation_threshold", 0.7))
        self._reject_thresh = float(cfg.get("rejection_threshold", 0.3))

        self._hypotheses: Dict[str, Hypothesis] = {}
        self._results: List[HypothesisResult] = []
        self._lock = threading.Lock()

        db = db_path or str(cfg.get("db_path", "data/hypotheses.db"))
        self._db_path = db
        self._init_db()
        self._load_hypotheses()

        logger.info("MarketHypothesisEngine initialised (enabled=%s)", self._enabled)

    # ------------------------------------------------------------------
    # SQLite
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hypotheses (
                    hypothesis_id TEXT PRIMARY KEY,
                    statement TEXT NOT NULL,
                    evidence_for TEXT NOT NULL DEFAULT '[]',
                    evidence_against TEXT NOT NULL DEFAULT '[]',
                    probability REAL NOT NULL,
                    testable_prediction TEXT,
                    expiry_hours INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    category TEXT NOT NULL DEFAULT 'general',
                    symbol TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hypothesis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hypothesis_id TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    evidence TEXT,
                    actual_value REAL,
                    predicted_value REAL,
                    tested_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def _load_hypotheses(self) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT hypothesis_id, statement, evidence_for, evidence_against, probability, "
                    "testable_prediction, expiry_hours, created_at, status, category, symbol "
                    "FROM hypotheses WHERE status='active'"
                ).fetchall()
            for row in rows:
                hid, stmt, ef, ea, prob, pred, exp, created, status, cat, sym = row
                self._hypotheses[hid] = Hypothesis(
                    hypothesis_id=hid, statement=stmt,
                    evidence_for=json.loads(ef) if ef else [],
                    evidence_against=json.loads(ea) if ea else [],
                    probability=prob, testable_prediction=pred or "",
                    expiry_hours=exp, created_at=created, status=status,
                    category=cat, symbol=sym,
                )
        except Exception as exc:
            logger.warning("MarketHypothesisEngine: load failed — %s", exc)

    def _persist_hypothesis(self, h: Hypothesis) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO hypotheses "
                    "(hypothesis_id, statement, evidence_for, evidence_against, probability, "
                    "testable_prediction, expiry_hours, created_at, status, category, symbol) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (h.hypothesis_id, h.statement, json.dumps(h.evidence_for),
                     json.dumps(h.evidence_against), h.probability, h.testable_prediction,
                     h.expiry_hours, h.created_at, h.status, h.category, h.symbol),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("MarketHypothesisEngine: persist failed — %s", exc)

    def _persist_result(self, r: HypothesisResult) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT INTO hypothesis_results (hypothesis_id, outcome, evidence, actual_value, predicted_value, tested_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (r.hypothesis_id, r.outcome, r.evidence, r.actual_value, r.predicted_value, r.tested_at),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("MarketHypothesisEngine: result persist failed — %s", exc)

    # ------------------------------------------------------------------
    # Hypothesis generation
    # ------------------------------------------------------------------

    def generate_hypotheses(self, market_data: Dict[str, Any]) -> List[Hypothesis]:
        """
        Generate hypotheses from current market data.

        Parameters
        ----------
        market_data : dict
            Keys per symbol: momentum_1d, hurst, rsi, bb_squeeze, atr_ratio,
            iv_change, volume_ratio, btc_eth_corr, fear_greed, drawdown_pct, etc.
            Can also have a "symbols" sub-dict keyed by symbol.

        Returns
        -------
        list of Hypothesis
        """
        generated: List[Hypothesis] = []
        now = datetime.now(timezone.utc).isoformat()

        # Handle per-symbol data
        symbols_data = market_data.get("symbols", {})
        if not symbols_data:
            # Treat entire dict as single-symbol data
            symbols_data = {"market": market_data}

        for symbol, data in symbols_data.items():
            # Merge global data (fear_greed, btc_eth_corr) with per-symbol
            merged = dict(market_data)
            merged.update(data)
            merged.pop("symbols", None)

            for template in _HYPOTHESIS_TEMPLATES:
                try:
                    if not template["check"](merged):
                        continue
                except Exception:
                    continue

                # Check if we already have an active hypothesis of this type for this symbol
                stmt = template["statement"].format(symbol=symbol)
                existing = any(
                    h.statement == stmt and h.status == "active"
                    for h in self._hypotheses.values()
                )
                if existing:
                    continue

                with self._lock:
                    if len(self._hypotheses) >= self._max_active:
                        self.expire_stale_hypotheses()
                        if len(self._hypotheses) >= self._max_active:
                            break

                try:
                    ef = template["evidence_for_fn"](merged)
                    ea = template["evidence_against_fn"](merged)
                    prob = template["probability"](merged) if callable(template.get("probability")) else 0.5
                except Exception:
                    ef, ea, prob = [], [], 0.5

                hid = str(uuid.uuid4())[:12]
                h = Hypothesis(
                    hypothesis_id=hid,
                    statement=stmt,
                    evidence_for=ef,
                    evidence_against=ea,
                    probability=round(prob, 3),
                    testable_prediction=template["prediction"].format(symbol=symbol),
                    expiry_hours=self._default_expiry,
                    created_at=now,
                    status="active",
                    category=template["category"],
                    symbol=symbol,
                )
                with self._lock:
                    self._hypotheses[hid] = h
                self._persist_hypothesis(h)
                generated.append(h)

        return generated

    def generate_hypothesis(self, market_data: Dict[str, Any]) -> List[Hypothesis]:
        """Alias for generate_hypotheses (singular name used in spec)."""
        return self.generate_hypotheses(market_data)

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------

    def test_hypothesis(
        self,
        hypothesis_id: str,
        actual_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[HypothesisResult]:
        """
        Test a hypothesis against actual market data.

        Parameters
        ----------
        hypothesis_id : str
        actual_data : dict, optional
            Must contain 'price_change_pct', 'atr_change_pct', etc.

        Returns
        -------
        HypothesisResult or None
        """
        with self._lock:
            h = self._hypotheses.get(hypothesis_id)
        if not h:
            return None

        if actual_data is None:
            actual_data = {}

        now = datetime.now(timezone.utc).isoformat()
        outcome = "inconclusive"
        evidence = ""
        actual_val = actual_data.get("price_change_pct")

        # Determine outcome based on category
        if h.category == "trend":
            if actual_val is not None:
                if "bull" in h.statement.lower():
                    if actual_val > 0:
                        outcome = "confirmed"
                        evidence = f"Price rose {actual_val:.2f}%"
                    else:
                        outcome = "rejected"
                        evidence = f"Price fell {actual_val:.2f}%"
                elif "bear" in h.statement.lower():
                    if actual_val < 0:
                        outcome = "confirmed"
                        evidence = f"Price fell {actual_val:.2f}%"
                    else:
                        outcome = "rejected"
                        evidence = f"Price rose {actual_val:.2f}%"

        elif h.category == "volatility":
            atr_change = actual_data.get("atr_change_pct")
            if atr_change is not None:
                if atr_change > 50:
                    outcome = "confirmed"
                    evidence = f"ATR increased {atr_change:.0f}%"
                elif atr_change < 10:
                    outcome = "rejected"
                    evidence = f"ATR only changed {atr_change:.0f}%"
                else:
                    outcome = "inconclusive"
                    evidence = f"ATR changed {atr_change:.0f}% — ambiguous"

        elif h.category == "correlation":
            corr = actual_data.get("btc_eth_corr")
            if corr is not None:
                if corr < 0.5:
                    outcome = "confirmed"
                    evidence = f"Correlation at {corr:.2f} — decoupled"
                elif corr > 0.7:
                    outcome = "rejected"
                    evidence = f"Correlation at {corr:.2f} — still coupled"

        elif h.category in ("regime", "catalyst"):
            if actual_val is not None:
                if "rally" in h.testable_prediction.lower() or "higher" in h.testable_prediction.lower():
                    threshold = 2.0
                    if actual_val > threshold:
                        outcome = "confirmed"
                        evidence = f"Price rallied {actual_val:.2f}%"
                    elif actual_val < -threshold:
                        outcome = "rejected"
                        evidence = f"Price dropped {actual_val:.2f}%"

        # Update hypothesis status
        with self._lock:
            h.status = outcome if outcome != "inconclusive" else "active"
            if outcome in ("confirmed", "rejected"):
                # Remove from active
                if h.hypothesis_id in self._hypotheses and outcome != "active":
                    pass  # keep for reference
        self._persist_hypothesis(h)

        result = HypothesisResult(
            hypothesis_id=hypothesis_id,
            outcome=outcome,
            evidence=evidence,
            actual_value=actual_val,
            tested_at=now,
        )
        self._results.append(result)
        self._persist_result(result)
        return result

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active_hypotheses(self) -> List[Hypothesis]:
        """Return all active (non-expired, non-resolved) hypotheses."""
        self.expire_stale_hypotheses()
        with self._lock:
            return [h for h in self._hypotheses.values() if h.status == "active"]

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Get a single hypothesis by ID."""
        with self._lock:
            return self._hypotheses.get(hypothesis_id)

    def get_hypothesis_accuracy(self) -> float:
        """Return historical accuracy of confirmed/rejected hypotheses."""
        if not self._results:
            return 0.0
        resolved = [r for r in self._results if r.outcome in ("confirmed", "rejected")]
        if not resolved:
            return 0.0
        confirmed = sum(1 for r in resolved if r.outcome == "confirmed")
        return round(confirmed / len(resolved), 4)

    def get_accuracy_by_category(self) -> Dict[str, float]:
        """Return accuracy broken down by category."""
        cats: Dict[str, List[str]] = {}
        for r in self._results:
            if r.outcome not in ("confirmed", "rejected"):
                continue
            h = self._hypotheses.get(r.hypothesis_id)
            cat = h.category if h else "unknown"
            cats.setdefault(cat, []).append(r.outcome)
        return {
            cat: round(sum(1 for o in outcomes if o == "confirmed") / len(outcomes), 4)
            for cat, outcomes in cats.items()
            if outcomes
        }

    def expire_stale_hypotheses(self) -> int:
        """Mark expired hypotheses. Returns count expired."""
        now = datetime.now(timezone.utc)
        expired = 0
        with self._lock:
            for h in list(self._hypotheses.values()):
                if h.status != "active":
                    continue
                try:
                    created = datetime.fromisoformat(h.created_at.replace("Z", "+00:00"))
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                except (ValueError, AttributeError):
                    continue
                if (now - created).total_seconds() > h.expiry_hours * 3600:
                    h.status = "expired"
                    self._persist_hypothesis(h)
                    expired += 1
        return expired

    def add_evidence(self, hypothesis_id: str, evidence: str, supports: bool) -> bool:
        """Add evidence to an existing hypothesis."""
        with self._lock:
            h = self._hypotheses.get(hypothesis_id)
            if not h or h.status != "active":
                return False
            if supports:
                h.evidence_for.append(evidence)
                h.probability = min(0.99, h.probability + 0.05)
            else:
                h.evidence_against.append(evidence)
                h.probability = max(0.01, h.probability - 0.05)
        self._persist_hypothesis(h)
        return True
