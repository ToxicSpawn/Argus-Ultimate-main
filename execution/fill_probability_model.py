"""
Limit Order Fill Probability Model — predicts the probability that a limit
order at a given price will be filled, and recommends optimal limit prices
for a target fill probability.

Uses logistic regression on distance-from-mid and spread features.  Falls back
to a manual sigmoid when scikit-learn is unavailable.

Persistence: SQLite at ``data/fill_probability.db``.

Usage:
    model = FillProbabilityModel()
    model.record_limit_order("BTC/USD", price=64950, side="buy",
                             book_mid=65000, spread_bps=3.2,
                             filled=True, time_to_fill_ms=1200)
    prob = model.predict_fill_probability("BTC/USD", price=64980,
                                          side="buy", book_mid=65000,
                                          spread_bps=3.0)
    optimal = model.get_optimal_limit_price("BTC/USD", "buy",
                                             book_mid=65000,
                                             target_fill_prob=0.7)
"""
from __future__ import annotations

import logging
import math
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional sklearn
# ---------------------------------------------------------------------------
try:
    from sklearn.linear_model import LogisticRegression  # type: ignore[import]
    import numpy as np  # type: ignore[import]
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    LogisticRegression = None  # type: ignore[assignment,misc]
    np = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DB_DIR = Path("data")
_DEFAULT_DB_PATH = _DB_DIR / "fill_probability.db"
_MIN_SAMPLES_FOR_FIT = 30
_RETRAIN_INTERVAL_S = 600  # re-fit model every 10 minutes if new data arrived


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FillProbabilityModel:
    """
    Logistic-regression model predicting limit-order fill probability.

    Features used for prediction:
        1. ``distance_bps`` — signed distance from mid in basis points
           (positive = aggressive / crossing spread, negative = passive / deep)
        2. ``spread_bps`` — current bid-ask spread in basis points
        3. ``distance_over_spread`` — ratio of distance to spread

    Parameters
    ----------
    db_path : str or Path
        SQLite database path (default ``data/fill_probability.db``).
    min_samples : int
        Minimum recorded outcomes before logistic model is trained.
    retrain_interval_s : float
        Seconds between automatic model re-fits.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        min_samples: int = _MIN_SAMPLES_FOR_FIT,
        retrain_interval_s: float = _RETRAIN_INTERVAL_S,
    ) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._min_samples = max(5, min_samples)
        self._retrain_interval = retrain_interval_s

        self._lock = threading.Lock()
        # Per-symbol sklearn model (or None → use sigmoid fallback)
        self._models: Dict[str, Any] = {}
        # Tracks last fit time per symbol
        self._last_fit_time: Dict[str, float] = {}
        # Count of new records since last fit
        self._new_records: Dict[str, int] = {}

        self._init_db()
        logger.info(
            "FillProbabilityModel initialised  db=%s  sklearn=%s  min_samples=%d",
            self._db_path, _HAS_SKLEARN, self._min_samples,
        )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the database and table if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS limit_orders (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          REAL    NOT NULL,
                    symbol      TEXT    NOT NULL,
                    price       REAL    NOT NULL,
                    side        TEXT    NOT NULL,
                    book_mid    REAL    NOT NULL,
                    spread_bps  REAL    NOT NULL,
                    distance_bps REAL   NOT NULL,
                    filled      INTEGER NOT NULL,
                    time_to_fill_ms REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lo_symbol ON limit_orders(symbol)
            """)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_distance_bps(price: float, side: str, book_mid: float) -> float:
        """
        Signed distance from mid in basis points.

        For buys:  positive distance = price above mid (aggressive)
        For sells: positive distance = price below mid (aggressive)
        """
        if book_mid <= 0:
            return 0.0
        raw_bps = ((price - book_mid) / book_mid) * 10_000
        if side == "sell":
            raw_bps = -raw_bps
        return raw_bps

    @staticmethod
    def _build_features(distance_bps: float, spread_bps: float) -> Tuple[float, float, float]:
        """Return (distance_bps, spread_bps, distance_over_spread)."""
        dos = distance_bps / spread_bps if spread_bps > 0 else 0.0
        return (distance_bps, spread_bps, dos)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_limit_order(
        self,
        symbol: str,
        price: float,
        side: str,
        book_mid: float,
        spread_bps: float,
        filled: bool,
        time_to_fill_ms: Optional[float] = None,
    ) -> None:
        """
        Record the outcome of a limit order for model training.

        Parameters
        ----------
        symbol : str
        price : float       Limit price placed.
        side : str          "buy" or "sell".
        book_mid : float    Mid price at time of placement.
        spread_bps : float  Bid-ask spread in basis points at placement.
        filled : bool       Whether the order was fully filled.
        time_to_fill_ms : float or None
            Milliseconds from placement to fill (None if not filled).
        """
        side = side.lower()
        distance_bps = self._compute_distance_bps(price, side, book_mid)
        now = time.time()

        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO limit_orders
                   (ts, symbol, price, side, book_mid, spread_bps, distance_bps, filled, time_to_fill_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (now, symbol, price, side, book_mid, spread_bps, distance_bps,
                 1 if filled else 0, time_to_fill_ms),
            )
            conn.commit()

        with self._lock:
            self._new_records[symbol] = self._new_records.get(symbol, 0) + 1

        logger.debug(
            "Recorded limit order  symbol=%s  side=%s  dist_bps=%.1f  filled=%s  ttf_ms=%s",
            symbol, side, distance_bps, filled, time_to_fill_ms,
        )

    def predict_fill_probability(
        self,
        symbol: str,
        price: float,
        side: str,
        book_mid: float,
        spread_bps: float,
    ) -> float:
        """
        Predict the probability that a limit order will be filled.

        Returns
        -------
        float
            Probability in [0, 1].
        """
        side = side.lower()
        distance_bps = self._compute_distance_bps(price, side, book_mid)
        features = self._build_features(distance_bps, spread_bps)

        self._maybe_refit(symbol)

        with self._lock:
            model = self._models.get(symbol)

        if model is not None and _HAS_SKLEARN:
            try:
                X = np.array([features])
                prob = float(model.predict_proba(X)[0, 1])
                return max(0.0, min(1.0, prob))
            except Exception as exc:
                logger.warning("sklearn predict failed, using sigmoid fallback: %s", exc)

        # Sigmoid fallback
        return self._sigmoid_fallback(distance_bps, spread_bps)

    def get_optimal_limit_price(
        self,
        symbol: str,
        side: str,
        book_mid: float,
        target_fill_prob: float = 0.7,
    ) -> float:
        """
        Find the best limit price that achieves *target_fill_prob*.

        Uses binary search over distance-from-mid.

        Parameters
        ----------
        target_fill_prob : float
            Desired fill probability (0-1, default 0.7).

        Returns
        -------
        float
            Recommended limit price.
        """
        side = side.lower()
        if book_mid <= 0:
            return book_mid

        # Estimate current spread from recent data
        spread_bps = self._estimate_spread(symbol)

        # Binary search: find distance_bps where fill_prob ≈ target
        # More aggressive (positive distance) → higher fill prob
        # More passive (negative distance) → lower fill prob
        lo_dist, hi_dist = -100.0, 50.0  # bps range

        for _ in range(50):
            mid_dist = (lo_dist + hi_dist) / 2.0
            features = self._build_features(mid_dist, spread_bps)

            with self._lock:
                model = self._models.get(symbol)

            if model is not None and _HAS_SKLEARN:
                try:
                    X = np.array([features])
                    prob = float(model.predict_proba(X)[0, 1])
                except Exception:
                    prob = self._sigmoid_fallback(mid_dist, spread_bps)
            else:
                prob = self._sigmoid_fallback(mid_dist, spread_bps)

            if prob < target_fill_prob:
                lo_dist = mid_dist  # need more aggressive
            else:
                hi_dist = mid_dist

            if abs(hi_dist - lo_dist) < 0.1:
                break

        optimal_dist_bps = (lo_dist + hi_dist) / 2.0

        # Convert distance_bps back to price
        if side == "buy":
            price = book_mid * (1.0 + optimal_dist_bps / 10_000)
        else:
            price = book_mid * (1.0 - optimal_dist_bps / 10_000)

        logger.debug(
            "Optimal limit  symbol=%s  side=%s  mid=%.2f  target_prob=%.2f  "
            "dist_bps=%.1f  price=%.2f",
            symbol, side, book_mid, target_fill_prob, optimal_dist_bps, price,
        )
        return round(price, 8)

    def get_fill_stats(self, symbol: str, side: Optional[str] = None) -> Dict[str, Any]:
        """
        Return aggregate fill statistics.

        Returns
        -------
        dict
            fill_rate : float               Fraction of orders filled.
            avg_time_to_fill_ms : float     Average fill time (filled orders only).
            avg_price_improvement_bps : float
                Average distance from mid for filled orders (positive = paid less).
            total_orders : int
        """
        query = "SELECT filled, time_to_fill_ms, distance_bps FROM limit_orders WHERE symbol = ?"
        params: list = [symbol]
        if side:
            query += " AND side = ?"
            params.append(side.lower())

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()

        if not rows:
            return {
                "fill_rate": 0.0,
                "avg_time_to_fill_ms": 0.0,
                "avg_price_improvement_bps": 0.0,
                "total_orders": 0,
            }

        total = len(rows)
        filled_rows = [r for r in rows if r[0] == 1]
        fill_rate = len(filled_rows) / total if total > 0 else 0.0

        ttfs = [r[1] for r in filled_rows if r[1] is not None]
        avg_ttf = sum(ttfs) / len(ttfs) if ttfs else 0.0

        # Price improvement = negative distance (passive fill)
        improvements = [-r[2] for r in filled_rows]
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0

        return {
            "fill_rate": round(fill_rate, 4),
            "avg_time_to_fill_ms": round(avg_ttf, 1),
            "avg_price_improvement_bps": round(avg_improvement, 2),
            "total_orders": total,
        }

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def _maybe_refit(self, symbol: str) -> None:
        """Re-fit the logistic model if enough new data has accumulated."""
        with self._lock:
            new_count = self._new_records.get(symbol, 0)
            last_fit = self._last_fit_time.get(symbol, 0.0)

        if new_count < 5 and (time.time() - last_fit) < self._retrain_interval:
            return

        self._fit_model(symbol)

    def _fit_model(self, symbol: str) -> None:
        """Fit logistic regression on all recorded outcomes for symbol."""
        if not _HAS_SKLEARN:
            return

        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT distance_bps, spread_bps, filled FROM limit_orders WHERE symbol = ?",
                (symbol,),
            ).fetchall()

        if len(rows) < self._min_samples:
            logger.debug(
                "Not enough samples to fit model for %s (%d < %d)",
                symbol, len(rows), self._min_samples,
            )
            return

        X = np.array([[r[0], r[1], r[0] / r[1] if r[1] > 0 else 0.0] for r in rows])
        y = np.array([r[2] for r in rows])

        # Need both classes
        if len(set(y)) < 2:
            logger.debug("Only one class in data for %s, skipping fit", symbol)
            return

        try:
            model = LogisticRegression(max_iter=500, solver="lbfgs")
            model.fit(X, y)
            with self._lock:
                self._models[symbol] = model
                self._last_fit_time[symbol] = time.time()
                self._new_records[symbol] = 0
            logger.info(
                "Fitted logistic model for %s  samples=%d  coef=%s",
                symbol, len(rows), model.coef_.tolist(),
            )
        except Exception as exc:
            logger.warning("Failed to fit model for %s: %s", symbol, exc)

    # ------------------------------------------------------------------
    # Fallbacks
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid_fallback(distance_bps: float, spread_bps: float) -> float:
        """
        Manual sigmoid estimate of fill probability.

        More aggressive (positive distance) → higher probability.
        Wider spread → slightly lower probability for passive orders.
        """
        # Centre sigmoid around 0 distance, steepness depends on spread
        k = 0.15 if spread_bps < 5 else 0.10
        z = k * distance_bps
        try:
            prob = 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            prob = 0.0 if z < 0 else 1.0
        return max(0.0, min(1.0, prob))

    def _estimate_spread(self, symbol: str) -> float:
        """Estimate typical spread from recent recorded orders."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT AVG(spread_bps) FROM limit_orders WHERE symbol = ? "
                "ORDER BY ts DESC LIMIT 50",
                (symbol,),
            ).fetchone()
        if row and row[0] is not None:
            return float(row[0])
        return 5.0  # default 5 bps
