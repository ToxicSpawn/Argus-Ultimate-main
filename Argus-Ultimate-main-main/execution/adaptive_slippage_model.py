"""
Adaptive Slippage Model — learns from historical fills to predict slippage.

Reads fill records from the FillTracker SQLite DB and fits a regression model
(linear fallback if sklearn unavailable) to predict slippage_bps from:
  - side (buy/sell encoded as +1/-1)
  - quantity_norm (normalised by 30d avg volume)
  - time_of_day (hour 0-23)
  - spread_bps (ask-bid spread at time of fill)
  - market_volatility (30m realised vol at fill time)

Output: predicted_slippage_bps, confidence_interval, recommended_limit_offset_bps

The model retrains automatically when the elapsed time since last training
exceeds ``retrain_interval`` seconds.  If fewer than ``min_samples`` fills are
available the module returns a conservative fallback prediction (3.0 bps, wide
CI) so that callers always receive a usable estimate.
"""
from __future__ import annotations

import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional sklearn import
# ---------------------------------------------------------------------------

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
    logger.debug("adaptive_slippage_model: sklearn available — will use GBR")
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.debug("adaptive_slippage_model: sklearn not available — using linear fallback")

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

_FALLBACK_PREDICTED_BPS: float = 3.0
_FALLBACK_CI_HALF: float = 4.0  # wide ± interval when model is cold


@dataclass
class SlippageFeatures:
    """Input features for a single slippage prediction request."""

    side: str                   # "buy" or "sell"
    quantity_norm: float        # quantity / 30d_avg_volume  (dimensionless, >0)
    hour: int                   # local hour 0-23
    spread_bps: float           # ask-bid spread in basis points at fill time
    volatility_30m: float       # 30-minute realised volatility (annualised fraction)


@dataclass
class SlippagePrediction:
    """Prediction output from AdaptiveSlippageModel."""

    predicted_bps: float            # point estimate of expected slippage
    ci_low: float                   # lower 95 % confidence bound
    ci_high: float                  # upper 95 % confidence bound
    limit_offset_bps: float         # recommended limit price offset = ci_high * 1.1
    model_type: str                 # "gbr" | "linear" | "fallback"
    n_samples: int                  # number of training samples used


# ---------------------------------------------------------------------------
# AdaptiveSlippageModel
# ---------------------------------------------------------------------------

class AdaptiveSlippageModel:
    """
    Online-learning slippage predictor backed by FillTracker SQLite data.

    Thread-safety note: predict() may trigger a retrain() call.  For
    single-threaded or asyncio usage this is fine; for multi-threaded callers
    use an external lock around predict().
    """

    def __init__(
        self,
        db_path: str = "data/fills.db",
        min_samples: int = 50,
        retrain_interval: float = 3600.0,
    ) -> None:
        self.db_path = db_path
        self.min_samples = min_samples
        self.retrain_interval = retrain_interval

        self._last_trained: float = 0.0
        self._n_samples: int = 0
        self._model_type: str = "fallback"
        self._mae_bps: float = float("nan")

        # Linear model state
        self._linear_coeffs: Dict[str, float] = {}
        self._linear_intercept: float = _FALLBACK_PREDICTED_BPS
        self._residual_std: float = _FALLBACK_CI_HALF / 1.96

        # sklearn model state
        self._sklearn_model: Optional[Any] = None
        self._scaler: Optional[Any] = None

        logger.info(
            "AdaptiveSlippageModel initialised: db=%s min_samples=%d interval=%.0fs",
            db_path, min_samples, retrain_interval,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, features: SlippageFeatures) -> SlippagePrediction:
        """
        Predict slippage for the given feature set.

        Triggers a retrain if the model is stale (elapsed > retrain_interval).
        Returns a fallback prediction when insufficient data exists.
        """
        now = time.time()
        if now - self._last_trained >= self.retrain_interval:
            try:
                self._n_samples = self.retrain()
            except Exception:
                logger.exception("AdaptiveSlippageModel: retrain failed; using cached model")

        if self._n_samples < self.min_samples:
            logger.debug(
                "AdaptiveSlippageModel: cold model (n=%d < min=%d) — returning fallback",
                self._n_samples, self.min_samples,
            )
            return SlippagePrediction(
                predicted_bps=_FALLBACK_PREDICTED_BPS,
                ci_low=max(0.0, _FALLBACK_PREDICTED_BPS - _FALLBACK_CI_HALF),
                ci_high=_FALLBACK_PREDICTED_BPS + _FALLBACK_CI_HALF,
                limit_offset_bps=(_FALLBACK_PREDICTED_BPS + _FALLBACK_CI_HALF) * 1.1,
                model_type="fallback",
                n_samples=self._n_samples,
            )

        x = self._features_to_vector(features)

        if self._model_type == "gbr" and self._sklearn_model is not None:
            predicted_bps = self._predict_sklearn(x)
        else:
            predicted_bps = self._predict_linear(x)

        # Enforce non-negative prediction
        predicted_bps = max(0.0, predicted_bps)
        ci_half = 1.96 * self._residual_std
        ci_low = max(0.0, predicted_bps - ci_half)
        ci_high = predicted_bps + ci_half
        limit_offset_bps = ci_high * 1.1

        return SlippagePrediction(
            predicted_bps=predicted_bps,
            ci_low=ci_low,
            ci_high=ci_high,
            limit_offset_bps=limit_offset_bps,
            model_type=self._model_type,
            n_samples=self._n_samples,
        )

    def retrain(self) -> int:
        """
        Load fills from SQLite and refit the model.

        Returns the number of samples used.  Updates internal model state.
        Raises on DB read errors (caller should catch).
        """
        rows = self._load_fills()
        n = len(rows)
        self._last_trained = time.time()

        if n < self.min_samples:
            logger.info(
                "AdaptiveSlippageModel: only %d fills available (min=%d); "
                "model stays as fallback",
                n, self.min_samples,
            )
            return n

        X, y = self._build_dataset(rows)

        if _SKLEARN_AVAILABLE:
            try:
                model = self._fit_sklearn(X, y)
                self._sklearn_model = model
                # Compute residuals for CI
                preds = model.predict(self._scale_X(X))
                residuals = [yi - pi for yi, pi in zip(y, preds)]
                self._residual_std = self._std(residuals)
                self._model_type = "gbr"
                self._mae_bps = sum(abs(r) for r in residuals) / len(residuals)
                logger.info(
                    "AdaptiveSlippageModel: GBR retrained n=%d MAE=%.3f bps", n, self._mae_bps
                )
            except Exception:
                logger.exception(
                    "AdaptiveSlippageModel: GBR fit failed; falling back to linear"
                )
                self._fit_linear_and_store(X, y)
        else:
            self._fit_linear_and_store(X, y)

        return n

    def get_stats(self) -> Dict[str, Any]:
        """Return a summary dict of current model state."""
        return {
            "model_type": self._model_type,
            "n_samples": self._n_samples,
            "last_trained": self._last_trained,
            "mae_bps": self._mae_bps,
            "residual_std": self._residual_std,
            "sklearn_available": _SKLEARN_AVAILABLE,
        }

    # ------------------------------------------------------------------
    # Auto-populate from trade ledger
    # ------------------------------------------------------------------

    @classmethod
    def auto_populate_from_ledger(
        cls,
        ledger_path: str = "data/trade_ledger.db",
        fills_db_path: str = "data/fills.db",
    ) -> int:
        """
        Read the trade ledger SQLite and populate the fills database for the
        slippage model.

        Extracts: symbol, side, expected_price, actual_price, quantity, timestamp.
        Computes slippage_bps = (actual - expected) / expected * 10000 for buys
        (reversed for sells).

        Parameters
        ----------
        ledger_path:
            Path to the trade ledger SQLite database.
        fills_db_path:
            Path to the fills SQLite database to write to.

        Returns
        -------
        int
            Number of fill records written.
        """
        ledger_file = Path(ledger_path)
        if not ledger_file.exists():
            logger.warning(
                "auto_populate_from_ledger: ledger not found at %s", ledger_path
            )
            return 0

        # Read trades from ledger
        rows = []
        try:
            conn = sqlite3.connect(ledger_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            try:
                # Probe available columns
                cursor = conn.execute("PRAGMA table_info(trades)")
                col_names = {row[1] for row in cursor.fetchall()}

                # Build SELECT with best available expected_price source
                if "expected_price" in col_names and "signal_price" in col_names:
                    ep_expr = "COALESCE(expected_price, signal_price, price)"
                elif "expected_price" in col_names:
                    ep_expr = "COALESCE(expected_price, price)"
                elif "signal_price" in col_names:
                    ep_expr = "COALESCE(signal_price, price)"
                else:
                    ep_expr = "price"

                raw = conn.execute(
                    f"SELECT symbol, side, price, quantity, timestamp, "
                    f"{ep_expr} AS expected_price "
                    f"FROM trades ORDER BY timestamp DESC LIMIT 10000"
                ).fetchall()
                rows = [dict(r) for r in raw]
            except sqlite3.OperationalError:
                logger.warning("auto_populate_from_ledger: no 'trades' table found")
            finally:
                conn.close()
        except Exception:
            logger.exception("auto_populate_from_ledger: failed to read ledger %s", ledger_path)
            return 0

        if not rows:
            return 0

        # Write to fills DB
        fills_file = Path(fills_db_path)
        fills_file.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        try:
            conn = sqlite3.connect(fills_db_path, timeout=10.0)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fills (
                    side TEXT,
                    quantity_usd REAL,
                    slippage_bps REAL,
                    timestamp REAL,
                    spread_bps REAL DEFAULT 5.0,
                    volatility_30m REAL DEFAULT 0.02
                )
                """
            )
            for row in rows:
                try:
                    side = str(row.get("side", "buy")).lower()
                    actual = float(row["price"])
                    expected = float(row["expected_price"])
                    qty = float(row["quantity"])
                    ts = float(row.get("timestamp", 0.0))

                    if expected <= 0 or actual <= 0:
                        continue

                    # Slippage: positive = unfavourable
                    if side == "buy":
                        slippage_bps = (actual - expected) / expected * 10000.0
                    else:
                        slippage_bps = (expected - actual) / expected * 10000.0

                    qty_usd = actual * qty

                    conn.execute(
                        "INSERT INTO fills (side, quantity_usd, slippage_bps, timestamp) "
                        "VALUES (?, ?, ?, ?)",
                        (side, qty_usd, slippage_bps, ts),
                    )
                    written += 1
                except (TypeError, ValueError, KeyError):
                    continue

            conn.commit()
            conn.close()
        except Exception:
            logger.exception("auto_populate_from_ledger: failed to write fills DB")
            return 0

        logger.info(
            "auto_populate_from_ledger: wrote %d fills from %s to %s",
            written, ledger_path, fills_db_path,
        )
        return written

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_fills(self) -> List[Dict[str, Any]]:
        """
        Load all fill records from the SQLite fills table.

        Returns an empty list when the DB file does not exist yet.
        """
        db_file = Path(self.db_path)
        if not db_file.exists():
            logger.debug("AdaptiveSlippageModel: DB not found at %s", self.db_path)
            return []

        rows: List[Dict[str, Any]] = []
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            try:
                raw = conn.execute(
                    """
                    SELECT side, quantity_usd, slippage_bps, timestamp,
                           -- spread_bps and volatility_30m are optional extended columns
                           COALESCE(spread_bps, 5.0)       AS spread_bps,
                           COALESCE(volatility_30m, 0.02)  AS volatility_30m
                    FROM fills
                    ORDER BY timestamp DESC
                    LIMIT 5000
                    """
                ).fetchall()
            except sqlite3.OperationalError:
                # Table may not have extended columns yet — fall back to base columns
                raw = conn.execute(
                    "SELECT side, quantity_usd, slippage_bps, timestamp FROM fills "
                    "ORDER BY timestamp DESC LIMIT 5000"
                ).fetchall()
            finally:
                conn.close()

            for row in raw:
                d: Dict[str, Any] = dict(row)
                d.setdefault("spread_bps", 5.0)
                d.setdefault("volatility_30m", 0.02)
                rows.append(d)

        except Exception:
            logger.exception("AdaptiveSlippageModel: failed to load fills from %s", self.db_path)

        logger.debug("AdaptiveSlippageModel: loaded %d fill rows from DB", len(rows))
        return rows

    def _build_dataset(
        self, rows: List[Dict[str, Any]]
    ) -> Tuple[List[List[float]], List[float]]:
        """Convert raw fill rows into (X, y) training arrays."""
        X: List[List[float]] = []
        y: List[float] = []
        for row in rows:
            try:
                side_enc = 1.0 if str(row.get("side", "buy")).lower() == "buy" else -1.0
                qty = float(row.get("quantity_usd", 0.0))
                # Normalise by a nominal ADV of $50M — same default as market_impact.py
                qty_norm = qty / 50_000_000.0
                ts = float(row.get("timestamp", 0.0))
                hour = int((ts % 86400) / 3600)
                spread = float(row.get("spread_bps", 5.0))
                vol = float(row.get("volatility_30m", 0.02))
                slip = float(row.get("slippage_bps", 0.0))
                X.append([side_enc, qty_norm, float(hour), spread, vol])
                y.append(slip)
            except (TypeError, ValueError):
                continue
        return X, y

    def _features_to_vector(self, f: SlippageFeatures) -> List[float]:
        """Convert a SlippageFeatures object to a plain float vector."""
        side_enc = 1.0 if f.side.lower() == "buy" else -1.0
        return [side_enc, f.quantity_norm, float(f.hour), f.spread_bps, f.volatility_30m]

    # -- Linear model -------------------------------------------------------

    def _fit_linear(self, X: List[List[float]], y: List[float]) -> Dict[str, float]:
        """
        Ordinary least-squares via normal equations (pure stdlib).

        Returns a dict of {feature_index: coefficient} plus 'intercept'.
        """
        n = len(X)
        p = len(X[0]) if X else 0

        # Build X matrix with bias column prepended
        X_aug = [[1.0] + row for row in X]

        # Normal equations: beta = (X^T X)^{-1} X^T y
        # Using a simple Gaussian elimination to avoid numpy dependency
        XtX = [[0.0] * (p + 1) for _ in range(p + 1)]
        Xty = [0.0] * (p + 1)
        for i in range(n):
            xi = X_aug[i]
            for j in range(p + 1):
                Xty[j] += xi[j] * y[i]
                for k in range(p + 1):
                    XtX[j][k] += xi[j] * xi[k]

        beta = self._solve_linear_system(XtX, Xty)
        coeffs: Dict[str, float] = {"intercept": beta[0]}
        for idx, coef in enumerate(beta[1:]):
            coeffs[str(idx)] = coef
        return coeffs

    def _fit_linear_and_store(
        self, X: List[List[float]], y: List[float]
    ) -> None:
        """Fit linear model and update internal state."""
        coeffs = self._fit_linear(X, y)
        self._linear_coeffs = coeffs
        self._linear_intercept = coeffs.get("intercept", _FALLBACK_PREDICTED_BPS)

        # Compute residuals
        preds = [self._predict_linear(xi) for xi in X]
        residuals = [yi - pi for yi, pi in zip(y, preds)]
        self._residual_std = self._std(residuals)
        self._mae_bps = sum(abs(r) for r in residuals) / max(len(residuals), 1)
        self._model_type = "linear"
        logger.info(
            "AdaptiveSlippageModel: linear model fitted n=%d MAE=%.3f bps",
            len(X), self._mae_bps,
        )

    def _predict_linear(self, x: List[float]) -> float:
        """Apply stored linear coefficients to feature vector x."""
        result = self._linear_intercept
        for idx, xi in enumerate(x):
            coef = self._linear_coeffs.get(str(idx), 0.0)
            result += coef * xi
        return result

    # -- sklearn model -------------------------------------------------------

    def _fit_sklearn(self, X: List[List[float]], y: List[float]) -> Any:
        """Fit a GradientBoostingRegressor; raises if sklearn unavailable."""
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("sklearn not available")

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_scaled, y)
        return model

    def _scale_X(self, X: List[List[float]]) -> Any:
        """Apply the stored scaler; returns a plain list if scaler is None."""
        if self._scaler is not None:
            return self._scaler.transform(X)
        return X

    def _predict_sklearn(self, x: List[float]) -> float:
        """Predict with the sklearn GBR model."""
        if self._sklearn_model is None:
            return _FALLBACK_PREDICTED_BPS
        try:
            x_scaled = self._scaler.transform([x]) if self._scaler else [x]
            return float(self._sklearn_model.predict(x_scaled)[0])
        except Exception:
            logger.exception("AdaptiveSlippageModel: sklearn predict failed; using fallback")
            return _FALLBACK_PREDICTED_BPS

    # -- Math utilities ------------------------------------------------------

    @staticmethod
    def _std(values: List[float]) -> float:
        """Population standard deviation of a list (pure stdlib)."""
        n = len(values)
        if n < 2:
            return _FALLBACK_CI_HALF / 1.96
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        return math.sqrt(max(variance, 0.0))

    @staticmethod
    def _solve_linear_system(
        A: List[List[float]], b: List[float]
    ) -> List[float]:
        """
        Solve Ax = b via Gaussian elimination with partial pivoting.

        Returns x (a list of floats).  Returns zeros on singular matrix.
        """
        n = len(b)
        # Augmented matrix
        M = [A[i][:] + [b[i]] for i in range(n)]

        for col in range(n):
            # Partial pivot
            pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
            M[col], M[pivot] = M[pivot], M[col]
            denom = M[col][col]
            if abs(denom) < 1e-12:
                logger.warning(
                    "AdaptiveSlippageModel: singular matrix in linear solve; "
                    "returning zeros"
                )
                return [0.0] * n
            for row in range(col + 1, n):
                factor = M[row][col] / denom
                M[row] = [M[row][k] - factor * M[col][k] for k in range(n + 1)]

        # Back substitution
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = M[i][n] / M[i][i]
            for j in range(i + 1, n):
                x[i] -= M[i][j] * x[j] / M[i][i]

        return x
