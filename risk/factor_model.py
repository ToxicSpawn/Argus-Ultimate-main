#!/usr/bin/env python3
"""
Return Factor Decomposition — attribute asset returns to systematic factors.

Factors modelled:
    1. Market (beta to aggregate crypto index)
    2. Momentum (trailing return loading)
    3. Volatility (rolling vol loading)
    4. Residual / idiosyncratic alpha

Uses OLS regression (numpy.linalg.lstsq) with a pure-Python fallback for
environments without numpy.

Standalone usage:
    fm = FactorModel()
    fm.update_returns("BTC/AUD", 0.023, 0.018)
    decomp = fm.decompose("BTC/AUD")
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

try:
    import numpy as np  # type: ignore[import-untyped]

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False
    np = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FactorDecomposition:
    """Result of a single-asset factor decomposition."""

    symbol: str
    alpha: float  # annualised intercept (daily alpha * 365)
    market_beta: float  # sensitivity to market returns
    momentum_loading: float  # sensitivity to momentum factor
    volatility_loading: float  # sensitivity to volatility factor
    residual_pct: float  # fraction of variance unexplained by factors
    r_squared: float  # goodness of fit (0–1)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Pure-Python OLS helpers
# ---------------------------------------------------------------------------

def _py_mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _py_ols_simple(y: List[float], X: List[List[float]]) -> Tuple[List[float], float]:
    """
    Ordinary Least Squares via normal equations (pure Python).

    Returns (coefficients, r_squared).  X is column-major list-of-columns.
    """
    n = len(y)
    k = len(X)
    if n < k + 1:
        return [0.0] * (k + 1), 0.0

    # Add intercept column
    cols = [[1.0] * n] + X  # intercept first
    m = k + 1

    # XtX = X^T @ X  (m x m)
    XtX = [[sum(cols[i][t] * cols[j][t] for t in range(n)) for j in range(m)] for i in range(m)]
    # XtY = X^T @ y  (m,)
    XtY = [sum(cols[i][t] * y[t] for t in range(n)) for i in range(m)]

    # Solve via Gaussian elimination
    aug = [XtX[i][:] + [XtY[i]] for i in range(m)]
    for i in range(m):
        # Pivot
        max_row = max(range(i, m), key=lambda r: abs(aug[r][i]))
        aug[i], aug[max_row] = aug[max_row], aug[i]
        piv = aug[i][i]
        if abs(piv) < 1e-15:
            return [0.0] * (k + 1), 0.0
        for j in range(i, m + 1):
            aug[i][j] /= piv
        for r in range(m):
            if r != i:
                factor = aug[r][i]
                for j in range(i, m + 1):
                    aug[r][j] -= factor * aug[i][j]

    beta = [aug[i][m] for i in range(m)]

    # R-squared
    y_mean = _py_mean(y)
    ss_tot = sum((y[t] - y_mean) ** 2 for t in range(n))
    y_hat = [sum(beta[j] * cols[j][t] for j in range(m)) for t in range(n)]
    ss_res = sum((y[t] - y_hat[t]) ** 2 for t in range(n))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    r2 = max(0.0, min(1.0, r2))

    return beta, r2


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FactorModel:
    """
    Multi-factor return decomposition for crypto assets.

    Factors:
        - Market (crypto aggregate return)
        - Momentum (5-period rolling return)
        - Volatility (5-period rolling standard deviation)

    Parameters
    ----------
    db_path : str
        Path to SQLite database for persistence (default ``data/factor_model.db``).
    momentum_window : int
        Lookback for momentum factor construction (default 5).
    vol_window : int
        Lookback for volatility factor construction (default 5).
    """

    def __init__(
        self,
        db_path: str = "data/factor_model.db",
        momentum_window: int = 5,
        vol_window: int = 5,
    ):
        self.db_path = db_path
        self.momentum_window = momentum_window
        self.vol_window = vol_window
        self._lock = threading.Lock()

        # In-memory return storage: symbol -> [(timestamp, return_pct, market_return_pct)]
        self._returns: Dict[str, List[Tuple[float, float, float]]] = {}

        # Cache of latest decompositions
        self._cache: Dict[str, FactorDecomposition] = {}

        self._init_db()
        logger.info(
            "FactorModel initialised (db=%s, momentum_window=%d, vol_window=%d, numpy=%s)",
            db_path, momentum_window, vol_window, _HAS_NUMPY,
        )

    # ------------------------------------------------------------------
    # DB setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the SQLite tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS factor_returns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    return_pct REAL NOT NULL,
                    market_return_pct REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS factor_decompositions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    alpha REAL NOT NULL,
                    market_beta REAL NOT NULL,
                    momentum_loading REAL NOT NULL,
                    volatility_loading REAL NOT NULL,
                    residual_pct REAL NOT NULL,
                    r_squared REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fr_symbol_ts ON factor_returns(symbol, timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fd_symbol_ts ON factor_decompositions(symbol, timestamp)"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_returns(
        self,
        symbol: str,
        return_pct: float,
        market_return_pct: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record a new return observation for *symbol*.

        Parameters
        ----------
        symbol : str
            Asset identifier (e.g. "BTC/AUD").
        return_pct : float
            Period return as a decimal (0.02 = 2%).
        market_return_pct : float
            Aggregate market return for the same period.
        timestamp : float, optional
            Epoch seconds (defaults to now).
        """
        ts = timestamp or time.time()
        with self._lock:
            self._returns.setdefault(symbol, []).append((ts, return_pct, market_return_pct))

        # Persist
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO factor_returns (symbol, timestamp, return_pct, market_return_pct) VALUES (?, ?, ?, ?)",
                    (symbol, ts, return_pct, market_return_pct),
                )
        except Exception:
            logger.warning("FactorModel: failed to persist return for %s", symbol, exc_info=True)

    def decompose(self, symbol: str, lookback_days: int = 30) -> Optional[FactorDecomposition]:
        """
        Decompose recent returns for *symbol* into factor loadings.

        Parameters
        ----------
        symbol : str
            Asset to analyse.
        lookback_days : int
            How many periods of data to use (default 30).

        Returns
        -------
        FactorDecomposition or None
            Decomposition result, or None if insufficient data.
        """
        with self._lock:
            raw = list(self._returns.get(symbol, []))

        if len(raw) < max(lookback_days, self.momentum_window + 2, self.vol_window + 2):
            logger.info(
                "decompose(%s): insufficient data (%d obs, need %d)",
                symbol, len(raw), lookback_days,
            )
            return None

        # Use last *lookback_days* observations
        raw = raw[-lookback_days:]
        y = [r[1] for r in raw]  # asset returns
        mkt = [r[2] for r in raw]  # market returns

        # Construct factor columns
        momentum = self._build_momentum(y)
        vol_factor = self._build_vol_factor(y)

        # Align all series (trim leading NaNs from momentum/vol construction)
        start = max(self.momentum_window, self.vol_window)
        if start >= len(y):
            return None

        y_trim = y[start:]
        mkt_trim = mkt[start:]
        mom_trim = momentum[start:]
        vol_trim = vol_factor[start:]
        n = len(y_trim)
        if n < 5:
            return None

        # Run regression
        if _HAS_NUMPY:
            beta, r2 = self._numpy_ols(y_trim, mkt_trim, mom_trim, vol_trim)
        else:
            beta, r2 = _py_ols_simple(y_trim, [mkt_trim, mom_trim, vol_trim])

        # beta: [intercept, market_beta, momentum_loading, volatility_loading]
        alpha_daily = beta[0]
        alpha_ann = alpha_daily * 365  # annualise

        decomp = FactorDecomposition(
            symbol=symbol,
            alpha=alpha_ann,
            market_beta=beta[1],
            momentum_loading=beta[2],
            volatility_loading=beta[3],
            residual_pct=1.0 - r2,
            r_squared=r2,
        )

        self._cache[symbol] = decomp
        self._persist_decomposition(decomp)
        logger.debug(
            "Factor decomposition %s: alpha=%.4f, beta=%.3f, r2=%.3f",
            symbol, decomp.alpha, decomp.market_beta, decomp.r_squared,
        )
        return decomp

    def get_portfolio_factor_exposure(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Aggregate factor exposures across a weighted portfolio.

        Parameters
        ----------
        weights : dict
            Symbol -> portfolio weight.

        Returns
        -------
        dict
            Factor name -> weighted total exposure.
        """
        exposure = {"alpha": 0.0, "market_beta": 0.0, "momentum": 0.0, "volatility": 0.0}
        for symbol, w in weights.items():
            decomp = self._cache.get(symbol)
            if decomp is None:
                logger.debug("get_portfolio_factor_exposure: no decomposition cached for %s", symbol)
                continue
            exposure["alpha"] += w * decomp.alpha
            exposure["market_beta"] += w * decomp.market_beta
            exposure["momentum"] += w * decomp.momentum_loading
            exposure["volatility"] += w * decomp.volatility_loading
        return exposure

    def get_alpha_ranked(self) -> List[Tuple[str, float]]:
        """
        Return all cached assets ranked by alpha (descending).

        Returns
        -------
        list of (symbol, alpha)
        """
        items = [(s, d.alpha) for s, d in self._cache.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return items

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_momentum(self, returns: List[float]) -> List[float]:
        """Trailing cumulative return over *momentum_window* periods."""
        out = [0.0] * len(returns)
        for i in range(self.momentum_window, len(returns)):
            cum = 1.0
            for j in range(i - self.momentum_window, i):
                cum *= (1.0 + returns[j])
            out[i] = cum - 1.0
        return out

    def _build_vol_factor(self, returns: List[float]) -> List[float]:
        """Rolling standard deviation over *vol_window* periods."""
        out = [0.0] * len(returns)
        for i in range(self.vol_window, len(returns)):
            window = returns[i - self.vol_window: i]
            mu = sum(window) / len(window)
            var = sum((x - mu) ** 2 for x in window) / len(window)
            out[i] = math.sqrt(var)
        return out

    def _numpy_ols(
        self,
        y: List[float],
        mkt: List[float],
        mom: List[float],
        vol: List[float],
    ) -> Tuple[List[float], float]:
        """OLS via numpy.linalg.lstsq."""
        n = len(y)
        Y = np.array(y)
        X = np.column_stack([np.ones(n), mkt, mom, vol])
        beta, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)

        y_mean = float(Y.mean())
        ss_tot = float(np.sum((Y - y_mean) ** 2))
        Y_hat = X @ beta
        ss_res = float(np.sum((Y - Y_hat) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
        r2 = max(0.0, min(1.0, r2))

        return beta.tolist(), r2

    def _persist_decomposition(self, d: FactorDecomposition) -> None:
        """Save decomposition to SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO factor_decompositions
                       (symbol, alpha, market_beta, momentum_loading, volatility_loading,
                        residual_pct, r_squared, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (d.symbol, d.alpha, d.market_beta, d.momentum_loading,
                     d.volatility_loading, d.residual_pct, d.r_squared, d.timestamp),
                )
        except Exception:
            logger.warning("FactorModel: failed to persist decomposition for %s", d.symbol, exc_info=True)
