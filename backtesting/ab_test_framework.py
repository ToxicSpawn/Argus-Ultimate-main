"""
A/B Testing Framework for Trading Strategies — statistical experiment management.

Creates controlled experiments comparing a control strategy against a variant,
records per-trade outcomes, and uses Welch's t-test to determine whether the
variant is statistically significantly better.

Auto-promotion logic: if the variant wins with p < 0.05 and has enough
samples, it can be auto-promoted to replace the control.

Usage:
    from backtesting.ab_test_framework import ABTestFramework

    ab = ABTestFramework()
    eid = ab.create_experiment("momentum_v2_vs_v3", control_params={...}, variant_params={...})
    ab.record_trade(eid, "control", pnl=12.5, slippage_bps=3.2)
    ab.record_trade(eid, "variant", pnl=15.0, slippage_bps=2.1)
    result = ab.get_results(eid)
    print(result.winner, result.p_value)
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Attempt scipy for Welch's t-test; fall back to manual implementation
try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

_DEFAULT_SIGNIFICANCE = 0.05  # p-value threshold


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VariantMetrics:
    """Aggregated metrics for one side of an A/B experiment."""

    sample_size: int = 0
    mean_pnl: float = 0.0
    std_pnl: float = 0.0
    total_pnl: float = 0.0
    mean_slippage_bps: float = 0.0
    win_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ABTestResult:
    """Statistical result of an A/B experiment."""

    name: str
    control_metrics: VariantMetrics = field(default_factory=VariantMetrics)
    variant_metrics: VariantMetrics = field(default_factory=VariantMetrics)
    p_value: float = 1.0
    significant: bool = False
    winner: str = "inconclusive"  # "control" / "variant" / "inconclusive"
    sample_sizes: Dict[str, int] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Welch's t-test (manual fallback)
# ---------------------------------------------------------------------------

def _welch_t_test(
    mean1: float, std1: float, n1: int,
    mean2: float, std2: float, n2: int,
) -> Tuple[float, float]:
    """Compute Welch's t-statistic and approximate p-value.

    Returns
    -------
    tuple of (t_stat, p_value)
    """
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    se1 = (std1 ** 2) / n1
    se2 = (std2 ** 2) / n2
    se_total = se1 + se2
    if se_total < 1e-15:
        return 0.0, 1.0

    t_stat = (mean1 - mean2) / math.sqrt(se_total)

    # Welch-Satterthwaite degrees of freedom
    num = se_total ** 2
    denom = (se1 ** 2) / (n1 - 1) + (se2 ** 2) / (n2 - 1)
    if denom < 1e-15:
        df = min(n1, n2) - 1
    else:
        df = num / denom
    df = max(df, 1.0)

    # Approximate p-value using t-distribution CDF via regularised incomplete
    # beta function.  This is a rough but serviceable approximation.
    if _SCIPY_AVAILABLE:
        p_value = _scipy_stats.t.sf(abs(t_stat), df) * 2  # Two-tailed
        return float(t_stat), float(max(0.0, min(1.0, p_value)))

    # Manual approximation: use normal approximation for large df
    # For df > 30 this is reasonably accurate
    if df > 30:
        # Normal approximation
        z = abs(t_stat)
        # Abramowitz & Stegun approximation of erfc
        p = 0.3275911
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        t_val = 1.0 / (1.0 + p * z / math.sqrt(2))
        erf_approx = 1.0 - (a1 * t_val + a2 * t_val ** 2 + a3 * t_val ** 3 +
                             a4 * t_val ** 4 + a5 * t_val ** 5) * math.exp(-z ** 2 / 2)
        p_value = 1.0 - erf_approx
    else:
        # Crude approximation for small df — wider distribution
        z = abs(t_stat)
        expansion = 1 + z ** 2 / df
        p_value = expansion ** (-(df + 1) / 2) * 0.75  # Rough scaling

    return float(t_stat), float(max(0.0, min(1.0, p_value)))


# ---------------------------------------------------------------------------
# ABTestFramework
# ---------------------------------------------------------------------------

class ABTestFramework:
    """A/B testing framework for comparing trading strategy variants.

    Parameters
    ----------
    db_path : str or Path
        SQLite database for persistence.
    significance_level : float
        p-value threshold for declaring significance (default 0.05).
    """

    def __init__(
        self,
        db_path: str = "data/ab_tests.db",
        *,
        significance_level: float = _DEFAULT_SIGNIFICANCE,
    ) -> None:
        self.db_path = Path(db_path)
        self.significance_level = significance_level
        self._lock = threading.Lock()
        self._ensure_db()

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _ensure_db(self) -> None:
        """Create SQLite tables if absent."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id   TEXT PRIMARY KEY,
                    name            TEXT NOT NULL,
                    control_params  TEXT DEFAULT '{}',
                    variant_params  TEXT DEFAULT '{}',
                    min_trades      INTEGER DEFAULT 50,
                    created_ts      TEXT NOT NULL,
                    status          TEXT DEFAULT 'running'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id   TEXT NOT NULL,
                    variant         TEXT NOT NULL,
                    pnl             REAL NOT NULL,
                    slippage_bps    REAL DEFAULT 0.0,
                    ts              TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_exp_var ON trades(experiment_id, variant)"
            )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id   TEXT NOT NULL,
                    winner          TEXT NOT NULL,
                    p_value         REAL NOT NULL,
                    significant     INTEGER NOT NULL,
                    ts              TEXT NOT NULL
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        control_params: Dict[str, Any],
        variant_params: Dict[str, Any],
        min_trades: int = 50,
    ) -> str:
        """Create a new A/B test experiment.

        Parameters
        ----------
        name : str
            Human-readable experiment name.
        control_params : dict
            Configuration parameters for the control strategy.
        variant_params : dict
            Configuration parameters for the variant strategy.
        min_trades : int
            Minimum trades per variant before results are considered valid.

        Returns
        -------
        str
            Unique experiment ID.
        """
        experiment_id = str(uuid.uuid4())[:12]
        ts = datetime.now(timezone.utc).isoformat()

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO experiments (experiment_id, name, control_params, variant_params, min_trades, created_ts)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (experiment_id, name, json.dumps(control_params), json.dumps(variant_params),
                     min_trades, ts),
                )

        logger.info(
            "ABTestFramework: created experiment '%s' id=%s min_trades=%d",
            name, experiment_id, min_trades,
        )
        return experiment_id

    def record_trade(
        self,
        experiment_id: str,
        variant: str,
        pnl: float,
        slippage_bps: float = 0.0,
    ) -> None:
        """Record a trade result for one side of the experiment.

        Parameters
        ----------
        experiment_id : str
            Experiment identifier.
        variant : str
            ``control`` or ``variant``.
        pnl : float
            Profit/loss of the trade.
        slippage_bps : float
            Slippage in basis points.

        Raises
        ------
        ValueError
            If variant is not ``control`` or ``variant``.
        """
        if variant not in ("control", "variant"):
            raise ValueError(f"variant must be 'control' or 'variant', got '{variant}'")

        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO trades (experiment_id, variant, pnl, slippage_bps, ts) VALUES (?, ?, ?, ?, ?)",
                    (experiment_id, variant, pnl, slippage_bps, ts),
                )

        logger.debug(
            "ABTestFramework: recorded %s trade for %s — pnl=%.4f slip=%.2fbps",
            variant, experiment_id, pnl, slippage_bps,
        )

    def get_results(self, experiment_id: str) -> ABTestResult:
        """Compute current A/B test results with statistical significance.

        Uses Welch's t-test to compare PnL distributions between control
        and variant.  If scipy is available it uses ``scipy.stats.ttest_ind``;
        otherwise falls back to a manual implementation.

        Parameters
        ----------
        experiment_id : str
            Experiment identifier.

        Returns
        -------
        ABTestResult
        """
        with self._lock:
            with self._connect() as conn:
                # Experiment metadata
                cursor = conn.execute(
                    "SELECT name, min_trades FROM experiments WHERE experiment_id = ?",
                    (experiment_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return ABTestResult(name="unknown", winner="inconclusive")
                name, min_trades = row

                # Control trades
                cursor = conn.execute(
                    "SELECT pnl, slippage_bps FROM trades WHERE experiment_id = ? AND variant = 'control'",
                    (experiment_id,),
                )
                control_trades = cursor.fetchall()

                # Variant trades
                cursor = conn.execute(
                    "SELECT pnl, slippage_bps FROM trades WHERE experiment_id = ? AND variant = 'variant'",
                    (experiment_id,),
                )
                variant_trades = cursor.fetchall()

        control_metrics = self._compute_variant_metrics(control_trades)
        variant_metrics = self._compute_variant_metrics(variant_trades)

        # Statistical test
        n_c = control_metrics.sample_size
        n_v = variant_metrics.sample_size

        if n_c < 2 or n_v < 2:
            p_value = 1.0
            t_stat = 0.0
        elif _SCIPY_AVAILABLE:
            control_pnls = [t[0] for t in control_trades]
            variant_pnls = [t[0] for t in variant_trades]
            t_stat, p_value = _scipy_stats.ttest_ind(
                variant_pnls, control_pnls, equal_var=False
            )
            p_value = float(p_value)
        else:
            t_stat, p_value = _welch_t_test(
                variant_metrics.mean_pnl, variant_metrics.std_pnl, n_v,
                control_metrics.mean_pnl, control_metrics.std_pnl, n_c,
            )

        significant = p_value < self.significance_level
        enough_samples = n_c >= min_trades and n_v >= min_trades

        if significant and enough_samples:
            if variant_metrics.mean_pnl > control_metrics.mean_pnl:
                winner = "variant"
            elif control_metrics.mean_pnl > variant_metrics.mean_pnl:
                winner = "control"
            else:
                winner = "inconclusive"
        else:
            winner = "inconclusive"

        result = ABTestResult(
            name=name,
            control_metrics=control_metrics,
            variant_metrics=variant_metrics,
            p_value=round(p_value, 6),
            significant=significant,
            winner=winner,
            sample_sizes={"control": n_c, "variant": n_v},
        )

        # Persist result
        self._persist_result(experiment_id, result)

        logger.info(
            "ABTestFramework: results for '%s' — winner=%s p=%.6f sig=%s (n_c=%d n_v=%d)",
            name, winner, p_value, significant, n_c, n_v,
        )
        return result

    def auto_promote(self, experiment_id: str) -> bool:
        """Auto-promote the variant if it wins significantly.

        Parameters
        ----------
        experiment_id : str
            Experiment identifier.

        Returns
        -------
        bool
            True if variant was promoted (i.e., it won with statistical
            significance and sufficient sample size).
        """
        result = self.get_results(experiment_id)
        if result.winner == "variant" and result.significant:
            with self._lock:
                with self._connect() as conn:
                    conn.execute(
                        "UPDATE experiments SET status = 'promoted' WHERE experiment_id = ?",
                        (experiment_id,),
                    )
            logger.info(
                "ABTestFramework: auto-promoted variant in experiment '%s' (p=%.6f)",
                result.name, result.p_value,
            )
            return True

        logger.info(
            "ABTestFramework: variant not promoted in '%s' — winner=%s sig=%s",
            result.name, result.winner, result.significant,
        )
        return False

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments with their current status.

        Returns
        -------
        list of dict
        """
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    "SELECT experiment_id, name, min_trades, status, created_ts FROM experiments ORDER BY created_ts DESC"
                )
                return [
                    {
                        "experiment_id": r[0],
                        "name": r[1],
                        "min_trades": r[2],
                        "status": r[3],
                        "created_ts": r[4],
                    }
                    for r in cursor.fetchall()
                ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_variant_metrics(trades: List[Tuple[float, float]]) -> VariantMetrics:
        """Compute aggregate metrics from a list of (pnl, slippage_bps) tuples."""
        if not trades:
            return VariantMetrics()

        pnls = [t[0] for t in trades]
        slippages = [t[1] for t in trades]
        n = len(pnls)
        total_pnl = sum(pnls)
        mean_pnl = total_pnl / n
        std_pnl = (sum((p - mean_pnl) ** 2 for p in pnls) / max(n - 1, 1)) ** 0.5
        mean_slip = sum(slippages) / n
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / n

        return VariantMetrics(
            sample_size=n,
            mean_pnl=round(mean_pnl, 6),
            std_pnl=round(std_pnl, 6),
            total_pnl=round(total_pnl, 6),
            mean_slippage_bps=round(mean_slip, 4),
            win_rate=round(win_rate, 4),
        )

    def _persist_result(self, experiment_id: str, result: ABTestResult) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO results_log (experiment_id, winner, p_value, significant, ts) VALUES (?, ?, ?, ?, ?)",
                    (experiment_id, result.winner, result.p_value, int(result.significant), ts),
                )
