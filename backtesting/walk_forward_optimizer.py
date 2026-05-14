"""
Walk-Forward Optimiser for Argus Ultimate.

Performs rolling in-sample / out-of-sample optimisation across a full
price history to detect parameter overfitting and measure true OOS
performance.  Uses Optuna (TPE sampler) for hyperparameter search with
support for multiple objective functions.

Key concepts
------------
* **Window**: a contiguous slice of the data split into IS (in-sample)
  and OOS (out-of-sample) segments.
* **Rolling mode**: the IS window slides forward with each iteration
  (fixed IS size, anchor moves).
* **Anchored mode**: the IS window always starts at index 0 and grows
  (expanding window).
* **Overfitting coefficient**: ratio of OOS Sharpe to IS Sharpe across
  windows.  Values near 1.0 indicate robust parameters; values < 0.5
  suggest severe overfitting.

Quick start
-----------
::

    from backtesting.walk_forward_optimizer import WalkForwardOptimizer, WFOConfig
    from backtesting.walk_forward_optimizer import CalmarLoss

    config = WFOConfig(
        n_windows=8,
        is_fraction=0.7,
        n_trials=50,
        objective=CalmarLoss(),
    )
    wfo = WalkForwardOptimizer(config=config)
    report = wfo.run(
        ohlcv=df,           # pandas DataFrame with OHLCV columns
        strategy_cls=MyStrategy,
        param_space={
            "rsi_period": ("int", 7, 28),
            "rsi_overbought": ("float", 65.0, 85.0),
            "atr_multiplier": ("float", 1.5, 4.0),
        },
    )
    print(report.summary())

CLI
---
::

    python -m backtesting.walk_forward_optimizer \\
        --data data/BTC_1h.parquet \\
        --strategy strategies.rsi_regime.RSIRegimeStrategy \\
        --n-windows 8 --is-fraction 0.7 --n-trials 100
"""

from __future__ import annotations

import importlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Type

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation so the module loads without them
# ---------------------------------------------------------------------------

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    pd = None  # type: ignore
    _PANDAS = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA = True
except ImportError:
    optuna = None  # type: ignore
    _OPTUNA = False


# ---------------------------------------------------------------------------
# Strategy protocol — duck-typed so any Argus strategy works
# ---------------------------------------------------------------------------


class StrategyProtocol(Protocol):
    """
    Minimal interface expected from an Argus strategy for WFO.

    A strategy must accept keyword params at construction time and expose
    a ``backtest()`` method that returns a dict of performance metrics.
    """

    def __init__(self, **params: Any) -> None: ...

    def backtest(
        self,
        ohlcv: Any,
        initial_capital: float = 10_000.0,
    ) -> Dict[str, float]:
        """
        Run a backtest on the supplied OHLCV data.

        Must return a dict containing at least::

            {
                "total_return": float,     # e.g. 0.15 for 15%
                "sharpe_ratio": float,
                "max_drawdown": float,     # positive fraction, e.g. 0.12
                "calmar_ratio": float,
                "profit_factor": float,
                "n_trades": int,
            }
        """
        ...


# ---------------------------------------------------------------------------
# Loss / objective functions
# ---------------------------------------------------------------------------


class _BaseLoss:
    """Base class for WFO objective functions (Optuna minimises, so negate gains)."""

    name: str = "base"

    def __call__(self, metrics: Dict[str, float]) -> float:
        raise NotImplementedError

    def direction(self) -> str:
        """Optuna direction string: always 'minimize' (we negate positives)."""
        return "minimize"


class SharpeLoss(_BaseLoss):
    """Maximise Sharpe ratio."""

    name = "sharpe"

    def __call__(self, metrics: Dict[str, float]) -> float:
        sharpe = metrics.get("sharpe_ratio", 0.0)
        # Penalise if fewer than 5 trades — unreliable
        if metrics.get("n_trades", 0) < 5:
            return 1e6
        return -sharpe


class CalmarLoss(_BaseLoss):
    """Maximise Calmar ratio (CAGR / MaxDD).  Best for drawdown-sensitive use."""

    name = "calmar"

    def __call__(self, metrics: Dict[str, float]) -> float:
        calmar = metrics.get("calmar_ratio", 0.0)
        if metrics.get("n_trades", 0) < 5:
            return 1e6
        return -calmar


class SortinoLoss(_BaseLoss):
    """Maximise Sortino ratio."""

    name = "sortino"

    def __call__(self, metrics: Dict[str, float]) -> float:
        sortino = metrics.get("sortino_ratio", metrics.get("sharpe_ratio", 0.0))
        if metrics.get("n_trades", 0) < 5:
            return 1e6
        return -sortino


class ProfitFactorLoss(_BaseLoss):
    """Maximise profit factor (gross profit / gross loss)."""

    name = "profit_factor"

    def __call__(self, metrics: Dict[str, float]) -> float:
        pf = metrics.get("profit_factor", 0.0)
        if metrics.get("n_trades", 0) < 5:
            return 1e6
        return -pf


class ReturnOverDrawdownLoss(_BaseLoss):
    """Maximise total return / max drawdown (RoMaD)."""

    name = "romad"

    def __call__(self, metrics: Dict[str, float]) -> float:
        ret = metrics.get("total_return", 0.0)
        dd = max(metrics.get("max_drawdown", 1.0), 1e-6)
        if metrics.get("n_trades", 0) < 5:
            return 1e6
        return -(ret / dd)


_LOSS_REGISTRY: Dict[str, _BaseLoss] = {
    "sharpe": SharpeLoss(),
    "calmar": CalmarLoss(),
    "sortino": SortinoLoss(),
    "profit_factor": ProfitFactorLoss(),
    "romad": ReturnOverDrawdownLoss(),
}


# ---------------------------------------------------------------------------
# Window dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WFOWindow:
    """Indices describing a single IS/OOS window."""

    window_idx: int
    is_start: int
    is_end: int  # exclusive
    oos_start: int
    oos_end: int  # exclusive


@dataclass
class WFOWindowResult:
    """Results for one IS/OOS window."""

    window: WFOWindow
    best_params: Dict[str, Any]
    is_metrics: Dict[str, float]
    oos_metrics: Dict[str, float]
    is_loss: float
    oos_loss: float
    n_trials: int
    elapsed_seconds: float

    @property
    def is_sharpe(self) -> float:
        return self.is_metrics.get("sharpe_ratio", 0.0)

    @property
    def oos_sharpe(self) -> float:
        return self.oos_metrics.get("sharpe_ratio", 0.0)

    @property
    def overfitting_ratio(self) -> float:
        """OOS Sharpe / IS Sharpe.  ~1.0 = robust, <0.5 = overfit."""
        if abs(self.is_sharpe) < 1e-6:
            return 0.0
        return self.oos_sharpe / self.is_sharpe


@dataclass
class WFOReport:
    """Aggregated report across all windows."""

    config: WFOConfig
    windows: List[WFOWindowResult] = field(default_factory=list)
    strategy_name: str = ""
    data_rows: int = 0
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "=" * 60,
            f"Walk-Forward Optimisation Report — {self.strategy_name}",
            f"Windows: {len(self.windows)}  |  Data rows: {self.data_rows}",
            f"Objective: {self.config.objective.name}  |  "
            f"Trials/window: {self.config.n_trials}",
            f"IS fraction: {self.config.is_fraction:.0%}  |  "
            f"Mode: {'anchored' if self.config.anchored else 'rolling'}",
            "-" * 60,
        ]
        for r in self.windows:
            ovr = r.overfitting_ratio
            flag = "✓" if ovr >= 0.5 else "⚠ OVERFIT"
            lines.append(
                f"  Win {r.window.window_idx:02d}  "
                f"IS Sharpe={r.is_sharpe:+.3f}  "
                f"OOS Sharpe={r.oos_sharpe:+.3f}  "
                f"OvR={ovr:.2f} {flag}  "
                f"trades={r.oos_metrics.get('n_trades', 0)}"
            )
        lines.append("-" * 60)
        lines.append(f"Mean OOS Sharpe : {self.mean_oos_sharpe:+.3f}")
        lines.append(f"Mean Overfitting: {self.mean_overfitting_ratio:.3f}")
        lines.append(f"Robust windows  : {self.n_robust_windows}/{len(self.windows)}")
        lines.append(f"Total elapsed   : {self.elapsed_seconds:.1f}s")
        lines.append("=" * 60)
        return "\n".join(lines)

    @property
    def mean_oos_sharpe(self) -> float:
        if not self.windows:
            return 0.0
        return float(np.mean([w.oos_sharpe for w in self.windows]))

    @property
    def mean_overfitting_ratio(self) -> float:
        if not self.windows:
            return 0.0
        return float(np.mean([w.overfitting_ratio for w in self.windows]))

    @property
    def n_robust_windows(self) -> int:
        return sum(1 for w in self.windows if w.overfitting_ratio >= 0.5)

    def best_params_by_oos_sharpe(self) -> Dict[str, Any]:
        """Return the best_params from the window with the highest OOS Sharpe."""
        if not self.windows:
            return {}
        best = max(self.windows, key=lambda w: w.oos_sharpe)
        return best.best_params

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "data_rows": self.data_rows,
            "mean_oos_sharpe": self.mean_oos_sharpe,
            "mean_overfitting_ratio": self.mean_overfitting_ratio,
            "n_robust_windows": self.n_robust_windows,
            "windows": [
                {
                    "idx": r.window.window_idx,
                    "best_params": r.best_params,
                    "is_sharpe": r.is_sharpe,
                    "oos_sharpe": r.oos_sharpe,
                    "overfitting_ratio": r.overfitting_ratio,
                    "oos_metrics": r.oos_metrics,
                }
                for r in self.windows
            ],
        }

    def save(self, path: str) -> None:
        """Save report as JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("WFO report saved to %s", path)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WFOConfig:
    """
    Configuration for the walk-forward optimiser.

    Attributes:
        n_windows:          Number of IS/OOS windows.
        is_fraction:        Fraction of each window used for IS (e.g. 0.7 = 70%).
        n_trials:           Optuna trials per window per IS segment.
        objective:          Loss function instance (default: CalmarLoss).
        anchored:           If True, IS always starts at row 0 (expanding window).
                            If False (default), rolling window of fixed IS size.
        min_is_rows:        Minimum rows required in IS segment (skip if below).
        min_oos_rows:       Minimum rows required in OOS segment (skip if below).
        initial_capital:    Capital passed to strategy.backtest().
        n_jobs:             Optuna parallel jobs (-1 = all cores).
        sampler:            Optuna sampler name: "tpe" | "cmaes" | "random".
        pruner:             Optuna pruner name: "median" | "none".
        seed:               Random seed for reproducibility.
        verbose:            Log level for Optuna (0=silent, 1=warnings, 2=info).
    """

    n_windows: int = 8
    is_fraction: float = 0.7
    n_trials: int = 50
    objective: _BaseLoss = field(default_factory=CalmarLoss)
    anchored: bool = False
    min_is_rows: int = 100
    min_oos_rows: int = 30
    initial_capital: float = 10_000.0
    n_jobs: int = 1
    sampler: str = "tpe"
    pruner: str = "median"
    seed: int = 42
    verbose: int = 0


# ---------------------------------------------------------------------------
# Param space helpers
# ---------------------------------------------------------------------------


def _suggest_param(trial: Any, name: str, spec: Tuple) -> Any:
    """
    Suggest a parameter value from an Optuna trial.

    Param spec formats::

        ("int",   low, high)                  -> trial.suggest_int
        ("int",   low, high, step)            -> trial.suggest_int(..., step=step)
        ("float", low, high)                  -> trial.suggest_float
        ("float", low, high, "log")           -> trial.suggest_float(..., log=True)
        ("categorical", [val1, val2, ...])    -> trial.suggest_categorical
    """
    kind = spec[0]
    if kind == "int":
        low, high = int(spec[1]), int(spec[2])
        step = int(spec[3]) if len(spec) > 3 else 1
        return trial.suggest_int(name, low, high, step=step)
    elif kind == "float":
        low, high = float(spec[1]), float(spec[2])
        log = len(spec) > 3 and spec[3] == "log"
        return trial.suggest_float(name, low, high, log=log)
    elif kind == "categorical":
        return trial.suggest_categorical(name, spec[1])
    else:
        raise ValueError(f"Unknown param spec kind: {kind!r}")


# ---------------------------------------------------------------------------
# Core optimiser
# ---------------------------------------------------------------------------


class WalkForwardOptimizer:
    """
    Walk-Forward Optimiser — rolls IS/OOS windows over OHLCV data and
    optimises strategy parameters using Optuna on each IS window.

    See module docstring for usage examples.
    """

    def __init__(self, config: Optional[WFOConfig] = None):
        self.config = config or WFOConfig()
        if not _OPTUNA:
            raise ImportError(
                "optuna is required for WalkForwardOptimizer. "
                "Install with: pip install optuna"
            )
        if not _PANDAS:
            raise ImportError(
                "pandas is required for WalkForwardOptimizer. "
                "Install with: pip install pandas"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        ohlcv: Any,
        strategy_cls: Type,
        param_space: Dict[str, Tuple],
    ) -> WFOReport:
        """
        Run the full walk-forward optimisation.

        Args:
            ohlcv:        pandas DataFrame with at minimum columns
                          [open, high, low, close, volume].  Must be
                          sorted oldest-first with a DatetimeIndex.
            strategy_cls: Class implementing StrategyProtocol.
            param_space:  Dict of parameter name → spec tuple.
                          See ``_suggest_param`` for spec formats.

        Returns:
            WFOReport with per-window and aggregate results.
        """
        t0 = time.perf_counter()
        cfg = self.config
        n = len(ohlcv)
        report = WFOReport(
            config=cfg,
            strategy_name=strategy_cls.__name__,
            data_rows=n,
        )

        windows = self._build_windows(n, cfg)
        logger.info(
            "WFO: %d windows, %d rows, strategy=%s, objective=%s, trials=%d",
            len(windows), n, strategy_cls.__name__, cfg.objective.name, cfg.n_trials,
        )

        for w in windows:
            wr = self._run_window(ohlcv, strategy_cls, param_space, w)
            if wr is not None:
                report.windows.append(wr)
                logger.info(
                    "WFO win %02d: IS_sharpe=%.3f OOS_sharpe=%.3f OvR=%.2f params=%s",
                    w.window_idx, wr.is_sharpe, wr.oos_sharpe,
                    wr.overfitting_ratio, wr.best_params,
                )

        report.elapsed_seconds = time.perf_counter() - t0
        logger.info("WFO complete in %.1fs.  Mean OOS Sharpe=%.3f",
                    report.elapsed_seconds, report.mean_oos_sharpe)
        return report

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------

    def _build_windows(
        self, n_rows: int, cfg: WFOConfig
    ) -> List[WFOWindow]:
        """Build IS/OOS window index pairs."""
        windows = []
        window_size = n_rows // (cfg.n_windows + 1 if not cfg.anchored else 1)
        oos_size = max(cfg.min_oos_rows, int(window_size * (1.0 - cfg.is_fraction)))
        is_size = max(cfg.min_is_rows, int(window_size * cfg.is_fraction))

        if cfg.anchored:
            # Expanding IS window
            step = (n_rows - is_size - oos_size) // max(cfg.n_windows - 1, 1)
            for i in range(cfg.n_windows):
                oos_start = is_size + i * step
                oos_end = oos_start + oos_size
                if oos_end > n_rows:
                    break
                windows.append(WFOWindow(
                    window_idx=i,
                    is_start=0,
                    is_end=oos_start,
                    oos_start=oos_start,
                    oos_end=oos_end,
                ))
        else:
            # Rolling window
            total_window = is_size + oos_size
            step = (n_rows - total_window) // max(cfg.n_windows - 1, 1)
            for i in range(cfg.n_windows):
                is_start = i * step
                is_end = is_start + is_size
                oos_start = is_end
                oos_end = oos_start + oos_size
                if oos_end > n_rows:
                    break
                windows.append(WFOWindow(
                    window_idx=i,
                    is_start=is_start,
                    is_end=is_end,
                    oos_start=oos_start,
                    oos_end=oos_end,
                ))

        return windows

    # ------------------------------------------------------------------
    # Single window optimisation
    # ------------------------------------------------------------------

    def _run_window(
        self,
        ohlcv: Any,
        strategy_cls: Type,
        param_space: Dict[str, Tuple],
        window: WFOWindow,
    ) -> Optional[WFOWindowResult]:
        """Run Optuna optimisation on the IS slice, evaluate on OOS."""
        cfg = self.config
        is_data = ohlcv.iloc[window.is_start:window.is_end]
        oos_data = ohlcv.iloc[window.oos_start:window.oos_end]

        if len(is_data) < cfg.min_is_rows or len(oos_data) < cfg.min_oos_rows:
            logger.warning(
                "WFO win %02d: skipping — IS=%d rows, OOS=%d rows (minimums: %d/%d)",
                window.window_idx, len(is_data), len(oos_data),
                cfg.min_is_rows, cfg.min_oos_rows,
            )
            return None

        t0 = time.perf_counter()

        # Build Optuna sampler
        if cfg.sampler == "cmaes":
            sampler = optuna.samplers.CmaEsSampler(seed=cfg.seed)
        elif cfg.sampler == "random":
            sampler = optuna.samplers.RandomSampler(seed=cfg.seed)
        else:
            sampler = optuna.samplers.TPESampler(
                seed=cfg.seed, n_startup_trials=max(5, cfg.n_trials // 5)
            )

        # Build Optuna pruner
        pruner = (
            optuna.pruners.NopPruner()
            if cfg.pruner == "none"
            else optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        )

        study = optuna.create_study(
            direction=cfg.objective.direction(),
            sampler=sampler,
            pruner=pruner,
        )

        def objective(trial: Any) -> float:
            params = {k: _suggest_param(trial, k, v) for k, v in param_space.items()}
            try:
                strategy = strategy_cls(**params)
                metrics = strategy.backtest(
                    is_data, initial_capital=cfg.initial_capital
                )
                return cfg.objective(metrics)
            except Exception as exc:
                logger.debug("WFO trial failed: %s", exc)
                return 1e6

        study.optimize(
            objective,
            n_trials=cfg.n_trials,
            n_jobs=cfg.n_jobs,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_is_loss = study.best_value

        # Evaluate best params on IS and OOS
        try:
            is_strategy = strategy_cls(**best_params)
            is_metrics = is_strategy.backtest(is_data, initial_capital=cfg.initial_capital)
        except Exception as exc:
            logger.warning("WFO win %02d IS eval failed: %s", window.window_idx, exc)
            is_metrics = {"sharpe_ratio": 0.0, "n_trades": 0}

        try:
            oos_strategy = strategy_cls(**best_params)
            oos_metrics = oos_strategy.backtest(oos_data, initial_capital=cfg.initial_capital)
        except Exception as exc:
            logger.warning("WFO win %02d OOS eval failed: %s", window.window_idx, exc)
            oos_metrics = {"sharpe_ratio": 0.0, "n_trades": 0}

        oos_loss = cfg.objective(oos_metrics)

        return WFOWindowResult(
            window=window,
            best_params=best_params,
            is_metrics=is_metrics,
            oos_metrics=oos_metrics,
            is_loss=best_is_loss,
            oos_loss=oos_loss,
            n_trials=len(study.trials),
            elapsed_seconds=time.perf_counter() - t0,
        )


# ---------------------------------------------------------------------------
# Stability analysis utilities
# ---------------------------------------------------------------------------


def compute_parameter_stability(report: WFOReport) -> Dict[str, Dict[str, float]]:
    """
    Compute per-parameter stability statistics across windows.

    Returns a dict: param_name → {"mean", "std", "cv", "stable"}.
    ``cv`` is coefficient of variation (std/mean).  ``stable`` is True
    if cv < 0.3 (arbitrarily — parameters that don't drift much).
    """
    if not report.windows:
        return {}

    all_params = [w.best_params for w in report.windows]
    param_names = list(all_params[0].keys())
    result: Dict[str, Dict[str, float]] = {}

    for name in param_names:
        vals = [p[name] for p in all_params if isinstance(p.get(name), (int, float))]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        cv = std / abs(mean) if abs(mean) > 1e-9 else float("inf")
        result[name] = {
            "mean": mean,
            "std": std,
            "cv": cv,
            "stable": cv < 0.3,
        }

    return result


def recommended_params(report: WFOReport) -> Dict[str, Any]:
    """
    Return recommended live-trading parameters derived from the WFO report.

    Strategy: use the median value of each parameter across all windows
    (median is more robust to outlier windows than the mean).
    Only includes windows with overfitting_ratio >= 0.5.
    """
    robust_windows = [w for w in report.windows if w.overfitting_ratio >= 0.5]
    if not robust_windows:
        logger.warning("No robust windows found — using all windows for recommendation")
        robust_windows = report.windows
    if not robust_windows:
        return {}

    all_params = [w.best_params for w in robust_windows]
    param_names = list(all_params[0].keys())
    result: Dict[str, Any] = {}

    for name in param_names:
        vals = [p[name] for p in all_params]
        if all(isinstance(v, (int, float)) for v in vals):
            median_val = float(np.median(vals))
            # Preserve int type if original was int
            if all(isinstance(v, int) for v in vals):
                result[name] = int(round(median_val))
            else:
                result[name] = median_val
        else:
            # Categorical: most common value
            from collections import Counter
            result[name] = Counter(vals).most_common(1)[0][0]

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Argus Walk-Forward Optimiser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True,
                        help="Path to OHLCV parquet or CSV file")
    parser.add_argument("--strategy", required=True,
                        help="Dotted import path to strategy class, e.g. strategies.rsi_regime.RSIRegimeStrategy")
    parser.add_argument("--n-windows", type=int, default=8)
    parser.add_argument("--is-fraction", type=float, default=0.7)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--objective", choices=list(_LOSS_REGISTRY.keys()),
                        default="calmar")
    parser.add_argument("--anchored", action="store_true",
                        help="Use anchored (expanding) IS window instead of rolling")
    parser.add_argument("--sampler", choices=["tpe", "cmaes", "random"], default="tpe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="Save JSON report to this path")
    parser.add_argument("--capital", type=float, default=10_000.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    # Load data
    if not _PANDAS:
        print("ERROR: pandas is required. pip install pandas", file=sys.stderr)
        sys.exit(1)

    data_path = Path(args.data)
    if data_path.suffix == ".parquet":
        ohlcv = pd.read_parquet(data_path)
    else:
        ohlcv = pd.read_csv(data_path, index_col=0, parse_dates=True)
    ohlcv = ohlcv.sort_index()

    # Load strategy class
    module_path, class_name = args.strategy.rsplit(".", 1)
    module = importlib.import_module(module_path)
    strategy_cls = getattr(module, class_name)

    # Build config
    config = WFOConfig(
        n_windows=args.n_windows,
        is_fraction=args.is_fraction,
        n_trials=args.n_trials,
        objective=_LOSS_REGISTRY[args.objective],
        anchored=args.anchored,
        sampler=args.sampler,
        seed=args.seed,
        initial_capital=args.capital,
    )

    wfo = WalkForwardOptimizer(config=config)

    # Derive param_space from strategy class if it exposes WFO_PARAM_SPACE
    param_space = getattr(strategy_cls, "WFO_PARAM_SPACE", {})
    if not param_space:
        print(
            f"WARNING: {class_name} does not define WFO_PARAM_SPACE — "
            "no parameters will be optimised.  Add a class-level dict::",
            file=sys.stderr,
        )
        print("  WFO_PARAM_SPACE = {\n"
              "    'rsi_period': ('int', 7, 28),\n"
              "    'rsi_ob': ('float', 65.0, 85.0),\n"
              "  }", file=sys.stderr)

    report = wfo.run(ohlcv=ohlcv, strategy_cls=strategy_cls, param_space=param_space)
    print(report.summary())

    stability = compute_parameter_stability(report)
    if stability:
        print("\nParameter Stability:")
        for pname, stats in stability.items():
            flag = "✓ stable" if stats["stable"] else "⚠ drifting"
            print(f"  {pname:<25} mean={stats['mean']:>10.4f}  "
                  f"cv={stats['cv']:.3f}  {flag}")

    rec = recommended_params(report)
    if rec:
        print(f"\nRecommended live params: {json.dumps(rec, indent=2)}")

    if args.output:
        report.save(args.output)


if __name__ == "__main__":
    _cli()
