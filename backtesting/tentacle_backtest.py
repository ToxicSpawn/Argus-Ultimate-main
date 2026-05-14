"""
tentacle_backtest.py — Walk-forward backtest runner wired to the Tentacle MatrixEvaluator.

Bridges the WalkForwardEngine with the OctoBot-style tentacle system:
  - Builds candle arrays from price data
  - Feeds each OOS window bar-by-bar through MatrixEvaluator
  - Converts MatrixResult action -> position {-1, 0, 1}
  - Delegates P&L accounting to WalkForwardEngine
  - Produces per-tentacle attribution report

Usage
-----
    runner = TentacleBacktestRunner(
        prices=close_prices,      # np.ndarray shape (T,)
        ohlcv=ohlcv_array,        # optional np.ndarray shape (T, 6)
        train_bars=500,
        test_bars=100,
    )
    result = runner.run()
    print(result.summary())
    print(runner.attribution_report())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .walk_forward import WalkForwardEngine, WalkForwardResult, KRAKEN_MAKER_FEE
from strategies.tentacles.matrix_evaluator import (
    MatrixEvaluator, MatrixResult, AggregationMode, Action,
)
from strategies.tentacles.base_tentacle import TENTACLE_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class TentacleAttributionRow:
    tentacle_name: str
    total_signals: int
    buy_signals: int
    sell_signals: int
    neutral_signals: int
    mean_signal: float
    mean_confidence: float


@dataclass
class TentacleBacktestResult:
    wf_result: WalkForwardResult
    attribution: List[TentacleAttributionRow] = field(default_factory=list)
    matrix_log: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def summary(self) -> Dict[str, Any]:
        s = self.wf_result.summary()
        s["tentacle_count"] = len(self.attribution)
        return s

    def attribution_report(self) -> List[Dict[str, Any]]:
        return [
            {
                "tentacle"        : r.tentacle_name,
                "total_signals"   : r.total_signals,
                "buy_pct"         : round(r.buy_signals  / max(r.total_signals, 1) * 100, 1),
                "sell_pct"        : round(r.sell_signals / max(r.total_signals, 1) * 100, 1),
                "neutral_pct"     : round(r.neutral_signals / max(r.total_signals, 1) * 100, 1),
                "mean_signal"     : round(r.mean_signal, 4),
                "mean_confidence" : round(r.mean_confidence, 4),
            }
            for r in sorted(self.attribution, key=lambda x: abs(x.mean_signal), reverse=True)
        ]


class TentacleBacktestRunner:
    """
    Runs a walk-forward backtest using the MatrixEvaluator as the strategy.

    Parameters
    ----------
    prices          : 1-D close price array
    ohlcv           : optional (T, 6) OHLCV array [ts, open, high, low, close, vol]
                      If None, a synthetic OHLCV is constructed from prices.
    train_bars      : in-sample bars per fold
    test_bars       : OOS bars per fold
    obs_window      : candle history passed to each tentacle evaluation
    fee_rate        : Kraken maker fee (default 0.16%)
    anchored        : anchored (True) or rolling (False) WFA
    matrix_kwargs   : forwarded to MatrixEvaluator constructor
    log_matrix      : if True, store every MatrixResult in matrix_log (memory-heavy)
    """

    def __init__(
        self,
        prices: np.ndarray,
        ohlcv: Optional[np.ndarray] = None,
        train_bars: int = 500,
        test_bars: int = 100,
        obs_window: int = 50,
        fee_rate: float = KRAKEN_MAKER_FEE,
        anchored: bool = True,
        matrix_kwargs: Optional[Dict[str, Any]] = None,
        log_matrix: bool = False,
    ) -> None:
        self._prices     = np.asarray(prices, dtype=np.float64)
        self._ohlcv      = ohlcv
        self._train_bars = train_bars
        self._test_bars  = test_bars
        self._obs_window = obs_window
        self._fee_rate   = fee_rate
        self._anchored   = anchored
        self._log_matrix = log_matrix
        self._matrix_kwargs = matrix_kwargs or {}

        # Build OHLCV if not provided
        if self._ohlcv is None:
            T = len(self._prices)
            ts = np.arange(T, dtype=np.float64)
            self._ohlcv = np.column_stack([
                ts,
                self._prices,   # open = close (simplified)
                self._prices,   # high
                self._prices,   # low
                self._prices,   # close
                np.ones(T),     # volume placeholder
            ])

        # Attribution accumulators keyed by tentacle name
        self._attr_acc: Dict[str, Dict[str, Any]] = {}
        self._matrix_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> TentacleBacktestResult:
        t0 = time.time()

        engine = WalkForwardEngine(
            prices=self._prices,
            strategy_fn=self._make_strategy(),
            train_bars=self._train_bars,
            test_bars=self._test_bars,
            fee_rate=self._fee_rate,
            anchored=self._anchored,
        )
        wf_result = engine.run()

        attribution = self._build_attribution()
        elapsed = time.time() - t0
        logger.info(
            "TentacleBacktest complete: %d folds in %.2fs | mean_sharpe=%.3f",
            len(wf_result.folds),
            elapsed,
            wf_result.summary().get("mean_sharpe", 0.0),
        )

        return TentacleBacktestResult(
            wf_result=wf_result,
            attribution=attribution,
            matrix_log=self._matrix_log,
            elapsed_sec=elapsed,
        )

    def attribution_report(self) -> List[Dict[str, Any]]:
        """Shortcut: run() must be called first."""
        return self._build_attribution_report()

    # ------------------------------------------------------------------
    # Strategy factory (closure over self)
    # ------------------------------------------------------------------

    def _make_strategy(self):
        """
        Returns a strategy_fn compatible with WalkForwardEngine:
            fn(test_prices, train_prices) -> positions array

        Creates a fresh MatrixEvaluator per fold to avoid state leakage.
        """
        ohlcv     = self._ohlcv
        obs_win   = self._obs_window
        mk_kwargs = self._matrix_kwargs
        attr_acc  = self._attr_acc
        log_matrix= self._log_matrix
        matrix_log= self._matrix_log

        def strategy_fn(test_prices: np.ndarray, train_prices: np.ndarray) -> np.ndarray:
            # Fresh matrix per fold
            matrix = MatrixEvaluator(**mk_kwargs)
            n      = len(test_prices)
            pos    = np.zeros(n)

            # Find offset of test_prices in global price array
            # (best-effort via value matching on first element)
            global_prices = np.concatenate([train_prices, test_prices])
            global_len    = len(global_prices)

            # Build a synthetic local OHLCV for this fold
            fold_ohlcv = np.column_stack([
                np.arange(global_len, dtype=np.float64),
                global_prices,
                global_prices,
                global_prices,
                global_prices,
                np.ones(global_len),
            ])

            for i in range(obs_win, n):
                global_i  = len(train_prices) + i
                start     = max(0, global_i - obs_win)
                candle_win = fold_ohlcv[start:global_i + 1]

                m_result: MatrixResult = matrix.evaluate(candle_win)

                if m_result.action == Action.BUY:
                    pos[i] = 1.0
                elif m_result.action == Action.SELL:
                    pos[i] = -1.0

                # Accumulate attribution
                for eval_r in m_result.breakdown:
                    name = eval_r.tentacle_name
                    if name not in attr_acc:
                        attr_acc[name] = {
                            "signals": [], "confidences": [],
                            "buys": 0, "sells": 0, "neutrals": 0,
                        }
                    attr_acc[name]["signals"].append(eval_r.signal)
                    attr_acc[name]["confidences"].append(eval_r.confidence)
                    if eval_r.signal > 0.05:
                        attr_acc[name]["buys"] += 1
                    elif eval_r.signal < -0.05:
                        attr_acc[name]["sells"] += 1
                    else:
                        attr_acc[name]["neutrals"] += 1

                if log_matrix:
                    matrix_log.append({
                        "bar": i,
                        "signal": m_result.signal,
                        "conviction": m_result.conviction,
                        "action": m_result.action,
                    })

            return pos

        return strategy_fn

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------

    def _build_attribution(self) -> List[TentacleAttributionRow]:
        rows = []
        for name, acc in self._attr_acc.items():
            sigs = acc["signals"]
            confs = acc["confidences"]
            rows.append(TentacleAttributionRow(
                tentacle_name    = name,
                total_signals    = len(sigs),
                buy_signals      = acc["buys"],
                sell_signals     = acc["sells"],
                neutral_signals  = acc["neutrals"],
                mean_signal      = float(np.mean(sigs)) if sigs else 0.0,
                mean_confidence  = float(np.mean(confs)) if confs else 0.0,
            ))
        return rows

    def _build_attribution_report(self) -> List[Dict[str, Any]]:
        attr = self._build_attribution()
        return [
            {
                "tentacle"        : r.tentacle_name,
                "total_signals"   : r.total_signals,
                "buy_pct"         : round(r.buy_signals  / max(r.total_signals, 1) * 100, 1),
                "sell_pct"        : round(r.sell_signals / max(r.total_signals, 1) * 100, 1),
                "neutral_pct"     : round(r.neutral_signals / max(r.total_signals, 1) * 100, 1),
                "mean_signal"     : round(r.mean_signal, 4),
                "mean_confidence" : round(r.mean_confidence, 4),
            }
            for r in sorted(attr, key=lambda x: abs(x.mean_signal), reverse=True)
        ]
