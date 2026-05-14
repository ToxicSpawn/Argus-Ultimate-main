"""
Performance Attribution Engine for ARGUS.

Tracks where profits and losses come from by breaking down PnL across
multiple dimensions: strategy, symbol, regime, time-of-day, and holding
period. Identifies alpha sources (high Sharpe strategies) and bleeders
(consistently losing strategies/symbols) with actionable recommendations.

Also tracks execution quality metrics (slippage, fill rate, latency) by venue.

Usage:
    engine = PerformanceEngine()
    engine.record_trade(
        trade_id="t1", strategy="momentum", symbol="BTC/USD", side="buy",
        entry_price=50000, exit_price=50500, size=0.01,
        entry_time=t0, exit_time=t1, regime="TRENDING_UP",
        slippage_bps=2.5, fees=0.35,
    )
    attribution = engine.attribute_pnl(period_hours=24)
    bleeders = engine.identify_bleeders()
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CompletedTrade:
    """A fully closed trade with entry/exit details."""
    trade_id: str
    strategy: str
    symbol: str
    side: str              # "buy" or "sell"
    entry_price: float
    exit_price: float
    size: float            # quantity
    entry_time: float      # unix timestamp
    exit_time: float       # unix timestamp
    regime: str
    slippage_bps: float
    fees: float
    pnl: float = 0.0      # computed on record

    @property
    def hold_time_seconds(self) -> float:
        return max(0.0, self.exit_time - self.entry_time)

    @property
    def hold_time_hours(self) -> float:
        return self.hold_time_seconds / 3600.0

    @property
    def entry_hour(self) -> int:
        """Hour of day (UTC) when the trade was entered."""
        import datetime
        dt = datetime.datetime.utcfromtimestamp(self.entry_time)
        return dt.hour


# ---------------------------------------------------------------------------
# PerformanceEngine
# ---------------------------------------------------------------------------

class PerformanceEngine:
    """
    Tracks and attributes PnL across multiple dimensions.

    Maintains a rolling history of completed trades and provides
    analytical methods for strategy evaluation, bleeder detection,
    and execution quality assessment.

    Parameters
    ----------
    max_history : int
        Maximum number of trades to retain (FIFO).
    bleeder_min_trades : int
        Minimum trades before a strategy/symbol can be flagged as a bleeder.
    bleeder_sharpe_threshold : float
        Sharpe ratio below which a strategy is considered a bleeder.
    """

    def __init__(
        self,
        max_history: int = 5000,
        bleeder_min_trades: int = 10,
        bleeder_sharpe_threshold: float = -0.3,
    ) -> None:
        self.max_history = max(100, int(max_history))
        self.bleeder_min_trades = max(3, int(bleeder_min_trades))
        self.bleeder_sharpe_threshold = float(bleeder_sharpe_threshold)

        self._trades: List[CompletedTrade] = []
        self._venue_stats: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

    # ── Recording ─────────────────────────────────────────────────────────

    def record_trade(
        self,
        trade_id: str,
        strategy: str,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        size: float,
        entry_time: float,
        exit_time: float,
        regime: str = "UNKNOWN",
        slippage_bps: float = 0.0,
        fees: float = 0.0,
        venue: str = "kraken",
    ) -> CompletedTrade:
        """
        Record a completed trade with full entry/exit details.

        PnL is computed as: (exit - entry) * size for long, inverted for short.
        Fees are subtracted from PnL.
        """
        entry_p = float(entry_price)
        exit_p = float(exit_price)
        sz = float(size)

        _side = str(side).lower()
        if _side in ("buy", "long"):
            raw_pnl = (exit_p - entry_p) * sz
        else:
            raw_pnl = (entry_p - exit_p) * sz

        pnl = raw_pnl - float(fees)

        trade = CompletedTrade(
            trade_id=str(trade_id),
            strategy=str(strategy),
            symbol=str(symbol),
            side=_side,
            entry_price=entry_p,
            exit_price=exit_p,
            size=sz,
            entry_time=float(entry_time),
            exit_time=float(exit_time),
            regime=str(regime),
            slippage_bps=float(slippage_bps),
            fees=float(fees),
            pnl=pnl,
        )

        self._trades.append(trade)

        # FIFO trimming
        if len(self._trades) > self.max_history:
            self._trades = self._trades[-self.max_history:]

        # Record venue execution stats
        venue_key = str(venue).lower()
        self._venue_stats[venue_key]["slippage_bps"].append(float(slippage_bps))
        if entry_p > 0:
            latency_proxy = abs(exit_p - entry_p) / entry_p * 10000
            self._venue_stats[venue_key]["latency_proxy_bps"].append(latency_proxy)

        return trade

    # ── PnL Attribution ───────────────────────────────────────────────────

    def attribute_pnl(self, period_hours: float = 24.0) -> Dict[str, Any]:
        """
        Break down PnL over the last period_hours by multiple dimensions.

        Returns dict with keys:
          - by_strategy: {strategy_name: {pnl, trades, win_rate}}
          - by_symbol: {symbol: {pnl, trades, win_rate}}
          - by_regime: {regime: {pnl, trades}}
          - by_time_of_day: {hour: {pnl, trades}}
          - by_holding_period: {"<1h": {pnl}, "1-4h": {pnl}, "4-24h": {pnl}, ">24h": {pnl}}
          - total_pnl: float
          - total_trades: int
        """
        cutoff = time.time() - period_hours * 3600
        recent = [t for t in self._trades if t.exit_time >= cutoff]

        if not recent:
            return {
                "by_strategy": {},
                "by_symbol": {},
                "by_regime": {},
                "by_time_of_day": {},
                "by_holding_period": {},
                "total_pnl": 0.0,
                "total_trades": 0,
            }

        # By strategy
        by_strategy: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"pnl": 0.0, "trades": 0, "wins": 0}
        )
        for t in recent:
            by_strategy[t.strategy]["pnl"] += t.pnl
            by_strategy[t.strategy]["trades"] += 1
            if t.pnl > 0:
                by_strategy[t.strategy]["wins"] += 1

        for v in by_strategy.values():
            v["win_rate"] = v["wins"] / max(v["trades"], 1)
            del v["wins"]

        # By symbol
        by_symbol: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"pnl": 0.0, "trades": 0, "wins": 0}
        )
        for t in recent:
            by_symbol[t.symbol]["pnl"] += t.pnl
            by_symbol[t.symbol]["trades"] += 1
            if t.pnl > 0:
                by_symbol[t.symbol]["wins"] += 1

        for v in by_symbol.values():
            v["win_rate"] = v["wins"] / max(v["trades"], 1)
            del v["wins"]

        # By regime
        by_regime: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"pnl": 0.0, "trades": 0}
        )
        for t in recent:
            by_regime[t.regime]["pnl"] += t.pnl
            by_regime[t.regime]["trades"] += 1

        # By time of day
        by_time: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {"pnl": 0.0, "trades": 0}
        )
        for t in recent:
            by_time[t.entry_hour]["pnl"] += t.pnl
            by_time[t.entry_hour]["trades"] += 1

        # By holding period
        by_hold: Dict[str, Dict[str, float]] = {
            "<1h": {"pnl": 0.0, "trades": 0},
            "1-4h": {"pnl": 0.0, "trades": 0},
            "4-24h": {"pnl": 0.0, "trades": 0},
            ">24h": {"pnl": 0.0, "trades": 0},
        }
        for t in recent:
            hours = t.hold_time_hours
            if hours < 1:
                bucket = "<1h"
            elif hours < 4:
                bucket = "1-4h"
            elif hours < 24:
                bucket = "4-24h"
            else:
                bucket = ">24h"
            by_hold[bucket]["pnl"] += t.pnl
            by_hold[bucket]["trades"] += 1

        total_pnl = sum(t.pnl for t in recent)

        return {
            "by_strategy": dict(by_strategy),
            "by_symbol": dict(by_symbol),
            "by_regime": dict(by_regime),
            "by_time_of_day": dict(by_time),
            "by_holding_period": by_hold,
            "total_pnl": total_pnl,
            "total_trades": len(recent),
        }

    # ── Alpha source identification ───────────────────────────────────────

    def identify_alpha_sources(self) -> List[Dict[str, Any]]:
        """
        Rank strategies by Sharpe ratio and return contribution percentages.

        Returns list of {strategy, sharpe, contribution_pct}, sorted by
        Sharpe ratio descending.
        """
        strategy_pnls: Dict[str, List[float]] = defaultdict(list)
        for t in self._trades:
            strategy_pnls[t.strategy].append(t.pnl)

        total_positive_pnl = sum(
            sum(pnls) for pnls in strategy_pnls.values()
            if sum(pnls) > 0
        )
        if total_positive_pnl == 0:
            total_positive_pnl = 1.0  # avoid division by zero

        results = []
        for strat, pnls in strategy_pnls.items():
            if len(pnls) < 2:
                continue
            arr = np.array(pnls, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std())
            sharpe = mean / std if std > 1e-9 else 0.0
            total = float(arr.sum())
            contribution = max(0.0, total) / total_positive_pnl * 100.0

            results.append({
                "strategy": strat,
                "sharpe": round(sharpe, 4),
                "contribution_pct": round(contribution, 2),
                "total_pnl": round(total, 4),
                "trades": len(pnls),
            })

        results.sort(key=lambda x: x["sharpe"], reverse=True)
        return results

    # ── Bleeder identification ────────────────────────────────────────────

    def identify_bleeders(self) -> List[Dict[str, Any]]:
        """
        Find consistently losing strategies or symbols.

        Returns list of {name, type, loss, recommendation} where
        recommendation is one of: 'disable', 'reduce_size', 'monitor', 'ok'.
        """
        bleeders: List[Dict[str, Any]] = []

        # Check strategies
        strategy_pnls: Dict[str, List[float]] = defaultdict(list)
        for t in self._trades:
            strategy_pnls[t.strategy].append(t.pnl)

        for strat, pnls in strategy_pnls.items():
            if len(pnls) < self.bleeder_min_trades:
                continue
            arr = np.array(pnls, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std())
            sharpe = mean / std if std > 1e-9 else 0.0
            total_loss = float(arr.sum())
            win_rate = float(np.sum(arr > 0)) / len(arr)

            rec = self._classify_bleeder(sharpe, win_rate, total_loss)
            if rec != "ok":
                bleeders.append({
                    "name": strat,
                    "type": "strategy",
                    "loss": round(total_loss, 4),
                    "sharpe": round(sharpe, 4),
                    "win_rate": round(win_rate, 4),
                    "trades": len(pnls),
                    "recommendation": rec,
                })

        # Check symbols
        symbol_pnls: Dict[str, List[float]] = defaultdict(list)
        for t in self._trades:
            symbol_pnls[t.symbol].append(t.pnl)

        for sym, pnls in symbol_pnls.items():
            if len(pnls) < self.bleeder_min_trades:
                continue
            arr = np.array(pnls, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std())
            sharpe = mean / std if std > 1e-9 else 0.0
            total_loss = float(arr.sum())
            win_rate = float(np.sum(arr > 0)) / len(arr)

            rec = self._classify_bleeder(sharpe, win_rate, total_loss)
            if rec != "ok":
                bleeders.append({
                    "name": sym,
                    "type": "symbol",
                    "loss": round(total_loss, 4),
                    "sharpe": round(sharpe, 4),
                    "win_rate": round(win_rate, 4),
                    "trades": len(pnls),
                    "recommendation": rec,
                })

        # Sort by loss ascending (biggest losers first)
        bleeders.sort(key=lambda x: x["loss"])
        return bleeders

    def _classify_bleeder(
        self,
        sharpe: float,
        win_rate: float,
        total_loss: float,
    ) -> str:
        """Classify a strategy/symbol based on performance metrics."""
        # Consistently negative with zero variance (all losses identical)
        if total_loss < 0 and win_rate == 0.0:
            return "disable"
        if sharpe < -0.5 and total_loss < 0:
            return "disable"
        if sharpe < self.bleeder_sharpe_threshold and total_loss < 0:
            return "reduce_size"
        if total_loss < 0 and win_rate < 0.30:
            return "reduce_size"
        if sharpe < 0.0 and win_rate < 0.40:
            return "monitor"
        if total_loss < 0 and win_rate < 0.40:
            return "monitor"
        return "ok"

    # ── Execution quality ─────────────────────────────────────────────────

    def get_execution_quality(self) -> Dict[str, Any]:
        """
        Return execution quality metrics.

        Returns dict with:
          - avg_slippage_bps: overall average slippage in basis points
          - fill_rate: fraction of trades with positive size (always 1.0 for now)
          - by_venue: {venue: {avg_slippage_bps, trade_count}}
        """
        if not self._trades:
            return {
                "avg_slippage_bps": 0.0,
                "fill_rate": 0.0,
                "by_venue": {},
            }

        all_slippage = [t.slippage_bps for t in self._trades]
        avg_slip = float(np.mean(all_slippage)) if all_slippage else 0.0

        by_venue: Dict[str, Dict[str, Any]] = {}
        for venue, stats in self._venue_stats.items():
            slips = stats.get("slippage_bps", [])
            by_venue[venue] = {
                "avg_slippage_bps": float(np.mean(slips)) if slips else 0.0,
                "trade_count": len(slips),
            }

        return {
            "avg_slippage_bps": round(avg_slip, 4),
            "fill_rate": 1.0,  # all recorded trades are filled by definition
            "by_venue": by_venue,
        }

    # ── Snapshot ──────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Return current state for dashboard/logging."""
        return {
            "total_trades": len(self._trades),
            "alpha_sources": self.identify_alpha_sources()[:5],
            "bleeders": self.identify_bleeders()[:5],
            "execution_quality": self.get_execution_quality(),
        }
