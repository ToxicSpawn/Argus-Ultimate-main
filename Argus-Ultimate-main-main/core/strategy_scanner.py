"""
Continuous Strategy Scanner — scans all symbols for the best strategy/parameter
combinations using rolling historical data. Updates strategy allocations every
N cycles based on what's actually working NOW.

This is the core of ARGUS's adaptive edge: instead of fixed strategies,
it continuously re-evaluates which strategy + parameters + symbols produce
positive expectancy, and rotates capital toward them.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    """Result of backtesting one strategy variant on one symbol."""
    symbol: str
    strategy: str
    params: Dict[str, float]
    trades: int
    win_rate: float
    total_return_pct: float
    sharpe: float
    avg_win_pct: float
    avg_loss_pct: float
    max_drawdown_pct: float


@dataclass
class ScanResult:
    """Complete scan result across all symbols."""
    timestamp: float
    best_per_symbol: Dict[str, StrategyResult]
    top_opportunities: List[StrategyResult]
    scan_duration_ms: float


class StrategyScanner:
    """
    Scans historical data for the best strategy + parameter combinations.

    Runs backtests across multiple strategy types and parameter grids
    for each symbol, ranks by risk-adjusted return, and recommends
    the top opportunities.

    Designed to be called periodically (every 50-100 cycles) to adapt
    to changing market conditions.
    """

    def __init__(
        self,
        fee_pct: float = 0.26,
        min_trades: int = 5,
        min_win_rate: float = 0.40,
        min_sharpe: float = 0.0,
        lookback_hours: int = 720,  # 30 days
    ):
        self._fee = fee_pct
        self._min_trades = min_trades
        self._min_wr = min_win_rate
        self._min_sharpe = min_sharpe
        self._lookback = lookback_hours
        self._last_scan: Optional[ScanResult] = None
        self._scan_count = 0

    def scan(self, market_data: Dict[str, Dict[str, np.ndarray]]) -> ScanResult:
        """
        Scan all symbols for best strategies.

        Args:
            market_data: {symbol: {"close": array, "high": array, "low": array, "volume": array}}

        Returns:
            ScanResult with best strategy per symbol and top opportunities.
        """
        t0 = time.time()
        best_per_symbol: Dict[str, StrategyResult] = {}

        for symbol, data in market_data.items():
            close = data.get("close")
            high = data.get("high")
            low = data.get("low")
            volume = data.get("volume")

            if close is None or len(close) < 50:
                continue

            best = self._find_best_strategy(symbol, close, high, low, volume)
            if best is not None:
                best_per_symbol[symbol] = best

        # Rank all opportunities by Sharpe, filter by minimum criteria
        all_results = list(best_per_symbol.values())
        top = sorted(
            [r for r in all_results if r.total_return_pct > 0 and r.sharpe >= self._min_sharpe],
            key=lambda r: r.sharpe,
            reverse=True,
        )

        scan_ms = (time.time() - t0) * 1000
        self._scan_count += 1

        result = ScanResult(
            timestamp=time.time(),
            best_per_symbol=best_per_symbol,
            top_opportunities=top[:10],
            scan_duration_ms=scan_ms,
        )
        self._last_scan = result

        logger.info(
            "StrategyScanner: scanned %d symbols in %.0fms — %d profitable opportunities",
            len(market_data), scan_ms, len(top),
        )

        return result

    def _find_best_strategy(
        self, symbol: str, close: np.ndarray, high: np.ndarray,
        low: np.ndarray, volume: np.ndarray,
    ) -> Optional[StrategyResult]:
        """Find the best strategy + params for a single symbol."""
        best: Optional[StrategyResult] = None
        best_sharpe = -999.0

        # Strategy 1: Breakout
        for lookback in [10, 20, 30, 48]:
            for tp in [1.5, 2.0, 3.0]:
                for sl in [1.0, 1.5, 2.0]:
                    result = self._backtest_breakout(symbol, close, high, low, lookback, tp, sl)
                    if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                        best = result
                        best_sharpe = result.sharpe

        # Strategy 2: Volume spike reversal
        for vol_mult in [1.5, 2.0, 2.5, 3.0]:
            for tp in [0.8, 1.0, 1.5, 2.0]:
                for sl in [1.5, 2.0, 3.0]:
                    result = self._backtest_vol_spike(symbol, close, volume, vol_mult, tp, sl)
                    if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                        best = result
                        best_sharpe = result.sharpe

        # Strategy 3: Mean reversion (BB)
        for bb_std in [1.5, 2.0, 2.5]:
            for sl in [1.0, 1.5, 2.0]:
                result = self._backtest_mean_reversion(symbol, close, bb_std, sl)
                if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                    best = result
                    best_sharpe = result.sharpe

        # Strategy 4: Momentum (SMA crossover with trailing stop)
        for fast in [10, 20]:
            for trail in [1.5, 2.0, 3.0]:
                result = self._backtest_momentum(symbol, close, fast, trail)
                if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                    best = result
                    best_sharpe = result.sharpe

        # Strategy 5: RSI mean reversion
        for rsi_buy in [20, 25, 30]:
            for rsi_sell in [50, 60, 70]:
                for sl in [1.0, 1.5, 2.0]:
                    result = self._backtest_rsi_mr(symbol, close, rsi_buy, rsi_sell, sl)
                    if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                        best = result
                        best_sharpe = result.sharpe

        # Strategy 6: MACD crossover
        for fast in [8, 12]:
            for slow in [21, 26]:
                for sl in [1.0, 1.5, 2.0]:
                    result = self._backtest_macd(symbol, close, fast, slow, sl)
                    if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                        best = result
                        best_sharpe = result.sharpe

        # Strategy 7: Range fade (sell at resistance, buy at support)
        for lookback in [24, 48, 72]:
            for tp in [1.0, 1.5, 2.0]:
                for sl in [1.0, 1.5]:
                    result = self._backtest_range_fade(symbol, close, high, low, lookback, tp, sl)
                    if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                        best = result
                        best_sharpe = result.sharpe

        # Strategy 8: EMA ribbon (fast EMA crosses slow EMA)
        for fast in [5, 8, 13]:
            for slow in [21, 34, 55]:
                for trail in [1.5, 2.0, 3.0]:
                    if fast >= slow:
                        continue
                    result = self._backtest_ema_ribbon(symbol, close, fast, slow, trail)
                    if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                        best = result
                        best_sharpe = result.sharpe

        # Strategy 9: Keltner channel breakout
        for atr_mult in [1.5, 2.0, 2.5]:
            for tp in [1.5, 2.0, 3.0]:
                for sl in [1.0, 1.5]:
                    result = self._backtest_keltner(symbol, close, high, low, atr_mult, tp, sl)
                    if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                        best = result
                        best_sharpe = result.sharpe

        # Strategy 10: Heikin Ashi trend
        for consec in [3, 4, 5]:
            for trail in [1.5, 2.0, 3.0]:
                result = self._backtest_heikin_ashi(symbol, close, high, low, consec, trail)
                if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                    best = result
                    best_sharpe = result.sharpe

        # Strategy 11: Pivot point bounce
        for tp in [1.0, 1.5, 2.0]:
            for sl in [0.5, 1.0, 1.5]:
                result = self._backtest_pivot_bounce(symbol, close, high, low, tp, sl)
                if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                    best = result
                    best_sharpe = result.sharpe

        # Strategy 12: Inside bar breakout
        for tp in [1.5, 2.0, 3.0]:
            for sl in [0.5, 1.0, 1.5]:
                result = self._backtest_inside_bar(symbol, close, high, low, tp, sl)
                if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                    best = result
                    best_sharpe = result.sharpe

        # Strategy 13: Williams %R reversal
        for period in [14, 21]:
            for buy_thresh in [-80, -85, -90]:
                for sl in [1.0, 1.5, 2.0]:
                    result = self._backtest_williams_r(symbol, close, high, low, period, buy_thresh, sl)
                    if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                        best = result
                        best_sharpe = result.sharpe

        # Strategy 14: Stochastic oversold bounce
        for k_period in [14, 21]:
            for buy_level in [15, 20, 25]:
                for sl in [1.0, 1.5, 2.0]:
                    result = self._backtest_stochastic(symbol, close, high, low, k_period, buy_level, sl)
                    if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                        best = result
                        best_sharpe = result.sharpe

        # Strategy 15: ADX trend strength filter + momentum
        for adx_thresh in [20, 25, 30]:
            for trail in [1.5, 2.0, 3.0]:
                result = self._backtest_adx_momentum(symbol, close, high, low, adx_thresh, trail)
                if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                    best = result
                    best_sharpe = result.sharpe

        # Strategy 16: Donchian channel (turtle trading)
        for entry_period in [20, 30, 55]:
            for exit_period in [10, 15, 20]:
                result = self._backtest_donchian(symbol, close, high, low, entry_period, exit_period)
                if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                    best = result
                    best_sharpe = result.sharpe

        # Strategy 17: Gap continuation (large candle follow-through)
        for gap_pct in [1.0, 1.5, 2.0]:
            for tp in [1.0, 1.5, 2.0]:
                for sl in [0.5, 1.0]:
                    result = self._backtest_gap_continuation(symbol, close, gap_pct, tp, sl)
                    if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                        best = result
                        best_sharpe = result.sharpe

        # Strategy 18: Three bar play (reversal pattern)
        for tp in [1.0, 1.5, 2.0]:
            for sl in [0.5, 1.0, 1.5]:
                result = self._backtest_three_bar_play(symbol, close, high, low, tp, sl)
                if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                    best = result
                    best_sharpe = result.sharpe

        # Strategy 19: Dual thrust (volatility breakout, popular in Asia)
        for k in [0.4, 0.5, 0.6, 0.7]:
            for sl in [1.0, 1.5, 2.0]:
                result = self._backtest_dual_thrust(symbol, close, high, low, k, sl)
                if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                    best = result
                    best_sharpe = result.sharpe

        # Strategy 20: Opening range breakout (first 4 hours)
        for tp in [1.5, 2.0, 3.0]:
            for sl in [0.5, 1.0, 1.5]:
                result = self._backtest_orb(symbol, close, high, low, tp, sl)
                if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                    best = result
                    best_sharpe = result.sharpe

        # Strategy 21: Quantum entropy regime (high entropy = mean revert, low = trend)
        for entropy_thresh in [0.5, 0.6, 0.7]:
            for tp in [1.5, 2.0, 3.0]:
                for sl in [1.0, 1.5, 2.0]:
                    result = self._backtest_quantum_entropy(symbol, close, entropy_thresh, tp, sl)
                    if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                        best = result
                        best_sharpe = result.sharpe

        # Strategy 22: Quantum tunneling breakout (probability of breaching barrier)
        for barrier_std in [1.5, 2.0, 2.5]:
            for decay in [0.9, 0.95, 0.99]:
                for sl in [1.0, 1.5, 2.0]:
                    result = self._backtest_quantum_tunneling(symbol, close, barrier_std, decay, sl)
                    if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                        best = result
                        best_sharpe = result.sharpe

        # Strategy 23: Quantum superposition (simultaneous long/short bias weighted by probability)
        for lookback in [20, 30, 50]:
            for tp in [1.5, 2.0, 3.0]:
                for sl in [1.0, 1.5]:
                    result = self._backtest_quantum_superposition(symbol, close, lookback, tp, sl)
                    if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                        best = result
                        best_sharpe = result.sharpe

        # Strategy 24: Quantum walk momentum (random walk deviation detection)
        for walk_len in [20, 50, 100]:
            for sigma_thresh in [1.5, 2.0, 2.5]:
                for trail in [1.5, 2.0, 3.0]:
                    result = self._backtest_quantum_walk(symbol, close, walk_len, sigma_thresh, trail)
                    if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                        best = result
                        best_sharpe = result.sharpe

        # Strategy 25: Quantum entanglement (correlated pair divergence)
        # Uses close array as single-asset proxy — measures autocorrelation regime
        for lag in [1, 2, 4]:
            for tp in [1.0, 1.5, 2.0]:
                for sl in [1.0, 1.5]:
                    result = self._backtest_quantum_entanglement(symbol, close, lag, tp, sl)
                    if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                        best = result
                        best_sharpe = result.sharpe

        # Strategy 26: Quantum annealing optimizer (simulated annealing for entry timing)
        for temp_decay in [0.95, 0.97, 0.99]:
            for tp in [1.5, 2.0, 3.0]:
                for sl in [1.0, 1.5]:
                    result = self._backtest_quantum_annealing(symbol, close, high, low, temp_decay, tp, sl)
                    if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                        best = result
                        best_sharpe = result.sharpe

        # Strategy 27: Quantum decoherence (regime stability measurement)
        for window in [20, 30, 50]:
            for stability_thresh in [0.3, 0.4, 0.5]:
                for trail in [1.5, 2.0, 3.0]:
                    result = self._backtest_quantum_decoherence(symbol, close, window, stability_thresh, trail)
                    if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                        best = result
                        best_sharpe = result.sharpe

        # Strategy 28: Quantum interference (constructive/destructive signal overlap)
        for fast in [5, 8]:
            for mid in [13, 21]:
                for slow in [34, 55]:
                    for sl in [1.0, 1.5, 2.0]:
                        if fast >= mid or mid >= slow:
                            continue
                        result = self._backtest_quantum_interference(symbol, close, fast, mid, slow, sl)
                        if result and result.sharpe > best_sharpe and result.trades >= self._min_trades:
                            best = result
                            best_sharpe = result.sharpe

        return best

    def _compute_result(self, symbol: str, strategy: str, params: Dict,
                        trades: List[float]) -> Optional[StrategyResult]:
        """Compute metrics from a list of trade P&L percentages."""
        if len(trades) < self._min_trades:
            return None

        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        wr = len(wins) / len(trades)

        if wr < self._min_wr:
            return None

        total = sum(trades)
        mean = np.mean(trades)
        std = max(np.std(trades), 1e-9)
        sharpe = mean / std

        # Max drawdown
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in trades:
            equity += t
            peak = max(peak, equity)
            dd = (peak - equity) / max(peak, 1e-9) if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        return StrategyResult(
            symbol=symbol, strategy=strategy, params=params,
            trades=len(trades), win_rate=wr, total_return_pct=total,
            sharpe=sharpe,
            avg_win_pct=np.mean(wins) if wins else 0.0,
            avg_loss_pct=np.mean(losses) if losses else 0.0,
            max_drawdown_pct=max_dd,
        )

    def _backtest_breakout(self, symbol, close, high, low, lookback, tp, sl):
        trades = []
        pos = 0
        entry = 0
        for i in range(lookback + 1, len(close)):
            h = max(high[i - lookback:i])
            if pos == 0 and close[i] > h:
                pos = 1
                entry = close[i]
            elif pos == 1:
                gain = (close[i] / entry - 1) * 100
                if gain >= tp:
                    trades.append(gain - 2 * self._fee)
                    pos = 0
                elif gain <= -sl:
                    trades.append(gain - 2 * self._fee)
                    pos = 0
        return self._compute_result(
            symbol, "breakout",
            {"lookback": lookback, "tp_pct": tp, "sl_pct": sl},
            trades,
        )

    def _backtest_vol_spike(self, symbol, close, volume, vol_mult, tp, sl):
        import pandas as pd
        vol_sma = pd.Series(volume).rolling(24).mean().values
        trades = []
        pos = 0
        entry = 0
        for i in range(25, len(close)):
            if pos == 0 and volume[i] > vol_sma[i] * vol_mult and close[i] < close[i - 1]:
                pos = 1
                entry = close[i]
            elif pos == 1:
                gain = (close[i] / entry - 1) * 100
                if gain >= tp:
                    trades.append(gain - 2 * self._fee)
                    pos = 0
                elif gain <= -sl:
                    trades.append(gain - 2 * self._fee)
                    pos = 0
        return self._compute_result(
            symbol, "vol_spike_reversal",
            {"vol_mult": vol_mult, "tp_pct": tp, "sl_pct": sl},
            trades,
        )

    def _backtest_mean_reversion(self, symbol, close, bb_std, sl):
        import pandas as pd
        sma = pd.Series(close).rolling(20).mean().values
        std = pd.Series(close).rolling(20).std().values
        lower = sma - bb_std * std
        trades = []
        pos = 0
        entry = 0
        for i in range(21, len(close)):
            if pos == 0 and close[i] <= lower[i]:
                pos = 1
                entry = close[i]
            elif pos == 1:
                gain = (close[i] / entry - 1) * 100
                if close[i] >= sma[i]:
                    trades.append(gain - 2 * self._fee)
                    pos = 0
                elif gain <= -sl:
                    trades.append(gain - 2 * self._fee)
                    pos = 0
        return self._compute_result(
            symbol, "mean_reversion",
            {"bb_std": bb_std, "sl_pct": sl},
            trades,
        )

    def _backtest_momentum(self, symbol, close, fast_period, trail_pct):
        import pandas as pd
        sma = pd.Series(close).rolling(fast_period).mean().values
        trades = []
        pos = 0
        entry = 0
        peak = 0
        for i in range(fast_period + 1, len(close)):
            if pos == 0 and close[i] > sma[i] and close[i - 1] <= sma[i - 1]:
                pos = 1
                entry = close[i]
                peak = close[i]
            elif pos == 1:
                peak = max(peak, close[i])
                trail_stop = peak * (1 - trail_pct / 100)
                if close[i] < trail_stop:
                    gain = (close[i] / entry - 1) * 100
                    trades.append(gain - 2 * self._fee)
                    pos = 0
        return self._compute_result(
            symbol, "momentum",
            {"fast_period": fast_period, "trail_pct": trail_pct},
            trades,
        )

    def _backtest_rsi_mr(self, symbol, close, rsi_buy, rsi_sell, sl):
        import pandas as pd
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss_s = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss_s.replace(0, np.nan)
        rsi = (100 - (100 / (1 + rs))).values
        trades = []; pos = 0; entry = 0
        for i in range(20, len(close)):
            if np.isnan(rsi[i]): continue
            if pos == 0 and rsi[i] < rsi_buy:
                pos = 1; entry = close[i]
            elif pos == 1:
                gain_pct = (close[i]/entry-1)*100
                if rsi[i] > rsi_sell: trades.append(gain_pct - 2*self._fee); pos = 0
                elif gain_pct <= -sl: trades.append(gain_pct - 2*self._fee); pos = 0
        return self._compute_result(symbol, "rsi_mean_reversion", {"rsi_buy": rsi_buy, "rsi_sell": rsi_sell, "sl_pct": sl}, trades)

    def _backtest_macd(self, symbol, close, fast, slow, sl):
        import pandas as pd
        ema_f = pd.Series(close).ewm(span=fast).mean().values
        ema_s = pd.Series(close).ewm(span=slow).mean().values
        macd = ema_f - ema_s
        signal = pd.Series(macd).ewm(span=9).mean().values
        trades = []; pos = 0; entry = 0
        for i in range(slow+10, len(close)):
            if pos == 0 and macd[i] > signal[i] and macd[i-1] <= signal[i-1]:
                pos = 1; entry = close[i]
            elif pos == 1:
                gain_pct = (close[i]/entry-1)*100
                if macd[i] < signal[i] and macd[i-1] >= signal[i-1]: trades.append(gain_pct - 2*self._fee); pos = 0
                elif gain_pct <= -sl: trades.append(gain_pct - 2*self._fee); pos = 0
        return self._compute_result(symbol, "macd_crossover", {"fast": fast, "slow": slow, "sl_pct": sl}, trades)

    def _backtest_range_fade(self, symbol, close, high, low, lookback, tp, sl):
        trades = []; pos = 0; entry = 0
        for i in range(lookback+1, len(close)):
            h_range = max(high[i-lookback:i]); l_range = min(low[i-lookback:i])
            mid = (h_range + l_range) / 2
            if pos == 0 and close[i] <= l_range * 1.005:
                pos = 1; entry = close[i]
            elif pos == 1:
                gain_pct = (close[i]/entry-1)*100
                if gain_pct >= tp: trades.append(gain_pct - 2*self._fee); pos = 0
                elif gain_pct <= -sl: trades.append(gain_pct - 2*self._fee); pos = 0
        return self._compute_result(symbol, "range_fade", {"lookback": lookback, "tp_pct": tp, "sl_pct": sl}, trades)

    def _backtest_ema_ribbon(self, symbol, close, fast, slow, trail):
        import pandas as pd
        ema_f = pd.Series(close).ewm(span=fast).mean().values
        ema_s = pd.Series(close).ewm(span=slow).mean().values
        trades = []; pos = 0; entry = 0; peak = 0
        for i in range(slow+1, len(close)):
            if pos == 0 and ema_f[i] > ema_s[i] and ema_f[i-1] <= ema_s[i-1]:
                pos = 1; entry = close[i]; peak = close[i]
            elif pos == 1:
                peak = max(peak, close[i])
                if close[i] < peak * (1 - trail/100): trades.append((close[i]/entry-1)*100 - 2*self._fee); pos = 0
        return self._compute_result(symbol, "ema_ribbon", {"fast": fast, "slow": slow, "trail_pct": trail}, trades)

    def _backtest_keltner(self, symbol, close, high, low, atr_mult, tp, sl):
        import pandas as pd
        sma = pd.Series(close).rolling(20).mean().values
        atr = pd.Series(high - low).rolling(20).mean().values
        upper = sma + atr_mult * atr
        trades = []; pos = 0; entry = 0
        for i in range(21, len(close)):
            if pos == 0 and close[i] > upper[i] and close[i-1] <= upper[i-1]:
                pos = 1; entry = close[i]
            elif pos == 1:
                gain_pct = (close[i]/entry-1)*100
                if gain_pct >= tp: trades.append(gain_pct - 2*self._fee); pos = 0
                elif gain_pct <= -sl: trades.append(gain_pct - 2*self._fee); pos = 0
        return self._compute_result(symbol, "keltner_breakout", {"atr_mult": atr_mult, "tp_pct": tp, "sl_pct": sl}, trades)

    def _backtest_heikin_ashi(self, symbol, close, high, low, consec_green, trail):
        ha_close = np.zeros(len(close)); ha_open = np.zeros(len(close))
        ha_close[0] = close[0]; ha_open[0] = close[0]
        for i in range(1, len(close)):
            ha_close[i] = (close[i] + high[i] + low[i] + close[i]) / 4
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
        trades = []; pos = 0; entry = 0; peak = 0
        for i in range(consec_green+1, len(close)):
            greens = all(ha_close[i-j] > ha_open[i-j] for j in range(consec_green))
            if pos == 0 and greens:
                pos = 1; entry = close[i]; peak = close[i]
            elif pos == 1:
                peak = max(peak, close[i])
                if close[i] < peak * (1 - trail/100): trades.append((close[i]/entry-1)*100 - 2*self._fee); pos = 0
        return self._compute_result(symbol, "heikin_ashi_trend", {"consec_green": consec_green, "trail_pct": trail}, trades)

    def _backtest_pivot_bounce(self, symbol, close, high, low, tp, sl):
        trades = []; pos = 0; entry = 0
        for i in range(25, len(close)):
            h24 = max(high[i-24:i]); l24 = min(low[i-24:i]); c24 = close[i-1]
            pivot = (h24 + l24 + c24) / 3; s1 = 2*pivot - h24
            if pos == 0 and close[i] <= s1 * 1.002 and close[i] >= s1 * 0.998:
                pos = 1; entry = close[i]
            elif pos == 1:
                gain_pct = (close[i]/entry-1)*100
                if gain_pct >= tp: trades.append(gain_pct - 2*self._fee); pos = 0
                elif gain_pct <= -sl: trades.append(gain_pct - 2*self._fee); pos = 0
        return self._compute_result(symbol, "pivot_bounce", {"tp_pct": tp, "sl_pct": sl}, trades)

    def _backtest_inside_bar(self, symbol, close, high, low, tp, sl):
        trades = []; pos = 0; entry = 0
        for i in range(2, len(close)):
            inside = high[i-1] < high[i-2] and low[i-1] > low[i-2]
            if pos == 0 and inside and close[i] > high[i-1]:
                pos = 1; entry = close[i]
            elif pos == 1:
                gain_pct = (close[i]/entry-1)*100
                if gain_pct >= tp: trades.append(gain_pct - 2*self._fee); pos = 0
                elif gain_pct <= -sl: trades.append(gain_pct - 2*self._fee); pos = 0
        return self._compute_result(symbol, "inside_bar_breakout", {"tp_pct": tp, "sl_pct": sl}, trades)

    def _backtest_williams_r(self, symbol, close, high, low, period, buy_thresh, sl):
        trades = []; pos = 0; entry = 0
        for i in range(period+1, len(close)):
            hh = max(high[i-period:i+1]); ll = min(low[i-period:i+1])
            wr = -100 * (hh - close[i]) / max(hh - ll, 1e-9)
            if pos == 0 and wr < buy_thresh:
                pos = 1; entry = close[i]
            elif pos == 1:
                gain_pct = (close[i]/entry-1)*100
                wr_now = -100 * (max(high[i-period:i+1]) - close[i]) / max(max(high[i-period:i+1]) - min(low[i-period:i+1]), 1e-9)
                if wr_now > -20: trades.append(gain_pct - 2*self._fee); pos = 0
                elif gain_pct <= -sl: trades.append(gain_pct - 2*self._fee); pos = 0
        return self._compute_result(symbol, "williams_r_reversal", {"period": period, "buy_thresh": buy_thresh, "sl_pct": sl}, trades)

    def _backtest_stochastic(self, symbol, close, high, low, k_period, buy_level, sl):
        trades = []; pos = 0; entry = 0
        for i in range(k_period+1, len(close)):
            hh = max(high[i-k_period:i+1]); ll = min(low[i-k_period:i+1])
            k_val = 100 * (close[i] - ll) / max(hh - ll, 1e-9)
            if pos == 0 and k_val < buy_level:
                pos = 1; entry = close[i]
            elif pos == 1:
                gain_pct = (close[i]/entry-1)*100
                k_now = 100 * (close[i] - min(low[i-k_period:i+1])) / max(max(high[i-k_period:i+1]) - min(low[i-k_period:i+1]), 1e-9)
                if k_now > 80: trades.append(gain_pct - 2*self._fee); pos = 0
                elif gain_pct <= -sl: trades.append(gain_pct - 2*self._fee); pos = 0
        return self._compute_result(symbol, "stochastic_oversold", {"k_period": k_period, "buy_level": buy_level, "sl_pct": sl}, trades)

    def _backtest_adx_momentum(self, symbol, close, high, low, adx_thresh, trail):
        import pandas as pd
        # Simplified ADX
        tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
        atr = pd.Series(tr).rolling(14).mean().values
        plus_dm = np.maximum(high[1:] - high[:-1], 0); minus_dm = np.maximum(low[:-1] - low[1:], 0)
        plus_di = pd.Series(plus_dm / np.maximum(atr, 1e-9) * 100).rolling(14).mean().values
        minus_di = pd.Series(minus_dm / np.maximum(atr, 1e-9) * 100).rolling(14).mean().values
        dx = abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-9) * 100
        adx = pd.Series(dx).rolling(14).mean().values
        sma20 = pd.Series(close[1:]).rolling(20).mean().values
        trades = []; pos = 0; entry = 0; peak = 0
        for i in range(35, len(close)-1):
            idx = i - 1
            if idx >= len(adx) or np.isnan(adx[idx]): continue
            if pos == 0 and adx[idx] > adx_thresh and plus_di[idx] > minus_di[idx] and close[i] > sma20[idx]:
                pos = 1; entry = close[i]; peak = close[i]
            elif pos == 1:
                peak = max(peak, close[i])
                if close[i] < peak * (1 - trail/100): trades.append((close[i]/entry-1)*100 - 2*self._fee); pos = 0
        return self._compute_result(symbol, "adx_momentum", {"adx_thresh": adx_thresh, "trail_pct": trail}, trades)

    def _backtest_donchian(self, symbol, close, high, low, entry_period, exit_period):
        trades = []; pos = 0; entry = 0
        for i in range(max(entry_period, exit_period)+1, len(close)):
            h_entry = max(high[i-entry_period:i]); l_exit = min(low[i-exit_period:i])
            if pos == 0 and close[i] > h_entry:
                pos = 1; entry = close[i]
            elif pos == 1 and close[i] < l_exit:
                trades.append((close[i]/entry-1)*100 - 2*self._fee); pos = 0
        return self._compute_result(symbol, "donchian_turtle", {"entry_period": entry_period, "exit_period": exit_period}, trades)

    def _backtest_gap_continuation(self, symbol, close, gap_pct, tp, sl):
        trades = []; pos = 0; entry = 0
        for i in range(1, len(close)):
            move = (close[i]/close[i-1]-1)*100
            if pos == 0 and move > gap_pct:
                pos = 1; entry = close[i]
            elif pos == 1:
                gain_pct = (close[i]/entry-1)*100
                if gain_pct >= tp: trades.append(gain_pct - 2*self._fee); pos = 0
                elif gain_pct <= -sl: trades.append(gain_pct - 2*self._fee); pos = 0
        return self._compute_result(symbol, "gap_continuation", {"gap_pct": gap_pct, "tp_pct": tp, "sl_pct": sl}, trades)

    def _backtest_three_bar_play(self, symbol, close, high, low, tp, sl):
        trades = []; pos = 0; entry = 0
        for i in range(3, len(close)):
            bar1_bear = close[i-2] < close[i-3]  # big down bar
            bar2_inside = high[i-1] < high[i-2] and low[i-1] > low[i-2]  # inside bar
            bar3_bull = close[i] > high[i-1]  # breakout up
            if pos == 0 and bar1_bear and bar2_inside and bar3_bull:
                pos = 1; entry = close[i]
            elif pos == 1:
                gain_pct = (close[i]/entry-1)*100
                if gain_pct >= tp: trades.append(gain_pct - 2*self._fee); pos = 0
                elif gain_pct <= -sl: trades.append(gain_pct - 2*self._fee); pos = 0
        return self._compute_result(symbol, "three_bar_play", {"tp_pct": tp, "sl_pct": sl}, trades)

    def _backtest_dual_thrust(self, symbol, close, high, low, k, sl):
        trades = []; pos = 0; entry = 0
        for i in range(25, len(close)):
            hh = max(high[i-24:i]); ll = min(low[i-24:i]); hc = max(close[i-24:i]); lc = min(close[i-24:i])
            range_val = max(hh - lc, hc - ll)
            upper_trigger = close[i-1] + k * range_val
            if pos == 0 and close[i] > upper_trigger:
                pos = 1; entry = close[i]
            elif pos == 1:
                gain_pct = (close[i]/entry-1)*100
                lower_trigger = close[i-1] - k * range_val
                if close[i] < lower_trigger: trades.append(gain_pct - 2*self._fee); pos = 0
                elif gain_pct <= -sl: trades.append(gain_pct - 2*self._fee); pos = 0
        return self._compute_result(symbol, "dual_thrust", {"k": k, "sl_pct": sl}, trades)

    def _backtest_orb(self, symbol, close, high, low, tp, sl):
        trades = []; pos = 0; entry = 0
        for i in range(5, len(close)):
            # Use first 4 bars as "opening range"
            if i % 24 == 4:  # 4th hour of the "day"
                orb_high = max(high[i-4:i]); orb_low = min(low[i-4:i])
            if i % 24 > 4 and i % 24 < 20:  # trade during "day"
                if pos == 0 and close[i] > orb_high if 'orb_high' in dir() else False:
                    pos = 1; entry = close[i]
                elif pos == 1:
                    gain_pct = (close[i]/entry-1)*100
                    if gain_pct >= tp: trades.append(gain_pct - 2*self._fee); pos = 0
                    elif gain_pct <= -sl: trades.append(gain_pct - 2*self._fee); pos = 0
            if i % 24 == 23 and pos == 1:  # close at end of "day"
                trades.append((close[i]/entry-1)*100 - 2*self._fee); pos = 0
        return self._compute_result(symbol, "opening_range_breakout", {"tp_pct": tp, "sl_pct": sl}, trades)

    def _backtest_quantum_entropy(self, symbol, close, entropy_thresh, tp, sl):
        """High Shannon entropy in returns = uncertainty = mean revert. Low entropy = trend."""
        import pandas as pd
        returns = np.diff(np.log(np.maximum(close, 1e-9)))
        trades = []; pos = 0; entry = 0
        for i in range(30, len(close)-1):
            window = returns[max(0,i-20):i]
            # Shannon entropy of binned returns
            hist, _ = np.histogram(window, bins=5)
            probs = hist / max(hist.sum(), 1)
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs)) / np.log2(5)  # normalized 0-1

            if pos == 0:
                if entropy > entropy_thresh and close[i] < close[i-1]:  # high entropy + dip = buy reversal
                    pos = 1; entry = close[i]
                elif entropy < (1 - entropy_thresh) and close[i] > close[i-1]:  # low entropy + up = trend follow
                    pos = 1; entry = close[i]
            elif pos == 1:
                gain = (close[i]/entry-1)*100
                if gain >= tp: trades.append(gain - 2*self._fee); pos = 0
                elif gain <= -sl: trades.append(gain - 2*self._fee); pos = 0
        return self._compute_result(symbol, "quantum_entropy", {"entropy_thresh": entropy_thresh, "tp_pct": tp, "sl_pct": sl}, trades)

    def _backtest_quantum_tunneling(self, symbol, close, barrier_std, decay, sl):
        """Price near a resistance level has a 'tunneling probability' of breaking through."""
        import pandas as pd
        sma = pd.Series(close).rolling(20).mean().values
        std = pd.Series(close).rolling(20).std().values
        trades = []; pos = 0; entry = 0
        for i in range(25, len(close)):
            if np.isnan(sma[i]) or np.isnan(std[i]) or std[i] < 1e-9: continue
            barrier = sma[i] + barrier_std * std[i]
            distance = (barrier - close[i]) / std[i]  # distance to barrier in std devs
            # Quantum tunneling probability: exp(-2 * distance^2)
            tunnel_prob = np.exp(-2 * max(distance, 0)**2) if distance > 0 else 1.0

            if pos == 0 and tunnel_prob > 0.5 and close[i] > sma[i]:
                pos = 1; entry = close[i]
            elif pos == 1:
                gain = (close[i]/entry-1)*100
                if gain >= barrier_std * 1.0: trades.append(gain - 2*self._fee); pos = 0
                elif gain <= -sl: trades.append(gain - 2*self._fee); pos = 0
        return self._compute_result(symbol, "quantum_tunneling", {"barrier_std": barrier_std, "decay": decay, "sl_pct": sl}, trades)

    def _backtest_quantum_superposition(self, symbol, close, lookback, tp, sl):
        """Maintain probability-weighted bias from multiple timeframe signals."""
        import pandas as pd
        sma_short = pd.Series(close).rolling(max(lookback//3, 5)).mean().values
        sma_mid = pd.Series(close).rolling(lookback).mean().values
        sma_long = pd.Series(close).rolling(min(lookback*2, len(close)-1)).mean().values
        trades = []; pos = 0; entry = 0
        for i in range(lookback*2+1, len(close)):
            if np.isnan(sma_short[i]) or np.isnan(sma_mid[i]) or np.isnan(sma_long[i]): continue
            # Superposition: weighted probability of UP from 3 timeframes
            p_up = 0.0
            if close[i] > sma_short[i]: p_up += 0.5
            if close[i] > sma_mid[i]: p_up += 0.3
            if close[i] > sma_long[i]: p_up += 0.2
            # "Collapse" the wavefunction when probability > 0.7
            if pos == 0 and p_up >= 0.7:
                pos = 1; entry = close[i]
            elif pos == 1:
                gain = (close[i]/entry-1)*100
                if gain >= tp: trades.append(gain - 2*self._fee); pos = 0
                elif gain <= -sl: trades.append(gain - 2*self._fee); pos = 0
                elif p_up < 0.3: trades.append(gain - 2*self._fee); pos = 0  # wavefunction collapsed bearish
        return self._compute_result(symbol, "quantum_superposition", {"lookback": lookback, "tp_pct": tp, "sl_pct": sl}, trades)

    def _backtest_quantum_walk(self, symbol, close, walk_len, sigma_thresh, trail):
        """Detect when price deviates from a quantum random walk (drift detection)."""
        trades = []; pos = 0; entry = 0; peak = 0
        returns = np.diff(np.log(np.maximum(close, 1e-9)))
        for i in range(walk_len+1, len(close)-1):
            window = returns[i-walk_len:i]
            # Expected random walk: sqrt(n) * sigma
            sigma = np.std(window)
            expected_displacement = np.sqrt(walk_len) * sigma
            actual_displacement = abs(np.sum(window))
            # If actual >> expected, price is trending (not random)
            drift_ratio = actual_displacement / max(expected_displacement, 1e-9)

            if pos == 0 and drift_ratio > sigma_thresh and np.sum(window) > 0:
                pos = 1; entry = close[i]; peak = close[i]
            elif pos == 1:
                peak = max(peak, close[i])
                if close[i] < peak * (1 - trail/100): trades.append((close[i]/entry-1)*100 - 2*self._fee); pos = 0
        return self._compute_result(symbol, "quantum_walk_drift", {"walk_len": walk_len, "sigma_thresh": sigma_thresh, "trail_pct": trail}, trades)

    def _backtest_quantum_entanglement(self, symbol, close, lag, tp, sl):
        """Exploit autocorrelation (self-entanglement) regime shifts."""
        import pandas as pd
        returns = np.diff(np.log(np.maximum(close, 1e-9)))
        autocorr = pd.Series(returns).rolling(30).apply(lambda x: np.corrcoef(x[:-lag], x[lag:])[0,1] if len(x) > lag+1 else 0).values
        trades = []; pos = 0; entry = 0
        for i in range(35, len(close)-1):
            if np.isnan(autocorr[i-1]): continue
            # Positive autocorrelation = momentum regime
            if pos == 0 and autocorr[i-1] > 0.3 and close[i] > close[i-1]:
                pos = 1; entry = close[i]
            # Negative autocorrelation = mean reversion regime
            elif pos == 0 and autocorr[i-1] < -0.3 and close[i] < close[i-1]:
                pos = 1; entry = close[i]
            elif pos == 1:
                gain = (close[i]/entry-1)*100
                if gain >= tp: trades.append(gain - 2*self._fee); pos = 0
                elif gain <= -sl: trades.append(gain - 2*self._fee); pos = 0
        return self._compute_result(symbol, "quantum_entanglement", {"lag": lag, "tp_pct": tp, "sl_pct": sl}, trades)

    def _backtest_quantum_annealing(self, symbol, close, high, low, temp_decay, tp, sl):
        """Simulated annealing for optimal entry: accept worse entries early, get pickier over time."""
        trades = []; pos = 0; entry = 0
        import pandas as pd
        sma = pd.Series(close).rolling(20).mean().values
        std = pd.Series(close).rolling(20).std().values
        temperature = 1.0
        for i in range(25, len(close)):
            if np.isnan(sma[i]) or np.isnan(std[i]) or std[i] < 1e-9: continue
            # Z-score of current price
            z = (close[i] - sma[i]) / std[i]
            # Acceptance probability: higher temperature = accept more signals
            accept_prob = np.exp(-abs(z) / max(temperature, 0.01))

            if pos == 0 and z < -1.0 and accept_prob > 0.5:
                pos = 1; entry = close[i]
                temperature *= temp_decay  # cool down
            elif pos == 1:
                gain = (close[i]/entry-1)*100
                if gain >= tp: trades.append(gain - 2*self._fee); pos = 0; temperature = min(temperature * 1.1, 1.0)
                elif gain <= -sl: trades.append(gain - 2*self._fee); pos = 0; temperature = min(temperature * 1.1, 1.0)
        return self._compute_result(symbol, "quantum_annealing", {"temp_decay": temp_decay, "tp_pct": tp, "sl_pct": sl}, trades)

    def _backtest_quantum_decoherence(self, symbol, close, window, stability_thresh, trail):
        """Trade only when regime is stable (low decoherence). Exit when regime destabilizes."""
        import pandas as pd
        returns = np.diff(np.log(np.maximum(close, 1e-9)))
        # Rolling regime stability: ratio of |mean| to std (high = stable trend)
        roll_mean = pd.Series(returns).rolling(window).mean().values
        roll_std = pd.Series(returns).rolling(window).std().values
        trades = []; pos = 0; entry = 0; peak = 0
        for i in range(window+2, len(close)-1):
            if np.isnan(roll_mean[i-1]) or np.isnan(roll_std[i-1]) or roll_std[i-1] < 1e-9: continue
            stability = abs(roll_mean[i-1]) / roll_std[i-1]  # signal-to-noise ratio

            if pos == 0 and stability > stability_thresh and roll_mean[i-1] > 0:
                pos = 1; entry = close[i]; peak = close[i]
            elif pos == 1:
                peak = max(peak, close[i])
                cur_stability = abs(roll_mean[i-1]) / max(roll_std[i-1], 1e-9) if not np.isnan(roll_std[i-1]) else 0
                if close[i] < peak * (1 - trail/100): trades.append((close[i]/entry-1)*100 - 2*self._fee); pos = 0
                elif cur_stability < stability_thresh * 0.5: trades.append((close[i]/entry-1)*100 - 2*self._fee); pos = 0  # decoherence
        return self._compute_result(symbol, "quantum_decoherence", {"window": window, "stability_thresh": stability_thresh, "trail_pct": trail}, trades)

    def _backtest_quantum_interference(self, symbol, close, fast, mid, slow, sl):
        """Three EMA waves: constructive interference (all aligned) = strong signal."""
        import pandas as pd
        ema_f = pd.Series(close).ewm(span=fast).mean().values
        ema_m = pd.Series(close).ewm(span=mid).mean().values
        ema_s = pd.Series(close).ewm(span=slow).mean().values
        trades = []; pos = 0; entry = 0
        for i in range(slow+1, len(close)):
            # Constructive interference: all three aligned bullish
            constructive_bull = ema_f[i] > ema_m[i] > ema_s[i]
            # Destructive: mixed signals
            if pos == 0 and constructive_bull and not (ema_f[i-1] > ema_m[i-1] > ema_s[i-1]):
                pos = 1; entry = close[i]
            elif pos == 1:
                gain = (close[i]/entry-1)*100
                destructive = not (ema_f[i] > ema_m[i] > ema_s[i])
                if destructive: trades.append(gain - 2*self._fee); pos = 0
                elif gain <= -sl: trades.append(gain - 2*self._fee); pos = 0
        return self._compute_result(symbol, "quantum_interference", {"fast": fast, "mid": mid, "slow": slow, "sl_pct": sl}, trades)

    def get_last_scan(self) -> Optional[ScanResult]:
        return self._last_scan

    def get_recommended_symbols(self, top_n: int = 5) -> List[str]:
        """Get top N symbols by opportunity quality."""
        if not self._last_scan:
            return []
        return [r.symbol for r in self._last_scan.top_opportunities[:top_n]]

    def get_recommended_params(self, symbol: str) -> Optional[Dict]:
        """Get recommended strategy + params for a symbol."""
        if not self._last_scan:
            return None
        result = self._last_scan.best_per_symbol.get(symbol)
        if result:
            return {
                "strategy": result.strategy,
                "params": result.params,
                "sharpe": result.sharpe,
                "win_rate": result.win_rate,
                "return_pct": result.total_return_pct,
            }
        return None

    def get_advisory(self) -> Dict[str, Any]:
        """Get advisory dict for the trading loop."""
        if not self._last_scan:
            return {"scan_count": 0, "opportunities": 0}

        return {
            "scan_count": self._scan_count,
            "opportunities": len(self._last_scan.top_opportunities),
            "top_symbols": self.get_recommended_symbols(5),
            "scan_age_seconds": time.time() - self._last_scan.timestamp,
            "top_strategies": [
                {
                    "symbol": r.symbol,
                    "strategy": r.strategy,
                    "sharpe": round(r.sharpe, 3),
                    "return_pct": round(r.total_return_pct, 2),
                    "win_rate": round(r.win_rate, 2),
                    "trades": r.trades,
                }
                for r in self._last_scan.top_opportunities[:5]
            ],
        }
