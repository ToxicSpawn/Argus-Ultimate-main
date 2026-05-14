"""
Unified event-driven backtester.

Goal: run the *same* signal -> sizing -> risk -> execution flow as the unified
runtime, but driven by historical candles.

Includes: realistic slippage per venue, configurable latency simulation,
out-of-sample window reporting.

NOTE: This is the canonical implementation. The stub at backtest/unified_event_backtester.py
has been replaced with a redirect — import from backtesting.unified_event_backtester.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from execution.state_store import ExecutionStateStore
from strategies.unified.strategy_engine import StrategyEngine
from unified_capital_optimizer import CapitalOptimizer1K

try:
    from risk.stop_loss.profit_target_stop import ProfitTargetStop
    _TP_LADDER_AVAILABLE = True
except Exception:
    _TP_LADDER_AVAILABLE = False

try:
    from execution.market_impact import MarketImpactModel
    _MARKET_IMPACT_AVAILABLE = True
except Exception:
    _MARKET_IMPACT_AVAILABLE = False

logger = logging.getLogger(__name__)


class HistoricalMarketDataService:
    """
    Minimal MarketDataService adapter over an in-memory OHLCV DataFrame.

    StrategyEngine only needs:
    - fetch_ohlcv_df(symbol, timeframe, limit)
    - fetch_ticker(symbol) (optional)
    - fetch_order_book(symbol, limit) (optional)
    """

    def __init__(self, *, symbol: str, ohlcv: pd.DataFrame):
        self.symbol = str(symbol)
        self.ohlcv = ohlcv.copy()
        self._cursor = 0

    def set_cursor(self, idx: int) -> None:
        self._cursor = max(0, int(idx))

    async def fetch_ohlcv_df(self, symbol: str, timeframe: str = "1m", limit: int = 200) -> Optional[pd.DataFrame]:
        if str(symbol) != self.symbol:
            return None
        lim = max(1, int(limit))
        end = max(0, self._cursor + 1)
        start = max(0, end - lim)
        df = self.ohlcv.iloc[start:end].copy()
        return df if not df.empty else None

    async def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        if str(symbol) != self.symbol:
            return None
        if self._cursor < 0 or self._cursor >= len(self.ohlcv):
            return None
        close = float(self.ohlcv.iloc[self._cursor]["close"])
        return {"symbol": symbol, "last": close, "bid": None, "ask": None, "timestamp": None, "price": close}

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
        return None


@dataclass
class BacktestResult:
    symbol: str
    start_equity_aud: float
    end_equity_aud: float
    total_return_pct: float
    max_drawdown_pct: float
    trades: int
    wins: int
    losses: int


def apply_slippage_bps(price: float, side: str, slippage_bps: float = 5.0) -> float:
    bps = slippage_bps / 10000.0
    if side.upper() == "BUY":
        return price * (1.0 + bps)
    return price * (1.0 - bps)


def simulate_latency_delay_ms(latency_ms: float) -> float:
    if latency_ms <= 0:
        return 0.0
    return latency_ms / 1000.0


class UnifiedEventBacktester:
    def __init__(self, config: Any):
        self.config = config
        self.strategy_engine = StrategyEngine(config)
        self.capital_optimizer = CapitalOptimizer1K(config)
        self.state = ExecutionStateStore(db_path="data/unified_backtest_state.db")

        self.cash_aud = float(getattr(config, "starting_capital_aud", 1000.0) or 1000.0)
        self.start_cash_aud = self.cash_aud

        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.peak_equity = self.cash_aud
        self.max_drawdown = 0.0
        _backtest = getattr(config, "backtest", None) or {}
        if isinstance(_backtest, dict):
            self.slippage_bps = float(_backtest.get("slippage_bps", 5.0))
            self.latency_sim_ms = float(_backtest.get("latency_ms", 0.0))
            self.oos_train_ratio = float(_backtest.get("oos_train_ratio", 0.0))
            self.rate_limit_reject_pct = float(_backtest.get("rate_limit_reject_pct", 0.0))
            self.fill_probability = float(_backtest.get("fill_probability", 1.0))
            self.trailing_stop_enabled = bool(_backtest.get("trailing_stop_enabled", False))
            self.trailing_stop_pct = float(_backtest.get("trailing_stop_pct", 0.02))
            self.static_stop_enabled = bool(_backtest.get("stop_loss_enabled", True))
        else:
            self.slippage_bps = float(getattr(config, "backtest_slippage_bps", 5.0))
            self.latency_sim_ms = float(getattr(config, "backtest_latency_ms", 0.0))
            self.oos_train_ratio = float(getattr(config, "backtest_oos_train_ratio", 0.0) or 0.0)
            self.rate_limit_reject_pct = float(getattr(config, "backtest_rate_limit_reject_pct", 0.0) or 0.0)
            self.fill_probability = float(getattr(config, "backtest_fill_probability", 1.0) or 1.0)
            self.trailing_stop_enabled = bool(getattr(config, "trailing_stop_enabled", False))
            self.trailing_stop_pct = float(getattr(config, "trailing_stop_pct", 0.02) or 0.02)
            self.static_stop_enabled = bool(getattr(config, "stop_loss_enabled", True))

        self.commission_rate = float(
            _backtest.get("commission_rate", 0.0) if isinstance(_backtest, dict)
            else getattr(config, "backtest_commission_rate", 0.0) or 0.0
        )
        if self.commission_rate <= 0:
            self.commission_rate = float(getattr(config, "kraken_taker_fee", 0.0026) or 0.0026)
        self.total_commissions_aud = 0.0
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self._tp_ladders: Dict[str, Any] = {}
        _tp_cfg = _backtest.get("tp_ladder", None) if isinstance(_backtest, dict) else None
        self._tp_ladder_enabled = bool((_tp_cfg.get("enabled", False)) if isinstance(_tp_cfg, dict) else False)
        self._tp_ladder_tiers = (
            [tuple(t) for t in _tp_cfg["tiers"]]
            if isinstance(_tp_cfg, dict) and "tiers" in _tp_cfg
            else [(0.01, 0.25), (0.02, 0.25), (0.03, 0.25), (0.05, 0.25)]
        )
        self._impact_model: Optional[Any] = None
        if _MARKET_IMPACT_AVAILABLE:
            self._impact_model = MarketImpactModel()
        self._use_dynamic_slippage = bool(
            (_backtest.get("use_dynamic_slippage", False)) if isinstance(_backtest, dict)
            else getattr(config, "backtest_use_dynamic_slippage", False)
        )

    async def initialize(self) -> None:
        await self.capital_optimizer.initialize()
        try:
            setattr(self.config, "run_mode", "backtest")
        except Exception as _e:
            logger.debug("unified_event_backtester error: %s", _e)

    def _dynamic_slippage(self, price: float, qty: float, side: str, volume: float = 0.0) -> float:
        if self._use_dynamic_slippage and self._impact_model is not None:
            if volume > 0:
                self._impact_model.update_market_data(volatility=0.02, volume=volume)
            impact = self._impact_model.estimate_impact(
                order_size=qty, price=price, side=side.lower(), urgency=0.5,
            )
            impact_pct = float(impact.get("total_impact_pct", 0.0))
            min_slip = self.slippage_bps / 10000.0
            slip = max(min_slip, impact_pct)
            if side.upper() == "BUY":
                return price * (1.0 + slip)
            return price * (1.0 - slip)
        return apply_slippage_bps(price, side, self.slippage_bps)

    def _equity(self, last_price: float) -> float:
        pos = self.state.get_positions().get(self._symbol(), {}) or {}
        qty = float(pos.get("quantity") or 0.0)
        return float(self.cash_aud + qty * float(last_price))

    def _symbol(self) -> str:
        pairs = list(getattr(self.config, "trading_pairs", []) or [])
        return str(pairs[0] if pairs else "BTC/USD")

    async def run(self, *, symbol: str, ohlcv: pd.DataFrame) -> BacktestResult:
        symbol = str(symbol)
        df = ohlcv.copy()
        if "close" not in df.columns:
            raise ValueError("OHLCV DataFrame requires 'close' column")

        mds = HistoricalMarketDataService(symbol=symbol, ohlcv=df)
        warmup = 200

        for i in range(len(df)):
            mds.set_cursor(i)
            last_price = float(df.iloc[i]["close"])
            self.state.update_position_price(symbol, last_price)

            if i < warmup:
                continue

            if symbol in self.open_positions:
                self.open_positions[symbol]["peak_price"] = max(
                    self.open_positions[symbol].get("peak_price", last_price), last_price
                )

            if self.trailing_stop_enabled or self.static_stop_enabled:
                exit_reason = self._check_exits(symbol, last_price)
                if exit_reason:
                    self._execute_exit(symbol, last_price)

            if self.trailing_stop_enabled or self.static_stop_enabled:
                exit_reason = self._check_exits(symbol, last_price)
                if exit_reason:
                    self._execute_exit(symbol, last_price)

            if hasattr(self.strategy_engine, "update_drawdown"):
                dd_pct = (self.peak_equity - self._equity(last_price)) / max(self.peak_equity, 1e-9)
                self.strategy_engine.update_drawdown(max(0.0, dd_pct))

            signals = await self.strategy_engine.generate_signals(mds)
            if not signals:
                self._update_dd(last_price)
                continue

            optimized = await self.capital_optimizer.optimize_signals(signals)
            if not optimized:
                self._update_dd(last_price)
                continue

            if getattr(self, "rate_limit_reject_pct", 0.0) > 0 and random.random() < (self.rate_limit_reject_pct / 100.0):
                self._update_dd(last_price)
                continue

            _latency_bars = 0
            if getattr(self, "latency_sim_ms", 0) > 0:
                candle_ms = 60000.0
                _latency_bars = max(1, int(self.latency_sim_ms / candle_ms))
            _fill_bar_idx = min(i + _latency_bars, len(df) - 1)
            _fill_price = float(df.iloc[_fill_bar_idx]["close"])

            for s in optimized:
                action = str(getattr(s, "action", "") or "").upper()
                qty = float(getattr(s, "quantity", 0.0) or 0.0)
                px = float(_fill_price)
                pos = (self.state.get_positions().get(symbol) or {})
                held = float(pos.get("quantity") or 0.0)

                if action == "BUY" and held <= 0 and qty > 0:
                    if random.random() > getattr(self, "fill_probability", 1.0):
                        continue
                    bar_vol = float(df.iloc[i].get("volume", 0)) if "volume" in df.columns else 0.0
                    fill_px = self._dynamic_slippage(px, qty, "BUY", volume=bar_vol)
                    notional = qty * fill_px
                    commission_aud = self._quote_to_aud_notional(symbol, notional * self.commission_rate)
                    cost_aud = self._quote_to_aud_notional(symbol, notional) + commission_aud
                    if self.cash_aud >= cost_aud:
                        self.cash_aud -= cost_aud
                        self.total_commissions_aud += commission_aud
                        self.state.apply_fill(symbol=symbol, side="BUY", quantity=qty, price=fill_px)
                        self.trades += 1
                        sig_sl = getattr(s, "stop_loss", None)
                        sig_tp = getattr(s, "take_profit", None)
                        self.open_positions[symbol] = {
                            "entry_price": fill_px,
                            "peak_price": fill_px,
                            "stop_loss": float(sig_sl) if sig_sl else None,
                            "take_profit": float(sig_tp) if sig_tp else None,
                        }

                if action == "SELL" and held > 0:
                    if random.random() > getattr(self, "fill_probability", 1.0):
                        continue
                    sell_qty = min(held, qty if qty > 0 else held)
                    bar_vol_s = float(df.iloc[i].get("volume", 0)) if "volume" in df.columns else 0.0
                    fill_px = self._dynamic_slippage(px, sell_qty, "SELL", volume=bar_vol_s)
                    notional_sell = sell_qty * fill_px
                    commission_sell_aud = self._quote_to_aud_notional(symbol, notional_sell * self.commission_rate)
                    proceeds_aud = self._quote_to_aud_notional(symbol, notional_sell) - commission_sell_aud
                    self.total_commissions_aud += commission_sell_aud
                    entry_px = float(pos.get("avg_price") or px)
                    pnl_aud = self._quote_to_aud_notional(symbol, sell_qty * (fill_px - entry_px)) - commission_sell_aud
                    self.cash_aud += proceeds_aud
                    self.state.apply_fill(symbol=symbol, side="SELL", quantity=sell_qty, price=fill_px)
                    self.trades += 1
                    if pnl_aud >= 0:
                        self.wins += 1
                    else:
                        self.losses += 1
                    self.open_positions.pop(symbol, None)
                    self._tp_ladders.pop(symbol, None)

            self._update_dd(last_price)

        last_close = float(df.iloc[-1]["close"])
        for sym, pos in (self.state.get_positions() or {}).items():
            held = float(pos.get("quantity") or 0.0)
            if held > 0:
                entry_px = float(pos.get("avg_price") or last_close)
                fill_px = apply_slippage_bps(last_close, "SELL", self.slippage_bps)
                notional_close = held * fill_px
                commission_close_aud = self._quote_to_aud_notional(sym, notional_close * self.commission_rate)
                proceeds_aud = self._quote_to_aud_notional(sym, notional_close) - commission_close_aud
                self.total_commissions_aud += commission_close_aud
                pnl_aud = self._quote_to_aud_notional(sym, held * (fill_px - entry_px)) - commission_close_aud
                self.cash_aud += proceeds_aud
                self.state.apply_fill(symbol=sym, side="SELL", quantity=held, price=fill_px)
                self.trades += 1
                if pnl_aud >= 0:
                    self.wins += 1
                else:
                    self.losses += 1

        end_eq = self._equity(last_close)
        total_ret = (end_eq - self.start_cash_aud) / max(self.start_cash_aud, 1e-9) * 100.0
        return BacktestResult(
            symbol=symbol,
            start_equity_aud=float(self.start_cash_aud),
            end_equity_aud=float(end_eq),
            total_return_pct=float(total_ret),
            max_drawdown_pct=float(self.max_drawdown) * 100.0,
            trades=int(self.trades),
            wins=int(self.wins),
            losses=int(self.losses),
        )

    def _update_dd(self, last_price: float) -> None:
        eq = self._equity(last_price)
        if eq > self.peak_equity:
            self.peak_equity = eq
        dd = (self.peak_equity - eq) / max(self.peak_equity, 1e-9)
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    def _check_exits(self, symbol: str, current_price: float) -> Optional[str]:
        pos_data = self.open_positions.get(symbol)
        if not pos_data:
            return None
        entry = pos_data.get("entry_price", 0.0)
        peak = pos_data.get("peak_price", entry)
        sl = pos_data.get("stop_loss")
        tp = pos_data.get("take_profit")
        if tp and current_price >= tp:
            return "take_profit"
        if self.static_stop_enabled and sl and current_price <= sl:
            return "stop_loss"
        if self.trailing_stop_enabled and peak > entry:
            trail_price = peak * (1.0 - self.trailing_stop_pct)
            if current_price <= trail_price:
                return "trailing_stop"
        return None

    def _execute_exit(self, symbol: str, current_price: float) -> None:
        pos = self.state.get_positions().get(symbol, {})
        held = float(pos.get("quantity") or 0.0)
        if held <= 0:
            return
        fill_px = self._dynamic_slippage(current_price, held, "SELL")
        entry_px = float(pos.get("avg_price") or current_price)
        notional_exit = held * fill_px
        commission_exit_aud = self._quote_to_aud_notional(symbol, notional_exit * self.commission_rate)
        proceeds_aud = self._quote_to_aud_notional(symbol, notional_exit) - commission_exit_aud
        self.total_commissions_aud += commission_exit_aud
        pnl_aud = self._quote_to_aud_notional(symbol, held * (fill_px - entry_px)) - commission_exit_aud
        self.cash_aud += proceeds_aud
        self.state.apply_fill(symbol=symbol, side="SELL", quantity=held, price=fill_px)
        self.trades += 1
        if pnl_aud >= 0:
            self.wins += 1
        else:
            self.losses += 1
        self.open_positions.pop(symbol, None)
        self._tp_ladders.pop(symbol, None)

    def _quote_to_aud_notional(self, symbol: str, notional_quote: float) -> float:
        sym = str(symbol or "")
        quote = sym.split("/")[-1].upper() if "/" in sym else "USD"
        aud_to_usd = float(getattr(self.config, "aud_to_usd", 0.65) or 0.65)
        if quote == "AUD":
            return float(notional_quote)
        return float(notional_quote) / max(aud_to_usd, 1e-9)


def run_backtest_sync(*, config: Any, symbol: str, ohlcv: pd.DataFrame) -> BacktestResult:
    bt = UnifiedEventBacktester(config)

    async def _run() -> BacktestResult:
        await bt.initialize()
        return await bt.run(symbol=symbol, ohlcv=ohlcv)

    return asyncio.run(_run())


def run_backtest_oos(
    config: Any,
    symbol: str,
    ohlcv: pd.DataFrame,
    train_ratio: float = 0.7,
) -> Dict[str, Any]:
    n = len(ohlcv)
    if n < 100:
        return {"error": "insufficient_data", "oos_sharpe": None, "oos_drawdown_pct": None}
    split = int(n * train_ratio)
    oos_df = ohlcv.iloc[split:]
    bt = UnifiedEventBacktester(config)

    async def _run_oos() -> BacktestResult:
        await bt.initialize()
        return await bt.run(symbol=symbol, ohlcv=oos_df)

    result = asyncio.run(_run_oos())
    oos_ret_pct = result.total_return_pct
    oos_dd_pct = result.max_drawdown_pct
    n_bars = max(len(oos_df), 1)
    vol_proxy = max(oos_dd_pct / 100.0 / 2.0, 1e-9)
    oos_sharpe = (oos_ret_pct / 100.0) / vol_proxy if vol_proxy > 1e-9 else 0.0
    oos_calmar = (oos_ret_pct / 100.0) / max(oos_dd_pct / 100.0, 1e-9) if oos_dd_pct else 0.0
    return {
        "oos_return_pct": oos_ret_pct,
        "oos_drawdown_pct": oos_dd_pct,
        "oos_sharpe": oos_sharpe,
        "oos_calmar": oos_calmar,
        "oos_trades": result.trades,
        "train_bars": split,
        "oos_bars": len(oos_df),
    }
