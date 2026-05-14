"""
Portfolio state manager — single source of truth for positions, cash, P&L.

Thread-safe via threading.Lock. All state mutations go through this class.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PortfolioSnapshot:
    """Immutable snapshot of portfolio state at a point in time."""
    portfolio_value_aud: float
    cash_balance_aud: float
    positions: Dict[str, Dict]
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl_aud: float
    realized_pnl_aud: float
    daily_pnl_aud: float
    total_fees_aud: float
    max_drawdown_aud: float
    peak_equity_aud: float
    timestamp: float = field(default_factory=time.time)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / max(self.total_trades, 1)

    @property
    def open_position_count(self) -> int:
        return sum(1 for p in self.positions.values() if p and float(p.get("quantity", 0) or 0) > 0)


class PortfolioManager:
    """
    Thread-safe portfolio state manager.

    All position/cash/P&L mutations must go through this class.
    Readers get immutable snapshots via snapshot().
    """

    def __init__(self, starting_capital_aud: float = 1000.0, aud_to_usd: float = 0.65):
        self._lock = threading.Lock()
        self._aud_to_usd = max(0.01, aud_to_usd)

        # Core state
        self._portfolio_value_aud = starting_capital_aud
        self._cash_balance_aud = starting_capital_aud
        self._positions: Dict[str, Dict] = {}
        self._trade_history: deque = deque(maxlen=10000)

        # Counters
        self._total_trades = 0
        self._winning_trades = 0
        self._losing_trades = 0
        self._total_pnl_aud = 0.0
        self._realized_pnl_aud = 0.0
        self._daily_pnl_aud = 0.0
        self._total_fees_aud = 0.0
        self._max_drawdown_aud = 0.0
        self._peak_equity_aud = starting_capital_aud

        logger.info("PortfolioManager initialized: capital=%.2f AUD", starting_capital_aud)

    def snapshot(self) -> PortfolioSnapshot:
        """Return an immutable snapshot of current state. Thread-safe."""
        with self._lock:
            return PortfolioSnapshot(
                portfolio_value_aud=self._portfolio_value_aud,
                cash_balance_aud=self._cash_balance_aud,
                positions=dict(self._positions),  # shallow copy
                total_trades=self._total_trades,
                winning_trades=self._winning_trades,
                losing_trades=self._losing_trades,
                total_pnl_aud=self._total_pnl_aud,
                realized_pnl_aud=self._realized_pnl_aud,
                daily_pnl_aud=self._daily_pnl_aud,
                total_fees_aud=self._total_fees_aud,
                max_drawdown_aud=self._max_drawdown_aud,
                peak_equity_aud=self._peak_equity_aud,
            )

    def record_buy(self, symbol: str, quantity: float, price_usd: float,
                   commission_usd: float = 0.0, strategy: str = "unknown",
                   order_id: str = "", **extra) -> Dict[str, Any]:
        """Record a BUY fill. Returns trade record dict. Thread-safe."""
        with self._lock:
            aud_to_usd = self._aud_to_usd
            notional_usd = quantity * price_usd
            cost_aud = (notional_usd + commission_usd) / aud_to_usd

            # Update position
            pos = self._positions.get(symbol) or {}
            held_qty = float(pos.get("quantity", 0) or 0)
            avg_px = float(pos.get("avg_price", 0) or 0)

            new_qty = held_qty + quantity
            if new_qty > 0:
                held_cost = held_qty * avg_px
                buy_cost = quantity * price_usd + commission_usd
                new_avg = (held_cost + buy_cost) / new_qty
            else:
                new_avg = price_usd

            self._positions[symbol] = {
                "quantity": new_qty,
                "avg_price": new_avg,
                "entry_price": new_avg,
                "current_price": price_usd,
                "side": "BUY",
                "symbol": symbol,
            }

            # Update cash
            self._cash_balance_aud -= cost_aud
            self._total_fees_aud += commission_usd / aud_to_usd
            self._total_trades += 1

            # Trade record
            record = {
                "symbol": symbol, "side": "buy", "quantity": quantity,
                "price": price_usd, "commission": commission_usd,
                "order_id": order_id, "strategy": strategy,
                "timestamp": time.time(), "pnl": 0.0,
                **extra,
            }
            self._trade_history.append(record)
            return record

    def record_sell(self, symbol: str, quantity: float, price_usd: float,
                    commission_usd: float = 0.0, strategy: str = "unknown",
                    order_id: str = "", **extra) -> Dict[str, Any]:
        """Record a SELL fill. Returns trade record dict with realized P&L. Thread-safe."""
        with self._lock:
            aud_to_usd = self._aud_to_usd
            pos = self._positions.get(symbol) or {}
            held_qty = float(pos.get("quantity", 0) or 0)
            avg_px = float(pos.get("avg_price", 0) or 0)

            sell_qty = min(quantity, held_qty)
            if sell_qty <= 0:
                return {"symbol": symbol, "side": "sell", "status": "no_position"}

            # P&L calculation
            entry_usd = sell_qty * avg_px
            proceeds_usd = sell_qty * price_usd - commission_usd
            realized_usd = proceeds_usd - entry_usd
            realized_aud = realized_usd / aud_to_usd

            # Update position
            remaining = held_qty - sell_qty
            if remaining > 0.0001:
                self._positions[symbol] = {
                    **pos,
                    "quantity": remaining,
                    "current_price": price_usd,
                }
            else:
                if symbol in self._positions:
                    del self._positions[symbol]

            # Update cash and P&L
            self._cash_balance_aud += (proceeds_usd + commission_usd) / aud_to_usd
            self._total_fees_aud += commission_usd / aud_to_usd
            self._total_pnl_aud += realized_aud
            self._realized_pnl_aud = self._total_pnl_aud
            self._daily_pnl_aud += realized_aud
            self._total_trades += 1

            if realized_aud > 0:
                self._winning_trades += 1
            elif realized_aud < 0:
                self._losing_trades += 1

            # Trade record
            record = {
                "symbol": symbol, "side": "sell", "quantity": sell_qty,
                "price": price_usd, "commission": commission_usd,
                "avg_entry_price": avg_px, "pnl": realized_aud,
                "pnl_usd": realized_usd, "order_id": order_id,
                "strategy": strategy, "timestamp": time.time(),
                **extra,
            }
            self._trade_history.append(record)
            return record

    def update_price(self, symbol: str, price: float) -> None:
        """Update current price for a position. Thread-safe."""
        with self._lock:
            pos = self._positions.get(symbol)
            if pos is not None:
                pos["current_price"] = price

    def update_portfolio_value(self) -> float:
        """Recalculate portfolio value from cash + positions. Thread-safe. Returns new value."""
        with self._lock:
            positions_value = 0.0
            for sym, pos in self._positions.items():
                if pos is None:
                    continue
                qty = float(pos.get("quantity", 0) or 0)
                px = float(pos.get("current_price", 0) or 0)
                if qty > 0 and px > 0:
                    positions_value += qty * px / self._aud_to_usd

            self._portfolio_value_aud = self._cash_balance_aud + positions_value

            # Peak and drawdown tracking
            if self._portfolio_value_aud > self._peak_equity_aud:
                self._peak_equity_aud = self._portfolio_value_aud

            if self._peak_equity_aud > 0:
                current_dd = self._peak_equity_aud - self._portfolio_value_aud
                if current_dd > self._max_drawdown_aud:
                    self._max_drawdown_aud = current_dd

            return self._portfolio_value_aud

    def reset_daily(self) -> None:
        """Reset daily P&L counter. Thread-safe."""
        with self._lock:
            self._daily_pnl_aud = 0.0

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for symbol. Thread-safe."""
        with self._lock:
            pos = self._positions.get(symbol)
            return dict(pos) if pos else None

    def get_positions(self) -> Dict[str, Dict]:
        """Get copy of all positions. Thread-safe."""
        with self._lock:
            return dict(self._positions)

    def get_trade_history(self, last_n: int = 50) -> List[Dict]:
        """Get recent trade history. Thread-safe."""
        with self._lock:
            history = list(self._trade_history)
            return history[-last_n:] if last_n else history

    def reconcile_position(self, symbol: str, exchange_qty: float, price: float) -> None:
        """Reconcile internal position with exchange balance. Thread-safe."""
        with self._lock:
            if exchange_qty > 0.0001:
                pos = self._positions.get(symbol, {})
                internal_qty = float(pos.get("quantity", 0) or 0)
                if abs(internal_qty - exchange_qty) > 0.0001:
                    logger.warning(
                        "Position reconciliation: %s internal=%.8f exchange=%.8f — adjusting",
                        symbol, internal_qty, exchange_qty,
                    )
                self._positions[symbol] = {
                    **pos,
                    "quantity": exchange_qty,
                    "current_price": price,
                    "side": "BUY",
                    "symbol": symbol,
                }
            else:
                if symbol in self._positions:
                    del self._positions[symbol]

    def set_aud_to_usd(self, rate: float) -> None:
        """Update AUD/USD rate. Thread-safe."""
        with self._lock:
            self._aud_to_usd = max(0.01, rate)
