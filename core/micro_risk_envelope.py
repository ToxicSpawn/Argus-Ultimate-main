"""
MicroRiskEnvelope — $1k-specific risk management layer with fee-aware position sizing.

Enforces per-pair position limits, daily loss hard stops, global drawdown halts,
and exchange-specific fee-aware viability checks for micro-capital HFT.

Fee break-even model (round-trip):
  - Bybit spot:  0 bps  (zero maker fee — any spread is profitable)  ← PREFERRED
  - Bybit perp:  2 bps  (0.01% taker × 2 sides, or 0 maker)         ← PREFERRED
  - Kraken:     32 bps  (0.16% × 2 sides)  — funding only, not spread
  - Coinbase:   80 bps  (0.40% × 2 sides)  — AVOID at MICRO tier

Position sizing modes:
  - fixed_fraction: size = capital × fraction × signal_confidence
  - kelly: f* = (win_rate × win/loss_ratio - (1-win_rate)) / win_loss_ratio,
           capped at kelly_fraction (fifth Kelly by default at MICRO)

MICRO-tier optimized defaults ($1k AUD / $620 USD):
  - max_position_per_pair_pct = 6%   → max $37.20 per pair
  - max_open_pairs = 2               → max 12% portfolio heat
  - max_daily_loss_usd = $6.20       → 1% of capital hard stop
  - max_drawdown_pct = 12%           → halt before -$74.40
  - position_sizing_mode = kelly     → adapts after 10 fills
  - kelly_fraction = 0.20            → tighter cap vs default 0.25
  - position_size_pct = 0.015        → 1.5% fixed-fraction fallback
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NS_PER_SECOND: int = 1_000_000_000
UTC_RESET_HOUR: int = 0  # midnight UTC daily reset

# Exchange minimum order sizes in USD
EXCHANGE_MIN_ORDER_USD: Dict[str, float] = {
    "bybit_spot": 1.0,
    "bybit_perp": 1.0,
    "kraken": 5.0,   # Kraken minimum varies per asset, $5 is conservative
    "coinbase": 1.0,
}

# Round-trip fee in basis points per exchange (entry + exit)
EXCHANGE_ROUND_TRIP_FEES_BPS: Dict[str, float] = {
    "bybit_spot": 0.0,    # zero maker fee program
    "bybit_perp": 2.0,    # 0.01% taker × 2
    "kraken": 32.0,       # 0.16% × 2
    "coinbase": 80.0,     # 0.40% × 2
}

# Preferred exchange routing order for MICRO tier (lowest fee first)
# Kraken is viable only for funding-rate strategies (spread_bps=0 path)
# Coinbase should be avoided entirely at MICRO tier
PREFERRED_EXCHANGE_ORDER: list = [
    "bybit_spot",   # 0 bps — always viable
    "bybit_perp",   # 2 bps — viable for funding harvest
    "kraken",       # 32 bps — funding harvest only, NOT spread strategies
    # "coinbase",   # 80 bps — avoid at MICRO tier
]

# Risk status thresholds (drawdown %) — tightened for MICRO tier
_YELLOW_THRESHOLD: float = 7.0   # was 10.0 — warn earlier
_RED_THRESHOLD: float = 10.0     # was 13.0
_HALT_THRESHOLD: float = 12.0    # was 15.0 — matches max_drawdown_pct default

# Fixed fraction default per trade (1.5% of allocated capital) — was 2%
DEFAULT_POSITION_PCT: float = 0.015

# For Kelly calculation: default prior win statistics (conservative)
_DEFAULT_WIN_RATE: float = 0.52
_DEFAULT_AVG_WIN: float = 1.0   # normalised units
_DEFAULT_AVG_LOSS: float = 0.8  # normalised units

# Minimum fills required before switching from prior to live Kelly stats
_KELLY_MIN_FILLS: int = 10


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MicroRiskConfig:
    """Configuration for MicroRiskEnvelope — optimized for $1k AUD / $620 USD MICRO tier."""
    total_capital_usd: float = 620.0
    max_drawdown_pct: float = 12.0          # was 15.0 — tighter capital preservation
    max_position_per_pair_pct: float = 6.0  # was 20.0 — fixes heat guard overflow
    max_open_pairs: int = 2                 # was 5 — 2 × 6% = 12% max portfolio heat
    max_daily_loss_usd: float = 6.20        # was 10.0 — 1% of $620 capital
    position_sizing_mode: str = "kelly"     # was fixed_fraction — adapts after 10 fills
    kelly_fraction: float = 0.20            # was 0.25 — tighter cap at MICRO
    fee_rates: Dict[str, float] = field(default_factory=lambda: {
        "bybit_spot": 0.0,
        "bybit_perp": 0.0001,   # 0.01% taker
        "kraken": 0.0016,       # 0.16% maker
        "coinbase": 0.004,      # 0.40% taker
    })
    # Fixed fraction position size as % of allocated capital per trade (fallback)
    position_size_pct: float = DEFAULT_POSITION_PCT  # 1.5%, was 2%

    def __post_init__(self) -> None:
        valid_modes = {"fixed_fraction", "kelly"}
        if self.position_sizing_mode not in valid_modes:
            raise ValueError(
                f"position_sizing_mode must be one of {valid_modes}, "
                f"got '{self.position_sizing_mode}'"
            )
        if self.kelly_fraction <= 0 or self.kelly_fraction > 1:
            raise ValueError("kelly_fraction must be in (0, 1]")
        if self.max_drawdown_pct <= 0 or self.max_drawdown_pct > 100:
            raise ValueError("max_drawdown_pct must be in (0, 100]")


@dataclass
class OrderSizing:
    """Result of a position sizing calculation."""
    size_usd: float          # notional size in USD
    size_base: float         # size in base currency (size_usd / price if price supplied)
    exchange: str
    fee_usd: float           # estimated fee for this order (one-way)
    net_cost_usd: float      # size_usd + fee_usd (total capital at risk)
    is_viable: bool          # True if expected profit > fee
    reason: str              # explanation of viability decision
    expected_profit_bps: float  # spread_bps - round_trip_fee_bps

    def to_dict(self) -> Dict:
        return {
            "size_usd": round(self.size_usd, 6),
            "size_base": round(self.size_base, 8),
            "exchange": self.exchange,
            "fee_usd": round(self.fee_usd, 6),
            "net_cost_usd": round(self.net_cost_usd, 6),
            "is_viable": self.is_viable,
            "reason": self.reason,
            "expected_profit_bps": round(self.expected_profit_bps, 2),
        }


@dataclass
class PreTradeCheck:
    """Result of a pre-trade risk validation."""
    allowed: bool
    reason: str
    risk_utilisation_pct: float   # how much of risk budget is consumed (0–100)

    def to_dict(self) -> Dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "risk_utilisation_pct": round(self.risk_utilisation_pct, 2),
        }


@dataclass
class _FillRecord:
    """Internal record of a fill for PnL tracking."""
    symbol: str
    exchange: str
    side: str        # "buy" or "sell"
    size: float
    price: float
    fee_usd: float
    timestamp_ns: int


@dataclass
class _PositionRecord:
    """Internal position state for a symbol."""
    symbol: str
    exchange: str
    size_usd: float      # signed: positive = long, negative = short
    avg_entry: float
    fee_paid_usd: float

    @property
    def abs_size_usd(self) -> float:
        return abs(self.size_usd)


# ---------------------------------------------------------------------------
# MicroRiskEnvelope
# ---------------------------------------------------------------------------

class MicroRiskEnvelope:
    """
    Risk management layer for $620 USD (~$1000 AUD) micro-capital HFT.

    Optimized MICRO-tier enforcements:
      - Per-pair position limits: 6% of capital ($37.20 max per pair)
      - Maximum concurrent open pairs: 2 (caps portfolio heat at 12%)
      - Daily loss hard stop: $6.20 USD (1% of capital)
      - Global drawdown halt: 12% ($74.40 loss from peak)
      - Fee-aware position viability (reject if fee > 50% of expected profit)
      - Exchange minimum order sizes
      - Kelly sizing (adapts after 10 fills; falls back to 1.5% fixed fraction)
      - Preferred exchange routing: Bybit spot/perp > Kraken > Coinbase (avoid)

    Thread-safety: NOT thread-safe. Use asyncio.Lock in async contexts.
    """

    def __init__(self, config: Optional[MicroRiskConfig] = None) -> None:
        self._cfg = config or MicroRiskConfig()
        self._daily_pnl: float = 0.0
        self._total_realised_pnl: float = 0.0
        self._total_fees_paid: float = 0.0
        self._fills: List[_FillRecord] = []
        self._day_reset_ns: int = self._next_midnight_ns()
        self._halted: bool = False

        # Positions: symbol -> _PositionRecord
        self._positions: Dict[str, _PositionRecord] = {}

        # Kelly win-rate tracker per symbol: symbol -> {wins, losses, total_win, total_loss}
        self._kelly_stats: Dict[str, Dict] = defaultdict(lambda: {
            "wins": 0,
            "losses": 0,
            "total_win": 0.0,
            "total_loss": 0.0,
        })

        logger.info(
            "MicroRiskEnvelope initialised: capital=$%.2f max_dd=%.1f%% "
            "daily_limit=$%.2f mode=%s max_pairs=%d per_pair_pct=%.1f%%",
            self._cfg.total_capital_usd,
            self._cfg.max_drawdown_pct,
            self._cfg.max_daily_loss_usd,
            self._cfg.position_sizing_mode,
            self._cfg.max_open_pairs,
            self._cfg.max_position_per_pair_pct,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_order_size(
        self,
        symbol: str,
        exchange: str,
        capital_allocated: float,
        signal_confidence: float = 1.0,
        spread_bps: float = 0.0,
        current_price: float = 1.0,
    ) -> OrderSizing:
        """
        Calculate fee-aware order sizing for a trade.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            exchange: Exchange key from fee_rates (e.g., "bybit_spot")
            capital_allocated: USD capital dedicated to this strategy/pair
            signal_confidence: [0, 1] confidence multiplier (1.0 = full size)
            spread_bps: Expected spread capture in basis points (for viability check).
                        Pass 0.0 for funding-rate strategies (not spread-based).
            current_price: Current asset price for base-currency sizing

        Returns:
            OrderSizing with size, fees, viability, and reason.
        """
        signal_confidence = max(0.0, min(1.0, signal_confidence))

        # Total fills across all symbols for Kelly readiness check
        total_fills = sum(
            s["wins"] + s["losses"] for s in self._kelly_stats.values()
        )

        if self._cfg.position_sizing_mode == "kelly" and total_fills >= _KELLY_MIN_FILLS:
            raw_size_usd = self._kelly_size(symbol, capital_allocated, signal_confidence)
        else:
            # Fixed fraction fallback (also used before 10 fills even in kelly mode)
            raw_size_usd = (
                capital_allocated
                * self._cfg.position_size_pct
                * signal_confidence
            )

        # Clamp to max position per pair
        max_position_usd = (
            self._cfg.total_capital_usd
            * self._cfg.max_position_per_pair_pct
            / 100.0
        )
        size_usd = min(raw_size_usd, max_position_usd)

        # Enforce exchange minimum
        min_order = EXCHANGE_MIN_ORDER_USD.get(exchange, 1.0)

        # Fee calculation (one-way entry fee)
        fee_rate = self._cfg.fee_rates.get(exchange, 0.004)
        fee_usd = size_usd * fee_rate
        net_cost_usd = size_usd + fee_usd

        # Round-trip fee for viability check
        rt_fee_bps = EXCHANGE_ROUND_TRIP_FEES_BPS.get(exchange, 80.0)
        rt_fee_usd = size_usd * rt_fee_bps / 10_000.0  # bps → fraction

        # Expected profit = spread capture − round-trip fee
        spread_usd = size_usd * spread_bps / 10_000.0
        expected_profit_bps = spread_bps - rt_fee_bps
        expected_profit_usd = spread_usd - rt_fee_usd

        # Viability checks
        if size_usd < min_order:
            return OrderSizing(
                size_usd=size_usd,
                size_base=size_usd / max(current_price, 1e-12),
                exchange=exchange,
                fee_usd=fee_usd,
                net_cost_usd=net_cost_usd,
                is_viable=False,
                reason=(
                    f"size_usd={size_usd:.4f} < exchange_minimum={min_order:.2f} "
                    f"for {exchange}"
                ),
                expected_profit_bps=expected_profit_bps,
            )

        if spread_bps > 0 and rt_fee_usd > 0.5 * spread_usd:
            # Fee would eat > 50% of expected spread capture → reject
            return OrderSizing(
                size_usd=size_usd,
                size_base=size_usd / max(current_price, 1e-12),
                exchange=exchange,
                fee_usd=fee_usd,
                net_cost_usd=net_cost_usd,
                is_viable=False,
                reason=(
                    f"fee={rt_fee_usd:.6f} > 50% of spread={spread_usd:.6f} "
                    f"on {exchange} (rt_fee={rt_fee_bps:.1f}bps, spread={spread_bps:.1f}bps)"
                ),
                expected_profit_bps=expected_profit_bps,
            )

        if spread_bps > 0 and expected_profit_bps < 0:
            return OrderSizing(
                size_usd=size_usd,
                size_base=size_usd / max(current_price, 1e-12),
                exchange=exchange,
                fee_usd=fee_usd,
                net_cost_usd=net_cost_usd,
                is_viable=False,
                reason=(
                    f"expected_profit_bps={expected_profit_bps:.2f} < 0 "
                    f"(spread={spread_bps:.1f}bps < rt_fee={rt_fee_bps:.1f}bps)"
                ),
                expected_profit_bps=expected_profit_bps,
            )

        return OrderSizing(
            size_usd=size_usd,
            size_base=size_usd / max(current_price, 1e-12),
            exchange=exchange,
            fee_usd=fee_usd,
            net_cost_usd=net_cost_usd,
            is_viable=True,
            reason=f"viable: profit≈{expected_profit_bps:.1f}bps on {exchange}",
            expected_profit_bps=expected_profit_bps,
        )

    def check_pre_trade(
        self,
        symbol: str,
        exchange: str,
        side: str,
        size_usd: float,
        current_positions: Optional[Dict[str, float]] = None,
    ) -> PreTradeCheck:
        """
        Validate a proposed trade against all risk limits.

        Args:
            symbol: Trading pair
            exchange: Exchange key
            side: "buy" or "sell"
            size_usd: Proposed notional size in USD
            current_positions: Dict of symbol→size_usd for existing positions
                              (uses internal positions if None)

        Returns:
            PreTradeCheck with allowed/denied and reason.
        """
        # Refresh daily reset if needed
        self._maybe_daily_reset()

        # 1. Halted state
        if self._halted:
            return PreTradeCheck(
                allowed=False,
                reason="system_halted: global_killswitch_active",
                risk_utilisation_pct=100.0,
            )

        # 2. Global drawdown check
        if self._is_drawdown_exceeded():
            self._halted = True
            return PreTradeCheck(
                allowed=False,
                reason=(
                    f"drawdown_halt: total_pnl={self._total_realised_pnl:.4f} "
                    f"exceeds max_drawdown={self._cfg.max_drawdown_pct}%"
                ),
                risk_utilisation_pct=100.0,
            )

        # 3. Daily loss hard stop
        if self._daily_pnl <= -self._cfg.max_daily_loss_usd:
            return PreTradeCheck(
                allowed=False,
                reason=(
                    f"daily_loss_halt: daily_pnl={self._daily_pnl:.4f} <= "
                    f"-{self._cfg.max_daily_loss_usd:.2f} USD"
                ),
                risk_utilisation_pct=100.0,
            )

        # 4. Per-pair position limit
        max_pair_usd = (
            self._cfg.total_capital_usd * self._cfg.max_position_per_pair_pct / 100.0
        )

        positions = current_positions or {
            sym: pos.size_usd for sym, pos in self._positions.items()
        }

        existing_size = abs(positions.get(symbol, 0.0))
        new_total_size = existing_size + size_usd

        if new_total_size > max_pair_usd:
            risk_util = (new_total_size / max_pair_usd) * 100.0
            return PreTradeCheck(
                allowed=False,
                reason=(
                    f"position_limit_exceeded: {symbol} new_total={new_total_size:.2f} > "
                    f"max={max_pair_usd:.2f} USD ({self._cfg.max_position_per_pair_pct}%)"
                ),
                risk_utilisation_pct=min(risk_util, 100.0),
            )

        # 5. Max open pairs check (only if opening a new position)
        open_pairs = set(positions.keys())
        is_new_pair = symbol not in open_pairs or positions.get(symbol, 0.0) == 0.0

        if is_new_pair and len(open_pairs) >= self._cfg.max_open_pairs:
            return PreTradeCheck(
                allowed=False,
                reason=(
                    f"max_pairs_exceeded: {len(open_pairs)}/{self._cfg.max_open_pairs} "
                    f"pairs open, cannot add {symbol}"
                ),
                risk_utilisation_pct=100.0,
            )

        # Compute risk utilisation
        total_exposure = sum(abs(v) for v in positions.values()) + size_usd
        max_exposure = (
            self._cfg.total_capital_usd
            * (1.0 - 0.05)  # minus 5% reserve
        )
        risk_util = min((total_exposure / max(max_exposure, 1.0)) * 100.0, 100.0)

        return PreTradeCheck(
            allowed=True,
            reason=f"pre_trade_ok: {symbol} {side} ${size_usd:.2f} on {exchange}",
            risk_utilisation_pct=risk_util,
        )

    def record_fill(
        self,
        symbol: str,
        exchange: str,
        side: str,
        size: float,
        price: float,
        fee_usd: float,
        timestamp_ns: int,
    ) -> None:
        """
        Record a completed fill. Updates daily PnL, positions, and Kelly stats.

        Args:
            symbol: Trading pair
            exchange: Exchange key
            side: "buy" or "sell"
            size: Base currency quantity
            price: Fill price in USD
            fee_usd: Actual fee charged in USD
            timestamp_ns: Nanosecond timestamp
        """
        self._maybe_daily_reset()

        notional_usd = size * price
        fill = _FillRecord(
            symbol=symbol,
            exchange=exchange,
            side=side,
            size=size,
            price=price,
            fee_usd=fee_usd,
            timestamp_ns=timestamp_ns,
        )
        self._fills.append(fill)
        self._total_fees_paid += fee_usd

        # Update position
        if symbol not in self._positions:
            self._positions[symbol] = _PositionRecord(
                symbol=symbol,
                exchange=exchange,
                size_usd=0.0,
                avg_entry=price,
                fee_paid_usd=0.0,
            )

        pos = self._positions[symbol]
        pos.fee_paid_usd += fee_usd

        if side == "buy":
            # Long entry or short cover
            if pos.size_usd < 0:
                # Covering short: realise PnL
                cover_size = min(abs(pos.size_usd), notional_usd)
                realised = (pos.avg_entry - price) * (cover_size / max(price, 1e-12))
                realised -= fee_usd
                self._daily_pnl += realised
                self._total_realised_pnl += realised
                self._update_kelly_stats(symbol, realised)
                pos.size_usd += notional_usd
            else:
                # Adding to long
                total = pos.size_usd + notional_usd
                pos.avg_entry = (
                    (pos.avg_entry * pos.size_usd + price * notional_usd) / total
                    if total > 0 else price
                )
                pos.size_usd = total
                self._daily_pnl -= fee_usd
                self._total_realised_pnl -= fee_usd
        else:  # sell
            # Short entry or long exit
            if pos.size_usd > 0:
                # Closing long: realise PnL
                exit_size = min(pos.size_usd, notional_usd)
                realised = (price - pos.avg_entry) * (exit_size / max(pos.avg_entry, 1e-12))
                realised -= fee_usd
                self._daily_pnl += realised
                self._total_realised_pnl += realised
                self._update_kelly_stats(symbol, realised)
                pos.size_usd -= exit_size
            else:
                # Adding to short
                total = pos.size_usd - notional_usd
                pos.avg_entry = (
                    (pos.avg_entry * abs(pos.size_usd) + price * notional_usd)
                    / (abs(pos.size_usd) + notional_usd)
                    if abs(pos.size_usd) + notional_usd > 0 else price
                )
                pos.size_usd = total
                self._daily_pnl -= fee_usd
                self._total_realised_pnl -= fee_usd

        # Clean up zero positions
        if abs(pos.size_usd) < 1e-8:
            del self._positions[symbol]

        logger.debug(
            "Fill recorded: %s %s %.6f@%.4f fee=%.6f daily_pnl=%.4f",
            side.upper(), symbol, size, price, fee_usd, self._daily_pnl,
        )

    def get_daily_pnl(self) -> float:
        """Return today's realised PnL in USD."""
        self._maybe_daily_reset()
        return self._daily_pnl

    def get_risk_status(self) -> Dict:
        """
        Return comprehensive risk dashboard.

        Status levels (MICRO-tier tightened thresholds):
          green  — drawdown < 7%
          yellow — 7% ≤ drawdown < 10%
          red    — 10% ≤ drawdown < 12%
          halted — drawdown ≥ 12% OR daily loss limit hit
        """
        self._maybe_daily_reset()

        total_capital = self._cfg.total_capital_usd
        drawdown_pct = (
            abs(self._total_realised_pnl) / total_capital * 100.0
            if self._total_realised_pnl < 0 else 0.0
        )

        # Determine status
        daily_halted = self._daily_pnl <= -self._cfg.max_daily_loss_usd
        if self._halted or drawdown_pct >= _HALT_THRESHOLD or daily_halted:
            status = "halted"
        elif drawdown_pct >= _RED_THRESHOLD:
            status = "red"
        elif drawdown_pct >= _YELLOW_THRESHOLD:
            status = "yellow"
        else:
            status = "green"

        open_pairs = list(self._positions.keys())
        total_exposure_usd = sum(abs(p.size_usd) for p in self._positions.values())
        daily_limit_remaining = max(
            0.0, self._cfg.max_daily_loss_usd + self._daily_pnl
        )

        # Kelly readiness
        total_fills = sum(
            s["wins"] + s["losses"] for s in self._kelly_stats.values()
        )
        kelly_active = (
            self._cfg.position_sizing_mode == "kelly"
            and total_fills >= _KELLY_MIN_FILLS
        )

        return {
            "status": status,
            "daily_pnl": round(self._daily_pnl, 4),
            "daily_limit_usd": self._cfg.max_daily_loss_usd,
            "daily_limit_remaining": round(daily_limit_remaining, 4),
            "total_realised_pnl": round(self._total_realised_pnl, 4),
            "total_fees_paid": round(self._total_fees_paid, 4),
            "drawdown_pct": round(drawdown_pct, 4),
            "max_drawdown_pct": self._cfg.max_drawdown_pct,
            "open_positions": len(self._positions),
            "open_pairs": open_pairs,
            "total_exposure_usd": round(total_exposure_usd, 4),
            "halted": self._halted or daily_halted,
            "kelly_active": kelly_active,
            "total_fills": total_fills,
            "kelly_fills_needed": max(0, _KELLY_MIN_FILLS - total_fills),
        }

    def reset_daily(self) -> None:
        """
        Reset daily loss counter. Called at UTC midnight.
        Also unhalts if only daily limit was triggered (global drawdown remains).
        """
        logger.info(
            "Daily reset: daily_pnl=%.4f → 0.0 (total_realised=%.4f)",
            self._daily_pnl, self._total_realised_pnl,
        )
        self._daily_pnl = 0.0
        self._fills = [f for f in self._fills if not self._is_today(f.timestamp_ns)]

        # Re-check global drawdown to decide if we remain halted
        if self._is_drawdown_exceeded():
            logger.warning("Daily reset: global drawdown still exceeded — system remains halted")
        else:
            self._halted = False

        self._day_reset_ns = self._next_midnight_ns()

    def min_spread_for_profit(self, exchange: str) -> float:
        """
        Return the minimum spread in basis points to break even after fees.

        Returns:
            Minimum spread in bps (round-trip). Zero means any spread is profitable.
        """
        return EXCHANGE_ROUND_TRIP_FEES_BPS.get(exchange, 80.0)

    # ------------------------------------------------------------------
    # Kelly helpers
    # ------------------------------------------------------------------

    def _kelly_size(
        self,
        symbol: str,
        capital_allocated: float,
        signal_confidence: float,
    ) -> float:
        """
        Compute Kelly-optimal fraction of allocated capital.

        f* = (p × b - q) / b
        where b = avg_win / avg_loss, p = win_rate, q = 1 - p

        Capped at kelly_fraction (0.20 at MICRO tier).
        Falls back to prior statistics until _KELLY_MIN_FILLS fills are recorded.
        """
        stats = self._kelly_stats[symbol]
        wins = stats["wins"]
        losses = stats["losses"]
        total_win = stats["total_win"]
        total_loss = stats["total_loss"]

        # Fall back to priors if insufficient data
        if wins + losses < _KELLY_MIN_FILLS:
            win_rate = _DEFAULT_WIN_RATE
            avg_win = _DEFAULT_AVG_WIN
            avg_loss = _DEFAULT_AVG_LOSS
        else:
            win_rate = wins / (wins + losses)
            avg_win = total_win / wins if wins > 0 else _DEFAULT_AVG_WIN
            avg_loss = total_loss / losses if losses > 0 else _DEFAULT_AVG_LOSS

        avg_loss = max(avg_loss, 1e-8)  # avoid division by zero
        b = avg_win / avg_loss
        q = 1.0 - win_rate

        kelly_f = (win_rate * b - q) / b if b > 0 else 0.0
        kelly_f = max(0.0, kelly_f)  # never negative (never bet against self)
        kelly_f = min(kelly_f, self._cfg.kelly_fraction)  # cap at MICRO kelly_fraction

        # Apply signal confidence
        kelly_f *= signal_confidence

        return capital_allocated * kelly_f

    def _update_kelly_stats(self, symbol: str, pnl: float) -> None:
        """Update win/loss statistics for Kelly sizing."""
        stats = self._kelly_stats[symbol]
        if pnl > 0:
            stats["wins"] += 1
            stats["total_win"] += pnl
        elif pnl < 0:
            stats["losses"] += 1
            stats["total_loss"] += abs(pnl)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_drawdown_exceeded(self) -> bool:
        """Return True if total drawdown exceeds max_drawdown_pct."""
        if self._total_realised_pnl >= 0:
            return False
        drawdown_pct = abs(self._total_realised_pnl) / self._cfg.total_capital_usd * 100.0
        return drawdown_pct >= self._cfg.max_drawdown_pct

    def _maybe_daily_reset(self) -> None:
        """Auto-reset daily PnL at UTC midnight."""
        now_ns = time.time_ns()
        if now_ns >= self._day_reset_ns:
            self.reset_daily()

    @staticmethod
    def _next_midnight_ns() -> int:
        """Return nanosecond timestamp of the next UTC midnight."""
        import datetime
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        next_midnight = (now_utc + datetime.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return int(next_midnight.timestamp() * NS_PER_SECOND)

    @staticmethod
    def _is_today(timestamp_ns: int) -> bool:
        """Return True if timestamp_ns is from today (UTC)."""
        import datetime
        ts = datetime.datetime.fromtimestamp(
            timestamp_ns / NS_PER_SECOND, tz=datetime.timezone.utc
        )
        now = datetime.datetime.now(datetime.timezone.utc)
        return ts.date() == now.date()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        s = self.get_risk_status()
        return (
            f"MicroRiskEnvelope("
            f"status={s['status']}, "
            f"daily_pnl={s['daily_pnl']:.4f}, "
            f"drawdown={s['drawdown_pct']:.2f}%, "
            f"open_pairs={s['open_positions']}, "
            f"kelly_active={s['kelly_active']})"
        )
