"""Execution pipeline — Batch 1 upgrade: maker-only limit orders with
fill-or-cancel monitoring and resting order management.

All previous validation logic (TTL, confidence, stop-loss, sizing, caps)
is preserved verbatim.  The new behaviour activates in the submit step:

  1. A limit BUY is posted at bid+1-tick (inside spread) → maker fee 0.16%.
  2. A background task monitors fill status for up to `maker_fill_timeout_s`.
  3. If unfilled, the order is cancelled and the result is marked as rejected
     so run_ultimate.py can retry next cycle.
  4. dry_run=True skips all exchange calls as before.

Batch 2 upgrade: SmallCapitalManager + KellyPositionSizer layered sizing.

  - SmallCapitalManager is the PRIMARY gate: fee-aware, bandit-adapted.
    If it returns pos_usd==0, the trade is skipped immediately.
  - KellyPositionSizer is the SECONDARY narrowing gate: takes the
    conservative minimum of SCM and Kelly sizes.
  - Both sizers receive post-fill feedback to close the learning loop.
  - Legacy position_sizer kwarg still accepted as fallback when neither
    SCM nor Kelly is supplied.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from argus_live.execution.router_policy import RouteDecision, select_route
from argus_live.execution.slippage_model import SlippageEstimate, estimate_slippage
from argus_live.risk.liquidity_gate import LiquidityDecision, apply_liquidity_haircut

logger = logging.getLogger(__name__)

# Maker tick: post limit 0.01% inside best bid to qualify for maker rebate
MAKER_INSIDE_SPREAD_PCT = 0.0001
# Maximum seconds to wait for a maker fill before cancelling
DEFAULT_MAKER_FILL_TIMEOUT_S = 30.0
# Poll interval when monitoring resting order
FILL_POLL_INTERVAL_S = 1.0


@dataclass(frozen=True)
class ExecutionPipelineInput:
    symbol: str
    quantity: float
    reference_price: float
    top_of_book_notional: float
    spread_bps: float
    volatility_bps: float
    allow_market_orders: bool = False


@dataclass(frozen=True)
class ExecutionPipelinePlan:
    liquidity: LiquidityDecision
    route: RouteDecision
    slippage: SlippageEstimate


@dataclass
class PipelineConfig:
    """Runtime limits for ExecutionPipeline.execute_signal()."""
    min_confidence: float = 0.50
    min_strength: float = 0.30
    max_position_value_aud: float = 25_000.0
    max_portfolio_risk_pct: float = 0.05
    require_stop_loss: bool = True
    signal_ttl_seconds: float = 30.0
    dry_run: bool = True
    # Maker-only settings
    maker_only: bool = True
    maker_fill_timeout_s: float = DEFAULT_MAKER_FILL_TIMEOUT_S
    # SCM config path — set to None to skip SmallCapitalManager
    scm_config_path: Optional[str] = "config/small_capital_config.yaml"


@dataclass
class ExecutionResult:
    """Result returned by execute_signal()."""
    success: bool
    signal: Any
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    cost: float = 0.0
    fee: float = 0.0
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    venue_order_id: Optional[str] = None
    reason: str = ""
    maker_order: bool = False


class ExecutionPipeline:
    """Signal-to-order execution pipeline with maker-only order logic."""

    def __init__(
        self,
        exchange: Any,
        config: Optional[PipelineConfig] = None,
        position_sizer: Optional[Any] = None,
        venue_adapter: Optional[Any] = None,
        small_capital_manager: Optional[Any] = None,
        kelly_sizer: Optional[Any] = None,
    ) -> None:
        self.exchange = exchange
        self.config = config or PipelineConfig()
        self.position_sizer = position_sizer
        self.venue_adapter = venue_adapter

        # --- Layered sizing: SCM (primary) + Kelly (secondary) ---
        # Prefer explicitly passed instances; fall back to auto-init from config.
        self._scm = small_capital_manager
        self._kelly = kelly_sizer

        if self._scm is None and self.config.scm_config_path is not None:
            try:
                from risk.small_capital_manager import SmallCapitalManager
                self._scm = SmallCapitalManager(self.config.scm_config_path)
                logger.info("SmallCapitalManager initialised from %s", self.config.scm_config_path)
            except Exception as exc:
                logger.warning("SmallCapitalManager init failed — disabled: %s", exc)

        if self._kelly is None:
            try:
                from risk.kelly_position_sizer import KellyPositionSizer
                self._kelly = KellyPositionSizer()
                logger.info("KellyPositionSizer initialised")
            except Exception as exc:
                logger.warning("KellyPositionSizer init failed — disabled: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _fetch_best_bid(self, symbol: str) -> Optional[float]:
        """Fetch best bid price for maker order placement."""
        try:
            ob = await self.exchange.fetch_order_book(symbol, limit=5)
            bids = ob.get("bids", [])
            if bids:
                return float(bids[0][0])
        except Exception as exc:
            logger.warning("fetch_order_book failed for %s: %s", symbol, exc)
        return None

    async def _place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        limit_price: float,
    ) -> Optional[str]:
        """Place a limit order on Kraken. Returns order id or None."""
        try:
            params = {"postOnly": True}  # Kraken: reject if would trade as taker
            order = await self.exchange.create_limit_buy_order(
                symbol, quantity, limit_price, params
            ) if side == "buy" else await self.exchange.create_limit_sell_order(
                symbol, quantity, limit_price, params
            )
            return order.get("id")
        except Exception as exc:
            logger.warning("create_limit_order failed %s %s: %s", side, symbol, exc)
            return None

    async def _monitor_fill(
        self,
        symbol: str,
        order_id: str,
        timeout_s: float,
    ) -> tuple[bool, float, float]:
        """
        Poll order status until filled or timeout.
        Returns (filled: bool, filled_qty: float, avg_price: float).
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                order = await self.exchange.fetch_order(order_id, symbol)
                status = str(order.get("status", "")).lower()
                if status == "closed":  # fully filled
                    return True, float(order.get("filled", 0)), float(order.get("average", 0))
                if status in ("canceled", "cancelled", "expired", "rejected"):
                    return False, 0.0, 0.0
            except Exception as exc:
                logger.debug("fetch_order poll error %s: %s", order_id, exc)
            await asyncio.sleep(FILL_POLL_INTERVAL_S)
        return False, 0.0, 0.0

    async def _cancel_order(self, symbol: str, order_id: str) -> None:
        try:
            await self.exchange.cancel_order(order_id, symbol)
            logger.info("Cancelled resting maker order %s for %s", order_id, symbol)
        except Exception as exc:
            logger.warning("cancel_order failed %s %s: %s", order_id, symbol, exc)

    def _notify_fill(
        self,
        signal: Any,
        entry_price: float,
        filled_price: float,
        filled_qty: float,
        capital: float,
    ) -> None:
        """Feed trade outcome back to SCM bandit and Kelly model."""
        if entry_price <= 0 or filled_qty <= 0:
            return
        pnl_pct = (filled_price - entry_price) / entry_price
        side = str(getattr(signal, "action", "buy")).lower()
        if side in ("sell", "-1", "signalaction.sell"):
            pnl_pct = -pnl_pct

        if self._kelly is not None:
            try:
                self._kelly.record_trade(win=(pnl_pct > 0), pnl_pct=pnl_pct)
                self._kelly.update_capital(capital)
            except Exception as exc:
                logger.debug("Kelly feedback error: %s", exc)

        if self._scm is not None:
            strategy_id = getattr(signal, "strategy_id", getattr(signal, "source", "default"))
            try:
                self._scm.update_bandit(strategy_id=strategy_id, reward_pct=pnl_pct)
            except Exception as exc:
                logger.debug("SCM bandit feedback error: %s", exc)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def execute_signal(
        self,
        signal: Any,
        capital: float,
    ) -> ExecutionResult:
        cfg = self.config
        now = datetime.now(timezone.utc)

        # 1. Signal TTL
        generated_at = getattr(signal, "generated_at", None) or getattr(signal, "timestamp", None)
        if generated_at is not None:
            age = (now - generated_at).total_seconds()
            if age > cfg.signal_ttl_seconds:
                return ExecutionResult(
                    success=False, signal=signal,
                    reason=f"signal expired (age={age:.1f}s > ttl={cfg.signal_ttl_seconds}s)",
                )

        # 2. Confidence / strength
        confidence = getattr(signal, "confidence", 1.0)
        strength = getattr(signal, "strength", 1.0)
        if confidence < cfg.min_confidence:
            return ExecutionResult(success=False, signal=signal,
                reason=f"confidence {confidence:.3f} < min {cfg.min_confidence}")
        if strength < cfg.min_strength:
            return ExecutionResult(success=False, signal=signal,
                reason=f"strength {strength:.3f} < min {cfg.min_strength}")

        # 3. Stop-loss
        stop_loss = getattr(signal, "stop_loss", None)
        if cfg.require_stop_loss and (stop_loss is None or stop_loss <= 0):
            return ExecutionResult(success=False, signal=signal,
                reason="stop_loss missing or zero (require_stop_loss=True)")

        # 4. Entry price
        entry_price = getattr(signal, "entry_price", None)
        if not entry_price:
            try:
                ticker = await self.exchange.fetch_ticker(signal.symbol)
                entry_price = float(ticker.get("last", 0)) if ticker else 0.0
            except Exception as exc:
                logger.warning("fetch_ticker failed for %s: %s", signal.symbol, exc)
                entry_price = 0.0
        if entry_price <= 0:
            return ExecutionResult(success=False, signal=signal,
                reason="could not determine entry price")

        # 5. Position sizing — layered: SCM (primary) → Kelly (secondary) → legacy fallback
        quantity = 0.0
        final_usd = 0.0

        if self._scm is not None:
            # Gate 1: SmallCapitalManager — fee-aware, bandit-adapted
            try:
                self._scm.update_capital(capital)
                pos_usd, _pos_units = self._scm.fee_aware_position_size(
                    signal_strength=strength,
                    price=entry_price,
                    use_taker=(not cfg.maker_only),
                )
            except Exception as exc:
                logger.warning("SCM sizing error — falling back to legacy: %s", exc)
                pos_usd = 0.0

            if pos_usd <= 0.0:
                return ExecutionResult(success=False, signal=signal,
                    reason="SCM blocked trade — fee drag exceeds expected edge or below min threshold")

            final_usd = pos_usd

            # Gate 2: Kelly narrowing — take conservative minimum
            if self._kelly is not None:
                try:
                    self._kelly.update_capital(capital)
                    kelly_result = self._kelly.position_size(price=entry_price)
                    kelly_usd = getattr(kelly_result, "position_usd", getattr(kelly_result, "usd", pos_usd))
                    if kelly_usd > 0:
                        final_usd = min(final_usd, kelly_usd)
                        logger.debug(
                            "SCM=%.2f USD  Kelly=%.2f USD  final=%.2f USD",
                            pos_usd, kelly_usd, final_usd,
                        )
                except Exception as exc:
                    logger.debug("Kelly sizing error — using SCM size only: %s", exc)

            quantity = final_usd / entry_price if entry_price > 0 else 0.0

        elif self.position_sizer is not None:
            # Legacy path: existing position_sizer injected at construction
            from core.types import MarketRegime
            regime = getattr(signal, "regime", MarketRegime.UNKNOWN)
            size_result = self.position_sizer.calculate_position_size(
                capital=capital,
                entry_price=entry_price,
                stop_loss=stop_loss or (entry_price * 0.97),
                confidence=confidence,
                regime=regime,
            )
            quantity = size_result.quantity

        else:
            # Bare fallback: 2% risk per trade
            risk_aud = capital * 0.02
            stop_dist = abs(entry_price - (stop_loss or entry_price * 0.97))
            quantity = (risk_aud / stop_dist) if stop_dist > 0 else 0.0

        if quantity <= 0:
            return ExecutionResult(success=False, signal=signal,
                reason="position sizer returned zero quantity")

        # 6. Position value cap
        position_value = quantity * entry_price
        if position_value > cfg.max_position_value_aud:
            quantity = cfg.max_position_value_aud / entry_price
            position_value = cfg.max_position_value_aud

        # 7. Latency stamp
        _journey_id: Optional[str] = None
        try:
            from alpha.microstructure import LatencyTelemetry, LatencyStage
            base = signal.symbol.split("/")[0].upper()
            _journey_id = LatencyTelemetry.get_instance().start_journey(base)
            LatencyTelemetry.get_instance().mark(_journey_id, LatencyStage.ORDER_SENT)
        except Exception:
            pass

        side = (
            "buy"
            if str(getattr(signal, "action", "buy")).lower() in ("buy", "1", "signaaction.buy")
            else "sell"
        )

        # 8. Submit — maker-only path vs venue_adapter / dry-run
        success = False
        venue_order_id: Optional[str] = None
        filled_qty = 0.0
        filled_price = 0.0
        reason = ""
        maker_order = False

        if cfg.dry_run:
            logger.info("DRY-RUN order: %s %s x %.6f @ %.2f", side, signal.symbol, quantity, entry_price)
            success = True
            venue_order_id = f"dryrun_{int(time.time() * 1000)}"
            filled_qty = quantity
            filled_price = entry_price
            reason = "dry-run accepted"
            maker_order = cfg.maker_only

        elif self.venue_adapter is not None:
            venue_result = self.venue_adapter.submit_limit_order(
                symbol=signal.symbol, side=side,
                quantity=quantity, price=entry_price,
            )
            success = venue_result.success
            venue_order_id = venue_result.venue_order_id
            filled_qty = quantity if success else 0.0
            filled_price = entry_price if success else 0.0
            reason = venue_result.message

        elif cfg.maker_only:
            # Real maker-only: post limit inside spread, monitor fill, cancel if stale
            best_bid = await self._fetch_best_bid(signal.symbol)
            if best_bid and best_bid > 0:
                limit_price = round(best_bid * (1 + MAKER_INSIDE_SPREAD_PCT), 8)
            else:
                limit_price = entry_price

            order_id = await self._place_limit_order(signal.symbol, side, quantity, limit_price)
            if order_id:
                maker_order = True
                logger.info(
                    "Maker order placed: %s %s qty=%.6f @ %.8f  id=%s",
                    side, signal.symbol, quantity, limit_price, order_id,
                )
                filled, filled_qty, filled_price = await self._monitor_fill(
                    signal.symbol, order_id, cfg.maker_fill_timeout_s
                )
                if filled:
                    success = True
                    venue_order_id = order_id
                    reason = f"maker fill @ {filled_price:.8f}"
                else:
                    await self._cancel_order(signal.symbol, order_id)
                    success = False
                    reason = f"maker order unfilled after {cfg.maker_fill_timeout_s:.0f}s — cancelled"
            else:
                success = False
                reason = "failed to place maker limit order"
        else:
            # Fallback market order
            try:
                order = await self.exchange.create_market_buy_order(
                    signal.symbol, quantity
                ) if side == "buy" else await self.exchange.create_market_sell_order(
                    signal.symbol, quantity
                )
                success = True
                venue_order_id = order.get("id")
                filled_qty = quantity
                filled_price = entry_price
                reason = "market order submitted"
            except Exception as exc:
                success = False
                reason = f"market order failed: {exc}"

        # 9. Post-fill feedback — close the SCM + Kelly learning loop
        if success and filled_qty > 0:
            self._notify_fill(
                signal=signal,
                entry_price=entry_price,
                filled_price=filled_price,
                filled_qty=filled_qty,
                capital=capital,
            )

        # Latency complete
        try:
            if _journey_id is not None:
                from alpha.microstructure import LatencyTelemetry, LatencyStage
                LatencyTelemetry.get_instance().mark(_journey_id, LatencyStage.ORDER_SUBMIT)
                LatencyTelemetry.get_instance().complete_journey(_journey_id)
        except Exception:
            pass

        # 10. Fee — maker 0.16%
        fee = (filled_qty * filled_price) * 0.0016 if success else 0.0

        return ExecutionResult(
            success=success,
            signal=signal,
            filled_quantity=filled_qty,
            filled_price=filled_price,
            cost=filled_qty * filled_price if success else 0.0,
            fee=fee,
            timestamp=now,
            venue_order_id=venue_order_id,
            reason=reason,
            maker_order=maker_order,
        )

    # ------------------------------------------------------------------
    # Low-level plan builder
    # ------------------------------------------------------------------

    def build_plan(self, data: ExecutionPipelineInput) -> ExecutionPipelinePlan:
        liquidity = apply_liquidity_haircut(
            requested_quantity=data.quantity,
            reference_price=data.reference_price,
            top_of_book_notional=data.top_of_book_notional,
            max_book_take_ratio=0.20,
        )
        participation_ratio = 0.0
        if data.top_of_book_notional > 0:
            participation_ratio = liquidity.approved_notional / data.top_of_book_notional
        route = select_route(
            symbol=data.symbol,
            spread_bps=data.spread_bps,
            volatility_bps=data.volatility_bps,
            allow_market_orders=data.allow_market_orders,
        )
        slippage = estimate_slippage(
            spread_bps=data.spread_bps,
            volatility_bps=data.volatility_bps,
            participation_ratio=participation_ratio,
        )
        return ExecutionPipelinePlan(liquidity=liquidity, route=route, slippage=slippage)
