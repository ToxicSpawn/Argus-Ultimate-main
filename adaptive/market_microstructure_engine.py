from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MicrostructureState:
    symbol: str
    spread_bps: float
    rolling_spread_bps: float
    order_book_imbalance: float
    microprice: float
    mid_price: float
    trade_velocity: float
    buy_sell_flow_ratio: float
    liquidity_vacuum_flag: bool
    adverse_selection_risk: float
    microstructure_bias: str  # up | down | neutral
    ts: float


class MarketMicrostructureEngine:
    """Deterministic microstructure feature extraction for runtime-safe use."""

    def __init__(
        self,
        *,
        rolling_window: int = 20,
        vacuum_spread_jump_bps: float = 4.0,
        vacuum_depth_drop_ratio: float = 0.5,
        high_adverse_selection_threshold: float = 0.7,
        use_in_execution_alpha: bool = True,
        use_in_liquidity_risk: bool = True,
    ) -> None:
        self.rolling_window = max(5, int(rolling_window or 20))
        self.vacuum_spread_jump_bps = max(0.01, float(vacuum_spread_jump_bps or 4.0))
        self.vacuum_depth_drop_ratio = max(0.0, min(1.0, float(vacuum_depth_drop_ratio or 0.5)))
        self.high_adverse_selection_threshold = max(0.0, min(1.0, float(high_adverse_selection_threshold or 0.7)))
        self.use_in_execution_alpha = bool(use_in_execution_alpha)
        self.use_in_liquidity_risk = bool(use_in_liquidity_risk)

        self._spread_hist: Dict[str, deque] = {}
        self._depth_hist: Dict[str, deque] = {}
        self._states: Dict[str, MicrostructureState] = {}

    @staticmethod
    def _signal_get(sig: Any, name: str, default: Any = None) -> Any:
        if isinstance(sig, dict):
            return sig.get(name, default)
        return getattr(sig, name, default)

    def _hist(self, row_map: Dict[str, deque], symbol: str) -> deque:
        q = row_map.get(symbol)
        if q is None:
            q = deque(maxlen=self.rolling_window)
            row_map[symbol] = q
        return q

    def update_symbol(
        self,
        *,
        symbol: str,
        best_bid: float,
        best_ask: float,
        bid_size: float,
        ask_size: float,
        spread_bps: Optional[float] = None,
        trade_velocity: float = 0.0,
        buy_sell_flow_ratio: Optional[float] = None,
        ts: Optional[float] = None,
    ) -> MicrostructureState:
        sym = str(symbol or "")
        bid = max(0.0, float(best_bid or 0.0))
        ask = max(0.0, float(best_ask or 0.0))
        bs = max(0.0, float(bid_size or 0.0))
        a_s = max(0.0, float(ask_size or 0.0))

        mid = 0.0
        if bid > 0.0 and ask > 0.0:
            mid = (bid + ask) / 2.0
        elif bid > 0.0:
            mid = bid
        elif ask > 0.0:
            mid = ask

        if spread_bps is None:
            if bid > 0.0 and ask > 0.0 and mid > 0.0:
                spread = ((ask - bid) / max(mid, 1e-9)) * 10_000.0
            else:
                spread = 0.0
        else:
            spread = max(0.0, float(spread_bps or 0.0))

        spread_hist = self._hist(self._spread_hist, sym)
        prev_rolling = float(sum(spread_hist) / len(spread_hist)) if spread_hist else spread
        spread_hist.append(float(spread))
        rolling_spread = float(sum(spread_hist) / len(spread_hist)) if spread_hist else spread

        depth = max(0.0, bs + a_s)
        depth_hist = self._hist(self._depth_hist, sym)
        prev_depth = float(depth_hist[-1]) if depth_hist else depth
        depth_hist.append(float(depth))
        depth_drop_ratio = float(depth / max(prev_depth, 1e-9)) if prev_depth > 0.0 else 1.0

        denom = max(bs + a_s, 1e-9)
        obi = (bs - a_s) / denom

        if denom > 0.0 and bid > 0.0 and ask > 0.0:
            microprice = ((ask * bs) + (bid * a_s)) / denom
        else:
            microprice = mid

        if buy_sell_flow_ratio is None:
            flow = 1.0 + max(-0.9, min(0.9, obi))
        else:
            flow = max(0.0, float(buy_sell_flow_ratio or 0.0))

        spread_jump = float(spread - prev_rolling)
        liquidity_vacuum_flag = bool(
            spread_jump >= self.vacuum_spread_jump_bps
            or depth_drop_ratio <= self.vacuum_depth_drop_ratio
        )

        if mid <= 0.0:
            bias = "neutral"
        else:
            mp_delta_bps = ((microprice - mid) / max(mid, 1e-9)) * 10_000.0
            if mp_delta_bps > 0.5 or obi > 0.1:
                bias = "up"
            elif mp_delta_bps < -0.5 or obi < -0.1:
                bias = "down"
            else:
                bias = "neutral"

        spread_stress = max(0.0, (spread / max(rolling_spread, 0.1)) - 1.0)
        depth_stress = max(0.0, 1.0 - max(0.0, min(1.0, depth_drop_ratio)))
        risk = (
            0.40 * min(1.0, abs(obi))
            + 0.30 * min(1.0, spread_stress)
            + 0.20 * min(1.0, depth_stress)
            + (0.20 if liquidity_vacuum_flag else 0.0)
        )
        adverse_selection_risk = max(0.0, min(1.0, float(risk)))

        state = MicrostructureState(
            symbol=sym,
            spread_bps=float(spread),
            rolling_spread_bps=float(rolling_spread),
            order_book_imbalance=float(obi),
            microprice=float(microprice),
            mid_price=float(mid),
            trade_velocity=max(0.0, float(trade_velocity or 0.0)),
            buy_sell_flow_ratio=float(flow),
            liquidity_vacuum_flag=bool(liquidity_vacuum_flag),
            adverse_selection_risk=float(adverse_selection_risk),
            microstructure_bias=str(bias),
            ts=float(ts if ts is not None else time.time()),
        )
        self._states[sym] = state
        return state

    def update_from_signals(self, signals: Iterable[Any]) -> Dict[str, MicrostructureState]:
        for sig in list(signals or []):
            symbol = str(self._signal_get(sig, "symbol", "") or "")
            if not symbol:
                continue
            best_bid = float(
                self._signal_get(sig, "best_bid", None)
                or self._signal_get(sig, "bid", None)
                or self._signal_get(sig, "entry_price", 0.0)
                or self._signal_get(sig, "price", 0.0)
                or 0.0
            )
            best_ask = float(
                self._signal_get(sig, "best_ask", None)
                or self._signal_get(sig, "ask", None)
                or self._signal_get(sig, "entry_price", 0.0)
                or self._signal_get(sig, "price", 0.0)
                or 0.0
            )
            bid_size = float(
                self._signal_get(sig, "top_of_book_bid_size", None)
                or self._signal_get(sig, "bid_size_1", None)
                or self._signal_get(sig, "bid_size", 0.0)
                or 0.0
            )
            ask_size = float(
                self._signal_get(sig, "top_of_book_ask_size", None)
                or self._signal_get(sig, "ask_size_1", None)
                or self._signal_get(sig, "ask_size", 0.0)
                or 0.0
            )
            spread_bps = self._signal_get(sig, "spread_bps", None)
            trade_velocity = float(self._signal_get(sig, "trade_velocity", 0.0) or 0.0)
            flow_ratio = self._signal_get(sig, "buy_sell_flow_ratio", None)
            state = self.update_symbol(
                symbol=symbol,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_size=bid_size,
                ask_size=ask_size,
                spread_bps=float(spread_bps) if spread_bps is not None else None,
                trade_velocity=trade_velocity,
                buy_sell_flow_ratio=(float(flow_ratio) if flow_ratio is not None else None),
            )
            logger.info("microstructure updated for %s", symbol)
            if state.liquidity_vacuum_flag:
                logger.info("liquidity vacuum detected for %s", symbol)
        return dict(self._states)

    def annotate_signals(self, signals: Iterable[Any]) -> List[Any]:
        out: List[Any] = []
        for sig in list(signals or []):
            symbol = str(self._signal_get(sig, "symbol", "") or "")
            state = self._states.get(symbol)
            if state is None:
                out.append(sig)
                continue
            mid = max(state.mid_price, 1e-9)
            microprice_drift_bps = ((state.microprice - state.mid_price) / mid) * 10_000.0
            adverse_high = state.adverse_selection_risk >= self.high_adverse_selection_threshold
            if isinstance(sig, dict):
                sig["spread_bps"] = float(state.spread_bps)
                sig["order_book_imbalance"] = float(state.order_book_imbalance)
                sig["depth_imbalance"] = float(state.order_book_imbalance)
                sig["microprice"] = float(state.microprice)
                sig["trade_velocity"] = float(state.trade_velocity)
                sig["liquidity_vacuum_flag"] = bool(state.liquidity_vacuum_flag)
                sig["adverse_selection_risk"] = float(state.adverse_selection_risk)
                sig["microstructure_bias"] = str(state.microstructure_bias)
                sig["buy_sell_flow_ratio"] = float(state.buy_sell_flow_ratio)
                sig["microprice_drift_bps"] = float(microprice_drift_bps)
                if self.use_in_execution_alpha:
                    sig["adverse_selection_detected"] = bool(adverse_high)
                    if state.liquidity_vacuum_flag:
                        sig["liquidity_quality"] = min(1.0, float(sig.get("liquidity_quality", 1.0) or 1.0) * 0.75)
                    if adverse_high:
                        sig["liquidity_quality"] = min(1.0, float(sig.get("liquidity_quality", 1.0) or 1.0) * 0.85)
                        logger.info("adverse selection elevated for maker routing on %s", symbol)
                if self.use_in_liquidity_risk:
                    bid_sz = float(sig.get("top_of_book_bid_size", sig.get("bid_size_1", sig.get("bid_size", 0.0))) or 0.0)
                    ask_sz = float(sig.get("top_of_book_ask_size", sig.get("ask_size_1", sig.get("ask_size", 0.0))) or 0.0)
                    depth_est = max(0.0, bid_sz + ask_sz)
                    sig["orderbook_depth_estimate"] = float(
                        sig.get("orderbook_depth_estimate", 0.0) or depth_est
                    )
            else:
                setattr(sig, "spread_bps", float(state.spread_bps))
                setattr(sig, "order_book_imbalance", float(state.order_book_imbalance))
                setattr(sig, "depth_imbalance", float(state.order_book_imbalance))
                setattr(sig, "microprice", float(state.microprice))
                setattr(sig, "trade_velocity", float(state.trade_velocity))
                setattr(sig, "liquidity_vacuum_flag", bool(state.liquidity_vacuum_flag))
                setattr(sig, "adverse_selection_risk", float(state.adverse_selection_risk))
                setattr(sig, "microstructure_bias", str(state.microstructure_bias))
                setattr(sig, "buy_sell_flow_ratio", float(state.buy_sell_flow_ratio))
                setattr(sig, "microprice_drift_bps", float(microprice_drift_bps))
                if self.use_in_execution_alpha:
                    setattr(sig, "adverse_selection_detected", bool(adverse_high))
                    liq_q = float(getattr(sig, "liquidity_quality", 1.0) or 1.0)
                    if state.liquidity_vacuum_flag:
                        liq_q *= 0.75
                    if adverse_high:
                        liq_q *= 0.85
                        logger.info("adverse selection elevated for maker routing on %s", symbol)
                    setattr(sig, "liquidity_quality", min(1.0, max(0.0, liq_q)))
                if self.use_in_liquidity_risk:
                    if float(getattr(sig, "orderbook_depth_estimate", 0.0) or 0.0) <= 0.0:
                        bid_sz = float(
                            getattr(sig, "top_of_book_bid_size", None)
                            or getattr(sig, "bid_size_1", None)
                            or getattr(sig, "bid_size", 0.0)
                            or 0.0
                        )
                        ask_sz = float(
                            getattr(sig, "top_of_book_ask_size", None)
                            or getattr(sig, "ask_size_1", None)
                            or getattr(sig, "ask_size", 0.0)
                            or 0.0
                        )
                        setattr(sig, "orderbook_depth_estimate", max(0.0, bid_sz + ask_sz))
            out.append(sig)
        return out

    def state_for_symbol(self, symbol: str) -> Optional[MicrostructureState]:
        return self._states.get(str(symbol or ""))

    def snapshot(self) -> Dict[str, MicrostructureState]:
        return dict(self._states)
