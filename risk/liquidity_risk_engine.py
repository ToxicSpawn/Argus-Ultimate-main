from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(slots=True)
class LiquidityRiskState:
    symbol: str
    spread_bps: float
    top_of_book_bid_size: float
    top_of_book_ask_size: float
    orderbook_depth_estimate: float
    liquidity_score: float
    slippage_estimate_bps: float
    max_safe_trade_size: float
    liquidity_state: str  # normal | thin | danger


class LiquidityRiskEngine:
    """
    Deterministic liquidity-aware risk clamps for target deltas.

    Designed to be additive and fail-safe: if data is incomplete, it prefers
    conservative defaults without breaking the existing risk pipeline.
    """

    def __init__(self, config: Any, *, depth_history_window: int = 200) -> None:
        self.config = config
        self.depth_history_window = max(20, int(depth_history_window or 200))
        self._depth_history: Dict[str, deque] = {}

    @staticmethod
    def _cfg_float(config: Any, name: str, default: float) -> float:
        raw = getattr(config, name, default)
        try:
            return float(raw if raw is not None else default)
        except Exception:
            return float(default)

    @staticmethod
    def _signal_get(sig: Any, name: str, default: Any = None) -> Any:
        if isinstance(sig, dict):
            return sig.get(name, default)
        return getattr(sig, name, default)

    def _weights(self) -> Tuple[float, float, float]:
        weights = dict(getattr(self.config, "liquidity_risk_score_weights", {}) or {})
        w_depth = float(weights.get("depth", 1.0) or 1.0)
        w_spread = float(weights.get("spread", 1.0) or 1.0)
        w_fill = float(weights.get("fill_ratio", 0.75) or 0.75)
        return (max(0.0, w_depth), max(0.0, w_spread), max(0.0, w_fill))

    def _median_depth(self, symbol: str, depth: float) -> float:
        sym = str(symbol or "")
        row = self._depth_history.get(sym)
        if row is None:
            row = deque(maxlen=self.depth_history_window)
            self._depth_history[sym] = row
        if depth > 0.0:
            row.append(float(depth))
        if not row:
            return max(depth, 1.0)
        try:
            return float(np.median(np.asarray(list(row), dtype=float)))
        except Exception:
            return float(max(depth, 1.0))

    @staticmethod
    def _price_aud(symbol: str, price: float, aud_to_usd: float) -> float:
        quote = str(symbol or "").split("/")[-1].upper() if "/" in str(symbol or "") else "USD"
        return float(price) if quote == "AUD" else float(price) / max(float(aud_to_usd), 1e-9)

    def evaluate_state(
        self,
        *,
        symbol: str,
        spread_bps: float,
        bid_size: float,
        ask_size: float,
        depth_estimate: float,
        slippage_estimate_bps: float,
        maker_fill_ratio: float,
    ) -> LiquidityRiskState:
        spread = max(0.0, float(spread_bps or 0.0))
        bid = max(0.0, float(bid_size or 0.0))
        ask = max(0.0, float(ask_size or 0.0))
        depth = max(0.0, float(depth_estimate or 0.0))
        fill_ratio = max(0.0, min(1.0, float(maker_fill_ratio or 0.0)))
        slip = max(0.0, float(slippage_estimate_bps or 0.0))

        depth_fraction_limit = max(0.001, self._cfg_float(self.config, "liquidity_risk_depth_fraction_limit", 0.04))
        thin_spread = max(0.01, self._cfg_float(self.config, "liquidity_risk_thin_spread_threshold_bps", 6.0))
        danger_spread = max(thin_spread, self._cfg_float(self.config, "liquidity_risk_danger_spread_threshold_bps", 12.0))
        min_depth_threshold = max(0.0, self._cfg_float(self.config, "liquidity_risk_min_depth_threshold", 0.5))
        slip_threshold = max(0.01, self._cfg_float(self.config, "liquidity_risk_slippage_threshold_bps", 10.0))
        min_liquidity_score = max(0.0, min(1.0, self._cfg_float(self.config, "liquidity_risk_min_liquidity_score", 0.2)))

        has_depth_signal = depth > 0.0 or bid > 0.0 or ask > 0.0

        # When no L2 data AND no meaningful spread data is available (common in
        # paper trading without orderbook feeds), assume reasonable defaults so
        # liquidity checks don't block all trades.  If a spread is present it
        # may still indicate thin/danger, so we only short-circuit when
        # spread is also below the thin threshold.
        # The caller can disable via config: liquidity_risk_assume_normal_without_l2 = 0.
        assume_normal = bool(
            self._cfg_float(self.config, "liquidity_risk_assume_normal_without_l2", 0.0) > 0.0
        )
        if not has_depth_signal and assume_normal and spread <= thin_spread:
            return LiquidityRiskState(
                symbol=str(symbol or ""),
                spread_bps=float(spread),
                top_of_book_bid_size=0.0,
                top_of_book_ask_size=0.0,
                orderbook_depth_estimate=0.0,
                liquidity_score=1.0,
                slippage_estimate_bps=float(slip),
                max_safe_trade_size=float("inf"),
                liquidity_state="normal",
            )

        effective_depth = max(depth, bid + ask, min(bid, ask))
        median_depth = self._median_depth(symbol, effective_depth)
        normalized_depth = min(2.0, max(0.0, effective_depth / max(median_depth, 1e-9)))
        spread_quality = min(1.5, max(0.0, thin_spread / max(spread, 0.1)))

        w_depth, w_spread, w_fill = self._weights()
        denom = max(1e-9, w_depth + w_spread + w_fill)
        liquidity_score = (
            (w_depth * normalized_depth)
            + (w_spread * spread_quality)
            + (w_fill * fill_ratio)
        ) / denom

        liquidity_state = "normal"
        if spread > danger_spread or (has_depth_signal and effective_depth < min_depth_threshold):
            liquidity_state = "danger"
        elif spread > thin_spread:
            liquidity_state = "thin"

        top_size = min(bid, ask) if (bid > 0.0 and ask > 0.0) else max(bid, ask, depth, 0.0)
        max_safe_trade_size = max(0.0, depth_fraction_limit * top_size)

        # Slippage-aware down-scaling.
        if slip > slip_threshold and slip > 0.0:
            scale = max(0.0, min(1.0, slip_threshold / max(slip, 1e-9)))
            max_safe_trade_size *= scale

        if liquidity_state == "thin":
            max_safe_trade_size *= 0.60
        elif liquidity_state == "danger":
            max_safe_trade_size *= 0.25
        if liquidity_score < min_liquidity_score:
            # Conservative fail-safe: below minimum liquidity quality, avoid new exposure.
            max_safe_trade_size = 0.0
            if liquidity_state == "normal":
                liquidity_state = "thin"

        # Deterministic consistency rule:
        # - normal liquidity must always have positive safe size
        # - zero safe size must be thin or danger (never normal)
        if max_safe_trade_size <= 0.0 and liquidity_state == "normal":
            liquidity_state = "thin"

        return LiquidityRiskState(
            symbol=str(symbol or ""),
            spread_bps=float(spread),
            top_of_book_bid_size=float(bid),
            top_of_book_ask_size=float(ask),
            orderbook_depth_estimate=float(effective_depth),
            liquidity_score=float(liquidity_score),
            slippage_estimate_bps=float(slip),
            max_safe_trade_size=float(max_safe_trade_size),
            liquidity_state=str(liquidity_state),
        )

    def adjust_targets(
        self,
        *,
        targets: Iterable[Any],
        symbol_market_state: Dict[str, Dict[str, Any]],
        execution_telemetry: Dict[str, Dict[str, Any]],
        equity_aud: float,
        aud_to_usd: float,
    ) -> Tuple[List[Any], Dict[str, LiquidityRiskState], int]:
        eq = max(0.0, float(equity_aud or 0.0))
        adjusted: List[Any] = []
        states: Dict[str, LiquidityRiskState] = {}
        clamp_count = 0

        for t in list(targets or []):
            symbol = str(self._signal_get(t, "symbol", "") or "")
            ctx = dict(symbol_market_state.get(symbol) or {})
            telemetry = dict(execution_telemetry.get(symbol) or {})
            state = self.evaluate_state(
                symbol=symbol,
                spread_bps=float(ctx.get("spread_bps", 0.0) or 0.0),
                bid_size=float(ctx.get("top_of_book_bid_size", 0.0) or 0.0),
                ask_size=float(ctx.get("top_of_book_ask_size", 0.0) or 0.0),
                depth_estimate=float(ctx.get("orderbook_depth_estimate", 0.0) or 0.0),
                slippage_estimate_bps=float(
                    ctx.get("slippage_estimate_bps", telemetry.get("slippage_p90", 0.0) or 0.0) or 0.0
                ),
                maker_fill_ratio=float(telemetry.get("maker_fill_ratio", 0.0) or 0.0),
            )
            states[symbol] = state
            min_liquidity_score = max(
                0.0,
                min(
                    1.0,
                    self._cfg_float(self.config, "liquidity_risk_min_liquidity_score", 0.2),
                ),
            )

            # Default metadata for auditability even without clamp.
            setattr(t, "liquidity_score", float(state.liquidity_score))
            setattr(t, "liquidity_state", str(state.liquidity_state))
            setattr(t, "max_safe_trade_size", float(state.max_safe_trade_size))
            setattr(t, "slippage_estimate_bps", float(state.slippage_estimate_bps))
            setattr(t, "liquidity_clamp_flag", False)

            if eq <= 0.0:
                setattr(t, "adjusted_target_exposure_pct", float(getattr(t, "target_exposure_pct", 0.0) or 0.0))
                adjusted.append(t)
                continue

            ref_price = float(
                self._signal_get(t, "reference_price", 0.0)
                or self._signal_get(t, "price", 0.0)
                or ctx.get("price", 0.0)
                or 0.0
            )
            px_aud = self._price_aud(symbol, ref_price if ref_price > 0.0 else 0.0, aud_to_usd)
            if px_aud <= 0.0:
                setattr(t, "adjusted_target_exposure_pct", float(getattr(t, "target_exposure_pct", 0.0) or 0.0))
                adjusted.append(t)
                continue

            delta_qty = float(
                self._signal_get(t, "delta_qty", 0.0)
                or ((float(self._signal_get(t, "delta_exposure_pct", 0.0) or 0.0) * eq) / max(px_aud, 1e-9))
            )
            safe_qty = max(0.0, float(state.max_safe_trade_size))
            abs_delta = abs(delta_qty)
            clamped_qty = abs_delta

            if state.liquidity_state == "danger" and safe_qty <= 1e-12:
                clamped_qty = 0.0
            elif state.liquidity_score < min_liquidity_score:
                clamped_qty = 0.0
            elif safe_qty > 0.0 and abs_delta > safe_qty:
                clamped_qty = safe_qty
            elif safe_qty <= 1e-12 and abs_delta > 0.0:
                clamped_qty = 0.0

            if clamped_qty < abs_delta - 1e-12:
                clamp_count += 1
                setattr(t, "liquidity_clamp_flag", True)
                reasons = list(self._signal_get(t, "reasons", []) or [])
                if "liquidity_clamp" not in reasons:
                    reasons.append("liquidity_clamp")
                if clamped_qty <= 1e-12:
                    if state.liquidity_score < min_liquidity_score:
                        reasons.append("suppressed:liquidity_score_low")
                    if state.liquidity_state == "danger":
                        reasons.append("suppressed:liquidity_danger")
                    elif state.liquidity_state == "thin":
                        reasons.append("suppressed:liquidity_thin")
                setattr(t, "reasons", reasons)

            signed_qty = (1.0 if delta_qty >= 0.0 else -1.0) * clamped_qty
            adjusted_delta_pct = (signed_qty * px_aud) / max(eq, 1e-9)
            current_pct = float(self._signal_get(t, "current_exposure_pct", 0.0) or 0.0)
            adjusted_target_pct = max(0.0, current_pct + adjusted_delta_pct)

            setattr(t, "delta_qty", float(signed_qty))
            setattr(t, "target_qty", float(self._signal_get(t, "current_qty", 0.0) or 0.0) + float(signed_qty))
            setattr(t, "delta_exposure_pct", float(adjusted_delta_pct))
            setattr(t, "target_exposure_pct", float(adjusted_target_pct))
            setattr(t, "adjusted_target_exposure_pct", float(adjusted_target_pct))
            adjusted.append(t)

        return adjusted, states, int(clamp_count)
