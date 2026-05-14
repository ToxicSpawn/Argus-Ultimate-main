"""Governance execution alpha — the ExecutionAlphaTuningPack."""
from __future__ import annotations

from typing import Any, Dict

from .types import (
    AggressionConfig,
    AbandonConfig,
    ExecutionAlphaConfig,
    ExecutionContext,
    ExecutionDecision,
    RoutingConfig,
    SlicingConfig,
    clamp,
)


class ExecutionAlphaTuningPack:
    def __init__(self, config: ExecutionAlphaConfig) -> None:
        self.config = config

    def apply_overrides(self, overrides: Dict[str, Any]) -> "ExecutionAlphaTuningPack":
        cfg = self.config
        a = cfg.aggression
        s = cfg.slicing
        r = cfg.routing
        ab = cfg.abandon

        new_cfg = ExecutionAlphaConfig(
            aggression=AggressionConfig(
                min_edge_to_cross_bps=a.min_edge_to_cross_bps * overrides.get("taker_min_edge_bps_multiplier", 1.0),
                max_cross_spread_bps=a.max_cross_spread_bps,
                urgency_multiplier=overrides.get("urgency_multiplier", a.urgency_multiplier),
                volatility_penalty=a.volatility_penalty,
                imbalance_boost=a.imbalance_boost,
                adverse_selection_penalty=a.adverse_selection_penalty,
            ),
            slicing=SlicingConfig(
                base_slice_notional=s.base_slice_notional,
                min_slice_notional=s.min_slice_notional,
                max_slice_pct_top_book=s.max_slice_pct_top_book * overrides.get("max_slice_pct_top_book_multiplier", 1.0),
                slice_growth_factor=s.slice_growth_factor,
                slice_decay_on_reject=s.slice_decay_on_reject,
                slice_decay_on_slippage=overrides.get("slice_decay_on_slippage_override", s.slice_decay_on_slippage),
                max_child_orders=s.max_child_orders,
            ),
            routing=RoutingConfig(
                maker_min_fill_prob=r.maker_min_fill_prob,
                maker_max_wait_ms=r.maker_max_wait_ms,
                taker_min_edge_bps=r.taker_min_edge_bps * overrides.get("taker_min_edge_bps_multiplier", 1.0),
                queue_position_threshold=r.queue_position_threshold,
                drift_escape_threshold=r.drift_escape_threshold,
                maker_retry_limit=overrides.get("maker_retry_limit_override", r.maker_retry_limit),
            ),
            abandon=ab,
        )
        return ExecutionAlphaTuningPack(new_cfg)

    def decide(self, ctx: ExecutionContext) -> ExecutionDecision:
        if self._should_cancel(ctx):
            return ExecutionDecision(
                mode="CANCEL",
                aggression_score=0.0,
                slice_notional=0.0,
                cancel=True,
                reason="edge/volatility/spread/fill-probability abandon condition met",
            )

        aggression = self._aggression_score(ctx)
        slice_notional = self._slice_size(ctx, aggression)

        use_taker = self._should_cross(ctx, aggression)
        mode = "TAKER" if use_taker else "MAKER"
        return ExecutionDecision(
            mode=mode,
            aggression_score=aggression,
            slice_notional=slice_notional,
            cancel=False,
            reason=self._decision_reason(ctx, aggression, mode),
        )

    def _aggression_score(self, ctx: ExecutionContext) -> float:
        c = self.config.aggression
        score = 0.0
        score += ctx.expected_edge_bps
        score += c.imbalance_boost * ctx.imbalance_score
        score += c.urgency_multiplier * ctx.urgency_score
        score += 0.50 * ctx.short_horizon_drift_bps
        score -= c.volatility_penalty * ctx.volatility_score
        score -= c.adverse_selection_penalty * ctx.adverse_selection_score
        score -= 0.50 * max(0.0, ctx.spread_bps - c.max_cross_spread_bps)
        return score

    def _slice_size(self, ctx: ExecutionContext, aggression_score: float) -> float:
        s = self.config.slicing
        base = s.base_slice_notional
        top_book_cap = max(s.min_slice_notional, ctx.top_book_notional * s.max_slice_pct_top_book)
        slice_size = min(base, top_book_cap, ctx.remaining_notional)

        if aggression_score > self.config.routing.taker_min_edge_bps:
            slice_size *= s.slice_growth_factor

        if ctx.fill_probability < self.config.routing.maker_min_fill_prob:
            slice_size *= s.slice_decay_on_reject

        if ctx.spread_widen_bps > self.config.abandon.abandon_on_spread_widen_bps:
            slice_size *= s.slice_decay_on_slippage

        return round(clamp(slice_size, s.min_slice_notional, max(s.min_slice_notional, ctx.remaining_notional)), 4)

    def _should_cross(self, ctx: ExecutionContext, aggression_score: float) -> bool:
        r = self.config.routing
        a = self.config.aggression
        edge_ok = ctx.expected_edge_bps >= max(a.min_edge_to_cross_bps, r.taker_min_edge_bps)
        spread_ok = ctx.spread_bps <= a.max_cross_spread_bps
        drift_escape = ctx.short_horizon_drift_bps >= r.drift_escape_threshold
        maker_unfavorable = (
            ctx.fill_probability < r.maker_min_fill_prob
            or ctx.queue_position_score < r.queue_position_threshold
            or ctx.maker_retries_used >= r.maker_retry_limit
            or ctx.elapsed_wait_ms > r.maker_max_wait_ms
        )
        return (edge_ok and spread_ok and aggression_score > 0.0 and (drift_escape or maker_unfavorable))

    def _should_cancel(self, ctx: ExecutionContext) -> bool:
        ab = self.config.abandon
        if ctx.fill_probability < ab.cancel_if_fill_prob_below:
            return True
        if ctx.elapsed_wait_ms > ab.abandon_wait_ms:
            return True
        if ctx.current_edge_decay_bps > ab.abandon_on_edge_decay_bps:
            return True
        if ctx.spread_widen_bps > ab.abandon_on_spread_widen_bps:
            return True
        if ctx.volatility_score > ab.abandon_on_vol_spike:
            return True
        return False

    @staticmethod
    def _decision_reason(ctx: ExecutionContext, aggression_score: float, mode: str) -> str:
        return (
            f"mode={mode}; edge={ctx.expected_edge_bps:.2f}bps; spread={ctx.spread_bps:.2f}bps; "
            f"drift={ctx.short_horizon_drift_bps:.2f}bps; fill_prob={ctx.fill_probability:.2f}; "
            f"aggression={aggression_score:.2f}"
        )
