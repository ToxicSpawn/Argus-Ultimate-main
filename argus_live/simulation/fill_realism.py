from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any

from argus_live.execution.fill_tracker import Fill
from .hostile_scenarios import MarketState


@dataclass(frozen=True)
class FillSimulationProfile:
    maker_base_fill_probability: float = 0.72
    taker_base_fill_probability: float = 0.97
    impact_bps_per_book_take: float = 18.0
    partial_fill_floor: float = 0.35


class FillRealismEngine:
    def __init__(self, profile: FillSimulationProfile | None = None, seed: int = 7) -> None:
        self.profile = profile or FillSimulationProfile()
        self.seed = seed

    def simulate(self, intent_id: str, quantity: float, price: float, metadata: dict[str, Any] | None = None) -> Fill:
        md = metadata or {}
        state: MarketState | None = md.get("market_state")
        if state is None:
            return Fill(intent_id, quantity, price)
        mode = str(md.get("mode") or "maker").lower()
        side = str(md.get("side") or "buy").lower()
        top_book_notional = max(float(md.get("top_of_book_notional") or state.top_of_book_notional), 1e-9)
        requested_notional = float(quantity) * float(price)
        take_ratio = min(3.0, requested_notional / top_book_notional)
        rng = self._rng(intent_id, state, md)

        base_prob = self.profile.taker_base_fill_probability if mode == "taker" else self.profile.maker_base_fill_probability
        fill_prob = base_prob * state.fill_probability_multiplier * state.venue_quality
        fill_prob *= max(0.1, 1.0 - 0.25 * take_ratio)
        if state.stale_quote:
            fill_prob *= 0.65
        if rng.random() < state.reject_probability:
            return Fill(intent_id, 0.0, price, latency_ms=state.latency_ms, rejected=True, reason="scenario_reject")
        if rng.random() > max(0.0, min(1.0, fill_prob)):
            partial = max(self.profile.partial_fill_floor, min(0.95, fill_prob))
            qty = round(float(quantity) * partial, 12)
            if qty <= 0:
                return Fill(intent_id, 0.0, price, latency_ms=state.latency_ms, rejected=True, reason="no_fill")
            price_out = self._fill_price(side, price, state, take_ratio, mode, rng)
            return Fill(intent_id, qty, price_out, latency_ms=state.latency_ms, partial_fill=True, reason="partial_fill")
        price_out = self._fill_price(side, price, state, take_ratio, mode, rng)
        return Fill(intent_id, quantity, price_out, latency_ms=state.latency_ms)

    def _fill_price(self, side: str, price: float, state: MarketState, take_ratio: float, mode: str, rng: random.Random) -> float:
        impact = self.profile.impact_bps_per_book_take * take_ratio * max(0.25, state.venue_quality)
        spread_component = state.spread_bps * (0.35 if mode == "maker" else 0.85)
        volatility_component = state.volatility_bps * 0.15
        stale_component = state.spread_bps * 0.50 if state.stale_quote else 0.0
        noise = rng.uniform(-0.15, 0.15) * max(1.0, state.spread_bps)
        total_bps = max(0.0, spread_component + volatility_component + impact + stale_component + noise)
        if side == "buy":
            return float(price) * (1.0 + total_bps / 10000.0)
        return float(price) * (1.0 - total_bps / 10000.0)

    def _rng(self, intent_id: str, state: MarketState, metadata: dict[str, Any]) -> random.Random:
        payload = f"{self.seed}|{intent_id}|{state.symbol}|{state.spread_bps}|{state.volatility_bps}|{metadata.get('mode')}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return random.Random(int(digest[:16], 16))
