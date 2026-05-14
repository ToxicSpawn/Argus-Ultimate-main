"""
L2 Orderbook Alpha Pipeline
VPIN toxicity detection, order flow imbalance, and book pressure analysis.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class OrderbookLevel:
    price: float
    size: float
    count: int = 1


@dataclass
class OrderbookSnapshot:
    bids: List[OrderbookLevel]
    asks: List[OrderbookLevel]
    timestamp: float = 0.0


class VPINCalculator:
    """Volume-synchronized Probability of Informed Trading.

    Buckets trades by volume rather than time, then measures buy/sell
    imbalance across N buckets to estimate information toxicity.
    """

    def __init__(self, bucket_size: float = 1000.0, n_buckets: int = 50):
        self.bucket_size = float(bucket_size)
        self.n_buckets = int(n_buckets)
        self._current_bucket_buy: float = 0.0
        self._current_bucket_sell: float = 0.0
        self._current_bucket_volume: float = 0.0
        self._buckets: deque = deque(maxlen=n_buckets)

    def update(self, price: float, size: float, side: str) -> Optional[float]:
        vol = abs(float(size))
        if side == "buy":
            self._current_bucket_buy += vol
        else:
            self._current_bucket_sell += vol
        self._current_bucket_volume += vol

        if self._current_bucket_volume >= self.bucket_size:
            buy_frac = self._current_bucket_buy / max(self._current_bucket_volume, 1e-12)
            sell_frac = self._current_bucket_sell / max(self._current_bucket_volume, 1e-12)
            imbalance = abs(buy_frac - sell_frac)
            self._buckets.append(imbalance)
            self._current_bucket_buy = 0.0
            self._current_bucket_sell = 0.0
            self._current_bucket_volume = 0.0

        if len(self._buckets) >= 5:
            return float(np.mean(list(self._buckets)))
        return None

    def get_vpin(self) -> float:
        if not self._buckets:
            return 0.0
        return float(np.mean(list(self._buckets)))

    def is_toxic(self, threshold: float = 0.7) -> bool:
        return self.get_vpin() > threshold


class OrderFlowImbalance:
    """Track order flow imbalance from trade stream."""

    def __init__(self, lookback: int = 100, ewm_span: int = 20):
        self.lookback = int(lookback)
        self.ewm_span = int(ewm_span)
        self._buy_volumes: deque = deque(maxlen=lookback)
        self._sell_volumes: deque = deque(maxlen=lookback)
        self._net_flows: deque = deque(maxlen=lookback)

    def update(self, price: float, size: float, side: str) -> Dict[str, float]:
        vol = abs(float(size))
        if side == "buy":
            self._buy_volumes.append(vol)
            self._sell_volumes.append(0.0)
            self._net_flows.append(vol)
        else:
            self._buy_volumes.append(0.0)
            self._sell_volumes.append(vol)
            self._net_flows.append(-vol)

        total_buy = sum(self._buy_volumes)
        total_sell = sum(self._sell_volumes)
        total = total_buy + total_sell

        ofi = (total_buy - total_sell) / max(total, 1e-12)

        if len(self._net_flows) >= self.ewm_span:
            flows = np.array(list(self._net_flows))
            alpha = 2.0 / (self.ewm_span + 1)
            weights = np.array([(1 - alpha) ** i for i in range(len(flows) - 1, -1, -1)])
            weights /= weights.sum()
            ewm_flow = float(np.sum(weights * flows))
        else:
            ewm_flow = float(np.mean(list(self._net_flows))) if self._net_flows else 0.0

        return {
            "ofi": ofi,
            "ewm_flow": ewm_flow,
            "buy_volume": total_buy,
            "sell_volume": total_sell,
            "pressure": "buy" if ofi > 0.1 else ("sell" if ofi < -0.1 else "neutral"),
        }


class BookPressureAnalyzer:
    """Analyze L2 book for support/resistance pressure."""

    def __init__(self, depth_levels: int = 10):
        self.depth_levels = int(depth_levels)

    def analyze(self, snapshot: OrderbookSnapshot) -> Dict[str, Any]:
        if not snapshot.bids or not snapshot.asks:
            return {"bid_pressure": 0.0, "ask_pressure": 0.0, "imbalance": 0.0, "spread": 0.0}

        bids = snapshot.bids[:self.depth_levels]
        asks = snapshot.asks[:self.depth_levels]

        bid_volume = sum(l.size for l in bids)
        ask_volume = sum(l.size for l in asks)
        total = bid_volume + ask_volume

        imbalance = (bid_volume - ask_volume) / max(total, 1e-12)

        best_bid = bids[0].price if bids else 0
        best_ask = asks[0].price if asks else 0
        spread = (best_ask - best_bid) / max(best_ask, 1e-12) if best_ask > 0 else 0

        bid_wall = max(bids, key=lambda x: x.size) if bids else None
        ask_wall = max(asks, key=lambda x: x.size) if asks else None

        return {
            "bid_pressure": bid_volume,
            "ask_pressure": ask_volume,
            "imbalance": imbalance,
            "spread_pct": spread,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_wall_price": bid_wall.price if bid_wall else 0,
            "bid_wall_size": bid_wall.size if bid_wall else 0,
            "ask_wall_price": ask_wall.price if ask_wall else 0,
            "ask_wall_size": ask_wall.size if ask_wall else 0,
            "signal": "bullish" if imbalance > 0.2 else ("bearish" if imbalance < -0.2 else "neutral"),
        }


class LiquidationCascadeDetector:
    """Detect potential liquidation cascades from sudden volume/price moves."""

    def __init__(self, volume_spike_mult: float = 5.0, price_move_threshold: float = 0.02, lookback: int = 50):
        self.volume_spike_mult = float(volume_spike_mult)
        self.price_move_threshold = float(price_move_threshold)
        self._volumes: deque = deque(maxlen=lookback)
        self._prices: deque = deque(maxlen=lookback)

    def update(self, price: float, volume: float) -> Dict[str, Any]:
        self._prices.append(float(price))
        self._volumes.append(float(volume))

        if len(self._volumes) < 10:
            return {"cascade_risk": 0.0, "alert": False}

        avg_vol = float(np.mean(list(self._volumes)[:-1]))
        current_vol = float(self._volumes[-1])
        vol_ratio = current_vol / max(avg_vol, 1e-12)

        prices = list(self._prices)
        if len(prices) >= 3:
            recent_move = abs(prices[-1] - prices[-3]) / max(abs(prices[-3]), 1e-12)
        else:
            recent_move = 0.0

        vol_signal = min(vol_ratio / self.volume_spike_mult, 1.0)
        price_signal = min(recent_move / self.price_move_threshold, 1.0)
        cascade_risk = (vol_signal * 0.6 + price_signal * 0.4)

        return {
            "cascade_risk": cascade_risk,
            "volume_ratio": vol_ratio,
            "price_move": recent_move,
            "alert": cascade_risk > 0.7,
            "direction": "down" if prices[-1] < prices[-2] else "up" if len(prices) >= 2 else "unknown",
        }


class FundingRateSignal:
    """Track funding rate for sentiment and mean-reversion signals."""

    def __init__(self, extreme_threshold: float = 0.001, lookback: int = 100):
        self.extreme_threshold = float(extreme_threshold)
        self._rates: deque = deque(maxlen=lookback)

    def update(self, rate: float) -> Dict[str, Any]:
        self._rates.append(float(rate))

        if len(self._rates) < 5:
            return {"signal": 0.0, "extreme": False, "direction": "neutral"}

        rates = np.array(list(self._rates))
        current = float(rates[-1])
        avg = float(np.mean(rates))
        std = float(np.std(rates))
        z_score = (current - avg) / max(std, 1e-12) if std > 1e-12 else 0.0

        extreme = abs(current) > self.extreme_threshold
        signal = 0.0
        direction = "neutral"
        if extreme:
            signal = -np.sign(current) * min(abs(z_score) / 3.0, 1.0)
            direction = "short" if current > 0 else "long"

        return {
            "signal": signal,
            "funding_rate": current,
            "z_score": z_score,
            "extreme": extreme,
            "direction": direction,
            "avg_rate": avg,
        }


class OrderbookPipeline:
    """Unified pipeline combining all orderbook alpha sources."""

    def __init__(self, bucket_size: float = 1000.0):
        self.vpin = VPINCalculator(bucket_size=bucket_size)
        self.ofi = OrderFlowImbalance()
        self.book_pressure = BookPressureAnalyzer()
        self.liquidation = LiquidationCascadeDetector()
        self.funding = FundingRateSignal()

    def on_trade(self, price: float, size: float, side: str) -> Dict[str, Any]:
        vpin_val = self.vpin.update(price, size, side)
        ofi_result = self.ofi.update(price, size, side)
        liq_result = self.liquidation.update(price, abs(size))
        return {
            "vpin": vpin_val if vpin_val is not None else self.vpin.get_vpin(),
            "vpin_toxic": self.vpin.is_toxic(),
            "order_flow": ofi_result,
            "liquidation": liq_result,
        }

    def on_book_update(self, snapshot: OrderbookSnapshot) -> Dict[str, Any]:
        return self.book_pressure.analyze(snapshot)

    def on_funding_rate(self, rate: float) -> Dict[str, Any]:
        return self.funding.update(rate)

    def get_composite_signal(self) -> Dict[str, Any]:
        vpin = self.vpin.get_vpin()
        toxic = self.vpin.is_toxic()
        funding = self.funding._rates[-1] if self.funding._rates else 0.0

        risk_score = 0.0
        if toxic:
            risk_score += 0.4
        liq = self.liquidation.update(
            self.liquidation._prices[-1] if self.liquidation._prices else 0,
            self.liquidation._volumes[-1] if self.liquidation._volumes else 0,
        )
        risk_score += liq["cascade_risk"] * 0.3
        if abs(funding) > self.funding.extreme_threshold:
            risk_score += 0.3

        return {
            "vpin": vpin,
            "toxic": toxic,
            "cascade_risk": liq["cascade_risk"],
            "funding_extreme": abs(funding) > self.funding.extreme_threshold,
            "composite_risk": min(risk_score, 1.0),
            "trade_caution": risk_score > 0.5,
        }
