"""
Universal Data Brain — ARGUS reads EVERY signal the market produces.

This is the omniscient data layer. It ingests, normalises, and scores
every available market data source into a unified signal vector that
the strategy engine, evolver, and predictor can consume.

Data sources (17 categories):
 1. Price/OHLCV (already have)
 2. Order book depth + imbalance
 3. Trade tape (time & sales)
 4. Funding rates (perp futures)
 5. Open interest changes
 6. Liquidation cascades
 7. Options flow (put/call ratio, IV skew, max pain)
 8. On-chain: whale wallets, exchange inflows/outflows
 9. On-chain: active addresses, transaction count
10. Stablecoin flows (USDT/USDC mint/burn)
11. Mining: hash rate, difficulty, miner outflows
12. DeFi: TVL, DEX volume, lending rates
13. Macro: DXY, S&P500, gold, bond yields
14. Sentiment: Fear & Greed, social media volume
15. News: event classification (bullish/bearish/neutral)
16. Correlation: BTC dominance, cross-asset correlation shifts
17. Technical structure: support/resistance levels, volume profile

Each source produces a normalised score from -1.0 (extremely bearish)
to +1.0 (extremely bullish). The brain aggregates all scores into a
single MarketIntelligence object with per-source breakdown.

The key insight: no single signal is reliable. But when 10+ independent
signals agree, the probability of a correct trade is very high.
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataSignal:
    """One normalised signal from any data source."""
    source: str             # e.g. "funding_rate", "whale_flow", "options_skew"
    category: str           # e.g. "on_chain", "derivatives", "macro"
    value: float            # raw value
    score: float            # normalised -1.0 to +1.0
    confidence: float       # 0.0 to 1.0 (data quality / freshness)
    timestamp: float
    description: str = ""


@dataclass
class MarketIntelligence:
    """Aggregated market intelligence from all data sources."""
    symbol: str
    timestamp: float
    # Aggregate scores
    composite_score: float = 0.0        # -1 to +1
    composite_confidence: float = 0.0   # 0 to 1
    signal_count: int = 0
    agreeing_signals: int = 0           # how many agree with composite direction
    # Per-category scores
    price_action_score: float = 0.0
    orderbook_score: float = 0.0
    derivatives_score: float = 0.0      # funding + OI + options + liquidations
    onchain_score: float = 0.0          # whales + addresses + stablecoins
    macro_score: float = 0.0            # DXY + S&P + gold + bonds
    sentiment_score: float = 0.0        # fear&greed + social + news
    technical_score: float = 0.0        # support/resistance + volume profile
    # Raw signals
    signals: List[DataSignal] = field(default_factory=list)
    # Derived
    regime_hint: str = "NEUTRAL"        # "STRONG_BULL", "BULL", "NEUTRAL", "BEAR", "STRONG_BEAR"
    conviction_level: str = "LOW"       # "LOW", "MEDIUM", "HIGH", "EXTREME"


class UniversalDataBrain:
    """
    Omniscient market data aggregator.

    Ingests all available data sources, normalises them to -1/+1 scores,
    and produces a unified MarketIntelligence for each symbol.

    Sources are pluggable: call inject_signal() from any data provider.
    The brain doesn't fetch data itself — it receives and aggregates.
    """

    def __init__(self, signal_ttl_seconds: float = 300.0, min_signals_for_confidence: int = 3):
        self._ttl = signal_ttl_seconds
        self._min_signals = min_signals_for_confidence
        self._signals: Dict[str, Dict[str, DataSignal]] = defaultdict(dict)  # symbol → {source: signal}
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        # Source reliability tracking
        self._source_accuracy: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._source_weights: Dict[str, float] = {}
        # Category weights (configurable)
        self._category_weights = {
            "price_action": 0.20,
            "orderbook": 0.15,
            "derivatives": 0.15,
            "onchain": 0.12,
            "macro": 0.10,
            "sentiment": 0.08,
            "technical": 0.10,
            "mining": 0.05,
            "defi": 0.05,
        }

    def inject_signal(self, symbol: str, signal: DataSignal) -> None:
        """Inject a signal from any data source."""
        self._signals[symbol][signal.source] = signal
        self._history[symbol].append(signal)

    def inject_raw(
        self,
        symbol: str,
        source: str,
        category: str,
        value: float,
        score: float,
        confidence: float = 0.8,
        description: str = "",
    ) -> None:
        """Convenience: inject a raw signal without building DataSignal."""
        sig = DataSignal(
            source=source, category=category, value=value,
            score=max(-1, min(1, score)), confidence=max(0, min(1, confidence)),
            timestamp=time.time(), description=description,
        )
        self.inject_signal(symbol, sig)

    def record_outcome(self, source: str, was_correct: bool) -> None:
        """Record whether a signal source's prediction was correct."""
        self._source_accuracy[source].append(1.0 if was_correct else 0.0)
        # Update weight based on accuracy
        acc = list(self._source_accuracy[source])
        if len(acc) >= 5:
            self._source_weights[source] = sum(acc) / len(acc)

    def compute(self, symbol: str) -> MarketIntelligence:
        """Compute aggregated market intelligence for a symbol."""
        now = time.time()
        signals = []

        # Collect fresh signals
        for source, sig in self._signals.get(symbol, {}).items():
            age = now - sig.timestamp
            if age <= self._ttl:
                # Decay confidence by age
                freshness = max(0, 1 - age / self._ttl)
                adjusted_conf = sig.confidence * freshness
                # Apply source reliability weight
                source_w = self._source_weights.get(sig.source, 1.0)
                signals.append(DataSignal(
                    source=sig.source, category=sig.category,
                    value=sig.value, score=sig.score,
                    confidence=adjusted_conf * source_w,
                    timestamp=sig.timestamp, description=sig.description,
                ))

        if not signals:
            return MarketIntelligence(symbol=symbol, timestamp=now)

        # Group by category
        cat_scores: Dict[str, List[Tuple[float, float]]] = defaultdict(list)  # category → [(score, confidence)]
        for sig in signals:
            cat_scores[sig.category].append((sig.score, sig.confidence))

        # Compute per-category weighted scores
        def _weighted_avg(pairs: List[Tuple[float, float]]) -> float:
            if not pairs:
                return 0.0
            total_w = sum(c for _, c in pairs)
            if total_w <= 0:
                return 0.0
            return sum(s * c for s, c in pairs) / total_w

        price_score = _weighted_avg(cat_scores.get("price_action", []))
        ob_score = _weighted_avg(cat_scores.get("orderbook", []))
        deriv_score = _weighted_avg(cat_scores.get("derivatives", []))
        onchain_score = _weighted_avg(cat_scores.get("onchain", []))
        macro_score = _weighted_avg(cat_scores.get("macro", []))
        sentiment_score = _weighted_avg(cat_scores.get("sentiment", []))
        technical_score = _weighted_avg(cat_scores.get("technical", []))

        # Composite: category-weighted sum
        composite = (
            price_score * self._category_weights.get("price_action", 0.2)
            + ob_score * self._category_weights.get("orderbook", 0.15)
            + deriv_score * self._category_weights.get("derivatives", 0.15)
            + onchain_score * self._category_weights.get("onchain", 0.12)
            + macro_score * self._category_weights.get("macro", 0.10)
            + sentiment_score * self._category_weights.get("sentiment", 0.08)
            + technical_score * self._category_weights.get("technical", 0.10)
        )

        # Composite confidence: based on signal count + agreement
        direction = "UP" if composite > 0 else "DOWN"
        agreeing = sum(1 for s in signals if (s.score > 0) == (composite > 0))
        total = len(signals)
        agreement_ratio = agreeing / max(total, 1)

        confidence = min(1.0, (
            agreement_ratio * 0.5
            + min(total / max(self._min_signals * 2, 1), 1.0) * 0.3
            + sum(s.confidence for s in signals) / max(total, 1) * 0.2
        ))

        # Regime hint
        if composite > 0.5:
            regime = "STRONG_BULL"
        elif composite > 0.2:
            regime = "BULL"
        elif composite < -0.5:
            regime = "STRONG_BEAR"
        elif composite < -0.2:
            regime = "BEAR"
        else:
            regime = "NEUTRAL"

        # Conviction level
        if agreement_ratio > 0.8 and total >= 5:
            conviction = "EXTREME"
        elif agreement_ratio > 0.65 and total >= 3:
            conviction = "HIGH"
        elif agreement_ratio > 0.5:
            conviction = "MEDIUM"
        else:
            conviction = "LOW"

        return MarketIntelligence(
            symbol=symbol, timestamp=now,
            composite_score=composite, composite_confidence=confidence,
            signal_count=total, agreeing_signals=agreeing,
            price_action_score=price_score, orderbook_score=ob_score,
            derivatives_score=deriv_score, onchain_score=onchain_score,
            macro_score=macro_score, sentiment_score=sentiment_score,
            technical_score=technical_score,
            signals=signals, regime_hint=regime, conviction_level=conviction,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Built-in signal generators (from raw market data)
    # ──────────────────────────────────────────────────────────────────────

    def process_price_data(self, symbol: str, close: np.ndarray,
                           high: np.ndarray, low: np.ndarray,
                           volume: np.ndarray) -> None:
        """Generate price-action signals from OHLCV data."""
        if len(close) < 30:
            return

        # Trend signal: price above/below SMA(50)
        sma50 = np.mean(close[-50:]) if len(close) >= 50 else np.mean(close)
        trend_score = (close[-1] / sma50 - 1) * 10  # amplify
        self.inject_raw(symbol, "price_trend", "price_action",
                        close[-1], max(-1, min(1, trend_score)), 0.9,
                        f"Price vs SMA50: {close[-1]:.2f} vs {sma50:.2f}")

        # Volume signal: above/below average
        avg_vol = np.mean(volume[-20:])
        vol_ratio = volume[-1] / max(avg_vol, 1)
        vol_score = (vol_ratio - 1) * 0.5  # above avg = bullish momentum
        self.inject_raw(symbol, "volume_momentum", "price_action",
                        vol_ratio, max(-1, min(1, vol_score)), 0.8,
                        f"Volume ratio: {vol_ratio:.2f}x avg")

        # Volatility signal: expanding vol = caution
        returns = np.diff(np.log(close[-30:])) if len(close) >= 30 else np.array([0])
        vol = np.std(returns) if len(returns) > 1 else 0
        vol_20 = np.std(np.diff(np.log(close[-20:]))) if len(close) >= 20 else vol
        vol_change = (vol / max(vol_20, 1e-9)) - 1
        vol_caution = -vol_change * 0.5  # expanding vol = bearish signal
        self.inject_raw(symbol, "volatility_regime", "price_action",
                        vol, max(-1, min(1, vol_caution)), 0.7,
                        f"Vol change: {vol_change:.1%}")

        # Support/resistance
        recent_high = np.max(high[-20:])
        recent_low = np.min(low[-20:])
        price = close[-1]
        range_pos = (price - recent_low) / max(recent_high - recent_low, 1e-9)
        # Near support (range_pos < 0.2) = bullish; near resistance (> 0.8) = bearish
        sr_score = (0.5 - range_pos) * 2
        self.inject_raw(symbol, "support_resistance", "technical",
                        range_pos, max(-1, min(1, sr_score)), 0.75,
                        f"Range position: {range_pos:.0%}")

    def process_funding_rate(self, symbol: str, funding_rate: float) -> None:
        """Generate signal from perpetual futures funding rate."""
        # Positive funding = longs pay shorts (market overleveraged long = bearish contrarian)
        # Negative funding = shorts pay longs (market overleveraged short = bullish contrarian)
        score = -funding_rate * 100  # contrarian: high funding = bearish
        self.inject_raw(symbol, "funding_rate", "derivatives",
                        funding_rate, max(-1, min(1, score)), 0.85,
                        f"Funding: {funding_rate:.4%}")

    def process_open_interest(self, symbol: str, oi_change_pct: float) -> None:
        """Rising OI + rising price = trend confirmation. Rising OI + falling price = bearish."""
        self.inject_raw(symbol, "open_interest", "derivatives",
                        oi_change_pct, max(-1, min(1, oi_change_pct * 0.5)), 0.7,
                        f"OI change: {oi_change_pct:.1%}")

    def process_liquidations(self, symbol: str, long_liq_usd: float, short_liq_usd: float) -> None:
        """Large long liquidations = bearish cascade. Large short liquidations = bullish squeeze."""
        net = short_liq_usd - long_liq_usd  # positive = more shorts liquidated = bullish
        total = long_liq_usd + short_liq_usd
        if total < 100:
            return
        score = net / max(total, 1) * 2
        self.inject_raw(symbol, "liquidations", "derivatives",
                        net, max(-1, min(1, score)), 0.8,
                        f"Liq net: ${net:,.0f} (long=${long_liq_usd:,.0f}, short=${short_liq_usd:,.0f})")

    def process_whale_flow(self, symbol: str, exchange_inflow_usd: float,
                           exchange_outflow_usd: float) -> None:
        """Whale deposits to exchange = selling pressure. Withdrawals = accumulation."""
        net_flow = exchange_outflow_usd - exchange_inflow_usd  # positive = withdrawal = bullish
        total = exchange_inflow_usd + exchange_outflow_usd
        if total < 100:
            return
        score = net_flow / max(total, 1)
        self.inject_raw(symbol, "whale_flow", "onchain",
                        net_flow, max(-1, min(1, score)), 0.75,
                        f"Net flow: ${net_flow:,.0f}")

    def process_stablecoin_flow(self, mint_usd: float, burn_usd: float) -> None:
        """USDT/USDC minting = new capital entering crypto = bullish."""
        net = mint_usd - burn_usd
        score = net / max(abs(net) + 1e6, 1) * 2  # normalise
        # Apply to all symbols (market-wide signal)
        for symbol in list(self._signals.keys()) or ["BTC/USD"]:
            self.inject_raw(symbol, "stablecoin_flow", "onchain",
                            net, max(-1, min(1, score)), 0.7,
                            f"Stablecoin net: ${net:,.0f}")

    def process_fear_greed(self, index_value: int) -> None:
        """Fear & Greed: 0=extreme fear (contrarian bullish), 100=extreme greed (contrarian bearish)."""
        # Contrarian: extreme fear = buy, extreme greed = sell
        score = (50 - index_value) / 50  # 0→+1, 50→0, 100→-1
        for symbol in list(self._signals.keys()) or ["BTC/USD"]:
            self.inject_raw(symbol, "fear_greed", "sentiment",
                            index_value, max(-1, min(1, score)), 0.65,
                            f"Fear&Greed: {index_value}")

    def process_macro(self, dxy: float, dxy_prev: float,
                      sp500_ret: float = 0.0, gold_ret: float = 0.0) -> None:
        """Macro signals: DXY up = crypto down. S&P up = risk-on = crypto up."""
        dxy_change = (dxy / max(dxy_prev, 1) - 1) * 100
        dxy_score = -dxy_change * 0.3  # DXY up = bearish for crypto
        sp_score = sp500_ret * 2  # risk-on = bullish
        gold_score = gold_ret * 0.5  # mild correlation
        macro_score = dxy_score * 0.5 + sp_score * 0.3 + gold_score * 0.2
        for symbol in list(self._signals.keys()) or ["BTC/USD"]:
            self.inject_raw(symbol, "macro_dxy", "macro",
                            dxy, max(-1, min(1, macro_score)), 0.6,
                            f"DXY: {dxy:.1f} ({dxy_change:+.1f}%)")

    def process_options_flow(self, symbol: str, put_call_ratio: float,
                             iv_skew: float = 0.0, max_pain: float = 0.0) -> None:
        """Options flow: high put/call ratio = bearish hedge. IV skew = direction bias."""
        pcr_score = (1 - put_call_ratio) * 0.5  # PCR > 1 = bearish, < 1 = bullish
        skew_score = -iv_skew * 0.3  # positive skew = downside fear = bearish
        combined = pcr_score * 0.6 + skew_score * 0.4
        self.inject_raw(symbol, "options_flow", "derivatives",
                        put_call_ratio, max(-1, min(1, combined)), 0.7,
                        f"P/C ratio: {put_call_ratio:.2f}, IV skew: {iv_skew:.2f}")

    def process_btc_dominance(self, dominance_pct: float, prev_dominance: float) -> None:
        """BTC dominance rising = altcoins bearish. Falling = altcoin season."""
        change = dominance_pct - prev_dominance
        # For BTC: rising dominance = slightly bullish
        self.inject_raw("BTC/USD", "btc_dominance", "technical",
                        dominance_pct, max(-1, min(1, change * 0.2)), 0.6,
                        f"BTC dom: {dominance_pct:.1f}% ({change:+.1f}%)")
        # For alts: rising BTC dominance = bearish
        for symbol in self._signals:
            if symbol != "BTC/USD":
                self.inject_raw(symbol, "btc_dominance", "technical",
                                dominance_pct, max(-1, min(1, -change * 0.3)), 0.6,
                                f"BTC dom: {dominance_pct:.1f}% ({change:+.1f}%)")

    # ──────────────────────────────────────────────────────────────────────
    # Queries
    # ──────────────────────────────────────────────────────────────────────

    def get_all_intelligence(self) -> Dict[str, MarketIntelligence]:
        """Get intelligence for all tracked symbols."""
        return {sym: self.compute(sym) for sym in self._signals}

    def get_strongest_signal(self, symbol: str) -> Optional[DataSignal]:
        """Get the single strongest signal for a symbol."""
        intel = self.compute(symbol)
        if not intel.signals:
            return None
        return max(intel.signals, key=lambda s: abs(s.score) * s.confidence)

    def get_signal_agreement(self, symbol: str) -> float:
        """Get signal agreement ratio (0-1). Higher = more conviction."""
        intel = self.compute(symbol)
        if intel.signal_count == 0:
            return 0.0
        return intel.agreeing_signals / intel.signal_count

    def get_stats(self) -> Dict[str, Any]:
        return {
            "symbols_tracked": len(self._signals),
            "total_signals": sum(len(sigs) for sigs in self._signals.values()),
            "source_weights": dict(self._source_weights),
            "categories": list(self._category_weights.keys()),
        }
