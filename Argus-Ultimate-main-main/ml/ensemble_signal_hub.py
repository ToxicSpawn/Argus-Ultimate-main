"""
Ensemble Signal Hub — central aggregation of all dormant signal sources.

Polls FearGreedIndex, LLMSignal, WhaleTracker, NewsSentimentSignal,
AlphaModel, VolatilityForecaster, and FundingRatePredictor each cycle and
returns a weighted composite bias that feeds the signal stacker.

Design principles
-----------------
- Every import is guarded with try/except; the module never fails to import.
- Results are cached per symbol for CACHE_TTL seconds (default 60 s) to avoid
  hammering external APIs on every cycle.
- Failed/missing sources are silently skipped with DEBUG logging.
- VolatilityForecaster does NOT contribute to the directional composite; it
  only affects size_multiplier.
- FundingRatePredictor is informational only (weight 0 by default) but its
  reading is recorded in the sources dict.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency imports — each wrapped independently
# ---------------------------------------------------------------------------

try:
    from data.fear_greed import FearGreedIndex as _FearGreedIndex
    _FG_AVAILABLE = True
except Exception:
    _FearGreedIndex = None  # type: ignore[assignment,misc]
    _FG_AVAILABLE = False
    logger.debug("ensemble_signal_hub: FearGreedIndex unavailable")

try:
    from ml.llm_signal import LLMSignalGenerator as _LLMSignalGenerator
    _LLM_AVAILABLE = True
except Exception:
    _LLMSignalGenerator = None  # type: ignore[assignment,misc]
    _LLM_AVAILABLE = False
    logger.debug("ensemble_signal_hub: LLMSignalGenerator unavailable")

try:
    from data.onchain.whale_tracker import WhaleTracker as _WhaleTracker
    _WHALE_AVAILABLE = True
except Exception:
    _WhaleTracker = None  # type: ignore[assignment,misc]
    _WHALE_AVAILABLE = False
    logger.debug("ensemble_signal_hub: WhaleTracker unavailable")

try:
    from data.sentiment.news_signal import NewsSentimentSignal as _NewsSentimentSignal
    _NEWS_AVAILABLE = True
except Exception:
    _NewsSentimentSignal = None  # type: ignore[assignment,misc]
    _NEWS_AVAILABLE = False
    logger.debug("ensemble_signal_hub: NewsSentimentSignal unavailable")

try:
    from ml.alpha_model import AlphaModel as _AlphaModel
    _ALPHA_AVAILABLE = True
except Exception:
    _AlphaModel = None  # type: ignore[assignment,misc]
    _ALPHA_AVAILABLE = False
    logger.debug("ensemble_signal_hub: AlphaModel unavailable")

try:
    from ml.volatility_forecaster import VolatilityForecaster as _VolatilityForecaster
    _VOL_AVAILABLE = True
except Exception:
    _VolatilityForecaster = None  # type: ignore[assignment,misc]
    _VOL_AVAILABLE = False
    logger.debug("ensemble_signal_hub: VolatilityForecaster unavailable")

try:
    from data.funding_predictor import FundingRatePredictor as _FundingRatePredictor
    _FUNDING_AVAILABLE = True
except Exception:
    _FundingRatePredictor = None  # type: ignore[assignment,misc]
    _FUNDING_AVAILABLE = False
    logger.debug("ensemble_signal_hub: FundingRatePredictor unavailable")

try:
    from data.onchain.chain_metrics import ChainMetricsProvider as _ChainMetricsProvider
    _CHAIN_METRICS_AVAILABLE = True
except Exception:
    _ChainMetricsProvider = None  # type: ignore[assignment,misc]
    _CHAIN_METRICS_AVAILABLE = False
    logger.debug("ensemble_signal_hub: ChainMetricsProvider unavailable")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_TTL: int = 60  # seconds

# Volatility regime thresholds that reduce size multiplier
_VOL_REDUCE_REGIMES = {"HIGH", "EXTREME", "ELEVATED"}  # cover both naming conventions

# Size multiplier range
_SIZE_MIN = 0.5
_SIZE_MAX = 1.5
_SIZE_NEUTRAL = 1.0

# Default weights (vol_regime excluded from directional composite)
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "fear_greed": 0.13,
    "llm": 0.18,
    "whale": 0.13,
    "news": 0.09,
    "alpha": 0.20,
    "vol_regime": 0.10,  # reserved but not used in directional composite
    "funding": 0.00,
    "chain_metrics": 0.07,
    "graph": 0.10,  # Phase W3: GCN/GAT signal from asset correlation graph
}

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class EnsembleSignal:
    """Aggregated signal from all enabled sources."""

    composite: float        # weighted directional bias, clamped to [-1.0, +1.0]
    confidence: float       # 0–1, fraction of enabled sources that responded
    size_multiplier: float  # position size adjustment [0.5, 1.5]
    regime_bias: str        # human-readable label: BULLISH / BEARISH / NEUTRAL
    sources: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


def _neutral_signal() -> EnsembleSignal:
    """Return a safe neutral EnsembleSignal."""
    return EnsembleSignal(
        composite=0.0,
        confidence=0.0,
        size_multiplier=_SIZE_NEUTRAL,
        regime_bias="NEUTRAL",
        sources={},
    )


# ---------------------------------------------------------------------------
# Hub
# ---------------------------------------------------------------------------


class EnsembleSignalHub:
    """
    Central signal aggregation hub.

    Instantiate once; call ``update()`` each trading cycle.

    Parameters
    ----------
    config : dict | None
        Section from ``unified_config.yaml`` → ``ensemble_signal_hub``.
        Supports keys: ``cache_ttl``, ``weights`` (nested dict),
        ``enabled`` dict (per-source boolean).
    fear_greed : optional pre-built FearGreedIndex
    llm : optional pre-built LLMSignalGenerator
    whale : optional pre-built WhaleTracker
    news : optional pre-built NewsSentimentSignal
    alpha : optional pre-built AlphaModel
    vol : optional pre-built VolatilityForecaster
    funding : optional pre-built FundingRatePredictor
    chain_metrics : optional pre-built ChainMetricsProvider
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        fear_greed: Any = None,
        llm: Any = None,
        whale: Any = None,
        news: Any = None,
        alpha: Any = None,
        vol: Any = None,
        funding: Any = None,
        chain_metrics: Any = None,
    ) -> None:
        cfg = config or {}

        self._cache_ttl: int = int(cfg.get("cache_ttl", CACHE_TTL))

        # --- FIX 27: Configurable ensemble thresholds ---
        self._bullish_threshold: float = float(cfg.get("bullish_threshold", 0.5))
        self._strong_composite_threshold: float = float(cfg.get("strong_composite_threshold", 0.3))
        self._strong_agreement_threshold: float = float(cfg.get("strong_agreement_threshold", 0.7))

        # --- weights ---
        raw_weights = cfg.get("weights", {})
        self._weights: Dict[str, float] = {
            **_DEFAULT_WEIGHTS,
            **{k: float(v) for k, v in raw_weights.items()},
        }

        # --- enabled flags ---
        enabled_cfg = cfg.get("enabled", {})
        self._enabled: Dict[str, bool] = {
            "fear_greed": bool(enabled_cfg.get("fear_greed", True)),
            "llm": bool(enabled_cfg.get("llm", True)),
            "whale": bool(enabled_cfg.get("whale", True)),
            "news": bool(enabled_cfg.get("news", True)),
            "alpha": bool(enabled_cfg.get("alpha", True)),
            "vol_regime": bool(enabled_cfg.get("vol_regime", True)),
            "funding": bool(enabled_cfg.get("funding", True)),
            "chain_metrics": bool(enabled_cfg.get("chain_metrics", True)),
        }

        # --- source instances ---
        self._fg: Any = fear_greed or (
            _FearGreedIndex() if _FG_AVAILABLE else None
        )
        self._llm: Any = llm or (
            _LLMSignalGenerator() if _LLM_AVAILABLE else None
        )
        self._whale: Any = whale or (
            _WhaleTracker() if _WHALE_AVAILABLE else None
        )
        self._news: Any = news or (
            _NewsSentimentSignal() if _NEWS_AVAILABLE else None
        )
        self._alpha: Any = alpha or (
            _AlphaModel(min_bars=1) if _ALPHA_AVAILABLE else None
        )
        self._vol: Any = vol or (
            _VolatilityForecaster() if _VOL_AVAILABLE else None
        )
        self._funding: Any = funding or (
            _FundingRatePredictor() if _FUNDING_AVAILABLE else None
        )
        self._chain_metrics: Any = chain_metrics or (
            _ChainMetricsProvider() if _CHAIN_METRICS_AVAILABLE else None
        )

        # Cache: symbol → (timestamp, EnsembleSignal)
        self._cache: Dict[str, tuple] = {}

        # Last source readings for snapshot()
        self._last_source_values: Dict[str, Dict[str, Any]] = {}

        # Signal quality metrics from last _compute() call
        self._last_signal_quality: Dict[str, Any] = {}

        # Phase W3: graph signal injection (per-symbol)
        # Populated by ``push_graph_signal()`` and consumed in ``_compute()``.
        self._graph_signals: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        symbol: str,
        prices: Sequence[float],
        regime: str = "UNKNOWN",
    ) -> EnsembleSignal:
        """
        Poll all enabled sources and compute composite signal.

        Caches result for ``cache_ttl`` seconds per symbol.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. ``"BTC/USD"``.
        prices : sequence of float
            Recent close prices (newest last).
        regime : str
            Current market regime label from regime detector.
        """
        now = time.time()
        cached = self._cache.get(symbol)
        if cached is not None and (now - cached[0]) < self._cache_ttl:
            return cached[1]

        signal = self._compute(symbol, prices, regime)
        self._cache[symbol] = (now, signal)
        return signal

    def get_last(self, symbol: str) -> EnsembleSignal:
        """Return the most recent cached signal, or a neutral default."""
        cached = self._cache.get(symbol)
        if cached is not None:
            return cached[1]
        return _neutral_signal()

    def get_signal_quality(self) -> Dict[str, Any]:
        """
        Return signal quality metrics from the most recent computation.

        Returns dict with:
            agreement_ratio (0-1): fraction of sources agreeing on direction
            conflict_score (0-1): magnitude of disagreement between bull/bear sources
            strongest_source: which source has highest absolute signal
            strongest_value: the value of the strongest source
            recommendation: "strong", "moderate", "weak", or "conflicted"
            n_sources: total number of responding sources
            n_agreeing: how many sources agree with composite direction
            conflict_detected: whether strong bull and bear sources both exist
        """
        if not self._last_signal_quality:
            return {
                "agreement_ratio": 0.0,
                "conflict_score": 0.0,
                "strongest_source": "",
                "strongest_value": 0.0,
                "recommendation": "weak",
                "n_sources": 0,
                "n_agreeing": 0,
                "conflict_detected": False,
            }
        return dict(self._last_signal_quality)

    def get_size_multiplier(self, symbol: str) -> float:
        """
        Return the size multiplier for the most recent signal.

        Maps composite signal strength to a position size adjustment
        in [0.5, 1.5].  Also considers vol regime.
        """
        return self.get_last(symbol).size_multiplier

    # ------------------------------------------------------------------
    # Phase W3 — graph signal injection
    # ------------------------------------------------------------------

    def push_graph_signal(
        self,
        symbol: str,
        graph_signal: float,
        *,
        confidence: float = 1.0,
        source: str = "gcn",
    ) -> None:
        """
        Push a graph-derived directional signal for ``symbol``.

        Called by a graph-signal producer (typically a GCN or GAT forward
        pass over the asset correlation graph). The signal is consumed on
        the next ``update()`` call for this symbol.

        Parameters
        ----------
        symbol : str
        graph_signal : float
            Directional value in [-1, +1]. Positive = bullish, negative = bearish.
        confidence : float, default 1.0
            Confidence in [0, 1]. Scales the signal before fusion.
        source : str, default "gcn"
            Tag identifying which graph model produced the signal
            ("gcn" or "gat").
        """
        clipped = max(-1.0, min(1.0, float(graph_signal) * float(confidence)))
        self._graph_signals[symbol] = {
            "signal": clipped,
            "confidence": float(confidence),
            "source": str(source),
            "timestamp": time.time(),
        }

    def clear_graph_signal(self, symbol: Optional[str] = None) -> None:
        """Clear graph signal for one symbol or all."""
        if symbol is None:
            self._graph_signals.clear()
        else:
            self._graph_signals.pop(symbol, None)

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a dict of all source statuses and last cached values.

        Useful for monitoring dashboards and the ``snapshot()`` API used
        by the signal stacker.
        """
        out: Dict[str, Any] = {
            "sources": {
                "fear_greed": {
                    "available": self._fg is not None,
                    "enabled": self._enabled["fear_greed"],
                    "weight": self._weights["fear_greed"],
                },
                "llm": {
                    "available": self._llm is not None,
                    "enabled": self._enabled["llm"],
                    "weight": self._weights["llm"],
                },
                "whale": {
                    "available": self._whale is not None,
                    "enabled": self._enabled["whale"],
                    "weight": self._weights["whale"],
                },
                "news": {
                    "available": self._news is not None,
                    "enabled": self._enabled["news"],
                    "weight": self._weights["news"],
                },
                "alpha": {
                    "available": self._alpha is not None,
                    "enabled": self._enabled["alpha"],
                    "weight": self._weights["alpha"],
                },
                "vol_regime": {
                    "available": self._vol is not None,
                    "enabled": self._enabled["vol_regime"],
                    "weight": self._weights["vol_regime"],
                },
                "funding": {
                    "available": self._funding is not None,
                    "enabled": self._enabled["funding"],
                    "weight": self._weights["funding"],
                },
                "chain_metrics": {
                    "available": self._chain_metrics is not None,
                    "enabled": self._enabled.get("chain_metrics", False),
                    "weight": self._weights.get("chain_metrics", 0.08),
                },
            },
            "cache": {
                symbol: {
                    "age_s": round(time.time() - ts, 1),
                    "composite": sig.composite,
                    "confidence": sig.confidence,
                    "size_multiplier": sig.size_multiplier,
                    "regime_bias": sig.regime_bias,
                }
                for symbol, (ts, sig) in self._cache.items()
            },
            "last_source_values": dict(self._last_source_values),
        }
        return out

    # ------------------------------------------------------------------
    # Adaptive weight management (FIX 7)
    # ------------------------------------------------------------------

    def update_source_weights(self, source_pnl: Dict[str, float]) -> None:
        """
        Adjust source weights using EMA based on realized P&L per source.

        Winning sources get more weight, losing sources get less.
        Weights are clamped to [MIN_WEIGHT, MAX_WEIGHT] and normalized to sum=1.0.

        Parameters
        ----------
        source_pnl : dict
            Mapping of source name -> realized P&L (positive = profitable).
        """
        if not source_pnl:
            return

        alpha = _DEFAULT_EMA_ALPHA

        for source, pnl in source_pnl.items():
            if source not in self._weights:
                continue
            current_w = self._weights[source]
            # EMA update: move weight toward higher value if profitable, lower if not
            # Scale factor: +10% per unit of PnL, capped by clamp below
            adjustment = alpha * pnl
            new_w = current_w + adjustment
            # Clamp to [min, max]
            new_w = max(_MIN_ADAPTIVE_WEIGHT, min(_MAX_ADAPTIVE_WEIGHT, new_w))
            self._weights[source] = new_w

        # Normalize so all weights sum to 1.0
        # FIX #19: Guard against NaN/Inf before normalization
        import math as _math
        for k in list(self._weights):
            v = self._weights[k]
            if not isinstance(v, (int, float)) or _math.isnan(v) or _math.isinf(v) or v < 0:
                self._weights[k] = _DEFAULT_WEIGHTS.get(k, 0.1)
        total = sum(self._weights.values())
        if total > 0:
            for k in self._weights:
                self._weights[k] = self._weights[k] / total

        logger.debug(
            "ensemble_signal_hub: adaptive weight update — weights=%s",
            {k: round(v, 4) for k, v in self._weights.items()},
        )

    def get_weights(self) -> Dict[str, float]:
        """Return a copy of current source weights."""
        return dict(self._weights)

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute(
        self,
        symbol: str,
        prices: Sequence[float],
        regime: str,
    ) -> EnsembleSignal:
        """Collect readings, compute weighted composite, build EnsembleSignal."""
        contributions: Dict[str, float] = {}  # source → raw value [-1,+1]
        weights_used: Dict[str, float] = {}
        source_meta: Dict[str, Any] = {}

        # ----------------------------------------------------------------
        # 1. Fear & Greed (async → run in executor or sync fallback)
        # ----------------------------------------------------------------
        if self._enabled["fear_greed"] and self._fg is not None:
            try:
                reading = self._run_async(self._fg.get())
                value = reading.value if reading is not None else 50
                # Contrarian: fear (low) → bullish (+1), greed (high) → bearish (-1)
                # Linear map: 0→+1, 50→0, 100→-1
                fg_signal = 1.0 - (value / 50.0)
                fg_signal = max(-1.0, min(1.0, fg_signal))
                contributions["fear_greed"] = fg_signal
                weights_used["fear_greed"] = self._weights["fear_greed"]
                source_meta["fear_greed"] = {
                    "value": value,
                    "signal": round(fg_signal, 4),
                    "classification": getattr(reading, "classification", ""),
                }
            except Exception:
                logger.debug(
                    "ensemble_signal_hub: fear_greed source failed", exc_info=True
                )

        # ----------------------------------------------------------------
        # 2. LLM Signal
        # ----------------------------------------------------------------
        if self._enabled["llm"] and self._llm is not None:
            try:
                # Use inference_timeout from the LLMSignalGenerator (primary guard
                # via asyncio.wait_for inside generate_signal) plus a small extra
                # buffer for the thread overhead as a secondary guard.
                _llm_timeout = getattr(self._llm, "inference_timeout", 10.0)
                _thread_timeout = _llm_timeout + 2.0  # secondary defence-in-depth
                llm_result = self._run_async(
                    self._llm.generate_signal(
                        symbol=symbol,
                        regime=regime,
                        price_data=list(prices)[-10:],
                    ),
                    timeout=_thread_timeout,
                )
                # LLMSignal has .as_numeric property
                llm_val = float(llm_result.as_numeric)
                contributions["llm"] = llm_val
                weights_used["llm"] = self._weights["llm"]
                source_meta["llm"] = {
                    "direction": llm_result.direction,
                    "confidence": round(llm_result.confidence, 4),
                    "signal": llm_val,
                }
            except Exception:
                logger.debug(
                    "ensemble_signal_hub: llm source failed", exc_info=True
                )

        # ----------------------------------------------------------------
        # 3. Whale Tracker
        # ----------------------------------------------------------------
        if self._enabled["whale"] and self._whale is not None:
            try:
                # Extract asset from symbol (e.g. "BTC/USD" → "BTC")
                asset = symbol.split("/")[0] if "/" in symbol else symbol
                whale_sig = self._whale.get_signal(asset)
                direction_map = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}
                whale_val = direction_map.get(whale_sig.direction, 0.0)
                # Scale by signal strength
                whale_val *= whale_sig.strength
                contributions["whale"] = whale_val
                weights_used["whale"] = self._weights["whale"]
                source_meta["whale"] = {
                    "direction": whale_sig.direction,
                    "strength": round(whale_sig.strength, 4),
                    "signal": round(whale_val, 4),
                }
            except Exception:
                logger.debug(
                    "ensemble_signal_hub: whale source failed", exc_info=True
                )

        # ----------------------------------------------------------------
        # 4. News Sentiment
        # ----------------------------------------------------------------
        if self._enabled["news"] and self._news is not None:
            try:
                news_result = self._run_async(self._news.get_signal(symbol))
                if news_result is not None:
                    # SentimentScore.aggregate_score is already in [-1, +1]
                    if hasattr(news_result, "aggregate_score"):
                        news_val = float(news_result.aggregate_score)
                    elif hasattr(news_result, "score"):
                        news_val = float(news_result.score)
                    else:
                        news_val = float(news_result)
                    news_val = max(-1.0, min(1.0, news_val))
                    contributions["news"] = news_val
                    weights_used["news"] = self._weights["news"]
                    source_meta["news"] = {"signal": round(news_val, 4)}
            except Exception:
                logger.debug(
                    "ensemble_signal_hub: news source failed", exc_info=True
                )

        # ----------------------------------------------------------------
        # 5. Alpha Model
        # ----------------------------------------------------------------
        if self._enabled["alpha"] and self._alpha is not None:
            try:
                # Feed price history into the model
                for price in prices:
                    self._alpha.update(symbol, float(price))
                alpha_score = self._alpha.score(symbol)
                alpha_val = float(alpha_score.composite)
                alpha_val = max(-1.0, min(1.0, alpha_val))
                contributions["alpha"] = alpha_val
                weights_used["alpha"] = self._weights["alpha"]
                source_meta["alpha"] = {
                    "composite": round(alpha_val, 4),
                    "signal": alpha_score.signal,
                    "confidence": round(alpha_score.confidence, 4),
                }
            except Exception:
                logger.debug(
                    "ensemble_signal_hub: alpha source failed", exc_info=True
                )

        # ----------------------------------------------------------------
        # 5b. Graph signal (Phase W3 — GCN/GAT on asset correlation graph)
        # ----------------------------------------------------------------
        graph_record = self._graph_signals.get(symbol)
        if graph_record is not None:
            # Staleness guard: graph signals older than 5 minutes are ignored
            age_s = time.time() - float(graph_record.get("timestamp", 0.0))
            if age_s < 300.0:
                graph_val = float(graph_record.get("signal", 0.0))
                graph_val = max(-1.0, min(1.0, graph_val))
                contributions["graph"] = graph_val
                weights_used["graph"] = self._weights.get("graph", 0.10)
                source_meta["graph"] = {
                    "signal": round(graph_val, 4),
                    "confidence": round(float(graph_record.get("confidence", 1.0)), 4),
                    "producer": graph_record.get("source", "gcn"),
                    "age_s": round(age_s, 1),
                }

        # ----------------------------------------------------------------
        # 6. Volatility Regime (size only — NOT directional)
        # ----------------------------------------------------------------
        vol_regime = "NORMAL"
        if self._enabled["vol_regime"] and self._vol is not None:
            try:
                # Feed prices into forecaster
                for price in prices:
                    self._vol.update(symbol, float(price))
                vf = self._vol.forecast(symbol)
                vol_regime = vf.regime
                source_meta["vol_regime"] = {
                    "regime": vol_regime,
                    "forecast_vol_1d": round(vf.forecast_vol_1d, 4),
                }
            except Exception:
                logger.debug(
                    "ensemble_signal_hub: vol_regime source failed", exc_info=True
                )

        # ----------------------------------------------------------------
        # 7. Funding Rate Predictor (informational — weight 0 by default)
        # ----------------------------------------------------------------
        if self._enabled["funding"] and self._funding is not None:
            try:
                fp = self._funding.predict(symbol)
                funding_val = 0.0
                if self._weights["funding"] > 0:
                    # Positive rate → longs pay → slightly bearish
                    funding_val = -float(fp.predicted_rate_pct) * 10.0
                    funding_val = max(-1.0, min(1.0, funding_val))
                    contributions["funding"] = funding_val
                    weights_used["funding"] = self._weights["funding"]
                source_meta["funding"] = {
                    "predicted_rate_pct": round(fp.predicted_rate_pct, 6),
                    "direction": fp.direction,
                    "signal": round(funding_val, 4),
                }
            except Exception:
                logger.debug(
                    "ensemble_signal_hub: funding source failed", exc_info=True
                )

        # ----------------------------------------------------------------
        # 8. Chain Metrics (on-chain: MVRV, SOPR, exchange flows)
        # ----------------------------------------------------------------
        if self._enabled.get("chain_metrics") and self._chain_metrics is not None:
            try:
                chain_snapshot = self._run_async(self._chain_metrics.get_metrics())
                if chain_snapshot is not None:
                    chain_val = float(chain_snapshot.signal_bias)
                    chain_val = max(-1.0, min(1.0, chain_val))
                    contributions["chain_metrics"] = chain_val
                    weights_used["chain_metrics"] = self._weights.get("chain_metrics", 0.08)
                    source_meta["chain_metrics"] = {
                        "mvrv_zscore": round(chain_snapshot.mvrv_zscore, 4),
                        "sopr": round(chain_snapshot.sopr, 4),
                        "net_exchange_flow_btc": round(chain_snapshot.net_exchange_flow_btc, 2),
                        "signal": round(chain_val, 4),
                    }
            except Exception:
                logger.debug(
                    "ensemble_signal_hub: chain_metrics source failed", exc_info=True
                )

        # ----------------------------------------------------------------
        # Weighted composite
        # ----------------------------------------------------------------
        composite = self._weighted_composite(contributions, weights_used)

        # ----------------------------------------------------------------
        # Conflict detection & minimum source agreement
        # ----------------------------------------------------------------
        bullish_sources = {k: v for k, v in contributions.items() if v > self._bullish_threshold}
        bearish_sources = {k: v for k, v in contributions.items() if v < -self._bullish_threshold}
        conflict_detected = bool(bullish_sources and bearish_sources)

        if conflict_detected:
            # Strong bullish and strong bearish disagree — reduce confidence by 40%
            composite *= 0.6
            composite = max(-1.0, min(1.0, composite))

        # Direction agreement: count sources pointing same direction as composite
        if contributions and abs(composite) > 0.01:
            same_dir = sum(
                1 for v in contributions.values()
                if (v > 0 and composite > 0) or (v < 0 and composite < 0)
            )
            n_sources_total = len(contributions)
            agreement_ratio = same_dir / n_sources_total if n_sources_total > 0 else 0.0
        else:
            same_dir = 0
            n_sources_total = len(contributions)
            agreement_ratio = 0.0

        # Minimum source agreement: need 2+ sources in same direction for strong signal
        if abs(composite) > self._strong_composite_threshold and same_dir < 2:
            # Not enough agreement for a strong signal — dampen
            composite *= 0.5
            composite = max(-1.0, min(1.0, composite))

        # ----------------------------------------------------------------
        # Signal quality metrics (stored for external access)
        # ----------------------------------------------------------------
        conflict_score = 0.0
        if bullish_sources and bearish_sources:
            strongest_bull = max(bullish_sources.values())
            strongest_bear = abs(min(bearish_sources.values()))
            conflict_score = min(1.0, (strongest_bull + strongest_bear) / 2.0)

        strongest_source = ""
        strongest_value = 0.0
        for k, v in contributions.items():
            if abs(v) > abs(strongest_value):
                strongest_value = v
                strongest_source = k

        if conflict_score > self._bullish_threshold:
            recommendation = "conflicted"
        elif agreement_ratio >= self._strong_agreement_threshold and abs(composite) >= self._strong_composite_threshold:
            recommendation = "strong"
        elif agreement_ratio >= 0.5 or abs(composite) >= 0.15:
            recommendation = "moderate"
        else:
            recommendation = "weak"

        self._last_signal_quality = {
            "agreement_ratio": round(agreement_ratio, 4),
            "conflict_score": round(conflict_score, 4),
            "strongest_source": strongest_source,
            "strongest_value": round(strongest_value, 4),
            "recommendation": recommendation,
            "n_sources": n_sources_total,
            "n_agreeing": same_dir,
            "conflict_detected": conflict_detected,
        }

        # ----------------------------------------------------------------
        # Confidence: fraction of enabled+available directional sources
        # that responded (vol and funding are excluded — not directional).
        # ----------------------------------------------------------------
        _directional_sources = ("fear_greed", "llm", "whale", "news", "alpha", "chain_metrics")
        n_enabled = sum(
            1
            for src in _directional_sources
            if self._enabled.get(src)
            and getattr(self, f"_{_SRC_ATTR[src]}") is not None
        )
        n_responded = sum(1 for src in _directional_sources if src in contributions)
        confidence = (n_responded / n_enabled) if n_enabled > 0 else 0.0
        confidence = round(min(1.0, confidence), 4)

        # ----------------------------------------------------------------
        # Size multiplier
        # ----------------------------------------------------------------
        size_multiplier = self._compute_size_multiplier(composite, vol_regime)

        # ----------------------------------------------------------------
        # Regime bias label
        # ----------------------------------------------------------------
        regime_bias = self._label(composite)

        # Store for snapshot
        self._last_source_values[symbol] = source_meta

        return EnsembleSignal(
            composite=round(composite, 6),
            confidence=confidence,
            size_multiplier=round(size_multiplier, 4),
            regime_bias=regime_bias,
            sources=source_meta,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_composite(
        contributions: Dict[str, float],
        weights: Dict[str, float],
    ) -> float:
        """Compute normalised weighted average of source contributions."""
        if not contributions:
            return 0.0
        total_weight = sum(weights.get(k, 0.0) for k in contributions)
        if total_weight < 1e-9:
            # All enabled weights are zero — simple average
            return sum(contributions.values()) / len(contributions)
        weighted_sum = sum(
            contributions[k] * weights.get(k, 0.0) for k in contributions
        )
        composite = weighted_sum / total_weight
        return max(-1.0, min(1.0, composite))

    @staticmethod
    def _compute_size_multiplier(composite: float, vol_regime: str) -> float:
        """
        Map composite signal strength and vol regime to position size factor.

        Base mapping (absolute composite → size):
          0.0  → 1.0  (neutral)
          1.0  → 1.5  (strong signal)
          -1.0 → 0.5  (strong opposing signal — reduce size)

        Wait — directional ambiguity: negative composite means short bias.
        We use |composite| to scale size up (strong signal = bigger),
        but vol regime can push size down regardless of direction.
        """
        strength = abs(composite)
        # Linear interpolation: 0 → neutral, 1 → max
        base = _SIZE_NEUTRAL + strength * (_SIZE_MAX - _SIZE_NEUTRAL)
        base = min(_SIZE_MAX, base)

        # Vol regime override
        if vol_regime in _VOL_REDUCE_REGIMES:
            if vol_regime == "EXTREME":
                base = min(base, 0.5)
            else:
                base = min(base, 0.7)

        return max(_SIZE_MIN, min(_SIZE_MAX, base))

    def _label(self, composite: float) -> str:
        # Uses strong_composite_threshold / 2 as the label boundary
        _label_threshold = self._strong_composite_threshold / 2.0
        if composite >= _label_threshold:
            return "BULLISH"
        if composite <= -_label_threshold:
            return "BEARISH"
        return "NEUTRAL"

    @staticmethod
    def _run_async(coro: Any, timeout: float = 15.0) -> Any:
        """
        Run an awaitable synchronously, regardless of event-loop context.

        Handles both sync values (returned immediately) and coroutines.

        Parameters
        ----------
        coro :
            Coroutine or plain value to resolve.
        timeout : float
            Wall-clock timeout in seconds applied to the background thread
            that runs the coroutine.  This is a defence-in-depth guard; the
            primary timeout for LLM calls is ``LLMSignalGenerator.inference_timeout``
            which is enforced via ``asyncio.wait_for()`` inside
            ``generate_signal()``.  Defaults to 15 s.
        """
        if not asyncio.iscoroutine(coro):
            return coro
        try:
            loop = asyncio.get_running_loop()
            # We are inside a running async loop.  Run in a separate thread
            # to avoid deadlocking the existing loop.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(asyncio.run, coro)
                return future.result(timeout=timeout)
        except RuntimeError:
            # No running loop — safe to call asyncio.run directly.
            return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Adaptive weight constants
# ---------------------------------------------------------------------------

_MIN_ADAPTIVE_WEIGHT: float = 0.05
_MAX_ADAPTIVE_WEIGHT: float = 0.50
_DEFAULT_EMA_ALPHA: float = 0.1


# Mapping from source name → instance attribute suffix
_SRC_ATTR: Dict[str, str] = {
    "fear_greed": "fg",
    "llm": "llm",
    "whale": "whale",
    "news": "news",
    "alpha": "alpha",
    "chain_metrics": "chain_metrics",
}
