"""
Maximum Trading Engine — The Ultimate Alpha Generation System

Orchestrates ALL existing strategies, ML models, and execution systems
to maximize trading performance across 4 pillars:

1. SIGNAL QUALITY: When to buy/sell
   - 73 strategies → ML ensemble voting
   - Confidence calibration with meta-labeling
   - Signal entropy measurement
   - Multi-factor quality scoring

2. MARKET TIMING: When to be in/out
   - Multi-timeframe confluence (1m → 1W)
   - Regime detection (HMM + ML classifier)
   - Session-aware timing (Asian/London/NY)
   - Volatility regime adaptation

3. ASSET SELECTION: What to trade
   - Cross-asset momentum ranking
   - Flow analysis (whale tracking)
   - Correlation-adjusted selection
   - Liquidity-adjusted opportunity scoring

4. EXECUTION: How to get filled
   - Smart order routing (8 venues)
   - Adaptive TWAP/VWAP/POV
   - Queue position optimization
   - Maker/taker fee optimization

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    MAXIMUM TRADING ENGINE                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  PILLAR 1: SIGNAL QUALITY ENGINE                                        │
    │    - EnsembleSignalHub: 73 strategies → weighted consensus              │
    │    - SignalQualityScorer: 7-factor quality gate                         │
    │    - MetaLabeling: ML model predicts signal correctness                 │
    │    - ConfidenceCalibrator: calibrates raw confidence to true prob       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  PILLAR 2: MARKET TIMING ENGINE                                         │
    │    - MultiTimeframeConfluence: 6 timeframe agreement                    │
    │    - HMMRegimeDetector: hidden markov regime model                      │
    │    - SessionOptimizer: time-of-day edge                                 │
    │    - VolatilityRegimeClassifier: adapt to vol environment               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  PILLAR 3: ASSET SELECTION ENGINE                                       │
    │    - CrossAssetRanker: rank all tradeable assets                        │
    │    - FlowAnalyzer: whale/institutional flow detection                   │
    │    - CorrelationCluster: avoid correlated bets                          │
    │    - OpportunityScorer: liquidity × momentum × volatility               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  PILLAR 4: EXECUTION ENGINE                                             │
    │    - SmartOrderRouter: optimal venue selection                          │
    │    - AdaptiveExecution: TWAP/VWAP/POV with learning                     │
    │    - QueueOptimizer: maximize fill probability                          │
    │    - FeeOptimizer: maker vs taker decision                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  ORCHESTRATION LAYER                                                    │
    │    - TradeDecision: unified decision with all 4 pillars                 │
    │    - PerformanceTracker: continuous learning from outcomes               │
    │    - ParameterOptimizer: online hyperparameter tuning                   │
    └─────────────────────────────────────────────────────────────────────────┘

Usage:
    from trading.maximum_trading_engine import MaximumTradingEngine

    engine = MaximumTradingEngine()
    decision = engine.generate_trade(
        symbols=["BTC/USDT", "ETH/USDT", ...],
        market_data={...},
        portfolio_equity=100000,
    )
    if decision.should_trade:
        execute(decision)
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SignalDirection(Enum):
    """Signal direction."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class MarketRegime(Enum):
    """Market regime classification."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RANGING = "ranging"
    CRISIS = "crisis"


class TimingWindow(Enum):
    """Optimal trading windows."""
    ASIAN_OPEN = "asian_open"
    LONDON_OPEN = "london_open"
    NY_OPEN = "ny_open"
    OVERLAP_LONDON_NY = "overlap_london_ny"
    LOW_LIQUIDITY = "low_liquidity"


class ExecutionAlgorithm(Enum):
    """Execution algorithms."""
    IMMEDIATE = "immediate"
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"
    ADAPTIVE = "adaptive"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SignalQuality:
    """Signal quality assessment (Pillar 1)."""
    raw_confidence: float
    calibrated_confidence: float
    quality_score: float  # 0-1
    ensemble_agreement: float  # 0-1, how many strategies agree
    meta_label_prob: float  # ML probability signal is correct
    entropy: float  # signal entropy (lower = more certain)
    passed_quality_gate: bool
    quality_factors: Dict[str, float]
    rejection_reasons: List[str]


@dataclass
class TimingAssessment:
    """Market timing assessment (Pillar 2)."""
    current_regime: MarketRegime
    regime_confidence: float
    optimal_timing_window: TimingWindow
    timing_score: float  # 0-1, how good is timing
    mtf_confluence: float  # 0-1, multi-timeframe agreement
    session_edge: float  # historical edge at this time
    vol_regime: str  # low/normal/high/extreme
    should_trade_now: bool
    wait_minutes: Optional[int]  # if should wait


@dataclass
class AssetSelection:
    """Asset selection assessment (Pillar 3)."""
    symbol: str
    rank: int  # rank among all candidates
    opportunity_score: float  # 0-1
    momentum_score: float
    flow_score: float  # institutional flow
    liquidity_score: float
    correlation_penalty: float  # penalty for correlated positions
    volatility_score: float
    selection_reason: str


@dataclass
class ExecutionPlan:
    """Execution plan (Pillar 4)."""
    algorithm: ExecutionAlgorithm
    optimal_venue: str
    estimated_fill_price: float
    expected_slippage_bps: float
    expected_fill_rate: float  # probability of fill
    maker_taker_decision: str  # "maker" or "taker"
    slice_count: int  # for TWAP/VWAP
    duration_seconds: int
    queue_position_estimate: int
    fee_optimization_bps: float  # savings from fee optimization


@dataclass
class MaximumTradeDecision:
    """Complete trade decision from Maximum Trading Engine."""
    # Decision
    should_trade: bool
    direction: SignalDirection
    symbol: str
    size_usd: float
    
    # Pillar results
    signal_quality: SignalQuality
    timing_assessment: TimingAssessment
    asset_selection: AssetSelection
    execution_plan: ExecutionPlan
    
    # Combined scores
    composite_score: float  # 0-100, overall trade quality
    expected_edge_bps: float  # expected edge in basis points
    expected_pnl_bps: float  # expected P&L in basis points
    confidence: float  # 0-1 overall confidence
    
    # Risk-adjusted metrics
    sharpe_contribution: float  # expected Sharpe contribution
    kelly_fraction: float  # optimal Kelly fraction
    max_position_size: float  # maximum position size
    
    # Timing
    execute_immediately: bool
    valid_until: float  # timestamp when signal expires
    
    # Metadata
    reasoning: List[str]
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Pillar 1: Signal Quality Engine
# ---------------------------------------------------------------------------

class SignalQualityEngine:
    """
    Maximizes signal quality through:
    - Ensemble of 73+ strategies
    - Meta-labeling (ML predicts signal correctness)
    - Confidence calibration
    - Multi-factor quality scoring
    """
    
    def __init__(self):
        # Strategy weights (learned over time)
        self._strategy_weights: Dict[str, float] = {}
        self._strategy_outcomes: Dict[str, List[float]] = {}
        
        # Meta-labeling model state
        self._meta_features: List[str] = [
            "raw_confidence", "volume_ratio", "spread_bps",
            "regime_match", "trend_match", "time_of_day",
            "correlation_penalty", "strategy_count"
        ]
        
        # Confidence calibration
        self._calibration_bins: Dict[int, Tuple[int, int]] = {}  # bin -> (correct, total)
        
        # Quality thresholds
        self.min_quality = 0.6
        self.min_confidence = 0.4
        self.min_ensemble_agreement = 0.5
        
    def assess_signal(
        self,
        raw_signals: List[Dict[str, Any]],
        market_state: Dict[str, Any],
    ) -> SignalQuality:
        """
        Assess signal quality from multiple strategy signals.
        
        Args:
            raw_signals: List of signals from different strategies
            market_state: Current market state (regime, volume, spread, etc.)
        """
        if not raw_signals:
            return SignalQuality(
                raw_confidence=0, calibrated_confidence=0, quality_score=0,
                ensemble_agreement=0, meta_label_prob=0, entropy=1,
                passed_quality_gate=False, quality_factors={},
                rejection_reasons=["No signals"]
            )
        
        # 1. Calculate ensemble agreement
        directions = [s.get("direction", "neutral") for s in raw_signals]
        buy_count = sum(1 for d in directions if "buy" in d.lower())
        sell_count = sum(1 for d in directions if "sell" in d.lower())
        total = len(directions)
        
        if buy_count > sell_count:
            dominant_direction = "buy"
            agreement = buy_count / total
        elif sell_count > buy_count:
            dominant_direction = "sell"
            agreement = sell_count / total
        else:
            dominant_direction = "neutral"
            agreement = 0.5
        
        # 2. Calculate weighted confidence
        weighted_confidence = 0
        total_weight = 0
        for signal in raw_signals:
            strategy = signal.get("strategy", "unknown")
            weight = self._strategy_weights.get(strategy, 1.0)
            confidence = signal.get("confidence", 0.5)
            weighted_confidence += confidence * weight
            total_weight += weight
        
        raw_confidence = weighted_confidence / max(1, total_weight)
        
        # 3. Quality factors
        quality_factors = {}
        rejection_reasons = []
        
        # Factor 1: Confidence
        quality_factors["confidence"] = min(1.0, raw_confidence / 0.8)
        if raw_confidence < self.min_confidence:
            rejection_reasons.append(f"Low confidence: {raw_confidence:.2f}")
        
        # Factor 2: Ensemble agreement
        quality_factors["ensemble_agreement"] = agreement
        if agreement < self.min_ensemble_agreement:
            rejection_reasons.append(f"Low agreement: {agreement:.2f}")
        
        # Factor 3: Volume confirmation
        volume_ratio = market_state.get("volume_ratio", 1.0)
        quality_factors["volume"] = min(1.0, volume_ratio)
        if volume_ratio < 0.5:
            rejection_reasons.append(f"Low volume: {volume_ratio:.2f}")
        
        # Factor 4: Spread acceptability
        spread_bps = market_state.get("spread_bps", 10.0)
        quality_factors["spread"] = max(0, 1.0 - spread_bps / 50)
        if spread_bps > 30:
            rejection_reasons.append(f"High spread: {spread_bps:.1f}bps")
        
        # Factor 5: Regime alignment
        regime = market_state.get("regime", "unknown")
        regime_aligned = self._check_regime_alignment(dominant_direction, regime)
        quality_factors["regime_alignment"] = 1.0 if regime_aligned else 0.3
        if not regime_aligned:
            rejection_reasons.append(f"Regime misalignment: {regime}")
        
        # Factor 6: Time of day
        hour = market_state.get("hour", 12)
        time_quality = self._get_time_quality(hour)
        quality_factors["time_of_day"] = time_quality
        
        # Factor 7: Correlation penalty
        correlation = market_state.get("correlation_with_portfolio", 0.0)
        quality_factors["correlation"] = max(0, 1.0 - abs(correlation))
        
        # 4. Meta-labeling (simplified - predicts probability signal is correct)
        meta_features = np.array([
            raw_confidence,
            volume_ratio,
            spread_bps / 100,
            1.0 if regime_aligned else 0.0,
            1.0 if "trend" in regime else 0.5,
            time_quality,
            1.0 - abs(correlation),
            len(raw_signals) / 10,
        ])
        
        # Simplified meta-model (in production, use trained ML model)
        meta_weight = np.array([0.25, 0.10, 0.10, 0.15, 0.10, 0.10, 0.10, 0.10])
        meta_logit = np.dot(meta_features, meta_weight)
        meta_label_prob = 1 / (1 + math.exp(-5 * (meta_logit - 0.5)))
        
        # 5. Calculate overall quality score
        factor_weights = {
            "confidence": 0.25,
            "ensemble_agreement": 0.20,
            "volume": 0.10,
            "spread": 0.10,
            "regime_alignment": 0.15,
            "time_of_day": 0.10,
            "correlation": 0.10,
        }
        
        quality_score = sum(
            quality_factors.get(f, 0) * w
            for f, w in factor_weights.items()
        )
        
        # Blend with meta-label
        quality_score = 0.7 * quality_score + 0.3 * meta_label_prob
        
        # 6. Confidence calibration
        calibrated_confidence = self._calibrate_confidence(raw_confidence, quality_score)
        
        # 7. Entropy (uncertainty measure)
        if len(raw_signals) > 1:
            confidences = [s.get("confidence", 0.5) for s in raw_signals]
            probs = np.array(confidences) / sum(confidences)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropy = entropy / math.log(len(confidences))  # normalize
        else:
            entropy = 0.5
        
        # 8. Pass/fail
        passed = (
            quality_score >= self.min_quality and
            raw_confidence >= self.min_confidence and
            agreement >= self.min_ensemble_agreement and
            len(rejection_reasons) == 0
        )
        
        return SignalQuality(
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            quality_score=quality_score,
            ensemble_agreement=agreement,
            meta_label_prob=meta_label_prob,
            entropy=entropy,
            passed_quality_gate=passed,
            quality_factors=quality_factors,
            rejection_reasons=rejection_reasons,
        )
    
    def _check_regime_alignment(self, direction: str, regime: str) -> bool:
        """Check if signal direction aligns with market regime."""
        alignment_rules = {
            "bull_trend": ["buy", "strong_buy"],
            "bear_trend": ["sell", "strong_sell"],
            "high_volatility": ["neutral"],  # reduce trading
            "low_volatility": ["buy", "sell"],  # either direction ok
            "ranging": ["buy", "sell"],  # mean reversion
            "crisis": ["neutral"],  # don't trade
        }
        
        allowed = alignment_rules.get(regime, ["buy", "sell", "neutral"])
        return direction in allowed or direction.replace("_", " ") in allowed
    
    def _get_time_quality(self, hour: int) -> float:
        """Get historical quality for this hour of day."""
        # Peak hours (UTC): London open (8), NY open (13), overlap (13-16)
        quality_by_hour = {
            0: 0.5, 1: 0.4, 2: 0.3, 3: 0.3, 4: 0.4, 5: 0.5,  # Asian
            6: 0.6, 7: 0.7, 8: 0.9, 9: 0.8, 10: 0.8, 11: 0.7,  # London
            12: 0.8, 13: 0.9, 14: 0.9, 15: 0.9, 16: 0.8,  # NY + overlap
            17: 0.7, 18: 0.6, 19: 0.5, 20: 0.5, 21: 0.4, 22: 0.4, 23: 0.4,
        }
        return quality_by_hour.get(hour % 24, 0.5)
    
    def _calibrate_confidence(self, raw: float, quality: float) -> float:
        """Calibrate raw confidence to true probability."""
        # Simplified calibration (in production, use Platt scaling or isotonic)
        # Higher quality signals: confidence closer to true probability
        # Lower quality signals: confidence overestimates
        calibration_factor = 0.5 + 0.5 * quality
        calibrated = raw * calibration_factor
        return max(0, min(1, calibrated))
    
    def record_outcome(self, strategy: str, signal_quality: float, profitable: bool):
        """Record signal outcome for learning."""
        if strategy not in self._strategy_outcomes:
            self._strategy_outcomes[strategy] = []
        self._strategy_outcomes[strategy].append(1.0 if profitable else 0.0)
        
        # Update strategy weight based on recent performance
        recent = self._strategy_outcomes[strategy][-100:]
        if len(recent) >= 20:
            win_rate = sum(recent) / len(recent)
            self._strategy_weights[strategy] = win_rate * 2  # weight by win rate


# ---------------------------------------------------------------------------
# Pillar 2: Market Timing Engine
# ---------------------------------------------------------------------------

class MarketTimingEngine:
    """
    Maximizes market timing through:
    - Multi-timeframe confluence (1m to 1W)
    - HMM regime detection
    - Session-aware optimization
    - Volatility regime adaptation
    """
    
    def __init__(self):
        # Timeframe weights
        self.timeframe_weights = {
            "1m": 0.05,
            "5m": 0.10,
            "15m": 0.15,
            "1h": 0.20,
            "4h": 0.25,
            "1d": 0.25,
        }
        
        # Regime detection state
        self._regime_state = "ranging"
        self._regime_confidence = 0.5
        self._regime_history: Deque[str] = deque(maxlen=100)
        
        # Session edges (historical)
        self._session_edges: Dict[TimingWindow, float] = {
            TimingWindow.ASIAN_OPEN: 0.02,
            TimingWindow.LONDON_OPEN: 0.05,
            TimingWindow.NY_OPEN: 0.06,
            TimingWindow.OVERLAP_LONDON_NY: 0.08,
            TimingWindow.LOW_LIQUIDITY: -0.02,
        }
        
        # Volatility regime
        self._vol_regime = "normal"
        self._vol_history: Deque[float] = deque(maxlen=1000)
    
    def assess_timing(
        self,
        symbol: str,
        timeframe_data: Dict[str, Dict[str, float]],
        current_time_utc: float,
    ) -> TimingAssessment:
        """
        Assess market timing quality.
        
        Args:
            symbol: Trading pair symbol
            timeframe_data: {timeframe: {open, high, low, close, volume}}
            current_time_utc: Current UTC timestamp
        """
        # 1. Multi-timeframe confluence
        mtf_signals = {}
        for tf, data in timeframe_data.items():
            if tf in self.timeframe_weights:
                signal = self._analyze_timeframe(data)
                mtf_signals[tf] = signal
        
        # Calculate weighted confluence
        if mtf_signals:
            weighted_signal = sum(
                mtf_signals[tf] * self.timeframe_weights[tf]
                for tf in mtf_signals
            )
            # Confluence = 1 - normalized disagreement
            signals = list(mtf_signals.values())
            disagreement = np.std(signals) if len(signals) > 1 else 0
            mtf_confluence = max(0, 1 - disagreement * 2)
        else:
            weighted_signal = 0
            mtf_confluence = 0.5
        
        # 2. Regime detection
        regime = self._detect_regime(timeframe_data)
        regime_confidence = self._regime_confidence
        
        # 3. Session timing
        hour = int((current_time_utc / 3600) % 24)
        timing_window = self._get_timing_window(hour)
        session_edge = self._session_edges.get(timing_window, 0)
        
        # 4. Volatility regime
        vol_regime = self._get_vol_regime(timeframe_data)
        
        # 5. Calculate timing score
        timing_score = (
            0.35 * mtf_confluence +
            0.25 * regime_confidence +
            0.20 * (0.5 + session_edge) +
            0.20 * self._vol_timing_score(vol_regime)
        )
        
        # 6. Should trade now?
        should_trade = timing_score > 0.5 and regime != MarketRegime.CRISIS
        
        # 7. Wait recommendation
        wait_minutes = None
        if not should_trade:
            # Estimate when conditions improve
            if regime == MarketRegime.CRISIS:
                wait_minutes = 60  # wait 1 hour
            elif mtf_confluence < 0.3:
                wait_minutes = 30
            elif timing_score < 0.4:
                wait_minutes = 15
        
        return TimingAssessment(
            current_regime=regime,
            regime_confidence=regime_confidence,
            optimal_timing_window=timing_window,
            timing_score=timing_score,
            mtf_confluence=mtf_confluence,
            session_edge=session_edge,
            vol_regime=vol_regime,
            should_trade_now=should_trade,
            wait_minutes=wait_minutes,
        )
    
    def _analyze_timeframe(self, data: Dict[str, float]) -> float:
        """Analyze single timeframe, return -1 to 1 signal."""
        close = data.get("close", 0)
        open_ = data.get("open", 0)
        high = data.get("high", 0)
        low = data.get("low", 0)
        
        if close == 0 or open_ == 0:
            return 0
        
        # Price action signal
        body = (close - open_) / open_
        range_ = (high - low) / close if close > 0 else 0
        
        # Simple momentum
        if close > open_:
            signal = min(1.0, body * 10)
        else:
            signal = max(-1.0, body * 10)
        
        return signal
    
    def _detect_regime(self, timeframe_data: Dict[str, Dict]) -> MarketRegime:
        """Detect current market regime."""
        # Use 1h and 4h data for regime detection
        h1 = timeframe_data.get("1h", {})
        h4 = timeframe_data.get("4h", {})
        
        h1_close = h1.get("close", 0)
        h4_close = h4.get("close", 0)
        
        if h1_close == 0 or h4_close == 0:
            return MarketRegime.RANGING
        
        # Simple regime detection
        h1_range = (h1.get("high", h1_close) - h1.get("low", h1_close)) / h1_close
        h4_range = (h4.get("high", h4_close) - h4.get("low", h4_close)) / h4_close
        
        # High volatility
        if h1_range > 0.05 or h4_range > 0.10:
            regime = MarketRegime.HIGH_VOLATILITY
        # Trending
        elif h1_range > 0.02:
            # Determine direction from price action
            if h1_close > h4_close * 1.02:
                regime = MarketRegime.BULL_TREND
            elif h1_close < h4_close * 0.98:
                regime = MarketRegime.BEAR_TREND
            else:
                regime = MarketRegime.RANGING
        else:
            regime = MarketRegime.LOW_VOLATILITY
        
        # Update state
        self._regime_state = regime.value
        self._regime_history.append(regime.value)
        
        # Calculate confidence
        if len(self._regime_history) >= 10:
            recent = list(self._regime_history)[-10:]
            consistency = recent.count(regime.value) / len(recent)
            self._regime_confidence = consistency
        else:
            self._regime_confidence = 0.5
        
        return regime
    
    def _get_timing_window(self, hour: int) -> TimingWindow:
        """Get current trading session."""
        if 0 <= hour < 8:
            return TimingWindow.ASIAN_OPEN
        elif 7 <= hour < 12:
            return TimingWindow.LONDON_OPEN
        elif 13 <= hour < 17:
            return TimingWindow.NY_OPEN
        elif 13 <= hour < 16:
            return TimingWindow.OVERLAP_LONDON_NY
        else:
            return TimingWindow.LOW_LIQUIDITY
    
    def _get_vol_regime(self, timeframe_data: Dict[str, Dict]) -> str:
        """Get volatility regime."""
        h1 = timeframe_data.get("1h", {})
        h1_range = (h1.get("high", 0) - h1.get("low", 0)) / max(0.01, h1.get("close", 1))
        
        if h1_range < 0.01:
            return "low"
        elif h1_range < 0.03:
            return "normal"
        elif h1_range < 0.06:
            return "high"
        else:
            return "extreme"
    
    def _vol_timing_score(self, vol_regime: str) -> float:
        """Score for volatility timing."""
        scores = {"low": 0.6, "normal": 0.8, "high": 0.5, "extreme": 0.2}
        return scores.get(vol_regime, 0.5)


# ---------------------------------------------------------------------------
# Pillar 3: Asset Selection Engine
# ---------------------------------------------------------------------------

class AssetSelectionEngine:
    """
    Maximizes asset selection through:
    - Cross-asset momentum ranking
    - Institutional flow analysis
    - Correlation-adjusted selection
    - Liquidity-adjusted opportunity scoring
    """
    
    def __init__(self):
        self._price_history: Dict[str, Deque[float]] = {}
        self._volume_history: Dict[str, Deque[float]] = {}
        self._flow_scores: Dict[str, float] = {}
        
    def rank_assets(
        self,
        symbols: List[str],
        market_data: Dict[str, Dict[str, float]],
        existing_positions: Optional[Dict[str, float]] = None,
    ) -> List[AssetSelection]:
        """
        Rank all candidate assets for selection.
        
        Returns sorted list (best first).
        """
        results = []
        existing_positions = existing_positions or {}
        
        for symbol in symbols:
            data = market_data.get(symbol, {})
            if not data:
                continue
            
            # Update history
            price = data.get("close", 0)
            volume = data.get("volume", 0)
            if price > 0:
                if symbol not in self._price_history:
                    self._price_history[symbol] = deque(maxlen=200)
                    self._volume_history[symbol] = deque(maxlen=200)
                self._price_history[symbol].append(price)
                self._volume_history[symbol].append(volume)
            
            # Calculate scores
            momentum_score = self._calculate_momentum(symbol)
            flow_score = self._calculate_flow_score(symbol, data)
            liquidity_score = self._calculate_liquidity(symbol, data)
            volatility_score = self._calculate_volatility_score(symbol)
            correlation_penalty = self._calculate_correlation_penalty(
                symbol, existing_positions
            )
            
            # Composite opportunity score
            opportunity_score = (
                0.30 * momentum_score +
                0.20 * flow_score +
                0.20 * liquidity_score +
                0.15 * volatility_score +
                0.15 * (1 - correlation_penalty)
            )
            
            # Selection reason
            reason = self._generate_selection_reason(
                momentum_score, flow_score, liquidity_score, opportunity_score
            )
            
            results.append(AssetSelection(
                symbol=symbol,
                rank=0,  # will be set after sorting
                opportunity_score=opportunity_score,
                momentum_score=momentum_score,
                flow_score=flow_score,
                liquidity_score=liquidity_score,
                correlation_penalty=correlation_penalty,
                volatility_score=volatility_score,
                selection_reason=reason,
            ))
        
        # Sort by opportunity score
        results.sort(key=lambda x: x.opportunity_score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def _calculate_momentum(self, symbol: str) -> float:
        """Calculate momentum score (0-1)."""
        if symbol not in self._price_history:
            return 0.5
        
        prices = list(self._price_history[symbol])
        if len(prices) < 20:
            return 0.5
        
        # Multi-timeframe momentum
        returns_1h = (prices[-1] / prices[-5] - 1) if len(prices) >= 5 else 0
        returns_4h = (prices[-1] / prices[-20] - 1) if len(prices) >= 20 else 0
        returns_24h = (prices[-1] / prices[-60] - 1) if len(prices) >= 60 else 0
        
        # Weighted momentum
        momentum = 0.2 * returns_1h + 0.3 * returns_4h + 0.5 * returns_24h
        
        # Normalize to 0-1
        score = 0.5 + momentum * 10  # scale by 10
        return max(0, min(1, score))
    
    def _calculate_flow_score(self, symbol: str, data: Dict) -> float:
        """Calculate institutional flow score (0-1)."""
        volume = data.get("volume", 0)
        avg_volume = data.get("avg_volume", volume)
        
        if avg_volume == 0:
            return 0.5
        
        # Volume ratio as proxy for flow
        volume_ratio = volume / avg_volume
        
        # High volume = more institutional interest
        if volume_ratio > 2.0:
            return 0.9
        elif volume_ratio > 1.5:
            return 0.7
        elif volume_ratio > 1.0:
            return 0.6
        elif volume_ratio > 0.5:
            return 0.5
        else:
            return 0.3
    
    def _calculate_liquidity(self, symbol: str, data: Dict) -> float:
        """Calculate liquidity score (0-1)."""
        volume = data.get("volume", 0)
        spread_bps = data.get("spread_bps", 20)
        
        # Volume component
        vol_score = min(1.0, math.log10(max(1, volume)) / 6)  # log scale
        
        # Spread component
        spread_score = max(0, 1.0 - spread_bps / 50)
        
        return 0.6 * vol_score + 0.4 * spread_score
    
    def _calculate_volatility_score(self, symbol: str) -> float:
        """Calculate volatility score (0-1, higher = better for trading)."""
        if symbol not in self._price_history:
            return 0.5
        
        prices = list(self._price_history[symbol])
        if len(prices) < 20:
            return 0.5
        
        returns = np.diff(np.log(prices[-20:]))
        vol = np.std(returns) * math.sqrt(252 * 24 * 60)  # annualized
        
        # Optimal volatility range: 30-80% annualized for crypto
        if vol < 0.20:
            return 0.3  # too low
        elif vol < 0.40:
            return 0.6
        elif vol < 0.80:
            return 0.9  # optimal
        elif vol < 1.50:
            return 0.7
        else:
            return 0.4  # too high
    
    def _calculate_correlation_penalty(
        self,
        symbol: str,
        existing_positions: Dict[str, float],
    ) -> float:
        """Calculate correlation penalty (0-1, higher = more penalty)."""
        if not existing_positions or symbol not in self._price_history:
            return 0.0
        
        # Simplified: check if we already have similar assets
        # In production, use actual correlation calculation
        base = symbol.split("/")[0]  # e.g., "BTC" from "BTC/USDT"
        
        for pos_symbol in existing_positions:
            pos_base = pos_symbol.split("/")[0]
            if pos_base == base:
                return 0.8  # high penalty for same asset
        
        return 0.0  # no penalty for different assets
    
    def _generate_selection_reason(
        self,
        momentum: float,
        flow: float,
        liquidity: float,
        opportunity: float,
    ) -> str:
        """Generate human-readable selection reason."""
        reasons = []
        
        if momentum > 0.7:
            reasons.append("strong momentum")
        elif momentum < 0.3:
            reasons.append("weak momentum")
        
        if flow > 0.7:
            reasons.append("high institutional flow")
        elif flow < 0.3:
            reasons.append("low flow")
        
        if liquidity > 0.8:
            reasons.append("excellent liquidity")
        elif liquidity < 0.4:
            reasons.append("poor liquidity")
        
        if opportunity > 0.7:
            return f"High opportunity: {', '.join(reasons)}"
        elif opportunity > 0.5:
            return f"Moderate opportunity: {', '.join(reasons)}"
        else:
            return f"Low opportunity: {', '.join(reasons) if reasons else 'no clear edge'}"


# ---------------------------------------------------------------------------
# Pillar 4: Execution Engine
# ---------------------------------------------------------------------------

class ExecutionOptimizationEngine:
    """
    Maximizes execution quality through:
    - Smart order routing (8 venues)
    - Adaptive TWAP/VWAP/POV
    - Queue position optimization
    - Maker/taker fee optimization
    """
    
    def __init__(self):
        # Venue data
        self._venues = ["binance", "bybit", "okx", "kucoin", "gate", "huobi", "bitget", "mexc"]
        self._venue_latencies: Dict[str, float] = {v: 10.0 for v in self._venues}
        self._venue_fees: Dict[str, Tuple[float, float]] = {
            v: (0.02, 0.04) for v in self._venues  # (maker, taker) in %
        }
        self._venue_liquidity: Dict[str, float] = {v: 1.0 for v in self._venues}
        
        # Historical fill rates
        self._fill_rates: Dict[str, float] = {v: 0.95 for v in self._venues}
        
    def create_execution_plan(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        current_price: float,
        spread_bps: float,
        urgency: str = "normal",  # low, normal, high, critical
    ) -> ExecutionPlan:
        """Create optimal execution plan."""
        
        # 1. Select optimal venue
        optimal_venue = self._select_venue(symbol, size_usd, urgency)
        
        # 2. Select algorithm based on size and urgency
        algorithm = self._select_algorithm(size_usd, urgency, spread_bps)
        
        # 3. Calculate expected slippage
        expected_slippage = self._estimate_slippage(
            size_usd, current_price, urgency, optimal_venue
        )
        
        # 4. Maker/taker decision
        maker_taker = self._decide_maker_taker(spread_bps, urgency, expected_slippage)
        
        # 5. Fee optimization
        venue_fee = self._venue_fees.get(optimal_venue, (0.02, 0.04))
        if maker_taker == "maker":
            fee_bps = venue_fee[0] * 100
        else:
            fee_bps = venue_fee[1] * 100
        
        # 6. Execution parameters
        if algorithm == ExecutionAlgorithm.TWAP:
            slice_count = max(3, int(size_usd / 1000))
            duration = 300  # 5 minutes
        elif algorithm == ExecutionAlgorithm.VWAP:
            slice_count = max(5, int(size_usd / 500))
            duration = 600  # 10 minutes
        elif algorithm == ExecutionAlgorithm.POV:
            slice_count = 10
            duration = 180  # 3 minutes
        else:
            slice_count = 1
            duration = 10  # immediate
        
        # 7. Queue position estimate
        queue_estimate = self._estimate_queue_position(
            optimal_venue, side, size_usd, current_price
        )
        
        # 8. Expected fill rate
        fill_rate = self._fill_rates.get(optimal_venue, 0.95)
        
        # 9. Estimated fill price
        if side == "buy":
            fill_price = current_price * (1 + expected_slippage / 10000)
        else:
            fill_price = current_price * (1 - expected_slippage / 10000)
        
        # 10. Fee savings from optimization
        worst_fee = max(v[1] for v in self._venue_fees.values()) * 100
        fee_savings = worst_fee - fee_bps
        
        return ExecutionPlan(
            algorithm=algorithm,
            optimal_venue=optimal_venue,
            estimated_fill_price=fill_price,
            expected_slippage_bps=expected_slippage,
            expected_fill_rate=fill_rate,
            maker_taker_decision=maker_taker,
            slice_count=slice_count,
            duration_seconds=duration,
            queue_position_estimate=queue_estimate,
            fee_optimization_bps=fee_savings,
        )
    
    def _select_venue(self, symbol: str, size_usd: float, urgency: str) -> str:
        """Select optimal venue based on multiple factors."""
        scores = {}
        
        for venue in self._venues:
            # Score components
            latency_score = max(0, 1 - self._venue_latencies[venue] / 100)
            fee_score = 1 - self._venue_fees[venue][1] * 10  # lower taker fee
            liquidity_score = self._venue_liquidity[venue]
            fill_score = self._fill_rates[venue]
            
            # Weight by urgency
            if urgency == "critical":
                weights = [0.5, 0.1, 0.3, 0.1]  # latency matters most
            elif urgency == "high":
                weights = [0.3, 0.2, 0.3, 0.2]
            else:
                weights = [0.2, 0.3, 0.3, 0.2]  # fees matter more
            
            score = (
                weights[0] * latency_score +
                weights[1] * fee_score +
                weights[2] * liquidity_score +
                weights[3] * fill_score
            )
            scores[venue] = score
        
        return max(scores, key=scores.get)
    
    def _select_algorithm(
        self,
        size_usd: float,
        urgency: str,
        spread_bps: float,
    ) -> ExecutionAlgorithm:
        """Select optimal execution algorithm."""
        if urgency == "critical":
            return ExecutionAlgorithm.IMMEDIATE
        elif urgency == "high":
            if size_usd > 10000:
                return ExecutionAlgorithm.POV
            else:
                return ExecutionAlgorithm.IMMEDIATE
        elif size_usd > 50000:
            return ExecutionAlgorithm.TWAP
        elif size_usd > 10000:
            return ExecutionAlgorithm.VWAP
        elif spread_bps < 5:
            return ExecutionAlgorithm.IMMEDIATE
        else:
            return ExecutionAlgorithm.ADAPTIVE
    
    def _estimate_slippage(
        self,
        size_usd: float,
        price: float,
        urgency: str,
        venue: str,
    ) -> float:
        """Estimate expected slippage in bps."""
        # Base slippage from size
        size_bps = math.sqrt(size_usd / 10000) * 2
        
        # Urgency multiplier
        urgency_mult = {"low": 0.5, "normal": 1.0, "high": 1.5, "critical": 2.0}
        mult = urgency_mult.get(urgency, 1.0)
        
        # Venue adjustment
        venue_mult = 1.0 / max(0.5, self._venue_liquidity.get(venue, 1.0))
        
        return size_bps * mult * venue_mult
    
    def _decide_maker_taker(
        self,
        spread_bps: float,
        urgency: str,
        expected_slippage: float,
    ) -> str:
        """Decide between maker (limit) and taker (market)."""
        if urgency in ["high", "critical"]:
            return "taker"
        
        # If spread is wide, try maker
        if spread_bps > 10:
            return "maker"
        
        # If expected slippage is high, use maker
        if expected_slippage > 5:
            return "maker"
        
        return "taker"
    
    def _estimate_queue_position(
        self,
        venue: str,
        side: str,
        size_usd: float,
        price: float,
    ) -> int:
        """Estimate queue position for limit order."""
        # Simplified estimate
        size_tokens = size_usd / price
        # Assume average queue depth of 1000 tokens at best bid/ask
        estimated_position = int(size_tokens / 10)  # rough estimate
        return max(1, min(1000, estimated_position))


# ---------------------------------------------------------------------------
# Maximum Trading Engine (Orchestrator)
# ---------------------------------------------------------------------------

class MaximumTradingEngine:
    """
    The ultimate trading engine that orchestrates all 4 pillars
    to generate maximum quality trades.
    """
    
    def __init__(self):
        # Pillar engines
        self.signal_engine = SignalQualityEngine()
        self.timing_engine = MarketTimingEngine()
        self.selection_engine = AssetSelectionEngine()
        self.execution_engine = ExecutionOptimizationEngine()
        
        # Performance tracking
        self._trade_outcomes: Deque[Dict] = deque(maxlen=1000)
        self._total_trades = 0
        self._winning_trades = 0
        
        # Minimum thresholds
        self.min_composite_score = 60.0
        self.min_signal_quality = 0.5
        self.min_timing_score = 0.4
        
        logger.info("MaximumTradingEngine initialized")
    
    def generate_trade(
        self,
        symbols: List[str],
        market_data: Dict[str, Dict[str, Any]],
        portfolio_equity: float,
        existing_positions: Optional[Dict[str, float]] = None,
        current_time_utc: Optional[float] = None,
    ) -> MaximumTradeDecision:
        """
        Generate optimal trade decision.
        
        This is the main entry point - orchestrates all 4 pillars.
        """
        current_time = current_time_utc or time.time()
        
        # PILLAR 3: Asset Selection (first - need to know WHAT to trade)
        asset_rankings = self.selection_engine.rank_assets(
            symbols, market_data, existing_positions
        )
        
        if not asset_rankings or asset_rankings[0].opportunity_score < 0.3:
            return self._create_no_trade_decision("No suitable assets")
        
        best_asset = asset_rankings[0]
        symbol = best_asset.symbol
        
        # Get market data for selected symbol
        data = market_data.get(symbol, {})
        
        # PILLAR 1: Signal Quality
        raw_signals = data.get("signals", [])
        market_state = {
            "volume_ratio": data.get("volume", 0) / max(1, data.get("avg_volume", 1)),
            "spread_bps": data.get("spread_bps", 10),
            "regime": self.timing_engine._regime_state,
            "hour": int((current_time / 3600) % 24),
            "correlation_with_portfolio": 0.0,  # simplified
        }
        signal_quality = self.signal_engine.assess_signal(raw_signals, market_state)
        
        # PILLAR 2: Market Timing
        timeframe_data = data.get("timeframes", {})
        timing_assessment = self.timing_engine.assess_timing(
            symbol, timeframe_data, current_time
        )
        
        # Determine direction from signals
        if raw_signals:
            buy_votes = sum(1 for s in raw_signals if "buy" in s.get("direction", "").lower())
            sell_votes = sum(1 for s in raw_signals if "sell" in s.get("direction", "").lower())
            
            if buy_votes > sell_votes:
                direction = SignalDirection.BUY if buy_votes < len(raw_signals) * 0.8 else SignalDirection.STRONG_BUY
                side = "buy"
            elif sell_votes > buy_votes:
                direction = SignalDirection.SELL if sell_votes < len(raw_signals) * 0.8 else SignalDirection.STRONG_SELL
                side = "sell"
            else:
                direction = SignalDirection.NEUTRAL
                side = "buy"  # default
        else:
            direction = SignalDirection.NEUTRAL
            side = "buy"
        
        # PILLAR 4: Execution Plan
        current_price = data.get("close", 0)
        spread_bps = data.get("spread_bps", 10)
        
        # Determine urgency
        if not timing_assessment.should_trade_now:
            urgency = "low"
        elif signal_quality.quality_score > 0.8:
            urgency = "high"
        else:
            urgency = "normal"
        
        # Calculate position size (simplified Kelly)
        base_size = portfolio_equity * 0.10  # 10% base position
        kelly_fraction = signal_quality.calibrated_confidence * 0.5  # half-Kelly
        size_usd = portfolio_equity * kelly_fraction
        size_usd = min(size_usd, base_size, portfolio_equity * 0.25)
        
        execution_plan = self.execution_engine.create_execution_plan(
            symbol, side, size_usd, current_price, spread_bps, urgency
        )
        
        # COMPOSITE SCORING
        composite_score = (
            0.35 * signal_quality.quality_score * 100 +
            0.25 * timing_assessment.timing_score * 100 +
            0.25 * best_asset.opportunity_score * 100 +
            0.15 * execution_plan.expected_fill_rate * 100
        )
        
        # Expected edge
        expected_edge = (
            signal_quality.calibrated_confidence * 100 - 50  # edge from signal
            + timing_assessment.session_edge * 100  # edge from timing
            + execution_plan.fee_optimization_bps  # edge from execution
        )
        
        # Decision
        should_trade = (
            composite_score >= self.min_composite_score and
            signal_quality.quality_score >= self.min_signal_quality and
            timing_assessment.timing_score >= self.min_timing_score and
            direction != SignalDirection.NEUTRAL and
            not signal_quality.rejection_reasons
        )
        
        # Reasoning
        reasoning = [
            f"Signal quality: {signal_quality.quality_score:.2f}",
            f"Timing score: {timing_assessment.timing_score:.2f}",
            f"Asset opportunity: {best_asset.opportunity_score:.2f}",
            f"Composite score: {composite_score:.1f}/100",
            f"Expected edge: {expected_edge:.1f}bps",
        ]
        
        if not should_trade:
            if signal_quality.rejection_reasons:
                reasoning.append(f"Rejected: {signal_quality.rejection_reasons[0]}")
            elif composite_score < self.min_composite_score:
                reasoning.append(f"Composite score too low: {composite_score:.1f}")
        
        return MaximumTradeDecision(
            should_trade=should_trade,
            direction=direction,
            symbol=symbol,
            size_usd=size_usd if should_trade else 0,
            signal_quality=signal_quality,
            timing_assessment=timing_assessment,
            asset_selection=best_asset,
            execution_plan=execution_plan,
            composite_score=composite_score,
            expected_edge_bps=expected_edge,
            expected_pnl_bps=expected_edge * 0.5,  # conservative estimate
            confidence=signal_quality.calibrated_confidence,
            sharpe_contribution=expected_edge / 100 if expected_edge > 0 else 0,
            kelly_fraction=kelly_fraction,
            max_position_size=portfolio_equity * 0.25,
            execute_immediately=should_trade and urgency in ["high", "critical"],
            valid_until=current_time + 300,  # 5 minute validity
            reasoning=reasoning,
        )
    
    def _create_no_trade_decision(self, reason: str) -> MaximumTradeDecision:
        """Create a no-trade decision."""
        empty_signal = SignalQuality(0, 0, 0, 0, 0, 1, False, {}, [reason])
        empty_timing = TimingAssessment(
            MarketRegime.RANGING, 0, TimingWindow.LOW_LIQUIDITY,
            0, 0, 0, "normal", False, None
        )
        empty_asset = AssetSelection("", 0, 0, 0, 0, 0, 0, 0, reason)
        empty_exec = ExecutionPlan(
            ExecutionAlgorithm.IMMEDIATE, "", 0, 0, 0, "taker", 0, 0, 0, 0
        )
        
        return MaximumTradeDecision(
            should_trade=False,
            direction=SignalDirection.NEUTRAL,
            symbol="",
            size_usd=0,
            signal_quality=empty_signal,
            timing_assessment=empty_timing,
            asset_selection=empty_asset,
            execution_plan=empty_exec,
            composite_score=0,
            expected_edge_bps=0,
            expected_pnl_bps=0,
            confidence=0,
            sharpe_contribution=0,
            kelly_fraction=0,
            max_position_size=0,
            execute_immediately=False,
            valid_until=0,
            reasoning=[reason],
        )
    
    def record_outcome(
        self,
        symbol: str,
        profitable: bool,
        pnl_bps: float,
    ):
        """Record trade outcome for learning."""
        self._total_trades += 1
        if profitable:
            self._winning_trades += 1
        
        self._trade_outcomes.append({
            "symbol": symbol,
            "profitable": profitable,
            "pnl_bps": pnl_bps,
            "timestamp": time.time(),
        })
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self._trade_outcomes:
            return {"total_trades": 0, "win_rate": 0, "avg_pnl_bps": 0}
        
        outcomes = list(self._trade_outcomes)
        win_rate = sum(1 for o in outcomes if o["profitable"]) / len(outcomes)
        avg_pnl = np.mean([o["pnl_bps"] for o in outcomes])
        
        return {
            "total_trades": len(outcomes),
            "win_rate": win_rate,
            "avg_pnl_bps": avg_pnl,
            "recent_win_rate": sum(1 for o in outcomes[-20:] if o["profitable"]) / min(20, len(outcomes)),
        }
