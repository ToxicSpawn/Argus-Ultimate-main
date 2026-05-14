"""
ARGUS Superintelligence — autonomous reasoning about markets.

This is not another signal processor. This is a REASONING ENGINE that
thinks about markets the way the best human traders do:

1. CAUSAL REASONING: "BTC dropped because DXY spiked, not because of selling"
   → Different cause = different response (hedge vs buy the dip)

2. COUNTERFACTUAL ANALYSIS: "If we had NOT taken that trade, P&L would be..."
   → Learns from paths not taken, not just paths taken

3. REGIME ANTICIPATION: "Conditions are transitioning from ranging → trending"
   → Acts on regime TRANSITIONS, not just current regime

4. ADVERSARIAL THINKING: "If I'm buying, who's selling and why?"
   → Models counterparty behavior to avoid being the dumb money

5. TEMPORAL ABSTRACTION: "This 5-min dip is noise; the 4-hour trend is real"
   → Separates signal from noise at multiple time scales

6. HYPOTHESIS GENERATION: "If funding stays negative for 3 more hours,
   shorts will get squeezed" → Generates and tests market hypotheses

7. META-COGNITION: "My confidence in this prediction is only 40% because
   I've never seen this exact pattern" → Knows what it doesn't know

This module is the bridge between mechanical signal processing and
genuine market understanding.
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


# ════════════════════════════════════════════════════════════════════════════
# Causal Reasoning Engine
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class CausalLink:
    """A learned cause-effect relationship."""
    cause: str                  # e.g. "dxy_spike", "whale_deposit", "funding_flip"
    effect: str                 # e.g. "btc_drop", "vol_expansion", "short_squeeze"
    strength: float             # -1 to +1 (correlation * consistency)
    lag_bars: int               # how many bars between cause and effect
    observations: int           # how many times seen
    last_seen: float


class CausalReasoningEngine:
    """
    Learns cause-effect relationships between market events.

    Instead of just correlating signals, discovers CAUSAL chains:
    DXY spike → USD strengthens → crypto sells off → funding goes negative
    → overleveraged shorts accumulate → short squeeze → price recovery

    This lets ARGUS understand WHY prices move, not just THAT they move.
    """

    def __init__(self, min_observations: int = 5, max_lag: int = 20):
        self._min_obs = min_observations
        self._max_lag = max_lag
        self._event_history: deque = deque(maxlen=500)
        self._causal_links: Dict[Tuple[str, str], CausalLink] = {}
        self._event_counts: Dict[str, int] = defaultdict(int)

    def record_event(self, event_type: str, magnitude: float = 1.0,
                     timestamp: Optional[float] = None) -> None:
        """Record a market event (e.g. 'dxy_spike', 'whale_deposit')."""
        ts = timestamp or time.time()
        self._event_history.append((ts, event_type, magnitude))
        self._event_counts[event_type] += 1
        self._update_links(event_type, ts)

    def _update_links(self, new_event: str, ts: float) -> None:
        """Check if recent events caused this one."""
        for hist_ts, hist_event, hist_mag in reversed(list(self._event_history)[:-1]):
            lag = int((ts - hist_ts) / 10)  # assume 10s bars
            if lag > self._max_lag or lag < 1:
                continue
            if hist_event == new_event:
                continue

            key = (hist_event, new_event)
            if key not in self._causal_links:
                self._causal_links[key] = CausalLink(
                    cause=hist_event, effect=new_event,
                    strength=0.0, lag_bars=lag, observations=0, last_seen=ts,
                )
            link = self._causal_links[key]
            link.observations += 1
            link.last_seen = ts
            # Update strength: more observations + consistent timing = stronger
            consistency = 1.0 / (1.0 + abs(link.lag_bars - lag))
            link.strength = link.strength * 0.9 + consistency * 0.1
            link.lag_bars = int(link.lag_bars * 0.8 + lag * 0.2)  # running avg

    def predict_effects(self, current_event: str) -> List[Tuple[str, float, int]]:
        """Given a current event, predict what effects will follow.
        Returns [(effect_name, probability, expected_lag_bars), ...]"""
        predictions = []
        for (cause, effect), link in self._causal_links.items():
            if cause == current_event and link.observations >= self._min_obs:
                # Probability based on how often cause→effect vs cause alone
                cause_count = max(self._event_counts.get(cause, 1), 1)
                prob = min(0.95, link.observations / cause_count * link.strength)
                if prob > 0.1:
                    predictions.append((effect, prob, link.lag_bars))
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:5]

    def explain(self, event: str) -> List[Tuple[str, float]]:
        """Explain WHY an event happened by finding likely causes.
        Returns [(cause_name, probability), ...]"""
        explanations = []
        for (cause, effect), link in self._causal_links.items():
            if effect == event and link.observations >= self._min_obs:
                # Check if cause happened recently
                recent = any(e == cause and (time.time() - t) < link.lag_bars * 15
                             for t, e, _ in self._event_history)
                if recent:
                    explanations.append((cause, link.strength))
        explanations.sort(key=lambda x: x[1], reverse=True)
        return explanations[:5]

    def get_chain(self, start_event: str, depth: int = 3) -> List[List[str]]:
        """Trace causal chains from a starting event.
        Returns chains like [['dxy_spike', 'btc_drop', 'funding_flip']]"""
        chains = []
        self._trace_chain([start_event], depth, chains)
        return chains[:5]

    def _trace_chain(self, current_chain: List[str], depth: int,
                     chains: List[List[str]]) -> None:
        if depth <= 0:
            if len(current_chain) > 1:
                chains.append(list(current_chain))
            return
        last = current_chain[-1]
        effects = self.predict_effects(last)
        if not effects:
            if len(current_chain) > 1:
                chains.append(list(current_chain))
            return
        for effect, prob, _ in effects:
            if effect not in current_chain and prob > 0.15:
                self._trace_chain(current_chain + [effect], depth - 1, chains)

    def get_stats(self) -> Dict[str, Any]:
        strong = [(k, v) for k, v in self._causal_links.items()
                  if v.observations >= self._min_obs and v.strength > 0.3]
        return {
            "total_links": len(self._causal_links),
            "strong_links": len(strong),
            "events_recorded": len(self._event_history),
            "top_chains": [f"{c}→{e} (str={v.strength:.2f}, n={v.observations})"
                          for (c, e), v in sorted(strong, key=lambda x: x[1].strength, reverse=True)[:5]],
        }


# ════════════════════════════════════════════════════════════════════════════
# Counterfactual Analyzer
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class CounterfactualResult:
    """What would have happened on the path not taken."""
    trade_id: str
    actual_action: str          # "BUY", "SELL", "SKIP"
    actual_pnl: float
    counterfactual_action: str  # what we WOULD have done
    counterfactual_pnl: float   # estimated P&L of the path not taken
    regret: float               # actual_pnl - counterfactual_pnl (negative = we chose wrong)
    lesson: str                 # what we should learn


class CounterfactualAnalyzer:
    """
    Learns from paths NOT taken.

    For every trade decision, records what would have happened if we
    had done the opposite. Over time, identifies systematic biases:
    - "We skip too many BUY signals that would have been profitable"
    - "We hold too long — selling earlier would have been +2% better"
    - "In ranging markets, our SELL signals are actually buy opportunities"
    """

    def __init__(self):
        self._results: deque = deque(maxlen=500)
        self._bias_tracker: Dict[str, List[float]] = defaultdict(list)

    def record_decision(
        self,
        trade_id: str,
        action_taken: str,
        actual_pnl: float,
        alternative_pnl: float,
        regime: str = "normal",
    ) -> CounterfactualResult:
        """Record a decision and its counterfactual outcome."""
        counterfactual = "SKIP" if action_taken in ("BUY", "SELL") else "BUY"
        regret = actual_pnl - alternative_pnl

        # Generate lesson
        if regret < -1.0:
            lesson = f"Should have {counterfactual} instead of {action_taken} (saved {-regret:.2f}%)"
        elif regret > 1.0:
            lesson = f"Good call: {action_taken} was {regret:.2f}% better than {counterfactual}"
        else:
            lesson = "Decision was approximately neutral"

        result = CounterfactualResult(
            trade_id=trade_id, actual_action=action_taken,
            actual_pnl=actual_pnl, counterfactual_action=counterfactual,
            counterfactual_pnl=alternative_pnl, regret=regret, lesson=lesson,
        )
        self._results.append(result)

        # Track biases by action type and regime
        self._bias_tracker[f"{action_taken}_{regime}"].append(regret)

        return result

    def get_biases(self) -> Dict[str, float]:
        """Identify systematic decision biases.
        Negative avg regret = we're consistently making wrong choices."""
        biases = {}
        for key, regrets in self._bias_tracker.items():
            if len(regrets) >= 5:
                biases[key] = sum(regrets) / len(regrets)
        return biases

    def should_override(self, proposed_action: str, regime: str) -> Optional[str]:
        """Based on counterfactual analysis, should we override the proposed action?
        Returns override action or None."""
        key = f"{proposed_action}_{regime}"
        regrets = self._bias_tracker.get(key, [])
        if len(regrets) < 10:
            return None
        avg_regret = sum(regrets[-20:]) / len(regrets[-20:])
        if avg_regret < -2.0:
            # We're consistently wrong with this action in this regime
            if proposed_action == "BUY":
                return "SKIP"
            elif proposed_action == "SKIP":
                return "BUY"
        return None

    def get_stats(self) -> Dict[str, Any]:
        if not self._results:
            return {"total_decisions": 0, "avg_regret": 0}
        results = list(self._results)
        return {
            "total_decisions": len(results),
            "avg_regret": sum(r.regret for r in results) / len(results),
            "good_decisions": sum(1 for r in results if r.regret >= 0),
            "bad_decisions": sum(1 for r in results if r.regret < -1.0),
            "biases": self.get_biases(),
        }


# ════════════════════════════════════════════════════════════════════════════
# Hypothesis Engine
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class MarketHypothesis:
    """A testable market hypothesis."""
    hypothesis_id: str
    description: str            # human-readable
    condition: str              # what must be true for this to hold
    prediction: str             # what the hypothesis predicts
    confidence: float           # 0-1
    created_at: float
    expires_at: float           # when to evaluate
    outcome: Optional[bool] = None  # True if correct
    actual_result: str = ""


class HypothesisEngine:
    """
    Generates and tests market hypotheses.

    Instead of just processing signals, ARGUS forms hypotheses:
    "If funding stays negative for 3 more hours, shorts will squeeze"
    "If BTC breaks 60K, altcoins will rally 5-10%"
    "If DXY reverses, crypto will bottom within 2 days"

    Hypotheses are tested against reality. Correct ones increase
    confidence in the reasoning chain. Wrong ones update the causal model.
    """

    def __init__(self, max_active: int = 20):
        self._active: Dict[str, MarketHypothesis] = {}
        self._completed: deque = deque(maxlen=200)
        self._max_active = max_active
        self._accuracy_by_type: Dict[str, List[bool]] = defaultdict(list)

    def generate(
        self,
        hypothesis_id: str,
        description: str,
        condition: str,
        prediction: str,
        confidence: float,
        ttl_seconds: float = 3600,
    ) -> MarketHypothesis:
        """Generate a new hypothesis to be tested."""
        now = time.time()
        hyp = MarketHypothesis(
            hypothesis_id=hypothesis_id, description=description,
            condition=condition, prediction=prediction,
            confidence=confidence, created_at=now, expires_at=now + ttl_seconds,
        )
        # Evict oldest if at capacity
        if len(self._active) >= self._max_active:
            oldest = min(self._active.values(), key=lambda h: h.created_at)
            self._resolve(oldest.hypothesis_id, False, "expired_unresolved")
        self._active[hypothesis_id] = hyp
        logger.debug("Hypothesis: %s (conf=%.0f%%, ttl=%.0fs)", description, confidence * 100, ttl_seconds)
        return hyp

    def resolve(self, hypothesis_id: str, was_correct: bool, actual: str = "") -> None:
        """Resolve a hypothesis with its actual outcome."""
        self._resolve(hypothesis_id, was_correct, actual)

    def _resolve(self, hypothesis_id: str, was_correct: bool, actual: str) -> None:
        hyp = self._active.pop(hypothesis_id, None)
        if hyp is None:
            return
        hyp.outcome = was_correct
        hyp.actual_result = actual
        self._completed.append(hyp)
        # Track accuracy by type
        h_type = hyp.condition.split("_")[0] if hyp.condition else "unknown"
        self._accuracy_by_type[h_type].append(was_correct)

    def check_expired(self) -> List[MarketHypothesis]:
        """Check for expired hypotheses that need resolution."""
        now = time.time()
        expired = [h for h in self._active.values() if now >= h.expires_at]
        for h in expired:
            self._resolve(h.hypothesis_id, False, "expired")
        return expired

    def get_active(self) -> List[MarketHypothesis]:
        return list(self._active.values())

    def get_accuracy(self) -> Dict[str, float]:
        result = {}
        for h_type, outcomes in self._accuracy_by_type.items():
            if len(outcomes) >= 3:
                result[h_type] = sum(outcomes) / len(outcomes)
        return result

    def get_stats(self) -> Dict[str, Any]:
        completed = list(self._completed)
        return {
            "active": len(self._active),
            "completed": len(completed),
            "accuracy": sum(1 for h in completed if h.outcome) / max(len(completed), 1),
            "accuracy_by_type": self.get_accuracy(),
        }


# ════════════════════════════════════════════════════════════════════════════
# Meta-Cognition (knows what it doesn't know)
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ConfidenceAssessment:
    """Self-assessment of prediction confidence."""
    overall_confidence: float       # 0-1
    data_sufficiency: float         # do we have enough data?
    pattern_familiarity: float      # have we seen this before?
    model_agreement: float          # do models agree?
    regime_stability: float         # is the regime stable?
    reasoning: str                  # human-readable explanation
    known_unknowns: List[str]       # what we know we don't know
    recommendation: str             # "TRADE", "REDUCE_SIZE", "SKIP", "WAIT"


class MetaCognition:
    """
    ARGUS knows what it doesn't know.

    Before every trade, assesses its own confidence by checking:
    - Data sufficiency: enough history to form reliable signals?
    - Pattern familiarity: have we seen this market setup before?
    - Model agreement: are our models telling the same story?
    - Regime stability: is the current regime stable or transitioning?

    When confidence is low, ARGUS reduces position size or skips entirely.
    This prevents overconfident trading in unfamiliar conditions.
    """

    def __init__(self, min_history_bars: int = 100, min_model_agreement: float = 0.5):
        self._min_history = min_history_bars
        self._min_agreement = min_model_agreement
        self._pattern_library: Dict[str, int] = defaultdict(int)  # pattern_hash → count
        self._decision_outcomes: deque = deque(maxlen=200)

    def record_pattern(self, pattern_key: str) -> None:
        """Record seeing a market pattern."""
        self._pattern_library[pattern_key] += 1

    def record_outcome(self, confidence_was: float, pnl: float) -> None:
        """Record whether high-confidence predictions were actually correct."""
        self._decision_outcomes.append((confidence_was, pnl > 0))

    def assess(
        self,
        symbol: str,
        history_bars: int,
        model_predictions: Dict[str, float],  # model_name → predicted_return
        regime: str,
        regime_age_bars: int,
        pattern_key: str = "",
        advisory: Optional[Dict[str, Any]] = None,
    ) -> ConfidenceAssessment:
        """Assess confidence in the current trading decision."""
        known_unknowns = []

        # 1. Data sufficiency
        data_suff = min(1.0, history_bars / self._min_history)
        if data_suff < 0.5:
            known_unknowns.append(f"Only {history_bars} bars of history (need {self._min_history})")

        # 2. Pattern familiarity
        pattern_count = self._pattern_library.get(pattern_key, 0) if pattern_key else 0
        familiarity = min(1.0, pattern_count / 10)  # seen 10+ times = familiar
        if familiarity < 0.3:
            known_unknowns.append("Unfamiliar market pattern — limited historical reference")

        # 3. Model agreement
        if model_predictions:
            directions = [1 if v > 0 else -1 for v in model_predictions.values()]
            if len(set(directions)) == 1:
                agreement = 1.0
            else:
                majority = max(directions.count(1), directions.count(-1))
                agreement = majority / len(directions)
        else:
            agreement = 0.5
            known_unknowns.append("No model predictions available")

        if agreement < self._min_agreement:
            known_unknowns.append(f"Models disagree: {agreement:.0%} agreement")

        # 4. Regime stability
        regime_stability = min(1.0, regime_age_bars / 50)  # stable after 50 bars
        if regime_stability < 0.3:
            known_unknowns.append(f"Regime '{regime}' only {regime_age_bars} bars old — may be transitioning")

        # 5. Calibration: are our confidence levels actually predictive?
        calibration = 1.0
        if len(self._decision_outcomes) >= 20:
            high_conf = [(c, o) for c, o in self._decision_outcomes if c > 0.6]
            if high_conf:
                actual_accuracy = sum(1 for _, o in high_conf if o) / len(high_conf)
                calibration = actual_accuracy  # if we're 70% confident but only 50% right, cal = 0.5
                if calibration < 0.4:
                    known_unknowns.append(f"Confidence calibration poor: {calibration:.0%} actual vs predicted")

        # Overall confidence: geometric mean of all factors
        factors = [data_suff, familiarity, agreement, regime_stability, calibration]
        overall = 1.0
        for f in factors:
            overall *= max(0.01, f)
        overall = overall ** (1 / len(factors))

        # Recommendation
        if overall > 0.7 and not known_unknowns:
            rec = "TRADE"
        elif overall > 0.5:
            rec = "REDUCE_SIZE"
        elif overall > 0.3:
            rec = "WAIT"
        else:
            rec = "SKIP"

        # Reasoning
        reasoning_parts = []
        if data_suff >= 0.8:
            reasoning_parts.append("sufficient data")
        if familiarity >= 0.5:
            reasoning_parts.append("familiar pattern")
        if agreement >= 0.7:
            reasoning_parts.append("models agree")
        if regime_stability >= 0.5:
            reasoning_parts.append("stable regime")
        reasoning = f"Confidence {overall:.0%}: " + (", ".join(reasoning_parts) if reasoning_parts else "low confidence across all factors")

        return ConfidenceAssessment(
            overall_confidence=overall,
            data_sufficiency=data_suff,
            pattern_familiarity=familiarity,
            model_agreement=agreement,
            regime_stability=regime_stability,
            reasoning=reasoning,
            known_unknowns=known_unknowns,
            recommendation=rec,
        )

    def get_stats(self) -> Dict[str, Any]:
        outcomes = list(self._decision_outcomes)
        return {
            "patterns_known": len(self._pattern_library),
            "decisions_tracked": len(outcomes),
            "calibration": (sum(1 for c, o in outcomes if c > 0.6 and o) /
                           max(sum(1 for c, _ in outcomes if c > 0.6), 1))
            if outcomes else 0,
        }


# ════════════════════════════════════════════════════════════════════════════
# Temporal Abstraction (multi-scale signal separation)
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TemporalSignal:
    """Signal decomposed into time scales."""
    symbol: str
    noise: float            # 1-min scale: random walk component
    micro_trend: float      # 5-15 min: short-term momentum
    meso_trend: float       # 1-4 hour: medium-term trend
    macro_trend: float      # 1+ day: long-term direction
    dominant_scale: str     # which scale has the strongest signal
    alignment: float        # 0-1: how aligned are all scales (1 = all agree)


class TemporalAbstraction:
    """
    Separates signal from noise at multiple time scales.

    A 1-minute dip might be noise. A 4-hour downtrend is real.
    This module decomposes price action into time scales and identifies
    which scale is dominant for the current market.

    Uses exponential moving average decomposition:
    noise = price - EMA(5)
    micro = EMA(5) - EMA(30)
    meso = EMA(30) - EMA(200)
    macro = EMA(200) trend direction
    """

    def __init__(self):
        self._prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))

    def update(self, symbol: str, price: float) -> None:
        self._prices[symbol].append(price)

    def decompose(self, symbol: str) -> TemporalSignal:
        """Decompose current price action into time scales."""
        prices = list(self._prices.get(symbol, []))
        if len(prices) < 30:
            return TemporalSignal(symbol, 0, 0, 0, 0, "insufficient", 0)

        arr = np.array(prices, dtype=float)

        # EMAs at different scales
        ema5 = self._ema(arr, 5)
        ema30 = self._ema(arr, 30)
        ema200 = self._ema(arr, min(200, len(arr) - 1))

        current = arr[-1]
        # Decompose into scales (normalised as % of price)
        noise = (current - ema5) / max(current, 1e-9) * 100
        micro = (ema5 - ema30) / max(current, 1e-9) * 100
        meso = (ema30 - ema200) / max(current, 1e-9) * 100
        macro_dir = (ema200 - arr[0]) / max(arr[0], 1e-9) * 100 if len(arr) > 200 else meso

        # Which scale is dominant?
        scales = {"noise": abs(noise), "micro": abs(micro),
                  "meso": abs(meso), "macro": abs(macro_dir)}
        dominant = max(scales, key=scales.get)

        # Alignment: all scales agree on direction
        directions = [1 if x > 0 else -1 for x in [noise, micro, meso, macro_dir]]
        agree = max(directions.count(1), directions.count(-1))
        alignment = agree / 4.0

        return TemporalSignal(
            symbol=symbol, noise=noise, micro_trend=micro,
            meso_trend=meso, macro_trend=macro_dir,
            dominant_scale=dominant, alignment=alignment,
        )

    def _ema(self, arr: np.ndarray, period: int) -> float:
        if len(arr) < period:
            return float(arr[-1])
        alpha = 2.0 / (period + 1)
        ema = float(arr[0])
        for i in range(1, len(arr)):
            ema = alpha * arr[i] + (1 - alpha) * ema
        return ema

    def get_stats(self) -> Dict[str, Any]:
        return {
            "symbols_tracked": len(self._prices),
            "decompositions": {sym: self.decompose(sym).dominant_scale
                              for sym in list(self._prices.keys())[:5]},
        }


# ════════════════════════════════════════════════════════════════════════════
# Adversarial Thinker
# ════════════════════════════════════════════════════════════════════════════

class AdversarialThinker:
    """
    Models counterparty behavior.

    When ARGUS buys, someone is selling. WHY are they selling?
    - Retail panic? → Buy more (they're wrong)
    - Whale distribution? → Be cautious (they know something)
    - Market maker inventory? → Neutral (just providing liquidity)

    Tracks patterns: when large orders appear on the opposite side,
    who tends to be right — us or them?
    """

    def __init__(self):
        self._counterparty_outcomes: Dict[str, List[float]] = defaultdict(list)
        # Track: "we bought when X sold" → was our buy profitable?

    def record_counterparty(
        self,
        our_side: str,
        counterparty_type: str,     # "retail", "whale", "market_maker", "unknown"
        our_pnl: float,
    ) -> None:
        """Record the outcome when we traded against a certain counterparty type."""
        key = f"{our_side}_vs_{counterparty_type}"
        self._counterparty_outcomes[key].append(our_pnl)
        if len(self._counterparty_outcomes[key]) > 100:
            self._counterparty_outcomes[key] = self._counterparty_outcomes[key][-100:]

    def assess_counterparty(self, our_side: str, counterparty_type: str) -> float:
        """Should we trade against this counterparty? Returns -1 (avoid) to +1 (confident)."""
        key = f"{our_side}_vs_{counterparty_type}"
        outcomes = self._counterparty_outcomes.get(key, [])
        if len(outcomes) < 5:
            return 0.0  # insufficient data
        avg_pnl = sum(outcomes) / len(outcomes)
        win_rate = sum(1 for p in outcomes if p > 0) / len(outcomes)
        # Positive avg PnL + high win rate = we tend to be right against them
        return max(-1, min(1, avg_pnl * 0.3 + (win_rate - 0.5) * 2))

    def get_stats(self) -> Dict[str, Any]:
        return {
            "counterparty_types_tracked": len(self._counterparty_outcomes),
            "assessments": {k: self.assess_counterparty(k.split("_vs_")[0], k.split("_vs_")[1])
                           for k in self._counterparty_outcomes if len(self._counterparty_outcomes[k]) >= 5},
        }
