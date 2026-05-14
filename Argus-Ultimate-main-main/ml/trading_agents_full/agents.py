from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .llm_integration import LLMProvider

logger = logging.getLogger(__name__)

_VALID_ACTIONS = {"buy", "sell", "hold"}


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]


@dataclass(slots=True)
class MarketContext:
    symbol: str
    timestamp: float = field(default_factory=time.time)
    market_data: Dict[str, float] = field(default_factory=dict)
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    fundamentals: Dict[str, float] = field(default_factory=dict)
    sentiment_data: Dict[str, float] = field(default_factory=dict)
    news_data: Dict[str, float] = field(default_factory=dict)
    portfolio_state: Dict[str, float] = field(default_factory=dict)
    risk_limits: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    news_headlines: List[str] = field(default_factory=list)
    regime: str = "unknown"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MarketContext":
        return cls(
            symbol=str(payload.get("symbol", "UNKNOWN")),
            timestamp=_safe_float(payload.get("timestamp"), time.time()),
            market_data={str(k): _safe_float(v) for k, v in _to_dict(payload.get("market_data")).items()},
            technical_indicators={str(k): _safe_float(v) for k, v in _to_dict(payload.get("technical_indicators")).items()},
            fundamentals={str(k): _safe_float(v) for k, v in _to_dict(payload.get("fundamentals")).items()},
            sentiment_data={str(k): _safe_float(v) for k, v in _to_dict(payload.get("sentiment_data")).items()},
            news_data={str(k): _safe_float(v) for k, v in _to_dict(payload.get("news_data")).items()},
            portfolio_state={str(k): _safe_float(v) for k, v in _to_dict(payload.get("portfolio_state")).items()},
            risk_limits={str(k): _safe_float(v) for k, v in _to_dict(payload.get("risk_limits")).items()},
            metadata=dict(payload.get("metadata", {}) or {}),
            news_headlines=_to_list(payload.get("news_headlines")),
            regime=str(payload.get("regime", payload.get("market_regime", "unknown")) or "unknown"),
        )


@dataclass(slots=True)
class AgentAnalysis:
    agent_name: str
    action: str
    confidence: float
    score: float
    thesis: str
    reasoning: List[str] = field(default_factory=list)
    evidence: Dict[str, float] = field(default_factory=dict)
    risk_flags: List[str] = field(default_factory=list)
    llm_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.action = str(self.action or "hold").lower().strip()
        if self.action not in _VALID_ACTIONS:
            self.action = "hold"
        self.confidence = _clamp(self.confidence, 0.0, 1.0)
        self.score = _clamp(self.score, -1.0, 1.0)


@dataclass(slots=True)
class AgentDebate:
    agent_name: str
    stance: str
    thesis: str
    rebuttals: List[str] = field(default_factory=list)
    concessions: List[str] = field(default_factory=list)
    argument_score: float = 0.0
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.stance = str(self.stance or "hold").lower().strip()
        if self.stance not in _VALID_ACTIONS:
            self.stance = "hold"
        self.argument_score = _clamp(self.argument_score, 0.0, 1.0)
        self.confidence = _clamp(self.confidence, 0.0, 1.0)


@dataclass(slots=True)
class AgentSynthesis:
    agent_name: str
    final_action: str
    confidence: float
    summary: str
    key_points: List[str] = field(default_factory=list)
    follow_ups: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.final_action = str(self.final_action or "hold").lower().strip()
        if self.final_action not in _VALID_ACTIONS:
            self.final_action = "hold"
        self.confidence = _clamp(self.confidence, 0.0, 1.0)


class BaseAgent(ABC):
    """Base class for all trading agents."""
    agent_name = "base_agent"
    role_name = "Generic analyst"

    def __init__(self, llm_provider: Optional[LLMProvider] = None) -> None:
        self.llm_provider = llm_provider

    def analyze(self, context: MarketContext | Mapping[str, Any]) -> AgentAnalysis:
        parsed = self._ensure_context(context)
        heuristic = self._analyze_heuristic(parsed)
        try:
            enriched = self._apply_llm_overlay(parsed, heuristic)
            logger.debug("%s analysis complete for %s", self.agent_name, parsed.symbol)
            return enriched
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s llm overlay failed for %s: %s", self.agent_name, parsed.symbol, exc)
            heuristic.risk_flags.append("llm_overlay_unavailable")
            return heuristic

    @abstractmethod
    def _analyze_heuristic(self, context: MarketContext) -> AgentAnalysis:
        """Implement heuristic analysis in subclasses."""
        ...

    def debate(
        self,
        analysis: AgentAnalysis,
        opposing_views: Sequence[AgentAnalysis],
        context: MarketContext | Mapping[str, Any],
    ) -> AgentDebate:
        parsed = self._ensure_context(context)
        strongest_opp = sorted(opposing_views, key=lambda item: item.confidence * abs(item.score), reverse=True)
        rebuttals = [
            f"Counters {item.agent_name}: {item.thesis}"
            for item in strongest_opp[:2]
            if item.action != analysis.action
        ]
        concessions = []
        if parsed.regime.lower() in {"stress_bear", "high_volatility_range"} and analysis.action == "buy":
            concessions.append("Regime is unsupportive for aggressive long exposure.")
        if "risk_limit_breached" in analysis.risk_flags:
            concessions.append("Risk capacity is already constrained.")
        argument_score = _clamp((analysis.confidence * 0.55) + (abs(analysis.score) * 0.30) - (0.08 * len(concessions)), 0.0, 1.0)
        return AgentDebate(
            agent_name=self.agent_name,
            stance=analysis.action,
            thesis=analysis.thesis,
            rebuttals=rebuttals,
            concessions=concessions,
            argument_score=argument_score,
            confidence=analysis.confidence,
        )

    def synthesize(
        self,
        analysis: AgentAnalysis,
        debate: Optional[AgentDebate],
        context: MarketContext | Mapping[str, Any],
    ) -> AgentSynthesis:
        parsed = self._ensure_context(context)
        key_points = list(analysis.reasoning[:3])
        if debate is not None:
            key_points.extend(debate.rebuttals[:1])
        follow_ups = [f"Re-check {self.role_name.lower()} inputs next cycle for {parsed.symbol}."]
        if analysis.action != "hold":
            follow_ups.append("Validate execution sizing against risk budget before entry.")
        return AgentSynthesis(
            agent_name=self.agent_name,
            final_action=analysis.action,
            confidence=analysis.confidence,
            summary=f"{self.role_name} synthesis remains {analysis.action} for {parsed.symbol}.",
            key_points=key_points,
            follow_ups=follow_ups,
        )

    def _apply_llm_overlay(self, context: MarketContext, heuristic: AgentAnalysis) -> AgentAnalysis:
        if self.llm_provider is None:
            return heuristic
        prompt = self.llm_provider.build_prompt(self.agent_name, context, heuristic)
        response = self.llm_provider.generate(prompt, role=self.agent_name, context=context, heuristic=heuristic)
        if not response.used_llm:
            heuristic.metadata["llm_fallback_reason"] = response.metadata.get("fallback_reason", "heuristic")
            return heuristic
        adjusted_score = _clamp((heuristic.score * 0.65) + (response.score * 0.35), -1.0, 1.0)
        adjusted_confidence = _clamp((heuristic.confidence * 0.60) + (response.confidence * 0.40), 0.0, 1.0)
        heuristic.score = adjusted_score
        heuristic.confidence = adjusted_confidence
        heuristic.action = self._action_from_score(adjusted_score)
        heuristic.llm_used = True
        heuristic.metadata["llm_provider"] = response.provider
        heuristic.metadata["llm_model"] = response.model
        heuristic.metadata["token_usage"] = {
            "prompt_tokens": response.token_usage.prompt_tokens,
            "completion_tokens": response.token_usage.completion_tokens,
            "estimated_cost_usd": response.token_usage.estimated_cost_usd,
        }
        if response.reasoning:
            heuristic.reasoning.append(f"LLM overlay: {response.reasoning}")
        return heuristic

    def _analyze_heuristic(self, context: MarketContext) -> AgentAnalysis:
        raise NotImplementedError

    def _ensure_context(self, context: MarketContext | Mapping[str, Any]) -> MarketContext:
        if isinstance(context, MarketContext):
            return context
        return MarketContext.from_mapping(context)

    def _build_analysis(
        self,
        *,
        score: float,
        thesis: str,
        reasoning: Sequence[str],
        evidence: Mapping[str, float],
        risk_flags: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AgentAnalysis:
        score = _clamp(score, -1.0, 1.0)
        return AgentAnalysis(
            agent_name=self.agent_name,
            action=self._action_from_score(score),
            confidence=self._confidence(score, evidence),
            score=score,
            thesis=thesis,
            reasoning=list(reasoning),
            evidence=dict(evidence),
            risk_flags=list(risk_flags or []),
            metadata=dict(metadata or {}),
        )

    def _action_from_score(self, score: float) -> str:
        if score > 0.15:
            return "buy"
        if score < -0.15:
            return "sell"
        return "hold"

    def _confidence(self, score: float, evidence: Mapping[str, float]) -> float:
        support = min(sum(abs(_safe_float(value)) for value in evidence.values()) / max(len(evidence), 1), 1.0)
        return _clamp(0.25 + (0.45 * abs(score)) + (0.30 * support), 0.05, 0.98)


class FundamentalAnalyst(BaseAgent):
    agent_name = "fundamental_analyst"
    role_name = "Fundamental analyst"

    def _analyze_heuristic(self, context: MarketContext) -> AgentAnalysis:
        revenue_growth = _safe_float(context.fundamentals.get("revenue_growth"))
        earnings_growth = _safe_float(context.fundamentals.get("earnings_growth"))
        valuation_discount = _safe_float(context.fundamentals.get("valuation_discount"))
        debt_to_equity = _safe_float(context.fundamentals.get("debt_to_equity"))
        free_cash_flow_margin = _safe_float(context.fundamentals.get("free_cash_flow_margin"))
        score = _clamp(
            (0.28 * revenue_growth)
            + (0.26 * earnings_growth)
            + (0.22 * valuation_discount)
            + (0.18 * free_cash_flow_margin)
            - (0.20 * debt_to_equity),
            -1.0,
            1.0,
        )
        risk_flags = ["fundamental_leverage_risk"] if debt_to_equity > 0.85 else []
        return self._build_analysis(
            score=score,
            thesis=f"Fundamentals support a {self._action_from_score(score)} bias in {context.symbol}.",
            reasoning=[
                f"Revenue growth={revenue_growth:.2f} and earnings growth={earnings_growth:.2f} drive the quality view.",
                f"Valuation discount={valuation_discount:.2f} and FCF margin={free_cash_flow_margin:.2f} affect upside durability.",
                f"Debt-to-equity={debt_to_equity:.2f} is treated as a balance-sheet drag.",
            ],
            evidence={
                "revenue_growth": revenue_growth,
                "earnings_growth": earnings_growth,
                "valuation_discount": valuation_discount,
                "free_cash_flow_margin": free_cash_flow_margin,
                "debt_to_equity": debt_to_equity,
            },
            risk_flags=risk_flags,
        )


class SentimentAnalyst(BaseAgent):
    agent_name = "sentiment_analyst"
    role_name = "Sentiment analyst"

    def _analyze_heuristic(self, context: MarketContext) -> AgentAnalysis:
        news_score = _safe_float(context.sentiment_data.get("news_score"))
        social_score = _safe_float(context.sentiment_data.get("social_score"))
        fear_greed = _safe_float(context.sentiment_data.get("fear_greed_index"))
        crowding = _safe_float(context.market_data.get("crowding_risk"))
        normalized_fg = (fear_greed - 50.0) / 50.0 if fear_greed else 0.0
        score = _clamp((0.40 * news_score) + (0.35 * social_score) + (0.20 * normalized_fg) - (0.15 * crowding), -1.0, 1.0)
        return self._build_analysis(
            score=score,
            thesis=f"Narrative and crowd sentiment are {self._action_from_score(score)} for {context.symbol}.",
            reasoning=[
                f"News score={news_score:.2f}, social score={social_score:.2f}.",
                f"Fear/greed normalization={normalized_fg:.2f} with crowding penalty={crowding:.2f}.",
            ],
            evidence={
                "news_score": news_score,
                "social_score": social_score,
                "fear_greed_normalized": normalized_fg,
                "crowding_risk": crowding,
            },
            risk_flags=["sentiment_extreme"] if abs(normalized_fg) > 0.85 else [],
        )


class TechnicalAnalyst(BaseAgent):
    agent_name = "technical_analyst"
    role_name = "Technical analyst"

    def _analyze_heuristic(self, context: MarketContext) -> AgentAnalysis:
        rsi = _safe_float(context.technical_indicators.get("rsi"), 50.0)
        macd = _safe_float(context.technical_indicators.get("macd"))
        trend_strength = _safe_float(context.technical_indicators.get("trend_strength"))
        breakout_score = _safe_float(context.technical_indicators.get("breakout_score"))
        volume_confirmation = _safe_float(context.technical_indicators.get("volume_confirmation"))
        normalized_rsi = (rsi - 50.0) / 50.0
        score = _clamp(
            (0.22 * normalized_rsi)
            + (0.22 * macd)
            + (0.28 * trend_strength)
            + (0.18 * breakout_score)
            + (0.10 * volume_confirmation),
            -1.0,
            1.0,
        )
        risk_flags = []
        if rsi >= 75:
            risk_flags.append("overbought")
        if rsi <= 25:
            risk_flags.append("oversold")
        return self._build_analysis(
            score=score,
            thesis=f"Price structure and indicators lean {self._action_from_score(score)} for {context.symbol}.",
            reasoning=[
                f"RSI={rsi:.2f}, MACD={macd:.2f}, trend={trend_strength:.2f}.",
                f"Breakout={breakout_score:.2f}, volume confirmation={volume_confirmation:.2f}.",
            ],
            evidence={
                "rsi": normalized_rsi,
                "macd": macd,
                "trend_strength": trend_strength,
                "breakout_score": breakout_score,
                "volume_confirmation": volume_confirmation,
            },
            risk_flags=risk_flags,
        )


class NewsAnalyst(BaseAgent):
    agent_name = "news_analyst"
    role_name = "News analyst"

    def _analyze_heuristic(self, context: MarketContext) -> AgentAnalysis:
        event_impact = _safe_float(context.news_data.get("event_impact"))
        surprise = _safe_float(context.news_data.get("surprise_score"))
        regulatory_risk = _safe_float(context.news_data.get("regulatory_risk"))
        headline_density = min(len(context.news_headlines) / 8.0, 1.0)
        score = _clamp((0.40 * event_impact) + (0.30 * surprise) - (0.30 * regulatory_risk), -1.0, 1.0)
        return self._build_analysis(
            score=score,
            thesis=f"Event flow and breaking news are {self._action_from_score(score)} for {context.symbol}.",
            reasoning=[
                f"Event impact={event_impact:.2f}, surprise score={surprise:.2f}, regulatory risk={regulatory_risk:.2f}.",
                f"Headline density factor={headline_density:.2f} from {len(context.news_headlines)} headlines.",
            ],
            evidence={
                "event_impact": event_impact,
                "surprise_score": surprise,
                "regulatory_risk": regulatory_risk,
                "headline_density": headline_density,
            },
            risk_flags=["regulatory_overhang"] if regulatory_risk > 0.7 else [],
        )


class BullResearcher(BaseAgent):
    agent_name = "bull_researcher"
    role_name = "Bull researcher"

    def _analyze_heuristic(self, context: MarketContext) -> AgentAnalysis:
        growth = max(_safe_float(context.fundamentals.get("revenue_growth")), 0.0)
        trend = max(_safe_float(context.technical_indicators.get("trend_strength")), 0.0)
        narrative = max(_safe_float(context.sentiment_data.get("news_score")), 0.0)
        liquidity = max(_safe_float(context.market_data.get("liquidity_score"), 0.5) - 0.3, 0.0)
        catalysts = max(_safe_float(context.news_data.get("event_impact")), 0.0)
        score = _clamp((0.24 * growth) + (0.24 * trend) + (0.18 * narrative) + (0.17 * liquidity) + (0.17 * catalysts), 0.0, 1.0)
        return self._build_analysis(
            score=score,
            thesis=f"Upside catalysts create a bullish asymmetric case in {context.symbol}.",
            reasoning=[
                f"Growth={growth:.2f}, trend={trend:.2f}, narrative={narrative:.2f}.",
                f"Liquidity support={liquidity:.2f}, event catalyst strength={catalysts:.2f}.",
            ],
            evidence={
                "growth": growth,
                "trend": trend,
                "narrative": narrative,
                "liquidity": liquidity,
                "catalysts": catalysts,
            },
        )


class BearResearcher(BaseAgent):
    agent_name = "bear_researcher"
    role_name = "Bear researcher"

    def _analyze_heuristic(self, context: MarketContext) -> AgentAnalysis:
        drawdown_risk = max(_safe_float(context.market_data.get("drawdown_risk")), 0.0)
        volatility = max(_safe_float(context.market_data.get("realized_volatility")), 0.0)
        leverage = max(_safe_float(context.fundamentals.get("debt_to_equity")), 0.0)
        negative_news = max(-_safe_float(context.sentiment_data.get("news_score")), 0.0)
        regulatory_risk = max(_safe_float(context.news_data.get("regulatory_risk")), 0.0)
        downside = _clamp((0.24 * drawdown_risk) + (0.22 * volatility) + (0.18 * leverage) + (0.18 * negative_news) + (0.18 * regulatory_risk), 0.0, 1.0)
        return self._build_analysis(
            score=-downside,
            thesis=f"Downside risk factors dominate the bearish case in {context.symbol}.",
            reasoning=[
                f"Drawdown risk={drawdown_risk:.2f}, volatility={volatility:.2f}, leverage={leverage:.2f}.",
                f"Negative narrative={negative_news:.2f}, regulatory risk={regulatory_risk:.2f}.",
            ],
            evidence={
                "drawdown_risk": drawdown_risk,
                "volatility": volatility,
                "leverage": leverage,
                "negative_news": negative_news,
                "regulatory_risk": regulatory_risk,
            },
            risk_flags=["tail_risk_elevated"] if downside > 0.7 else [],
        )


class RiskManager(BaseAgent):
    agent_name = "risk_manager"
    role_name = "Risk manager"

    def _analyze_heuristic(self, context: MarketContext) -> AgentAnalysis:
        gross_exposure = abs(_safe_float(context.portfolio_state.get("gross_exposure")))
        position_exposure = abs(_safe_float(context.portfolio_state.get("symbol_exposure")))
        value_at_risk = abs(_safe_float(context.portfolio_state.get("value_at_risk")))
        drawdown = abs(_safe_float(context.portfolio_state.get("drawdown")))
        max_exposure = max(_safe_float(context.risk_limits.get("max_exposure"), 1.0), 1e-6)
        max_position = max(_safe_float(context.risk_limits.get("max_position"), max_exposure), 1e-6)
        max_var = max(_safe_float(context.risk_limits.get("max_value_at_risk"), 1.0), 1e-6)
        max_drawdown = max(_safe_float(context.risk_limits.get("max_drawdown"), 1.0), 1e-6)
        utilization = max(
            gross_exposure / max_exposure,
            position_exposure / max_position,
            value_at_risk / max_var,
            drawdown / max_drawdown,
        )
        buffer_score = _clamp(0.45 - utilization, -1.0, 1.0)
        risk_flags: List[str] = []
        if utilization >= 1.0:
            risk_flags.append("risk_limit_breached")
        elif utilization >= 0.8:
            risk_flags.append("risk_limit_near")
        if drawdown / max_drawdown >= 0.85:
            risk_flags.append("drawdown_pressure")
        return self._build_analysis(
            score=buffer_score,
            thesis=f"Risk budget implies {self._action_from_score(buffer_score)} capacity for {context.symbol}.",
            reasoning=[
                f"Exposure utilization={gross_exposure / max_exposure:.2f}, single-name utilization={position_exposure / max_position:.2f}.",
                f"VaR utilization={value_at_risk / max_var:.2f}, drawdown utilization={drawdown / max_drawdown:.2f}.",
            ],
            evidence={
                "gross_exposure_utilization": gross_exposure / max_exposure,
                "position_utilization": position_exposure / max_position,
                "var_utilization": value_at_risk / max_var,
                "drawdown_utilization": drawdown / max_drawdown,
            },
            risk_flags=risk_flags,
            metadata={"risk_utilization": utilization},
        )
