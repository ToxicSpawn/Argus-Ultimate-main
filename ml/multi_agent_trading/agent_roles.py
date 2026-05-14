from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

logger = logging.getLogger(__name__)

_VALID_STANCES = {"buy", "sell", "hold"}


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_list(values: Any) -> List[Any]:
    if values is None:
        return []
    if isinstance(values, list):
        return values
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        return list(values)
    return [values]


@dataclass(slots=True)
class AgentContext:
    symbol: str
    timestamp: float = field(default_factory=time.time)
    market_data: Dict[str, float] = field(default_factory=dict)
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    fundamentals: Dict[str, float] = field(default_factory=dict)
    sentiment_data: Dict[str, float] = field(default_factory=dict)
    portfolio_state: Dict[str, float] = field(default_factory=dict)
    risk_limits: Dict[str, float] = field(default_factory=dict)
    macro_data: Dict[str, float] = field(default_factory=dict)
    market_regime: str = "unknown"
    news_headlines: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AgentContext":
        return cls(
            symbol=str(payload.get("symbol", "UNKNOWN")),
            timestamp=_safe_float(payload.get("timestamp"), time.time()),
            market_data=dict(payload.get("market_data", {}) or {}),
            technical_indicators=dict(payload.get("technical_indicators", {}) or {}),
            fundamentals=dict(payload.get("fundamentals", {}) or {}),
            sentiment_data=dict(payload.get("sentiment_data", {}) or {}),
            portfolio_state=dict(payload.get("portfolio_state", {}) or {}),
            risk_limits=dict(payload.get("risk_limits", {}) or {}),
            macro_data=dict(payload.get("macro_data", {}) or {}),
            market_regime=str(payload.get("market_regime", "unknown") or "unknown"),
            news_headlines=[str(item) for item in _to_list(payload.get("news_headlines"))],
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(slots=True)
class AgentAnalysis:
    agent_name: str
    stance: str
    confidence: float
    score: float
    summary: str
    rationale: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    evidence: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.stance = str(self.stance or "hold").lower().strip()
        if self.stance not in _VALID_STANCES:
            self.stance = "hold"
        self.confidence = _clamp(self.confidence, 0.0, 1.0)
        self.score = _clamp(self.score, -1.0, 1.0)
        self.summary = str(self.summary or "")

    @property
    def signed_confidence(self) -> float:
        direction = {"buy": 1.0, "hold": 0.0, "sell": -1.0}[self.stance]
        return direction * self.confidence


@dataclass(slots=True)
class AgentSynthesis:
    agent_name: str
    conclusion: str
    action_bias: str
    confidence: float
    key_points: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.action_bias = str(self.action_bias or "hold").lower().strip()
        if self.action_bias not in _VALID_STANCES:
            self.action_bias = "hold"
        self.confidence = _clamp(self.confidence, 0.0, 1.0)


class BaseTradingAgent(ABC):
    """Base class for all trading agents."""
    agent_name = "base_agent"
    role_description = "generic market analyst"

    def analyze(self, context: AgentContext | Mapping[str, Any]) -> AgentAnalysis:
        parsed = self._ensure_context(context)
        try:
            analysis = self._analyze_impl(parsed)
            analysis = self._apply_prompt_variant(analysis, parsed)
            logger.debug("%s generated analysis for %s", self.agent_name, parsed.symbol)
            return analysis
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s analyze failed for %s: %s", self.agent_name, parsed.symbol, exc)
            return AgentAnalysis(
                agent_name=self.agent_name,
                stance="hold",
                confidence=0.05,
                score=0.0,
                summary=f"{self.agent_name} failed and defaulted to neutral.",
                rationale=[f"Error during analysis: {exc}"],
                risk_flags=["analysis_failure"],
            )

    @abstractmethod
    def _analyze_impl(self, context: AgentContext) -> AgentAnalysis:
        """Implement analysis logic in subclasses."""
        ...

    def debate(
        self,
        context: AgentContext | Mapping[str, Any],
        peer_analysis: Optional[AgentAnalysis] = None,
    ) -> AgentAnalysis:
        base_analysis = self.analyze(context)
        if peer_analysis is not None:
            peer_note = (
                f"Counterpoint from {peer_analysis.agent_name}: {peer_analysis.summary} "
                f"(confidence={peer_analysis.confidence:.2f})."
            )
            base_analysis.rationale.append(peer_note)
            if peer_analysis.stance != base_analysis.stance and peer_analysis.confidence > base_analysis.confidence:
                base_analysis.risk_flags.append("strong_opposition")
                base_analysis.confidence = _clamp(base_analysis.confidence * 0.9, 0.0, 1.0)
        return base_analysis

    def synthesize(
        self,
        analyses: Sequence[AgentAnalysis],
        context: AgentContext | Mapping[str, Any],
    ) -> AgentSynthesis:
        parsed = self._ensure_context(context)
        if not analyses:
            return AgentSynthesis(
                agent_name=self.agent_name,
                conclusion="No peer analyses available; remain neutral.",
                action_bias="hold",
                confidence=0.0,
                key_points=[f"No analyses supplied for {parsed.symbol}."],
            )

        avg_score = sum(item.score for item in analyses) / len(analyses)
        avg_confidence = sum(item.confidence for item in analyses) / len(analyses)
        action_bias = "buy" if avg_score > 0.15 else "sell" if avg_score < -0.15 else "hold"
        ranked = sorted(analyses, key=lambda item: abs(item.score) * item.confidence, reverse=True)
        key_points = [item.summary for item in ranked[:3]]
        return AgentSynthesis(
            agent_name=self.agent_name,
            conclusion=f"Synthesis for {parsed.symbol}: consensus bias is {action_bias}.",
            action_bias=action_bias,
            confidence=_clamp(avg_confidence, 0.0, 1.0),
            key_points=key_points,
            next_actions=["Cross-check with execution and risk rules before trading."],
        )

    def _ensure_context(self, context: AgentContext | Mapping[str, Any]) -> AgentContext:
        if isinstance(context, AgentContext):
            return context
        return AgentContext.from_mapping(context)

    def _build_analysis(
        self,
        *,
        stance: str,
        confidence: float,
        score: float,
        summary: str,
        rationale: Sequence[str],
        risk_flags: Optional[Sequence[str]] = None,
        evidence: Optional[Mapping[str, float]] = None,
        metadata: Optional[MutableMapping[str, Any]] = None,
    ) -> AgentAnalysis:
        return AgentAnalysis(
            agent_name=self.agent_name,
            stance=stance,
            confidence=confidence,
            score=score,
            summary=summary,
            rationale=list(rationale),
            risk_flags=list(risk_flags or []),
            evidence=dict(evidence or {}),
            metadata=dict(metadata or {}),
        )

    def _stance_from_score(self, score: float) -> str:
        if score > 0.15:
            return "buy"
        if score < -0.15:
            return "sell"
        return "hold"

    def _confidence_from_score(self, score: float, support: float) -> float:
        magnitude = min(abs(score), 1.0)
        return _clamp(0.2 + (0.55 * magnitude) + (0.25 * _clamp(support, 0.0, 1.0)), 0.0, 1.0)

    def _analyze_impl(self, context: AgentContext) -> AgentAnalysis:
        raise NotImplementedError

    def _apply_prompt_variant(self, analysis: AgentAnalysis, context: AgentContext) -> AgentAnalysis:
        raw_prompt_variants = context.metadata.get("prompt_variants", {})
        if not isinstance(raw_prompt_variants, Mapping):
            return analysis
        variant = raw_prompt_variants.get(self.agent_name)
        if not isinstance(variant, Mapping):
            return analysis

        raw_metadata = variant.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raw_metadata = {}

        metadata = {str(key): value for key, value in raw_metadata.items()}
        prompt_text = str(variant.get("prompt_text", "") or "")
        score_bias = _safe_float(metadata.get("score_bias"))
        confidence_bias = _safe_float(metadata.get("confidence_bias"))
        emphasis = str(metadata.get("emphasis", "") or "").lower()

        if "conservative" in prompt_text.lower() or emphasis == "risk":
            confidence_bias -= 0.05
            if analysis.score > 0.0:
                score_bias -= 0.05
        if "aggressive" in prompt_text.lower() or emphasis == "momentum":
            confidence_bias += 0.05
            if analysis.score > 0.0:
                score_bias += 0.05

        analysis.score = _clamp(analysis.score + score_bias, -1.0, 1.0)
        analysis.confidence = _clamp(analysis.confidence + confidence_bias, 0.0, 1.0)
        analysis.stance = self._stance_from_score(analysis.score)
        analysis.metadata["prompt_variant_id"] = str(variant.get("variant_id", ""))
        analysis.metadata["prompt_variant_name"] = str(variant.get("name", ""))
        analysis.metadata["prompt_emphasis"] = emphasis
        if prompt_text:
            analysis.rationale.append(f"Prompt variant influence applied: {variant.get('name', 'default')}.")
        return analysis


class FundamentalAnalyst(BaseTradingAgent):
    agent_name = "fundamental_analyst"
    role_description = "evaluates intrinsic value, growth, and balance-sheet strength"

    def _analyze_impl(self, context: AgentContext) -> AgentAnalysis:
        fundamentals = context.fundamentals
        revenue_growth = _safe_float(fundamentals.get("revenue_growth"))
        earnings_growth = _safe_float(fundamentals.get("earnings_growth"))
        valuation = _safe_float(fundamentals.get("valuation_discount"))
        debt_ratio = _safe_float(fundamentals.get("debt_to_equity"))
        cash_flow = _safe_float(fundamentals.get("free_cash_flow_margin"))
        quality = (0.30 * revenue_growth) + (0.25 * earnings_growth) + (0.20 * valuation) + (0.15 * cash_flow) - (0.20 * debt_ratio)
        score = _clamp(quality, -1.0, 1.0)
        rationale = [
            f"Revenue growth={revenue_growth:.2f}, earnings growth={earnings_growth:.2f}.",
            f"Valuation discount={valuation:.2f}, free cash flow margin={cash_flow:.2f}.",
            f"Debt-to-equity penalty applied from {debt_ratio:.2f}.",
        ]
        risk_flags = ["balance_sheet_stress"] if debt_ratio > 0.8 else []
        return self._build_analysis(
            stance=self._stance_from_score(score),
            confidence=self._confidence_from_score(score, 0.7 if fundamentals else 0.2),
            score=score,
            summary=f"Fundamentals imply a {self._stance_from_score(score)} bias for {context.symbol}.",
            rationale=rationale,
            risk_flags=risk_flags,
            evidence={
                "revenue_growth": revenue_growth,
                "earnings_growth": earnings_growth,
                "valuation_discount": valuation,
                "free_cash_flow_margin": cash_flow,
                "debt_to_equity": debt_ratio,
            },
        )


class SentimentAnalyst(BaseTradingAgent):
    agent_name = "sentiment_analyst"
    role_description = "interprets news flow, crowd positioning, and narrative momentum"

    def _analyze_impl(self, context: AgentContext) -> AgentAnalysis:
        sentiment = context.sentiment_data
        news_score = _safe_float(sentiment.get("news_score"))
        social_score = _safe_float(sentiment.get("social_score"))
        analyst_score = _safe_float(sentiment.get("analyst_revision_score"))
        headline_boost = min(len(context.news_headlines), 10) / 10.0
        score = _clamp((0.45 * news_score) + (0.35 * social_score) + (0.20 * analyst_score), -1.0, 1.0)
        rationale = [
            f"News tone={news_score:.2f}, social sentiment={social_score:.2f}.",
            f"Analyst revision score={analyst_score:.2f} with headline coverage factor={headline_boost:.2f}.",
        ]
        risk_flags = ["headline_density_low"] if headline_boost < 0.2 else []
        return self._build_analysis(
            stance=self._stance_from_score(score),
            confidence=self._confidence_from_score(score, headline_boost),
            score=score,
            summary=f"Sentiment flow is {self._stance_from_score(score)} for {context.symbol}.",
            rationale=rationale,
            risk_flags=risk_flags,
            evidence={
                "news_score": news_score,
                "social_score": social_score,
                "analyst_revision_score": analyst_score,
                "headline_factor": headline_boost,
            },
        )


class TechnicalAnalyst(BaseTradingAgent):
    agent_name = "technical_analyst"
    role_description = "reads momentum, trend structure, and tactical price action"

    def _analyze_impl(self, context: AgentContext) -> AgentAnalysis:
        indicators = context.technical_indicators
        rsi = _safe_float(indicators.get("rsi"), 50.0)
        macd = _safe_float(indicators.get("macd"))
        trend = _safe_float(indicators.get("trend_strength"))
        breakout = _safe_float(indicators.get("breakout_score"))
        vol_penalty = _safe_float(indicators.get("volatility_penalty"))
        normalized_rsi = (rsi - 50.0) / 50.0
        score = _clamp((0.25 * normalized_rsi) + (0.25 * macd) + (0.30 * trend) + (0.25 * breakout) - (0.15 * vol_penalty), -1.0, 1.0)
        risk_flags = []
        if rsi > 75:
            risk_flags.append("overbought")
        elif rsi < 25:
            risk_flags.append("oversold")
        return self._build_analysis(
            stance=self._stance_from_score(score),
            confidence=self._confidence_from_score(score, abs(trend)),
            score=score,
            summary=f"Technical setup leans {self._stance_from_score(score)} for {context.symbol}.",
            rationale=[
                f"RSI={rsi:.2f}, MACD={macd:.2f}, trend strength={trend:.2f}.",
                f"Breakout score={breakout:.2f}, volatility penalty={vol_penalty:.2f}.",
            ],
            risk_flags=risk_flags,
            evidence={
                "rsi": rsi,
                "macd": macd,
                "trend_strength": trend,
                "breakout_score": breakout,
                "volatility_penalty": vol_penalty,
            },
        )


class BullResearcher(BaseTradingAgent):
    agent_name = "bull_researcher"
    role_description = "constructs the strongest bullish case from the evidence"

    def _analyze_impl(self, context: AgentContext) -> AgentAnalysis:
        momentum = _safe_float(context.technical_indicators.get("trend_strength"))
        growth = _safe_float(context.fundamentals.get("revenue_growth"))
        sentiment = _safe_float(context.sentiment_data.get("news_score"))
        liquidity = _safe_float(context.market_data.get("liquidity_score"), 0.5)
        score = _clamp(max(momentum, 0.0) * 0.35 + max(growth, 0.0) * 0.35 + max(sentiment, 0.0) * 0.20 + max(liquidity - 0.5, 0.0) * 0.20, -1.0, 1.0)
        return self._build_analysis(
            stance="buy" if score > 0.10 else "hold",
            confidence=self._confidence_from_score(score, liquidity),
            score=max(score, 0.0),
            summary=f"Bull case focuses on upside convexity in {context.symbol}.",
            rationale=[
                f"Positive trend contribution={momentum:.2f}.",
                f"Growth contribution={growth:.2f}, sentiment contribution={sentiment:.2f}.",
                f"Liquidity support={liquidity:.2f}.",
            ],
            evidence={
                "trend_strength": momentum,
                "revenue_growth": growth,
                "news_score": sentiment,
                "liquidity_score": liquidity,
            },
        )


class BearResearcher(BaseTradingAgent):
    agent_name = "bear_researcher"
    role_description = "constructs the strongest bearish case from the evidence"

    def _analyze_impl(self, context: AgentContext) -> AgentAnalysis:
        drawdown_risk = _safe_float(context.market_data.get("drawdown_risk"))
        debt_ratio = _safe_float(context.fundamentals.get("debt_to_equity"))
        volatility = _safe_float(context.market_data.get("realized_volatility"))
        sentiment = _safe_float(context.sentiment_data.get("news_score"))
        downside = (0.35 * max(drawdown_risk, 0.0)) + (0.25 * max(debt_ratio, 0.0)) + (0.25 * max(volatility, 0.0)) + (0.15 * max(-sentiment, 0.0))
        score = -_clamp(downside, 0.0, 1.0)
        risk_flags = ["downside_tail_risk"] if volatility > 0.7 or drawdown_risk > 0.7 else []
        return self._build_analysis(
            stance="sell" if score < -0.10 else "hold",
            confidence=self._confidence_from_score(score, min(downside, 1.0)),
            score=score,
            summary=f"Bear case highlights asymmetric downside risks in {context.symbol}.",
            rationale=[
                f"Drawdown risk={drawdown_risk:.2f}, realized volatility={volatility:.2f}.",
                f"Debt stress={debt_ratio:.2f}, sentiment drag={sentiment:.2f}.",
            ],
            risk_flags=risk_flags,
            evidence={
                "drawdown_risk": drawdown_risk,
                "debt_to_equity": debt_ratio,
                "realized_volatility": volatility,
                "news_score": sentiment,
            },
        )


class RiskManager(BaseTradingAgent):
    agent_name = "risk_manager"
    role_description = "constrains exposure using portfolio and market risk limits"

    def _analyze_impl(self, context: AgentContext) -> AgentAnalysis:
        portfolio = context.portfolio_state
        limits = context.risk_limits
        current_exposure = abs(_safe_float(portfolio.get("gross_exposure")))
        max_exposure = max(_safe_float(limits.get("max_exposure"), 1.0), 1e-6)
        value_at_risk = _safe_float(portfolio.get("value_at_risk"))
        max_var = max(_safe_float(limits.get("max_value_at_risk"), 1.0), 1e-6)
        daily_loss = abs(_safe_float(portfolio.get("daily_loss")))
        max_daily_loss = max(_safe_float(limits.get("max_daily_loss"), 1.0), 1e-6)
        utilization = max(current_exposure / max_exposure, value_at_risk / max_var, daily_loss / max_daily_loss)
        score = _clamp(0.4 - utilization, -1.0, 1.0)
        stance = "buy" if utilization < 0.45 else "hold" if utilization < 0.80 else "sell"
        risk_flags = []
        if utilization >= 1.0:
            risk_flags.append("risk_limit_breached")
        elif utilization >= 0.80:
            risk_flags.append("risk_limit_near")
        return self._build_analysis(
            stance=stance,
            confidence=self._confidence_from_score(score, 1.0 - min(utilization, 1.0)),
            score=score,
            summary=f"Risk capacity assessment is {stance} for additional {context.symbol} exposure.",
            rationale=[
                f"Exposure utilization={current_exposure / max_exposure:.2f} of limit.",
                f"VaR utilization={value_at_risk / max_var:.2f}, daily-loss utilization={daily_loss / max_daily_loss:.2f}.",
            ],
            risk_flags=risk_flags,
            evidence={
                "exposure_utilization": current_exposure / max_exposure,
                "var_utilization": value_at_risk / max_var,
                "daily_loss_utilization": daily_loss / max_daily_loss,
            },
        )


class FundManager(BaseTradingAgent):
    agent_name = "fund_manager"
    role_description = "translates research into portfolio-level sizing and action"

    def _analyze_impl(self, context: AgentContext) -> AgentAnalysis:
        alpha = _safe_float(context.metadata.get("alpha_score"))
        liquidity = _safe_float(context.market_data.get("liquidity_score"), 0.5)
        regime_support = 0.3 if str(context.market_regime).lower() in {"trending_bull", "risk_on"} else -0.1
        crowding = _safe_float(context.market_data.get("crowding_risk"))
        score = _clamp((0.45 * alpha) + (0.25 * liquidity) + regime_support - (0.25 * crowding), -1.0, 1.0)
        stance = self._stance_from_score(score)
        target_weight = _clamp(abs(score) * (0.08 if stance != "hold" else 0.02), 0.0, 0.10)
        return self._build_analysis(
            stance=stance,
            confidence=self._confidence_from_score(score, liquidity),
            score=score,
            summary=f"Portfolio construction recommends {stance} with size discipline for {context.symbol}.",
            rationale=[
                f"Alpha score={alpha:.2f}, liquidity={liquidity:.2f}, crowding risk={crowding:.2f}.",
                f"Regime overlay contributed {regime_support:.2f}; target weight={target_weight:.2%}.",
            ],
            evidence={
                "alpha_score": alpha,
                "liquidity_score": liquidity,
                "crowding_risk": crowding,
                "target_weight": target_weight,
            },
            metadata={"target_weight": target_weight},
        )


class MarketRegimeAgent(BaseTradingAgent):
    agent_name = "market_regime_agent"
    role_description = "classifies market regime and its impact on allocation"

    def _analyze_impl(self, context: AgentContext) -> AgentAnalysis:
        trend = _safe_float(context.technical_indicators.get("trend_strength"))
        volatility = _safe_float(context.market_data.get("realized_volatility"))
        breadth = _safe_float(context.market_data.get("market_breadth"))
        macro = _safe_float(context.macro_data.get("risk_on_score"))
        regime_score = (0.35 * trend) - (0.30 * volatility) + (0.20 * breadth) + (0.25 * macro)
        score = _clamp(regime_score, -1.0, 1.0)
        if score > 0.35:
            regime = "trending_bull"
        elif score < -0.35:
            regime = "stress_bear"
        elif volatility > 0.65:
            regime = "high_volatility_range"
        else:
            regime = "range_bound"
        stance = "buy" if regime == "trending_bull" else "sell" if regime == "stress_bear" else "hold"
        return self._build_analysis(
            stance=stance,
            confidence=self._confidence_from_score(score, 1.0 - min(volatility, 1.0)),
            score=score,
            summary=f"Detected regime for {context.symbol} is {regime}.",
            rationale=[
                f"Trend={trend:.2f}, volatility={volatility:.2f}, breadth={breadth:.2f}, macro risk-on={macro:.2f}.",
                f"Regime score={score:.2f} mapped to {regime}.",
            ],
            evidence={
                "trend_strength": trend,
                "realized_volatility": volatility,
                "market_breadth": breadth,
                "risk_on_score": macro,
            },
            metadata={"detected_regime": regime},
        )
