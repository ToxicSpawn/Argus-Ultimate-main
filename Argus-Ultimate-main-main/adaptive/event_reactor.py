"""Real-time event reaction system for news and breaking events."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MarketEvent:
    """Represents a market-moving event."""
    event_id: str
    event_type: str  # "news", "earnings", "regulatory", "macro", "technical", "social"
    title: str
    content: str
    symbols: List[str]
    sentiment: float  # -1 to 1
    impact_score: float  # 0-100
    urgency: str  # "low", "medium", "high", "critical"
    timestamp: datetime
    source: str

    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        self.sentiment = max(-1.0, min(1.0, self.sentiment))
        self.impact_score = max(0.0, min(100.0, self.impact_score))


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    score: float  # -1 to 1
    confidence: float  # 0-1
    label: str  # "positive", "negative", "neutral"
    keywords: List[str] = field(default_factory=list)


@dataclass
class EventImpact:
    """Estimated impact of an event on assets."""
    event: MarketEvent
    expected_impact: Dict[str, float]  # symbol -> expected price change %
    confidence: float
    affected_assets: List[str]
    recommended_action: str
    time_horizon: str  # "immediate", "hours", "days"


@dataclass
class ReactionAction:
    """A specific action to take in response to an event."""
    action_type: str  # "reduce_position", "hedge", "close", "alert", "pause_trading"
    symbol: str
    size_pct: float  # percentage to reduce
    reason: str


@dataclass
class ReactionPlan:
    """Plan for reacting to an event."""
    event: MarketEvent
    actions: List[ReactionAction]
    urgency: str
    expires_at: datetime
    confidence: float


@dataclass
class ReactionResult:
    """Result of executing a reaction plan."""
    plan: ReactionPlan
    executed_actions: List[ReactionAction]
    success: bool
    timestamp: datetime
    message: str = ""


@dataclass
class Alert:
    """An alert for a market event."""
    alert_id: str
    event: MarketEvent
    impact: EventImpact
    message: str
    created_at: datetime
    sent: bool = False

    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = str(uuid.uuid4())


@dataclass
class AlertBatch:
    """A batch of alerts."""
    batch_id: str
    alerts: List[Alert]
    created_at: datetime
    total_count: int = 0

    def __post_init__(self):
        if not self.batch_id:
            self.batch_id = str(uuid.uuid4())
        self.total_count = len(self.alerts)


class EventClassifier:
    """Classifies market events and computes impact scores."""

    TYPE_WEIGHTS = {
        "regulatory": 0.9,
        "macro": 0.85,
        "earnings": 0.8,
        "news": 0.7,
        "technical": 0.6,
        "social": 0.5,
    }

    URGENCY_KEYWORDS = {
        "critical": ["emergency", "halt", "crisis", "collapse", "crash", "ban", "sanction"],
        "high": ["breaking", "urgent", "warning", "alert", "surge", "plunge", "spike"],
        "medium": ["report", "data", "release", "update", "change", "shift"],
        "low": ["routine", "scheduled", "regular", "minor", "update"],
    }

    def classify_event(self, event: MarketEvent) -> MarketEvent:
        """Classify and enrich a market event."""
        event.impact_score = self.compute_impact_score(event)
        event.urgency = self.compute_urgency(event)
        logger.info(
            "Classified event: %s (type=%s, impact=%.1f, urgency=%s)",
            event.title, event.event_type, event.impact_score, event.urgency,
        )
        return event

    def compute_impact_score(self, event: MarketEvent) -> float:
        """Compute impact score based on event characteristics."""
        base_weight = self.TYPE_WEIGHTS.get(event.event_type, 0.5)
        sentiment_magnitude = abs(event.sentiment)
        symbol_count_factor = min(len(event.symbols) / 10.0, 1.0)

        content_length_factor = min(len(event.content) / 1000.0, 1.0)

        score = (
            base_weight * 40
            + sentiment_magnitude * 30
            + symbol_count_factor * 15
            + content_length_factor * 15
        )

        return max(0.0, min(100.0, score))

    def compute_urgency(self, event: MarketEvent) -> str:
        """Determine event urgency based on content and type."""
        text = f"{event.title} {event.content}".lower()

        for urgency_level in ["critical", "high", "medium", "low"]:
            keywords = self.URGENCY_KEYWORDS.get(urgency_level, [])
            if any(kw in text for kw in keywords):
                return urgency_level

        if event.impact_score >= 80:
            return "critical"
        elif event.impact_score >= 60:
            return "high"
        elif event.impact_score >= 40:
            return "medium"
        else:
            return "low"

    def identify_affected_assets(self, event: MarketEvent) -> List[str]:
        """Identify assets affected by an event."""
        affected = list(event.symbols)

        sector_correlations = {
            "tech": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
            "finance": ["JPM", "BAC", "GS", "MS", "C"],
            "energy": ["XOM", "CVX", "COP", "SLB", "OXY"],
            "healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
            "crypto": ["BTC", "ETH", "SOL", "BNB", "XRP"],
        }

        if event.event_type == "macro":
            for sector_symbols in sector_correlations.values():
                affected.extend(sector_symbols)
        elif event.event_type == "regulatory":
            for symbol in event.symbols:
                for sector, sector_symbols in sector_correlations.items():
                    if symbol in sector_symbols:
                        affected.extend(sector_symbols)
                        break

        return list(dict.fromkeys(affected))


class SentimentAnalyzer:
    """Analyzes sentiment in text and tracks sentiment history."""

    POSITIVE_WORDS = {
        "surge", "jump", "rally", "gain", "rise", "beat", "upgrade", "bullish",
        "growth", "profit", "success", "positive", "strong", "record", "high",
        "breakthrough", "approval", "deal", "partnership", "expansion",
    }

    NEGATIVE_WORDS = {
        "crash", "plunge", "drop", "fall", "miss", "downgrade", "bearish",
        "loss", "decline", "risk", "warning", "negative", "weak", "low",
        "investigation", "fine", "penalty", "lawsuit", "fraud", "scandal",
    }

    def __init__(self):
        self._sentiment_history: Dict[str, List[float]] = {}

    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text, returning score from -1 to 1."""
        if not text:
            return 0.0

        words = set(text.lower().split())
        positive_count = len(words & self.POSITIVE_WORDS)
        negative_count = len(words & self.NEGATIVE_WORDS)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        score = (positive_count - negative_count) / total
        return max(-1.0, min(1.0, score))

    def analyze_headline(self, headline: str) -> SentimentResult:
        """Analyze sentiment of a headline."""
        score = self.analyze_sentiment(headline)

        abs_score = abs(score)
        if abs_score > 0.5:
            confidence = 0.8
        elif abs_score > 0.2:
            confidence = 0.6
        else:
            confidence = 0.4

        if score > 0.2:
            label = "positive"
        elif score < -0.2:
            label = "negative"
        else:
            label = "neutral"

        words = headline.lower().split()
        keywords = [
            w for w in words
            if w in self.POSITIVE_WORDS or w in self.NEGATIVE_WORDS
        ]

        return SentimentResult(
            score=score,
            confidence=confidence,
            label=label,
            keywords=keywords,
        )

    def track_sentiment_history(self, symbol: str) -> List[float]:
        """Get sentiment history for a symbol."""
        return list(self._sentiment_history.get(symbol, []))

    def add_sentiment(self, symbol: str, score: float):
        """Add a sentiment score to history."""
        if symbol not in self._sentiment_history:
            self._sentiment_history[symbol] = []
        self._sentiment_history[symbol].append(score)

        if len(self._sentiment_history[symbol]) > 1000:
            self._sentiment_history[symbol] = self._sentiment_history[symbol][-500:]

    def detect_sentiment_shift(self, history: List[float]) -> Optional[float]:
        """Detect significant shift in sentiment history."""
        if len(history) < 10:
            return None

        recent = history[-5:]
        older = history[-10:-5]

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        shift = recent_avg - older_avg

        if abs(shift) > 0.3:
            return shift
        return None


class EventImpactEstimator:
    """Estimates the impact of events on asset prices."""

    def __init__(self):
        self._historical_events: List[MarketEvent] = []

    def estimate_impact(
        self, event: MarketEvent, historical_events: Optional[List[MarketEvent]] = None
    ) -> EventImpact:
        """Estimate the impact of an event."""
        history = historical_events or self._historical_events
        similar = self.find_similar_events(event, history)

        avg_impact = self.compute_average_impact(similar) if similar else 2.5

        expected_impact = {}
        for symbol in event.symbols:
            symbol_impact = avg_impact * abs(event.sentiment) * (event.impact_score / 100.0)
            if event.sentiment < 0:
                symbol_impact = -symbol_impact
            expected_impact[symbol] = round(symbol_impact, 2)

        confidence = min(0.9, 0.3 + (len(similar) * 0.1))

        if event.urgency == "critical":
            time_horizon = "immediate"
            recommended_action = "immediate_review"
        elif event.urgency == "high":
            time_horizon = "hours"
            recommended_action = "reduce_exposure"
        elif event.urgency == "medium":
            time_horizon = "hours"
            recommended_action = "monitor_closely"
        else:
            time_horizon = "days"
            recommended_action = "monitor"

        return EventImpact(
            event=event,
            expected_impact=expected_impact,
            confidence=confidence,
            affected_assets=list(expected_impact.keys()),
            recommended_action=recommended_action,
            time_horizon=time_horizon,
        )

    def find_similar_events(
        self, event: MarketEvent, history: List[MarketEvent]
    ) -> List[MarketEvent]:
        """Find similar historical events."""
        similar = []
        for historical in history:
            if historical.event_type != event.event_type:
                continue

            symbol_overlap = set(historical.symbols) & set(event.symbols)
            if symbol_overlap:
                similar.append(historical)
                continue

            sentiment_diff = abs(historical.sentiment - event.sentiment)
            if sentiment_diff < 0.3:
                similar.append(historical)

        return similar[:20]

    def compute_average_impact(self, similar_events: List[MarketEvent]) -> float:
        """Compute average impact from similar events."""
        if not similar_events:
            return 2.5

        impacts = [e.impact_score for e in similar_events]
        return sum(impacts) / len(impacts) / 20.0

    def adjust_for_market_conditions(
        self, base_impact: float, market_state: Dict[str, Any]
    ) -> float:
        """Adjust impact estimate based on market conditions."""
        volatility = market_state.get("volatility", 1.0)
        trend = market_state.get("trend", 0.0)
        liquidity = market_state.get("liquidity", 1.0)

        adjusted = base_impact * volatility
        adjusted *= (1.0 + abs(trend) * 0.5)
        if liquidity < 0.5:
            adjusted *= 1.5

        return round(adjusted, 2)


class EventReactor:
    """Reacts to market events with trading actions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._hedge_threshold = self.config.get("hedge_threshold", 50.0)
        self._position_reduce_pct = self.config.get("position_reduce_pct", 25.0)

    def react_to_event(
        self, event: MarketEvent, portfolio: Optional[Dict[str, Any]] = None
    ) -> ReactionPlan:
        """Generate a reaction plan for an event."""
        portfolio = portfolio or {}
        actions = []

        for symbol in event.symbols:
            position = portfolio.get(symbol, {})
            position_size = position.get("size", 0)

            if event.sentiment < -0.5 and position_size > 0:
                actions.append(
                    ReactionAction(
                        action_type="reduce_position",
                        symbol=symbol,
                        size_pct=self._position_reduce_pct,
                        reason=f"Negative sentiment ({event.sentiment:.2f}) on {event.title}",
                    )
                )

            if event.urgency in ("critical", "high") and position_size > 0:
                actions.append(
                    ReactionAction(
                        action_type="hedge",
                        symbol=symbol,
                        size_pct=min(50.0, abs(event.sentiment) * 100),
                        reason=f"High urgency event: {event.title}",
                    )
                )

            if event.impact_score >= 90 and event.sentiment < -0.7:
                actions.append(
                    ReactionAction(
                        action_type="close",
                        symbol=symbol,
                        size_pct=100.0,
                        reason=f"Critical negative event: {event.title}",
                    )
                )

        if event.event_type == "regulatory" and event.urgency == "critical":
            actions.append(
                ReactionAction(
                    action_type="pause_trading",
                    symbol="ALL",
                    size_pct=0.0,
                    reason=f"Critical regulatory event: {event.title}",
                )
            )

        if not actions:
            actions.append(
                ReactionAction(
                    action_type="alert",
                    symbol=event.symbols[0] if event.symbols else "UNKNOWN",
                    size_pct=0.0,
                    reason=f"Event detected: {event.title}",
                )
            )

        urgency = event.urgency
        if urgency == "critical":
            expires_at = datetime.now() + timedelta(minutes=15)
        elif urgency == "high":
            expires_at = datetime.now() + timedelta(hours=1)
        elif urgency == "medium":
            expires_at = datetime.now() + timedelta(hours=4)
        else:
            expires_at = datetime.now() + timedelta(hours=24)

        confidence = min(0.95, (event.impact_score / 100.0) * 0.7 + 0.3)

        plan = ReactionPlan(
            event=event,
            actions=actions,
            urgency=urgency,
            expires_at=expires_at,
            confidence=confidence,
        )

        logger.info(
            "Generated reaction plan: %d actions for event %s",
            len(actions), event.event_id,
        )
        return plan

    def should_hedge(self, event: MarketEvent, exposure: float) -> bool:
        """Determine if hedging is needed."""
        return (
            event.impact_score >= self._hedge_threshold
            and exposure > 0
            and event.sentiment < -0.3
        )

    def should_reduce_position(self, event: MarketEvent, position: Dict[str, Any]) -> bool:
        """Determine if position should be reduced."""
        size = position.get("size", 0)
        if size <= 0:
            return False

        if event.sentiment < -0.5:
            return True

        if event.urgency in ("critical", "high") and event.sentiment < -0.2:
            return True

        return False

    def compute_hedge_size(self, exposure: float, impact: float) -> float:
        """Compute hedge size based on exposure and impact."""
        base_hedge = exposure * (impact / 100.0)
        return min(base_hedge, exposure * 0.5)


class EventAlertManager:
    """Manages alerts for market events."""

    def __init__(self, alert_callback: Optional[Callable] = None):
        self._alert_callback = alert_callback
        self._alerts: List[Alert] = []

    def create_alert(self, event: MarketEvent, impact: EventImpact) -> Alert:
        """Create an alert for an event."""
        message = (
            f"[{event.urgency.upper()}] {event.title}\n"
            f"Type: {event.event_type} | Sentiment: {event.sentiment:.2f}\n"
            f"Impact: {impact.expected_impact}\n"
            f"Action: {impact.recommended_action}"
        )

        alert = Alert(
            alert_id="",
            event=event,
            impact=impact,
            message=message,
            created_at=datetime.now(),
        )

        self._alerts.append(alert)
        logger.info("Created alert: %s", alert.alert_id)
        return alert

    def should_alert_urgently(self, event: MarketEvent) -> bool:
        """Determine if an alert should be sent immediately."""
        return event.urgency in ("critical", "high")

    def send_immediate_alert(self, alert: Alert) -> None:
        """Send an immediate alert."""
        alert.sent = True

        if self._alert_callback:
            try:
                self._alert_callback(alert)
            except Exception as e:
                logger.error("Failed to send alert: %s", e)

        logger.warning(
            "IMMEDIATE ALERT: %s [%s]",
            alert.event.title, alert.event.urgency,
        )

    def batch_alerts(self, events: List[MarketEvent]) -> AlertBatch:
        """Create a batch of alerts for multiple events."""
        alerts = []
        for event in events:
            impact = EventImpact(
                event=event,
                expected_impact={s: 0.0 for s in event.symbols},
                confidence=0.5,
                affected_assets=event.symbols,
                recommended_action="monitor",
                time_horizon="days",
            )
            alert = self.create_alert(event, impact)
            alerts.append(alert)

        batch = AlertBatch(
            batch_id="",
            alerts=alerts,
            created_at=datetime.now(),
        )

        logger.info("Created alert batch: %d alerts", len(alerts))
        return batch


class EventMonitor:
    """Main event monitoring system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._event_sources: Dict[str, Callable] = {}
        self._event_history: List[MarketEvent] = []
        self._pending_reactions: List[ReactionPlan] = []
        self._executed_results: List[ReactionResult] = []

        self.classifier = EventClassifier()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.impact_estimator = EventImpactEstimator()
        self.reactor = EventReactor(self.config.get("reactor", {}))
        self.alert_manager = EventAlertManager(
            self.config.get("alert_callback")
        )

        logger.info("EventMonitor initialized")

    def register_event_source(self, name: str, fetch_fn: Callable) -> None:
        """Register an event source with a fetch function."""
        self._event_sources[name] = fetch_fn
        logger.info("Registered event source: %s", name)

    async def monitor_events(self) -> List[MarketEvent]:
        """Monitor all registered event sources for new events."""
        new_events = []

        for name, fetch_fn in self._event_sources.items():
            try:
                events = fetch_fn()
                if isinstance(events, list):
                    for event in events:
                        if isinstance(event, MarketEvent):
                            processed = self.process_event(event)
                            new_events.append(processed)
                logger.debug("Fetched %d events from source: %s", len(new_events), name)
            except Exception as e:
                logger.error("Error fetching events from %s: %s", name, e)

        self._event_history.extend(new_events)
        self._prune_history()

        return new_events

    def process_event(self, event: MarketEvent) -> EventImpact:
        """Process a single event through the pipeline."""
        event = self.classifier.classify_event(event)

        sentiment = self.sentiment_analyzer.analyze_sentiment(event.content)
        if abs(sentiment) > 0.1:
            for symbol in event.symbols:
                self.sentiment_analyzer.add_sentiment(symbol, sentiment)

        impact = self.impact_estimator.estimate_impact(event, self._event_history)

        if self.alert_manager.should_alert_urgently(event):
            alert = self.alert_manager.create_alert(event, impact)
            self.alert_manager.send_immediate_alert(alert)

        if event.impact_score >= 30:
            plan = self.reactor.react_to_event(event)
            self._pending_reactions.append(plan)
            self._prune_pending_reactions()

        logger.info(
            "Processed event: %s (impact=%.1f, sentiment=%.2f)",
            event.title, impact.confidence, sentiment,
        )

        return impact

    def get_pending_reactions(self) -> List[ReactionPlan]:
        """Get all pending reaction plans."""
        now = datetime.now()
        return [
            plan for plan in self._pending_reactions
            if plan.expires_at > now
        ]

    def execute_reaction(self, plan: ReactionPlan) -> ReactionResult:
        """Execute a reaction plan."""
        executed = []

        for action in plan.actions:
            try:
                logger.info(
                    "Executing action: %s on %s (%.1f%%)",
                    action.action_type, action.symbol, action.size_pct,
                )
                executed.append(action)
            except Exception as e:
                logger.error("Failed to execute action: %s", e)

        result = ReactionResult(
            plan=plan,
            executed_actions=executed,
            success=len(executed) > 0,
            timestamp=datetime.now(),
            message=f"Executed {len(executed)}/{len(plan.actions)} actions",
        )

        self._executed_results.append(result)
        if plan in self._pending_reactions:
            self._pending_reactions.remove(plan)

        return result

    def get_event_history(self, hours: int = 24) -> List[MarketEvent]:
        """Get event history for the specified time period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            event for event in self._event_history
            if event.timestamp >= cutoff
        ]

    def _prune_history(self) -> None:
        """Prune old events from history."""
        cutoff = datetime.now() - timedelta(hours=72)
        self._event_history = [
            event for event in self._event_history
            if event.timestamp >= cutoff
        ]

    def _prune_pending_reactions(self) -> None:
        """Prune expired reaction plans."""
        now = datetime.now()
        self._pending_reactions = [
            plan for plan in self._pending_reactions
            if plan.expires_at > now
        ]
