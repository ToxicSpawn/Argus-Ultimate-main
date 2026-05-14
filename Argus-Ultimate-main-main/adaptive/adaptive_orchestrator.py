"""
Adaptive Orchestrator - Master orchestrator for real-time adaptation.

Ties all adaptive components together for coordinated, real-time system adaptation
including regime detection, strategy rotation, model health monitoring, and
position sizing adjustments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from adaptive.regime import MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveConfig:
    """Configuration for the adaptive orchestrator."""
    regime_check_interval_seconds: int = 10
    strategy_rotation_interval_minutes: int = 60
    model_health_check_interval_minutes: int = 30
    position_sizing_update_seconds: int = 5
    enable_auto_rotation: bool = True
    enable_auto_retrain: bool = True
    enable_dynamic_sizing: bool = True
    min_regime_confidence: float = 0.7


@dataclass
class PortfolioState:
    """Snapshot of current portfolio state."""
    total_value: float = 0.0
    cash: float = 0.0
    positions: Dict[str, float] = field(default_factory=dict)
    daily_pnl: float = 0.0
    drawdown: float = 0.0
    sharpe_ratio: float = 0.0


@dataclass
class Alert:
    """Adaptation alert record."""
    timestamp: datetime
    level: str
    message: str
    component: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketState:
    """Comprehensive snapshot of current market conditions."""
    timestamp: datetime
    regime: MarketRegime
    regime_confidence: float
    volatility: float
    trend_strength: float
    liquidity_score: float
    active_strategies: List[str]
    portfolio_state: PortfolioState
    alerts: List[Alert] = field(default_factory=list)


@dataclass
class AdaptationDecision:
    """Record of a single adaptation decision."""
    timestamp: datetime
    decision_type: str
    trigger: str
    old_state: Dict[str, Any]
    new_state: Dict[str, Any]
    confidence: float
    expected_impact: float


@dataclass
class StrategyChange:
    """Proposed change to strategy allocation."""
    strategy_name: str
    action: str
    current_weight: float
    target_weight: float
    reason: str
    confidence: float


@dataclass
class HealingAction:
    """Model healing/retraining action."""
    model_name: str
    action: str
    reason: str
    confidence: float
    expected_improvement: float


@dataclass
class AdaptationResult:
    """Result of a single adaptation cycle."""
    timestamp: datetime
    cycle_duration_ms: float
    regime_detected: MarketRegime
    regime_confidence: float
    strategy_changes: List[StrategyChange]
    model_actions: List[HealingAction]
    position_adjustments: Dict[str, float]
    alerts_generated: List[Alert]
    adaptation_decisions: List[AdaptationDecision]


@dataclass
class SystemHealth:
    """Overall system health status."""
    overall_status: str
    regime_detector_status: str
    strategy_selector_status: str
    position_sizer_status: str
    model_manager_status: str
    last_adaptation: datetime
    adaptations_today: int
    errors_today: int


@dataclass
class AdaptationRecord:
    """Full audit record of an adaptation event."""
    timestamp: datetime
    trigger: str
    decisions: List[AdaptationDecision]
    result: AdaptationResult
    success: bool
    notes: str = ""


@dataclass
class AdaptationStats:
    """Aggregated adaptation statistics."""
    total_adaptations: int
    successful_adaptations: int
    failed_adaptations: int
    avg_cycle_duration_ms: float
    avg_regime_confidence: float
    most_common_trigger: str
    adaptations_by_type: Dict[str, int]


class MarketDataAdapter:
    """Adapts raw market data into normalized features for the orchestrator."""

    def __init__(self, feature_window: int = 100):
        self.feature_window = feature_window
        self._last_data: Optional[Dict[str, Any]] = None

    def collect_market_data(self) -> Dict[str, Any]:
        """Collect current market data snapshot."""
        return {
            "timestamp": datetime.utcnow(),
            "prices": {},
            "volumes": {},
            "order_book_depth": {},
            "funding_rates": {},
            "open_interest": {},
        }

    def compute_features(self, market_data: Dict[str, Any]) -> MarketState:
        """Compute market state features from raw data."""
        return MarketState(
            timestamp=market_data.get("timestamp", datetime.utcnow()),
            regime=MarketRegime.RANGE,
            regime_confidence=0.5,
            volatility=0.0,
            trend_strength=0.0,
            liquidity_score=0.5,
            active_strategies=[],
            portfolio_state=PortfolioState(),
        )

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to zero mean, unit variance."""
        if data.size == 0:
            return data
        mean = np.nanmean(data)
        std = np.nanstd(data)
        if std < 1e-10:
            return np.zeros_like(data)
        return (data - mean) / std

    def handle_missing_data(self, data: np.ndarray) -> np.ndarray:
        """Handle missing values via forward-fill then median imputation."""
        if data.size == 0:
            return data
        cleaned = data.copy().astype(float)
        mask = np.isnan(cleaned)
        if not np.any(mask):
            return cleaned
        last_valid = None
        for i in range(len(cleaned)):
            if np.isnan(cleaned[i]):
                if last_valid is not None:
                    cleaned[i] = last_valid
            else:
                last_valid = cleaned[i]
        median_val = np.nanmedian(cleaned)
        cleaned[np.isnan(cleaned)] = median_val
        return cleaned


class AdaptationOrchestrator:
    """Orchestrates adaptation cycles across all components."""

    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self._adapter = MarketDataAdapter()
        self._current_state: Optional[MarketState] = None
        self._previous_state: Optional[MarketState] = None
        self._adaptation_history: deque = deque(maxlen=10000)
        self._circuit_breaker = CircuitBreaker()
        self._running = False
        self._cycle_task: Optional[asyncio.Task] = None
        self._errors_today = 0
        self._adaptations_today = 0
        self._last_adaptation: Optional[datetime] = None

    async def initialize(self) -> None:
        """Initialize orchestrator and all sub-components."""
        logger.info("AdaptationOrchestrator initializing")
        self._running = False
        self._adaptation_history.clear()
        logger.info("AdaptationOrchestrator initialized")

    async def run_cycle(self) -> AdaptationResult:
        """Execute a single adaptation cycle."""
        start = time.monotonic()
        ts = datetime.utcnow()

        try:
            market_data = self._adapter.collect_market_data()
            current_state = self._adapter.compute_features(market_data)
            self._previous_state = self._current_state
            self._current_state = current_state

            should = self.should_adapt(current_state, self._previous_state)
            urgency = self.compute_adaptation_urgency(current_state)

            decisions: List[AdaptationDecision] = []
            strategy_changes: List[StrategyChange] = []
            model_actions: List[HealingAction] = []
            position_adjustments: Dict[str, float] = {}
            alerts: List[Alert] = []

            if should and not self._circuit_breaker.is_tripped():
                decision = AdaptationDecision(
                    timestamp=ts,
                    decision_type="regime_change",
                    trigger="cycle_check",
                    old_state=self._state_to_dict(self._previous_state),
                    new_state=self._state_to_dict(current_state),
                    confidence=current_state.regime_confidence,
                    expected_impact=urgency * 0.1,
                )
                decisions.append(decision)
                self._circuit_breaker.check_and_increment()
                self._adaptations_today += 1
                self._last_adaptation = ts

            duration_ms = (time.monotonic() - start) * 1000.0

            result = AdaptationResult(
                timestamp=ts,
                cycle_duration_ms=duration_ms,
                regime_detected=current_state.regime,
                regime_confidence=current_state.regime_confidence,
                strategy_changes=strategy_changes,
                model_actions=model_actions,
                position_adjustments=position_adjustments,
                alerts_generated=alerts,
                adaptation_decisions=decisions,
            )

            self._adaptation_history.append(result)
            return result

        except Exception as e:
            self._errors_today += 1
            logger.error("Adaptation cycle failed: %s", e)
            duration_ms = (time.monotonic() - start) * 1000.0
            return AdaptationResult(
                timestamp=ts,
                cycle_duration_ms=duration_ms,
                regime_detected=MarketRegime.RANGE,
                regime_confidence=0.0,
                strategy_changes=[],
                model_actions=[],
                position_adjustments={},
                alerts_generated=[
                    Alert(
                        timestamp=ts,
                        level="error",
                        message=f"Adaptation cycle failed: {e}",
                        component="AdaptationOrchestrator",
                    )
                ],
                adaptation_decisions=[],
            )

    def should_adapt(self, current_state: MarketState, previous_state: Optional[MarketState]) -> bool:
        """Determine if adaptation is needed based on state change."""
        if previous_state is None:
            return True
        if current_state.regime != previous_state.regime:
            return True
        if abs(current_state.volatility - previous_state.volatility) > 0.1:
            return True
        if abs(current_state.regime_confidence - previous_state.regime_confidence) > 0.2:
            return True
        return False

    def compute_adaptation_urgency(self, state: MarketState) -> float:
        """Compute urgency score 0-1 for adaptation."""
        urgency = 0.0
        if state.regime == MarketRegime.HIGH_VOL:
            urgency += 0.4
        urgency += (1.0 - state.regime_confidence) * 0.3
        if abs(state.trend_strength) > 0.7:
            urgency += 0.2
        if state.liquidity_score < 0.3:
            urgency += 0.1
        return float(np.clip(urgency, 0.0, 1.0))

    async def execute_adaptation(self, decision: AdaptationDecision) -> AdaptationResult:
        """Execute a specific adaptation decision."""
        start = time.monotonic()
        ts = datetime.utcnow()

        logger.info(
            "Executing adaptation: type=%s trigger=%s confidence=%.3f",
            decision.decision_type,
            decision.trigger,
            decision.confidence,
        )

        strategy_changes: List[StrategyChange] = []
        model_actions: List[HealingAction] = []
        position_adjustments: Dict[str, float] = {}
        alerts: List[Alert] = []

        if decision.decision_type == "regime_change":
            alerts.append(
                Alert(
                    timestamp=ts,
                    level="info",
                    message=f"Regime changed to {decision.new_state.get('regime', 'unknown')}",
                    component="AdaptationOrchestrator",
                )
            )
        elif decision.decision_type == "strategy_rotation":
            alerts.append(
                Alert(
                    timestamp=ts,
                    level="info",
                    message="Strategy rotation executed",
                    component="AdaptationOrchestrator",
                )
            )
        elif decision.decision_type == "model_retrain":
            model_actions.append(
                HealingAction(
                    model_name="ensemble",
                    action="retrain",
                    reason=decision.trigger,
                    confidence=decision.confidence,
                    expected_improvement=decision.expected_impact,
                )
            )
        elif decision.decision_type == "position_adjust":
            position_adjustments = decision.new_state.get("position_adjustments", {})

        duration_ms = (time.monotonic() - start) * 1000.0

        result = AdaptationResult(
            timestamp=ts,
            cycle_duration_ms=duration_ms,
            regime_detected=MarketRegime(decision.new_state.get("regime", "range")),
            regime_confidence=decision.confidence,
            strategy_changes=strategy_changes,
            model_actions=model_actions,
            position_adjustments=position_adjustments,
            alerts_generated=alerts,
            adaptation_decisions=[decision],
        )

        self._adaptation_history.append(result)
        self._last_adaptation = ts
        self._adaptations_today += 1

        return result

    def get_current_state(self) -> Optional[MarketState]:
        """Get the current market state."""
        return self._current_state

    def get_adaptation_history(self, hours: int = 24) -> List[AdaptationResult]:
        """Get adaptation history for the specified time window."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [r for r in self._adaptation_history if r.timestamp >= cutoff]

    def get_health_status(self) -> SystemHealth:
        """Get current system health status."""
        now = datetime.utcnow()
        statuses = {
            "regime_detector_status": "healthy",
            "strategy_selector_status": "healthy",
            "position_sizer_status": "healthy",
            "model_manager_status": "healthy",
        }

        error_count = self._errors_today
        if error_count > 10:
            overall = "critical"
            for k in statuses:
                statuses[k] = "critical"
        elif error_count > 3:
            overall = "degraded"
            for k in statuses:
                statuses[k] = "degraded"
        else:
            overall = "healthy"

        return SystemHealth(
            overall_status=overall,
            last_adaptation=self._last_adaptation or now,
            adaptations_today=self._adaptations_today,
            errors_today=error_count,
            **statuses,
        )

    @staticmethod
    def _state_to_dict(state: Optional[MarketState]) -> Dict[str, Any]:
        """Convert MarketState to a serializable dict."""
        if state is None:
            return {}
        return {
            "regime": state.regime.value,
            "regime_confidence": state.regime_confidence,
            "volatility": state.volatility,
            "trend_strength": state.trend_strength,
            "liquidity_score": state.liquidity_score,
            "active_strategies": state.active_strategies,
        }


class CrossComponentCoordinator:
    """Coordinates adaptations across regime, strategy, model, and sizing components."""

    def coordinate_regime_strategy(
        self, regime: MarketRegime, strategies: List[str]
    ) -> List[StrategyChange]:
        """Coordinate regime change with strategy adjustments."""
        changes: List[StrategyChange] = []
        regime_strategy_map = {
            MarketRegime.TREND_UP: {"momentum": 0.6, "trend_following": 0.3, "mean_reversion": 0.1},
            MarketRegime.TREND_DOWN: {"short_momentum": 0.5, "mean_reversion": 0.3, "trend_following": 0.2},
            MarketRegime.RANGE: {"mean_reversion": 0.5, "statistical_arbitrage": 0.3, "market_making": 0.2},
            MarketRegime.HIGH_VOL: {"volatility_breakout": 0.4, "mean_reversion": 0.3, "tail_hedge": 0.3},
        }

        target_weights = regime_strategy_map.get(regime, {})
        for strategy in strategies:
            target = target_weights.get(strategy, 0.0)
            changes.append(
                StrategyChange(
                    strategy_name=strategy,
                    action="adjust_weight",
                    current_weight=0.0,
                    target_weight=target,
                    reason=f"Regime changed to {regime.value}",
                    confidence=0.8,
                )
            )
        return changes

    def coordinate_strategy_sizing(
        self, strategies: List[str], market_state: MarketState
    ) -> Dict[str, float]:
        """Coordinate position sizing across strategies based on market state."""
        sizing: Dict[str, float] = {}
        base_size = 1.0 / max(len(strategies), 1)

        vol_factor = 1.0
        if market_state.volatility > 0.8:
            vol_factor = 0.5
        elif market_state.volatility > 0.5:
            vol_factor = 0.75

        confidence_factor = market_state.regime_confidence

        for strategy in strategies:
            sizing[strategy] = base_size * vol_factor * confidence_factor

        total = sum(sizing.values())
        if total > 0:
            sizing = {k: v / total for k, v in sizing.items()}

        return sizing

    def coordinate_model_strategy(
        self, models: List[str], strategies: List[str]
    ) -> List[HealingAction]:
        """Coordinate model health with strategy requirements."""
        actions: List[HealingAction] = []
        for model in models:
            actions.append(
                HealingAction(
                    model_name=model,
                    action="health_check",
                    reason="routine_coordination",
                    confidence=0.9,
                    expected_improvement=0.05,
                )
            )
        return actions

    def resolve_conflicts(
        self, decisions: List[AdaptationDecision]
    ) -> List[AdaptationDecision]:
        """Resolve conflicting adaptation decisions."""
        if not decisions:
            return []

        by_type: Dict[str, List[AdaptationDecision]] = {}
        for d in decisions:
            by_type.setdefault(d.decision_type, []).append(d)

        resolved: List[AdaptationDecision] = []
        for dtype, dtype_decisions in by_type.items():
            best = max(dtype_decisions, key=lambda d: d.confidence)
            resolved.append(best)

        priority_order = ["regime_change", "strategy_rotation", "model_retrain", "position_adjust"]
        resolved.sort(key=lambda d: priority_order.index(d.decision_type) if d.decision_type in priority_order else 99)

        return resolved


class AdaptationLogger:
    """Logs and tracks all adaptation decisions and results."""

    def __init__(self, max_records: int = 10000):
        self._records: deque = deque(maxlen=max_records)
        self._decisions: deque = deque(maxlen=max_records)

    def log_decision(self, decision: AdaptationDecision) -> None:
        """Log an adaptation decision."""
        self._decisions.append(decision)
        logger.info(
            "Adaptation decision: type=%s trigger=%s confidence=%.3f impact=%.3f",
            decision.decision_type,
            decision.trigger,
            decision.confidence,
            decision.expected_impact,
        )

    def log_result(self, result: AdaptationResult) -> None:
        """Log an adaptation result."""
        self._records.append(result)
        logger.info(
            "Adaptation result: regime=%s confidence=%.3f duration=%.1fms changes=%d",
            result.regime_detected.value,
            result.regime_confidence,
            result.cycle_duration_ms,
            len(result.strategy_changes),
        )

    def get_adaptation_history(self, hours: int = 24) -> List[AdaptationRecord]:
        """Get adaptation history for the specified time window."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        records: List[AdaptationRecord] = []
        for result in self._records:
            if result.timestamp >= cutoff:
                records.append(
                    AdaptationRecord(
                        timestamp=result.timestamp,
                        trigger="cycle",
                        decisions=result.adaptation_decisions,
                        result=result,
                        success=True,
                    )
                )
        return records

    def compute_adaptation_stats(self) -> AdaptationStats:
        """Compute aggregated adaptation statistics."""
        if not self._records:
            return AdaptationStats(
                total_adaptations=0,
                successful_adaptations=0,
                failed_adaptations=0,
                avg_cycle_duration_ms=0.0,
                avg_regime_confidence=0.0,
                most_common_trigger="none",
                adaptations_by_type={},
            )

        total = len(self._records)
        durations = [r.cycle_duration_ms for r in self._records]
        confidences = [r.regime_confidence for r in self._records]

        type_counts: Dict[str, int] = {}
        trigger_counts: Dict[str, int] = {}
        for r in self._records:
            for d in r.adaptation_decisions:
                type_counts[d.decision_type] = type_counts.get(d.decision_type, 0) + 1
                trigger_counts[d.trigger] = trigger_counts.get(d.trigger, 0) + 1

        most_common = max(trigger_counts, key=trigger_counts.get) if trigger_counts else "none"

        return AdaptationStats(
            total_adaptations=total,
            successful_adaptations=total,
            failed_adaptations=0,
            avg_cycle_duration_ms=float(np.mean(durations)),
            avg_regime_confidence=float(np.mean(confidences)),
            most_common_trigger=most_common,
            adaptations_by_type=type_counts,
        )

    def export_log(self, format: str = "json") -> str:
        """Export adaptation log in the specified format."""
        records = list(self._records)
        if format == "json":
            data = []
            for r in records:
                data.append({
                    "timestamp": r.timestamp.isoformat(),
                    "cycle_duration_ms": r.cycle_duration_ms,
                    "regime_detected": r.regime_detected.value,
                    "regime_confidence": r.regime_confidence,
                    "strategy_changes": len(r.strategy_changes),
                    "model_actions": len(r.model_actions),
                    "position_adjustments": r.position_adjustments,
                    "alerts": len(r.alerts_generated),
                    "decisions": len(r.adaptation_decisions),
                })
            return json.dumps(data, indent=2)
        else:
            lines = []
            for r in records:
                lines.append(
                    f"{r.timestamp.isoformat()} | {r.regime_detected.value} | "
                    f"conf={r.regime_confidence:.3f} | dur={r.cycle_duration_ms:.1f}ms"
                )
            return "\n".join(lines)


class CircuitBreaker:
    """Circuit breaker to prevent excessive adaptations."""

    def __init__(
        self,
        max_adaptations_per_hour: int = 10,
        min_time_between_adaptations: int = 60,
    ):
        self.max_adaptations_per_hour = max_adaptations_per_hour
        self.min_time_between_adaptations = min_time_between_adaptations
        self._timestamps: deque = deque(maxlen=max_adaptations_per_hour + 10)
        self._tripped = False
        self._trip_time: Optional[datetime] = None

    def check_and_increment(self) -> bool:
        """Check if adaptation is allowed and record it. Returns True if allowed."""
        now = datetime.utcnow()

        if self._tripped:
            if self._trip_time and (now - self._trip_time).total_seconds() > 3600:
                self.reset()
            else:
                return False

        hour_ago = now - timedelta(hours=1)
        self._timestamps = deque(
            [t for t in self._timestamps if t > hour_ago],
            maxlen=self.max_adaptations_per_hour + 10,
        )

        if len(self._timestamps) >= self.max_adaptations_per_hour:
            self._tripped = True
            self._trip_time = now
            logger.warning("Circuit breaker tripped: max adaptations per hour reached")
            return False

        if self._timestamps:
            last = self._timestamps[-1]
            if (now - last).total_seconds() < self.min_time_between_adaptations:
                return False

        self._timestamps.append(now)
        return True

    def reset(self) -> None:
        """Reset the circuit breaker."""
        self._timestamps.clear()
        self._tripped = False
        self._trip_time = None
        logger.info("Circuit breaker reset")

    def is_tripped(self) -> bool:
        """Check if the circuit breaker is currently tripped."""
        if self._tripped:
            if self._trip_time and (datetime.utcnow() - self._trip_time).total_seconds() > 3600:
                self.reset()
                return False
            return True
        return False


class AdaptiveOrchestrator:
    """
    Master orchestrator that ties all adaptive components together.

    Continuously monitors market regime, rotates strategies, adjusts position sizes,
    monitors model health, and coordinates all adaptations to avoid conflicts.
    """

    def __init__(
        self,
        config: AdaptiveConfig,
        components: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.components = components or {}

        self.regime_detector = self.components.get("regime_detector")
        self.strategy_selector = self.components.get("strategy_selector")
        self.position_sizer = self.components.get("position_sizer")
        self.model_manager = self.components.get("model_manager")

        self._orchestrator = AdaptationOrchestrator(config)
        self._coordinator = CrossComponentCoordinator()
        self._adaptation_logger = AdaptationLogger()
        self._circuit_breaker = CircuitBreaker()

        self._running = False
        self._cycle_task: Optional[asyncio.Task] = None
        self._adaptation_records: deque = deque(maxlen=10000)

    async def initialize(self) -> None:
        """Initialize all components and the adaptation loop."""
        logger.info("AdaptiveOrchestrator initializing")

        await self._orchestrator.initialize()

        if self.regime_detector and hasattr(self.regime_detector, "initialize"):
            try:
                if asyncio.iscoroutinefunction(self.regime_detector.initialize):
                    await self.regime_detector.initialize()
                else:
                    self.regime_detector.initialize()
                logger.info("Regime detector initialized")
            except Exception as e:
                logger.warning("Regime detector init failed: %s", e)

        if self.strategy_selector and hasattr(self.strategy_selector, "initialize"):
            try:
                if asyncio.iscoroutinefunction(self.strategy_selector.initialize):
                    await self.strategy_selector.initialize()
                else:
                    self.strategy_selector.initialize()
                logger.info("Strategy selector initialized")
            except Exception as e:
                logger.warning("Strategy selector init failed: %s", e)

        if self.position_sizer and hasattr(self.position_sizer, "initialize"):
            try:
                if asyncio.iscoroutinefunction(self.position_sizer.initialize):
                    await self.position_sizer.initialize()
                else:
                    self.position_sizer.initialize()
                logger.info("Position sizer initialized")
            except Exception as e:
                logger.warning("Position sizer init failed: %s", e)

        if self.model_manager and hasattr(self.model_manager, "initialize"):
            try:
                if asyncio.iscoroutinefunction(self.model_manager.initialize):
                    await self.model_manager.initialize()
                else:
                    self.model_manager.initialize()
                logger.info("Model manager initialized")
            except Exception as e:
                logger.warning("Model manager init failed: %s", e)

        logger.info("AdaptiveOrchestrator initialized successfully")

    async def start(self) -> None:
        """Start the continuous adaptation loop."""
        if self._running:
            logger.warning("AdaptiveOrchestrator already running")
            return

        self._running = True
        self._cycle_task = asyncio.create_task(self._adaptation_loop())
        logger.info("AdaptiveOrchestrator started")

    async def stop(self) -> None:
        """Gracefully stop the adaptation loop."""
        self._running = False
        if self._cycle_task:
            self._cycle_task.cancel()
            try:
                await self._cycle_task
            except asyncio.CancelledError:
                pass
            self._cycle_task = None
        logger.info("AdaptiveOrchestrator stopped")

    async def _adaptation_loop(self) -> None:
        """Main adaptation loop that runs continuously."""
        while self._running:
            try:
                result = await self.run_cycle()
                self._adaptation_logger.log_result(result)

                for decision in result.adaptation_decisions:
                    self._adaptation_logger.log_decision(decision)

                if result.adaptation_decisions:
                    record = AdaptationRecord(
                        timestamp=result.timestamp,
                        trigger=result.adaptation_decisions[0].trigger,
                        decisions=result.adaptation_decisions,
                        result=result,
                        success=True,
                    )
                    self._adaptation_records.append(record)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Adaptation loop error: %s", e)

            await asyncio.sleep(self.config.regime_check_interval_seconds)

    async def run_cycle(self) -> AdaptationResult:
        """Execute a single adaptation cycle."""
        start = time.monotonic()
        ts = datetime.utcnow()

        if not self._circuit_breaker.check_and_increment():
            return AdaptationResult(
                timestamp=ts,
                cycle_duration_ms=(time.monotonic() - start) * 1000.0,
                regime_detected=MarketRegime.RANGE,
                regime_confidence=0.0,
                strategy_changes=[],
                model_actions=[],
                position_adjustments={},
                alerts_generated=[
                    Alert(
                        timestamp=ts,
                        level="warning",
                        message="Circuit breaker preventing adaptation",
                        component="AdaptiveOrchestrator",
                    )
                ],
                adaptation_decisions=[],
            )

        regime = MarketRegime.RANGE
        regime_confidence = 0.5
        volatility = 0.0
        trend_strength = 0.0
        active_strategies: List[str] = []

        if self.regime_detector and hasattr(self.regime_detector, "detect"):
            try:
                snapshot = self.regime_detector.detect(
                    self._get_market_data()
                )
                if snapshot:
                    regime = snapshot.regime
                    regime_confidence = getattr(snapshot, "confidence", 0.8)
                    volatility = getattr(snapshot, "vol_annualized", 0.0)
                    trend_strength = getattr(snapshot, "trend_score", 0.0)
            except Exception as e:
                logger.warning("Regime detection failed: %s", e)

        if self.strategy_selector and hasattr(self.strategy_selector, "select"):
            try:
                selected = self.strategy_selector.select(regime)
                if selected:
                    active_strategies = selected if isinstance(selected, list) else [selected]
            except Exception as e:
                logger.warning("Strategy selection failed: %s", e)

        current_state = MarketState(
            timestamp=ts,
            regime=regime,
            regime_confidence=regime_confidence,
            volatility=volatility,
            trend_strength=trend_strength,
            liquidity_score=0.5,
            active_strategies=active_strategies,
            portfolio_state=PortfolioState(),
        )

        previous_state = self._orchestrator.get_current_state()
        self._orchestrator._current_state = current_state
        self._orchestrator._previous_state = previous_state

        should = self._orchestrator.should_adapt(current_state, previous_state)
        urgency = self._orchestrator.compute_adaptation_urgency(current_state)

        decisions: List[AdaptationDecision] = []
        strategy_changes: List[StrategyChange] = []
        model_actions: List[HealingAction] = []
        position_adjustments: Dict[str, float] = {}
        alerts: List[Alert] = []

        if should and regime_confidence >= self.config.min_regime_confidence:
            if previous_state and current_state.regime != previous_state.regime:
                decision = AdaptationDecision(
                    timestamp=ts,
                    decision_type="regime_change",
                    trigger=f"regime_{previous_state.regime.value}_to_{regime.value}",
                    old_state=self._orchestrator._state_to_dict(previous_state),
                    new_state=self._orchestrator._state_to_dict(current_state),
                    confidence=regime_confidence,
                    expected_impact=urgency * 0.15,
                )
                decisions.append(decision)

                if self.config.enable_auto_rotation and active_strategies:
                    strategy_changes = self._coordinator.coordinate_regime_strategy(
                        regime, active_strategies
                    )

            if self.config.enable_dynamic_sizing and active_strategies:
                position_adjustments = self._coordinator.coordinate_strategy_sizing(
                    active_strategies, current_state
                )

            if self.config.enable_auto_retrain and self.model_manager:
                model_actions = self._coordinator.coordinate_model_strategy(
                    ["ensemble_model"], active_strategies
                )

        if len(decisions) > 1:
            decisions = self._coordinator.resolve_conflicts(decisions)

        duration_ms = (time.monotonic() - start) * 1000.0

        result = AdaptationResult(
            timestamp=ts,
            cycle_duration_ms=duration_ms,
            regime_detected=regime,
            regime_confidence=regime_confidence,
            strategy_changes=strategy_changes,
            model_actions=model_actions,
            position_adjustments=position_adjustments,
            alerts_generated=alerts,
            adaptation_decisions=decisions,
        )

        return result

    def get_current_state(self) -> Optional[MarketState]:
        """Get the current market state."""
        return self._orchestrator.get_current_state()

    def get_adaptation_history(self, hours: int = 24) -> List[AdaptationRecord]:
        """Get adaptation history for the specified time window."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [r for r in self._adaptation_records if r.timestamp >= cutoff]

    async def force_adaptation(self, reason: str) -> AdaptationResult:
        """Force an adaptation cycle regardless of normal triggers."""
        ts = datetime.utcnow()
        logger.info("Forced adaptation triggered: reason=%s", reason)

        decision = AdaptationDecision(
            timestamp=ts,
            decision_type="regime_change",
            trigger=f"forced:{reason}",
            old_state=self._orchestrator._state_to_dict(
                self._orchestrator.get_current_state()
            ),
            new_state={},
            confidence=1.0,
            expected_impact=0.1,
        )

        result = await self._orchestrator.execute_adaptation(decision)

        record = AdaptationRecord(
            timestamp=ts,
            trigger=f"forced:{reason}",
            decisions=[decision],
            result=result,
            success=True,
            notes=reason,
        )
        self._adaptation_records.append(record)

        return result

    def get_health_status(self) -> SystemHealth:
        """Get the overall system health status."""
        orchestrator_health = self._orchestrator.get_health_status()

        component_status = {
            "regime_detector_status": "healthy",
            "strategy_selector_status": "healthy",
            "position_sizer_status": "healthy",
            "model_manager_status": "healthy",
        }

        for name, key in [
            ("regime_detector", "regime_detector_status"),
            ("strategy_selector", "strategy_selector_status"),
            ("position_sizer", "position_sizer_status"),
            ("model_manager", "model_manager_status"),
        ]:
            comp = self.components.get(name)
            if comp is None:
                component_status[key] = "not_configured"
            elif hasattr(comp, "get_health"):
                try:
                    health = comp.get_health()
                    component_status[key] = health if isinstance(health, str) else "healthy"
                except Exception:
                    component_status[key] = "error"

        statuses = list(component_status.values())
        if "error" in statuses or "critical" in statuses:
            overall = "critical"
        elif "degraded" in statuses or "not_configured" in statuses:
            overall = "degraded"
        else:
            overall = "healthy"

        return SystemHealth(
            overall_status=overall,
            last_adaptation=orchestrator_health.last_adaptation,
            adaptations_today=orchestrator_health.adaptations_today,
            errors_today=orchestrator_health.errors_today,
            **component_status,
        )

    @staticmethod
    def _get_market_data() -> Any:
        """Get market data for regime detection. Override or inject as needed."""
        return None
