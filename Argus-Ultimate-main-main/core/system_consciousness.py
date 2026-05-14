#!/usr/bin/env python3
"""
System Consciousness — self-awareness module for ARGUS.

Knows its own state, capabilities, limitations, strengths, and weaknesses.
Generates holistic trade/no-trade decisions and natural-language daily briefings.
Updates every cycle based on component feedback.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SelfAssessment:
    """ARGUS self-assessment snapshot."""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    current_state: str = "learning"  # confident, cautious, stressed, recovering, learning
    confidence_level: float = 0.5  # 0..1
    adaptation_rate: float = 0.5  # how fast the system is learning
    risk_appetite: float = 0.5  # 0..1 scale
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComponentFeedback:
    """Performance feedback from a single component."""
    name: str
    score: float  # 0..1 capability score
    recent_accuracy: float = 0.0
    error_rate: float = 0.0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# SystemConsciousness
# ---------------------------------------------------------------------------

class SystemConsciousness:
    """
    Self-awareness module that aggregates all component feedback into a
    holistic view of the system's state and capabilities.

    Parameters
    ----------
    config : dict, optional
        ``system_consciousness`` section from unified_config.yaml.
    """

    _DEFAULTS: Dict[str, Any] = {
        "enabled": True,
        "confidence_decay_rate": 0.01,
        "confidence_boost_rate": 0.02,
        "stressed_drawdown_pct": 5.0,
        "confident_win_streak": 5,
        "cautious_loss_streak": 3,
        "min_trades_for_assessment": 10,
        "capability_smoothing": 0.3,
    }

    # Default capability scores (before any real data)
    _DEFAULT_CAPABILITIES = {
        "regime_detection": 0.50,
        "signal_generation": 0.50,
        "execution_quality": 0.50,
        "risk_management": 0.50,
        "portfolio_management": 0.50,
        "market_analysis": 0.50,
        "timing_accuracy": 0.50,
        "position_sizing": 0.50,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(self._DEFAULTS)
        if config:
            cfg.update(config)
        self._cfg = cfg
        self._enabled = bool(cfg.get("enabled", True))
        self._decay = float(cfg.get("confidence_decay_rate", 0.01))
        self._boost = float(cfg.get("confidence_boost_rate", 0.02))
        self._stressed_dd = float(cfg.get("stressed_drawdown_pct", 5.0))
        self._confident_streak = int(cfg.get("confident_win_streak", 5))
        self._cautious_streak = int(cfg.get("cautious_loss_streak", 3))
        self._min_trades = int(cfg.get("min_trades_for_assessment", 10))
        self._smoothing = float(cfg.get("capability_smoothing", 0.3))

        # State
        self._capabilities: Dict[str, float] = dict(self._DEFAULT_CAPABILITIES)
        self._performance_history: List[Dict[str, Any]] = []
        self._current_state = "learning"
        self._confidence = 0.5
        self._risk_appetite = 0.5
        self._adaptation_rate = 0.5
        self._component_feedback: Dict[str, ComponentFeedback] = {}
        self._recent_trades: List[Dict[str, Any]] = []  # [{pnl, win, timestamp}, ...]
        self._daily_stats: Dict[str, Any] = {}
        self._goals_summary: Dict[str, Any] = {}
        self._regime: str = "unknown"
        self._drawdown_pct: float = 0.0
        self._total_pnl: float = 0.0

        logger.info("SystemConsciousness initialised (enabled=%s)", self._enabled)

    # ------------------------------------------------------------------
    # External data injection
    # ------------------------------------------------------------------

    def update_component(self, feedback: ComponentFeedback) -> None:
        """Record performance feedback from a component."""
        self._component_feedback[feedback.name] = feedback

        # Update capability if it maps to a known capability
        cap_mapping = {
            "regime_classifier": "regime_detection",
            "regime_store": "regime_detection",
            "hmm_regime": "regime_detection",
            "strategy_engine": "signal_generation",
            "signal_stacker": "signal_generation",
            "ensemble_signal_hub": "signal_generation",
            "alpha_model": "signal_generation",
            "smart_order_execution": "execution_quality",
            "fill_tracker": "execution_quality",
            "maker_enforcement": "execution_quality",
            "unified_risk_manager": "risk_management",
            "intraday_var": "risk_management",
            "stress_tester": "risk_management",
            "portfolio_optimizer": "portfolio_management",
            "position_registry": "portfolio_management",
            "volatility_forecaster": "market_analysis",
            "fear_greed": "market_analysis",
        }
        cap_key = cap_mapping.get(feedback.name)
        if cap_key and cap_key in self._capabilities:
            # Exponential moving average update
            old = self._capabilities[cap_key]
            self._capabilities[cap_key] = round(
                old * (1 - self._smoothing) + feedback.score * self._smoothing, 4
            )

    def record_trade_result(self, pnl: float, win: bool) -> None:
        """Record a trade outcome."""
        self._recent_trades.append({
            "pnl": pnl,
            "win": win,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._recent_trades) > 200:
            self._recent_trades = self._recent_trades[-200:]
        self._total_pnl += pnl

        # Update confidence
        if win:
            self._confidence = min(1.0, self._confidence + self._boost)
        else:
            self._confidence = max(0.0, self._confidence - self._decay * 2)

    def update_market_state(
        self,
        regime: str = "unknown",
        drawdown_pct: float = 0.0,
        daily_stats: Optional[Dict[str, Any]] = None,
        goals_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Inject current market/portfolio state."""
        self._regime = regime
        self._drawdown_pct = drawdown_pct
        if daily_stats:
            self._daily_stats = daily_stats
        if goals_summary:
            self._goals_summary = goals_summary

    # ------------------------------------------------------------------
    # Self-assessment
    # ------------------------------------------------------------------

    def get_self_assessment(self) -> SelfAssessment:
        """
        Generate a comprehensive self-assessment.

        Returns
        -------
        SelfAssessment
        """
        self._update_state()

        strengths = self._identify_strengths()
        weaknesses = self._identify_weaknesses()

        return SelfAssessment(
            strengths=strengths,
            weaknesses=weaknesses,
            current_state=self._current_state,
            confidence_level=round(self._confidence, 4),
            adaptation_rate=round(self._adaptation_rate, 4),
            risk_appetite=round(self._risk_appetite, 4),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _update_state(self) -> None:
        """Determine current system state from all signals."""
        # Win/loss streak
        recent = self._recent_trades[-20:]
        if not recent:
            self._current_state = "learning"
            return

        wins = [t for t in recent if t["win"]]
        losses = [t for t in recent if not t["win"]]
        win_rate = len(wins) / max(len(recent), 1)

        # Streak detection
        streak = 0
        streak_type = "neutral"
        for t in reversed(recent):
            if streak == 0:
                streak_type = "win" if t["win"] else "loss"
                streak = 1
            elif (streak_type == "win" and t["win"]) or (streak_type == "loss" and not t["win"]):
                streak += 1
            else:
                break

        # Determine state
        if self._drawdown_pct > self._stressed_dd:
            self._current_state = "stressed"
            self._risk_appetite = max(0.1, self._risk_appetite - 0.05)
        elif streak_type == "loss" and streak >= self._cautious_streak:
            self._current_state = "cautious"
            self._risk_appetite = max(0.2, self._risk_appetite - 0.03)
        elif streak_type == "win" and streak >= self._confident_streak:
            self._current_state = "confident"
            self._risk_appetite = min(0.9, self._risk_appetite + 0.02)
        elif self._drawdown_pct > self._stressed_dd * 0.5 and self._drawdown_pct <= self._stressed_dd:
            self._current_state = "recovering"
            self._risk_appetite = min(0.6, self._risk_appetite + 0.01)
        elif len(recent) < self._min_trades:
            self._current_state = "learning"
        else:
            self._current_state = "confident" if win_rate > 0.55 else "cautious"

        # Update adaptation rate based on how much capability scores are changing
        if len(self._performance_history) >= 2:
            prev = self._performance_history[-1]
            changes = []
            for cap, score in self._capabilities.items():
                old_score = prev.get(cap, score)
                changes.append(abs(score - old_score))
            self._adaptation_rate = min(1.0, sum(changes) / max(len(changes), 1) * 10)

        # Record snapshot for adaptation tracking
        self._performance_history.append(dict(self._capabilities))
        if len(self._performance_history) > 100:
            self._performance_history = self._performance_history[-100:]

    def _identify_strengths(self) -> List[str]:
        """Identify system strengths from capabilities and performance."""
        strengths: List[str] = []

        # Capability-based
        for cap, score in sorted(self._capabilities.items(), key=lambda x: x[1], reverse=True):
            if score >= 0.7:
                strengths.append(f"Strong {cap.replace('_', ' ')} ({score:.0%})")

        # Performance-based
        recent = self._recent_trades[-50:]
        if len(recent) >= 10:
            win_rate = sum(1 for t in recent if t["win"]) / len(recent)
            if win_rate > 0.55:
                strengths.append(f"High recent win rate ({win_rate:.0%})")
            avg_pnl = sum(t["pnl"] for t in recent) / len(recent)
            if avg_pnl > 0:
                strengths.append(f"Positive average trade PnL ({avg_pnl:.4f})")

        # Regime-based
        if self._regime in ("mean_revert", "ranging"):
            cap = self._capabilities.get("signal_generation", 0.5)
            if cap > 0.6:
                strengths.append(f"Good in {self._regime} regime")

        if not strengths:
            strengths.append("System is still learning — insufficient data")

        return strengths[:8]

    def _identify_weaknesses(self) -> List[str]:
        """Identify system weaknesses."""
        weaknesses: List[str] = []

        # Capability-based
        for cap, score in sorted(self._capabilities.items(), key=lambda x: x[1]):
            if score < 0.4:
                weaknesses.append(f"Weak {cap.replace('_', ' ')} ({score:.0%})")

        # Performance-based
        recent = self._recent_trades[-50:]
        if len(recent) >= 10:
            win_rate = sum(1 for t in recent if t["win"]) / len(recent)
            if win_rate < 0.45:
                weaknesses.append(f"Low recent win rate ({win_rate:.0%})")
            losses = [t["pnl"] for t in recent if not t["win"]]
            if losses:
                avg_loss = sum(losses) / len(losses)
                weaknesses.append(f"Average loss size: {avg_loss:.4f}")

        # Component errors
        for name, fb in self._component_feedback.items():
            if fb.error_rate > 0.1:
                weaknesses.append(f"High error rate in {name} ({fb.error_rate:.0%})")
            if fb.latency_ms > 500:
                weaknesses.append(f"High latency in {name} ({fb.latency_ms:.0f}ms)")

        if not weaknesses:
            weaknesses.append("No significant weaknesses identified")

        return weaknesses[:8]

    # ------------------------------------------------------------------
    # Capability matrix
    # ------------------------------------------------------------------

    def get_capability_matrix(self) -> Dict[str, float]:
        """Return capability scores for all capabilities."""
        return dict(self._capabilities)

    def update_capability(self, capability: str, score: float) -> None:
        """Directly update a capability score."""
        if capability in self._capabilities:
            old = self._capabilities[capability]
            self._capabilities[capability] = round(
                old * (1 - self._smoothing) + score * self._smoothing, 4
            )

    # ------------------------------------------------------------------
    # Should we trade?
    # ------------------------------------------------------------------

    def should_trade(self) -> Tuple[bool, str]:
        """
        Holistic decision on whether conditions are right for trading.

        Returns
        -------
        (should_trade, reason)
        """
        reasons_no: List[str] = []
        reasons_yes: List[str] = []

        # Check drawdown
        if self._drawdown_pct > self._stressed_dd * 1.5:
            reasons_no.append(f"Drawdown {self._drawdown_pct:.1f}% exceeds safe limit")

        # Check confidence
        if self._confidence < 0.2:
            reasons_no.append(f"System confidence too low ({self._confidence:.0%})")

        # Check system state
        if self._current_state == "stressed":
            reasons_no.append("System is in stressed state — reduce activity")

        # Check capability minimums
        critical_caps = ["signal_generation", "risk_management", "execution_quality"]
        for cap in critical_caps:
            score = self._capabilities.get(cap, 0.5)
            if score < 0.3:
                reasons_no.append(f"Critical capability '{cap}' below minimum ({score:.0%})")

        # Check error rates
        high_error = [
            fb.name for fb in self._component_feedback.values()
            if fb.error_rate > 0.2
        ]
        if high_error:
            reasons_no.append(f"High error rates in: {', '.join(high_error)}")

        # Positive signals
        if self._confidence > 0.5:
            reasons_yes.append(f"Good confidence ({self._confidence:.0%})")
        if self._current_state in ("confident",):
            reasons_yes.append("System in confident state")
        avg_cap = sum(self._capabilities.values()) / max(len(self._capabilities), 1)
        if avg_cap > 0.6:
            reasons_yes.append(f"Average capability strong ({avg_cap:.0%})")

        # Decision
        if reasons_no:
            return False, "; ".join(reasons_no)
        if not reasons_yes:
            return True, "No objections — proceed with normal trading"
        return True, "; ".join(reasons_yes)

    # ------------------------------------------------------------------
    # Daily briefing
    # ------------------------------------------------------------------

    def get_daily_briefing(self) -> str:
        """Generate a natural-language daily briefing of system state."""
        assessment = self.get_self_assessment()
        should, reason = self.should_trade()
        caps = self.get_capability_matrix()

        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("ARGUS DAILY BRIEFING")
        lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("=" * 60)

        # System state
        lines.append(f"\nSystem State: {assessment.current_state.upper()}")
        lines.append(f"Confidence: {assessment.confidence_level:.0%}")
        lines.append(f"Risk Appetite: {assessment.risk_appetite:.0%}")
        lines.append(f"Adaptation Rate: {assessment.adaptation_rate:.0%}")

        # Market context
        lines.append(f"\nMarket Regime: {self._regime}")
        lines.append(f"Current Drawdown: {self._drawdown_pct:.2f}%")
        lines.append(f"Total PnL: {self._total_pnl:.4f}")

        # Trade decision
        lines.append(f"\nShould Trade: {'YES' if should else 'NO'}")
        lines.append(f"Reason: {reason}")

        # Strengths
        lines.append("\nStrengths:")
        for s in assessment.strengths:
            lines.append(f"  + {s}")

        # Weaknesses
        lines.append("\nWeaknesses:")
        for w in assessment.weaknesses:
            lines.append(f"  - {w}")

        # Capabilities
        lines.append("\nCapability Matrix:")
        for cap, score in sorted(caps.items(), key=lambda x: x[1], reverse=True):
            bar = "#" * int(score * 20)
            lines.append(f"  {cap:<25s} {bar:<20s} {score:.0%}")

        # Recent performance
        recent = self._recent_trades[-20:]
        if recent:
            wins = sum(1 for t in recent if t["win"])
            total = len(recent)
            total_pnl = sum(t["pnl"] for t in recent)
            lines.append(f"\nRecent Trades (last {total}):")
            lines.append(f"  Win Rate: {wins}/{total} ({wins/total:.0%})")
            lines.append(f"  Total PnL: {total_pnl:.4f}")

        # Goals
        if self._goals_summary:
            lines.append("\nGoals:")
            for key, val in self._goals_summary.items():
                lines.append(f"  {key}: {val}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
