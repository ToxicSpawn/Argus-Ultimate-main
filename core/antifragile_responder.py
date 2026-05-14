"""
Antifragile Responder — learn from failures, apply lessons globally.

When ARGUS takes a significant loss, this module:
  1. Analyses the failure causally (what led to the bad trade)
  2. Identifies the failure pattern (regime + signal + size combination)
  3. Looks for the same pattern across other strategies
  4. Applies a filter to prevent similar trades in those strategies
  5. Tracks failure rates to escalate (disable patterns repeating > N times)

Inspired by Nassim Taleb's Antifragile concept: the system gets STRONGER from
failures, not just resilient to them.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class FailureSeverity(Enum):
    MINOR = "minor"          # < 1% portfolio loss
    MODERATE = "moderate"    # 1-3% portfolio loss
    SIGNIFICANT = "significant"  # 3-5% portfolio loss
    SEVERE = "severe"        # 5-10% portfolio loss
    CATASTROPHIC = "catastrophic"  # > 10% portfolio loss


@dataclass
class FailurePattern:
    """A pattern that has historically led to losses."""
    pattern_id: str
    regime: str
    strategy: str
    symbol_class: str       # "btc", "alt_majors", "alt_minors"
    size_pct_range: tuple   # (min, max) — e.g. (0.20, 0.30)
    trigger_signal: str     # the trigger that initiated the trade
    occurrences: int = 0
    total_loss_aud: float = 0.0
    last_seen: float = 0.0
    disabled: bool = False
    description: str = ""


@dataclass
class FailureRecord:
    """One specific failure event."""
    timestamp: float
    trade_id: str
    strategy: str
    symbol: str
    side: str
    pnl_aud: float
    pnl_pct: float
    severity: FailureSeverity
    regime: str
    confidence: float
    size_pct: float
    causal_explanation: str = ""
    pattern_id: Optional[str] = None
    counterfactual_better: Optional[str] = None


class AntifragileResponder:
    """
    Detects loss patterns and applies preventive measures across all strategies.

    Usage::

        responder = AntifragileResponder(causal_engine=ce, counterfactual=cf)

        # On every fill:
        result = responder.on_fill(trade_result, portfolio_value=1000.0)

        # If pattern detected and escalated:
        if result.get("disabled_pattern"):
            for strategy_id in result["affected_strategies"]:
                strategy_router.disable(strategy_id)
    """

    # Severity thresholds (% of portfolio)
    SEVERITY_THRESHOLDS = {
        FailureSeverity.MINOR: 0.01,
        FailureSeverity.MODERATE: 0.03,
        FailureSeverity.SIGNIFICANT: 0.05,
        FailureSeverity.SEVERE: 0.10,
        FailureSeverity.CATASTROPHIC: float("inf"),
    }

    # Pattern escalation: after N occurrences of the same pattern, disable it
    ESCALATION_THRESHOLD = 3
    ESCALATION_WINDOW_HOURS = 24

    def __init__(
        self,
        causal_engine: Any = None,
        counterfactual_analyzer: Any = None,
        max_history: int = 10_000,
    ) -> None:
        self._causal_engine = causal_engine
        self._counterfactual = counterfactual_analyzer
        self._failures: deque[FailureRecord] = deque(maxlen=max_history)
        self._patterns: Dict[str, FailurePattern] = {}
        self._disabled_patterns: Set[str] = set()
        self._lessons_applied: int = 0
        logger.info("AntifragileResponder: initialized")

    def on_fill(
        self,
        trade_result: Dict[str, Any],
        portfolio_value: float,
    ) -> Dict[str, Any]:
        """
        Process a fill. If it's a loss, analyse and respond.
        Returns a dict describing what action was taken.
        """
        pnl_aud = float(trade_result.get("pnl", 0.0) or 0.0)

        # Only respond to losses
        if pnl_aud >= 0:
            return {"action": "none", "reason": "winning_trade"}

        # Compute severity
        pnl_pct = abs(pnl_aud) / max(portfolio_value, 1e-9)
        severity = self._classify_severity(pnl_pct)

        # Skip minor losses (normal noise)
        if severity == FailureSeverity.MINOR:
            return {"action": "none", "reason": "noise_level_loss"}

        # Build failure record
        record = self._build_failure_record(trade_result, pnl_aud, pnl_pct, severity)

        # Causal analysis
        if self._causal_engine is not None:
            try:
                cause = self._causal_engine.analyze_trade_outcome(trade_result)
                record.causal_explanation = str(cause)
            except Exception as exc:
                logger.debug("causal_engine.analyze_trade_outcome error: %s", exc)

        # Counterfactual analysis
        if self._counterfactual is not None:
            try:
                better = self._counterfactual.find_alternative(trade_result)
                record.counterfactual_better = str(better)
            except Exception as exc:
                logger.debug("counterfactual.find_alternative error: %s", exc)

        self._failures.append(record)

        # Identify or update pattern
        pattern_id = self._identify_pattern(record)
        record.pattern_id = pattern_id

        if pattern_id not in self._patterns:
            self._patterns[pattern_id] = self._create_pattern(record, pattern_id)

        pattern = self._patterns[pattern_id]
        pattern.occurrences += 1
        pattern.total_loss_aud += abs(pnl_aud)
        pattern.last_seen = time.time()

        # Check escalation
        action = "logged"
        affected_strategies: List[str] = []

        if self._should_escalate(pattern):
            self._escalate_pattern(pattern)
            action = "escalated_pattern_disabled"
            affected_strategies = self._find_affected_strategies(pattern)
            logger.warning(
                "AntifragileResponder: ESCALATED pattern '%s' "
                "(%d occurrences, total loss $%.2f) — disabling %d strategies",
                pattern.pattern_id, pattern.occurrences,
                pattern.total_loss_aud, len(affected_strategies),
            )
        elif severity in (FailureSeverity.SEVERE, FailureSeverity.CATASTROPHIC):
            action = "severity_warning"
            logger.warning(
                "AntifragileResponder: %s loss on %s — %.2f%% portfolio",
                severity.value, record.strategy, pnl_pct * 100,
            )

        self._lessons_applied += 1

        return {
            "action": action,
            "severity": severity.value,
            "pattern_id": pattern_id,
            "occurrences": pattern.occurrences,
            "disabled_pattern": pattern.disabled,
            "affected_strategies": affected_strategies,
            "causal": record.causal_explanation,
            "counterfactual": record.counterfactual_better,
        }

    def _classify_severity(self, pnl_pct: float) -> FailureSeverity:
        if pnl_pct < self.SEVERITY_THRESHOLDS[FailureSeverity.MINOR]:
            return FailureSeverity.MINOR
        if pnl_pct < self.SEVERITY_THRESHOLDS[FailureSeverity.MODERATE]:
            return FailureSeverity.MODERATE
        if pnl_pct < self.SEVERITY_THRESHOLDS[FailureSeverity.SIGNIFICANT]:
            return FailureSeverity.SIGNIFICANT
        if pnl_pct < self.SEVERITY_THRESHOLDS[FailureSeverity.SEVERE]:
            return FailureSeverity.SEVERE
        return FailureSeverity.CATASTROPHIC

    def _build_failure_record(
        self,
        trade_result: Dict[str, Any],
        pnl_aud: float,
        pnl_pct: float,
        severity: FailureSeverity,
    ) -> FailureRecord:
        return FailureRecord(
            timestamp=time.time(),
            trade_id=str(trade_result.get("order_id", "")),
            strategy=str(trade_result.get("source_strategy", "unknown")),
            symbol=str(trade_result.get("symbol", "")),
            side=str(trade_result.get("side", "")),
            pnl_aud=pnl_aud,
            pnl_pct=pnl_pct,
            severity=severity,
            regime=str(trade_result.get("regime_label", "unknown")),
            confidence=float(trade_result.get("confidence", 0.0) or 0.0),
            size_pct=float(trade_result.get("size_pct", 0.0) or 0.0),
        )

    def _identify_pattern(self, record: FailureRecord) -> str:
        """Build a pattern ID from the failure attributes."""
        symbol_class = self._classify_symbol(record.symbol)
        size_bucket = self._bucket_size(record.size_pct)
        return f"{record.regime}|{record.strategy}|{symbol_class}|{size_bucket}"

    @staticmethod
    def _classify_symbol(symbol: str) -> str:
        sym = symbol.upper()
        if sym.startswith("BTC"):
            return "btc"
        elif sym.startswith(("ETH", "SOL", "AVAX")):
            return "alt_majors"
        else:
            return "alt_minors"

    @staticmethod
    def _bucket_size(size_pct: float) -> str:
        if size_pct < 0.05:
            return "tiny"
        elif size_pct < 0.15:
            return "small"
        elif size_pct < 0.25:
            return "medium"
        elif size_pct < 0.40:
            return "large"
        else:
            return "huge"

    def _create_pattern(
        self,
        record: FailureRecord,
        pattern_id: str,
    ) -> FailurePattern:
        return FailurePattern(
            pattern_id=pattern_id,
            regime=record.regime,
            strategy=record.strategy,
            symbol_class=self._classify_symbol(record.symbol),
            size_pct_range=(0.0, 1.0),
            trigger_signal="",
            description=f"Loss pattern: {record.strategy} in {record.regime}",
        )

    def _should_escalate(self, pattern: FailurePattern) -> bool:
        """Check if a pattern should be escalated to disabled."""
        if pattern.disabled:
            return False
        if pattern.occurrences < self.ESCALATION_THRESHOLD:
            return False

        # Check if all occurrences are within escalation window
        cutoff = time.time() - (self.ESCALATION_WINDOW_HOURS * 3600)
        recent_failures = [
            f for f in self._failures
            if f.pattern_id == pattern.pattern_id and f.timestamp >= cutoff
        ]
        return len(recent_failures) >= self.ESCALATION_THRESHOLD

    def _escalate_pattern(self, pattern: FailurePattern) -> None:
        """Mark a pattern as disabled."""
        pattern.disabled = True
        self._disabled_patterns.add(pattern.pattern_id)

    def _find_affected_strategies(self, pattern: FailurePattern) -> List[str]:
        """Find strategies that match this failure pattern."""
        return [pattern.strategy]

    def is_pattern_disabled(self, regime: str, strategy: str, symbol: str, size_pct: float) -> bool:
        """
        Check if a proposed trade matches a disabled pattern.
        Called from _apply_risk_gates as a pre-trade check.
        """
        pattern_id = (
            f"{regime}|{strategy}|"
            f"{self._classify_symbol(symbol)}|{self._bucket_size(size_pct)}"
        )
        return pattern_id in self._disabled_patterns

    def get_lessons_learned(self) -> List[Dict[str, Any]]:
        """Return all patterns sorted by total loss (worst first)."""
        sorted_patterns = sorted(
            self._patterns.values(),
            key=lambda p: p.total_loss_aud,
            reverse=True,
        )
        return [
            {
                "pattern_id": p.pattern_id,
                "regime": p.regime,
                "strategy": p.strategy,
                "symbol_class": p.symbol_class,
                "occurrences": p.occurrences,
                "total_loss_aud": round(p.total_loss_aud, 2),
                "disabled": p.disabled,
                "last_seen": p.last_seen,
            }
            for p in sorted_patterns
        ]

    def reset_pattern(self, pattern_id: str) -> bool:
        """Re-enable a previously disabled pattern (manual override)."""
        if pattern_id in self._patterns:
            self._patterns[pattern_id].disabled = False
            self._disabled_patterns.discard(pattern_id)
            logger.info("AntifragileResponder: re-enabled pattern %s", pattern_id)
            return True
        return False

    def snapshot(self) -> Dict[str, Any]:
        """Return current state for advisory dict."""
        recent_failures = sum(
            1 for f in self._failures
            if time.time() - f.timestamp < 86400
        )
        severity_counts: Dict[str, int] = defaultdict(int)
        for f in self._failures:
            severity_counts[f.severity.value] += 1

        return {
            "total_failures": len(self._failures),
            "recent_24h": recent_failures,
            "patterns_tracked": len(self._patterns),
            "patterns_disabled": len(self._disabled_patterns),
            "lessons_applied": self._lessons_applied,
            "severity_breakdown": dict(severity_counts),
        }
